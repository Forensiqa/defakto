# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import tempfile
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from defakto import __version__
from defakto.analyzer import Analyzer, load_detector_config
from defakto.encoders import Encoder, build_encoder
from defakto.errors import DefaktoError
from defakto.fitting import TrajectoryStore
from defakto.fitting import fit as fit_artifact
from defakto.fitting.config import FitConfig, load_examples, pin_revisions
from defakto.results import Result

app = typer.Typer(
    help="Open-source toolkit for detecting AI-generated and unreliable content, from Forensiqa.",
    no_args_is_help=True,
    add_completion=False,
)


class Show(StrEnum):
    sentence = "sentence"
    token = "token"
    none = "none"


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"defakto {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            callback=_show_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Open-source toolkit for detecting AI-generated and unreliable content, from Forensiqa."""


@app.command()
def analyze(
    source: Annotated[
        str | None,
        typer.Argument(help="File to analyze, or '-' to read text from standard input."),
    ] = None,
    text: Annotated[
        str | None, typer.Option("--text", "-t", help="Analyze this text instead of a file.")
    ] = None,
    artifact: Annotated[
        list[str] | None,
        typer.Option(
            "--artifact",
            "-a",
            help="Artifact folder or Hugging Face Hub id. Repeat to run several.",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="YAML file choosing artifacts per media type.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    media_type: Annotated[
        str | None,
        typer.Option("--media-type", "-m", help="Media type, if the file extension is unclear."),
    ] = None,
    threshold: Annotated[
        float | None, typer.Option(help="Override the flagging threshold of every artifact.")
    ] = None,
    device: Annotated[
        str | None, typer.Option(help="Device for the encoders, for example cpu or cuda.")
    ] = None,
    show: Annotated[Show, typer.Option(help="Which flagged segments to list.")] = Show.sentence,
    as_json: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
) -> None:
    """Analyze a file or text with the chosen artifacts and report what was flagged."""
    if (source is None) == (text is None):
        _fail("Give a file to analyze, '-' to read standard input, or --text.")
    if bool(artifact) == (config is not None):
        _fail("Choose artifacts with --artifact or with --config (one of the two).")

    content: str | Path
    if text is not None:
        content = text
    elif source == "-":
        content = sys.stdin.read()
    else:
        assert source is not None
        content = Path(source)

    try:
        refs = load_detector_config(config) if config is not None else list(artifact or [])
        results = Analyzer(refs, device=device).analyze(
            content, media_type=media_type, threshold=threshold
        )
    except (DefaktoError, FileNotFoundError, ValueError) as e:
        _fail(str(e))

    if as_json:
        typer.echo(json.dumps([r.to_dict() for r in results], indent=2))
        return
    for result in results:
        _print_result(result, show)


@app.command()
def fit(
    config: Annotated[
        Path,
        typer.Argument(
            help="YAML file describing the artifact, its encoder and its data.",
            exists=True,
            dir_okay=False,
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Folder to write the artifact to.")
    ],
    work_dir: Annotated[
        Path | None,
        typer.Option(
            help="Folder to keep encoded examples in. A later run with the same encoder and "
            "data reuses them instead of encoding again."
        ),
    ] = None,
    device: Annotated[
        str | None, typer.Option(help="Device for encoding and fitting, for example cpu or cuda.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an artifact already in the output folder.")
    ] = False,
) -> None:
    """Fit an artifact from a labeled Hugging Face dataset."""
    if (output / "manifest.yaml").exists() and not force:
        _fail(f"'{output}' already holds an artifact. Use --force to overwrite it.")

    try:
        cfg = pin_revisions(FitConfig.load(config))
        data = cfg.data
        with tempfile.TemporaryDirectory(prefix="defakto-fit-") as temp_dir:
            root = work_dir or Path(temp_dir)
            encoders: list[Encoder] = []
            stores: dict[str, TrajectoryStore] = {}
            for split in data.splits:
                meta = {"encoder": cfg.encoder_raw, "source": data.source(split)}
                store = TrajectoryStore.open_if_current(root / split, meta)
                if store is not None:
                    typer.echo(f"Reusing encoded {split} examples from {root / split}")
                else:
                    examples = load_examples(data, split)
                    if not encoders:
                        encoders.append(build_encoder(cfg.encoder, device=device))
                    with typer.progressbar(length=len(examples), label=f"Encoding {split}") as bar:
                        store = TrajectoryStore.encode(
                            root / split,
                            encoders[0],
                            examples,
                            meta=meta,
                            on_progress=lambda _: bar.update(1),
                        )
                stores[split] = store
                typer.echo(
                    f"{split}: {store.num_positive} {cfg.positive_label}, "
                    f"{store.num_negative} {cfg.negative_label}"
                )

            typer.echo("Fitting…")
            result = fit_artifact(
                stores["train"],
                encoder=cfg.encoder,
                output=output,
                artifact_id=cfg.artifact_id,
                validation=stores.get("validation"),
                test=stores.get("test"),
                positive_label=cfg.positive_label,
                negative_label=cfg.negative_label,
                task=cfg.task,
                version=cfg.version,
                lift_modes=cfg.lift_modes,
                observable_modes=cfg.observable_modes,
                min_steps=cfg.min_steps,
                device=device,
                training_metadata={"data": asdict(data)},
            )
    except (DefaktoError, FileNotFoundError, ValueError) as e:
        _fail(str(e))

    typer.echo(f"\nWrote {result.artifact.manifest.id} to {output}")
    typer.echo(f"  threshold {result.threshold:+.4f}")
    for name, metrics in (("validation", result.validation), ("test", result.test)):
        for m in metrics:
            typer.echo(
                f"  {name:<10} steps ≥ {m.min_steps:<3}  n {m.count:<5}  "
                f"balanced acc {m.balanced_accuracy:.3f}  acc {m.accuracy:.3f}  "
                f"TPR {m.tpr:.3f}  FPR {m.fpr:.3f}  AUC {m.auc:.3f}"
            )


def _print_result(result: Result, show: Show) -> None:
    typer.echo(f"\n{result.artifact_id} {result.artifact_version} · {result.task}")
    if result.flagged:
        typer.secho(f"  {result.positive_label} (flagged)", fg=typer.colors.YELLOW, bold=True)
    else:
        typer.secho(f"  {result.negative_label}", fg=typer.colors.GREEN, bold=True)
    typer.echo(
        f"  score {result.score:+.4f}  threshold {result.threshold:+.4f}  steps {result.num_steps}"
    )
    for warning in result.warnings:
        typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)

    if show is Show.none:
        return
    flagged = [s for s in result.segments_of(show.value) if s.flagged]
    if not flagged:
        typer.echo(f"  No flagged {show.value}s.")
        return
    typer.echo(f"  Flagged {show.value}s:")
    for segment in flagged:
        score = segment.score if segment.score is not None else 0.0
        typer.echo(f"    {score:+.3f}  {_shorten(segment.text)}")


def _shorten(text: str, limit: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fail(message: str) -> NoReturn:
    typer.echo(f"defakto: {message}", err=True)
    raise typer.Exit(code=1)
