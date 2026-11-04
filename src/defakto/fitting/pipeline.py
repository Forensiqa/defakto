# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from defakto.artifacts import Artifact, EncoderSpec, load_artifact, save_artifact
from defakto.encoders import Encoder, build_encoder
from defakto.fitting.calibrate import Metrics, choose_threshold, evaluate, score_trajectories
from defakto.fitting.solver import fit_koopman
from defakto.fitting.store import TrajectoryStore

Examples = Iterable[tuple[Any, bool]]
"""``(content, positive)`` pairs: ``positive`` is True for the class the artifact flags."""


@dataclass(frozen=True)
class FitResult:
    """A fitted artifact and how it did.

    ``validation`` and ``test`` hold metrics for all examples first, then for longer
    examples only (see ``LENGTH_BUCKETS``). They are empty when that split wasn't given.
    """

    artifact: Artifact
    threshold: float
    validation: list[Metrics] = field(default_factory=list)
    test: list[Metrics] = field(default_factory=list)


def fit(
    train: Examples | TrajectoryStore,
    *,
    encoder: EncoderSpec | dict[str, Any],
    output: str | Path,
    artifact_id: str,
    validation: Examples | TrajectoryStore | None = None,
    test: Examples | TrajectoryStore | None = None,
    positive_label: str = "flagged",
    negative_label: str = "not flagged",
    task: str = "",
    version: str = "0.1.0",
    lift_modes: int = 40,
    observable_modes: int = 500,
    min_steps: int = 2,
    device: str | None = None,
    training_metadata: dict[str, Any] | None = None,
    on_progress: Callable[[str, int], None] | None = None,
) -> FitResult:
    """Fit an artifact from labeled examples and write it to ``output``.

    Each split is either ``(content, positive)`` examples, which are encoded with
    ``encoder`` first, or a ``TrajectoryStore`` that was already encoded with it.

    The threshold is the one with the best balanced accuracy on ``validation``; without
    validation data it is 0, where both operators predict equally well. ``test`` is only
    measured, never used for choices, so its metrics estimate how the artifact does on
    unseen data.

    The manifest's ``training`` section records how the artifact was made: the defakto
    version, the date, the example counts, how the threshold was chosen and the metrics.
    ``training_metadata`` adds what ``fit`` can't know, such as the dataset's name; it
    can't replace the counts, the threshold rule or the metrics. ``on_progress`` is
    called with a split name and the number of examples encoded so far.
    """
    spec = encoder if isinstance(encoder, EncoderSpec) else EncoderSpec.from_dict(encoder)
    built: list[Encoder] = []

    def get_encoder() -> Encoder:
        if not built:
            built.append(build_encoder(spec, device=device))
        return built[0]

    # Splits given as examples are encoded to disk here, too big to keep in memory, and
    # deleted when fitting ends. Pass a TrajectoryStore to keep and reuse encodings.
    with tempfile.TemporaryDirectory(prefix="defakto-fit-") as work_dir:

        def as_store(name: str, data: Examples | TrajectoryStore) -> TrajectoryStore:
            if isinstance(data, TrajectoryStore):
                return data
            progress = None if on_progress is None else lambda n: on_progress(name, n)
            return TrajectoryStore.encode(
                Path(work_dir) / name, get_encoder(), data, on_progress=progress
            )

        train_store = as_store("train", train)
        model = fit_koopman(
            train_store, lift_modes=lift_modes, observable_modes=observable_modes, device=device
        )

        threshold = 0.0
        validation_metrics: list[Metrics] = []
        if validation is not None:
            scores, labels, lengths = score_trajectories(model, as_store("validation", validation))
            threshold = choose_threshold(scores, labels)
            validation_metrics = evaluate(scores, labels, lengths, threshold)
        test_metrics: list[Metrics] = []
        if test is not None:
            test_metrics = evaluate(
                *score_trajectories(model, as_store("test", test)), threshold=threshold
            )

        media_type = train_store.meta.get("media_type") or get_encoder().media_type

        provenance: dict[str, Any] = {
            "defakto_version": _defakto_version(),
            "date": datetime.now(UTC).date().isoformat(),
            **(training_metadata or {}),
            "examples": {
                "train": {
                    "positive": train_store.num_positive,
                    "negative": train_store.num_negative,
                }
            },
            "threshold": (
                "best balanced accuracy on validation" if validation is not None else "default (0)"
            ),
        }
        if validation_metrics:
            provenance["validation"] = [m.to_dict() for m in validation_metrics]
        if test_metrics:
            provenance["test"] = [m.to_dict() for m in test_metrics]

    directory = save_artifact(
        output,
        model,
        artifact_id=artifact_id,
        media_type=media_type,
        encoder=spec,
        positive_label=positive_label,
        negative_label=negative_label,
        task=task,
        version=version,
        threshold=threshold,
        min_steps=min_steps,
        training=provenance,
    )
    return FitResult(
        artifact=load_artifact(directory),
        threshold=threshold,
        validation=validation_metrics,
        test=test_metrics,
    )


def _defakto_version() -> str:
    from defakto import __version__

    return __version__
