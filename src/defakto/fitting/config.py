# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Self

import yaml

from defakto.artifacts import EncoderSpec
from defakto.fitting.solver import FittingError

_INSTALL_HINT = "Install it with: pip install 'defakto[fit]'"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class DataConfig:
    """Where the labeled examples come from: a Hugging Face dataset.

    Rows whose text or label is empty are skipped. A label equal to ``positive`` marks the
    flagged class; every other label is the other class. ``limit`` caps how many usable
    training examples are taken, in dataset order.
    """

    dataset: str
    text_column: str
    label_column: str
    positive: str
    splits: dict[str, str] = field(default_factory=lambda: {"train": "train"})
    revision: str | None = None
    limit: int | None = None

    def source(self, split: str) -> dict[str, Any]:
        """What identifies one split's examples, to tell whether an encoded copy is current."""
        data = {k: v for k, v in asdict(self).items() if k != "splits"}
        if split != "train":
            data.pop("limit")
        return {**data, "split": self.splits[split]}


@dataclass(frozen=True)
class FitConfig:
    """A fitting run, read from YAML: the artifact to make, its encoder, and its data."""

    artifact_id: str
    encoder: EncoderSpec
    encoder_raw: dict[str, Any]
    data: DataConfig
    positive_label: str = "flagged"
    negative_label: str = "not flagged"
    task: str = ""
    version: str = "0.1.0"
    min_steps: int = 2
    lift_modes: int = 40
    observable_modes: int = 500

    @classmethod
    def load(cls, path: str | Path) -> Self:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise FittingError(f"{path}: expected a YAML mapping")
        return cls.from_dict(data, source=str(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "fit config") -> Self:
        def section(name: str, required: bool = True) -> dict[str, Any]:
            value = data.get(name)
            if value is None and not required:
                return {}
            if not isinstance(value, dict):
                raise FittingError(f"{source}: '{name}' must be a mapping")
            return value

        def need(mapping: dict[str, Any], key: str, where: str) -> Any:
            if mapping.get(key) in (None, ""):
                raise FittingError(f"{source}: missing '{where}'")
            return mapping[key]

        artifact, encoder, data_section = section("artifact"), section("encoder"), section("data")
        labels = artifact.get("labels") or {}
        dims = section("dimensions", required=False)
        splits = data_section.get("splits") or {"train": "train"}
        if not isinstance(splits, dict) or "train" not in splits:
            raise FittingError(f"{source}: 'data.splits' needs at least a 'train' split")
        unknown = set(splits) - set(SPLITS)
        if unknown:
            raise FittingError(
                f"{source}: unknown split role(s) {sorted(unknown)}; use {', '.join(SPLITS)}"
            )

        try:
            limit = data_section.get("limit")
            return cls(
                artifact_id=str(need(artifact, "id", "artifact.id")),
                encoder=EncoderSpec.from_dict(encoder),
                encoder_raw=dict(encoder),
                data=DataConfig(
                    dataset=str(need(data_section, "dataset", "data.dataset")),
                    text_column=str(need(data_section, "text_column", "data.text_column")),
                    label_column=str(need(data_section, "label_column", "data.label_column")),
                    positive=str(need(data_section, "positive", "data.positive")),
                    splits={str(k): str(v) for k, v in splits.items()},
                    revision=data_section.get("revision"),
                    limit=None if limit is None else int(limit),
                ),
                positive_label=str(labels.get("positive", "flagged")),
                negative_label=str(labels.get("negative", "not flagged")),
                task=str(artifact.get("task", "")),
                version=str(artifact.get("version", "0.1.0")),
                min_steps=int(artifact.get("min_steps", 2)),
                lift_modes=int(dims.get("lift_modes", 40)),
                observable_modes=int(dims.get("observable_modes", 500)),
            )
        except (TypeError, ValueError) as e:
            raise FittingError(f"{source}: invalid value ({e})") from e


def pin_revisions(cfg: FitConfig) -> FitConfig:
    """Pin the dataset and the encoder's model to the Hub commits they have now.

    Revisions already set are kept. Pinned revisions end up in the artifact's manifest,
    so the run can be reproduced and the artifact always gets the model it was fitted with.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    if not cfg.data.revision:
        cfg = replace(cfg, data=replace(cfg.data, revision=api.dataset_info(cfg.data.dataset).sha))
    if (
        cfg.encoder.type == "hf-token-embeddings"
        and cfg.encoder.model_name
        and not cfg.encoder.get("model_revision")
    ):
        sha = api.model_info(cfg.encoder.model_name).sha
        raw = {**cfg.encoder_raw, "model_revision": sha}
        cfg = replace(cfg, encoder=EncoderSpec.from_dict(raw), encoder_raw=raw)
    return cfg


def load_examples(data: DataConfig, split: str) -> list[tuple[str, bool]]:
    """Load one split's ``(text, positive)`` examples from the Hugging Face Hub."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise FittingError(
            f"Loading Hub datasets needs the 'datasets' package. {_INSTALL_HINT}"
        ) from e

    rows = load_dataset(data.dataset, split=data.splits[split], revision=data.revision)
    for column in (data.text_column, data.label_column):
        if column not in rows.column_names:
            raise FittingError(
                f"{data.dataset} has no column '{column}' (columns: {', '.join(rows.column_names)})"
            )

    limit = data.limit if split == "train" else None
    examples = []
    for text, label in zip(rows[data.text_column], rows[data.label_column], strict=True):
        if not text or label is None or str(label) == "":
            continue
        examples.append((str(text), str(label) == data.positive))
        if limit is not None and len(examples) >= limit:
            break
    return examples
