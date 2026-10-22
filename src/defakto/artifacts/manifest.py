# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Self

from defakto.errors import ArtifactError

SCHEMA = "defakto-artifact/v1"
METHOD_TYPE = "koopman-two-operator"
LIFTING = "quadratic"
REQUIRED_TENSORS = ("lift_modes", "observable_modes", "operator_negative", "operator_positive")


@dataclass(frozen=True)
class EncoderSpec:
    """The encoder an artifact was fitted with: its type, model and settings.

    An artifact is only valid with exactly this encoder, so defakto builds the encoder
    from the artifact rather than letting it be chosen separately.
    """

    type: str
    model_name: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        return self.options.get(name, default)

    def key(self) -> tuple[str, str | None, str]:
        """Artifacts whose encoders have equal keys can share one encoder and one encoding."""
        return (self.type, self.model_name, repr(sorted(self.options.items())))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if "type" not in data:
            raise ArtifactError("manifest 'encoder' section is missing 'type'")
        model_name = data.get("model_name")
        return cls(
            type=str(data["type"]),
            model_name=None if model_name is None else str(model_name),
            options={k: v for k, v in data.items() if k not in ("type", "model_name")},
        )


@dataclass(frozen=True)
class Manifest:
    """What an artifact is, what it needs, and how to check its weights."""

    id: str
    version: str
    media_type: str
    task: str
    positive_label: str
    negative_label: str
    encoder: EncoderSpec
    embedding_dim: int
    lift_modes: int
    observable_modes: int
    threshold: float
    min_steps: int
    weights_file: str
    weights_sha256: str | None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def name(self) -> str:
        return self.id.rsplit("/", 1)[-1]

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "manifest") -> Self:
        def need(section: Any, key: str, where: str) -> Any:
            if not isinstance(section, dict) or key not in section:
                raise ArtifactError(f"{source}: missing '{where}'")
            return section[key]

        schema = data.get("schema")
        if schema != SCHEMA:
            raise ArtifactError(
                f"{source}: unsupported schema {schema!r}; this version of defakto reads {SCHEMA!r}"
            )
        method = need(data, "method", "method")
        if need(method, "type", "method.type") != METHOD_TYPE:
            raise ArtifactError(f"{source}: unsupported method {method.get('type')!r}")
        if method.get("lifting", LIFTING) != LIFTING:
            raise ArtifactError(f"{source}: unsupported lifting {method.get('lifting')!r}")

        labels = need(data, "labels", "labels")
        dims = need(data, "dimensions", "dimensions")
        weights = need(data, "weights", "weights")
        encoder = need(data, "encoder", "encoder")
        if not isinstance(encoder, dict) or not isinstance(weights, dict):
            raise ArtifactError(f"{source}: 'encoder' and 'weights' must be mappings")

        try:
            return cls(
                id=str(need(data, "id", "id")),
                version=str(data.get("version", "0")),
                media_type=str(need(data, "media_type", "media_type")),
                task=str(data.get("task", "")),
                positive_label=str(need(labels, "positive", "labels.positive")),
                negative_label=str(labels.get("negative", "not flagged")),
                encoder=EncoderSpec.from_dict(encoder),
                embedding_dim=int(need(dims, "embedding_dim", "dimensions.embedding_dim")),
                lift_modes=int(need(dims, "lift_modes", "dimensions.lift_modes")),
                observable_modes=int(need(dims, "observable_modes", "dimensions.observable_modes")),
                threshold=float(data.get("threshold", 0.0)),
                min_steps=int(data.get("min_steps", data.get("min_tokens", 2))),
                weights_file=str(weights.get("file", "weights.safetensors")),
                weights_sha256=weights.get("sha256"),
                raw=data,
            )
        except (TypeError, ValueError) as e:
            raise ArtifactError(f"{source}: invalid value ({e})") from e
