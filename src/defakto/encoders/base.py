# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any, Protocol

from defakto.artifacts.manifest import EncoderSpec
from defakto.errors import EncoderError
from defakto.trajectory import Trajectory


class Encoder(Protocol):
    """Turns media content into a trajectory of embeddings."""

    @property
    def media_type(self) -> str: ...

    def encode(self, content: Any) -> Trajectory: ...


EncoderFactory = Callable[[EncoderSpec, str | None], Encoder]


def _hf_token_embeddings(spec: EncoderSpec, device: str | None) -> Encoder:
    from defakto.encoders.text import HFTokenEncoder

    return HFTokenEncoder(spec, device=device)


_REGISTRY: dict[str, EncoderFactory] = {
    "hf-token-embeddings": _hf_token_embeddings,
}


def register_encoder(encoder_type: str, factory: EncoderFactory) -> None:
    """Make an encoder type available to artifacts that declare it in their manifest."""
    _REGISTRY[encoder_type] = factory


def available_encoders() -> list[str]:
    return sorted(_REGISTRY)


def build_encoder(spec: EncoderSpec, device: str | None = None) -> Encoder:
    """Build the encoder an artifact's manifest asks for."""
    factory = _REGISTRY.get(spec.type)
    if factory is None:
        raise EncoderError(
            f"Unknown encoder type '{spec.type}'. Available: {', '.join(available_encoders())}"
        )
    return factory(spec, device)
