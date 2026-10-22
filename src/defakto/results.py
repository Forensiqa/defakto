# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Segment:
    """A part of the input with its own score: a token or a sentence for text.

    ``start``/``end`` use the media's span unit (character offsets for text).
    ``score`` and ``flagged`` are ``None`` for parts that can't be scored, such as
    the first token, which nothing comes before.
    """

    kind: str
    text: str
    start: float
    end: float
    score: float | None
    flagged: bool | None


@dataclass
class Result:
    """The outcome of running one artifact on one input. Same shape for every media type."""

    media_type: str
    source: str | None
    artifact_id: str
    artifact_version: str
    task: str
    positive_label: str
    negative_label: str
    score: float
    threshold: float
    flagged: bool
    num_steps: int
    residual_negative: float
    residual_positive: float
    segments: list[Segment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.positive_label if self.flagged else self.negative_label

    def segments_of(self, kind: str) -> list[Segment]:
        return [s for s in self.segments if s.kind == kind]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"label": self.label}
