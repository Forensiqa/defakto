# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Step:
    """One step of a trajectory, and where it sits in the original media.

    For text a step is a token and ``start``/``end`` are character offsets. Other media
    use their own units, recorded in ``Trajectory.span_unit`` (for example seconds).
    """

    label: str
    start: float
    end: float


@dataclass
class Trajectory:
    """A sequence of embeddings: what every encoder produces and every model scores.

    ``embeddings`` has shape ``(T, d)``: one ``d``-dimensional embedding per step, in
    order. ``steps[i]`` describes where embedding ``i`` came from.
    """

    media_type: str
    embeddings: np.ndarray
    steps: list[Step]
    span_unit: str
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D (steps, dim), got shape {self.embeddings.shape}"
            )
        if len(self.steps) != self.embeddings.shape[0]:
            raise ValueError(
                f"got {len(self.steps)} steps for {self.embeddings.shape[0]} embeddings"
            )

    @property
    def num_steps(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])
