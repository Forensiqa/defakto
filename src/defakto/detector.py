# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

from defakto.artifacts import Artifact, load_artifact
from defakto.dynamics import DynamicsScore
from defakto.encoders import Encoder, build_encoder
from defakto.errors import ArtifactError, EncoderError, UnsupportedMediaError
from defakto.localize import localize_text
from defakto.results import Result, Segment
from defakto.trajectory import Trajectory

Localizer = Callable[[Any, Trajectory, DynamicsScore, float], list[Segment]]

_LOCALIZERS: dict[str, Localizer] = {
    "text": localize_text,
}


class Detector:
    """One artifact paired with the encoder it was fitted with.

    ``Detector.from_artifact(...)`` builds the right encoder from the artifact's manifest,
    so a detector is chosen by choosing an artifact.
    """

    def __init__(self, artifact: Artifact, encoder: Encoder) -> None:
        manifest = artifact.manifest
        if encoder.media_type != manifest.media_type:
            raise EncoderError(
                f"Encoder produces {encoder.media_type} trajectories, but artifact "
                f"'{manifest.id}' is for {manifest.media_type}"
            )
        self.artifact = artifact
        self.encoder = encoder

    @classmethod
    def from_artifact(cls, ref: str | Path | Artifact, *, device: str | None = None) -> Self:
        artifact = ref if isinstance(ref, Artifact) else load_artifact(ref)
        return cls(artifact, build_encoder(artifact.manifest.encoder, device=device))

    @property
    def media_type(self) -> str:
        return self.artifact.manifest.media_type

    def analyze(
        self, content: Any, *, threshold: float | None = None, source: str | None = None
    ) -> Result:
        """Encode ``content`` and score it."""
        trajectory = self.encoder.encode(content)
        return self.analyze_trajectory(
            trajectory, content=content, threshold=threshold, source=source
        )

    def analyze_trajectory(
        self,
        trajectory: Trajectory,
        *,
        content: Any = None,
        threshold: float | None = None,
        source: str | None = None,
    ) -> Result:
        """Score a trajectory that was already encoded, for example one shared by several
        artifacts that use the same encoder."""
        manifest = self.artifact.manifest
        if trajectory.media_type != manifest.media_type:
            raise UnsupportedMediaError(
                f"Artifact '{manifest.id}' is for {manifest.media_type}, "
                f"got a {trajectory.media_type} trajectory"
            )
        if trajectory.dim != manifest.embedding_dim:
            raise ArtifactError(
                f"Embeddings have {trajectory.dim} dimensions, but artifact '{manifest.id}' "
                f"expects {manifest.embedding_dim}. The encoder doesn't match the artifact."
            )
        if trajectory.num_steps < 2:
            raise ValueError(
                f"Input is too short to analyze: got {trajectory.num_steps} step(s), need at "
                f"least 2"
            )

        threshold = manifest.threshold if threshold is None else threshold
        dynamics = self.artifact.model.score(trajectory.embeddings)

        warnings = list(trajectory.notes)
        if trajectory.num_steps < manifest.min_steps:
            warnings.append(
                f"Only {trajectory.num_steps} steps (fewer than {manifest.min_steps}); "
                f"the score may be unreliable."
            )

        localize = _LOCALIZERS.get(manifest.media_type)
        segments: list[Segment] = []
        if localize is not None:
            segments = localize(content, trajectory, dynamics, threshold)

        return Result(
            media_type=manifest.media_type,
            source=source,
            artifact_id=manifest.id,
            artifact_version=manifest.version,
            task=manifest.task,
            positive_label=manifest.positive_label,
            negative_label=manifest.negative_label,
            score=dynamics.score,
            threshold=float(threshold),
            flagged=dynamics.score > threshold,
            num_steps=trajectory.num_steps,
            residual_negative=dynamics.residual_negative,
            residual_positive=dynamics.residual_positive,
            segments=segments,
            warnings=warnings,
        )
