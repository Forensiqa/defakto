# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from defakto.artifacts import Artifact, load_artifact
from defakto.detector import Detector
from defakto.encoders import Encoder, build_encoder
from defakto.errors import ArtifactError, UnsupportedMediaError
from defakto.media import load_media
from defakto.results import Result
from defakto.trajectory import Trajectory

ArtifactRef = str | Path | Artifact
ArtifactChoice = ArtifactRef | Sequence[ArtifactRef] | Mapping[str, Sequence[ArtifactRef]]


class Analyzer:
    """Runs the artifacts chosen for each media type on any input.

    Pass a list of artifacts (each is used for the media type its manifest declares), or a
    mapping from media type to artifacts. Artifacts that share an encoder share one encoder
    instance and one encoding of each input, so adding a second text artifact doesn't
    embed the text twice.
    """

    def __init__(self, artifacts: ArtifactChoice, *, device: str | None = None) -> None:
        self.device = device
        self._by_media: dict[str, list[Artifact]] = {}
        self._encoders: dict[tuple[str, str | None, str], Encoder] = {}

        if isinstance(artifacts, Mapping):
            for media_type, refs in artifacts.items():
                for ref in _as_list(refs):
                    artifact = _load(ref)
                    if artifact.manifest.media_type != media_type:
                        raise ArtifactError(
                            f"Artifact '{artifact.manifest.id}' is for "
                            f"{artifact.manifest.media_type}, but it's listed under {media_type}"
                        )
                    self._by_media.setdefault(media_type, []).append(artifact)
        else:
            for ref in _as_list(artifacts):
                artifact = _load(ref)
                self._by_media.setdefault(artifact.manifest.media_type, []).append(artifact)

        if not any(self._by_media.values()):
            raise ArtifactError("No artifacts given")

    @property
    def media_types(self) -> list[str]:
        return sorted(media for media, artifacts in self._by_media.items() if artifacts)

    def artifacts_for(self, media_type: str) -> list[Artifact]:
        return list(self._by_media.get(media_type, []))

    def analyze(
        self,
        source: str | Path,
        *,
        media_type: str | None = None,
        threshold: float | None = None,
    ) -> list[Result]:
        """Analyze a file, or a string of text, with every artifact for its media type."""
        media = load_media(source, media_type)
        artifacts = self._by_media.get(media.media_type)
        if not artifacts:
            configured = ", ".join(self.media_types)
            raise UnsupportedMediaError(
                f"No artifacts are configured for {media.media_type} (configured: {configured})"
            )

        trajectories: dict[tuple[str, str | None, str], Trajectory] = {}
        results: list[Result] = []
        for artifact in artifacts:
            spec = artifact.manifest.encoder
            encoder = self._encoder_for(artifact)
            if spec.key() not in trajectories:
                trajectories[spec.key()] = encoder.encode(media.content)
            results.append(
                Detector(artifact, encoder).analyze_trajectory(
                    trajectories[spec.key()],
                    content=media.content,
                    threshold=threshold,
                    source=media.name,
                )
            )
        return results

    def _encoder_for(self, artifact: Artifact) -> Encoder:
        spec = artifact.manifest.encoder
        if spec.key() not in self._encoders:
            self._encoders[spec.key()] = build_encoder(spec, device=self.device)
        return self._encoders[spec.key()]


def analyze(
    source: str | Path,
    artifacts: ArtifactChoice,
    *,
    media_type: str | None = None,
    threshold: float | None = None,
    device: str | None = None,
) -> list[Result]:
    """Analyze one input with the given artifacts. See ``Analyzer`` to reuse loaded models."""
    analyzer = Analyzer(artifacts, device=device)
    return analyzer.analyze(source, media_type=media_type, threshold=threshold)


def load_detector_config(path: str | Path) -> dict[str, list[str]]:
    """Read a YAML file that chooses artifacts per media type::

        detectors:
          text:
            - ./artifacts/text-safety-jina-v2-base
            - forensiqa/text-safety-qwen3-embedding-0.6b

    Relative folder paths are resolved against the config file's own folder.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    detectors = data.get("detectors") if isinstance(data, dict) else None
    if not isinstance(detectors, dict) or not detectors:
        raise ArtifactError(
            f"{path}: expected a 'detectors' section mapping media types to artifact lists"
        )

    config: dict[str, list[str]] = {}
    for media_type, refs in detectors.items():
        refs = [refs] if isinstance(refs, str) else refs
        if not isinstance(refs, list):
            raise ArtifactError(f"{path}: detectors.{media_type} must be a list of artifacts")
        config[str(media_type)] = [_resolve_relative(str(ref), path.parent) for ref in refs]
    return config


def _resolve_relative(ref: str, base: Path) -> str:
    candidate = Path(ref).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    local = base / candidate
    if ref.startswith(".") or local.exists():
        return str(local.resolve())
    return ref


def _as_list(refs: ArtifactRef | Sequence[ArtifactRef]) -> list[ArtifactRef]:
    if isinstance(refs, str | Path | Artifact):
        return [refs]
    return list(refs)


def _load(ref: ArtifactRef) -> Artifact:
    return ref if isinstance(ref, Artifact) else load_artifact(ref)
