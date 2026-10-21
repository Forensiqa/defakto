# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from safetensors.numpy import load_file

from defakto.artifacts.manifest import REQUIRED_TENSORS, Manifest
from defakto.dynamics import KoopmanModel
from defakto.errors import ArtifactError

MANIFEST_FILE = "manifest.yaml"
_HUB_ID = re.compile(r"[\w.-]+/[\w.-]+(@[\w./-]+)?")


@dataclass(frozen=True)
class Artifact:
    """A loaded artifact: its manifest, its model, and the folder it came from."""

    manifest: Manifest
    model: KoopmanModel
    path: Path


def resolve_artifact_dir(ref: str | Path) -> Path:
    """Find the folder for an artifact reference.

    ``ref`` is either a local folder containing ``manifest.yaml``, or a Hugging Face Hub
    id such as ``forensiqa/text-safety-jina-v2-base``, optionally pinned to a revision
    with ``@revision``. Hub artifacts are downloaded once and then cached.
    """
    path = Path(ref).expanduser()
    if path.is_dir():
        return path

    text = str(ref)
    if isinstance(ref, str) and not text.startswith((".", "/", "~")) and _HUB_ID.fullmatch(text):
        repo_id, _, revision = text.partition("@")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ArtifactError(
                f"'{ref}' looks like a Hugging Face Hub id, but huggingface_hub isn't installed"
            ) from e
        try:
            return Path(snapshot_download(repo_id=repo_id, revision=revision or None))
        except Exception as e:
            raise ArtifactError(
                f"Couldn't download artifact '{ref}' from the Hugging Face Hub: {e}"
            ) from e

    raise ArtifactError(
        f"No artifact found at '{ref}'. Pass a folder containing {MANIFEST_FILE}, "
        f"or a Hugging Face Hub id like 'forensiqa/text-safety-jina-v2-base'."
    )


def load_artifact(ref: str | Path, *, verify_checksum: bool = True) -> Artifact:
    """Load an artifact and check it: checksum, required arrays and dimensions."""
    directory = resolve_artifact_dir(ref)
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ArtifactError(f"No {MANIFEST_FILE} in '{directory}'")

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ArtifactError(f"{manifest_path}: expected a YAML mapping")
    manifest = Manifest.from_dict(data, source=str(manifest_path))

    weights_path = directory / manifest.weights_file
    if not weights_path.is_file():
        raise ArtifactError(f"{manifest.id}: weights file '{weights_path}' is missing")
    if verify_checksum and manifest.weights_sha256:
        actual = _sha256(weights_path)
        if actual != manifest.weights_sha256:
            raise ArtifactError(
                f"{manifest.id}: checksum mismatch for '{weights_path.name}' "
                f"(manifest {manifest.weights_sha256[:12]}…, file {actual[:12]}…). "
                f"The file may be corrupted or modified."
            )

    tensors = load_file(str(weights_path))
    missing = [name for name in REQUIRED_TENSORS if name not in tensors]
    if missing:
        raise ArtifactError(f"{manifest.id}: weights are missing {', '.join(missing)}")
    try:
        model = KoopmanModel(
            lift_modes=np.asarray(tensors["lift_modes"]),
            observable_modes=np.asarray(tensors["observable_modes"]),
            operator_negative=np.asarray(tensors["operator_negative"]),
            operator_positive=np.asarray(tensors["operator_positive"]),
        )
    except ValueError as e:
        raise ArtifactError(f"{manifest.id}: {e}") from e

    for field_name, declared, actual in (
        ("embedding_dim", manifest.embedding_dim, model.embedding_dim),
        ("lift_modes", manifest.lift_modes, model.n_lift_modes),
        ("observable_modes", manifest.observable_modes, model.n_observable_modes),
    ):
        if declared != actual:
            raise ArtifactError(
                f"{manifest.id}: manifest says {field_name}={declared}, "
                f"but the weights have {actual}"
            )

    return Artifact(manifest=manifest, model=model, path=directory)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
