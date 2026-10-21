# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

"""Artifacts: fitted models stored as weights.safetensors plus a manifest.yaml."""

from defakto.artifacts.loader import Artifact, load_artifact, resolve_artifact_dir
from defakto.artifacts.manifest import EncoderSpec, Manifest

__all__ = ["Artifact", "EncoderSpec", "Manifest", "load_artifact", "resolve_artifact_dir"]
