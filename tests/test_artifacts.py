# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import pytest

from defakto.artifacts import load_artifact
from defakto.errors import ArtifactError


def test_loads_a_valid_artifact(make_artifact):
    artifact = load_artifact(make_artifact())
    assert artifact.manifest.id == "forensiqa/test-artifact"
    assert artifact.manifest.media_type == "text"
    assert artifact.manifest.encoder.type == "test-words"
    assert artifact.model.embedding_dim == 6
    assert artifact.model.lifted_dim == 4 + 3


def test_rejects_modified_weights(make_artifact):
    bad = {"weights": {"file": "weights.safetensors", "sha256": "0" * 64}}
    folder = make_artifact(manifest_overrides=bad)
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        load_artifact(folder)
    assert load_artifact(folder, verify_checksum=False).manifest.id == "forensiqa/test-artifact"


def test_rejects_dimensions_that_dont_match_the_weights(make_artifact):
    dims = {"dimensions": {"embedding_dim": 99, "lift_modes": 2, "observable_modes": 4}}
    with pytest.raises(ArtifactError, match="embedding_dim=99"):
        load_artifact(make_artifact(manifest_overrides=dims))


def test_rejects_unknown_schema(make_artifact):
    with pytest.raises(ArtifactError, match="unsupported schema"):
        load_artifact(make_artifact(manifest_overrides={"schema": "something-else/v9"}))


def test_rejects_missing_folder(tmp_path):
    with pytest.raises(ArtifactError, match="No artifact found"):
        load_artifact(tmp_path / "missing")


def test_reads_min_tokens_from_older_manifests(make_artifact):
    folder = make_artifact()
    manifest = folder / "manifest.yaml"
    manifest.write_text(manifest.read_text().replace("min_steps: 3", "min_tokens: 5"))
    assert load_artifact(folder).manifest.min_steps == 5
