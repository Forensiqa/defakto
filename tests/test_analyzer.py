# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from defakto import Analyzer, analyze, load_detector_config
from defakto.errors import ArtifactError, UnsupportedMediaError

TEXT = "This is a short test sentence. It has two sentences in total."


def test_artifacts_sharing_an_encoder_encode_once(make_artifact, word_encoder):
    first = make_artifact("a", artifact_id="forensiqa/test-a", seed=1)
    second = make_artifact("b", artifact_id="forensiqa/test-b", seed=2)
    analyzer = Analyzer([first, second])

    before = word_encoder.encode_calls
    results = analyzer.analyze(TEXT)
    assert word_encoder.encode_calls - before == 1
    assert [r.artifact_id for r in results] == ["forensiqa/test-a", "forensiqa/test-b"]
    assert results[0].score != results[1].score


def test_analyzes_a_text_file(make_artifact, tmp_path):
    path = tmp_path / "statement.txt"
    path.write_text(TEXT)
    (result,) = analyze(path, [make_artifact()])
    assert result.source == str(path)
    assert result.num_steps == len(TEXT.split())


def test_artifact_listed_under_the_wrong_media_type(make_artifact):
    with pytest.raises(ArtifactError, match="listed under audio"):
        Analyzer({"audio": [make_artifact()]})


def test_media_type_without_artifacts(make_artifact):
    analyzer = Analyzer([make_artifact(media_type="audio")])
    with pytest.raises(UnsupportedMediaError, match="No artifacts are configured for text"):
        analyzer.analyze(TEXT)


def test_media_type_not_supported_yet(make_artifact, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    with pytest.raises(UnsupportedMediaError, match="Video analysis isn't available yet"):
        Analyzer([make_artifact()]).analyze(clip)


def test_missing_file(make_artifact, tmp_path):
    with pytest.raises(FileNotFoundError):
        Analyzer([make_artifact()]).analyze(tmp_path / "missing.txt")
    with pytest.raises(FileNotFoundError):
        Analyzer([make_artifact()]).analyze("missing.mp4")


def test_config_resolves_relative_paths(make_artifact, tmp_path):
    folder = make_artifact("artifacts/safety")
    config = tmp_path / "defakto.yaml"
    config.write_text("detectors:\n  text:\n    - ./artifacts/safety\n")
    assert load_detector_config(config) == {"text": [str(folder.resolve())]}


def test_config_keeps_hub_ids(tmp_path):
    config = tmp_path / "defakto.yaml"
    config.write_text("detectors:\n  text: forensiqa/text-safety-jina-v2-base\n")
    assert load_detector_config(config) == {"text": ["forensiqa/text-safety-jina-v2-base"]}


def test_config_needs_a_detectors_section(tmp_path):
    config = Path(tmp_path / "defakto.yaml")
    config.write_text("something: else\n")
    with pytest.raises(ArtifactError, match="'detectors' section"):
        load_detector_config(config)
