# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import pytest

from defakto import Detector
from defakto.errors import DefaktoError

TEXT = "This is a short test sentence. It has two sentences in total."


def test_analyze_returns_a_complete_result(make_artifact):
    result = Detector.from_artifact(make_artifact()).analyze(TEXT)
    assert result.media_type == "text"
    assert result.artifact_id == "forensiqa/test-artifact"
    assert result.num_steps == len(TEXT.split())
    assert result.flagged == (result.score > result.threshold)
    assert result.label == ("unsafe" if result.flagged else "safe")
    assert len(result.segments_of("token")) == len(TEXT.split())
    assert len(result.segments_of("sentence")) == 2


def test_single_sentence_score_matches_overall_score(make_artifact):
    result = Detector.from_artifact(make_artifact()).analyze("I can show you how to do that.")
    (sentence,) = result.segments_of("sentence")
    assert sentence.score == pytest.approx(result.score)
    assert sentence.flagged == result.flagged


def test_threshold_override(make_artifact):
    detector = Detector.from_artifact(make_artifact())
    score = detector.analyze(TEXT).score
    assert detector.analyze(TEXT, threshold=score - 1).flagged
    assert not detector.analyze(TEXT, threshold=score + 1).flagged


def test_warns_on_short_input(make_artifact):
    result = Detector.from_artifact(make_artifact(min_steps=50)).analyze(TEXT)
    assert any("unreliable" in w for w in result.warnings)


def test_rejects_input_that_is_too_short(make_artifact):
    with pytest.raises(ValueError, match="too short"):
        Detector.from_artifact(make_artifact()).analyze("Hi")


def test_rejects_artifact_for_another_media_type(make_artifact):
    with pytest.raises(DefaktoError, match="is for audio"):
        Detector.from_artifact(make_artifact(media_type="audio"))


def test_to_dict_is_json_ready(make_artifact):
    data = Detector.from_artifact(make_artifact()).analyze(TEXT).to_dict()
    assert data["label"] in ("safe", "unsafe")
    assert data.keys() >= {"score", "threshold", "segments", "warnings"}
