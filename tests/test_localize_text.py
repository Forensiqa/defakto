# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pytest

from defakto.dynamics import DynamicsScore
from defakto.localize import sentence_segments, split_sentences, token_segments
from defakto.trajectory import Step, Trajectory

TEXT = "Hello world. Bad words here."


def _trajectory(text: str) -> Trajectory:
    steps = [Step(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    return Trajectory("text", np.zeros((len(steps), 2)), steps, span_unit="char")


def _dynamics(negative: list[float], positive: list[float]) -> DynamicsScore:
    neg, pos = np.array(negative), np.array(positive)
    return DynamicsScore(
        score=float(np.linalg.norm(neg) - np.linalg.norm(pos)),
        residual_negative=float(np.linalg.norm(neg)),
        residual_positive=float(np.linalg.norm(pos)),
        step_scores=neg - pos,
        step_residual_negative=neg,
        step_residual_positive=pos,
    )


# Step residuals for "world." "Bad" "words" "here." (the first token has none).
# Step scores (neg - pos): 1, 2, 3, -1.
DYNAMICS = _dynamics([2.0, 3.0, 4.0, 1.0], [1.0, 1.0, 1.0, 2.0])


def test_split_sentences():
    assert [s for _, _, s in split_sentences(TEXT)] == ["Hello world.", " Bad words here."]


def test_first_token_has_no_score():
    segments = token_segments(_trajectory(TEXT), DYNAMICS, threshold=1.5)
    assert segments[0].text == "Hello" and segments[0].score is None
    assert [s.score for s in segments[1:]] == [1.0, 2.0, 3.0, -1.0]
    assert [s.flagged for s in segments[1:]] == [False, True, True, False]


def test_sentence_score_uses_the_overall_formula():
    segments = sentence_segments(TEXT, _trajectory(TEXT), DYNAMICS, threshold=2.0)
    first, second = segments
    # "Hello world.": sqrt(2²) - sqrt(1²)
    assert (first.text, first.score, first.flagged) == ("Hello world.", 1.0, False)
    # "Bad words here.": sqrt(3² + 4² + 1²) - sqrt(1² + 1² + 2²)
    assert second.text == "Bad words here."
    assert second.score == pytest.approx(np.sqrt(26) - np.sqrt(6))
    assert second.flagged is True


def test_single_sentence_gets_the_overall_score():
    text = "Bad words here."
    dynamics = _dynamics([3.0, 4.0], [1.0, 2.0])
    (segment,) = sentence_segments(text, _trajectory(text), dynamics, threshold=0.0)
    assert segment.score == pytest.approx(dynamics.score)


def test_sentence_without_scored_tokens_has_no_score():
    text = "Hi. There you go."
    dynamics = _dynamics([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    first, second = sentence_segments(text, _trajectory(text), dynamics, threshold=0.0)
    assert first.text == "Hi." and first.score is None and first.flagged is None
    assert second.score == pytest.approx(np.sqrt(3) - np.sqrt(12))
