# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import re
from bisect import bisect_right

import numpy as np

from defakto.dynamics import DynamicsScore
from defakto.results import Segment
from defakto.trajectory import Trajectory

# A sentence-ish span: text up to [.!?], or the rest of the line for markdown headers and
# bullets without terminal punctuation. Good enough to group tokens; not a linguistic
# sentence splitter.
_SENTENCE_RE = re.compile(r"[^.!?\n]*[.!?]+(?=\s|$)|[^\n]+")


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, text)`` for each sentence-ish span in ``text``."""
    return [
        (m.start(), m.end(), m.group()) for m in _SENTENCE_RE.finditer(text) if m.group().strip()
    ]


def token_segments(
    trajectory: Trajectory, dynamics: DynamicsScore, threshold: float
) -> list[Segment]:
    """One segment per token. The first token has no score: nothing comes before it."""
    _check_lengths(trajectory, dynamics)
    first = trajectory.steps[0]
    segments = [Segment("token", first.label, first.start, first.end, None, None)]
    for step, score in zip(trajectory.steps[1:], dynamics.step_scores, strict=True):
        flagged = bool(score > threshold)
        segments.append(Segment("token", step.label, step.start, step.end, float(score), flagged))
    return segments


def sentence_segments(
    text: str, trajectory: Trajectory, dynamics: DynamicsScore, threshold: float
) -> list[Segment]:
    """One segment per sentence, scored like the whole input but over its own tokens only.

    A sentence's score is ``sqrt(sum(neg**2)) - sqrt(sum(pos**2))`` over the step
    residuals of its tokens: the same formula as the overall score, so both use the same
    scale and threshold. A single-sentence input gets exactly the overall score.
    Sentences without a scored token have no score.
    """
    _check_lengths(trajectory, dynamics)
    sentences = split_sentences(text)
    if not sentences:
        return []

    span_starts = [start for start, _, _ in sentences]
    squared_negative = np.zeros(len(sentences))
    squared_positive = np.zeros(len(sentences))
    scored_steps = np.zeros(len(sentences), dtype=int)
    for step, residual_negative, residual_positive in zip(
        trajectory.steps[1:],
        dynamics.step_residual_negative,
        dynamics.step_residual_positive,
        strict=True,
    ):
        index = min(max(bisect_right(span_starts, step.start) - 1, 0), len(sentences) - 1)
        squared_negative[index] += residual_negative**2
        squared_positive[index] += residual_positive**2
        scored_steps[index] += 1

    segments: list[Segment] = []
    for (start, end, body), sq_neg, sq_pos, count in zip(
        sentences, squared_negative, squared_positive, scored_steps, strict=True
    ):
        if count == 0:
            segments.append(Segment("sentence", body.strip(), start, end, None, None))
            continue
        score = float(np.sqrt(sq_neg) - np.sqrt(sq_pos))
        segments.append(Segment("sentence", body.strip(), start, end, score, score > threshold))
    return segments


def localize_text(
    content: str | None, trajectory: Trajectory, dynamics: DynamicsScore, threshold: float
) -> list[Segment]:
    """Token segments, plus sentence segments when the original text is available."""
    segments = token_segments(trajectory, dynamics, threshold)
    if content is not None:
        segments += sentence_segments(content, trajectory, dynamics, threshold)
    return segments


def _check_lengths(trajectory: Trajectory, dynamics: DynamicsScore) -> None:
    if len(dynamics.step_scores) != trajectory.num_steps - 1:
        raise ValueError(
            f"expected {trajectory.num_steps - 1} step scores for {trajectory.num_steps} "
            f"steps, got {len(dynamics.step_scores)}"
        )
