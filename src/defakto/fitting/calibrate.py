# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from defakto.dynamics import KoopmanModel
from defakto.fitting.solver import FittingError, LabeledTrajectory

LENGTH_BUCKETS = (1, 50, 100, 150)
"""Minimum lengths (in steps) that metrics are also reported for."""


@dataclass(frozen=True)
class Metrics:
    """How well a threshold separates the two classes.

    The positive class is the flagged one, so ``tpr`` is the share of positive examples
    flagged and ``fpr`` the share of negative examples flagged. ``auc`` doesn't depend on
    the threshold. ``min_steps`` is the shortest length counted (1 means all).
    """

    min_steps: int
    count: int
    positives: int
    negatives: int
    accuracy: float
    balanced_accuracy: float
    tpr: float
    fpr: float
    auc: float

    def to_dict(self) -> dict[str, float | int]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def score_trajectories(
    model: KoopmanModel, trajectories: Iterable[LabeledTrajectory]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every trajectory with 2 or more steps, as ``Detector`` would.

    Returns ``(scores, labels, lengths)``; shorter trajectories are skipped.
    """
    scores, labels, lengths = [], [], []
    for embeddings, positive in trajectories:
        if len(embeddings) < 2:
            continue
        scores.append(model.score(embeddings).score)
        labels.append(bool(positive))
        lengths.append(len(embeddings))
    return np.asarray(scores), np.asarray(labels, dtype=bool), np.asarray(lengths, dtype=np.int64)


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """The threshold with the best balanced accuracy, flagging ``score > threshold``.

    Balanced accuracy weighs both classes equally, so an imbalanced validation set doesn't
    push the threshold towards the majority class. The threshold sits halfway between
    two neighbouring scores, so it doesn't depend on how finely a grid is drawn.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise FittingError("calibration needs examples of both classes")

    order = np.argsort(scores, kind="stable")
    s, y = scores[order], labels[order]
    # Cut k flags everything from s[k] up: k = 0 flags everything, k = n flags nothing.
    pos_below = np.concatenate([[0], np.cumsum(y)])
    neg_below = np.arange(len(s) + 1) - pos_below
    balanced = ((n_pos - pos_below) / n_pos + neg_below / n_neg) / 2
    # Cuts between two equal scores can't be realised by any threshold.
    valid = np.ones(len(s) + 1, dtype=bool)
    valid[1:-1] = s[1:] > s[:-1]
    k = int(np.argmax(np.where(valid, balanced, -np.inf)))

    if k == 0:
        return float(s[0] - 1.0)
    if k == len(s):
        return float(s[-1] + 1.0)
    return float((s[k - 1] + s[k]) / 2)


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
    *,
    buckets: Sequence[int] = LENGTH_BUCKETS,
) -> list[Metrics]:
    """Metrics at ``threshold``, for all examples and for each minimum length in ``buckets``.

    Buckets with no example of one of the classes are left out.
    """
    out = []
    for min_steps in buckets:
        keep = lengths >= min_steps
        s, y = scores[keep], labels[keep]
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        flagged = s > threshold
        tpr = float((flagged & y).sum() / n_pos)
        fpr = float((flagged & ~y).sum() / n_neg)
        out.append(
            Metrics(
                min_steps=int(min_steps),
                count=len(s),
                positives=n_pos,
                negatives=n_neg,
                accuracy=float((flagged == y).mean()),
                balanced_accuracy=(tpr + 1 - fpr) / 2,
                tpr=tpr,
                fpr=fpr,
                auc=_auc(s, y),
            )
        )
    return out


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve: the chance a positive example outscores a negative one."""
    ranks = _average_ranks(scores)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, with ties sharing their average rank."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            ranks[order[start:end]] = (start + end + 1) / 2
            start = end
    return ranks
