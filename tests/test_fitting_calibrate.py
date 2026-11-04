# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from defakto.fitting import FittingError, choose_threshold, evaluate


def balanced_accuracy(scores, labels, threshold):
    flagged = scores > threshold
    return (flagged[labels].mean() + (~flagged[~labels]).mean()) / 2


def test_threshold_separates_separable_scores():
    scores = np.array([-3.0, -2.0, -1.0, 1.0, 2.0])
    labels = np.array([False, False, False, True, True])
    assert choose_threshold(scores, labels) == 0.0


def test_threshold_is_as_good_as_any_on_a_fine_grid():
    rng = np.random.default_rng(0)
    labels = rng.random(300) < 0.25
    scores = rng.normal(size=300) + labels * 1.0
    best = choose_threshold(scores, labels)
    grid = max(balanced_accuracy(scores, labels, t) for t in np.linspace(-5, 5, 20001))
    assert balanced_accuracy(scores, labels, best) >= grid - 1e-12


def test_threshold_is_not_pulled_towards_the_majority_class():
    # 90 negatives around 0 and 10 positives around 2: plain accuracy would flag almost
    # nothing, balanced accuracy still finds the gap.
    scores = np.concatenate([np.linspace(-1, 1, 90), np.linspace(1.5, 2.5, 10)])
    labels = np.array([False] * 90 + [True] * 10)
    assert 1.0 < choose_threshold(scores, labels) < 1.5


def test_threshold_never_splits_tied_scores():
    scores = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
    labels = np.array([False, False, True, True, True])
    threshold = choose_threshold(scores, labels)
    assert threshold != 1.0
    assert balanced_accuracy(scores, labels, threshold) == pytest.approx(0.75)


def test_threshold_needs_both_classes():
    with pytest.raises(FittingError, match="both classes"):
        choose_threshold(np.array([1.0, 2.0]), np.array([True, True]))


def test_evaluate_reports_each_length_bucket():
    scores = np.array([-1.0, 1.0, -1.0, 1.0, 1.0])
    labels = np.array([False, True, False, True, False])
    lengths = np.array([10, 10, 60, 60, 120])
    metrics = evaluate(scores, labels, lengths, threshold=0.0)

    assert [m.min_steps for m in metrics] == [1, 50]  # 100+ has no positive example
    everything = metrics[0]
    assert (everything.count, everything.positives, everything.negatives) == (5, 2, 3)
    assert everything.tpr == 1.0
    assert everything.fpr == pytest.approx(1 / 3)
    assert everything.accuracy == pytest.approx(4 / 5)


def test_auc_counts_ties_as_half():
    scores = np.array([0.0, 1.0, 1.0, 2.0])
    labels = np.array([False, False, True, True])
    # Pairs (positive, negative): 1>0, 1=1, 2>0, 2>1 -> (1 + 0.5 + 1 + 1) / 4
    assert evaluate(scores, labels, np.full(4, 10), 0.5)[0].auc == pytest.approx(0.875)
