# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from defakto.dynamics import KoopmanModel
from defakto.fitting import FittingError, fit_koopman, score_trajectories
from defakto.fitting.calibrate import evaluate

pytest.importorskip("torch")

DIM, N_LIFT, N_OBS = 8, 3, 5


def make_trajectories(seed: int, count: int = 40, dim: int = DIM) -> list[tuple[np.ndarray, bool]]:
    """Two classes with different linear dynamics, plus noise."""
    rng = np.random.default_rng(seed)
    dynamics = {
        False: np.linalg.qr(np.random.default_rng(1).normal(size=(dim, dim)))[0] * 0.9,
        True: np.linalg.qr(np.random.default_rng(2).normal(size=(dim, dim)))[0] * 0.9,
    }
    out = []
    for i in range(count):
        positive = i % 2 == 1
        x = rng.normal(size=dim)
        steps = [x]
        for _ in range(int(rng.integers(6, 14))):
            x = dynamics[positive] @ x + rng.normal(scale=0.05, size=dim)
            steps.append(x)
        out.append((np.array(steps), positive))
    return out


def reference_fit(trajectories, n_lift: int, n_obs: int) -> KoopmanModel:
    """The method written out directly, holding every transition in memory."""

    def stack(positive: bool, part: slice) -> np.ndarray:
        return np.hstack([e[part].T for e, p in trajectories if p == positive and len(e) >= 2])

    X_neg, Y_neg = stack(False, slice(None, -1)), stack(False, slice(1, None))
    X_pos, Y_pos = stack(True, slice(None, -1)), stack(True, slice(1, None))
    basis = np.linalg.svd(np.hstack([X_neg, X_pos]), full_matrices=False)[0]
    lift_modes, observable_modes = basis[:, :n_lift], basis[:, :n_obs]

    def lifted(X: np.ndarray) -> np.ndarray:
        a = lift_modes.T @ X
        lift = [a[row] * a[col] for row in range(n_lift) for col in range(row + 1)]
        return np.vstack([observable_modes.T @ X, *lift])

    def pinv(X: np.ndarray) -> np.ndarray:
        s_max = np.linalg.svd(X, compute_uv=False)[0]
        return np.linalg.pinv(X, rcond=max(X.shape) * np.spacing(s_max) / s_max)

    A_neg = (lifted(Y_neg) @ pinv(lifted(X_neg)))[:n_obs]
    A_pos = (lifted(Y_pos) @ pinv(lifted(X_pos)))[:n_obs]
    return KoopmanModel(
        lift_modes, observable_modes, operator_negative=A_neg, operator_positive=A_pos
    )


def assert_same_scores(model: KoopmanModel, reference: KoopmanModel, trajectories) -> None:
    ours = score_trajectories(model, trajectories)[0]
    theirs = score_trajectories(reference, trajectories)[0]
    np.testing.assert_allclose(ours, theirs, rtol=1e-7, atol=1e-9)


def test_matches_the_reference_fit():
    train, held_out = make_trajectories(seed=0), make_trajectories(seed=1)
    model = fit_koopman(train, lift_modes=N_LIFT, observable_modes=N_OBS)
    reference = reference_fit(train, N_LIFT, N_OBS)

    assert model.operator_negative.shape == (N_OBS, N_OBS + N_LIFT * (N_LIFT + 1) // 2)
    # SVD signs are arbitrary, so compare the basis up to sign, and the scores exactly.
    overlap = np.abs(np.sum(model.observable_modes * reference.observable_modes, axis=0))
    np.testing.assert_allclose(overlap, 1.0, atol=1e-8)
    assert_same_scores(model, reference, held_out)


def test_matches_the_reference_when_the_data_is_rank_deficient():
    # Embeddings that only use 4 of 8 dimensions, with 6 observable modes: two modes carry
    # no data, so pinv has to drop singular values.
    basis = np.linalg.qr(np.random.default_rng(3).normal(size=(DIM, 4)))[0]
    train = [(e @ basis @ basis.T, p) for e, p in make_trajectories(seed=0)]
    held_out = [(e @ basis @ basis.T, p) for e, p in make_trajectories(seed=1)]
    model = fit_koopman(train, lift_modes=N_LIFT, observable_modes=6)
    assert_same_scores(model, reference_fit(train, N_LIFT, 6), held_out)


def test_streaming_does_not_change_the_result():
    train, held_out = make_trajectories(seed=0), make_trajectories(seed=1)
    one_chunk = fit_koopman(train, lift_modes=N_LIFT, observable_modes=N_OBS, chunk_rows=10**6)
    small_chunks = fit_koopman(train, lift_modes=N_LIFT, observable_modes=N_OBS, chunk_rows=7)
    assert_same_scores(small_chunks, one_chunk, held_out)


def test_fitted_model_separates_the_classes():
    model = fit_koopman(make_trajectories(seed=0), lift_modes=N_LIFT, observable_modes=N_OBS)
    scores, labels, lengths = score_trajectories(model, make_trajectories(seed=1))
    assert evaluate(scores, labels, lengths, threshold=0.0)[0].auc > 0.95


def test_skips_trajectories_too_short_to_have_a_transition():
    train = make_trajectories(seed=0)
    padded = [*train, (np.zeros((1, DIM)), True), (np.zeros((0, DIM)), False)]
    held_out = make_trajectories(seed=1)
    assert_same_scores(
        fit_koopman(padded, lift_modes=N_LIFT, observable_modes=N_OBS),
        fit_koopman(train, lift_modes=N_LIFT, observable_modes=N_OBS),
        held_out,
    )


def test_needs_both_classes():
    only_negative = [(e, False) for e, _ in make_trajectories(seed=0)]
    with pytest.raises(FittingError, match="no positive trajectory"):
        fit_koopman(only_negative, lift_modes=N_LIFT, observable_modes=N_OBS)


def test_rejects_more_modes_than_dimensions():
    with pytest.raises(FittingError, match="only 8 dimensions"):
        fit_koopman(make_trajectories(seed=0), lift_modes=N_LIFT, observable_modes=9)


def test_rejects_a_one_shot_iterator():
    with pytest.raises(TypeError, match="reads the trajectories twice"):
        fit_koopman(iter(make_trajectories(seed=0)), lift_modes=N_LIFT, observable_modes=N_OBS)
