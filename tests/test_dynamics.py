# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from defakto.dynamics import KoopmanModel, quadratic_lift, quadratic_lift_indices


def _model(seed: int = 1, emb_dim: int = 5, n_lift: int = 2, n_obs: int = 3) -> KoopmanModel:
    rng = np.random.default_rng(seed)
    observable = rng.normal(size=(emb_dim, n_obs))
    lifted = n_obs + n_lift * (n_lift + 1) // 2
    return KoopmanModel(
        lift_modes=observable[:, :n_lift],
        observable_modes=observable,
        operator_negative=rng.normal(size=(n_obs, lifted)),
        operator_positive=rng.normal(size=(n_obs, lifted)),
    )


def test_quadratic_lift_is_all_pairwise_products():
    a = np.array([[1.0, 2.0, 3.0]])
    rows, cols = quadratic_lift_indices(3)
    # pairs (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
    assert quadratic_lift(a, rows, cols).tolist() == [[1.0, 2.0, 4.0, 3.0, 6.0, 9.0]]


def test_score_matches_reference_formula():
    model = _model()
    E = np.random.default_rng(7).normal(size=(9, model.embedding_dim))

    a = E @ model.lift_modes
    quad = np.stack([a[:, i] * a[:, j] for i in range(2) for j in range(i + 1)], axis=1)
    Z = E @ model.observable_modes
    X = np.hstack([Z, quad])
    r_neg = np.linalg.norm(X[:-1] @ model.operator_negative.T - Z[1:])
    r_pos = np.linalg.norm(X[:-1] @ model.operator_positive.T - Z[1:])

    s = model.score(E)
    assert s.score == pytest.approx(r_neg - r_pos)
    assert s.residual_negative == pytest.approx(r_neg)
    assert s.residual_positive == pytest.approx(r_pos)


def test_step_residuals_combine_into_the_totals():
    model = _model()
    s = model.score(np.random.default_rng(3).normal(size=(12, model.embedding_dim)))
    assert len(s.step_scores) == 11
    assert np.sqrt(np.sum(s.step_residual_negative**2)) == pytest.approx(s.residual_negative)
    assert np.allclose(s.step_scores, s.step_residual_negative - s.step_residual_positive)


def test_rejects_wrong_embedding_size():
    model = _model()
    with pytest.raises(ValueError, match="expected embeddings of shape"):
        model.score(np.zeros((4, model.embedding_dim + 1)))


def test_rejects_single_step():
    model = _model()
    with pytest.raises(ValueError, match="at least 2 steps"):
        model.score(np.zeros((1, model.embedding_dim)))


def test_rejects_inconsistent_operator_shape():
    observable = np.zeros((5, 3))
    with pytest.raises(ValueError, match="operator_negative has shape"):
        KoopmanModel(observable[:, :2], observable, np.zeros((3, 5)), np.zeros((3, 6)))
