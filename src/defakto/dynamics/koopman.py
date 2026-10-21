# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from defakto.dynamics.lifting import quadratic_lift, quadratic_lift_indices


@dataclass(frozen=True)
class DynamicsScore:
    """How well each operator predicts a trajectory, overall and step by step.

    ``score`` is ``residual_negative - residual_positive``: positive values mean the
    trajectory behaves more like the positive (flagged) class.

    ``step_scores[i]`` describes the transition from step ``i`` to step ``i + 1``, so it
    belongs to step ``i + 1``. The first step has no score. The overall residuals are the
    norms of the step residuals combined: ``residual = sqrt(sum(step_residual ** 2))``.
    """

    score: float
    residual_negative: float
    residual_positive: float
    step_scores: np.ndarray
    step_residual_negative: np.ndarray
    step_residual_positive: np.ndarray


class KoopmanModel:
    """A two-operator Koopman model over embedding trajectories.

    Each embedding is projected onto ``observable_modes`` (the observed state) and onto
    ``lift_modes``, whose pairwise products add a quadratic lift. Two linear operators,
    fitted on negative and on positive examples, each predict the next observed state
    from the current lifted state. Whichever predicts a trajectory better tells you
    which class it behaves like.

    Nothing here depends on the media type: text, audio and video all reduce to an
    embedding trajectory first.
    """

    def __init__(
        self,
        lift_modes: np.ndarray,
        observable_modes: np.ndarray,
        operator_negative: np.ndarray,
        operator_positive: np.ndarray,
    ) -> None:
        arrays = {
            "lift_modes": np.asarray(lift_modes, dtype=np.float64),
            "observable_modes": np.asarray(observable_modes, dtype=np.float64),
            "operator_negative": np.asarray(operator_negative, dtype=np.float64),
            "operator_positive": np.asarray(operator_positive, dtype=np.float64),
        }
        for name, array in arrays.items():
            if array.ndim != 2:
                raise ValueError(f"{name} must be 2-D, got shape {array.shape}")

        embedding_dim, n_lift = arrays["lift_modes"].shape
        if arrays["observable_modes"].shape[0] != embedding_dim:
            raise ValueError(
                f"lift_modes and observable_modes disagree on the embedding size: "
                f"{embedding_dim} vs {arrays['observable_modes'].shape[0]}"
            )
        n_observable = arrays["observable_modes"].shape[1]
        lifted_dim = n_observable + n_lift * (n_lift + 1) // 2
        expected_shape = (n_observable, lifted_dim)
        for name in ("operator_negative", "operator_positive"):
            if arrays[name].shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {arrays[name].shape}, expected {expected_shape}"
                )

        self.lift_modes = arrays["lift_modes"]
        self.observable_modes = arrays["observable_modes"]
        self.operator_negative = arrays["operator_negative"]
        self.operator_positive = arrays["operator_positive"]
        self.embedding_dim = int(embedding_dim)
        self.n_lift_modes = int(n_lift)
        self.n_observable_modes = int(n_observable)
        self.lifted_dim = int(lifted_dim)
        self._rows, self._cols = quadratic_lift_indices(self.n_lift_modes)

    def score(self, embeddings: np.ndarray) -> DynamicsScore:
        """Score a ``(T, embedding_dim)`` trajectory with ``T >= 2``."""
        E = np.asarray(embeddings, dtype=np.float64)
        if E.ndim != 2 or E.shape[1] != self.embedding_dim:
            raise ValueError(
                f"expected embeddings of shape (T, {self.embedding_dim}), got {E.shape}"
            )
        if E.shape[0] < 2:
            raise ValueError(f"need at least 2 steps to score a trajectory, got {E.shape[0]}")

        Z = E @ self.observable_modes
        lift = quadratic_lift(E @ self.lift_modes, self._rows, self._cols)
        X = np.concatenate([Z, lift], axis=1)

        X_t, Z_next = X[:-1], Z[1:]
        diff_negative = X_t @ self.operator_negative.T - Z_next
        diff_positive = X_t @ self.operator_positive.T - Z_next

        residual_negative = float(np.linalg.norm(diff_negative))
        residual_positive = float(np.linalg.norm(diff_positive))
        step_negative = np.linalg.norm(diff_negative, axis=1)
        step_positive = np.linalg.norm(diff_positive, axis=1)

        return DynamicsScore(
            score=residual_negative - residual_positive,
            residual_negative=residual_negative,
            residual_positive=residual_positive,
            step_scores=step_negative - step_positive,
            step_residual_negative=step_negative,
            step_residual_positive=step_positive,
        )
