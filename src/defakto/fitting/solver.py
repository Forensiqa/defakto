# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from defakto.dynamics import KoopmanModel, quadratic_lift_indices
from defakto.errors import DefaktoError

if TYPE_CHECKING:
    import torch

LabeledTrajectory = tuple[np.ndarray, bool]
"""A ``(T, embedding_dim)`` trajectory and whether it belongs to the positive (flagged) class."""

_INSTALL_HINT = "Install it with: pip install 'defakto[fit]'"


class FittingError(DefaktoError):
    """The training data can't produce a model, or fitting can't run."""


def fit_koopman(
    trajectories: Iterable[LabeledTrajectory],
    *,
    lift_modes: int = 40,
    observable_modes: int = 500,
    device: str | None = None,
    chunk_rows: int = 8192,
) -> KoopmanModel:
    """Fit a two-operator Koopman model from labeled embedding trajectories.

    1. Every transition ``x_t -> x_{t+1}`` of every trajectory is a training pair.
    2. One SVD of the current states ``x_t`` of both classes gives the basis: its first
       ``lift_modes`` left singular vectors are the lift modes, its first
       ``observable_modes`` the observable modes.
    3. Each state is lifted to ``[observable_modes.T @ x, quadratic_lift(lift_modes.T @ x)]``.
    4. For each class, DMD regresses the next observed state on the current lifted state

    Neither step holds all the data at once: both stream the rows through a QR
    factorization, so memory depends on the dimensions, not on the number of tokens.
    ``trajectories`` is read twice, so it must be a collection (a list, or a
    ``TrajectoryStore``), not a one-shot iterator.
    """
    if iter(trajectories) is trajectories:
        raise TypeError(
            "fit_koopman reads the trajectories twice; pass a list or a TrajectoryStore, "
            "not an iterator"
        )
    if lift_modes < 1 or observable_modes < 1:
        raise ValueError("lift_modes and observable_modes must be at least 1")
    torch = _import_torch()
    dev = torch.device(device or "cpu")

    # Pass 1: SVD basis of the current states x_t of both classes.
    states = _StreamingQR(chunk_rows)
    for embeddings, _ in trajectories:
        if len(embeddings) >= 2:
            states.add(_as_tensor(embeddings[:-1], dev))
    if states.n_rows == 0:
        raise FittingError("no trajectory has 2 or more steps, so there is nothing to fit")
    embedding_dim = states.n_cols
    if observable_modes > embedding_dim or lift_modes > embedding_dim:
        raise FittingError(
            f"asked for {observable_modes} observable and {lift_modes} lift modes, but the "
            f"embeddings have only {embedding_dim} dimensions"
        )
    # The R factor has the same right singular vectors as the stacked states, which are
    # the left singular vectors of the (embedding_dim x tokens) data matrix.
    _, _, vh = torch.linalg.svd(states.finish(), full_matrices=False)
    if vh.shape[0] < max(lift_modes, observable_modes):
        raise FittingError(
            f"only {vh.shape[0]} training states, too few for {observable_modes} observable modes"
        )
    basis = vh.T
    lift_basis = basis[:, :lift_modes].contiguous()
    observable_basis = basis[:, :observable_modes].contiguous()

    # Pass 2: one least-squares problem per class, X_t -> Z_{t+1}.
    rows, cols = (torch.as_tensor(i, device=dev) for i in quadratic_lift_indices(lift_modes))
    lifted_dim = observable_modes + len(rows)
    systems = {
        False: _StreamingQR(chunk_rows),
        True: _StreamingQR(chunk_rows),
    }
    for embeddings, positive in trajectories:
        if len(embeddings) < 2:
            continue
        e = _as_tensor(embeddings, dev)
        z = e @ observable_basis
        a = e @ lift_basis
        lifted = torch.cat([z, a[:, rows] * a[:, cols]], dim=1)
        systems[bool(positive)].add(torch.cat([lifted[:-1], z[1:]], dim=1))

    operators = {}
    for positive, system in systems.items():
        name = "positive" if positive else "negative"
        if system.n_rows == 0:
            raise FittingError(f"no {name} trajectory has 2 or more steps")
        r = system.finish()
        operators[positive] = _min_norm_solve(
            r[:, :lifted_dim], r[:, lifted_dim:], max(lifted_dim, system.n_rows)
        ).T

    return KoopmanModel(
        lift_modes=lift_basis.cpu().numpy(),
        observable_modes=observable_basis.cpu().numpy(),
        operator_negative=operators[False].cpu().numpy(),
        operator_positive=operators[True].cpu().numpy(),
    )


class _StreamingQR:
    """The R factor of a tall matrix whose rows arrive in batches.

    ``R.T @ R`` equals the Gram matrix of all rows seen, but R is computed by QR rather
    than by squaring, so it keeps the conditioning of the original rows.
    """

    def __init__(self, chunk_rows: int) -> None:
        self._chunk_rows = chunk_rows
        self._pending: list[torch.Tensor] = []
        self._pending_rows = 0
        self._r: torch.Tensor | None = None
        self.n_rows = 0
        self.n_cols = 0

    def add(self, rows: "torch.Tensor") -> None:
        if self.n_cols and rows.shape[1] != self.n_cols:
            raise FittingError(
                f"trajectories disagree on the embedding size: {rows.shape[1]} vs {self.n_cols}"
            )
        self.n_cols = int(rows.shape[1])
        self._pending.append(rows)
        self._pending_rows += int(rows.shape[0])
        self.n_rows += int(rows.shape[0])
        if self._pending_rows >= self._chunk_rows:
            self._flush()

    def finish(self) -> "torch.Tensor":
        self._flush()
        if self._r is None:
            raise FittingError("no rows were added")
        return self._r

    def _flush(self) -> None:
        import torch

        if not self._pending:
            return
        blocks = self._pending if self._r is None else [self._r, *self._pending]
        self._r = torch.linalg.qr(torch.cat(blocks), mode="r").R
        self._pending, self._pending_rows = [], 0


def _min_norm_solve(a: "torch.Tensor", b: "torch.Tensor", size: int) -> "torch.Tensor":
    """``pinv(a) @ b``, dropping singular values below ``size * eps(largest singular value)``.

    ``size`` is the larger dimension of the original data matrix.
    """
    import torch

    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    if s.numel() == 0:
        return torch.zeros((a.shape[1], b.shape[1]), dtype=a.dtype, device=a.device)
    tol = size * np.spacing(float(s[0]))
    keep = s > tol
    return vh[keep].T @ ((u[:, keep].T @ b) / s[keep, None])


def _as_tensor(array: np.ndarray, device: "torch.device") -> "torch.Tensor":
    import torch

    # Stored trajectories are read-only memory maps, which torch won't wrap.
    a = np.asarray(array, dtype=np.float64)
    if not a.flags.writeable:
        a = a.copy()
    return torch.as_tensor(a, device=device)


def _import_torch():
    """Import torch, or explain how to install it. Later ``import torch`` lines are free."""
    try:
        import torch
    except ImportError as e:
        raise FittingError(f"Fitting needs PyTorch. {_INSTALL_HINT}") from e
    return torch
