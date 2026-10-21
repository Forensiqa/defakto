# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def quadratic_lift_indices(n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of every pair ``(i, j)`` with ``j <= i``."""
    rows, cols = np.tril_indices(n_modes)
    return rows, cols


def quadratic_lift(a: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """All pairwise products of the columns of ``a``.

    ``a`` has shape ``(T, n_modes)``; the result has shape ``(T, n_modes * (n_modes + 1) // 2)``.
    """
    return a[:, rows] * a[:, cols]
