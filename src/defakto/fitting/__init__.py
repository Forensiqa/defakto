# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

"""Fitting: learn a Koopman artifact from labeled examples.

``fit_koopman`` and the encoders need PyTorch: ``pip install 'defakto[fit]'``.
"""

from defakto.fitting.calibrate import Metrics, choose_threshold, evaluate, score_trajectories
from defakto.fitting.pipeline import FitResult, fit
from defakto.fitting.solver import FittingError, LabeledTrajectory, fit_koopman
from defakto.fitting.store import TrajectoryStore

__all__ = [
    "FitResult",
    "FittingError",
    "LabeledTrajectory",
    "Metrics",
    "TrajectoryStore",
    "choose_threshold",
    "evaluate",
    "fit",
    "fit_koopman",
    "score_trajectories",
]
