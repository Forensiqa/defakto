# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

"""The detection method: Koopman dynamics over embedding trajectories, for any media type."""

from defakto.dynamics.koopman import DynamicsScore, KoopmanModel
from defakto.dynamics.lifting import quadratic_lift, quadratic_lift_indices

__all__ = ["DynamicsScore", "KoopmanModel", "quadratic_lift", "quadratic_lift_indices"]
