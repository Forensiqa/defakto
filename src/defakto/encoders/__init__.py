# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

"""Encoders turn media into embedding trajectories. Each artifact names the one it needs."""

from defakto.encoders.base import Encoder, available_encoders, build_encoder, register_encoder

__all__ = ["Encoder", "available_encoders", "build_encoder", "register_encoder"]
