# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0


class DefaktoError(Exception):
    """Base class for errors raised by defakto."""


class ArtifactError(DefaktoError):
    """An artifact is missing, malformed, or doesn't match what it describes."""


class EncoderError(DefaktoError):
    """An encoder can't be built or doesn't fit the artifact it's paired with."""


class UnsupportedMediaError(DefaktoError):
    """The input's media type isn't supported, or no artifact is configured for it."""
