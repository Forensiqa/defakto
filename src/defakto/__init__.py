# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

"""Defakto: an open-source toolkit for detecting AI-generated and unreliable content.

One method for every media type: an encoder turns the input into a trajectory of
embeddings, and a Koopman dynamics model scores how that trajectory evolves. Artifacts
hold the fitted models, one per media type and task.

Developed by Forensiqa.
"""

from importlib.metadata import PackageNotFoundError, version

from defakto.analyzer import Analyzer, analyze, load_detector_config
from defakto.artifacts import Artifact, load_artifact
from defakto.detector import Detector
from defakto.results import Result, Segment
from defakto.trajectory import Step, Trajectory

# The version lives only in pyproject.toml; this reads it from the installed package.
try:
    __version__ = version("defakto")
except PackageNotFoundError:  # running from a source checkout that isn't installed
    __version__ = "0.0.0+unknown"
__all__ = [
    "Analyzer",
    "Artifact",
    "Detector",
    "Result",
    "Segment",
    "Step",
    "Trajectory",
    "__version__",
    "analyze",
    "load_artifact",
    "load_detector_config",
]
