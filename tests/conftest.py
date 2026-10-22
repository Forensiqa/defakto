# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from safetensors.numpy import save_file

from defakto.artifacts import EncoderSpec
from defakto.encoders import register_encoder
from defakto.trajectory import Step, Trajectory

EMB_DIM, N_LIFT, N_OBS = 6, 2, 4


class WordEncoder:
    """Stand-in for a real text encoder: one fixed pseudo-random embedding per word.

    Fast and deterministic, so tests run without torch or a model download.
    """

    media_type = "text"
    encode_calls = 0

    def __init__(self, spec: EncoderSpec, device: str | None = None) -> None:
        self.dim = int(spec.get("dim", EMB_DIM))

    def encode(self, content: str) -> Trajectory:
        WordEncoder.encode_calls += 1
        steps = [Step(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", content)]
        rows = [
            np.random.default_rng(int(hashlib.sha256(s.label.encode()).hexdigest()[:8], 16)).normal(
                size=self.dim
            )
            for s in steps
        ]
        embeddings = np.array(rows, dtype=np.float64).reshape(len(steps), self.dim)
        return Trajectory("text", embeddings, steps, span_unit="char")


register_encoder("test-words", WordEncoder)


def write_artifact(
    directory: Path,
    *,
    artifact_id: str = "forensiqa/test-artifact",
    media_type: str = "text",
    seed: int = 0,
    threshold: float = 0.0,
    min_steps: int = 3,
    manifest_overrides: dict[str, Any] | None = None,
) -> Path:
    rng = np.random.default_rng(seed)
    observable = rng.normal(size=(EMB_DIM, N_OBS))
    lifted = N_OBS + N_LIFT * (N_LIFT + 1) // 2
    tensors = {
        "lift_modes": np.ascontiguousarray(observable[:, :N_LIFT]),
        "observable_modes": observable,
        "operator_negative": rng.normal(scale=0.1, size=(N_OBS, lifted)),
        "operator_positive": rng.normal(scale=0.1, size=(N_OBS, lifted)),
    }
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "weights.safetensors"
    save_file(tensors, str(weights))

    manifest: dict[str, Any] = {
        "schema": "defakto-artifact/v1",
        "id": artifact_id,
        "version": "0.1.0",
        "media_type": media_type,
        "task": "safety",
        "labels": {"positive": "unsafe", "negative": "safe"},
        "method": {"type": "koopman-two-operator", "lifting": "quadratic"},
        "encoder": {"type": "test-words", "dim": EMB_DIM},
        "dimensions": {"embedding_dim": EMB_DIM, "lift_modes": N_LIFT, "observable_modes": N_OBS},
        "threshold": threshold,
        "min_steps": min_steps,
        "weights": {
            "file": "weights.safetensors",
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        },
    }
    manifest.update(manifest_overrides or {})
    (directory / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return directory


@pytest.fixture
def make_artifact(tmp_path):
    """Write a small, valid artifact under ``tmp_path`` and return its folder."""

    def make(name: str = "artifact", **kwargs: Any) -> Path:
        return write_artifact(tmp_path / name, **kwargs)

    return make


@pytest.fixture
def word_encoder():
    return WordEncoder
