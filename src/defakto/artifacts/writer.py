# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from safetensors.numpy import save_file

from defakto.artifacts.manifest import LIFTING, METHOD_TYPE, SCHEMA, EncoderSpec
from defakto.dynamics import KoopmanModel

WEIGHTS_FILE = "weights.safetensors"


def save_artifact(
    directory: str | Path,
    model: KoopmanModel,
    *,
    artifact_id: str,
    media_type: str,
    encoder: EncoderSpec,
    positive_label: str,
    negative_label: str,
    task: str = "",
    version: str = "0.1.0",
    threshold: float = 0.0,
    min_steps: int = 2,
    training: dict[str, Any] | None = None,
) -> Path:
    """Write ``model`` as an artifact folder: ``weights.safetensors`` plus ``manifest.yaml``.

    ``training`` is stored as-is in the manifest, to record how the artifact was made.
    The result loads with ``load_artifact`` like any official artifact.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / WEIGHTS_FILE
    save_file(
        {
            "lift_modes": np.ascontiguousarray(model.lift_modes),
            "observable_modes": np.ascontiguousarray(model.observable_modes),
            "operator_negative": np.ascontiguousarray(model.operator_negative),
            "operator_positive": np.ascontiguousarray(model.operator_positive),
        },
        str(weights),
    )

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "id": artifact_id,
        "version": version,
        "media_type": media_type,
        "task": task,
        "labels": {"positive": positive_label, "negative": negative_label},
        "method": {"type": METHOD_TYPE, "lifting": LIFTING},
        "encoder": {"type": encoder.type, "model_name": encoder.model_name, **encoder.options},
        "dimensions": {
            "embedding_dim": model.embedding_dim,
            "lift_modes": model.n_lift_modes,
            "observable_modes": model.n_observable_modes,
        },
        "threshold": float(threshold),
        "min_steps": int(min_steps),
        "weights": {
            "file": WEIGHTS_FILE,
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        },
    }
    if training:
        manifest["training"] = training
    (directory / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return directory
