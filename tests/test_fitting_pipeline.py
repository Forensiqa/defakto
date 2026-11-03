# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

import defakto
from defakto.artifacts import load_artifact
from defakto.cli import app
from defakto.encoders import build_encoder
from defakto.fitting import FittingError, TrajectoryStore
from defakto.fitting.config import DataConfig, FitConfig

pytest.importorskip("torch")

ENCODER = {"type": "test-words", "dim": 6}
SAFE_WORDS = "the weather is calm and the garden is green today".split()
UNSAFE_WORDS = "attack weapon harm steal poison threat danger break".split()


def make_examples(seed: int, count: int = 60) -> list[tuple[str, bool]]:
    """Sentences drawn from two vocabularies, so the classes have different trajectories."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        positive = i % 3 == 0
        words = UNSAFE_WORDS if positive else SAFE_WORDS
        out.append((" ".join(rng.choice(words, size=int(rng.integers(5, 15)))), positive))
    return out


def test_store_round_trips_trajectories(tmp_path):
    trajectories = [(np.arange(12.0).reshape(4, 3), True), (np.ones((2, 3)), False)]
    store = TrajectoryStore.write(tmp_path / "store", trajectories, meta={"note": "x"})

    reopened = TrajectoryStore(tmp_path / "store")
    assert len(reopened) == 2
    assert reopened.meta["note"] == "x"
    assert list(reopened.lengths) == [4, 2]
    assert (reopened.num_positive, reopened.num_negative) == (1, 1)
    for (expected, label), (actual, actual_label) in zip(trajectories, store, strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert actual_label == label


def test_unfinished_store_is_not_reused(tmp_path):
    TrajectoryStore.write(tmp_path / "s", [(np.ones((2, 3)), True)], meta={"encoder": "a"})
    assert TrajectoryStore.open_if_current(tmp_path / "s", {"encoder": "a"}) is not None
    assert TrajectoryStore.open_if_current(tmp_path / "s", {"encoder": "b"}) is None
    (tmp_path / "s" / "meta.json").unlink()
    assert TrajectoryStore.open_if_current(tmp_path / "s", {"encoder": "a"}) is None
    with pytest.raises(FittingError, match="not a finished trajectory store"):
        TrajectoryStore(tmp_path / "s")


def test_fit_writes_an_artifact_that_detects(tmp_path):
    result = defakto.fit(
        make_examples(0),
        encoder=ENCODER,
        output=tmp_path / "artifact",
        artifact_id="me/test-fitted",
        validation=make_examples(1),
        test=make_examples(2),
        positive_label="unsafe",
        negative_label="safe",
        task="safety",
        lift_modes=2,
        observable_modes=4,
        min_steps=3,
    )

    manifest = load_artifact(tmp_path / "artifact").manifest
    assert manifest.id == "me/test-fitted"
    assert manifest.threshold == result.threshold
    assert (manifest.lift_modes, manifest.observable_modes, manifest.min_steps) == (2, 4, 3)
    training = manifest.raw["training"]
    assert training["examples"]["train"] == {"positive": 20, "negative": 40}
    assert training["validation"][0]["count"] == 60
    assert result.test[0].auc > 0.9

    detector = defakto.Detector.from_artifact(tmp_path / "artifact")
    assert detector.analyze("weapon attack poison threat harm").flagged
    assert not detector.analyze("the garden is calm and green today").flagged


def test_fit_accepts_already_encoded_stores(tmp_path):
    encoder = build_encoder(defakto.artifacts.EncoderSpec.from_dict(ENCODER))
    train = TrajectoryStore.encode(tmp_path / "train", encoder, make_examples(0))
    from_store = defakto.fit(
        train,
        encoder=ENCODER,
        output=tmp_path / "a",
        artifact_id="me/a",
        lift_modes=2,
        observable_modes=4,
    )
    from_examples = defakto.fit(
        make_examples(0),
        encoder=ENCODER,
        output=tmp_path / "b",
        artifact_id="me/b",
        lift_modes=2,
        observable_modes=4,
    )
    np.testing.assert_allclose(
        from_store.artifact.model.operator_positive, from_examples.artifact.model.operator_positive
    )
    assert from_store.threshold == 0.0
    assert from_store.artifact.manifest.media_type == "text"


def fit_config(**data_overrides) -> dict:
    return {
        "artifact": {
            "id": "me/test-fitted",
            "task": "safety",
            "labels": {"positive": "unsafe", "negative": "safe"},
        },
        "encoder": ENCODER,
        "dimensions": {"lift_modes": 2, "observable_modes": 4},
        "data": {
            "dataset": "someone/some-dataset",
            "text_column": "response",
            "label_column": "response_label",
            "positive": "unsafe",
            "splits": {"train": "train", "validation": "validation"},
            **data_overrides,
        },
    }


def test_config_reads_every_section():
    cfg = FitConfig.from_dict(fit_config(limit=100))
    assert cfg.artifact_id == "me/test-fitted"
    assert (cfg.positive_label, cfg.negative_label) == ("unsafe", "safe")
    assert (cfg.lift_modes, cfg.observable_modes) == (2, 4)
    assert cfg.data.limit == 100
    assert "limit" in cfg.data.source("train") and "limit" not in cfg.data.source("validation")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c["artifact"].pop("id"), "artifact.id"),
        (lambda c: c["data"].pop("positive"), "data.positive"),
        (lambda c: c["data"].update(splits={"validation": "v"}), "'train' split"),
        (lambda c: c["data"]["splits"].update(holdout="h"), "unknown split"),
    ],
)
def test_config_explains_mistakes(change, message):
    config = fit_config()
    change(config)
    with pytest.raises(FittingError, match=message):
        FitConfig.from_dict(config)


def test_fit_command_encodes_once_then_reuses(tmp_path, monkeypatch):
    loaded = []

    def fake_load_examples(data: DataConfig, split: str):
        loaded.append(split)
        return make_examples({"train": 0, "validation": 1}[split])

    monkeypatch.setattr("defakto.cli.load_examples", fake_load_examples)
    monkeypatch.setattr("defakto.cli.pin_revisions", lambda cfg: cfg)
    config = tmp_path / "fit.yaml"
    config.write_text(yaml.safe_dump(fit_config()))
    args = [
        "fit",
        str(config),
        "-o",
        str(tmp_path / "artifact"),
        "--work-dir",
        str(tmp_path / "work"),
    ]

    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "Wrote me/test-fitted" in first.output
    assert "validation" in first.output and "AUC" in first.output
    assert loaded == ["train", "validation"]

    refused = CliRunner().invoke(app, args)
    assert refused.exit_code == 1
    assert "--force" in refused.output

    second = CliRunner().invoke(app, [*args, "--force"])
    assert second.exit_code == 0, second.output
    assert "Reusing encoded train examples" in second.output
    assert loaded == ["train", "validation"]
