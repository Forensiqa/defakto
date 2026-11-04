# Changelog

All notable changes to defakto are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Fitting: train a two-operator Koopman artifact on your own labeled data.
- `defakto.fit()`: fit from `(content, positive)` examples from any source, or from already encoded trajectories. Returns the artifact, its threshold and its metrics.
- `defakto fit` command: fit from a labeled Hugging Face Hub dataset described in a YAML file. `--work-dir` keeps the encoded examples, so a later run with the same encoder and data skips encoding.
- Threshold calibration: the best balanced accuracy on the validation split.
- Provenance in the artifact's manifest: the pinned dataset and model commits, the defakto version, the date, the example counts and the metrics.

## [0.1.0] - 2026-10-22

First release. Text detection (e.g., deepfake, safety, hallucination)

### Added

- Koopman dynamics detection: an encoder turns the input into a trajectory of embeddings, and two fitted operators score how it evolves.
- Scores broken down by token and by sentence, so flagged spans can be located.
- Two official text artifacts: `Forensiqa/text-safety-jina-v2-base` and `Forensiqa/text-safety-qwen3-embedding-0.6b`.
- Python API: `defakto.analyze()` and a reusable `Analyzer`. Artifacts that share an encoder share one encoding.
- `defakto analyze` command: text from an argument, a file or stdin, artifacts per media type from a YAML config, and `--json` output.
- `text` extra that installs PyTorch and Transformers.

[Unreleased]: https://github.com/Forensiqa/defakto/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Forensiqa/defakto/releases/tag/v0.1.0
