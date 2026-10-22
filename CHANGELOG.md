# Changelog

All notable changes to defakto are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-22

First release. Text detection (e.g., deepfake, safety, hallucination)

### Added

- Koopman dynamics detection: an encoder turns the input into a trajectory of embeddings, and two fitted operators score how it evolves.
- Scores broken down by token and by sentence, so flagged spans can be located.
- Two official text artifacts: `Forensiqa/text-safety-jina-v2-base` and `Forensiqa/text-safety-qwen3-embedding-0.6b`.
- Python API: `defakto.analyze()` and a reusable `Analyzer`. Artifacts that share an encoder share one encoding.
- `defakto analyze` command: text from an argument, a file or stdin, artifacts per media type from a YAML config, and `--json` output.
- `text` extra that installs PyTorch and Transformers.

[0.1.0]: https://github.com/Forensiqa/defakto/releases/tag/v0.1.0
