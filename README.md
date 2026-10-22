# Defakto

An open-source toolkit from [Forensiqa](https://forensiqa.com) for detecting AI-generated and unreliable content: deepfake media, and hallucinated or unsafe LLM responses.

> **Status: early development.** Text analysis (LLM response safety) works today. Image, audio and video support is planned.

Defakto uses one method for every media type: an encoder turns the input into a trajectory of embeddings, and a Koopman dynamics model scores how that trajectory evolves. Inputs whose dynamics don't match the reliable class get flagged, and the score is broken down so you can see *where*: which tokens and sentences, and later which frames or seconds.

## Install

```
pip install "defakto[text]"
```

Requires Python 3.12 or later. The `text` extra installs PyTorch and Transformers for text analysis.

## Usage

Detection runs with **artifacts**: fitted models for one media type and one task, for example unsafe content in English text. Each artifact names the encoder it was fitted with, and defakto builds that encoder for you.

### From Python

```python
import defakto

results = defakto.analyze(
    "An LLM response you want to check...",
    artifacts=["./artifacts/text-safety-jina-v2-base"],
)
for r in results:
    print(r.artifact_id, r.label, round(r.score, 3))
    for sentence in r.segments_of("sentence"):
        if sentence.flagged:
            print("  flagged:", sentence.text)
```

To analyze many inputs, create an `Analyzer` once and reuse it, so models load only once:

```python
analyzer = defakto.Analyzer(["./artifacts/text-safety-jina-v2-base"])
results = analyzer.analyze("response.txt")
```

### From the command line

```
defakto analyze --text "An LLM response..." -a ./artifacts/text-safety-jina-v2-base
defakto analyze response.txt -a ./artifacts/text-safety-jina-v2-base --json
cat response.txt | defakto analyze - -a ./artifacts/text-safety-jina-v2-base
```

To choose artifacts per media type, use a config file:

```yaml
# defakto.yaml
detectors:
  text:
    - ./artifacts/text-safety-jina-v2-base
    - ./artifacts/text-safety-qwen3-embedding-0.6b
```

```
defakto analyze response.txt --config defakto.yaml
```

Artifacts that share an encoder share one encoding, so adding a second text artifact doesn't embed the text twice.

## Artifacts

An artifact is a folder with two files:

- `weights.safetensors`: the model's arrays (`lift_modes`, `observable_modes`, `operator_negative`, `operator_positive`).
- `manifest.yaml`: what the artifact detects, the exact encoder it needs, its dimensions, threshold, and a checksum of the weights.

Defakto checks the checksum and dimensions before using an artifact. Artifacts can be local folders or Hugging Face Hub ids. Official artifacts are published under the Forensiqa organization, for example `forensiqa/text-safety-jina-v2-base`. Add `@revision` (a tag or commit) to pin an exact version, for example `forensiqa/text-safety-jina-v2-base@v0.1.0`.

## Development

```
uv sync --all-extras
uv run pre-commit install
uv run pytest
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
