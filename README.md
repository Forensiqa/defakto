# Defakto

An open-source toolkit from [Forensiqa](https://forensiqa.com) for detecting AI-generated and unreliable content: deepfake media, and hallucinated or unsafe LLM responses.

> **Status: early development.** Text analysis (LLM response safety) works today. Image, audio and video support is planned.

Defakto uses one method for every media type: an encoder turns the input into a trajectory of embeddings, and a Koopman dynamics model scores how that trajectory evolves. Inputs whose dynamics don't match the reliable class get flagged, and the score is broken down so you can see *where*: which tokens and sentences, and later which frames or seconds.

## Install

```
pip install "defakto[text]"
```

Requires Python 3.12 or later. The `text` extra installs PyTorch and Transformers for text analysis. To [fit your own artifacts](#fit-your-own-artifacts) on text, install `defakto[text,fit]`.

## Usage

Detection runs with **artifacts**: fitted models for one media type and one task, for example unsafe generated content. Each artifact names the encoder it was fitted with, and defakto builds that encoder for you. The first run downloads the artifact and its embedding model; both are cached afterwards.

### From Python

```python
import defakto

results = defakto.analyze(
    "An LLM response you want to check...",
    artifacts=["Forensiqa/text-safety-jina-v2-base"],
)
for r in results:
    print(r.artifact_id, r.label, round(r.score, 3))
    for sentence in r.segments_of("sentence"):
        if sentence.flagged:
            print("  flagged:", sentence.text)
```

To analyze many inputs, create an `Analyzer` once and reuse it, so models load only once:

```python
analyzer = defakto.Analyzer(["Forensiqa/text-safety-jina-v2-base"])
results = analyzer.analyze("response.txt")
```

### From the command line

```
defakto analyze --text "An LLM response..." -a Forensiqa/text-safety-jina-v2-base
defakto analyze response.txt -a Forensiqa/text-safety-jina-v2-base --json
cat response.txt | defakto analyze - -a Forensiqa/text-safety-jina-v2-base
```

To choose artifacts per media type, use a config file:

```yaml
# defakto.yaml
detectors:
  text:
    - Forensiqa/text-safety-jina-v2-base
    - Forensiqa/text-safety-qwen3-embedding-0.6b
```

```
defakto analyze response.txt --config defakto.yaml
```

Artifacts that share an encoder share one encoding, so adding a second text artifact doesn't embed the text twice. Each artifact's score is on its own scale: compare a score with its artifact's threshold, not with another artifact's score.

## Artifacts

An artifact is a folder with two files:

- `weights.safetensors`: the model's arrays (`lift_modes`, `observable_modes`, `operator_negative`, `operator_positive`).
- `manifest.yaml`: what the artifact detects, the exact encoder it needs, its dimensions, threshold, and a checksum of the weights.

Defakto checks the checksum and dimensions before using an artifact. Official artifacts are published on the Hugging Face Hub under the Forensiqa organization:

| Artifact | Task | Encoder |
|---|---|---|
| `Forensiqa/text-safety-jina-v2-base` | unsafe content in text | `jinaai/jina-embeddings-v2-base-en` |
| `Forensiqa/text-safety-qwen3-embedding-0.6b` | unsafe content in text | `Qwen/Qwen3-Embedding-0.6B` |

Add `@revision` (a tag or commit) to pin an exact version, for example `Forensiqa/text-safety-jina-v2-base@v0.1.0`. Local artifact folders work too: pass the folder's path instead of a Hub id.

## Fit your own artifacts

You can fit an artifact on your own labeled data. Each example is either in the class the artifact flags (positive) or not. Fitting learns one operator per class.

### From Python

`defakto.fit()` takes `(content, positive)` pairs from any source:

```python
import defakto

train = [
    ("A response that should be flagged...", True),
    ("A response that shouldn't be flagged...", False),
    # ...
]

result = defakto.fit(
    train,
    validation=validation,  # pairs like train, used to choose the threshold
    test=test,  # pairs like train, for eval
    encoder={
        "type": "hf-token-embeddings",
        "model_name": "jinaai/jina-embeddings-v2-base-en",
        "max_length": 512,
        "strip_special_tokens": True,
        "dtype": "float32",
        "trust_remote_code": True,
    },
    output="my-artifact",
    artifact_id="my-org/my-artifact",
    positive_label="unsafe",
    negative_label="safe",
)
print(result.threshold, result.test[0].auc)

defakto.analyze("A new response...", artifacts=["my-artifact"])
```

The threshold is the one with the best balanced accuracy on `validation`. Without validation data it is 0, where both operators predict equally well. `test` plays no part in any choice, so its metrics estimate how the artifact does on unseen data.

### From the command line

`defakto fit` reads a labeled dataset from the Hugging Face Hub, described in a YAML file. For local data, use `defakto.fit()`.

```yaml
# fit.yaml
artifact:
  id: my-org/text-safety-jina-v2-base
  task: safety
  labels:
    positive: unsafe
    negative: safe

encoder:
  type: hf-token-embeddings
  model_name: jinaai/jina-embeddings-v2-base-en
  max_length: 512
  strip_special_tokens: true
  dtype: float32
  trust_remote_code: true

data:
  dataset: nvidia/Aegis-AI-Content-Safety-Dataset-2.0
  text_column: response
  label_column: response_label
  positive: unsafe        # this label is the flagged class; any other label is not
  splits:
    train: train
    validation: validation
    test: test
```

```
defakto fit fit.yaml -o my-artifact --work-dir .fit-cache
```

Complete configs are in [`examples/fit`](https://github.com/Forensiqa/defakto/tree/main/examples/fit).

The dataset and the encoder's model are pinned to their current Hub commits when fitting starts. The artifact's manifest records them, along with the defakto version, the date, the example counts and the metrics.


## Development

```
uv sync --all-extras
uv run pre-commit install
uv run pytest
```

## License

Apache License 2.0. See [LICENSE](https://github.com/Forensiqa/defakto/blob/main/LICENSE).
