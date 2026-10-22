# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from defakto.artifacts.manifest import EncoderSpec
from defakto.errors import EncoderError
from defakto.trajectory import Step, Trajectory

_INSTALL_HINT = "Install them with: pip install 'defakto[text]'"


class HFTokenEncoder:
    """Per-token embeddings from a Hugging Face model: one trajectory step per token.

    Settings come from the artifact's ``encoder`` section: ``model_name``,
    ``model_revision``, ``max_length``, ``strip_special_tokens``, ``dtype`` and
    ``trust_remote_code``.
    """

    media_type = "text"

    def __init__(self, spec: EncoderSpec, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise EncoderError(
                f"Text analysis needs torch and transformers. {_INSTALL_HINT}"
            ) from e

        if not spec.model_name:
            raise EncoderError("hf-token-embeddings encoder needs a 'model_name'")
        dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype_name = str(spec.get("dtype", "float32"))
        if dtype_name not in dtypes:
            raise EncoderError(f"Unsupported dtype '{dtype_name}' (use one of {sorted(dtypes)})")

        self.spec = spec
        self.max_length = int(spec.get("max_length", 512))
        self.strip_special_tokens = bool(spec.get("strip_special_tokens", True))
        revision = spec.get("model_revision")
        trust_remote_code = bool(spec.get("trust_remote_code", False))

        self._torch = torch
        self.device = torch.device(device or "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_name, revision=revision, trust_remote_code=trust_remote_code
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise EncoderError(
                f"'{spec.model_name}' has no fast tokenizer, which is needed to map tokens "
                f"back to the text"
            )
        self.model = AutoModel.from_pretrained(
            spec.model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            dtype=dtypes[dtype_name],
        ).to(self.device)
        self.model.eval()

    def encode(self, content: str) -> Trajectory:
        if not isinstance(content, str):
            raise TypeError(f"text encoder expects a string, got {type(content).__name__}")

        inputs = self.tokenizer(
            content,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        special = inputs.pop("special_tokens_mask")[0].tolist()
        offsets = inputs.pop("offset_mapping")[0].tolist()
        ids = inputs["input_ids"][0].tolist()

        with self._torch.no_grad():
            outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        hidden = outputs.last_hidden_state[0]

        drop_special = self.strip_special_tokens
        keep = [i for i, is_special in enumerate(special) if not (drop_special and is_special)]
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        embeddings = hidden[keep].to(self._torch.float64).cpu().numpy()
        steps = [Step(label=tokens[i], start=offsets[i][0], end=offsets[i][1]) for i in keep]

        notes: list[str] = []
        last_end = max((offsets[i][1] for i in keep), default=0)
        if len(ids) >= self.max_length and last_end < len(content.rstrip()):
            notes.append(
                f"Text was cut to the first {self.max_length} tokens; the rest wasn't analyzed."
            )
        return Trajectory(
            media_type="text", embeddings=embeddings, steps=steps, span_unit="char", notes=notes
        )
