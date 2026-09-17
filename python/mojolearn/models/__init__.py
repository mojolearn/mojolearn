# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Checkpoint loading for Hugging Face-shaped causal language models
(lane/model-loader, 2026-09-17).

    from mojolearn.models import CausalLM, Tokenizer
    lm = CausalLM.load("path/to/model", weight_format="bfloat16")
    tok = Tokenizer.from_pretrained("path/to/model")
    ids = tok.encode("The capital of France is")
    out = lm.generate(Array.from_list([ids], "<i4"), 8)

What is here, and what each piece claims:

  `safetensors`   a pure-Python reader of the `.safetensors` format and its
                  sharded index; tensors come back as `mojolearn.Array`
  `config`        `HFConfig` and the OPTION MATRIX: which `model_type`s are
                  known, which config fields map onto which block options,
                  which tensor names carry each weight, and which fields are
                  refused BY NAME
  `causal_lm`     `CausalLM`: embedding, N blocks, final norm and head
                  assembled from a config and its shards, with `forward`,
                  `allocate_state`, `step` and greedy `generate`
  `tokenizer`     `Tokenizer.from_pretrained` over `tokenizer.json` for the
                  byte-level BPE families, with the pre-tokenization
                  pattern a PARAMETER (GPT-2, Llama 3, Qwen 2)

NO REAL CHECKPOINT HAS BEEN LOADED THROUGH THIS PACKAGE YET. Every test in
`tests/test_models_loader.py` writes a tiny synthetic checkpoint and loads
that; loading a published model is lane C's run, and until it is recorded
this package claims the format reader, the name mapping and the refusals,
not a model's logits. `README.md` beside this file carries the matrix.
"""
from .config import FAMILIES, HFConfig, ModelPlan, UnsupportedModel, plan_for
from .safetensors import Checkpoint, SafetensorsFile
from .causal_lm import CausalLM, CausalLMState
from .tokenizer import PATTERNS, Tokenizer

__all__ = ["CausalLM", "CausalLMState", "Checkpoint", "FAMILIES", "HFConfig", "ModelPlan",
           "PATTERNS", "SafetensorsFile", "Tokenizer", "UnsupportedModel", "plan_for"]
