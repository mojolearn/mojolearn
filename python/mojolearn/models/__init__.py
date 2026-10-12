# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Load and run Hugging Face-format causal language models.

Reads ``.safetensors`` checkpoints (single file or sharded) and their
``config.json``, maps the weights onto mojolearn's transformer blocks, and
runs them on the GPU.

- :class:`CausalLM`: embedding, decoder blocks, final norm and head built
  from a checkpoint, with ``forward``, ``allocate_state``, ``step`` and greedy
  ``generate``.
- :class:`ParallelCausalLM`: the same model with each layer assigned to a
  device (``layer_devices``).
- :class:`Tokenizer`: ``Tokenizer.from_pretrained`` over ``tokenizer.json``
  for byte-level BPE tokenizers, with the pre-tokenization pattern
  (``PATTERNS``: GPT-2, Llama 3, Qwen 2) as a parameter.
- :class:`HFConfig`, :func:`plan_for`, ``FAMILIES``: which ``model_type``
  values are supported (Llama 2/3 and SmolLM, Mistral, Qwen 2, Qwen 3, ...),
  how config fields map onto block options, and which fields are refused by
  name with :class:`UnsupportedModel`.
- :class:`Checkpoint`, :class:`SafetensorsFile`: the ``.safetensors``
  reader; tensors come back as ``mojolearn.Array``.

Examples
--------
>>> from mojolearn import Array
>>> from mojolearn.models import CausalLM, Tokenizer
>>> lm = CausalLM.load("path/to/model", weight_format="bfloat16")
>>> tok = Tokenizer.from_pretrained("path/to/model")
>>> ids = tok.encode("The capital of France is")
>>> out = lm.generate(Array.from_list([ids], "<i4"), 8)
"""
# The option matrix (known model_types, config-field mapping, tensor names,
# refusals by name) is in README.md beside this file and in config.py.
from .config import FAMILIES, HFConfig, ModelPlan, UnsupportedModel, plan_for
from .safetensors import Checkpoint, SafetensorsFile
from .causal_lm import CausalLM, CausalLMState
from .parallel_causal_lm import ParallelCausalLM
from .tokenizer import PATTERNS, Tokenizer

__all__ = ["ParallelCausalLM", "CausalLM", "CausalLMState", "Checkpoint", "FAMILIES", "HFConfig", "ModelPlan",
           "PATTERNS", "SafetensorsFile", "Tokenizer", "UnsupportedModel", "plan_for"]
