# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Load and run Hugging Face-format causal language models.

Reads ``.safetensors`` checkpoints (single file or sharded) and their
``config.json``, maps the weights onto mojolearn's transformer blocks, and
runs them on the GPU.

- :class:`CausalLM`: embedding, decoder blocks, final norm and head built
  from a checkpoint, with ``forward``, ``allocate_state``, ``step`` and greedy
  ``generate``.
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
from .tokenizer import PATTERNS, Tokenizer
# perf2-neural (2026-11-06): `causal_lm` connects the GPU transformer, mamba,
# training and linalg bindings when it is imported (9d3b9ba18b). Importing it
# here charged that connection to `import mojolearn` and to every Tokenizer,
# config or checkpoint user. The causal classes now load on first attribute
# access (`__getattr__` below): the first `models.CausalLM` (construct or
# `load`) pays the connection, and the standalone blocks still connect when
# `_transformer_impl` / `_mamba_impl` are imported. Import glue only.
_CAUSAL_NAMES = {"CausalLM": ".causal_lm", "CausalLMState": ".causal_lm",
                 "GenerationStream": ".causal_lm", "ParallelCausalLM": ".parallel_causal_lm"}


def _connect_runtime_codec():
    # The cheap, process-wide part of the causal connection stays at import:
    # the shared operation-lock protocol and the state codec's storage classes
    # in the tokenizer host. The parallel MLP, neural, byte and forecast pools
    # pack values through that codec without importing `causal_lm`, so it must
    # not wait for the first CausalLM. `causal_lm` repeats both calls; each is
    # idempotent for the same runtime and classes.
    from .. import lowbit
    from .._array import Array
    from .._native_operation_lock import connect
    from .tokenizer import _NATIVE as runtime
    connect(runtime)
    runtime.native_codec_call("initialize", (Array, lowbit.BF16Weight, lowbit.Int8Weight))


_connect_runtime_codec()
# catalog / verify_catalog_entry: listed in docs-site/api_exclude.txt until the first verified catalog
# entry lands; then add them to the docstring above and drop those two lines.
# Catalog verification depends on the optional ops/importer surface. Keep
# its import-time connection lazy, like mojolearn.ops itself, so unrelated
# native models and estimators remain importable without NumPy.

__all__ = ["ParallelCausalLM", "CausalLM", "CausalLMState", "GenerationStream", "Checkpoint", "FAMILIES", "HFConfig", "ModelPlan",
           "PATTERNS", "SafetensorsFile", "Tokenizer", "UnsupportedModel", "catalog", "plan_for",
           "verify_catalog_entry"]


def __getattr__(name):
    if name in _CAUSAL_NAMES:
        import importlib
        value = getattr(importlib.import_module(_CAUSAL_NAMES[name], __name__), name)
        globals()[name] = value
        return value
    if name in ("catalog", "verify_catalog_entry"):
        from . import _catalog
        value = getattr(_catalog, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))  # glue: include lazy public module exports
