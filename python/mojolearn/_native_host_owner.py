# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The loader of the shared host-owner binding."""
# `_mojolearn_tokenizer_host` is the host owner every runtime owner binding is
# connected into (host_surface.HOST_IMPORT_ALIASES); it registers the attribute,
# parameter, constructor, array, optimizer and embedding-layer entries every
# family's estimators call, and it is in the `shared` release family
# (bindings/release_families.json). Its loader lived in `tokenizer.py` until
# 2026-11-28 (lane lane-loads-2), beside the names of the GPU tokenizer
# (`_mojolearn_x_tokenizer`) and the fast one, so every bridge that imported
# `_binding` from there counted as loading the tokenizer family: 956 of 1,022
# verify lanes. The code below is that loader, unchanged; `tokenizer.py`
# re-exports every name, so `mojolearn.tokenizer._binding` is the same object.
# This module names the shared host owner and nothing else: do not add another
# binding name here.
from . import _backend

_EXTENSION = "_mojolearn_tokenizer_host"

#: The binding's entries were `gpt2_*` until 2026-09-18 and are `bpe_*`
#: since. A binding built before the rename (an older host directory under
#: MOJOLEARN_HOST_DIR) is read through its old names so it keeps loading;
#: the ids are the same arithmetic under either name.
_ABI = ("load", "n_vocab", "max_token_bytes", "encode", "encode_batch")


class _Entries:
    """The binding module with its tokenizer entries under their `bpe_*`
    names; every other attribute is the module's own."""

    def __init__(self, module):
        self._module = module
        for name in _ABI:  # glue: looks up the binding ABI entries
            f = getattr(module, "bpe_" + name, None) or getattr(module, "gpt2_" + name, None)
            if f is None:
                raise ImportError(f"mojolearn: the tokenizer binding exports neither bpe_{name} nor gpt2_{name}")
            setattr(self, "bpe_" + name, f)

    def __getattr__(self, name):
        return getattr(self._module, name)


def _binding():
    return _Entries(_backend.load_host_module(_EXTENSION))  # import-time: tokenizer host binding module loaded once by load_host_module (text input step)
