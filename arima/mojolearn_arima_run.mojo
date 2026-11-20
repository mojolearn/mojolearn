# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN of `bindings/_mojolearn_arima.mojo` (wave 8 run split): the typed
address-to-pointer helpers the binding's doors use. Imports no Python
(tools/python_free_dirs.sh); bodies moved unchanged (lane
rs-sequence-tokenizer, 2026-11-12)."""
from core.hostptr import f32_ptr, i32_ptr


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)
