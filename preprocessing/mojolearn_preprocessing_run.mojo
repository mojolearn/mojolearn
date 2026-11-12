# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""RUN of `bindings/_mojolearn_preprocessing.mojo` (wave 8 run split): the
borrowed Float32 address check as typed Mojo. Imports no Python."""
from core.hostptr import f32_ptr


def ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("preprocessing: null Float32 pointer")
    return f32_ptr(addr)
