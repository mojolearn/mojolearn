# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host copies for the Mamba decode sessions and Mamba3 state handoff.

Byte copies only: no arithmetic, no Python. The host arm of a decode session
keeps its own copy of the weights and the state (the session's ownership
clause), and `sync_state` / `load_state` / `set_input_states` move whole
float32 buffers between those copies and the caller's state pieces.
"""
from std.memory import memcpy
from core.hostptr import f32_ptr


def mamba_copy_f32(dst_addr: Int, src_addr: Int, count: Int) raises:
    """Copy `count` float32 values from `src_addr` to `dst_addr` (disjoint)."""
    if count > 0:
        memcpy(dest=f32_ptr(dst_addr), src=f32_ptr(src_addr), count=count)


def mamba_host_copy(src_addr: Int, count: Int) raises -> List[Float32]:
    """A new owned buffer holding the `count` float32 values at `src_addr`."""
    var out = List[Float32](length=count, fill=0)
    if count > 0:
        memcpy(dest=out.unsafe_ptr(), src=f32_ptr(src_addr), count=count)
    return out^
