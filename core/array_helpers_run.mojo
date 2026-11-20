# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of `bindings/array_helpers.mojo` (wave 8 run split): the typed
address helper the doors use and the shard top-k merge the core host image's
parallel table calls. Typed Mojo only: no Python import."""
from core.array_helpers_impl import shard_topk_merge_f32


@always_inline
def _addr_ptr[dt: DType](addr: Int) raises -> MutPointer[Scalar[dt], MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=addr)


def shard_topk_merge_f32_host_run(ta: Int, s_count: Int, nq: Int, kk: Int, od: Int, oi: Int) raises -> Int:
    """`parallel_neighbors_reference._merge` (lane pyglue-sweep: out of the
    Python per-query loop): per query row, every reference shard's
    candidates as composite keys (neighbors/checks/select_radix_identical
    `composite_key`: twiddled float32 distance bits << 32 | global index),
    the k smallest keys, then estimator.mojo's host insertion sort on the
    float distance (ties by index). `table` is int64[5 * n_shards]: per
    shard the float32 distance address, the int64 local-id address, the
    first and end global rows and the width (columns per query row).
    Writes the winning float bits and uint32 global indices. Returns 0,
    1 when a local id is outside its shard, 2 for fewer than k candidates.
    Reached through the core host image's parallel table entry
    (bindings/parallel_base_entry_host.mojo); the Python door had no caller
    left (lane calltable-parallel-archive, 2026-11-11)."""
    if s_count < 1 or nq < 0 or kk < 1:
        raise Error("shard_topk_merge_f32: bad dimensions")
    if nq == 0:
        return 0
    var tp = _addr_ptr[DType.int64](ta)
    var dp = _addr_ptr[DType.uint32](od)
    var ip = _addr_ptr[DType.uint32](oi)
    return shard_topk_merge_f32(tp, dp, ip, s_count, nq, kk)
