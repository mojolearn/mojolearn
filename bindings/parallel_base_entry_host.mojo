# SPDX-License-Identifier: Apache-2.0
"""The core host image's entry of the parallel base table
(core/parallel_entry_table.mojo, family PARALLEL_TABLE_BASE; lane
calltable-parallel-archive, 2026-11-11).

The same host bodies the former Python doors `fold_pair_f32`,
`msel_rebase_offsets_i32` and `shard_topk_merge_f32` ran, reached natively by
the parallel owners (bindings/cross_vendor_native.mojo,
bindings/parallel_shard_native.mojo) with the GIL released by the caller.
No kernel, fold order or route changed.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.hotpath_impl_rows import fold_pair_f32
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.parallel_entry_table import (
    PARALLEL_TABLE_BASE, PB_FOLD_PAIR_F32, PB_REBASE_OFFSETS, ParallelEntryKey, parallel_check_spans,
    parallel_table_publish,
)
from core.shared_slot import SharedSlot
from core.array_helpers_run import shard_topk_merge_f32_host_run
from bindings.msel_host import msel_rebase_offsets_i32_host_run


def parallel_base_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        parallel_check_spans(PARALLEL_TABLE_BASE, operation, aa, na, pp, np, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == PB_FOLD_PAIR_F32:
            r[] = fold_pair_f32(a[0], a[1], p[0])
        elif operation == PB_REBASE_OFFSETS:
            msel_rebase_offsets_i32_host_run(a[0], p[0], a[1], p[1], p[2], a[2])
            r[] = 0
        else:
            r[] = shard_topk_merge_f32_host_run(a[0], p[0], p[1], p[2], a[1], a[2])
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ParallelBaseHostOwner(Defaultable, Movable):
    """The image-lifetime publisher word of this image's parallel base table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime PARALLEL_BASE_HOST_OWNER = SharedSlot[ParallelBaseHostOwner,
    "MojolearnParallelBaseHostOwnerIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnParallelBaseHostOwnerFastV1"]


def publish_parallel_base_host() raises:
    """Publish at module init of `_mojolearn_core_host` (vendor word 0)."""
    var owner = PARALLEL_BASE_HOST_OWNER.get_or_create_ptr()
    parallel_table_publish(PARALLEL_TABLE_BASE, ParallelEntryKey(Int(GLOBAL_NUMERIC_MODE), 0), Int(owner),
                           parallel_base_host_v1)
