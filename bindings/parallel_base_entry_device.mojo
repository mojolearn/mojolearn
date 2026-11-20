# SPDX-License-Identifier: Apache-2.0
"""The GPU base image's entry of the parallel base table
(core/parallel_entry_table.mojo, family PARALLEL_TABLE_BASE; lane
calltable-parallel-archive, 2026-11-11).

The same device routes the former Python doors `fold_pair_f32`,
`msel_rebase_offsets_i32` and `shard_topk_merge_f32` of `_mojolearn` ran
(each with its switch and host fallback unchanged), reached natively by the
parallel owners with the GIL released by the caller. No kernel, fold order
or route changed.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neural_context import process_ctx
from core.parallel_entry_table import (
    PARALLEL_TABLE_BASE, PB_FOLD_PAIR_F32, PB_REBASE_OFFSETS, ParallelEntryKey, parallel_check_spans,
    parallel_entry_vendor, parallel_table_publish,
)
from core.shard_merge_device import device_shard_topk_merge_f32
from core.shared_slot import SharedSlot
from core.hotpath_device_run import fold_pair_f32_device_run
from core.msel_device_run import msel_rebase_offsets_i32_device_run

#: The GPU base binding's ONE process-lifetime DeviceContext
#: (core/neural_context.mojo, lane/devctx-lifetime): a context per call
#: exhausts Metal command queues. `bindings/_mojolearn.mojo` names it
#: `_DEVCTX_SLOT`; it is defined here once so this entry shares it.
comptime CORE_DEVCTX_SLOT = "MojolearnCoreContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnCoreContextFastV3"


def shard_topk_merge_f32_device_run(ta: Int, s_count: Int, nq: Int, kk: Int, od: Int, oi: Int) raises -> Int:
    """`shard_topk_merge_f32` (bindings/array_helpers.mojo's contract: 0, 1
    a local id outside its shard, 2 fewer than k candidates) on the device
    (lane cpu4-python, core/shard_merge_device.mojo); the host loop is the
    core host image's (`shard_topk_merge_f32_host_run`). Moved here from the
    former Python door of `_mojolearn` unchanged."""
    if s_count < 1 or nq < 0 or kk < 1:
        raise Error("shard_topk_merge_f32: bad dimensions")
    if nq == 0:
        return 0
    if ta == 0 or od == 0 or oi == 0:
        raise Error("shard_topk_merge_f32: null buffer address")
    var tp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=ta)
    var ctx = process_ctx[CORE_DEVCTX_SLOT]()
    return device_shard_topk_merge_f32(ctx, tp, s_count, nq, kk, od, oi)


def parallel_base_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        parallel_check_spans(PARALLEL_TABLE_BASE, operation, aa, na, pp, np, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == PB_FOLD_PAIR_F32:
            r[] = fold_pair_f32_device_run(a[0], a[1], p[0])
        elif operation == PB_REBASE_OFFSETS:
            msel_rebase_offsets_i32_device_run(a[0], p[0], a[1], p[1], p[2], a[2])
            r[] = 0
        else:
            r[] = shard_topk_merge_f32_device_run(a[0], p[0], p[1], p[2], a[1], a[2])
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ParallelBaseDeviceOwner(Defaultable, Movable):
    """The image-lifetime publisher word of this image's parallel base table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime PARALLEL_BASE_DEVICE_OWNER = SharedSlot[ParallelBaseDeviceOwner,
    "MojolearnParallelBaseDeviceOwnerIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnParallelBaseDeviceOwnerFastV1"]


def publish_parallel_base_device() raises:
    """Publish at module init of `_mojolearn` under its compiled vendor. A
    build with no GPU vendor publishes nothing; its callers refuse with the
    table's "provider is not published"."""
    var word = parallel_entry_vendor(COMPILED_VENDOR)
    if word < 1:
        return
    var owner = PARALLEL_BASE_DEVICE_OWNER.get_or_create_ptr()
    parallel_table_publish(PARALLEL_TABLE_BASE, ParallelEntryKey(Int(GLOBAL_NUMERIC_MODE), word), Int(owner),
                           parallel_base_device_v1)
