# SPDX-License-Identifier: Apache-2.0
"""The GPU IVF image's entries of the parallel IVF table and the archive
export table (core/parallel_entry_table.mojo family PARALLEL_TABLE_IVF,
core/archive_entry_table.mojo family ARCHIVE_TABLE_IVF; lane
calltable-parallel-archive, 2026-11-11).

The same device bodies the former Python doors `ivf_shard_plan` and
`ivf_merge_shards` of `_mojolearn_ivf` ran, and the bodies of its
`ivf_finalize_distances` and `ivf_flat_index_export` doors (which stay for
their other Python callers), reached natively by the distributed IVF owner
(bindings/parallel_shard_native.mojo) and the archive writer
(bindings/archive_schema_native.mojo) with the GIL released by the caller.
No kernel or route changed.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.archive_entry_table import ARCHIVE_TABLE_IVF, archive_check_spans, archive_table_publish
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neural_context import process_ctx
from core.parallel_entry_table import (
    PARALLEL_TABLE_IVF, PI_SHARD_PLAN, PI_MERGE_SHARDS, ParallelEntryKey, parallel_check_spans,
    parallel_entry_vendor, parallel_table_publish,
)
from core.shard_merge_device import device_ivf_merge_shards, device_ivf_shard_plan, device_root_f32
from core.shared_slot import SharedSlot
from ivf.impl.neighbors.ivf_common import postprocess_distances_is_identity
from ivf.resident import ivf_resident_export_list_data

#: The IVF binding's ONE process-lifetime DeviceContext
#: (core/neural_context.mojo). `bindings/_mojolearn_ivf.mojo` names it
#: `_DEVCTX_SLOT`; it is defined here once so these entries share it.
comptime IVF_DEVCTX_SLOT = "MojolearnIvfContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnIvfContextFastV3"


def ivf_finalize_distances_device_run(address: Int, n: Int, metric: Int) raises:
    """`postprocess_distances` over the merged distances, on the device
    (lane cpu4-python: `ftz(identical_sqrt(d))` per word, the host
    statement; the host walk stays the host bindings')."""
    if n < 0:
        raise Error("distance count must be nonnegative")
    if postprocess_distances_is_identity(metric) or n == 0:
        return
    if address == 0:
        raise Error("ivf_finalize_distances: null buffer address")
    var ctx = process_ctx[IVF_DEVCTX_SLOT]()
    device_root_f32(ctx, address, n)


def ivf_merge_shards_device_run(aa: Int, pp: Int) raises -> Int:
    """`ivf_merge_shards` (bindings/ivf_index_arrays.mojo's contract) on the
    device (lane cpu4-python, core/shard_merge_device.mojo) over the
    PI_MERGE_SHARDS layout (spans already checked): the merge status."""
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    var shards = p[0]
    var m = p[1]
    var k = p[2]
    if shards < 1 or m < 0 or k < 1:
        raise Error("ivf_merge_shards: need at least one shard and k >= 1")
    var d_addrs = List[Int]()
    var i_addrs = List[Int]()
    var c_addrs = List[Int]()
    var m_addrs = List[Int]()
    var sizes = List[Int]()
    for s in range(shards):  # small-loop(shards: one shard per device): reads the shard addresses, launch arguments only
        d_addrs.append(a[3 + 4 * s])
        i_addrs.append(a[4 + 4 * s])
        c_addrs.append(a[5 + 4 * s])
        m_addrs.append(a[6 + 4 * s])
        sizes.append(p[3 + s])
    var ctx = process_ctx[IVF_DEVCTX_SLOT]()
    var r = device_ivf_merge_shards(ctx, m, k, d_addrs, i_addrs, c_addrs, m_addrs, sizes, a[0], a[1], a[2])
    return Int(r[0])


def ivf_shard_plan_device_run(ind: Int, offs: Int, soff: Int, local_ids: Int, id_map: Int,
                              n: Int, n_lists: Int, shards: Int) raises -> Int:
    """`ivf_shard_plan` (bindings/ivf_index_arrays.mojo's contract) on the
    device (lane cpu4-python, core/shard_merge_device.mojo)."""
    if n < 0 or n_lists < 1 or shards < 1:
        raise Error("ivf_shard_plan: needs 5 addresses, n >= 0, n_lists >= 1 and shards >= 1")
    var ctx = process_ctx[IVF_DEVCTX_SLOT]()
    return device_ivf_shard_plan(ctx, ind, offs, soff, local_ids, id_map, n, n_lists, shards)


def ivf_flat_index_export_device_run(handle: Int, address: Int, count: Int) raises:
    var ctx = process_ctx[IVF_DEVCTX_SLOT]()
    ivf_resident_export_list_data(ctx, handle, address, count)
    _ = ctx^


def parallel_ivf_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        parallel_check_spans(PARALLEL_TABLE_IVF, operation, aa, na, pp, np, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == PI_SHARD_PLAN:
            r[] = ivf_shard_plan_device_run(a[0], a[1], a[2], a[3], a[4], p[0], p[1], p[2])
        elif operation == PI_MERGE_SHARDS:
            r[] = ivf_merge_shards_device_run(aa, pp)
        else:
            ivf_finalize_distances_device_run(a[0], p[0], p[1])
            r[] = 0
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def archive_ivf_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        archive_check_spans(operation, aa, na, pp, np, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        ivf_flat_index_export_device_run(p[0], a[0], p[1])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 0
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ParallelIvfDeviceOwner(Defaultable, Movable):
    """The image-lifetime publisher word of this image's IVF tables."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime PARALLEL_IVF_DEVICE_OWNER = SharedSlot[ParallelIvfDeviceOwner,
    "MojolearnParallelIvfDeviceOwnerIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnParallelIvfDeviceOwnerFastV1"]


def publish_parallel_ivf_device() raises:
    """Publish at module init of `_mojolearn_ivf` under its compiled vendor
    (none for a build with no GPU vendor)."""
    var word = parallel_entry_vendor(COMPILED_VENDOR)
    if word < 1:
        return
    var owner = PARALLEL_IVF_DEVICE_OWNER.get_or_create_ptr()
    var key = ParallelEntryKey(Int(GLOBAL_NUMERIC_MODE), word)
    parallel_table_publish(PARALLEL_TABLE_IVF, key.copy(), Int(owner), parallel_ivf_device_v1)
    archive_table_publish(ARCHIVE_TABLE_IVF, key^, Int(owner), archive_ivf_device_v1)
