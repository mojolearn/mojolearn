# SPDX-License-Identifier: Apache-2.0
"""The CPU IVF image's entry of the parallel IVF table
(core/parallel_entry_table.mojo, family PARALLEL_TABLE_IVF; lane
calltable-parallel-archive, 2026-11-11).

Published by `_mojolearn_ivf_host` alone (the search-only host binding
`_mojolearn_ivf_search_host` publishes nothing, so two host images in one
process never claim the same slot). The same host bodies the former Python
doors `ivf_shard_plan` and `ivf_merge_shards` ran, and the body of the
`ivf_finalize_distances` door, reached natively by the distributed IVF owner.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.parallel_entry_table import (
    PARALLEL_TABLE_IVF, PI_SHARD_PLAN, PI_MERGE_SHARDS, ParallelEntryKey, parallel_check_spans,
    parallel_table_publish,
)
from core.shared_slot import SharedSlot
from bindings.ivf_index_arrays import ivf_merge_shards_host_run, ivf_shard_plan_host_run
from ivf.index_arrays_impl import ivf_finalize_distances_host


def parallel_ivf_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        parallel_check_spans(PARALLEL_TABLE_IVF, operation, aa, na, pp, np, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == PI_SHARD_PLAN:
            r[] = ivf_shard_plan_host_run(a[0], a[1], a[2], a[3], a[4], p[0], p[1], p[2])
        elif operation == PI_MERGE_SHARDS:
            r[] = ivf_merge_shards_host_run(aa, pp)
        else:
            if p[0] < 0:
                raise Error("distance count must be nonnegative")
            ivf_finalize_distances_host(a[0], p[0], p[1])
            r[] = 0
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ParallelIvfHostOwner(Defaultable, Movable):
    """The image-lifetime publisher word of this image's IVF table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime PARALLEL_IVF_HOST_OWNER = SharedSlot[ParallelIvfHostOwner,
    "MojolearnParallelIvfHostOwnerIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnParallelIvfHostOwnerFastV1"]


def publish_parallel_ivf_host() raises:
    """Publish at module init of `_mojolearn_ivf_host` (vendor word 0)."""
    var owner = PARALLEL_IVF_HOST_OWNER.get_or_create_ptr()
    parallel_table_publish(PARALLEL_TABLE_IVF, ParallelEntryKey(Int(GLOBAL_NUMERIC_MODE), 0), Int(owner),
                           parallel_ivf_host_v1)
