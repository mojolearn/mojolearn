# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Stable partition oracle above the NVIDIA single-pass routing threshold.

Run IDENTICAL with and without MOJOLEARN_IDENTICAL_SINGLE_PASS_PARTITION.
The banner distinguishes actual single-pass reach from a fallback-only run.
Permuted leaf slots, an empty leaf, a short leaf, all-zero/all-one flags,
ragged tails and sentinel gaps check the complete permutation and its bounds.
"""
from max.gpu.host import DeviceContext
from std.testing import assert_equal
from checks.kernel_matrix import TARGET_COLUMN, reorder_single_pass_for
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from gbdt.methods.greedy_subsets_searcher.kernel.split_points import PARTITION_BLOCK
from gbdt.gpu_util.kernel.reorder_single_pass import launch_stable_partition_routed


def run_checks() raises:
    var ctx = DeviceContext()
    comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime ROUTED = reorder_single_pass_for[TARGET_COLUMN, IDENTICAL]()
    var large = 1_000_003
    var short = 777
    var total = 17 + large + 11 + short + 19
    print("numeric_mode", numeric_mode_name(), "device", ctx.name())
    print("single_pass_reached", ROUTED, "max_leaf_rows", large)
    var sizes: List[Int] = [large, short, 0]
    var offsets: List[Int] = [17, 17 + large + 11, total - 1]
    var slots: List[UInt32] = [1, 0, 2]
    var hoff: List[UInt32] = [UInt32(offsets[0]), UInt32(offsets[1]), UInt32(offsets[2])]
    var hsize: List[UInt32] = [UInt32(sizes[0]), UInt32(sizes[1]), UInt32(sizes[2])]
    var lids = upload[DType.uint32](ctx, slots)
    var off = upload[DType.uint32](ctx, hoff)
    var size = upload[DType.uint32](ctx, hsize)
    var flags = ctx.enqueue_create_buffer[DType.uint8](total)
    var gmap = ctx.enqueue_create_buffer[DType.uint32](total)
    var sorted_flags = ctx.enqueue_create_buffer[DType.uint8](total)
    var chunks = (large + PARTITION_BLOCK - 1) // PARTITION_BLOCK
    var chunk_zeros = ctx.enqueue_create_buffer[DType.uint32](3 * chunks)
    var chunk_offsets = ctx.enqueue_create_buffer[DType.uint32](3 * chunks)
    var leaf_zeros = ctx.enqueue_create_buffer[DType.uint32](3)
    for pattern in range(3):
        var hflags = List[UInt8](length=total, fill=UInt8(9))
        for leaf in range(3):
            for r in range(sizes[leaf]):
                var flag = (r * 13 + r // 7 + leaf) & 1
                if pattern < 2:
                    flag = pattern
                hflags[offsets[leaf] + r] = UInt8(flag)
        upload_into[DType.uint8](ctx, flags, hflags)
        ctx.enqueue_memset(gmap, UInt32(0xDEADBEEF))
        ctx.enqueue_memset(sorted_flags, UInt8(9))
        launch_stable_partition_routed[IDENTICAL](
            ctx, 3, large, lids, off, size, flags, chunk_zeros,
            chunk_offsets, leaf_zeros, gmap, sorted_flags, sm_count=4,
        )
        var got = download[DType.uint32](ctx, gmap, total)
        var gotflags = download[DType.uint8](ctx, sorted_flags, total)
        var gotzeros = download[DType.uint32](ctx, leaf_zeros, 3)
        for leaf in range(3):
            var cursor = offsets[leaf]
            var zero_count = 0
            for wanted in range(2):
                for r in range(sizes[leaf]):
                    if Int(hflags[offsets[leaf] + r]) == wanted:
                        assert_equal(got[cursor], UInt32(r))
                        assert_equal(gotflags[cursor], UInt8(wanted))
                        cursor += 1
                        if wanted == 0:
                            zero_count += 1
            var slot = 1 - leaf if leaf < 2 else leaf
            assert_equal(gotzeros[slot], UInt32(zero_count))
        for i in range(total):
            if hflags[i] == UInt8(9):
                assert_equal(got[i], UInt32(0xDEADBEEF))
                assert_equal(gotflags[i], UInt8(9))
        print("PASS stable partition pattern", pattern)
    # the launch inputs and scratch are read by launches the downloads above drained
    _ = lids^
    _ = off^
    _ = size^
    _ = flags^
    _ = chunk_zeros^
    _ = chunk_offsets^


# Registered by lane checks-toplevel (2026-11-16): a standalone main() under
# checks/ that no CI job ran. One process per check name in CI keeps any
# fresh-process precondition. Device column only: the suite needs a GPU, so the CPU build never instantiates it.
from checks.native_tests.runner import Check, Column, Report, CPU_COLUMN_BUILD
from checks.native_tests.kit import download, upload, upload_into


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("gbdt_large_partition_check", cpu=False, gpu=True, serial="", run=run_cases)
