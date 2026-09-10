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


def main() raises:
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
    var hflags = ctx.enqueue_create_host_buffer[DType.uint8](total)
    var hlids = ctx.enqueue_create_host_buffer[DType.uint32](3)
    var hoff = ctx.enqueue_create_host_buffer[DType.uint32](3)
    var hsize = ctx.enqueue_create_host_buffer[DType.uint32](3)
    var got = ctx.enqueue_create_host_buffer[DType.uint32](total)
    var gotflags = ctx.enqueue_create_host_buffer[DType.uint8](total)
    var gotzeros = ctx.enqueue_create_host_buffer[DType.uint32](3)
    var flags = ctx.enqueue_create_buffer[DType.uint8](total)
    var lids = ctx.enqueue_create_buffer[DType.uint32](3)
    var off = ctx.enqueue_create_buffer[DType.uint32](3)
    var size = ctx.enqueue_create_buffer[DType.uint32](3)
    var gmap = ctx.enqueue_create_buffer[DType.uint32](total)
    var sorted_flags = ctx.enqueue_create_buffer[DType.uint8](total)
    var chunks = (large + PARTITION_BLOCK - 1) // PARTITION_BLOCK
    var chunk_zeros = ctx.enqueue_create_buffer[DType.uint32](3 * chunks)
    var chunk_offsets = ctx.enqueue_create_buffer[DType.uint32](3 * chunks)
    var leaf_zeros = ctx.enqueue_create_buffer[DType.uint32](3)
    ctx.synchronize()
    for i in range(3):
        hlids.unsafe_ptr()[i] = UInt32(1 - i if i < 2 else i)
        hoff.unsafe_ptr()[i] = UInt32(offsets[i])
        hsize.unsafe_ptr()[i] = UInt32(sizes[i])
    ctx.enqueue_copy(dst_buf=lids, src_ptr=hlids.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=off, src_ptr=hoff.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=size, src_ptr=hsize.unsafe_ptr())
    for pattern in range(3):
        for i in range(total):
            hflags.unsafe_ptr()[i] = UInt8(9)
        for leaf in range(3):
            for r in range(sizes[leaf]):
                var flag = (r * 13 + r // 7 + leaf) & 1
                if pattern < 2:
                    flag = pattern
                hflags.unsafe_ptr()[offsets[leaf] + r] = UInt8(flag)
        ctx.enqueue_copy(dst_buf=flags, src_ptr=hflags.unsafe_ptr())
        ctx.enqueue_memset(gmap, UInt32(0xDEADBEEF))
        ctx.enqueue_memset(sorted_flags, UInt8(9))
        launch_stable_partition_routed[IDENTICAL](
            ctx, 3, large, lids, off, size, flags, chunk_zeros,
            chunk_offsets, leaf_zeros, gmap, sorted_flags, sm_count=4,
        )
        ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=gmap)
        ctx.enqueue_copy(dst_ptr=gotflags.unsafe_ptr(), src_buf=sorted_flags)
        ctx.enqueue_copy(dst_ptr=gotzeros.unsafe_ptr(), src_buf=leaf_zeros)
        ctx.synchronize()
        for leaf in range(3):
            var cursor = offsets[leaf]
            var zero_count = 0
            for wanted in range(2):
                for r in range(sizes[leaf]):
                    if Int(hflags.unsafe_ptr()[offsets[leaf] + r]) == wanted:
                        assert_equal(got.unsafe_ptr()[cursor], UInt32(r))
                        assert_equal(gotflags.unsafe_ptr()[cursor], UInt8(wanted))
                        cursor += 1
                        if wanted == 0:
                            zero_count += 1
            var slot = 1 - leaf if leaf < 2 else leaf
            assert_equal(gotzeros.unsafe_ptr()[slot], UInt32(zero_count))
        for i in range(total):
            if hflags.unsafe_ptr()[i] == UInt8(9):
                assert_equal(got.unsafe_ptr()[i], UInt32(0xDEADBEEF))
                assert_equal(gotflags.unsafe_ptr()[i], UInt8(9))
        print("PASS stable partition pattern", pattern)
