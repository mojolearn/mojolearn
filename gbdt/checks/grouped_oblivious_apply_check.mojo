# SPDX-License-Identifier: Apache-2.0
"""Exact analytic gate for four-tree resident oblivious apply grouping."""

from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from gbdt.models.kernel.add_bin_values import (
    compute_bins_and_add_four_kernel,
    compute_bins_and_add_kernel,
)


def main() raises:
    var ctx = DeviceContext()
    comptime N = 1_000_003
    comptime DIM = 2
    var depths: List[Int] = [1, 2, 3, 1]
    var total_levels = 7
    var total_leaves = 16
    var h_ci = ctx.enqueue_create_host_buffer[DType.uint32](N)
    for r in range(N):
        h_ci[r] = UInt32((r * 37 + r // 11) & 255)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](total_levels)
    var h_shift = ctx.enqueue_create_host_buffer[DType.uint32](total_levels)
    var h_mask = ctx.enqueue_create_host_buffer[DType.uint32](total_levels)
    var h_bin = ctx.enqueue_create_host_buffer[DType.uint32](total_levels)
    var h_eq = ctx.enqueue_create_host_buffer[DType.uint8](total_levels)
    for j in range(total_levels):
        h_off[j] = UInt32(0)
        h_shift[j] = UInt32(0)
        h_mask[j] = UInt32(255)
        h_bin[j] = UInt32((j * 29 + 17) & 255)
        h_eq[j] = UInt8(1) if j % 3 == 0 else UInt8(0)
    var h_leaf = ctx.enqueue_create_host_buffer[DType.float32](total_leaves * DIM)
    for i in range(total_leaves * DIM):
        h_leaf[i] = Float32((i * 13) % 31 - 15) / Float32(16.0)
    var d_ci = ctx.enqueue_create_buffer[DType.uint32](N)
    var d_off = ctx.enqueue_create_buffer[DType.uint32](total_levels)
    var d_shift = ctx.enqueue_create_buffer[DType.uint32](total_levels)
    var d_mask = ctx.enqueue_create_buffer[DType.uint32](total_levels)
    var d_bin = ctx.enqueue_create_buffer[DType.uint32](total_levels)
    var d_eq = ctx.enqueue_create_buffer[DType.uint8](total_levels)
    var d_leaf = ctx.enqueue_create_buffer[DType.float32](total_leaves * DIM)
    ctx.enqueue_copy(dst_buf=d_ci, src_buf=h_ci)
    ctx.enqueue_copy(dst_buf=d_off, src_buf=h_off)
    ctx.enqueue_copy(dst_buf=d_shift, src_buf=h_shift)
    ctx.enqueue_copy(dst_buf=d_mask, src_buf=h_mask)
    ctx.enqueue_copy(dst_buf=d_bin, src_buf=h_bin)
    ctx.enqueue_copy(dst_buf=d_eq, src_buf=h_eq)
    ctx.enqueue_copy(dst_buf=d_leaf, src_buf=h_leaf)
    var a = ctx.enqueue_create_buffer[DType.float32](N * DIM)
    var b = ctx.enqueue_create_buffer[DType.float32](N * DIM)
    ctx.enqueue_memset(a, 0)
    ctx.enqueue_memset(b, 0)
    var level = 0
    var leaf = 0
    for t in range(4):
        var depth = depths[t]
        ctx.enqueue_function[compute_bins_and_add_kernel](
            d_ci.unsafe_ptr(), d_off.unsafe_ptr() + level,
            d_shift.unsafe_ptr() + level, d_mask.unsafe_ptr() + level,
            d_bin.unsafe_ptr() + level, d_eq.unsafe_ptr() + level,
            Int32(depth), d_leaf.unsafe_ptr() + leaf * DIM, Int32(N),
            a.unsafe_ptr(), Int32(DIM), Int32(N),
            grid_dim=((N + 255) // 256, DIM, 1), block_dim=(256, 1, 1),
        )
        level += depth
        leaf += 1 << depth
    ctx.enqueue_function[compute_bins_and_add_four_kernel](
        d_ci.unsafe_ptr(), d_off.unsafe_ptr(), d_shift.unsafe_ptr(),
        d_mask.unsafe_ptr(), d_bin.unsafe_ptr(), d_eq.unsafe_ptr(),
        Int32(1), Int32(2), Int32(3), Int32(1), Int32(4),
        d_leaf.unsafe_ptr(), Int32(N), b.unsafe_ptr(), Int32(DIM), Int32(N),
        grid_dim=((N + 255) // 256, DIM, 1), block_dim=(256, 1, 1),
    )
    with a.map_to_host() as ah, b.map_to_host() as bh:
        for i in range(N * DIM):
            if ah[i].to_bits() != bh[i].to_bits():
                raise Error("grouped oblivious apply changed cell " + String(i))
    print("grouped oblivious apply: exact", N * DIM, "cells")
    # One hundred identical ordered trees model a realistic boosted ensemble.
    # Reusing the four-tree descriptor is intentional: this measures launch
    # and cursor traffic rather than model preparation.
    for rep in range(5):
        ctx.enqueue_memset(a, 0)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for group in range(25):
            level = 0
            leaf = 0
            for t in range(4):
                var depth = depths[t]
                ctx.enqueue_function[compute_bins_and_add_kernel](
                    d_ci.unsafe_ptr(), d_off.unsafe_ptr() + level,
                    d_shift.unsafe_ptr() + level, d_mask.unsafe_ptr() + level,
                    d_bin.unsafe_ptr() + level, d_eq.unsafe_ptr() + level,
                    Int32(depth), d_leaf.unsafe_ptr() + leaf * DIM, Int32(N),
                    a.unsafe_ptr(), Int32(DIM), Int32(N),
                    grid_dim=((N + 255) // 256, DIM, 1), block_dim=(256, 1, 1),
                )
                level += depth
                leaf += 1 << depth
        ctx.synchronize()
        var t1 = perf_counter_ns()
        ctx.enqueue_memset(b, 0)
        ctx.synchronize()
        var t2 = perf_counter_ns()
        for group in range(25):
            ctx.enqueue_function[compute_bins_and_add_four_kernel](
                d_ci.unsafe_ptr(), d_off.unsafe_ptr(), d_shift.unsafe_ptr(),
                d_mask.unsafe_ptr(), d_bin.unsafe_ptr(), d_eq.unsafe_ptr(),
                Int32(1), Int32(2), Int32(3), Int32(1), Int32(4),
                d_leaf.unsafe_ptr(), Int32(N), b.unsafe_ptr(), Int32(DIM),
                Int32(N), grid_dim=((N + 255) // 256, DIM, 1),
                block_dim=(256, 1, 1),
            )
        ctx.synchronize()
        var t3 = perf_counter_ns()
        print("TIMING_MS", rep, Float64(t1 - t0) / 1.0e6,
              Float64(t3 - t2) / 1.0e6)
    with a.map_to_host() as ah, b.map_to_host() as bh:
        for i in range(N * DIM):
            if ah[i].to_bits() != bh[i].to_bits():
                raise Error("100-tree grouped apply changed cell " + String(i))
