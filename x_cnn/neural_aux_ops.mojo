# SPDX-License-Identifier: Apache-2.0
"""Uncompiled opt-in neural message-passing and channel-dropout schedules.

NI55/58/60 change ownership of independent cells, preserving each cell's exact
operation sequence. NI59 uses one supported thread-block barrier to share the
existing channel draw. The public mask stays dense and unchanged.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz, identical_mul, identical_div
from x_cnn.ops import FP, IP, dropout2d_mask_val
from x_cnn.neural_aux_contract import (
    NEURAL_AUX_FEATURE_TILE, NI57_GRAPH_TREE64, graph_reduce_tree64,
)


@always_inline
def spmm_feature4_at(tile: Int, vals: FP, h: FP, dst: FP, unused: FP, q: IP, p: IP):
    var n = Int(p.unsafe_load(0))
    var features = Int(p.unsafe_load(1))
    var mode = Int(p.unsafe_load(3))
    var tiles = (features + NEURAL_AUX_FEATURE_TILE - 1) // NEURAL_AUX_FEATURE_TILE
    var row = tile // tiles
    var first = (tile % tiles) * NEURAL_AUX_FEATURE_TILE
    var lo = Int(q.unsafe_load(row))
    var hi = Int(q.unsafe_load(row + 1))
    if mode == 3:
        comptime for lane in range(NEURAL_AUX_FEATURE_TILE):
            if first + lane < features:
                var i = row * features + first + lane
                var value = Float32(0)
                if hi > lo:
                    value = ftz(identical_div(ftz(h.unsafe_load(i)), Float32(hi - lo)))
                dst.unsafe_store(i, value)
        return
    comptime if NI57_GRAPH_TREE64:
        # Both flags use the NI57 graph; do not accidentally restore the old
        # arithmetic when the independent feature-schedule candidate is enabled.
        comptime for lane in range(NEURAL_AUX_FEATURE_TILE):
            if first + lane < features:
                dst.unsafe_store(row * features + first + lane,
                    graph_reduce_tree64(row, first + lane, n, features, mode, vals, h, q))
        return
    var acc = SIMD[DType.float32, NEURAL_AUX_FEATURE_TILE](0.0)
    for edge in range(lo, hi):
        var column = Int(q.unsafe_load(n + 1 + edge))
        var weight = Float32(0)
        if mode == 0 or mode == 2:
            weight = ftz(vals.unsafe_load(edge))
        comptime for lane in range(NEURAL_AUX_FEATURE_TILE):
            if first + lane < features:
                var value = ftz(h.unsafe_load(column * features + first + lane))
                if mode == 0:
                    value = ftz(identical_mul(weight, value))
                elif mode == 2:
                    value = ftz(identical_div(value, weight))
                acc[lane] = ftz(acc[lane] + value)
    comptime for lane in range(NEURAL_AUX_FEATURE_TILE):
        if first + lane < features:
            var value = acc[lane]
            if mode == 1 and hi > lo:
                value = ftz(identical_div(value, Float32(hi - lo)))
            dst.unsafe_store(row * features + first + lane, value)


@always_inline
def sage_max_feature4_at(tile: Int, h: FP, unused: FP, aux: FP, dst: FP, q: IP, p: IP):
    var n = Int(p.unsafe_load(0))
    var features = Int(p.unsafe_load(1))
    var tiles = (features + NEURAL_AUX_FEATURE_TILE - 1) // NEURAL_AUX_FEATURE_TILE
    var row = tile // tiles
    var first = (tile % tiles) * NEURAL_AUX_FEATURE_TILE
    var lo = Int(q.unsafe_load(row))
    var hi = Int(q.unsafe_load(row + 1))
    var maxima = SIMD[DType.float32, NEURAL_AUX_FEATURE_TILE](0.0)
    var counts = SIMD[DType.int32, NEURAL_AUX_FEATURE_TILE](0)
    for edge in range(lo, hi):
        var column = Int(q.unsafe_load(n + 1 + edge))
        comptime for lane in range(NEURAL_AUX_FEATURE_TILE):
            if first + lane < features:
                var value = ftz(h.unsafe_load(column * features + first + lane))
                if edge == lo or value > maxima[lane] or value != value:
                    maxima[lane] = value
                    counts[lane] = 1 if value == value else 0
                elif value == maxima[lane]:
                    counts[lane] += 1
    comptime for lane in range(NEURAL_AUX_FEATURE_TILE):
        if first + lane < features:
            var i = row * features + first + lane
            aux.unsafe_store(i, maxima[lane])
            aux.unsafe_store(n * features + i, Float32(counts[lane]))
            dst.unsafe_store(i, maxima[lane])


@always_inline
def dropout_apply4_at(tile: Int, x: FP, mask: FP, dst: FP, table: FP, unused: IP, p: IP):
    var spatial = Int(p.unsafe_load(2))
    var tiles = (spatial + NEURAL_AUX_FEATURE_TILE - 1) // NEURAL_AUX_FEATURE_TILE
    var channel = tile // tiles
    var first = (tile % tiles) * NEURAL_AUX_FEATURE_TILE
    var value = table.unsafe_load(channel)
    comptime for lane in range(NEURAL_AUX_FEATURE_TILE):
        if first + lane < spatial:
            var i = channel * spatial + first + lane
            mask.unsafe_store(i, value)
            dst.unsafe_store(i, ftz(identical_mul(ftz(x.unsafe_load(i)), value)))


def dropout_channel_kernel(x: FP, mask: FP, dst: FP, hyper: FP, p: IP):
    """One block owns a channel, independent of spatial shape or board rows."""
    var channel = Int(block_idx.x)
    var spatial = Int(p.unsafe_load(2))
    var lane = Int(thread_idx.x)
    var shared_mask = stack_allocation[1, Float32, address_space=AddressSpace.SHARED]()
    if lane == 0:
        shared_mask[0] = dropout2d_mask_val(channel, hyper, p)
    barrier()
    var value = shared_mask[0]
    var offset = lane
    while offset < spatial:
        var i = channel * spatial + offset
        mask.unsafe_store(i, value)
        dst.unsafe_store(i, ftz(identical_mul(ftz(x.unsafe_load(i)), value)))
        offset += Int(block_dim.x)
