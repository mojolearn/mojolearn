# SPDX-License-Identifier: Apache-2.0
"""Opt-in exact tiled-attention-v2 forward primitive.

This stage consumes Q/K/V directly and never materializes scores. It is not
wired into v1. One device thread owns one row so TILE=32 traversal and every
state fold have a single, vendor-independent logical order.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import identical_fmax, identical_mul, identical_mul_add, portable_expf

comptime ATTENTION_V2_TILE = 32

@no_inline
def _v2_mul(a: Float32, b: Float32) -> Float32:
    return identical_mul(a, b)

@no_inline
def _v2_add(a: Float32, b: Float32) -> Float32:
    return identical_mul_add(Float32(1.0), a, b)


def attention_v2_forward_kernel(
    queries: MutPointer[Float32, MutAnyOrigin],
    key_vectors: MutPointer[Float32, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin],
    visible_lo: MutPointer[Int32, MutAnyOrigin],
    visible_hi: MutPointer[Int32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    row_max: MutPointer[Float32, MutAnyOrigin],
    denominator: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    keys_in: Int32,
    width_in: Int32,
    head_dim_in: Int32,
    queries_per_group_in: Int32,
    scale: Float32,
):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var rows = Int(rows_in)
    if row >= rows:
        return
    var keys = Int(keys_in)
    var width = Int(width_in)
    var head_dim = Int(head_dim_in)
    var group = row // Int(queries_per_group_in)
    var lo = Int(visible_lo.unsafe_load(row))
    var hi = Int(visible_hi.unsafe_load(row))
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var m = neg_inf
    var z = Float32(0.0)
    for d in range(width):
        output.unsafe_store(row * width + d, Float32(0.0))
    var tile_lo = lo
    while tile_lo < hi:
        var tile_hi = min(tile_lo + ATTENTION_V2_TILE, hi)
        var tm = neg_inf
        for j in range(tile_lo, tile_hi):
            var score = Float32(0.0)
            for d in range(head_dim):
                score = identical_mul_add(queries.unsafe_load(row * head_dim + d), key_vectors.unsafe_load((group * keys + j) * head_dim + d), score)
            score = _v2_mul(score, scale)
            tm = identical_fmax(tm, score)
        var nm = identical_fmax(m, tm)
        var corr = Float32(0.0) if m == neg_inf else portable_expf(m - nm)
        z = _v2_mul(z, corr)
        for d in range(width):
            var oi = row * width + d
            output.unsafe_store(oi, _v2_mul(output.unsafe_load(oi), corr))
        for j in range(tile_lo, tile_hi):
            var score = Float32(0.0)
            for d in range(head_dim):
                score = identical_mul_add(queries.unsafe_load(row * head_dim + d), key_vectors.unsafe_load((group * keys + j) * head_dim + d), score)
            score = _v2_mul(score, scale)
            var w = portable_expf(score - nm)
            z = _v2_add(w, z)
            for d in range(width):
                var oi = row * width + d
                var vi = (group * keys + j) * width + d
                output.unsafe_store(oi, identical_mul_add(w, values.unsafe_load(vi), output.unsafe_load(oi)))
        m = nm
        tile_lo += ATTENTION_V2_TILE
    for d in range(width):
        var oi = row * width + d
        output.unsafe_store(oi, output.unsafe_load(oi) / z)
    row_max.unsafe_store(row, m)
    denominator.unsafe_store(row, z)


def enqueue_attention_v2_forward(
    ctx: DeviceContext,
    mut queries: DeviceBuffer[DType.float32],
    mut key_vectors: DeviceBuffer[DType.float32],
    mut values: DeviceBuffer[DType.float32],
    mut visible_lo: DeviceBuffer[DType.int32],
    mut visible_hi: DeviceBuffer[DType.int32],
    mut output: DeviceBuffer[DType.float32],
    mut row_max: DeviceBuffer[DType.float32],
    mut denominator: DeviceBuffer[DType.float32],
    rows: Int,
    keys: Int,
    width: Int,
    head_dim: Int,
    queries_per_group: Int,
    scale: Float32,
) raises:
    ctx.enqueue_function[attention_v2_forward_kernel](
        queries.unsafe_ptr(), key_vectors.unsafe_ptr(), values.unsafe_ptr(), visible_lo.unsafe_ptr(),
        visible_hi.unsafe_ptr(), output.unsafe_ptr(), row_max.unsafe_ptr(),
        denominator.unsafe_ptr(), Int32(rows), Int32(keys), Int32(width), Int32(head_dim), Int32(queries_per_group), scale,
        grid_dim=((rows + 63) // 64, 1, 1), block_dim=(64, 1, 1),
    )
