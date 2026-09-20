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


def _v2_score(
    queries: MutPointer[Float32, MutAnyOrigin],
    key_vectors: MutPointer[Float32, MutAnyOrigin],
    row: Int, group: Int, key: Int, keys: Int, head_dim: Int, scale: Float32,
) -> Float32:
    var score = Float32(0.0)
    for d in range(head_dim):
        score = identical_mul_add(
            queries.unsafe_load(row * head_dim + d),
            key_vectors.unsafe_load((group * keys + key) * head_dim + d), score,
        )
    return _v2_mul(score, scale)


def _v2_normalizer(
    queries: MutPointer[Float32, MutAnyOrigin],
    key_vectors: MutPointer[Float32, MutAnyOrigin],
    row: Int, group: Int, lo: Int, hi: Int, keys: Int, head_dim: Int,
    scale: Float32,
) -> Tuple[Float32, Float32]:
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var m = neg_inf
    var z = Float32(0.0)
    var tile_lo = lo
    while tile_lo < hi:
        var tile_hi = min(tile_lo + ATTENTION_V2_TILE, hi)
        var tm = neg_inf
        for j in range(tile_lo, tile_hi):
            tm = identical_fmax(tm, _v2_score(queries, key_vectors, row, group, j, keys, head_dim, scale))
        var nm = identical_fmax(m, tm)
        var corr = Float32(0.0) if m == neg_inf else portable_expf(m - nm)
        z = _v2_mul(z, corr)
        for j in range(tile_lo, tile_hi):
            z = _v2_add(portable_expf(_v2_score(queries, key_vectors, row, group, j, keys, head_dim, scale) - nm), z)
        m = nm
        tile_lo += ATTENTION_V2_TILE
    return m, z


def _v2_dyv(
    values: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    row: Int, group: Int, key: Int, keys: Int, width: Int,
) -> Float32:
    var acc = Float32(0.0)
    for d in range(width):
        acc = identical_mul_add(
            dy.unsafe_load(row * width + d),
            values.unsafe_load((group * keys + key) * width + d), acc,
        )
    return acc


def _v2_zdot(
    queries: MutPointer[Float32, MutAnyOrigin],
    key_vectors: MutPointer[Float32, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    row: Int, group: Int, lo: Int, hi: Int, keys: Int, width: Int,
    head_dim: Int, scale: Float32, m: Float32, z: Float32,
) -> Float32:
    var acc = Float32(0.0)
    for j in range(lo, hi):
        var p = portable_expf(_v2_score(queries, key_vectors, row, group, j, keys, head_dim, scale) - m) / z
        acc = identical_mul_add(p, _v2_dyv(values, dy, row, group, j, keys, width), acc)
    return acc


def attention_v2_dq_kernel(
    queries: MutPointer[Float32, MutAnyOrigin], key_vectors: MutPointer[Float32, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin], dy: MutPointer[Float32, MutAnyOrigin],
    visible_lo: MutPointer[Int32, MutAnyOrigin], visible_hi: MutPointer[Int32, MutAnyOrigin],
    row_max: MutPointer[Float32, MutAnyOrigin], denominator: MutPointer[Float32, MutAnyOrigin],
    row_zdot: MutPointer[Float32, MutAnyOrigin],
    dq: MutPointer[Float32, MutAnyOrigin], rows_in: Int32, keys_in: Int32,
    width_in: Int32, head_dim_in: Int32, queries_per_group_in: Int32, scale: Float32,
):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var rows = Int(rows_in)
    if row >= rows: return
    var keys = Int(keys_in); var width = Int(width_in); var hd = Int(head_dim_in)
    var group = row // Int(queries_per_group_in)
    var lo = Int(visible_lo.unsafe_load(row)); var hi = Int(visible_hi.unsafe_load(row))
    var m=row_max.unsafe_load(row); var z=denominator.unsafe_load(row); var zdot=row_zdot.unsafe_load(row)
    for d in range(hd): dq.unsafe_store(row * hd + d, Float32(0.0))
    for j in range(lo, hi):
        var p = portable_expf(_v2_score(queries, key_vectors, row, group, j, keys, hd, scale) - m) / z
        var ds = _v2_mul(_v2_mul(p, _v2_dyv(values, dy, row, group, j, keys, width) - zdot), scale)
        for d in range(hd):
            var oi = row * hd + d
            dq.unsafe_store(oi, identical_mul_add(ds, key_vectors.unsafe_load((group * keys + j) * hd + d), dq.unsafe_load(oi)))


def attention_v2_dk_kernel(
    queries: MutPointer[Float32, MutAnyOrigin], key_vectors: MutPointer[Float32, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin], dy: MutPointer[Float32, MutAnyOrigin],
    visible_lo: MutPointer[Int32, MutAnyOrigin], visible_hi: MutPointer[Int32, MutAnyOrigin],
    dk: MutPointer[Float32, MutAnyOrigin], rows_in: Int32, keys_in: Int32,
    width_in: Int32, head_dim_in: Int32, queries_per_group_in: Int32, scale: Float32,
):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var keys = Int(keys_in); var hd = Int(head_dim_in); var rows = Int(rows_in)
    var groups = (rows + Int(queries_per_group_in) - 1) // Int(queries_per_group_in)
    if idx >= groups * keys * hd: return
    var d = idx % hd; var key = (idx // hd) % keys; var group = idx // (hd * keys)
    var begin = group * Int(queries_per_group_in); var end = min(begin + Int(queries_per_group_in), rows)
    var acc = Float32(0.0)
    for row in range(begin, end):
        var lo = Int(visible_lo.unsafe_load(row)); var hi = Int(visible_hi.unsafe_load(row))
        if key >= lo and key < hi:
            var norm = _v2_normalizer(queries, key_vectors, row, group, lo, hi, keys, hd, scale)
            var zdot = _v2_zdot(queries, key_vectors, values, dy, row, group, lo, hi, keys, Int(width_in), hd, scale, norm[0], norm[1])
            var p = portable_expf(_v2_score(queries, key_vectors, row, group, key, keys, hd, scale) - norm[0]) / norm[1]
            var ds = _v2_mul(_v2_mul(p, _v2_dyv(values, dy, row, group, key, keys, Int(width_in)) - zdot), scale)
            acc = identical_mul_add(ds, queries.unsafe_load(row * hd + d), acc)
    dk.unsafe_store(idx, acc)


def attention_v2_dv_kernel(
    queries: MutPointer[Float32, MutAnyOrigin], key_vectors: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin], visible_lo: MutPointer[Int32, MutAnyOrigin],
    visible_hi: MutPointer[Int32, MutAnyOrigin], dv: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32, keys_in: Int32, width_in: Int32, head_dim_in: Int32,
    queries_per_group_in: Int32, scale: Float32,
):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var keys = Int(keys_in); var width = Int(width_in); var hd = Int(head_dim_in); var rows = Int(rows_in)
    var groups = (rows + Int(queries_per_group_in) - 1) // Int(queries_per_group_in)
    if idx >= groups * keys * width: return
    var d = idx % width; var key = (idx // width) % keys; var group = idx // (width * keys)
    var begin = group * Int(queries_per_group_in); var end = min(begin + Int(queries_per_group_in), rows)
    var acc = Float32(0.0)
    for row in range(begin, end):
        var lo = Int(visible_lo.unsafe_load(row)); var hi = Int(visible_hi.unsafe_load(row))
        if key >= lo and key < hi:
            var norm = _v2_normalizer(queries, key_vectors, row, group, lo, hi, keys, hd, scale)
            var p = portable_expf(_v2_score(queries, key_vectors, row, group, key, keys, hd, scale) - norm[0]) / norm[1]
            acc = identical_mul_add(p, dy.unsafe_load(row * width + d), acc)
    dv.unsafe_store(idx, acc)


def attention_v2_dkdv_kernel(
    queries: MutPointer[Float32, MutAnyOrigin], key_vectors: MutPointer[Float32, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin], dy: MutPointer[Float32, MutAnyOrigin],
    visible_lo: MutPointer[Int32, MutAnyOrigin], visible_hi: MutPointer[Int32, MutAnyOrigin],
    row_max: MutPointer[Float32, MutAnyOrigin], denominator: MutPointer[Float32, MutAnyOrigin],
    row_zdot: MutPointer[Float32, MutAnyOrigin],
    dk: MutPointer[Float32, MutAnyOrigin], dv: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32, keys_in: Int32, width_in: Int32, head_dim_in: Int32,
    queries_per_group_in: Int32, scale: Float32,
):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var keys = Int(keys_in); var width = Int(width_in); var hd = Int(head_dim_in); var rows = Int(rows_in)
    var qpg = Int(queries_per_group_in); var groups = (rows + qpg - 1) // qpg
    if idx >= groups * keys: return
    var key = idx % keys; var group = idx // keys
    for d in range(hd): dk.unsafe_store((group*keys+key)*hd+d, Float32(0.0))
    for d in range(width): dv.unsafe_store((group*keys+key)*width+d, Float32(0.0))
    var begin = group*qpg; var end = min(begin+qpg, rows)
    for row in range(begin,end):
        var lo=Int(visible_lo.unsafe_load(row)); var hi=Int(visible_hi.unsafe_load(row))
        if key>=lo and key<hi:
            var m=row_max.unsafe_load(row);var z=denominator.unsafe_load(row);var zdot=row_zdot.unsafe_load(row)
            var p=portable_expf(_v2_score(queries,key_vectors,row,group,key,keys,hd,scale)-m)/z
            var ds=_v2_mul(_v2_mul(p,_v2_dyv(values,dy,row,group,key,keys,width)-zdot),scale)
            for d in range(hd):
                var oi=(group*keys+key)*hd+d
                dk.unsafe_store(oi,identical_mul_add(ds,queries.unsafe_load(row*hd+d),dk.unsafe_load(oi)))
            for d in range(width):
                var oi=(group*keys+key)*width+d
                dv.unsafe_store(oi,identical_mul_add(p,dy.unsafe_load(row*width+d),dv.unsafe_load(oi)))


def attention_v2_backward_prepare_kernel(
    queries: MutPointer[Float32, MutAnyOrigin], key_vectors: MutPointer[Float32, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin], dy: MutPointer[Float32, MutAnyOrigin],
    visible_lo: MutPointer[Int32, MutAnyOrigin], visible_hi: MutPointer[Int32, MutAnyOrigin],
    row_max: MutPointer[Float32, MutAnyOrigin], denominator: MutPointer[Float32, MutAnyOrigin],
    row_zdot: MutPointer[Float32, MutAnyOrigin], rows_in:Int32,keys_in:Int32,width_in:Int32,
    head_dim_in:Int32,queries_per_group_in:Int32,scale:Float32,
):
    var row=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x);var rows=Int(rows_in)
    if row>=rows:return
    var keys=Int(keys_in);var width=Int(width_in);var hd=Int(head_dim_in);var group=row//Int(queries_per_group_in)
    var lo=Int(visible_lo.unsafe_load(row));var hi=Int(visible_hi.unsafe_load(row))
    var norm=_v2_normalizer(queries,key_vectors,row,group,lo,hi,keys,hd,scale)
    row_max.unsafe_store(row,norm[0]);denominator.unsafe_store(row,norm[1])
    row_zdot.unsafe_store(row,_v2_zdot(queries,key_vectors,values,dy,row,group,lo,hi,keys,width,hd,scale,norm[0],norm[1]))


def enqueue_attention_v2_backward(
    ctx: DeviceContext, mut queries: DeviceBuffer[DType.float32],
    mut key_vectors: DeviceBuffer[DType.float32], mut values: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32], mut visible_lo: DeviceBuffer[DType.int32],
    mut visible_hi: DeviceBuffer[DType.int32], mut dq: DeviceBuffer[DType.float32],
    mut dk: DeviceBuffer[DType.float32], mut dv: DeviceBuffer[DType.float32],
    mut row_max: DeviceBuffer[DType.float32], mut denominator: DeviceBuffer[DType.float32],
    mut row_zdot: DeviceBuffer[DType.float32],
    rows: Int, keys: Int, width: Int, head_dim: Int, queries_per_group: Int, scale: Float32,
) raises:
    var groups = (rows + queries_per_group - 1) // queries_per_group
    ctx.enqueue_function[attention_v2_backward_prepare_kernel](queries.unsafe_ptr(),key_vectors.unsafe_ptr(),values.unsafe_ptr(),dy.unsafe_ptr(),visible_lo.unsafe_ptr(),visible_hi.unsafe_ptr(),row_max.unsafe_ptr(),denominator.unsafe_ptr(),row_zdot.unsafe_ptr(),Int32(rows),Int32(keys),Int32(width),Int32(head_dim),Int32(queries_per_group),scale,grid_dim=((rows+63)//64,1,1),block_dim=(64,1,1))
    ctx.enqueue_function[attention_v2_dq_kernel](queries.unsafe_ptr(), key_vectors.unsafe_ptr(), values.unsafe_ptr(), dy.unsafe_ptr(), visible_lo.unsafe_ptr(), visible_hi.unsafe_ptr(),row_max.unsafe_ptr(),denominator.unsafe_ptr(),row_zdot.unsafe_ptr(), dq.unsafe_ptr(), Int32(rows), Int32(keys), Int32(width), Int32(head_dim), Int32(queries_per_group), scale, grid_dim=((rows+63)//64,1,1), block_dim=(64,1,1))
    ctx.enqueue_function[attention_v2_dkdv_kernel](queries.unsafe_ptr(), key_vectors.unsafe_ptr(), values.unsafe_ptr(), dy.unsafe_ptr(), visible_lo.unsafe_ptr(), visible_hi.unsafe_ptr(),row_max.unsafe_ptr(),denominator.unsafe_ptr(),row_zdot.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(), Int32(rows), Int32(keys), Int32(width), Int32(head_dim), Int32(queries_per_group), scale, grid_dim=((groups*keys+63)//64,1,1), block_dim=(64,1,1))
