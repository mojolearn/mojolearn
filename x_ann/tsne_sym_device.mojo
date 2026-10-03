# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`tsne_symmetrize` on the device (lane cgr4-download-loop, 2026-10-03).

The t-SNE fit used to download the k-NN graph and its conditional
probabilities, symmetrize them on the host and upload the CSR. Here every
step runs on the device, in parallel, and gives the host function's CSR:

  1. the 2 * n * nn directed entries (i -> j for every slot of row i, and
     the reverse j -> i), each named by its entry number e;
  2. two stable device radix sorts (by column, then by row) put them in
     (row, column) order;
  3. a boundary flag, an exclusive scan and an emit keep one edge per
     (row, column), with `ftz(a + b)`: a = row i's probability for j, b =
     row j's probability for i (the LAST matching slot, 0.0 when none),
     exactly the host loops' values;
  4. the row pointers by a device histogram and scan;
  5. the total by `core/device_fold.mojo`'s fixed fold (the host function
     now folds the same order, `host_sum_f32_fixed`), clamped below at
     2^-23, and every value divided by it with `identical_div`.

All integer except steps 3 and 5, whose float operations are the host
function's, operand for operand.
"""

from std.atomic import Atomic
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz, identical_div
from core.device_fold import device_count_nonzero_i32, device_exclusive_scan_total, device_sum_f32_fixed
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len, frs_exclusive_scan, frs_scan_blocks

comptime _TPB = 256
comptime _I32P = MutPointer[Int32, MutAnyOrigin]
comptime _U32P = MutPointer[UInt32, MutAnyOrigin]
comptime _F32P = MutPointer[Float32, MutAnyOrigin]


def _g(n: Int) -> Int:
    return max((n + _TPB - 1) // _TPB, 1)


@always_inline
def _entry_row(e: Int, nn_i: _I32P, nn: Int) -> Int:
    var t = e // 2
    return t // nn if e % 2 == 0 else Int(nn_i[t])


@always_inline
def _entry_col(e: Int, nn_i: _I32P, nn: Int) -> Int:
    var t = e // 2
    return Int(nn_i[t]) if e % 2 == 0 else t // nn


def _col_keys_kernel(nn_i: _I32P, m_in: Int32, nn_in: Int32, keys: _U32P, vals: _U32P):
    var e = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if e < Int(m_in):
        keys[e] = UInt32(_entry_col(e, nn_i, Int(nn_in)))
        vals[e] = UInt32(e)


def _row_keys_kernel(nn_i: _I32P, m_in: Int32, nn_in: Int32, vals: _U32P, keys: _U32P):
    var k = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if k < Int(m_in):
        keys[k] = UInt32(_entry_row(Int(vals[k]), nn_i, Int(nn_in)))


def _edge_flag_kernel(nn_i: _I32P, m_in: Int32, nn_in: Int32, vals: _U32P, flag: _I32P, scan: _I32P):
    var k = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if k >= Int(m_in):
        return
    var nn = Int(nn_in)
    var f = Int32(1)
    if k > 0:
        var e = Int(vals[k])
        var e0 = Int(vals[k - 1])
        if _entry_row(e, nn_i, nn) == _entry_row(e0, nn_i, nn) and _entry_col(e, nn_i, nn) == _entry_col(e0, nn_i, nn):
            f = Int32(0)
    flag[k] = f
    scan[k] = f


def _edge_emit_kernel(
    nn_i: _I32P, p: _F32P, m_in: Int32, nn_in: Int32, vals: _U32P, flag: _I32P, pos: _I32P,
    indices: _I32P, values: _F32P, rowcnt: _I32P,
):
    var k = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if k >= Int(m_in) or flag[k] == 0:
        return
    var nn = Int(nn_in)
    var e = Int(vals[k])
    var i = _entry_row(e, nn_i, nn)
    var j = _entry_col(e, nn_i, nn)
    var a = Float32(0.0)
    var b = Float32(0.0)
    for s in range(nn):
        if Int(nn_i[i * nn + s]) == j:
            a = p[i * nn + s]
        if Int(nn_i[j * nn + s]) == i:
            b = p[j * nn + s]
    var at = Int(pos[k])
    indices[at] = Int32(j)
    values[at] = ftz(a + b)
    _ = Atomic.fetch_add(rowcnt.unsafe_offset(i), Int32(1))


def _zero_kernel(buf: _I32P, n_in: Int32):
    var i = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if i < Int(n_in):
        buf[i] = Int32(0)


def _normalize_kernel(values: _F32P, n_in: Int32, total: Float32):
    var i = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if i < Int(n_in):
        values[i] = ftz(identical_div(values[i], total))


struct TsneGraph(Movable):
    """The symmetrized P as a device CSR."""

    var indptr: DeviceBuffer[DType.int32]
    var indices: DeviceBuffer[DType.int32]
    var values: DeviceBuffer[DType.float32]
    var nnz: Int

    def __init__(
        out self,
        var indptr: DeviceBuffer[DType.int32],
        var indices: DeviceBuffer[DType.int32],
        var values: DeviceBuffer[DType.float32],
        nnz: Int,
    ):
        self.indptr = indptr^
        self.indices = indices^
        self.values = values^
        self.nnz = nnz


def tsne_symmetrize_device(
    ctx: DeviceContext,
    mut nn_i: DeviceBuffer[DType.int32],
    mut p_cond: DeviceBuffer[DType.float32],
    n: Int,
    nn: Int,
) raises -> TsneGraph:
    """`tsne_symmetrize`'s CSR, on the device (the module docstring)."""
    var m = 2 * n * nn
    var keys = ctx.enqueue_create_buffer[DType.uint32](max(m, 1))
    var vals = ctx.enqueue_create_buffer[DType.uint32](max(m, 1))
    var tk = ctx.enqueue_create_buffer[DType.uint32](max(m, 1))
    var tv = ctx.enqueue_create_buffer[DType.uint32](max(m, 1))
    var cnt = ctx.enqueue_create_buffer[DType.int32](max(frs_counts_len(m), 1))
    ctx.enqueue_function[_col_keys_kernel](
        nn_i.unsafe_ptr(), Int32(m), Int32(nn), keys.unsafe_ptr(), vals.unsafe_ptr(),
        grid_dim=_g(m), block_dim=_TPB,
    )
    fast_radix_sort_pairs_u32(ctx, m, keys, vals, tk, tv, cnt)
    ctx.enqueue_function[_row_keys_kernel](
        nn_i.unsafe_ptr(), Int32(m), Int32(nn), vals.unsafe_ptr(), keys.unsafe_ptr(),
        grid_dim=_g(m), block_dim=_TPB,
    )
    fast_radix_sort_pairs_u32(ctx, m, keys, vals, tk, tv, cnt)
    var flag = ctx.enqueue_create_buffer[DType.int32](max(m, 1))
    var pos = ctx.enqueue_create_buffer[DType.int32](max(m, 1))
    ctx.enqueue_function[_edge_flag_kernel](
        nn_i.unsafe_ptr(), Int32(m), Int32(nn), vals.unsafe_ptr(), flag.unsafe_ptr(), pos.unsafe_ptr(),
        grid_dim=_g(m), block_dim=_TPB,
    )
    var bsum = ctx.enqueue_create_buffer[DType.int32](max(frs_scan_blocks(m), 1))
    frs_exclusive_scan(ctx, pos, m, bsum)
    var nnz = device_count_nonzero_i32(ctx, flag, m)
    var indices = ctx.enqueue_create_buffer[DType.int32](max(nnz, 1))
    var values = ctx.enqueue_create_buffer[DType.float32](max(nnz, 1))
    var rowcnt = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var indptr = ctx.enqueue_create_buffer[DType.int32](n + 1)
    ctx.enqueue_function[_zero_kernel](rowcnt.unsafe_ptr(), Int32(n + 1), grid_dim=_g(n + 1), block_dim=_TPB)
    ctx.enqueue_function[_edge_emit_kernel](
        nn_i.unsafe_ptr(), p_cond.unsafe_ptr(), Int32(m), Int32(nn), vals.unsafe_ptr(), flag.unsafe_ptr(),
        pos.unsafe_ptr(), indices.unsafe_ptr(), values.unsafe_ptr(), rowcnt.unsafe_ptr(),
        grid_dim=_g(m), block_dim=_TPB,
    )
    device_exclusive_scan_total(ctx, rowcnt.unsafe_ptr(), indptr, n)
    var total = device_sum_f32_fixed(ctx, values, nnz)
    if total < Float32(1.1920929e-07):
        total = Float32(1.1920929e-07)
    if nnz > 0:
        ctx.enqueue_function[_normalize_kernel](
            values.unsafe_ptr(), Int32(nnz), total, grid_dim=_g(nnz), block_dim=_TPB,
        )
    ctx.synchronize()
    _ = keys^
    _ = vals^
    _ = tk^
    _ = tv^
    _ = cnt^
    _ = flag^
    _ = pos^
    _ = bsum^
    _ = rowcnt^
    return TsneGraph(indptr^, indices^, values^, nnz)
