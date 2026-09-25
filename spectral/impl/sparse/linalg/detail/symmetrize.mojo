# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`raft/sparse/linalg/detail/symmetrize.cuh::coo_symmetrize_kernel`
(`:44-115`) and the HANDLE overload of `coo_symmetrize` (`:168-234`), for
the kNN connectivity graph. (There are two `coo_symmetrize` overloads; the
one cuVS calls takes `raft::resources` and a `device_coo_matrix_view`,
`spectral_embedding.cu:90-93`. A previous version of this header cited
`:44-107` and `:119-`, which are the kernel truncated and the OTHER
overload.)

The kernel is ONE THREAD PER ROW of a ROW-SORTED input COO: for each entry
`(r, c, v)` of the row it looks up the transposed entry `(c, r)` inside row
`c`'s segment (`found_match`), forms `res = reduction_op(r, c, v, transpose)`
-- cuVS passes `0.5f * (a + b)` (`spectral_embedding.cuh:186-188`) -- and
writes `(c, r, res)` when no transpose existed and `v != 0`, then `(r, c,
res)` when `res != 0`, into an output of `2 * nnz` slots starting at `2 *
start_idx`. **THE CALLER ZERO-FILLS THAT OUTPUT AND IT IS THEIRS**:
`coo_symmetrize` runs three `raft::matrix::fill`s over `out_nnz = 2 * nnz`
rows, cols and values (`:205-209`) before the launch, so the slots no row
wrote come back as `(0, 0, 0.0)` and the caller compacts them with
`coo_sort` then `coo_remove_scalar(0)`. DEVIATION 777 exists because a
repeated-key refusal was once spelled inside `coo_sort`, where THIS
padding lives. No float fold anywhere: one add and one multiply per
entry, on values that for the kNN graph are exactly `0`, `0.5` or `1`.

`get_stop_idx(row, m, nnz, ind)` (`raft/sparse/detail/utils.h:97-105`) is
`ind[row + 1]` for every row but the last, which stops at `nnz`;
`coo_symmetrize` allocates `in_row_ind` at length `n` with no terminator
(`:188`). Ours carries the `n + 1` form and reads `row_ind[row + 1]`
always, the same numbers.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from spectral.impl.sparse.coo import CooGraph
from spectral.checks.device_io import download_f32, download_i32, upload_f32, upload_i32
from spectral.impl.sparse.op.coo_ops import sorted_coo_to_csr

comptime SYMMETRIZE_TPB = 128

#: FAST on Apple: each row's (column << 12 | position) keys are sorted once
#: (`fs_sort_rows_kernel`, a bitonic sort in threadgroup memory) and the
#: transpose lookup is a binary search instead of a scan of the whole row
#: (k = 3,000 neighbors per row made the scan 22 s at 30,000 rows). The
#: search returns the LOWEST position among equal columns, the entry the
#: ascending scan stops at, so the output is the same. Rows longer than
#: FS_MAX_ROW or n >= 2^20 keep the scan.
#: `-D MOJOLEARN_SYMMETRIZE_FAST_OFF` keeps the scan.
comptime SYMMETRIZE_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SYMMETRIZE_FAST_OFF"]()
)
comptime FS_MAX_ROW = 4096
comptime FS_TPB = 1024


def fs_sort_rows_kernel(
    row_ind: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    skeys: MutPointer[UInt32, MutAnyOrigin],
    kp_in: Int32,
):
    var row = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var kp = Int(kp_in)
    var start = Int(row_ind[row])
    var ln = Int(row_ind[row + 1]) - start
    var sh = stack_allocation[
        FS_MAX_ROW, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var i = t
    while i < kp:
        sh[i] = (UInt32(Int(cols[start + i])) << 12) | UInt32(i) if i < ln else UInt32(0xFFFFFFFF)
        i += FS_TPB
    barrier()
    var size = 2
    while size <= kp:
        var stride = size // 2
        while stride > 0:
            var e = t
            while e < kp // 2:
                var lo = 2 * e - (e & (stride - 1))
                var hi = lo + stride
                var up = (lo & size) == 0
                var a = sh[lo]
                var b = sh[hi]
                if (a > b) == up:
                    sh[lo] = b
                    sh[hi] = a
                e += FS_TPB
            barrier()
            stride //= 2
        size *= 2
    i = t
    while i < ln:
        skeys[row * kp + i] = sh[i]
        i += FS_TPB


def fs_symmetrize_kernel(
    row_ind: MutPointer[Int32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    skeys: MutPointer[UInt32, MutAnyOrigin],
    orows: MutPointer[Int32, MutAnyOrigin],
    ocols: MutPointer[Int32, MutAnyOrigin],
    ovals: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    kp_in: Int32,
):
    """`coo_symmetrize_kernel` with the transpose found by binary search in
    the lookup row's sorted keys."""
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    if row >= n:
        return
    var kp = Int(kp_in)
    var start_idx = Int(row_ind.unsafe_load(row))
    var stop_idx = Int(row_ind.unsafe_load(row + 1))
    var row_nnz = 0
    var out_start_idx = start_idx * 2
    for idx in range(0, stop_idx - start_idx):
        var cur_row = rows.unsafe_load(start_idx + idx)
        var cur_col = cols.unsafe_load(start_idx + idx)
        var cur_val = vals.unsafe_load(start_idx + idx)
        var lookup_row = Int(cur_col)
        var t_start = Int(row_ind.unsafe_load(lookup_row))
        var t_len = Int(row_ind.unsafe_load(lookup_row + 1)) - t_start
        var target = UInt32(Int(cur_row)) << 12
        var lo = 0
        var hi = t_len
        var base = lookup_row * kp
        while lo < hi:
            var mid = (lo + hi) // 2
            if skeys[base + mid] < target:
                lo = mid + 1
            else:
                hi = mid
        var transpose = Float32(0.0)
        var found_match = False
        if lo < t_len:
            var key = skeys[base + lo]
            if (key >> 12) == UInt32(Int(cur_row)):
                var t_idx = t_start + Int(key & UInt32(0xFFF))
                if rows.unsafe_load(t_idx) == cur_col:
                    transpose = vals.unsafe_load(t_idx)
                    found_match = True
        var res = ftz(Float32(0.5) * ftz(cur_val + transpose))
        if (not found_match) and cur_val != Float32(0.0):
            orows.unsafe_store(out_start_idx + row_nnz, cur_col)
            ocols.unsafe_store(out_start_idx + row_nnz, cur_row)
            ovals.unsafe_store(out_start_idx + row_nnz, res)
            row_nnz += 1
        if res != Float32(0.0):
            orows.unsafe_store(out_start_idx + row_nnz, cur_row)
            ocols.unsafe_store(out_start_idx + row_nnz, cur_col)
            ovals.unsafe_store(out_start_idx + row_nnz, res)
            row_nnz += 1  # `coo_symmetrize<128, ...>` at the call site


def coo_symmetrize_kernel(
    row_ind: MutPointer[Int32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    orows: MutPointer[Int32, MutAnyOrigin],
    ocols: MutPointer[Int32, MutAnyOrigin],
    ovals: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`symmetrize.cuh:44-107` with cuVS's `0.5f * (a + b)` reducer inlined
    (a lambda in theirs; the only reducer this lane reaches)."""
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    if row >= n:
        return
    var start_idx = Int(row_ind.unsafe_load(row))
    var stop_idx = Int(row_ind.unsafe_load(row + 1))
    var row_nnz = 0
    var out_start_idx = start_idx * 2
    for idx in range(0, stop_idx - start_idx):
        var cur_row = rows.unsafe_load(start_idx + idx)
        var cur_col = cols.unsafe_load(start_idx + idx)
        var cur_val = vals.unsafe_load(start_idx + idx)
        var lookup_row = Int(cur_col)
        var t_start = Int(row_ind.unsafe_load(lookup_row))
        var t_stop = Int(row_ind.unsafe_load(lookup_row + 1))
        var transpose = Float32(0.0)
        var found_match = False
        for t_idx in range(t_start, t_stop):
            if (
                cols.unsafe_load(t_idx) == cur_row
                and rows.unsafe_load(t_idx) == cur_col
            ):
                transpose = vals.unsafe_load(t_idx)
                found_match = True
                break
        # reduction_op(row, col, a, b) = 0.5f * (a + b)
        var res = ftz(Float32(0.5) * ftz(cur_val + transpose))
        if (not found_match) and cur_val != Float32(0.0):
            orows.unsafe_store(out_start_idx + row_nnz, cur_col)
            ocols.unsafe_store(out_start_idx + row_nnz, cur_row)
            ovals.unsafe_store(out_start_idx + row_nnz, res)
            row_nnz += 1
        if res != Float32(0.0):
            orows.unsafe_store(out_start_idx + row_nnz, cur_row)
            ocols.unsafe_store(out_start_idx + row_nnz, cur_col)
            ovals.unsafe_store(out_start_idx + row_nnz, res)
            row_nnz += 1


def coo_symmetrize(
    ctx: DeviceContext, g_in: CooGraph, tpb: Int = SYMMETRIZE_TPB
) raises -> CooGraph:
    """`coo_symmetrize` (`symmetrize.cuh:119-`): `sorted_coo_to_csr` of the
    ROW-SORTED input, a zero-filled `2 * nnz` output, the kernel, then the
    caller's `coo_sort` + `coo_remove_scalar(0)` (`spectral_embedding.cuh:
    190-204`) -- both of which live in `coo_ops.mojo` and are applied by
    the caller, not here. Returns the RAW kernel output (with its zero
    slots), because that is what theirs returns."""
    var n = g_in.n
    var nnz = g_in.nnz()
    var row_ind = sorted_coo_to_csr(g_in)
    var d_row_ind = upload_i32(ctx, row_ind)
    var d_rows = upload_i32(ctx, g_in.rows)
    var d_cols = upload_i32(ctx, g_in.cols)
    var d_vals = upload_f32(ctx, g_in.vals)
    var out_n = 2 * nnz if nnz > 0 else 1
    var d_orows = ctx.enqueue_create_buffer[DType.int32](out_n)
    var d_ocols = ctx.enqueue_create_buffer[DType.int32](out_n)
    var d_ovals = ctx.enqueue_create_buffer[DType.float32](out_n)
    ctx.enqueue_memset(d_orows, Int32(0))
    ctx.enqueue_memset(d_ocols, Int32(0))
    ctx.enqueue_memset(d_ovals, Float32(0.0))
    var fast_done = False
    comptime if SYMMETRIZE_FAST:
        var max_len = 0
        for r in range(n):
            var l = Int(row_ind[r + 1]) - Int(row_ind[r])
            if l > max_len:
                max_len = l
        if max_len > 0 and max_len <= FS_MAX_ROW and n < (1 << 20):
            var kp = 2
            while kp < max_len:
                kp *= 2
            var d_skeys = ctx.enqueue_create_buffer[DType.uint32](n * kp)
            ctx.enqueue_function[fs_sort_rows_kernel](
                d_row_ind.unsafe_ptr(), d_cols.unsafe_ptr(), d_skeys.unsafe_ptr(),
                Int32(kp), grid_dim=(n, 1, 1), block_dim=(FS_TPB, 1, 1),
            )
            ctx.enqueue_function[fs_symmetrize_kernel](
                d_row_ind.unsafe_ptr(), d_rows.unsafe_ptr(), d_cols.unsafe_ptr(),
                d_vals.unsafe_ptr(), d_skeys.unsafe_ptr(), d_orows.unsafe_ptr(),
                d_ocols.unsafe_ptr(), d_ovals.unsafe_ptr(), Int32(n), Int32(kp),
                grid_dim=((n + tpb - 1) // tpb, 1, 1), block_dim=(tpb, 1, 1),
            )
            ctx.synchronize()
            _ = d_skeys^
            fast_done = True
    if not fast_done:
        ctx.enqueue_function[coo_symmetrize_kernel](
            d_row_ind.unsafe_ptr(),
            d_rows.unsafe_ptr(),
            d_cols.unsafe_ptr(),
            d_vals.unsafe_ptr(),
            d_orows.unsafe_ptr(),
            d_ocols.unsafe_ptr(),
            d_ovals.unsafe_ptr(),
            Int32(n),
            grid_dim=((n + tpb - 1) // tpb, 1, 1),
            block_dim=(tpb, 1, 1),
        )
    var orows = download_i32(ctx, d_orows, 2 * nnz)
    var ocols = download_i32(ctx, d_ocols, 2 * nnz)
    var ovals = download_f32(ctx, d_ovals, 2 * nnz)
    _ = d_row_ind^
    _ = d_rows^
    _ = d_cols^
    _ = d_vals^
    _ = d_orows^
    _ = d_ocols^
    _ = d_ovals^
    return CooGraph(n, orows^, ocols^, ovals^)
