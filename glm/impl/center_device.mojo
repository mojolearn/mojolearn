# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LinearRegression's and Ridge's centering on the device (lane
hr-small-passes, 2026-10-02): the exact column sums, the center and the
row scale of glm/impl/center_items.mojo, which replace the host pool
helpers of bindings/_mojolearn.mojo (`column_mean_f64`,
`center_columns_f32`, `scale_rows_f32`) on a GPU install.

Column sums: thread (row block, column) adds its CS_RB-row run of one
column into nine exact Int64 digits (`cs_partial_kernel`); block c folds
column c's partials (`cs_fold_kernel`, a threadgroup tree per digit: an
integer sum, so any order is the same total) and its thread 0 rounds the
total once to float64 bits. Center and scale: one thread per cell."""
from experiments.classical_identical_ideas.shared_controls import C02_LINEAR_PAIR
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for
from checks.soft_f64 import sf64_div, sf64_from_int, sf64_to_f32
from glm.impl.center_items import (
    CS_LIMBS, CS_WORDS, exact_add, exact_finish, center_cell, scale_cell,
)

comptime F32P = MutPointer[Float32, MutAnyOrigin]
comptime I64P = MutPointer[Int64, MutAnyOrigin]
comptime U64P = MutPointer[UInt64, MutAnyOrigin]

comptime CS_TPB = 256
#: the fewest rows a partial covers; more when the row-block count would
#: pass CS_MAX_BLOCKS
comptime CS_RB = 256
comptime CS_MAX_BLOCKS = 65536
#: one Int64 per thread plus the ten folded words: 2,128 bytes
comptime CS_SMEM_FITS = lib_smem_page_fits_for[TARGET_COLUMN, CS_TPB * 8 + CS_WORDS * 8]()


def cs_rows_per_block(rows: Int) -> Int:
    var rb = CS_RB
    var need = (rows + CS_MAX_BLOCKS - 1) // CS_MAX_BLOCKS
    if need > rb:
        rb = need
    return rb


def cs_partial_kernel(x: F32P, part: I64P, rows_: Int64, cols_: Int64, rb_: Int64):
    var rows = Int(rows_)
    var cols = Int(cols_)
    var rb = Int(rb_)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nrb = (rows + rb - 1) // rb
    if tid >= nrb * cols:
        return
    var c = tid % cols
    var blk = tid // cols
    var r0 = blk * rb
    var r1 = min(r0 + rb, rows)
    var acc = InlineArray[Int64, CS_WORDS](fill=Int64(0))
    for r in range(r0, r1):
        exact_add(acc, x.unsafe_load(r * cols + c))
    var o = tid * CS_WORDS
    comptime for L in range(CS_WORDS):
        part.unsafe_store(o + L, acc[L])


def cs_fold_kernel(part: I64P, dst: U64P, nrb_: Int64, cols_: Int64):
    """Block c: column c's nrb partials (partial (blk, c) at blk * cols + c)."""
    var nrb = Int(nrb_)
    var cols = Int(cols_)
    var c = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var sh = stack_allocation[CS_TPB, Scalar[DType.int64], address_space=AddressSpace.SHARED]()
    var tot = stack_allocation[CS_WORDS, Scalar[DType.int64], address_space=AddressSpace.SHARED]()
    for L in range(CS_WORDS):
        var s = Int64(0)
        var blk = t
        while blk < nrb:
            var v = part.unsafe_load((blk * cols + c) * CS_WORDS + L)
            if L == CS_LIMBS:
                s = s | v
            else:
                s = s + v
            blk += CS_TPB
        sh[t] = s
        barrier()
        var h = CS_TPB // 2
        while h > 0:
            if t < h:
                if L == CS_LIMBS:
                    sh[t] = sh[t] | sh[t + h]
                else:
                    sh[t] = sh[t] + sh[t + h]
            barrier()
            h //= 2
        if t == 0:
            tot[L] = sh[0]
        barrier()
    if t == 0:
        var acc = InlineArray[Int64, CS_WORDS](fill=Int64(0))
        for L in range(CS_WORDS):
            acc[L] = tot[L]
        dst.unsafe_store(c, exact_finish(acc))


def center_kernel(x: F32P, mu: F32P, dst: F32P, rows_: Int64, cols_: Int64):
    var cols = Int(cols_)
    var total = Int(rows_) * cols
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < total:
        dst.unsafe_store(i, center_cell(x.unsafe_load(i), mu.unsafe_load(i % cols)))


def scale_rows_kernel(x: F32P, w: F32P, dst: F32P, rows_: Int64, cols_: Int64):
    var cols = Int(cols_)
    var total = Int(rows_) * cols
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < total:
        dst.unsafe_store(i, scale_cell(x.unsafe_load(i), w.unsafe_load(i // cols)))


def col_sums_device(ctx: DeviceContext, x: Int, dst: Int, rows: Int, cols: Int) raises:
    """dst (float64[cols], written as its bits): the correctly rounded exact
    sum of every column of the float32 [rows, cols] matrix at x."""
    comptime assert CS_SMEM_FITS, "col_sums_device: a 2 KB threadgroup page must fit"
    if rows <= 0 or cols <= 0:
        raise Error("col_sums: rows and cols must be positive")
    var cells = rows * cols
    var rb = cs_rows_per_block(rows)
    var nrb = (rows + rb - 1) // rb
    var d_x = ctx.enqueue_create_buffer[DType.float32](cells)
    ctx.enqueue_copy(dst_buf=d_x, src_ptr=F32P(unsafe_from_address=x))
    var d_p = ctx.enqueue_create_buffer[DType.int64](nrb * cols * CS_WORDS)
    var d_o = ctx.enqueue_create_buffer[DType.uint64](cols)
    var threads = nrb * cols
    ctx.enqueue_function[cs_partial_kernel](d_x.unsafe_ptr(), d_p.unsafe_ptr(), Int64(rows), Int64(cols), Int64(rb),
                                            grid_dim=(threads + CS_TPB - 1) // CS_TPB, block_dim=CS_TPB)
    ctx.enqueue_function[cs_fold_kernel](d_p.unsafe_ptr(), d_o.unsafe_ptr(), Int64(nrb), Int64(cols),
                                         grid_dim=cols, block_dim=CS_TPB)
    ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=dst), src_buf=d_o)
    ctx.synchronize()
    _ = d_x^
    _ = d_p^
    _ = d_o^


def center_device(ctx: DeviceContext, x: Int, mu: Int, dst: Int, rows: Int, cols: Int) raises:
    """dst[r, c] = center_cell(x[r, c], mu[c]); mu float32[cols]."""
    var cells = rows * cols
    if cells <= 0:
        return
    var d_x = ctx.enqueue_create_buffer[DType.float32](cells)
    var d_m = ctx.enqueue_create_buffer[DType.float32](cols)
    var d_o = ctx.enqueue_create_buffer[DType.float32](cells)
    ctx.enqueue_copy(dst_buf=d_x, src_ptr=F32P(unsafe_from_address=x))
    ctx.enqueue_copy(dst_buf=d_m, src_ptr=F32P(unsafe_from_address=mu))
    ctx.enqueue_function[center_kernel](d_x.unsafe_ptr(), d_m.unsafe_ptr(), d_o.unsafe_ptr(), Int64(rows), Int64(cols),
                                        grid_dim=(cells + CS_TPB - 1) // CS_TPB, block_dim=CS_TPB)
    ctx.enqueue_copy(dst_ptr=F32P(unsafe_from_address=dst), src_buf=d_o)
    ctx.synchronize()
    _ = d_x^
    _ = d_m^
    _ = d_o^


def scale_rows_device(ctx: DeviceContext, x: Int, w: Int, dst: Int, rows: Int, cols: Int) raises:
    """dst[r, c] = scale_cell(x[r, c], w[r]); w float32[rows]."""
    var cells = rows * cols
    if cells <= 0:
        return
    var d_x = ctx.enqueue_create_buffer[DType.float32](cells)
    var d_w = ctx.enqueue_create_buffer[DType.float32](rows)
    var d_o = ctx.enqueue_create_buffer[DType.float32](cells)
    ctx.enqueue_copy(dst_buf=d_x, src_ptr=F32P(unsafe_from_address=x))
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=F32P(unsafe_from_address=w))
    ctx.enqueue_function[scale_rows_kernel](d_x.unsafe_ptr(), d_w.unsafe_ptr(), d_o.unsafe_ptr(), Int64(rows), Int64(cols),
                                            grid_dim=(cells + CS_TPB - 1) // CS_TPB, block_dim=CS_TPB)
    ctx.enqueue_copy(dst_ptr=F32P(unsafe_from_address=dst), src_buf=d_o)
    ctx.synchronize()
    _ = d_x^
    _ = d_w^
    _ = d_o^


# ---- resident forms (lane apple-fast-olsne, 2026-10-03) --------------------
# The same two kernels on buffers already on the device, so LinearRegression's
# normal-equations route uploads X and y ONCE (`ols_fit_resident_host`)
# instead of once per helper (col sums, center, solve). Same kernels, same
# launch shapes, same inputs: the same words as col_sums_device and
# center_device.


def col_sums_buf(
    ctx: DeviceContext,
    mut d_x: DeviceBuffer[DType.float32],
    mut d_o: DeviceBuffer[DType.uint64],
    rows: Int,
    cols: Int,
) raises:
    """d_o (uint64[cols], float64 bits): col_sums_device on a resident
    [rows, cols] buffer. Enqueued only; the caller synchronizes."""
    comptime assert CS_SMEM_FITS, "col_sums_buf: a 2 KB threadgroup page must fit"
    if rows <= 0 or cols <= 0:
        raise Error("col_sums: rows and cols must be positive")
    var rb = cs_rows_per_block(rows)
    var nrb = (rows + rb - 1) // rb
    var d_p = ctx.enqueue_create_buffer[DType.int64](nrb * cols * CS_WORDS)
    var threads = nrb * cols
    ctx.enqueue_function[cs_partial_kernel](d_x.unsafe_ptr(), d_p.unsafe_ptr(), Int64(rows), Int64(cols), Int64(rb),
                                            grid_dim=(threads + CS_TPB - 1) // CS_TPB, block_dim=CS_TPB)
    ctx.enqueue_function[cs_fold_kernel](d_p.unsafe_ptr(), d_o.unsafe_ptr(), Int64(nrb), Int64(cols),
                                         grid_dim=cols, block_dim=CS_TPB)
    ctx.synchronize()
    _ = d_p^


def center_buf(
    ctx: DeviceContext,
    mut d_x: DeviceBuffer[DType.float32],
    mut d_m: DeviceBuffer[DType.float32],
    mut d_o: DeviceBuffer[DType.float32],
    rows: Int,
    cols: Int,
) raises:
    """d_o = center_device(d_x, d_m) on resident buffers. Enqueued only."""
    var cells = rows * cols
    if cells <= 0:
        return
    ctx.enqueue_function[center_kernel](d_x.unsafe_ptr(), d_m.unsafe_ptr(), d_o.unsafe_ptr(), Int64(rows), Int64(cols),
                                        grid_dim=(cells + CS_TPB - 1) // CS_TPB, block_dim=CS_TPB)


# ---- device means (lane apple-fast-purity, 2026-10-03) ---------------------
# The resident Ridge / OLS routes used to download the n_features + 1 exact
# sums, divide them by rows on the host and upload the float32 means again.
# One thread per column now does that division on the device in software
# binary64 (checks/soft_f64.mojo: the correctly rounded IEEE quotient and
# the round-to-nearest-even narrowing, the same words the host's float64
# `/` and `.cast[float32]` produce on every vendor). The means are still
# downloaded once, because they are the fit's outputs (mu_, the y mean).


def col_means_kernel(sx: U64P, sy: U64P, mx: F32P, my: F32P, m64: U64P, cols_: Int64, rows_: Int64):
    """Thread j < cols: column j of X; thread cols: y. m64[j] = sum / rows
    (binary64 bits), mx / my = its float32 rounding."""
    var cols = Int(cols_)
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j > cols:
        return
    var s = sx.unsafe_load(j) if j < cols else sy.unsafe_load(0)
    var m = sf64_div(s, sf64_from_int(Int(rows_)))
    m64.unsafe_store(j, m)
    var f = sf64_to_f32(m)
    if j < cols:
        mx.unsafe_store(j, f)
    else:
        my.unsafe_store(0, f)


def col_means_buf(
    ctx: DeviceContext,
    mut d_sx: DeviceBuffer[DType.uint64],
    mut d_sy: DeviceBuffer[DType.uint64],
    mut d_mx: DeviceBuffer[DType.float32],
    mut d_my: DeviceBuffer[DType.float32],
    mut d_m64: DeviceBuffer[DType.uint64],
    rows: Int,
    cols: Int,
) raises:
    """d_mx[cols], d_my[1] (float32) and d_m64[cols + 1] (binary64 bits):
    the means of col_sums_buf's sums of X (d_sx) and y (d_sy). Enqueued only."""
    var n = cols + 1
    ctx.enqueue_function[col_means_kernel](
        d_sx.unsafe_ptr(), d_sy.unsafe_ptr(), d_mx.unsafe_ptr(), d_my.unsafe_ptr(), d_m64.unsafe_ptr(),
        Int64(cols), Int64(rows), grid_dim=(n + CS_TPB - 1) // CS_TPB, block_dim=CS_TPB,
    )


# C02 pairs exact X/y streams in one launch and invocation-owned partial.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
def cs_pair_partial_kernel(x: F32P, y: F32P, part: I64P, rows_: Int64, cols_: Int64, rb_: Int64):
    var rows = Int(rows_)
    var cols = Int(cols_)
    var rb = Int(rb_)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nrb = (rows + rb - 1) // rb
    if tid >= nrb * (cols + 1):
        return
    var c = tid % (cols + 1)
    var lo = (tid // (cols + 1)) * rb
    var acc = InlineArray[Int64, CS_WORDS](fill=Int64(0))
    for r in range(lo, min(lo + rb, rows)):
        var v = y.unsafe_load(r) if c == cols else x.unsafe_load(r * cols + c)
        exact_add(acc, v)
    comptime for L in range(CS_WORDS):
        part.unsafe_store(tid * CS_WORDS + L, acc[L])


def col_sums_pair_buf(ctx: DeviceContext, mut x: DeviceBuffer[DType.float32],
                      mut y: DeviceBuffer[DType.float32], mut sx: DeviceBuffer[DType.uint64],
                      mut sy: DeviceBuffer[DType.uint64], rows: Int, cols: Int) raises:
    comptime if not C02_LINEAR_PAIR:
        col_sums_buf(ctx, x, sx, rows, cols)
        col_sums_buf(ctx, y, sy, rows, 1)
        return
    comptime assert CS_SMEM_FITS, "paired column sums require the incumbent shared page"
    if rows <= 0 or cols <= 0:
        raise Error("col_sums: rows and cols must be positive")
    var rb = cs_rows_per_block(rows)
    var nrb = (rows + rb - 1) // rb
    var tasks = nrb * (cols + 1)
    var part = ctx.enqueue_create_buffer[DType.int64](tasks * CS_WORDS)
    ctx.enqueue_function[cs_pair_partial_kernel](x.unsafe_ptr(), y.unsafe_ptr(), part.unsafe_ptr(),
        Int64(rows), Int64(cols), Int64(rb), grid_dim=(tasks + CS_TPB - 1) // CS_TPB, block_dim=CS_TPB)
    ctx.enqueue_function[cs_fold_kernel](part.unsafe_ptr(), sx.unsafe_ptr(), Int64(nrb), Int64(cols + 1),
        grid_dim=cols, block_dim=CS_TPB)
    ctx.enqueue_function[cs_fold_kernel](part.unsafe_ptr() + cols * CS_WORDS, sy.unsafe_ptr(),
        Int64(nrb), Int64(cols + 1), grid_dim=1, block_dim=CS_TPB)
    ctx.synchronize()
    _ = part^


def col_sums_pair_device(ctx: DeviceContext, x: Int, y: Int, sx: Int, sy: Int, rows: Int, cols: Int) raises:
    if rows <= 0 or cols <= 0:
        raise Error("col_sums: rows and cols must be positive")
    var dx = ctx.enqueue_create_buffer[DType.float32](rows * cols)
    var dy = ctx.enqueue_create_buffer[DType.float32](rows)
    var dsx = ctx.enqueue_create_buffer[DType.uint64](cols)
    var dsy = ctx.enqueue_create_buffer[DType.uint64](1)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=F32P(unsafe_from_address=x))
    ctx.enqueue_copy(dst_buf=dy, src_ptr=F32P(unsafe_from_address=y))
    col_sums_pair_buf(ctx, dx, dy, dsx, dsy, rows, cols)
    ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=sx), src_buf=dsx)
    ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=sy), src_buf=dsy)
    ctx.synchronize()
    _ = dx^
    _ = dy^
    _ = dsx^
    _ = dsy^
