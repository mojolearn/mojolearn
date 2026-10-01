# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""geqrf and orgqr on the host as ROW-STREAMING walks (lane neural-pass37,
2026-10-01): `geqrf_serial`'s and `orgqr_col`'s cells, statement for
statement, with the trailing columns' chains advanced one row at a time
side by side instead of one column at a time.

The serial loop walks column by column: for each trailing column j it reads
the whole column (a stride of n floats per row) for the dot, then again for
the update, so at 200,000 x 220 it touches 48,000 columns' worth of strided
rows per step and takes 300 s on an M4 core. The device's one-block-per-
column staged chains take 17 s there (the chain is 200,000 dependent
multiply-adds per column, which no GPU runs faster than its fma latency).
Here the matrix is copied once into a BLOCK-INTERLEAVED layout: the columns
in blocks of W = HOST_FW, each block holding its W columns row-interleaved
(row i of block b is W consecutive words), the walk runs there, and the
result is copied back. A block's W chains `w_j = ftz(fma(ftz(a[i, k]),
ftz(a[i, j]), w_j))` then advance together one row at a time: one vector
load, `identical_mul_add_simd` and `ftz_lanes`, which are the scalar fma
and flush lane by lane, so each column's chain is the serial loop's (the
same operands in the same order from the same `ftz(a[k, j])`); column k is
read down its own block at stride W. (A first version streamed the
row-major rows and was latency-bound on the 880-byte stride, 17 s at
200,000 x 220; a column-major version was instruction-bound on the scalar
flushes, 10.5 s; the serial loop takes 300 s.) The trailing blocks go over
host tasks in contiguous groups (a task's blocks are its own; column k is
read by every task and written by none during the step); the block that
holds column k, and the last block's padding, mask the lanes that are not
trailing columns, whose words are left exactly as they were. The head and
the multipliers are `geqrf_head`'s and `geqrf_scale_elem`'s statements
over the strided column (`_head_col`, `_norm_col`), the row-k update is
`sub`. `orgqr` is the same shape with the reflectors applied last to first
to the columns of Q.
"""
from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add, identical_mul_add_simd
from std.math import iota
from core.host_lanes import F32V, HOST_FW, ftz_lanes, host_row_tasks
from core.host_parallel import host_parallelize
from std.os import getenv
from x_decomp.cells import F32Ptr, div0, sqrt0, sub


def xd_qr_serial() -> Bool:
    """MOJOLEARN_XD_QR_SERIAL=1: the host column's column-by-column serial
    loop (the A/B arm); default the row-streaming walk."""
    return String(getenv("MOJOLEARN_XD_QR_SERIAL")) == "1"


comptime XD_QR_HOST_MIN = 1 << 16


def xd_qr_on_host(m: Int) -> Bool:
    """Whether the device kit runs geqrf/orgqr as the host walk: by default
    for m >= XD_QR_HOST_MIN rows (the device's one-chain-per-column kernels
    are bound by m dependent multiply-adds per column, 17 s at 200,000 x 220
    on an M4 against this walk's host time); MOJOLEARN_XD_QR_HOST=0/1
    forces either. The same cells either way."""
    var v = String(getenv("MOJOLEARN_XD_QR_HOST"))
    if v == "0":
        return False
    if v == "1":
        return True
    return m >= XD_QR_HOST_MIN

comptime _FP = MutPointer[Float32, MutUntrackedOrigin]


comptime _W = HOST_FW
comptime _B32 = SIMD[DType.bool, _W]


@always_inline
def _blk(base: _FP, m: Int, j: Int) -> _FP:
    """Block j // W's base: row i of the block is W words at + i * W."""
    return base + (j // _W) * m * _W


def _pack_in(src: F32Ptr, rows: Int, cols: Int, mut dst: List[Float32]):
    """The block-interleaved copy of a row-major rows x cols matrix (the
    padding columns of the last block zero), the blocks over host tasks."""
    var nb = (cols + _W - 1) // _W
    var dp = _FP(unsafe_from_address=Int(dst.unsafe_ptr()))
    var tasks = host_row_tasks(nb, 2 * rows * _W)
    var chunk = (nb + tasks - 1) // tasks
    def _t(task: Int) {imm src, imm dp, imm rows, imm cols, imm chunk, imm nb}:
        var b0 = task * chunk
        var b1 = min(b0 + chunk, nb)
        for b in range(b0, b1):
            var out = dp + b * rows * _W
            var j0 = b * _W
            for i in range(rows):
                for l in range(_W):
                    var j = j0 + l
                    out.unsafe_store(i * _W + l, src.unsafe_load(i * cols + j) if j < cols else Float32(0))
    if tasks <= 1:
        _t(0)
    else:
        host_parallelize(_t, tasks)


def _pack_out(src: List[Float32], rows: Int, cols: Int, dst: F32Ptr):
    """The row-major matrix back from the block-interleaved copy, the rows
    over host tasks."""
    var sp = _FP(unsafe_from_address=Int(src.unsafe_ptr()))
    var tasks = host_row_tasks(rows, 2 * cols)
    var chunk = (rows + tasks - 1) // tasks
    def _t(task: Int) {imm sp, imm dst, imm rows, imm cols, imm chunk}:
        var r0 = task * chunk
        var r1 = min(r0 + chunk, rows)
        for i in range(r0, r1):
            for j in range(cols):
                dst.unsafe_store(i * cols + j, sp.unsafe_load((j // _W) * rows * _W + i * _W + (j % _W)))
    if tasks <= 1:
        _t(0)
    else:
        host_parallelize(_t, tasks)


@always_inline
def _norm_col(col: _FP, k: Int, m: Int) -> Float32:
    """`reflector_norm` over column k read at stride W: the largest |entry|
    first, the sum of squares ascending in the row index."""
    var mx = Float32(0)
    for i in range(k, m):
        var v = abs(ftz(col.unsafe_load(i * _W)))
        if v > mx:
            mx = v
    if mx == Float32(0):
        return Float32(0)
    var acc = Float32(0)
    for i in range(k, m):
        var v = ftz(identical_div(ftz(col.unsafe_load(i * _W)), mx))
        acc = ftz(identical_mul_add(v, v, acc))
    return ftz(identical_mul(sqrt0(acc), mx))


@always_inline
def _head_col(col: _FP, tau: F32Ptr, scal: F32Ptr, k: Int, m: Int):
    """`geqrf_head` over column k read at stride W (the same statements)."""
    var alpha = ftz(col.unsafe_load(k * _W))
    var xmax = Float32(0)
    for i in range(k + 1, m):
        var v = abs(ftz(col.unsafe_load(i * _W)))
        if v > xmax:
            xmax = v
    if xmax == Float32(0):
        tau.unsafe_store(k, Float32(0))
        scal.unsafe_store(0, Float32(1))
        scal.unsafe_store(1, Float32(0))
        return
    var nrm = _norm_col(col, k, m)
    var beta = -nrm if alpha >= Float32(0) else nrm
    tau.unsafe_store(k, div0(sub(beta, alpha), beta))
    scal.unsafe_store(0, sub(alpha, beta))
    scal.unsafe_store(1, Float32(1))
    col.unsafe_store(k * _W, beta)


@always_inline
def _apply_block(blk: _FP, vk: _FP, t: Float32, w: F32V, k: Int, m: Int, mask: _B32):
    """One block's update from its chains' results w: tw = ftz(mul(t, w));
    row k: sub(row, tw); rows i > k: ftz(fma(-tw, vk[i], ftz(row))); the
    lanes outside `mask` keep their words."""
    var tw = F32V(0)
    for l in range(_W):
        tw[l] = ftz(identical_mul(t, w[l]))
    var rowk = blk.unsafe_load[width=_W](k * _W)
    var newk = F32V(0)
    for l in range(_W):
        newk[l] = sub(rowk[l], tw[l])
    blk.unsafe_store(k * _W, mask.select(newk, rowk))
    var ntw = -tw
    for i in range(k + 1, m):
        var v = F32V(vk.unsafe_load(i))
        var old = blk.unsafe_load[width=_W](i * _W)
        var upd = ftz_lanes(identical_mul_add_simd[_W](ntw, v, ftz_lanes(old)))
        blk.unsafe_store(i * _W, mask.select(upd, old))


@always_inline
def _step_block(blk: _FP, vk: _FP, t: Float32, k: Int, m: Int, mask: _B32):
    """One block's W chains and update for step k (`vk[i]` = ftz of column
    k's entry, computed once per step); `mask` names the lanes that are
    trailing columns (the others' words are left as they were)."""
    var w = ftz_lanes(blk.unsafe_load[width=_W](k * _W))
    for i in range(k + 1, m):
        var v = F32V(vk.unsafe_load(i))
        w = ftz_lanes(identical_mul_add_simd[_W](v, ftz_lanes(blk.unsafe_load[width=_W](i * _W)), w))
    _apply_block(blk, vk, t, w, k, m, mask)


@always_inline
def _step_block4(b0: _FP, b1: _FP, b2: _FP, b3: _FP, vk: _FP, t: Float32, k: Int, m: Int,
                 m0: _B32, m1: _B32, m2: _B32, m3: _B32):
    """Four blocks' chains advanced in one loop (four independent chains,
    each `_step_block`'s), then each block's update."""
    var w0 = ftz_lanes(b0.unsafe_load[width=_W](k * _W))
    var w1 = ftz_lanes(b1.unsafe_load[width=_W](k * _W))
    var w2 = ftz_lanes(b2.unsafe_load[width=_W](k * _W))
    var w3 = ftz_lanes(b3.unsafe_load[width=_W](k * _W))
    for i in range(k + 1, m):
        var v = F32V(vk.unsafe_load(i))
        w0 = ftz_lanes(identical_mul_add_simd[_W](v, ftz_lanes(b0.unsafe_load[width=_W](i * _W)), w0))
        w1 = ftz_lanes(identical_mul_add_simd[_W](v, ftz_lanes(b1.unsafe_load[width=_W](i * _W)), w1))
        w2 = ftz_lanes(identical_mul_add_simd[_W](v, ftz_lanes(b2.unsafe_load[width=_W](i * _W)), w2))
        w3 = ftz_lanes(identical_mul_add_simd[_W](v, ftz_lanes(b3.unsafe_load[width=_W](i * _W)), w3))
    _apply_block(b0, vk, t, w0, k, m, m0)
    _apply_block(b1, vk, t, w1, k, m, m1)
    _apply_block(b2, vk, t, w2, k, m, m2)
    _apply_block(b3, vk, t, w3, k, m, m3)


def _flushed_col(colk: _FP, k: Int, m: Int, mut vk: List[Float32]):
    """vk[i] = ftz(colk[i]) for i > k: the chain's and the update's flushed
    operand, computed once per step instead of once per block."""
    var vp = _FP(unsafe_from_address=Int(vk.unsafe_ptr()))
    for i in range(k + 1, m):
        vp.unsafe_store(i, ftz(colk.unsafe_load(i * _W)))


@always_inline
def _run_blocks(base: _FP, vk: _FP, t: Float32, k: Int, m: Int, cols: Int, b_lo: Int, b_hi: Int, all_cols: Bool):
    """The blocks [b_lo, b_hi), four at a time then one at a time."""
    var b = b_lo
    while b + 4 <= b_hi:
        var l0 = iota[DType.int32, _W]() + Int32(b * _W)
        var l1 = l0 + Int32(_W)
        var l2 = l0 + Int32(2 * _W)
        var l3 = l0 + Int32(3 * _W)
        var m0 = l0.lt(Int32(cols)) if all_cols else (l0.gt(Int32(k)) & l0.lt(Int32(cols)))
        var m1 = l1.lt(Int32(cols)) if all_cols else (l1.gt(Int32(k)) & l1.lt(Int32(cols)))
        var m2 = l2.lt(Int32(cols)) if all_cols else (l2.gt(Int32(k)) & l2.lt(Int32(cols)))
        var m3 = l3.lt(Int32(cols)) if all_cols else (l3.gt(Int32(k)) & l3.lt(Int32(cols)))
        _step_block4(base + b * m * _W, base + (b + 1) * m * _W, base + (b + 2) * m * _W, base + (b + 3) * m * _W,
                     vk, t, k, m, m0, m1, m2, m3)
        b += 4
    while b < b_hi:
        var lane = iota[DType.int32, _W]() + Int32(b * _W)
        var mask = lane.lt(Int32(cols)) if all_cols else (lane.gt(Int32(k)) & lane.lt(Int32(cols)))
        _step_block(base + b * m * _W, vk, t, k, m, mask)
        b += 1


def _trailing_blocks(base: _FP, vk: _FP, t: Float32, k: Int, m: Int, cols: Int, all_cols: Bool):
    """Every block holding a column of step k (the trailing ones k < j <
    cols; with `all_cols`, orgqr's every column j < cols), the blocks over
    host tasks in contiguous groups."""
    var nb = (cols + _W - 1) // _W
    var b_lo = 0 if all_cols else (k + 1) // _W
    if b_lo >= nb:
        return
    var count = nb - b_lo
    var tasks = host_row_tasks(count, 2 * (m - k) * _W)
    var chunk = (count + tasks - 1) // tasks
    def _run(task: Int) {imm base, imm vk, imm t, imm k, imm m, imm cols, imm chunk, imm b_lo, imm nb, imm all_cols}:
        var b0 = b_lo + task * chunk
        var b1 = min(b0 + chunk, nb)
        if b1 > b0:
            _run_blocks(base, vk, t, k, m, cols, b0, b1, all_cols)
    if tasks <= 1:
        _run(0)
    else:
        host_parallelize(_run, tasks)


def geqrf_host_rows(a: F32Ptr, tau: F32Ptr, m: Int, n: Int):
    """`geqrf_serial` on the block-interleaved copy (the module docstring)."""
    var kk = m if m < n else n
    var nb = (n + _W - 1) // _W
    var at = List[Float32](unsafe_uninit_length=max(nb * _W * m, 1))
    _pack_in(a, m, n, at)
    var base = _FP(unsafe_from_address=Int(at.unsafe_ptr()))
    var scal = InlineArray[Float32, 2](fill=Float32(0))
    var sp = F32Ptr(unsafe_from_address=Int(scal.unsafe_ptr()))
    var vk = List[Float32](length=max(m, 1), fill=Float32(0))
    var vkp = _FP(unsafe_from_address=Int(vk.unsafe_ptr()))
    for k in range(kk):
        var colk = _blk(base, m, k) + (k % _W)
        _head_col(colk, tau, sp, k, m)
        if scal[1] != Float32(0):
            for i in range(k + 1, m):
                colk.unsafe_store(i * _W, div0(colk.unsafe_load(i * _W), scal[0]))
        if scal[1] == Float32(0):
            continue
        _flushed_col(colk, k, m, vk)
        _trailing_blocks(base, vkp, tau.unsafe_load(k), k, m, n, False)
    _pack_out(at, m, n, a)
    _ = at^
    _ = vk^


def orgqr_host_rows(h: F32Ptr, tau: F32Ptr, q: F32Ptr, m: Int, n: Int, kk: Int, qc: Int):
    """`orgqr_col` for every column of Q on block-interleaved copies: Q = I,
    then the reflectors last to first (the module docstring)."""
    if qc <= 0:
        return
    var nbh = (n + _W - 1) // _W
    var ht = List[Float32](unsafe_uninit_length=max(nbh * _W * m, 1))
    _pack_in(h, m, n, ht)
    var hb = _FP(unsafe_from_address=Int(ht.unsafe_ptr()))
    var nbq = (qc + _W - 1) // _W
    var qt = List[Float32](length=max(nbq * _W * m, 1), fill=Float32(0))
    var qb = _FP(unsafe_from_address=Int(qt.unsafe_ptr()))
    for j in range(qc):
        if j < m:
            (_blk(qb, m, j) + (j % _W)).unsafe_store(j * _W, Float32(1))
    var vk = List[Float32](length=max(m, 1), fill=Float32(0))
    var vkp = _FP(unsafe_from_address=Int(vk.unsafe_ptr()))
    for r in range(kk):
        var k = kk - 1 - r
        var t = ftz(tau.unsafe_load(k))
        if t == Float32(0):
            continue
        var colk = _blk(hb, m, k) + (k % _W)
        # every column of Q is updated by every reflector (orgqr_col's loop
        # starts at column 0): the mask is "j < qc", not "j > k"
        _flushed_col(colk, k, m, vk)
        _trailing_blocks(qb, vkp, t, k, m, qc, True)
    _pack_out(qt, m, qc, q)
    _ = ht^
    _ = qt^
    _ = vk^
