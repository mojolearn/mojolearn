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
Here the matrix is transposed once into a column-major copy (every column
contiguous), the walk runs there, and the result is transposed back. A
column's chain `w_j = ftz(fma(ftz(a[i, k]), ftz(a[i, j]), w_j))` is then two
sequential streams (column k and column j), which the prefetcher follows
(a first version streamed the row-major rows and was latency-bound on the
880-byte stride: 17 s at 200,000 x 220, the device's time). Each chain is
the serial loop's: the same operands in the same order from the same
`ftz(a[k, j])`; the chains of four columns are advanced in one loop so the
four independent fma chains overlap, which changes no chain. The trailing
columns go over host tasks in contiguous groups (a task's columns are its
own; column k is read by every task and written by none during the step).
The head and the multipliers are `geqrf_head`'s and `geqrf_scale_elem`'s
statements over the contiguous column (`_head_col`, `_norm_col`), the row-k
update is `sub`. `orgqr` is the same shape with the reflectors applied last
to first to the columns of Q.
"""
from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add
from core.host_lanes import host_row_tasks
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


comptime _ILP = 4


def _transpose_in(src: F32Ptr, rows: Int, cols: Int, mut dst: List[Float32]):
    """dst[j * rows + i] = src[i * cols + j], tiles of 64 x 64, the
    destination columns over host tasks. A copy."""
    var dp = _FP(unsafe_from_address=Int(dst.unsafe_ptr()))
    var tasks = host_row_tasks(cols, 2 * rows)
    var chunk = (cols + tasks - 1) // tasks
    def _t(task: Int) {imm src, imm dp, imm rows, imm cols, imm chunk}:
        var c0 = task * chunk
        var c1 = min(c0 + chunk, cols)
        var i0 = 0
        while i0 < rows:
            var i1 = min(i0 + 64, rows)
            var j0 = c0
            while j0 < c1:
                var j1 = min(j0 + 64, c1)
                for i in range(i0, i1):
                    for j in range(j0, j1):
                        dp.unsafe_store(j * rows + i, src.unsafe_load(i * cols + j))
                j0 = j1
            i0 = i1
    if tasks <= 1:
        _t(0)
    else:
        host_parallelize(_t, tasks)


def _transpose_out(src: List[Float32], rows: Int, cols: Int, dst: F32Ptr):
    """dst[i * cols + j] = src[j * rows + i], the destination rows over
    host tasks. A copy."""
    var sp = _FP(unsafe_from_address=Int(src.unsafe_ptr()))
    var tasks = host_row_tasks(rows, 2 * cols)
    var chunk = (rows + tasks - 1) // tasks
    def _t(task: Int) {imm sp, imm dst, imm rows, imm cols, imm chunk}:
        var r0 = task * chunk
        var r1 = min(r0 + chunk, rows)
        var i0 = r0
        while i0 < r1:
            var i1 = min(i0 + 64, r1)
            var j0 = 0
            while j0 < cols:
                var j1 = min(j0 + 64, cols)
                for j in range(j0, j1):
                    for i in range(i0, i1):
                        dst.unsafe_store(i * cols + j, sp.unsafe_load(j * rows + i))
                j0 = j1
            i0 = i1
    if tasks <= 1:
        _t(0)
    else:
        host_parallelize(_t, tasks)


@always_inline
def _norm_col(col: _FP, k: Int, m: Int) -> Float32:
    """`reflector_norm` over the contiguous column: the largest |entry| first,
    the sum of squares ascending in the row index."""
    var mx = Float32(0)
    for i in range(k, m):
        var v = abs(ftz(col.unsafe_load(i)))
        if v > mx:
            mx = v
    if mx == Float32(0):
        return Float32(0)
    var acc = Float32(0)
    for i in range(k, m):
        var v = ftz(identical_div(ftz(col.unsafe_load(i)), mx))
        acc = ftz(identical_mul_add(v, v, acc))
    return ftz(identical_mul(sqrt0(acc), mx))


@always_inline
def _head_col(col: _FP, tau: F32Ptr, scal: F32Ptr, k: Int, m: Int):
    """`geqrf_head` over the contiguous column k (the same statements)."""
    var alpha = ftz(col.unsafe_load(k))
    var xmax = Float32(0)
    for i in range(k + 1, m):
        var v = abs(ftz(col.unsafe_load(i)))
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
    col.unsafe_store(k, beta)


@always_inline
def _chain4(colk: _FP, c0: _FP, c1: _FP, c2: _FP, c3: _FP, k: Int, m: Int, mut w: InlineArray[Float32, 4]):
    """Four columns' dot chains in one loop: w[c] = ftz(c[k]) then, for i in
    k + 1 .. m - 1 ascending, w[c] = ftz(fma(ftz(colk[i]), ftz(c[i]), w[c])).
    Four independent chains, each the serial loop's."""
    var w0 = ftz(c0.unsafe_load(k))
    var w1 = ftz(c1.unsafe_load(k))
    var w2 = ftz(c2.unsafe_load(k))
    var w3 = ftz(c3.unsafe_load(k))
    for i in range(k + 1, m):
        var v = ftz(colk.unsafe_load(i))
        w0 = ftz(identical_mul_add(v, ftz(c0.unsafe_load(i)), w0))
        w1 = ftz(identical_mul_add(v, ftz(c1.unsafe_load(i)), w1))
        w2 = ftz(identical_mul_add(v, ftz(c2.unsafe_load(i)), w2))
        w3 = ftz(identical_mul_add(v, ftz(c3.unsafe_load(i)), w3))
    w[0] = w0
    w[1] = w1
    w[2] = w2
    w[3] = w3


@always_inline
def _chain1(colk: _FP, c: _FP, k: Int, m: Int) -> Float32:
    var w = ftz(c.unsafe_load(k))
    for i in range(k + 1, m):
        w = ftz(identical_mul_add(ftz(colk.unsafe_load(i)), ftz(c.unsafe_load(i)), w))
    return w


@always_inline
def _apply_col(colk: _FP, c: _FP, t: Float32, w: Float32, k: Int, m: Int):
    """tw = ftz(mul(t, w)); c[k] = sub(c[k], tw); for i > k: c[i] =
    ftz(fma(-tw, ftz(colk[i]), ftz(c[i])))."""
    var tw = ftz(identical_mul(t, w))
    c.unsafe_store(k, sub(c.unsafe_load(k), tw))
    for i in range(k + 1, m):
        c.unsafe_store(i, ftz(identical_mul_add(-tw, ftz(colk.unsafe_load(i)), ftz(c.unsafe_load(i)))))


@always_inline
def _step_cols(base: _FP, colk: _FP, t: Float32, k: Int, m: Int, j0: Int, j1: Int):
    """The trailing columns [j0, j1) of a column-major buffer (column j at
    base + j * m): each column's chain, then its update."""
    var j = j0
    var w4 = InlineArray[Float32, 4](fill=Float32(0))
    while j + _ILP <= j1:
        var c0 = base + j * m
        var c1 = base + (j + 1) * m
        var c2 = base + (j + 2) * m
        var c3 = base + (j + 3) * m
        _chain4(colk, c0, c1, c2, c3, k, m, w4)
        _apply_col(colk, c0, t, w4[0], k, m)
        _apply_col(colk, c1, t, w4[1], k, m)
        _apply_col(colk, c2, t, w4[2], k, m)
        _apply_col(colk, c3, t, w4[3], k, m)
        j += _ILP
    while j < j1:
        var c = base + j * m
        _apply_col(colk, c, t, _chain1(colk, c, k, m), k, m)
        j += 1


def geqrf_host_rows(a: F32Ptr, tau: F32Ptr, m: Int, n: Int):
    """`geqrf_serial` on a column-major copy (the module docstring)."""
    var kk = m if m < n else n
    var at = List[Float32](unsafe_uninit_length=max(m * n, 1))
    _transpose_in(a, m, n, at)
    var base = _FP(unsafe_from_address=Int(at.unsafe_ptr()))
    var scal = InlineArray[Float32, 2](fill=Float32(0))
    var sp = F32Ptr(unsafe_from_address=Int(scal.unsafe_ptr()))
    for k in range(kk):
        var colk = base + k * m
        _head_col(colk, tau, sp, k, m)
        if scal[1] != Float32(0):
            for i in range(k + 1, m):
                colk.unsafe_store(i, div0(colk.unsafe_load(i), scal[0]))
        if scal[1] == Float32(0):
            continue
        var cols = n - k - 1
        if cols <= 0:
            continue
        var t = tau.unsafe_load(k)
        var tasks = host_row_tasks(cols, 4 * (m - k))
        var chunk = (cols + tasks - 1) // tasks
        def _step(task: Int) {imm base, imm colk, imm t, imm k, imm m, imm chunk, imm cols}:
            var j0 = k + 1 + task * chunk
            var j1 = min(j0 + chunk, k + 1 + cols)
            if j1 > j0:
                _step_cols(base, colk, t, k, m, j0, j1)
        if tasks <= 1:
            _step(0)
        else:
            host_parallelize(_step, tasks)
    _transpose_out(at, m, n, a)
    _ = at^


def orgqr_host_rows(h: F32Ptr, tau: F32Ptr, q: F32Ptr, m: Int, n: Int, kk: Int, qc: Int):
    """`orgqr_col` for every column of Q on column-major copies: Q = I, then
    the reflectors last to first (the module docstring)."""
    if qc <= 0:
        return
    var ht = List[Float32](unsafe_uninit_length=max(m * n, 1))
    _transpose_in(h, m, n, ht)
    var hb = _FP(unsafe_from_address=Int(ht.unsafe_ptr()))
    var qt = List[Float32](unsafe_uninit_length=max(m * qc, 1))
    var qb = _FP(unsafe_from_address=Int(qt.unsafe_ptr()))
    for j in range(qc):
        for i in range(m):
            qb.unsafe_store(j * m + i, Float32(1) if i == j else Float32(0))
    for r in range(kk):
        var k = kk - 1 - r
        var t = ftz(tau.unsafe_load(k))
        if t == Float32(0):
            continue
        var colk = hb + k * m
        var tasks = host_row_tasks(qc, 4 * (m - k))
        var chunk = (qc + tasks - 1) // tasks
        def _step(task: Int) {imm qb, imm colk, imm t, imm k, imm m, imm qc, imm chunk}:
            var j0 = task * chunk
            var j1 = min(j0 + chunk, qc)
            if j1 > j0:
                _step_cols(qb, colk, t, k, m, j0, j1)
        if tasks <= 1:
            _step(0)
        else:
            host_parallelize(_step, tasks)
    _transpose_out(qt, m, qc, q)
    _ = ht^
    _ = qt^
