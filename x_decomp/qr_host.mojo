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
Here a step is two passes over the rows: the dots pass carries every
trailing column's chain `w_j = ftz(fma(ftz(a[i, k]), ftz(a[i, j]), w_j))`
down the rows ascending, so each column's chain is the serial loop's (the
same operands in the same order, started from the same `ftz(a[k, j])`); the
update pass applies each cell's one statement. The columns go over host
tasks in contiguous groups (a task's cells are its own; column k is read by
every task and written by none during the passes), and within a task the
columns advance HOST_FW at a time through `identical_mul_add_simd` and
`ftz_lanes`, which are the scalar fma and flush lane by lane. The head
(`geqrf_head`), the multipliers (`geqrf_scale_elem`) and the row-k update
(`sub`) stay the cells themselves. `orgqr` is the same shape with the
reflectors applied last to first to the columns of Q.
"""
from checks.numerics import ftz, identical_mul, identical_mul_add, identical_mul_add_simd
from core.host_lanes import F32V, HOST_FW, ftz_lanes, host_row_tasks
from core.host_parallel import host_parallelize
from std.os import getenv
from x_decomp.cells import F32Ptr, geqrf_head, geqrf_scale_elem, orgqr_init_elem, sub


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


@always_inline
def _dots_rows(
    src_col: F32Ptr, col_stride: Int, x: F32Ptr, x_stride: Int, w: _FP,
    c0: Int, c1: Int, i_lo: Int, i_hi: Int, k: Int,
):
    """w[j] = ftz(x[k, j]) then, for i ascending over [i_lo, i_hi), w[j] =
    ftz(fma(ftz(src_col[i]), ftz(x[i, j]), w[j])) for j in [c0, c1)."""
    var j = c0
    while j + HOST_FW <= c1:
        w.unsafe_store(j, ftz_lanes(x.unsafe_load[width=HOST_FW](k * x_stride + j)))
        j += HOST_FW
    while j < c1:
        w.unsafe_store(j, ftz(x.unsafe_load(k * x_stride + j)))
        j += 1
    for i in range(i_lo, i_hi):
        var v = ftz(src_col.unsafe_load(i * col_stride))
        var vv = F32V(v)
        j = c0
        while j + HOST_FW <= c1:
            w.unsafe_store(
                j, ftz_lanes(identical_mul_add_simd[HOST_FW](
                    vv, ftz_lanes(x.unsafe_load[width=HOST_FW](i * x_stride + j)), w.unsafe_load[width=HOST_FW](j)))
            )
            j += HOST_FW
        while j < c1:
            w.unsafe_store(j, ftz(identical_mul_add(v, ftz(x.unsafe_load(i * x_stride + j)), w.unsafe_load(j))))
            j += 1


@always_inline
def _update_rows(
    src_col: F32Ptr, col_stride: Int, x: F32Ptr, x_stride: Int, w: _FP, t: Float32,
    c0: Int, c1: Int, i_lo: Int, i_hi: Int, k: Int,
):
    """tw[j] = ftz(mul(t, w[j])); x[k, j] = sub(x[k, j], tw[j]); for i in
    [i_lo, i_hi): x[i, j] = ftz(fma(-tw[j], ftz(src_col[i]), ftz(x[i, j])))."""
    for j in range(c0, c1):
        var tw = ftz(identical_mul(t, w.unsafe_load(j)))
        w.unsafe_store(j, tw)
        x.unsafe_store(k * x_stride + j, sub(x.unsafe_load(k * x_stride + j), tw))
    for i in range(i_lo, i_hi):
        var v = ftz(src_col.unsafe_load(i * col_stride))
        var vv = F32V(v)
        var j = c0
        while j + HOST_FW <= c1:
            var tw = w.unsafe_load[width=HOST_FW](j)
            x.unsafe_store(
                i * x_stride + j,
                ftz_lanes(identical_mul_add_simd[HOST_FW](-tw, vv, ftz_lanes(x.unsafe_load[width=HOST_FW](i * x_stride + j)))),
            )
            j += HOST_FW
        while j < c1:
            var tw = w.unsafe_load(j)
            x.unsafe_store(i * x_stride + j, ftz(identical_mul_add(-tw, v, ftz(x.unsafe_load(i * x_stride + j)))))
            j += 1


def geqrf_host_rows(a: F32Ptr, tau: F32Ptr, m: Int, n: Int):
    """`geqrf_serial` as row-streaming passes (the module docstring)."""
    var kk = m if m < n else n
    var scal = InlineArray[Float32, 2](fill=Float32(0))
    var sp = F32Ptr(unsafe_from_address=Int(scal.unsafe_ptr()))
    var wbuf = List[Float32](length=n if n > 0 else 1, fill=Float32(0))
    var wp = _FP(unsafe_from_address=Int(wbuf.unsafe_ptr()))
    for k in range(kk):
        geqrf_head(a, tau, sp, k, m, n)
        for i in range(k + 1, m):
            geqrf_scale_elem(a, sp, k, i, n)
        if scal[1] == Float32(0):
            # the cells return 0 and leave the trailing columns as they are
            continue
        var cols = n - k - 1
        if cols <= 0:
            continue
        var t = tau.unsafe_load(k)
        var col_k = F32Ptr(unsafe_from_address=Int(a) + k * 4)
        var tasks = host_row_tasks(cols, 4 * (m - k))
        var chunk = (cols + tasks - 1) // tasks
        def _dots(task: Int) {imm a, imm col_k, imm wp, imm k, imm m, imm n, imm chunk, imm cols}:
            var c0 = k + 1 + task * chunk
            var c1 = min(c0 + chunk, k + 1 + cols)
            if c1 > c0:
                _dots_rows(col_k, n, a, n, wp, c0, c1, k + 1, m, k)
        if tasks <= 1:
            _dots(0)
        else:
            host_parallelize(_dots, tasks)
        def _upd(task: Int) {imm a, imm col_k, imm wp, imm t, imm k, imm m, imm n, imm chunk, imm cols}:
            var c0 = k + 1 + task * chunk
            var c1 = min(c0 + chunk, k + 1 + cols)
            if c1 > c0:
                _update_rows(col_k, n, a, n, wp, t, c0, c1, k + 1, m, k)
        if tasks <= 1:
            _upd(0)
        else:
            host_parallelize(_upd, tasks)
    _ = wbuf^


def orgqr_host_rows(h: F32Ptr, tau: F32Ptr, q: F32Ptr, m: Int, n: Int, kk: Int, qc: Int):
    """`orgqr_col` for every column of Q as row-streaming passes: Q = I,
    then the reflectors last to first (the module docstring)."""
    for i in range(m):
        for j in range(qc):
            orgqr_init_elem(q, i, j, qc)
    if qc <= 0:
        return
    var wbuf = List[Float32](length=qc, fill=Float32(0))
    var wp = _FP(unsafe_from_address=Int(wbuf.unsafe_ptr()))
    for r in range(kk):
        var k = kk - 1 - r
        var t = ftz(tau.unsafe_load(k))
        if t == Float32(0):
            continue
        var col_k = F32Ptr(unsafe_from_address=Int(h) + k * 4)
        var tasks = host_row_tasks(qc, 4 * (m - k))
        var chunk = (qc + tasks - 1) // tasks
        def _dots(task: Int) {imm q, imm col_k, imm wp, imm k, imm m, imm n, imm qc, imm chunk}:
            var c0 = task * chunk
            var c1 = min(c0 + chunk, qc)
            if c1 > c0:
                _dots_rows(col_k, n, q, qc, wp, c0, c1, k + 1, m, k)
        if tasks <= 1:
            _dots(0)
        else:
            host_parallelize(_dots, tasks)
        def _upd(task: Int) {imm q, imm col_k, imm wp, imm t, imm k, imm m, imm n, imm qc, imm chunk}:
            var c0 = task * chunk
            var c1 = min(c0 + chunk, qc)
            if c1 > c0:
                _update_rows(col_k, n, q, qc, wp, t, c0, c1, k + 1, m, k)
        if tasks <= 1:
            _upd(0)
        else:
            host_parallelize(_upd, tasks)
    _ = wbuf^
