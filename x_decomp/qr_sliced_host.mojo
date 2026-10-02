# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SLICED geqrf / orgqr, THE HOST REPLAY (lane hr-qr, 2026-10-02): the host
column's (and a CPU-only install's) run of x_decomp/qr_sliced.mojo's order,
the same chains as x_decomp/qr_sliced_device.mojo bit for bit.

The reflector products run row-streaming: a host task owns whole slices,
and walks each slice's rows once, advancing the chains of every column of
the slice side by side (a row of A is contiguous, so the columns' chains
step W lanes at a time with `identical_mul_add_simd` and `ftz_lanes`, the
scalar fma and flush lane by lane). Each (slice, column) chain is the
device thread's: the same operands, rows ascending, from 0. The norms, the
trees, the reflector and the updates are the shared cells.
"""
from checks.numerics import ftz, identical_mul_add, identical_mul_add_simd
from core.host_lanes import F32V, HOST_FW, ftz_lanes, host_row_tasks
from core.host_parallel import host_parallelize
from x_decomp.cells import F32Ptr, geqrf_scale_elem, geqrf_update_elem, orgqr_update_elem
from x_decomp.qr_sliced import QS_ROWS, qs_dot_finish, qs_head, qs_slice_hi, qs_slice_lo, qs_slice_ssq, qs_slices

comptime _W = HOST_FW


def _partials(x: F32Ptr, xs: Int, y: F32Ptr, ys: Int, k: Int, m: Int, j0: Int, ncols: Int, ns: Int, part: F32Ptr, pstride: Int):
    """part[c * pstride + jo] = slice c's chain of ftz(x[i, k]) ftz(y[i, j0 + jo])
    (rows ascending, from 0), every slice c < ns and column jo < ncols."""
    var tasks = host_row_tasks(ns, QS_ROWS * ncols)
    var chunk = (ns + tasks - 1) // tasks

    def _t(task: Int) {imm x, imm xs, imm y, imm ys, imm k, imm m, imm j0, imm ncols, imm ns, imm part, imm pstride, imm chunk}:
        var c_lo = task * chunk
        var c_hi = min(c_lo + chunk, ns)
        for c in range(c_lo, c_hi):
            var acc = part.unsafe_offset(c * pstride)
            for jo in range(ncols):
                acc.unsafe_store(jo, Float32(0))
            for i in range(qs_slice_lo(k, c), qs_slice_hi(k, c, m)):
                var v = ftz(x.unsafe_load(i * xs + k))
                var vv = F32V(v)
                var row = y.unsafe_offset(i * ys + j0)
                var jo = 0
                while jo + _W <= ncols:
                    var old = acc.unsafe_load[width=_W](jo)
                    acc.unsafe_store(jo, ftz_lanes(identical_mul_add_simd[_W](vv, ftz_lanes(row.unsafe_load[width=_W](jo)), old)))
                    jo += _W
                while jo < ncols:
                    acc.unsafe_store(jo, ftz(identical_mul_add(v, ftz(row.unsafe_load(jo)), acc.unsafe_load(jo))))
                    jo += 1

    if tasks <= 1:
        _t(0)
    else:
        host_parallelize(_t, tasks)


def _folds(y: F32Ptr, ys: Int, k: Int, j0: Int, ncols: Int, ns: Int, part: F32Ptr, pstride: Int, col: F32Ptr, w: F32Ptr):
    """w[j0 + jo] = `qs_dot_finish`(y[k, j0 + jo], column jo's slice partials)."""
    for jo in range(ncols):
        for c in range(ns):
            col.unsafe_store(c, part.unsafe_load(c * pstride + jo))
        w.unsafe_store(j0 + jo, qs_dot_finish(y.unsafe_load(k * ys + j0 + jo), col, ns))


def qs_geqrf_host(a: F32Ptr, tau: F32Ptr, m: Int, n: Int):
    """geqrf in the sliced order, in place (x_decomp/qr_sliced.mojo)."""
    var kk = m if m < n else n
    var nsmax = max(qs_slices(m - 1), 1)
    var ps = List[Float32](length=nsmax, fill=Float32(0))
    var pq = List[Float32](length=nsmax, fill=Float32(0))
    var col = List[Float32](length=nsmax, fill=Float32(0))
    var part = List[Float32](length=nsmax * max(n, 1), fill=Float32(0))
    var w = List[Float32](length=max(n, 1), fill=Float32(0))
    var scal = InlineArray[Float32, 2](fill=Float32(0))
    var psp = F32Ptr(unsafe_from_address=Int(ps.unsafe_ptr()))
    var pqp = F32Ptr(unsafe_from_address=Int(pq.unsafe_ptr()))
    var colp = F32Ptr(unsafe_from_address=Int(col.unsafe_ptr()))
    var pp = F32Ptr(unsafe_from_address=Int(part.unsafe_ptr()))
    var wp = F32Ptr(unsafe_from_address=Int(w.unsafe_ptr()))
    var sp = F32Ptr(unsafe_from_address=Int(scal.unsafe_ptr()))
    for k in range(kk):
        var ns = qs_slices(m - k - 1)
        for c in range(ns):
            var r = qs_slice_ssq(a, n, k, qs_slice_lo(k, c), qs_slice_hi(k, c, m))
            psp.unsafe_store(c, r[0])
            pqp.unsafe_store(c, r[1])
        qs_head(a, tau, sp, psp, pqp, k, n, ns)
        if scal[1] == Float32(0):
            continue
        for i in range(k + 1, m):
            geqrf_scale_elem(a, sp, k, i, n)
        var cols = n - k - 1
        if cols <= 0:
            continue
        _partials(a, n, a, n, k, m, k + 1, cols, ns, pp, cols)
        _folds(a, n, k, k + 1, cols, ns, pp, cols, colp, wp)
        var rows = m - k
        var tasks = host_row_tasks(rows, cols)
        var chunk = (rows + tasks - 1) // tasks

        def _u(task: Int) {imm a, imm tau, imm sp, imm wp, imm k, imm m, imm n, imm chunk}:
            var r0 = k + task * chunk
            var r1 = min(r0 + chunk, m)
            for i in range(r0, r1):
                for j in range(k + 1, n):
                    geqrf_update_elem(a, tau, sp, k, i, j, n, wp.unsafe_load(j))

        if tasks <= 1:
            _u(0)
        else:
            host_parallelize(_u, tasks)
    _ = ps^
    _ = pq^
    _ = col^
    _ = part^
    _ = w^


def qs_orgqr_host(h: F32Ptr, tau: F32Ptr, q: F32Ptr, m: Int, n: Int, kk: Int, qc: Int):
    """The first qc columns of Q = H_0 ... H_{kk-1} in the sliced order."""
    if qc <= 0:
        return
    for t in range(m * qc):
        q.unsafe_store(t, Float32(1) if t // qc == t % qc else Float32(0))
    var nsmax = max(qs_slices(m - 1), 1)
    var col = List[Float32](length=nsmax, fill=Float32(0))
    var part = List[Float32](length=nsmax * qc, fill=Float32(0))
    var w = List[Float32](length=qc, fill=Float32(0))
    var colp = F32Ptr(unsafe_from_address=Int(col.unsafe_ptr()))
    var pp = F32Ptr(unsafe_from_address=Int(part.unsafe_ptr()))
    var wp = F32Ptr(unsafe_from_address=Int(w.unsafe_ptr()))
    for r in range(kk):
        var k = kk - 1 - r
        if ftz(tau.unsafe_load(k)) == Float32(0):
            continue
        var ns = qs_slices(m - k - 1)
        _partials(h, n, q, qc, k, m, 0, qc, ns, pp, qc)
        _folds(q, qc, k, 0, qc, ns, pp, qc, colp, wp)
        var rows = m - k
        var tasks = host_row_tasks(rows, qc)
        var chunk = (rows + tasks - 1) // tasks

        def _u(task: Int) {imm h, imm tau, imm q, imm wp, imm k, imm m, imm n, imm qc, imm chunk}:
            var r0 = k + task * chunk
            var r1 = min(r0 + chunk, m)
            for i in range(r0, r1):
                for j in range(qc):
                    orgqr_update_elem(h, tau, q, k, i, j, n, qc, wp.unsafe_load(j))

        if tasks <= 1:
            _u(0)
        else:
            host_parallelize(_u, tasks)
    _ = col^
    _ = part^
    _ = w^
