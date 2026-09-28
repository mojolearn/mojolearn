# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU column's one-sided Jacobi SVD, `host_one_sided_jacobi_svd`'s
words at CPU speed (lane decomp-cpu, 2026-09-28). Host only: compiled into
the CPU host binding.

`decomposition/host/pca_full_oracle.mojo::host_one_sided_jacobi_svd`
replays `one_sided_jacobi_svd_kernel` on the row-major R: every pair (p, q)
reads and rotates two COLUMNS, a stride-n walk. Here R and V are held
TRANSPOSED (a column is a contiguous row), so:
  * the three Gram folds of a pair keep their SVD_TPB lanes, lane t
    folding rows t, t + SVD_TPB, ... ascending; lanes t..t+W-1 of one
    stride are W consecutive words, so W lanes advance per vector op, each
    its own chain in its own order; then the same halving trees;
  * the rotation of the two columns is elementwise (`_rot_sub`/`_rot_add`
    lane by lane, the same flushes and the same single rounding);
  * the pair order, the threshold, the rotation angle
    (`host_jacobi_rotation_cs`), the sweep count and the final norms are
    the oracle's.
A transpose moves words, never rounds them. Under MOJOLEARN_HOST_SABOTAGE
the rotation is the oracle's split-FMA arm, as `_rot_sub`/`_rot_add` are.

Proof: x_decomp/checks/dense_check.mojo holds `fast_one_sided_jacobi_svd`
to `host_one_sided_jacobi_svd` bit for bit (values, V, sweep count), and
arm host_svd_fold_order.patch must make it fail.
"""
from std.math import ceildiv, fma
from std.sys.info import simd_width_of

from checks.numerics import ftz, identical_mul_add, identical_sqrt
from decomposition.host.pca_full_oracle import SVD_TPB
from decomposition.host.pca_oracle import PCA_ORACLE_HOST_SABOTAGE, host_jacobi_rotation_cs
from x_decomp.cells import F32Ptr
from x_decomp.host_simd import ftz_v, mul_add_v

comptime W = simd_width_of[DType.float32]()
comptime V = SIMD[DType.float32, W]


@fieldwise_init
struct FastSvdResult(Movable):
    var v: List[Float32]  # n x n row major, vector i in COLUMN i
    var s: List[Float32]
    var converged: Bool
    var executed: Int


@always_inline
def _halving(p: F32Ptr) -> Float32:
    """`host_halving_sum` over SVD_TPB partials (clobbers them)."""
    var step = SVD_TPB // 2
    while step > 0:
        for t in range(step):
            p.unsafe_store(t, p.unsafe_load(t) + p.unsafe_load(t + step))
        step //= 2
    return p.unsafe_load(0)


@always_inline
def _rot_sub_v(c: V, x: V, s: V, y: V) -> V:
    comptime if PCA_ORACLE_HOST_SABOTAGE:
        return ftz_v[W](ftz_v[W](c * x) - ftz_v[W](s * y))
    return ftz_v[W](mul_add_v[W](c, x, -ftz_v[W](s * y)))


@always_inline
def _rot_add_v(s: V, x: V, c: V, y: V) -> V:
    comptime if PCA_ORACLE_HOST_SABOTAGE:
        return ftz_v[W](ftz_v[W](s * x) + ftz_v[W](c * y))
    return ftz_v[W](mul_add_v[W](s, x, ftz_v[W](c * y)))


@always_inline
def _rot_sub_1(c: Float32, x: Float32, s: Float32, y: Float32) -> Float32:
    comptime if PCA_ORACLE_HOST_SABOTAGE:
        return ftz(ftz(c * x) - ftz(s * y))
    return ftz(identical_mul_add(c, x, -ftz(s * y)))


@always_inline
def _rot_add_1(s: Float32, x: Float32, c: Float32, y: Float32) -> Float32:
    comptime if PCA_ORACLE_HOST_SABOTAGE:
        return ftz(ftz(s * x) + ftz(c * y))
    return ftz(identical_mul_add(s, x, ftz(c * y)))


def _rotate(a: F32Ptr, b: F32Ptr, n: Int, c: Float32, s: Float32):
    """Rows a and b (two columns of the untransposed matrix), elementwise."""
    var cv = V(c)
    var sv = V(s)
    var k = 0
    while k + W <= n:
        var x = ftz_v[W](a.unsafe_load[width=W](k))
        var y = ftz_v[W](b.unsafe_load[width=W](k))
        a.unsafe_store(k, _rot_sub_v(cv, x, sv, y))
        b.unsafe_store(k, _rot_add_v(sv, x, cv, y))
        k += W
    while k < n:
        var x1 = ftz(a.unsafe_load(k))
        var y1 = ftz(b.unsafe_load(k))
        a.unsafe_store(k, _rot_sub_1(c, x1, s, y1))
        b.unsafe_store(k, _rot_add_1(s, x1, c, y1))
        k += 1


def _fold3(xp: F32Ptr, xq: F32Ptr, n: Int, lp: F32Ptr, lq: F32Ptr, lpq: F32Ptr):
    """The pair's SVD_TPB lane partials of app, aqq, apq: lane t folds
    elements t, t + SVD_TPB, ... ascending."""
    for t in range(SVD_TPB):
        lp.unsafe_store(t, Float32(0))
        lq.unsafe_store(t, Float32(0))
        lpq.unsafe_store(t, Float32(0))
    var j0 = 0
    while j0 < n:
        var t = 0
        while t + W <= SVD_TPB and j0 + t + W <= n:
            var a = ftz_v[W](xp.unsafe_load[width=W](j0 + t))
            var b = ftz_v[W](xq.unsafe_load[width=W](j0 + t))
            lp.unsafe_store(t, ftz_v[W](mul_add_v[W](a, a, lp.unsafe_load[width=W](t))))
            lq.unsafe_store(t, ftz_v[W](mul_add_v[W](b, b, lq.unsafe_load[width=W](t))))
            lpq.unsafe_store(t, ftz_v[W](mul_add_v[W](a, b, lpq.unsafe_load[width=W](t))))
            t += W
        while t < SVD_TPB and j0 + t < n:
            var a1 = ftz(xp.unsafe_load(j0 + t))
            var b1 = ftz(xq.unsafe_load(j0 + t))
            lp.unsafe_store(t, ftz(identical_mul_add(a1, a1, lp.unsafe_load(t))))
            lq.unsafe_store(t, ftz(identical_mul_add(b1, b1, lq.unsafe_load(t))))
            lpq.unsafe_store(t, ftz(identical_mul_add(a1, b1, lpq.unsafe_load(t))))
            t += 1
        j0 += SVD_TPB


def fast_one_sided_jacobi_svd(r: List[Float32], n: Int, max_sweeps: Int, tol: Float32) -> FastSvdResult:
    """`host_one_sided_jacobi_svd(r, n, max_sweeps, tol)` (r is not modified)."""
    var rt = List[Float32](length=n * n, fill=Float32(0))
    var vt = List[Float32](length=n * n, fill=Float32(0))
    for i in range(n):
        for j in range(n):
            rt[j * n + i] = r[i * n + j]
        vt[i * n + i] = Float32(1.0)
    var prt = F32Ptr(unsafe_from_address=Int(rt.unsafe_ptr()))
    var pvt = F32Ptr(unsafe_from_address=Int(vt.unsafe_ptr()))
    var lanes = List[Float32](length=3 * SVD_TPB, fill=Float32(0))
    var pl = F32Ptr(unsafe_from_address=Int(lanes.unsafe_ptr()))
    var lp = pl
    var lq = pl.unsafe_offset(SVD_TPB)
    var lpq = pl.unsafe_offset(2 * SVD_TPB)

    var executed = 0
    var converged = False
    for _sweep in range(max_sweeps):
        var rots = 0
        for p in range(n):
            for q in range(p + 1, n):
                var rp = prt.unsafe_offset(p * n)
                var rq = prt.unsafe_offset(q * n)
                _fold3(rp, rq, n, lp, lq, lpq)
                var app = _halving(lp)
                var aqq = _halving(lq)
                var apq = _halving(lpq)
                var np_ = ftz(identical_sqrt(app))
                var nq_ = ftz(identical_sqrt(aqq))
                var thresh = ftz(tol * ftz(np_ * nq_))
                if abs(apq) > thresh:
                    rots += 1
                    var cs = host_jacobi_rotation_cs(app, aqq, apq)
                    _rotate(rp, rq, n, cs[0], cs[1])
                    _rotate(pvt.unsafe_offset(p * n), pvt.unsafe_offset(q * n), n, cs[0], cs[1])
        executed += 1
        if rots == 0:
            converged = True
            break

    var sv = List[Float32](length=n, fill=Float32(0.0))
    for j in range(n):
        var col = prt.unsafe_offset(j * n)
        for t in range(SVD_TPB):
            lp.unsafe_store(t, Float32(0))
        var j0 = 0
        while j0 < n:
            var t = 0
            while t < SVD_TPB and j0 + t < n:
                var vv = ftz(col.unsafe_load(j0 + t))
                lp.unsafe_store(t, ftz(identical_mul_add(vv, vv, lp.unsafe_load(t))))
                t += 1
            j0 += SVD_TPB
        sv[j] = ftz(identical_sqrt(_halving(lp)))
    var v = List[Float32](length=n * n, fill=Float32(0))
    for k in range(n):
        for p in range(n):
            v[k * n + p] = vt[p * n + k]
    _ = rt^
    _ = vt^
    _ = lanes^
    return FastSvdResult(v^, sv^, converged, executed)
