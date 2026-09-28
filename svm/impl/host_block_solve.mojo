# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple: the SMO block solve on the host cores (lane
neighbors-apple3, 2026-09-28).

The block solve is a serial loop: every inner iteration picks two elements
of the working set by three reductions over it and updates `f` over it. On
the GPU that is one block whose every iteration is three barriers and two
cross-simdgroup folds, about 3.7 us an iteration on the M4 Pro (SVC taxi
10k: 99,358 inner iterations, 0.37 to 0.41 s of a 0.44 s FAST fit). On the
host the same three scans and the update are three vector passes (the
first two scans share one) over at most 2,048 floats that sit in the first
level cache.

Apple's memory is unified, so what crosses per OUTER iteration is one copy
of the square kernel tile, `f` and the working set indices down, and
`alpha` and `delta_alpha` up.

The selections are the device kernel's total order (value, then the smaller
training index; `fast_block_solve.mojo`, `smo_block_solve_kernel`) and every
update is its expression. Under FAST the products and sums are the
compiler's, so a word may differ from the device solve's by a rounding; the
paired quality check is bench/x_neighbors_fast_quality.py (svc, svr).
"""
from std.math import inf
from std.sys.compile import is_defined

from checks.numerics import ftz, ftz_simd, identical_mul_add, identical_mul_add_simd
from svm.impl.smo_sets import in_lower, in_upper

#: Lanes per vector pass. A vector's lanes are different elements of the
#: working set, so the width moves no word; 16 (four NEON registers) keeps
#: the pass short against the select chain it carries.
#: `-D MOJOLEARN_SVM_HBS_W4` / `_W8` are the measurement arms.
comptime HBS_W = 4 if is_defined["MOJOLEARN_SVM_HBS_W4"]() else (
    8 if is_defined["MOJOLEARN_SVM_HBS_W8"]() else 16
)
comptime HbsF = SIMD[DType.float32, HBS_W]
comptime HbsI = SIMD[DType.int32, HBS_W]
comptime HbsB = SIMD[DType.bool, HBS_W]
comptime HbsFP = MutPointer[Float32, MutAnyOrigin]
comptime HbsIP = MutPointer[Int32, MutAnyOrigin]
comptime HBS_ETA_EPS = Float32(1.0e-12)
comptime HBS_NO_KEY = Int32(2147483647)


@always_inline
def _upper_v(a: HbsF, y: HbsF, C: HbsF) -> HbsB:
    return (y.lt(HbsF(0.0)) & a.gt(HbsF(0.0))) | (y.gt(HbsF(0.0)) & a.lt(C))


@always_inline
def _lower_v(a: HbsF, y: HbsF, C: HbsF) -> HbsB:
    return (y.lt(HbsF(0.0)) & a.lt(C)) | (y.gt(HbsF(0.0)) & a.gt(HbsF(0.0)))


@always_inline
def _gain(f_u: Float32, ft: Float32, kdt: Float32, kdu: Float32, kui: Float32) -> Float32:
    var eta = ftz(ftz(kdt + kdu) - ftz(Float32(2.0) * kui))
    if eta < HBS_ETA_EPS:
        eta = HBS_ETA_EPS
    var d = ftz(f_u - ft)
    return ftz(ftz(d * d) / eta)


@always_inline
def _lane_ids() -> HbsI:
    var lanes = HbsI(0)
    comptime for l in range(HBS_W):
        lanes[l] = Int32(l)
    return lanes


def _scan_extremes(
    fp: HbsFP, ap: HbsFP, yp: HbsFP, cp: HbsFP, keyp: HbsIP, n: Int
) -> Tuple[Float32, Int, Float32]:
    """The first two reductions in one pass: (f_u, u, f_max). f_u is the
    least f over the upper set and u its position, f_max the greatest f over
    the lower set; each the best of the strict total order (value, then the
    smaller key)."""
    var pos_inf = inf[DType.float32]()
    var neg_inf = -inf[DType.float32]()
    var uv = HbsF(pos_inf)
    var uk = HbsI(HBS_NO_KEY)
    var ut = HbsI(-1)
    var mv = HbsF(neg_inf)
    var mk = HbsI(HBS_NO_KEY)
    var lanes = _lane_ids()
    var whole = (n // HBS_W) * HBS_W
    var t0 = 0
    while t0 < whole:
        var f = fp.unsafe_load[width=HBS_W](t0)
        var a = ap.unsafe_load[width=HBS_W](t0)
        var y = yp.unsafe_load[width=HBS_W](t0)
        var C = cp.unsafe_load[width=HBS_W](t0)
        var key = keyp.unsafe_load[width=HBS_W](t0)
        var vu = _upper_v(a, y, C).select(f, HbsF(pos_inf))
        var vm = _lower_v(a, y, C).select(f, HbsF(neg_inf))
        var bu = vu.lt(uv) | (vu.eq(uv) & key.lt(uk))
        uv = bu.select(vu, uv)
        uk = bu.select(key, uk)
        ut = bu.select(lanes + HbsI(Int32(t0)), ut)
        var bm = vm.gt(mv) | (vm.eq(mv) & key.lt(mk))
        mv = bm.select(vm, mv)
        mk = bm.select(key, mk)
        t0 += HBS_W
    var f_u = pos_inf
    var u_key = HBS_NO_KEY
    var u = -1
    var f_max = neg_inf
    var m_key = HBS_NO_KEY
    for l in range(HBS_W):
        if uv[l] < f_u or (uv[l] == f_u and uk[l] < u_key):
            f_u = uv[l]
            u_key = uk[l]
            u = Int(ut[l])
        if mv[l] > f_max or (mv[l] == f_max and mk[l] < m_key):
            f_max = mv[l]
            m_key = mk[l]
    for t in range(whole, n):
        var a1 = ap.unsafe_load(t)
        var y1 = yp.unsafe_load(t)
        var C1 = cp.unsafe_load(t)
        var k1 = keyp.unsafe_load(t)
        var vu1 = fp.unsafe_load(t) if in_upper(a1, y1, C1) else pos_inf
        var vm1 = fp.unsafe_load(t) if in_lower(a1, y1, C1) else neg_inf
        if vu1 < f_u or (vu1 == f_u and k1 < u_key):
            f_u = vu1
            u_key = k1
            u = t
        if vm1 > f_max or (vm1 == f_max and k1 < m_key):
            f_max = vm1
            m_key = k1
    return (f_u, u, f_max)


def _scan_gain(
    fp: HbsFP, ap: HbsFP, yp: HbsFP, cp: HbsFP, keyp: HbsIP, n: Int,
    f_u: Float32, kdp: HbsFP, kdu: Float32, trow: HbsFP,
) -> Int:
    """The third reduction: the position of the greatest
    `(f_u - f)^2 / eta` over the lower set with f_u < f (value, then the
    smaller key)."""
    var neg_inf = -inf[DType.float32]()
    var bv = HbsF(neg_inf)
    var bk = HbsI(HBS_NO_KEY)
    var bt = HbsI(-1)
    var lanes = _lane_ids()
    var whole = (n // HBS_W) * HBS_W
    var t0 = 0
    while t0 < whole:
        var f = fp.unsafe_load[width=HBS_W](t0)
        var a = ap.unsafe_load[width=HBS_W](t0)
        var y = yp.unsafe_load[width=HBS_W](t0)
        var C = cp.unsafe_load[width=HBS_W](t0)
        var key = keyp.unsafe_load[width=HBS_W](t0)
        var kd = kdp.unsafe_load[width=HBS_W](t0)
        var kui = trow.unsafe_load[width=HBS_W](t0)
        var eta = ftz_simd[HBS_W](
            ftz_simd[HBS_W](kd + HbsF(kdu)) - ftz_simd[HBS_W](HbsF(2.0) * kui)
        )
        eta = eta.lt(HbsF(HBS_ETA_EPS)).select(HbsF(HBS_ETA_EPS), eta)
        var d = ftz_simd[HBS_W](HbsF(f_u) - f)
        var g = ftz_simd[HBS_W](ftz_simd[HBS_W](d * d) / eta)
        var ok = HbsF(f_u).lt(f) & _lower_v(a, y, C)
        var v = ok.select(g, HbsF(neg_inf))
        var better = v.gt(bv) | (v.eq(bv) & key.lt(bk))
        bv = better.select(v, bv)
        bk = better.select(key, bk)
        bt = better.select(lanes + HbsI(Int32(t0)), bt)
        t0 += HBS_W
    var best_v = neg_inf
    var best_k = HBS_NO_KEY
    var best_t = -1
    for l in range(HBS_W):
        if bv[l] > best_v or (bv[l] == best_v and bk[l] < best_k):
            best_v = bv[l]
            best_k = bk[l]
            best_t = Int(bt[l])
    for t in range(whole, n):
        var v1 = neg_inf
        if f_u < fp.unsafe_load(t) and in_lower(ap.unsafe_load(t), yp.unsafe_load(t), cp.unsafe_load(t)):
            v1 = _gain(f_u, fp.unsafe_load(t), kdp.unsafe_load(t), kdu, trow.unsafe_load(t))
        var k1 = keyp.unsafe_load(t)
        if v1 > best_v or (v1 == best_v and k1 < best_k):
            best_v = v1
            best_k = k1
            best_t = t
    return best_t


def host_block_solve(
    ws_idx: HbsIP,
    n_ws: Int,
    y_array: HbsFP,
    alpha: HbsFP,
    f_array: HbsFP,
    tile: HbsFP,
    C_vec: HbsFP,
    eps: Float32,
    max_iter: Int,
    delta_alpha: HbsFP,
) -> Tuple[Float32, Int]:
    """`smo_block_solve_kernel` over host memory. `y_array`, `alpha`,
    `f_array` and `C_vec` are the n_train vectors, `tile` the n_ws x n_ws
    square kernel tile, `ws_idx` the working set (indices into n_train).
    Writes `alpha` at the working set and `delta_alpha[0, n_ws)`; returns
    (the first iteration's diff, the inner iterations)."""
    var y = List[Float32](length=n_ws, fill=Float32(0.0))
    var f = List[Float32](length=n_ws, fill=Float32(0.0))
    var a = List[Float32](length=n_ws, fill=Float32(0.0))
    var a_save = List[Float32](length=n_ws, fill=Float32(0.0))
    var C = List[Float32](length=n_ws, fill=Float32(0.0))
    var Kd = List[Float32](length=n_ws, fill=Float32(0.0))
    var key = List[Int32](length=n_ws, fill=Int32(0))
    for t in range(n_ws):
        var idx = Int(ws_idx.unsafe_load(t))
        y[t] = y_array.unsafe_load(idx)
        f[t] = f_array.unsafe_load(idx)
        a[t] = alpha.unsafe_load(idx)
        a_save[t] = a[t]
        C[t] = C_vec.unsafe_load(idx)
        Kd[t] = tile.unsafe_load(t + t * n_ws)
        key[t] = Int32(idx)
    var fp = HbsFP(unsafe_from_address=Int(f.unsafe_ptr()))
    var ap = HbsFP(unsafe_from_address=Int(a.unsafe_ptr()))
    var yp = HbsFP(unsafe_from_address=Int(y.unsafe_ptr()))
    var cp = HbsFP(unsafe_from_address=Int(C.unsafe_ptr()))
    var kdp = HbsFP(unsafe_from_address=Int(Kd.unsafe_ptr()))
    var keyp = HbsIP(unsafe_from_address=Int(key.unsafe_ptr()))

    var diff0 = Float32(0.0)
    var diff_end = Float32(0.0)
    var n_iter = 0
    while n_iter < max_iter:
        var ru = _scan_extremes(fp, ap, yp, cp, keyp, n_ws)
        var f_u = ru[0]
        var u = ru[1]
        if u < 0:
            u = 0
        var f_max = ru[2]
        var diff = ftz(f_max - f_u)
        if n_iter == 0:
            diff0 = diff
            var d10 = ftz(Float32(0.1) * diff)
            diff_end = eps if eps > d10 else d10
        if diff < diff_end:
            break
        var tu = tile.unsafe_offset(u * n_ws)
        var kdu = kdp.unsafe_load(u)
        var l = _scan_gain(fp, ap, yp, cp, keyp, n_ws, f_u, kdp, kdu, tu)
        if l < 0:
            l = 0
        var tl = tile.unsafe_offset(l * n_ws)
        var au = ap.unsafe_load(u)
        var al = ap.unsafe_load(l)
        var yu = yp.unsafe_load(u)
        var yl = yp.unsafe_load(l)
        var tmp_u = cp.unsafe_load(u) - au if yu > Float32(0.0) else au
        var tmp_l = al if yl > Float32(0.0) else cp.unsafe_load(l) - al
        var eta_ul = ftz(ftz(kdu + kdp.unsafe_load(l)) - ftz(Float32(2.0) * tu.unsafe_load(l)))
        if eta_ul < HBS_ETA_EPS:
            eta_ul = HBS_ETA_EPS
        var q_l = ftz(ftz(fp.unsafe_load(l) - f_u) / eta_ul)
        if q_l < tmp_l:
            tmp_l = q_l
        var q = tmp_u if tmp_u < tmp_l else tmp_l
        # the device kernel updates u, then l (the same element when u == l)
        ap.unsafe_store(u, ftz(ap.unsafe_load(u) + q * yu))
        ap.unsafe_store(l, ftz(ap.unsafe_load(l) - q * yl))
        var qv = HbsF(q)
        var t = 0
        while t + HBS_W <= n_ws:
            var dk = ftz_simd[HBS_W](tu.unsafe_load[width=HBS_W](t) - tl.unsafe_load[width=HBS_W](t))
            fp.unsafe_store(
                t, ftz_simd[HBS_W](identical_mul_add_simd[HBS_W](qv, dk, fp.unsafe_load[width=HBS_W](t)))
            )
            t += HBS_W
        while t < n_ws:
            var dk1 = ftz(tu.unsafe_load(t) - tl.unsafe_load(t))
            fp.unsafe_store(t, ftz(identical_mul_add(q, dk1, fp.unsafe_load(t))))
            t += 1
        if q == Float32(0.0):
            break
        n_iter += 1

    for t in range(n_ws):
        var idx = Int(ws_idx.unsafe_load(t))
        var a_new = ap.unsafe_load(t)
        alpha.unsafe_store(idx, a_new)
        delta_alpha.unsafe_store(t, ftz(ftz(a_new - a_save[t]) * y[t]))
    _ = y^
    _ = f^
    _ = a^
    _ = a_save^
    _ = C^
    _ = Kd^
    _ = key^
    return (diff0, n_iter)
