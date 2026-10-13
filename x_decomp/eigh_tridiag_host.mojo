# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU column of the IDENTICAL tridiagonal eigh (lane
apple-neighbors-gpu-cpu, 2026-10-13): `DevExec._eigh_td_try`
(x_decomp/device.mojo) and `eigh_td_on` (x_decomp/eigh_tridiag.mojo) on the
host, the same words.

Why: since 2026-10-11 every IDENTICAL device eigh from n = TD_MIN_N takes the
tridiagonal route (EIGH_TD_IDN) on NVIDIA, AMD and Apple alike, while
`HostExec.eigh` kept the round-robin Jacobi. So the CPU column of every eigh
consumer from n = 64 differed from the GPU columns; on Apple it surfaced as
python/mojolearn/tests/test_x_neighbors_repeat.py's GPU/CPU digest split
(KernelPCA's dense solve at n = 96). This module makes the host take the same
route, so `HostExec.eigh` and `DevExec.eigh` agree again.

How each launch is reproduced (the kernels are in x_decomp/eigh_tridiag.mojo;
the shared statements in x_decomp/eigh_tridiag_cells.mojo):
  * one-thread-per-item kernels (`td_bisect_item`, `td_gap_item`,
    `td_vec_item`, `td_tfac_item`): the same items, one call per index.
  * the block folds of `td_col_kernel`, `td_gemv_kernel`, `td_p_kernel`,
    `td_w_kernel` and `td_gram_kernel`: each thread's strided partial
    (rows / partials at stride TD_TPB, the kernel's own statements), then
    `_td_tree`, the kernels' shared-memory tree over TD_TPB slots
    (`red[t] = tf_add(red[t], red[t + w])`, w = TD_TPB / 2 .. 1). Every
    result is flushed (`tf_*`, the denormal policy of eigh_tridiag_cells.mojo).
  * the identical GEMMs (`blk_gemm_sub`, `identical_gemm_into`): the
    contract's host answer, `host_gemm_oracle` (cluster/host/host_gemm_cells.mojo,
    bit for bit `gemm_oracle`), on the same packed operands; the block
    update's epilogue `C = ftz(ftz(C) - ftz(G))` (gemm/block_update.mojo
    `blk_sub_kernel`).
  * the tail: the range scale (`host_es_scale` / `host_es_unscale`, the twins
    of `enqueue_es_scale` / `enqueue_es_unscale`), `host_sign_flip`
    (`sign_flip_kernel`'s twin) and `eigh_ascending`
    (`enqueue_eigh_ascending`'s twin): the Jacobi route's host tail.
Every task writes only its own cells, so the host thread count moves no word.
"""
from checks.numerics import ftz
from cluster.host.host_cells import host_cells
from cluster.host.host_gemm_cells import host_gemm_oracle
from decomposition.host.linalg_public import eigh_ascending
from decomposition.host.pca_oracle import host_sign_flip
from gemm.contract import OP_NN, OP_NT, OP_TN
from x_decomp.cells import F32Ptr
from x_decomp.eigh_scale import host_es_scale, host_es_unscale
from x_decomp.eigh_tridiag_cells import (
    TD_NB,
    tf_add,
    tf_fma,
    tf_mul,
    TD_TPB,
    _td_cell_kern,
    _td_house,
    _td_v,
    _td_y,
    td_bisect_item,
    td_gap_item,
    td_tfac_item,
    td_vec_item,
)


@always_inline
def _lp(l: List[Float32]) -> F32Ptr:
    """A List's storage as the items' pointer type. The List must outlive
    every use (the callers keep each in a local until the end)."""
    return F32Ptr(unsafe_from_address=Int(l.unsafe_ptr()))


@always_inline
def _td_grid(count: Int) -> Int:
    """x_decomp/eigh_tridiag.mojo `_td_grid(count, TD_TPB)`."""
    return (count + TD_TPB - 1) // TD_TPB if count > 0 else 1


@always_inline
def _td_tree(mut red: InlineArray[Float32, TD_TPB]) -> Float32:
    """The kernels' shared-memory fold: w = TD_TPB / 2 .. 1, slot t < w
    takes tf_add(red[t], red[t + w]); slot 0 is the block's value."""
    var w = TD_TPB // 2
    while w > 0:
        for t in range(w):
            red[t] = tf_add(red[t], red[t + w])
        w = w // 2
    return red[0]


@always_inline
def _td_fold_parts(part: F32Ptr, np_: Int) -> Float32:
    """`td_gemv_kernel` / `td_w_kernel`'s fold of the np_ block partials:
    thread t sums part[t], part[t + TD_TPB], ... from +0.0, then the tree."""
    var red = InlineArray[Float32, TD_TPB](fill=Float32(0.0))
    for t in range(TD_TPB):
        var s = Float32(0.0)
        var q = t
        while q < np_:
            s = tf_add(s, part.unsafe_load(q))
            q += TD_TPB
        red[t] = s
    return _td_tree(red)


def _td_block_sub(mut c: List[Float32], ldc: Int, r0: Int, c0: Int, g: List[Float32], m: Int, n: Int):
    """`blk_sub_kernel` (GUARD off, lower off): C[r0 + i, c0 + j] =
    ftz(ftz(C) - ftz(G[i, j]))."""
    var cp = _lp(c)
    var gp = _lp(g)

    def cell(t: Int) {imm cp, imm gp, imm ldc, imm r0, imm c0, imm n}:
        var i = t // n
        var j = t - i * n
        var at = (r0 + i) * ldc + c0 + j
        cp.unsafe_store(at, ftz(ftz(cp.unsafe_load(at)) - ftz(gp.unsafe_load(t))))

    host_cells(cell, m * n, 2)


def _td_tridiag(
    a: F32Ptr, n: Int, mut vl: List[Float32], mut wl: List[Float32], xcol: F32Ptr, dd: F32Ptr, ee: F32Ptr,
    tau: F32Ptr, vv: F32Ptr, y: F32Ptr, tv: F32Ptr, pb: F32Ptr, part: F32Ptr, part2: F32Ptr,
) raises:
    """Stage 1 of `eigh_td_on`: the panels of TD_NB reflectors, per column
    `td_col_kernel`, `td_gemv_kernel`, `td_p_kernel`, `td_w_kernel`, per
    panel the EIGH_TD_IDN rank-2k update `A[kend:, kend:] -= [V | W] [W | V]^T`
    (one OP_NT product at k = 2 TD_NB)."""
    var nref = n - 1
    var vp = _lp(vl)
    var wp = _lp(wl)
    var k = 0
    while k < nref:
        var cnt = min(TD_NB, nref - k)
        for t in range(n * TD_NB):
            vl[t] = Float32(0.0)
            wl[t] = Float32(0.0)
        for jj in range(cnt):
            var j = k + jj
            var m = n - j - 1
            var nbm = _td_grid(m)

            # td_col_kernel: xcol, the block partials of ||x[1:]||^2, d_j
            def col_block(b: Int) {imm a, imm vp, imm wp, imm xcol, imm part, imm n, imm j, imm jj}:
                var red = InlineArray[Float32, TD_TPB](fill=Float32(0.0))
                for tid in range(TD_TPB):
                    var i = j + 1 + b * TD_TPB + tid
                    var s = Float32(0.0)
                    if i < n:
                        var x = _td_cell_kern(a, vp, wp, n, i, j, jj)
                        xcol.unsafe_store(i, x)
                        if i >= j + 2:
                            s = tf_mul(x, x)
                    red[tid] = s
                part.unsafe_store(b, _td_tree(red))

            host_cells(col_block, nbm, TD_TPB * (2 * jj + 2))
            dd.unsafe_store(j, _td_cell_kern(a, vp, wp, n, j, j, jj))
            if j == n - 2:
                dd.unsafe_store(n - 1, _td_cell_kern(a, vp, wp, n, n - 1, n - 1, jj))

            # td_gemv_kernel: the reflector (every block folds the same
            # partials), y = A_stale v rows, the 2 jj panel dots, the stores
            var hh = _td_house(xcol.unsafe_load(j + 1), _td_fold_parts(part, nbm))
            var scale = hh[2]

            def y_row(b: Int) {imm a, imm xcol, imm y, imm n, imm j, imm scale}:
                var red = InlineArray[Float32, TD_TPB](fill=Float32(0.0))
                var row = (j + 1 + b) * n
                for tid in range(TD_TPB):
                    var s = Float32(0.0)
                    var c = j + 1 + tid
                    while c < n:
                        s = tf_fma(a.unsafe_load(row + c), _td_v(xcol, j, scale, c), s)
                        c += TD_TPB
                    red[tid] = s
                y.unsafe_store(j + 1 + b, _td_tree(red))

            host_cells(y_row, m, 2 * m)

            def tv_dot(q: Int) {imm vp, imm wp, imm xcol, imm tv, imm n, imm j, imm jj, imm scale}:
                var red = InlineArray[Float32, TD_TPB](fill=Float32(0.0))
                for tid in range(TD_TPB):
                    var s = Float32(0.0)
                    var i = j + 1 + tid
                    if q < jj:
                        while i < n:
                            s = tf_fma(wp.unsafe_load(i * TD_NB + q), _td_v(xcol, j, scale, i), s)
                            i += TD_TPB
                    else:
                        var q2 = q - jj
                        while i < n:
                            s = tf_fma(vp.unsafe_load(i * TD_NB + q2), _td_v(xcol, j, scale, i), s)
                            i += TD_TPB
                    red[tid] = s
                tv.unsafe_store(q, _td_tree(red))

            host_cells(tv_dot, 2 * jj, 2 * m)
            for i in range(j + 1, n):
                var v = _td_v(xcol, j, scale, i)
                vv.unsafe_store(i, v)
                vp.unsafe_store(i * TD_NB + jj, v)
                a.unsafe_store(i * n + j, v)
            ee.unsafe_store(j, hh[0])
            tau.unsafe_store(j, hh[1])

            # td_p_kernel: pb = tau (y - V (W^T v) - W (V^T v)), block partials of pb . v
            def p_block(b: Int) {imm vp, imm wp, imm vv, imm y, imm tv, imm tau, imm pb, imm part2, imm n, imm j, imm jj}:
                var red = InlineArray[Float32, TD_TPB](fill=Float32(0.0))
                for tid in range(TD_TPB):
                    var i = j + 1 + b * TD_TPB + tid
                    var s = Float32(0.0)
                    if i < n:
                        var u = y.unsafe_load(i)
                        for p in range(jj):
                            u = tf_fma(-vp.unsafe_load(i * TD_NB + p), tv.unsafe_load(p), u)
                            u = tf_fma(-wp.unsafe_load(i * TD_NB + p), tv.unsafe_load(jj + p), u)
                        var pv = tf_mul(tau.unsafe_load(j), u)
                        pb.unsafe_store(i, pv)
                        s = tf_mul(pv, vv.unsafe_load(i))
                    red[tid] = s
                part2.unsafe_store(b, _td_tree(red))

            host_cells(p_block, nbm, TD_TPB * (4 * jj + 3))

            # td_w_kernel: W[:, jj] = pb - tau/2 (p . v) v
            var alpha2 = tf_mul(tf_mul(Float32(-0.5), tau.unsafe_load(j)), _td_fold_parts(part2, nbm))
            for i in range(j + 1, n):
                wp.unsafe_store(i * TD_NB + jj, tf_fma(alpha2, vv.unsafe_load(i), pb.unsafe_load(i)))
        var kend = k + cnt
        var mm = n - kend
        if mm > 0:
            # the four blk_pack_kernel copies: P1 = [V | W], P2 = [W | V]
            # (rows kend .. n - 1, 2 TD_NB columns), then blk_gemm_sub at OP_NT
            var p1 = List[Float32](length=mm * 2 * TD_NB, fill=Float32(0.0))
            var p2 = List[Float32](length=mm * 2 * TD_NB, fill=Float32(0.0))
            for i in range(mm):
                for q in range(TD_NB):
                    var vq = vl[(kend + i) * TD_NB + q]
                    var wq = wl[(kend + i) * TD_NB + q]
                    p1[i * 2 * TD_NB + q] = vq
                    p1[i * 2 * TD_NB + TD_NB + q] = wq
                    p2[i * 2 * TD_NB + q] = wq
                    p2[i * 2 * TD_NB + TD_NB + q] = vq
            var g = host_gemm_oracle(p1, p2, OP_NT, mm, mm, 2 * TD_NB)
            _td_block_sub_ptr(a, n, kend, kend, g, mm, mm)
            _ = p1^
            _ = p2^
            _ = g^
        k = kend


def _td_block_sub_ptr(c: F32Ptr, ldc: Int, r0: Int, c0: Int, g: List[Float32], m: Int, n: Int):
    """`_td_block_sub` on a raw matrix pointer."""
    var gp = _lp(g)

    def cell(t: Int) {imm c, imm gp, imm ldc, imm r0, imm c0, imm n}:
        var i = t // n
        var j = t - i * n
        var at = (r0 + i) * ldc + c0 + j
        c.unsafe_store(at, ftz(ftz(c.unsafe_load(at)) - ftz(gp.unsafe_load(t))))

    host_cells(cell, m * n, 2)


def host_eigh_td(mut m: List[Float32], n: Int, w: F32Ptr, v: F32Ptr) raises -> Bool:
    """`DevExec._eigh_td_try` on the host. `m` (n x n row major, the
    triangle already mirrored) is consumed: range-scaled, then overwritten
    by the reflectors. True: w (n, ascending) and v (n x n, vector c in
    COLUMN c) written. False: refused (clustered spectrum, nonfinite T or
    vector norm); nothing written, the caller runs the Jacobi on its own
    copy of the input, as the device does. n >= 2."""
    # td_copy_kernel's flush of the input (the denormal policy of
    # x_decomp/eigh_tridiag_cells.mojo), then the range scale
    for i in range(n * n):
        m[i] = ftz(m[i])
    var fac = host_es_scale(m, n)
    var a = _lp(m)
    var vl = List[Float32](length=n * TD_NB, fill=Float32(0.0))
    var wl = List[Float32](length=n * TD_NB, fill=Float32(0.0))
    var vvl = List[Float32](length=n, fill=Float32(0.0))
    var yl = List[Float32](length=n, fill=Float32(0.0))
    var tvl = List[Float32](length=2 * TD_NB, fill=Float32(0.0))
    var taul = List[Float32](length=n, fill=Float32(0.0))
    var ddl = List[Float32](length=n, fill=Float32(0.0))
    var eel = List[Float32](length=n, fill=Float32(0.0))
    var xl = List[Float32](length=n, fill=Float32(0.0))
    var pbl = List[Float32](length=n, fill=Float32(0.0))
    var partl = List[Float32](length=_td_grid(n), fill=Float32(0.0))
    var part2l = List[Float32](length=_td_grid(n), fill=Float32(0.0))
    var xcol = _lp(xl)
    var dd = _lp(ddl)
    var ee = _lp(eel)
    var tau = _lp(taul)
    # 1. tridiagonalize
    _td_tridiag(a, n, vl, wl, xcol, dd, ee, tau, _lp(vvl), _lp(yl), _lp(tvl), _lp(pbl), _lp(partl), _lp(part2l))
    # 2. eigenvalues: td_bisect_kernel, td_gap_kernel (info[0] = sc, info[2] = refused)
    var infol = List[Float32](length=4, fill=Float32(0.0))
    var whl = List[Float32](length=n, fill=Float32(0.0))
    var wll = List[Float32](length=n, fill=Float32(0.0))
    var wtl = List[Float32](length=n, fill=Float32(0.0))
    var info = _lp(infol)
    var wh = _lp(whl)
    var wlo = _lp(wll)
    var wt = _lp(wtl)

    def bis(k: Int) {imm dd, imm ee, imm info, imm wh, imm wlo, imm n}:
        td_bisect_item(dd, ee, info, wh, wlo, n, k)

    # every index writes only its own wh / wl cells; info[0] (index 0) and
    # info[2] (the same 1.0 from any index) are read only after the join
    host_cells(bis, n, 64 * 40 * n)

    def gap(k: Int) {imm wh, imm wlo, imm info, imm wt, imm n}:
        td_gap_item(wh, wlo, info, wt, n, k)

    host_cells(gap, n, 16)
    # 3. eigenvectors of T: td_vec_kernel (column k of z per index)
    var zl = List[Float32](length=n * n, fill=Float32(0.0))
    var dphl = List[Float32](length=n * n, fill=Float32(0.0))
    var dpll = List[Float32](length=n * n, fill=Float32(0.0))
    var dmhl = List[Float32](length=n * n, fill=Float32(0.0))
    var dmll = List[Float32](length=n * n, fill=Float32(0.0))
    var z = _lp(zl)
    var dph = _lp(dphl)
    var dpl = _lp(dpll)
    var dmh = _lp(dmhl)
    var dml = _lp(dmll)

    def vec(k: Int) {imm dd, imm ee, imm info, imm wh, imm wlo, imm dph, imm dpl, imm dmh, imm dml, imm z, imm n}:
        td_vec_item(dd, ee, info, wh, wlo, dph, dpl, dmh, dml, z, n, k)

    host_cells(vec, n, 120 * n)
    _ = dphl^
    _ = dpll^
    _ = dmhl^
    _ = dmll^
    if infol[2] != Float32(0.0):
        _ = vl^
        _ = wl^
        _ = vvl^
        _ = yl^
        _ = tvl^
        _ = taul^
        _ = ddl^
        _ = eel^
        _ = xl^
        _ = pbl^
        _ = partl^
        _ = part2l^
        _ = infol^
        _ = whl^
        _ = wll^
        _ = wtl^
        _ = zl^
        return False
    # 4. V = Q Z: td_gram_kernel, td_tfac_kernel, then panels last to first
    # Z[kp + 1:, :] -= Y (T_p (Y^T Z[kp + 1:, :])) as the three identical GEMMs
    var nref = n - 1
    var npan = (nref + TD_NB - 1) // TD_NB
    var gml = List[Float32](length=npan * TD_NB * TD_NB, fill=Float32(0.0))
    var tfl = List[Float32](length=npan * TD_NB * TD_NB, fill=Float32(0.0))
    var gm = _lp(gml)
    var tf = _lp(tfl)

    def gram(t: Int) {imm a, imm gm, imm n}:
        var panel = t // (TD_NB * TD_NB)
        var b = t - panel * TD_NB * TD_NB
        var k = panel * TD_NB
        var cnt = min(TD_NB, n - 1 - k)
        var qa = b // TD_NB
        var qb = b - qa * TD_NB
        var red = InlineArray[Float32, TD_TPB](fill=Float32(0.0))
        for tid in range(TD_TPB):
            var s = Float32(0.0)
            if qa < cnt and qb < cnt:
                var i = k + 1 + tid
                while i < n:
                    s = tf_fma(_td_y(a, n, k, cnt, i, qa), _td_y(a, n, k, cnt, i, qb), s)
                    i += TD_TPB
            red[tid] = s
        gm.unsafe_store(panel * TD_NB * TD_NB + b, _td_tree(red))

    host_cells(gram, npan * TD_NB * TD_NB, 2 * n)
    for t in range(npan * TD_NB):
        var panel = t // TD_NB
        td_tfac_item(gm, tau, tf, n, panel, t - panel * TD_NB)
    var p = npan - 1
    while p >= 0:
        var kp = p * TD_NB
        var cnt = min(TD_NB, nref - kp)
        var mrows = n - kp - 1
        # td_pack_y_kernel: Y (mrows x TD_NB)
        var yb = List[Float32](length=mrows * TD_NB, fill=Float32(0.0))
        for i in range(mrows):
            for q in range(TD_NB):
                yb[i * TD_NB + q] = _td_y(a, n, kp, cnt, kp + 1 + i, q)
        # the Z rows kp + 1 .. n - 1 as the OP_TN right operand (mrows x n)
        var zr = List[Float32](length=mrows * n, fill=Float32(0.0))
        for t in range(mrows * n):
            zr[t] = zl[(kp + 1) * n + t]
        var s32 = host_gemm_oracle(yb, zr, OP_TN, TD_NB, n, mrows)
        var tp = List[Float32](length=TD_NB * TD_NB, fill=Float32(0.0))
        for t in range(TD_NB * TD_NB):
            tp[t] = tfl[p * TD_NB * TD_NB + t]
        var s2 = host_gemm_oracle(tp, s32, OP_NN, TD_NB, n, TD_NB)
        var g = host_gemm_oracle(yb, s2, OP_NN, mrows, n, TD_NB)
        _td_block_sub(zl, n, kp + 1, 0, g, mrows, n)
        _ = yb^
        _ = zr^
        _ = s32^
        _ = tp^
        _ = s2^
        _ = g^
        p -= 1
    # the tail of `_eigh_td_try`: sign_flip_kernel, the ascending
    # permutation keyed by td_gap_kernel's values, the range unscale
    host_sign_flip(zl, n)
    var got = eigh_ascending(wtl, zl, n, True, 0)
    var wout = got.w.copy()
    host_es_unscale(wout, fac)
    for i in range(n):
        w.unsafe_store(i, wout[i])
    for i in range(n * n):
        v.unsafe_store(i, got.v[i])
    _ = vl^
    _ = wl^
    _ = vvl^
    _ = yl^
    _ = tvl^
    _ = taul^
    _ = ddl^
    _ = eel^
    _ = xl^
    _ = pbl^
    _ = partl^
    _ = part2l^
    _ = infol^
    _ = whl^
    _ = wll^
    _ = wtl^
    _ = zl^
    _ = gml^
    _ = tfl^
    return True
