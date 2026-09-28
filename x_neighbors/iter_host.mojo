# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU column of `x_neighbors/iter_device.mojo`: the same loop over the
same items. HOST ONLY."""
from std.memory import bitcast
from std.sys.compile import is_defined

from x_neighbors.items import (
    FP, IP, absdiff_sum_item, matmul_item, lp_clamp_item, ls_clamp_item,
    pagerank_step_item, cc_step_item, pcs_item, knn_sq_item,
)

from x_neighbors.host_ops import op_knn_impute_cells

comptime _SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def op_lp_iterate(
    g: Int, ld: Int, ystatic: Int, unlabeled: Int, info: Int,
    n: Int, c: Int, max_iter: Int, variant: Int, tol_hi: Int, tol_lo: Int,
    alpha: Float32,
) raises:
    var tol = bitcast[DType.float64]((UInt64(tol_hi) << UInt64(32)) | UInt64(tol_lo))
    var nc = n * c
    var pg = FP(unsafe_from_address=g)
    var pys = FP(unsafe_from_address=ystatic)
    var punl = IP(unsafe_from_address=unlabeled)
    var a = List[Float32](length=nc if nc > 0 else 1, fill=Float32(0))
    var b = List[Float32](length=nc if nc > 0 else 1, fill=Float32(0))
    var nxt = List[Float32](length=nc if nc > 0 else 1, fill=Float32(0))
    var s = List[Float32](length=1, fill=Float32(0))
    var pld = FP(unsafe_from_address=ld)
    for i in range(nc):
        a[i] = pld.unsafe_load(i)
    var cur = FP(unsafe_from_address=Int(a.unsafe_ptr()))
    var prev = FP(unsafe_from_address=Int(b.unsafe_ptr()))
    var pn = FP(unsafe_from_address=Int(nxt.unsafe_ptr()))
    var ps = FP(unsafe_from_address=Int(s.unsafe_ptr()))
    var n_iter = 0
    var converged = False
    for it in range(max_iter):
        n_iter = it
        absdiff_sum_item(0, cur, prev, ps, nc)
        if Float64(ps.unsafe_load(0)) < tol:
            converged = True
            break
        for t in range(nc):
            matmul_item(t, pg, cur, pn, n, n, c)
        if variant == 0:
            for t in range(n):
                lp_clamp_item(t, pn, pys, punl, prev, n, c)
        else:
            for t in range(nc):
                ls_clamp_item(t, pn, pys, prev, nc, alpha)
        var tmp = cur
        cur = prev
        prev = tmp
    if not converged:
        n_iter += 1
    for i in range(nc):
        pld.unsafe_store(i, cur.unsafe_load(i))
    comptime if _SABOTAGE:
        if nc > 0:
            pld.unsafe_store(0, pld.unsafe_load(0) + Float32(1e-3))
    var inf = IP(unsafe_from_address=info)
    inf.unsafe_store(0, Int32(n_iter))
    inf.unsafe_store(1, Int32(1 if converged else 0))
    _ = a^
    _ = b^
    _ = nxt^
    _ = s^


def op_pr_iterate(
    q: Int, x: Int, p: Int, dw: Int, dangling: Int, info: Int,
    n: Int, max_iter: Int, thr_hi: Int, thr_lo: Int, alpha: Float32,
) raises:
    var thr = bitcast[DType.float64]((UInt64(thr_hi) << UInt64(32)) | UInt64(thr_lo))
    var a = List[Float32](length=n if n > 0 else 1, fill=Float32(0))
    var b = List[Float32](length=n if n > 0 else 1, fill=Float32(0))
    var s = List[Float32](length=1, fill=Float32(0))
    var px = FP(unsafe_from_address=x)
    for i in range(n):
        a[i] = px.unsafe_load(i)
    var cur = FP(unsafe_from_address=Int(a.unsafe_ptr()))
    var nxt = FP(unsafe_from_address=Int(b.unsafe_ptr()))
    var ps = FP(unsafe_from_address=Int(s.unsafe_ptr()))
    var pq = FP(unsafe_from_address=q)
    var pp = FP(unsafe_from_address=p)
    var pdw = FP(unsafe_from_address=dw)
    var pdg = IP(unsafe_from_address=dangling)
    var n_iter = 0
    var converged = False
    for it in range(max_iter):
        for t in range(n):
            pagerank_step_item(t, pq, cur, pp, pdw, pdg, nxt, n, alpha)
        absdiff_sum_item(0, nxt, cur, ps, n)
        var tmp = cur
        cur = nxt
        nxt = tmp
        n_iter = it + 1
        if Float64(ps.unsafe_load(0)) < thr:
            converged = True
            break
    for i in range(n):
        px.unsafe_store(i, cur.unsafe_load(i))
    comptime if _SABOTAGE:
        if n > 0:
            px.unsafe_store(0, px.unsafe_load(0) + Float32(1e-3))
    var inf = IP(unsafe_from_address=info)
    inf.unsafe_store(0, Int32(n_iter))
    inf.unsafe_store(1, Int32(1 if converged else 0))
    _ = a^
    _ = b^
    _ = s^


def op_cc_iterate(a: Int, lab: Int, info: Int, n: Int) raises:
    var l0 = List[Int32](length=n if n > 0 else 1, fill=Int32(0))
    var l1 = List[Int32](length=n if n > 0 else 1, fill=Int32(0))
    var pl = IP(unsafe_from_address=lab)
    for i in range(n):
        l0[i] = pl.unsafe_load(i)
    var cur = IP(unsafe_from_address=Int(l0.unsafe_ptr()))
    var nxt = IP(unsafe_from_address=Int(l1.unsafe_ptr()))
    var pa = FP(unsafe_from_address=a)
    var steps = 0
    while True:
        for t in range(n):
            cc_step_item(t, pa, cur, nxt, n)
        steps += 1
        var same = True
        for i in range(n):
            if nxt.unsafe_load(i) != cur.unsafe_load(i):
                same = False
                break
        if same:
            break
        var tmp = cur
        cur = nxt
        nxt = tmp
    for i in range(n):
        pl.unsafe_store(i, cur.unsafe_load(i))
    IP(unsafe_from_address=info).unsafe_store(0, Int32(steps))
    _ = l0^
    _ = l1^


def op_pcs_resident(
    x: Int, hidx: Int, hbit: Int, res: Int,
    n: Int, d_in: Int, nf: Int, nc: Int, degree: Int, gamma: Float32, coef0: Float32,
) raises:
    """The CPU column: `pcs_item` itself (the GPU's split runs its statements)."""
    var scr = List[Float32](length=n * 2 * nc if n * nc > 0 else 1, fill=Float32(0))
    var ps = FP(unsafe_from_address=Int(scr.unsafe_ptr()))
    for t in range(n):
        pcs_item(t, FP(unsafe_from_address=x), IP(unsafe_from_address=hidx), IP(unsafe_from_address=hbit),
                 FP(unsafe_from_address=res), ps, n, d_in, nf, nc, degree, gamma, coef0)
    comptime if _SABOTAGE:
        if n * nc > 0:
            FP(unsafe_from_address=res).unsafe_store(0, FP(unsafe_from_address=res).unsafe_load(0) + Float32(1e-3))
    _ = scr^


def op_knn_sq_tiled(
    x: Int, y: Int, dist: Int, idx: Int, n: Int, m: Int, d: Int, k: Int, exclude_self: Int,
) raises:
    """The CPU column: `knn_sq_item` itself."""
    for t in range(n):
        knn_sq_item(t, FP(unsafe_from_address=x), FP(unsafe_from_address=y), FP(unsafe_from_address=dist),
                    IP(unsafe_from_address=idx), n, m, d, k, exclude_self)
    comptime if _SABOTAGE:
        if n * k > 0:
            FP(unsafe_from_address=dist).unsafe_store(0, FP(unsafe_from_address=dist).unsafe_load(0) + Float32(1e-3))


def op_knn_impute_tiled(
    cells: Int, x: Int, fx: Int, res: Int,
    n: Int, m: Int, d: Int, k: Int, weights: Int, nc: Int,
) raises:
    """The CPU column: `knn_impute_cells` (the item per missing cell)."""
    op_knn_impute_cells(cells, x, fx, res, n, m, d, k, weights, nc)
