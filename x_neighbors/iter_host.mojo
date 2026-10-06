from experiments.classical_identical_ideas.graph_controls import C43_RESIDENT_NORMALIZATION
from x_neighbors.classical_graph import classical_graph_degree, classical_graph_product
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU column of `x_neighbors/iter_device.mojo`: the same loop over the
same items. HOST ONLY."""
from x_neighbors.svgp_ff import matmul_tn_acc_ff_item, svgp_ff_solve
from std.memory import bitcast
from std.sys.compile import is_defined
from std.python import PythonObject

from x_neighbors.cc_sparse import cc_iterate_sparse, cc_iterate_csr
from x_neighbors.nan_cells import nan_cells_host
from core.host_lanes import host_row_tasks
from core.host_parallel import host_parallelize
from x_neighbors.items import (
    FP, IP, absdiff_sum_item, matmul_item, lp_clamp_item, ls_clamp_item,
    pagerank_step_item, cc_step_item, pcs_item, knn_sq_item, nc_stats_item,
)

from x_neighbors.host_ops import op_knn_impute_cells, X_NEIGHBORS_HOST_SABOTAGE
from x_neighbors.items import (
    kernel_item, rowsum_item, scale_div_item, kpca_center_item, unary_item, svgp_var_item,
    K_RBF, U_IDENTITY,
)
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count

comptime _SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def op_lp_iterate(
    g: Int, ld: Int, ystatic: Int, unlabeled: Int, info: Int,
    n: Int, c: Int, max_iter: Int, variant: Int, tol_hi: Int, tol_lo: Int,
    alpha: Float32,
) raises:
    var tol = bitcast[DType.float64]((UInt64(tol_hi) << UInt64(32)) | UInt64(tol_lo))
    var raw_graph = C43_RESIDENT_NORMALIZATION and variant >= 2
    var clamp_variant = variant % 2
    var nc = n * c
    var pg = FP(unsafe_from_address=g)
    var degree = List[Float32](length=max(n,1) if raw_graph else 1,fill=Float32(0))
    var pd = degree.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    if raw_graph:
        for row in range(n):
            classical_graph_degree(row,pg,pd,n,clamp_variant)
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
            if raw_graph:
                classical_graph_product(t,pg,pd,cur,pn,n,c,clamp_variant)
            else:
                matmul_item(t, pg, cur, pn, n, n, c)
        if clamp_variant == 0:
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
    _ = degree^
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



def op_pr_iterate_sparse(
    a: Int, x: Int, p: Int, dw: Int, info: Int,
    n: Int, max_iter: Int, thr_hi: Int, thr_lo: Int, binary: Int, alpha: Float32,
) raises:
    """`op_pr_iterate` over the column lists of the dense adjacency: the
    host column of the device's items in the same order
    (x_neighbors/graph_par.mojo `pr_drive`, lane hr-graph)."""
    from x_neighbors.graph_host import pr_iterate_cpu

    var thr = bitcast[DType.float64]((UInt64(thr_hi) << UInt64(32)) | UInt64(thr_lo))
    pr_iterate_cpu(a, x, p, dw, info, n, max_iter, thr, binary, alpha)

def op_nan_cells(x: Int, cells: Int, colmiss: Int, info: Int, n: Int, d: Int) raises:
    """lane/neural-pass71: the NaN cells of x (n x d): flat indices
    ascending into `cells`, the NaN count per column, the total in info[0].
    The host pass on every column (x_neighbors/nan_cells.mojo)."""
    nan_cells_host(FP(unsafe_from_address=x), IP(unsafe_from_address=cells),
                   IP(unsafe_from_address=colmiss), IP(unsafe_from_address=info), n, d)


def op_cc_iterate_csr(indptr: Int, indices: Int, lab: Int, info: Int, n: Int, nnz: Int) raises:
    """lane/neural-pass69: `op_cc_iterate` from a CSR adjacency (indptr n + 1,
    indices nnz): the host walk of x_neighbors/cc_sparse.mojo on every
    column, no dense matrix."""
    cc_iterate_csr(IP(unsafe_from_address=indptr), IP(unsafe_from_address=indices),
                   IP(unsafe_from_address=lab), IP(unsafe_from_address=info), n, nnz)


def op_cc_iterate(a: Int, lab: Int, info: Int, n: Int) raises:
    """connected_components' min-label rounds as the sparse walk of
    x_neighbors/cc_sparse.mojo (lane neural-pass22): the same rounds over the
    graph's edges, the same labels and round count; `cc_iterate_dense` below
    is the dense loop it is held to (cc_sparse_check)."""
    cc_iterate_sparse(FP(unsafe_from_address=a), IP(unsafe_from_address=lab), IP(unsafe_from_address=info), n)


def cc_iterate_dense(a: Int, lab: Int, info: Int, n: Int) raises:
    """The rounds of `cc_step_item` over every cell of the n x n matrix: the
    reference the sparse walk is checked against."""
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


# ============================================================================
# FUSED KERNEL CHAINS, THE CPU COLUMN (lane/py-dn-kern, 2026-09-28): the
# items of `x_neighbors/iter_device.mojo`'s fused drivers in the same order
# over the same row tiles. Independent cells are split over host tasks
# (each cell written by one task), so the bits do not depend on the task
# count.
# ============================================================================

comptime XN_FUSED_CELLS = 1 << 24
#: `kind` of `kpca_transform` when q IS the precomputed kernel (d == nf)
comptime XN_PRECOMPUTED_KIND = 100


def _tile_rows(n: Int, width: Int) -> Int:
    var t = XN_FUSED_CELLS // max(width, 1)
    return max(1, min(n, t))


def _cells[F: def(Int) -> None](ref f: F, count: Int):
    """f(t) for t in [0, count), cut into host tasks by count only."""
    if count <= 0:
        return
    var tasks = host_predict_task_count(count)
    var part = host_predict_chunk(count, tasks)

    def _task(k: Int) {imm f, imm part, imm count}:
        var lo = k * part
        var hi = min(lo + part, count)
        for t in range(lo, hi):
            f(t)

    if tasks <= 1:
        _task(0)
    else:
        host_parallelize(_task, tasks)


def _hsab(addr: Int, count: Int):
    comptime if _SABOTAGE:
        if count > 0:
            FP(unsafe_from_address=addr).unsafe_store(0, FP(unsafe_from_address=addr).unsafe_load(0) + Float32(1e-3))


def _kernel_rows(q: FP, y: FP, k: FP, rows: Int, m: Int, d: Int, kind: Int, degree: Int, gamma: Float32, coef0: Float32):
    def f(t: Int) {imm q, imm y, imm k, imm rows, imm m, imm d, imm kind, imm degree, imm gamma, imm coef0}:
        kernel_item(t, q, y, k, rows, m, d, kind, gamma, coef0, degree)
    _cells(f, rows * m)


def _matmul_rows(a: FP, b: FP, res: FP, n: Int, k: Int, m: Int):
    def f(t: Int) {imm a, imm b, imm res, imm n, imm k, imm m}:
        matmul_item(t, a, b, res, n, k, m)
    _cells(f, n * m)


def op_kpca_transform(
    q: Int, fitx: Int, fit_cols: Int, fit_all: Int, alphas: Int, res: Int,
    nq: Int, nf: Int, d: Int, c: Int, kind: Int, degree: Int, gamma: Float32, coef0: Float32, s: Float32,
) raises:
    var pre = kind == XN_PRECOMPUTED_KIND
    var tr = _tile_rows(nq, nf)
    var kb = List[Float32](length=1 if pre else tr * nf, fill=Float32(0))
    var kcb = List[Float32](length=max(tr * nf, 1), fill=Float32(0))
    var rsb = List[Float32](length=tr, fill=Float32(0))
    var prb = List[Float32](length=tr, fill=Float32(0))
    var kp = FP(unsafe_from_address=Int(kb.unsafe_ptr()))
    var kcp = FP(unsafe_from_address=Int(kcb.unsafe_ptr()))
    var rsp = FP(unsafe_from_address=Int(rsb.unsafe_ptr()))
    var prp = FP(unsafe_from_address=Int(prb.unsafe_ptr()))
    var colp = FP(unsafe_from_address=fit_cols)
    var allp = FP(unsafe_from_address=fit_all)
    var r0 = 0
    while r0 < nq:
        var rows = min(tr, nq - r0)
        var K = FP(unsafe_from_address=q) + r0 * d
        if not pre:
            _kernel_rows(K, FP(unsafe_from_address=fitx), kp, rows, nf, d, kind, degree, gamma, coef0)
            K = kp
        var Kc = K

        def rsum(t: Int) {imm Kc, imm rsp, imm rows, imm nf}:
            rowsum_item(t, Kc, rsp, rows, nf)
        _cells(rsum, rows)
        for t in range(rows):
            scale_div_item(t, rsp, prp, rows, s)

        def cen(t: Int) {imm Kc, imm colp, imm prp, imm allp, imm kcp, imm rows, imm nf}:
            kpca_center_item(t, Kc, colp, prp, allp, kcp, rows, nf)
        _cells(cen, rows * nf)
        _matmul_rows(kcp, FP(unsafe_from_address=alphas), FP(unsafe_from_address=res) + r0 * c, rows, nf, c)
        r0 += rows
    _hsab(res, nq * c)
    _ = kb^
    _ = kcb^
    _ = rsb^
    _ = prb^


def op_kernel_matmul(
    q: Int, y: Int, w: Int, res: Int,
    n: Int, m: Int, d: Int, c: Int, kind: Int, degree: Int, gamma: Float32, coef0: Float32,
) raises:
    var tr = _tile_rows(n, m)
    var kb = List[Float32](length=max(tr * m, 1), fill=Float32(0))
    var kp = FP(unsafe_from_address=Int(kb.unsafe_ptr()))
    var r0 = 0
    while r0 < n:
        var rows = min(tr, n - r0)
        _kernel_rows(FP(unsafe_from_address=q) + r0 * d, FP(unsafe_from_address=y), kp, rows, m, d, kind, degree,
                     gamma, coef0)
        _matmul_rows(kp, FP(unsafe_from_address=w), FP(unsafe_from_address=res) + r0 * c, rows, m, c)
        r0 += rows
    _hsab(res, n * c)
    _ = kb^


def _scaled_rbf_rows(q: FP, z: FP, kp: FP, ksp: FP, rows: Int, m: Int, d: Int, gamma: Float32, variance: Float32):
    _kernel_rows(q, z, kp, rows, m, d, K_RBF, 0, gamma, Float32(0))

    def f(t: Int) {imm kp, imm ksp, imm rows, imm m, imm variance}:
        unary_item(t, kp, ksp, rows * m, U_IDENTITY, variance, Float32(0))
    _cells(f, rows * m)


def op_svgp_fit_ff(
    x: Int, z: Int, y: Int, alpha: Int, cmat: Int, qmu: Int, qsqrt: Int, info: Int, n: Int, m: Int, d: Int,
    gamma: Float32, variance: Float32, noise: Float32, jitter: Float32, kdiag: Float32,
) raises:
    """The device op's chain on the host: Kuu, the float-float statistics
    over the same tiles, cells and order, then the same solve items."""
    var mm = m * m
    var tr = max(_tile_rows(n, m), m)
    var kb = List[Float32](length=max(tr * m, 1), fill=Float32(0))
    var ksb = List[Float32](length=max(tr * m, 1), fill=Float32(0))
    var kuub = List[Float32](length=max(mm, 1), fill=Float32(0))
    var bhb = List[Float32](length=max(mm, 1), fill=Float32(0))
    var blb = List[Float32](length=max(mm, 1), fill=Float32(0))
    var vhb = List[Float32](length=max(m, 1), fill=Float32(0))
    var vlb = List[Float32](length=max(m, 1), fill=Float32(0))
    var kp = FP(unsafe_from_address=Int(kb.unsafe_ptr()))
    var ksp = FP(unsafe_from_address=Int(ksb.unsafe_ptr()))
    var kuup = FP(unsafe_from_address=Int(kuub.unsafe_ptr()))
    var bhp = FP(unsafe_from_address=Int(bhb.unsafe_ptr()))
    var blp = FP(unsafe_from_address=Int(blb.unsafe_ptr()))
    var vhp = FP(unsafe_from_address=Int(vhb.unsafe_ptr()))
    var vlp = FP(unsafe_from_address=Int(vlb.unsafe_ptr()))
    var zp = FP(unsafe_from_address=z)
    if mm > 0:
        _scaled_rbf_rows(zp, zp, kp, kuup, m, m, d, gamma, variance)
    var tile = _tile_rows(n, m)
    var r0 = 0
    while r0 < n:
        var rows = min(tile, n - r0)
        _scaled_rbf_rows(FP(unsafe_from_address=x) + r0 * d, zp, kp, ksp, rows, m, d, gamma, variance)
        var yp = FP(unsafe_from_address=y) + r0

        def bb(t: Int) {imm ksp, imm bhp, imm blp, imm rows, imm m}:
            matmul_tn_acc_ff_item(t, ksp, ksp, bhp, blp, rows, m, m)
        _cells(bb, mm)
        for t in range(m):
            matmul_tn_acc_ff_item(t, ksp, yp, vhp, vlp, rows, m, 1)
        r0 += rows
    svgp_ff_solve(kuup, bhp, blp, vhp, vlp, FP(unsafe_from_address=y),
                  FP(unsafe_from_address=alpha), FP(unsafe_from_address=cmat), FP(unsafe_from_address=qmu),
                  FP(unsafe_from_address=qsqrt), FP(unsafe_from_address=info), m, n, noise, jitter, kdiag)
    _ = kb^
    _ = ksb^
    _ = kuub^
    _ = bhb^
    _ = blb^
    _ = vhb^
    _ = vlb^


def op_svgp_predict(
    q: Int, z: Int, alpha: Int, cmat: Int, mean: Int, var_: Int,
    n: Int, m: Int, d: Int, gamma: Float32, variance: Float32, kdiag: Float32,
) raises:
    var tr = _tile_rows(n, m)
    var kb = List[Float32](length=max(tr * m, 1), fill=Float32(0))
    var ksb = List[Float32](length=max(tr * m, 1), fill=Float32(0))
    var kp = FP(unsafe_from_address=Int(kb.unsafe_ptr()))
    var ksp = FP(unsafe_from_address=Int(ksb.unsafe_ptr()))
    var cp = FP(unsafe_from_address=cmat)
    var r0 = 0
    while r0 < n:
        var rows = min(tr, n - r0)
        _scaled_rbf_rows(FP(unsafe_from_address=q) + r0 * d, FP(unsafe_from_address=z), kp, ksp, rows, m, d, gamma,
                         variance)
        _matmul_rows(ksp, FP(unsafe_from_address=alpha), FP(unsafe_from_address=mean) + r0, rows, m, 1)
        var vp = FP(unsafe_from_address=var_) + r0

        def vv(t: Int) {imm ksp, imm cp, imm vp, imm rows, imm m, imm kdiag}:
            svgp_var_item(t, ksp, cp, vp, rows, m, kdiag)
        _cells(vv, rows)
        r0 += rows
    _hsab(mean, n)
    _ = kb^
    _ = ksb^


from x_neighbors.lp_knn import op_lp_knn_graph, lp_knn_product_item, lp_knn_finite


def op_lp_knn_product(cols: Int, vals: Int, x: Int, res: Int, n: Int, m: Int, k: Int, c: Int) raises:
    var px = FP(unsafe_from_address=x)
    var finite = lp_knn_finite(px, m * c)
    for t in range(n * c):
        lp_knn_product_item(t, IP(unsafe_from_address=cols), FP(unsafe_from_address=vals),
                            px, FP(unsafe_from_address=res), n, m, k, c, finite)


from x_neighbors.items import XN_NC_IDN_CHUNKED, XN_NC_CHUNK_ROWS, _add as _nc_add, _sub as _nc_sub
from checks.numerics import (
    ftz as _nc_ftz, identical_div as _nc_div, identical_sqrt as _nc_sqrt, identical_mul_add as _nc_fma,
)


def _nc_stats_chunked(f: Int, x: FP, lab: IP, cent: FP, std: FP, dsc: FP, n: Int, d: Int, n_classes: Int):
    """Feature f's statistics in the device's chunked order (lane/fam-neighbors,
    XN_NC_IDN_CHUNKED; x_neighbors/iter_device.mojo `nc_means_part_kernel`,
    `nc_means_red_kernel`, `nc_std_part_kernel`, `nc_std_red_kernel`): every
    chain's rows ascending inside a chunk of XN_NC_CHUNK_ROWS, then the chunk
    partials ascending."""
    var nch = (n + XN_NC_CHUNK_ROWS - 1) // XN_NC_CHUNK_ROWS
    for g in range(n_classes + 1):
        var a = Float32(0)
        var m = Float32(0)
        for ch in range(nch):
            var lo = ch * XN_NC_CHUNK_ROWS
            var hi = min(n, lo + XN_NC_CHUNK_ROWS)
            var pa = Float32(0)
            var pc = 0
            for i in range(lo, hi):
                if g >= n_classes or Int(lab.unsafe_load(i)) == g:
                    pa = _nc_add(pa, x.unsafe_load(i * d + f))
                    pc += 1
            a = _nc_add(a, pa)
            m = _nc_add(m, Float32(pc))
        if g < n_classes:
            if m > 0:
                cent.unsafe_store(g * d + f, _nc_ftz(_nc_div(a, m)))
            else:
                cent.unsafe_store(g * d + f, Float32(0))
        else:
            dsc.unsafe_store(f, _nc_ftz(_nc_div(a, Float32(n))))
    var ss = Float32(0)
    for ch in range(nch):
        var lo = ch * XN_NC_CHUNK_ROWS
        var hi = min(n, lo + XN_NC_CHUNK_ROWS)
        var ps = Float32(0)
        for i in range(lo, hi):
            var df = _nc_sub(x.unsafe_load(i * d + f), cent.unsafe_load(Int(lab.unsafe_load(i)) * d + f))
            ps = _nc_ftz(_nc_fma(df, df, ps))
        ss = _nc_add(ss, ps)
    var dof = n - n_classes
    if dof > 0:
        std.unsafe_store(f, _nc_ftz(_nc_sqrt(_nc_ftz(_nc_div(ss, Float32(dof))))))
    else:
        std.unsafe_store(f, Float32(0))


def op_nc_stats(x: Int, lab: Int, nk: Int, cent: Int, std: Int, dsc: Int, n: Int, d: Int, n_classes: Int) raises:
    """nc_stats_item over the features (what the generated op ran); the
    device's chunked fold when XN_NC_IDN_CHUNKED and n > XN_NC_CHUNK_ROWS."""
    var xp = FP(unsafe_from_address=x)
    var lp = IP(unsafe_from_address=lab)
    var kp = FP(unsafe_from_address=nk)
    var cp = FP(unsafe_from_address=cent)
    var sp = FP(unsafe_from_address=std)
    var dp = FP(unsafe_from_address=dsc)
    var chunked = False
    comptime if XN_NC_IDN_CHUNKED:
        chunked = n > XN_NC_CHUNK_ROWS
    for t in range(d):
        if chunked:
            _nc_stats_chunked(t, xp, lp, cp, sp, dp, n, d, n_classes)
        else:
            nc_stats_item(t, xp, lp, kp, cp, sp, dp, n, d, n_classes)
    comptime if X_NEIGHBORS_HOST_SABOTAGE:
        if (n_classes * d) > 0:
            cp.unsafe_store(0, cp.unsafe_load(0) + Float32(1e-3))


def op_lp_iterate_knn(
    cols: Int, vals: Int, ld: Int, ystatic: Int, unlabeled: Int, info: Int,
    n: Int, k: Int, c: Int, max_iter: Int, variant: Int, tol_hi: Int, tol_lo: Int, alpha: Float32,
) raises:
    """The CPU column of the resident kNN-graph loop (lane/apple-fast-neighbors2):
    `_LabelPropagationBase.fit`'s Python loop over the same items
    (absdiff_sum, lp_knn_product, lp_clamp / ls_clamp)."""
    var tol = bitcast[DType.float64]((UInt64(tol_hi) << UInt64(32)) | UInt64(tol_lo))
    var nc = n * c
    var pc = IP(unsafe_from_address=cols)
    var pv = FP(unsafe_from_address=vals)
    var pld = FP(unsafe_from_address=ld)
    var pys = FP(unsafe_from_address=ystatic)
    var pun = IP(unsafe_from_address=unlabeled)
    var cur = List[Float32](length=nc if nc > 0 else 1, fill=Float32(0))
    var prev = List[Float32](length=nc if nc > 0 else 1, fill=Float32(0))
    var nxt = List[Float32](length=nc if nc > 0 else 1, fill=Float32(0))
    var hs = List[Float32](length=1, fill=Float32(0))
    var p_cur = FP(unsafe_from_address=Int(cur.unsafe_ptr()))
    var p_prev = FP(unsafe_from_address=Int(prev.unsafe_ptr()))
    var p_nxt = FP(unsafe_from_address=Int(nxt.unsafe_ptr()))
    var p_hs = FP(unsafe_from_address=Int(hs.unsafe_ptr()))
    for q in range(nc):
        cur[q] = pld.unsafe_load(q)
    var n_iter = 0
    var converged = False
    for it in range(max_iter):
        n_iter = it
        absdiff_sum_item(0, p_cur, p_prev, p_hs, nc)
        if Float64(hs[0]) < tol:
            converged = True
            break
        for q in range(nc):
            prev[q] = cur[q]
        var finite = lp_knn_finite(p_cur, nc)
        for t in range(nc):
            lp_knn_product_item(t, pc, pv, p_cur, p_nxt, n, n, k, c, finite)
        if variant == 0:
            for t in range(n):
                lp_clamp_item(t, p_nxt, pys, pun, p_cur, n, c)
        else:
            for t in range(nc):
                ls_clamp_item(t, p_nxt, pys, p_cur, nc, alpha)
    if not converged:
        n_iter += 1
    for q in range(nc):
        pld.unsafe_store(q, cur[q])
    var inf = IP(unsafe_from_address=info)
    inf.unsafe_store(0, Int32(n_iter))
    inf.unsafe_store(1, Int32(1 if converged else 0))
    _ = cur^
    _ = prev^
    _ = nxt^
    _ = hs^


def lp_fast_resident_binding() raises -> PythonObject:
    """The host column never takes the resident kNN-graph loop by default
    (LP_FAST_RESIDENT is FAST + Apple; the Python loop runs here)."""
    return PythonObject(0)
