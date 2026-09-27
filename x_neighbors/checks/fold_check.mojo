# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5209 (the dense folds: matmul, row / column sums, group
means, the variance, row normalization, softmax) and 5210 (LocalOutlierFactor's
reach-distance mean and ratio mean) of x_neighbors/items.mojo.

    tools/with_identical_mode.sh pixi run mojo run -I . x_neighbors/checks/fold_check.mojo

Pattern as dist_check.mojo: separation first, then device == oracle and
host == oracle bit for bit under IDENTICAL, each stage on the card."""
from core.identity_trace import IdentityTrace
from x_neighbors.checks.oracles import (
    o_matmul, o_rowsum, o_colsum, o_group_mean, o_variance, o_row_normalize, o_softmax, o_log_softmax, o_sqdist, o_knn_select,
    o_lof_lrd, o_lof_score,
)
from x_neighbors.checks.seam_util import (
    seam_fixture, fa, ia, zf, zi, count_diff_f32, count_diff_i32, require_separates, same,
)
from x_neighbors.device_ops import (
    op_matmul, op_rowsum, op_colsum, op_group_mean, op_variance, op_row_normalize, op_softmax, op_log_softmax, op_lof_lrd,
    op_lof_score,
)
from x_neighbors.host_ops import (
    op_matmul as h_matmul, op_rowsum as h_rowsum, op_colsum as h_colsum, op_group_mean as h_group_mean,
    op_variance as h_variance, op_row_normalize as h_row_normalize, op_softmax as h_softmax, op_log_softmax as h_log_softmax,
    op_lof_lrd as h_lof_lrd, op_lof_score as h_lof_score,
)


def main() raises:
    var tr = IdentityTrace()
    var n = 33
    var k = 11
    var m = 6
    var a = seam_fixture(n, k, 7)
    var b = seam_fixture(k, m, 8)

    var w = o_matmul(a, b, n, k, m)
    require_separates("5209 matmul fold", count_diff_f32(w, o_matmul(a, b, n, k, m, 1)))
    var dv = zf(n * m)
    op_matmul(fa(a), fa(b), fa(dv), n, k, m)
    same("5209 matmul device", count_diff_f32(dv, w))
    var hv = zf(n * m)
    h_matmul(fa(a), fa(b), fa(hv), n, k, m)
    same("5209 matmul host", count_diff_f32(hv, w))
    tr.record_list_f32("x_neighbors.matmul", dv)

    var wr = o_rowsum(a, n, k)
    require_separates("5209 rowsum fold", count_diff_f32(wr, o_rowsum(a, n, k, 1)))
    var dr = zf(n)
    op_rowsum(fa(a), fa(dr), n, k)
    same("5209 rowsum device", count_diff_f32(dr, wr))
    var hr = zf(n)
    h_rowsum(fa(a), fa(hr), n, k)
    same("5209 rowsum host", count_diff_f32(hr, wr))

    var wc = o_colsum(a, n, k)
    require_separates("5209 colsum fold", count_diff_f32(wc, o_colsum(a, n, k, 1)))
    var dc = zf(k)
    op_colsum(fa(a), fa(dc), n, k)
    same("5209 colsum device", count_diff_f32(dc, wc))
    var hc = zf(k)
    h_colsum(fa(a), fa(hc), n, k)
    same("5209 colsum host", count_diff_f32(hc, wc))
    tr.record_list_f32("x_neighbors.sums", dr)

    var lab = zi(n)
    for i in range(n):
        lab[i] = Int32((i * 7) % 3)
    var wg = o_group_mean(a, lab, n, k, 4)          # group 3 is empty: its mean is 0
    require_separates("5209 group mean fold", count_diff_f32(wg, o_group_mean(a, lab, n, k, 4, 1)))
    var dg = zf(4 * k)
    op_group_mean(fa(a), ia(lab), fa(dg), n, k, 4)
    same("5209 group_mean device", count_diff_f32(dg, wg))
    var hg = zf(4 * k)
    h_group_mean(fa(a), ia(lab), fa(hg), n, k, 4)
    same("5209 group_mean host", count_diff_f32(hg, wg))
    tr.record_list_f32("x_neighbors.group_mean", dg)

    # one large deviation and many small ones: ascending, each small square rounds into the
    # large partial sum; descending, the small ones accumulate exactly first
    var vx = List[Float32]()
    vx.append(Float32(10000))
    for _ in range(300):
        vx.append(Float32(0.5))
    var wvar = List[Float32]()
    wvar.append(o_variance(vx))
    var alt = List[Float32]()
    alt.append(o_variance(vx, 1))
    require_separates("5209 variance fold", count_diff_f32(wvar, alt))
    var dvar = zf(1)
    op_variance(fa(vx), fa(dvar), len(vx))
    same("5209 variance device", count_diff_f32(dvar, wvar))
    var hvar = zf(1)
    h_variance(fa(vx), fa(hvar), len(vx))
    _ = vx^
    same("5209 variance host", count_diff_f32(hvar, wvar))

    var pos = a.copy()
    for i in range(len(pos)):
        pos[i] = abs(pos[i])
    for j in range(k):
        pos[3 * k + j] = Float32(0)                     # a zero row divides by 1
    var wn = o_row_normalize(pos, n, k)
    require_separates("5209 row normalize fold", count_diff_f32(wn, o_row_normalize(pos, n, k, 1)))
    var dn = zf(n * k)
    op_row_normalize(fa(pos), fa(dn), n, k)
    same("5209 row_normalize device", count_diff_f32(dn, wn))
    var hn = zf(n * k)
    h_row_normalize(fa(pos), fa(hn), n, k)
    same("5209 row_normalize host", count_diff_f32(hn, wn))

    var sm = a.copy()
    for i in range(len(sm)):
        sm[i] = sm[i] * Float32(0.01)
    var ws = o_softmax(sm, n, k)
    require_separates("5209 softmax sum fold", count_diff_f32(ws, o_softmax(sm, n, k, 1)))
    var ds = zf(n * k)
    op_softmax(fa(sm), fa(ds), n, k)
    same("5209 softmax device", count_diff_f32(ds, ws))
    var hs = zf(n * k)
    h_softmax(fa(sm), fa(hs), n, k)
    same("5209 softmax host", count_diff_f32(hs, ws))
    tr.record_list_f32("x_neighbors.softmax", ds)
    var lsm = a.copy()
    for i in range(len(lsm)):
        lsm[i] = lsm[i] * Float32(0.37)
    var wls = o_log_softmax(lsm, n, k)
    require_separates("5209 log_softmax sum fold", count_diff_f32(wls, o_log_softmax(lsm, n, k, 1)))
    var dls = zf(n * k)
    op_log_softmax(fa(lsm), fa(dls), n, k)
    same("5209 log_softmax device", count_diff_f32(dls, wls))
    var hls = zf(n * k)
    h_log_softmax(fa(lsm), fa(hls), n, k)
    same("5209 log_softmax host", count_diff_f32(hls, wls))
    tr.record_list_f32("x_neighbors.log_softmax", dls)
    _ = lsm^

    # ---- 5210: LOF on a neighbor table from the oracle's selection
    var nn = 48
    var d = 5
    var x = seam_fixture(nn, d, 9)
    var kk = 6
    var sq = o_sqdist(x, x, nn, nn, d)
    var sel = o_knn_select(sq, nn, nn, kk, True)
    var dist = sel[0].copy()
    for i in range(len(dist)):
        dist[i] = dist[i] * Float32(0.001) + Float32(i % 7)
    var idx = sel[1].copy()
    var wl = o_lof_lrd(dist, idx, dist, nn, kk)
    require_separates("5210 lrd rank fold", count_diff_f32(wl, o_lof_lrd(dist, idx, dist, nn, kk, 1)))
    var dl = zf(nn)
    op_lof_lrd(fa(dist), ia(idx), fa(dist), fa(dl), kk, nn, nn)
    same("5210 lof_lrd device", count_diff_f32(dl, wl))
    var hl = zf(nn)
    h_lof_lrd(fa(dist), ia(idx), fa(dist), fa(hl), kk, nn, nn)
    same("5210 lof_lrd host", count_diff_f32(hl, wl))
    var wsc = o_lof_score(idx, wl, wl, nn, kk)
    require_separates("5210 score rank fold", count_diff_f32(wsc, o_lof_score(idx, wl, wl, nn, kk, 1)))
    var dsc = zf(nn)
    op_lof_score(ia(idx), fa(wl), fa(wl), fa(dsc), kk, nn, nn)
    same("5210 lof_score device", count_diff_f32(dsc, wsc))
    var hsc = zf(nn)
    h_lof_score(ia(idx), fa(wl), fa(wl), fa(hsc), kk, nn, nn)
    same("5210 lof_score host", count_diff_f32(hsc, wsc))
    tr.record_list_f32("x_neighbors.lof", dsc)
    _ = a^
    _ = b^
    _ = lab^
    _ = pos^
    _ = sm^
    _ = dist^
    _ = idx^
    _ = wl^
    print("PASS x_neighbors fold_check")
