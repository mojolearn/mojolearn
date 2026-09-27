# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5200 (the one-class SMO in float32 and its tie rule), 5201 /
5212 (NearestCentroid's std, shrink and discriminant) and 5202 (KernelPCA's
centering and svd_flip sign rule) of x_neighbors/items.mojo.

    tools/with_identical_mode.sh pixi run mojo run -I . x_neighbors/checks/model_check.mojo

Pattern as dist_check.mojo."""
from core.identity_trace import IdentityTrace
from x_neighbors.checks.oracles import (
    o_kernel, o_ocsvm, o_nc_std, o_nc_shrink, o_nc_shrink_dev, o_nc_decision, o_kpca_center, o_svd_flip, o_group_mean,
)
from x_neighbors.checks.seam_util import (
    seam_fixture, fa, ia, zf, zi, count_diff_f32, count_diff_i32, require_separates, same,
)
from x_neighbors.device_ops import op_ocsvm, op_nc_std, op_nc_shrink, op_nc_decision, op_kpca_center, op_svd_flip
from x_neighbors.host_ops import (
    op_ocsvm as h_ocsvm, op_nc_std as h_nc_std, op_nc_shrink as h_nc_shrink, op_nc_decision as h_nc_decision,
    op_kpca_center as h_kpca_center, op_svd_flip as h_svd_flip,
)


def main() raises:
    var tr = IdentityTrace()
    # ---- 5200: the SMO on an rbf kernel matrix whose rows 1 and 2 are equal (tied gradients)
    var n = 30
    var d = 4
    var x = seam_fixture(n, d, 11)
    for i in range(len(x)):
        x[i] = x[i] * Float32(0.002)
    for r in range(1, n, 2):                          # every row has a twin: tied gradients everywhere
        for f in range(d):
            x[r * d + f] = x[(r - 1) * d + f]
    var q = o_kernel(x, x, n, n, d, 2, Float32(0.5), Float32(0), 0)
    var a0 = zf(n)
    for i in range(9):
        a0[i] = Float32(1)
    a0[9] = Float32(0.5)
    var ones = zf(n)
    for i in range(n):
        ones[i] = Float32(1)
    var want = o_ocsvm(q, ones, a0, n, Float32(1e-3), 100000)
    var alt = o_ocsvm(q, ones, a0, n, Float32(1e-3), 100000, 1)
    var wa = want[0].copy()
    wa.append(want[1])
    var aa = alt[0].copy()
    aa.append(alt[1])
    require_separates("5200 SMO tie rule", count_diff_f32(wa, aa))
    var da = a0.copy()
    var di = zf(1)
    var dit = zi(1)
    op_ocsvm(fa(q), fa(ones), fa(da), fa(di), ia(dit), n, Float32(1e-3), 100000)
    da.append(di[0])
    same("5200 ocsvm device", count_diff_f32(da, wa))
    var ha = a0.copy()
    var hi = zf(1)
    var hit = zi(1)
    h_ocsvm(fa(q), fa(ones), fa(ha), fa(hi), ia(hit), n, Float32(1e-3), 100000)
    ha.append(hi[0])
    same("5200 ocsvm host", count_diff_f32(ha, wa))
    tr.record_list_f32("x_neighbors.ocsvm", da)
    # 5200 with sample_weight: per-sample upper bounds C_i (libsvm solve_one_class's W)
    var cw = zf(n)
    for i in range(n):
        cw[i] = Float32(0.5) + Float32(i % 4) * Float32(0.5)
    var aw0 = zf(n)
    var nul = Float32(0)
    for i in range(n):
        nul += cw[i] * Float32(0.3)
    for i in range(n):
        if nul <= Float32(0):
            break
        aw0[i] = cw[i] if cw[i] < nul else nul
        nul -= aw0[i]
    var wwo = o_ocsvm(q, cw, aw0, n, Float32(1e-3), 100000)
    var wwa = wwo[0].copy()
    wwa.append(wwo[1])
    var wu = o_ocsvm(q, cw, aw0, n, Float32(1e-3), 100000, 2)
    var wua = wu[0].copy()
    wua.append(wu[1])
    require_separates("5200 SMO per-sample bound", count_diff_f32(wwa, wua))
    var dw = aw0.copy()
    var dwi = zf(1)
    var dwit = zi(1)
    op_ocsvm(fa(q), fa(cw), fa(dw), fa(dwi), ia(dwit), n, Float32(1e-3), 100000)
    dw.append(dwi[0])
    same("5200 ocsvm weighted device", count_diff_f32(dw, wwa))
    var hw = aw0.copy()
    var hwi = zf(1)
    var hwit = zi(1)
    h_ocsvm(fa(q), fa(cw), fa(hw), fa(hwi), ia(hwit), n, Float32(1e-3), 100000)
    hw.append(hwi[0])
    same("5200 ocsvm weighted host", count_diff_f32(hw, wwa))
    tr.record_list_f32("x_neighbors.ocsvm_weighted", dw)
    _ = ones^
    _ = cw^

    # ---- 5212 / 5201: NearestCentroid
    var nn = 36
    var dd = 5
    var xc = seam_fixture(nn, dd, 12)
    var lab = zi(nn)
    for i in range(nn):
        lab[i] = Int32(i % 3)
    var cent = o_group_mean(xc, lab, nn, dd, 3)
    var ws = o_nc_std(xc, lab, cent, nn, dd, 3)
    require_separates("5212 within-class std fold", count_diff_f32(ws, o_nc_std(xc, lab, cent, nn, dd, 3, 1)))
    var dstd = zf(dd)
    op_nc_std(fa(xc), ia(lab), fa(cent), fa(dstd), nn, dd, 3)
    same("5212 nc_std device", count_diff_f32(dstd, ws))
    var hstd = zf(dd)
    h_nc_std(fa(xc), ia(lab), fa(cent), fa(hstd), nn, dd, 3)
    same("5212 nc_std host", count_diff_f32(hstd, ws))
    var nk = zf(3)
    for i in range(3):
        nk[i] = Float32(12)
    var med = Float32(0.25)
    var shrink = Float32(0.1)
    var wsh = o_nc_shrink(xc, cent, nk, ws, nn, dd, 3, med, shrink)
    require_separates("5212 shrink dataset-centroid fold", count_diff_f32(wsh, o_nc_shrink(xc, cent, nk, ws, nn, dd, 3, med, shrink, 1)))
    var wdv = o_nc_shrink_dev(xc, cent, nk, ws, nn, dd, 3, med, shrink)[1].copy()
    require_separates("5212 deviations_ thresholded", count_diff_f32(wdv, o_nc_shrink_dev(xc, cent, nk, ws, nn, dd, 3, med, shrink, 2)[1].copy()))
    var dsh = zf(3 * dd)
    var ddv = zf(3 * dd)
    op_nc_shrink(fa(xc), fa(cent), fa(nk), fa(ws), fa(dsh), fa(ddv), nn, dd, 3, 1, med, shrink)
    same("5212/5201 nc_shrink device", count_diff_f32(dsh, wsh))
    same("5212/5201 nc deviations_ device", count_diff_f32(ddv, wdv))
    var hsh = zf(3 * dd)
    var hdv = zf(3 * dd)
    h_nc_shrink(fa(xc), fa(cent), fa(nk), fa(ws), fa(hsh), fa(hdv), nn, dd, 3, 1, med, shrink)
    same("5212/5201 nc_shrink host", count_diff_f32(hsh, wsh))
    same("5212/5201 nc deviations_ host", count_diff_f32(hdv, wdv))
    var wns = o_nc_shrink_dev(xc, cent, nk, ws, nn, dd, 3, med, shrink, 0, False)
    var dns = zf(3 * dd)
    var dnd = zf(3 * dd)
    op_nc_shrink(fa(xc), fa(cent), fa(nk), fa(ws), fa(dns), fa(dnd), nn, dd, 3, 0, med, shrink)
    same("5212 nc no-shrink device", count_diff_f32(dns, wns[0]) + count_diff_f32(dnd, wns[1]))
    var hns = zf(3 * dd)
    var hnd = zf(3 * dd)
    h_nc_shrink(fa(xc), fa(cent), fa(nk), fa(ws), fa(hns), fa(hnd), nn, dd, 3, 0, med, shrink)
    same("5212 nc no-shrink host", count_diff_f32(hns, wns[0]) + count_diff_f32(hnd, wns[1]))
    tr.record_list_f32("x_neighbors.nc_deviations", ddv)
    var prior = zf(3)
    prior[0] = Float32(0.5)
    prior[1] = Float32(0.3)
    prior[2] = Float32(0.2)
    var wdd = o_nc_decision(xc, cent, ws, prior, nn, dd, 3)
    require_separates("5212 discriminant fold", count_diff_f32(wdd, o_nc_decision(xc, cent, ws, prior, nn, dd, 3, 1)))
    var ddec = zf(nn * 3)
    op_nc_decision(fa(xc), fa(cent), fa(ws), fa(prior), fa(ddec), nn, dd, 3)
    same("5212 nc_decision device", count_diff_f32(ddec, wdd))
    var hdec = zf(nn * 3)
    h_nc_decision(fa(xc), fa(cent), fa(ws), fa(prior), fa(hdec), nn, dd, 3)
    same("5212 nc_decision host", count_diff_f32(hdec, wdd))
    tr.record_list_f32("x_neighbors.nc", ddec)

    # ---- 5202: centering and the svd_flip sign rule (column 1 plants a |max| tie of opposite signs)
    var km = seam_fixture(8, 8, 13)
    var cols = seam_fixture(1, 8, 14)
    var rows = seam_fixture(1, 8, 15)
    var all_ = zf(1)
    all_[0] = Float32(777.7771)             # the added constant dwarfs the small columns: order shows
    var wcn = o_kpca_center(km, cols, rows, all_[0], 8, 8)
    require_separates("5202 centering order", count_diff_f32(wcn, o_kpca_center(km, cols, rows, all_[0], 8, 8, 1)))
    var dcn = zf(64)
    op_kpca_center(fa(km), fa(cols), fa(rows), fa(all_), fa(dcn), 8, 8)
    same("5202 kpca_center device", count_diff_f32(dcn, wcn))
    var hcn = zf(64)
    h_kpca_center(fa(km), fa(cols), fa(rows), fa(all_), fa(hcn), 8, 8)
    same("5202 kpca_center host", count_diff_f32(hcn, wcn))
    var v = seam_fixture(6, 4, 16)
    v[0 * 4 + 1] = Float32(0.9)
    v[3 * 4 + 1] = Float32(-0.9)
    for i in range(6):
        if i != 0 and i != 3:
            v[i * 4 + 1] = Float32(0.1)
    var wf = o_svd_flip(v, 6, 4)
    require_separates("5202 svd_flip tie row", count_diff_f32(wf, o_svd_flip(v, 6, 4, 1)))
    var dfv = v.copy()
    op_svd_flip(fa(dfv), 6, 4)
    same("5202 svd_flip device", count_diff_f32(dfv, wf))
    var hfv = v.copy()
    h_svd_flip(fa(hfv), 6, 4)
    same("5202 svd_flip host", count_diff_f32(hfv, wf))
    tr.record_list_f32("x_neighbors.kpca", dcn)
    _ = q^
    _ = xc^
    _ = lab^
    _ = cent^
    _ = nk^
    _ = ws^
    _ = prior^
    _ = km^
    _ = cols^
    _ = rows^
    _ = all_^
    print("PASS x_neighbors model_check")
