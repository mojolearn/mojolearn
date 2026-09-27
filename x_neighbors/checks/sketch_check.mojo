# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5203 (PolynomialCountSketch's direct circular convolution)
and 5213 (the chi-squared samplers' transcendental compositions and the skewed
transform's fold) of x_neighbors/items.mojo.

    tools/with_identical_mode.sh pixi run mojo run -I . x_neighbors/checks/sketch_check.mojo

Pattern as dist_check.mojo."""
from core.identity_trace import IdentityTrace
from x_neighbors.checks.oracles import o_pcs, o_achi2, o_skew_weights, o_skew_transform
from x_neighbors.checks.seam_util import seam_fixture, positive_fixture, fa, ia, zf, zi, count_diff_f32, require_separates, same
from x_neighbors.device_ops import op_pcs, op_achi2, op_skew_weights, op_skew_transform
from x_neighbors.host_ops import (
    op_pcs as h_pcs, op_achi2 as h_achi2, op_skew_weights as h_skew_weights, op_skew_transform as h_skew_transform,
)


def main() raises:
    var tr = IdentityTrace()
    var n = 20
    var d = 6
    var nf = d + 1
    var nc = 16
    var deg = 3
    var x = seam_fixture(n, d, 21)
    var hidx = zi(deg * nf)
    var hbit = zi(deg * nf)
    for i in range(deg * nf):
        hidx[i] = Int32((i * 5 + 3) % nc)
        hbit[i] = Int32(1) if (i * 7) % 3 == 0 else Int32(-1)
    var g = Float32(0.7)
    var c0 = Float32(1.5)
    var w = o_pcs(x, hidx, hbit, n, d, nf, nc, deg, g, c0)
    require_separates("5203 convolution shift order", count_diff_f32(w, o_pcs(x, hidx, hbit, n, d, nf, nc, deg, g, c0, 1)))
    var dv = zf(n * nc)
    op_pcs(fa(x), ia(hidx), ia(hbit), fa(dv), n, d, nf, nc, deg, g, c0)
    same("5203 pcs device", count_diff_f32(dv, w))
    var hv = zf(n * nc)
    h_pcs(fa(x), ia(hidx), ia(hbit), fa(hv), n, d, nf, nc, deg, g, c0)
    same("5203 pcs host", count_diff_f32(hv, w))
    tr.record_list_f32("x_neighbors.pcs", dv)

    var xp = positive_fixture(n, d, 22)
    var steps = 3
    var wa = o_achi2(xp, n, d, steps, Float32(0.4))
    require_separates("5213 sech factor spelling", count_diff_f32(wa, o_achi2(xp, n, d, steps, Float32(0.4), 1)))
    var da = zf(n * d * (2 * steps - 1))
    op_achi2(fa(xp), fa(da), n, d, steps, Float32(0.4))
    same("5213 achi2 device", count_diff_f32(da, wa))
    var ha = zf(n * d * (2 * steps - 1))
    h_achi2(fa(xp), fa(ha), n, d, steps, Float32(0.4))
    same("5213 achi2 host", count_diff_f32(ha, wa))
    tr.record_list_f32("x_neighbors.achi2", da)

    var z = zf(d * nc)
    for i in range(d * nc):
        z[i] = Float32(0.02) + Float32(1.5) * Float32(i) / Float32(d * nc)
    var wz = o_skew_weights(z)
    require_separates("5213 inverse sech spelling", count_diff_f32(wz, o_skew_weights(z, 1)))
    var dz = zf(d * nc)
    op_skew_weights(fa(z), fa(dz), d * nc)
    same("5213 skew_weights device", count_diff_f32(dz, wz))
    var hz = zf(d * nc)
    h_skew_weights(fa(z), fa(hz), d * nc)
    same("5213 skew_weights host", count_diff_f32(hz, wz))
    var off = zf(nc)
    for c in range(nc):
        off[c] = Float32(c) * Float32(0.37)
    var wt = o_skew_transform(x, wz, off, n, d, nc)
    require_separates("5213 skewed projection fold", count_diff_f32(wt, o_skew_transform(x, wz, off, n, d, nc, 1)))
    var dt = zf(n * nc)
    op_skew_transform(fa(x), fa(wz), fa(off), fa(dt), n, d, nc)
    same("5213 skew_transform device", count_diff_f32(dt, wt))
    var ht = zf(n * nc)
    h_skew_transform(fa(x), fa(wz), fa(off), fa(ht), n, d, nc)
    same("5213 skew_transform host", count_diff_f32(ht, wt))
    tr.record_list_f32("x_neighbors.skew", dt)
    _ = x^
    _ = hidx^
    _ = hbit^
    _ = xp^
    _ = z^
    _ = wz^
    _ = off^
    print("PASS x_neighbors sketch_check")
