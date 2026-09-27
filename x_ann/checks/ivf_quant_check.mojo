# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ann Additions' per-seam proof (DEVIATIONS 5830-5832, 5840-5842, 5850,
5855; IDENTITY_PATHS rows 183-186): IVF-SQ, IVF-RaBitQ, refine and the
sample filter on the device equal the independent host oracle
(x_ann/checks/ivf_quant_oracle.mojo) BIT FOR BIT, on fixtures first shown to
SEPARATE each seam. Under IDENTICAL:

    tools/with_identical_mode.sh pixi run mojo run -I . x_ann/checks/ivf_quant_check.mojo

Card stages: quant.sq.*, quant.rq.*, quant.refine.*, quant.filter.*."""

from std.memory import bitcast
from std.sys import exit
from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul, identical_mul_add
from x_ann.ivf_pq_device import (
    ivf_pq_search_device, ivf_rabitq_build_device, ivf_rabitq_search_device, ivf_sq_build_device,
    ivf_sq_search_device, refine_device, rq_encode_given_device, sq_encode_given_device, sq_range_device,
)
from x_ann.checks.ann_check_fixtures import fixture_ties, fixture_wide, hash_u, report, same_f32, same_i32
from x_ann.checks.ivf_pq_oracle import or_search
from x_ann.checks.ivf_quant_oracle import (
    oq_refine, oq_rq_encode, oq_rq_search, oq_sq_code, oq_sq_range, oq_sq_search,
)


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "run under tools/with_identical_mode.sh"
    var failed = 0
    var trace = IdentityTrace()
    trace.header("x_ann ivf_quant_check")

    # ---------------- SQ range (5830): column 0's minimum is 0.0 at row 3
    # and -0.0 at row 7 (first-wins keeps +0.0, last-wins -0.0); the margin
    # hides the sign there, so column 2 carries it through a zero range
    var rn = 16
    var rd = 3
    var r = List[Float32]()
    for i in range(rn):
        for c in range(rd):
            r.append(Float32(Int(hash_u(i * rd + c, 31) % UInt64(50)) + 1) * Float32(0.1))
    r[3 * rd] = Float32(0.0)
    r[7 * rd] = -Float32(0.0)
    # column 2 is constant zero with one -0.0: margin 0, so vmin keeps the sign
    for i in range(rn):
        r[i * rd + 2] = Float32(0.0)
    r[(rn - 1) * rd + 2] = -Float32(0.0)
    var ov = List[Float32]()
    var odl = List[Float32]()
    var lv = List[Float32]()
    var ldl = List[Float32]()
    oq_sq_range(r, rn, rd, ov, odl)
    oq_sq_range(r, rn, rd, lv, ldl, last_wins=True)
    var sep_range = 0 if same_f32(ov, lv) else 1
    var dv = List[Float32]()
    var dd = List[Float32]()
    sq_range_device(r, rn, rd, dv, dd)
    report("SQ range == oracle (5830)", same_f32(dv, ov) and same_f32(dd, odl), failed)
    # ---------------- SQ rounding (5831): exact halves under delta 1
    var hv = List[Float32]()
    for i in range(rn * rd):
        hv.append(Float32(i % 7) + Float32(0.5))
    var one = List[Float32](length=rd, fill=Float32(1.0))
    var zero = List[Float32](length=rd, fill=Float32(0.0))
    var sep_round = 0
    var round_ok = True
    var hc = sq_encode_given_device(hv, rn, rd, zero, one)
    for e in range(rn * rd):
        var want = oq_sq_code(hv[e], Float32(0.0), Float32(1.0))
        if want != oq_sq_code(hv[e], Float32(0.0), Float32(1.0), half_up_strict=True):
            sep_round += 1
        if Int(hc[e]) != want:
            round_ok = False
    report("SQ rounding of exact halves == oracle (5831)", round_ok, failed)

    # ---------------- end-to-end SQ build + search (5832 in the decode)
    var n = 512
    var dim = 10
    var wide = fixture_wide(n, dim)
    var ties = fixture_ties(n, dim)
    var m = 32
    var q = List[Float32]()
    for e in range(m * dim):
        q.append(wide[(e * 7 + 3) % (n * dim)])
    var c = List[Float32]()
    var off = List[Int32]()
    var li = List[Int32]()
    var vmin = List[Float32]()
    var delta = List[Float32]()
    var codes = List[Int32]()
    ivf_sq_build_device(wide, n, dim, 6, 4, 1, c, off, li, vmin, delta, codes)
    trace.record_list_i32("quant.sq.codes", codes)
    var mask = List[Int32](length=n, fill=Int32(1))
    var sd = List[Float32]()
    var si = List[Int32]()
    var sn = List[Int32]()
    ivf_sq_search_device(c, off, li, vmin, delta, codes, mask, 6, dim, q, m, 5, 2, sd, si, sn)
    var od = List[Float32]()
    var oi = List[Int32]()
    var on = List[Int32]()
    oq_sq_search(c, off, li, vmin, delta, codes, mask, 6, dim, q, m, 5, 2, od, oi, on)
    trace.record_list_f32("quant.sq.search.dist", sd)
    report("SQ search == oracle (5832)", same_f32(sd, od) and same_i32(si, oi) and same_i32(sn, on), failed)
    var sep_decode = 0
    for e in range(n * dim):
        var cc = e % dim
        var fused = ftz(identical_mul_add(Float32(Int(codes[e])), delta[cc], vmin[cc]))
        var split = ftz(ftz(identical_mul(Float32(Int(codes[e])), delta[cc])) + vmin[cc])
        if bitcast[DType.uint32](fused) != bitcast[DType.uint32](split):
            sep_decode += 1

    # ---------------- RaBitQ: a planted zero residual (row 0 IS centre 0), 5840-5842
    var rq_c = List[Float32]()
    for t in range(6 * dim):
        rq_c.append(wide[t + 5 * dim])
    for t in range(dim):
        rq_c[t] = wide[t]
    var rq_l = List[Int32]()
    for i in range(n):
        rq_l.append(Int32(0 if i == 0 else (i % 6)))
    var dco = List[Int32]()
    var dno = List[Float32]()
    var dip = List[Float32]()
    rq_encode_given_device(wide, n, dim, rq_c, rq_l, 3, dco, dno, dip)
    var oco = List[Int32]()
    var ono = List[Float32]()
    var oip = List[Float32]()
    oq_rq_encode(wide, n, dim, rq_c, rq_l, 3, oco, ono, oip)
    report("RaBitQ codes/norms/factors == oracle (5840, 5841)", same_i32(dco, oco) and same_f32(dno, ono) and same_f32(dip, oip), failed)
    var sep_zero = 1 if ono[0] == Float32(0.0) else 0
    var rc = List[Float32]()
    var roff = List[Int32]()
    var rli = List[Int32]()
    var rcodes = List[Int32]()
    var rnorm = List[Float32]()
    var rips = List[Float32]()
    ivf_rabitq_build_device(wide, n, dim, 6, 4, 3, rc, roff, rli, rcodes, rnorm, rips)
    trace.record_list_i32("quant.rq.codes", rcodes)
    var rd2 = List[Float32]()
    var ri2 = List[Int32]()
    var rn2 = List[Int32]()
    ivf_rabitq_search_device(rc, roff, rli, rcodes, rnorm, rips, mask, 6, dim, 3, q, m, 5, 2, rd2, ri2, rn2)
    var ord2 = List[Float32]()
    var ori2 = List[Int32]()
    var orn2 = List[Int32]()
    oq_rq_search(rc, roff, rli, rcodes, rnorm, rips, mask, 6, dim, 3, q, m, 5, 2, ord2, ori2, orn2)
    trace.record_list_f32("quant.rq.search.dist", rd2)
    report("RaBitQ search == oracle (5842)", same_f32(rd2, ord2) and same_i32(ri2, ori2) and same_i32(rn2, orn2), failed)
    # the planted zero residual searched: centre 0 alone, row 0 its only member
    var zc = List[Float32]()
    for t in range(dim):
        zc.append(wide[t])
    var zoff: List[Int32] = [0, 1]
    var zli: List[Int32] = [0]
    var zm = List[Int32](length=1, fill=Int32(1))
    var zcodes = List[Int32]()
    var znorm = List[Float32]()
    var zips = List[Float32]()
    oq_rq_encode(wide, 1, dim, zc, List[Int32](length=1, fill=Int32(0)), 3, zcodes, znorm, zips)
    var zd = List[Float32]()
    var zi = List[Int32]()
    var zn = List[Int32]()
    ivf_rabitq_search_device(zc, zoff, zli, zcodes, znorm, zips, zm, 1, dim, 3, q, 4, 1, 1, zd, zi, zn)
    var ozd = List[Float32]()
    var ozi = List[Int32]()
    var ozn = List[Int32]()
    oq_rq_search(zc, zoff, zli, zcodes, znorm, zips, zm, 1, dim, 3, q, 4, 1, 1, ozd, ozi, ozn)
    report("RaBitQ zero-residual estimate == oracle (5841)", same_f32(zd, ozd), failed)

    # ---------------- refine (5850): padding and repeats in the candidate rows
    var k0 = 12
    var cand = List[Int32]()
    for qi in range(m):
        for t in range(k0):
            cand.append(Int32(Int(hash_u(qi * k0 + t, 41) % UInt64(n))))
        cand[qi * k0 + 4] = Int32(-1)
        cand[qi * k0 + 9] = cand[qi * k0 + 2]
    var fd = List[Float32]()
    var fi = List[Int32]()
    refine_device(ties, n, dim, q, m, cand, k0, 8, fd, fi)
    var ofd = List[Float32]()
    var ofi = List[Int32]()
    oq_refine(ties, n, dim, q, m, cand, k0, 8, ofd, ofi)
    var rfd = List[Float32]()
    var rfi = List[Int32]()
    oq_refine(ties, n, dim, q, m, cand, k0, 8, rfd, rfi, keep_repeats=True)
    var sep_refine = 0 if same_i32(ofi, rfi) else 1
    trace.record_list_i32("quant.refine.idx", fi)
    report("refine == oracle (5850)", same_f32(fd, ofd) and same_i32(fi, ofi), failed)

    # ---------------- the sample filter (5855) on IVF-PQ and IVF-SQ
    var keep = List[Int32]()
    for i in range(n):
        keep.append(Int32(0 if i % 3 == 0 else 1))
    var fsd = List[Float32]()
    var fsi = List[Int32]()
    var fsn = List[Int32]()
    ivf_sq_search_device(c, off, li, vmin, delta, codes, keep, 6, dim, q, m, 5, 2, fsd, fsi, fsn)
    var ofsd = List[Float32]()
    var ofsi = List[Int32]()
    var ofsn = List[Int32]()
    oq_sq_search(c, off, li, vmin, delta, codes, keep, 6, dim, q, m, 5, 2, ofsd, ofsi, ofsn)
    var sep_filter = 0 if same_i32(ofsi, oi) else 1
    trace.record_list_i32("quant.filter.sq.idx", fsi)
    report("filtered SQ search == oracle (5855)", same_f32(fsd, ofsd) and same_i32(fsi, ofsi) and same_i32(fsn, ofsn), failed)
    var pd = List[Float32]()
    var pi = List[Int32]()
    var pn = List[Int32]()
    var pcb = List[Float32]()
    for e in range(2 * 8 * 5):
        pcb.append(Float32(Int(hash_u(e, 43) % UInt64(9))) * Float32(0.25) - Float32(1.0))
    var pcodes = List[Int32]()
    for e in range(n * 2):
        pcodes.append(Int32(Int(hash_u(e, 47) % UInt64(8))))
    ivf_pq_search_device(c, off, li, pcb, pcodes, keep, 6, dim, 2, 3, q, m, 5, 2, pd, pi, pn)
    var opd = List[Float32]()
    var opi = List[Int32]()
    var opn = List[Int32]()
    or_search(c, off, li, pcb, pcodes, keep, 6, dim, 2, 3, q, m, 5, 2, opd, opi, opn)
    report("filtered PQ search == oracle (5855)", same_f32(pd, opd) and same_i32(pi, opi) and same_i32(pn, opn), failed)

    print("separation: sq-range", sep_range, "sq-round", sep_round, "sq-decode", sep_decode, "rq-zero", sep_zero,
          "refine", sep_refine, "filter", sep_filter)
    if sep_range == 0 or sep_round == 0 or sep_decode == 0 or sep_zero == 0 or sep_refine == 0 or sep_filter == 0:
        print("VACUOUS: a fixture does not separate its seam")
        exit(2)
    if failed > 0:
        print("ivf_quant_check: FAILED", failed)
        exit(1)
    print("ivf_quant_check: ALL OK")
