# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""t-SNE's per-seam proof (DEVIATIONS 5810-5817, IDENTITY_PATHS row 181):
the device fit's embedding and KL equal the independent host oracle
(x_ann/checks/tsne_oracle.mojo) BIT FOR BIT on fixtures first shown to
SEPARATE each seam's pinned spelling from its unpinned one. Under IDENTICAL:

    tools/with_identical_mode.sh pixi run mojo run -I . x_ann/checks/tsne_check.mojo

With MOJOLEARN_IDENTITY_TRACE set, the card holds tsne.<fixture>.embedding
and tsne.<fixture>.kl."""

from std.memory import bitcast
from std.sys import exit
from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_exp, identical_mul
from x_ann.tsne_device import tsne_fit_device
from x_ann.checks.tsne_oracle import to_fit, to_knn, to_q, to_sqdist, to_sum, to_symmetrize, to_perplexity
from x_ann.checks.ann_check_fixtures import fixture_ties, fixture_wide, hash_u, report, same_f32


def y_start(n: Int) -> List[Float32]:
    var y = List[Float32](capacity=2 * n)
    for e in range(2 * n):
        y.append(Float32(Int(hash_u(e, 21) % UInt64(2001)) - 1000) * Float32(1e-7))
    return y^


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "run under tools/with_identical_mode.sh"
    var failed = 0
    var trace = IdentityTrace()
    trace.header("x_ann tsne_check")
    var n = 60
    var d = 5
    var perp = Float32(5.0)
    var nn = 16
    var ties = fixture_ties(n, d)
    var wide = fixture_wide(n, d)
    var y0 = y_start(n)

    # ---- separation
    var sep_knn = 0
    for i in range(n):
        var dist = List[Float32]()
        for j in range(n):
            if j != i:
                dist.append(to_sqdist(ties, i, j, d))
        # a tie at the membership boundary: the nn-th smallest equals the (nn+1)-th
        for a in range(len(dist)):
            var below = 0
            var equal = 0
            for b in range(len(dist)):
                if dist[b] < dist[a]:
                    below += 1
                elif dist[b] == dist[a]:
                    equal += 1
            if below < nn and below + equal > nn:
                sep_knn += 1
                break
    var sep_perp = 0
    var nd = List[Float32]()
    var ni = List[Int]()
    to_knn(wide, n, d, nn, nd, ni)
    for i in range(n):
        var ex = List[Float32]()
        for j in range(nn):
            ex.append(ftz(identical_exp(ftz(-identical_mul(nd[i * nn + j], Float32(0.01))))))
        if bitcast[DType.uint32](to_sum(ex, 0, nn)) != bitcast[DType.uint32](to_sum(ex, 0, nn, rev=True)):
            sep_perp += 1
    var p = to_perplexity(nd, n, nn, Float32(1.609438))
    var ip1 = List[Int]()
    var ix1 = List[Int]()
    var v1 = List[Float32]()
    var ip2 = List[Int]()
    var ix2 = List[Int]()
    var v2 = List[Float32]()
    to_symmetrize(ni, p, n, nn, ip1, ix1, v1)
    to_symmetrize(ni, p, n, nn, ip2, ix2, v2, rev_total=True)
    var sep_norm = 0 if same_f32(v1, v2) else 1
    var yw = List[Float32]()
    for e in range(2 * n):
        yw.append(wide[e % (n * d)])
    # the schedule's own seams are shown by the configurations below: the
    # nc = 3 run has dof 2 (q * sqrt(q), 5816), the 120-step run crosses the
    # phase change with a reset (5817), the large min_grad_norm run stops at
    # a check before its last step (5817)
    var sep_rep = 0
    var sep_attr = 0
    for i in range(n):
        var qs = List[Float32]()
        for j in range(n):
            if j != i:
                qs.append(to_q(yw, i, j, 2, 1))
        if bitcast[DType.uint32](to_sum(qs, 0, len(qs))) != bitcast[DType.uint32](to_sum(qs, 0, len(qs), rev=True)):
            sep_rep += 1
        var at = List[Float32]()
        for s in range(ip1[i], ip1[i + 1]):
            at.append(ftz(identical_mul(v1[s], ftz(ftz(yw[2 * i]) - ftz(yw[2 * ix1[s]])))))
        if bitcast[DType.uint32](to_sum(at, 0, len(at))) != bitcast[DType.uint32](to_sum(at, 0, len(at), rev=True)):
            sep_attr += 1
    # 5815: at the first step update == 0, so update * grad == 0 for every
    # coordinate: `< 0` and `<= 0` take opposite branches on all 2n of them
    var sep_gain = 2 * n
    print("separation: knn-boundary-tie", sep_knn, "perplexity-sum", sep_perp, "P-total", sep_norm,
          "repulsion", sep_rep, "attraction", sep_attr, "gain-branch", sep_gain)
    if sep_knn == 0 or sep_perp == 0 or sep_norm == 0 or sep_rep == 0 or sep_attr == 0:
        print("VACUOUS: a fixture does not separate its seam")
        exit(2)

    # ---- device == oracle, four configurations
    for f in range(4):
        var name = String("ties") if f == 0 else (String("wide") if f == 1 else (String("nc3-exact") if f == 2 else String("early-stop")))
        var x = ties.copy() if f == 0 else wide.copy()
        var nc = 3 if f == 2 else 2
        var iters = 120 if f >= 2 else 30
        var explo = 60 if f >= 2 else 10
        var exact = f == 2
        var mgn = Float32(1000.0) if f == 3 else Float32(1e-7)
        var y0f = List[Float32]()
        for e in range(nc * n):
            y0f.append(Float32(Int(hash_u(e, 21) % UInt64(2001)) - 1000) * Float32(1e-7))
        var yd = List[Float32]()
        var kd = Float32(0.0)
        var nd_it = 0
        tsne_fit_device(x, n, d, nc, y0f, perp, Float32(12.0), Float32(50.0), iters, explo, exact, 300, mgn, yd, kd, nd_it)
        var ko = Float32(0.0)
        var no_it = 0
        var yo = to_fit(x, n, d, nc, y0f, perp, Float32(12.0), Float32(50.0), iters, explo, exact, 300, mgn, ko, no_it)
        trace.record_list_f32(String("tsne.") + name + ".embedding", yd)
        trace.record_scalar_f32(String("tsne.") + name + ".kl", kd)
        report(name + ": embedding == oracle (5810-5817)", same_f32(yd, yo), failed)
        report(name + ": KL == oracle", bitcast[DType.uint32](kd) == bitcast[DType.uint32](ko), failed)
        report(name + ": n_iter == oracle (" + String(nd_it) + ")", nd_it == no_it, failed)
        if f == 3 and nd_it >= iters - 1:
            print("VACUOUS: the early-stop configuration did not stop early")
            exit(2)
    if failed > 0:
        print("tsne_check: FAILED", failed)
        exit(1)
    print("tsne_check: ALL OK")
