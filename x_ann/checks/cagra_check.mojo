# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CAGRA's per-seam proof (DEVIATIONS 5820-5824, IDENTITY_PATHS row 182):
the device build's graph and the device search equal the independent host
oracle (x_ann/checks/cagra_oracle.mojo, the k-NN from tsne_oracle) BIT FOR
BIT, on fixtures first shown to SEPARATE each seam. Under IDENTICAL:

    tools/with_identical_mode.sh pixi run mojo run -I . x_ann/checks/cagra_check.mojo

Card stages: cagra.<fixture>.graph, cagra.<fixture>.search.{dist,idx}."""

from std.sys import exit
from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_ann.cagra_device import cagra_build_device, cagra_search_device
from x_ann.checks.cagra_oracle import co_prune, co_reverse_merge, co_search
from x_ann.checks.tsne_oracle import to_knn
from x_ann.checks.fixtures import fixture_ties, fixture_wide, report, same_f32, same_i32


def to_i32(v: List[Int]) -> List[Int32]:
    var o = List[Int32](capacity=len(v))
    for e in range(len(v)):
        o.append(Int32(v[e]))
    return o^


def same_int(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for e in range(len(a)):
        if a[e] != b[e]:
            return False
    return True


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "run under tools/with_identical_mode.sh"
    var failed = 0
    var trace = IdentityTrace()
    trace.header("x_ann cagra_check")
    var n = 300
    var d = 6
    var kdeg = 16
    var deg = 8
    var m = 24
    var k = 5
    var L = 12
    var width = 2
    var iters = 6
    var n_seeds = 6
    var sep_prune = 0
    var sep_rev = 0
    var sep_topk = 0
    var sep_parent = 0
    var sep_seed = 0
    for f in range(2):
        var name = String("ties") if f == 0 else String("wide")
        var x = fixture_ties(n, d) if f == 0 else fixture_wide(n, d)
        var nd = List[Float32]()
        var ni = List[Int]()
        to_knn(x, n, d, kdeg, nd, ni)
        var pr = co_prune(n, kdeg, ni, deg)
        if not same_int(pr, co_prune(n, kdeg, ni, deg, high=True)):
            sep_prune += 1
        var g = co_reverse_merge(n, deg, pr)
        if not same_int(g, co_reverse_merge(n, deg, pr, high=True)):
            sep_rev += 1
        var q = List[Float32]()
        for e in range(m * d):
            q.append(x[(e * 13 + 5) % (n * d)])
        var od = List[Float32]()
        var oi = List[Int32]()
        co_search(x, n, d, g, deg, q, m, k, L, width, iters, n_seeds, od, oi)
        for qi in range(m):
            for s in range(k - 1):
                if od[qi * k + s] == od[qi * k + s + 1] and oi[qi * k + s] != oi[qi * k + s + 1]:
                    sep_topk += 1
        var pd = List[Float32]()
        var pi = List[Int32]()
        co_search(x, n, d, g, deg, q, m, k, L, width, iters, n_seeds, pd, pi, parents_from_back=True)
        if not same_i32(oi, pi):
            sep_parent += 1
        co_search(x, n, d, g, deg, q, m, k, L, width, iters, n_seeds, pd, pi, seed_offset=1)
        if not same_i32(oi, pi):
            sep_seed += 1
        # ---- device == oracle
        var dg = cagra_build_device(x, n, d, kdeg, deg)
        trace.record_list_i32(String("cagra.") + name + ".graph", dg)
        report(name + ": graph == oracle (5810, 5820, 5821)", same_i32(dg, to_i32(g)), failed)
        var dd = List[Float32]()
        var di = List[Int32]()
        cagra_search_device(x, n, d, to_i32(g), deg, q, m, k, L, width, iters, n_seeds, dd, di)
        trace.record_list_f32(String("cagra.") + name + ".search.dist", dd)
        trace.record_list_i32(String("cagra.") + name + ".search.idx", di)
        report(name + ": search == oracle (5822-5824)", same_f32(dd, od) and same_i32(di, oi), failed)
    print("separation: prune", sep_prune, "reverse", sep_rev, "topk-tie", sep_topk, "parents", sep_parent,
          "seeds", sep_seed)
    if sep_prune == 0 or sep_rev == 0 or sep_topk == 0 or sep_parent == 0 or sep_seed == 0:
        print("VACUOUS: a fixture does not separate its seam")
        exit(2)
    if failed > 0:
        print("cagra_check: FAILED", failed)
        exit(1)
    print("cagra_check: ALL OK")
