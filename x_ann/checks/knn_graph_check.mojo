# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The all-rows k-NN graph (`mojolearn.knn_graph`, lane sc-ann-1) on
hand-sized fixtures, under IDENTICAL:

  host (CPU column, `run_host_checks`):
    * the exact graph equals the one-thread cell (`ts_knn_cell`) word for word;
    * IVF with n_probes == n_lists equals the exact graph word for word;
    * IVF with fewer probes: every row's list is sorted under (distance,
      index), holds no self and no repeat, and its recall against the exact
      graph is at least 0.8 on the clustered fixture (a test quantity, not a
      tuning target: the fixture's eight well separated clusters put every
      true neighbor in the row's own or a probed list);
    * a probe pool shorter than k + 1 refuses back to the exact graph;
    * the plan's default rule (small n -> exact; n // 384 lists, 16 probes).
  device (`run_checks`, a GPU column): the device graph equals the host
    graph word for word, exact and IVF, sqeuclidean and euclidean.

    tools/with_identical_mode.sh pixi run mojo run -I . checks/native_tests/x_ann_knn_graph_check.mojo"""

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_ann.tsne_core import F32P, I32P, ts_knn_cell
from x_ann.host.ivf_pq_host import fp, ip
from x_ann.host.knn_graph_host import knn_graph_host, ivf_graph_host
from x_ann.knn_graph_device import knn_graph_device_ptr
from x_ann.knn_graph_plan import (
    KnnGraphPlan, knn_graph_plan, KG_METHOD_EXACT, KG_METHOD_IVF, KG_AUTO,
)
from x_ann.checks.ann_check_fixtures import fixture_ties, fixture_wide, report, same_f32, same_i32


def kg_fixture_clusters(n: Int, d: Int) -> List[Float32]:
    """Eight clusters 100 apart on the first feature, a small deterministic
    spread inside each (row i in cluster 8 i / n: contiguous blocks, so the
    stride seeds put one seed in each cluster)."""
    var x = List[Float32](capacity=n * d)
    for i in range(n):
        var c = (i * 8) // n
        for f in range(d):
            var h = ((i * 2654435761 + f * 40503) >> 7) % 1000
            var v = Float32(h) / Float32(1000.0)
            if f == 0:
                v += Float32(100 * c)
            x.append(v)
    return x^


def _cell_graph(mut x: List[Float32], n: Int, d: Int, k: Int, mut od: List[Float32], mut oi: List[Int32]):
    for i in range(n):
        ts_knn_cell(i, fp(x), n, d, k, fp(od), ip(oi))


def _rows_valid(oi: List[Int32], od: List[Float32], n: Int, k: Int) -> Bool:
    for i in range(n):
        for s in range(k):
            var j = Int(oi[i * k + s])
            if j < 0 or j >= n or j == i:
                return False
            for t in range(s):
                if Int(oi[i * k + t]) == j:
                    return False
            if s > 0:
                var pd = od[i * k + s - 1]
                var pj = Int(oi[i * k + s - 1])
                if not (pd < od[i * k + s] or (pd == od[i * k + s] and pj < j)):
                    return False
    return True


def _recall(approx: List[Int32], exact: List[Int32], n: Int, k: Int) -> Float64:
    var hit = 0
    for i in range(n):
        for s in range(k):
            var j = approx[i * k + s]
            for t in range(k):
                if exact[i * k + t] == j:
                    hit += 1
                    break
    return Float64(hit) / Float64(n * k)


def run_host_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "run under tools/with_identical_mode.sh"
    var failed = 0
    var d = 5
    var k = 6
    for f in range(3):
        var name = String("ties") if f == 0 else (String("wide") if f == 1 else String("clusters"))
        var n = 400
        var x = fixture_ties(n, d) if f == 0 else (fixture_wide(n, d) if f == 1 else kg_fixture_clusters(n, d))
        var cd = List[Float32](length=n * k, fill=Float32(0.0))
        var ci = List[Int32](length=n * k, fill=Int32(0))
        _cell_graph(x, n, d, k, cd, ci)
        var ed = List[Float32](length=n * k, fill=Float32(0.0))
        var ei = List[Int32](length=n * k, fill=Int32(0))
        var used = knn_graph_host(fp(x), n, d, k, KnnGraphPlan(KG_METHOD_EXACT, 0, 0, 0), fp(ed), ip(ei), False)
        report("knn_graph." + name + ".exact_is_cell", used == KG_METHOD_EXACT and same_f32(ed, cd)
               and same_i32(ei, ci), failed)
        var ad = List[Float32](length=n * k, fill=Float32(0.0))
        var ai = List[Int32](length=n * k, fill=Int32(0))
        var ok_all = ivf_graph_host(fp(x), n, d, k, 8, 8, 17, fp(ad), ip(ai))
        report("knn_graph." + name + ".ivf_all_probes_is_exact", ok_all and same_f32(ad, cd) and same_i32(ai, ci),
               failed)
        var pd = List[Float32](length=n * k, fill=Float32(0.0))
        var pi = List[Int32](length=n * k, fill=Int32(0))
        var ok_p = ivf_graph_host(fp(x), n, d, k, 8, 2, 17, fp(pd), ip(pi))
        report("knn_graph." + name + ".ivf_valid_rows", ok_p and _rows_valid(pi, pd, n, k), failed)
        if f == 2:
            var r = _recall(pi, ci, n, k)
            print("  knn_graph.clusters.recall", r)
            report("knn_graph.clusters.ivf_recall_at_least_0.8", r >= 0.8, failed)
        _ = x^
    # a short pool: 64 lists of a 100-row set, one probe, k = 20 -> refused
    var xs = kg_fixture_clusters(100, d)
    var sd = List[Float32](length=100 * 20, fill=Float32(0.0))
    var si = List[Int32](length=100 * 20, fill=Int32(0))
    report("knn_graph.short_pool_refuses", not ivf_graph_host(fp(xs), 100, d, 20, 64, 1, 0, fp(sd), ip(si)), failed)
    # the default rule
    var small = knn_graph_plan(5000, 8, 15, KG_METHOD_IVF, KG_AUTO, KG_AUTO, 0)
    report("knn_graph.plan.small_n_exact", small.method == KG_METHOD_EXACT, failed)
    var big = knn_graph_plan(1000000, 50, 15, KG_METHOD_IVF, KG_AUTO, KG_AUTO, 0)
    report("knn_graph.plan.default_lists_probes", big.method == KG_METHOD_IVF and big.n_lists == 2604
           and big.n_probes == 16, failed)
    var huge = knn_graph_plan(20000000, 50, 15, KG_METHOD_IVF, KG_AUTO, KG_AUTO, 0)
    report("knn_graph.plan.list_cap", huge.n_lists == 16384, failed)
    var given = knn_graph_plan(500, 8, 5, KG_METHOD_IVF, 4, 2, 3)
    report("knn_graph.plan.given_kept", given.method == KG_METHOD_IVF and given.n_lists == 4
           and given.n_probes == 2 and given.seed_off >= 0 and given.seed_off < 500, failed)
    var refused = False
    try:
        _ = knn_graph_plan(500, 8, 5, KG_METHOD_IVF, 4, 5, 3)
    except:
        refused = True
    report("knn_graph.plan.probes_over_lists_refused", refused, failed)
    _ = xs^
    if failed > 0:
        raise Error("knn_graph_check (host): FAILED " + String(failed))
    print("knn_graph_check (host): ALL OK")


def run_checks() raises:
    """The device column against the host column, word for word."""
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "run under tools/with_identical_mode.sh"
    var failed = 0
    var n = 2000
    var d = 7
    var k = 9
    for f in range(3):
        var name = String("ties") if f == 0 else (String("wide") if f == 1 else String("clusters"))
        var x = fixture_ties(n, d) if f == 0 else (fixture_wide(n, d) if f == 1 else kg_fixture_clusters(n, d))
        var xp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
        for p in range(3):
            var plan = KnnGraphPlan(KG_METHOD_EXACT, 0, 0, 0)
            if p == 1:
                plan = KnnGraphPlan(KG_METHOD_IVF, 16, 4, 5)
            elif p == 2:
                plan = KnnGraphPlan(KG_METHOD_IVF, 16, 16, 5)
            for root in range(2):
                var hd = List[Float32](length=n * k, fill=Float32(0.0))
                var hi = List[Int32](length=n * k, fill=Int32(0))
                var gd = List[Float32](length=n * k, fill=Float32(0.0))
                var gi = List[Int32](length=n * k, fill=Int32(0))
                var hu = knn_graph_host(fp(x), n, d, k, plan.copy(), fp(hd), ip(hi), root == 1)
                var gu = knn_graph_device_ptr(
                    xp, n, d, k, plan.copy(),
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(gd.unsafe_ptr())),
                    MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(gi.unsafe_ptr())),
                    root == 1,
                )
                report("knn_graph." + name + ".plan" + String(p) + ".root" + String(root) + ".device_is_host",
                       hu == gu and same_f32(gd, hd) and same_i32(gi, hi), failed)
        _ = x^
    if failed > 0:
        raise Error("knn_graph_check (device): FAILED " + String(failed))
    print("knn_graph_check (device): ALL OK")
