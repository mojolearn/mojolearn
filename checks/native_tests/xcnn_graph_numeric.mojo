# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""GCNConv and SAGEConv forward and backward against float64 references of PyG's semantics, on the host column
(lane py2mojo-A, 2026-11-22; ported from python/mojolearn/tests/test_x_cnn_gnn.py `test_gcn`, `test_sage`,
`test_sage_max_normalize` and `test_sage_project`).

The inputs are the original tests' NumPy draws (x, the edge index of `_graph`, the edge weights, the bias and the
upstream gradient), and the references are the original's float64 `ref_gcn`, `ref_sage`, `ref_sage_max_norm` and
project composition, stored as raw words in `fixtures/xcnn_graph_numeric/` by `tools/py2mojo_native_fixtures_b8.py`.
The original read the layer's own seeded initial weights into its reference; the initial draw is not what these tests
assert, so the weights are NumPy draws stored beside the inputs and handed to the same RUN calls.

Product side: the host image's x_cnn entry through x_cnn/graph_layer_run.mojo's `GraphTableCalls`, with the calls
and words of the host forms in bindings/xcnn_graph_native_door.mojo: GCN `_gcn_graph` (`gcn_graph_run`: self loops,
both CSR views, the normalization; the graph route bit XCNN_F_GRAPH is 0 on the host image, which exports no
`x_cnn_idn_flags`), `xcnn_gcn_forward` (gemm op 1, spmm mode 0 with the normalized values, the bias rows, add) and
`_gcn_backward_host` (`gcn_backward_host_run`); SAGE `xcnn_sage_forward` (`graph_views_run`, the projection's linear
and ReLU, the max graph op 0 or the spmm over ones with mode 1 mean / 0 sum, lin_l's linear, the root gemm and add,
the normalization graph op 2) and `_sage_backward_host` (`sage_backward_host_run`). The composition repeats the
door's call order; the arithmetic under test is the entry's.

    test_gcn                  improved False unweighted, improved True weighted: y, dx, grad_weight_, grad_bias_
                              to rtol = atol = 1e-5
    test_sage                 aggr mean, sum: y, dx, lin_l grads, grad_weight_r_ to rtol = atol = 1e-5
    test_sage_max_normalize   aggr max with ties, normalize False, True: y to 1e-5, dx to rtol 1e-4, atol 1e-5
    test_sage_project         project=True, aggr mean, max: y to 1e-5, dx and lin's grad_weight_ to rtol 1e-4,
                              atol 1e-5

The cached-graph bit tests, the torch oracle of the max reference and the public layer objects stay in Python. The
units are independent and run across the native host worker pool.
"""
from std.math import isnan
from x_cnn.host.entry_runs_host import publish_x_cnn_host_entries
from x_cnn.graph_layer_run import (
    GraphTableCalls, GraphEdges, SageBackwardHost, graph_views_run, gcn_graph_run, gcn_backward_host_run,
    sage_backward_host_run,
)
from x_cnn.graph_run import (
    graph_run_gemm, graph_run_spmm, graph_run_add, graph_run_bias_rows, graph_run_linear_forward,
    graph_run_relu_forward, graph_run_graph_op, graph_run_aggr_kind,
)
from checks.native_tests.cnn_cases import cnn_entry
from checks.native_tests.tsa_metrics_fixture_words import fixture_f32, fixture_i32
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases

comptime GNN_DIR = "checks/native_tests/fixtures/xcnn_graph_numeric/"


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def addr(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def close_all(actual: List[Float32], expected: List[Float64], rtol: Float64, atol: Float64, label: String) raises:
    """numpy.testing.assert_allclose over a block: NaN only against NaN; names the first differing word."""
    require(len(actual) == len(expected), label + ": extent " + String(len(actual)) + " vs " + String(len(expected)))
    for i in range(len(expected)):  # small-loop(oracle: one block)
        var a = Float64(actual[i])
        var e = expected[i]
        var ok: Bool
        if isnan(a) or isnan(e):
            ok = isnan(a) and isnan(e)
        else:
            ok = abs(a - e) <= atol + rtol * abs(e)
        require(ok, label + ": word " + String(i) + " is " + String(actual[i]) + ", reference " + String(e))


def edges_into(mut calls: GraphTableCalls, prefix: String, count: Int) raises -> GraphEdges:
    """The stored (2, E) edge index as the int32 rows the door's `_edges_run` builds."""
    var src = fixture_i32(GNN_DIR + prefix + "-src.i32", count)
    var dst = fixture_i32(GNN_DIR + prefix + "-dst.i32", count)
    var hs = calls.alloc_i32(count)
    var hd = calls.alloc_i32(count)
    for k in range(count):  # small-loop(fixture: edge words)
        calls.i32[hs // 2][k] = src[k]
        calls.i32[hd // 2][k] = dst[k]
    return GraphEdges(hs, hd, count)


# ---- GCNConv (in 6, out 4, 40 nodes, 119 edges) ----
def gcn_case(unit: Int) raises:
    var n = 40
    var d = 6
    var f = 4
    var e = 119
    var improved = unit == 1
    var weighted = unit == 1
    var p = String("gcn") + String(unit)
    var label = String("gcn improved ") + String(improved) + " weighted " + String(weighted)
    var calls = GraphTableCalls(cnn_entry())
    var edges = edges_into(calls, p, e)
    var x = fixture_f32(GNN_DIR + p + "-x.f32", n * d)
    var w = fixture_f32(GNN_DIR + p + "-w.f32", f * d)
    var b = fixture_f32(GNN_DIR + p + "-b.f32", f)
    var g = fixture_f32(GNN_DIR + p + "-g.f32", n * f)
    var ew = fixture_f32(GNN_DIR + p + "-ew.f32", e) if weighted else List[Float32]()
    var gv = gcn_graph_run(calls, edges, n, addr(ew) if weighted else 0, e if weighted else 0, True, True, improved,
                           False)
    var nnz = gv.views.nnz
    var h = List[Float32](length=n * f, fill=0)
    graph_run_gemm(calls.entry, addr(x), addr(w), addr(h), n, f, d, 1)
    var spread = List[Float32](length=n * f, fill=0)
    graph_run_spmm(calls.entry, calls.address(gv.vals), addr(h), addr(spread), calls.address(gv.views.csr_f), n, f,
                   nnz, 0)
    var rows = List[Float32](length=n * f, fill=0)
    graph_run_bias_rows(addr(b), addr(rows), n, f)
    var y = List[Float32](length=n * f, fill=0)
    graph_run_add(calls.entry, addr(spread), addr(rows), addr(y), n * f)
    var gw = List[Float32](length=f * d, fill=0)
    var gb = List[Float32](length=f, fill=0)
    var dx = List[Float32](length=n * d, fill=0)
    gcn_backward_host_run(calls, calls.address(gv.views.csr_t), calls.address(gv.views.order_f),
                          calls.address(gv.views.order_t), calls.address(gv.vals), nnz, n, d, f, addr(g), addr(x),
                          addr(w), True, addr(b), addr(gw), addr(gb), addr(dx))
    close_all(y, fixture_f64(GNN_DIR + p + "-y.f64", n * f), 1e-5, 1e-5, label + ": forward")
    close_all(dx, fixture_f64(GNN_DIR + p + "-dx.f64", n * d), 1e-5, 1e-5, label + ": dx")
    close_all(gw, fixture_f64(GNN_DIR + p + "-dw.f64", f * d), 1e-5, 1e-5, label + ": grad_weight_")
    close_all(gb, fixture_f64(GNN_DIR + p + "-db.f64", f), 1e-5, 1e-5, label + ": grad_bias_")
    _ = x^
    _ = w^
    _ = b^
    _ = g^
    _ = ew^
    _ = h^
    _ = spread^
    _ = rows^
    _ = calls^


# ---- SAGEConv: the host form of xcnn_sage_forward and _sage_backward_host ----
struct SageResult(Movable):
    var y: List[Float32]
    var dx: List[Float32]
    var dw: List[Float32]
    var db: List[Float32]
    var gwr: List[Float32]
    var dwp: List[Float32]

    def __init__(out self, n: Int, d: Int, f: Int):
        self.y = List[Float32](length=n * f, fill=0)
        self.dx = List[Float32](length=n * d, fill=0)
        self.dw = List[Float32](length=f * d, fill=0)
        self.db = List[Float32](length=f, fill=0)
        self.gwr = List[Float32](length=f * d, fill=0)
        self.dwp = List[Float32](length=d * d, fill=0)


def sage_run(p: String, n: Int, d: Int, f: Int, e: Int, aggr: String, normalize: Bool,
             project: Bool) raises -> SageResult:
    """SAGEConv(d, f, aggr, normalize, project), root_weight and bias on (the defaults), on the stored inputs."""
    var calls = GraphTableCalls(cnn_entry())
    var edges = edges_into(calls, p, e)
    var views = graph_views_run(calls, edges.src, edges.dst, n, e)
    var x = fixture_f32(GNN_DIR + p + "-x.f32", n * d)
    var wl = fixture_f32(GNN_DIR + p + "-wl.f32", f * d)
    var bl = fixture_f32(GNN_DIR + p + "-bl.f32", f)
    var wr = fixture_f32(GNN_DIR + p + "-wr.f32", f * d)
    var g = fixture_f32(GNN_DIR + p + "-g.f32", n * f)
    var wp = fixture_f32(GNN_DIR + p + "-wp.f32", d * d) if project else List[Float32]()
    var bp = fixture_f32(GNN_DIR + p + "-bp.f32", d) if project else List[Float32]()
    var hp = List[Float32](length=n * d if project else 0, fill=0)
    var xs = x.copy()
    if project:
        graph_run_linear_forward(calls.entry, addr(x), addr(wp), addr(bp), addr(hp), n, d, d)
        graph_run_relu_forward(calls.entry, addr(hp), addr(xs), n * d)
    var kind = graph_run_aggr_kind(aggr)
    var agg = List[Float32](length=n * d, fill=0)
    var aux = List[Float32](length=2 * n * d if kind == 2 else 0, fill=0)
    if kind == 2:
        graph_run_graph_op(calls.entry, addr(xs), addr(xs), addr(aux), addr(agg), calls.address(views.csr_f), n, d,
                           e, 0)
    else:
        var ones = List[Float32](length=e, fill=1)
        graph_run_spmm(calls.entry, addr(ones), addr(xs), addr(agg), calls.address(views.csr_f), n, d, e,
                       1 if aggr == "mean" else 0)
        _ = ones^
    var lin = List[Float32](length=n * f, fill=0)
    graph_run_linear_forward(calls.entry, addr(agg), addr(wl), addr(bl), addr(lin), n, d, f)
    var root = List[Float32](length=n * f, fill=0)
    graph_run_gemm(calls.entry, addr(x), addr(wr), addr(root), n, f, d, 1)
    var pre = List[Float32](length=n * f, fill=0)
    graph_run_add(calls.entry, addr(lin), addr(root), addr(pre), n * f)
    var result = SageResult(n, d, f)
    var den = List[Float32](length=n if normalize else 0, fill=0)
    var nograph = List[Int32](length=n + 1 if normalize else 0, fill=0)
    if normalize:
        graph_run_graph_op(calls.entry, addr(pre), addr(pre), addr(den), addr(result.y), Int(nograph.unsafe_ptr()),
                           n, f, 0, 2)
    else:
        result.y = pre.copy()
    var io = SageBackwardHost(
        addr(g), addr(x), addr(agg), addr(wl), addr(bl), addr(result.dw), addr(result.db), addr(result.dx),
        addr(result.y) if normalize else 0, addr(pre) if normalize else 0, addr(den) if normalize else 0,
        Int(nograph.unsafe_ptr()) if normalize else 0, addr(aux) if kind == 2 else 0, addr(wr), addr(result.gwr),
        addr(xs) if project else 0, addr(hp) if project else 0, addr(wp) if project else 0,
        addr(bp) if project else 0, addr(result.dwp) if project else 0,
        0)
    var dbp = List[Float32](length=d if project else 0, fill=0)
    if project:
        io.dbp = addr(dbp)
    sage_backward_host_run(calls, calls.address(views.csr_f), calls.address(views.csr_t), e, n, d, f, kind,
                           normalize, True, project, io)
    _ = x^
    _ = wl^
    _ = bl^
    _ = wr^
    _ = g^
    _ = wp^
    _ = bp^
    _ = hp^
    _ = xs^
    _ = agg^
    _ = aux^
    _ = lin^
    _ = root^
    _ = pre^
    _ = den^
    _ = nograph^
    _ = dbp^
    _ = calls^
    return result^


def sage_case(unit: Int) raises:
    var aggr = String("mean") if unit == 0 else String("sum")
    var p = String("sage") + String(unit)
    var label = String("sage aggr ") + aggr
    var n = 40
    var d = 6
    var f = 5
    var r = sage_run(p, n, d, f, 119, aggr, False, False)
    close_all(r.y, fixture_f64(GNN_DIR + p + "-y.f64", n * f), 1e-5, 1e-5, label + ": forward")
    close_all(r.dx, fixture_f64(GNN_DIR + p + "-dx.f64", n * d), 1e-5, 1e-5, label + ": dx")
    close_all(r.dw, fixture_f64(GNN_DIR + p + "-dwl.f64", f * d), 1e-5, 1e-5, label + ": lin_l grad_weight_")
    close_all(r.db, fixture_f64(GNN_DIR + p + "-dbl.f64", f), 1e-5, 1e-5, label + ": lin_l grad_bias_")
    close_all(r.gwr, fixture_f64(GNN_DIR + p + "-dwr.f64", f * d), 1e-5, 1e-5, label + ": grad_weight_r_")


def sage_max_case(unit: Int) raises:
    var normalize = unit == 1
    var p = String("smax") + String(unit)
    var label = String("sage max normalize ") + String(normalize)
    var n = 30
    var d = 5
    var f = 4
    var r = sage_run(p, n, d, f, 91, "max", normalize, False)
    close_all(r.y, fixture_f64(GNN_DIR + p + "-y.f64", n * f), 1e-5, 1e-5, label + ": forward")
    close_all(r.dx, fixture_f64(GNN_DIR + p + "-dx.f64", n * d), 1e-4, 1e-5, label + ": dx")


def sage_project_case(unit: Int) raises:
    var aggr = String("mean") if unit == 0 else String("max")
    var p = String("sproj") + String(unit)
    var label = String("sage project aggr ") + aggr
    var n = 30
    var d = 5
    var f = 4
    var r = sage_run(p, n, d, f, 93, aggr, False, True)
    close_all(r.y, fixture_f64(GNN_DIR + p + "-y.f64", n * f), 1e-5, 1e-5, label + ": forward")
    close_all(r.dx, fixture_f64(GNN_DIR + p + "-dx.f64", n * d), 1e-4, 1e-5, label + ": dx")
    close_all(r.dwp, fixture_f64(GNN_DIR + p + "-dwp.f64", d * d), 1e-4, 1e-5, label + ": lin grad_weight_")


def test_gcn() raises:
    # The publication _mojolearn_x_cnn_host makes at module init (idempotent: each test publishes first).
    publish_x_cnn_host_entries()
    check_cases("gcn", 2, gcn_case)


def test_sage() raises:
    publish_x_cnn_host_entries()
    check_cases("sage", 2, sage_case)


def test_sage_max_normalize() raises:
    publish_x_cnn_host_entries()
    check_cases("sage_max_normalize", 2, sage_max_case)


def test_sage_project() raises:
    publish_x_cnn_host_entries()
    check_cases("sage_project", 2, sage_project_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
