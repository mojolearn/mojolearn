# SPDX-License-Identifier: Apache-2.0
"""SAGEConv max aggregation: forward maxima and the input gradient against
an independent tie-splitting adjoint, on the host column (lane
tests-to-mojo-spatial, 2026-11-16; ported from
`test_cnn_max_gradient_ownership.py::test_max_backward_matches_independent_tie_scatter`).

Four units: nodes in {37, 53} x ties in {False, True}; 5 features;
x[i, f] = (5 i + f) * 0.125 - 1 (exact in float32), with x[1] = x[0] when
ties; the eight edges (0->2, 1->2, 2->3, 3->4, 4->0, 1->3, 0->4, 3->0);
upstream g[i, f] = (5 i + f + 1) / 8. The layer is aggr='max',
root_weight=False, bias=False with lin_l's weight the identity, so its
output is the aggregate itself (the Python test asserted the layer output;
here the aggregate the door feeds lin_l is asserted, and the backward runs
the whole host-form tape, lin_l included). Expected values are rebuilt here
in the test's own order: the per-target maximum (0 for a target with no
source), and for every target in ascending order each winning source gets
float32(g[t, f] / count) added in float32. Both comparisons are exact, as
the Python test's assert_array_equal was.

Product side: the graph views, the forward op (kind 0) and the host-form
backward tape (kind 2) through x_cnn/graph_layer_run.mojo's GraphTableCalls
on the host image's x_cnn entry, the calls and words
bindings/xcnn_graph_native_door.mojo uses (`xcnn_sage_forward`,
`_sage_backward_host`). The units are independent and run across the
native host worker pool.
"""
from x_cnn.host.entry_runs_host import publish_x_cnn_host_entries
from x_cnn.graph_layer_run import GraphTableCalls, SageBackwardHost, graph_views_run, sage_backward_host_run
from x_cnn.graph_run import graph_run_graph_op, graph_run_aggr_kind
from checks.native_tests.cnn_cases import cnn_entry
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime SAGE_MAX_UNITS = 4
comptime SAGE_MAX_FEATURES = 5
comptime SAGE_MAX_EDGES = 8


def _case(unit: Int) raises:
    var nodes = 37 if unit < 2 else 53
    var ties = unit % 2 == 1
    var feats = SAGE_MAX_FEATURES
    var size = nodes * feats
    var what = "sage max nodes " + String(nodes) + (" ties" if ties else "")
    var x = List[Float32](length=size, fill=0)
    var g = List[Float32](length=size, fill=0)
    for i in range(size):  # small-loop(fixture words)
        x[i] = Float32(i) * Float32(0.125) - Float32(1)
        g[i] = (Float32(i) + Float32(1)) / Float32(8)
    if ties:
        for f in range(feats):  # small-loop(fixture words)
            x[feats + f] = x[f]
    var src: List[Int] = [0, 1, 2, 3, 4, 1, 0, 3]
    var dst: List[Int] = [2, 2, 3, 4, 0, 3, 4, 0]

    # ---- product: graph views, forward aggregate, host-form backward tape
    var calls = GraphTableCalls(cnn_entry())
    var hs = calls.alloc_i32(SAGE_MAX_EDGES)
    var hd = calls.alloc_i32(SAGE_MAX_EDGES)
    for k in range(SAGE_MAX_EDGES):  # small-loop(fixture edge words)
        calls.i32[hs // 2][k] = Int32(src[k])
        calls.i32[hd // 2][k] = Int32(dst[k])
    var views = graph_views_run(calls, hs, hd, nodes, SAGE_MAX_EDGES)
    var agg = List[Float32](length=size, fill=0)
    var aux = List[Float32](length=2 * size, fill=0)
    graph_run_graph_op(calls.entry, Int(x.unsafe_ptr()), Int(x.unsafe_ptr()), Int(aux.unsafe_ptr()),
                       Int(agg.unsafe_ptr()), calls.address(views.csr_f), nodes, feats, SAGE_MAX_EDGES, 0)
    var w = List[Float32](length=feats * feats, fill=0)
    for f in range(feats):  # small-loop(identity weight words)
        w[f * feats + f] = 1
    var b = List[Float32](length=feats, fill=0)
    var dw = List[Float32](length=feats * feats, fill=0)
    var db = List[Float32](length=feats, fill=0)
    var dx = List[Float32](length=size, fill=0)
    var kind = graph_run_aggr_kind("max")
    var io = SageBackwardHost(
        Int(g.unsafe_ptr()), Int(x.unsafe_ptr()), Int(agg.unsafe_ptr()), Int(w.unsafe_ptr()), Int(b.unsafe_ptr()),
        Int(dw.unsafe_ptr()), Int(db.unsafe_ptr()), Int(dx.unsafe_ptr()), 0, 0, 0, 0, Int(aux.unsafe_ptr()),
        0, 0, 0, 0, 0, 0, 0, 0)
    sage_backward_host_run(calls, calls.address(views.csr_f), calls.address(views.csr_t), SAGE_MAX_EDGES, nodes,
                           feats, feats, kind, False, False, False, io)

    # ---- independent reference, in the Python test's order
    var want_y = List[Float32](length=size, fill=0)
    var want_dx = List[Float32](length=size, fill=0)
    for t in range(nodes):  # small-loop(reference over targets ascending, the test's order)
        var sources = List[Int]()
        for k in range(SAGE_MAX_EDGES):  # small-loop(edge list)
            if dst[k] == t:
                sources.append(src[k])
        if len(sources) == 0:
            continue
        for f in range(feats):  # small-loop(features)
            var m = x[sources[0] * feats + f]
            for s in sources:  # small-loop(sources)
                if x[s * feats + f] > m:
                    m = x[s * feats + f]
            want_y[t * feats + f] = m
            var count = 0
            for s in sources:  # small-loop(sources)
                if x[s * feats + f] == m:
                    count += 1
            var share = g[t * feats + f] / Float32(count)
            for s in sources:  # small-loop(winners in source order)
                if x[s * feats + f] == m:
                    want_dx[s * feats + f] = want_dx[s * feats + f] + share

    for i in range(size):  # small-loop(exact comparison words)
        require(agg[i] == want_y[i], what + ": forward word " + String(i) + " is " + String(agg[i])
                + ", expected " + String(want_y[i]))
        require(dx[i] == want_dx[i], what + ": input gradient word " + String(i) + " is " + String(dx[i])
                + ", expected " + String(want_dx[i]))
    _ = x^
    _ = g^
    _ = agg^
    _ = aux^
    _ = w^
    _ = b^
    _ = dw^
    _ = db^
    _ = calls^


def _run_cpu(mut report: Report) raises:
    # The publication _mojolearn_x_cnn_host makes at module init.
    publish_x_cnn_host_entries()
    var cases = report.cases(SAGE_MAX_UNITS)
    def task(unit: Int) {imm cases}:
        try:
            _case(unit)
        except problem:
            cases.fail(unit, problem)
    report.parallel(task, SAGE_MAX_UNITS)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("xcnn_sage_max_adjoint", cpu=True, gpu=False, serial="", run=run_cases)
