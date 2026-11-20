# SPDX-License-Identifier: Apache-2.0
"""The CNN trainer's Linear layer forward and backward against float64, on
the host column (lane tmig-sequence-1, 2026-11-14; ported from
`test_x_cnn_trainer.py::test_linear_softmax_sgd`).

x (9, 5), a (3, 5) weight and a bias through the linear entries the layer
door calls (bindings/xcnn_graph_native_door.mojo `_linear_forward` and the
backward): out = x W^T + b within 1e-5; for an arbitrary upstream gradient
g (9, 3): dx = g W, dW = g^T x and db = sum_rows g within rtol 1e-5 / atol
1e-6. (The softmax cross-entropy and SGD steps run inside CNNClassifier's one
native fit; `test_cnn_classifier_learns` stays in Python as a public-fit
test.) One fixture: it runs on the calling thread (a single unit; the host
pool has nothing to split).
"""
from x_cnn.host.entry_runs_host import publish_x_cnn_host_entries
from x_cnn.graph_run import graph_run_linear_forward, graph_run_linear_backward
from checks.native_tests.cnn_cases import cnn_entry, addr, zeros32, normals, check_close
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The host column (was cnn_trainer_cpu_suite.mojo) ----
def _linear() raises:
    var n = 9
    var d = 5
    var f = 3
    var x = normals(n * d, 0)
    var w = normals(f * d, 1, 0.4)
    var b = normals(f, 2, 0.4)
    var out = zeros32(n * f)
    graph_run_linear_forward(cnn_entry(), addr(x), addr(w), addr(b), addr(out), n, d, f)
    var want = List[Float64](length=n * f, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        for o in range(f):  # small-loop(fixture outputs)
            var acc = Float64(b[o])
            for k in range(d):  # small-loop(fixture features)
                acc += Float64(x[i * d + k]) * Float64(w[o * d + k])
            want[i * f + o] = acc
    check_close(out, want, 1e-5, 1e-5, "linear out")
    var g = normals(n * f, 3)
    var dx = zeros32(n * d)
    var dw = zeros32(f * d)
    var db = zeros32(f)
    graph_run_linear_backward(cnn_entry(), addr(x), addr(w), addr(g), addr(dx), addr(dw), addr(db), n, d, f)
    var want_dx = List[Float64](length=n * d, fill=0.0)
    var want_dw = List[Float64](length=f * d, fill=0.0)
    var want_db = List[Float64](length=f, fill=0.0)
    for i in range(n):  # small-loop(fixture rows)
        for o in range(f):  # small-loop(fixture outputs)
            var gv = Float64(g[i * f + o])
            want_db[o] += gv
            for k in range(d):  # small-loop(fixture features)
                want_dx[i * d + k] += gv * Float64(w[o * d + k])
                want_dw[o * d + k] += gv * Float64(x[i * d + k])
    check_close(dx, want_dx, 1e-5, 1e-6, "linear dx")
    check_close(dw, want_dw, 1e-5, 1e-6, "linear grad_weight_")
    check_close(db, want_db, 1e-5, 1e-6, "linear grad_bias_")
    _ = x^
    _ = w^
    _ = b^
    _ = g^


def _run_cpu(mut report: Report) raises:
    # The publication _mojolearn_x_cnn_host makes at module init.
    publish_x_cnn_host_entries()
    _linear()
    print("PASS cnn trainer linear", 1)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("cnn_trainer", cpu=True, gpu=False, serial="", run=run_cases)
