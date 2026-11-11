# SPDX-License-Identifier: Apache-2.0
"""The graph layers' RUN half (`x_cnn/graph_run.mojo`), reached without
Python (lane doorsplit-cnn-xseq, 2026-11-11). One case per group of RUN
functions: literal words, host word moves and the layers' own refusal
texts. The native entries the RUN calls are covered by
`xcnn_entry_table_suite.mojo`.

The cases are independent units and run across the native host worker pool.
"""
from std.math import sqrt
from core.host_parallel import host_parallelize
from core.cast_in import CAST_KIND_SIGNED
from x_cnn.graph_run import (
    graph_run_same_shape, graph_run_csr_words, graph_run_check_edge_layout, graph_run_check_cache_dims,
    graph_run_check_index_layout, graph_run_check_gemm, graph_run_check_graph_x, graph_run_check_x,
    graph_run_check_edge_weight, graph_run_check_bias, graph_run_check_gcn_weight, graph_run_check_root_weight,
    graph_run_check_linear_input, graph_run_check_linear_weights, graph_run_check_linear_gradient,
    graph_run_check_gradient, graph_run_check_channels, graph_run_check_features, graph_run_check_aggr,
    graph_run_aggr_kind, graph_run_sage_res_form, graph_run_gcn_bound, graph_run_fan_in_bound, graph_run_ones,
    graph_run_slice, graph_run_edges, graph_run_bias_rows,
)

comptime XCNN_GRAPH_RUN_CASES = 12


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _refuses(problem: Error, text: String) raises:
    require(String(problem) == text, "refusal text: got '" + String(problem) + "', expected '" + text + "'")


def case(unit: Int) raises:
    if unit == 0:
        require(graph_run_same_shape([2, 3], [2, 3]) and not graph_run_same_shape([2, 3], [3, 2])
                and not graph_run_same_shape([2], [2, 1]), "same shape")
        require(graph_run_csr_words(4, 5) == 15, "CSR words")
    elif unit == 1:
        graph_run_check_edge_layout([2, 7])
        try:
            graph_run_check_edge_layout([3, 7])
            require(False, "(3, E) edge index accepted")
        except problem:
            _refuses(problem, "mojolearn: edge_index must be (2, E)")
        try:
            graph_run_check_cache_dims(-1, 0)
            require(False, "negative node count accepted")
        except problem:
            _refuses(problem, "graph cache dimensions must be nonnegative")
        graph_run_check_index_layout([15], 15)
        try:
            graph_run_check_index_layout([14], 15)
            require(False, "short CSR view accepted")
        except problem:
            _refuses(problem, "cached graph index layout differs from its node/edge metadata")
    elif unit == 2:
        graph_run_check_gemm(6, 12, 2, 4, 3)
        try:
            graph_run_check_gemm(6, 11, 2, 4, 3)
            require(False, "GEMM size mismatch accepted")
        except problem:
            _refuses(problem, "GEMM inputs differ from the declared graph operation dimensions")
        try:
            graph_run_check_graph_x([5, 3], 4)
            require(False, "graph x channel mismatch accepted")
        except problem:
            _refuses(problem, "mojolearn: graph x must be (n, in_channels)")
        try:
            graph_run_check_x([5], 4)
            require(False, "rank-one resident x accepted")
        except problem:
            _refuses(problem, "ValueError: mojolearn: x must be (n, 4)")
    elif unit == 3:
        try:
            graph_run_check_edge_weight(3, 4)
            require(False, "edge weight count mismatch accepted")
        except problem:
            _refuses(problem, "ValueError: mojolearn: edge_weight must have one entry per edge")
        try:
            graph_run_check_bias(3, 4)
            require(False, "bias size mismatch accepted")
        except problem:
            _refuses(problem, "ValueError: mojolearn: bias_ has the wrong size")
        graph_run_check_gcn_weight([4, 3], 4, 3)
        try:
            graph_run_check_gcn_weight([3, 4], 4, 3)
            require(False, "transposed GCN weight accepted")
        except problem:
            _refuses(problem, "GCN weight shape differs from configured channels")
        try:
            graph_run_check_root_weight([4], 4, 3)
            require(False, "rank-one root weight accepted")
        except problem:
            _refuses(problem, "SAGE root weight shape differs from configured channels")
    elif unit == 4:
        graph_run_check_linear_input([5, 3], 3)
        try:
            graph_run_check_linear_input([5, 2], 3)
            require(False, "linear input mismatch accepted")
        except problem:
            _refuses(problem, "linear input shape differs from in_features")
        graph_run_check_linear_weights([4, 3], [4], 4, 3)
        try:
            graph_run_check_linear_weights([4, 3], [3], 4, 3)
            require(False, "linear bias mismatch accepted")
        except problem:
            _refuses(problem, "linear weights differ from configured dimensions")
        graph_run_check_linear_gradient([5, 3], [5, 4], 3, 4)
        try:
            graph_run_check_linear_gradient([5, 3], [6, 4], 3, 4)
            require(False, "linear gradient rows mismatch accepted")
        except problem:
            _refuses(problem, "linear gradient shape differs from forward output")
    elif unit == 5:
        graph_run_check_gradient([5, 3], [5, 4], 4)
        try:
            graph_run_check_gradient([5, 3], [5, 3], 4)
            require(False, "graph gradient mismatch accepted")
        except problem:
            _refuses(problem, "mojolearn: graph grad_out shape differs from forward output")
        try:
            graph_run_check_channels(0, 4)
            require(False, "zero channels accepted")
        except problem:
            _refuses(problem, "graph layer channels must be positive")
        try:
            graph_run_check_features(3, -1)
            require(False, "negative features accepted")
        except problem:
            _refuses(problem, "linear feature dimensions must be positive")
    elif unit == 6:
        graph_run_check_aggr("add")
        try:
            graph_run_check_aggr("min")
            require(False, "min aggregation accepted")
        except problem:
            _refuses(problem, "NotImplementedError: SAGE aggregation must be mean, sum, add or max")
        require(graph_run_aggr_kind("max") == 2 and graph_run_aggr_kind("mean") == 1
                and graph_run_aggr_kind("sum") == 0 and graph_run_aggr_kind("add") == 0, "aggregation kind")
    elif unit == 7:
        require(not graph_run_sage_res_form(False, False, True, False, "mean"), "host image is not resident")
        require(not graph_run_sage_res_form(True, True, True, False, "mean"), "projection is not resident")
        require(graph_run_sage_res_form(True, False, True, True, "max"), "graph lane switch is resident")
        require(graph_run_sage_res_form(True, False, False, False, "sum"), "plain sum is resident")
        require(not graph_run_sage_res_form(True, False, False, True, "sum"), "normalize off the lane is host")
        require(not graph_run_sage_res_form(True, False, False, False, "max"), "max off the lane is host")
    elif unit == 8:
        require(graph_run_gcn_bound(2, 4) == sqrt(1.0), "GCN bound")
        require(graph_run_fan_in_bound(4) == 0.5, "fan-in bound")
    elif unit == 9:
        var ones = List[Float32](length=4, fill=0.0)
        graph_run_ones(Int(ones.unsafe_ptr()), 3)
        require(ones[0] == 1.0 and ones[2] == 1.0 and ones[3] == 0.0, "ones words")
        var src: List[Int32] = [7, 8, 9, 10]
        var dst = List[Int32](length=2, fill=0)
        graph_run_slice(Int(dst.unsafe_ptr()), Int(src.unsafe_ptr()), 1, 2)
        require(dst[0] == 8 and dst[1] == 9, "slice words")
        _ = ones
        _ = src
        _ = dst
    elif unit == 10:
        var edges: List[Int64] = [0, 1, 1, 2]     # (2, 2) C-ordered int64: sources 0 1, targets 1 2
        var s = List[Int32](length=2, fill=-1)
        var t = List[Int32](length=2, fill=-1)
        graph_run_edges(Int(edges.unsafe_ptr()), 16, 8, 2, 8, CAST_KIND_SIGNED, True, 3,
                        Int(s.unsafe_ptr()), Int(t.unsafe_ptr()))
        require(s[0] == 0 and s[1] == 1 and t[0] == 1 and t[1] == 2, "edge rows")
        try:
            graph_run_edges(Int(edges.unsafe_ptr()), 16, 8, 2, 8, CAST_KIND_SIGNED, True, 2,
                            Int(s.unsafe_ptr()), Int(t.unsafe_ptr()))
            require(False, "out-of-range node accepted")
        except problem:
            _refuses(problem, "mojolearn: edge_index refers to a node that does not exist")
        _ = edges
        _ = s
        _ = t
    else:
        var bias: List[Float32] = [1.0, 2.0]
        var rows = List[Float32](length=6, fill=0.0)
        graph_run_bias_rows(Int(bias.unsafe_ptr()), Int(rows.unsafe_ptr()), 3, 2)
        require(rows[0] == 1.0 and rows[1] == 2.0 and rows[4] == 1.0 and rows[5] == 2.0, "bias rows")
        _ = bias
        _ = rows


def run_checks() raises:
    var failures = List[Int](length=XCNN_GRAPH_RUN_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL x_cnn graph run case", unit, String(problem))
    host_parallelize(task, XCNN_GRAPH_RUN_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "x_cnn graph run cases failed: " + String(failed))
    print("PASS x_cnn graph run", XCNN_GRAPH_RUN_CASES)
