# SPDX-License-Identifier: Apache-2.0
"""Neighbors graph door RUN (`neighbors/neighbors_graph_run.mojo`) without
Python or a GPU.

Each fixture covers typed checks (accepted values, refusals with their exact
wording), the operation requests (word order and float32 rounding) and the
info-word decoding, plus the host CSR merge and dense-to-CSR passes.
Independent fixtures run across the host's cores.
"""
from std.math import isinf
from core.host_parallel import host_parallelize
from neighbors.neighbors_graph_run import (
    XnOpRequest, XN_KERNEL_RBF, XN_U_IDENTITY, xn_f32_word, xn_i32_word, xn_f32_value, xn_double_words,
    xn_rows_match_check, xn_relabel_request, xn_relabel_decode, xn_fill_request, xn_iota_request,
    xn_graph_square_check, xn_graph_triples_check, xn_graph_weights_check, xn_graph_merge_triples,
    xn_graph_merged_copy, xn_graph_dense_counts, xn_graph_dense_fill, xn_graph_csr_check, xn_unit_length_check,
    xn_unit_request, xn_unit_check, xn_unit_missing, xn_pagerank_threshold, xn_pagerank_csr_request,
    xn_pagerank_dense_request, xn_pagerank_decode, xn_pagerank_converged_check, xn_cc_csr_request,
    xn_cc_dense_request, xn_cc_csr_components, xn_cc_relabel_check, xn_louvain_csr_request, xn_louvain_request,
    xn_graph_symmetry_request, xn_louvain_symmetry_check, xn_louvain_device_message, xn_louvain_relabel_check,
    xn_louvain_decode, xn_svgp_gamma, xn_svgp_even_rows, xn_take_rows_request, xn_svgp_fit_request,
    xn_svgp_fit_decode, xn_svgp_definite_check, xn_svgp_predict_request, xn_kernel_rbf_request,
    xn_unary_scale_request, xn_unary_shift_request, xn_matmul_request, xn_svgp_var_request,
)

comptime NEIGHBORS_GRAPH_RUN_FIXTURES = 17


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def ints_equal(got: List[Int], expected: List[Int]) -> Bool:
    if len(got) != len(expected):
        return False
    for i in range(len(got)):
        if got[i] != expected[i]:
            return False
    return True


def floats_equal(got: List[Float64], expected: List[Float64]) -> Bool:
    if len(got) != len(expected):
        return False
    for i in range(len(got)):
        if got[i] != expected[i]:
            return False
    return True


def same(request: XnOpRequest, name: String, a: List[Int], p: List[Int], f: List[Float64]) -> Bool:
    return request.name == name and ints_equal(request.a, a) and ints_equal(request.p, p) and floats_equal(request.f, f)


def refusal(check: Int, expected: String, prefix_only: Bool = False) raises:
    """Run refusing check `check` and compare its wording."""
    try:
        if check == 0:
            xn_rows_match_check(3, 4)
        elif check == 1:
            xn_graph_square_check(2, 3)
        elif check == 2:
            xn_graph_triples_check(2, 3)
        elif check == 3:
            xn_graph_weights_check(1, 2)
        elif check == 4:
            xn_unit_length_check(2, 3, "nstart")
        elif check == 5:
            xn_unit_missing("dangling")
        elif check == 6:
            xn_pagerank_converged_check(False, "100")
        elif check == 7:
            xn_cc_relabel_check(False)
        elif check == 8:
            xn_louvain_relabel_check(False)
        elif check == 9:
            xn_svgp_definite_check(False)
        elif check == 10:
            _ = xn_svgp_gamma(0.0, 14)
        else:
            _ = xn_svgp_gamma(0.0, 12)
    except e:
        var text = String(e)
        require(text.startswith(expected) if prefix_only else text == expected,
                "refusal " + String(check) + " wording: " + text)
        return
    raise Error("refusal " + String(check) + " accepted")


def fixture(unit: Int) raises:
    if unit == 0:
        require(xn_f32_word(0.5) == 0.5, "exact float32 value kept")
        require(xn_f32_word(0.1) == Float64(Float32(0.1)) and xn_f32_word(0.1) != 0.1, "0.1 rounded once")
        require(isinf(xn_f32_word(1.0e39)), "float32 overflow to infinity")
        var one = xn_double_words(1.0)
        require(one[0] == 0x3FF00000 and one[1] == 0, "IEEE words of 1.0")
        var thr = xn_pagerank_threshold(4, 0.25)
        require(thr[0] == 0x3FF00000 and thr[1] == 0, "n * tol words")
        var words: List[Int32] = [7, -3]
        var floats: List[Float32] = [1.5, -2.25]
        require(xn_i32_word(Int(words.unsafe_ptr()), 1) == -3, "int32 word")
        require(xn_f32_value(Int(floats.unsafe_ptr()), 1) == -2.25, "float32 word")
        _ = words
        _ = floats
    elif unit == 1:
        xn_rows_match_check(4, 4)
        refusal(0, "X and y have different numbers of rows")
        xn_graph_square_check(3, 3)
        refusal(1, "the adjacency matrix must be square")
        xn_graph_triples_check(3, 3)
        refusal(2, "the sparse adjacency's row and col differ in length")
        xn_graph_weights_check(2, 2)
        refusal(3, "the CSR adjacency's data must hold one weight per entry")
    elif unit == 2:
        require(same(xn_relabel_request(1, 2, 3, 9), "p2m_relabel", [1, 2, 3], [9], List[Float64]()),
                "relabel request")
        var good: List[Int32] = [3, 0]
        var bad: List[Int32] = [2, 1]
        var g = xn_relabel_decode(Int(good.unsafe_ptr()))
        var b = xn_relabel_decode(Int(bad.unsafe_ptr()))
        require(g[0] == 3 and g[1] and not b[1], "relabel decode")
        _ = good
        _ = bad
        refusal(7, "RuntimeError: connected_components: a component label fell outside [0, n)")
        refusal(8, "RuntimeError: Louvain: a community label fell outside [0, n)")
        xn_cc_relabel_check(True)
        xn_louvain_relabel_check(True)
    elif unit == 3:
        require(same(xn_fill_request(5, 4), "p2m_fill", [5], [4], [0.25]), "fill request: 1/n unrounded")
        require(xn_fill_request(5, 3).f[0] == 1.0 / 3.0, "fill 1/3 in double")
        require(same(xn_iota_request(6, 7), "p2m_iota", [6], [7], List[Float64]()), "iota request")
    elif unit == 4:
        # duplicates (0, 1) summed in stored order, rows grouped
        var rows: List[Int32] = [0, 1, 0]
        var cols: List[Int32] = [1, 0, 1]
        var vals: List[Float32] = [1.0, 2.0, 3.0]
        var indptr = List[Int32](length=3, fill=0)
        var merged = xn_graph_merge_triples(2, 3, Int(rows.unsafe_ptr()), Int(cols.unsafe_ptr()),
                                            Int(vals.unsafe_ptr()), Int(indptr.unsafe_ptr()), True)
        require(merged.count == 2, "merged count")
        require(indptr[0] == 0 and indptr[1] == 1 and indptr[2] == 2, "merged indptr")
        var indices = List[Int32](length=2, fill=-1)
        var weights = List[Float32](length=2, fill=-1.0)
        xn_graph_merged_copy(merged, Int(indices.unsafe_ptr()), Int(weights.unsafe_ptr()))
        require(indices[0] == 1 and indices[1] == 0, "merged indices")
        require(weights[0] == 4.0 and weights[1] == 2.0, "merged weights summed")
        _ = merged^
        _ = rows
        _ = cols
        _ = vals
        _ = indptr
        _ = indices
        _ = weights
    elif unit == 5:
        var rows: List[Int32] = [0, 5]
        var cols: List[Int32] = [1, 0]
        var indptr = List[Int32](length=3, fill=0)
        try:
            _ = xn_graph_merge_triples(2, 2, Int(rows.unsafe_ptr()), Int(cols.unsafe_ptr()), 0,
                                       Int(indptr.unsafe_ptr()), False)
            raise Error("accepted")
        except e:
            require(String(e) == "the sparse adjacency has an entry outside its shape", "outside-shape refusal")
        _ = rows
        _ = cols
        _ = indptr
    elif unit == 6:
        var dense: List[Float32] = [0.0, 1.0, 2.0, 0.0]
        var indptr = List[Int32](length=3, fill=0)
        var m = xn_graph_dense_counts(2, Int(dense.unsafe_ptr()), Int(indptr.unsafe_ptr()))
        require(m == 2 and indptr[1] == 1 and indptr[2] == 2, "dense counts")
        var indices = List[Int32](length=2, fill=-1)
        var weights = List[Float32](length=2, fill=-1.0)
        xn_graph_dense_fill(2, Int(dense.unsafe_ptr()), Int(indptr.unsafe_ptr()), Int(indices.unsafe_ptr()),
                            Int(weights.unsafe_ptr()), m)
        require(indices[0] == 1 and indices[1] == 0 and weights[0] == 1.0 and weights[1] == 2.0, "dense fill")
        _ = dense
        _ = indptr
        _ = indices
        _ = weights
    elif unit == 7:
        var indptr: List[Int32] = [0, 1, 3]
        var address = Int(indptr.unsafe_ptr())
        xn_graph_csr_check(3, 2, True, address, 3, 3)
        xn_graph_csr_check(3, 2, True, address, 3, -1)
        xn_graph_csr_check(3, 2, False, 0, 0, -1)
        try:
            xn_graph_csr_check(2, 2, True, 0, 3, -1)
            raise Error("accepted")
        except e:
            require(String(e) == "the CSR adjacency's indptr must hold n + 1 entries", "graph indptr wording")
        try:
            xn_graph_csr_check(2, 2, False, 0, 0, -1)
            raise Error("accepted")
        except e:
            require(String(e) == "connected_components: indptr must hold n + 1 entries", "cc indptr wording")
        try:
            xn_graph_csr_check(3, 2, True, address, 2, -1)
            raise Error("accepted")
        except e:
            require(String(e) == "the CSR adjacency's indices must hold indptr[n] entries", "indices wording")
        try:
            xn_graph_csr_check(3, 2, True, address, 3, 2)
            raise Error("accepted")
        except e:
            require(String(e) == "the CSR adjacency's data must hold one weight per entry", "data wording")
        _ = indptr
    elif unit == 8:
        xn_unit_length_check(3, 3, "nstart")
        refusal(4, "nstart must be n non-negative values, not all zero")
        refusal(5, "NotImplementedError: mojolearn PageRank dangling: this binary lacks x_neighbors_unit_ff "
                + "(rebuild without -D MOJOLEARN_IDN_XN_UNIT_DEV_OFF)")
        require(same(xn_unit_request(1, 2, 3, 4), "unit_ff", [1, 2, 3], [4], List[Float64]()), "unit request")
        var ok: List[Int32] = [0, 0]
        var negative: List[Int32] = [1, 0]
        xn_unit_check(Int(ok.unsafe_ptr()), "personalization")
        try:
            xn_unit_check(Int(negative.unsafe_ptr()), "personalization")
            raise Error("accepted")
        except e:
            require(String(e) == "personalization must be n non-negative values, not all zero", "unit check wording")
        _ = ok
        _ = negative
    elif unit == 9:
        var r = xn_pagerank_csr_request(1, 2, 3, 4, 5, 6, 7, 8, 9, True, 100, 0.125, False, True, False, True, 0.85)
        require(same(r, "pr_csr", [1, 2, 3, 4, 5, 6, 7], [8, 9, 1, 100, 0x3FF00000, 0, 0, 1, 0, 1],
                     [Float64(Float32(0.85))]), "pagerank csr request")
        var dense = xn_pagerank_dense_request(1, 2, 3, 4, 5, 2, 50, 0.5, True, 0.85)
        require(same(dense, "pr_iterate_sparse", [1, 2, 3, 4, 5], [2, 50, 0x3FF00000, 0, 1],
                     [Float64(Float32(0.85))]), "pagerank dense request")
    elif unit == 10:
        var converged: List[Int32] = [12, 1]
        var failed: List[Int32] = [100, 0]
        var c = xn_pagerank_decode(Int(converged.unsafe_ptr()))
        var f = xn_pagerank_decode(Int(failed.unsafe_ptr()))
        require(c[0] == 12 and c[1] and not f[1], "pagerank decode")
        xn_pagerank_converged_check(True, "100")
        refusal(6, "RuntimeError: PageRank: power iteration failed to converge within 100 iterations")
        _ = converged
        _ = failed
    elif unit == 11:
        require(same(xn_cc_csr_request(1, 2, 3, 4, 5, 6), "cc_iterate_csr", [1, 2, 3, 4], [5, 6], List[Float64]()),
                "cc csr request")
        require(same(xn_cc_dense_request(1, 2, 3, 4), "cc_iterate", [1, 2, 3], [4], List[Float64]()),
                "cc dense request")
        var known: List[Int32] = [0, 3]
        var unknown: List[Int32] = [0, 0]
        require(xn_cc_csr_components(Int(known.unsafe_ptr())) == 2, "cc count from k + 1")
        require(xn_cc_csr_components(Int(unknown.unsafe_ptr())) == -1, "cc needs the relabel pass")
        _ = known
        _ = unknown
    elif unit == 12:
        var csr = xn_louvain_csr_request(1, 2, 3, 4, 5, 6, 7, False, 0, 1.0, 0.1)
        require(same(csr, "louvain_csr", [1, 2, 3, 4, 5], [6, 7, 0, 0], [1.0, Float64(Float32(0.1))]),
                "louvain csr request")
        require(same(xn_louvain_request(1, 2, 3, 4, 2, 0.3, 1.0e-7), "louvain", [1, 2, 3], [4, 2],
                     [Float64(Float32(0.3)), Float64(Float32(1.0e-7))]), "louvain dense request")
        require(same(xn_graph_symmetry_request(1, 2, 3), "graph_symmetry", [1, 2], [3], List[Float64]()),
                "symmetry request")
        var info: List[Float32] = [0.375, 3.0]
        var d = xn_louvain_decode(Int(info.unsafe_ptr()))
        require(d[0] == 0.375 and d[1] == 3.0, "louvain decode")
        _ = info
    elif unit == 13:
        var good: List[Int32] = [0, 1]
        var asym: List[Int32] = [1, 1]
        var empty: List[Int32] = [0, 0]
        xn_louvain_symmetry_check(Int(good.unsafe_ptr()))
        try:
            xn_louvain_symmetry_check(Int(asym.unsafe_ptr()))
            raise Error("accepted")
        except e:
            require(String(e) == "Louvain: the adjacency matrix must be symmetric (an undirected graph)",
                    "symmetry wording")
        try:
            xn_louvain_symmetry_check(Int(empty.unsafe_ptr()))
            raise Error("accepted")
        except e:
            require(String(e) == "Louvain: the graph has no edges", "no-edge wording")
        _ = good
        _ = asym
        _ = empty
        require(xn_louvain_device_message("x_neighbors: Louvain: bad weight") == "ValueError: Louvain: bad weight",
                "device message from Louvain: on")
        require(xn_louvain_device_message("other failure").byte_length() == 0, "other messages pass through")
    elif unit == 14:
        require(xn_svgp_gamma(1.0, 12) == 0.5 and xn_svgp_gamma(2.0, 14) == 0.125, "svgp gamma")
        refusal(10, "ZeroDivisionError: division by zero")
        refusal(11, "ZeroDivisionError: float division by zero")
        refusal(9, "SVGP: the inducing system is not positive definite; raise jitter or noise_variance")
        xn_svgp_definite_check(True)
    elif unit == 15:
        var rows = List[Int32](length=3, fill=-1)
        xn_svgp_even_rows(Int(rows.unsafe_ptr()), 10, 3)
        require(rows[0] == 0 and rows[1] == 3 and rows[2] == 6, "even inducing rows")
        _ = rows
        require(same(xn_take_rows_request(1, 2, 3, 4, 5, 6), "take_rows", [1, 2, 3], [4, 5, 6], List[Float64]()),
                "take_rows request")
        var fit = xn_svgp_fit_request(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0.1, 0.2, 0.3, 0.4)
        require(same(fit, "svgp_fit_ff", [1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11],
                     [Float64(Float32(0.1)), Float64(Float32(0.2)), Float64(Float32(0.3)), Float64(Float32(0.4)),
                      Float64(Float32(0.2))]), "svgp fit request")
        var info: List[Float32] = [-1.5, 1.0]
        var singular: List[Float32] = [0.0, 0.0]
        var fitted = xn_svgp_fit_decode(Int(info.unsafe_ptr()))
        require(fitted[0] == -1.5 and fitted[1], "svgp fit decode")
        require(not xn_svgp_fit_decode(Int(singular.unsafe_ptr()))[1], "svgp singular decode")
        _ = info
        _ = singular
    else:
        require(same(xn_svgp_predict_request(1, 2, 3, 4, 5, 6, 7, 8, 9, 0.1, 0.2), "svgp_predict",
                     [1, 2, 3, 4, 5, 6], [7, 8, 9],
                     [Float64(Float32(0.1)), Float64(Float32(0.2)), Float64(Float32(0.2))]), "svgp predict request")
        require(same(xn_kernel_rbf_request(1, 2, 3, 4, 5, 6, 0.1), "kernel", [1, 2, 3], [4, 5, 6, XN_KERNEL_RBF, 0],
                     [Float64(Float32(0.1)), 0.0]), "rbf kernel request")
        require(same(xn_unary_scale_request(1, 2, 3, 0.1), "unary", [1, 2], [3, XN_U_IDENTITY],
                     [Float64(Float32(0.1)), 0.0]), "unary scale request")
        require(same(xn_unary_shift_request(1, 2, 3, 0.1), "unary", [1, 2], [3, XN_U_IDENTITY],
                     [1.0, Float64(Float32(0.1))]), "unary shift request")
        require(same(xn_matmul_request(1, 2, 3, 4, 5, 1), "matmul", [1, 2, 3], [4, 5, 1], List[Float64]()),
                "matmul request")
        require(same(xn_svgp_var_request(1, 2, 3, 4, 5, 0.1), "svgp_var", [1, 2, 3], [4, 5],
                     [Float64(Float32(0.1))]), "svgp var request")


def run_checks() raises:
    var failures = List[Int](length=NEIGHBORS_GRAPH_RUN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL neighbors_graph_run fixture", unit, String(error))
    host_parallelize(task, NEIGHBORS_GRAPH_RUN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    require(failed == 0, "neighbors_graph_run failed " + String(failed) + " fixtures")
    print("PASS neighbors_graph_run:", NEIGHBORS_GRAPH_RUN_FIXTURES, "typed check, request and decode fixtures")
