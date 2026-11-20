# SPDX-License-Identifier: Apache-2.0
"""Neighbors semi-supervised and chi2 sampler door RUN
(`neighbors/neighbors_semi_run.mojo`) without Python or a GPU.

Each fixture covers typed checks (accepted values, refusals with their exact
wording), the operation requests (word order and float32 rounding), the
info-word decoding and the host dense-to-CSR pass. Independent fixtures run
across the host's cores.
"""
from core.host_parallel import host_parallelize
from neighbors.neighbors_graph_run import XnOpRequest
from neighbors.neighbors_semi_run import (
    XN_SEMI_U_LOG, xn_achi2_negative_check, xn_achi2_fit_negative, xn_achi2_width, xn_achi2_request,
    xn_achi2_csr_count, xn_achi2_csr_fill, xn_schi2_skew_check, xn_schi2_missing_fit, xn_schi2_log_request,
    xn_schi2_transform_request, xn_semi_knn_k, xn_semi_sqdist_request, xn_semi_knn_select_request,
    xn_semi_knn_sq_request, xn_lp_knn_graph_request, xn_lp_knn_product_request, xn_semi_matmul_request,
    xn_semi_rbf_request, xn_semi_row_normalize_request, xn_semi_row_argmax_request, xn_lp_laplacian_request,
    xn_lp_col_degree_request, xn_lp_laplacian_deg_request, xn_lp_alpha_check, xn_lp_static_weight,
    xn_lp_labels_request, xn_lp_clamp_request, xn_lp_mode, xn_lp_iterate_knn_request, xn_lp_iterate_request,
    xn_lp_iterations, xn_lp_dense_proba_request,
)

comptime NEIGHBORS_SEMI_RUN_FIXTURES = 10


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


def fixture(unit: Int) raises:
    if unit == 0:
        xn_achi2_negative_check(False)
        try:
            xn_achi2_negative_check(True)
            raise Error("accepted")
        except e:
            require(String(e) == "Negative values in data passed to AdditiveChi2Sampler", "achi2 wording")
        require(xn_achi2_fit_negative(0, 4, 1.0) and not xn_achi2_fit_negative(-1, 4, -1.0),
                "the device scan's answer wins")
        require(xn_achi2_fit_negative(-2, 4, -0.5) and not xn_achi2_fit_negative(-2, 4, 0.0),
                "the host minimum without the scan")
        require(not xn_achi2_fit_negative(-2, 0, -1.0), "an empty X has no negative value")
        require(xn_achi2_width(3, 2) == 9, "achi2 width")
    elif unit == 1:
        require(same(xn_achi2_request(1, 2, 3, 4, 2, 0.1), "achi2", [1, 2], [3, 4, 2], [Float64(Float32(0.1))]),
                "achi2 request")
        var dense: List[Float32] = [0.0, 1.0, -0.0, 2.0, 0.0, 3.0]
        var nnz = xn_achi2_csr_count(Int(dense.unsafe_ptr()), 6)
        require(nnz == 3, "achi2 csr count (-0.0 dropped)")
        var data = List[Float32](length=3, fill=0.0)
        var indices = List[Int32](length=3, fill=-1)
        var indptr = List[Int32](length=3, fill=-1)
        xn_achi2_csr_fill(Int(dense.unsafe_ptr()), 2, 3, Int(data.unsafe_ptr()), Int(indices.unsafe_ptr()),
                          Int(indptr.unsafe_ptr()))
        require(indptr[0] == 0 and indptr[1] == 1 and indptr[2] == 3, "achi2 csr indptr")
        require(indices[0] == 1 and indices[1] == 0 and indices[2] == 2, "achi2 csr indices")
        require(data[0] == 1.0 and data[1] == 2.0 and data[2] == 3.0, "achi2 csr data")
        _ = dense
        _ = data
        _ = indices
        _ = indptr
    elif unit == 2:
        xn_schi2_skew_check(-0.5, 1.0)
        try:
            xn_schi2_skew_check(-1.0, 1.0)
            raise Error("accepted")
        except e:
            require(String(e) == "X may not contain entries smaller than -skewedness.", "skew wording")
        try:
            xn_schi2_missing_fit(0, 3)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn SkewedChi2Sampler: X needs at least one feature and n_components >= 1",
                    "schi2 shape wording")
        try:
            xn_schi2_missing_fit(2, 3)
            raise Error("accepted")
        except e:
            require(String(e) == "NotImplementedError: mojolearn SkewedChi2Sampler: this binary lacks "
                    + "x_neighbors_kfeat_schi2_fit_idn (rebuild without -D MOJOLEARN_IDN_XN_SKETCH_CTR_OFF)",
                    "schi2 missing entry wording")
    elif unit == 3:
        require(same(xn_schi2_log_request(1, 2, 6, 0.1), "unary", [1, 2], [6, XN_SEMI_U_LOG],
                     [1.0, Float64(Float32(0.1))]), "schi2 log request")
        require(same(xn_schi2_transform_request(1, 2, 3, 4, 5, 6, 7), "skew_transform", [1, 2, 3, 4], [5, 6, 7],
                     List[Float64]()), "schi2 transform request")
    elif unit == 4:
        require(xn_semi_knn_k(7, 5) == 5 and xn_semi_knn_k(3, 5) == 3, "knn k")
        require(same(xn_semi_sqdist_request(1, 2, 3, 4, 5, 6, 7), "sqdist", [1, 2, 3], [4, 5, 6, 7], List[Float64]()),
                "sqdist request")
        require(same(xn_semi_knn_select_request(1, 2, 3, 4, 5, 6), "knn_select", [1, 2, 3], [4, 5, 6, 0],
                     List[Float64]()), "knn_select request")
        require(same(xn_semi_knn_sq_request(False, 1, 2, 3, 4, 5, 6, 7, 8, 9), "knn_sq_tiled", [1, 2, 3, 4],
                     [5, 6, 7, 8, 0, 9], List[Float64]()), "fused knn request")
        require(xn_semi_knn_sq_request(True, 1, 2, 3, 4, 5, 6, 7, 8, 9).name == "knn_sq", "old-items knn request")
    elif unit == 5:
        require(same(xn_lp_knn_graph_request(1, 2, 3, 4, 5, 6, 2), "lp_knn_graph", [1, 2, 3], [4, 5, 6, 2],
                     List[Float64]()), "knn graph request")
        require(same(xn_lp_knn_product_request(1, 2, 3, 4, 5, 6, 7, 8), "lp_knn_product", [1, 2, 3, 4],
                     [5, 6, 7, 8], List[Float64]()), "knn product request")
        require(same(xn_semi_matmul_request(1, 2, 3, 4, 5, 6), "matmul", [1, 2, 3], [4, 5, 6], List[Float64]()),
                "matmul request")
        require(same(xn_semi_rbf_request(1, 2, 3, 4, 5, 6, 2, 9, 0.1), "kernel", [1, 2, 3], [4, 5, 6, 2, 0, 9],
                     [Float64(Float32(0.1)), 0.0]), "rbf request")
    elif unit == 6:
        require(same(xn_semi_row_normalize_request(1, 2, 3, 4), "row_normalize", [1, 2], [3, 4], List[Float64]()),
                "row_normalize request")
        require(same(xn_semi_row_argmax_request(1, 2, 3, 4), "row_argmax", [1, 2], [3, 4], List[Float64]()),
                "row_argmax request")
        require(same(xn_lp_laplacian_request(1, 2, 3), "ls_laplacian", [1, 2], [3], List[Float64]()),
                "laplacian request")
        require(same(xn_lp_col_degree_request(1, 2, 3), "col_degree", [1, 2], [3], List[Float64]()),
                "col_degree request")
        require(same(xn_lp_laplacian_deg_request(1, 2, 3, 4), "ls_laplacian_deg", [1, 2, 3], [4], List[Float64]()),
                "laplacian_deg request")
    elif unit == 7:
        xn_lp_alpha_check(0.2)
        for bad in range(2):
            try:
                xn_lp_alpha_check(Float64(bad))
                raise Error("accepted")
            except e:
                require(String(e) == "alpha must be in (0, 1)", "alpha wording")
        require(xn_lp_static_weight(0.2) == Float64(Float32(0.8)), "static weight float32(1 - alpha)")
        require(same(xn_lp_labels_request(1, 2, 3, 4, 5, 6, 7, 0.5), "p2m_lp_labels", [1, 2, 3, 4], [5, 6, 7],
                     [0.5]), "labels request")
        require(same(xn_lp_clamp_request(1, 2, 3, 4, 5), "lp_clamp", [1, 1, 2, 3], [4, 5], List[Float64]()),
                "clamp request: ld as both inputs")
        require(xn_lp_mode(False, False) == 0 and xn_lp_mode(True, False) == 1 and xn_lp_mode(True, True) == 3,
                "iterate mode")
    elif unit == 8:
        var knn = xn_lp_iterate_knn_request(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, True, 1.0, 0.25)
        require(same(knn, "lp_iterate_knn", [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 1, 0x3FF00000, 0], [0.25]),
                "iterate knn request")
        var dense = xn_lp_iterate_request(1, 2, 3, 4, 5, 6, 7, 8, 2, 1.0, 0.0)
        require(same(dense, "lp_iterate", [1, 2, 3, 4, 5], [6, 7, 8, 2, 0x3FF00000, 0], [0.0]), "iterate request")
        var info: List[Int32] = [17, 0]
        require(xn_lp_iterations(Int(info.unsafe_ptr())) == 17, "iterations")
        _ = info
    else:
        require(same(xn_lp_dense_proba_request(1, 2, 3, 4, 5, 6, 7, 8, 9, 2, True, 11, 0.1), "lp_dense_proba",
                     [1, 2, 3, 4, 5], [6, 7, 8, 9, 2, 1, 11], [Float64(Float32(0.1)), 0.0]),
                "dense proba request")
        require(xn_lp_dense_proba_request(1, 2, 3, 4, 5, 6, 7, 8, 9, 2, False, 11, 0.1).p[5] == 0,
                "proba wants no codes")


def run_checks() raises:
    var failures = List[Int](length=NEIGHBORS_SEMI_RUN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL neighbors_semi_run fixture", unit, String(error))
    host_parallelize(task, NEIGHBORS_SEMI_RUN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    require(failed == 0, "neighbors_semi_run failed " + String(failed) + " fixtures")
    print("PASS neighbors_semi_run:", NEIGHBORS_SEMI_RUN_FIXTURES, "typed check, request and decode fixtures")
