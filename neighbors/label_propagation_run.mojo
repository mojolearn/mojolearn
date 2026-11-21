# SPDX-License-Identifier: Apache-2.0
"""LabelPropagation / LabelSpreading RUN: typed requests in, typed results
out.

The boundary (bindings/neighbors_semi_native_door.mojo) admits the public
arguments into these requests, releases the GIL once for each function here
and publishes the results. Nothing in this module imports Python. Every
address in a request is admitted storage the boundary keeps alive through
the call; every intermediate (the label matrices, the graph, the kernel
matrix, the neighbor indices) is native scratch of this module, never a
public array. The operations, their words and their order are the public
method's (the requests are neighbors/neighbors_semi_run.mojo's).
"""
from core.neighbors_native_driver import XnDriver
from neighbors.xn_request_run import xn_request_run, xn_scratch_f32, xn_scratch_i32
from neighbors.neighbors_semi_run import (
    xn_semi_sqdist_request, xn_semi_knn_select_request, xn_semi_knn_sq_request, xn_lp_knn_graph_request,
    xn_lp_knn_product_request, xn_semi_matmul_request, xn_semi_rbf_request, xn_semi_row_normalize_request,
    xn_semi_row_argmax_request, xn_lp_laplacian_request, xn_lp_col_degree_request, xn_lp_laplacian_deg_request,
    xn_lp_labels_request, xn_lp_clamp_request, xn_lp_mode, xn_lp_iterate_knn_request, xn_lp_iterate_request,
    xn_lp_iterations, xn_lp_dense_proba_request,
)


def _f32_at(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def _i32_at(values: List[Int32]) -> Int:
    return Int(values.unsafe_ptr())


def lp_knn_indices_run(driver: XnDriver, q: Int, r: Int, n: Int, m: Int, d: Int, k: Int, unfused: Bool,
                       old_items: Bool, status: Int, idx: Int) raises:
    """`_XNeighbors._knn_sq(Q, R, k, False)`'s indices into `idx` (n x k
    int32): the fused `knn_sq_tiled` (`knn_sq` under MOJOLEARN_XN_OLD_ITEMS),
    or `sqdist` then `knn_select` under MOJOLEARN_XN_UNFUSED_KNN. `status`:
    the address of Q's finite status word, answered by the first operation."""
    var dist = xn_scratch_f32(n * k)
    if unfused:
        var dmat = xn_scratch_f32(n * m)
        _ = xn_request_run(driver, xn_semi_sqdist_request(q, r, _f32_at(dmat), n, m, d, status))
        _ = xn_request_run(driver, xn_semi_knn_select_request(_f32_at(dmat), _f32_at(dist), idx, n, m, k))
        _ = dmat^
    else:
        _ = xn_request_run(driver, xn_semi_knn_sq_request(old_items, q, r, _f32_at(dist), idx, n, m, d, k, status))
    _ = dist^


@fieldwise_init
struct LpFitRequest(Copyable, Movable):
    # `x`: the admitted float32 rows; `codes`: the int32 label codes over
    # every label seen, `marker` the code of the unlabeled marker (the class
    # count when no row is unlabeled). `final` (rows x classes float32) and
    # `labels` (rows int32) receive `label_distributions_` and the
    # transduction's class codes. `knn`: the compact kNN graph (`neighbors`
    # already limited to the row count), else the RBF kernel (`kind`,
    # `gamma`); `raw`: the image keeps the raw affinity (the
    # classical-normalization switch). `weight`: the clamp weight of the
    # label step; `alpha`: the float32 word of the loop (0.0 for
    # propagation). `unfused` / `old_items`: the import-time arms
    # MOJOLEARN_XN_UNFUSED_KNN and MOJOLEARN_XN_OLD_ITEMS.
    var x: Int
    var codes: Int
    var final: Int
    var labels: Int
    var rows: Int
    var features: Int
    var classes: Int
    var marker: Int
    var spreading: Bool
    var knn: Bool
    var neighbors: Int
    var kind: Int
    var gamma: Float64
    var raw: Bool
    var weight: Float64
    var alpha: Float64
    var tol: Float64
    var max_iter: Int
    var unfused: Bool
    var old_items: Bool


struct LpFitOutcome(Copyable, Movable):
    # `status`: X's deferred finite word from the graph's first operation
    # (0 = every word finite; otherwise nothing after it ran). `iterations`:
    # the fitted `n_iter_`.
    var status: Int
    var iterations: Int

    def __init__(out self):
        self.status = 0
        self.iterations = 0


def lp_fit_run(driver: XnDriver, request: LpFitRequest) raises -> LpFitOutcome:
    """The label matrices, the graph, the resident loop, the final row
    normalization and the transduction codes."""
    var outcome = LpFitOutcome()
    var n = request.rows
    var nc = request.classes
    var ld = xn_scratch_f32(n * nc)
    var ys = xn_scratch_f32(n * nc)
    var unlabeled = xn_scratch_i32(n)
    if n != 0 and nc != 0:
        _ = xn_request_run(driver, xn_lp_labels_request(request.codes, _f32_at(ld), _f32_at(ys), _i32_at(unlabeled),
                                                        n, nc, request.marker, request.weight))
    # X's finite test rides behind the upload of the graph's first operation.
    var status = xn_scratch_i32(1)
    var k = request.neighbors
    var cols = xn_scratch_i32(n * k if request.knn else 0)
    var vals = xn_scratch_f32(n * k if request.knn else 0)
    var affinity = xn_scratch_f32(0 if request.knn else n * n)
    var graph = xn_scratch_f32(0 if (request.knn or request.raw) else n * n)
    var dense = _f32_at(affinity)
    if request.knn:
        var idx = xn_scratch_i32(n * k)
        lp_knn_indices_run(driver, request.x, request.x, n, n, request.features, k, request.unfused,
                           request.old_items, _i32_at(status), _i32_at(idx))
        outcome.status = Int(status[0])
        if outcome.status == 0:
            _ = xn_request_run(driver, xn_lp_knn_graph_request(_i32_at(idx), _i32_at(cols), _f32_at(vals), n, n, k,
                                                               1 if request.spreading else 0))
        _ = idx^
    else:
        _ = xn_request_run(driver, xn_semi_rbf_request(request.x, request.x, _f32_at(affinity), n, n,
                                                       request.features, request.kind, _i32_at(status),
                                                       request.gamma))
        outcome.status = Int(status[0])
        if outcome.status == 0 and not request.raw:
            dense = _f32_at(graph)
            if not request.spreading:
                _ = xn_request_run(driver, xn_semi_row_normalize_request(_f32_at(affinity), dense, n, n))
            elif request.old_items:
                _ = xn_request_run(driver, xn_lp_laplacian_request(_f32_at(affinity), dense, n))
            else:
                var deg = xn_scratch_f32(n)
                _ = xn_request_run(driver, xn_lp_col_degree_request(_f32_at(affinity), _f32_at(deg), n))
                _ = xn_request_run(driver, xn_lp_laplacian_deg_request(_f32_at(affinity), _f32_at(deg), dense, n))
                _ = deg^
    if outcome.status == 0:
        # propagation's static rows: `lp_clamp` with ld as both inputs;
        # spreading's are the label step's.
        var clamped = xn_scratch_f32(0 if request.spreading else n * nc)
        var ystatic = _f32_at(ys)
        if not request.spreading:
            ystatic = _f32_at(clamped)
            _ = xn_request_run(driver, xn_lp_clamp_request(_f32_at(ld), _i32_at(unlabeled), ystatic, n, nc))
        # the loop as one resident operation
        var info = xn_scratch_i32(2)
        if request.knn:
            _ = xn_request_run(driver, xn_lp_iterate_knn_request(_i32_at(cols), _f32_at(vals), _f32_at(ld), ystatic,
                                                                 _i32_at(unlabeled), _i32_at(info), n, k, nc,
                                                                 request.max_iter, request.spreading, request.tol,
                                                                 request.alpha))
        else:
            _ = xn_request_run(driver, xn_lp_iterate_request(dense, _f32_at(ld), ystatic, _i32_at(unlabeled),
                                                             _i32_at(info), n, nc, request.max_iter,
                                                             xn_lp_mode(request.spreading, request.raw), request.tol,
                                                             request.alpha))
        outcome.iterations = xn_lp_iterations(_i32_at(info))
        # `_finish_fit`: the final row normalization, then each row's first maximum
        _ = xn_request_run(driver, xn_semi_row_normalize_request(_f32_at(ld), request.final, n, nc))
        if n > 0:
            _ = xn_request_run(driver, xn_semi_row_argmax_request(request.final, request.labels, n, nc))
        _ = clamped^
        _ = info^
    _ = ld^
    _ = ys^
    _ = unlabeled^
    _ = status^
    _ = cols^
    _ = vals^
    _ = affinity^
    _ = graph^
    return outcome


@fieldwise_init
struct LpQueryRequest(Copyable, Movable):
    # `q`: the admitted float32 queries; `fit_x`, `distributions`: the
    # fitted `X_` (fit_rows x features) and `label_distributions_` (fit_rows
    # x classes). `want_labels`: write each row's class code to `labels`
    # (rows int32), else the probabilities to `dst` (rows x classes
    # float32); the other address is 0. `fused`: the bound image carries
    # `lp_dense_proba` (kernel='rbf' only).
    var q: Int
    var fit_x: Int
    var distributions: Int
    var dst: Int
    var labels: Int
    var rows: Int
    var fit_rows: Int
    var features: Int
    var classes: Int
    var knn: Bool
    var neighbors: Int
    var kind: Int
    var gamma: Float64
    var fused: Bool
    var want_labels: Bool
    var unfused: Bool
    var old_items: Bool


def lp_query_fused(request: LpQueryRequest) -> Bool:
    """Whether the one-session operation serves: the RBF kernel, the image
    carries it, and no side is empty."""
    return (request.fused and not request.knn and request.rows != 0 and request.fit_rows != 0
            and request.classes != 0)


def lp_query_run(driver: XnDriver, request: LpQueryRequest) raises -> Int:
    """predict_proba or predict's class codes of new rows. Returns the
    queries' deferred finite word; on a non-zero word nothing after the
    first operation ran."""
    var nq = request.rows
    var n = request.fit_rows
    var c = request.classes
    var status = xn_scratch_i32(1)
    if lp_query_fused(request):
        # The unused output is a one-word placeholder (the descriptor
        # refuses a null address).
        var spare_f = xn_scratch_f32(1)
        var spare_i = xn_scratch_i32(1)
        var proba = _f32_at(spare_f) if request.want_labels else request.dst
        var codes = request.labels if request.want_labels else _i32_at(spare_i)
        _ = xn_request_run(driver, xn_lp_dense_proba_request(request.q, request.fit_x, request.distributions, proba,
                                                             codes, nq, n, request.features, c, request.kind,
                                                             request.want_labels, _i32_at(status), request.gamma))
        var word = Int(status[0])
        _ = spare_f^
        _ = spare_i^
        _ = status^
        return word
    # the operation chain: the weights, their product with the fitted
    # distributions, the row normalization
    var product = xn_scratch_f32(nq * c)
    if request.knn:
        var k = request.neighbors
        var idx = xn_scratch_i32(nq * k)
        lp_knn_indices_run(driver, request.q, request.fit_x, nq, n, request.features, k, request.unfused,
                           request.old_items, _i32_at(status), _i32_at(idx))
        if status[0] == 0:
            var cols = xn_scratch_i32(nq * k)
            var vals = xn_scratch_f32(nq * k)
            _ = xn_request_run(driver, xn_lp_knn_graph_request(_i32_at(idx), _i32_at(cols), _f32_at(vals), nq, n, k,
                                                               2))
            _ = xn_request_run(driver, xn_lp_knn_product_request(_i32_at(cols), _f32_at(vals), request.distributions,
                                                                 _f32_at(product), nq, n, k, c))
            _ = cols^
            _ = vals^
        _ = idx^
    else:
        var weights = xn_scratch_f32(nq * n)
        _ = xn_request_run(driver, xn_semi_rbf_request(request.q, request.fit_x, _f32_at(weights), nq, n,
                                                       request.features, request.kind, _i32_at(status),
                                                       request.gamma))
        if status[0] == 0:
            _ = xn_request_run(driver, xn_semi_matmul_request(_f32_at(weights), request.distributions,
                                                              _f32_at(product), nq, n, c))
        _ = weights^
    var word = Int(status[0])
    if word == 0:
        var normalized = xn_scratch_f32(nq * c if request.want_labels else 0)
        var proba = _f32_at(normalized) if request.want_labels else request.dst
        _ = xn_request_run(driver, xn_semi_row_normalize_request(_f32_at(product), proba, nq, c))
        if request.want_labels and nq > 0:
            _ = xn_request_run(driver, xn_semi_row_argmax_request(proba, request.labels, nq, c))
        _ = normalized^
    _ = product^
    _ = status^
    return word
