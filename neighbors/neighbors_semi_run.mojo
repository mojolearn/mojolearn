# SPDX-License-Identifier: Apache-2.0
"""LabelPropagation / LabelSpreading and AdditiveChi2Sampler /
SkewedChi2Sampler door RUN: typed checks, the x_neighbors operation requests
and the decoding of their info words.

`bindings/neighbors_semi_native_door.mojo` is the UNPACK/PACK side: it reads the
estimator's fields and the inputs, allocates the public arrays, calls these
checks at the point the public method made them (so the refusal order and
wording are unchanged) and hands each request to the bound x_neighbors
binary. Nothing here imports Python.
"""
from core.p3nc_scan import p3nc_dense_nnz_f32, p3nc_dense_to_csr_f32
from neighbors.neighbors_graph_run import XnOpRequest, xn_f32_word, xn_i32_word, xn_double_words

#: `_expansion_neighbors._U_LOG`, the unary op code of `log(a * x + b)`.
comptime XN_SEMI_U_LOG = 1
comptime _SEMI_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _SEMI_I32P = MutPointer[Int32, MutUntrackedOrigin]


# ---------------------------------------------------------------- AdditiveChi2Sampler


def xn_achi2_negative_check(negative: Bool) raises:
    if negative:
        raise Error("Negative values in data passed to AdditiveChi2Sampler")


def xn_achi2_fit_negative(first: Int, size: Int, lo: Float64) -> Bool:
    """Whether X holds a negative value: the device scan's first negative
    index (`first`; -2 when the scan did not run), else the host minimum
    `lo` of a non-empty X."""
    if first != -2:
        return first >= 0
    return size != 0 and lo < 0.0


def xn_achi2_width(d: Int, steps: Int) -> Int:
    """The map's width: d * (2 * sample_steps - 1)."""
    return d * (2 * steps - 1)


def xn_achi2_request(x: Int, dst: Int, n: Int, d: Int, steps: Int, interval: Float64) -> XnOpRequest:
    return XnOpRequest("achi2", [x, dst], [n, d, steps], [xn_f32_word(interval)])


def xn_achi2_csr_count(dense: Int, count: Int) -> Int:
    """The entries SciPy's dense-to-CSR conversion keeps (NaN counts, -0.0
    does not)."""
    return p3nc_dense_nnz_f32(_SEMI_F32P(unsafe_from_address=dense), count)


def xn_achi2_csr_fill(dense: Int, rows: Int, cols: Int, data: Int, indices: Int, indptr: Int):
    """The CSR arrays of the dense map, row-major, columns ascending."""
    p3nc_dense_to_csr_f32(_SEMI_F32P(unsafe_from_address=dense), rows, cols, _SEMI_F32P(unsafe_from_address=data),
                          _SEMI_I32P(unsafe_from_address=indices), _SEMI_I32P(unsafe_from_address=indptr))


# ---------------------------------------------------------------- SkewedChi2Sampler


def xn_schi2_skew_check(lo: Float64, skewedness: Float64) raises:
    if lo <= -skewedness:
        raise Error("X may not contain entries smaller than -skewedness.")


def xn_schi2_missing_fit(d: Int, nc: Int) raises:
    """The refusal when no fit entry serves: a shape the draw cannot take,
    else the binary lacks the counter-based draw."""
    if d < 1 or nc < 1:
        raise Error("mojolearn SkewedChi2Sampler: X needs at least one feature and n_components >= 1")
    raise Error("NotImplementedError: mojolearn SkewedChi2Sampler: this binary lacks x_neighbors_kfeat_schi2_fit_idn "
                + "(rebuild without -D MOJOLEARN_IDN_XN_SKETCH_CTR_OFF)")


def xn_schi2_log_request(x: Int, dst: Int, size: Int, skewedness: Float64) -> XnOpRequest:
    """`_unary(X, _U_LOG, 1.0, skewedness)`."""
    return XnOpRequest("unary", [x, dst], [size, XN_SEMI_U_LOG], [1.0, xn_f32_word(skewedness)])


def xn_schi2_transform_request(lx: Int, w: Int, offset: Int, dst: Int, n: Int, d: Int, nc: Int) -> XnOpRequest:
    return XnOpRequest("skew_transform", [lx, w, offset, dst], [n, d, nc], List[Float64]())


# ---------------------------------------------------------------- label propagation ops


def xn_semi_knn_k(n_neighbors: Int, n: Int) -> Int:
    return min(n_neighbors, n)


def xn_semi_sqdist_request(q: Int, r: Int, dmat: Int, n: Int, m: Int, d: Int, status: Int) -> XnOpRequest:
    return XnOpRequest("sqdist", [q, r, dmat], [n, m, d, status], List[Float64]())


def xn_semi_knn_select_request(dmat: Int, dist: Int, idx: Int, n: Int, m: Int, k: Int) -> XnOpRequest:
    return XnOpRequest("knn_select", [dmat, dist, idx], [n, m, k, 0], List[Float64]())


def xn_semi_knn_sq_request(old_items: Bool, q: Int, r: Int, dist: Int, idx: Int, n: Int, m: Int, d: Int, k: Int,
                           status: Int) -> XnOpRequest:
    """The fused `knn_sq_tiled` (`knn_sq` under MOJOLEARN_XN_OLD_ITEMS)."""
    var name = String("knn_sq") if old_items else String("knn_sq_tiled")
    return XnOpRequest(name, [q, r, dist, idx], [n, m, d, k, 0, status], List[Float64]())


def xn_lp_knn_graph_request(idx: Int, cols: Int, vals: Int, n: Int, n_reference: Int, k: Int,
                            variant: Int) -> XnOpRequest:
    return XnOpRequest("lp_knn_graph", [idx, cols, vals], [n, n_reference, k, variant], List[Float64]())


def xn_lp_knn_product_request(cols: Int, vals: Int, labels: Int, dst: Int, n: Int, m: Int, k: Int,
                              c: Int) -> XnOpRequest:
    return XnOpRequest("lp_knn_product", [cols, vals, labels, dst], [n, m, k, c], List[Float64]())


def xn_semi_matmul_request(a: Int, b: Int, dst: Int, n: Int, k: Int, m: Int) -> XnOpRequest:
    return XnOpRequest("matmul", [a, b, dst], [n, k, m], List[Float64]())


def xn_semi_rbf_request(a: Int, b: Int, dst: Int, n: Int, m: Int, d: Int, kind: Int, status: Int,
                        gamma: Float64) -> XnOpRequest:
    """`_XNeighbors._kernel(A, B, "rbf", gamma, 0.0, 0)` with A's finite
    status slot (0 not asked)."""
    return XnOpRequest("kernel", [a, b, dst], [n, m, d, kind, 0, status], [xn_f32_word(gamma), xn_f32_word(0.0)])


def xn_semi_row_normalize_request(p: Int, dst: Int, rows: Int, cols: Int) -> XnOpRequest:
    return XnOpRequest("row_normalize", [p, dst], [rows, cols], List[Float64]())


def xn_semi_row_argmax_request(m: Int, codes: Int, n: Int, c: Int) -> XnOpRequest:
    return XnOpRequest("row_argmax", [m, codes], [n, c], List[Float64]())


def xn_lp_laplacian_request(a: Int, g: Int, n: Int) -> XnOpRequest:
    return XnOpRequest("ls_laplacian", [a, g], [n], List[Float64]())


def xn_lp_col_degree_request(a: Int, deg: Int, n: Int) -> XnOpRequest:
    return XnOpRequest("col_degree", [a, deg], [n], List[Float64]())


def xn_lp_laplacian_deg_request(a: Int, deg: Int, g: Int, n: Int) -> XnOpRequest:
    return XnOpRequest("ls_laplacian_deg", [a, deg, g], [n], List[Float64]())


# ---------------------------------------------------------------- label propagation fit


def xn_lp_alpha_check(alpha: Float64) raises:
    if not (0.0 < alpha and alpha < 1.0):
        raise Error("alpha must be in (0, 1)")


def xn_lp_static_weight(alpha: Float64) -> Float64:
    """LabelSpreading's clamp weight, float32(1 - alpha); propagation
    passes 1.0 and never reads alpha."""
    return xn_f32_word(1.0 - alpha)


def xn_lp_labels_request(codes: Int, ld: Int, ys: Int, unlabeled: Int, n: Int, nc: Int, marker: Int,
                         weight: Float64) -> XnOpRequest:
    return XnOpRequest("p2m_lp_labels", [codes, ld, ys, unlabeled], [n, nc, marker], [weight])


def xn_lp_clamp_request(ld: Int, unlabeled: Int, ystatic: Int, n: Int, nc: Int) -> XnOpRequest:
    """Propagation's static rows: `lp_clamp` with ld as both inputs."""
    return XnOpRequest("lp_clamp", [ld, ld, unlabeled, ystatic], [n, nc], List[Float64]())


def xn_lp_mode(spreading: Bool, raw: Bool) -> Int:
    return (1 if spreading else 0) + (2 if raw else 0)


def xn_lp_iterate_knn_request(cols: Int, vals: Int, ld: Int, ystatic: Int, unlabeled: Int, info: Int, n: Int,
                              k: Int, nc: Int, max_iter: Int, spreading: Bool, tol: Float64,
                              alpha: Float64) -> XnOpRequest:
    """The kNN-graph loop as one resident op; `alpha` is the float32 word
    (0.0 for propagation)."""
    var words = xn_double_words(tol)
    return XnOpRequest("lp_iterate_knn", [cols, vals, ld, ystatic, unlabeled, info],
        [n, k, nc, max_iter, 1 if spreading else 0, words[0], words[1]], [alpha])


def xn_lp_iterate_request(dense: Int, ld: Int, ystatic: Int, unlabeled: Int, info: Int, n: Int, nc: Int,
                          max_iter: Int, mode: Int, tol: Float64, alpha: Float64) -> XnOpRequest:
    """The dense loop as one resident op."""
    var words = xn_double_words(tol)
    return XnOpRequest("lp_iterate", [dense, ld, ystatic, unlabeled, info],
        [n, nc, max_iter, mode, words[0], words[1]], [alpha])


def xn_lp_iterations(info: Int) -> Int:
    return xn_i32_word(info, 0)


def xn_lp_dense_proba_request(q: Int, xfit: Int, dist: Int, proba: Int, codes: Int, nq: Int, n: Int, d: Int,
                              c: Int, kind: Int, want_codes: Bool, status: Int, gamma: Float64) -> XnOpRequest:
    """kernel='rbf' predict_proba or predict codes through `lp_dense_proba`:
    the same kind, gamma and coef0 words as the rbf kernel op."""
    return XnOpRequest("lp_dense_proba", [q, xfit, dist, proba, codes],
        [nq, n, d, c, kind, 1 if want_codes else 0, status], [xn_f32_word(gamma), xn_f32_word(0.0)])
