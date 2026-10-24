# SPDX-License-Identifier: Apache-2.0
"""Frozen b32103c00 Python paths of LabelPropagation / LabelSpreading and the
chi2 samplers' public methods, used only as an independent test oracle for
their one-call native doors (bindings/neighbors_semi_native.mojo). Each
function takes the estimator in place of `self`; the primitives
(`_op`, `_knn_sq`, `_kernel`, `_matmul`, `_unary`) are the package's own."""
import struct

from mojolearn._buffer import addr, addr_ro, empty
from mojolearn._lazy_out import _empty_out
from mojolearn import _expansion_neighbors as xn
from mojolearn._expansion_neighbors import (
    _f32, _kfeat_flags, _seed_word, _f32_scalar, _lp_fast_resident,
    _classical_graph_normalization, _argmax_labels, _accuracy, _U_LOG,
)


# ------------------------------------------------------------ AdditiveChi2Sampler

def achi2_interval(est):
    if est.sample_interval is not None:
        return float(est.sample_interval)
    table = {1: 0.8, 2: 0.5, 3: 0.4}
    if est.sample_steps not in table:
        raise ValueError("If sample_steps is not in [1, 2, 3], you need to provide sample_interval")
    return table[est.sample_steps]


def achi2_fit(est, X):
    X = _f32(X)
    first = -2
    if X.size and _kfeat_flags(est) & 1:
        first = int(est._bind().x_neighbors_kfeat_first_negative(addr_ro(X, name="X"), X.size))
    if first != -2:
        neg = first >= 0
    else:
        neg = X.size and X.min() < 0
    if neg:
        raise ValueError("Negative values in data passed to AdditiveChi2Sampler")
    achi2_interval(est)
    est.n_features_in_ = X.shape[1]
    return est


def achi2_transform(est, X):
    sparse = X if (hasattr(X, "toarray") and hasattr(X, "nnz")) else None
    X = _f32(X)
    n, d = X.shape
    steps = int(est.sample_steps)
    out = _empty_out((n, d * (2 * steps - 1)), "<f4")
    if X.size and X.min() < 0:
        raise ValueError("Negative values in data passed to AdditiveChi2Sampler")
    est._op("achi2", [(X, 0), (out, 1)], (n, d, steps), (_f32_scalar(achi2_interval(est)),))
    if sparse is not None:
        import scipy.sparse as sp
        m = sp.csr_matrix(out.tolist(), dtype="float32")
        m.eliminate_zeros()
        return m
    return out


def achi2_fit_transform(est, X):
    return achi2_transform(achi2_fit(est, X), X)


# ------------------------------------------------------------ SkewedChi2Sampler

def schi2_fit(est, X):
    X = _f32(X)
    d = X.shape[1]
    nc = int(est.n_components)
    seed = _seed_word(est.random_state)
    flags = (_kfeat_flags(est) if isinstance(seed, int) and not isinstance(seed, bool)
             and d > 0 and nc > 0 else 0)
    est.__dict__.pop("_schi2_z", None)
    fit_idn = (getattr(est._bind(), "x_neighbors_kfeat_schi2_fit_idn", None)
               if isinstance(seed, int) and not isinstance(seed, bool) and d > 0 and nc > 0 else None)
    if fit_idn is not None:
        w = _empty_out((d, nc), "<f4")
        off = empty((nc,), "<f4")
        fit_idn([seed & 0xFFFFFFFF, d, nc], addr(w, name="random_weights_"), addr(off, name="random_offset_"))
        est.random_weights_ = w
        est.random_offset_ = off
        est.n_features_in_ = d
        return est
    if flags & 2:
        w = _empty_out((d, nc), "<f4")
        off = empty((nc,), "<f4")
        est._bind().x_neighbors_kfeat_schi2_fit([seed & 0xFFFFFFFF, d, nc], addr(w, name="random_weights_"),
                                                addr(off, name="random_offset_"))
        est.random_weights_ = w
        est.random_offset_ = off
        est.n_features_in_ = d
        return est
    if d < 1 or nc < 1:
        raise ValueError("mojolearn SkewedChi2Sampler: X needs at least one feature and n_components >= 1")
    raise NotImplementedError(
        "mojolearn SkewedChi2Sampler: this binary lacks x_neighbors_kfeat_schi2_fit_idn "
        "(rebuild without -D MOJOLEARN_IDN_XN_SKETCH_CTR_OFF)")


def schi2_transform(est, X):
    X = _f32(X)
    n, d = X.shape
    if X.size and X.min() <= -float(est.skewedness):
        raise ValueError("X may not contain entries smaller than -skewedness.")
    nc = int(est.n_components)
    z = est.__dict__.get("_schi2_z")
    if (z is not None or _kfeat_flags(est) & 4) and d == est.__dict__.get("n_features_in_"):
        w = _empty_out((d, nc), "<f4") if z is not None else est.random_weights_
        out = _empty_out((n, nc), "<f4")
        est._bind().x_neighbors_kfeat_schi2_transform(
            [n, d, nc, 0 if z is None else 1],
            [addr_ro(X, name="X"), addr_ro(w if z is None else z, name="random_weights_"),
             0 if z is None else addr(w, name="random_weights_"),
             addr_ro(est.random_offset_, name="random_offset_"), addr(out, name="schi2 output")],
            [_f32_scalar(est.skewedness)])
        if z is not None:
            est.random_weights_ = w
        return out
    lx = est._unary(X, _U_LOG, 1.0, _f32_scalar(est.skewedness))
    out = _empty_out((n, nc), "<f4")
    est._op("skew_transform", [(lx, 0), (est.random_weights_, 0), (est.random_offset_, 0), (out, 1)], (n, d, nc))
    return out


def schi2_fit_transform(est, X):
    return schi2_transform(schi2_fit(est, X), X)


# ------------------------------------------------------------ label propagation

def _compact_graph(est, idx, n_reference, variant):
    n, k = idx.shape
    cols = empty((n, k), "<i4")
    vals = empty((n, k), "<f4")
    est._op("lp_knn_graph", [(idx, 0), (cols, 1), (vals, 1)], (n, n_reference, k, variant))
    return cols, vals, n_reference


def _graph_product(est, G, labels):
    if not isinstance(G, tuple):
        return est._matmul(G, labels)
    cols, vals, m = G
    n, k = cols.shape
    c = labels.shape[1]
    out = _empty_out((n, c), "<f4")
    est._op("lp_knn_product", [(cols, 0), (vals, 0), (labels, 0), (out, 1)], (n, m, k, c))
    return out


def _graph_affinity(est, X):
    n = X.shape[0]
    if est.kernel == "rbf":
        return est._kernel(X, X, "rbf", est.gamma, 0.0, 0)
    if est.kernel == "knn":
        k = min(int(est.n_neighbors), n)
        _, idx = est._knn_sq(X, X, k, False)
        g = _empty_out((n, n), "<f4")
        est._op("knn_graph", [(idx, 0), (g, 1)], (n, n, k))
        return g
    raise NotImplementedError(f"{type(est).__name__}: kernel={est.kernel!r} is not implemented ('rbf' or 'knn')")


def _build_graph(est, X):
    A = _graph_affinity(est, X)
    est._classical_raw_graph = _classical_graph_normalization(est)
    if est._classical_raw_graph:
        return A
    if est._variant == "propagation":
        G = empty(A.shape, "<f4")
        est._op("row_normalize", [(A, 0), (G, 1)], A.shape)
        return G
    n = A.shape[0]
    G = _empty_out((n, n), "<f4")
    if xn._OLD_ITEMS:
        est._op("ls_laplacian", [(A, 0), (G, 1)], (n,))
    else:
        deg = _empty_out((n,), "<f4")
        est._op("col_degree", [(A, 0), (deg, 1)], (n,))
        est._op("ls_laplacian_deg", [(A, 0), (deg, 0), (G, 1)], (n,))
    return G


def _finish_fit(est, X, classes, ld, n, C, n_iter):
    final = _empty_out((n, C), "<f4")
    est._op("row_normalize", [(ld, 0), (final, 1)], (n, C))
    est.X_ = X
    est.classes_ = classes
    est.label_distributions_ = final
    est.n_iter_ = n_iter
    est.transduction_ = _argmax_labels(est, classes, final)
    est.n_features_in_ = X.shape[1]
    return est


def lp_fit(est, X, y):
    if est._variant != "propagation" and not (0.0 < float(est.alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    X = _f32(X)
    n = X.shape[0]
    from mojolearn._labels import encode_labels
    allc, codes = encode_labels(y)
    if codes.size != n:
        raise ValueError("X and y have different numbers of rows")
    skip = next((i for i, c in enumerate(allc) if c == -1 and not isinstance(c, str)), -1)
    classes = [c for i, c in enumerate(allc) if i != skip]
    C = len(classes)
    ld = _empty_out((n, C), "<f4")
    ys = _empty_out((n, C), "<f4")
    unlabeled = empty((n,), "<i4")
    a = _f32_scalar(1.0 - float(est.alpha)) if est._variant != "propagation" else 1.0
    if n and C:
        est._op("p2m_lp_labels", [(codes, 0), (ld, 1), (ys, 1), (unlabeled, 1)],
                (n, C, skip if skip >= 0 else len(allc)), (a,))
    if est._variant == "propagation":
        ys = None
    if est.kernel == "knn":
        k = min(int(est.n_neighbors), n)
        _, idx = est._knn_sq(X, X, k, False)
        G = _compact_graph(est, idx, n, 0 if est._variant == "propagation" else 1)
    else:
        G = _build_graph(est, X)
    if ys is None:
        ystatic = _empty_out((n, C), "<f4")
        est._op("lp_clamp", [(ld, 0), (ld, 0), (unlabeled, 0), (ystatic, 1)], (n, C))
    else:
        ystatic = ys
    if isinstance(G, tuple) and _lp_fast_resident(est):
        cols, vals, _ = G
        info = empty((2,), "<i4")
        tol_bits = struct.unpack("<Q", struct.pack("<d", float(est.tol)))[0]
        est._op("lp_iterate_knn", [(cols, 0), (vals, 0), (ld, 1), (ystatic, 0), (unlabeled, 0), (info, 1)],
                (n, cols.shape[1], C, int(est.max_iter), 0 if est._variant == "propagation" else 1,
                 tol_bits >> 32, tol_bits & 0xFFFFFFFF),
                (_f32_scalar(est.alpha) if est._variant != "propagation" else 0.0,))
        return _finish_fit(est, X, classes, ld, n, C, int(info.tolist()[0]))
    if not isinstance(G, tuple):
        info = empty((2,), "<i4")
        tol_bits = struct.unpack("<Q", struct.pack("<d", float(est.tol)))[0]
        est._op("lp_iterate", [(G, 0), (ld, 1), (ystatic, 0), (unlabeled, 0), (info, 1)],
                (n, C, int(est.max_iter),
                 (0 if est._variant == "propagation" else 1) +
                 (2 if getattr(est, "_classical_raw_graph", False) else 0),
                 tol_bits >> 32, tol_bits & 0xFFFFFFFF),
                (_f32_scalar(est.alpha) if est._variant != "propagation" else 0.0,))
        return _finish_fit(est, X, classes, ld, n, C, int(info.tolist()[0]))
    prev = _empty_out((n, C), "<f4")
    s = _empty_out((1,), "<f4")
    n_iter = 0
    converged = False
    for it in range(int(est.max_iter)):
        n_iter = it
        est._op("absdiff_sum", [(ld, 0), (prev, 0), (s, 1)], (n * C,))
        if s.tolist()[0] < float(est.tol):
            converged = True
            break
        prev = ld
        nxt = _graph_product(est, G, ld)
        out = _empty_out((n, C), "<f4")
        if est._variant == "propagation":
            est._op("lp_clamp", [(nxt, 0), (ystatic, 0), (unlabeled, 0), (out, 1)], (n, C))
        else:
            est._op("ls_clamp", [(nxt, 0), (ystatic, 0), (out, 1)], (n * C,), (_f32_scalar(est.alpha),))
        ld = out
    if not converged:
        n_iter += 1
    return _finish_fit(est, X, classes, ld, n, C, n_iter)


def lp_predict_proba(est, X):
    Q = _f32(X)
    n = est.X_.shape[0]
    if est.kernel == "knn":
        k = min(int(est.n_neighbors), n)
        _, idx = est._knn_sq(Q, est.X_, k, False)
        W = _compact_graph(est, idx, n, 2)
    else:
        W = est._kernel(Q, est.X_, "rbf", est.gamma, 0.0, 0)
    P = _graph_product(est, W, est.label_distributions_)
    out = empty(P.shape, "<f4")
    est._op("row_normalize", [(P, 0), (out, 1)], P.shape)
    return out


def lp_predict(est, X):
    return _argmax_labels(est, est.classes_, lp_predict_proba(est, X))


def lp_score(est, X, y, sample_weight=None):
    return _accuracy(y, lp_predict(est, X), sample_weight)
