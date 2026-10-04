# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE NEIGHBORS LANE'S PUBLIC DOOR.

Owned by the `neighbors` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`; a name
that is already public is refused at import. Put the classes here, or import
them here from the lane's own modules. Two optional hooks route a SAVED model
to the CPU (`mojolearn.host_model`); `_classical_host.py` merges them when it is
first imported, after the package, so both may rely on every module existing:

  CLASSICAL_HOST_BASENAMES     {"_mojolearn_x_neighbors": "_mojolearn_x_neighbors_host"}
  def classical_host_formats() -> {"mojolearn-<format>-1": {"Estimator": HostEstimator}}
                               import `_classical_host` INSIDE it and subclass
                               its `_HostBound`; never import it at module level

HOW THESE ESTIMATORS ARE BUILT. Every floating-point operation runs in Mojo,
in the primitives of `x_neighbors/items.mojo` (one source for the GPU binding
`_mojolearn_x_neighbors` and the CPU host binding `_mojolearn_x_neighbors_host`,
generated from one op table by `x_neighbors/gen.py`). Python here only moves
buffers between those calls, keeps integer bookkeeping (class encodings,
index lists, column orders) and compares values it was handed, all of which
is exact. Where a scalar is derived in Python (a percentile, gamma from a
variance) it is IEEE double arithmetic on float32 inputs, rounded once to
float32, which is the same on every box.
"""
from . import _portable_math as math
import os
import struct

from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, as_i32_c, empty, full, zeros
from ._lazy_out import _empty_out
from ._mode import NumericModeMixin

__all__ = ["LocalOutlierFactor", "NearestCentroid", "OneClassSVM", "KernelPCA", "PolynomialCountSketch",
           "AdditiveChi2Sampler", "SkewedChi2Sampler", "LabelPropagation", "LabelSpreading", "KNNImputer",
           "PageRank", "connected_components", "Louvain", "SVGP"]

# x_neighbors/items.mojo's codes
#: x_neighbors/iter_device.mojo XN_PRECOMPUTED_KIND: `kpca_transform` reads q as the kernel
_XN_PRECOMPUTED_KIND = 100
_KERNELS = {"linear": 0, "poly": 1, "polynomial": 1, "rbf": 2, "sigmoid": 3, "laplacian": 4,
            "cosine": 5, "chi2": 6, "additive_chi2": 7}
_U_EXP, _U_LOG, _U_SQRT, _U_TANH, _U_COS, _U_SIN, _U_IDENTITY, _U_RECIP = range(8)


def _dense(X):
    """A scipy.sparse matrix densified (an exact copy: the implicit entries are
    zeros), anything else unchanged. Every x_neighbors primitive is dense, so a
    sparse input takes the same path, with the same bits, as its dense twin."""
    if hasattr(X, "toarray") and hasattr(X, "nnz") and hasattr(X, "tocsr"):
        return X.toarray()
    return X


def _f32(X, name="X"):
    a, _ = as_f32_c(_dense(X), ndim=2, name=name)
    return a


def _feature_names_in(est, input_features):
    d = est.n_features_in_
    if input_features is None:
        return [f"x{i}" for i in range(d)]
    names = [str(v) for v in input_features]
    if len(names) != d:
        raise ValueError(f"input_features should have length equal to number of features ({d}), got {len(names)}")
    return names


def _names_out(names):
    """sklearn returns an object ndarray of str; an Array holds numbers only,
    so the names come back as a list of str, in the same order."""
    return list(names)


def _prefixed_names(est, count):
    """sklearn's ClassNamePrefixFeaturesOutMixin: `<classname lower><i>`."""
    base = type(est).__name__.lower()
    return _names_out([f"{base}{i}" for i in range(count)])


def _accuracy(y_true, y_pred, sample_weight=None):
    """sklearn's accuracy_score: the (weighted) fraction of exact label
    matches (`_expansion_metrics.accuracy_fraction`: grouped sums in the
    x_metrics binding, no per-row Python)."""
    from ._expansion_metrics import accuracy_fraction
    return accuracy_fraction(y_true, y_pred, sample_weight)


def _f32_1d(x, name):
    a, _ = as_f32_c(x, ndim=1, name=name)
    return a


def _i32(x, name):
    a, _ = as_i32_c(x, ndim=1, name=name)
    return a


def _f32_scalar(v):
    """A Python double rounded once to float32, as a Python float."""
    return Array.from_list([float(v)], "<f4").tolist()[0]


def _kpca_resident(est):
    """lane/apple-fast-kapprox: whether KernelPCA.fit builds, centers and
    solves its kernel matrix on x_decomp's resident kit with one upload of X
    (FAST + Apple, rollback `-D MOJOLEARN_KPCA_RESIDENT_OFF`), read back from the
    x_neighbors binding's compile-time constant (no env read)."""
    fn = getattr(est._bind(), "x_neighbors_kpca_resident", None)
    return fn is not None and int(fn()) != 0


#: the kernels the resident route builds from the kit's sqdist / gemm and
#: elementwise launches (any other kernel takes main's path)
_KPCA_RESIDENT_KERNELS = ("rbf", "linear", "poly", "sigmoid")


def _kpca_resident_center(kit, X, n, kernel, gamma, coef0, degree):
    """lane/apple-fast-kapprox: X (an Array) uploaded once from its own
    buffer; the kernel matrix, sklearn's KernelCenterer (K - rows - cols +
    all) and the two fit vectors on the device. Returns (Kc, cols, all_)
    as resident `_M` matrices, or None when this kit has no resident path
    (the caller then runs main's code)."""
    from ._expansion_decomp import _M, _DevBuf
    if not kit._res():
        return None
    d = X.shape[1]
    Xm = _M._on_device(_DevBuf(kit._raw(), X.size), n, d)
    kit.b.x_decomp_dev_upload(Xm._d.id, addr_ro(X, name="X"), X.size)
    if kernel == "rbf":
        K = kit.ew("exp", kit.ew("scale", kit.sqdist(Xm, Xm), s=-gamma))
    else:
        K = kit.mm(Xm, Xm, tb=True)
        if kernel == "poly":
            K = kit.ew("adds", kit.ew("scale", K, s=gamma), s=coef0)
            K = kit.ew("sq" if degree == 2 else "cube", K)
        elif kernel == "sigmoid":
            K = kit.ew("tanh", kit.ew("adds", kit.ew("scale", K, s=gamma), s=coef0))
    cols = kit.ew("scale", kit.colsum(K), s=1.0 / n)          # K_fit_rows_ (1 x n)
    all_ = kit.ew("scale", kit.rowsum(cols), s=1.0 / n)       # K_fit_all_ (1 x 1)
    colv = _M._on_device(cols._d, n, 1)                        # the same buffer as a column
    Kc = kit.ew("add", kit.ew("sub", kit.ew("sub", K, cols), colv), all_)
    return Kc, cols, all_


def _argmax_labels(est, classes, M):
    """The class of every row's first maximum: the argmax on the device
    (`row_argmax`), the labels by `_class_array`."""
    n, C = M.shape
    codes = empty((n,), "<i4")
    if n > 0:
        est._op("row_argmax", [(M, 0), (codes, 1)], (n, C))
    return _class_array(classes, codes)


def _class_array(classes, codes):
    """classes[codes] for int32 codes (an Array): int or bool classes an
    int64 Array, real classes a float64 Array (the native gather of
    `decode_labels`), other labels a list (`decode_labels`)."""
    from ._labels import decode_labels
    if all(isinstance(v, (bool, int)) for v in classes):
        return decode_labels([int(v) for v in classes], codes)   # glue: the k class values
    if all(isinstance(v, (int, float)) for v in classes):
        return decode_labels([float(v) for v in classes], codes)
    return decode_labels(classes, codes)


#: A/B arm (lane neighbors-apple2): the k-NN primitive as the two ops it
#: fuses, `sqdist` then `knn_select` through an n x m matrix. Same bits.
_UNFUSED_KNN = os.environ.get("MOJOLEARN_XN_UNFUSED_KNN", "") == "1"
#: A/B arm: KNNImputer.transform over every cell instead of the missing ones.
_UNCOMPACT_IMPUTE = os.environ.get("MOJOLEARN_XN_UNCOMPACT_IMPUTE", "") == "1"
#: A/B arm: the one-item forms these replaced (PolynomialCountSketch's
#: per-row `pcs`, LabelSpreading's per-cell-degree `ls_laplacian`, PageRank's
#: dangling rows in Python).
_OLD_ITEMS = os.environ.get("MOJOLEARN_XN_OLD_ITEMS", "") == "1"
#: lane/apple-fast-neighbors2 (2026-10-02): LabelPropagation /
#: LabelSpreading's kNN-graph loop as one resident op (`lp_iterate_knn`) is
#: the FAST + Apple default, read back from the binding (`_lp_fast_resident`,
#: no env read).


def _kfeat_flags(est):
    """lane apple-fast-w2-kfeat: the chi2 samplers' FAST + Apple fit entries
    compiled into the bound binary (x_neighbors/kfeat_dev.mojo, all default,
    each off under its _OFF define: bit 1 ACHI2_DEVSCAN, bit 2 SCHI2_MOJO_MT,
    bit 4 SCHI2_LAZYW);
    0 on the FAST tier without them, on IDENTICAL and on the host column."""
    if not est._fast_tier():
        return 0
    fn = getattr(est._bind(), "x_neighbors_kfeat_flags", None)
    return int(fn()) if fn is not None else 0


def _lp_fast_resident(est):
    """Whether the bound binary takes the resident kNN-graph loop
    (x_neighbors/iter_device.mojo LP_FAST_RESIDENT: FAST + Apple, off with
    `-D MOJOLEARN_LP_FAST_RESIDENT_OFF`). 0 on IDENTICAL and the host column."""
    fn = getattr(est._bind(), "x_neighbors_lp_fast_resident", None)
    return fn is not None and int(fn()) != 0


def _p2m_relabel(est, lab, name=None):
    """(count, labels): lab renumbered by first occurrence on the device
    (`xn_p2m_relabel`), or None when a label falls outside [0, n)."""
    n = lab.shape[0]
    out = empty((max(n, 1),), "<i4")
    info = empty((2,), "<i4")
    getattr(est._bind(name), "xn_p2m_relabel")(
        [addr_ro(lab, name="xn_p2m_relabel input"), addr(out, name="xn_p2m_relabel output"),
         addr(info, name="xn_p2m_relabel output")], [n], [])
    cnt, bad = info.tolist()
    if bad:
        return None
    return int(cnt), out


class _XNeighbors(NumericModeMixin):
    """The primitives, one method each. Every buffer is an owned Array held
    in a local for the duration of the call (the `_buffer` contract)."""

    _BINDING = "_mojolearn_x_neighbors"

    def _op(self, name, bufs, ints=(), floats=()):
        addrs = [addr(a, name=f"xn_{name} output") if w else addr_ro(a, name=f"xn_{name} input")
                 for a, w in bufs]
        getattr(self._bind(), "xn_" + name)(addrs, [int(v) for v in ints], [float(v) for v in floats])

    def _sqdist(self, A, B):
        n, d = A.shape
        m = B.shape[0]
        out = _empty_out((n, m), "<f4")
        self._op("sqdist", [(A, 0), (B, 0), (out, 1)], (n, m, d))
        return out

    def _l1dist(self, A, B):
        n, d = A.shape
        m = B.shape[0]
        out = _empty_out((n, m), "<f4")
        self._op("l1dist", [(A, 0), (B, 0), (out, 1)], (n, m, d))
        return out

    def _fast_tier(self):
        """True on the FAST tier (the lane/apple-fast-neighbors2 switches
        apply there only; IDENTICAL keeps its ops)."""
        try:
            return self.numeric_mode_used() == "fast"
        except Exception:
            return False

    def _kernel(self, A, B, kind, gamma, coef0, degree):
        n, d = A.shape
        m = B.shape[0]
        out = _empty_out((n, m), "<f4")
        self._op("kernel", [(A, 0), (B, 0), (out, 1)], (n, m, d, _KERNELS[kind], int(degree)),
                 (_f32_scalar(gamma), _f32_scalar(coef0)))
        return out

    def _matmul(self, A, B):
        n, k = A.shape
        m = B.shape[1]
        out = _empty_out((n, m), "<f4")
        self._op("matmul", [(A, 0), (B, 0), (out, 1)], (n, k, m))
        return out

    def _unary(self, X, op, a=1.0, b=0.0):
        out = empty(X.shape, "<f4")
        self._op("unary", [(X, 0), (out, 1)], (X.size, op), (a, b))
        return out

    def _knn_select(self, D, k, exclude_self):
        n, m = D.shape
        dist = _empty_out((n, k), "<f4")
        idx = empty((n, k), "<i4")
        self._op("knn_select", [(D, 0), (dist, 1), (idx, 1)], (n, m, k, 1 if exclude_self else 0))
        return dist, idx

    def _rowsum(self, A):
        out = _empty_out((A.shape[0],), "<f4")
        self._op("rowsum", [(A, 0), (out, 1)], A.shape)
        return out

    def _colsum(self, A):
        out = _empty_out((A.shape[1],), "<f4")
        self._op("colsum", [(A, 0), (out, 1)], A.shape)
        return out

    def _scale_div(self, X, s):
        out = empty(X.shape, "<f4")
        self._op("scale_div", [(X, 0), (out, 1)], (X.size,), (s,))
        return out

    def _take_rows(self, X, rows):
        rows = _i32(rows, "rows")
        out = _empty_out((len(rows), X.shape[1]), "<f4")
        self._op("take_rows", [(X, 0), (rows, 0), (out, 1)], (len(rows), X.shape[1], X.shape[0]))
        return out

    def _take_cols(self, X, cols):
        cols = _i32(cols, "cols")
        out = _empty_out((X.shape[0], len(cols)), "<f4")
        self._op("take_cols", [(X, 0), (cols, 0), (out, 1)], (X.shape[0], X.shape[1], len(cols)))
        return out

    def _variance(self, X):
        out = _empty_out((1,), "<f4")
        self._op("variance", [(X, 0), (out, 1)], (X.size,))
        return out.tolist()[0]

    def _knn_sq(self, Q, R, k, exclude_self):
        """(squared distances, indices) of the k nearest rows of R to each row
        of Q, ascending by (value, index): `knn_select(sqdist(Q, R))` without
        the n x m matrix (the fused `knn_sq` item runs the same statements;
        MOJOLEARN_XN_UNFUSED_KNN=1 restores the two ops, an A/B arm)."""
        if _UNFUSED_KNN:
            return self._knn_select(self._sqdist(Q, R), k, exclude_self)
        n, d = Q.shape
        m = R.shape[0]
        dist = _empty_out((n, k), "<f4")
        idx = empty((n, k), "<i4")
        self._op("knn_sq" if _OLD_ITEMS else "knn_sq_tiled", [(Q, 0), (R, 0), (dist, 1), (idx, 1)],
                 (n, m, d, k, 1 if exclude_self else 0))
        return dist, idx

    def _knn(self, Q, R, k, exclude_self):
        """Exact k-NN, euclidean: (distances, indices), ascending by (distance,
        index)."""
        sq, idx = self._knn_sq(Q, R, k, exclude_self)
        return self._unary(sq, _U_SQRT), idx


def _p2m_sort_rows(est, indptr, cols, dists, name=None):
    """(cols, dists) of a CSR's rows each sorted ascending by distance, ties
    in position order (Python's stable `sorted`), on the device
    (`xn_p2m_row_sort`, a segmented bitonic sort). int32 / float32 Arrays."""
    nq = indptr.shape[0] - 1
    nnz = dists.shape[0]
    p = 1
    while p < nnz:
        p *= 2
    lg = p.bit_length() - 1
    n_steps = lg * (lg + 1) // 2
    oc = empty((nnz,), "<i4")
    od = empty((nnz,), "<f4")
    getattr(est._bind(name), "xn_p2m_row_sort")(
        [addr_ro(indptr, name="xn_p2m_row_sort input"), addr_ro(cols, name="xn_p2m_row_sort input"),
         addr_ro(dists, name="xn_p2m_row_sort input"), addr(oc, name="xn_p2m_row_sort output"),
         addr(od, name="xn_p2m_row_sort output")], [nq, nnz, p, n_steps], [])
    return oc, od


# ====================================================================== LOF
class LocalOutlierFactor(_XNeighbors):
    """Unsupervised outlier detection by the local outlier factor.

    Reference: scikit-learn `neighbors/_lof.py` (1.9.0): `fit` (k = min(
    n_neighbors, n_samples - 1), `kneighbors` on the training set without the
    sample itself), `_local_reachability_density`, `negative_outlier_factor_`
    = -mean(lrd[neighbors] / lrd), `offset_` (-1.5 at contamination='auto',
    else the contamination percentile), `score_samples` / `decision_function`
    / `predict` under novelty=True, `fit_predict` under novelty=False.

    EXACT brute-force neighbors, euclidean (metric 'minkowski' with p=2 or
    'euclidean'); any other metric is refused by name. `algorithm`,
    `leaf_size` and `n_jobs` are accepted and change nothing: every search is
    exact. Ties between equal distances go to the lower training index, and a
    duplicate of a training row is its neighbor (sklearn drops the query's
    own index, and so does this). Float32 where sklearn is float64.
    """

    def __init__(self, n_neighbors=20, *, algorithm="auto", leaf_size=30, metric="minkowski", p=2,
                 metric_params=None, contamination="auto", novelty=False, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.contamination = contamination
        self.novelty = novelty
        self.n_jobs = n_jobs

    def _check_params(self):
        if not (self.metric == "euclidean" or (self.metric == "minkowski" and self.p == 2)):
            raise NotImplementedError(
                f"LocalOutlierFactor: metric={self.metric!r} (p={self.p!r}) is not implemented; "
                "euclidean (or minkowski with p=2) only")
        if self.metric_params:
            raise NotImplementedError("LocalOutlierFactor: metric_params is not implemented")
        if self.contamination != "auto" and not (0.0 < float(self.contamination) <= 0.5):
            raise ValueError("contamination must be 'auto' or in (0, 0.5]")
        if int(self.n_neighbors) < 1:
            raise ValueError("n_neighbors must be >= 1")

    def fit(self, X, y=None):
        self._check_params()
        X = _f32(X)
        n, d = X.shape
        if n < 2:
            raise ValueError("LocalOutlierFactor needs at least 2 samples")
        k = min(int(self.n_neighbors), n - 1)
        dist, idx = self._knn(X, X, k, True)
        lrd = _empty_out((n,), "<f4")
        self._op("lof_lrd", [(dist, 0), (idx, 0), (dist, 0), (lrd, 1)], (k, n, n))
        score = _empty_out((n,), "<f4")
        self._op("lof_score", [(idx, 0), (lrd, 0), (lrd, 0), (score, 1)], (k, n, n))
        self._fit_X, self._fit_dist, self._lrd = X, dist, lrd
        self.n_neighbors_ = k
        self.n_features_in_ = d
        self.n_samples_fit_ = n
        self.negative_outlier_factor_ = score
        if self.contamination == "auto":
            self.offset_ = -1.5
        else:
            q = 100.0 * float(self.contamination)
            self.offset_ = _f32_scalar(self._sorted_percentile(score, q))
        return self

    def _sorted_percentile(self, score, q):
        """numpy.percentile (method 'linear') of score with the sort on the device (`p2m_row_sort`
        over one row: ascending by value, ties by position, as `sorted`);
        only the two order statistics it reads come back to Python."""
        n = score.shape[0]
        srt = _p2m_sort_rows(self, _i32([0, n], "indptr"), zeros((n,), "<i4"), score)[1]
        pos = (q / 100.0) * (n - 1)
        lo = math.floor(pos)
        hi = min(lo + 1, n - 1)
        t = pos - lo
        a, b = srt[lo], srt[hi]
        diff = b - a
        return b - diff * (1.0 - t) if t >= 0.5 else a + diff * t

    def fit_predict(self, X, y=None):
        if self.novelty:
            raise AttributeError("fit_predict is not available when novelty=True; use predict on new data")
        self.fit(X)
        off = self.offset_
        score = self.negative_outlier_factor_
        lab = empty((score.size,), "<i4")
        if score.size:
            self._op("p2m_sign_label", [(score, 0), (lab, 1)], (score.size, 0), (off,))
        return lab.astype("<i8")

    def _novelty(self, what):
        if not self.novelty:
            raise AttributeError(f"{what} is not available when novelty=False; use fit_predict")

    def score_samples(self, X):
        self._novelty("score_samples")
        Q = _f32(X)
        k, nf = self.n_neighbors_, self.n_samples_fit_
        dist, idx = self._knn(Q, self._fit_X, k, False)
        nq = Q.shape[0]
        lrd = _empty_out((nq,), "<f4")
        self._op("lof_lrd", [(dist, 0), (idx, 0), (self._fit_dist, 0), (lrd, 1)], (k, nq, nf))
        score = _empty_out((nq,), "<f4")
        self._op("lof_score", [(idx, 0), (self._lrd, 0), (lrd, 0), (score, 1)], (k, nq, nf))
        return score

    def decision_function(self, X):
        self._novelty("decision_function")
        return self._unary(self.score_samples(X), _U_IDENTITY, 1.0, -self.offset_)

    def predict(self, X):
        self._novelty("predict")
        dec = self.decision_function(X)
        lab = empty((dec.size,), "<i4")
        if dec.size:
            self._op("p2m_sign_label", [(dec, 0), (lab, 1)], (dec.size, 1), (0.0,))
        return lab.astype("<i8")


# ====================================================================== NearestCentroid
class NearestCentroid(_XNeighbors):
    """Nearest centroid classifier.

    Reference: scikit-learn `neighbors/_nearest_centroid.py` (1.9.0): per-class
    mean (euclidean) or per-class median (manhattan) centroids,
    `within_class_std_dev_`, the shrunken centroids (`shrink_threshold`),
    `class_prior_` ('uniform', 'empirical' or given), `predict` (the nearest
    centroid when the priors are uniform, else the discriminant),
    `decision_function` and `predict_proba` (euclidean only, as theirs).
    `deviations_`, `predict_log_proba` (their log-softmax of the
    discriminant) and `score`. Sparse input is densified (the same values).
    Float32 where sklearn is float64. DEVIATION 5201: a feature whose shrink
    scale m*s is zero gets deviation 0 where theirs divides by zero.
    """

    def __init__(self, metric="euclidean", *, shrink_threshold=None, priors="uniform"):
        self.metric = metric
        self.shrink_threshold = shrink_threshold
        self.priors = priors

    def fit(self, X, y):
        if self.metric not in ("euclidean", "manhattan"):
            raise ValueError("NearestCentroid: metric must be 'euclidean' or 'manhattan'")
        X = _f32(X)
        n, d = X.shape
        # the native encoder's int32 codes, the class counts from the device
        # (`p2m_class_counts`, an atomic add per row; lane pyglue-numeric:
        # a Python counting loop on every column but FAST + Apple)
        from ._labels import encode_labels
        classes, lab = encode_labels(y)
        if lab.size != n:
            raise ValueError("X and y have different numbers of rows")
        C = len(classes)
        if C < 2:
            raise ValueError(f"The number of classes has to be greater than one; got {C} class")
        nk = _empty_out((C,), "<f4")
        info = empty((1,), "<i4")
        self._op("p2m_class_counts", [(lab, 0), (nk, 1), (info, 1)], (n, C))
        counts = [int(v) for v in nk.tolist()]          # glue: the k class counts (priors, offsets)
        if self.priors == "empirical":
            prior = [c / float(n) for c in counts]
        elif self.priors == "uniform":
            prior = [1.0 / C] * C
        else:
            prior = [float(v) for v in (self.priors.tolist() if hasattr(self.priors, "tolist") else self.priors)]
            if len(prior) != C:
                raise ValueError("priors must have one entry per class")
            if any(p < 0 for p in prior):
                raise ValueError("priors must be non-negative")
            tot = math.fsum(prior)
            if not math.isclose(tot, 1.0, rel_tol=1e-5, abs_tol=1e-8):
                prior = [p / tot for p in prior]
        self.class_prior_ = Array.from_list(prior, "<f8")
        if self.metric == "euclidean":
            cent = _empty_out((C, d), "<f4")
            if os.environ.get("MOJOLEARN_NC_SPLIT_OPS", "") == "1":
                self._op("group_mean", [(X, 0), (lab, 0), (cent, 1)], (n, d, C))
        else:
            # the per-class medians on the device (cpu-gpu-cleanup w2-pyglue,
            # x_neighbors/sort_items.mojo): a bitonic sort of every feature's
            # rows by (class, value, row) over a power-of-two pad, then each
            # class's middle value(s) at its offset; offsets are the class
            # counts' prefix (integer bookkeeping over the classes)
            start = [0] * (C + 1)
            for c in range(C):
                start[c + 1] = start[c] + counts[c]
            p = 1
            while p < n:
                p *= 2
            lg = p.bit_length() - 1
            cent = _empty_out((C, d), "<f4")
            self._op("nc_median", [(X, 0), (lab, 0), (_i32(start, "start"), 0), (cent, 1)],
                     (n, d, C, p, lg * (lg + 1) // 2))
        stats = _empty_out((d,), "<f4")
        new_cent = _empty_out((C, d), "<f4")
        devs = _empty_out((C, d), "<f4")
        dsc = None
        if self.metric == "euclidean" and os.environ.get("MOJOLEARN_NC_SPLIT_OPS", "") != "1":
            # lane/neural-pass95: the class means, the within-class std and the
            # dataset centroid in one op (x_neighbors/items.mojo nc_stats_item):
            # X goes to the device once; MOJOLEARN_NC_SPLIT_OPS=1 restores the three ops
            dsc = _empty_out((d,), "<f4")
            self._op("nc_stats", [(X, 0), (lab, 0), (nk, 0), (cent, 1), (stats, 1), (dsc, 1)], (n, d, C))
        else:
            self._op("nc_std", [(X, 0), (lab, 0), (cent, 0), (stats, 1)], (n, d, C))
        # the median of the d stds and the all-zero test on the device
        # (x_neighbors/sort_items.mojo nc_med_std: a bitonic sort, then the
        # middle value(s)); two floats come back
        p = 1
        while p < d:
            p *= 2
        lg = p.bit_length() - 1
        ms = _empty_out((2,), "<f4")
        self._op("nc_med_std", [(stats, 0), (ms, 1)], (d, p, lg * (lg + 1) // 2))
        med_std, all_zero = ms.tolist()
        if all_zero and self._ptp_zero(X):
            raise ValueError("All features have zero variance. Division by zero.")
        shrink = float(self.shrink_threshold) if self.shrink_threshold else 0.0
        if dsc is not None:
            self._op("nc_shrink_d", [(dsc, 0), (cent, 0), (nk, 0), (stats, 0), (new_cent, 1), (devs, 1)],
                     (n, d, C, 1 if shrink else 0), (med_std, shrink))
        else:
            self._op("nc_shrink", [(X, 0), (cent, 0), (nk, 0), (stats, 0), (new_cent, 1), (devs, 1)],
                     (n, d, C, 1 if shrink else 0), (med_std, shrink))
        self.centroids_ = new_cent
        self.deviations_ = devs
        self.within_class_std_dev_ = stats
        self.classes_ = classes
        self._codes_classes = classes
        self.n_features_in_ = d
        return self

    def _ptp_zero(self, X):
        """Every feature constant (sklearn's ptp == 0): `p2m_const_cols`."""
        n, d = X.shape
        flag = empty((1,), "<i4")
        self._op("p2m_const_cols", [(X, 0), (flag, 1)], (n, d))
        return flag.tolist()[0] == 0

    def _uniform(self):
        C = len(self.classes_)
        return all(math.isclose(p, 1.0 / C, rel_tol=1e-5, abs_tol=1e-8) for p in self.class_prior_.tolist())

    def predict(self, X):
        Q = _f32(X)
        if self._uniform():
            D = self._sqdist(Q, self.centroids_) if self.metric == "euclidean" else self._l1dist(Q, self.centroids_)
            _, idx = self._knn_select(D, 1, False)
            return _class_array(self.classes_, idx.reshape((idx.shape[0],)))
        return _argmax_labels(self, self.classes_, self.decision_function(Q))

    def decision_function(self, X):
        if self.metric != "euclidean":
            raise AttributeError("decision_function is available for metric='euclidean' only")
        Q = _f32(X)
        C = len(self.classes_)
        prior = Array.from_list(self.class_prior_.tolist(), "<f4")
        std = self.within_class_std_dev_
        out = _empty_out((Q.shape[0], C), "<f4")
        self._op("nc_decision", [(Q, 0), (self.centroids_, 0), (std, 0), (prior, 0), (out, 1)],
                 (Q.shape[0], Q.shape[1], C))
        return out

    def predict_proba(self, X):
        dec = self.decision_function(X)
        out = empty(dec.shape, "<f4")
        self._op("softmax", [(dec, 0), (out, 1)], dec.shape)
        return out

    def predict_log_proba(self, X):
        dec = self.decision_function(X)
        out = empty(dec.shape, "<f4")
        self._op("log_softmax", [(dec, 0), (out, 1)], dec.shape)
        return out

    def score(self, X, y, sample_weight=None):
        return _accuracy(y, self.predict(X), sample_weight)


def _resolve_gamma(gamma, kernel, X, est):
    d = X.shape[1]
    if gamma is None or gamma == "auto":
        return 1.0 / d
    if gamma == "scale":
        var = est._variance(X)
        return 1.0 / (d * var) if var != 0 else 1.0
    return float(gamma)


# ====================================================================== OneClassSVM
class OneClassSVM(_XNeighbors):
    """Unsupervised outlier detection, the nu one-class SVM.

    Reference: scikit-learn `svm/_classes.py` (OneClassSVM) over libsvm
    `svm.cpp` (`solve_one_class`, `Solver::Solve` with WSS3 working-set
    selection, `calculate_rho`). On the GPU each SMO iteration is three grid
    launches (x_neighbors/ocsvm_dev.mojo); the host column runs the same
    order as one item (x_neighbors/items.mojo `ocsvm_smo_item`) over the kernel matrix,
    in float32 with the pinned spellings (DEVIATION 5200; libsvm is double).
    `shrinking` and `cache_size` are accepted and change nothing (no
    shrinking, the whole kernel matrix is formed). max_iter=-1 caps at
    10_000_000 iterations. `sample_weight` is libsvm's per-sample bound C_i
    (samples of weight 0 are dropped first, as libsvm's remove_zero_weight;
    `support_` indexes the caller's rows). kernel='precomputed' takes the
    n x n Gram matrix at fit and the n_test x n_train one after. Callable
    kernels are refused (a user arithmetic outside the pinned items).
    """

    def __init__(self, *, kernel="rbf", degree=3, gamma="scale", coef0=0.0, tol=1e-3, nu=0.5,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter

    def fit(self, X, y=None, sample_weight=None):
        if callable(self.kernel) or self.kernel not in ("linear", "poly", "rbf", "sigmoid", "precomputed"):
            raise NotImplementedError(f"OneClassSVM: kernel={self.kernel!r} is not implemented")
        if not (0.0 < float(self.nu) <= 1.0):
            raise ValueError("nu must be in (0, 1]")
        X = _f32(X)
        n, d = X.shape
        if self.kernel == "precomputed" and n != d:
            raise ValueError("Precomputed matrix must be a square matrix.")
        if sample_weight is None:
            rows = None                                   # every row
            cv = full((n,), 1.0, "<f4")
        else:
            wv = _f32_1d(sample_weight, "sample_weight")
            if wv.shape[0] != n:
                raise ValueError("sample_weight and X have different numbers of samples")
            if n and wv.min() < 0:
                raise ValueError("negative sample_weight is not supported")
            # libsvm's remove_zero_weight as a device compaction (cpu-gpu-cleanup
            # w2-pyglue, x_neighbors/sort_items.mojo pos_compact): the rows of
            # positive weight in row order and their float32 weights
            prow = empty((n,), "<i4")
            pval = _empty_out((n,), "<f4")
            pinfo = empty((1,), "<i4")
            self._op("pos_compact", [(wv, 0), (prow, 1), (pval, 1), (pinfo, 1)], (n,))
            kept = pinfo.tolist()[0]
            if not kept:
                raise ValueError("Invalid input - all samples have zero or negative weights.")
            rows, cv = prow[:kept], pval[:kept]
        m = n if rows is None else rows.shape[0]
        if self.kernel == "precomputed":
            self._gamma = 0.0
            Q = X if rows is None else self._take_cols(self._take_rows(X, rows), rows)
        else:
            self._gamma = _f32_scalar(_resolve_gamma(self.gamma, self.kernel, X, self))
            Xw = X if rows is None else self._take_rows(X, rows)
            # lane/apple-fast-gap-cls2 (FAST + Apple default; -D MOJOLEARN_XN_FAST_CLS2_OCSVM_RES_OFF off):
            # the binding forms the same Gram on the device and solves over it
            # there (no 400 MB download into a fresh host array and upload back)
            res_fn = getattr(self._bind(), "x_neighbors_ocsvm_resident", None) if self._fast_tier() else None
            Q = None if res_fn is not None else self._kernel(Xw, Xw, self.kernel, self._gamma, self.coef0, self.degree)
        # libsvm's start (solve_one_class) by the base binding's
        # `ocsvm_alpha_init_f32` (lane pyglue-numeric: a Python loop)
        from ._buffer import _native
        alpha = empty((m,), "<f4")
        _native("ocsvm_alpha_init_f32")(addr_ro(cv, name="C"), m, float(self.nu),
                                        0 if sample_weight is None else 1, addr(alpha, name="alpha"))
        info = _empty_out((1,), "<f4")
        iters = empty((1,), "<i4")
        cap = 10_000_000 if int(self.max_iter) < 0 else int(self.max_iter)
        if Q is None:
            res_fn([addr_ro(Xw, name="xn_ocsvm X"), addr_ro(cv, name="xn_ocsvm cv"),
                    addr(alpha, name="xn_ocsvm alpha"), addr(info, name="xn_ocsvm info"),
                    addr(iters, name="xn_ocsvm iters")],
                   [m, d, _KERNELS[self.kernel], int(self.degree), cap],
                   [float(self._gamma), float(_f32_scalar(self.coef0)), float(_f32_scalar(self.tol))])
        else:
            self._op("ocsvm", [(Q, 0), (cv, 0), (alpha, 1), (info, 1), (iters, 1)], (m, cap), (_f32_scalar(self.tol),), )
        # the op's scalar order is (n, eps, max_iter): ints (n, max_iter), floats (eps,)
        rho = info.tolist()[0]
        # the support (alpha > 0, in order) and its coefficients by the
        # device compaction (pos_compact), the training rows by the core gather
        srow = empty((max(m, 1),), "<i4")
        sval = _empty_out((max(m, 1),), "<f4")
        sinfo = empty((1,), "<i4")
        if m:
            self._op("pos_compact", [(alpha, 0), (srow, 1), (sval, 1), (sinfo, 1)], (m,))
        n_sv = sinfo.tolist()[0] if m else 0
        local = srow[:n_sv]
        if rows is None or not n_sv:
            support = local
        else:
            support = empty((n_sv,), "<i4")
            _native("gather_i32")(addr_ro(rows, name="rows"), rows.shape[0], addr_ro(local, name="support"),
                                  n_sv, addr(support, name="support_"))
        self.support_ = support
        if self.kernel == "precomputed":
            self.support_vectors_ = _empty_out((n_sv, 0), "<f4")
        else:
            self.support_vectors_ = self._take_rows(X, support)
        self.dual_coef_ = sval[:n_sv].reshape((1, n_sv))
        self.intercept_ = Array.from_list([-rho], "<f4")
        self.offset_ = rho
        self.n_iter_ = iters.tolist()[0]
        self.n_features_in_ = d
        self.fit_status_ = 0
        return self

    def score_samples(self, X):
        Q = _f32(X)
        n_sv = self.dual_coef_.shape[1]
        coef = as_f32_c(self.dual_coef_, ndim=2, name="dual_coef_")[0].reshape((n_sv, 1))
        if self.kernel != "precomputed" and n_sv and Q.shape[0] and not _OLD_ITEMS:
            # the fused chain (lane/py-dn-kern): kernel then matmul per row
            # tile on the device, only the scores downloaded (`xn_kernel_matmul`)
            nq, d = Q.shape
            s = _empty_out((nq, 1), "<f4")
            self._op("kernel_matmul", [(Q, 0), (self.support_vectors_, 0), (coef, 0), (s, 1)],
                     (nq, n_sv, d, 1, _KERNELS[self.kernel], int(self.degree)),
                     (_f32_scalar(self._gamma), _f32_scalar(self.coef0)))
            return s.reshape((nq,))
        if self.kernel == "precomputed":
            K = self._take_cols(Q, self.support_)
        else:
            K = self._kernel(Q, self.support_vectors_, self.kernel, self._gamma, self.coef0, self.degree)
        s = self._matmul(K, coef)
        return s.reshape((Q.shape[0],))

    def decision_function(self, X):
        return self._unary(self.score_samples(X), _U_IDENTITY, 1.0, -self.offset_)

    def predict(self, X):
        dec = self.decision_function(X)
        lab = empty((dec.size,), "<i4")
        if dec.size:
            self._op("p2m_sign_label", [(dec, 0), (lab, 1)], (dec.size, 2), (0.0,))
        return lab.astype("<i8")

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)


# ====================================================================== KernelPCA
class KernelPCA(_XNeighbors):
    """Kernel principal component analysis.

    Reference: scikit-learn `decomposition/_kernel_pca.py` (1.9.0) with
    `preprocessing.KernelCenterer`: the kernel matrix, centered in their
    order, the dense eigendecomposition (eigen_solver 'dense'; here
    x_decomp's eigh, the round-robin Jacobi on the device, ascending, then
    sklearn's `svd_flip(u, None)` sign rule), eigenvalues
    below zero set to zero, components sorted by decreasing eigenvalue (equal
    eigenvalues: the higher solver index first, as their reversed argsort),
    zero components removed when n_components is None or remove_zero_eig.
    Every eigen_solver is served by the dense solve (DEVIATION 5202: arpack
    and randomized are approximations of it). kernel='precomputed' takes the
    Gram matrix; fit_inverse_transform and callables are refused by name.
    """

    def __init__(self, n_components=None, *, kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver="auto",
                 tol=0, max_iter=None, iterated_power="auto", remove_zero_eig=False,
                 random_state=None, copy_X=True, n_jobs=None):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.remove_zero_eig = remove_zero_eig
        self.random_state = random_state
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    def _k(self, A, B):
        if self.kernel == "precomputed":
            return A
        return self._kernel(A, B, self.kernel, self._gamma, self.coef0, self.degree)

    def fit(self, X, y=None):
        if callable(self.kernel) or (self.kernel not in _KERNELS and self.kernel != "precomputed"):
            raise NotImplementedError(f"KernelPCA: kernel={self.kernel!r} is not implemented")
        if self.fit_inverse_transform:
            raise NotImplementedError("KernelPCA: fit_inverse_transform is not implemented")
        if self.kernel_params:
            raise NotImplementedError("KernelPCA: kernel_params is not implemented")
        X = _f32(X)
        n, d = X.shape
        if self.kernel == "precomputed" and n != d:
            raise ValueError("Precomputed matrix must be a square matrix.")
        self._gamma = _f32_scalar(1.0 / d if self.gamma is None else float(self.gamma))
        Kc_M = None
        kit = None
        # w2-w4d-kpca-q validates RBF with the auto top-k solver only.
        # Keep unmeasured kernels and dense-solver routes on their old path.
        if (self.kernel == "rbf" and self.eigen_solver == "auto" and n > 200
                and self.n_components is not None and 0 < int(self.n_components) < 10
                and _kpca_resident(self)):
            # lane/apple-fast-kapprox: the kernel matrix never leaves the
            # device (main's path moved the n x n matrix through the host
            # five times: the kernel op's download, colsum's upload, the
            # center op's upload and download, tobytes + the kit's upload)
            from ._expansion_decomp import _Kit
            kit = _Kit(self.numeric_mode_used())
            got = _kpca_resident_center(kit, X, n, self.kernel, self._gamma, _f32_scalar(self.coef0),
                                        int(self.degree))
            if got is not None:
                Kc_M, cols_m, all_m = got
                cols = cols_m.out((n,))
                all_ = all_m.out((1,))
        if Kc_M is None:
            K = self._k(X, X)
            cols = self._scale_div(self._colsum(K), float(n))              # K_fit_rows_
            all_ = self._scale_div(self._colsum(cols.reshape((1, n))), float(n))  # K_fit_all_
            Kc = _empty_out((n, n), "<f4")
            self._op("kpca_center", [(K, 0), (cols, 0), (cols, 0), (all_, 0), (Kc, 1)], (n, n))
        # Top-k GPU Lanczos instead of the full n-by-n host eigensolve when
        # auto asks for a few components (on by default since 2026-09-30:
        # L40S CUDA, Apple Metal and the CPU column bit-identical, residual < 1e-5,
        # float64 reference match; lane pyglue-numeric deleted the
        # MOJOLEARN_XN_KPCA_LANCZOS route switch). Anything outside that
        # scope, or a basis that does not converge, takes the exact dense
        # path below.
        c = 0 if self.n_components is None else int(self.n_components)
        lanczos = self.eigen_solver == "auto" and n > 200 and 0 < c < 10
        if lanczos and kit is None:
            from ._expansion_decomp import _Kit
            # Every column (CUDA, HIP, Metal and the CPU host binding) takes
            # this route, so a CPU-only install gives the GPUs' bits.
            kit = _Kit(self.numeric_mode_used())
        result = None
        if lanczos:
            import array
            from ._expansion_decomp import _M, _lanczos_top
            if Kc_M is None:
                # Array's buffer is float32 in row order; no Python float list of
                # n*n cells. The kit uploads it once and retains the device store.
                store = array.array("f")
                store.frombytes(Kc.tobytes())
                Kc_M = _M(store, n, n)
            result = _lanczos_top(kit, Kc_M, c)
        if result is not None:
            values, vectors = result
            vectors = vectors.neg_cols(kit.absmax_flags(vectors, True))
            vals = [max(float(v), 0.0) for v in values.s]
            keep = [i for i, v in enumerate(vals) if not self.remove_zero_eig or v > 0]
            vectors = vectors.take_cols(keep)
            self.eigenvalues_ = Array._from_flat([vals[i] for i in keep], (len(keep),), "<f4")
            self.eigenvectors_ = Array._from_flat(vectors.s, (n, len(keep)), "<f4")
            self._fit_X, self._fit_cols, self._fit_all = X, cols, all_
            self.n_features_in_ = d
            return self
        # lane hr2-kpca-seq: the dense eigendecomposition is x_decomp's
        # eigh (the pinned round-robin Jacobi, x_decomp/rr.mojo, on the
        # device; the host binding runs the same rounds), never the host
        # Jacobi of x_neighbors/eigh.mojo inside the GPU binding.
        c = n if self.n_components is None else min(n, int(self.n_components))
        import array
        from ._expansion_decomp import _Kit, _M
        if kit is None:
            kit = _Kit(self.numeric_mode_used())
        if Kc_M is None:
            store = array.array("f")
            store.frombytes(Kc.tobytes())
            Kc_M = _M(store, n, n)
        wm, Vm = kit.eigh(Kc_M)
        wl = list(wm.s)
        order = list(range(n - 1, -1, -1))           # descending; equal values: higher index first
        order = order[:c]
        vals = [max(wl[i], 0.0) for i in order]
        if self.n_components is None or self.remove_zero_eig:
            keep = [j for j in range(c) if vals[j] > 0]
            order = [order[j] for j in keep]
            vals = [vals[j] for j in keep]
        self.eigenvalues_ = Array.from_list(vals, "<f4")
        # sklearn's svd_flip(u, None) on the kept columns: each column's
        # largest-|.| entry (ties to the lower row) made positive
        vecs = Vm.take_cols(order)
        if order:
            vecs = vecs.neg_cols(kit.absmax_flags(vecs, True))
        self.eigenvectors_ = Array._from_flat(vecs.s, (n, len(order)), "<f4")
        self._fit_X, self._fit_cols, self._fit_all = X, cols, all_
        self.n_features_in_ = d
        return self

    def fit_transform(self, X, y=None, **params):
        self.fit(X)
        return self._alpha_scale(0)

    def _alpha_scale(self, divide):
        V = self.eigenvectors_
        out = empty(V.shape, "<f4")
        self._op("kpca_alpha_scale", [(V, 0), (self.eigenvalues_, 0), (out, 1)], (V.shape[0], V.shape[1], divide))
        return out

    def transform(self, X):
        Q = _f32(X)
        nq = Q.shape[0]
        nf = self._fit_X.shape[0]
        alphas = self._alpha_scale(1)
        c = alphas.shape[1]
        pre = self.kernel == "precomputed"
        if pre and Q.shape[1] != nf:
            raise ValueError(f"KernelPCA: a precomputed kernel needs {nf} columns (the fitted rows), got {Q.shape[1]}")
        if nq and c and not _OLD_ITEMS:
            # the fused chain (lane/py-dn-kern): kernel, rowsum, / n_fit,
            # center and the product with the scaled alphas per row tile on
            # the device, only the (nq, c) result downloaded (`xn_kpca_transform`)
            out = _empty_out((nq, c), "<f4")
            self._op("kpca_transform",
                     [(Q, 0), (Q if pre else self._fit_X, 0), (self._fit_cols, 0), (self._fit_all, 0),
                      (alphas, 0), (out, 1)],
                     (nq, nf, Q.shape[1], c, _XN_PRECOMPUTED_KIND if pre else _KERNELS[self.kernel], int(self.degree)),
                     (0.0 if pre else _f32_scalar(self._gamma), 0.0 if pre else _f32_scalar(self.coef0), float(nf)))
            return out
        K = self._k(Q, self._fit_X)
        pred = self._scale_div(self._rowsum(K), float(nf))
        Kc = _empty_out((nq, nf), "<f4")
        self._op("kpca_center", [(K, 0), (self._fit_cols, 0), (pred, 0), (self._fit_all, 0), (Kc, 1)], (nq, nf))
        return self._matmul(Kc, alphas)

    def inverse_transform(self, X):
        raise NotImplementedError("KernelPCA: inverse_transform needs fit_inverse_transform, which is not implemented")

    def get_feature_names_out(self, input_features=None):
        return _prefixed_names(self, self.eigenvalues_.shape[0])


# ====================================================================== RandomState
class _LegacyRandomState:
    """numpy's legacy `RandomState(seed)` stream (MT19937 seeded by
    init_genrand, what `check_random_state(int)` builds), in integers and
    IEEE doubles only, so a sampler draws exactly scikit-learn's numbers on
    every box and needs no NumPy. Python's `random.Random` IS MT19937 with
    numpy's 53-bit double (`genrand_res53`); only the seeding differs, so the
    state is set directly."""

    def __init__(self, seed):
        import random
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"{seed!r} cannot be used to seed a RandomState instance")
        mt = [0] * 624
        mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self._r = random.Random()
        self._r.setstate((3, tuple(mt) + (624,), None))

    def random_sample(self, count):
        return [self._r.random() for _ in range(count)]

    def uniform(self, low, high, count):
        return [low + (high - low) * self._r.random() for _ in range(count)]

    def randint(self, high, count):
        """`randint(0, high, size)`, the legacy masked rejection draw."""
        rng = high - 1
        if rng == 0:
            return [0] * count
        mask = rng
        for sh in (1, 2, 4, 8, 16):
            mask |= mask >> sh
        out = []
        for _ in range(count):
            while True:
                v = self._r.getrandbits(32) & mask
                if v <= rng:
                    break
            out.append(v)
        return out


def _random_state(seed):
    """sklearn's check_random_state: an int seeds the legacy stream (drawn here
    with no NumPy); a caller's numpy RandomState is drawn from directly; None
    is numpy's global RandomState, as theirs (not reproducible, as theirs).
    Every draw is an integer or an IEEE double from the same generator
    sklearn would use, so the fitted parameters are theirs exactly."""
    if isinstance(seed, int) and not isinstance(seed, bool):
        return _LegacyRandomState(seed)
    if seed is None:
        from ._optional_numpy import require_numpy
        np = require_numpy('_expansion_neighbors')
        return _NumpyRandomState(np.random.mtrand._rand)
    if hasattr(seed, "randint") and hasattr(seed, "random_sample") and hasattr(seed, "uniform"):
        return _NumpyRandomState(seed)
    raise ValueError(f"{seed!r} cannot be used to seed a numpy.random.RandomState instance")


class _NumpyRandomState:
    """The `_LegacyRandomState` draws, from a numpy RandomState."""

    def __init__(self, rs):
        self._rs = rs

    def random_sample(self, count):
        return [float(v) for v in self._rs.random_sample(count)]

    def uniform(self, low, high, count):
        return [float(v) for v in self._rs.uniform(low, high, size=count)]

    def randint(self, high, count):
        return [int(v) for v in self._rs.randint(0, high, size=count)]


# ====================================================================== PolynomialCountSketch
class PolynomialCountSketch(_XNeighbors):
    """Polynomial kernel approximation by tensor sketch.

    Reference: scikit-learn `kernel_approximation.py` (PolynomialCountSketch,
    1.9.0): `indexHash_` and `bitHash_` are drawn exactly as theirs
    (`randint(0, n_components, (degree, n_features))`, then
    `choice([-1, 1], (degree, n_features))`, one legacy RandomState stream);
    the transform is the count sketches' circular convolution, computed
    directly instead of through an FFT (DEVIATION 5203). random_state is an
    int, a numpy RandomState or None, as sklearn's check_random_state. Sparse
    input is densified (the same values).
    """

    def __init__(self, *, gamma=1.0, degree=2, coef0=0, n_components=100, random_state=None):
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = _f32(X)
        d = X.shape[1]
        nf = d + (1 if self.coef0 != 0 else 0)
        deg, nc = int(self.degree), int(self.n_components)
        if deg < 1 or nc < 1:
            raise ValueError("degree and n_components must be >= 1")
        rs = _random_state(self.random_state)
        idx = rs.randint(nc, deg * nf)
        bits = [(-1, 1)[v] for v in rs.randint(2, deg * nf)]
        self.indexHash_ = Array.from_list([idx[p * nf:(p + 1) * nf] for p in range(deg)], "<i4")
        self.bitHash_ = Array.from_list([bits[p * nf:(p + 1) * nf] for p in range(deg)], "<i4")
        self.n_features_in_ = d
        return self

    def transform(self, X):
        X = _f32(X)
        n, d = X.shape
        if d != self.n_features_in_:
            raise ValueError("Number of features of test samples does not match that of training samples.")
        nf = self.indexHash_.shape[1]
        nc, deg = int(self.n_components), int(self.degree)
        out = _empty_out((n, nc), "<f4")
        self._op("pcs" if _OLD_ITEMS else "pcs_resident",
                 [(X, 0), (self.indexHash_, 0), (self.bitHash_, 0), (out, 1)],
                 (n, d, nf, nc, deg), (_f32_scalar(self.gamma), _f32_scalar(self.coef0)))
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        return _prefixed_names(self, int(self.n_components))


# ====================================================================== AdditiveChi2Sampler
class AdditiveChi2Sampler(_XNeighbors):
    """Approximate feature map for the additive chi-squared kernel.

    Reference: scikit-learn `kernel_approximation.py` (AdditiveChi2Sampler,
    `_transform_dense`, 1.9.0). Deterministic: no random state. Negative input
    is refused, as theirs. A sparse input gives a sparse (csr) output holding
    the dense map's nonzeros, as theirs (`_transform_sparse` maps only the
    stored entries, and a zero maps to zeros).
    """

    def __init__(self, *, sample_steps=2, sample_interval=None):
        self.sample_steps = sample_steps
        self.sample_interval = sample_interval

    def _interval(self):
        if self.sample_interval is not None:
            return float(self.sample_interval)
        table = {1: 0.8, 2: 0.5, 3: 0.4}
        if self.sample_steps not in table:
            raise ValueError("If sample_steps is not in [1, 2, 3], you need to provide sample_interval")
        return table[self.sample_steps]

    def fit(self, X, y=None):
        X = _f32(X)
        first = -2
        if X.size and _kfeat_flags(self) & 1:
            # XN_FAST_ACHI2_DEVSCAN (default, rollback _OFF): one pooled upload and device
            # scan (x_neighbors_kfeat_first_negative), not a host X.min();
            # -2 = below the binding's size gate (XN_ACHI2_DEVSCAN_MIN,
            # x_neighbors/kfeat_dev.mojo): main's check below
            first = int(self._bind().x_neighbors_kfeat_first_negative(addr_ro(X, name="X"), X.size))
        if first != -2:
            neg = first >= 0
        else:
            neg = X.size and X.min() < 0
        if neg:
            raise ValueError("Negative values in data passed to AdditiveChi2Sampler")
        self._interval()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        sparse = X if (hasattr(X, "toarray") and hasattr(X, "nnz")) else None
        X = _f32(X)
        n, d = X.shape
        steps = int(self.sample_steps)
        out = _empty_out((n, d * (2 * steps - 1)), "<f4")
        if X.size and X.min() < 0:
            raise ValueError("Negative values in data passed to AdditiveChi2Sampler")
        self._op("achi2", [(X, 0), (out, 1)], (n, d, steps), (_f32_scalar(self._interval()),))
        if sparse is not None:
            # Preserve the caller's sparse container type; dense inputs need
            # neither SciPy nor NumPy. SciPy owns this optional format conversion.
            import scipy.sparse as sp
            m = sp.csr_matrix(out.tolist(), dtype="float32")
            m.eliminate_zeros()
            return m
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        names = _feature_names_in(self, input_features)
        base = type(self).__name__.lower()
        out = [f"{base}_{nm}_sqrt" for nm in names]
        for j in range(1, int(self.sample_steps)):
            out += [f"{base}_{nm}_cos{j}" for nm in names]
            out += [f"{base}_{nm}_sin{j}" for nm in names]
        return _names_out(out)


# ====================================================================== SkewedChi2Sampler
class SkewedChi2Sampler(_XNeighbors):
    """Approximate feature map for the skewed chi-squared kernel.

    Reference: scikit-learn `kernel_approximation.py` (SkewedChi2Sampler,
    1.9.0). The uniforms are theirs exactly (legacy RandomState, see
    `_LegacyRandomState`); pi/2 * u and the offsets are formed in IEEE double
    and rounded once to float32; the inverse sech CDF, the log, the product
    and the cosine run in float32 on the lane's portable spellings. Input at
    or below -skewedness is refused, as theirs.
    """

    def __init__(self, *, skewedness=1.0, n_components=100, random_state=None):
        self.skewedness = skewedness
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = _f32(X)
        d = X.shape[1]
        nc = int(self.n_components)
        seed = self.random_state
        flags = (_kfeat_flags(self) if isinstance(seed, int) and not isinstance(seed, bool)
                 and d > 0 and nc > 0 else 0)
        self.__dict__.pop("_schi2_z", None)
        if flags & 4:
            # XN_FAST_SCHI2_LAZYW (default, rollback _OFF; x_neighbors/kfeat_dev.mojo):
            # main's z = pi/2 * u and offsets drawn into our arrays, no device
            # work; the weights come from z in the first transform's one call
            # (or on the first read of random_weights_)
            z = empty((d, nc), "<f4")
            off = empty((nc,), "<f4")
            self._bind().x_neighbors_kfeat_schi2_draw([seed & 0xFFFFFFFF, d, nc], addr(z, name="schi2 z"),
                                                      addr(off, name="random_offset_"))
            self.__dict__.pop("_random_weights", None)
            self.__dict__["_schi2_z"] = z
            self.random_offset_ = off
            self.n_features_in_ = d
            return self
        if flags & 2:
            # SCHI2_MOJO_MT (default; _OFF rollback): _LegacyRandomState(seed)'s
            # stream, pi/2 * u, the weights kernel and the offsets in one
            # binding call (x_neighbors_kfeat_schi2_fit): main's words
            w = _empty_out((d, nc), "<f4")
            off = empty((nc,), "<f4")
            self._bind().x_neighbors_kfeat_schi2_fit([seed & 0xFFFFFFFF, d, nc], addr(w, name="random_weights_"),
                                                     addr(off, name="random_offset_"))
            self.random_weights_ = w
            self.random_offset_ = off
            self.n_features_in_ = d
            return self
        rs = _random_state(self.random_state)
        u = rs.random_sample(d * nc)
        z = Array.from_list([[math.pi / 2.0 * u[f * nc + c] for c in range(nc)] for f in range(d)], "<f4")
        w = _empty_out((d, nc), "<f4")
        self._op("skew_weights", [(z, 0), (w, 1)], (d * nc,))
        self.random_weights_ = w
        self.random_offset_ = Array.from_list(rs.uniform(0.0, 2.0 * math.pi, nc), "<f4")
        self.n_features_in_ = d
        return self

    @property
    def random_weights_(self):
        # SCHI2_LAZYW: made from z on the first read (one binding call, main's
        # kernel and words) when no transform has made it yet. Stored under
        # `_random_weights` (an older pickle's plain attribute still reads).
        st = self.__dict__
        z = st.get("_schi2_z")
        if z is not None:
            w = _empty_out(z.shape, "<f4")
            self._bind().x_neighbors_kfeat_schi2_weights(addr_ro(z, name="schi2 z"),
                                                         addr(w, name="random_weights_"), z.size)
            st["_random_weights"] = w
            del st["_schi2_z"]
        if "_random_weights" in st:
            return st["_random_weights"]
        if "random_weights_" in st:
            return st["random_weights_"]
        raise AttributeError("'SkewedChi2Sampler' object has no attribute 'random_weights_'")

    @random_weights_.setter
    def random_weights_(self, value):
        self.__dict__.pop("_schi2_z", None)
        self.__dict__["_random_weights"] = value

    def transform(self, X):
        X = _f32(X)
        n, d = X.shape
        if X.size and X.min() <= -float(self.skewedness):
            raise ValueError("X may not contain entries smaller than -skewedness.")
        nc = int(self.n_components)
        z = self.__dict__.get("_schi2_z")
        if (z is not None or _kfeat_flags(self) & 4) and d == self.__dict__.get("n_features_in_"):
            # SCHI2_LAZYW: log, the pending weights, the map: one call, one wait
            w = _empty_out((d, nc), "<f4") if z is not None else self.random_weights_
            out = _empty_out((n, nc), "<f4")
            self._bind().x_neighbors_kfeat_schi2_transform(
                [n, d, nc, 0 if z is None else 1],
                [addr_ro(X, name="X"), addr_ro(w if z is None else z, name="random_weights_"),
                 0 if z is None else addr(w, name="random_weights_"),
                 addr_ro(self.random_offset_, name="random_offset_"), addr(out, name="schi2 output")],
                [_f32_scalar(self.skewedness)])
            if z is not None:
                self.random_weights_ = w
            return out
        lx = self._unary(X, _U_LOG, 1.0, _f32_scalar(self.skewedness))
        out = _empty_out((n, nc), "<f4")
        self._op("skew_transform", [(lx, 0), (self.random_weights_, 0), (self.random_offset_, 0), (out, 1)], (n, d, nc))
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        return _prefixed_names(self, int(self.n_components))


# ====================================================================== LabelPropagation
class _LabelPropagationBase(_XNeighbors):
    """scikit-learn `semi_supervised/_label_propagation.py` (1.9.0): the dense
    RBF graph or compact kNN connectivity graph (including each row's own
    point as its first neighbor), the product / clamp iteration with their stopping
    rule (sum |L - L_prev| < tol, checked before each step), the final row
    normalization and `transduction_`. Callable kernels are refused. Float32
    where theirs is float64; the neighbor ties go to the lower index."""

    _variant = None

    def _compact_graph(self, idx, n_reference, variant):
        n, k = idx.shape
        cols = empty((n, k), "<i4")
        vals = empty((n, k), "<f4")
        self._op("lp_knn_graph", [(idx, 0), (cols, 1), (vals, 1)],
                 (n, n_reference, k, variant))
        return cols, vals, n_reference

    def _graph_product(self, G, labels):
        if not isinstance(G, tuple):
            return self._matmul(G, labels)
        cols, vals, m = G
        n, k = cols.shape
        c = labels.shape[1]
        out = _empty_out((n, c), "<f4")
        self._op("lp_knn_product", [(cols, 0), (vals, 0), (labels, 0), (out, 1)],
                 (n, m, k, c))
        return out

    def _graph_affinity(self, X):
        n = X.shape[0]
        if self.kernel == "rbf":
            return self._kernel(X, X, "rbf", self.gamma, 0.0, 0)
        if self.kernel == "knn":
            k = min(int(self.n_neighbors), n)
            _, idx = self._knn_sq(X, X, k, False)
            g = _empty_out((n, n), "<f4")
            self._op("knn_graph", [(idx, 0), (g, 1)], (n, n, k))
            return g
        raise NotImplementedError(f"{type(self).__name__}: kernel={self.kernel!r} is not implemented ('rbf' or 'knn')")

    def fit(self, X, y):
        X = _f32(X)
        n = X.shape[0]
        # the labels by the native encoder, then the one-hot rows, the
        # spreading rows and the unlabeled flags on the device
        # (`p2m_lp_labels`; lane pyglue-numeric: Python lists over the rows)
        from ._labels import encode_labels
        allc, codes = encode_labels(y)
        if codes.size != n:
            raise ValueError("X and y have different numbers of rows")
        skip = next((i for i, c in enumerate(allc) if c == -1 and not isinstance(c, str)), -1)   # glue: scan over the k class labels
        classes = [c for i, c in enumerate(allc) if i != skip]
        C = len(classes)
        ld = _empty_out((n, C), "<f4")
        ys = _empty_out((n, C), "<f4")
        unlabeled = empty((n,), "<i4")
        a = _f32_scalar(1.0 - float(self.alpha)) if self._variant != "propagation" else 1.0
        if n and C:
            self._op("p2m_lp_labels", [(codes, 0), (ld, 1), (ys, 1), (unlabeled, 1)], (n, C, skip), (a,))
        if self._variant == "propagation":
            ys = None
        if self.kernel == "knn":
            k = min(int(self.n_neighbors), n)
            _, idx = self._knn_sq(X, X, k, False)
            G = self._compact_graph(idx, n, 0 if self._variant == "propagation" else 1)
        else:
            G = self._build_graph(X)
        if ys is None:
            # propagation's static rows on the device: a labeled row keeps
            # its one-hot row, an unlabeled row (all zero in ld0) normalizes
            # to zero (`lp_clamp` with ld as both inputs)
            ystatic = _empty_out((n, C), "<f4")
            self._op("lp_clamp", [(ld, 0), (ld, 0), (unlabeled, 0), (ystatic, 1)], (n, C))
        else:
            ystatic = ys
        if isinstance(G, tuple) and self._fast_tier() and _lp_fast_resident(self):
            # lane/apple-fast-neighbors2: the loop below over the compact kNN
            # graph as ONE resident op (x_neighbors/iter_device.mojo
            # op_lp_iterate_knn), the graph uploaded once, the stopping sum
            # and test on the device
            cols, vals, _ = G
            info = empty((2,), "<i4")
            tol_bits = struct.unpack("<Q", struct.pack("<d", float(self.tol)))[0]
            self._op("lp_iterate_knn", [(cols, 0), (vals, 0), (ld, 1), (ystatic, 0), (unlabeled, 0), (info, 1)],
                     (n, cols.shape[1], C, int(self.max_iter), 0 if self._variant == "propagation" else 1,
                      tol_bits >> 32, tol_bits & 0xFFFFFFFF),
                     (_f32_scalar(self.alpha) if self._variant != "propagation" else 0.0,))
            n_iter = int(info.tolist()[0])
            return self._finish_fit(X, classes, ld, n, C, n_iter)
        if not isinstance(G, tuple):
            # The loop below as ONE resident op (x_neighbors/iter_device.mojo):
            # the same items in the same order, the graph uploaded once, tol
            # passed as its float64 bits so the stopping test is Python's.
            info = empty((2,), "<i4")
            tol_bits = struct.unpack("<Q", struct.pack("<d", float(self.tol)))[0]
            self._op("lp_iterate", [(G, 0), (ld, 1), (ystatic, 0), (unlabeled, 0), (info, 1)],
                     (n, C, int(self.max_iter), 0 if self._variant == "propagation" else 1,
                      tol_bits >> 32, tol_bits & 0xFFFFFFFF),
                     (_f32_scalar(self.alpha) if self._variant != "propagation" else 0.0,))
            n_iter = int(info.tolist()[0])
            return self._finish_fit(X, classes, ld, n, C, n_iter)
        prev = _empty_out((n, C), "<f4")
        s = _empty_out((1,), "<f4")
        n_iter = 0
        converged = False
        for it in range(int(self.max_iter)):
            n_iter = it
            self._op("absdiff_sum", [(ld, 0), (prev, 0), (s, 1)], (n * C,))
            if s.tolist()[0] < float(self.tol):
                converged = True
                break
            prev = ld
            nxt = self._graph_product(G, ld)
            out = _empty_out((n, C), "<f4")
            if self._variant == "propagation":
                self._op("lp_clamp", [(nxt, 0), (ystatic, 0), (unlabeled, 0), (out, 1)], (n, C))
            else:
                self._op("ls_clamp", [(nxt, 0), (ystatic, 0), (out, 1)], (n * C,), (_f32_scalar(self.alpha),))
            ld = out
        if not converged:
            n_iter += 1
        return self._finish_fit(X, classes, ld, n, C, n_iter)

    def _finish_fit(self, X, classes, ld, n, C, n_iter):
        final = _empty_out((n, C), "<f4")
        self._op("row_normalize", [(ld, 0), (final, 1)], (n, C))
        self.X_ = X
        self.classes_ = classes
        self.label_distributions_ = final
        self.n_iter_ = n_iter
        self.transduction_ = _argmax_labels(self, classes, final)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        Q = _f32(X)
        nq = Q.shape[0]
        n = self.X_.shape[0]
        if self.kernel == "knn":
            k = min(int(self.n_neighbors), n)
            _, idx = self._knn_sq(Q, self.X_, k, False)
            W = self._compact_graph(idx, n, 2)
        else:
            W = self._kernel(Q, self.X_, "rbf", self.gamma, 0.0, 0)
        P = self._graph_product(W, self.label_distributions_)
        out = empty(P.shape, "<f4")
        self._op("row_normalize", [(P, 0), (out, 1)], P.shape)
        return out

    def predict(self, X):
        return _argmax_labels(self, self.classes_, self.predict_proba(X))

    def score(self, X, y, sample_weight=None):
        return _accuracy(y, self.predict(X), sample_weight)


class LabelPropagation(_LabelPropagationBase):
    """Label propagation (hard clamping). See `_LabelPropagationBase`."""

    _variant = "propagation"

    def __init__(self, kernel="rbf", *, gamma=20, n_neighbors=7, max_iter=1000, tol=1e-3, n_jobs=None):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs

    def _build_graph(self, X):
        A = self._graph_affinity(X)
        G = empty(A.shape, "<f4")
        self._op("row_normalize", [(A, 0), (G, 1)], A.shape)
        return G


class LabelSpreading(_LabelPropagationBase):
    """Label spreading (the normalized graph Laplacian, soft clamping by
    alpha). See `_LabelPropagationBase`."""

    _variant = "spreading"

    def __init__(self, kernel="rbf", *, gamma=20, n_neighbors=7, alpha=0.2, max_iter=30, tol=1e-3, n_jobs=None):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        return super().fit(X, y)

    def _build_graph(self, X):
        A = self._graph_affinity(X)
        n = A.shape[0]
        G = _empty_out((n, n), "<f4")
        if _OLD_ITEMS:
            self._op("ls_laplacian", [(A, 0), (G, 1)], (n,))
        else:
            deg = _empty_out((n,), "<f4")
            self._op("col_degree", [(A, 0), (deg, 1)], (n,))
            self._op("ls_laplacian_deg", [(A, 0), (deg, 0), (G, 1)], (n,))
        return G


# ====================================================================== KNNImputer
class KNNImputer(_XNeighbors):
    """Imputation of missing values by k nearest neighbors.

    Reference: scikit-learn `impute/_knn.py` (1.9.0): nan_euclidean distances
    to the fit rows, donors = fit rows where the column is present, the k
    nearest (ties: the lower row index), 'uniform' or 'distance' weights,
    the masked column mean when no donor has a finite distance, all-missing
    columns dropped (or zero with keep_empty_features), `add_indicator`,
    `get_feature_names_out`. A numeric missing_values other than NaN is
    replaced by NaN before the Mojo call (an exact equality test), so it
    imputes as sklearn's mask does. metric 'nan_euclidean' only; callable
    metrics and weights are refused by name (a user arithmetic outside the
    pinned items).
    """

    def __init__(self, *, missing_values=float("nan"), n_neighbors=5, weights="uniform",
                 metric="nan_euclidean", copy=True, add_indicator=False, keep_empty_features=False):
        self.missing_values = missing_values
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def _check(self):
        mv = self.missing_values
        import numbers
        if mv is None or isinstance(mv, (str, bool)) or not isinstance(mv, numbers.Real):
            raise NotImplementedError("KNNImputer: missing_values must be a number or NaN (float input)")
        if self.metric != "nan_euclidean":
            raise NotImplementedError("KNNImputer: metric must be 'nan_euclidean'")
        if self.weights not in ("uniform", "distance"):
            raise NotImplementedError("KNNImputer: weights must be 'uniform' or 'distance'")

    def _nan_cells(self, X, colmiss_only=0):
        """(cells, colmiss, count): the flat indices of the NaN cells of X
        ascending (an int32 Array of n * d slots, the first `count` used), the
        NaN count per column as a list, and the count (xn_nan_cells).
        colmiss_only=1 (fit): a build with MOJOLEARN_XN_FAST_NAN_COLMISS_ONLY
        may skip the cell list; the counts are the same integers."""
        n, d = X.shape
        # FAST Apple default MOJOLEARN_XN_FAST_NAN_FIT_LEAN_GPU (_OFF rollback): a binary
        # whose fit-time op ignores the cell list gets a one-slot one (no
        # n * d allocation)
        lean = colmiss_only and int(getattr(self._bind(), "x_neighbors_nc_fit_lean", lambda: 0)())
        cells = empty((1 if lean else max(n * d, 1),), "<i4")
        colmiss = empty((max(d, 1),), "<i4")
        info = empty((1,), "<i4")
        self._op("nan_cells", [(X, 0), (cells, 1), (colmiss, 1), (info, 1)], (n, d, colmiss_only))
        return cells, colmiss.tolist(), int(info.tolist()[0])

    def _masked(self, X):
        X = _f32(X)
        mv = self.missing_values
        if mv == mv and X.size:           # a number: its cells become NaN
            out = empty(X.shape, "<f4")
            self._op("p2m_mask_value", [(X, 0), (out, 1)], (X.size,), (_f32_scalar(mv),))
            X = out
        return X

    def fit(self, X, y=None):
        self._check()
        X = self._masked(X)
        n, d = X.shape
        # lane/neural-pass71 (2026-10-01): the column flags from one native
        # pass over the cells (xn_nan_cells), no list of the matrix
        cm = self._nan_cells(X, 1)[1]
        self._valid = [cm[f] < n for f in range(d)]
        self._miss_cols = [f for f in range(d) if cm[f] > 0]
        self._fit_X = X
        self.n_features_in_ = d
        return self

    def transform(self, X):
        X = self._masked(X)
        n, d = X.shape
        if d != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        m = self._fit_X.shape[0]
        out = _empty_out((n, d), "<f4")
        k = int(self.n_neighbors)
        if k < 1:
            raise ValueError("n_neighbors must be >= 1")
        if _UNCOMPACT_IMPUTE:
            self._op("knn_impute", [(X, 0), (self._fit_X, 0), (out, 1)],
                     (n, m, d, k, 1 if self.weights == "distance" else 0))
        else:
            # one GPU thread per MISSING cell (`knn_impute_cells`): the same
            # item statements; a present cell keeps x, as the item stores it
            cells, _, nc = self._nan_cells(X)        # lane/neural-pass71: no Python walk
            out = X.copy()
            if nc:
                self._op("knn_impute_cells" if _OLD_ITEMS else "knn_impute_tiled",
                         [(cells, 0), (X, 0), (self._fit_X, 0), (out, 1)],
                         (n, m, d, k, 1 if self.weights == "distance" else 0, nc))
        keep = [f for f in range(d) if self._valid[f]]
        if self.keep_empty_features:
            if n and not all(self._valid):
                flags = _i32([0 if v else 1 for v in self._valid], "flags")   # glue: d fit flags
                z = _empty_out((n, d), "<f4")
                self._op("p2m_zero_cols", [(out, 0), (flags, 0), (z, 1)], (n, d))
                out = z
        elif len(keep) != d:
            out = self._take_cols(out, keep)
        if self.add_indicator and self._miss_cols and n:
            c = out.shape[1]
            q = len(self._miss_cols)
            res = _empty_out((n, c + q), "<f4")
            self._op("p2m_nan_indicator",
                     [(X, 0), (out if c else X, 0), (_i32(self._miss_cols, "cols"), 0), (res, 1)], (n, d, c, q))
            out = res
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        names = _feature_names_in(self, input_features)
        out = [nm for f, nm in enumerate(names) if self._valid[f] or self.keep_empty_features]
        if self.add_indicator:
            out += [f"missingindicator_{names[f]}" for f in self._miss_cols]
        return _names_out(out)


# ====================================================================== PageRank
def _adjacency(A):
    A = _f32(A, "adjacency")
    if A.shape[0] != A.shape[1]:
        raise ValueError("the adjacency matrix must be square")
    return A


class PageRank(_XNeighbors):
    """PageRank by power iteration on a dense weighted adjacency matrix
    (A[i, j] = weight of the edge i -> j).

    References: networkx `pagerank` (`_pagerank_scipy`: row-stochastic
    transition matrix, dangling nodes redistributed by the personalization,
    stop when sum |x - x_last| < n * tol) and cuGraph
    cpp/src/link_analysis/pagerank_impl.cuh. Each step is the pinned-fold
    GEMV of x_neighbors/items.mojo `pagerank_step_item`. `personalization`
    is a length-n vector (normalized to sum 1) or None (uniform); `nstart`
    (the starting vector, normalized) and `dangling` (where a dangling node's
    mass goes, normalized; the personalization when None) are length-n
    vectors, as networkx's dicts over nodes 0..n-1. `weight` False (or None,
    networkx's unweighted spelling) makes every edge weight 1; the dense
    matrix carries the weights, so no attribute name is needed.
    Non-convergence raises, as networkx's PowerIterationFailedConvergence.
    """

    def __init__(self, alpha=0.85, *, personalization=None, max_iter=100, tol=1e-6, nstart=None, weight=True,
                 dangling=None):
        self.alpha = alpha
        self.personalization = personalization
        self.max_iter = max_iter
        self.tol = tol
        self.nstart = nstart
        self.weight = weight
        self.dangling = dangling

    @staticmethod
    def _unit(v, n, what):
        """A caller's length-n non-negative vector divided by its sum (IEEE
        double, rounded once to float32), as networkx normalizes its dicts."""
        pv = [float(t) for t in (v.tolist() if hasattr(v, "tolist") else v)]
        if len(pv) != n or any(t < 0 for t in pv) or math.fsum(pv) == 0:
            raise ValueError(f"{what} must be n non-negative values, not all zero")
        tot = math.fsum(pv)
        return Array.from_list([t / tot for t in pv], "<f4")

    def fit(self, A, y=None):
        A = _adjacency(A)
        n = A.shape[0]
        binary = self.weight is None or self.weight is False
        # the iteration over the column lists of the adjacency, built
        # and iterated on the device (lane hr-graph,
        # x_neighbors/graph_par.mojo): the dense chains with their zero
        # terms left out, the dangling mass and |x' - x| blocked folds

        def uniform():
            # [1/n] * n: 1/n in IEEE double rounded once to float32 (the
            # binding's float argument), filled on the device
            u = empty((n,), "<f4")
            if n:
                self._op("p2m_fill", [(u, 1)], (n,), (1.0 / n,))
            return u

        if self.personalization is None:
            p = uniform()
        else:
            p = self._unit(self.personalization, n, "personalization")
        dw = p if self.dangling is None else self._unit(self.dangling, n, "dangling")
        x = uniform() if self.nstart is None else self._unit(self.nstart, n, "nstart")
        x = x.copy()
        info = empty((2,), "<i4")
        thr = struct.unpack("<Q", struct.pack("<d", n * float(self.tol)))[0]
        self._op("pr_iterate_sparse", [(A, 0), (x, 1), (p, 0), (dw, 0), (info, 1)],
                 (n, int(self.max_iter), thr >> 32, thr & 0xFFFFFFFF, 1 if binary else 0),
                 (_f32_scalar(self.alpha),))
        it, ok = info.tolist()
        if ok:
            self.pagerank_ = x
            self.n_iter_ = int(it)
            return self
        raise RuntimeError(f"PageRank: power iteration failed to converge within {self.max_iter} iterations")


# ====================================================================== connected components
def connected_components(A, directed=True, connection="weak", return_labels=True, numeric_mode=None):
    """(n_components, labels) of the graph with dense adjacency A, as
    `scipy.sparse.csgraph.connected_components`: an edge is any nonzero
    entry, `connection='weak'` ignores direction (for an undirected graph the
    two agree); labels are numbered by the lowest node of each component, so
    component 0 holds node 0 (scipy's numbering). The labels come from the
    min-label product iteration (DBSCAN's weak_cc; cuGraph
    weakly_connected_components_impl.cuh), integers only. connection='strong'
    on a directed graph is refused by name."""
    if directed and connection == "strong":
        raise NotImplementedError("connected_components: connection='strong' is not implemented")
    if connection not in ("weak", "strong"):
        raise ValueError("connection must be 'weak' or 'strong'")
    est = _XNeighbors()
    est.numeric_mode = numeric_mode
    csr = _csr_of(A)
    if csr is not None:
        # lane/neural-pass69 (2026-10-01): a sparse graph (scipy.sparse CSR, or
        # (indptr, indices, n)) walks its edge lists directly: the same rounds
        # and labels as the dense matrix's, without building or scanning it
        indptr, indices, n = csr
        lab = _p2m_iota(est, n)
        info = empty((1,), "<i4")
        est._op("cc_iterate_csr", [(indptr, 0), (indices, 0), (lab, 1), (info, 1)], (n, indices.shape[0]))
        return _cc_relabel(lab, return_labels, est)
    A = _adjacency(A)
    n = A.shape[0]
    lab = _p2m_iota(est, n)
    # the min-label rounds as ONE resident op, A uploaded once
    info = empty((1,), "<i4")
    est._op("cc_iterate", [(A, 0), (lab, 1), (info, 1)], (n,))
    return _cc_relabel(lab, return_labels, est)


def _p2m_iota(est, n):
    """[0, 1, ..., n - 1] as int32 (`xn_p2m_iota` on the device)."""
    lab = empty((max(n, 0),), "<i4")
    if n > 0:
        est._op("p2m_iota", [(lab, 1)], (n,))
    return lab


def _cc_relabel(lab, return_labels, est):
    """(count, labels) renumbered by first occurrence (`xn_p2m_relabel`)."""
    if not lab.shape[0]:
        return (0, lab) if return_labels else 0
    got = _p2m_relabel(est, lab)
    if got is None:
        raise RuntimeError("connected_components: a component label fell outside [0, n)")
    cnt, labels = got
    return (cnt, labels) if return_labels else cnt


def _csr_of(A):
    """(indptr, indices, n) as int32 Arrays for a scipy.sparse matrix (any
    format: converted to CSR) or a tuple (indptr, indices, n); None for a
    dense input. The edge weights do not matter to connectivity."""
    if isinstance(A, tuple) and len(A) == 3:
        indptr, indices, n = A
        n = int(n)
    elif hasattr(A, "tocsr") and hasattr(A, "shape"):
        if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("the adjacency matrix must be square")
        M = A.tocsr()
        indptr, indices, n = M.indptr, M.indices, int(M.shape[0])
    else:
        return None
    ip = _i32(indptr, "indptr")
    ix = _i32(indices, "indices")
    if ip.shape[0] != n + 1:
        raise ValueError("connected_components: indptr must hold n + 1 entries")
    return ip, ix, n


# ====================================================================== Louvain
class Louvain(_XNeighbors):
    """Louvain community detection on a dense symmetric weighted adjacency.

    References: networkx `louvain_communities` / `louvain_partitions`
    (`_one_level`, `_gen_graph`, `modularity`) and cuGraph
    cpp/src/community/louvain_impl.cuh. Parallel local moving in a PINNED
    order (x_neighbors/graph_par.mojo, DEVIATION 5204): the nodes of one
    colour of a fixed graph colouring move together instead of networkx's
    `seed` shuffle, candidate communities in ascending id, a strictly larger
    gain to move, so ties go to the lowest community id; community totals
    and the aggregation are fixed-order folds. `labels_` numbers the
    communities by their lowest node; `modularity_` is networkx's
    modularity of that partition, in float32. `seed` is accepted and unused.
    """

    def __init__(self, resolution=1.0, *, threshold=1e-7, max_level=None, seed=None):
        self.resolution = resolution
        self.threshold = threshold
        self.max_level = max_level
        self.seed = seed

    def fit(self, A, y=None):
        A = _adjacency(A)
        n = A.shape[0]
        # The symmetry and no-edge checks run natively (`xn_graph_symmetry`,
        # lane neural-pass14): over `A.tolist()` they were a 400-million-cell
        # Python scan at the board's 20,000 nodes, most of the race's minute.
        flags = _empty_out((2,), "<i4")
        self._op("graph_symmetry", [(A, 0), (flags, 1)], (n,))
        flags = flags.tolist()
        if flags[0]:
            raise ValueError("Louvain: the adjacency matrix must be symmetric (an undirected graph)")
        if not flags[1]:
            raise ValueError("Louvain: the graph has no edges")
        labels = empty((n,), "<i4")
        info = _empty_out((2,), "<f4")
        ml = 0 if self.max_level is None else int(self.max_level)
        if self.max_level is not None and ml < 1:
            raise ValueError("max_level must be a positive integer or None")
        self._op("louvain", [(A, 0), (labels, 1), (info, 1)], (n, ml),
                 (_f32_scalar(self.resolution), _f32_scalar(self.threshold)))
        got = _p2m_relabel(self, labels)
        if got is None:
            raise RuntimeError("Louvain: a community label fell outside [0, n)")
        self.n_communities_, self.labels_ = got
        info = info.tolist()
        self.modularity_ = info[0]
        self.n_levels_ = int(info[1])
        return self

    def fit_predict(self, A, y=None):
        return self.fit(A).labels_


# ====================================================================== SVGP
class SVGP(_XNeighbors):
    """Sparse variational Gaussian process regression with inducing points.

    Reference: GPflow `gpflow/models/svgp.py` (SVGP with a Gaussian
    likelihood, a squared-exponential kernel, zero mean). The variational
    distribution q(u) = N(q_mu, q_sqrt q_sqrt^T) is set to its OPTIMUM for
    the given hyperparameters (Titsias 2009), where GPflow's `elbo` equals the
    collapsed bound reported as `elbo_`; hyperparameters are the caller's,
    not optimized (DEVIATION 5205: GPflow trains them and q by gradient
    steps). q_mu / q_sqrt are the non-whitened parameters. Inducing points:
    `inducing_points`, or `n_inducing` training rows evenly spaced
    (row i * n // M). The m x m system runs as staged items
    (x_neighbors/items.mojo `svgp_*_item`: a Cholesky column per launch, one
    triangular solve per right-hand side); the kernel matrices and products
    are parallel items. Float32 throughout.
    """

    def __init__(self, n_inducing=32, *, inducing_points=None, kernel_variance=1.0, lengthscale=1.0,
                 noise_variance=1.0, jitter=1e-6):
        self.n_inducing = n_inducing
        self.inducing_points = inducing_points
        self.kernel_variance = kernel_variance
        self.lengthscale = lengthscale
        self.noise_variance = noise_variance
        self.jitter = jitter

    def _gamma_value(self):
        ls = float(self.lengthscale)
        return 1.0 / (2.0 * (ls * ls))  # ls ** 2 as one product, not the platform pow

    def _k(self, A, B):
        g = self._gamma_value()
        K = self._kernel(A, B, "rbf", g, 0.0, 0)
        return self._unary(K, _U_IDENTITY, _f32_scalar(self.kernel_variance), 0.0)

    def fit(self, X, y):
        X = _f32(X)
        n, d = X.shape
        yv = _f32_1d(y, "y")
        if len(yv) != n:
            raise ValueError("X and y have different numbers of rows")
        if self.inducing_points is not None:
            Z = _f32(self.inducing_points, "inducing_points")
        else:
            M = min(int(self.n_inducing), n)
            Z = self._take_rows(X, [i * n // M for i in range(M)])
        M = Z.shape[0]
        # one resident device chain (lane/cgr-kernel, `xn_svgp_fit_ff`):
        # Kuu, then B = Kuf Kfu and b = Kuf y in float-float per Kfu row tile
        # (Kuf is Kfu^T bit for bit, never formed), then the float-float
        # solve (x_neighbors/svgp_ff.mojo). Taxi's Sigma (eigenvalues 1e-6 ..
        # 9e5) defeats float32: B's float32 accumulation error alone exceeds
        # its smallest eigenvalue. Kuu, B and b never leave the device.
        alpha = _empty_out((M,), "<f4")
        C = _empty_out((M, M), "<f4")
        qmu = _empty_out((M,), "<f4")
        qsqrt = _empty_out((M, M), "<f4")
        info = _empty_out((2,), "<f4")
        self._op("svgp_fit_ff", [(X, 0), (Z, 0), (yv, 0), (alpha, 1), (C, 1), (qmu, 1), (qsqrt, 1), (info, 1)],
                 (n, M, d),
                 (_f32_scalar(self._gamma_value()), _f32_scalar(self.kernel_variance),
                  _f32_scalar(self.noise_variance), _f32_scalar(self.jitter), _f32_scalar(self.kernel_variance)))
        elbo, ok = info.tolist()
        self.precision_ = "float-float"
        if ok == 0:
            raise ValueError("SVGP: the inducing system is not positive definite; raise jitter or noise_variance")
        self.Z_, self._alpha, self._C = Z, alpha, C
        self.q_mu_ = qmu
        self.q_sqrt_ = qsqrt
        self.elbo_ = elbo
        self.n_features_in_ = d
        return self

    def predict_f(self, X):
        Q = _f32(X)
        M = self.Z_.shape[0]
        if not _OLD_ITEMS and Q.shape[0]:
            # the fused chain (lane/py-dn-kern): Ksu, the mean and the
            # variance per row tile on the device (`xn_svgp_predict`)
            nq = Q.shape[0]
            mean = _empty_out((nq,), "<f4")
            var = _empty_out((nq,), "<f4")
            self._op("svgp_predict", [(Q, 0), (self.Z_, 0), (self._alpha, 0), (self._C, 0), (mean, 1), (var, 1)],
                     (nq, M, Q.shape[1]),
                     (_f32_scalar(self._gamma_value()), _f32_scalar(self.kernel_variance),
                      _f32_scalar(self.kernel_variance)))
            return mean, var
        Ksu = self._k(Q, self.Z_)
        mean = self._matmul(Ksu, self._alpha.reshape((M, 1))).reshape((Q.shape[0],))
        var = _empty_out((Q.shape[0],), "<f4")
        self._op("svgp_var", [(Ksu, 0), (self._C, 0), (var, 1)], (Q.shape[0], M), (_f32_scalar(self.kernel_variance),))
        return mean, var

    def predict_y(self, X):
        mean, var = self.predict_f(X)
        return mean, self._unary(var, _U_IDENTITY, 1.0, _f32_scalar(self.noise_variance))

    def predict(self, X):
        return self.predict_f(X)[0]
