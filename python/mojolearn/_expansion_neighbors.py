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
from ._mode import NumericModeMixin, available_if
from ._mode import RegressorScoreMixin

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
        return [f"x{i}" for i in range(d)]  # glue: feature names over the d columns
    names = [str(v) for v in input_features]  # glue: feature names over the d columns
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
    return _names_out([f"{base}{i}" for i in range(count)])  # glue: feature names over the d columns


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
    if all(isinstance(v, (bool, int)) for v in classes):  # glue: type check over the k class labels
        return decode_labels([int(v) for v in classes], codes)   # glue: the k class values
    if all(isinstance(v, (int, float)) for v in classes):  # glue: type check over the k class labels
        return decode_labels([float(v) for v in classes], codes)  # glue: k class labels as floats
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
# TOMBSTONE: MOJOLEARN_XN_FAST_TILED_RBF (DROPPED-noise) deleted 2026-10-03 by f103d7381; code recoverable at f103d7381^.
# Restore: git apply experiments/removed/MOJOLEARN_XN_FAST_TILED_RBF.patch; record in docs/TOMBSTONES.md.
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


def _classical_graph_normalization(est):
    """C43 binary capability, IDENTICAL only and default OFF.

    NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    """
    flag = getattr(est._bind(), "x_neighbors_classical_graph_normalization", None)
    return flag is not None and bool(flag())


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
                 for a, w in bufs]  # glue: buffer addresses and binding params
        getattr(self._bind(), "xn_" + name)(addrs, [int(v) for v in ints], [float(v) for v in floats])  # glue: buffer addresses and binding params

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
# Reference: scikit-learn `neighbors/_lof.py` (1.9.0): `fit` (k = min(
# n_neighbors, n_samples - 1), `kneighbors` on the training set without the
# sample itself), `_local_reachability_density`, `negative_outlier_factor_`
# = -mean(lrd[neighbors] / lrd), `offset_` (-1.5 at contamination='auto',
# else the contamination percentile), `score_samples` / `decision_function`
# / `predict` under novelty=True, `fit_predict` under novelty=False.
class LocalOutlierFactor(_XNeighbors):
    """Unsupervised outlier detection by the local outlier factor (LOF).

    Scores each sample by how isolated it is relative to its k nearest
    neighbors, following scikit-learn's ``LocalOutlierFactor``. Neighbor
    searches are exact brute force with the euclidean metric; ties between
    equal distances go to the lower training index, and a duplicate of a
    training row counts as its neighbor. Computation is float32 (scikit-learn
    uses float64).

    With ``novelty=False`` (the default) use :meth:`fit_predict` to label the
    training data. With ``novelty=True`` fit on clean data and call
    :meth:`predict`, :meth:`decision_function` or :meth:`score_samples` on new
    data.

    Parameters
    ----------
    n_neighbors : int, default 20
        Number of neighbors. ``fit`` uses ``min(n_neighbors, n_samples - 1)``.
    algorithm : str, default 'auto'
        Accepted for scikit-learn compatibility; every search is exact brute
        force.
    leaf_size : int, default 30
        Accepted for scikit-learn compatibility; unused.
    metric : str, default 'minkowski'
        ``'euclidean'``, or ``'minkowski'`` with ``p=2``. Any other metric
        raises ``NotImplementedError``.
    p : int, default 2
        Minkowski power; only 2 is supported.
    metric_params : dict or None, default None
        Not supported; a non-empty value raises ``NotImplementedError``.
    contamination : 'auto' or float, default 'auto'
        Expected outlier fraction, in (0, 0.5]. ``'auto'`` sets ``offset_`` to
        -1.5; a float sets it to that percentile of the training scores.
    novelty : bool, default False
        Whether the model scores new data (``predict``, ``decision_function``,
        ``score_samples``) instead of the training data (``fit_predict``).
    n_jobs : int or None, default None
        Accepted for scikit-learn compatibility; unused.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator (``'fast'`` is Apple only). None uses
        the process default.

    Attributes
    ----------
    negative_outlier_factor_ : Array of shape (n_samples,)
        Negative LOF of each training sample; lower is more abnormal.
    offset_ : float
        Threshold on the score; samples below it are outliers.
    n_neighbors_ : int
        Number of neighbors actually used.
    n_features_in_ : int
        Number of features seen in ``fit``.
    n_samples_fit_ : int
        Number of training samples.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> labels = mojolearn.LocalOutlierFactor(n_neighbors=20).fit_predict(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "outlier_detector"

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
        """Fit the model on the training data.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data; at least 2 samples. Sparse input is densified.
        y : None
            Ignored.

        Returns
        -------
        self : LocalOutlierFactor
            The fitted estimator.
        """
        self._check_params()
        X = _f32(X)
        n, d = X.shape
        if n < 2:
            # scikit-learn's check_array wording (ensure_min_samples=2)
            raise ValueError(f"Found array with {n} sample(s) (shape=({n}, {d})) while a minimum of 2 is required.")
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

    @available_if(lambda self: not self.novelty)
    def fit_predict(self, X, y=None):
        """Fit the model and label the training samples.

        Available only with ``novelty=False``; otherwise raises
        ``AttributeError``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        labels : Array of shape (n_samples,), int64
            -1 for outliers, 1 for inliers.
        """
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

    @available_if(lambda self: self.novelty)
    def score_samples(self, X):
        """Negative local outlier factor of new samples.

        Available only with ``novelty=True``; otherwise raises
        ``AttributeError``. Lower scores are more abnormal.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            New samples.

        Returns
        -------
        scores : Array of shape (n_queries,), float32
        """
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

    @available_if(lambda self: self.novelty)
    def decision_function(self, X):
        """Shifted outlier score of new samples: ``score_samples(X) - offset_``.

        Available only with ``novelty=True``. Negative values are outliers.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            New samples.

        Returns
        -------
        scores : Array of shape (n_queries,), float32
        """
        self._novelty("decision_function")
        return self._unary(self.score_samples(X), _U_IDENTITY, 1.0, -self.offset_)

    @available_if(lambda self: self.novelty)
    def predict(self, X):
        """Label new samples as inliers or outliers.

        Available only with ``novelty=True``; otherwise raises
        ``AttributeError``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            New samples.

        Returns
        -------
        labels : Array of shape (n_queries,), int64
            -1 for outliers, 1 for inliers.
        """
        self._novelty("predict")
        dec = self.decision_function(X)
        lab = empty((dec.size,), "<i4")
        if dec.size:
            self._op("p2m_sign_label", [(dec, 0), (lab, 1)], (dec.size, 1), (0.0,))
        return lab.astype("<i8")


# ====================================================================== NearestCentroid
# Reference: scikit-learn `neighbors/_nearest_centroid.py` (1.9.0): per-class
# mean (euclidean) or per-class median (manhattan) centroids,
# `within_class_std_dev_`, the shrunken centroids (`shrink_threshold`),
# `class_prior_` ('uniform', 'empirical' or given), `predict` (the nearest
# centroid when the priors are uniform, else the discriminant),
# `decision_function` and `predict_proba` (euclidean only, as theirs).
# `deviations_`, `predict_log_proba` (their log-softmax of the
# discriminant) and `score`. DEVIATION 5201: a feature whose shrink
# scale m*s is zero gets deviation 0 where theirs divides by zero.
class NearestCentroid(_XNeighbors):
    """Nearest centroid classifier.

    Represents each class by its centroid (the per-class mean for
    ``metric='euclidean'``, the per-class median for ``'manhattan'``) and
    assigns each sample to the nearest one, following scikit-learn's
    ``NearestCentroid``, including shrunken centroids and class priors. With
    non-uniform priors, ``predict`` uses the discriminant of
    :meth:`decision_function`. Sparse input is densified. Computation is
    float32 (scikit-learn uses float64). A feature whose shrink scale is zero
    gets deviation 0 (scikit-learn divides by zero there).

    Parameters
    ----------
    metric : {'euclidean', 'manhattan'}, default 'euclidean'
        Distance used for centroids and prediction; anything else raises
        ``ValueError``.
    shrink_threshold : float or None, default None
        Threshold for shrinking centroids toward the overall centroid; None
        disables shrinking.
    priors : {'uniform', 'empirical'} or array-like of shape (n_classes,), default 'uniform'
        Class priors. ``'empirical'`` uses the class frequencies. Given priors
        must be non-negative, one per class, and are rescaled to sum to one.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator (``'fast'`` is Apple only). None uses
        the process default.

    Attributes
    ----------
    centroids_ : Array of shape (n_classes, n_features)
        Class centroids, after shrinking.
    deviations_ : Array of shape (n_classes, n_features)
        Shrunken deviations of each class centroid from the overall centroid.
    within_class_std_dev_ : Array of shape (n_features,)
        Pooled within-class standard deviation of each feature.
    class_prior_ : Array of shape (n_classes,), float64
        Class priors.
    classes_ : list
        Class labels.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > 0.5).astype(np.int64)
    >>> clf = mojolearn.NearestCentroid().fit(X, y)
    >>> pred = clf.predict(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "classifier"

    def __init__(self, metric="euclidean", *, shrink_threshold=None, priors="uniform"):
        self.metric = metric
        self.shrink_threshold = shrink_threshold
        self.priors = priors

    def fit(self, X, y):
        """Fit the class centroids.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Class labels; at least two classes.

        Returns
        -------
        self : NearestCentroid
            The fitted estimator.

        Raises
        ------
        ValueError
            If there are fewer than two classes, the priors are invalid, or
            every feature has zero variance.
        """
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
            # count / n per class in the base binding (lane py-runtime-b;
            # the same binary64 words as the Python quotient)
            from ._buffer import _native
            pa = empty((C,), "<f8")
            ca = Array.from_list(counts, "<i8")
            _native("class_ratio_f64")(addr_ro(ca, name="counts"), C, n, 1, addr(pa, name="class_prior_"))
            prior = pa.tolist()  # glue: k class priors for the checks below
        elif self.priors == "uniform":
            prior = [1.0 / C] * C
        else:
            prior = [float(v) for v in (self.priors.tolist() if hasattr(self.priors, "tolist") else self.priors)]  # glue: user priors argument checks
            if len(prior) != C:
                raise ValueError("priors must have one entry per class")
            if any(p < 0 for p in prior):  # glue: user priors argument checks
                raise ValueError("priors must be non-negative")
            tot = math.fsum(prior)  # glue: the native fsum (core fsum_f64) of the user priors argument
            if not math.isclose(tot, 1.0, rel_tol=1e-5, abs_tol=1e-8):
                prior = [p / tot for p in prior]  # glue: the user priors argument rescaled to sum one (k sized)
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
            for c in range(C):  # glue: class offsets prefix over k classes
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
        return all(math.isclose(p, 1.0 / C, rel_tol=1e-5, abs_tol=1e-8) for p in self.class_prior_.tolist())  # glue: uniform prior check over k classes

    def predict(self, X):
        """Predict the class of each sample.

        With uniform priors this is the class of the nearest centroid;
        otherwise the class with the largest :meth:`decision_function` value.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples to classify.

        Returns
        -------
        labels : Array or list of shape (n_queries,)
            Predicted class labels.
        """
        Q = _f32(X)
        if self._uniform():
            D = self._sqdist(Q, self.centroids_) if self.metric == "euclidean" else self._l1dist(Q, self.centroids_)
            _, idx = self._knn_select(D, 1, False)
            # TOMBSTONE: MOJOLEARN_NC_FAST_CLS1_PREDICT (DROPPED-noise) deleted 2026-10-03 by 9decae29f; code recoverable at 9decae29f^.
            # Restore: git apply experiments/removed/MOJOLEARN_NC_FAST_CLS1_PREDICT.patch; record in docs/TOMBSTONES.md.
            return _class_array(self.classes_, idx.reshape((idx.shape[0],)))
        return _argmax_labels(self, self.classes_, self.decision_function(Q))

    def decision_function(self, X):
        """Discriminant score of each sample for each class.

        Available for ``metric='euclidean'`` only; otherwise raises
        ``AttributeError``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples to score.

        Returns
        -------
        scores : Array of shape (n_queries, n_classes), float32
        """
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
        """Class probabilities: the softmax of :meth:`decision_function`.

        Available for ``metric='euclidean'`` only.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples to score.

        Returns
        -------
        proba : Array of shape (n_queries, n_classes)
            Probabilities, columns in the order of ``classes_``.
        """
        dec = self.decision_function(X)
        if hasattr(self._bind(), "xn_softmax64"):
            # lane apple-fast-q-clf (x_neighbors/proba64_nc.mojo, QUALITY-FIX, FAST
            # default; -D MOJOLEARN_PROBA64_QOLD has no export): float64 probabilities,
            # the winner's 1 - c exact (float32 saturated at 1.0: Istella log loss
            # 4.299 vs scikit-learn 4.118, accuracy equal)
            out = empty(dec.shape, "<f8")
            self._op("softmax64", [(dec, 0), (out, 1)], dec.shape)
            return out
        out = empty(dec.shape, "<f4")
        self._op("softmax", [(dec, 0), (out, 1)], dec.shape)
        return out

    def predict_log_proba(self, X):
        """Log class probabilities: the log-softmax of :meth:`decision_function`.

        Available for ``metric='euclidean'`` only.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples to score.

        Returns
        -------
        log_proba : Array of shape (n_queries, n_classes), float32
        """
        dec = self.decision_function(X)
        out = empty(dec.shape, "<f4")
        self._op("log_softmax", [(dec, 0), (out, 1)], dec.shape)
        return out

    def score(self, X, y, sample_weight=None):
        """Mean accuracy of :meth:`predict` on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,) or None, default None
            Sample weights.

        Returns
        -------
        score : float
            (Weighted) fraction of correct predictions.
        """
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
# Reference: scikit-learn `svm/_classes.py` (OneClassSVM) over libsvm
# `svm.cpp` (`solve_one_class`, `Solver::Solve` with WSS3 working-set
# selection, `calculate_rho`). On the GPU each SMO iteration is three grid
# launches (x_neighbors/ocsvm_dev.mojo); the host column runs the same
# order as one item (x_neighbors/items.mojo `ocsvm_smo_item`) over the kernel matrix,
# in float32 with the pinned spellings (DEVIATION 5200; libsvm is double).
# `sample_weight` is libsvm's per-sample bound C_i (remove_zero_weight first).
# Callable kernels are refused (a user arithmetic outside the pinned items).
class OneClassSVM(_XNeighbors):
    """Unsupervised outlier detection with the nu one-class SVM.

    Learns a boundary around the training data with libsvm's SMO solver for
    the one-class problem, following scikit-learn's ``OneClassSVM``.
    Computation is float32 (libsvm uses float64). No shrinking heuristic is
    used and the whole kernel matrix is formed, so ``shrinking`` and
    ``cache_size`` change nothing.

    Parameters
    ----------
    kernel : {'rbf', 'linear', 'poly', 'sigmoid', 'precomputed'}, default 'rbf'
        Kernel. With ``'precomputed'``, ``fit`` takes the n x n Gram matrix
        and later calls take the n_test x n_train kernel against the training
        rows. Callable kernels and other names raise ``NotImplementedError``.
    degree : int, default 3
        Degree of the ``'poly'`` kernel.
    gamma : {'scale', 'auto'} or float, default 'scale'
        Kernel coefficient. ``'scale'`` is ``1 / (n_features * X.var())``
        (1.0 when the variance is zero); ``'auto'`` is ``1 / n_features``.
    coef0 : float, default 0.0
        Independent term of the ``'poly'`` and ``'sigmoid'`` kernels.
    tol : float, default 1e-3
        Stopping tolerance of the solver.
    nu : float, default 0.5
        Upper bound on the fraction of training errors and lower bound on the
        fraction of support vectors, in (0, 1].
    shrinking : bool, default True
        Accepted for scikit-learn compatibility; unused.
    cache_size : float, default 200
        Accepted for scikit-learn compatibility; unused.
    verbose : bool, default False
        Accepted for scikit-learn compatibility; unused.
    max_iter : int, default -1
        Iteration limit of the solver; -1 means 10,000,000.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator (``'fast'`` is Apple only). None uses
        the process default.

    Attributes
    ----------
    support_ : Array of shape (n_SV,), int32
        Indices of the support vectors in the training data.
    support_vectors_ : Array of shape (n_SV, n_features)
        Support vectors (shape (n_SV, 0) for a precomputed kernel).
    dual_coef_ : Array of shape (1, n_SV)
        Coefficients of the support vectors.
    intercept_ : Array of shape (1,)
        Constant in the decision function.
    offset_ : float
        Offset subtracted from ``score_samples`` to give ``decision_function``.
    n_iter_ : int
        Solver iterations run.
    n_features_in_ : int
        Number of features seen in ``fit``.
    fit_status_ : int
        0 when fitted.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.OneClassSVM(nu=0.1).fit(X)
    >>> labels = model.predict(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "outlier_detector"

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
        """Fit the one-class SVM.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data, or the (n_samples, n_samples) Gram matrix for
            ``kernel='precomputed'``. Sparse input is densified.
        y : None
            Ignored.
        sample_weight : array-like of shape (n_samples,) or None, default None
            Per-sample weights that scale each sample's bound, as libsvm does.
            Samples of weight 0 are dropped; negative weights raise
            ``ValueError``, and so do all-zero weights.

        Returns
        -------
        self : OneClassSVM
            The fitted estimator.
        """
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
            # lane/fam2-neighbors: IDENTICAL takes the same resident solve on
            # every vendor (`x_neighbors_ocsvm_resident_idn`, registered unless
            # -D MOJOLEARN_IDN_OCSVM_RES_OFF; the host binding has none)
            res_fn = getattr(self._bind(), "x_neighbors_ocsvm_resident" if self._fast_tier()
                             else "x_neighbors_ocsvm_resident_idn", None)
            Q = None if res_fn is not None else self._kernel(Xw, Xw, self.kernel, self._gamma, self.coef0, self.degree)
        # libsvm's start (solve_one_class) by the base binding's
        # `ocsvm_alpha_init_f32` (lane pyglue-numeric: a Python loop)
        from ._buffer import _native
        alpha = empty((m,), "<f4")
        # lane/fam2-neighbors (cpu-gpu audit case 2): the start as three device
        # launches (x_neighbors/ocsvm_init.mojo; the host binding runs the same
        # items), registered in every mode since lane cpu2-l9-neighbors (no
        # host fallback)
        self._bind().x_neighbors_ocsvm_alpha_init(
            [addr_ro(cv, name="xn_ocsvm_init C"), addr(alpha, name="xn_ocsvm_init alpha")],
            [m], [float(self.nu)])
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
        """Raw scoring function of the samples (not shifted by ``offset_``).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples, or the (n_queries, n_train) kernel for
            ``kernel='precomputed'``.

        Returns
        -------
        scores : Array of shape (n_queries,), float32
        """
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
        """Signed distance to the boundary: ``score_samples(X) - offset_``.

        Positive for inliers, negative for outliers.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples (or kernel rows for ``kernel='precomputed'``).

        Returns
        -------
        scores : Array of shape (n_queries,), float32
        """
        return self._unary(self.score_samples(X), _U_IDENTITY, 1.0, -self.offset_)

    def predict(self, X):
        """Label samples as inliers (1) or outliers (-1).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples (or kernel rows for ``kernel='precomputed'``).

        Returns
        -------
        labels : Array of shape (n_queries,), int64
        """
        dec = self.decision_function(X)
        lab = empty((dec.size,), "<i4")
        if dec.size:
            self._op("p2m_sign_label", [(dec, 0), (lab, 1)], (dec.size, 2), (0.0,))
        return lab.astype("<i8")

    def fit_predict(self, X, y=None):
        """Fit on X and label X: ``fit(X).predict(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        labels : Array of shape (n_samples,), int64
            1 for inliers, -1 for outliers.
        """
        return self.fit(X).predict(X)


# ====================================================================== KernelPCA
def _kpca_clamp_count(kit, w):
    """KernelPCA's eigenvalue tail on the kit's cells (lane
    cpu2-l9-neighbors): `w` (1 x c, descending) clamped at zero (`maxs`,
    `x if x > 0 else 0`), and the count of positive clamped values (`gts`
    flags folded by `total`; 0/1 sums are exact in float32 below 2^24).
    Returns (clamped 1 x c matrix, count); only the count word is read."""
    wc = kit.ew("maxs", w, s=0.0)
    npos = int(kit.total(kit.ew("gts", wc, s=0.0)).s[0])
    return wc, npos


# Reference: scikit-learn `decomposition/_kernel_pca.py` (1.9.0) with
# `preprocessing.KernelCenterer`: the kernel matrix, centered in their
# order, the dense eigendecomposition (eigen_solver 'dense'; here
# x_decomp's eigh, ascending, then
# sklearn's `svd_flip(u, None)` sign rule), eigenvalues
# below zero set to zero, components sorted by decreasing eigenvalue (equal
# eigenvalues: the higher solver index first, as their reversed argsort),
# zero components removed when n_components is None or remove_zero_eig.
# Every eigen_solver is served by the dense solve (DEVIATION 5202: arpack
# and randomized are approximations of it), except the top-k Lanczos route
# for eigen_solver='auto', n > 200 and 0 < n_components < 10 (falls back to
# the dense solve when its basis does not converge).
class KernelPCA(_XNeighbors):
    """Kernel principal component analysis.

    Non-linear dimensionality reduction: forms the kernel matrix of the
    training data, centers it, and projects onto its leading eigenvectors,
    following scikit-learn's ``KernelPCA``. Negative eigenvalues are set to
    zero, components are sorted by decreasing eigenvalue, and each
    eigenvector's sign is fixed so its largest-magnitude entry is positive.
    Computation is float32.

    With ``eigen_solver='auto'``, more than 200 samples and
    ``0 < n_components < 10``, the top components come from an iterative
    (Lanczos) solver; every other case, including ``'arpack'`` and
    ``'randomized'``, uses an exact dense eigendecomposition.

    Parameters
    ----------
    n_components : int or None, default None
        Number of components. None keeps every component with a positive
        eigenvalue.
    kernel : str, default 'linear'
        ``'linear'``, ``'poly'`` (or ``'polynomial'``), ``'rbf'``,
        ``'sigmoid'``, ``'laplacian'``, ``'cosine'``, ``'chi2'``,
        ``'additive_chi2'`` or ``'precomputed'`` (``fit`` then takes the
        n x n Gram matrix and ``transform`` the n_queries x n_train kernel).
        Callable kernels raise ``NotImplementedError``.
    gamma : float or None, default None
        Kernel coefficient; None means ``1 / n_features``.
    degree : int, default 3
        Degree of the ``'poly'`` kernel.
    coef0 : float, default 1
        Independent term of the ``'poly'`` and ``'sigmoid'`` kernels.
    kernel_params : dict or None, default None
        Not supported; a non-empty value raises ``NotImplementedError``.
    alpha : float, default 1.0
        Accepted for scikit-learn compatibility; unused (no inverse transform).
    fit_inverse_transform : bool, default False
        Not supported; True raises ``NotImplementedError`` in ``fit``.
    eigen_solver : str, default 'auto'
        ``'auto'`` may use the iterative solver described above; any other
        value uses the dense solver.
    tol, max_iter, iterated_power, random_state, copy_X, n_jobs
        Accepted for scikit-learn compatibility; unused.
    remove_zero_eig : bool, default False
        Drop components with zero eigenvalue even when ``n_components`` is
        set.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator (``'fast'`` is Apple only). None uses
        the process default.

    Attributes
    ----------
    eigenvalues_ : Array of shape (n_components,)
        Eigenvalues of the centered kernel matrix, decreasing.
    eigenvectors_ : Array of shape (n_samples, n_components)
        Eigenvectors of the centered kernel matrix.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> Z = mojolearn.KernelPCA(n_components=2, kernel="rbf").fit_transform(X)
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
        """Fit the model on the training data.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data, or the (n_samples, n_samples) Gram matrix for
            ``kernel='precomputed'``. Sparse input is densified.
        y : None
            Ignored.

        Returns
        -------
        self : KernelPCA
            The fitted estimator.
        """
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
            vectors = kit.ew("mul", vectors, kit.absmax_signs(vectors, True))
            # lane cpu2-l9-neighbors: the clamp and the positive count on the
            # kit's cells (values come back descending, so the kept set is a
            # prefix); one count word read back, no host list of the values
            nv = values.r * values.c
            vals_m, npos = _kpca_clamp_count(kit, values if values.r == 1 else values.reshape(1, nv))
            kept = npos if self.remove_zero_eig else nv
            if kept < nv:
                vectors = vectors.cols(0, kept)
                vals_m = vals_m.cols(0, kept)
            self.eigenvalues_ = vals_m.out((kept,))
            self.eigenvectors_ = Array._from_flat(vectors.s, (n, kept), "<f4")
            self._fit_X, self._fit_cols, self._fit_all = X, cols, all_
            self.n_features_in_ = d
            return self
        # lane hr2-kpca-seq: the dense eigendecomposition is x_decomp's
        # eigh, never the host Jacobi of x_neighbors/eigh.mojo inside the
        # GPU binding. On every IDENTICAL device column (NVIDIA, AMD and
        # Apple alike, no vendor gate) eigh from n = 2 TD_NB (64) is the
        # tridiagonal route (x_decomp/eigh_tridiag.mojo EIGH_TD_IDN, default
        # since 2026-10-11), below that and on a clustered-spectrum refusal
        # the round-robin Jacobi (x_decomp/rr.mojo); the host binding takes
        # the same route with the same words since 2026-10-13
        # (x_decomp/eigh_tridiag_host.mojo; lane apple-neighbors-gpu-cpu).
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
        # descending (eigh is ascending; equal values: higher index first);
        # lane cpu2-l9-neighbors: only the top c values are taken, and the
        # clamp and the positive count run on the kit's cells (a prefix of
        # the descending order is kept); one count word read back
        order = range(n - 1, n - 1 - c, -1)
        kept = c
        vals_m = _M(array.array("f"), 1, 0)
        if c:
            vals_m, npos = _kpca_clamp_count(kit, wm.take_cols(order))  # wm is 1 x n
            if self.n_components is None or self.remove_zero_eig:
                kept = npos
            if kept < c:
                vals_m = vals_m.cols(0, kept)
        order = list(range(n - 1, n - 1 - kept, -1))  # glue: the kept column indices
        self.eigenvalues_ = vals_m.out((kept,))
        # sklearn's svd_flip(u, None) on the kept columns: each column's
        # largest-|.| entry (ties to the lower row) made positive
        vecs = Vm.take_cols(order)
        if order:
            vecs = kit.ew("mul", vecs, kit.absmax_signs(vecs, True))
        self.eigenvectors_ = Array._from_flat(vecs.s, (n, len(order)), "<f4")
        self._fit_X, self._fit_cols, self._fit_all = X, cols, all_
        self.n_features_in_ = d
        return self

    def fit_transform(self, X, y=None, **params):
        """Fit the model and return the projection of the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (or the Gram matrix for ``kernel='precomputed'``).
        y : None
            Ignored.
        **params
            Ignored.

        Returns
        -------
        X_new : Array of shape (n_samples, n_components), float32
        """
        self.fit(X)
        return self._alpha_scale(0)

    def _alpha_scale(self, divide):
        V = self.eigenvectors_
        out = empty(V.shape, "<f4")
        self._op("kpca_alpha_scale", [(V, 0), (self.eigenvalues_, 0), (out, 1)], (V.shape[0], V.shape[1], divide))
        return out

    def transform(self, X):
        """Project new samples onto the fitted components.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples, or for ``kernel='precomputed'`` the (n_queries, n_train)
            kernel against the training rows (a different column count raises
            ``ValueError``).

        Returns
        -------
        X_new : Array of shape (n_queries, n_components), float32
        """
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
        """Not supported: always raises ``NotImplementedError``."""
        raise NotImplementedError("KernelPCA: inverse_transform needs fit_inverse_transform, which is not implemented")

    def get_feature_names_out(self, input_features=None):
        """Output feature names: ``kernelpca0``, ``kernelpca1``, ...

        Parameters
        ----------
        input_features : None
            Ignored.

        Returns
        -------
        names : list of str
        """
        return _prefixed_names(self, self.eigenvalues_.shape[0])


# ====================================================================== RandomState
def _seed_word(seed):
    """The sketch samplers' seed (lane cpu2-l9-neighbors): an int as given;
    None (numpy's global RandomState) or a caller's numpy RandomState gives
    ONE draw, a word in [0, 2^31), and that word seeds the device
    counter-based generator exactly as an int random_state would. The
    O(d * n_components) table is then drawn on the device. Not
    scikit-learn's numbers for those two inputs (theirs are not reproducible
    for None either); the distribution is the same. Anything else is
    refused. (Lane py-runtime-b deleted the MT19937 Python stream that the
    _OFF arms drew from.)"""
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    if seed is None:
        from ._optional_numpy import require_numpy
        np = require_numpy('_expansion_neighbors')
        seed = np.random.mtrand._rand
    elif not (hasattr(seed, "randint") and hasattr(seed, "random_sample") and hasattr(seed, "uniform")):
        raise ValueError(f"{seed!r} cannot be used to seed a numpy.random.RandomState instance")
    return int(seed.randint(0, 1 << 31, size=1)[0])


# ====================================================================== PolynomialCountSketch
# Reference: scikit-learn `kernel_approximation.py` (PolynomialCountSketch,
# 1.9.0): `indexHash_` (`randint(0, n_components, (degree, n_features))`)
# and `bitHash_` (`choice([-1, 1], (degree, n_features))`); here drawn by the
# device counter-based generator (x_neighbors/kfeat_rng.mojo, see
# `_seed_word`), the same distribution, not their legacy RandomState words.
# The transform is the count sketches' circular convolution, computed
# directly instead of through an FFT (DEVIATION 5203).
class PolynomialCountSketch(_XNeighbors):
    """Polynomial kernel approximation by tensor sketch.

    Maps data to a feature space whose inner products approximate the
    polynomial kernel ``(gamma * <x, y> + coef0) ** degree``, following
    scikit-learn's ``PolynomialCountSketch``. The hash tables are drawn from
    mojolearn's counter-based generator seeded by ``random_state``: the same
    distribution as scikit-learn, reproducible for an int seed, but not
    scikit-learn's exact numbers. The sketches' circular convolution is
    computed directly rather than by FFT. Sparse input is densified;
    computation is float32.

    Parameters
    ----------
    gamma : float, default 1.0
        Kernel coefficient.
    degree : int, default 2
        Degree of the polynomial kernel; at least 1.
    coef0 : float, default 0
        Constant term of the kernel.
    n_components : int, default 100
        Dimension of the output feature space; at least 1.
    random_state : int, numpy.random.RandomState or None, default None
        Seed for the hash tables. A RandomState or None contributes one draw
        that seeds the generator. Anything else raises ``ValueError``.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator (``'fast'`` is Apple only). None uses
        the process default.

    Attributes
    ----------
    indexHash_ : Array of shape (degree, n_features_hashed), int32
        Output bucket of each input feature, per degree.
    bitHash_ : Array of shape (degree, n_features_hashed), int32
        Sign (-1 or 1) of each input feature, per degree.
        ``n_features_hashed`` is ``n_features + 1`` when ``coef0 != 0``,
        else ``n_features``.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> Z = mojolearn.PolynomialCountSketch(n_components=64, random_state=0).fit_transform(X)
    """

    def __init__(self, *, gamma=1.0, degree=2, coef0=0, n_components=100, random_state=None):
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Draw the hash tables.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data; only its number of features is used.
        y : None
            Ignored.

        Returns
        -------
        self : PolynomialCountSketch
            The fitted transformer.

        Raises
        ------
        ValueError
            If ``degree`` or ``n_components`` is below 1, or ``random_state`` is
            not an int, a numpy RandomState or None.
        """
        X = _f32(X)
        d = X.shape[1]
        nf = d + (1 if self.coef0 != 0 else 0)
        deg, nc = int(self.degree), int(self.n_components)
        if deg < 1 or nc < 1:
            raise ValueError("degree and n_components must be >= 1")
        seed = _seed_word(self.random_state)
        draw_idn = (getattr(self._bind(), "x_neighbors_kfeat_pcs_draw_idn", None)
                    if isinstance(seed, int) and not isinstance(seed, bool) and nf > 0 else None)
        if draw_idn is not None:
            # IDN_XN_SKETCH_CTR (IDENTICAL default with an _OFF A/B; FAST always;
            # x_neighbors/kfeat_rng.mojo): the tables from the counter-based
            # generator on the device, not the MT19937 Python loop
            ih = empty((deg, nf), "<i4")
            bh = empty((deg, nf), "<i4")
            draw_idn([seed & 0xFFFFFFFF, nc, deg * nf], addr(ih, name="indexHash_"), addr(bh, name="bitHash_"))
            self.indexHash_ = ih
            self.bitHash_ = bh
            self.n_features_in_ = d
            return self
        if nf > 0:
            # lane py-runtime-b: the MT19937 Python draw behind the _OFF
            # define is deleted (no Python compute in the runtime); a binary
            # without the device draw refuses
            raise NotImplementedError(
                "mojolearn PolynomialCountSketch: this binary lacks x_neighbors_kfeat_pcs_draw_idn "
                "(rebuild without -D MOJOLEARN_IDN_XN_SKETCH_CTR_OFF)")
        self.indexHash_ = zeros((deg, 0), "<i4")
        self.bitHash_ = zeros((deg, 0), "<i4")
        self.n_features_in_ = d
        return self

    def transform(self, X):
        """Map X to the sketched feature space.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Samples; the number of features must match ``fit`` (``ValueError``
            otherwise). Sparse input is densified.

        Returns
        -------
        X_new : Array of shape (n_samples, n_components), float32
        """
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
        """Fit to X, then transform X: ``fit(X).transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        X_new : Array of shape (n_samples, n_components), float32
        """
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Output feature names: ``polynomialcountsketch0``, ``polynomialcountsketch1``, ...

        Parameters
        ----------
        input_features : None
            Ignored.

        Returns
        -------
        names : list of str
        """
        return _prefixed_names(self, int(self.n_components))


# ====================================================================== AdditiveChi2Sampler
# Reference: scikit-learn `kernel_approximation.py` (AdditiveChi2Sampler,
# `_transform_dense`, 1.9.0). A sparse input gives the dense map's nonzeros
# as csr, as theirs (`_transform_sparse` maps only the stored entries, and a
# zero maps to zeros).
class AdditiveChi2Sampler(_XNeighbors):
    """Approximate feature map for the additive chi-squared kernel.

    A deterministic Fourier sampling of the kernel's feature map, following
    scikit-learn's ``AdditiveChi2Sampler``. Each input feature becomes
    ``2 * sample_steps - 1`` output features. Input must be non-negative;
    negative values raise ``ValueError``. A scipy.sparse input gives a sparse
    (CSR) output. Computation is float32.

    Parameters
    ----------
    sample_steps : int, default 2
        Number of sampling points.
    sample_interval : float or None, default None
        Sampling interval. Required when ``sample_steps`` is not 1, 2 or 3
        (``ValueError`` otherwise); by default 0.8, 0.5 or 0.4 for those.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator (``'fast'`` is Apple only). None uses
        the process default.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> Z = mojolearn.AdditiveChi2Sampler(sample_steps=2).fit_transform(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`: stateless, non-negative input
    _sklearn_input_tags = {"positive_only": True}
    _sklearn_requires_fit = False

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
        """Validate the input and the sampling settings.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data; must be non-negative (``ValueError`` otherwise).
        y : None
            Ignored.

        Returns
        -------
        self : AdditiveChi2Sampler
            The fitted transformer.
        """
        X = _f32(X)
        first = -2
        if X.size and _kfeat_flags(self) & 1:
            # XN_FAST_ACHI2_DEVSCAN (default, rollback _OFF): one pooled upload and device
            # scan (x_neighbors_kfeat_first_negative), not a host X.min();
            # -2 = below the binding's size gate (XN_ACHI2_DEVSCAN_MIN,
            # x_neighbors/kfeat_dev.mojo; 1 since 2026-10-04, the old 2^22
            # only under -D MOJOLEARN_LEGACY_NARROW_ACHI2_DEVSCAN): main's check below
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
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Non-negative samples (``ValueError`` otherwise).

        Returns
        -------
        X_new : Array or scipy.sparse.csr_matrix of shape (n_samples, n_features * (2 * sample_steps - 1))
            float32; sparse (CSR) when X is sparse.
        """
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
        """Fit to X, then transform X: ``fit(X).transform(X)``.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Non-negative training data.
        y : None
            Ignored.

        Returns
        -------
        X_new : Array or scipy.sparse.csr_matrix of shape (n_samples, n_features * (2 * sample_steps - 1))
        """
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Output feature names.

        For each input name ``f``: ``additivechi2sampler_f_sqrt``, then for each
        step ``j`` from 1 the ``..._cos{j}`` and ``..._sin{j}`` names.

        Parameters
        ----------
        input_features : list of str or None, default None
            Input feature names; None means ``x0``, ``x1``, ... Its length must
            equal ``n_features_in_`` (``ValueError`` otherwise).

        Returns
        -------
        names : list of str
        """
        names = _feature_names_in(self, input_features)
        base = type(self).__name__.lower()
        out = [f"{base}_{nm}_sqrt" for nm in names]  # glue: feature names over the d columns
        for j in range(1, int(self.sample_steps)):  # glue: feature names over the d columns
            out += [f"{base}_{nm}_cos{j}" for nm in names]  # glue: feature names over the d columns
            out += [f"{base}_{nm}_sin{j}" for nm in names]  # glue: feature names over the d columns
        return _names_out(out)


# ====================================================================== SkewedChi2Sampler
# Reference: scikit-learn `kernel_approximation.py` (SkewedChi2Sampler,
# 1.9.0). The weights and offsets come from the device counter-based
# generator (x_neighbors/kfeat_rng.mojo, `x_neighbors_kfeat_schi2_fit_idn`);
# the legacy-RandomState route (`x_neighbors_kfeat_schi2_fit`, sklearn's
# uniforms, pi/2 * u and the offsets in IEEE double rounded once to float32)
# runs only on a binary without the counter-based draw. The inverse sech CDF,
# the log, the product and the cosine run in float32 on the portable spellings.
class SkewedChi2Sampler(_XNeighbors):
    """Approximate feature map for the skewed chi-squared kernel.

    Monte Carlo approximation of the kernel's feature map, following
    scikit-learn's ``SkewedChi2Sampler``. The random weights and offsets are
    drawn from mojolearn's counter-based generator seeded by
    ``random_state``: the same distribution as scikit-learn, reproducible
    for an int seed, but not scikit-learn's exact numbers. Computation is
    float32. Input values at or below ``-skewedness`` raise ``ValueError``.

    Parameters
    ----------
    skewedness : float, default 1.0
        Skewedness parameter of the kernel.
    n_components : int, default 100
        Number of Monte Carlo samples per feature (output dimension); at
        least 1.
    random_state : int, numpy.random.RandomState or None, default None
        Seed for the weights and offsets. A RandomState or None contributes
        one draw that seeds the generator. Anything else raises
        ``ValueError``.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator (``'fast'`` is Apple only). None uses
        the process default.

    Attributes
    ----------
    random_weights_ : Array of shape (n_features, n_components)
        Random projection weights.
    random_offset_ : Array of shape (n_components,)
        Random offsets.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> Z = mojolearn.SkewedChi2Sampler(n_components=64, random_state=0).fit_transform(X)
    """

    def __init__(self, *, skewedness=1.0, n_components=100, random_state=None):
        self.skewedness = skewedness
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Draw the random weights and offsets.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data; only its number of features is used.
        y : None
            Ignored.

        Returns
        -------
        self : SkewedChi2Sampler
            The fitted transformer.

        Raises
        ------
        ValueError
            If X has no features, ``n_components`` is below 1, or
            ``random_state`` is not an int, a numpy RandomState or None.
        """
        X = _f32(X)
        d = X.shape[1]
        nc = int(self.n_components)
        seed = _seed_word(self.random_state)
        flags = (_kfeat_flags(self) if isinstance(seed, int) and not isinstance(seed, bool)
                 and d > 0 and nc > 0 else 0)
        self.__dict__.pop("_schi2_z", None)
        # lane cpu3-python: the counter-based device draw first, on every
        # tier that compiles it (IDENTICAL, and since this lane FAST on every
        # vendor: x_neighbors/kfeat_rng.mojo XN_IDN_SKETCH_CTR). FAST + Apple's
        # SCHI2_LAZYW route drew z and the offsets with sklearn's sequential
        # stream on the host (`x_neighbors_kfeat_schi2_draw`); it has left the
        # GPU route.
        fit_idn = (getattr(self._bind(), "x_neighbors_kfeat_schi2_fit_idn", None)
                   if isinstance(seed, int) and not isinstance(seed, bool) and d > 0 and nc > 0 else None)
        if fit_idn is not None:
            # IDN_XN_SKETCH_CTR (IDENTICAL default with an _OFF A/B; FAST always;
            # x_neighbors/kfeat_rng.mojo): z, the weights and the offsets
            # from the counter-based generator on the device, one call
            w = _empty_out((d, nc), "<f4")
            off = empty((nc,), "<f4")
            fit_idn([seed & 0xFFFFFFFF, d, nc], addr(w, name="random_weights_"),
                    addr(off, name="random_offset_"))
            self.random_weights_ = w
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
        # lane py-runtime-b: the MT19937 Python draw (and pi/2 * u in Python)
        # behind the _OFF defines is deleted; a binary with neither device
        # draw, or an empty X / n_components < 1, refuses
        if d < 1 or nc < 1:
            raise ValueError("mojolearn SkewedChi2Sampler: X needs at least one feature and n_components >= 1")
        raise NotImplementedError(
            "mojolearn SkewedChi2Sampler: this binary lacks x_neighbors_kfeat_schi2_fit_idn "
            "(rebuild without -D MOJOLEARN_IDN_XN_SKETCH_CTR_OFF)")

    @property
    def random_weights_(self):
        """Random projection weights, shape (n_features, n_components)."""
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
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Samples; every value must be greater than ``-skewedness``
            (``ValueError`` otherwise). Sparse input is densified.

        Returns
        -------
        X_new : Array of shape (n_samples, n_components), float32
        """
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
        """Fit to X, then transform X: ``fit(X).transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        X_new : Array of shape (n_samples, n_components), float32
        """
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Output feature names: ``skewedchi2sampler0``, ``skewedchi2sampler1``, ...

        Parameters
        ----------
        input_features : None
            Ignored.

        Returns
        -------
        names : list of str
        """
        return _prefixed_names(self, int(self.n_components))


# ====================================================================== LabelPropagation
# scikit-learn `semi_supervised/_label_propagation.py` (1.9.0): the dense
# RBF graph or compact kNN connectivity graph (including each row's own
# point as its first neighbor), the product / clamp iteration with their stopping
# rule (sum |L - L_prev| < tol, checked before each step), the final row
# normalization and `transduction_`. Callable kernels are refused. Float32
# where theirs is float64; the neighbor ties go to the lower index.
class _LabelPropagationBase(_XNeighbors):
    """Shared implementation of LabelPropagation and LabelSpreading."""
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "classifier"

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
        """Fit the model and propagate labels to the unlabeled samples.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data. Sparse input is densified.
        y : array-like of shape (n_samples,)
            Labels; -1 marks an unlabeled sample.

        Returns
        -------
        self
            The fitted estimator.
        """
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
        classes = [c for i, c in enumerate(allc) if i != skip]  # glue: class labels without the unlabeled marker
        C = len(classes)
        ld = _empty_out((n, C), "<f4")
        ys = _empty_out((n, C), "<f4")
        unlabeled = empty((n,), "<i4")
        a = _f32_scalar(1.0 - float(self.alpha)) if self._variant != "propagation" else 1.0
        if n and C:
            # The binding reads every int as a size (>= 0), so "no unlabeled
            # marker" is passed as a code no row carries (len(allc)), which
            # the item treats exactly as skip -1: no row unlabeled, codes kept.
            self._op("p2m_lp_labels", [(codes, 0), (ld, 1), (ys, 1), (unlabeled, 1)],
                     (n, C, skip if skip >= 0 else len(allc)), (a,))
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
        # lane/fam2-neighbors: IDENTICAL device binaries answer 1 too
        # (LP_IDN_RESIDENT: the same items, no per-iteration host round trip)
        if isinstance(G, tuple) and _lp_fast_resident(self):
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
                     (n, C, int(self.max_iter),
                      (0 if self._variant == "propagation" else 1) +
                      (2 if getattr(self, "_classical_raw_graph", False) else 0),
                      tol_bits >> 32, tol_bits & 0xFFFFFFFF),
                     (_f32_scalar(self.alpha) if self._variant != "propagation" else 0.0,))
            n_iter = int(info.tolist()[0])
            return self._finish_fit(X, classes, ld, n, C, n_iter)
        prev = _empty_out((n, C), "<f4")
        s = _empty_out((1,), "<f4")
        n_iter = 0
        converged = False
        for it in range(int(self.max_iter)):  # glue: one device propagation step per iteration
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
        """Class probabilities of new samples from the fitted label distributions.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples.

        Returns
        -------
        proba : Array of shape (n_queries, n_classes), float32
            Columns in the order of ``classes_``.
        """
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
        """Predict the class of new samples.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Samples.

        Returns
        -------
        labels : Array or list of shape (n_queries,)
            The most probable class of each sample.
        """
        return _argmax_labels(self, self.classes_, self.predict_proba(X))

    def score(self, X, y, sample_weight=None):
        """Mean accuracy of :meth:`predict` on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,) or None, default None
            Sample weights.

        Returns
        -------
        score : float
            (Weighted) fraction of correct predictions.
        """
        return _accuracy(y, self.predict(X), sample_weight)


class LabelPropagation(_LabelPropagationBase):
    """Semi-supervised classification by label propagation.

    Spreads the known labels over a similarity graph of all samples, keeping
    the labeled samples fixed (hard clamping), following scikit-learn's
    ``LabelPropagation``. Mark unlabeled samples with -1 in ``y``. The graph
    is a dense RBF affinity matrix or a k-nearest-neighbor connectivity
    graph (each sample is its own first neighbor; distance ties go to the
    lower index). Iteration stops when the summed absolute change of the
    label distributions falls below ``tol`` or after ``max_iter`` steps.
    Computation is float32 (scikit-learn uses float64).

    Parameters
    ----------
    kernel : {'rbf', 'knn'}, default 'rbf'
        Graph kernel. Other values, including callables, raise
        ``NotImplementedError``.
    gamma : float, default 20
        Coefficient of the ``'rbf'`` kernel.
    n_neighbors : int, default 7
        Number of neighbors of the ``'knn'`` kernel.
    max_iter : int, default 1000
        Maximum number of iterations.
    tol : float, default 1e-3
        Convergence tolerance.
    n_jobs : int or None, default None
        Accepted for scikit-learn compatibility; unused.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator (``'fast'`` is Apple only). None uses
        the process default.

    Attributes
    ----------
    X_ : Array of shape (n_samples, n_features)
        Training data.
    classes_ : list
        Class labels (without the -1 marker).
    label_distributions_ : Array of shape (n_samples, n_classes)
        Label distribution of each training sample.
    transduction_ : Array or list of shape (n_samples,)
        Label assigned to each training sample.
    n_iter_ : int
        Number of iterations run.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > 0.5).astype(np.int64)
    >>> y[100:] = -1
    >>> model = mojolearn.LabelPropagation(kernel="knn").fit(X, y)
    >>> labels = model.transduction_
    """

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
        self._classical_raw_graph = _classical_graph_normalization(self)
        if self._classical_raw_graph:
            return A
        G = empty(A.shape, "<f4")
        self._op("row_normalize", [(A, 0), (G, 1)], A.shape)
        return G


class LabelSpreading(_LabelPropagationBase):
    """Semi-supervised classification by label spreading.

    Spreads the known labels over the normalized graph Laplacian of a
    similarity graph, letting labeled samples change by a fraction ``alpha``
    (soft clamping), following scikit-learn's ``LabelSpreading``. Mark
    unlabeled samples with -1 in ``y``. The graph is a dense RBF affinity
    matrix or a k-nearest-neighbor graph (each sample is its own first
    neighbor; distance ties go to the lower index). Iteration stops when the
    summed absolute change of the label distributions falls below ``tol`` or
    after ``max_iter`` steps. Computation is float32 (scikit-learn uses
    float64).

    Parameters
    ----------
    kernel : {'rbf', 'knn'}, default 'rbf'
        Graph kernel. Other values, including callables, raise
        ``NotImplementedError``.
    gamma : float, default 20
        Coefficient of the ``'rbf'`` kernel.
    n_neighbors : int, default 7
        Number of neighbors of the ``'knn'`` kernel.
    alpha : float, default 0.2
        Clamping factor in (0, 1): the fraction of a sample's label taken
        from its neighbors each step. Other values raise ``ValueError``.
    max_iter : int, default 30
        Maximum number of iterations.
    tol : float, default 1e-3
        Convergence tolerance.
    n_jobs : int or None, default None
        Accepted for scikit-learn compatibility; unused.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator (``'fast'`` is Apple only). None uses
        the process default.

    Attributes
    ----------
    X_ : Array of shape (n_samples, n_features)
        Training data.
    classes_ : list
        Class labels (without the -1 marker).
    label_distributions_ : Array of shape (n_samples, n_classes)
        Label distribution of each training sample.
    transduction_ : Array or list of shape (n_samples,)
        Label assigned to each training sample.
    n_iter_ : int
        Number of iterations run.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > 0.5).astype(np.int64)
    >>> y[100:] = -1
    >>> model = mojolearn.LabelSpreading(kernel="knn", alpha=0.2).fit(X, y)
    >>> labels = model.transduction_
    """

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
        """Fit the model and spread labels to the unlabeled samples.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data. Sparse input is densified.
        y : array-like of shape (n_samples,)
            Labels; -1 marks an unlabeled sample.

        Returns
        -------
        self : LabelSpreading
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``alpha`` is not in (0, 1).
        """
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        return super().fit(X, y)

    def _build_graph(self, X):
        A = self._graph_affinity(X)
        self._classical_raw_graph = _classical_graph_normalization(self)
        if self._classical_raw_graph:
            return A
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
# Reference: scikit-learn `impute/_knn.py` (1.9.0): nan_euclidean distances
# to the fit rows, donors = fit rows where the column is present, the k
# nearest (ties: the lower row index), 'uniform' or 'distance' weights,
# the masked column mean when no donor has a finite distance, all-missing
# columns dropped (or zero with keep_empty_features), `add_indicator`,
# `get_feature_names_out`. A numeric missing_values other than NaN is
# replaced by NaN before the Mojo call (an exact equality test). Callable
# metrics and weights are refused (a user arithmetic outside the pinned items).
class KNNImputer(_XNeighbors):
    """Imputation of missing values by k nearest neighbors.

    Fills each missing value from the ``n_neighbors`` nearest training rows
    that have that feature, using NaN-aware euclidean distances, following
    scikit-learn's ``KNNImputer``. Distance ties go to the lower row index.
    When no donor is at a finite distance, the training column mean is used.
    Features that are entirely missing at ``fit`` are dropped from the output
    (or set to zero with ``keep_empty_features=True``). Computation is
    float32.

    Parameters
    ----------
    missing_values : float, default nan
        Value marking a missing entry: NaN or another number. Strings,
        None and booleans raise ``NotImplementedError``.
    n_neighbors : int, default 5
        Number of donor neighbors; at least 1.
    weights : {'uniform', 'distance'}, default 'uniform'
        Donor weighting: equal, or by inverse distance. Callables raise
        ``NotImplementedError``.
    metric : {'nan_euclidean'}, default 'nan_euclidean'
        Distance; any other value raises ``NotImplementedError``.
    copy : bool, default True
        Accepted for scikit-learn compatibility; the output is always a new
        array.
    add_indicator : bool, default False
        Append a 0/1 indicator column for each feature that had missing
        values at ``fit``.
    keep_empty_features : bool, default False
        Keep features that are entirely missing at ``fit``, imputed as zero.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator (``'fast'`` is Apple only). None uses
        the process default.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> X[::7, 3] = np.nan
    >>> X_filled = mojolearn.KNNImputer(n_neighbors=5).fit_transform(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"allow_nan": True}

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
        """Store the training data used as donors.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data, with missing values marked by ``missing_values``.
        y : None
            Ignored.

        Returns
        -------
        self : KNNImputer
            The fitted imputer.
        """
        self._check()
        X = self._masked(X)
        n, d = X.shape
        # lane/neural-pass71 (2026-10-01): the column flags from one native
        # pass over the cells (xn_nan_cells), no list of the matrix
        cm = self._nan_cells(X, 1)[1]
        self._valid = [cm[f] < n for f in range(d)]  # glue: column masks over the d features
        self._miss_cols = [f for f in range(d) if cm[f] > 0]  # glue: column masks over the d features
        self._fit_X = X
        self.n_features_in_ = d
        return self

    def transform(self, X):
        """Impute the missing values of X.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Data to impute; the number of features must match ``fit``
            (``ValueError`` otherwise).

        Returns
        -------
        X_imputed : Array of shape (n_samples, n_features_out), float32
            Imputed data, without the features that were entirely missing at
            ``fit`` (unless ``keep_empty_features``), followed by the indicator
            columns when ``add_indicator`` is True.
        """
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
        keep = [f for f in range(d) if self._valid[f]]  # glue: column masks over the d features
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
        """Fit to X, then impute X: ``fit(X).transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with missing values.
        y : None
            Ignored.

        Returns
        -------
        X_imputed : Array of shape (n_samples, n_features_out), float32
        """
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Output feature names.

        The kept input names, followed by ``missingindicator_<name>`` for each
        indicator column when ``add_indicator`` is True.

        Parameters
        ----------
        input_features : list of str or None, default None
            Input feature names; None means ``x0``, ``x1``, ... Its length must
            equal ``n_features_in_`` (``ValueError`` otherwise).

        Returns
        -------
        names : list of str
        """
        names = _feature_names_in(self, input_features)
        out = [nm for f, nm in enumerate(names) if self._valid[f] or self.keep_empty_features]  # glue: feature names over the d columns
        if self.add_indicator:
            out += [f"missingindicator_{names[f]}" for f in self._miss_cols]  # glue: feature names over the d columns
        return _names_out(out)


# ====================================================================== PageRank
def _adjacency(A):
    A = _f32(A, "adjacency")
    if A.shape[0] != A.shape[1]:
        raise ValueError("the adjacency matrix must be square")
    return A


# References: networkx `pagerank` (`_pagerank_scipy`: row-stochastic
# transition matrix, dangling nodes redistributed by the personalization,
# stop when sum |x - x_last| < n * tol) and cuGraph
# cpp/src/link_analysis/pagerank_impl.cuh. Each step is the pinned-fold
# GEMV of x_neighbors/items.mojo `pagerank_step_item`. A scipy.sparse input
# is read as CSR with sorted indices; the device checks the column order.
class PageRank(_XNeighbors):
    """PageRank scores of a directed graph by power iteration.

    Follows networkx's ``pagerank``: a row-stochastic transition matrix,
    dangling nodes' mass redistributed by ``dangling`` (or the
    personalization), and iteration until the summed absolute change is
    below ``n * tol``. The graph is a weighted adjacency matrix with
    ``A[i, j]`` the weight of the edge i -> j, given as a dense square
    array, a scipy.sparse matrix (any format; its indices must be sorted
    within each row), or an ``(indptr, indices, n)`` CSR tuple in which every
    edge weighs 1. Vectors over nodes are length-n arrays (networkx uses
    dicts over nodes 0..n-1). Computation is float32.

    Parameters
    ----------
    alpha : float, default 0.85
        Damping factor.
    personalization : array-like of shape (n,) or None, default None
        Teleport distribution, normalized to sum to one; None is uniform.
    max_iter : int, default 100
        Maximum number of iterations. Not converging raises ``RuntimeError``.
    tol : float, default 1e-6
        Per-node convergence tolerance.
    nstart : array-like of shape (n,) or None, default None
        Starting vector, normalized to sum to one; None is uniform.
    weight : bool or None, default True
        Use the edge weights. False or None treats every edge as weight 1.
    dangling : array-like of shape (n,) or None, default None
        Where the mass of nodes without out-edges goes, normalized to sum to
        one; None uses the personalization.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator (``'fast'`` is Apple only). None uses
        the process default.

    The ``personalization``, ``nstart`` and ``dangling`` vectors must have
    length n, be non-negative and not all zero (``ValueError`` otherwise).

    Attributes
    ----------
    pagerank_ : Array of shape (n,), float32
        PageRank score of each node.
    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> A = (np.random.default_rng(0).random((500, 500)) < 0.01).astype(np.float32)
    >>> scores = mojolearn.PageRank(alpha=0.85).fit(A).pagerank_
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

    def _unit(self, v, n, what):
        """A caller's length-n non-negative vector divided by its sum (IEEE
        double, rounded once to float32), as networkx normalizes its dicts."""
        # lane/fam2-neighbors: the sum, the sign test and the n divisions by
        # the binding (`x_neighbors_unit_ff`, device kernels; registered in
        # IDENTICAL unless -D MOJOLEARN_IDN_XN_UNIT_DEV_OFF)
        # (lane py-runtime-b: the Python fsum-and-divide route behind the
        # _OFF define is deleted; that binary refuses a weight vector)
        unit_fn = getattr(self._bind(), "x_neighbors_unit_ff", None)
        if unit_fn is None:
            raise NotImplementedError(
                f"mojolearn PageRank {what}: this binary lacks x_neighbors_unit_ff "
                "(rebuild without -D MOJOLEARN_IDN_XN_UNIT_DEV_OFF)")
        wv = _f32_1d(v, what)
        if wv.shape[0] != n:
            raise ValueError(f"{what} must be n non-negative values, not all zero")
        out = _empty_out((n,), "<f4")
        uinfo = empty((2,), "<i4")
        unit_fn([addr_ro(wv, name="xn_unit v"), addr(out, name="xn_unit out"),
                 addr(uinfo, name="xn_unit info")], [n], [])
        neg, zero = uinfo.tolist()
        if neg or zero:
            raise ValueError(f"{what} must be n non-negative values, not all zero")
        return out

    def fit(self, A, y=None):
        """Compute the PageRank scores of the graph.

        Parameters
        ----------
        A : array-like of shape (n, n), scipy.sparse matrix, or tuple
            Weighted adjacency matrix (``A[i, j]`` is the weight of i -> j): a
            dense square array, a scipy.sparse matrix, or an
            ``(indptr, indices, n)`` CSR tuple.
        y : None
            Ignored.

        Returns
        -------
        self : PageRank
            The fitted estimator, with ``pagerank_`` and ``n_iter_``.

        Raises
        ------
        RuntimeError
            If the iteration does not converge within ``max_iter`` steps.
        """
        binary = self.weight is None or self.weight is False
        csr = _csr_graph_of(A)
        if csr is not None:
            # lane gap-graph (2026-10-08, docs/plans/gaps-2026-10-08.md 5.2):
            # a scipy.sparse matrix or an (indptr, indices, n) tuple goes to
            # the device as it is (`xn_pr_csr`): the column lists are built
            # by one sort on the device, the vectors the caller left to the
            # default are filled there, no dense matrix and no separate
            # fill calls. Glue only: pointer handoff and flags.
            indptr, indices, vals, n = csr
        else:
            A = _adjacency(A)
            n = A.shape[0]
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

        info = empty((2,), "<i4")
        thr = struct.unpack("<Q", struct.pack("<d", n * float(self.tol)))[0]
        if csr is not None:
            p_uniform = self.personalization is None
            p = empty((n,), "<f4") if p_uniform else self._unit(self.personalization, n, "personalization")
            dw_uniform = self.dangling is None and p_uniform
            dw = p if self.dangling is None else self._unit(self.dangling, n, "dangling")
            x_uniform = self.nstart is None
            x = empty((n,), "<f4") if x_uniform else self._unit(self.nstart, n, "nstart").copy()
            has_vals = vals is not None
            if not has_vals:
                vals = empty((1,), "<f4")
            self._op("pr_csr", [(indptr, 0), (indices, 0), (vals, 0), (x, 1), (p, 0), (dw, 0), (info, 1)],
                     (n, indices.shape[0], 1 if has_vals else 0, int(self.max_iter), thr >> 32,
                      thr & 0xFFFFFFFF, 1 if binary else 0, 1 if x_uniform else 0, 1 if p_uniform else 0,
                      1 if dw_uniform else 0),
                     (_f32_scalar(self.alpha),))
        else:
            if self.personalization is None:
                p = uniform()
            else:
                p = self._unit(self.personalization, n, "personalization")
            dw = p if self.dangling is None else self._unit(self.dangling, n, "dangling")
            x = uniform() if self.nstart is None else self._unit(self.nstart, n, "nstart")
            x = x.copy()
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
# The labels come from the min-label product iteration (DBSCAN's
# weak_cc; cuGraph weakly_connected_components_impl.cuh), integers only.
def connected_components(A, directed=True, connection="weak", return_labels=True, numeric_mode=None):
    """Connected components of a graph.

    Follows ``scipy.sparse.csgraph.connected_components``. An edge is any
    nonzero entry of the adjacency matrix. Weak connectivity ignores edge
    direction. Components are numbered in order of their lowest node, so
    component 0 holds node 0, as scipy numbers them.

    Parameters
    ----------
    A : array-like of shape (n, n), scipy.sparse matrix, or tuple
        Adjacency matrix: a dense square array, a scipy.sparse matrix (any
        format), or an ``(indptr, indices, n)`` CSR tuple.
    directed : bool, default True
        Whether the graph is directed.
    connection : {'weak', 'strong'}, default 'weak'
        Connectivity type. ``'strong'`` on a directed graph raises
        ``NotImplementedError``; on an undirected graph it equals
        ``'weak'``. Other values raise ``ValueError``.
    return_labels : bool, default True
        Also return the component label of each node.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for the call (``'fast'`` is Apple only). None uses the
        process default.

    Returns
    -------
    n_components : int
        Number of connected components.
    labels : Array of shape (n,), int32
        Component of each node; returned only when ``return_labels`` is True.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> A = (np.random.default_rng(0).random((500, 500)) < 0.002).astype(np.float32)
    >>> n_components, labels = mojolearn.connected_components(A, directed=False)
    """
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
        # TOMBSTONE: MOJOLEARN_CC_FAST_OFF (removed) deleted 2026-10-08 by 2851a6627; code recoverable at 2851a6627^.
        # Restore: git apply experiments/removed/MOJOLEARN_CC_FAST_OFF.patch; record in docs/TOMBSTONES.md.
        # lane gap-graph (2026-10-08, docs/plans/gaps-2026-10-08.md 5.3): the
        # GPU binding on every vendor and tier runs the batched rounds and
        # compacts the labels itself in order of first appearance
        # (`_cc_relabel`'s integers; lane/apple-fast-graph's CC_FAST
        # promoted) and puts n_components + 1 in info[1]; info[1] == 0
        # means the host binding ran (it never writes info[1]) and `lab`
        # still needs `_cc_relabel` (zeros, not empty, for that reason).
        info = zeros((2,), "<i4")
        est._op("cc_iterate_csr", [(indptr, 0), (indices, 0), (lab, 1), (info, 1)], (n, indices.shape[0]))
        k1 = int(info.tolist()[1])
        if k1 > 0:
            return (k1 - 1, lab) if return_labels else k1 - 1
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


def _csr_graph_of(A):
    """(indptr, indices, vals, n) for PageRank and Louvain (lane gap-graph):
    `_csr_of`'s lists plus the weights as a float32 Array (a scipy matrix's
    `data`), or None for the weights when the caller gave an (indptr,
    indices, n) tuple (every edge then weighs 1.0 on the device); None for
    a dense input. Glue only: the device checks the column order and the
    symmetry, never Python."""
    if isinstance(A, tuple) and len(A) == 3:
        indptr, indices, n = A
        n = int(n)
        vals = None
    elif hasattr(A, "tocsr") and hasattr(A, "shape"):
        if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("the adjacency matrix must be square")
        M = A.tocsr()
        indptr, indices, n = M.indptr, M.indices, int(M.shape[0])
        vals = _f32_1d(M.data, "data")
    else:
        return None
    ip = _i32(indptr, "indptr")
    ix = _i32(indices, "indices")
    if ip.shape[0] != n + 1:
        raise ValueError("the CSR adjacency's indptr must hold n + 1 entries")
    if int(ip[n]) != ix.shape[0]:  # argument check: one scalar, the entry count
        raise ValueError("the CSR adjacency's indices must hold indptr[n] entries")
    if vals is not None and vals.shape[0] != ix.shape[0]:
        raise ValueError("the CSR adjacency's data must hold one weight per entry")
    return ip, ix, vals, n


# ====================================================================== Louvain
# References: networkx `louvain_communities` / `louvain_partitions`
# (`_one_level`, `_gen_graph`, `modularity`) and cuGraph
# cpp/src/community/louvain_impl.cuh. Parallel local moving in a PINNED
# order (x_neighbors/graph_par.mojo, DEVIATION 5204): the nodes of one
# colour of a fixed graph colouring move together instead of networkx's
# `seed` shuffle, candidate communities in ascending id, a strictly larger
# gain to move, so ties go to the lowest community id; community totals
# and the aggregation are fixed-order folds. A scipy.sparse input is read as
# CSR with sorted indices; the device checks the column order and symmetry.
class Louvain(_XNeighbors):
    """Louvain community detection on an undirected weighted graph.

    Finds communities by greedy modularity optimization, following
    networkx's ``louvain_communities``. Local moving runs in parallel in a
    fixed order (the nodes of one color of a fixed graph coloring move
    together, ties go to the lowest community id) instead of networkx's
    random shuffle, so the result is reproducible and does not depend on
    ``seed``; it can differ from networkx's partition. The graph is a
    symmetric weighted adjacency matrix given as a dense square array, a
    scipy.sparse matrix (any format; indices sorted within each row), or an
    ``(indptr, indices, n)`` CSR tuple in which every edge weighs 1. A
    non-symmetric matrix or a graph without edges raises ``ValueError``.

    Parameters
    ----------
    resolution : float, default 1.0
        Resolution of the modularity; larger values favor smaller
        communities.
    threshold : float, default 1e-7
        Minimum modularity gain for another level.
    max_level : int or None, default None
        Maximum number of levels (a positive integer); None means no limit.
    seed : int or None, default None
        Accepted for networkx compatibility; unused.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator (``'fast'`` is Apple only). None uses
        the process default.

    Attributes
    ----------
    labels_ : Array of shape (n,), int32
        Community of each node, numbered in order of each community's lowest
        node.
    n_communities_ : int
        Number of communities.
    modularity_ : float
        Modularity of the partition (computed in float32).
    n_levels_ : int
        Number of levels run.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> M = np.random.default_rng(0).random((500, 500)) < 0.02
    >>> A = (M | M.T).astype(np.float32)
    >>> labels = mojolearn.Louvain().fit_predict(A)
    """

    def __init__(self, resolution=1.0, *, threshold=1e-7, max_level=None, seed=None):
        self.resolution = resolution
        self.threshold = threshold
        self.max_level = max_level
        self.seed = seed

    def fit(self, A, y=None):
        """Detect communities in the graph.

        Parameters
        ----------
        A : array-like of shape (n, n), scipy.sparse matrix, or tuple
            Symmetric weighted adjacency matrix: a dense square array, a
            scipy.sparse matrix, or an ``(indptr, indices, n)`` CSR tuple.
        y : None
            Ignored.

        Returns
        -------
        self : Louvain
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``max_level`` is not a positive integer or None, the matrix is not
            symmetric, or the graph has no edges.
        """
        ml = 0 if self.max_level is None else int(self.max_level)
        if self.max_level is not None and ml < 1:
            raise ValueError("max_level must be a positive integer or None")
        csr = _csr_graph_of(A)
        if csr is not None:
            # lane gap-graph (2026-10-08, docs/plans/gaps-2026-10-08.md 5.1):
            # a scipy.sparse matrix or an (indptr, indices, n) tuple goes to
            # the device as it is (`xn_louvain_csr`): the graph slot is
            # filled from the lists, the symmetry, column-order and no-edge
            # checks run on the device (the op raises their ValueError
            # text), no dense matrix. Glue only: pointer handoff.
            indptr, indices, vals, n = csr
            has_vals = vals is not None
            if not has_vals:
                vals = empty((1,), "<f4")
            labels = empty((n,), "<i4")
            info = _empty_out((2,), "<f4")
            try:
                self._op("louvain_csr", [(indptr, 0), (indices, 0), (vals, 0), (labels, 1), (info, 1)],
                         (n, indices.shape[0], 1 if has_vals else 0, ml),
                         (_f32_scalar(self.resolution), _f32_scalar(self.threshold)))
            except Exception as e:  # the device checks' messages as ValueErrors
                msg = str(e)
                if "Louvain:" in msg:
                    raise ValueError(msg[msg.index("Louvain:"):]) from None
                raise
        else:
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
        """Detect communities and return ``labels_``.

        Parameters
        ----------
        A : array-like of shape (n, n), scipy.sparse matrix, or tuple
            Symmetric weighted adjacency matrix.
        y : None
            Ignored.

        Returns
        -------
        labels : Array of shape (n,), int32
            Community of each node.
        """
        return self.fit(A).labels_


# ====================================================================== SVGP
# Reference: GPflow `gpflow/models/svgp.py` (SVGP with a Gaussian
# likelihood, a squared-exponential kernel, zero mean). q(u) is set to its
# optimum for the given hyperparameters (Titsias 2009), where GPflow's `elbo`
# equals the collapsed bound reported as `elbo_` (DEVIATION 5205: GPflow
# trains the hyperparameters and q by gradient steps). The m x m system runs
# as staged items (x_neighbors/items.mojo `svgp_*_item`: a Cholesky column
# per launch, one triangular solve per right-hand side); the fit is the
# float-float resident chain `xn_svgp_fit_ff` (x_neighbors/svgp_ff.mojo).
class SVGP(RegressorScoreMixin, _XNeighbors):
    """Sparse variational Gaussian process regression with inducing points.

    A Gaussian process regressor with a Gaussian likelihood, a
    squared-exponential kernel and zero mean, in the form of GPflow's
    ``SVGP``. The variational distribution q(u) = N(q_mu, q_sqrt q_sqrt^T)
    is set to its optimum for the given hyperparameters (Titsias 2009), and
    ``elbo_`` is the resulting bound. The hyperparameters are used as given
    and are not optimized (GPflow trains them by gradient steps). Inputs and
    outputs are float32; the fit solves the inducing-point system in
    extended (float-float) precision.

    Parameters
    ----------
    n_inducing : int, default 32
        Number of inducing points when ``inducing_points`` is None: that
        many training rows, evenly spaced (row ``i * n // M``), at most n.
    inducing_points : array-like of shape (M, n_features) or None, default None
        Inducing point locations.
    kernel_variance : float, default 1.0
        Signal variance of the kernel.
    lengthscale : float, default 1.0
        Kernel lengthscale.
    noise_variance : float, default 1.0
        Variance of the Gaussian likelihood.
    jitter : float, default 1e-6
        Diagonal jitter added to the inducing-point kernel matrix.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator (``'fast'`` is Apple only). None uses
        the process default.

    Attributes
    ----------
    Z_ : Array of shape (M, n_features)
        Inducing points.
    q_mu_ : Array of shape (M,)
        Mean of q(u) (non-whitened).
    q_sqrt_ : Array of shape (M, M)
        Square root of the covariance of q(u) (non-whitened).
    elbo_ : float
        Evidence lower bound at the optimal q(u).
    precision_ : str
        Precision of the fit's linear algebra, ``'float-float'``.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 4), dtype=np.float32)
    >>> y = np.sin(6 * X[:, 0]).astype(np.float32)
    >>> model = mojolearn.SVGP(n_inducing=32, lengthscale=0.3, noise_variance=0.01).fit(X, y)
    >>> mean, var = model.predict_y(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "regressor"

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
        """Fit the variational distribution for the given hyperparameters.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training inputs.
        y : array-like of shape (n_samples,)
            Training targets.

        Returns
        -------
        self : SVGP
            The fitted model.

        Raises
        ------
        ValueError
            If X and y have different numbers of rows, or the inducing-point
            system is not positive definite (raise ``jitter`` or
            ``noise_variance``).
        """
        X = _f32(X)
        n, d = X.shape
        yv = _f32_1d(y, "y")
        if len(yv) != n:
            raise ValueError("X and y have different numbers of rows")
        if self.inducing_points is not None:
            Z = _f32(self.inducing_points, "inducing_points")
        else:
            M = min(int(self.n_inducing), n)
            Z = self._take_rows(X, [i * n // M for i in range(M)])  # glue: evenly spaced inducing row indices (M sized)
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
        """Mean and variance of the latent function at X.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query inputs.

        Returns
        -------
        mean : Array of shape (n_queries,), float32
        var : Array of shape (n_queries,), float32
        """
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
        """Mean and variance of the observations at X (latent variance plus
        ``noise_variance``).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query inputs.

        Returns
        -------
        mean : Array of shape (n_queries,), float32
        var : Array of shape (n_queries,), float32
        """
        mean, var = self.predict_f(X)
        return mean, self._unary(var, _U_IDENTITY, 1.0, _f32_scalar(self.noise_variance))

    def predict(self, X):
        """Predictive mean at X (the mean of :meth:`predict_f`).

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query inputs.

        Returns
        -------
        mean : Array of shape (n_queries,), float32
        """
        return self.predict_f(X)[0]
