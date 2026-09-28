# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE NEIGHBORS LANE'S PUBLIC DOOR (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

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
import math
import os
import struct

from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, as_i32_c, empty
from ._mode import NumericModeMixin

__all__ = ["LocalOutlierFactor", "NearestCentroid", "OneClassSVM", "KernelPCA", "PolynomialCountSketch",
           "AdditiveChi2Sampler", "SkewedChi2Sampler", "LabelPropagation", "LabelSpreading", "KNNImputer",
           "PageRank", "connected_components", "Louvain", "SVGP"]

# x_neighbors/items.mojo's codes
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
    """sklearn's accuracy_score: the (weighted) fraction of exact label matches,
    in IEEE double on labels compared exactly."""
    t = y_true.tolist() if hasattr(y_true, "tolist") else list(y_true)
    p = y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)
    if len(t) != len(p):
        raise ValueError("y_true and y_pred have different lengths")
    if sample_weight is None:
        return math.fsum(1.0 for a, b in zip(t, p) if a == b) / len(t)
    w = [float(v) for v in (sample_weight.tolist() if hasattr(sample_weight, "tolist") else sample_weight)]
    return math.fsum(wi for a, b, wi in zip(t, p, w) if a == b) / math.fsum(w)


def _f32_1d(x, name):
    a, _ = as_f32_c(x, ndim=1, name=name)
    return a


def _i32(x, name):
    a, _ = as_i32_c(x, ndim=1, name=name)
    return a


def _f32_scalar(v):
    """A Python double rounded once to float32, as a Python float."""
    return Array.from_list([float(v)], "<f4").tolist()[0]


def _labels_of(y):
    y = y.tolist() if hasattr(y, "tolist") else list(y)
    classes = sorted(set(y))
    code = {c: i for i, c in enumerate(classes)}
    return classes, [code[v] for v in y]


def _class_array(classes, codes):
    vals = [classes[c] for c in codes]
    if all(isinstance(v, bool) or isinstance(v, int) for v in classes):
        return Array.from_list([int(v) for v in vals], "<i8")
    if all(isinstance(v, (int, float)) for v in classes):
        return Array.from_list([float(v) for v in vals], "<f8")
    return vals


#: A/B arm (lane neighbors-apple2): the k-NN primitive as the two ops it
#: fuses, `sqdist` then `knn_select` through an n x m matrix. Same bits.
_UNFUSED_KNN = os.environ.get("MOJOLEARN_XN_UNFUSED_KNN", "") == "1"
#: A/B arm (lane neighbors-apple2): the fit loops of label propagation /
#: spreading, PageRank and connected_components in Python, one op per step,
#: instead of the resident `lp_iterate` / `pr_iterate` / `cc_iterate`.
_HOST_LOOP_LP = os.environ.get("MOJOLEARN_XN_HOST_LOOPS", "") == "1"
#: A/B arm: KNNImputer.transform over every cell instead of the missing ones.
_UNCOMPACT_IMPUTE = os.environ.get("MOJOLEARN_XN_UNCOMPACT_IMPUTE", "") == "1"
#: A/B arm: the one-item forms these replaced (PolynomialCountSketch's
#: per-row `pcs`, LabelSpreading's per-cell-degree `ls_laplacian`, PageRank's
#: dangling rows in Python).
_OLD_ITEMS = os.environ.get("MOJOLEARN_XN_OLD_ITEMS", "") == "1"


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
        out = empty((n, m), "<f4")
        self._op("sqdist", [(A, 0), (B, 0), (out, 1)], (n, m, d))
        return out

    def _l1dist(self, A, B):
        n, d = A.shape
        m = B.shape[0]
        out = empty((n, m), "<f4")
        self._op("l1dist", [(A, 0), (B, 0), (out, 1)], (n, m, d))
        return out

    def _kernel(self, A, B, kind, gamma, coef0, degree):
        n, d = A.shape
        m = B.shape[0]
        out = empty((n, m), "<f4")
        self._op("kernel", [(A, 0), (B, 0), (out, 1)], (n, m, d, _KERNELS[kind], int(degree)),
                 (_f32_scalar(gamma), _f32_scalar(coef0)))
        return out

    def _matmul(self, A, B):
        n, k = A.shape
        m = B.shape[1]
        out = empty((n, m), "<f4")
        self._op("matmul", [(A, 0), (B, 0), (out, 1)], (n, k, m))
        return out

    def _unary(self, X, op, a=1.0, b=0.0):
        out = empty(X.shape, "<f4")
        self._op("unary", [(X, 0), (out, 1)], (X.size, op), (a, b))
        return out

    def _knn_select(self, D, k, exclude_self):
        n, m = D.shape
        dist = empty((n, k), "<f4")
        idx = empty((n, k), "<i4")
        self._op("knn_select", [(D, 0), (dist, 1), (idx, 1)], (n, m, k, 1 if exclude_self else 0))
        return dist, idx

    def _rowsum(self, A):
        out = empty((A.shape[0],), "<f4")
        self._op("rowsum", [(A, 0), (out, 1)], A.shape)
        return out

    def _colsum(self, A):
        out = empty((A.shape[1],), "<f4")
        self._op("colsum", [(A, 0), (out, 1)], A.shape)
        return out

    def _scale_div(self, X, s):
        out = empty(X.shape, "<f4")
        self._op("scale_div", [(X, 0), (out, 1)], (X.size,), (s,))
        return out

    def _take_rows(self, X, rows):
        rows = _i32(rows, "rows")
        out = empty((len(rows), X.shape[1]), "<f4")
        self._op("take_rows", [(X, 0), (rows, 0), (out, 1)], (len(rows), X.shape[1], X.shape[0]))
        return out

    def _take_cols(self, X, cols):
        cols = _i32(cols, "cols")
        out = empty((X.shape[0], len(cols)), "<f4")
        self._op("take_cols", [(X, 0), (cols, 0), (out, 1)], (X.shape[0], X.shape[1], len(cols)))
        return out

    def _variance(self, X):
        out = empty((1,), "<f4")
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
        dist = empty((n, k), "<f4")
        idx = empty((n, k), "<i4")
        self._op("knn_sq", [(Q, 0), (R, 0), (dist, 1), (idx, 1)], (n, m, d, k, 1 if exclude_self else 0))
        return dist, idx

    def _knn(self, Q, R, k, exclude_self):
        """Exact k-NN, euclidean: (distances, indices), ascending by (distance,
        index)."""
        sq, idx = self._knn_sq(Q, R, k, exclude_self)
        return self._unary(sq, _U_SQRT), idx


def _percentile(values, q):
    """numpy.percentile, method 'linear', on Python doubles."""
    v = sorted(values)
    n = len(v)
    pos = (q / 100.0) * (n - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    t = pos - lo
    a, b = v[lo], v[hi]
    diff = b - a
    return b - diff * (1.0 - t) if t >= 0.5 else a + diff * t


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
        lrd = empty((n,), "<f4")
        self._op("lof_lrd", [(dist, 0), (idx, 0), (dist, 0), (lrd, 1)], (k, n, n))
        score = empty((n,), "<f4")
        self._op("lof_score", [(idx, 0), (lrd, 0), (lrd, 0), (score, 1)], (k, n, n))
        self._fit_X, self._fit_dist, self._lrd = X, dist, lrd
        self.n_neighbors_ = k
        self.n_features_in_ = d
        self.n_samples_fit_ = n
        self.negative_outlier_factor_ = score
        if self.contamination == "auto":
            self.offset_ = -1.5
        else:
            self.offset_ = _f32_scalar(_percentile(score.tolist(), 100.0 * float(self.contamination)))
        return self

    def fit_predict(self, X, y=None):
        if self.novelty:
            raise AttributeError("fit_predict is not available when novelty=True; use predict on new data")
        self.fit(X)
        off = self.offset_
        return Array.from_list([-1 if s < off else 1 for s in self.negative_outlier_factor_.tolist()], "<i8")

    def _novelty(self, what):
        if not self.novelty:
            raise AttributeError(f"{what} is not available when novelty=False; use fit_predict")

    def score_samples(self, X):
        self._novelty("score_samples")
        Q = _f32(X)
        k, nf = self.n_neighbors_, self.n_samples_fit_
        dist, idx = self._knn(Q, self._fit_X, k, False)
        nq = Q.shape[0]
        lrd = empty((nq,), "<f4")
        self._op("lof_lrd", [(dist, 0), (idx, 0), (self._fit_dist, 0), (lrd, 1)], (k, nq, nf))
        score = empty((nq,), "<f4")
        self._op("lof_score", [(idx, 0), (self._lrd, 0), (lrd, 0), (score, 1)], (k, nq, nf))
        return score

    def decision_function(self, X):
        self._novelty("decision_function")
        return self._unary(self.score_samples(X), _U_IDENTITY, 1.0, -self.offset_)

    def predict(self, X):
        self._novelty("predict")
        return Array.from_list([1 if v >= 0 else -1 for v in self.decision_function(X).tolist()], "<i8")


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
        classes, codes = _labels_of(y)
        if len(codes) != n:
            raise ValueError("X and y have different numbers of rows")
        C = len(classes)
        if C < 2:
            raise ValueError(f"The number of classes has to be greater than one; got {C} class")
        counts = [0] * C
        for c in codes:
            counts[c] += 1
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
        lab = _i32(codes, "y")
        if self.metric == "euclidean":
            cent = empty((C, d), "<f4")
            self._op("group_mean", [(X, 0), (lab, 0), (cent, 1)], (n, d, C))
        else:
            rows = X.tolist()
            med = []
            for c in range(C):
                members = [rows[i] for i in range(n) if codes[i] == c]
                med.append([_median([r[f] for r in members]) for f in range(d)])
            cent = Array.from_list(med, "<f4")
        stats = empty((d,), "<f4")
        new_cent = empty((C, d), "<f4")
        devs = empty((C, d), "<f4")
        self._op("nc_std", [(X, 0), (lab, 0), (cent, 0), (stats, 1)], (n, d, C))
        std = stats.tolist()
        if all(v == 0.0 for v in std) and self._ptp_zero(X):
            raise ValueError("All features have zero variance. Division by zero.")
        std_sorted = sorted(std)
        med_std = _f32_scalar(_median(std_sorted))
        nk = Array.from_list([float(c) for c in counts], "<f4")
        shrink = float(self.shrink_threshold) if self.shrink_threshold else 0.0
        self._op("nc_shrink", [(X, 0), (cent, 0), (nk, 0), (stats, 0), (new_cent, 1), (devs, 1)],
                 (n, d, C, 1 if shrink else 0), (med_std, shrink))
        self.centroids_ = new_cent
        self.deviations_ = devs
        self.within_class_std_dev_ = stats
        self.classes_ = classes
        self._codes_classes = classes
        self.n_features_in_ = d
        return self

    @staticmethod
    def _ptp_zero(X):
        rows = X.tolist()
        return all(min(col) == max(col) for col in zip(*rows))

    def _uniform(self):
        C = len(self.classes_)
        return all(math.isclose(p, 1.0 / C, rel_tol=1e-5, abs_tol=1e-8) for p in self.class_prior_.tolist())

    def predict(self, X):
        Q = _f32(X)
        if self._uniform():
            D = self._sqdist(Q, self.centroids_) if self.metric == "euclidean" else self._l1dist(Q, self.centroids_)
            _, idx = self._knn_select(D, 1, False)
            return _class_array(self.classes_, [r[0] for r in idx.tolist()])
        return _class_array(self.classes_, [_argmax(r) for r in self.decision_function(Q).tolist()])

    def decision_function(self, X):
        if self.metric != "euclidean":
            raise AttributeError("decision_function is available for metric='euclidean' only")
        Q = _f32(X)
        C = len(self.classes_)
        prior = Array.from_list(self.class_prior_.tolist(), "<f4")
        std = self.within_class_std_dev_
        out = empty((Q.shape[0], C), "<f4")
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


def _median(vals):
    v = sorted(vals)
    n = len(v)
    if n % 2:
        return v[n // 2]
    return (v[n // 2 - 1] + v[n // 2]) / 2.0


def _argmax(row):
    best, at = row[0], 0
    for i, v in enumerate(row):
        if v > best:
            best, at = v, i
    return at


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
    selection, `calculate_rho`). The dual is solved in ONE sequential Mojo
    item (x_neighbors/items.mojo `ocsvm_smo_item`) over the kernel matrix,
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
            rows = list(range(n))
            cvals = [1.0] * n
        else:
            w = [float(v) for v in (sample_weight.tolist() if hasattr(sample_weight, "tolist") else sample_weight)]
            if len(w) != n:
                raise ValueError("sample_weight and X have different numbers of samples")
            if any(v < 0 for v in w):
                raise ValueError("negative sample_weight is not supported")
            rows = [i for i in range(n) if w[i] > 0]
            if not rows:
                raise ValueError("Invalid input - all samples have zero or negative weights.")
            cvals = [w[i] for i in rows]
        m = len(rows)
        if self.kernel == "precomputed":
            self._gamma = 0.0
            Q = X if m == n else self._take_cols(self._take_rows(X, rows), rows)
        else:
            self._gamma = _f32_scalar(_resolve_gamma(self.gamma, self.kernel, X, self))
            Xw = X if m == n else self._take_rows(X, rows)
            Q = self._kernel(Xw, Xw, self.kernel, self._gamma, self.coef0, self.degree)
        cv = Array.from_list(cvals, "<f4")
        cf = cv.tolist()                                  # libsvm's C_i as the solver sees them
        nl = float(self.nu) * m                            # solve_one_class: nu_l = sum(C_i * nu) ...
        if sample_weight is not None:                      # ... accumulated in sample order, in double
            nl = 0.0
            for c in cf:
                nl += c * float(self.nu)
        init = [0.0] * m
        i = 0
        while nl > 0 and i < m:
            init[i] = min(cf[i], nl)
            nl -= init[i]
            i += 1
        alpha = Array.from_list(init, "<f4")
        info = empty((1,), "<f4")
        iters = empty((1,), "<i4")
        cap = 10_000_000 if int(self.max_iter) < 0 else int(self.max_iter)
        self._op("ocsvm", [(Q, 0), (cv, 0), (alpha, 1), (info, 1), (iters, 1)], (m, cap), (_f32_scalar(self.tol),), )
        # the op's scalar order is (n, eps, max_iter): ints (n, max_iter), floats (eps,)
        rho = info.tolist()[0]
        a = alpha.tolist()
        local = [i for i in range(m) if a[i] > 0]
        support = [rows[i] for i in local]
        self.support_ = Array.from_list(support, "<i4")
        if self.kernel == "precomputed":
            self.support_vectors_ = Array.from_list([[] for _ in support], "<f4") if support else empty((0, 0), "<f4")
        else:
            self.support_vectors_ = self._take_rows(X, support)
        self.dual_coef_ = Array.from_list([[a[i] for i in local]], "<f4")
        self.intercept_ = Array.from_list([-rho], "<f4")
        self.offset_ = rho
        self.n_iter_ = iters.tolist()[0]
        self.n_features_in_ = d
        self.fit_status_ = 0
        return self

    def score_samples(self, X):
        Q = _f32(X)
        if self.kernel == "precomputed":
            K = self._take_cols(Q, self.support_.tolist())
        else:
            K = self._kernel(Q, self.support_vectors_, self.kernel, self._gamma, self.coef0, self.degree)
        coef = Array.from_list([[v] for v in self.dual_coef_.tolist()[0]], "<f4")
        s = self._matmul(K, coef)
        return s.reshape((Q.shape[0],))

    def decision_function(self, X):
        return self._unary(self.score_samples(X), _U_IDENTITY, 1.0, -self.offset_)

    def predict(self, X):
        return Array.from_list([1 if v > 0 else -1 for v in self.decision_function(X).tolist()], "<i8")

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)


# ====================================================================== KernelPCA
class KernelPCA(_XNeighbors):
    """Kernel principal component analysis.

    Reference: scikit-learn `decomposition/_kernel_pca.py` (1.9.0) with
    `preprocessing.KernelCenterer`: the kernel matrix, centered in their
    order, the dense eigendecomposition (eigen_solver 'dense'; here the lane's
    host Jacobi, spectral/checks/symmetric_eig_host.mojo, ascending with
    pinned signs, then sklearn's `svd_flip(u, None)` sign rule), eigenvalues
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
        K = self._k(X, X)
        cols = self._scale_div(self._colsum(K), float(n))              # K_fit_rows_
        all_ = self._scale_div(self._colsum(cols.reshape((1, n))), float(n))  # K_fit_all_
        Kc = empty((n, n), "<f4")
        self._op("kpca_center", [(K, 0), (cols, 0), (cols, 0), (all_, 0), (Kc, 1)], (n, n))
        w = empty((n,), "<f4")
        V = empty((n, n), "<f4")
        self._op("eigh", [(Kc, 0), (w, 1), (V, 1)], (n,))
        self._op("svd_flip", [(V, 1)], (n, n))
        wl = w.tolist()
        order = list(range(n - 1, -1, -1))           # descending; equal values: higher index first
        c = n if self.n_components is None else min(n, int(self.n_components))
        order = order[:c]
        vals = [max(wl[i], 0.0) for i in order]
        if self.n_components is None or self.remove_zero_eig:
            keep = [j for j in range(c) if vals[j] > 0]
            order = [order[j] for j in keep]
            vals = [vals[j] for j in keep]
        self.eigenvalues_ = Array.from_list(vals, "<f4")
        self.eigenvectors_ = self._take_cols(V, order)
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
        K = self._k(Q, self._fit_X)
        pred = self._scale_div(self._rowsum(K), float(nf))
        Kc = empty((nq, nf), "<f4")
        self._op("kpca_center", [(K, 0), (self._fit_cols, 0), (pred, 0), (self._fit_all, 0), (Kc, 1)], (nq, nf))
        return self._matmul(Kc, self._alpha_scale(1))

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
        import numpy as np
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
        out = empty((n, nc), "<f4")
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
        if X.size and X.min() < 0:
            raise ValueError("Negative values in data passed to AdditiveChi2Sampler")
        self._interval()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        sparse = X if (hasattr(X, "toarray") and hasattr(X, "nnz")) else None
        X = _f32(X)
        if X.size and X.min() < 0:
            raise ValueError("Negative values in data passed to AdditiveChi2Sampler")
        n, d = X.shape
        steps = int(self.sample_steps)
        out = empty((n, d * (2 * steps - 1)), "<f4")
        self._op("achi2", [(X, 0), (out, 1)], (n, d, steps), (_f32_scalar(self._interval()),))
        if sparse is not None:
            import numpy as np
            import scipy.sparse as sp
            m = sp.csr_matrix(np.asarray(out.tolist(), dtype=np.float32))
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
        rs = _random_state(self.random_state)
        u = rs.random_sample(d * nc)
        z = Array.from_list([[math.pi / 2.0 * u[f * nc + c] for c in range(nc)] for f in range(d)], "<f4")
        w = empty((d, nc), "<f4")
        self._op("skew_weights", [(z, 0), (w, 1)], (d * nc,))
        self.random_weights_ = w
        self.random_offset_ = Array.from_list(rs.uniform(0.0, 2.0 * math.pi, nc), "<f4")
        self.n_features_in_ = d
        return self

    def transform(self, X):
        X = _f32(X)
        n, d = X.shape
        if X.size and X.min() <= -float(self.skewedness):
            raise ValueError("X may not contain entries smaller than -skewedness.")
        nc = int(self.n_components)
        lx = self._unary(X, _U_LOG, 1.0, _f32_scalar(self.skewedness))
        out = empty((n, nc), "<f4")
        self._op("skew_transform", [(lx, 0), (self.random_weights_, 0), (self.random_offset_, 0), (out, 1)], (n, d, nc))
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        return _prefixed_names(self, int(self.n_components))


# ====================================================================== LabelPropagation
class _LabelPropagationBase(_XNeighbors):
    """scikit-learn `semi_supervised/_label_propagation.py` (1.9.0): the dense
    graph (rbf, or the knn connectivity graph with each row's own point as
    its first neighbor), the product / clamp iteration with their stopping
    rule (sum |L - L_prev| < tol, checked before each step), the final row
    normalization and `transduction_`. Callable kernels are refused. Float32
    where theirs is float64; the neighbor ties go to the lower index."""

    _variant = None

    def _graph_affinity(self, X):
        n = X.shape[0]
        if self.kernel == "rbf":
            return self._kernel(X, X, "rbf", self.gamma, 0.0, 0)
        if self.kernel == "knn":
            k = min(int(self.n_neighbors), n)
            _, idx = self._knn_sq(X, X, k, False)
            g = empty((n, n), "<f4")
            self._op("knn_graph", [(idx, 0), (g, 1)], (n, n, k))
            return g
        raise NotImplementedError(f"{type(self).__name__}: kernel={self.kernel!r} is not implemented ('rbf' or 'knn')")

    def fit(self, X, y):
        X = _f32(X)
        n = X.shape[0]
        y = y.tolist() if hasattr(y, "tolist") else list(y)
        if len(y) != n:
            raise ValueError("X and y have different numbers of rows")
        classes = sorted(set(v for v in y if v != -1))
        C = len(classes)
        code = {c: i for i, c in enumerate(classes)}
        unl = [1 if v == -1 else 0 for v in y]
        ld0 = [[1.0 if (v != -1 and code[v] == j) else 0.0 for j in range(C)] for v in y]
        if self._variant == "propagation":
            ys = [[0.0] * C if unl[i] else ld0[i] for i in range(n)]
        else:
            a = _f32_scalar(1.0 - float(self.alpha))
            ys = [[a * v for v in row] for row in ld0]
        G = self._build_graph(X)
        ld = Array.from_list(ld0, "<f4")
        ystatic = Array.from_list(ys, "<f4")
        unlabeled = _i32(unl, "unlabeled")
        if not _HOST_LOOP_LP:
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
        prev = empty((n, C), "<f4")
        s = empty((1,), "<f4")
        n_iter = 0
        converged = False
        for it in range(int(self.max_iter)):
            n_iter = it
            self._op("absdiff_sum", [(ld, 0), (prev, 0), (s, 1)], (n * C,))
            if s.tolist()[0] < float(self.tol):
                converged = True
                break
            prev = ld
            nxt = self._matmul(G, ld)
            out = empty((n, C), "<f4")
            if self._variant == "propagation":
                self._op("lp_clamp", [(nxt, 0), (ystatic, 0), (unlabeled, 0), (out, 1)], (n, C))
            else:
                self._op("ls_clamp", [(nxt, 0), (ystatic, 0), (out, 1)], (n * C,), (_f32_scalar(self.alpha),))
            ld = out
        if not converged:
            n_iter += 1
        return self._finish_fit(X, classes, ld, n, C, n_iter)

    def _finish_fit(self, X, classes, ld, n, C, n_iter):
        final = empty((n, C), "<f4")
        self._op("row_normalize", [(ld, 0), (final, 1)], (n, C))
        self.X_ = X
        self.classes_ = classes
        self.label_distributions_ = final
        self.n_iter_ = n_iter
        self.transduction_ = _class_array(classes, [_argmax(r) for r in final.tolist()])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        Q = _f32(X)
        nq = Q.shape[0]
        n = self.X_.shape[0]
        if self.kernel == "knn":
            k = min(int(self.n_neighbors), n)
            _, idx = self._knn_sq(Q, self.X_, k, False)
            W = empty((nq, n), "<f4")
            self._op("knn_graph", [(idx, 0), (W, 1)], (nq, n, k))
        else:
            W = self._kernel(Q, self.X_, "rbf", self.gamma, 0.0, 0)
        P = self._matmul(W, self.label_distributions_)
        out = empty(P.shape, "<f4")
        self._op("row_normalize", [(P, 0), (out, 1)], P.shape)
        return out

    def predict(self, X):
        return _class_array(self.classes_, [_argmax(r) for r in self.predict_proba(X).tolist()])

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
        G = empty((n, n), "<f4")
        if _OLD_ITEMS:
            self._op("ls_laplacian", [(A, 0), (G, 1)], (n,))
        else:
            deg = empty((n,), "<f4")
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

    def _masked(self, X):
        X = _f32(X)
        mv = self.missing_values
        if mv == mv:                                     # a number: its cells become NaN
            want = _f32_scalar(mv)
            X = Array.from_list([[float("nan") if v == want else v for v in r] for r in X.tolist()], "<f4")
        return X

    def fit(self, X, y=None):
        self._check()
        X = self._masked(X)
        n, d = X.shape
        rows = X.tolist()
        miss = [[v != v for v in r] for r in rows]
        self._valid = [not all(miss[i][f] for i in range(n)) for f in range(d)]
        self._miss_cols = [f for f in range(d) if any(miss[i][f] for i in range(n))]
        self._fit_X = X
        self.n_features_in_ = d
        return self

    def transform(self, X):
        X = self._masked(X)
        n, d = X.shape
        if d != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        m = self._fit_X.shape[0]
        out = empty((n, d), "<f4")
        k = int(self.n_neighbors)
        if k < 1:
            raise ValueError("n_neighbors must be >= 1")
        if _UNCOMPACT_IMPUTE:
            self._op("knn_impute", [(X, 0), (self._fit_X, 0), (out, 1)],
                     (n, m, d, k, 1 if self.weights == "distance" else 0))
        else:
            # one GPU thread per MISSING cell (`knn_impute_cells`): the same
            # item statements; a present cell keeps x, as the item stores it
            flat = X.reshape((n * d,)).tolist()
            cells = [i for i, v in enumerate(flat) if v != v]
            out = Array.from_list(flat, "<f4").reshape((n, d))
            if cells:
                self._op("knn_impute_cells", [(_i32(cells, "cells"), 0), (X, 0), (self._fit_X, 0), (out, 1)],
                         (n, m, d, k, 1 if self.weights == "distance" else 0, len(cells)))
        keep = [f for f in range(d) if self._valid[f]]
        if self.keep_empty_features:
            empty_cols = [f for f in range(d) if not self._valid[f]]
            if empty_cols:
                rows = out.tolist()
                for r in rows:
                    for f in empty_cols:
                        r[f] = 0.0
                out = Array.from_list(rows, "<f4")
        elif len(keep) != d:
            out = self._take_cols(out, keep)
        if self.add_indicator and self._miss_cols:
            src = X.tolist()
            res = out.tolist()
            for i, r in enumerate(res):
                r.extend(1.0 if src[i][f] != src[i][f] else 0.0 for f in self._miss_cols)
            out = Array.from_list(res, "<f4")
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
        if self.weight is None or self.weight is False:
            A = Array.from_list([[1.0 if v != 0 else 0.0 for v in r] for r in A.tolist()], "<f4")
        Q = empty((n, n), "<f4")
        self._op("row_normalize", [(A, 0), (Q, 1)], (n, n))
        if _OLD_ITEMS:
            dangling = _i32([1 if all(v == 0 for v in r) else 0 for r in A.tolist()], "dangling")
        else:
            dangling = empty((n,), "<i4")
            self._op("row_all_zero", [(A, 0), (dangling, 1)], (n, n))
        if self.personalization is None:
            p = Array.from_list([1.0 / n] * n, "<f4")
        else:
            p = self._unit(self.personalization, n, "personalization")
        dw = p if self.dangling is None else self._unit(self.dangling, n, "dangling")
        x = Array.from_list([1.0 / n] * n, "<f4") if self.nstart is None else self._unit(self.nstart, n, "nstart")
        if not _HOST_LOOP_LP:
            # The loop below as ONE resident op (x_neighbors/iter_device.mojo),
            # Q uploaded once; n * tol passed as its float64 bits.
            info = empty((2,), "<i4")
            thr = struct.unpack("<Q", struct.pack("<d", n * float(self.tol)))[0]
            x = Array.from_list(x.tolist(), "<f4")
            self._op("pr_iterate", [(Q, 0), (x, 1), (p, 0), (dw, 0), (dangling, 0), (info, 1)],
                     (n, int(self.max_iter), thr >> 32, thr & 0xFFFFFFFF), (_f32_scalar(self.alpha),))
            it, ok = info.tolist()
            if ok:
                self.pagerank_ = x
                self.n_iter_ = int(it)
                return self
            raise RuntimeError(f"PageRank: power iteration failed to converge within {self.max_iter} iterations")
        s = empty((1,), "<f4")
        for it in range(int(self.max_iter)):
            nxt = empty((n,), "<f4")
            self._op("pagerank_step", [(Q, 0), (x, 0), (p, 0), (dw, 0), (dangling, 0), (nxt, 1)], (n,), (_f32_scalar(self.alpha),))
            self._op("absdiff_sum", [(nxt, 0), (x, 0), (s, 1)], (n,))
            x = nxt
            if s.tolist()[0] < n * float(self.tol):
                self.pagerank_ = x
                self.n_iter_ = it + 1
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
    A = _adjacency(A)
    n = A.shape[0]
    lab = _i32(list(range(n)), "labels")
    if not _HOST_LOOP_LP:
        # the loop below as ONE resident op, A uploaded once
        info = empty((1,), "<i4")
        est._op("cc_iterate", [(A, 0), (lab, 1), (info, 1)], (n,))
    while _HOST_LOOP_LP:
        nxt = empty((n,), "<i4")
        est._op("cc_step", [(A, 0), (lab, 0), (nxt, 1)], (n,))
        if nxt.tolist() == lab.tolist():
            break
        lab = nxt
    roots = {}
    out = []
    for v in lab.tolist():
        out.append(roots.setdefault(v, len(roots)))
    labels = Array.from_list(out, "<i4")
    return (len(roots), labels) if return_labels else len(roots)


# ====================================================================== Louvain
class Louvain(_XNeighbors):
    """Louvain community detection on a dense symmetric weighted adjacency.

    References: networkx `louvain_communities` / `louvain_partitions`
    (`_one_level`, `_gen_graph`, `modularity`) and cuGraph
    cpp/src/community/louvain_impl.cuh. The whole method is ONE sequential
    Mojo item (x_neighbors/items.mojo `louvain_item`) with a PINNED order
    (DEVIATION 5204): nodes in ascending id instead of networkx's `seed`
    shuffle, candidate communities in ascending id, a strictly larger gain
    to move, so ties go to the lowest community id. `labels_` numbers the
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
        rows = A.tolist()
        if any(rows[i][j] != rows[j][i] for i in range(n) for j in range(i + 1, n)):
            raise ValueError("Louvain: the adjacency matrix must be symmetric (an undirected graph)")
        if all(v == 0 for r in rows for v in r):
            raise ValueError("Louvain: the graph has no edges")
        labels = empty((n,), "<i4")
        info = empty((2,), "<f4")
        ml = 0 if self.max_level is None else int(self.max_level)
        if self.max_level is not None and ml < 1:
            raise ValueError("max_level must be a positive integer or None")
        self._op("louvain", [(A, 0), (labels, 1), (info, 1)], (n, ml),
                 (_f32_scalar(self.resolution), _f32_scalar(self.threshold)))
        roots = {}
        self.labels_ = Array.from_list([roots.setdefault(v, len(roots)) for v in labels.tolist()], "<i4")
        self.n_communities_ = len(roots)
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
    (row i * n // M). The m x m system is ONE sequential item
    (x_neighbors/items.mojo `svgp_item`); the kernel matrices and products are
    parallel items. Float32 throughout.
    """

    def __init__(self, n_inducing=32, *, inducing_points=None, kernel_variance=1.0, lengthscale=1.0,
                 noise_variance=1.0, jitter=1e-6):
        self.n_inducing = n_inducing
        self.inducing_points = inducing_points
        self.kernel_variance = kernel_variance
        self.lengthscale = lengthscale
        self.noise_variance = noise_variance
        self.jitter = jitter

    def _k(self, A, B):
        g = 1.0 / (2.0 * float(self.lengthscale) ** 2)
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
        Kuu = self._k(Z, Z)
        Kfu = self._k(X, Z)
        Kuf = self._k(Z, X)
        B = self._matmul(Kuf, Kfu)
        b = self._matmul(Kuf, yv.reshape((n, 1)))
        alpha = empty((M,), "<f4")
        C = empty((M, M), "<f4")
        qmu = empty((M,), "<f4")
        qsqrt = empty((M, M), "<f4")
        info = empty((2,), "<f4")
        self._op("svgp", [(Kuu, 0), (B, 0), (b, 0), (yv, 0), (alpha, 1), (C, 1), (qmu, 1), (qsqrt, 1), (info, 1)],
                 (M, n), (_f32_scalar(self.noise_variance), _f32_scalar(self.jitter), _f32_scalar(self.kernel_variance)))
        elbo, ok = info.tolist()
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
        Ksu = self._k(Q, self.Z_)
        M = self.Z_.shape[0]
        mean = self._matmul(Ksu, self._alpha.reshape((M, 1))).reshape((Q.shape[0],))
        var = empty((Q.shape[0],), "<f4")
        self._op("svgp_var", [(Ksu, 0), (self._C, 0), (var, 1)], (Q.shape[0], M), (_f32_scalar(self.kernel_variance),))
        return mean, var

    def predict_y(self, X):
        mean, var = self.predict_f(X)
        return mean, self._unary(var, _U_IDENTITY, 1.0, _f32_scalar(self.noise_variance))

    def predict(self, X):
        return self.predict_f(X)[0]
