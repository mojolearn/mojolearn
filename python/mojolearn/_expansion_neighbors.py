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

from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, as_i32_c, empty
from ._mode import NumericModeMixin

__all__ = ["LocalOutlierFactor", "NearestCentroid", "OneClassSVM", "KernelPCA", "PolynomialCountSketch", "AdditiveChi2Sampler", "SkewedChi2Sampler", "LabelPropagation", "LabelSpreading"]

# x_neighbors/items.mojo's codes
_KERNELS = {"linear": 0, "poly": 1, "polynomial": 1, "rbf": 2, "sigmoid": 3, "laplacian": 4,
            "cosine": 5, "chi2": 6, "additive_chi2": 7}
_U_EXP, _U_LOG, _U_SQRT, _U_TANH, _U_COS, _U_SIN, _U_IDENTITY, _U_RECIP = range(8)


def _f32(X, name="X"):
    a, _ = as_f32_c(X, ndim=2, name=name)
    return a


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

    def _knn(self, Q, R, k, exclude_self):
        """Exact k-NN, euclidean: (distances, indices), ascending by (distance,
        index)."""
        sq, idx = self._knn_select(self._sqdist(Q, R), k, exclude_self)
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
    Sparse input is not implemented. Float32 where sklearn is float64.
    DEVIATION 5201: a feature whose shrink scale m*s is zero gets deviation 0
    where theirs divides by zero.
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
        self._op("nc_std", [(X, 0), (lab, 0), (cent, 0), (stats, 1)], (n, d, C))
        std = stats.tolist()
        if all(v == 0.0 for v in std) and self._ptp_zero(X):
            raise ValueError("All features have zero variance. Division by zero.")
        std_sorted = sorted(std)
        med_std = _f32_scalar(_median(std_sorted))
        nk = Array.from_list([float(c) for c in counts], "<f4")
        shrink = float(self.shrink_threshold) if self.shrink_threshold else 0.0
        self._op("nc_shrink", [(X, 0), (cent, 0), (nk, 0), (stats, 0), (new_cent, 1)],
                 (n, d, C, 1 if shrink else 0), (med_std, shrink))
        self.centroids_ = new_cent
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
    10_000_000 iterations. kernel='precomputed' and callables are refused.
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
        if sample_weight is not None:
            raise NotImplementedError("OneClassSVM: sample_weight is not implemented")
        if self.kernel not in ("linear", "poly", "rbf", "sigmoid"):
            raise NotImplementedError(f"OneClassSVM: kernel={self.kernel!r} is not implemented")
        if not (0.0 < float(self.nu) <= 1.0):
            raise ValueError("nu must be in (0, 1]")
        X = _f32(X)
        n, d = X.shape
        self._gamma = _f32_scalar(_resolve_gamma(self.gamma, self.kernel, X, self))
        Q = self._kernel(X, X, self.kernel, self._gamma, self.coef0, self.degree)
        nl = float(self.nu) * n
        whole = int(math.floor(nl))
        init = [1.0] * min(whole, n) + [0.0] * (n - min(whole, n))
        if whole < n:
            init[whole] = nl - whole
        alpha = Array.from_list(init, "<f4")
        info = empty((1,), "<f4")
        iters = empty((1,), "<i4")
        cap = 10_000_000 if int(self.max_iter) < 0 else int(self.max_iter)
        self._op("ocsvm", [(Q, 0), (alpha, 1), (info, 1), (iters, 1)], (n, cap), (_f32_scalar(self.tol),), )
        # the op's scalar order is (n, eps, max_iter): ints (n, max_iter), floats (eps,)
        rho = info.tolist()[0]
        a = alpha.tolist()
        support = [i for i in range(n) if a[i] > 0]
        self.support_ = Array.from_list(support, "<i4")
        self.support_vectors_ = self._take_rows(X, support)
        self.dual_coef_ = Array.from_list([[a[i] for i in support]], "<f4")
        self.intercept_ = Array.from_list([-rho], "<f4")
        self.offset_ = rho
        self.n_iter_ = iters.tolist()[0]
        self.n_features_in_ = d
        self.fit_status_ = 0
        return self

    def score_samples(self, X):
        Q = _f32(X)
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
    and randomized are approximations of it); fit_inverse_transform,
    kernel='precomputed' and callables are refused by name.
    """

    def __init__(self, n_components=None, *, kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver="auto",
                 tol=0, max_iter=None, iterative_power="auto", remove_zero_eig=False,
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
        self.iterative_power = iterative_power
        self.remove_zero_eig = remove_zero_eig
        self.random_state = random_state
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    def _k(self, A, B):
        return self._kernel(A, B, self.kernel, self._gamma, self.coef0, self.degree)

    def fit(self, X, y=None):
        if self.kernel not in _KERNELS:
            raise NotImplementedError(f"KernelPCA: kernel={self.kernel!r} is not implemented")
        if self.fit_inverse_transform:
            raise NotImplementedError("KernelPCA: fit_inverse_transform is not implemented")
        if self.kernel_params:
            raise NotImplementedError("KernelPCA: kernel_params is not implemented")
        X = _f32(X)
        n, d = X.shape
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
        if seed is None:
            raise ValueError("random_state=None draws from the OS; pass an int for a reproducible fit")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise NotImplementedError("random_state must be an int here (a RandomState instance is not carried)")
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


# ====================================================================== PolynomialCountSketch
class PolynomialCountSketch(_XNeighbors):
    """Polynomial kernel approximation by tensor sketch.

    Reference: scikit-learn `kernel_approximation.py` (PolynomialCountSketch,
    1.9.0): `indexHash_` and `bitHash_` are drawn exactly as theirs
    (`randint(0, n_components, (degree, n_features))`, then
    `choice([-1, 1], (degree, n_features))`, one legacy RandomState stream);
    the transform is the count sketches' circular convolution, computed
    directly instead of through an FFT (DEVIATION 5203). random_state must be
    an int. Sparse input is not implemented.
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
        rs = _LegacyRandomState(self.random_state)
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
        self._op("pcs", [(X, 0), (self.indexHash_, 0), (self.bitHash_, 0), (out, 1)],
                 (n, d, nf, nc, deg), (_f32_scalar(self.gamma), _f32_scalar(self.coef0)))
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# ====================================================================== AdditiveChi2Sampler
class AdditiveChi2Sampler(_XNeighbors):
    """Approximate feature map for the additive chi-squared kernel.

    Reference: scikit-learn `kernel_approximation.py` (AdditiveChi2Sampler,
    `_transform_dense`, 1.9.0). Deterministic: no random state. Negative input
    is refused, as theirs. Sparse input is not implemented.
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
        X = _f32(X)
        if X.size and X.min() < 0:
            raise ValueError("Negative values in data passed to AdditiveChi2Sampler")
        n, d = X.shape
        steps = int(self.sample_steps)
        out = empty((n, d * (2 * steps - 1)), "<f4")
        self._op("achi2", [(X, 0), (out, 1)], (n, d, steps), (_f32_scalar(self._interval()),))
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


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
        rs = _LegacyRandomState(self.random_state)
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
            _, idx = self._knn_select(self._sqdist(X, X), k, False)
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
            _, idx = self._knn_select(self._sqdist(Q, self.X_), k, False)
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
        self._op("ls_laplacian", [(A, 0), (G, 1)], (n,))
        return G

