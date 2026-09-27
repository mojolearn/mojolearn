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

__all__ = ["LocalOutlierFactor"]

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

