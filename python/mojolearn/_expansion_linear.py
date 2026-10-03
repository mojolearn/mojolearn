# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINEAR LANE'S PUBLIC DOOR.

Owned by the `linear` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`; a name
that is already public is refused at import. Put the classes here, or import
them here from the lane's own modules. Two optional hooks route a SAVED model
to the CPU (`mojolearn.host_model`); `_classical_host.py` merges them when it is
first imported, after the package, so both may rely on every module existing:

  CLASSICAL_HOST_BASENAMES     {"_mojolearn_x_linear": "_mojolearn_x_linear_host"}
  def classical_host_formats() -> {"mojolearn-<format>-1": {"Estimator": HostEstimator}}
                               import `_classical_host` INSIDE it and subclass
                               its `_HostBound`; never import it at module level

Every fit here is one call into `_mojolearn_x_linear` (`x_linear_fit`), whose
source is x_linear/ (one sequential schedule, the same on the CPU host
binding and on the device); every score is `x_linear_decision`. Python only
validates, encodes labels, assigns CV folds (integers) and unpacks the flat
float32 result. NumPy-free (NUMPY_FREE_CONTRACT.md).
"""
import collections
import os
from . import _portable_math as _pm
from . import _backend
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty, full, zeros
from ._bufcheck import memcopy
from ._labels import argmax_rows, decode_labels, encode_labels, threshold_codes
from ._mode import NumericModeMixin

__all__ = ["SGDClassifier", "SGDRegressor", "PoissonRegressor", "GammaRegressor", "TweedieRegressor",
           "HuberRegressor",
           "BayesianRidge", "ARDRegression",
           "Lars", "LassoLars",
           "QuantileRegressor",
           "Perceptron", "PassiveAggressiveClassifier",
           "PassiveAggressiveRegressor", "SGDOneClassSVM",
           "RidgeClassifier", "RidgeCV",
           "LassoCV", "ElasticNetCV", "LogisticRegressionCV",
           "IsotonicRegression"]

_BINDING = "_mojolearn_x_linear"
ALGO_SGD, ALGO_GLM, ALGO_HUBER, ALGO_BAYES, ALGO_ARD = 1, 2, 3, 4, 5
ALGO_LARS, ALGO_QUANTILE, ALGO_RIDGE, ALGO_ENETCV, ALGO_LOGCV, ALGO_ISOTONIC = 6, 7, 8, 9, 10, 11
ALGO_ISOTONIC_PREDICT = 12
ALGO_RIDGE_KFOLD = 13
LINK_IDENTITY, LINK_EXP, LINK_SIGMOID = 0, 1, 2


# ------------------------------------------------------------------ plumbing

def _matrix(X, name="X"):
    a, _ = as_f32_c(X, ndim=2, name=name)
    n, d = a.shape
    if n == 0 or d == 0:
        raise ValueError(f"mojolearn: {name} must have at least one row and one column")
    # finiteness is checked by the binding (x_linear_fit), on both columns
    return a, n, d


def _vector(y, n, name="y"):
    a, _ = as_f32_c(y, ndim=1, name=name)
    if a.shape[0] != n:
        raise ValueError(f"mojolearn: X and {name} lengths differ")
    return a


def _fit_module(est, algo):
    """The estimator's binding: the GPU binding on a GPU install, the host
    binding on a CPU-only one (`_bind`). cpu-gpu-cleanup c-linear (2026-10-02):
    the host route table (`_HOST_ALGOS`, MOJOLEARN_X_LINEAR_DEVICE,
    MOJOLEARN_X_LINEAR_ISOTONIC_HOST) that sent fits to the host binding on a
    GPU install is deleted."""
    return est._bind(_BINDING)


def _run(est, algo, X, n, d, y, ip, fp, n_out, n_fw, n_iw):
    """One `x_linear_fit` call; the flat float32 result as a Python list."""
    out = empty((n_out,), "<f4")
    yy = y if y is not None else zeros((1,), "<f4")
    ip = [int(v) for v in ip]
    fp = [float(v) for v in fp]
    _fit_module(est, algo).x_linear_fit(
        int(algo), addr_ro(X, name="X"), addr_ro(yy, name="y"),
        [n, d, X.size, 0 if y is None else yy.size, n_out, max(n_fw, 1), max(n_iw, 1), len(ip), len(fp)],
        ip, fp, addr(out, name="out"))
    return out.tolist()


def _decision(est, X, coef_rows, intercepts, link=LINK_IDENTITY):
    """link(X @ W^T + b) through the binding, as an (n, k) float32 Array."""
    a, n, d = _matrix(X)
    if d != est.n_features_in_:
        raise ValueError(
            f"mojolearn {type(est).__name__}: X has {d} features, the model was fitted with {est.n_features_in_}")
    k = len(coef_rows)
    flat = []
    for row, b in zip(coef_rows, intercepts):
        flat.extend(row)
        flat.append(b)
    wb = Array.from_list(flat, "<f4")
    out = empty((n, k), "<f4")
    est._bind(_BINDING).x_linear_decision(
        addr_ro(a, name="X"), addr_ro(wb, name="coef"), [n, d, k, link], addr(out, name="out"))
    return out


def _weights_f32(sample_weight, n):
    """The checked float32 sample weights: finite-or-infinite non-negative
    with a positive sum (scikit-learn's _check_sample_weight), tested by the
    base binding's reductions (a NaN fails the sum test), no row loop here."""
    if isinstance(sample_weight, (int, float)):
        w = full((n,), float(sample_weight), "<f4")
    else:
        w = _vector(sample_weight, n, "sample_weight")
    if n == 0 or not (w.min() >= 0) or not (w.sum() > 0):
        raise ValueError("mojolearn: sample_weight must be non-negative with a positive sum")
    return w


def _concat_f32(a, b):
    """a's then b's float32 elements as one flat float32 Array (two byte
    copies)."""
    out = empty((a.size + b.size,), "<f4")
    o = addr(out, name="y")
    memcopy(o, addr_ro(a, name="y"), 4 * a.size)
    memcopy(o + 4 * a.size, addr_ro(b, name="y"), 4 * b.size)
    return out


def _with_weights(yv, sample_weight, n):
    """(targets | weights, 1) with a sample_weight, (targets, 0) without:
    the kernels take the weights as the second half of y."""
    if sample_weight is None:
        return yv, 0
    return _concat_f32(yv, _weights_f32(sample_weight, n)), 1


#: lane/apple-fast-py2mojo-linear: `py2mojo_rows` modes (core/py2mojo_rows.mojo)
_ROWS_SGD_PROBA, _ROWS_LRCV_PROBA, _ROWS_HUBER_OUT = 2, 3, 4
_PY2MOJO_ROWS = 2


def _py2mojo_proba(est, mode, scores, k):
    """predict_proba's per-row glue in the binding (the same words), or None
    when the binary predates it or was built with
    -D MOJOLEARN_PY2MOJO_linear_OFF (the old Python rows)."""
    b = est._bind(_BINDING)
    fn = getattr(b, "py2mojo_linear_flags", None)
    if fn is None or not (int(fn()) & _PY2MOJO_ROWS) or scores.dtype != "<f4":
        return None
    n = scores.shape[0]
    out = empty((n, 2 if k == 1 else k), "<f4")
    if n:
        b.py2mojo_rows(mode, addr_ro(scores, name="scores"), addr(out, name="proba"), [n, k])
    return out


_CLS1_RIDGE_CODES = 1


def _cls1_flags(est):
    """lane/apple-fast-gap-cls1: the bound binary's build-time switches
    (`x_linear_cls1_flags`; 0 on IDENTICAL, the host binding or a build
    without them, which is main's path)."""
    try:
        fn = getattr(est._bind(_BINDING), "x_linear_cls1_flags", None)
        return int(fn()) if fn is not None else 0
    except Exception:  # noqa: BLE001  (a binary without the entry: main's path)
        return 0


def _rows(values, k, d):
    return [values[c * d:(c + 1) * d] for c in range(k)]


def _check_fitted(est):
    if not hasattr(est, "coef_"):
        raise RuntimeError(f"mojolearn {type(est).__name__}: call fit first")


def _seed(random_state):
    """An int seed (None -> 0: a fit is reproducible by default)."""
    if random_state is None:
        return 0, 0
    if not isinstance(random_state, int) or isinstance(random_state, bool) or random_state < 0:
        raise ValueError("mojolearn: random_state must be None or a non-negative int")
    s = random_state & ((1 << 64) - 1)
    lo, hi = s & 0xFFFFFFFF, s >> 32
    # Int32 transport, two's complement
    return (lo - (1 << 32) if lo >= 1 << 31 else lo), (hi - (1 << 32) if hi >= 1 << 31 else hi)


def _classes(est, y, n):
    classes, codes = encode_labels(y)
    if len(codes) != n:
        raise ValueError(f"mojolearn {type(est).__name__}: X and y lengths differ")
    if len(classes) < 2:
        raise ValueError(f"mojolearn {type(est).__name__}: y has one class")
    return classes, codes.astype("<f4")


class _LinearClassifierMixin:
    _estimator_type = "classifier"

    def decision_function(self, X):
        _check_fitted(self)
        k = len(self.intercept_)
        coef = self.coef_.tolist()
        out = _decision(self, X, coef, self.intercept_.tolist())
        if k == 1:
            return out.reshape((out.shape[0],))
        return out

    def predict(self, X):
        scores = self.decision_function(X)
        codes = threshold_codes(scores) if scores.ndim == 1 else argmax_rows(scores)
        return decode_labels(self.classes_, codes)

    def score(self, X, y):
        from ._expansion_metrics import accuracy_fraction
        return accuracy_fraction(y, self.predict(X))


class _LinearRegressorMixin:
    _estimator_type = "regressor"
    _link = LINK_IDENTITY

    def predict(self, X):
        _check_fitted(self)
        out = _decision(self, X, [self.coef_.tolist()], [float(self.intercept_)
                                                         if not hasattr(self.intercept_, "tolist")
                                                         else self.intercept_.tolist()[0]], self._link)
        return out.reshape((out.shape[0],))

    def score(self, X, y):
        # R^2 on the device (the metrics binding's pinned-sum r2_score)
        from ._metrics_impl import r2_score
        pred, _ = as_f32_c(self.predict(X), ndim=1, name="prediction")
        truth, _ = as_f32_c(y, ndim=1, name="y")
        return r2_score(truth, pred)


# ---------------------------------------------------------------------- SGD
# Reference: scikit-learn sklearn/linear_model/_stochastic_gradient.py and
# _sgd_fast.pyx.tp (`_plain_sgd`); the Mojo side is x_linear/sgd.mojo.

_SGD_CLF_LOSS = {"hinge": 0, "log_loss": 1, "modified_huber": 2, "squared_hinge": 3, "perceptron": 4}
_SGD_REG_LOSS = {"squared_error": 10, "huber": 11, "epsilon_insensitive": 12,
                 "squared_epsilon_insensitive": 13}
_SGD_PENALTY = {None: 0, "l2": 1, "l1": 2, "elasticnet": 3}
_SGD_LR = {"constant": 0, "optimal": 1, "invscaling": 2, "adaptive": 3, "pa1": 5, "pa2": 6}


def _sgd_refuse(est, early_stopping, average, class_weight=None, warm_start=False):
    name = type(est).__name__
    if early_stopping:
        raise ValueError(f"mojolearn {name}: early_stopping is not implemented (x_linear/NOT_IMPLEMENTED.tsv)")
    if average:
        raise ValueError(f"mojolearn {name}: average is not implemented (x_linear/NOT_IMPLEMENTED.tsv)")
    if warm_start:
        raise ValueError(f"mojolearn {name}: warm_start is not implemented (x_linear/NOT_IMPLEMENTED.tsv)")


def _expanded_class_weight(class_weight, classes, codes, sample_weight=None):
    """scikit-learn's compute_class_weight: 'balanced' is total / (k * count)
    over the (weighted, when sample_weight is given) class counts, a dict
    maps labels to weights (1 for a label it does not name)."""
    labels = classes.tolist() if hasattr(classes, "tolist") else list(classes)
    k = len(labels)
    if class_weight == "balanced":
        counts = [0.0] * k
        ws = [1.0] * len(codes) if sample_weight is None else sample_weight
        for c, w in zip(codes, ws):
            counts[int(c)] += w
        total = sum(counts)
        return [total / (k * cnt) if cnt else 0.0 for cnt in counts]
    if isinstance(class_weight, dict):
        for key in class_weight:
            if key not in labels:
                raise ValueError(f"The classes, {[key]}, are not in class_weight")
        return [float(class_weight.get(lab, 1.0)) for lab in labels]
    raise ValueError("mojolearn: class_weight must be None, 'balanced' or a dict")


def _sgd_fit(est, X, y, n_classes, loss_code, penalty, lr, alpha, l1_ratio, eta0, power_t,
             epsilon, fit_intercept, max_iter, tol, n_iter_no_change, shuffle, random_state,
             sample_weight=None, class_weight=None, classes=None, codes=None, batch_size=0, batch_sum=False):
    a, n, d = X
    if penalty not in _SGD_PENALTY:
        raise ValueError(f"mojolearn {type(est).__name__}: penalty must be 'l2', 'l1', 'elasticnet' or None")
    if lr not in _SGD_LR:
        raise ValueError(f"mojolearn {type(est).__name__}: unknown learning_rate {lr!r}")
    if lr in ("constant", "invscaling", "adaptive") and not eta0 > 0:
        raise ValueError(f"mojolearn {type(est).__name__}: eta0 must be > 0")
    if lr == "optimal" and not alpha > 0:
        raise ValueError(f"mojolearn {type(est).__name__}: alpha must be > 0 for learning_rate='optimal'")
    lo, hi = _seed(random_state)
    problems = n_classes if n_classes > 2 else 1
    ip = [n_classes, loss_code, _SGD_PENALTY[penalty], _SGD_LR[lr], int(bool(fit_intercept)),
          int(max_iter), int(n_iter_no_change), int(bool(shuffle)), lo, hi]
    fp = [alpha, l1_ratio, eta0, power_t, epsilon, -3.0e38 if tol is None else tol]
    if y is None:
        y = zeros((n,), "<f4")
    y, has_sw = _with_weights(y, sample_weight, n)
    has_cw = 0
    if class_weight is not None and n_classes >= 2:
        cw = _expanded_class_weight(class_weight, classes, codes)
        if n_classes == 2:
            pos, neg = [cw[1]], [cw[0]]
        else:
            pos, neg = list(cw), [1.0] * n_classes
        fp += pos + neg
        has_cw = 1
    # lane/neural-pass103 (Andrew, 2026-10-01): SGDClassifier / SGDRegressor
    # train minibatch SGD (cuML MBSGD's form, a fixed in-batch combine order,
    # x_linear/sgd.mojo `sgd_mb_one`); since lane/neural-pass132 Perceptron,
    # the passive-aggressive pair too (and SGDOneClassSVM when asked), at batch_size=256
    # (taxi / istella 200k: Perceptron accuracy 0.72 / 0.904 at 256 vs 0.36 /
    # 0.879 at 4096 (sklearn 0.60 / 0.902); the one-class objective 0.5001 /
    # 0.5010 vs 0.5008 / 0.5062 (sklearn 0.5000 / 0.5001));
    # batch 0 is the per-sample fit (batch_size=0)
    if batch_size and (not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1):
        raise ValueError(f"mojolearn {type(est).__name__}: batch_size must be a positive int")
    # batch_sum (lane/neural-pass132): Perceptron's batch step is the SUM of
    # its rows' updates (the per-sample fit's scale, so `tol` reads the same
    # loss scale); MOJOLEARN_SGD_BATCH_SUM=0/1 overrides for an A/B
    if os.environ.get("MOJOLEARN_SGD_BATCH_SUM", "") in ("0", "1"):
        batch_sum = os.environ["MOJOLEARN_SGD_BATCH_SUM"] == "1"
    ip += [has_sw, has_cw, int(batch_size), int(bool(batch_sum))]
    vals = _run(est, ALGO_SGD, a, n, d, y, ip, fp, problems * d + problems + 2, problems * (n + d + 1), problems * n)
    if vals[-1] != 0:
        raise ValueError("Floating-point under-/overflow occurred. Scaling input data with "
                         "StandardScaler or MinMaxScaler might help.")
    coef = Array.from_list(_rows(vals, problems, d), "<f4")
    intercept = Array.from_list(vals[problems * d:problems * d + problems], "<f4")
    est.n_iter_ = int(vals[-2])
    est.t_ = float(est.n_iter_ * n + 1)
    est.n_features_in_ = d
    return coef, intercept


class SGDClassifier(_LinearClassifierMixin, NumericModeMixin):
    """Linear classifiers trained by plain SGD (scikit-learn's SGDClassifier,
    one-vs-rest for more than two classes). Reference:
    sklearn/linear_model/_stochastic_gradient.py; kernel x_linear/sgd.mojo."""

    _BINDING = _BINDING

    def __init__(self, loss="hinge", *, penalty="l2", alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True,
                 epsilon=0.1, random_state=None, learning_rate="optimal", eta0=0.0,
                 power_t=0.5, early_stopping=False, n_iter_no_change=5,
                 class_weight=None, average=False, warm_start=False, batch_size=4096):
        self.loss, self.penalty, self.alpha, self.l1_ratio = loss, penalty, alpha, l1_ratio
        self.fit_intercept, self.max_iter, self.tol, self.shuffle = fit_intercept, max_iter, tol, shuffle
        self.epsilon, self.random_state, self.learning_rate = epsilon, random_state, learning_rate
        self.eta0, self.power_t, self.early_stopping = eta0, power_t, early_stopping
        self.n_iter_no_change, self.class_weight, self.average = n_iter_no_change, class_weight, average
        self.warm_start = warm_start
        self.batch_size = batch_size

    def fit(self, X, y, sample_weight=None):
        _sgd_refuse(self, self.early_stopping, self.average, None, self.warm_start)
        if self.loss not in _SGD_CLF_LOSS:
            raise ValueError(f"mojolearn SGDClassifier: loss must be one of {sorted(_SGD_CLF_LOSS)}")
        Xm = _matrix(X)
        classes, codes = _classes(self, y, Xm[1])
        self.classes_ = classes
        self.coef_, self.intercept_ = _sgd_fit(
            self, Xm, codes, len(classes), _SGD_CLF_LOSS[self.loss], self.penalty, self.learning_rate,
            self.alpha, self.l1_ratio, self.eta0, self.power_t, self.epsilon, self.fit_intercept,
            self.max_iter, self.tol, self.n_iter_no_change, self.shuffle, self.random_state,
            sample_weight, self.class_weight, classes, codes.tolist(), batch_size=self.batch_size)
        return self

    def predict_proba(self, X):
        """log_loss: the sigmoid of each OvR score, normalized over classes
        when there are more than two (their `_predict_proba_lr`)."""
        if self.loss != "log_loss":
            raise AttributeError("probability estimates are not available for loss=%r" % self.loss)
        _check_fitted(self)
        p = _decision(self, X, self.coef_.tolist(), self.intercept_.tolist(), LINK_SIGMOID)
        out = _py2mojo_proba(self, _ROWS_SGD_PROBA, p, len(self.intercept_))
        if out is not None:
            return out
        rows = p.tolist()
        if len(self.intercept_) == 1:
            return Array.from_list([[1.0 - r[0], r[0]] for r in rows], "<f4")
        out = []
        for r in rows:
            s = sum(r)
            out.append([v / s for v in r] if s > 0 else [1.0 / len(r)] * len(r))
        return Array.from_list(out, "<f4")


class SGDRegressor(_LinearRegressorMixin, NumericModeMixin):
    """Linear regression trained by plain SGD (scikit-learn's SGDRegressor).
    Reference: sklearn/linear_model/_stochastic_gradient.py; kernel
    x_linear/sgd.mojo."""

    _BINDING = _BINDING

    def __init__(self, loss="squared_error", *, penalty="l2", alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, epsilon=0.1,
                 random_state=None, learning_rate="invscaling", eta0=0.01, power_t=0.25,
                 early_stopping=False, n_iter_no_change=5, average=False, warm_start=False, batch_size=4096):
        self.loss, self.penalty, self.alpha, self.l1_ratio = loss, penalty, alpha, l1_ratio
        self.fit_intercept, self.max_iter, self.tol, self.shuffle = fit_intercept, max_iter, tol, shuffle
        self.epsilon, self.random_state, self.learning_rate = epsilon, random_state, learning_rate
        self.eta0, self.power_t, self.early_stopping = eta0, power_t, early_stopping
        self.n_iter_no_change, self.average, self.warm_start = n_iter_no_change, average, warm_start
        self.batch_size = batch_size

    def fit(self, X, y, sample_weight=None):
        _sgd_refuse(self, self.early_stopping, self.average, None, self.warm_start)
        if self.loss not in _SGD_REG_LOSS:
            raise ValueError(f"mojolearn SGDRegressor: loss must be one of {sorted(_SGD_REG_LOSS)}")
        Xm = _matrix(X)
        yv = _vector(y, Xm[1])
        coef, intercept = _sgd_fit(
            self, Xm, yv, 0, _SGD_REG_LOSS[self.loss], self.penalty, self.learning_rate,
            self.alpha, self.l1_ratio, self.eta0, self.power_t, self.epsilon, self.fit_intercept,
            self.max_iter, self.tol, self.n_iter_no_change, self.shuffle, self.random_state,
            sample_weight, batch_size=self.batch_size)
        self.coef_ = coef.reshape((Xm[2],))
        self.intercept_ = intercept
        return self


# ---------------------------------------------------------------------- GLMs
# Reference: scikit-learn sklearn/linear_model/_glm/glm.py; kernel
# x_linear/glm.mojo (Newton-Cholesky with an Armijo line search).

class _GLMBase(_LinearRegressorMixin, NumericModeMixin):
    _BINDING = _BINDING
    _power = 0.0

    def _check_y(self, y):
        return None

    def _link_code(self):
        return 1

    def fit(self, X, y, sample_weight=None):
        if self.solver not in ("lbfgs", "newton-cholesky"):
            raise ValueError(f"mojolearn {type(self).__name__}: solver must be 'lbfgs' or 'newton-cholesky'")
        if self.warm_start:
            raise ValueError(f"mojolearn {type(self).__name__}: warm_start is not implemented")
        if not self.alpha >= 0:
            raise ValueError(f"mojolearn {type(self).__name__}: alpha must be >= 0")
        a, n, d = _matrix(X)
        yv = _vector(y, n)
        self._check_y(yv.tolist())
        link = self._link_code()
        m = d + 1
        yv, has_sw = _with_weights(yv, sample_weight, n)
        vals = _run(self, ALGO_GLM, a, n, d, yv, [self.max_iter, int(bool(self.fit_intercept)), link, has_sw],
                    [self._power_value(), self.alpha, self.tol], d + 3, 3 * n + m * m + 3 * m, 1)
        self.coef_ = Array.from_list(vals[:d], "<f4")
        self.intercept_ = float(vals[d])
        self.n_iter_ = int(vals[d + 1])
        self.n_features_in_ = d
        self._link = LINK_EXP if link == 1 else LINK_IDENTITY
        return self

    def _power_value(self):
        return self._power


def _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose):
    self.alpha, self.fit_intercept, self.solver = alpha, fit_intercept, solver
    self.max_iter, self.tol, self.warm_start, self.verbose = max_iter, tol, warm_start, verbose


class PoissonRegressor(_GLMBase):
    """Poisson GLM with log link (scikit-learn's PoissonRegressor)."""
    _power = 1.0

    def __init__(self, *, alpha=1.0, fit_intercept=True, solver="lbfgs", max_iter=100, tol=1e-4,
                 warm_start=False, verbose=0):
        _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose)

    def _check_y(self, y):
        if min(y) < 0 or sum(y) <= 0:
            raise ValueError("Some value(s) of y are out of the valid range of the loss 'HalfPoissonLoss'.")


class GammaRegressor(_GLMBase):
    """Gamma GLM with log link (scikit-learn's GammaRegressor)."""
    _power = 2.0

    def __init__(self, *, alpha=1.0, fit_intercept=True, solver="lbfgs", max_iter=100, tol=1e-4,
                 warm_start=False, verbose=0):
        _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose)

    def _check_y(self, y):
        if min(y) <= 0:
            raise ValueError("Some value(s) of y are out of the valid range of the loss 'HalfGammaLoss'.")


class TweedieRegressor(_GLMBase):
    """Tweedie GLM (scikit-learn's TweedieRegressor). power 0 (normal),
    1 (Poisson), (1, 2) (compound Poisson-Gamma), 2 (Gamma), > 2; link
    'auto' is identity for power 0 and log otherwise. power < 0 and
    power in (0, 1) are refused (x_linear/NOT_IMPLEMENTED.tsv), as is the
    identity link with power > 0."""

    def __init__(self, *, power=0.0, alpha=1.0, fit_intercept=True, link="auto", solver="lbfgs",
                 max_iter=100, tol=1e-4, warm_start=False, verbose=0):
        self.power, self.link = power, link
        _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose)

    def _power_value(self):
        return float(self.power)

    def _link_code(self):
        p = float(self.power)
        if p < 0 or 0 < p < 1:
            raise ValueError(f"mojolearn TweedieRegressor: power={p} is not implemented (power 0 or >= 1)")
        link = self.link
        if link == "auto":
            return 0 if p == 0 else 1
        if link == "identity":
            if p != 0:
                raise ValueError("mojolearn TweedieRegressor: the identity link is implemented for power=0 only")
            return 0
        if link == "log":
            return 1
        raise ValueError("mojolearn TweedieRegressor: link must be 'auto', 'identity' or 'log'")

    def _check_y(self, y):
        p = float(self.power)
        if 1 <= p < 2 and (min(y) < 0 or sum(y) <= 0):
            raise ValueError("Some value(s) of y are out of the valid range of the loss 'HalfTweedieLoss'.")
        if p >= 2 and min(y) <= 0:
            raise ValueError("Some value(s) of y are out of the valid range of the loss 'HalfTweedieLoss'.")


# -------------------------------------------------------------------- Huber
# Reference: scikit-learn sklearn/linear_model/_huber.py; kernel
# x_linear/huber.mojo (L-BFGS, x_linear/lbfgs.mojo, sigma = exp(s)).

_LBFGS_M = 10


def _lbfgs_work(p):
    return 4 * p + 2 * _LBFGS_M * p + 2 * _LBFGS_M


class HuberRegressor(_LinearRegressorMixin, NumericModeMixin):
    """L2-regularized linear regression with the Huber loss and a jointly
    estimated scale (scikit-learn's HuberRegressor)."""

    _BINDING = _BINDING

    def __init__(self, *, epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False,
                 fit_intercept=True, tol=1e-05):
        self.epsilon, self.max_iter, self.alpha = epsilon, max_iter, alpha
        self.warm_start, self.fit_intercept, self.tol = warm_start, fit_intercept, tol

    def fit(self, X, y, sample_weight=None):
        if not self.epsilon >= 1.0:
            raise ValueError("mojolearn HuberRegressor: epsilon must be >= 1.0")
        if self.warm_start:
            raise ValueError("mojolearn HuberRegressor: warm_start is not implemented")
        a, n, d = _matrix(X)
        yv = _vector(y, n)
        p = d + 2 if self.fit_intercept else d + 1
        yw, has_sw = _with_weights(yv, sample_weight, n)
        vals = _run(self, ALGO_HUBER, a, n, d, yw, [self.max_iter, int(bool(self.fit_intercept)), has_sw],
                    [self.epsilon, self.alpha, self.tol], d + 4 + p, _lbfgs_work(p) + n, 1)
        self.coef_ = Array.from_list(vals[:d], "<f4")
        self.intercept_ = float(vals[d])
        self.scale_ = float(vals[d + 1])
        self.n_iter_ = int(vals[d + 2])
        self.n_features_in_ = d
        pred_a = self.predict(a)
        thr = self.scale_ * self.epsilon
        b = self._bind(_BINDING)
        fn = getattr(b, "py2mojo_linear_flags", None)
        if (fn is not None and int(fn()) & _PY2MOJO_ROWS and yv.dtype == "<f4"
                and pred_a.dtype == "<f4"):
            # lane/apple-fast-py2mojo-linear: |y - pred| > thr in the binding
            # (core/py2mojo_rows.mojo ROWS_HUBER_OUT, the same binary64 test)
            flags = empty((n,), "<u1")
            if n:
                b.py2mojo_rows(_ROWS_HUBER_OUT, addr_ro(yv, name="y"), addr(flags, name="outliers_"),
                               [n, 1, addr_ro(pred_a, name="pred"), float(thr)])
            self.outliers_ = list(map(bool, flags.tolist()))
        else:
            pred = pred_a.tolist()
            self.outliers_ = [abs(t - q) > thr for t, q in zip(yv.tolist(), pred)]
        return self


# ---------------------------------------------------------- Bayesian / ARD
# Reference: scikit-learn sklearn/linear_model/_bayes.py; kernel
# x_linear/bayes.mojo (Jacobi eigenpairs of the centered Gram; Cholesky sigma).

def _bayes_refuse(est, return_std=False):
    if est.compute_score:
        raise ValueError(f"mojolearn {type(est).__name__}: compute_score is not implemented")
    if return_std:
        raise ValueError(f"mojolearn {type(est).__name__}: predict(return_std=True) is not implemented")


class BayesianRidge(_LinearRegressorMixin, NumericModeMixin):
    """Bayesian ridge regression by evidence maximization (scikit-learn's BayesianRidge)."""

    _BINDING = _BINDING

    def __init__(self, *, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6,
                 lambda_2=1e-6, alpha_init=None, lambda_init=None, compute_score=False,
                 fit_intercept=True, copy_X=True, verbose=False):
        self.max_iter, self.tol, self.alpha_1, self.alpha_2 = max_iter, tol, alpha_1, alpha_2
        self.lambda_1, self.lambda_2, self.alpha_init, self.lambda_init = lambda_1, lambda_2, alpha_init, lambda_init
        self.compute_score, self.fit_intercept, self.copy_X, self.verbose = compute_score, fit_intercept, copy_X, verbose

    def fit(self, X, y, sample_weight=None):
        _bayes_refuse(self)
        a, n, d = _matrix(X)
        yv = _vector(y, n)
        yv, has_sw = _with_weights(yv, sample_weight, n)
        vals = _run(self, ALGO_BAYES, a, n, d, yv, [self.max_iter, int(bool(self.fit_intercept)), has_sw],
                    [self.tol, self.alpha_1, self.alpha_2, self.lambda_1, self.lambda_2,
                     -1.0 if self.alpha_init is None else self.alpha_init,
                     -1.0 if self.lambda_init is None else self.lambda_init],
                    d + 4, 3 * d * d + 5 * d + n, 1)
        self.coef_ = Array.from_list(vals[:d], "<f4")
        self.intercept_ = float(vals[d])
        self.alpha_, self.lambda_ = float(vals[d + 1]), float(vals[d + 2])
        self.n_iter_ = int(vals[d + 3])
        self.n_features_in_ = d
        return self

    def predict(self, X, return_std=False):
        _bayes_refuse(self, return_std)
        return _LinearRegressorMixin.predict(self, X)


class ARDRegression(_LinearRegressorMixin, NumericModeMixin):
    """Automatic relevance determination regression (scikit-learn's ARDRegression)."""

    _BINDING = _BINDING

    def __init__(self, *, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6,
                 lambda_2=1e-6, compute_score=False, threshold_lambda=1e4, fit_intercept=True,
                 copy_X=True, verbose=False):
        self.max_iter, self.tol, self.alpha_1, self.alpha_2 = max_iter, tol, alpha_1, alpha_2
        self.lambda_1, self.lambda_2, self.compute_score = lambda_1, lambda_2, compute_score
        self.threshold_lambda, self.fit_intercept, self.copy_X, self.verbose = (
            threshold_lambda, fit_intercept, copy_X, verbose)

    def fit(self, X, y):
        _bayes_refuse(self)
        a, n, d = _matrix(X)
        if n < 2:
            raise ValueError("mojolearn ARDRegression: at least 2 samples are required")
        yv = _vector(y, n)
        vals = _run(self, ALGO_ARD, a, n, d, yv, [self.max_iter, int(bool(self.fit_intercept))],
                    [self.tol, self.alpha_1, self.alpha_2, self.lambda_1, self.lambda_2, self.threshold_lambda],
                    2 * d + 4, 3 * d * d + 4 * d + n, 2 * d)
        self.coef_ = Array.from_list(vals[:d], "<f4")
        self.intercept_ = float(vals[d])
        self.alpha_ = float(vals[d + 1])
        self.lambda_ = Array.from_list(vals[d + 2:2 * d + 2], "<f4")
        self.n_iter_ = int(vals[2 * d + 2])
        self.n_features_in_ = d
        return self

    def predict(self, X, return_std=False):
        _bayes_refuse(self, return_std)
        return _LinearRegressorMixin.predict(self, X)


# --------------------------------------------------------------------- LARS
# Reference: scikit-learn sklearn/linear_model/_least_angle.py; kernel
# x_linear/lars.mojo (the Gram form of _lars_path_solver).

def _lars_fit(est, X, y, max_iter, lasso, alpha_min):
    if getattr(est, "jitter", None) is not None:
        raise ValueError(f"mojolearn {type(est).__name__}: jitter is not implemented")
    a, n, d = _matrix(X)
    yv = _vector(y, n)
    vals = _run(est, ALGO_LARS, a, n, d, yv, [int(max_iter), int(bool(est.fit_intercept)), int(lasso),
                                              int(bool(getattr(est, "positive", False)))],
                [alpha_min], 2 * d + 4, 2 * d * d + 8 * d, 2 * d)
    est.coef_ = Array.from_list(vals[:d], "<f4")
    est.intercept_ = float(vals[d])
    est.n_iter_ = int(vals[d + 1])
    est.alpha_ = float(vals[d + 2])
    k = int(vals[d + 3])
    est.active_ = [int(v) for v in vals[d + 4:d + 4 + k]]
    est.n_features_in_ = d
    return est


class Lars(_LinearRegressorMixin, NumericModeMixin):
    """Least angle regression (scikit-learn's Lars)."""

    _BINDING = _BINDING

    def __init__(self, *, fit_intercept=True, verbose=False, precompute="auto", n_nonzero_coefs=500,
                 eps=2.220446049250313e-16, copy_X=True, fit_path=True, jitter=None, random_state=None):
        self.fit_intercept, self.verbose, self.precompute = fit_intercept, verbose, precompute
        self.n_nonzero_coefs, self.eps, self.copy_X = n_nonzero_coefs, eps, copy_X
        self.fit_path, self.jitter, self.random_state = fit_path, jitter, random_state

    def fit(self, X, y):
        return _lars_fit(self, X, y, self.n_nonzero_coefs, False, 0.0)


class LassoLars(_LinearRegressorMixin, NumericModeMixin):
    """Lasso fitted by least angle regression (scikit-learn's LassoLars)."""

    _BINDING = _BINDING

    def __init__(self, alpha=1.0, *, fit_intercept=True, verbose=False, precompute="auto", max_iter=500,
                 eps=2.220446049250313e-16, copy_X=True, fit_path=True, positive=False, jitter=None,
                 random_state=None):
        self.alpha, self.fit_intercept, self.verbose, self.precompute = alpha, fit_intercept, verbose, precompute
        self.max_iter, self.eps, self.copy_X, self.fit_path = max_iter, eps, copy_X, fit_path
        self.positive, self.jitter, self.random_state = positive, jitter, random_state

    def fit(self, X, y):
        if not self.alpha >= 0:
            raise ValueError("mojolearn LassoLars: alpha must be >= 0")
        return _lars_fit(self, X, y, self.max_iter, True, self.alpha)


# ----------------------------------------------------------------- Quantile
# Reference problem: scikit-learn sklearn/linear_model/_quantile.py; the
# solver is ADMM (x_linear/quantile.mojo), not their linear program.

class QuantileRegressor(_LinearRegressorMixin, NumericModeMixin):
    """L1-penalized quantile regression (scikit-learn's QuantileRegressor
    problem). `solver` accepts their names and always runs ADMM; `max_iter`
    and `tol` (ADMM's relative tolerance, eps_abs = tol / 100) are this
    implementation's own keywords."""

    _BINDING = _BINDING

    def __init__(self, *, quantile=0.5, alpha=1.0, fit_intercept=True, solver="highs",
                 solver_options=None, max_iter=5000, tol=1e-4):
        self.quantile, self.alpha, self.fit_intercept = quantile, alpha, fit_intercept
        self.solver, self.solver_options, self.max_iter, self.tol = solver, solver_options, max_iter, tol

    def fit(self, X, y, sample_weight=None):
        if not 0 < self.quantile < 1:
            raise ValueError("mojolearn QuantileRegressor: quantile must be strictly between 0 and 1")
        if not self.alpha >= 0:
            raise ValueError("mojolearn QuantileRegressor: alpha must be >= 0")
        a, n, d = _matrix(X)
        yv = _vector(y, n)
        m = d + 1
        yv, has_sw = _with_weights(yv, sample_weight, n)
        vals = _run(self, ALGO_QUANTILE, a, n, d, yv, [self.max_iter, int(bool(self.fit_intercept)), has_sw],
                    [self.quantile, self.alpha, self.tol / 100.0, self.tol,
                     0.0 if os.environ.get("MOJOLEARN_XQ_ABS_BALANCE", "") == "1" else 1.0], d + 3,
                    m * m + 3 * m + 4 * n + 3 * d, 1)
        self.coef_ = Array.from_list(vals[:d], "<f4")
        self.intercept_ = float(vals[d])
        self.n_iter_ = int(vals[d + 1])
        self.n_features_in_ = d
        return self


# --------------------------------------------------------------- Perceptron
# Reference: scikit-learn sklearn/linear_model/_perceptron.py: SGD with the
# perceptron loss, a constant rate eta0 and no penalty (x_linear/sgd.mojo).

class Perceptron(_LinearClassifierMixin, NumericModeMixin):
    """The perceptron (scikit-learn's Perceptron)."""

    _BINDING = _BINDING

    def __init__(self, *, penalty=None, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000,
                 tol=1e-3, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=0,
                 early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None,
                 warm_start=False, batch_size=256):
        self.batch_size = batch_size
        self.penalty, self.alpha, self.l1_ratio, self.fit_intercept = penalty, alpha, l1_ratio, fit_intercept
        self.max_iter, self.tol, self.shuffle, self.verbose, self.eta0 = max_iter, tol, shuffle, verbose, eta0
        self.n_jobs, self.random_state, self.early_stopping = n_jobs, random_state, early_stopping
        self.validation_fraction, self.n_iter_no_change = validation_fraction, n_iter_no_change
        self.class_weight, self.warm_start = class_weight, warm_start

    def fit(self, X, y, sample_weight=None):
        _sgd_refuse(self, self.early_stopping, False, None, self.warm_start)
        Xm = _matrix(X)
        classes, codes = _classes(self, y, Xm[1])
        self.classes_ = classes
        self.coef_, self.intercept_ = _sgd_fit(
            self, Xm, codes, len(classes), _SGD_CLF_LOSS["perceptron"], self.penalty, "constant",
            self.alpha, self.l1_ratio, self.eta0, 0.5, 0.1, self.fit_intercept,
            self.max_iter, self.tol, self.n_iter_no_change, self.shuffle, self.random_state,
            sample_weight, self.class_weight, classes, codes.tolist(), batch_size=self.batch_size,
            batch_sum=True)
        return self


# -------------------------------------------------------- Passive-aggressive
# Reference: scikit-learn sklearn/linear_model/_passive_aggressive.py: the
# SGD kernel with learning_rate pa1 (hinge / epsilon_insensitive) or pa2
# (squared_*), eta0 = C, no penalty (x_linear/sgd.mojo, `_plain_sgd`'s PA step).

class PassiveAggressiveClassifier(_LinearClassifierMixin, NumericModeMixin):
    """Passive-aggressive classifier, PA-I or PA-II (scikit-learn's)."""

    _BINDING = _BINDING

    def __init__(self, *, C=1.0, fit_intercept=True, max_iter=1000, tol=1e-3, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0, loss="hinge",
                 n_jobs=None, random_state=None, warm_start=False, class_weight=None, average=False,
                 batch_size=256):
        self.batch_size = batch_size
        self.C, self.fit_intercept, self.max_iter, self.tol = C, fit_intercept, max_iter, tol
        self.early_stopping, self.validation_fraction = early_stopping, validation_fraction
        self.n_iter_no_change, self.shuffle, self.verbose, self.loss = n_iter_no_change, shuffle, verbose, loss
        self.n_jobs, self.random_state, self.warm_start = n_jobs, random_state, warm_start
        self.class_weight, self.average = class_weight, average

    def fit(self, X, y, sample_weight=None):
        _sgd_refuse(self, self.early_stopping, self.average, None, self.warm_start)
        if self.loss not in ("hinge", "squared_hinge"):
            raise ValueError("mojolearn PassiveAggressiveClassifier: loss must be 'hinge' or 'squared_hinge'")
        if not self.C > 0:
            raise ValueError("mojolearn PassiveAggressiveClassifier: C must be > 0")
        Xm = _matrix(X)
        classes, codes = _classes(self, y, Xm[1])
        self.classes_ = classes
        lr = "pa1" if self.loss == "hinge" else "pa2"
        self.coef_, self.intercept_ = _sgd_fit(
            self, Xm, codes, len(classes), _SGD_CLF_LOSS["hinge"], None, lr,
            1.0, 0.0, self.C, 0.5, 0.1, self.fit_intercept,
            self.max_iter, self.tol, self.n_iter_no_change, self.shuffle, self.random_state,
            sample_weight, self.class_weight, classes, codes.tolist(), batch_size=self.batch_size)
        return self


class PassiveAggressiveRegressor(_LinearRegressorMixin, NumericModeMixin):
    """Passive-aggressive regressor, PA-I or PA-II (scikit-learn's)."""

    _BINDING = _BINDING

    def __init__(self, *, C=1.0, fit_intercept=True, max_iter=1000, tol=1e-3, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0,
                 loss="epsilon_insensitive", epsilon=0.1, random_state=None, warm_start=False,
                 average=False, batch_size=256):
        self.batch_size = batch_size
        self.C, self.fit_intercept, self.max_iter, self.tol = C, fit_intercept, max_iter, tol
        self.early_stopping, self.validation_fraction = early_stopping, validation_fraction
        self.n_iter_no_change, self.shuffle, self.verbose, self.loss = n_iter_no_change, shuffle, verbose, loss
        self.epsilon, self.random_state, self.warm_start, self.average = epsilon, random_state, warm_start, average

    def fit(self, X, y, sample_weight=None):
        _sgd_refuse(self, self.early_stopping, self.average, None, self.warm_start)
        if self.loss not in ("epsilon_insensitive", "squared_epsilon_insensitive"):
            raise ValueError("mojolearn PassiveAggressiveRegressor: loss must be 'epsilon_insensitive' "
                             "or 'squared_epsilon_insensitive'")
        if not self.C > 0:
            raise ValueError("mojolearn PassiveAggressiveRegressor: C must be > 0")
        Xm = _matrix(X)
        yv = _vector(y, Xm[1])
        lr = "pa1" if self.loss == "epsilon_insensitive" else "pa2"
        coef, intercept = _sgd_fit(
            self, Xm, yv, 0, _SGD_REG_LOSS["epsilon_insensitive"], None, lr,
            1.0, 0.0, self.C, 0.5, self.epsilon, self.fit_intercept,
            self.max_iter, self.tol, self.n_iter_no_change, self.shuffle, self.random_state,
            sample_weight, batch_size=self.batch_size)
        self.coef_ = coef.reshape((Xm[2],))
        self.intercept_ = intercept
        return self


# ---------------------------------------------------------- SGDOneClassSVM
# Reference: scikit-learn sklearn/linear_model/_stochastic_gradient.py
# (SGDOneClassSVM, `_fit_one_class`: y = 1, hinge loss, l2, alpha = nu,
# intercept = 1 - offset, the offset step `- eta * alpha`); x_linear/sgd.mojo.

class SGDOneClassSVM(NumericModeMixin):
    """Linear one-class SVM trained by SGD (scikit-learn's SGDOneClassSVM)."""

    _BINDING = _BINDING
    _estimator_type = "outlier_detector"

    def __init__(self, nu=0.5, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, verbose=0,
                 random_state=None, learning_rate="optimal", eta0=0.0, power_t=0.5, warm_start=False,
                 average=False, batch_size=0):
        # batch_size=0 (lane/neural-pass132): the per-sample fit, main's form.
        # The minibatch form oscillates the offset (with w near 0 every row
        # shares one score, so a batch moves all of them across the margin
        # together): board istella nu 0.1 flags 0.374 of the training rows at
        # batch 256, 0.024 at 64, sklearn 0.055. A positive batch_size opts in.
        self.batch_size = batch_size
        self.nu, self.fit_intercept, self.max_iter, self.tol = nu, fit_intercept, max_iter, tol
        self.shuffle, self.verbose, self.random_state = shuffle, verbose, random_state
        self.learning_rate, self.eta0, self.power_t = learning_rate, eta0, power_t
        self.warm_start, self.average = warm_start, average

    def fit(self, X, y=None, sample_weight=None):
        _sgd_refuse(self, False, self.average, None, self.warm_start)
        if not 0 < self.nu <= 1:
            raise ValueError("mojolearn SGDOneClassSVM: nu must be in (0, 1]")
        if self.learning_rate in ("pa1", "pa2"):
            raise ValueError("mojolearn SGDOneClassSVM: learning_rate must be constant, optimal, invscaling or adaptive")
        Xm = _matrix(X)
        coef, intercept = _sgd_fit(
            self, Xm, None, 1, _SGD_CLF_LOSS["hinge"], "l2", self.learning_rate,
            self.nu, 0.0, self.eta0, self.power_t, 0.1, self.fit_intercept,
            self.max_iter, self.tol, 5, self.shuffle, self.random_state, sample_weight,
            batch_size=self.batch_size)
        self.coef_ = coef.reshape((Xm[2],))
        # the binding returns offset_ itself for one class (lane/neural-pass139:
        # the per-sample fit carries the intercept near 1 as a float-float,
        # so 1 - float32(intercept) would drop every bit below 1.2e-7)
        self.offset_ = Array.from_list([intercept.tolist()[0]], "<f4")
        return self

    def decision_function(self, X):
        _check_fitted(self)
        out = _decision(self, X, [self.coef_.tolist()], [-self.offset_.tolist()[0]])
        return out.reshape((out.shape[0],))

    def score_samples(self, X):
        _check_fitted(self)
        out = _decision(self, X, [self.coef_.tolist()], [0.0])
        return out.reshape((out.shape[0],))

    def predict(self, X):
        return threshold_codes(self.decision_function(X), 0.0, strict=False, below=-1, above=1)


# ------------------------------------------------------------------- Ridge
# Reference: scikit-learn sklearn/linear_model/_ridge.py (`_solve_cholesky`,
# `_RidgeGCV`); kernel x_linear/ridge.mojo.

def _ridge_run(est, a, n, d, Y, T, alphas, sample_weight=None, codes_mode=False):
    A = len(alphas)
    has_sw = 0
    if sample_weight is not None:
        Y = _concat_f32(Y, _weights_f32(sample_weight, n))
        has_sw = 1
    ip = [T, int(bool(est.fit_intercept)), A, has_sw] + ([1] if codes_mode else [])
    vals = _run(est, ALGO_RIDGE, a, n, d, Y, ip, list(alphas),
                T * d + T + 2 + A + 1, 3 * d * d + 3 * d + T + d * T + 2 * n, 1)
    # lane/neural-pass93: status 2 = X'X + alpha I does not factor even in
    # float-float (x_linear/ridge.mojo); 1 never leaves a binding
    if vals[T * d + T + 2 + A] == 2.0:
        raise ValueError(f"mojolearn {type(est).__name__}: X'X + alpha I is singular even in float-float "
                         f"(alpha={vals[T * d + T]:g}); rescale X or use a larger alpha")
    return vals


def _ridge_refuse(est):
    name = type(est).__name__
    if getattr(est, "positive", False):
        raise ValueError(f"mojolearn {name}: positive=True is not implemented")
    if getattr(est, "solver", "auto") not in ("auto", "cholesky"):
        raise ValueError(f"mojolearn {name}: solver must be 'auto' or 'cholesky'")


class RidgeClassifier(_LinearClassifierMixin, NumericModeMixin):
    """Ridge regression on +-1 class targets (scikit-learn's RidgeClassifier)."""

    _BINDING = _BINDING

    def __init__(self, alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=1e-4,
                 class_weight=None, solver="auto", positive=False, random_state=None):
        self.alpha, self.fit_intercept, self.copy_X, self.max_iter = alpha, fit_intercept, copy_X, max_iter
        self.tol, self.class_weight, self.solver, self.positive = tol, class_weight, solver, positive
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        _ridge_refuse(self)
        if not self.alpha >= 0:
            raise ValueError("mojolearn RidgeClassifier: alpha must be >= 0")
        a, n, d = _matrix(X)
        if (sample_weight is None and self.class_weight is None
                and _cls1_flags(self) & _CLS1_RIDGE_CODES):
            # lane/apple-fast-gap-cls1 RIDGE_FAST_CLS1_CODES (FAST + Apple
            # default, off with -D MOJOLEARN_RIDGE_FAST_CLS1_CODES_OFF): the
            # int32 codes go to the binding as they are and the
            # +-1 targets are built on the device (x_linear/cls1_fast.mojo)
            classes, icodes = encode_labels(y)
            if icodes.size != n:
                raise ValueError("mojolearn RidgeClassifier: X and y lengths differ")
            if len(classes) < 2:
                raise ValueError("mojolearn RidgeClassifier: y has one class")
            k = len(classes)
            T = 1 if k == 2 else k
            vals = _ridge_run(self, a, n, d, icodes, T, [self.alpha], None, codes_mode=True)
            self.classes_ = classes
            self.coef_ = Array.from_list(_rows(vals, T, d), "<f4")
            self.intercept_ = Array.from_list(vals[T * d:T * d + T], "<f4")
            self.n_features_in_ = d
            return self
        classes, codes = _classes(self, y, n)
        k = len(classes)
        T = 1 if k == 2 else k
        cl = codes.tolist()
        if T == 1:
            Y = [1.0 if c == 1 else -1.0 for c in cl]
        else:
            Y = [1.0 if c == t else -1.0 for c in cl for t in range(T)]
        if self.class_weight is not None:
            # theirs: sample_weight * compute_sample_weight(class_weight, y)
            cw = _expanded_class_weight(self.class_weight, classes, cl)
            base = [1.0] * n if sample_weight is None else _vector(sample_weight, n, "sample_weight").tolist()
            sample_weight = [b * cw[int(c)] for b, c in zip(base, cl)]
        vals = _ridge_run(self, a, n, d, Array.from_list(Y, "<f4"), T, [self.alpha], sample_weight)
        self.classes_ = classes
        self.coef_ = Array.from_list(_rows(vals, T, d), "<f4")
        self.intercept_ = Array.from_list(vals[T * d:T * d + T], "<f4")
        self.n_features_in_ = d
        return self



class RidgeCV(_LinearRegressorMixin, NumericModeMixin):
    """Ridge with the alpha chosen by efficient leave-one-out (scikit-learn's
    RidgeCV with cv=None) or by k-fold cross-validation (cv=k: KFold(k),
    R^2, Ridge refit on every row). Other cv objects, scoring,
    alpha_per_target and a 2-D y are refused (x_linear/NOT_IMPLEMENTED.tsv)."""

    _BINDING = _BINDING

    def __init__(self, alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, scoring=None, cv=None,
                 gcv_mode=None, store_cv_results=False, alpha_per_target=False):
        self.alphas, self.fit_intercept, self.scoring, self.cv = alphas, fit_intercept, scoring, cv
        self.gcv_mode, self.store_cv_results, self.alpha_per_target = gcv_mode, store_cv_results, alpha_per_target

    def fit(self, X, y, sample_weight=None):
        if self.scoring is not None or self.alpha_per_target:
            raise ValueError("mojolearn RidgeCV: only scoring=None, alpha_per_target=False are implemented")
        alphas = [float(v) for v in (self.alphas if hasattr(self.alphas, "__len__") else [self.alphas])]
        if not alphas or any(not v > 0 for v in alphas):
            raise ValueError("mojolearn RidgeCV: alphas must be positive")
        a, n, d = _matrix(X)
        yv = _vector(y, n)
        if self.cv is not None:
            return self._fit_kfold(a, n, d, yv, alphas, sample_weight)
        vals = _ridge_run(self, a, n, d, yv, 1, alphas, sample_weight)
        self.coef_ = Array.from_list(vals[:d], "<f4")
        self.intercept_ = float(vals[d])
        self.alpha_ = float(vals[d + 1])
        self.best_score_ = float(vals[d + 2])
        if self.store_cv_results:
            self.cv_results_ = Array.from_list(vals[d + 3:d + 3 + len(alphas)], "<f4")
        self.n_features_in_ = d
        return self

    def _fit_kfold(self, a, n, d, yv, alphas, sample_weight):
        """cv = k (lane/neural-pass91): their GridSearchCV over the alphas with
        KFold(k) (no shuffle) and Ridge.score, then Ridge refit on every row
        with the first best alpha (x_linear/ridgecv.mojo)."""
        cv = self.cv
        if not isinstance(cv, int) or isinstance(cv, bool):
            raise ValueError("mojolearn RidgeCV: cv must be None or an int (KFold)")
        if not 2 <= cv <= n:
            raise ValueError("mojolearn RidgeCV: cv must be in [2, n_samples]")
        if sample_weight is not None:
            raise ValueError("mojolearn RidgeCV: sample_weight with cv is not implemented")
        if self.store_cv_results:
            raise ValueError("cv!=None and store_cv_results=True are incompatible")
        A = len(alphas)
        n_fw = 2 * d * d + 3 * d + 1 + A * (d + 2) + A * (n // cv + 1)
        scores = _run(self, ALGO_RIDGE_KFOLD, a, n, d, yv, [cv, int(bool(self.fit_intercept)), A], alphas, A, n_fw, 1)
        # an alpha whose system is singular in float32 scores NaN (x_linear/ridgecv.mojo)
        best = -1
        for i in range(A):
            if scores[i] == scores[i] and (best < 0 or scores[i] > scores[best]):
                best = i
        if best < 0:
            raise ValueError("mojolearn RidgeCV: X'X + alpha I is singular in float32 for every alpha "
                             "(rescale X, or use larger alphas)")
        vals = _ridge_run(self, a, n, d, yv, 1, [alphas[best]])
        self.coef_ = Array.from_list(vals[:d], "<f4")
        self.intercept_ = float(vals[d])
        self.alpha_ = alphas[best]
        self.best_score_ = float(scores[best])
        self.n_features_in_ = d
        return self


# ------------------------------------------------------ LassoCV / ElasticNetCV
# Reference: scikit-learn sklearn/linear_model/_coordinate_descent.py
# (LinearModelCV.fit, _alpha_grid, _path_residuals) and _cd_fast.pyx
# (enet_coordinate_descent_gram); kernel x_linear/cd.mojo.

def _kfold_ids_f32(n, k):
    """scikit-learn's KFold(n_splits=k, shuffle=False) fold of every row as
    float32: contiguous folds, the first n % k of them one row longer
    (the base binding's `fold_ids`, cast by `cast_elements`; lane
    cgr4-py-compute)."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 2 or k > n:
        raise ValueError("mojolearn: cv must be None or an int in [2, n_samples]")
    from ._buffer import _native
    ids = empty((n,), "<i4")
    counts = empty((k,), "<i8")
    _native("fold_ids")(0, n, 0, k, 0, addr(ids, name="folds"), addr(counts, name="folds"))
    out = empty((n,), "<f4")
    _native("cast_elements")(addr_ro(ids, name="folds"), 2, addr(out, name="folds"), 0, n)
    return out


def _enetcv_fit(est, X, y, l1_ratios):
    name = type(est).__name__
    if est.selection != "cyclic":
        raise ValueError(f"mojolearn {name}: selection='random' is not implemented")
    if any(not 0 < r <= 1 for r in l1_ratios):
        raise ValueError(f"mojolearn {name}: l1_ratio must be in (0, 1]")
    a, n, d = _matrix(X)
    yv = _vector(y, n)
    folds = 5 if est.cv is None else est.cv
    ids = _kfold_ids_f32(n, folds)
    alphas = est.alphas
    n_alphas = getattr(est, "n_alphas", None)
    if isinstance(alphas, int) and not isinstance(alphas, bool):
        explicit, grid = False, int(alphas)
    elif alphas is None or alphas in ("warn", "deprecated"):
        explicit, grid = False, int(n_alphas) if isinstance(n_alphas, int) else 100
    else:
        explicit = True
        values = sorted((float(v) for v in alphas), reverse=True)
        grid = len(values)
    if grid < 1:
        raise ValueError(f"mojolearn {name}: at least one alpha is required")
    yy = _concat_f32(yv, ids)
    L = len(l1_ratios)
    fp = [est.eps, est.tol] + [float(r) for r in l1_ratios] + (values if explicit else [])
    ip = [est.max_iter, int(bool(est.fit_intercept)), grid, folds, L, int(explicit), int(bool(est.positive))]
    vals = _run(est, ALGO_ENETCV, a, n, d, yy, ip, fp, d + 4 + L * grid + L * grid * folds,
                d * d + 4 * d + 3 + grid * (d + 2), 1)
    est.coef_ = Array.from_list(vals[:d], "<f4")
    est.intercept_ = float(vals[d])
    est.alpha_ = float(vals[d + 1])
    l1_best = float(vals[d + 2])
    est.n_iter_ = int(vals[d + 3])
    off = d + 4
    al = vals[off:off + L * grid]
    ms = vals[off + L * grid:off + L * grid + L * grid * folds]
    if L == 1:
        est.alphas_ = Array.from_list(al, "<f4")
        est.mse_path_ = Array.from_list([ms[k * folds:(k + 1) * folds] for k in range(grid)], "<f4")
    else:
        est.alphas_ = Array.from_list([al[l * grid:(l + 1) * grid] for l in range(L)], "<f4")
        est.mse_path_ = Array.from_list(
            [[ms[(l * grid + k) * folds:(l * grid + k + 1) * folds] for k in range(grid)] for l in range(L)], "<f4")
    est.n_features_in_ = d
    # the user's own value (the kernel carried it as float32)
    return min(l1_ratios, key=lambda r: abs(r - l1_best))


class LassoCV(_LinearRegressorMixin, NumericModeMixin):
    """Lasso with alpha chosen by K-fold cross-validation over a path
    (scikit-learn's LassoCV; cv None or an int, unshuffled KFold)."""

    _BINDING = _BINDING

    def __init__(self, *, eps=1e-3, n_alphas="deprecated", alphas=100, fit_intercept=True,
                 precompute="auto", max_iter=1000, tol=1e-4, copy_X=True, cv=None, verbose=False,
                 n_jobs=None, positive=False, random_state=None, selection="cyclic"):
        self.eps, self.n_alphas, self.alphas, self.fit_intercept = eps, n_alphas, alphas, fit_intercept
        self.precompute, self.max_iter, self.tol, self.copy_X = precompute, max_iter, tol, copy_X
        self.cv, self.verbose, self.n_jobs, self.positive = cv, verbose, n_jobs, positive
        self.random_state, self.selection = random_state, selection

    def fit(self, X, y):
        _enetcv_fit(self, X, y, [1.0])
        return self


class ElasticNetCV(_LinearRegressorMixin, NumericModeMixin):
    """Elastic net with (l1_ratio, alpha) chosen by K-fold cross-validation
    (scikit-learn's ElasticNetCV; cv None or an int, unshuffled KFold)."""

    _BINDING = _BINDING

    def __init__(self, *, l1_ratio=0.5, eps=1e-3, n_alphas="deprecated", alphas=100, fit_intercept=True,
                 precompute="auto", max_iter=1000, tol=1e-4, cv=None, copy_X=True, verbose=0,
                 n_jobs=None, positive=False, random_state=None, selection="cyclic"):
        self.l1_ratio, self.eps, self.n_alphas, self.alphas = l1_ratio, eps, n_alphas, alphas
        self.fit_intercept, self.precompute, self.max_iter, self.tol = fit_intercept, precompute, max_iter, tol
        self.cv, self.copy_X, self.verbose, self.n_jobs = cv, copy_X, verbose, n_jobs
        self.positive, self.random_state, self.selection = positive, random_state, selection

    def fit(self, X, y):
        ratios = list(self.l1_ratio) if hasattr(self.l1_ratio, "__len__") else [self.l1_ratio]
        self.l1_ratio_ = _enetcv_fit(self, X, y, [float(r) for r in ratios])
        return self


# ---------------------------------------------------- LogisticRegressionCV
# Reference: scikit-learn sklearn/linear_model/_logistic.py; kernel
# x_linear/logcv.mojo (L-BFGS on their LinearModelLoss objective).

def _check_stratified_folds(codes, k):
    """cv must be an int in [2, the largest class size] (StratifiedKFold's
    refusal). The fold ids themselves are built from the labels inside the
    binding (x_linear/logcv.mojo `lcv_fold_table`, on the grid on a GPU)."""
    largest = max(collections.Counter(codes).values())
    if not isinstance(k, int) or isinstance(k, bool) or k < 2 or k > largest:
        raise ValueError("mojolearn: cv must be None or an int in [2, the largest class size]")


class LogisticRegressionCV(_LinearClassifierMixin, NumericModeMixin):
    """L2 logistic regression with C chosen by stratified K-fold accuracy
    (scikit-learn's LogisticRegressionCV; binary or multinomial)."""

    _BINDING = _BINDING

    def __init__(self, *, Cs=10, fit_intercept=True, cv=None, dual=False, penalty="l2", scoring=None,
                 solver="lbfgs", tol=1e-4, max_iter=100, class_weight=None, n_jobs=None, verbose=0,
                 refit=True, intercept_scaling=1.0, random_state=None, l1_ratios=None):
        self.Cs, self.fit_intercept, self.cv, self.dual, self.penalty = Cs, fit_intercept, cv, dual, penalty
        self.scoring, self.solver, self.tol, self.max_iter = scoring, solver, tol, max_iter
        self.class_weight, self.n_jobs, self.verbose, self.refit = class_weight, n_jobs, verbose, refit
        self.intercept_scaling, self.random_state, self.l1_ratios = intercept_scaling, random_state, l1_ratios

    def fit(self, X, y, sample_weight=None):
        if self.penalty != "l2" or self.dual or self.l1_ratios is not None:
            raise ValueError("mojolearn LogisticRegressionCV: only penalty='l2' (primal) is implemented")
        if self.scoring is not None or not self.refit:
            raise ValueError("mojolearn LogisticRegressionCV: scoring and refit=False are not implemented")
        if self.solver not in ("lbfgs", "newton-cg", "newton-cholesky"):
            raise ValueError("mojolearn LogisticRegressionCV: solver must be lbfgs (newton-* run L-BFGS too)")
        a, n, d = _matrix(X)
        classes, codes = _classes(self, y, n)
        cl = [int(c) for c in codes.tolist()]
        K = len(classes)
        kp = 1 if K == 2 else K
        if isinstance(self.Cs, int) and not isinstance(self.Cs, bool):
            m = self.Cs
            # np.logspace(-4, 4, m); 10 ** y through `_pm.powr` (DEVIATION 6900), not the platform pow
            Cs = [_pm.powr(10.0, -4 + 8 * i / (m - 1)) for i in range(m)] if m > 1 else [1e-4]
        else:
            Cs = [float(c) for c in self.Cs]
        folds = 5 if self.cv is None else self.cv
        _check_stratified_folds(cl, folds)
        has_sw = 0
        tail = []
        if sample_weight is not None or self.class_weight is not None:
            raw = [1.0] * n if sample_weight is None else _with_weights(zeros((n,), "<f4"), sample_weight, n)[0].tolist()[n:]
            fitw = raw
            if self.class_weight is not None:
                # theirs: 'balanced' from the weighted counts of all of y
                cw = _expanded_class_weight(self.class_weight, classes, cl,
                                            None if sample_weight is None else raw)
                fitw = [b * cw[c] for b, c in zip(raw, cl)]
            tail, has_sw = fitw + raw, 1
        # the fold ids' slot is zeros: the binding builds them from the labels
        yy = Array.from_list([float(c) for c in cl] + [0.0] * n + tail, "<f4")
        p = kp * (d + 1)
        nc = len(Cs)
        vals = _run(self, ALGO_LOGCV, a, n, d, yy, [self.max_iter, int(bool(self.fit_intercept)), kp, nc, folds, has_sw],
                    [self.tol] + Cs, kp * d + kp + 2 + folds * nc, p + 1 + n * (kp + 1) + _lbfgs_work(p), 5)
        self.classes_ = classes
        self.coef_ = Array.from_list(_rows(vals, kp, d), "<f4")
        self.intercept_ = Array.from_list(vals[kp * d:kp * d + kp], "<f4")
        best_c = Cs[min(range(nc), key=lambda i: abs(Cs[i] - vals[kp * d + kp]))]
        self.Cs_ = Cs
        self.C_ = [best_c] * kp
        self.n_iter_ = int(vals[kp * d + kp + 1])
        off = kp * d + kp + 2
        grid = [vals[off + f * nc:off + (f + 1) * nc] for f in range(folds)]
        labels = classes.tolist() if hasattr(classes, "tolist") else list(classes)
        self.scores_ = {lab: Array.from_list(grid, "<f4") for lab in (labels[1:] if kp == 1 else labels)}
        self.n_features_in_ = d
        return self

    def predict_proba(self, X):
        # DEVIATION 6900: the pinned exp (was the platform exp) and the
        # CPython 3.12+ sum spelled out (`_pm.nsum`), the same bits on every host
        sc = self.decision_function(X)
        out = _py2mojo_proba(self, _ROWS_LRCV_PROBA, sc, len(self.intercept_))
        if out is not None:
            return out
        scores = sc.tolist()
        if len(self.intercept_) == 1:
            es = _pm.exp_array([-z if z >= 0 else z for z in scores])
            out = []
            for z, e in zip(scores, es):
                p = 1.0 / (1.0 + e) if z >= 0 else e / (1.0 + e)
                out.append([1.0 - p, p])
            return Array.from_list(out, "<f4")
        out = []
        for row in scores:
            m = max(row)
            e = _pm.exp_array([v - m for v in row])
            s = _pm.nsum(e)
            out.append([v / s for v in e])
        return Array.from_list(out, "<f4")


# --------------------------------------------------------------- Isotonic
# Reference: scikit-learn sklearn/isotonic.py and sklearn/_isotonic.pyx;
# kernel x_linear/isotonic.mojo (sequential PAVA; interp1d-linear predict).

def _column(X, name="X"):
    a, _ = as_f32_c(X, ndim=None, name=name)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a.reshape((a.shape[0],))
    if a.ndim != 1:
        raise ValueError(f"mojolearn IsotonicRegression: {name} must be 1-D or of shape (n, 1)")
    if a.shape[0] == 0:
        raise ValueError(f"mojolearn IsotonicRegression: {name} is empty")
    return a


class IsotonicRegression(NumericModeMixin):
    """Isotonic regression (scikit-learn's IsotonicRegression)."""

    _BINDING = _BINDING
    _estimator_type = "regressor"

    def __init__(self, *, y_min=None, y_max=None, increasing=True, out_of_bounds="nan"):
        self.y_min, self.y_max, self.increasing, self.out_of_bounds = y_min, y_max, increasing, out_of_bounds

    def fit(self, X, y, sample_weight=None):
        if self.out_of_bounds not in ("nan", "clip", "raise"):
            raise ValueError("mojolearn IsotonicRegression: out_of_bounds must be 'nan', 'clip' or 'raise'")
        # lane/neural-pass70 (2026-10-01): no lists and no Python sort; the
        # binding sorts the positive-weight rows by (x, y, row) itself
        xa = _column(X)
        n = xa.shape[0]
        yv = _vector(y, n)
        if self.increasing == "auto":
            self.increasing_ = _spearman_sign(xa.tolist(), yv.tolist()) >= 0
        else:
            self.increasing_ = bool(self.increasing)
        yy, has_w = _with_weights(yv, sample_weight, n)
        ip = [int(self.increasing_), int(self.y_min is not None), int(self.y_max is not None), int(has_w)]
        fp = [0.0 if self.y_min is None else self.y_min, 0.0 if self.y_max is None else self.y_max]
        vals = _run(self, ALGO_ISOTONIC, xa, n, 1, yy, ip, fp, 3 + 2 * n, 6 * n, 3 * n)
        k = int(vals[0])
        self.X_min_, self.X_max_ = float(vals[1]), float(vals[2])
        self.X_thresholds_ = Array.from_list(vals[3:3 + k], "<f4")
        self.y_thresholds_ = Array.from_list(vals[3 + n:3 + n + k], "<f4")
        self.n_features_in_ = 1
        return self

    def predict(self, T):
        return self.transform(T)

    def transform(self, T):
        if not hasattr(self, "X_thresholds_"):
            raise RuntimeError("mojolearn IsotonicRegression: call fit first")
        t = _column(T, "T")
        vals = t.tolist()
        if self.out_of_bounds == "raise" and any(v < self.X_min_ or v > self.X_max_ for v in vals):
            raise ValueError("A value in x_new is below/above the interpolation range.")
        n = len(vals)
        k = len(self.X_thresholds_)
        thr = Array.from_list(self.X_thresholds_.tolist() + self.y_thresholds_.tolist(), "<f4")
        q = t.reshape((n, 1))
        out = _run(self, ALGO_ISOTONIC_PREDICT, q, n, 1, thr, [k, 1 if self.out_of_bounds == "clip" else 0],
                   [self.X_min_, self.X_max_], n, 1, 1)
        return Array.from_list(out, "<f4")

    def fit_transform(self, X, y, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)


def _spearman_sign(x, y):
    """The sign of Spearman's rho (scikit-learn's check_increasing), with
    average ranks for ties, in float64 Python."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry))
