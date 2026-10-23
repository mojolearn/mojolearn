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
binding and on the device); every score is `x_linear_decision`. The public
methods are one call into the native owner (bindings/x_linear_native.mojo),
which validates, encodes labels and unpacks the flat float32 result; only
the cv-splitter fits (RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV)
still assemble their call here. NumPy-free (NUMPY_FREE_CONTRACT.md).
"""
from . import _portable_math as _pm
from . import _backend
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty, full, zeros
from ._bufcheck import memcopy
from ._labels import encode_labels
from ._mode import NumericModeMixin, available_if
from ._sample_weight import check_sample_weight
from ._mode import RegressorScoreMixin

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
# The native `score` of the regressors, connected once at import.
from . import _native_linear_score_bridge  # noqa: E402
_LINEAR_SCORE, _LINEAR_SCORE_CONTEXT = _native_linear_score_bridge.connect()
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
    return _run_array(est, algo, X, n, d, y, ip, fp, n_out, n_fw, n_iw).tolist()


def _run_array(est, algo, X, n, d, y, ip, fp, n_out, n_fw, n_iw):
    """One `x_linear_fit` call; the flat float32 result as an Array."""
    out = empty((n_out,), "<f4")
    yy = y if y is not None else zeros((1,), "<f4")
    ip = [int(v) for v in ip]  # glue: converts the int parameter list
    fp = [float(v) for v in fp]  # glue: converts the float parameter list
    _fit_module(est, algo).x_linear_fit(
        int(algo), addr_ro(X, name="X"), addr_ro(yy, name="y"),
        [n, d, X.size, 0 if y is None else yy.size, n_out, max(n_fw, 1), max(n_iw, 1), len(ip), len(fp)],
        ip, fp, addr(out, name="out"))
    return out


def _coef_block(coef, intercept):
    """[coef row | intercept] per class as one float32 Array (k rows of d + 1):
    byte copies of the fitted Arrays."""
    c = as_f32_c(coef, ndim=None, name="coef_")[0]
    b = as_f32_c(intercept, ndim=None, name="intercept_")[0]
    k = b.size
    d = c.size // max(k, 1)
    out = empty((k * (d + 1),), "<f4")
    o = addr(out, name="coef")
    ca, ba = addr_ro(c, name="coef_"), addr_ro(b, name="intercept_")
    for j in range(k):                  # glue: one row copy per class
        memcopy(o + 4 * j * (d + 1), ca + 4 * j * d, 4 * d)
        memcopy(o + 4 * (j * (d + 1) + d), ba + 4 * j, 4)
    return out


def _decision(est, X, coef, intercept, link=LINK_IDENTITY):
    """link(X @ W^T + b) through the binding, as an (n, k) float32 Array.
    `coef`: the float32 coef_ Array ((k, d) or (d,)); `intercept`: the
    intercept_ Array (k) or one Python number. The [w | b] block is byte
    copies (`_coef_block`; lane cpu2-l10-linear: was a Python list)."""
    a, n, d = _matrix(X)
    if d != est.n_features_in_:
        raise ValueError(
            f"mojolearn {type(est).__name__}: X has {d} features, the model was fitted with {est.n_features_in_}")
    if not hasattr(intercept, "shape"):
        intercept = full((1,), float(intercept), "<f4")
    wb = _coef_block(coef, intercept)
    k = wb.size // (d + 1)
    out = empty((n, k), "<f4")
    est._bind(_BINDING).x_linear_decision(
        addr_ro(a, name="X"), addr_ro(wb, name="coef"), [n, d, k, link], addr(out, name="out"))
    return out


def _weights_f32(sample_weight, n):
    """The checked float32 sample weights (scikit-learn's _check_sample_weight
    and its wording, `_sample_weight.check_sample_weight`): a scalar or a
    1-D vector of n finite non-negative entries, not all zero, tested by the
    base binding's reductions, no row loop here. A fresh copy, so no kernel
    can write into the caller's array."""
    return check_sample_weight(sample_weight, n, copy=True)


def _concat_f32(a, b):
    """a's then b's float32 elements as one flat float32 Array (two byte
    copies)."""
    out = empty((a.size + b.size,), "<f4")
    o = addr(out, name="y")
    memcopy(o, addr_ro(a, name="y"), 4 * a.size)
    memcopy(o + 4 * a.size, addr_ro(b, name="y"), 4 * b.size)
    return out


def _concat_f32_parts(parts):
    """The parts' float32 elements, in order, as one flat float32 Array (one
    byte copy per part)."""
    out = empty((sum(p.size for p in parts),), "<f4")  # glue: one size per part
    o = addr(out, name="y")
    at = 0
    for p in parts:                     # glue: one byte copy per part
        memcopy(o + 4 * at, addr_ro(p, name="y"), 4 * p.size)
        at += p.size
    return out


#: lane/apple-fast-py2mojo-linear: `py2mojo_rows` modes (core/py2mojo_rows.mojo)
_ROWS_SGD_PROBA, _ROWS_LRCV_PROBA, _ROWS_HUBER_OUT = 2, 3, 4


def _rows(values, k, d):
    return [values[c * d:(c + 1) * d] for c in range(k)]  # glue: per-class coefficient row views (k-sized: class count)


def _class_codes(est, y, n):
    """(classes, int32 codes): the labels encoded once at entry."""
    classes, codes = encode_labels(y)
    if len(codes) != n:
        raise ValueError(f"mojolearn {type(est).__name__}: X and y lengths differ")
    if len(classes) < 2:
        raise ValueError(f"mojolearn {type(est).__name__}: y has one class")
    return classes, codes


class _LinearClassifierMixin:
    _estimator_type = "classifier"

    def decision_function(self, X):
        """Signed distance of each sample to the separating hyperplane(s).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, converted to float32. The feature count must match the
            one seen in ``fit``.

        Returns
        -------
        scores : float32 array of shape (n_samples,) or (n_samples, n_classes)
            ``X @ coef_.T + intercept_``. One column (returned as a 1-D array)
            for a binary problem, one column per class otherwise.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If ``X`` has a different number of features than in ``fit``.
        """
        return _XL_CALL(self, "decision_function", (X,), _XL_CONTEXT)

    def predict(self, X):
        """Predict class labels.

        A binary model predicts ``classes_[1]`` where the decision score is
        positive and ``classes_[0]`` otherwise; a multiclass model predicts
        the class with the largest score (the first one on a tie).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, converted to float32.

        Returns
        -------
        labels : array of shape (n_samples,)
            Predicted labels, drawn from ``classes_``.
        """
        return _XL_CALL(self, "classifier_predict", (X,), _XL_CONTEXT)

    def score(self, X, y):
        """Mean accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Fraction of samples predicted correctly.
        """
        return _XL_CALL(self, "classifier_score", (X, y), _XL_CONTEXT)


class _LinearRegressorMixin:
    _estimator_type = "regressor"
    _link = LINK_IDENTITY

    def predict(self, X):
        """Predict targets for ``X``.

        Returns ``X @ coef_ + intercept_``; models with a log link (Poisson,
        Gamma and Tweedie with ``power > 0``) return its exponential.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, converted to float32. The feature count must match the
            one seen in ``fit``.

        Returns
        -------
        y_pred : float32 array of shape (n_samples,)
            Predicted values.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If ``X`` has a different number of features than in ``fit``.
        """
        return _XL_CALL(self, "regressor_predict", (X,), _XL_CONTEXT)

    def score(self, X, y):
        """Coefficient of determination R^2 of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True targets.

        Returns
        -------
        score : float
            R^2; 1.0 is a perfect fit.
        """
        # Predict and R^2 in one native call (bindings/linear_score_native.mojo):
        # the x_linear decision kernel, then the default metrics binding's r2_score.
        return _LINEAR_SCORE(self, "linear_regressor_score", (X, y), _LINEAR_SCORE_CONTEXT)


# ---------------------------------------------------------------------- SGD
# Reference: scikit-learn sklearn/linear_model/_stochastic_gradient.py and
# _sgd_fast.pyx.tp (`_plain_sgd`); the Mojo side is x_linear/sgd.mojo.

_SGD_CLF_LOSS = {"hinge": 0, "log_loss": 1, "modified_huber": 2, "squared_hinge": 3, "perceptron": 4}
_SGD_REG_LOSS = {"squared_error": 10, "huber": 11, "epsilon_insensitive": 12,
                 "squared_epsilon_insensitive": 13}
_SGD_PENALTY = {None: 0, "l2": 1, "l1": 2, "elasticnet": 3}
_SGD_LR = {"constant": 0, "optimal": 1, "invscaling": 2, "adaptive": 3, "pa1": 5, "pa2": 6}


def _class_prep(est, class_weight, classes, codes, sample_weight=None, weighted=False, rows=False):
    """scikit-learn's compute_class_weight and compute_sample_weight in the
    binding (`x_linear_class_prep`, x_linear/class_prep.mojo: on the device
    on a GPU install; lane cpu2-l10-linear removed the Python row loops).

    `codes`: the int32 class codes. 'balanced' is total / (k * count) over
    the class counts (weighted by `sample_weight` when `weighted`), computed
    in the binding; a dict maps labels to weights (1 for a label it does not
    name), None weighs every class 1. `sample_weight`: a checked float32
    Array (`_weights_f32`) or None. Returns (the k class weights as a
    float32 Array, the n row weights `sample_weight * cw[code]` as a float32
    Array when `rows` else None, the largest unweighted class count)."""
    labels = classes.tolist() if hasattr(classes, "tolist") else list(classes)
    k = len(labels)
    n = codes.shape[0]
    balanced = class_weight == "balanced"
    if balanced:
        cw = empty((k,), "<f4")
    elif class_weight is None:
        cw = full((k,), 1.0, "<f4")
    elif isinstance(class_weight, dict):
        for key in class_weight:  # glue: one entry per named label (label -> class map, G5)
            if key not in labels:
                raise ValueError(f"The classes, {[key]}, are not in class_weight")
        cw = Array.from_list([float(class_weight.get(lab, 1.0)) for lab in labels], "<f4")  # glue: label -> class weight
    else:
        raise ValueError("mojolearn: class_weight must be None, 'balanced' or a dict")
    out = empty((n,), "<f4") if rows else None
    largest = _fit_module(est, ALGO_LOGCV).x_linear_class_prep(
        addr_ro(codes, name="class codes"), 0 if sample_weight is None else addr_ro(sample_weight, name="sample_weight"),
        addr(cw, name="class weights"), [n, k, int(balanced), int(balanced and weighted and sample_weight is not None)],
        0 if out is None else addr(out, name="row weights"))
    return cw, out, int(largest)


#: lane sgd-quality (2026-10-17): the losses whose per-row gradient is bounded
#: (|dloss| <= 1, modified_huber <= 4), so a batch's summed per-sample steps
#: are the per-sample fit's steps taken at the batch's start and cannot
#: overshoot past what those steps would; squared_hinge's gradient grows with
#: the margin and keeps the mean step
_SGD_SUM_LOSSES = frozenset(("hinge", "log_loss", "modified_huber", "perceptron"))


# Reference: sklearn/linear_model/_stochastic_gradient.py; kernel x_linear/sgd.mojo.
class SGDClassifier(_LinearClassifierMixin, NumericModeMixin):
    """Linear classifier trained by stochastic gradient descent.

    Follows scikit-learn's SGDClassifier: a binary linear model, one-vs-rest
    for more than two classes. Two defaults differ from scikit-learn and
    change results: the fit runs minibatch SGD with ``batch_size='auto'``,
    which picks the batch from the row count (``n_samples // 256`` rows per
    step, at least 1, at most 4096; with the 'hinge', 'log_loss',
    'modified_huber' and 'perceptron' losses each step applies the sum of its
    rows' per-sample updates on scikit-learn's learning-rate schedule;
    ``batch_size=0`` gives scikit-learn's per-sample updates), and
    ``random_state=None`` means seed 0, so a fit is reproducible by default.

    Parameters
    ----------
    loss : {'hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'}, default 'hinge'
        Loss function.
    penalty : {'l2', 'l1', 'elasticnet'} or None, default 'l2'
        Regularization term.
    alpha : float, default 0.0001
        Regularization strength; must be > 0 with ``learning_rate='optimal'``.
    l1_ratio : float, default 0.15
        Elastic-net mixing (0 is L2, 1 is L1); used with ``penalty='elasticnet'``.
    fit_intercept : bool, default True
        Whether to fit an intercept.
    max_iter : int, default 1000
        Maximum number of passes over the data.
    tol : float or None, default 1e-3
        Stopping tolerance, as in scikit-learn; None disables it.
    shuffle : bool, default True
        Whether to shuffle the rows each pass.
    epsilon : float, default 0.1
        Epsilon of the ``'modified_huber'`` loss.
    random_state : int or None, default None
        Non-negative seed for shuffling. None means seed 0.
    learning_rate : {'constant', 'optimal', 'invscaling', 'adaptive'}, default 'optimal'
        Step-size schedule.
    eta0 : float, default 0.0
        Initial step size; must be > 0 for 'constant', 'invscaling' and 'adaptive'.
    power_t : float, default 0.5
        Exponent of the 'invscaling' schedule.
    early_stopping : bool, default False
        Not supported: True raises ValueError.
    n_iter_no_change : int, default 5
        Passes without improvement before stopping.
    class_weight : dict, 'balanced' or None, default None
        Per-class weights; 'balanced' weighs classes inversely to their frequency.
    average : bool, default False
        Not supported: True raises ValueError.
    warm_start : bool, default False
        Not supported: True raises ValueError.
    batch_size : 'auto' or int, default 'auto'
        Rows per minibatch step; 'auto' picks the batch from the row count
        (``n_samples // 256``, at least 1, at most 4096), 0 runs per-sample
        SGD. A positive batch size takes the mean of its rows' gradients each
        step.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels.
    coef_ : float32 array of shape (1, n_features) or (n_classes, n_features)
        Weights; one row for a binary problem.
    intercept_ : float32 array of shape (1,) or (n_classes,)
        Intercepts.
    n_iter_ : int
        Number of passes run.
    t_ : float
        Number of weight updates, ``n_iter_ * n_samples + 1``.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > 0.5).astype(np.int32)
    >>> model = mojolearn.SGDClassifier(loss="log_loss").fit(X, y)
    >>> proba = model.predict_proba(X)
    """

    _BINDING = _BINDING

    def __init__(self, loss="hinge", *, penalty="l2", alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True,
                 epsilon=0.1, random_state=None, learning_rate="optimal", eta0=0.0,
                 power_t=0.5, early_stopping=False, n_iter_no_change=5,
                 class_weight=None, average=False, warm_start=False, batch_size="auto"):
        self.loss, self.penalty, self.alpha, self.l1_ratio = loss, penalty, alpha, l1_ratio
        self.fit_intercept, self.max_iter, self.tol, self.shuffle = fit_intercept, max_iter, tol, shuffle
        self.epsilon, self.random_state, self.learning_rate = epsilon, random_state, learning_rate
        self.eta0, self.power_t, self.early_stopping = eta0, power_t, early_stopping
        self.n_iter_no_change, self.class_weight, self.average = n_iter_no_change, class_weight, average
        self.warm_start = warm_start
        self.batch_size = batch_size

    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Class labels; at least two classes.
        sample_weight : array-like of shape (n_samples,) or float, optional
            Non-negative weights with a positive sum.

        Returns
        -------
        self : SGDClassifier
            The fitted model.

        Raises
        ------
        ValueError
            On an unsupported option, a single class, invalid weights, or a
            floating-point overflow during training (scale the inputs).
        """
        return _XL_CALL(self, "fit_sgd_classifier", (X, y, sample_weight), _XL_CONTEXT)

    @available_if(lambda self: self.loss == "log_loss")
    def predict_proba(self, X):
        """Class probabilities, available with ``loss='log_loss'`` only.

        The sigmoid of each one-vs-rest score, normalized over the classes
        when there are more than two (as scikit-learn does).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, converted to float32.

        Returns
        -------
        proba : float32 array of shape (n_samples, n_classes)
            Probability of each class, in the order of ``classes_``.

        Raises
        ------
        AttributeError
            If ``loss`` is not ``'log_loss'``.
        """
        return _XL_CALL(self, "sgd_predict_proba", (X,), _XL_CONTEXT)


# Reference: sklearn/linear_model/_stochastic_gradient.py; kernel x_linear/sgd.mojo.
class SGDRegressor(_LinearRegressorMixin, NumericModeMixin):
    """Linear regression trained by stochastic gradient descent.

    Follows scikit-learn's SGDRegressor. Two defaults differ from
    scikit-learn and change results: the fit runs minibatch SGD with
    ``n_samples // 256`` rows per step by default, at most 4096
    (``batch_size=0`` gives scikit-learn's per-sample updates), and
    ``random_state=None`` means seed 0, so a fit is
    reproducible by default.

    Parameters
    ----------
    loss : {'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'}, default 'squared_error'
        Loss function.
    penalty : {'l2', 'l1', 'elasticnet'} or None, default 'l2'
        Regularization term.
    alpha : float, default 0.0001
        Regularization strength; must be > 0 with ``learning_rate='optimal'``.
    l1_ratio : float, default 0.15
        Elastic-net mixing (0 is L2, 1 is L1); used with ``penalty='elasticnet'``.
    fit_intercept : bool, default True
        Whether to fit an intercept.
    max_iter : int, default 1000
        Maximum number of passes over the data.
    tol : float or None, default 1e-3
        Stopping tolerance, as in scikit-learn; None disables it.
    shuffle : bool, default True
        Whether to shuffle the rows each pass.
    epsilon : float, default 0.1
        Epsilon of the huber and epsilon-insensitive losses.
    random_state : int or None, default None
        Non-negative seed for shuffling. None means seed 0.
    learning_rate : {'constant', 'optimal', 'invscaling', 'adaptive'}, default 'invscaling'
        Step-size schedule.
    eta0 : float, default 0.01
        Initial step size; must be > 0 for 'constant', 'invscaling' and 'adaptive'.
    power_t : float, default 0.25
        Exponent of the 'invscaling' schedule.
    early_stopping : bool, default False
        Not supported: True raises ValueError.
    n_iter_no_change : int, default 5
        Passes without improvement before stopping.
    average : bool, default False
        Not supported: True raises ValueError.
    warm_start : bool, default False
        Not supported: True raises ValueError.
    batch_size : 'auto' or int, default 'auto'
        Rows per minibatch step; 'auto' is ``n_samples // 256`` (at least 1,
        at most 4096), 0 runs per-sample SGD.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    coef_ : float32 array of shape (n_features,)
        Weights.
    intercept_ : float32 array of shape (1,)
        Intercept.
    n_iter_ : int
        Number of passes run.
    t_ : float
        Number of weight updates, ``n_iter_ * n_samples + 1``.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ np.arange(8, dtype=np.float32)
    >>> model = mojolearn.SGDRegressor().fit(X, y)
    >>> y_pred = model.predict(X)
    """

    _BINDING = _BINDING

    def __init__(self, loss="squared_error", *, penalty="l2", alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, epsilon=0.1,
                 random_state=None, learning_rate="invscaling", eta0=0.01, power_t=0.25,
                 early_stopping=False, n_iter_no_change=5, average=False, warm_start=False, batch_size="auto"):
        self.loss, self.penalty, self.alpha, self.l1_ratio = loss, penalty, alpha, l1_ratio
        self.fit_intercept, self.max_iter, self.tol, self.shuffle = fit_intercept, max_iter, tol, shuffle
        self.epsilon, self.random_state, self.learning_rate = epsilon, random_state, learning_rate
        self.eta0, self.power_t, self.early_stopping = eta0, power_t, early_stopping
        self.n_iter_no_change, self.average, self.warm_start = n_iter_no_change, average, warm_start
        self.batch_size = batch_size

    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Targets.
        sample_weight : array-like of shape (n_samples,) or float, optional
            Non-negative weights with a positive sum.

        Returns
        -------
        self : SGDRegressor
            The fitted model.

        Raises
        ------
        ValueError
            On an unsupported option, invalid weights, or a floating-point
            overflow during training (scale the inputs).
        """
        return _XL_CALL(self, "fit_sgd_regressor", (X, y, sample_weight), _XL_CONTEXT)


# ---------------------------------------------------------------------- GLMs
# Reference: scikit-learn sklearn/linear_model/_glm/glm.py; kernel
# x_linear/glm.mojo (Newton-Cholesky with an Armijo line search).

class _GLMBase(_LinearRegressorMixin, NumericModeMixin):
    _BINDING = _BINDING
    _power = 0.0

    #: the loss name in the out-of-range ValueError; None: every target is valid
    _loss_name = None


    def fit(self, X, y, sample_weight=None):
        """Fit the generalized linear model.

        Both accepted ``solver`` names run the same Newton-Cholesky solver
        with a line search.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Targets, inside the valid range of the model's distribution.
        sample_weight : array-like of shape (n_samples,) or float, optional
            Non-negative weights with a positive sum.

        Returns
        -------
        self : object
            The fitted model.

        Raises
        ------
        ValueError
            If ``solver`` is not 'lbfgs' or 'newton-cholesky', ``warm_start``
            is True, ``alpha`` is negative, or a value of ``y`` is outside
            the valid range of the loss.
        """
        return _XL_CALL(self, "fit_glm", (X, y, sample_weight), _XL_CONTEXT)


def _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose):
    self.alpha, self.fit_intercept, self.solver = alpha, fit_intercept, solver
    self.max_iter, self.tol, self.warm_start, self.verbose = max_iter, tol, warm_start, verbose


class PoissonRegressor(_GLMBase):
    """Generalized linear model with a Poisson distribution and a log link.

    Follows scikit-learn's PoissonRegressor; targets must be non-negative.
    ``predict`` returns ``exp(X @ coef_ + intercept_)``. Unlike scikit-learn,
    ``score`` returns R^2, not the deviance-based D^2.

    Parameters
    ----------
    alpha : float, default 1.0
        L2 regularization strength (>= 0).
    fit_intercept : bool, default True
        Whether to fit an intercept.
    solver : {'lbfgs', 'newton-cholesky'}, default 'lbfgs'
        Accepted for compatibility; both run a Newton-Cholesky solver.
    max_iter : int, default 100
        Maximum number of solver iterations.
    tol : float, default 1e-4
        Stopping tolerance.
    warm_start : bool, default False
        Not supported: True raises ValueError.
    verbose : int, default 0
        Accepted for compatibility; ignored.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    coef_ : float32 array of shape (n_features,)
        Weights.
    intercept_ : float
        Intercept.
    n_iter_ : int
        Number of solver iterations run.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.poisson(np.exp(X[:, 0])).astype(np.float32)
    >>> model = mojolearn.PoissonRegressor(alpha=0.01).fit(X, y)
    """
    _power = 1.0
    # y >= 0 (scikit-learn's positive_only target tag)
    _sklearn_target_tags = {"positive_only": True}

    def __init__(self, *, alpha=1.0, fit_intercept=True, solver="lbfgs", max_iter=100, tol=1e-4,
                 warm_start=False, verbose=0):
        _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose)

    _loss_name = "HalfPoissonLoss"


class GammaRegressor(_GLMBase):
    """Generalized linear model with a Gamma distribution and a log link.

    Follows scikit-learn's GammaRegressor; targets must be positive.
    ``predict`` returns ``exp(X @ coef_ + intercept_)``. Unlike scikit-learn,
    ``score`` returns R^2, not the deviance-based D^2.

    Parameters
    ----------
    alpha : float, default 1.0
        L2 regularization strength (>= 0).
    fit_intercept : bool, default True
        Whether to fit an intercept.
    solver : {'lbfgs', 'newton-cholesky'}, default 'lbfgs'
        Accepted for compatibility; both run a Newton-Cholesky solver.
    max_iter : int, default 100
        Maximum number of solver iterations.
    tol : float, default 1e-4
        Stopping tolerance.
    warm_start : bool, default False
        Not supported: True raises ValueError.
    verbose : int, default 0
        Accepted for compatibility; ignored.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    coef_ : float32 array of shape (n_features,)
        Weights.
    intercept_ : float
        Intercept.
    n_iter_ : int
        Number of solver iterations run.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.gamma(2.0, np.exp(X[:, 0])).astype(np.float32)
    >>> model = mojolearn.GammaRegressor(alpha=0.01).fit(X, y)
    """
    _power = 2.0
    # y > 0 (scikit-learn's positive_only target tag)
    _sklearn_target_tags = {"positive_only": True}

    def __init__(self, *, alpha=1.0, fit_intercept=True, solver="lbfgs", max_iter=100, tol=1e-4,
                 warm_start=False, verbose=0):
        _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose)

    _loss_name = "HalfGammaLoss"


# Refused powers and links are listed in x_linear/NOT_IMPLEMENTED.tsv.
class TweedieRegressor(_GLMBase):
    """Generalized linear model with a Tweedie distribution.

    Follows scikit-learn's TweedieRegressor. Supported powers are 0 (normal),
    1 (Poisson), between 1 and 2 (compound Poisson-Gamma), 2 (Gamma) and
    above 2. ``power < 0`` and
    ``0 < power < 1`` raise ValueError, as does the identity link with
    ``power > 0``. Unlike scikit-learn, ``score`` returns R^2, not the
    deviance-based D^2.

    Parameters
    ----------
    power : float, default 0.0
        Tweedie power; sets the distribution and the valid range of ``y``.
    alpha : float, default 1.0
        L2 regularization strength (>= 0).
    fit_intercept : bool, default True
        Whether to fit an intercept.
    link : {'auto', 'identity', 'log'}, default 'auto'
        Link function; 'auto' is identity for ``power=0`` and log otherwise.
    solver : {'lbfgs', 'newton-cholesky'}, default 'lbfgs'
        Accepted for compatibility; both run a Newton-Cholesky solver.
    max_iter : int, default 100
        Maximum number of solver iterations.
    tol : float, default 1e-4
        Stopping tolerance.
    warm_start : bool, default False
        Not supported: True raises ValueError.
    verbose : int, default 0
        Accepted for compatibility; ignored.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    coef_ : float32 array of shape (n_features,)
        Weights.
    intercept_ : float
        Intercept.
    n_iter_ : int
        Number of solver iterations run.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.poisson(np.exp(X[:, 0])).astype(np.float32)
    >>> model = mojolearn.TweedieRegressor(power=1.5, alpha=0.01).fit(X, y)
    """

    def __init__(self, *, power=0.0, alpha=1.0, fit_intercept=True, link="auto", solver="lbfgs",
                 max_iter=100, tol=1e-4, warm_start=False, verbose=0):
        self.power, self.link = power, link
        _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose)


    def _sklearn_instance_target_tags(self):
        # power > 0 needs y >= 0 (power >= 1) or y > 0 (power >= 2), as
        # scikit-learn's TweedieRegressor tags it; power 0 takes any y
        try:
            return {"positive_only": float(self.power) > 0}
        except (TypeError, ValueError):
            return {}


    _loss_name = "HalfTweedieLoss"


# -------------------------------------------------------------------- Huber
# Reference: scikit-learn sklearn/linear_model/_huber.py; kernel
# x_linear/huber.mojo (L-BFGS, x_linear/lbfgs.mojo, sigma = exp(s)).

_LBFGS_M = 10


def _lbfgs_work(p):
    return 4 * p + 2 * _LBFGS_M * p + 2 * _LBFGS_M


class HuberRegressor(_LinearRegressorMixin, NumericModeMixin):
    """Linear regression with the Huber loss, robust to outliers.

    Follows scikit-learn's HuberRegressor: an L2-regularized linear model
    whose scale is estimated jointly with the weights, solved by L-BFGS.

    Parameters
    ----------
    epsilon : float, default 1.35
        Residuals larger than ``epsilon * scale_`` count as outliers; must be >= 1.0.
    max_iter : int, default 100
        Maximum number of L-BFGS iterations.
    alpha : float, default 0.0001
        L2 regularization strength.
    warm_start : bool, default False
        Not supported: True raises ValueError.
    fit_intercept : bool, default True
        Whether to fit an intercept.
    tol : float, default 1e-05
        Stopping tolerance.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    coef_ : float32 array of shape (n_features,)
        Weights.
    intercept_ : float
        Intercept.
    scale_ : float
        Estimated scale of the residuals.
    n_iter_ : int
        Number of iterations run.
    outliers_ : list of bool
        For each training row, whether its absolute residual exceeds
        ``epsilon * scale_``.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ np.arange(8, dtype=np.float32)
    >>> model = mojolearn.HuberRegressor().fit(X, y)
    """

    _BINDING = _BINDING

    def __init__(self, *, epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False,
                 fit_intercept=True, tol=1e-05):
        self.epsilon, self.max_iter, self.alpha = epsilon, max_iter, alpha
        self.warm_start, self.fit_intercept, self.tol = warm_start, fit_intercept, tol

    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Targets.
        sample_weight : array-like of shape (n_samples,) or float, optional
            Non-negative weights with a positive sum.

        Returns
        -------
        self : HuberRegressor
            The fitted model.

        Raises
        ------
        ValueError
            If ``epsilon < 1``, ``warm_start`` is True or the weights are invalid.
        """
        return _XL_CALL(self, "fit_huber", (X, y, sample_weight), _XL_CONTEXT)


# ---------------------------------------------------------- Bayesian / ARD
# Reference: scikit-learn sklearn/linear_model/_bayes.py; kernel
# x_linear/bayes.mojo (Jacobi eigenpairs of the centered Gram; Cholesky sigma).


class BayesianRidge(_LinearRegressorMixin, NumericModeMixin):
    """Bayesian ridge regression.

    Follows scikit-learn's BayesianRidge: estimates the noise precision
    ``alpha_`` and the weight precision ``lambda_`` by maximizing the
    marginal likelihood (evidence). ``compute_score=True`` and
    ``predict(return_std=True)`` raise ValueError.

    Parameters
    ----------
    max_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 1e-3
        Stop when the weights change by less than this.
    alpha_1, alpha_2 : float, default 1e-6
        Shape and rate of the Gamma prior over ``alpha``.
    lambda_1, lambda_2 : float, default 1e-6
        Shape and rate of the Gamma prior over ``lambda``.
    alpha_init : float or None, default None
        Starting value of ``alpha``; None starts from the inverse variance of ``y``.
    lambda_init : float or None, default None
        Starting value of ``lambda``; None starts from 1.
    compute_score : bool, default False
        Not supported: True raises ValueError.
    fit_intercept : bool, default True
        Whether to fit an intercept.
    copy_X : bool, default True
        Accepted for compatibility; ``X`` is never modified.
    verbose : bool, default False
        Accepted for compatibility; ignored.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    coef_ : float32 array of shape (n_features,)
        Posterior mean of the weights.
    intercept_ : float
        Intercept.
    alpha_ : float
        Estimated noise precision.
    lambda_ : float
        Estimated weight precision.
    n_iter_ : int
        Number of iterations run.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ np.arange(8, dtype=np.float32)
    >>> model = mojolearn.BayesianRidge().fit(X, y)
    """

    _BINDING = _BINDING

    def __init__(self, *, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6,
                 lambda_2=1e-6, alpha_init=None, lambda_init=None, compute_score=False,
                 fit_intercept=True, copy_X=True, verbose=False):
        self.max_iter, self.tol, self.alpha_1, self.alpha_2 = max_iter, tol, alpha_1, alpha_2
        self.lambda_1, self.lambda_2, self.alpha_init, self.lambda_init = lambda_1, lambda_2, alpha_init, lambda_init
        self.compute_score, self.fit_intercept, self.copy_X, self.verbose = compute_score, fit_intercept, copy_X, verbose

    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Targets.
        sample_weight : array-like of shape (n_samples,) or float, optional
            Non-negative weights with a positive sum.

        Returns
        -------
        self : BayesianRidge
            The fitted model.

        Raises
        ------
        ValueError
            If ``compute_score`` is True or the weights are invalid.
        """
        return _XL_CALL(self, "fit_bayes", (X, y, sample_weight), _XL_CONTEXT)

    def predict(self, X, return_std=False):
        """Predict targets with the posterior mean of the weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, converted to float32.
        return_std : bool, default False
            Not supported: True raises ValueError.

        Returns
        -------
        y_pred : float32 array of shape (n_samples,)
            Predicted values.
        """
        return _XL_CALL(self, "regressor_predict", (X, return_std), _XL_CONTEXT)


class ARDRegression(_LinearRegressorMixin, NumericModeMixin):
    """Bayesian ARD (automatic relevance determination) regression.

    Follows scikit-learn's ARDRegression: one precision per weight,
    estimated by evidence maximization; features whose precision exceeds
    ``threshold_lambda`` are pruned (their weight is 0). Needs at least 2
    samples. ``compute_score=True`` and ``predict(return_std=True)`` raise
    ValueError; ``fit`` takes no ``sample_weight``.

    Parameters
    ----------
    max_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 1e-3
        Stop when the weights change by less than this.
    alpha_1, alpha_2 : float, default 1e-6
        Shape and rate of the Gamma prior over ``alpha``.
    lambda_1, lambda_2 : float, default 1e-6
        Shape and rate of the Gamma prior over each ``lambda``.
    compute_score : bool, default False
        Not supported: True raises ValueError.
    threshold_lambda : float, default 1e4
        Precision above which a weight is pruned.
    fit_intercept : bool, default True
        Whether to fit an intercept.
    copy_X : bool, default True
        Accepted for compatibility; ``X`` is never modified.
    verbose : bool, default False
        Accepted for compatibility; ignored.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    coef_ : float32 array of shape (n_features,)
        Posterior mean of the weights.
    intercept_ : float
        Intercept.
    alpha_ : float
        Estimated noise precision.
    lambda_ : float32 array of shape (n_features,)
        Estimated precision of each weight.
    n_iter_ : int
        Number of iterations run.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = 3 * X[:, 0] - 2 * X[:, 1]
    >>> model = mojolearn.ARDRegression().fit(X, y)
    """

    _BINDING = _BINDING

    def __init__(self, *, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6,
                 lambda_2=1e-6, compute_score=False, threshold_lambda=1e4, fit_intercept=True,
                 copy_X=True, verbose=False):
        self.max_iter, self.tol, self.alpha_1, self.alpha_2 = max_iter, tol, alpha_1, alpha_2
        self.lambda_1, self.lambda_2, self.compute_score = lambda_1, lambda_2, compute_score
        self.threshold_lambda, self.fit_intercept, self.copy_X, self.verbose = (
            threshold_lambda, fit_intercept, copy_X, verbose)

    def fit(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        self : ARDRegression
            The fitted model.

        Raises
        ------
        ValueError
            If ``compute_score`` is True or there are fewer than 2 samples.
        """
        return _XL_CALL(self, "fit_ard", (X, y), _XL_CONTEXT)

    def predict(self, X, return_std=False):
        """Predict targets with the posterior mean of the weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, converted to float32.
        return_std : bool, default False
            Not supported: True raises ValueError.

        Returns
        -------
        y_pred : float32 array of shape (n_samples,)
            Predicted values.
        """
        return _XL_CALL(self, "regressor_predict", (X, return_std), _XL_CONTEXT)


# --------------------------------------------------------------------- LARS
# Reference: scikit-learn sklearn/linear_model/_least_angle.py; kernel
# x_linear/lars.mojo (the Gram form of _lars_path_solver).


class Lars(_LinearRegressorMixin, NumericModeMixin):
    """Least angle regression (LARS).

    Follows scikit-learn's Lars: adds features to the active set one at a
    time, up to ``n_nonzero_coefs`` of them. ``jitter`` must be None
    (any other value raises ValueError).

    Parameters
    ----------
    fit_intercept : bool, default True
        Whether to fit an intercept.
    verbose : bool, default False
        Accepted for compatibility; ignored.
    precompute : bool, 'auto' or array, default 'auto'
        Accepted for compatibility; the fit always uses the Gram matrix.
    n_nonzero_coefs : int, default 500
        Maximum number of active features.
    eps : float, default 2.220446049250313e-16
        Accepted for compatibility; ignored.
    copy_X : bool, default True
        Accepted for compatibility; ``X`` is never modified.
    fit_path : bool, default True
        Accepted for compatibility; the path is not stored.
    jitter : None, default None
        Not supported: any value other than None raises ValueError.
    random_state : int or None, default None
        Accepted for compatibility; ignored.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    coef_ : float32 array of shape (n_features,)
        Weights at the end of the path.
    intercept_ : float
        Intercept.
    alpha_ : float
        Regularization value at the end of the path.
    active_ : list of int
        Indices of the active features.
    n_iter_ : int
        Number of steps run.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = 3 * X[:, 0] - 2 * X[:, 1]
    >>> model = mojolearn.Lars(n_nonzero_coefs=2).fit(X, y)
    """

    _BINDING = _BINDING

    def __init__(self, *, fit_intercept=True, verbose=False, precompute="auto", n_nonzero_coefs=500,
                 eps=2.220446049250313e-16, copy_X=True, fit_path=True, jitter=None, random_state=None):
        self.fit_intercept, self.verbose, self.precompute = fit_intercept, verbose, precompute
        self.n_nonzero_coefs, self.eps, self.copy_X = n_nonzero_coefs, eps, copy_X
        self.fit_path, self.jitter, self.random_state = fit_path, jitter, random_state

    def fit(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        self : Lars
            The fitted model.
        """
        return _XL_CALL(self, "fit_lars", (X, y), _XL_CONTEXT)


class LassoLars(_LinearRegressorMixin, NumericModeMixin):
    """Lasso fitted by least angle regression.

    Follows scikit-learn's LassoLars: the LARS path with the lasso
    modification, stopped at ``alpha``. ``jitter`` must be None (any other
    value raises ValueError).

    Parameters
    ----------
    alpha : float, default 1.0
        L1 regularization strength (>= 0); 0 runs the full path.
    fit_intercept : bool, default True
        Whether to fit an intercept.
    verbose : bool, default False
        Accepted for compatibility; ignored.
    precompute : bool, 'auto' or array, default 'auto'
        Accepted for compatibility; the fit always uses the Gram matrix.
    max_iter : int, default 500
        Maximum number of path steps.
    eps : float, default 2.220446049250313e-16
        Accepted for compatibility; ignored.
    copy_X : bool, default True
        Accepted for compatibility; ``X`` is never modified.
    fit_path : bool, default True
        Accepted for compatibility; the path is not stored.
    positive : bool, default False
        Restrict the weights to be non-negative.
    jitter : None, default None
        Not supported: any value other than None raises ValueError.
    random_state : int or None, default None
        Accepted for compatibility; ignored.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    coef_ : float32 array of shape (n_features,)
        Weights.
    intercept_ : float
        Intercept.
    alpha_ : float
        Regularization value where the path stopped.
    active_ : list of int
        Indices of the active features.
    n_iter_ : int
        Number of steps run.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = 3 * X[:, 0] - 2 * X[:, 1]
    >>> model = mojolearn.LassoLars(alpha=0.01).fit(X, y)
    """

    _BINDING = _BINDING

    def __init__(self, alpha=1.0, *, fit_intercept=True, verbose=False, precompute="auto", max_iter=500,
                 eps=2.220446049250313e-16, copy_X=True, fit_path=True, positive=False, jitter=None,
                 random_state=None):
        self.alpha, self.fit_intercept, self.verbose, self.precompute = alpha, fit_intercept, verbose, precompute
        self.max_iter, self.eps, self.copy_X, self.fit_path = max_iter, eps, copy_X, fit_path
        self.positive, self.jitter, self.random_state = positive, jitter, random_state

    def fit(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        self : LassoLars
            The fitted model.

        Raises
        ------
        ValueError
            If ``alpha`` is negative or ``jitter`` is not None.
        """
        return _XL_CALL(self, "fit_lasso_lars", (X, y), _XL_CONTEXT)


# ----------------------------------------------------------------- Quantile
# Reference problem: scikit-learn sklearn/linear_model/_quantile.py; the
# solver is ADMM (x_linear/quantile.mojo), not their linear program.

class QuantileRegressor(_LinearRegressorMixin, NumericModeMixin):
    """Linear regression that predicts a conditional quantile.

    Solves scikit-learn's QuantileRegressor problem (the pinball loss plus
    an L1 penalty) with ADMM instead of a linear program, then moves the
    answer onto the linear program's vertex (the rows the fit passes through
    solved exactly) whenever that does not raise the objective, so results
    agree with scikit-learn's exact solution. ``solver`` accepts
    scikit-learn's names but always runs ADMM; ``max_iter`` and ``tol`` are
    mojolearn's own keywords.

    Parameters
    ----------
    quantile : float, default 0.5
        Quantile to predict, strictly between 0 and 1.
    alpha : float, default 1.0
        L1 regularization strength (>= 0).
    fit_intercept : bool, default True
        Whether to fit an intercept.
    solver : str, default 'highs'
        Accepted for compatibility; the fit always runs ADMM.
    solver_options : dict or None, default None
        Accepted for compatibility; ignored.
    max_iter : int, default 5000
        Maximum number of ADMM iterations.
    tol : float, default 1e-4
        ADMM relative tolerance; the absolute tolerance is ``tol / 100``.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    coef_ : float32 array of shape (n_features,)
        Weights.
    intercept_ : float
        Intercept.
    n_iter_ : int
        Number of ADMM iterations run.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X[:, 0] + rng.standard_normal(1000).astype(np.float32)
    >>> model = mojolearn.QuantileRegressor(quantile=0.9, alpha=0.0).fit(X, y)
    """

    _BINDING = _BINDING

    def __init__(self, *, quantile=0.5, alpha=1.0, fit_intercept=True, solver="highs",
                 solver_options=None, max_iter=5000, tol=1e-4):
        self.quantile, self.alpha, self.fit_intercept = quantile, alpha, fit_intercept
        self.solver, self.solver_options, self.max_iter, self.tol = solver, solver_options, max_iter, tol

    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Targets.
        sample_weight : array-like of shape (n_samples,) or float, optional
            Non-negative weights with a positive sum.

        Returns
        -------
        self : QuantileRegressor
            The fitted model.

        Raises
        ------
        ValueError
            If ``quantile`` is not in (0, 1), ``alpha`` is negative or the
            weights are invalid.
        """
        return _XL_CALL(self, "fit_quantile", (X, y, sample_weight), _XL_CONTEXT)


# --------------------------------------------------------------- Perceptron
# Reference: scikit-learn sklearn/linear_model/_perceptron.py: SGD with the
# perceptron loss, a constant rate eta0 and no penalty (x_linear/sgd.mojo).

class Perceptron(_LinearClassifierMixin, NumericModeMixin):
    """Perceptron classifier.

    Follows scikit-learn's Perceptron: SGD with the perceptron loss and a
    constant step ``eta0``, one-vs-rest for more than two classes. Unlike
    scikit-learn, the default fit runs minibatches with ``batch_size='auto'``,
    which picks the batch from the row count (``n_samples // 256`` rows, at
    most 256), each step the sum of its rows' updates; ``batch_size=0`` gives
    scikit-learn's per-sample updates.

    Parameters
    ----------
    penalty : {'l2', 'l1', 'elasticnet'} or None, default None
        Regularization term.
    alpha : float, default 0.0001
        Regularization strength, used with a penalty.
    l1_ratio : float, default 0.15
        Elastic-net mixing; used with ``penalty='elasticnet'``.
    fit_intercept : bool, default True
        Whether to fit an intercept.
    max_iter : int, default 1000
        Maximum number of passes over the data.
    tol : float or None, default 1e-3
        Stopping tolerance, as in scikit-learn; None disables it.
    shuffle : bool, default True
        Whether to shuffle the rows each pass.
    verbose : int, default 0
        Accepted for compatibility; ignored.
    eta0 : float, default 1.0
        Constant step size (> 0).
    n_jobs : int or None, default None
        Accepted for compatibility; ignored.
    random_state : int or None, default 0
        Non-negative seed for shuffling. None means seed 0.
    early_stopping : bool, default False
        Not supported: True raises ValueError.
    validation_fraction : float, default 0.1
        Accepted for compatibility; ignored (early stopping is not supported).
    n_iter_no_change : int, default 5
        Passes without improvement before stopping.
    class_weight : dict, 'balanced' or None, default None
        Per-class weights; 'balanced' weighs classes inversely to their frequency.
    warm_start : bool, default False
        Not supported: True raises ValueError.
    batch_size : 'auto' or int, default 'auto'
        Rows per minibatch step; 'auto' picks the batch from the row count
        (``n_samples // 256``, at least 1, at most 256), 0 runs per-sample updates.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels.
    coef_ : float32 array of shape (1, n_features) or (n_classes, n_features)
        Weights; one row for a binary problem.
    intercept_ : float32 array of shape (1,) or (n_classes,)
        Intercepts.
    n_iter_ : int
        Number of passes run.
    t_ : float
        Number of weight updates, ``n_iter_ * n_samples + 1``.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > X[:, 1]).astype(np.int32)
    >>> model = mojolearn.Perceptron().fit(X, y)
    """

    _BINDING = _BINDING

    def __init__(self, *, penalty=None, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000,
                 tol=1e-3, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=0,
                 early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None,
                 warm_start=False, batch_size="auto"):
        self.batch_size = batch_size
        self.penalty, self.alpha, self.l1_ratio, self.fit_intercept = penalty, alpha, l1_ratio, fit_intercept
        self.max_iter, self.tol, self.shuffle, self.verbose, self.eta0 = max_iter, tol, shuffle, verbose, eta0
        self.n_jobs, self.random_state, self.early_stopping = n_jobs, random_state, early_stopping
        self.validation_fraction, self.n_iter_no_change = validation_fraction, n_iter_no_change
        self.class_weight, self.warm_start = class_weight, warm_start

    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Class labels; at least two classes.
        sample_weight : array-like of shape (n_samples,) or float, optional
            Non-negative weights with a positive sum.

        Returns
        -------
        self : Perceptron
            The fitted model.

        Raises
        ------
        ValueError
            On an unsupported option, a single class, invalid weights, or a
            floating-point overflow during training (scale the inputs).
        """
        return _XL_CALL(self, "fit_perceptron", (X, y, sample_weight), _XL_CONTEXT)


# -------------------------------------------------------- Passive-aggressive
# Reference: scikit-learn sklearn/linear_model/_passive_aggressive.py: the
# SGD kernel with learning_rate pa1 (hinge / epsilon_insensitive) or pa2
# (squared_*), eta0 = C, no penalty (x_linear/sgd.mojo, `_plain_sgd`'s PA step).

class PassiveAggressiveClassifier(_LinearClassifierMixin, NumericModeMixin):
    """Passive-aggressive classifier (PA-I or PA-II).

    Follows scikit-learn's PassiveAggressiveClassifier, one-vs-rest for more
    than two classes: ``loss='hinge'`` is PA-I and ``'squared_hinge'`` is
    PA-II. Unlike scikit-learn, the default fit runs minibatches of
    ``n_samples // 256`` rows (at most 256); ``batch_size=0`` gives scikit-learn's
    per-sample updates.

    Parameters
    ----------
    C : float, default 1.0
        Maximum step size (> 0).
    fit_intercept : bool, default True
        Whether to fit an intercept.
    max_iter : int, default 1000
        Maximum number of passes over the data.
    tol : float or None, default 1e-3
        Stopping tolerance, as in scikit-learn; None disables it.
    early_stopping : bool, default False
        Not supported: True raises ValueError.
    validation_fraction : float, default 0.1
        Accepted for compatibility; ignored (early stopping is not supported).
    n_iter_no_change : int, default 5
        Passes without improvement before stopping.
    shuffle : bool, default True
        Whether to shuffle the rows each pass.
    verbose : int, default 0
        Accepted for compatibility; ignored.
    loss : {'hinge', 'squared_hinge'}, default 'hinge'
        PA-I ('hinge') or PA-II ('squared_hinge').
    n_jobs : int or None, default None
        Accepted for compatibility; ignored.
    random_state : int or None, default None
        Non-negative seed for shuffling. None means seed 0.
    warm_start : bool, default False
        Not supported: True raises ValueError.
    class_weight : dict, 'balanced' or None, default None
        Per-class weights; 'balanced' weighs classes inversely to their frequency.
    average : bool, default False
        Not supported: True raises ValueError.
    batch_size : 'auto' or int, default 'auto'
        Rows per minibatch step; 'auto' is ``n_samples // 256`` (at least 1,
        at most 256), 0 runs per-sample updates.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels.
    coef_ : float32 array of shape (1, n_features) or (n_classes, n_features)
        Weights; one row for a binary problem.
    intercept_ : float32 array of shape (1,) or (n_classes,)
        Intercepts.
    n_iter_ : int
        Number of passes run.
    t_ : float
        Number of weight updates, ``n_iter_ * n_samples + 1``.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > X[:, 1]).astype(np.int32)
    >>> model = mojolearn.PassiveAggressiveClassifier().fit(X, y)
    """

    _BINDING = _BINDING

    def __init__(self, *, C=1.0, fit_intercept=True, max_iter=1000, tol=1e-3, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0, loss="hinge",
                 n_jobs=None, random_state=None, warm_start=False, class_weight=None, average=False,
                 batch_size="auto"):
        self.batch_size = batch_size
        self.C, self.fit_intercept, self.max_iter, self.tol = C, fit_intercept, max_iter, tol
        self.early_stopping, self.validation_fraction = early_stopping, validation_fraction
        self.n_iter_no_change, self.shuffle, self.verbose, self.loss = n_iter_no_change, shuffle, verbose, loss
        self.n_jobs, self.random_state, self.warm_start = n_jobs, random_state, warm_start
        self.class_weight, self.average = class_weight, average

    def fit(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Class labels; at least two classes.

        Returns
        -------
        self : PassiveAggressiveClassifier
            The fitted model.

        Raises
        ------
        ValueError
            On an unsupported option or loss, ``C <= 0``, a single class,
            invalid class weights, or a floating-point overflow during
            training.
        """
        return _XL_CALL(self, "fit_pa_classifier", (X, y), _XL_CONTEXT)


class PassiveAggressiveRegressor(_LinearRegressorMixin, NumericModeMixin):
    """Passive-aggressive regressor (PA-I or PA-II).

    Follows scikit-learn's PassiveAggressiveRegressor:
    ``loss='epsilon_insensitive'`` is PA-I and
    ``'squared_epsilon_insensitive'`` is PA-II. Unlike scikit-learn, the
    default fit runs minibatches of ``n_samples // 256`` rows (at most 256);
    ``batch_size=0`` gives scikit-learn's per-sample updates.

    Parameters
    ----------
    C : float, default 1.0
        Maximum step size (> 0).
    fit_intercept : bool, default True
        Whether to fit an intercept.
    max_iter : int, default 1000
        Maximum number of passes over the data.
    tol : float or None, default 1e-3
        Stopping tolerance, as in scikit-learn; None disables it.
    early_stopping : bool, default False
        Not supported: True raises ValueError.
    validation_fraction : float, default 0.1
        Accepted for compatibility; ignored (early stopping is not supported).
    n_iter_no_change : int, default 5
        Passes without improvement before stopping.
    shuffle : bool, default True
        Whether to shuffle the rows each pass.
    verbose : int, default 0
        Accepted for compatibility; ignored.
    loss : {'epsilon_insensitive', 'squared_epsilon_insensitive'}, default 'epsilon_insensitive'
        PA-I or PA-II.
    epsilon : float, default 0.1
        Residuals smaller than this cause no update.
    random_state : int or None, default None
        Non-negative seed for shuffling. None means seed 0.
    warm_start : bool, default False
        Not supported: True raises ValueError.
    average : bool, default False
        Not supported: True raises ValueError.
    batch_size : 'auto' or int, default 'auto'
        Rows per minibatch step; 'auto' is ``n_samples // 256`` (at least 1,
        at most 256), 0 runs per-sample updates.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    coef_ : float32 array of shape (n_features,)
        Weights.
    intercept_ : float32 array of shape (1,)
        Intercept.
    n_iter_ : int
        Number of passes run.
    t_ : float
        Number of weight updates, ``n_iter_ * n_samples + 1``.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ np.arange(8, dtype=np.float32)
    >>> model = mojolearn.PassiveAggressiveRegressor().fit(X, y)
    """

    _BINDING = _BINDING

    def __init__(self, *, C=1.0, fit_intercept=True, max_iter=1000, tol=1e-3, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0,
                 loss="epsilon_insensitive", epsilon=0.1, random_state=None, warm_start=False,
                 average=False, batch_size="auto"):
        self.batch_size = batch_size
        self.C, self.fit_intercept, self.max_iter, self.tol = C, fit_intercept, max_iter, tol
        self.early_stopping, self.validation_fraction = early_stopping, validation_fraction
        self.n_iter_no_change, self.shuffle, self.verbose, self.loss = n_iter_no_change, shuffle, verbose, loss
        self.epsilon, self.random_state, self.warm_start, self.average = epsilon, random_state, warm_start, average

    def fit(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Targets.

        Returns
        -------
        self : PassiveAggressiveRegressor
            The fitted model.

        Raises
        ------
        ValueError
            On an unsupported option or loss, ``C <= 0``, or a
            floating-point overflow during training.
        """
        return _XL_CALL(self, "fit_pa_regressor", (X, y), _XL_CONTEXT)


# ---------------------------------------------------------- SGDOneClassSVM
# Reference: scikit-learn sklearn/linear_model/_stochastic_gradient.py
# (SGDOneClassSVM, `_fit_one_class`: y = 1, hinge loss, l2, alpha = nu,
# intercept = 1 - offset, the offset step `- eta * alpha`); x_linear/sgd.mojo.

class SGDOneClassSVM(NumericModeMixin):
    """Linear one-class SVM trained by stochastic gradient descent.

    Follows scikit-learn's SGDOneClassSVM for outlier detection.
    ``predict`` returns +1 for inliers and -1 for outliers.
    ``random_state=None`` means seed 0, so a fit is reproducible by default.

    Parameters
    ----------
    nu : float, default 0.5
        Upper bound on the fraction of training errors, in (0, 1].
    fit_intercept : bool, default True
        Whether to fit the offset.
    max_iter : int, default 1000
        Maximum number of passes over the data.
    tol : float or None, default 1e-3
        Stopping tolerance, as in scikit-learn; None disables it.
    shuffle : bool, default True
        Whether to shuffle the rows each pass.
    verbose : int, default 0
        Accepted for compatibility; ignored.
    random_state : int or None, default None
        Non-negative seed for shuffling. None means seed 0.
    learning_rate : {'constant', 'optimal', 'invscaling', 'adaptive'}, default 'optimal'
        Step-size schedule.
    eta0 : float, default 0.0
        Initial step size; must be > 0 for 'constant', 'invscaling' and 'adaptive'.
    power_t : float, default 0.5
        Exponent of the 'invscaling' schedule.
    warm_start : bool, default False
        Not supported: True raises ValueError.
    average : bool, default False
        Not supported: True raises ValueError.
    batch_size : int or None, default None
        Rows per minibatch step. None chooses per numeric mode: minibatches
        of 4096 rows in ``'identical'`` mode, per-sample updates otherwise.
        0 asks for per-sample updates.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    coef_ : float32 array of shape (n_features,)
        Weights.
    offset_ : float32 array of shape (1,)
        Offset; ``decision_function = score_samples - offset_``.
    n_iter_ : int
        Number of passes run.
    t_ : float
        Number of weight updates, ``n_iter_ * n_samples + 1``.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.SGDOneClassSVM(nu=0.1).fit(X)
    >>> labels = model.predict(X)
    """

    _BINDING = _BINDING
    _estimator_type = "outlier_detector"

    def __init__(self, nu=0.5, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, verbose=0,
                 random_state=None, learning_rate="optimal", eta0=0.0, power_t=0.5, warm_start=False,
                 average=False, batch_size=None):
        # batch_size=None (fix-l1-linear, from lane/neural-pass139): "auto".
        # IDENTICAL builds fit the fixed-order minibatch at 4096 (the mean
        # step, a float-float intercept and an IMPLICIT intercept step,
        # x_linear/sgd.mojo `oc_solve`: the explicit step oscillated the
        # offset, board istella nu 0.1 flagged 0.374 of the training rows at
        # batch 256, sklearn 0.055); FAST builds, or
        # MOJOLEARN_SGD_OC_IDN_MB_DEFAULT_OFF, the per-sample fit
        # (lane/neural-pass132's default). 0 asks for the per-sample fit, a
        # positive batch_size for that minibatch.
        self.batch_size = batch_size
        self.nu, self.fit_intercept, self.max_iter, self.tol = nu, fit_intercept, max_iter, tol
        self.shuffle, self.verbose, self.random_state = shuffle, verbose, random_state
        self.learning_rate, self.eta0, self.power_t = learning_rate, eta0, power_t
        self.warm_start, self.average = warm_start, average

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : ignored
            Present for API compatibility.
        sample_weight : array-like of shape (n_samples,) or float, optional
            Non-negative weights with a positive sum.

        Returns
        -------
        self : SGDOneClassSVM
            The fitted model.

        Raises
        ------
        ValueError
            If ``nu`` is not in (0, 1], ``learning_rate`` is not supported,
            ``average`` or ``warm_start`` is True, or the weights are invalid.
        """
        return _XL_CALL(self, "fit_one_class", (X, y, sample_weight), _XL_CONTEXT)

    def decision_function(self, X):
        """Signed score of each sample: positive for inliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, converted to float32.

        Returns
        -------
        scores : float64 array of shape (n_samples,)
            ``X @ coef_ - offset_``.
        """
        return _XL_CALL(self, "one_class_decision_function", (X,), _XL_CONTEXT)

    def score_samples(self, X):
        """Raw score of each sample, ``X @ coef_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, converted to float32.

        Returns
        -------
        scores : float64 array of shape (n_samples,)
            Raw scores; higher is more normal.
        """
        return _XL_CALL(self, "one_class_score_samples", (X,), _XL_CONTEXT)

    def predict(self, X):
        """Label each sample as an inlier (+1) or an outlier (-1).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, converted to float32.

        Returns
        -------
        labels : int64 array of shape (n_samples,)
            +1 where ``decision_function`` is >= 0, -1 elsewhere.
        """
        return _XL_CALL(self, "one_class_predict", (X,), _XL_CONTEXT)


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


class RidgeClassifier(_LinearClassifierMixin, NumericModeMixin):
    """Classifier using ridge regression on -1/+1 targets.

    Follows scikit-learn's RidgeClassifier: each class becomes a -1/+1
    target column (one column for a binary problem), fitted by ridge
    regression with a Cholesky solve. Only ``solver='auto'`` or
    ``'cholesky'`` is accepted, and ``positive=True`` raises ValueError.

    Parameters
    ----------
    alpha : float, default 1.0
        L2 regularization strength (>= 0).
    fit_intercept : bool, default True
        Whether to fit an intercept.
    copy_X : bool, default True
        Accepted for compatibility; ``X`` is never modified.
    max_iter : int or None, default None
        Accepted for compatibility; ignored by the direct solver.
    tol : float, default 1e-4
        Accepted for compatibility; ignored by the direct solver.
    class_weight : dict, 'balanced' or None, default None
        Per-class weights, multiplied into ``sample_weight``; 'balanced'
        weighs classes inversely to their frequency.
    solver : {'auto', 'cholesky'}, default 'auto'
        Both run the Cholesky solver.
    positive : bool, default False
        Not supported: True raises ValueError.
    random_state : int or None, default None
        Accepted for compatibility; ignored.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels.
    coef_ : float32 array of shape (1, n_features) or (n_classes, n_features)
        Weights; one row for a binary problem.
    intercept_ : float32 array of shape (1,) or (n_classes,)
        Intercepts.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > X[:, 1]).astype(np.int32)
    >>> model = mojolearn.RidgeClassifier(alpha=1.0).fit(X, y)
    """

    _BINDING = _BINDING

    def __init__(self, alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=1e-4,
                 class_weight=None, solver="auto", positive=False, random_state=None):
        self.alpha, self.fit_intercept, self.copy_X, self.max_iter = alpha, fit_intercept, copy_X, max_iter
        self.tol, self.class_weight, self.solver, self.positive = tol, class_weight, solver, positive
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Class labels; at least two classes.
        sample_weight : array-like of shape (n_samples,) or float, optional
            Non-negative weights with a positive sum.

        Returns
        -------
        self : RidgeClassifier
            The fitted model.

        Raises
        ------
        ValueError
            On an unsupported ``solver`` or ``positive=True``, a negative
            ``alpha``, a single class, invalid weights, or when
            ``X'X + alpha I`` cannot be factored (rescale ``X`` or raise
            ``alpha``).
        """
        return _XL_CALL(self, "fit_ridge_classifier", (X, y, sample_weight), _XL_CONTEXT)


# Refused options are listed in x_linear/NOT_IMPLEMENTED.tsv.
class RidgeCV(_LinearRegressorMixin, NumericModeMixin):
    """Ridge regression with built-in selection of ``alpha``.

    Follows scikit-learn's RidgeCV. With ``cv=None`` the alpha is chosen by
    efficient leave-one-out cross-validation; otherwise by cross-validation
    over ``cv`` scored by R^2 (weighted by ``sample_weight`` when given),
    after which the model is refit on all rows with the best alpha (the
    first one on a tie). ``scoring``, ``alpha_per_target`` and a 2-D ``y``
    are not supported.

    Parameters
    ----------
    alphas : sequence of float, default (0.1, 1.0, 10.0)
        Candidate regularization strengths; all must be > 0.
    fit_intercept : bool, default True
        Whether to fit an intercept.
    scoring : None, default None
        Only None is supported; anything else raises ValueError.
    cv : int, cross-validation splitter, iterable or None, default None
        None for leave-one-out; an int for that many unshuffled K-fold folds
        (2 to n_samples); a splitter (an object with ``split``) or an
        iterable of ``(train, test)`` index arrays for those splits.
    gcv_mode : str or None, default None
        Accepted for compatibility; ignored.
    store_cv_results : bool, default False
        Store ``cv_results_``; only with ``cv=None``.
    alpha_per_target : bool, default False
        Only False is supported; True raises ValueError.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    alpha_ : float
        Selected alpha.
    best_score_ : float
        Score of the selected alpha: minus its mean squared leave-one-out
        error with ``cv=None``, otherwise its mean R^2 over the splits.
    coef_ : float32 array of shape (n_features,)
        Weights.
    intercept_ : float
        Intercept.
    cv_results_ : float32 array of shape (n_alphas,)
        Mean squared leave-one-out error of each alpha, when
        ``store_cv_results=True`` (scikit-learn stores the per-sample errors).
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ np.arange(8, dtype=np.float32)
    >>> model = mojolearn.RidgeCV(alphas=(0.01, 0.1, 1.0)).fit(X, y)
    >>> model.alpha_
    """

    _BINDING = _BINDING

    def __init__(self, alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, scoring=None, cv=None,
                 gcv_mode=None, store_cv_results=False, alpha_per_target=False):
        self.alphas, self.fit_intercept, self.scoring, self.cv = alphas, fit_intercept, scoring, cv
        self.gcv_mode, self.store_cv_results, self.alpha_per_target = gcv_mode, store_cv_results, alpha_per_target

    def fit(self, X, y, sample_weight=None):
        """Fit the model and select ``alpha``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Targets (1-D).
        sample_weight : array-like of shape (n_samples,) or float, optional
            Non-negative weights with a positive sum. With ``cv`` set, each
            split's fit uses its training rows' weights and its R^2 its test
            rows' weights.

        Returns
        -------
        self : RidgeCV
            The fitted model.

        Raises
        ------
        ValueError
            On an unsupported option, a non-positive alpha, an invalid
            ``cv``, ``store_cv_results`` with ``cv`` set, or when
            ``X'X + alpha I`` cannot be factored.
        """
        if self.scoring is not None or self.alpha_per_target:
            raise ValueError("mojolearn RidgeCV: only scoring=None, alpha_per_target=False are implemented")
        alphas = [float(v) for v in (self.alphas if hasattr(self.alphas, "__len__") else [self.alphas])]  # glue: converts the alphas argument
        if not alphas or any(not v > 0 for v in alphas):  # glue: validates the alphas argument
            raise ValueError("mojolearn RidgeCV: alphas must be positive")
        a, n, d = _matrix(X)
        yv = _vector(y, n)
        if self.cv is not None:
            # a k-fold with one-row test folds (n // k < 2) takes the split
            # walk, which scores such a fold NaN as scikit-learn does
            if _is_int_cv(self.cv) and sample_weight is None and int(self.cv) >= 2 and n // int(self.cv) >= 2:
                return self._fit_kfold(a, n, d, yv, alphas, sample_weight)
            return self._fit_splits(X, y, a, n, d, yv, alphas, sample_weight)
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
        cv = int(self.cv)
        if not 2 <= cv <= n:
            raise ValueError("mojolearn RidgeCV: cv must be in [2, n_samples]")
        if self.store_cv_results:
            raise ValueError("cv!=None and store_cv_results=True are incompatible")
        A = len(alphas)
        n_fw = 2 * d * d + 3 * d + 1 + A * (d + 2) + A * (n // cv + 1)
        scores = _run_array(self, ALGO_RIDGE_KFOLD, a, n, d, yv, [cv, int(bool(self.fit_intercept)), A], alphas, A,
                            n_fw, 1)
        # an alpha whose system is singular in float32 scores NaN (x_linear/ridgecv.mojo);
        # the first best finite score is chosen in Mojo (`x_linear_best_alpha`)
        best = int(_fit_module(self, ALGO_RIDGE_KFOLD).x_linear_best_alpha(addr_ro(scores, name="scores"), A))
        if best < 0:
            raise ValueError("mojolearn RidgeCV: X'X + alpha I is singular in float32 for every alpha "
                             "(rescale X, or use larger alphas)")
        vals = _ridge_run(self, a, n, d, yv, 1, [alphas[best]])
        self.coef_ = Array.from_list(vals[:d], "<f4")
        self.intercept_ = float(vals[d])
        self.alpha_ = alphas[best]
        self.best_score_ = float(scores[best])  # one element of the score Array
        self.n_features_in_ = d
        return self

    def _fit_splits(self, X, y, a, n, d, yv, alphas, sample_weight):
        """cv = a splitter, explicit (train, test) splits, or an int with
        sample_weight: scikit-learn's GridSearchCV over the alphas, each
        split's Ridge fit on its training rows with their weights, scored by
        Ridge.score on its test rows (r2_score weighted by the test rows'
        weights, as GridSearchCV routes sample_weight to a scorer that takes
        it), the mean over splits ranked, the first best alpha refit on every
        row with every weight. Python only walks the handful of splits and
        alphas: the row gathers (`gather_rows_bytes`), the fits (`_ridge_run`),
        the predictions (`_decision`), the scores (`r2_score`), the means
        (`row_means_f64`) and the choice (`reduce_stat` argmax) are native.
        A weight-w row is w repeated rows and a weight-0 row a removed one in
        every fit and every score."""
        from ._buffer import _native
        from ._metrics_impl import r2_score
        from .model_selection import _row_means
        if self.store_cv_results:
            raise ValueError("cv!=None and store_cv_results=True are incompatible")
        if _is_int_cv(self.cv):
            from .model_selection import KFold
            cv = int(self.cv)
            if not 2 <= cv <= n:
                raise ValueError("mojolearn RidgeCV: cv must be in [2, n_samples]")
            splits = _cv_splits(KFold(n_splits=cv), a, yv)
        else:
            splits = _cv_splits(self.cv, X, y)
        w = None if sample_weight is None else _weights_f32(sample_weight, n)
        gather = _native("gather_rows_bytes")

        def take(src, idx, width):
            # rows idx of src (width float32 words a row) as a new Array
            out = empty((idx.size, width) if width > 1 else (idx.size,), "<f4")
            gather(addr_ro(src, name="rows"), addr(out, name="rows"), addr_ro(idx, name="rows"), n, idx.size,
                   4 * width)
            return out

        self.n_features_in_ = d
        rows = [[] for _ in alphas]  # glue: one score list per alpha (alpha-grid sized)
        undefined = False
        for tr, te in splits:  # glue: one native gather, fit, predict and score per split and alpha
            if not tr.size or not te.size:
                raise ValueError("mojolearn RidgeCV: every cv split needs training and test rows")
            if te.size < 2:
                # scikit-learn's r2_score: NaN, with its warning, for fewer
                # than two test rows; every alpha's mean is then NaN, and
                # GridSearchCV ranks all-NaN means equal (the first alpha wins)
                from ._expansion_metrics import _undefined_warning
                _undefined_warning("R^2 score is not well-defined with less than two samples.")
                undefined = True
                continue
            xtr, ytr, xte, yte = take(a, tr, d), take(yv, tr, 1), take(a, te, d), take(yv, te, 1)
            wtr = None if w is None else take(w, tr, 1)
            wte = None if w is None else take(w, te, 1)
            for i, alpha in enumerate(alphas):  # glue: one native fit per alpha
                vals = _ridge_run(self, xtr, tr.size, d, ytr, 1, [alpha], wtr)
                pred = _decision(self, xte, Array.from_list(vals[:d], "<f4"), float(vals[d]))
                rows[i].append(float(r2_score(yte, pred.reshape((te.size,)), sample_weight=wte)))
        if undefined:
            best, best_score = 0, float("nan")
        else:
            means = Array.from_list(_row_means(rows), "<f8")
            best = int(_native("reduce_stat")(addr_ro(means, name="scores"), 1, len(alphas), 3))
            best_score = float(means[best])  # one element of the mean-score Array
        vals = _ridge_run(self, a, n, d, yv, 1, [alphas[best]], w)
        self.coef_ = Array.from_list(vals[:d], "<f4")
        self.intercept_ = float(vals[d])
        self.alpha_ = alphas[best]
        self.best_score_ = best_score
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
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        raise ValueError("mojolearn: cv must be None or an int in [2, n_samples]")
    if k > n:
        # scikit-learn's KFold wording (its check_fit2d_1sample reads it)
        raise ValueError(f"Cannot have number of splits n_splits={k} greater than the "
                         f"number of samples: n_samples={n}.")
    from ._buffer import _native
    ids = empty((n,), "<i4")
    counts = empty((k,), "<i8")
    _native("fold_ids")(0, n, 0, k, 0, addr(ids, name="folds"), addr(counts, name="folds"))
    out = empty((n,), "<f4")
    _native("cast_elements")(addr_ro(ids, name="folds"), 2, addr(out, name="folds"), 0, n)
    return out


def _is_int_cv(cv):
    """An integer fold count (not a bool): the kernels' own KFold /
    StratifiedKFold folds."""
    import numbers
    return isinstance(cv, numbers.Integral) and not isinstance(cv, bool)


def _cv_splits(cv, X, y):
    """scikit-learn's check_cv for a cv that is not None or an int: a
    splitter object (anything with `split`, called as ``cv.split(X, y)``) or
    an iterable of (train, test) index arrays, as a list of int64 Array
    pairs. Iterating the handful of splits is glue; each index array is
    converted natively (`as_index_i64`)."""
    from ._buffer import as_index_i64
    if hasattr(cv, "split"):
        pairs = cv.split(X, y)
    elif isinstance(cv, (str, bytes)) or not hasattr(cv, "__iter__"):
        # scikit-learn's check_cv wording
        raise ValueError("Expected cv as an integer, cross-validation object (from sklearn.model_selection) "
                         f"or an iterable. Got {cv!r}.")
    else:
        pairs = cv
    splits = []
    for train, test in pairs:  # glue: one (train, test) pair of index arrays per split
        splits.append((as_index_i64(train, name="train"), as_index_i64(test, name="test")))
    if not splits:
        raise ValueError("mojolearn: cv produced no (train, test) split")
    return splits


def _cv_fold_ids(splits, n):
    """Row -> fold id (an int32 Array) of explicit cv splits, checked by the
    base binding's helpers (no per-row Python; TargetEncoder's
    `_splitter_folds` rule), or None when the splits are not a partition:
    the test sets cover every row exactly once and each split's training
    rows are every other row (a fold's training rows are the rows with
    another fold id). KFold, StratifiedKFold, GroupKFold, LeaveOneGroupOut
    and LeaveOneOut splits all have it; ShuffleSplit, repeated and
    overlapping splits take per-split roles (`_cv_roles_f32`)."""
    from ._buffer import _native
    fold = full((n,), -1, "<i4")
    assign = _native("assign_fold_i64")  # cpu-route: input prep of a user cv splitter's index lists, before any fit
    covered = 0
    for k, (_, te) in enumerate(splits):  # glue: one native scatter per split
        if int(assign(addr_ro(te, name="test") if te.size else 0, te.size, n, k, addr(fold, name="folds"))) != 0:
            return None
        covered += te.size  # glue: one count per split (distinct rows: assign refuses a repeat)
    if covered != n:
        return None
    check = _native("check_indices_i64")
    hits = _native("count_fold_hits_i64")  # cpu-route: input prep of a user cv splitter's index lists, before any fit
    for k, (tr, te) in enumerate(splits):  # glue: one native check per split
        # every row outside fold k exactly once: as many as there are,
        # distinct, in range and none in fold k
        if (tr.size != n - te.size
                or tr.size and (int(check(addr_ro(tr, name="train"), tr.size, n)) != 0
                                or int(hits(addr_ro(tr, name="train"), tr.size, addr_ro(fold, name="folds"),
                                            n, k)) != 0)):
            return None
    return fold


def _cv_roles_f32(splits, n, name):
    """The per-split roles of cv splits that are not a partition
    (x_linear/cv_roles.mojo): the fold words (every one -(n + 1), which
    tells the kernels to read roles) and the (splits, n) roles, 1 for a
    training row, 2 for a test row, 0 for a row the split leaves out, written
    by the base binding (`mark_split_roles_f32`) from the index arrays."""
    from ._buffer import _native
    if n + 1 > 1 << 24:
        # the fold words carry n as an exact float32
        raise NotImplementedError(
            f"mojolearn {name}: cv splits that are not a partition are limited to 16,777,215 rows")
    mark = _native("mark_split_roles_f32")  # cpu-route: input prep of a user cv splitter's index lists, before any fit
    roles = zeros((len(splits) * n,), "<f4")
    for k, (tr, te) in enumerate(splits):  # glue: one native scatter per side of each split
        for side, role in ((tr, 1), (te, 2)):  # glue: the two sides of one split, one native scatter each
            status = int(mark(addr_ro(side, name="cv") if side.size else 0, side.size, n, k, role,
                              addr(roles, name="cv")))
            if status == 1:
                raise ValueError(f"mojolearn {name}: a cv split holds an index outside [0, n_samples)")
            if status == 2:
                raise ValueError(f"mojolearn {name}: a cv split repeats a row, or puts a row on both sides")
        if not tr.size or not te.size:
            raise ValueError(f"mojolearn {name}: every cv split needs training and test rows")
    return full((n,), -float(n + 1), "<f4"), roles


def _cv_fold_ids_f32(splits, n, name):
    """The fold description the kernels read after the targets, as float32
    (the kernels' y layout; `cast_elements`, as `_kfold_ids_f32`):
    (fold ids, None) for a partition, else (fold words, roles)
    (`_cv_roles_f32`); the roles ride after the weights."""
    from ._buffer import _native
    ids = _cv_fold_ids(splits, n)
    if ids is None:
        return _cv_roles_f32(splits, n, name)
    out = empty((n,), "<f4")
    _native("cast_elements")(addr_ro(ids, name="folds"), 2, addr(out, name="folds"), 0, n)
    return out, None


def _enetcv_fit(est, X, y, l1_ratios, sample_weight=None):
    name = type(est).__name__
    if est.selection != "cyclic":
        raise ValueError(f"mojolearn {name}: selection='random' is not implemented")
    if any(not 0 < r <= 1 for r in l1_ratios):  # glue: validates the l1_ratio argument
        raise ValueError(f"mojolearn {name}: l1_ratio must be in (0, 1]")
    a, n, d = _matrix(X)
    yv = _vector(y, n)
    roles = None
    if est.cv is None or _is_int_cv(est.cv):
        folds = 5 if est.cv is None else int(est.cv)
        ids = _kfold_ids_f32(n, folds)
    else:
        # a splitter or explicit (train, test) splits (scikit-learn's
        # check_cv): a partition's folds as row ids, any other splits as
        # per-split roles (x_linear/cv_roles.mojo)
        splits = _cv_splits(est.cv, X, y)
        folds = len(splits)
        ids, roles = _cv_fold_ids_f32(splits, n, name)
    alphas = est.alphas
    n_alphas = getattr(est, "n_alphas", None)
    if isinstance(alphas, int) and not isinstance(alphas, bool):
        explicit, grid = False, int(alphas)
    elif alphas is None or alphas in ("warn", "deprecated"):
        explicit, grid = False, int(n_alphas) if isinstance(n_alphas, int) else 100
    else:
        explicit = True
        values = sorted((float(v) for v in alphas), reverse=True)  # glue: sorts the user alphas argument
        grid = len(values)
    if grid < 1:
        raise ValueError(f"mojolearn {name}: at least one alpha is required")
    # scikit-learn's LinearModelCV.fit: a scalar sample_weight is no
    # weighting; an array is checked and rides after the fold ids, and ip[7]
    # selects the kernels' weighted arm (x_linear/cd.mojo, cd_grid.mojo)
    import numbers
    if isinstance(sample_weight, numbers.Number):
        sample_weight = None
    if roles is not None:
        # per-split roles ride the weighted arm, after the weights (unit
        # weights without a sample_weight: the same objective and scores)
        w = full((n,), 1.0, "<f4") if sample_weight is None else _weights_f32(sample_weight, n)
        yy, weighted = _concat_f32_parts([yv, ids, w, roles]), 1
    elif sample_weight is None:
        yy, weighted = _concat_f32(yv, ids), 0
    else:
        yy, weighted = _concat_f32_parts([yv, ids, _weights_f32(sample_weight, n)]), 1
    L = len(l1_ratios)
    fp = [est.eps, est.tol] + [float(r) for r in l1_ratios] + (values if explicit else [])  # glue: builds the float parameter list
    ip = [est.max_iter, int(bool(est.fit_intercept)), grid, folds, L, int(explicit), int(bool(est.positive)),
          weighted]
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
        est.mse_path_ = Array.from_list([ms[k * folds:(k + 1) * folds] for k in range(grid)], "<f4")  # glue: reshapes the returned mse path (grid-sized: alpha grid)
    else:
        est.alphas_ = Array.from_list([al[l * grid:(l + 1) * grid] for l in range(L)], "<f4")  # glue: reshapes the returned alpha path (grid-sized: alpha grid)
        est.mse_path_ = Array.from_list(
            [[ms[(l * grid + k) * folds:(l * grid + k + 1) * folds] for k in range(grid)] for l in range(L)], "<f4")  # glue: reshapes the returned mse path (grid-sized: alpha grid)
    est.n_features_in_ = d
    # the user's own value (the kernel carried it as float32)
    return min(l1_ratios, key=lambda r: abs(r - l1_best))  # glue: maps the returned best l1_ratio to the argument entry


class LassoCV(_LinearRegressorMixin, NumericModeMixin):
    """Lasso with ``alpha`` chosen by K-fold cross-validation along a path.

    Follows scikit-learn's LassoCV, solved by coordinate descent: each fold
    fits the regularization path, the alpha with the lowest mean squared
    error across folds is chosen, and the model is refit on all rows.
    ``selection='random'`` raises ValueError.

    Parameters
    ----------
    eps : float, default 1e-3
        Path length: ``alpha_min / alpha_max`` for a generated grid.
    n_alphas : int or 'deprecated', default 'deprecated'
        Grid size, used only when ``alphas`` is None; prefer an int ``alphas``.
    alphas : int or sequence of float, default 100
        Number of alphas on a generated grid, or the alphas to try.
    fit_intercept : bool, default True
        Whether to fit an intercept.
    precompute : bool, 'auto' or array, default 'auto'
        Accepted for compatibility; ignored.
    max_iter : int, default 1000
        Maximum number of coordinate-descent passes per alpha.
    tol : float, default 1e-4
        Coordinate-descent stopping tolerance.
    copy_X : bool, default True
        Accepted for compatibility; ``X`` is never modified.
    cv : int, cross-validation splitter, iterable or None, default None
        None means 5 folds; an int is the number of unshuffled K-fold folds;
        a splitter (an object with ``split``) or an iterable of
        ``(train, test)`` index arrays gives the splits (for example
        ``ShuffleSplit`` or ``RepeatedKFold``).
    verbose : bool, default False
        Accepted for compatibility; ignored.
    n_jobs : int or None, default None
        Accepted for compatibility; ignored.
    positive : bool, default False
        Restrict the weights to be non-negative.
    random_state : int or None, default None
        Accepted for compatibility; ignored.
    selection : {'cyclic'}, default 'cyclic'
        Only 'cyclic' is supported.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    alpha_ : float
        Selected alpha.
    alphas_ : float32 array of shape (n_alphas,)
        Alphas tried, in decreasing order.
    mse_path_ : float32 array of shape (n_alphas, n_folds)
        Mean squared error of each alpha on each fold.
    coef_ : float32 array of shape (n_features,)
        Weights of the refit model.
    intercept_ : float
        Intercept.
    n_iter_ : int
        Coordinate-descent passes of the refit.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = 3 * X[:, 0] - 2 * X[:, 1]
    >>> model = mojolearn.LassoCV(cv=5).fit(X, y)
    >>> model.alpha_
    """

    _BINDING = _BINDING

    def __init__(self, *, eps=1e-3, n_alphas="deprecated", alphas=100, fit_intercept=True,
                 precompute="auto", max_iter=1000, tol=1e-4, copy_X=True, cv=None, verbose=False,
                 n_jobs=None, positive=False, random_state=None, selection="cyclic"):
        self.eps, self.n_alphas, self.alphas, self.fit_intercept = eps, n_alphas, alphas, fit_intercept
        self.precompute, self.max_iter, self.tol, self.copy_X = precompute, max_iter, tol, copy_X
        self.cv, self.verbose, self.n_jobs, self.positive = cv, verbose, n_jobs, positive
        self.random_state, self.selection = random_state, selection

    def fit(self, X, y, sample_weight=None):
        """Fit the model and select ``alpha``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Targets.
        sample_weight : float or array-like of shape (n_samples,), default None
            Per-sample weights, as scikit-learn's: each fit rescales its
            weights to sum to its number of rows, and each held-out score is
            the weighted mean squared error. A single number means no
            weighting.

        Returns
        -------
        self : LassoCV
            The fitted model.

        Raises
        ------
        ValueError
            If ``selection`` is not 'cyclic', an int ``cv`` is outside
            [2, n_samples], or no alpha is given.
        """
        _enetcv_fit(self, X, y, [1.0], sample_weight)
        return self


class ElasticNetCV(_LinearRegressorMixin, NumericModeMixin):
    """Elastic net with ``l1_ratio`` and ``alpha`` chosen by K-fold cross-validation.

    Follows scikit-learn's ElasticNetCV, solved by coordinate descent: for
    each ``l1_ratio`` every fold fits the regularization path, the pair with
    the lowest mean squared error across folds is chosen, and the model is
    refit on all rows. ``selection='random'`` raises ValueError.

    Parameters
    ----------
    l1_ratio : float or sequence of float, default 0.5
        L1 share of the penalty, each in (0, 1]; a sequence is searched.
    eps : float, default 1e-3
        Path length: ``alpha_min / alpha_max`` for a generated grid.
    n_alphas : int or 'deprecated', default 'deprecated'
        Grid size, used only when ``alphas`` is None; prefer an int ``alphas``.
    alphas : int or sequence of float, default 100
        Number of alphas on a generated grid, or the alphas to try.
    fit_intercept : bool, default True
        Whether to fit an intercept.
    precompute : bool, 'auto' or array, default 'auto'
        Accepted for compatibility; ignored.
    max_iter : int, default 1000
        Maximum number of coordinate-descent passes per alpha.
    tol : float, default 1e-4
        Coordinate-descent stopping tolerance.
    cv : int, cross-validation splitter, iterable or None, default None
        None means 5 folds; an int is the number of unshuffled K-fold folds;
        a splitter (an object with ``split``) or an iterable of
        ``(train, test)`` index arrays gives the splits (for example
        ``ShuffleSplit`` or ``RepeatedKFold``).
    copy_X : bool, default True
        Accepted for compatibility; ``X`` is never modified.
    verbose : int, default 0
        Accepted for compatibility; ignored.
    n_jobs : int or None, default None
        Accepted for compatibility; ignored.
    positive : bool, default False
        Restrict the weights to be non-negative.
    random_state : int or None, default None
        Accepted for compatibility; ignored.
    selection : {'cyclic'}, default 'cyclic'
        Only 'cyclic' is supported.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    alpha_ : float
        Selected alpha.
    l1_ratio_ : float
        Selected l1_ratio (one of the values given).
    alphas_ : float32 array of shape (n_alphas,) or (n_l1_ratio, n_alphas)
        Alphas tried, in decreasing order; one row per l1_ratio when several.
    mse_path_ : float32 array of shape (n_alphas, n_folds) or (n_l1_ratio, n_alphas, n_folds)
        Mean squared error of each alpha on each fold.
    coef_ : float32 array of shape (n_features,)
        Weights of the refit model.
    intercept_ : float
        Intercept.
    n_iter_ : int
        Coordinate-descent passes of the refit.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = 3 * X[:, 0] - 2 * X[:, 1]
    >>> model = mojolearn.ElasticNetCV(l1_ratio=[0.2, 0.5, 0.9]).fit(X, y)
    >>> model.l1_ratio_, model.alpha_
    """

    _BINDING = _BINDING

    def __init__(self, *, l1_ratio=0.5, eps=1e-3, n_alphas="deprecated", alphas=100, fit_intercept=True,
                 precompute="auto", max_iter=1000, tol=1e-4, cv=None, copy_X=True, verbose=0,
                 n_jobs=None, positive=False, random_state=None, selection="cyclic"):
        self.l1_ratio, self.eps, self.n_alphas, self.alphas = l1_ratio, eps, n_alphas, alphas
        self.fit_intercept, self.precompute, self.max_iter, self.tol = fit_intercept, precompute, max_iter, tol
        self.cv, self.copy_X, self.verbose, self.n_jobs = cv, copy_X, verbose, n_jobs
        self.positive, self.random_state, self.selection = positive, random_state, selection

    def fit(self, X, y, sample_weight=None):
        """Fit the model and select ``l1_ratio`` and ``alpha``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Targets.
        sample_weight : float or array-like of shape (n_samples,), default None
            Per-sample weights, as scikit-learn's: each fit rescales its
            weights to sum to its number of rows, and each held-out score is
            the weighted mean squared error. A single number means no
            weighting.

        Returns
        -------
        self : ElasticNetCV
            The fitted model.

        Raises
        ------
        ValueError
            If an ``l1_ratio`` is outside (0, 1], ``selection`` is not
            'cyclic', an int ``cv`` is outside [2, n_samples], or no alpha
            is given.
        """
        ratios = list(self.l1_ratio) if hasattr(self.l1_ratio, "__len__") else [self.l1_ratio]
        self.l1_ratio_ = _enetcv_fit(self, X, y, [float(r) for r in ratios], sample_weight)  # glue: converts the l1_ratio argument
        return self


# ---------------------------------------------------- LogisticRegressionCV
# Reference: scikit-learn sklearn/linear_model/_logistic.py; kernel
# x_linear/logcv.mojo (L-BFGS on their LinearModelLoss objective).

def _check_stratified_folds(largest, k):
    """cv must be an int in [2, the largest class size] (StratifiedKFold's
    refusal; `largest` from the binding's class counts, `_class_prep`). The
    fold ids themselves are built from the labels inside the binding
    (x_linear/logcv.mojo `lcv_fold_table`, on the grid on a GPU)."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 2 or k > largest:
        raise ValueError("mojolearn: cv must be None or an int in [2, the largest class size]")


class LogisticRegressionCV(_LinearClassifierMixin, NumericModeMixin):
    """Logistic regression with ``C`` chosen by stratified K-fold cross-validation.

    Follows scikit-learn's LogisticRegressionCV with an L2 penalty, binary
    or multinomial, solved by L-BFGS. Each candidate ``C`` is scored by
    accuracy (weighted by ``sample_weight``) on the held-out folds
    (stratified folds by default), and the model is refit on all rows with
    the best one. Only ``penalty='l2'`` (primal), ``scoring=None`` and
    ``refit=True`` are supported; other values raise ValueError.

    Parameters
    ----------
    Cs : int or sequence of float, default 10
        Candidate inverse regularization strengths; an int ``m`` tries ``m``
        values log-spaced between 1e-4 and 1e4.
    fit_intercept : bool, default True
        Whether to fit an intercept.
    cv : int, cross-validation splitter, iterable or None, default None
        None means 5 stratified folds; an int is the number of stratified
        folds, from 2 to the size of the largest class; a splitter (an
        object with ``split``) or an iterable of ``(train, test)`` index
        arrays gives the splits (for example ``ShuffleSplit`` or
        ``RepeatedStratifiedKFold``).
    dual : bool, default False
        Only False is supported.
    penalty : {'l2'}, default 'l2'
        Only 'l2' is supported.
    scoring : None, default None
        Only None (accuracy) is supported.
    solver : {'lbfgs', 'newton-cg', 'newton-cholesky'}, default 'lbfgs'
        All three run L-BFGS.
    tol : float, default 1e-4
        L-BFGS stopping tolerance.
    max_iter : int, default 100
        Maximum number of L-BFGS iterations per fit.
    class_weight : dict, 'balanced' or None, default None
        Per-class weights; 'balanced' weighs classes inversely to their frequency.
    n_jobs : int or None, default None
        Accepted for compatibility; ignored.
    verbose : int, default 0
        Accepted for compatibility; ignored.
    refit : bool, default True
        Only True is supported.
    intercept_scaling : float, default 1.0
        Accepted for compatibility; ignored.
    random_state : int or None, default None
        Accepted for compatibility; ignored.
    l1_ratios : None, default None
        Only None is supported.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels.
    coef_ : float32 array of shape (1, n_features) or (n_classes, n_features)
        Weights; one row for a binary problem.
    intercept_ : float32 array of shape (1,) or (n_classes,)
        Intercepts.
    Cs_ : list of float
        Candidate C values.
    C_ : list of float
        Selected C, repeated once per row of ``coef_``.
    scores_ : dict
        For each class label (the positive class only, for a binary
        problem), a float32 array of shape (n_folds, n_Cs) of fold scores.
    n_iter_ : int
        L-BFGS iterations of the refit.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > X[:, 1]).astype(np.int32)
    >>> model = mojolearn.LogisticRegressionCV(Cs=5).fit(X, y)
    >>> proba = model.predict_proba(X)
    """

    _BINDING = _BINDING

    def __init__(self, *, Cs=10, fit_intercept=True, cv=None, dual=False, penalty="l2", scoring=None,
                 solver="lbfgs", tol=1e-4, max_iter=100, class_weight=None, n_jobs=None, verbose=0,
                 refit=True, intercept_scaling=1.0, random_state=None, l1_ratios=None):
        self.Cs, self.fit_intercept, self.cv, self.dual, self.penalty = Cs, fit_intercept, cv, dual, penalty
        self.scoring, self.solver, self.tol, self.max_iter = scoring, solver, tol, max_iter
        self.class_weight, self.n_jobs, self.verbose, self.refit = class_weight, n_jobs, verbose, refit
        self.intercept_scaling, self.random_state, self.l1_ratios = intercept_scaling, random_state, l1_ratios

    def fit(self, X, y, sample_weight=None):
        """Fit the model and select ``C``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32; values must be finite.
        y : array-like of shape (n_samples,)
            Class labels; at least two classes.
        sample_weight : array-like of shape (n_samples,) or float, optional
            Non-negative weights with a positive sum.

        Returns
        -------
        self : LogisticRegressionCV
            The fitted model.

        Raises
        ------
        ValueError
            On an unsupported option, a single class, invalid weights, or an
            int ``cv`` outside [2, the size of the largest class].
        """
        if self.penalty != "l2" or self.dual or self.l1_ratios is not None:
            raise ValueError("mojolearn LogisticRegressionCV: only penalty='l2' (primal) is implemented")
        if self.scoring is not None or not self.refit:
            raise ValueError("mojolearn LogisticRegressionCV: scoring and refit=False are not implemented")
        if self.solver not in ("lbfgs", "newton-cg", "newton-cholesky"):
            raise ValueError("mojolearn LogisticRegressionCV: solver must be lbfgs (newton-* run L-BFGS too)")
        a, n, d = _matrix(X)
        classes, icodes = _class_codes(self, y, n)
        K = len(classes)
        kp = 1 if K == 2 else K
        if isinstance(self.Cs, int) and not isinstance(self.Cs, bool):
            m = self.Cs
            # np.logspace(-4, 4, m); 10 ** y through `_pm.powr` (DEVIATION 6900), not the platform pow
            Cs = [_pm.powr(10.0, -4 + 8 * i / (m - 1)) for i in range(m)] if m > 1 else [1e-4]  # glue: builds the Cs parameter grid (m-sized: Cs grid size)
        else:
            Cs = [float(c) for c in self.Cs]  # glue: converts the Cs argument
        explicit = not (self.cv is None or _is_int_cv(self.cv))
        roles = None
        if explicit:
            # a splitter or explicit (train, test) splits (scikit-learn's
            # check_cv): their row fold ids ride in y, and ip[6] = 1 tells
            # the bindings not to build StratifiedKFold ids over them
            splits = _cv_splits(self.cv, X, y)
            folds = len(splits)
            fold_ids, roles = _cv_fold_ids_f32(splits, n, "LogisticRegressionCV")
        else:
            folds = 5 if self.cv is None else int(self.cv)
        # lane cpu2-l10-linear: the class counts, the 'balanced' weights
        # (theirs: from the weighted counts of all of y) and the row weights
        # in the binding (x_linear/class_prep.mojo); y is assembled by byte
        # copies: codes n | fold ids n (zeros: the binding builds them from
        # the labels) | fit weights n | raw weights n
        raw = None if sample_weight is None else _weights_f32(sample_weight, n)
        weigh = sample_weight is not None or self.class_weight is not None
        if roles is not None:
            # per-split roles (x_linear/cv_roles.mojo) sit after both weight
            # blocks, so such a fit always sends them (unit weights: the
            # same objective and accuracy)
            weigh = True
        _, fitw, largest = _class_prep(self, self.class_weight, classes, icodes, raw, weighted=True,
                                       rows=self.class_weight is not None)
        if not explicit:
            _check_stratified_folds(largest, folds)
        parts = [icodes.astype("<f4"), fold_ids if explicit else zeros((n,), "<f4")]
        has_sw = 0
        if weigh:
            if raw is None:
                raw = full((n,), 1.0, "<f4")
            parts += [raw if fitw is None else fitw, raw]
            has_sw = 1
        if roles is not None:
            parts.append(roles)
        yy = _concat_f32_parts(parts)
        p = kp * (d + 1)
        nc = len(Cs)
        vals = _run(self, ALGO_LOGCV, a, n, d, yy, [self.max_iter, int(bool(self.fit_intercept)), kp, nc, folds, has_sw]
                    + ([1] if explicit else []),
                    [self.tol] + Cs, kp * d + kp + 2 + folds * nc, p + 1 + n * (kp + 1) + _lbfgs_work(p), 5)
        self.classes_ = classes
        self.coef_ = Array.from_list(_rows(vals, kp, d), "<f4")
        self.intercept_ = Array.from_list(vals[kp * d:kp * d + kp], "<f4")
        best_c = Cs[min(range(nc), key=lambda i: abs(Cs[i] - vals[kp * d + kp]))]  # glue: maps the returned best C to its grid entry (nc-sized: Cs grid size)
        self.Cs_ = Cs
        self.C_ = [best_c] * kp
        self.n_iter_ = int(vals[kp * d + kp + 1])
        off = kp * d + kp + 2
        grid = [vals[off + f * nc:off + (f + 1) * nc] for f in range(folds)]  # glue: reshapes the returned score grid (folds-sized: CV folds)
        labels = classes.tolist() if hasattr(classes, "tolist") else list(classes)
        self.scores_ = {lab: Array.from_list(grid, "<f4") for lab in (labels[1:] if kp == 1 else labels)}  # glue: one score table per class label
        self.n_features_in_ = d
        return self

    def predict_proba(self, X):
        """Class probabilities.

        The sigmoid of the score for a binary problem, the softmax of the
        class scores otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, converted to float32.

        Returns
        -------
        proba : float32 array of shape (n_samples, n_classes)
            Probability of each class, in the order of ``classes_``.
        """
        return _XL_CALL(self, "lrcv_predict_proba", (X,), _XL_CONTEXT)


# --------------------------------------------------------------- Isotonic
# Reference: scikit-learn sklearn/isotonic.py and sklearn/_isotonic.pyx;
# kernel x_linear/isotonic.mojo (sequential PAVA; interp1d-linear predict).


class IsotonicRegression(RegressorScoreMixin, NumericModeMixin):
    """Isotonic regression: the best monotone fit of one feature.

    Follows scikit-learn's IsotonicRegression: fits a non-decreasing (or
    non-increasing) step function by pool-adjacent-violators and predicts
    by linear interpolation between its thresholds. ``X`` is 1-D or of
    shape (n_samples, 1); rows with zero weight are ignored.

    Parameters
    ----------
    y_min : float or None, default None
        Lower bound on the fitted values.
    y_max : float or None, default None
        Upper bound on the fitted values.
    increasing : bool or 'auto', default True
        Direction of the fit; 'auto' uses the sign of Spearman's rank
        correlation of ``X`` and ``y``.
    out_of_bounds : {'nan', 'clip', 'raise'}, default 'nan'
        Prediction outside the training range of ``X``: NaN, the nearest
        end value, or a ValueError.
    numeric_mode : str or None, default None
        Numeric mode for this model (``'identical'``
        or ``'fast'``, which runs on Apple GPUs only); None uses the process
        default (see :func:`mojolearn.set_numeric_mode`).

    Attributes
    ----------
    increasing_ : bool
        Direction used by the fit.
    X_min_ : float
        Smallest training ``X``.
    X_max_ : float
        Largest training ``X``.
    X_thresholds_ : float32 array of shape (n_thresholds,)
        Increasing ``X`` values of the interpolation knots.
    y_thresholds_ : float32 array of shape (n_thresholds,)
        Fitted values at ``X_thresholds_``.
    n_features_in_ : int
        Always 1.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> x = rng.random(1000, dtype=np.float32)
    >>> y = x + 0.1 * rng.standard_normal(1000).astype(np.float32)
    >>> model = mojolearn.IsotonicRegression(out_of_bounds="clip").fit(x, y)
    >>> y_fit = model.predict(x)
    """

    _BINDING = _BINDING
    _estimator_type = "regressor"
    # scikit-learn's IsotonicRegression tags: X is one feature, 1-D
    _sklearn_input_tags = {"one_d_array": True, "two_d_array": False}

    def __init__(self, *, y_min=None, y_max=None, increasing=True, out_of_bounds="nan"):
        self.y_min, self.y_max, self.increasing, self.out_of_bounds = y_min, y_max, increasing, out_of_bounds

    def fit(self, X, y, sample_weight=None):
        """Fit the monotone function.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Training feature, converted to float32.
        y : array-like of shape (n_samples,)
            Targets.
        sample_weight : array-like of shape (n_samples,) or float, optional
            Non-negative weights with a positive sum.

        Returns
        -------
        self : IsotonicRegression
            The fitted model.

        Raises
        ------
        ValueError
            If ``out_of_bounds`` is invalid, ``X`` is empty or not 1-D, or
            the weights are invalid.
        """
        return _XL_CALL(self, "isotonic_fit", (X, y, sample_weight), _XL_CONTEXT)

    def predict(self, T):
        """Predict by linear interpolation; same as ``transform``.

        Parameters
        ----------
        T : array-like of shape (n_samples,) or (n_samples, 1)
            Values to predict at.

        Returns
        -------
        y_pred : float32 array of shape (n_samples,)
            Interpolated fitted values.
        """
        return _XL_CALL(self, "isotonic_transform", (T,), _XL_CONTEXT)

    def transform(self, T):
        """Interpolate the fitted function at ``T``.

        Parameters
        ----------
        T : array-like of shape (n_samples,) or (n_samples, 1)
            Values to transform, converted to float32.

        Returns
        -------
        y_pred : float32 array of shape (n_samples,)
            Interpolated fitted values; values outside the training range
            follow ``out_of_bounds``.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            With ``out_of_bounds='raise'``, if a value of ``T`` is outside
            the training range.
        """
        return _XL_CALL(self, "isotonic_transform", (T,), _XL_CONTEXT)

    def fit_transform(self, X, y, sample_weight=None):
        """Fit, then return the fitted values at ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Training feature.
        y : array-like of shape (n_samples,)
            Targets.
        sample_weight : array-like of shape (n_samples,) or float, optional
            Non-negative weights with a positive sum.

        Returns
        -------
        y_fit : float32 array of shape (n_samples,)
            Fitted values.
        """
        return _XL_CALL(self, "isotonic_fit_transform", (X, y, sample_weight), _XL_CONTEXT)

    def score(self, X, y, sample_weight=None):
        """Coefficient of determination R^2 of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Test samples.
        y : array-like of shape (n_samples,)
            True targets.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            R^2; 1.0 is a perfect fit.
        """
        return _XL_CALL(self, "isotonic_score", (X, y, sample_weight), _XL_CONTEXT)


# The native owner of every public fit / predict / transform / score above
# except RidgeCV, LassoCV, ElasticNetCV and LogisticRegressionCV.fit (their
# cv splitters are Python objects): bindings/x_linear_native.mojo, connected
# once at import.
from . import _native_x_linear_bridge  # noqa: E402
_XL_CALL, _XL_CONTEXT = _native_x_linear_bridge.connect({
    "sgd_clf_loss": _SGD_CLF_LOSS, "sgd_reg_loss": _SGD_REG_LOSS, "sgd_penalty": _SGD_PENALTY,
    "sgd_lr": _SGD_LR, "sgd_sum_losses": _SGD_SUM_LOSSES})
