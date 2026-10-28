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
which validates, encodes labels, assigns CV folds (integers) and unpacks the
flat float32 result. NumPy-free (NUMPY_FREE_CONTRACT.md).
"""
from . import _backend
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._bufcheck import memcopy
from ._mode import NumericModeMixin, available_if
from ._mode import RegressorScoreMixin
from ._native_params_setter import NativeParamsSetter

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


#: lane/apple-fast-py2mojo-linear: `py2mojo_rows` modes (core/py2mojo_rows.mojo)
_ROWS_SGD_PROBA, _ROWS_LRCV_PROBA, _ROWS_HUBER_OUT = 2, 3, 4


class _LinearClassifierMixin(NativeParamsSetter):
    # One-call native public methods: x_linear_native owns the lifecycle.
    _mojolearn_native_public = True
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


class _LinearRegressorMixin(NativeParamsSetter):
    # One-call native public methods: x_linear_native owns the lifecycle.
    _mojolearn_native_public = True
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
                 class_weight=None, average=False, warm_start=False, batch_size="auto", numeric_mode=None):
        self.loss, self.penalty, self.alpha, self.l1_ratio = loss, penalty, alpha, l1_ratio
        self.fit_intercept, self.max_iter, self.tol, self.shuffle = fit_intercept, max_iter, tol, shuffle
        self.epsilon, self.random_state, self.learning_rate = epsilon, random_state, learning_rate
        self.eta0, self.power_t, self.early_stopping = eta0, power_t, early_stopping
        self.n_iter_no_change, self.class_weight, self.average = n_iter_no_change, class_weight, average
        self.warm_start = warm_start
        self.batch_size = batch_size
        self.numeric_mode = numeric_mode

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
                 early_stopping=False, n_iter_no_change=5, average=False, warm_start=False, batch_size="auto", numeric_mode=None):
        self.loss, self.penalty, self.alpha, self.l1_ratio = loss, penalty, alpha, l1_ratio
        self.fit_intercept, self.max_iter, self.tol, self.shuffle = fit_intercept, max_iter, tol, shuffle
        self.epsilon, self.random_state, self.learning_rate = epsilon, random_state, learning_rate
        self.eta0, self.power_t, self.early_stopping = eta0, power_t, early_stopping
        self.n_iter_no_change, self.average, self.warm_start = n_iter_no_change, average, warm_start
        self.batch_size = batch_size
        self.numeric_mode = numeric_mode

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
                 warm_start=False, verbose=0, numeric_mode=None):
        _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose)
        self.numeric_mode = numeric_mode

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
                 warm_start=False, verbose=0, numeric_mode=None):
        _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose)
        self.numeric_mode = numeric_mode

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
                 max_iter=100, tol=1e-4, warm_start=False, verbose=0, numeric_mode=None):
        self.power, self.link = power, link
        _glm_init(self, alpha, fit_intercept, solver, max_iter, tol, warm_start, verbose)
        self.numeric_mode = numeric_mode


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
                 fit_intercept=True, tol=1e-05, numeric_mode=None):
        self.epsilon, self.max_iter, self.alpha = epsilon, max_iter, alpha
        self.warm_start, self.fit_intercept, self.tol = warm_start, fit_intercept, tol
        self.numeric_mode = numeric_mode

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
                 fit_intercept=True, copy_X=True, verbose=False, numeric_mode=None):
        self.max_iter, self.tol, self.alpha_1, self.alpha_2 = max_iter, tol, alpha_1, alpha_2
        self.lambda_1, self.lambda_2, self.alpha_init, self.lambda_init = lambda_1, lambda_2, alpha_init, lambda_init
        self.compute_score, self.fit_intercept, self.copy_X, self.verbose = compute_score, fit_intercept, copy_X, verbose
        self.numeric_mode = numeric_mode

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
                 copy_X=True, verbose=False, numeric_mode=None):
        self.max_iter, self.tol, self.alpha_1, self.alpha_2 = max_iter, tol, alpha_1, alpha_2
        self.lambda_1, self.lambda_2, self.compute_score = lambda_1, lambda_2, compute_score
        self.threshold_lambda, self.fit_intercept, self.copy_X, self.verbose = (
            threshold_lambda, fit_intercept, copy_X, verbose)
        self.numeric_mode = numeric_mode

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
                 eps=2.220446049250313e-16, copy_X=True, fit_path=True, jitter=None, random_state=None, numeric_mode=None):
        self.fit_intercept, self.verbose, self.precompute = fit_intercept, verbose, precompute
        self.n_nonzero_coefs, self.eps, self.copy_X = n_nonzero_coefs, eps, copy_X
        self.fit_path, self.jitter, self.random_state = fit_path, jitter, random_state
        self.numeric_mode = numeric_mode

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
                 random_state=None, numeric_mode=None):
        self.alpha, self.fit_intercept, self.verbose, self.precompute = alpha, fit_intercept, verbose, precompute
        self.max_iter, self.eps, self.copy_X, self.fit_path = max_iter, eps, copy_X, fit_path
        self.positive, self.jitter, self.random_state = positive, jitter, random_state
        self.numeric_mode = numeric_mode

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
                 solver_options=None, max_iter=5000, tol=1e-4, numeric_mode=None):
        self.quantile, self.alpha, self.fit_intercept = quantile, alpha, fit_intercept
        self.solver, self.solver_options, self.max_iter, self.tol = solver, solver_options, max_iter, tol
        self.numeric_mode = numeric_mode

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
                 warm_start=False, batch_size="auto", numeric_mode=None):
        self.batch_size = batch_size
        self.penalty, self.alpha, self.l1_ratio, self.fit_intercept = penalty, alpha, l1_ratio, fit_intercept
        self.max_iter, self.tol, self.shuffle, self.verbose, self.eta0 = max_iter, tol, shuffle, verbose, eta0
        self.n_jobs, self.random_state, self.early_stopping = n_jobs, random_state, early_stopping
        self.validation_fraction, self.n_iter_no_change = validation_fraction, n_iter_no_change
        self.class_weight, self.warm_start = class_weight, warm_start
        self.numeric_mode = numeric_mode

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
                 batch_size="auto", numeric_mode=None):
        self.batch_size = batch_size
        self.C, self.fit_intercept, self.max_iter, self.tol = C, fit_intercept, max_iter, tol
        self.early_stopping, self.validation_fraction = early_stopping, validation_fraction
        self.n_iter_no_change, self.shuffle, self.verbose, self.loss = n_iter_no_change, shuffle, verbose, loss
        self.n_jobs, self.random_state, self.warm_start = n_jobs, random_state, warm_start
        self.class_weight, self.average = class_weight, average
        self.numeric_mode = numeric_mode

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
                 average=False, batch_size="auto", numeric_mode=None):
        self.batch_size = batch_size
        self.C, self.fit_intercept, self.max_iter, self.tol = C, fit_intercept, max_iter, tol
        self.early_stopping, self.validation_fraction = early_stopping, validation_fraction
        self.n_iter_no_change, self.shuffle, self.verbose, self.loss = n_iter_no_change, shuffle, verbose, loss
        self.epsilon, self.random_state, self.warm_start, self.average = epsilon, random_state, warm_start, average
        self.numeric_mode = numeric_mode

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
    _mojolearn_native_public = True
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
                 average=False, batch_size=None, *, numeric_mode=None):
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
        self.numeric_mode = numeric_mode

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
                 class_weight=None, solver="auto", positive=False, random_state=None, numeric_mode=None):
        self.alpha, self.fit_intercept, self.copy_X, self.max_iter = alpha, fit_intercept, copy_X, max_iter
        self.tol, self.class_weight, self.solver, self.positive = tol, class_weight, solver, positive
        self.random_state = random_state
        self.numeric_mode = numeric_mode

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
                 gcv_mode=None, store_cv_results=False, alpha_per_target=False, numeric_mode=None):
        self.alphas, self.fit_intercept, self.scoring, self.cv = alphas, fit_intercept, scoring, cv
        self.gcv_mode, self.store_cv_results, self.alpha_per_target = gcv_mode, store_cv_results, alpha_per_target
        self.numeric_mode = numeric_mode

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
        return _XL_CALL(self, "fit_ridge_cv", (X, y, sample_weight), _XL_CONTEXT)


# ------------------------------------------------------ LassoCV / ElasticNetCV
# Reference: scikit-learn sklearn/linear_model/_coordinate_descent.py
# (LinearModelCV.fit, _alpha_grid, _path_residuals) and _cd_fast.pyx
# (enet_coordinate_descent_gram); kernel x_linear/cd.mojo.


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
                 n_jobs=None, positive=False, random_state=None, selection="cyclic", numeric_mode=None):
        self.eps, self.n_alphas, self.alphas, self.fit_intercept = eps, n_alphas, alphas, fit_intercept
        self.precompute, self.max_iter, self.tol, self.copy_X = precompute, max_iter, tol, copy_X
        self.cv, self.verbose, self.n_jobs, self.positive = cv, verbose, n_jobs, positive
        self.random_state, self.selection = random_state, selection
        self.numeric_mode = numeric_mode

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
        return _XL_CALL(self, "fit_lasso_cv", (X, y, sample_weight), _XL_CONTEXT)


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
                 n_jobs=None, positive=False, random_state=None, selection="cyclic", numeric_mode=None):
        self.l1_ratio, self.eps, self.n_alphas, self.alphas = l1_ratio, eps, n_alphas, alphas
        self.fit_intercept, self.precompute, self.max_iter, self.tol = fit_intercept, precompute, max_iter, tol
        self.cv, self.copy_X, self.verbose, self.n_jobs = cv, copy_X, verbose, n_jobs
        self.positive, self.random_state, self.selection = positive, random_state, selection
        self.numeric_mode = numeric_mode

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
        return _XL_CALL(self, "fit_elastic_net_cv", (X, y, sample_weight), _XL_CONTEXT)


# ---------------------------------------------------- LogisticRegressionCV
# Reference: scikit-learn sklearn/linear_model/_logistic.py; kernel
# x_linear/logcv.mojo (L-BFGS on their LinearModelLoss objective).


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
                 refit=True, intercept_scaling=1.0, random_state=None, l1_ratios=None, numeric_mode=None):
        self.Cs, self.fit_intercept, self.cv, self.dual, self.penalty = Cs, fit_intercept, cv, dual, penalty
        self.scoring, self.solver, self.tol, self.max_iter = scoring, solver, tol, max_iter
        self.class_weight, self.n_jobs, self.verbose, self.refit = class_weight, n_jobs, verbose, refit
        self.intercept_scaling, self.random_state, self.l1_ratios = intercept_scaling, random_state, l1_ratios
        self.numeric_mode = numeric_mode

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
        return _XL_CALL(self, "fit_logistic_cv", (X, y, sample_weight), _XL_CONTEXT)

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
    _mojolearn_native_public = True
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

    def __init__(self, *, y_min=None, y_max=None, increasing=True, out_of_bounds="nan", numeric_mode=None):
        self.y_min, self.y_max, self.increasing, self.out_of_bounds = y_min, y_max, increasing, out_of_bounds
        self.numeric_mode = numeric_mode

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
        return _XL_CALL(self, "isotonic_predict", (T,), _XL_CONTEXT)

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
# (bindings/x_linear_native.mojo), connected once at import. A user cv
# splitter's `split(X, y)` is called once by it, as the Python did.
from . import _native_x_linear_bridge  # noqa: E402
_XL_CALL, _XL_CONTEXT = _native_x_linear_bridge.connect({
    "sgd_clf_loss": _SGD_CLF_LOSS, "sgd_reg_loss": _SGD_REG_LOSS, "sgd_penalty": _SGD_PENALTY,
    "sgd_lr": _SGD_LR, "sgd_sum_losses": _SGD_SUM_LOSSES})
