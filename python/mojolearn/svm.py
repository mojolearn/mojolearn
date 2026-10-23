# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Maintainer notes (moved from the module docstring, 2026-10-12):
# Reference for the linear pair: cuML's LinearSVC and LinearSVR
# (cuml/svm/linear_svc.py, linear_svr.py, linear.pyx, and cpp/src/svm/linear.cu,
# which fills qn_params and calls qnFit); the losses are
# cpp/src/glm/qn/glm_svm.cuh, here glm/impl/qn/glm_svm.mojo (DEVIATIONS 708 and
# 714). DEVIATION 714 stands on this surface: the squared hinge's gradient is
# the derivative of its value, 2 (z - s), so LinearSVC(loss='squared_hinge')
# minimizes the objective it documents. The objective's penalty_normalized =
# true is linear.cu:151-176; tol -> grad_tol = tol, change_tol = 0.1 * tol is
# linear.pyx:150-151.
"""Support vector machines.

``SVC`` and ``SVR`` are kernel machines trained by SMO. ``LinearSVC`` and
``LinearSVR`` are linear machines trained by the quasi-Newton solver of
``LogisticRegression`` (L-BFGS, or OWL-QN for an l1 penalty) on the hinge
family of losses. The linear machines minimize::

    mean_i loss(y_i, x_i @ w + b) + penalty(w) / (C * n_samples)

where ``penalty`` is the l1 norm or half the squared l2 norm: scikit-learn's
``LinearSVC`` / ``LinearSVR`` objective divided by ``C * n_samples``. The
intercept is never penalized. ``tol`` is the gradient tolerance; the
objective-change tolerance is ``0.1 * tol``. This follows cuML's
``LinearSVC`` and ``LinearSVR``.
"""

from ._mode import NumericModeMixin
from . import _native_family_bridge
from ._svm_impl import SVC, SVR
from .linear_model import _QN_LOSS_SVC_L1, _QN_LOSS_SVC_L2, _QN_LOSS_SVR_L1, _QN_LOSS_SVR_L2

__all__ = ["SVC", "SVR", "LinearSVC", "LinearSVR"]



# The compiled family owner, connected once at import: every public method of
# the two linear machines is one native call (bindings/svm_mixture_native.mojo
# for LinearSVC.decision_function and LinearSVR.predict,
# bindings/svc_linear_native.mojo for fit, predict and score), handed the
# estimators binding `_bind` resolves.
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()

class _LinearSVMBase(NumericModeMixin):
    """The fields and refusals the two linear machines share."""

    _BINDING = "_mojolearn_estimators"
    _LOSSES = {}

    # QN(...) default the Python door does not expose (a class constant, so
    # the instance carries only its constructor parameters)
    penalty_normalized = True

    def _init_common(self, penalty, loss, C, fit_intercept, penalized_intercept,
                     tol, max_iter, linesearch_max_iter, lbfgs_memory):
        # Stored verbatim (scikit-learn's convention); the native owner
        # (bindings/svc_linear_native.mojo) checks them when `fit` runs.
        self.penalty = penalty
        self.loss = loss
        self.C = C
        self.fit_intercept = fit_intercept
        self.penalized_intercept = penalized_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.linesearch_max_iter = linesearch_max_iter
        self.lbfgs_memory = lbfgs_memory

    # sample_weight (and LinearSVC's class_weight) are the solver's per-row
    # weights (glm_base.mojo::qn_row_weight_kernel; lane sk-kernels).


class LinearSVC(_LinearSVMBase):
    """Linear support vector classification.

    Minimizes the hinge or squared hinge loss plus an l2 or l1 penalty
    (see the ``mojolearn.svm`` module description for the objective).
    Two classes fit one machine; more classes fit one machine per class
    against the rest, in class order. The intercept is never penalized.

    ``X`` is read as float32. ``y`` is a 1-D array of labels;
    ``classes_`` is the sorted unique labels. Invalid parameter values
    raise ``ValueError``.

    Parameters
    ----------
    penalty : {'l2', 'l1'}, default 'l2'
        The regularization term; ``'l1'`` uses OWL-QN and gives exact
        zeros.
    loss : {'squared_hinge', 'hinge'}, default 'squared_hinge'
        The data-fit loss.
    C : float, default 1.0
        Inverse regularization strength, > 0.
    fit_intercept : bool, default True
        Whether to fit an (unpenalized) intercept.
    penalized_intercept : bool, default False
        Only False is supported; True raises ``NotImplementedError``.
    class_weight : dict, 'balanced' or None, default None
        Weights per class: row ``i`` of class ``c`` has its loss multiplied
        by ``class_weight[c]`` (times its ``sample_weight``) in every
        machine. ``'balanced'`` uses ``n_samples / (n_classes * count(c))``.
    tol : float, default 1e-4
        Gradient tolerance, > 0.
    max_iter : int, default 1000
        Maximum number of solver iterations per machine.
    linesearch_max_iter : int, default 100
        Maximum number of line-search steps per iteration.
    lbfgs_memory : int, default 5
        Number of past updates kept by L-BFGS.
    multi_class : {'ovr'}, default 'ovr'
        One-vs-rest, the only supported value.
    numeric_mode : {'identical', 'fast'} or None, default None
        The numeric mode for this estimator; None uses the process default
        (see ``mojolearn.set_numeric_mode``). ``'fast'`` is for Apple GPUs.

    Attributes
    ----------
    classes_ : list
        The sorted class labels.
    coef_ : array of shape (1, n_features) or (n_classes, n_features), float32
        Coefficients: one row for two classes, one row per class for more.
    intercept_ : array of shape (1,) or (n_classes,), float32
        Intercepts; zeros when ``fit_intercept`` is False.
    n_iter_ : array of shape (1,), int64
        The largest iteration count over the machines.
    objective_ : float or list of float
        The final objective value (one per machine when there are more
        than two classes).
    retcode_ : int or list of int
        The solver's return code, as on ``LogisticRegression`` (one per
        machine when there are more than two classes).
    n_features_in_ : int
        The number of features seen in ``fit``.
    input_copied_ : bool
        Whether ``fit`` had to copy ``X`` to float32 C order.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int64)
    >>> clf = mojolearn.LinearSVC(C=10.0).fit(X, y)
    >>> labels = clf.predict(X)
    """

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "classifier"

    _LOSSES = {"hinge": _QN_LOSS_SVC_L1, "squared_hinge": _QN_LOSS_SVC_L2}

    def __init__(self, *, penalty="l2", loss="squared_hinge", C=1.0,
                 fit_intercept=True, penalized_intercept=False,
                 class_weight=None, tol=1e-4, max_iter=1000,
                 linesearch_max_iter=100, lbfgs_memory=5, multi_class="ovr"):
        self._init_common(penalty, loss, C, fit_intercept, penalized_intercept,
                          tol, max_iter, linesearch_max_iter, lbfgs_memory)
        self.class_weight = class_weight
        self.multi_class = multi_class

    def fit(self, X, y, sample_weight=None):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, read as float32.
        y : array-like of shape (n_samples,)
            Class labels, at least two distinct values. A 2-D ``y``, a
            single class or a length mismatch raises ``ValueError``.
        sample_weight : array-like of shape (n_samples,) or None
            Per-sample weights; each row's loss is multiplied by its
            weight. None weighs every row 1.

        Returns
        -------
        self : LinearSVC
            The fitted estimator.
        """
        return _NATIVE(self, "family_svc_linear_svc_fit", (self._bind("_mojolearn_estimators"), X, y, sample_weight), _NATIVE_CONTEXT)

    def decision_function(self, X):
        """Return the linear scores ``X @ coef_.T + intercept_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, read as float32.

        Returns
        -------
        scores : array of shape (n_samples,) or (n_samples, n_classes), float32
            For two classes, one score per row (positive means
            ``classes_[1]``); for more, one score per class.

        Raises
        ------
        ValueError
            If the model is not fitted, or ``X`` has a different number of
            features than the training data.
        """
        return _NATIVE(self, "family_svm_linear_svc_decision", (self._bind("_mojolearn_estimators"), X), _NATIVE_CONTEXT)

    def predict(self, X):
        """Predict class labels.

        For two classes a row is ``classes_[1]`` when its decision score is
        positive and ``classes_[0]`` otherwise. For more classes it is the
        class with the highest score; a tie goes to the class that comes
        first in ``classes_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, read as float32.

        Returns
        -------
        labels : array of shape (n_samples,) or list
            An int64 array for integer labels, a float64 array for float
            labels, and a Python list for any other label type.
        """
        return _NATIVE(self, "family_svc_linear_svc_predict", (self._bind("_mojolearn_estimators"), X), _NATIVE_CONTEXT)

    def score(self, X, y):
        """Return the mean accuracy on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            The fraction of rows where ``predict(X) == y``.
        """
        return _NATIVE(self, "family_svc_linear_svc_score", (self._bind("_mojolearn_estimators"), X, y), _NATIVE_CONTEXT)


class LinearSVR(_LinearSVMBase):
    """Linear support vector regression.

    Minimizes the epsilon-insensitive or squared epsilon-insensitive loss
    plus an l1 or l2 penalty (see the ``mojolearn.svm`` module
    description for the objective). The intercept is never penalized.

    The default penalty is ``'l1'`` (cuML's default), which uses OWL-QN
    and gives exact zeros. scikit-learn's ``LinearSVR`` is l2 only: pass
    ``penalty='l2'`` for its objective.

    ``X`` is read as float32; ``y`` must be one-dimensional. Invalid
    parameter values raise ``ValueError``.

    Parameters
    ----------
    epsilon : float, default 0.0
        Width of the insensitive zone, finite and >= 0.
    penalty : {'l1', 'l2'}, default 'l1'
        The regularization term.
    loss : {'epsilon_insensitive', 'squared_epsilon_insensitive'}, default 'epsilon_insensitive'
        The data-fit loss.
    C : float, default 1.0
        Inverse regularization strength, > 0.
    fit_intercept : bool, default True
        Whether to fit an (unpenalized) intercept.
    penalized_intercept : bool, default False
        Only False is supported; True raises ``NotImplementedError``.
    tol : float, default 1e-4
        Gradient tolerance, > 0.
    max_iter : int, default 1000
        Maximum number of solver iterations.
    linesearch_max_iter : int, default 100
        Maximum number of line-search steps per iteration.
    lbfgs_memory : int, default 5
        Number of past updates kept by L-BFGS.
    numeric_mode : {'identical', 'fast'} or None, default None
        The numeric mode for this estimator; None uses the process default
        (see ``mojolearn.set_numeric_mode``). ``'fast'`` is for Apple GPUs.

    Attributes
    ----------
    coef_ : array of shape (n_features,), float32
        The fitted coefficients.
    intercept_ : float
        The fitted intercept; 0.0 when ``fit_intercept`` is False.
    n_iter_ : array of shape (1,), int64
        The number of solver iterations run.
    objective_ : float
        The final value of the minimized objective.
    retcode_ : int
        The solver's return code, as on ``LogisticRegression``.
    n_features_in_ : int
        The number of features seen in ``fit``.
    input_copied_ : bool
        Whether ``fit`` had to copy ``X`` to float32 C order.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ np.arange(8, dtype=np.float32) + 1.0
    >>> model = mojolearn.LinearSVR(penalty="l2", epsilon=0.1).fit(X, y)
    >>> pred = model.predict(X)
    """

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "regressor"

    _LOSSES = {"epsilon_insensitive": _QN_LOSS_SVR_L1,
               "squared_epsilon_insensitive": _QN_LOSS_SVR_L2}

    def __init__(self, *, epsilon=0.0, penalty="l1", loss="epsilon_insensitive",
                 C=1.0, fit_intercept=True, penalized_intercept=False,
                 tol=1e-4, max_iter=1000, linesearch_max_iter=100,
                 lbfgs_memory=5):
        self._init_common(penalty, loss, C, fit_intercept, penalized_intercept,
                          tol, max_iter, linesearch_max_iter, lbfgs_memory)
        self.epsilon = epsilon

    def fit(self, X, y, sample_weight=None):
        """Fit the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, read as float32.
        y : array-like of shape (n_samples,)
            Target values. A 2-D ``y`` or a length mismatch raises
            ``ValueError``.
        sample_weight : array-like of shape (n_samples,) or None
            Per-sample weights; each row's loss is multiplied by its
            weight. None weighs every row 1.

        Returns
        -------
        self : LinearSVR
            The fitted estimator.
        """
        return _NATIVE(self, "family_svc_linear_svr_fit", (self._bind("_mojolearn_estimators"), X, y, sample_weight), _NATIVE_CONTEXT)

    def predict(self, X):
        """Predict with the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, read as float32.

        Returns
        -------
        y_pred : array of shape (n_samples,), float32
            ``X @ coef_ + intercept_``.

        Raises
        ------
        ValueError
            If the model is not fitted, or ``X`` has a different number of
            features than the training data.
        """
        return _NATIVE(self, "family_svm_linear_svr_predict", (self._bind("_mojolearn_estimators"), X), _NATIVE_CONTEXT)

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            ``1 - SS_res / SS_tot``, computed in float64. A constant ``y``
            scores 0.0.
        """
        return _NATIVE(self, "family_svc_linear_svr_score", (self._bind("_mojolearn_estimators"), X, y), _NATIVE_CONTEXT)


# scikit-learn keeps more estimators in this module than it defines; they resolve from
# the package at first use (`_sklearn_modules`), so `from mojolearn.svm import ...`
# takes the same names as scikit-learn's module.
from ._sklearn_modules import LAYOUT as _SKLEARN_LAYOUT, lazy_exports as _lazy_exports  # noqa: E402
__getattr__ = _lazy_exports("svm")
__all__ += _SKLEARN_LAYOUT["svm"]
