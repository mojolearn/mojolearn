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

from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty, frombytes, zeros
from ._labels import argmax_rows, decode_labels, encode_labels, sorted_classes, threshold_codes
from ._mode import NumericModeMixin
from ._svm_impl import SVC, SVR
from .linear_model import (
    _QN_LOSS_SVC_L1, _QN_LOSS_SVC_L2, _QN_LOSS_SVR_L1, _QN_LOSS_SVR_L2,
    _accuracy_host, _check_qn_solver_fields, _labels_1d, _qn_fit_one_target,
    _qn_scores, _r2_host, _target_1d, _py2mojo_flags, _PY2MOJO_ROWS,
)

__all__ = ["SVC", "SVR", "LinearSVC", "LinearSVR"]


class _LinearSVMBase(NumericModeMixin):
    """The fields and refusals the two linear machines share."""

    _BINDING = "_mojolearn_estimators"
    _LOSSES = {}

    # QN(...) default the Python door does not expose (a class constant, so
    # the instance carries only its constructor parameters)
    penalty_normalized = True

    def _init_common(self, penalty, loss, C, fit_intercept, penalized_intercept,
                     tol, max_iter, linesearch_max_iter, lbfgs_memory):
        # Stored verbatim (scikit-learn's convention); `_check_common`
        # checks them when `fit` runs.
        self.penalty = penalty
        self.loss = loss
        self.C = C
        self.fit_intercept = fit_intercept
        self.penalized_intercept = penalized_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.linesearch_max_iter = linesearch_max_iter
        self.lbfgs_memory = lbfgs_memory

    def _check_common(self):
        name = type(self).__name__
        penalty, loss, C = self.penalty, self.loss, self.C
        if penalty not in ("l1", "l2"):
            raise ValueError(f"Expected penalty to be one of ['l1', 'l2'], got {penalty!r}")
        if loss not in self._LOSSES:
            raise ValueError(f"Expected loss to be one of {list(self._LOSSES)}, got {loss!r}")
        if not C > 0:
            raise ValueError(f"mojolearn {name}: C must be positive, got {C}")
        if self.penalized_intercept:
            raise NotImplementedError(
                f"mojolearn {name}: penalized_intercept=True is not implemented "
                "(the reference appends a column of ones to X and fits without "
                "a bias, linear.cu:150-163); the intercept is never penalized here")
        _check_qn_solver_fields(name, self.tol, self.max_iter, self.linesearch_max_iter,
                                self.lbfgs_memory)

    def _strengths(self):
        """`linear.cu:151,174-175`: `iC = 1 / C` on the chosen penalty."""
        inv_c = 1.0 / self.C
        return (inv_c, 0.0) if self.penalty == "l1" else (0.0, inv_c)

    # No `fit` here takes sample_weight: the quasi-Newton solver has no
    # weighted loss (GLMBase::add_sample_weights, glm_base.cuh:115; the qn_fit
    # binding refuses has_sample_weight by name; glm/NOT_IMPLEMENTED.tsv).
    # Advertising it only to raise would tell scikit-learn's
    # has_fit_parameter it works (lane sk-edges).


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
    class_weight : None
        Must be None; anything else raises ``NotImplementedError``.
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

    def _validate_params(self):
        """The constructor checks, run at the start of `fit`."""
        multi_class = self.multi_class
        if multi_class != "ovr":
            raise ValueError(
                f"mojolearn LinearSVC: multi_class must be 'ovr', got {multi_class!r}")
        if self.class_weight is not None:
            raise NotImplementedError(
                "mojolearn LinearSVC: class_weight is not implemented (it becomes "
                "a sample_weight in the reference, and sample_weight is not "
                "implemented; glm/NOT_IMPLEMENTED.tsv)")
        self._check_common()

    def fit(self, X, y):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, read as float32.
        y : array-like of shape (n_samples,)
            Class labels, at least two distinct values. A 2-D ``y``, a
            single class or a length mismatch raises ``ValueError``.

        Returns
        -------
        self : LinearSVC
            The fitted estimator.
        """
        self._validate_params()
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        rows, cols = x.shape
        labels, _ = _labels_1d(y)
        if labels is None:
            raise ValueError("mojolearn LinearSVC requires a 1-D y")
        if len(labels) != rows:
            raise ValueError("mojolearn LinearSVC X and y lengths differ")
        # lane py-shared: the native encoder (`sorted_classes` is its definition)
        self.classes_, codes = encode_labels(labels)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("mojolearn LinearSVC: y has one class")
        l1, l2 = self._strengths()
        # `OvrSelector` (linear.cu:50-54): y is 1 on the selected class and 0
        # elsewhere; two classes select class 1 and fit once.
        selected = [1] if n_classes == 2 else list(range(n_classes))
        blocks, n_iter, objectives, retcodes = [], 0, [], []
        ovr = getattr(self._bind("_mojolearn_estimators"), "qn_fit_ovr", None)
        if ovr is not None and codes.dtype == "<i4":
            # lane fam2-linear: X and the class codes cross the bus once;
            # the binding builds each 0/1 target on the device and runs the
            # same per-class fits (a binding without the entry keeps the
            # per-class calls below)
            n_fits = len(selected)
            n_coefs = cols + (1 if self.fit_intercept else 0)
            w_all = zeros((n_fits, n_coefs), "<f4")
            info = zeros((n_fits, 3), "<f4")
            codes_c = codes._as_c()
            ovr(
                addr_ro(x, name="X"), addr_ro(codes_c, name="codes"),
                addr(w_all, name="coef_"), addr(info, name="info"),
                [rows, cols, selected[0], n_fits, float(l1), float(l2),
                 float(self.tol), float(0.1 * self.tol), int(self.max_iter),
                 int(self.linesearch_max_iter), int(self.lbfgs_memory),
                 1 if self.fit_intercept else 0,
                 1 if self.penalty_normalized else 0,
                 int(self._LOSSES[self.loss])])
            for k, (fx, rc, it) in enumerate(info.tolist()):  # glue: unpacks one result row per fit
                blocks.append(w_all[k])
                n_iter = max(n_iter, int(it))
                objectives.append(float(fx))
                retcodes.append(int(rc))
            selected = []
        for cls in selected:  # glue: one binary fit per selected class
            # the 0/1 target through the core helpers (equal_elements, cast_elements)
            y_enc = (codes == cls).astype("<f4")
            w, k, fx, rc = _qn_fit_one_target(
                self, x, y_enc, 2, self._LOSSES[self.loss], l1, l2,
                self.tol, 0.1 * self.tol)
            blocks.append(w)
            n_iter = max(n_iter, k)
            objectives.append(fx)
            retcodes.append(rc)
        n_targets = len(blocks)
        n_coefs = cols + (1 if self.fit_intercept else 0)
        # the (C, n_coefs) block, joined as bytes; its column-major copy (one
        # Mojo strided copy) is the `w[c + C*j]` block `qn_decision_function` reads
        W = frombytes(b"".join(b.tobytes() for b in blocks), "<f4", (n_targets, n_coefs))  # glue: joins one coefficient block per target
        self._w = Array._view_of(W._as_order("F"), (n_targets * n_coefs,), "C")
        self.coef_ = W[:, :cols]
        self.intercept_ = W[:, cols] if self.fit_intercept else zeros((n_targets,), "<f4")
        self.n_iter_ = Array.from_list([n_iter], "<i8")
        self.objective_ = objectives[0] if n_targets == 1 else objectives
        self.retcode_ = retcodes[0] if n_targets == 1 else retcodes
        self.n_features_in_ = cols
        return self

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
        if not hasattr(self, "_w"):
            raise ValueError("mojolearn LinearSVC: call fit first")
        n_classes = len(self.classes_)
        return _qn_scores(self, X, self._w, 1 if n_classes == 2 else n_classes)

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
        b = self._bind("_mojolearn_estimators") if hasattr(self, "classes_") else None
        if (b is not None and len(self.classes_) == 2
                and _py2mojo_flags(b) & _PY2MOJO_ROWS):
            # lane/apple-fast-py2mojo-linear: the `z > 0` codes in the binding
            # (`qn_predict_binary`, the same decision function and threshold)
            if not hasattr(self, "_w"):
                raise ValueError("mojolearn LinearSVC: call fit first")
            x, _ = as_f32_c(X, ndim=2, name="X")
            if x.shape[1] != self.n_features_in_:
                raise ValueError(f"mojolearn {type(self).__name__} feature count differs from fit")
            codes = empty((x.shape[0],), "<i8")
            if x.shape[0]:
                b.qn_predict_binary(
                    addr_ro(x, name="X"), addr_ro(self._w, name="coef_"),
                    addr(codes, name="codes"),
                    [x.shape[0], x.shape[1], 1 if self.fit_intercept else 0])
            return decode_labels(self.classes_, codes)
        multi = getattr(b, "qn_predict_multiclass", None) if b is not None else None
        if multi is not None and len(self.classes_) > 2:
            # lane fam2-linear: the row argmax on the device beside the scores
            if not hasattr(self, "_w"):
                raise ValueError("mojolearn LinearSVC: call fit first")
            x, _ = as_f32_c(X, ndim=2, name="X")
            if x.shape[1] != self.n_features_in_:
                raise ValueError(f"mojolearn {type(self).__name__} feature count differs from fit")
            codes = empty((x.shape[0],), "<i8")
            if x.shape[0]:
                multi(
                    addr_ro(x, name="X"), addr_ro(self._w, name="coef_"),
                    addr(codes, name="codes"),
                    [x.shape[0], x.shape[1], 1 if self.fit_intercept else 0,
                     len(self.classes_)])
            return decode_labels(self.classes_, codes)
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            return decode_labels(self.classes_, threshold_codes(scores))
        return decode_labels(self.classes_, argmax_rows(scores))

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
        return _accuracy_host(self.predict(X), y)


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

    def _validate_params(self):
        """The constructor checks, run at the start of `fit`."""
        epsilon = self.epsilon
        if not (0.0 <= epsilon < float("inf")):
            raise ValueError(
                f"mojolearn LinearSVR: epsilon must be finite and non-negative, got {epsilon}")
        self._check_common()

    def fit(self, X, y):
        """Fit the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, read as float32.
        y : array-like of shape (n_samples,)
            Target values. A 2-D ``y`` or a length mismatch raises
            ``ValueError``.

        Returns
        -------
        self : LinearSVR
            The fitted estimator.
        """
        self._validate_params()
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        rows, cols = x.shape
        t = _target_1d(y, rows, "mojolearn LinearSVR requires a 1-D y",
                       "mojolearn LinearSVR X and y lengths differ")
        l1, l2 = self._strengths()
        w, n_iter, self.objective_, self.retcode_ = _qn_fit_one_target(
            self, x, t, 1, self._LOSSES[self.loss], l1, l2,
            self.tol, 0.1 * self.tol, svr_eps=self.epsilon)
        self._w = w
        self.coef_ = w[:cols]
        self.intercept_ = float(w[cols]) if self.fit_intercept else 0.0
        self.n_iter_ = Array.from_list([n_iter], "<i8")
        self.n_features_in_ = cols
        return self

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
        if not hasattr(self, "_w"):
            raise ValueError("mojolearn LinearSVR: call fit before predict")
        return _qn_scores(self, X, self._w)

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
        return _r2_host(self.predict(X), y)


# scikit-learn keeps more estimators in this module than it defines; they resolve from
# the package at first use (`_sklearn_modules`), so `from mojolearn.svm import ...`
# takes the same names as scikit-learn's module.
from ._sklearn_modules import LAYOUT as _SKLEARN_LAYOUT, lazy_exports as _lazy_exports  # noqa: E402
__getattr__ = _lazy_exports("svm")
__all__ += _SKLEARN_LAYOUT["svm"]
