# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.svm`: the kernel machines `SVC` and `SVR` (`_svm_impl.py`, the
SMO solver) and the linear machines `LinearSVC` and `LinearSVR`, which are
the quasi-Newton solver of `LogisticRegression` on the four hinge-family
losses.

Reference for the linear pair: cuML's `LinearSVC` and `LinearSVR`
(`cuml/svm/linear_svc.py`, `linear_svr.py`, `linear.pyx`, and
`cpp/src/svm/linear.cu`, which fills `qn_params` and calls `qnFit`); the
losses are `cpp/src/glm/qn/glm_svm.cuh`, here `glm/impl/qn/glm_svm.mojo`
(DEVIATIONS 708 and 714). DEVIATION 714 stands on this surface: the
squared hinge's gradient is the derivative of its value, `2 (z - s)`, so
`LinearSVC(loss='squared_hinge')` minimizes the objective it documents.

THE OBJECTIVE is `mean_i lz(y_i, x_i w + b) + (1 / (C n)) pen(w)`, `pen`
the l1 norm or half the squared l2 norm (`linear.cu:151-176`,
`penalty_normalized = true`): scikit-learn's `LinearSVC` / `LinearSVR`
objective divided by `C n`. `penalty='l1'` selects OWL-QN, as on
`LogisticRegression`. `tol` maps to `grad_tol = tol`, `change_tol = 0.1 *
tol` (`linear.pyx:150-151`).
"""

from ._array import Array
from ._buffer import as_f32_c
from ._labels import argmax_rows, decode_labels, sorted_classes
from ._mode import NumericModeMixin
from ._svm_impl import SVC, SVR
from .linear_model import (
    _QN_LOSS_SVC_L1, _QN_LOSS_SVC_L2, _QN_LOSS_SVR_L1, _QN_LOSS_SVR_L2,
    _accuracy_host, _check_qn_solver_fields, _labels_1d, _qn_fit_one_target,
    _qn_scores, _r2_host, _target_1d,
)

__all__ = ["SVC", "SVR", "LinearSVC", "LinearSVR"]


class _LinearSVMBase(NumericModeMixin):
    """The fields and refusals the two linear machines share."""

    _BINDING = "_mojolearn_estimators"
    _LOSSES = {}

    def _init_common(self, penalty, loss, C, fit_intercept, penalized_intercept,
                     tol, max_iter, linesearch_max_iter, lbfgs_memory):
        name = type(self).__name__
        if penalty not in ("l1", "l2"):
            raise ValueError(f"Expected penalty to be one of ['l1', 'l2'], got {penalty!r}")
        if loss not in self._LOSSES:
            raise ValueError(f"Expected loss to be one of {list(self._LOSSES)}, got {loss!r}")
        if not C > 0:
            raise ValueError(f"mojolearn {name}: C must be positive, got {C}")
        if penalized_intercept:
            raise NotImplementedError(
                f"mojolearn {name}: penalized_intercept=True is not implemented "
                "(the reference appends a column of ones to X and fits without "
                "a bias, linear.cu:150-163); the intercept is never penalized here")
        _check_qn_solver_fields(name, tol, max_iter, linesearch_max_iter, lbfgs_memory)
        self.penalty = penalty
        self.loss = loss
        self.C = C
        self.fit_intercept = fit_intercept
        self.penalized_intercept = False
        self.tol = tol
        self.max_iter = max_iter
        self.linesearch_max_iter = linesearch_max_iter
        self.lbfgs_memory = lbfgs_memory
        self.penalty_normalized = True

    def _strengths(self):
        """`linear.cu:151,174-175`: `iC = 1 / C` on the chosen penalty."""
        inv_c = 1.0 / self.C
        return (inv_c, 0.0) if self.penalty == "l1" else (0.0, inv_c)

    def _refuse_sample_weight(self, sample_weight):
        if sample_weight is not None:
            raise NotImplementedError(
                f"mojolearn {type(self).__name__}: sample_weight is not implemented "
                "(GLMBase::add_sample_weights, glm_base.cuh:115; "
                "glm/NOT_IMPLEMENTED.tsv)")


class LinearSVC(_LinearSVMBase):
    """Linear support vector classification.

        loss            'squared_hinge' (default) or 'hinge'
        penalty         'l2' (default) or 'l1' (OWL-QN)
        C               honored, > 0
        fit_intercept, tol, max_iter, linesearch_max_iter, lbfgs_memory
                        honored
        multi_class     'ovr' only, the only value the reference accepts:
                        more than two classes fit one machine per class
                        against the rest, in class order
        penalized_intercept   refused by name
        class_weight    refused   it becomes a sample_weight in the
                                  reference, and sample_weight is not
                                  implemented
        sample_weight   refused

    OUTPUTS: `classes_` (sorted, a Python list), `coef_` (1, n_features)
    float32 and `intercept_` (1,) for two classes, (C, n_features) and (C,)
    for C > 2, `n_iter_` (the largest count over the machines),
    `decision_function` (n,) or (n, C) float32. `predict` is `score > 0`
    for two classes and the row argmax, first maximum winning, for more.
    """

    _LOSSES = {"hinge": _QN_LOSS_SVC_L1, "squared_hinge": _QN_LOSS_SVC_L2}

    def __init__(self, *, penalty="l2", loss="squared_hinge", C=1.0,
                 fit_intercept=True, penalized_intercept=False,
                 class_weight=None, tol=1e-4, max_iter=1000,
                 linesearch_max_iter=100, lbfgs_memory=5, multi_class="ovr"):
        if multi_class != "ovr":
            raise ValueError(
                f"mojolearn LinearSVC: multi_class must be 'ovr', got {multi_class!r}")
        if class_weight is not None:
            raise NotImplementedError(
                "mojolearn LinearSVC: class_weight is not implemented (it becomes "
                "a sample_weight in the reference, and sample_weight is not "
                "implemented; glm/NOT_IMPLEMENTED.tsv)")
        self._init_common(penalty, loss, C, fit_intercept, penalized_intercept,
                          tol, max_iter, linesearch_max_iter, lbfgs_memory)
        self.class_weight = None
        self.multi_class = multi_class

    def fit(self, X, y, sample_weight=None):
        self._refuse_sample_weight(sample_weight)
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        rows, cols = x.shape
        labels, _ = _labels_1d(y)
        if labels is None:
            raise ValueError("mojolearn LinearSVC requires a 1-D y")
        if len(labels) != rows:
            raise ValueError("mojolearn LinearSVC X and y lengths differ")
        self.classes_, codes = sorted_classes(labels)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("mojolearn LinearSVC: y has one class")
        l1, l2 = self._strengths()
        # `OvrSelector` (linear.cu:50-54): y is 1 on the selected class and 0
        # elsewhere; two classes select class 1 and fit once.
        selected = [1] if n_classes == 2 else list(range(n_classes))
        blocks, n_iter, objectives, retcodes = [], 0, [], []
        for cls in selected:
            y_enc = Array.from_list([1.0 if c == cls else 0.0 for c in codes], "<f4")
            w, k, fx, rc = _qn_fit_one_target(
                self, x, y_enc, 2, self._LOSSES[self.loss], l1, l2,
                self.tol, 0.1 * self.tol)
            blocks.append(w.tolist())
            n_iter = max(n_iter, k)
            objectives.append(fx)
            retcodes.append(rc)
        n_targets = len(blocks)
        n_coefs = cols + (1 if self.fit_intercept else 0)
        # the column-major `w[c + C*j]` block `qn_decision_function` reads
        self._w = Array.from_list(
            [blocks[c][j] for j in range(n_coefs) for c in range(n_targets)], "<f4")
        self.coef_ = Array.from_list([b[:cols] for b in blocks], "<f4")
        self.intercept_ = Array.from_list(
            [b[cols] if self.fit_intercept else 0.0 for b in blocks], "<f4")
        self.n_iter_ = Array.from_list([n_iter], "<i8")
        self.objective_ = objectives[0] if n_targets == 1 else objectives
        self.retcode_ = retcodes[0] if n_targets == 1 else retcodes
        self.n_features_in_ = cols
        return self

    def decision_function(self, X):
        if not hasattr(self, "_w"):
            raise ValueError("mojolearn LinearSVC: call fit first")
        n_classes = len(self.classes_)
        return _qn_scores(self, X, self._w, 1 if n_classes == 2 else n_classes)

    def predict(self, X):
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            return decode_labels(self.classes_,
                                 [1 if s > 0.0 else 0 for s in scores.tolist()])
        return decode_labels(self.classes_, argmax_rows(scores))

    def score(self, X, y):
        """Accuracy, a host count over O(rows) labels."""
        return _accuracy_host(self.predict(X), y)


class LinearSVR(_LinearSVMBase):
    """Linear support vector regression.

        loss            'epsilon_insensitive' (default) or
                        'squared_epsilon_insensitive'
        epsilon         honored, >= 0 (default 0.0, the reference's)
        penalty         'l1' (the reference's default, OWL-QN) or 'l2'.
                        scikit-learn's LinearSVR is l2 only; pass
                        penalty='l2' for its objective
        C               honored, > 0
        fit_intercept, tol, max_iter, linesearch_max_iter, lbfgs_memory
                        honored
        penalized_intercept   refused by name
        sample_weight   refused

    OUTPUTS: `coef_` (n_features,) float32, `intercept_` a float,
    `n_iter_`, `objective_`, `retcode_`.
    """

    _LOSSES = {"epsilon_insensitive": _QN_LOSS_SVR_L1,
               "squared_epsilon_insensitive": _QN_LOSS_SVR_L2}

    def __init__(self, *, epsilon=0.0, penalty="l1", loss="epsilon_insensitive",
                 C=1.0, fit_intercept=True, penalized_intercept=False,
                 tol=1e-4, max_iter=1000, linesearch_max_iter=100,
                 lbfgs_memory=5):
        if not (0.0 <= epsilon < float("inf")):
            raise ValueError(
                f"mojolearn LinearSVR: epsilon must be finite and non-negative, got {epsilon}")
        self._init_common(penalty, loss, C, fit_intercept, penalized_intercept,
                          tol, max_iter, linesearch_max_iter, lbfgs_memory)
        self.epsilon = epsilon

    def fit(self, X, y, sample_weight=None):
        self._refuse_sample_weight(sample_weight)
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
        if not hasattr(self, "_w"):
            raise ValueError("mojolearn LinearSVR: call fit before predict")
        return _qn_scores(self, X, self._w)

    def score(self, X, y):
        """R^2 on the host, as `LinearRegression.score`."""
        return _r2_host(self.predict(X), y)
