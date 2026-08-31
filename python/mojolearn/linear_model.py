# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU linear models, mirroring cuML's `LinearRegression(algorithm='eig')`,
`Ridge(solver='eig')` and `LogisticRegression(solver='qn')`."""

import math

import numpy as np

from . import _mojolearn_estimators
from ._mode import NumericModeMixin
from ._arrays import _addr, _addr_ro, as_f32_c


class LinearRegression(NumericModeMixin):
    """Ordinary least squares through normal equations on the GPU.

    This is cuML's `algorithm='eig'` arm (`lstsqEig`, RAFT), which forms
    ``X.T @ X`` and so squares the condition number. It is less robust than
    an SVD-based solver and should not be used for badly conditioned
    designs; cuML's own SVD arms are not ported (glm/NOT_IMPLEMENTED.tsv).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        fit_intercept   honored   True (the default) centers X and y ON THE
                                  HOST and the device solves the centered
                                  system; see THE INTERCEPT IS A HOST
                                  REIMPLEMENTATION below. False sends the
                                  raw design to the device, which is the
                                  arm the ported `ols_fit` carries.
        sample_weight   refused   not ported (ols.cuh:99-110, a
                                  sqrt-scaling of both operands and its
                                  inverse; glm/NOT_IMPLEMENTED.tsv)
        n_features == 1 refused   cuML forces its SVD solver for one
                                  column (linear_regression.pyx:390) and
                                  that solver is not ported; the Mojo layer
                                  raises BY NAME (glm/derived/glm/ols.mojo)
        n_features > n  refused   A^T A is singular by construction;
                                  cuML's dispatch switches to SVD
                                  (ols.cuh:112-113), not ported; raised
                                  BY NAME by the same file
        y 2-D           refused   one target only, at this boundary

    THE INTERCEPT IS A HOST REIMPLEMENTATION, NOT A PORT. cuML's Python
    default `fit_intercept=True` wraps the solver in `preProcessData`
    (center X and y on the DEVICE) and `postProcessData` (intercept =
    mean(y) - mu_X . coef; preprocess.cuh:98-176). The ported `ols_fit`
    REFUSES `fit_intercept` by name because those two are not ported
    (glm/derived/glm/ols.mojo). This class therefore does the centering here,
    in numpy: column means and the y mean in float64, subtracted in float32,
    and the intercept as `mean(y) - sum(mu_X * coef)` with `math.fsum`
    (exactly rounded; NO BLAS dot, which would be a platform-dependent host
    reduction -- E2's first finding). The device sees a centered design;
    the arithmetic that reaches it is a function of the inputs alone. The
    mean-centering is mathematically what cuML does, but it runs on the
    host in float64 where theirs runs on the device in float32, so
    `coef_` CAN differ from cuML's in the last bits on ill-conditioned
    data. Named here because a hidden host step is the thing this library
    exists to not have.
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError(
                "mojolearn LinearRegression: sample_weight is not ported "
                "(ols.cuh:99-110; glm/NOT_IMPLEMENTED.tsv)"
            )
        x, self.input_copied_ = as_f32_c(X, "X")
        target = np.asarray(y)
        if target.ndim != 1:
            raise ValueError("mojolearn LinearRegression currently requires one target")
        if target.shape[0] != x.shape[0]:
            raise ValueError("mojolearn LinearRegression X and y lengths differ")
        target = np.ascontiguousarray(target, dtype=np.float32)
        if self.fit_intercept:
            # float64 column means -> float32, then a float32 subtraction.
            # numpy's axis-0 reduction over a C-order matrix is a sequential
            # row accumulation and its 1-D reduction is the pairwise C
            # kernel; neither goes through BLAS or libm. The dot below is
            # the one place a BLAS call would have slipped in, so it is an
            # exactly-rounded fsum instead.
            self._x_mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
            self._y_mean = float(target.mean(dtype=np.float64))
            work_x = np.ascontiguousarray(x - self._x_mean, dtype=np.float32)
            work_y = np.ascontiguousarray(target - self._y_mean, dtype=np.float32)
        else:
            work_x, work_y = x, target
            self._x_mean = np.zeros(x.shape[1], dtype=np.float32)
            self._y_mean = 0.0
        self.coef_ = np.empty(x.shape[1], dtype=np.float32)
        self._bind("_mojolearn_estimators").ols_fit(
            _addr_ro(work_x), _addr_ro(work_y), _addr(self.coef_),
            [x.shape[0], x.shape[1]],
        )
        if self.fit_intercept:
            dot = math.fsum(
                float(a) * float(b) for a, b in zip(self._x_mean, self.coef_)
            )
            self.intercept_ = float(self._y_mean - dot)
        else:
            self.intercept_ = 0.0
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("mojolearn LinearRegression: call fit before predict")
        x, _ = as_f32_c(X, "X")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("mojolearn LinearRegression feature count differs from fit")
        out = np.empty(x.shape[0], dtype=np.float32)
        self._bind("_mojolearn_estimators").ols_predict(
            _addr_ro(x), _addr_ro(self.coef_), _addr(out),
            [x.shape[0], x.shape[1], float(self.intercept_)],
        )
        return out

    def score(self, X, y):
        target = np.asarray(y, dtype=np.float64)
        residual = target - self.predict(X)
        denom = np.sum((target - target.mean()) ** 2)
        return 1.0 - float(np.sum(residual ** 2) / denom) if denom else 0.0


class Ridge(NumericModeMixin):
    """l2-regularized least squares on the GPU, cuML's `solver='eig'` arm.

    Mirrors `cuml/python/cuml/linear_model/ridge.pyx` on top of
    `cuml/cpp/src/glm/ridge.cuh::ridgeFit` (DEVIATION 545; the Mojo port is
    `glm/derived/glm/ridge.mojo` and the design note there is worth reading:
    their `eig` solver is an SVD through the eigendecomposition of `X.T @ X`
    followed by `ridgeSolve`, NOT "OLS with alpha added", and so is ours).
    Solves `min ||y - Xw||^2 + alpha ||w||^2`; the same objective as
    scikit-learn's `Ridge`, and the same minimizer. Forms `X.T @ X`, so the
    condition number is squared (see `LinearRegression`).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        alpha           honored   a non-negative float (ridge.pyx:296);
                                  0.0 is allowed and is least squares
                                  through the ridge program
        fit_intercept   honored   True centers X and y ON THE HOST exactly
                                  as `LinearRegression` does (that class's
                                  docstring: THE INTERCEPT IS A HOST
                                  REIMPLEMENTATION, DEVIATION 517); the
                                  ported `ridge_fit` sees a centered
                                  design with fit_intercept=False, which
                                  is `ridge.cuh:247`'s `intercept = 0` arm
        solver          'eig' only  cuML's 'auto' maps to 'eig'
                                  (ridge.pyx:304); 'svd' (ridgeSVD ->
                                  cuSOLVER gesvd, glm/NOT_IMPLEMENTED.tsv) and
                                  'cd' (cuml/solvers/cd.pyx, a different
                                  solver) are REFUSED by name
        normalize       refused   only reachable with fit_intercept on
                                  their side and is preProcessData's
                                  meanvar arm (preprocess.cuh:76-108),
                                  not ported
        sample_weight   refused   ridge.cuh:197-208 / 220-231, a sqrt-
                                  scaling of both operands and its exact
                                  inverse, not ported
        n_features == 1 refused   ridge.cuh:210 forces ridgeSVD for one
                                  column and the Python layer warns and
                                  switches (ridge.pyx:355); raised BY NAME
                                  by the Mojo layer
        y 2-D           refused   one target only, at this boundary
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    def __init__(self, *, alpha=1.0, solver="auto", fit_intercept=True,
                 normalize=False):
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if solver not in ("auto", "eig", "svd", "cd"):
            raise TypeError(f"solver {solver!r} is not supported")
        if solver in ("svd", "cd"):
            raise NotImplementedError(
                f"mojolearn Ridge: solver={solver!r} is not ported "
                "(ridgeSVD is raft::linalg::svdQR -> cuSOLVER gesvd; 'cd' "
                "is cuml/solvers/cd.pyx); solver='eig' (cuML's 'auto') is "
                "the ported arm. See glm/NOT_IMPLEMENTED.tsv"
            )
        if normalize:
            raise NotImplementedError(
                "mojolearn Ridge: normalize is not ported (preprocess.cuh:"
                "76-108, the meanvar arm of preProcessData; glm/NOT_IMPLEMENTED.tsv)"
            )
        self.alpha = alpha
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.normalize = normalize

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError(
                "mojolearn Ridge: sample_weight is not ported "
                "(ridge.cuh:197-208; glm/NOT_IMPLEMENTED.tsv)"
            )
        self.solver_ = "eig"
        x, self.input_copied_ = as_f32_c(X, "X")
        target = np.asarray(y)
        if target.ndim != 1:
            raise ValueError("mojolearn Ridge currently requires one target")
        if target.shape[0] != x.shape[0]:
            raise ValueError("mojolearn Ridge X and y lengths differ")
        target = np.ascontiguousarray(target, dtype=np.float32)
        if self.fit_intercept:
            # The same host centering as LinearRegression, for the same
            # reasons; read that class's fit.
            self._x_mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
            self._y_mean = float(target.mean(dtype=np.float64))
            work_x = np.ascontiguousarray(x - self._x_mean, dtype=np.float32)
            work_y = np.ascontiguousarray(target - self._y_mean, dtype=np.float32)
        else:
            work_x, work_y = x, target
            self._x_mean = np.zeros(x.shape[1], dtype=np.float32)
            self._y_mean = 0.0
        self.coef_ = np.empty(x.shape[1], dtype=np.float32)
        self._bind("_mojolearn_estimators").ridge_fit(
            _addr_ro(work_x), _addr_ro(work_y), _addr(self.coef_),
            [x.shape[0], x.shape[1], float(self.alpha)],
        )
        if self.fit_intercept:
            dot = math.fsum(
                float(a) * float(b) for a, b in zip(self._x_mean, self.coef_)
            )
            self.intercept_ = float(self._y_mean - dot)
        else:
            self.intercept_ = 0.0
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("mojolearn Ridge: call fit before predict")
        x, _ = as_f32_c(X, "X")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("mojolearn Ridge feature count differs from fit")
        out = np.empty(x.shape[0], dtype=np.float32)
        # The same gemv + intercept epilogue OLS predicts with
        # (`gemmPredict` upstream serves both, `base.pyx:134`).
        self._bind("_mojolearn_estimators").ols_predict(
            _addr_ro(x), _addr_ro(self.coef_), _addr(out),
            [x.shape[0], x.shape[1], float(self.intercept_)],
        )
        return out

    def score(self, X, y):
        target = np.asarray(y, dtype=np.float64)
        residual = target - self.predict(X)
        denom = np.sum((target - target.mean()) ** 2)
        return 1.0 - float(np.sum(residual ** 2) / denom) if denom else 0.0


# cuML's `qn_params.loss` ids (cuml/linear_model/qn.h); the Python door maps
# 'sigmoid' to QN_LOSS_LOGISTIC and 'softmax' to QN_LOSS_SOFTMAX.
_QN_OPT_RETCODE = {0: "OPT_SUCCESS", 1: "OPT_NUMERIC_ERROR",
                   2: "OPT_LS_FAILED", 3: "OPT_MAX_ITERS_REACHED",
                   4: "OPT_INVALID_ARGS"}


class LogisticRegression(NumericModeMixin):
    """Binary logistic regression on the GPU, cuML's quasi-Newton solver.

    Mirrors `cuml/python/cuml/linear_model/logistic_regression.py` on top of
    `cuml/python/cuml/solvers/qn.pyx` and `cuml/cpp/src/glm/qn/` (DEVIATIONS
    546-549; the Mojo port is `glm/derived/glm/qn/*.mojo`, one file per
    theirs). The objective is `mean_i logloss_i + (1/(2 C n)) ||w||^2`
    (`penalty_normalized=True`: cuML divides the penalty by n so that its
    minimizer is scikit-learn's `LogisticRegression(C)` minimizer), the
    solver is L-BFGS (`lbfgs_memory=5`) with a backtracking Armijo line
    search, and convergence is `max|grad| <= tol * max(loss, tol)` or an
    objective change below `tol * 0.01 * max(loss, tol)` over 10 iterations
    (`qn_util.cuh::check_convergence`).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        penalty         'l2' / None honored   'l2' is Tikhonov with
                                  l2 = 1/C (logistic_regression.py:640);
                                  None is the unregularized arm (qn.cuh:61)
                        'l1' / 'elasticnet' REFUSED  they select OWL-QN
                                  (`min_owlqn`, qn_solvers.cuh:246), NOT
                                  PORTED; glm/NOT_IMPLEMENTED.tsv
        C               honored   inverse regularization strength, > 0
        tol             honored   grad_tol = tol, change_tol = tol * 0.01,
                                  ftol = change_tol * 0.1 (qn.pyx:504-506,
                                  qn_util.cuh:88-98)
        fit_intercept   honored   the bias is a parameter of the solver
                                  (GLMDims, glm_base.cuh:96); it is NOT
                                  penalized (glm_regularizer.cuh:45); no
                                  host centering here, unlike the linear
                                  models
        max_iter        honored   the L-BFGS iteration cap; reaching it is
                                  a WARNING upstream and `n_iter_ ==
                                  max_iter` with `retcode_ == 3` here
        linesearch_max_iter honored  default 50 as theirs
        class_weight    refused   becomes a sample_weight upstream
                                  (logistic_regression.py:400-436), and
                                  sample_weight is not ported
        sample_weight   refused   GLMBase::add_sample_weights and the
                                  weighted getLossAndDZ arm, not ported
        l1_ratio        refused with penalty='elasticnet' (OWL-QN)
        solver          'qn' only the only value cuML accepts either
        > 2 classes     refused   softmax (glm_softmax.cuh) is not ported;
                                  the Mojo layer raises by name
        warm_start      absent    cuML's QN has it, LogisticRegression
                                  does not expose it; w0 = 0 always

    OUTPUTS: `coef_` (1, n_features) float32, `intercept_` (1,) float32,
    `classes_` (the two labels, sorted), `n_iter_` array([k]), plus
    `objective_` (the final value of the objective the solver minimized)
    and `retcode_` (cuML's OPT_RETCODE, 0 = converged). `predict` is
    `classes_[score > 0]`, `predict_proba` is float64 (n, 2) through
    `identical_exp64` on the host (DEVIATION 549; cuML computes it in
    float32 on the device and stores float64).

    IDENTITY: under MOJOLEARN_NUMERIC_MODE=identical every reduction in
    the objective, the gradient and the solver's dot products is a pinned
    fold, `exp`/`log` are the portable pair, and the line search's one
    host multiply-add is an fma -- so the accepted steps, the L-BFGS
    history, the ITERATION COUNT and the coefficients are a function of
    the inputs alone; the card (`qn.iterNNNN.*`, `qn.n_iter`, `qn.coef`)
    records all of them. Under the default FAST mode the reductions are
    the vendor's and the count may differ across GPUs.
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    def __init__(self, *, penalty="l2", tol=1e-4, C=1.0, fit_intercept=True,
                 class_weight=None, max_iter=1000, linesearch_max_iter=50,
                 l1_ratio=None, solver="qn"):
        if penalty not in ("l1", "l2", "elasticnet", None):
            raise ValueError(f"`penalty` {penalty!r} not supported.")
        if solver != "qn":
            raise ValueError(
                "Only quasi-newton `qn` solver is supported, not %s" % solver)
        if penalty in ("l1", "elasticnet"):
            raise NotImplementedError(
                f"mojolearn LogisticRegression: penalty={penalty!r} selects "
                "OWL-QN (min_owlqn, qn_solvers.cuh:246), which is NOT PORTED; "
                "penalty='l2' or None (L-BFGS) are the ported arms. See "
                "glm/NOT_IMPLEMENTED.tsv"
            )
        if class_weight is not None:
            raise NotImplementedError(
                "mojolearn LogisticRegression: class_weight is not ported "
                "(it becomes a sample_weight upstream, logistic_regression.py"
                ":400-436, and sample_weight is not ported; glm/NOT_IMPLEMENTED.tsv)"
            )
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.class_weight = None
        self.max_iter = max_iter
        self.linesearch_max_iter = linesearch_max_iter
        self.l1_ratio = None
        self.solver = solver
        # QN(...) defaults the Python door does not expose
        self.lbfgs_memory = 5
        self.penalty_normalized = True

    def _get_qn_params(self):
        """`_get_qn_params`, logistic_regression.py:631-648."""
        if self.penalty is None:
            return 0.0, 0.0
        return 0.0, 1.0 / self.C

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError(
                "mojolearn LogisticRegression: sample_weight is not ported "
                "(GLMBase::add_sample_weights, glm_base.cuh:115; "
                "glm/NOT_IMPLEMENTED.tsv)"
            )
        x, self.input_copied_ = as_f32_c(X, "X")
        target = np.asarray(y)
        if target.ndim != 1:
            raise ValueError("mojolearn LogisticRegression requires a 1-D y")
        if target.shape[0] != x.shape[0]:
            raise ValueError("mojolearn LogisticRegression X and y lengths differ")
        # LabelEncoder: sorted unique classes -> 0..k-1 (logistic_regression.py:383)
        self.classes_ = np.unique(target)
        n_classes = int(self.classes_.shape[0])
        if n_classes > 2:
            raise NotImplementedError(
                f"mojolearn LogisticRegression: {n_classes} classes need the "
                "softmax loss (glm_softmax.cuh, QN_LOSS_SOFTMAX), which is NOT "
                "PORTED; binary only. See glm/NOT_IMPLEMENTED.tsv"
            )
        if n_classes < 2:
            raise ValueError("mojolearn LogisticRegression: y has one class")
        y_enc = np.ascontiguousarray(
            (target == self.classes_[1]).astype(np.float32))
        l1, l2 = self._get_qn_params()
        n_param = x.shape[1] + (1 if self.fit_intercept else 0)
        w = np.zeros(n_param, dtype=np.float32)
        info = np.zeros(2, dtype=np.float32)
        n_iter = self._bind("_mojolearn_estimators").qn_fit(
            _addr_ro(x), _addr_ro(y_enc), _addr(w), _addr(info),
            [x.shape[0], x.shape[1], n_classes,
             float(l1), float(l2), float(self.tol), float(self.tol * 0.01),
             int(self.max_iter), int(self.linesearch_max_iter),
             int(self.lbfgs_memory), 1 if self.fit_intercept else 0,
             1 if self.penalty_normalized else 0, 0],
        )
        self._w = w
        self.coef_ = w[:x.shape[1]].reshape(1, -1).copy()
        self.intercept_ = (w[x.shape[1]:x.shape[1] + 1].copy()
                           if self.fit_intercept
                           else np.zeros(1, dtype=np.float32))
        self.n_iter_ = np.asarray([int(n_iter)])
        self.objective_ = float(info[0])
        self.retcode_ = int(info[1])
        self.n_features_in_ = x.shape[1]
        return self

    def decision_function(self, X):
        if not hasattr(self, "_w"):
            raise ValueError("mojolearn LogisticRegression: call fit first")
        x, _ = as_f32_c(X, "X")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("mojolearn LogisticRegression feature count differs from fit")
        out = np.empty(x.shape[0], dtype=np.float32)
        self._bind("_mojolearn_estimators").qn_decision_function(
            _addr_ro(x), _addr_ro(self._w), _addr(out),
            [x.shape[0], x.shape[1], 1 if self.fit_intercept else 0],
        )
        return out

    def predict(self, X):
        """`qn_predict`: `z > 0 ? 1 : 0` (qn.cuh:276), mapped to classes_."""
        scores = self.decision_function(X)
        return self.classes_[(scores > 0).astype(np.intp)]

    def predict_proba(self, X):
        scores = self.decision_function(X)
        out = np.empty((scores.shape[0], 2), dtype=np.float64)
        self._bind("_mojolearn_estimators").qn_sigmoid(
            _addr_ro(scores), _addr(out), [scores.shape[0]])
        return out

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
