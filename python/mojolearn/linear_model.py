# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU linear models, mirroring cuML's `LinearRegression(algorithm='eig')`,
`Ridge(solver='eig')` and `LogisticRegression(solver='qn')`."""

import math

import numpy as np

from . import _mojolearn_estimators
from ._mode import NumericModeMixin
from ._arrays import _addr, _addr_ro, as_f32_c



def _check_sample_weight(sample_weight, n_rows, estimator):
    """Validate `sample_weight` and return it as a C-order float32 vector,
    or None when there are no weights.

    cuML validates nothing here: `olsFit` takes a raw pointer and trusts it.
    These three checks are OURS and they are input validation, which is the
    one kind of refusal that is correct rather than a gap -- a length
    mismatch would read past the end of the array on the device, and a
    negative weight makes `sqrt(w)` a NaN that silently poisons the whole
    fit. Non-finite weights are refused for the same reason.
    """
    if sample_weight is None:
        return None
    w = np.ascontiguousarray(np.asarray(sample_weight), dtype=np.float32)
    if w.ndim != 1:
        raise ValueError(
            f"mojolearn {estimator}: sample_weight must be 1-D, got "
            f"{w.ndim} dimensions"
        )
    if w.shape[0] != n_rows:
        raise ValueError(
            f"mojolearn {estimator}: sample_weight has {w.shape[0]} entries "
            f"but X has {n_rows} rows"
        )
    if not np.all(np.isfinite(w)):
        raise ValueError(
            f"mojolearn {estimator}: sample_weight contains a non-finite "
            "value; sqrt of it would poison every row of the fit"
        )
    if np.any(w < 0.0):
        raise ValueError(
            f"mojolearn {estimator}: sample_weight has a negative entry; "
            "a weighted least squares scales rows by sqrt(w) (ols.cuh:100) "
            "and sqrt of a negative is a NaN"
        )
    return w


def _column_means(x, weights):
    """Column means in float64, narrowed to float32 -- weighted when
    `weights` is not None.

    Unweighted this is `x.mean(axis=0, dtype=np.float64)`, the reduction
    this file has always used (a sequential row accumulation in numpy's C
    kernel, not BLAS, not libm). Weighted it is cuML's
    `raft::stats::weightedMean`: `sum_i w_i x_ij / sum_i w_i`, one division
    by the same scalar for every column (`raft/stats/detail/weighted_mean
    .cuh:49-64`). Note that theirs divides by the SUM OF THE WEIGHTS and not
    by the row count, so a uniform weight of 2 leaves the mean unchanged --
    which is the property that makes duplicating a row equal doubling its
    weight, and is what the gate checks.
    """
    if weights is None:
        return x.mean(axis=0, dtype=np.float64).astype(np.float32)
    w64 = weights.astype(np.float64)
    total = float(w64.sum())
    if total <= 0.0:
        raise ValueError(
            "mojolearn: sample_weight sums to zero, so the weighted mean "
            "cuML's preProcessData forms is a division by zero"
        )
    return ((x.astype(np.float64) * w64[:, None]).sum(axis=0) / total).astype(
        np.float32
    )


def _vector_mean(v, weights):
    """The scalar mean of the target, weighted when `weights` is not None.
    The 1-D half of `_column_means`; see that docstring."""
    if weights is None:
        return float(v.mean(dtype=np.float64))
    w64 = weights.astype(np.float64)
    total = float(w64.sum())
    if total <= 0.0:
        raise ValueError(
            "mojolearn: sample_weight sums to zero, so the weighted mean "
            "cuML's preProcessData forms is a division by zero"
        )
    return float((v.astype(np.float64) * w64).sum() / total)


class LinearRegression(NumericModeMixin):
    """Ordinary least squares through normal equations on the GPU.

    This is cuML's `algorithm='eig'` arm (`lstsqEig`, RAFT), which forms
    ``X.T @ X`` and so squares the condition number. It is less robust than
    an SVD-based solver and should not be used for badly conditioned
    designs; cuML's SVD and QR solvers (`lstsqSvdJacobi`, `lstsqSvdQR`,
    `lstsqQR`) are not written here (glm/NOT_IMPLEMENTED.tsv). A design with
    more features than samples takes a second Gram route instead of an SVD;
    see the `n_features > n` row below.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        fit_intercept   honored   True (the default) centers X and y ON THE
                                  HOST and the device solves the centered
                                  system; see THE INTERCEPT IS A HOST
                                  REIMPLEMENTATION below. False sends the
                                  raw design to the device, which is the
                                  arm the ported `ols_fit` carries.
        sample_weight   honored   a weighted least squares IS an unweighted
                                  one on rows rescaled by sqrt(w), which is
                                  exactly what cuML does (ols.cuh:99-110)
                                  before it dispatches; see SAMPLE WEIGHTS
                                  below
        n_features == 1 honored   a single column is scalar least squares
                                  and the Gram is 1 x 1 with condition
                                  number 1; cuML switches to its SVD solver
                                  here because THEIR eigensolver does not
                                  take one column (linear_regression.pyx:
                                  390-394), a limitation the device Jacobi
                                  does not share (DEVIATION 551)
        n_features > n  honored   the minimum-norm solution
                                  w = X.T (X X.T)^+ y, through the Gram of
                                  the ROWS, which is n x n and nonsingular
                                  at full row rank (DEVIATION 550,
                                  glm/impl/linalg/detail/lstsq_min_norm.mojo).
                                  It squares the condition number exactly as
                                  the tall route does, so it is no more
                                  accurate than this class already is and no
                                  less; a true SVD or LQ route would be
                                  better and is not written
        y 2-D           refused   one target only, at this boundary

    SAMPLE WEIGHTS ARE A HOST RESCALE HERE, AND THAT IS A DEVIATION WITH A
    REASON. `olsFit` takes `sqrt` of the weights, multiplies row `i` of X
    and entry `i` of y by it, solves, and undoes the scaling
    (ols.cuh:99-110, 129-141). Both halves of the scaling are ported and run
    ON THE DEVICE in `glm/impl/glm/ols.mojo::ols_fit_weighted`; what is not
    yet in place is a BINDING that can hand a weight pointer across
    (`bindings/_mojolearn_estimators.mojo::ols_fit_binding` takes a fixed
    `params` of length 2). Until it is, this class applies the same two
    operations in numpy and calls the unweighted entry.

    That is defensible where a host reimplementation usually is not, and the
    reason is arithmetic rather than convenience: `np.sqrt` on float32 is
    the IEEE correctly-rounded square root, the row multiply is one
    float32 rounding, and both are the same operations the device kernels
    perform. The two routes are therefore expected to agree BIT FOR BIT
    except on denormals, where the device flushes and numpy does not, and
    `check_ols_sample_weight_host_rescale_matches_device` in
    `glm/checks/ols_check.mojo` is the gate on that rather than this
    paragraph. `sample_weight` with `fit_intercept=True` uses WEIGHTED
    column means, which is what cuML does too (`raft::stats::weightedMean`,
    preprocess.cuh:95-97, 110-112).

    THE INTERCEPT IS A HOST REIMPLEMENTATION, NOT A PORT. cuML's Python
    default `fit_intercept=True` wraps the solver in `preProcessData`
    (center X and y on the DEVICE) and `postProcessData` (intercept =
    mean(y) - mu_X . coef; preprocess.cuh:98-176). The ported `ols_fit`
    REFUSES `fit_intercept` by name because those two are not ported
    (glm/impl/glm/ols.mojo). This class therefore does the centering here,
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
        x, self.input_copied_ = as_f32_c(X, "X")
        target = np.asarray(y)
        if target.ndim != 1:
            raise ValueError("mojolearn LinearRegression currently requires one target")
        if target.shape[0] != x.shape[0]:
            raise ValueError("mojolearn LinearRegression X and y lengths differ")
        target = np.ascontiguousarray(target, dtype=np.float32)
        weights = _check_sample_weight(sample_weight, x.shape[0],
                                       "LinearRegression")
        if self.fit_intercept:
            # float64 column means -> float32, then a float32 subtraction.
            # numpy's axis-0 reduction over a C-order matrix is a sequential
            # row accumulation and its 1-D reduction is the pairwise C
            # kernel; neither goes through BLAS or libm. The dot below is
            # the one place a BLAS call would have slipped in, so it is an
            # exactly-rounded fsum instead.
            #
            # WITH WEIGHTS THE MEANS ARE WEIGHTED, which is cuML's
            # preProcessData (`raft::stats::weightedMean`, sum(w*x)/sum(w),
            # preprocess.cuh:95-97 and 110-112) and NOT the unweighted mean
            # applied to weighted rows. The two differ, and using the wrong
            # one puts the intercept in the wrong place without moving any
            # coefficient enough to notice.
            self._x_mean = _column_means(x, weights)
            self._y_mean = _vector_mean(target, weights)
            work_x = np.ascontiguousarray(x - self._x_mean, dtype=np.float32)
            work_y = np.ascontiguousarray(target - self._y_mean, dtype=np.float32)
        else:
            work_x, work_y = x, target
            self._x_mean = np.zeros(x.shape[1], dtype=np.float32)
            self._y_mean = 0.0
        if weights is not None:
            # `olsFit`, ols.cuh:99-110, on the host. See SAMPLE WEIGHTS in
            # the class docstring for why this is here and not in the Mojo
            # layer, and for the bit-for-bit claim the gate checks.
            root = np.sqrt(weights, dtype=np.float32)
            work_x = np.ascontiguousarray(work_x * root[:, None],
                                          dtype=np.float32)
            work_y = np.ascontiguousarray(work_y * root, dtype=np.float32)
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
    `glm/impl/glm/ridge.mojo` and the design note there is worth reading:
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
    546-549; the Mojo port is `glm/impl/glm/qn/*.mojo`, one file per
    theirs). The objective is `mean_i logloss_i + (1/(2 C n)) ||w||^2`
    (`penalty_normalized=True`: cuML divides the penalty by n so that its
    minimizer is scikit-learn's `LogisticRegression(C)` minimizer), the
    solver is L-BFGS (`lbfgs_memory=5`) with a backtracking Armijo line
    search -- or OWL-QN with a PROJECTED Armijo line search when the penalty
    has an l1 part (`qn_solvers.cuh:420`, DEVIATION 552) -- and convergence
    is `max|grad| <= tol * max(loss, tol)` or an objective change below
    `tol * 0.01 * max(loss, tol)` over 10 iterations
    (`qn_util.cuh::check_convergence`).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        penalty         all four honored   'l2' is Tikhonov with
                                  l2 = 1/C (logistic_regression.py:640);
                                  None is the unregularized arm (qn.cuh:61);
                                  'l1' and 'elasticnet' set l1 != 0, which
                                  selects OWL-QN (qn_solvers.cuh:420-445),
                                  ported 2026-09-01 as
                                  glm/impl/glm/qn/qn_solvers.mojo::min_owlqn
                                  (DEVIATION 552)
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
        l1_ratio        honored, and REQUIRED with penalty='elasticnet'
                                  (logistic_regression.py:310-316): the
                                  split is l1 = l1_ratio / C,
                                  l2 = (1 - l1_ratio) / C
        solver          'qn' only the only value cuML accepts either
        > 2 classes     refused   softmax (glm_softmax.cuh) is not ported;
                                  the Mojo layer raises by name
        warm_start      absent    cuML's QN has it, LogisticRegression
                                  does not expose it; w0 = 0 always

    THE l1 ARM IS A DIFFERENT SOLVER, NOT A DIFFERENT PENALTY. `|w|` has no
    gradient at zero, which is where an l1 solution sits, so cuML switches
    from L-BFGS to OWL-QN whenever `l1 != 0` (`qn_solvers.cuh:420`) and so
    does this port. OWL-QN keeps the L-BFGS history and replaces three
    things: the objective carries `l1 * ||w||_1` in its VALUE, the direction
    is built from a PSEUDO-gradient, and every step is projected back into
    the orthant it started in. That projection is what produces coefficients
    that are EXACTLY zero, which is the thing an l1 fit is asked for -- and
    it means the identity claim on this arm is about the SPARSITY PATTERN as
    well as about bits, because every branch that zeroes a coefficient is a
    float comparison with a discrete output. The intercept is not
    l1-penalized, exactly as it is not l2-penalized (`pg_limit = D * C`,
    `qn_solvers.cuh:447`).

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
        # `logistic_regression.py:310-316`, copied including the two
        # messages: `l1_ratio` is REQUIRED for elasticnet and IGNORED
        # (set to None) for every other penalty.
        self.l1_ratio = None
        if penalty == "elasticnet":
            if l1_ratio is None:
                raise ValueError(
                    "l1_ratio has to be specified for loss='elasticnet'"
                )
            if l1_ratio < 0.0 or l1_ratio > 1.0:
                raise ValueError(
                    "l1_ratio value has to be between 0.0 and 1.0"
                )
            self.l1_ratio = l1_ratio
        self.solver = solver
        # QN(...) defaults the Python door does not expose
        self.lbfgs_memory = 5
        self.penalty_normalized = True

    def _get_qn_params(self):
        """`_get_qn_params`, logistic_regression.py:632-649, all four arms.

        Returns `(l1_strength, l2_strength)`. `qn_fit` then divides both by
        `n` when `penalty_normalized` (qn.cuh:54-59), so what the solver sees
        is `(1/C)/n`, and `l1 != 0` is what selects OWL-QN there.
        """
        if self.penalty is None:
            return 0.0, 0.0
        if self.penalty == "l1":
            return 1.0 / self.C, 0.0
        if self.penalty == "l2":
            return 0.0, 1.0 / self.C
        strength = 1.0 / self.C
        return self.l1_ratio * strength, (1.0 - self.l1_ratio) * strength

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
