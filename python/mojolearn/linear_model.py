"""GPU linear models, mirroring cuML's `LinearRegression(algorithm='eig')`."""

import math

import numpy as np

from . import _mojolearn_estimators
from ._arrays import _addr, _addr_ro, as_f32_c


class LinearRegression:
    """Ordinary least squares through normal equations on the GPU.

    This is cuML's `algorithm='eig'` arm (`lstsqEig`, RAFT), which forms
    ``X.T @ X`` and so squares the condition number. It is less robust than
    an SVD-based solver and should not be used for badly conditioned
    designs; cuML's own SVD arms are not ported (glm/UNPORTED.tsv).

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
                                  inverse; glm/UNPORTED.tsv)
        n_features == 1 refused   cuML forces its SVD solver for one
                                  column (linear_regression.pyx:390) and
                                  that solver is not ported; the Mojo layer
                                  raises BY NAME (glm/ported/glm/ols.mojo)
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
    (glm/ported/glm/ols.mojo). This class therefore does the centering here,
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

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError(
                "mojolearn LinearRegression: sample_weight is not ported "
                "(ols.cuh:99-110; glm/UNPORTED.tsv)"
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
        _mojolearn_estimators.ols_fit(
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
        _mojolearn_estimators.ols_predict(
            _addr_ro(x), _addr_ro(self.coef_), _addr(out),
            [x.shape[0], x.shape[1], float(self.intercept_)],
        )
        return out

    def score(self, X, y):
        target = np.asarray(y, dtype=np.float64)
        residual = target - self.predict(X)
        denom = np.sum((target - target.mean()) ** 2)
        return 1.0 - float(np.sum(residual ** 2) / denom) if denom else 0.0
