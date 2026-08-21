"""GPU linear models."""

import numpy as np

from . import _mojolearn_estimators
from ._arrays import _addr, _addr_ro, as_f32_c


class LinearRegression:
    """Ordinary least squares through normal equations on the GPU.

    This fast tall-and-narrow solver forms ``X.T @ X``, which squares the
    condition number. It is less numerically robust than an SVD-based least
    squares solver and should not be used for badly conditioned designs.
    """

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("mojolearn LinearRegression has no sample_weight yet")
        x, self.input_copied_ = as_f32_c(X, "X")
        target = np.asarray(y)
        if target.ndim != 1:
            raise ValueError("mojolearn LinearRegression currently requires one target")
        if target.shape[0] != x.shape[0]:
            raise ValueError("mojolearn LinearRegression X and y lengths differ")
        target = np.ascontiguousarray(target, dtype=np.float32)
        if self.fit_intercept:
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
        self.intercept_ = float(self._y_mean - self._x_mean @ self.coef_)
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
