"""GPU dimensionality reduction."""

import numpy as np

from . import _mojolearn_estimators
from ._arrays import _addr, _addr_ro, as_f32_c


def _component_count(n_components, shape):
    value = min(shape) if n_components is None else int(n_components)
    if value < 1 or value > shape[1]:
        raise ValueError(
            f"mojolearn n_components must be in [1, {shape[1]}], got {value}"
        )
    return value


class PCA:
    """Covariance-eigendecomposition PCA on the GPU.

    The preliminary API supports full-component fitting and transformation.
    Whitening, randomized solvers, incremental fitting, and sparse input are
    not implemented.
    """

    def __init__(self, n_components=None, *, whiten=False, svd_solver="auto"):
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver

    def fit(self, X, y=None):
        if self.whiten:
            raise NotImplementedError("mojolearn PCA does not yet support whiten=True")
        if self.svd_solver not in ("auto", "covariance_eigh"):
            raise ValueError("mojolearn PCA supports only covariance_eigh")
        x, self.input_copied_ = as_f32_c(X, "X")
        if x.shape[0] < 2 or x.shape[1] < 2:
            raise ValueError("mojolearn PCA requires at least 2 rows and 2 features")
        nc = _component_count(self.n_components, x.shape)
        self.components_ = np.empty((nc, x.shape[1]), dtype=np.float32)
        self.mean_ = np.empty(x.shape[1], dtype=np.float32)
        self.explained_variance_ = np.empty(nc, dtype=np.float32)
        self.explained_variance_ratio_ = np.empty(nc, dtype=np.float32)
        self.singular_values_ = np.empty(nc, dtype=np.float32)
        self.noise_variance_ = float(_mojolearn_estimators.pca_fit(
            _addr_ro(x), _addr(self.components_), _addr(self.mean_),
            _addr(self.explained_variance_), _addr(self.explained_variance_ratio_),
            _addr(self.singular_values_), [x.shape[0], x.shape[1], nc],
        ))
        self.n_components_ = nc
        self.n_features_in_ = x.shape[1]
        self.n_samples_ = x.shape[0]
        return self

    def transform(self, X):
        if not hasattr(self, "components_"):
            raise ValueError("mojolearn PCA: call fit before transform")
        x, _ = as_f32_c(X, "X")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("mojolearn PCA feature count differs from fit")
        out = np.empty((x.shape[0], self.n_components_), dtype=np.float32)
        _mojolearn_estimators.pca_transform(
            _addr_ro(x), _addr_ro(self.mean_), _addr_ro(self.components_),
            _addr(out), [x.shape[0], x.shape[1], self.n_components_],
        )
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

    def inverse_transform(self, X):
        z, _ = as_f32_c(X, "X")
        if z.shape[1] != self.n_components_:
            raise ValueError("mojolearn PCA component count differs from fit")
        out = np.empty((z.shape[0], self.n_features_in_), dtype=np.float32)
        _mojolearn_estimators.inverse_transform(
            _addr_ro(z), _addr_ro(self.components_), _addr_ro(self.mean_),
            _addr(out), [z.shape[0], self.n_features_in_, self.n_components_, 1],
        )
        return out


class TruncatedSVD:
    """Uncentered truncated SVD on the GPU.

    Components and singular values are exposed. Explained variance is omitted
    because the current fit kernel does not compute cuML's transformed-data
    definition of that quantity.
    """

    def __init__(self, n_components=2, *, algorithm="covariance_eigh"):
        self.n_components = n_components
        self.algorithm = algorithm

    def fit(self, X, y=None):
        if self.algorithm != "covariance_eigh":
            raise ValueError("mojolearn TruncatedSVD supports covariance_eigh only")
        x, self.input_copied_ = as_f32_c(X, "X")
        if x.shape[0] < 2 or x.shape[1] < 2:
            raise ValueError("mojolearn TruncatedSVD requires at least 2 rows and 2 features")
        nc = _component_count(self.n_components, x.shape)
        self.components_ = np.empty((nc, x.shape[1]), dtype=np.float32)
        self.singular_values_ = np.empty(nc, dtype=np.float32)
        _mojolearn_estimators.tsvd_fit(
            _addr_ro(x), _addr(self.components_), _addr(self.singular_values_),
            [x.shape[0], x.shape[1], nc],
        )
        self.n_components_ = nc
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "components_"):
            raise ValueError("mojolearn TruncatedSVD: call fit before transform")
        x, _ = as_f32_c(X, "X")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("mojolearn TruncatedSVD feature count differs from fit")
        out = np.empty((x.shape[0], self.n_components_), dtype=np.float32)
        _mojolearn_estimators.tsvd_transform(
            _addr_ro(x), _addr_ro(self.components_), _addr(out),
            [x.shape[0], x.shape[1], self.n_components_],
        )
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

    def inverse_transform(self, X):
        z, _ = as_f32_c(X, "X")
        if z.shape[1] != self.n_components_:
            raise ValueError("mojolearn TruncatedSVD component count differs from fit")
        out = np.empty((z.shape[0], self.n_features_in_), dtype=np.float32)
        # The mean pointer is unused when add_mean is false; components is a
        # valid non-null float32 address for the boundary contract.
        _mojolearn_estimators.inverse_transform(
            _addr_ro(z), _addr_ro(self.components_), _addr_ro(self.components_),
            _addr(out), [z.shape[0], self.n_features_in_, self.n_components_, 0],
        )
        return out
