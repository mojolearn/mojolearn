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
    """Covariance-eigendecomposition PCA on the GPU: cuML's `pcaFit` route
    (covariance, then an eigensolver) with the eigensolver being cuML's
    JACOBI arm (`svd_solver='jacobi'`, cuSOLVER syevj there, a device
    Jacobi here). cuML's own 'auto' reaches the divide-and-conquer `eigDC`
    (syevd), which is NOT ported (decomposition/UNPORTED.tsv); this class
    accepts 'auto' and runs the Jacobi arm, and says so here.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        n_components  honored   1..n_features; None means min(n_samples,
                                n_features). The eigendecomposition is of
                                the FULL covariance on every setting; the
                                count selects how many eigenpairs are kept
                                and sets `noise_variance_` (the mean of the
                                discarded eigenvalues, cuML's
                                truncCompExpVars)
        svd_solver    refused   anything but 'auto' / 'covariance_eigh'
                                (both name the one ported arm, the device
                                Jacobi); 'full' / 'randomized' / 'arpack'
                                are different algorithms that are not
                                ported (decomposition/UNPORTED.tsv)
        whiten        refused   True is not ported: it needs the
                                `explained_var` rescale of the components
                                on transform and its inverse on
                                inverse_transform (cuml pca.cuh's
                                whitening arms); refused by name rather
                                than silently unwhitened
        tol, iterated_power, random_state, n_oversamples -- not on this
                                surface: the device Jacobi runs RAFT's own
                                defaults (tol 1e-7, 15 sweeps; see
                                decomposition/ported/linalg/detail/pca.mojo)
        n_features > 128 refused UNDER NUMERIC_IDENTICAL ONLY: the pinned
                                split-K Gram kernel's capacity
                                (IDENTITY_PATHS row 27; the refusal names
                                the shape). FAST runs any width through
                                the vendor matmul.

    Incremental fitting and sparse input are not implemented.
    """

    def __init__(self, n_components=None, *, whiten=False, svd_solver="auto"):
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver

    def fit(self, X, y=None):
        if self.whiten:
            raise NotImplementedError(
                "mojolearn PCA: whiten=True is refused; the whitening "
                "rescale of the components (and its inverse on "
                "inverse_transform) is not ported, and an unwhitened "
                "transform returned for whiten=True would be a wrong answer"
            )
        if self.svd_solver not in ("auto", "covariance_eigh"):
            raise ValueError(
                f"mojolearn PCA: svd_solver={self.svd_solver!r} is refused; "
                "only 'auto' / 'covariance_eigh' (cuML's covariance + device "
                "Jacobi arm) is ported; 'full', 'randomized' and 'arpack' are "
                "different algorithms (decomposition/UNPORTED.tsv)"
            )
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
    """Uncentered truncated SVD on the GPU, mirroring cuML's `tsvdFit`
    (Gram matrix + the same device Jacobi eigensolver PCA uses).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        n_components  honored   1..n_features, the eigenpairs kept
        algorithm     refused   anything but 'covariance_eigh' (the Gram +
                                Jacobi arm). scikit-learn's 'arpack' and
                                'randomized' are different algorithms that
                                are not ported. NOTE the default is NOT
                                scikit-learn's 'randomized', because
                                accepting that name for a different
                                algorithm would be a silent substitution
        n_iter, random_state, tol -- not on this surface (they belong to the
                                refused solvers)
        n_features > 128 refused UNDER NUMERIC_IDENTICAL ONLY, as for PCA
                                (IDENTITY_PATHS row 27)

    Components and singular values are exposed. `explained_variance_` is
    omitted because the fit kernel does not compute cuML's
    transformed-data definition of that quantity (tsvd.cuh's
    `explained_var` comes from the transformed matrix's column variances,
    a second pass this port does not make).
    """

    def __init__(self, n_components=2, *, algorithm="covariance_eigh"):
        self.n_components = n_components
        self.algorithm = algorithm

    def fit(self, X, y=None):
        if self.algorithm != "covariance_eigh":
            raise ValueError(
                f"mojolearn TruncatedSVD: algorithm={self.algorithm!r} is "
                "refused; only 'covariance_eigh' (the Gram + device Jacobi "
                "arm cuML's tsvdFit takes) is ported; 'arpack' and "
                "'randomized' are different algorithms"
            )
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
