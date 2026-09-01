# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU dimensionality reduction."""

import numpy as np

from . import _mojolearn_estimators
from ._mode import NumericModeMixin
from ._arrays import _addr, _addr_ro, as_f32_c


def _component_count(n_components, shape):
    value = min(shape) if n_components is None else int(n_components)
    if value < 1 or value > shape[1]:
        raise ValueError(
            f"mojolearn n_components must be in [1, {shape[1]}], got {value}"
        )
    return value


class PCA(NumericModeMixin):
    """Covariance-eigendecomposition PCA on the GPU: cuML's `pcaFit` route
    (covariance, then an eigensolver) with the eigensolver being cuML's
    JACOBI arm (`svd_solver='jacobi'`, cuSOLVER syevj there, a device
    Jacobi here). cuML's own 'auto' reaches the divide-and-conquer `eigDC`
    (syevd), which is NOT ported (decomposition/NOT_IMPLEMENTED.tsv); this class
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
        whiten        honored   the transform's columns come out with unit
                                variance and inverse_transform undoes it.
                                cuML's rescale of a COPY of the components
                                (pca.cuh:292-302 and :232-243), in all three
                                numeric modes, pinned at the seam through
                                identical_mul / identical_div / ftz.
                                DEVIATIONS 580-585 in
                                decomposition/impl/linalg/detail/pca.mojo;
                                gated by decomposition/checks/pca_check.mojo
                                (unit variance, round trip, row-subset
                                agreement, and the planted-bit edges)
        svd_solver    honored   'auto', 'covariance_eigh' and 'jacobi',
                                which all name THE ONE ARM THIS CLASS RUNS:
                                the covariance plus the device Jacobi.
                                'jacobi' is cuML's own name for it
                                (pca.pyx:398 maps it to COV_EIG_JACOBI).
                                'auto' is accepted with the substitution
                                stated: cuML maps 'auto' to COV_EIG_DQ and
                                we run the Jacobi arm
        svd_solver    refused   'full', 'randomized' and 'arpack' are NOT
                                YET IMPLEMENTED, which is a different
                                statement from the one this docstring used
                                to make. They are not out of reach and they
                                are not blocked by a closed vendor library
                                being closed; each needs a portable
                                tall-skinny QR this tree does not have yet.
                                decomposition/NOT_IMPLEMENTED.tsv carries
                                the route and what is missing. Accepting
                                the name and running the covariance arm
                                would be a silent substitution and is
                                refused for that reason, not for the
                                algorithm's sake
        tol, iterated_power, random_state, n_oversamples -- not on this
                                surface: the device Jacobi runs RAFT's own
                                defaults (tol 1e-7, 15 sweeps; see
                                decomposition/impl/linalg/detail/pca.mojo)
        n_features > 128 refused UNDER NUMERIC_IDENTICAL ONLY: the pinned
                                split-K Gram kernel's capacity
                                (IDENTITY_PATHS row 27; the refusal names
                                the shape). FAST runs any width through
                                the vendor matmul.

    Incremental fitting and sparse input are not implemented.
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    #: The three names for the ONE arm this class runs. 'covariance_eigh' is
    #: scikit-learn's name for it, 'jacobi' is cuML's (pca.pyx:398 ->
    #: Solver.COV_EIG_JACOBI), and 'auto' is accepted with the substitution
    #: stated in the class docstring: cuML's 'auto' is COV_EIG_DQ, ours is
    #: the Jacobi arm.
    _SOLVERS = ("auto", "covariance_eigh", "jacobi")

    def __init__(self, n_components=None, *, whiten=False, svd_solver="auto"):
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver

    def _whiten_binding(self):
        """The binding, checked for the whitened pair.

        `pca_whiten_transform` / `pca_whiten_inverse_transform` are ADDITIVE
        exports beside the unwhitened ones, so an older `_mojolearn_estimators`
        that predates them is a build that simply does not carry the pair.
        That is a build state, not a user error, and it says so rather than
        failing with an arity message from the extension.
        """
        binding = self._bind("_mojolearn_estimators")
        if not hasattr(binding, "pca_whiten_transform"):
            raise NotImplementedError(
                "mojolearn PCA: whiten=True needs the whitened transform pair "
                "and this build of _mojolearn_estimators does not export it. "
                "The kernel, the host surfaces and the gates are present "
                "(decomposition/impl/linalg/detail/pca.mojo::whiten_components, "
                "decomposition/estimator.mojo::pca_whiten_transform_host and "
                "pca_whiten_inverse_transform_host, "
                "decomposition/checks/pca_check.mojo::check_whiten_*); what is "
                "missing is the two def_function lines in "
                "bindings/_mojolearn_estimators.mojo and a rebuild via "
                "bindings/build_estimators.sh"
            )
        return binding

    def fit(self, X, y=None):
        if self.svd_solver not in self._SOLVERS:
            raise NotImplementedError(
                f"mojolearn PCA: svd_solver={self.svd_solver!r} is not "
                "implemented. This class runs one arm, the covariance plus the "
                "device Jacobi, named 'auto' / 'covariance_eigh' / 'jacobi'. "
                "'full', 'randomized' and 'arpack' are a dense SVD of the data "
                "matrix rather than of its covariance; the route is portable "
                "and is written down in decomposition/NOT_IMPLEMENTED.tsv, and "
                "what it waits on is a tall-skinny QR this tree does not have "
                "yet. Accepting the name and running the covariance arm would "
                "be a silent substitution, which is why this raises instead"
            )
        if self.whiten:
            self._whiten_binding()
        x, self.input_copied_ = as_f32_c(X, "X")
        if x.shape[0] < 2 or x.shape[1] < 2:
            raise ValueError("mojolearn PCA requires at least 2 rows and 2 features")
        nc = _component_count(self.n_components, x.shape)
        self.components_ = np.empty((nc, x.shape[1]), dtype=np.float32)
        self.mean_ = np.empty(x.shape[1], dtype=np.float32)
        self.explained_variance_ = np.empty(nc, dtype=np.float32)
        self.explained_variance_ratio_ = np.empty(nc, dtype=np.float32)
        self.singular_values_ = np.empty(nc, dtype=np.float32)
        self.noise_variance_ = float(self._bind("_mojolearn_estimators").pca_fit(
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
        if self.whiten:
            # `n_samples_` and NOT `x.shape[0]`. DEVIATION 580: cuML's dense
            # path scales by the row count of THIS CALL (pca.pyx:770), so
            # their `transform(X[:100])` disagrees with `transform(X)[:100]`.
            # Their own sparse path uses the fit's count and so does
            # scikit-learn, and so does this.
            self._whiten_binding().pca_whiten_transform(
                _addr_ro(x), _addr_ro(self.mean_), _addr_ro(self.components_),
                _addr_ro(self.singular_values_), _addr(out),
                [x.shape[0], x.shape[1], self.n_components_, self.n_samples_],
            )
            return out
        self._bind("_mojolearn_estimators").pca_transform(
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
        if self.whiten:
            self._whiten_binding().pca_whiten_inverse_transform(
                _addr_ro(z), _addr_ro(self.components_),
                _addr_ro(self.singular_values_), _addr_ro(self.mean_),
                _addr(out),
                [z.shape[0], self.n_features_in_, self.n_components_,
                 self.n_samples_],
            )
            return out
        self._bind("_mojolearn_estimators").inverse_transform(
            _addr_ro(z), _addr_ro(self.components_), _addr_ro(self.mean_),
            _addr(out), [z.shape[0], self.n_features_in_, self.n_components_, 1],
        )
        return out


class TruncatedSVD(NumericModeMixin):
    """Uncentered truncated SVD on the GPU, mirroring cuML's `tsvdFit`
    (Gram matrix + the same device Jacobi eigensolver PCA uses).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        n_components  honored   1..n_features, the eigenpairs kept
        algorithm     honored   'covariance_eigh' and 'jacobi', both naming
                                the one arm that runs: the Gram matrix plus
                                the device Jacobi, which is what cuML's
                                `tsvdFit` takes
        algorithm     refused   'arpack' and 'randomized' are NOT YET
                                IMPLEMENTED. 'randomized' is reachable from
                                here and the route is written down in
                                decomposition/NOT_IMPLEMENTED.tsv: RAFT's own
                                rsvd has a variant that needs no dense SVD at
                                all, only a Gaussian sketch, a QR, and the
                                eigendecomposition this class already ships
                                (raft/linalg/detail/rsvd.cuh:364 reaches
                                eigJacobi). The one primitive missing is a
                                portable tall-skinny QR. NOTE the default is
                                NOT scikit-learn's 'randomized', because
                                accepting that name for a different algorithm
                                would be a silent substitution
        n_iter, random_state, tol -- not on this surface (they belong to the
                                unimplemented solvers)
        n_features > 128 refused UNDER NUMERIC_IDENTICAL ONLY, as for PCA
                                (IDENTITY_PATHS row 27)

    Components and singular values are exposed. `explained_variance_` is
    omitted because the fit kernel does not compute cuML's
    transformed-data definition of that quantity (tsvd.cuh's
    `explained_var` comes from the transformed matrix's column variances,
    a second pass this port does not make).
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    #: The two names for the ONE arm this class runs; see `PCA._SOLVERS`.
    _ALGORITHMS = ("covariance_eigh", "jacobi")

    def __init__(self, n_components=2, *, algorithm="covariance_eigh"):
        self.n_components = n_components
        self.algorithm = algorithm

    def fit(self, X, y=None):
        if self.algorithm not in self._ALGORITHMS:
            raise NotImplementedError(
                f"mojolearn TruncatedSVD: algorithm={self.algorithm!r} is not "
                "implemented. This class runs one arm, the Gram matrix plus "
                "the device Jacobi that cuML's tsvdFit takes, named "
                "'covariance_eigh' or 'jacobi'. 'randomized' and 'arpack' are "
                "different algorithms and the route to 'randomized' is "
                "written down in decomposition/NOT_IMPLEMENTED.tsv; what it "
                "waits on is a portable tall-skinny QR, not a closed vendor "
                "library. Accepting the name and running this arm would be a "
                "silent substitution, which is why this raises instead"
            )
        x, self.input_copied_ = as_f32_c(X, "X")
        if x.shape[0] < 2 or x.shape[1] < 2:
            raise ValueError("mojolearn TruncatedSVD requires at least 2 rows and 2 features")
        nc = _component_count(self.n_components, x.shape)
        self.components_ = np.empty((nc, x.shape[1]), dtype=np.float32)
        self.singular_values_ = np.empty(nc, dtype=np.float32)
        self._bind("_mojolearn_estimators").tsvd_fit(
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
        self._bind("_mojolearn_estimators").tsvd_transform(
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
        self._bind("_mojolearn_estimators").inverse_transform(
            _addr_ro(z), _addr_ro(self.components_), _addr_ro(self.components_),
            _addr(out), [z.shape[0], self.n_features_in_, self.n_components_, 0],
        )
        return out
