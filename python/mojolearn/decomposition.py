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
    """PCA on the GPU, in TWO ARMS as of 2026-09-01.

    The default is cuML's `pcaFit` route (covariance, then an eigensolver)
    with the eigensolver being cuML's JACOBI arm (`svd_solver='jacobi'`,
    cuSOLVER syevj there, a device Jacobi here). cuML's own 'auto' reaches
    the divide-and-conquer `eigDC` (syevd), which is NOT ported
    (decomposition/NOT_IMPLEMENTED.tsv); this class accepts 'auto' and runs
    the Jacobi arm, and says so here.

    Wide IDENTICAL full fits use transposed QR, at most min(X.shape)
    components, and a 64-sweep/1e-6 Jacobi budget. Tall fits retain their
    existing 15-sweep/1e-7 budget. Nonconvergence still raises.

    `svd_solver='full'` is the second arm and a genuinely different
    algorithm: an R-SVD of the centered data that never forms the
    covariance, so it never squares the condition number. That is
    scikit-learn's meaning for the name and it is why scikit-learn keeps it
    beside 'covariance_eigh'; cuML collapses the two and we deliberately do
    not. THE SENTENCE THAT USED TO OPEN THIS DOCSTRING SAID THIS CLASS WAS A
    COVARIANCE EIGENDECOMPOSITION FULL STOP, AND IT IS DELETED RATHER THAN
    QUALIFIED.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        n_components  honored   1..n_features; None means min(n_samples,
                                n_features). The eigendecomposition is of
                                the FULL covariance on every setting; the
                                count selects how many eigenpairs are kept
                                and sets `noise_variance_` (the mean of the
                                discarded eigenvalues, cuML's
                                truncCompExpVars)
        whiten        honored   nondegenerate fitted score columns have unit
                                sample variance; inverse_transform applies
                                the reverse component scaling.
                                cuML's rescale of a COPY of the components
                                (pca.cuh:292-302 and :232-243), in all three
                                numeric modes, pinned at the seam through
                                identical_mul / identical_div / ftz.
                                DEVIATIONS 580-585 in
                                decomposition/impl/linalg/detail/pca.mojo;
                                gated by decomposition/checks/pca_check.mojo
                                (unit variance, round trip, row-subset
                                agreement, and the planted-bit edges)
        svd_solver    honored   'auto', 'covariance_eigh' and 'jacobi' all
                                name THE COVARIANCE ARM: the covariance plus
                                the device Jacobi. 'jacobi' is cuML's own
                                name for it (pca.pyx:398 maps it to
                                COV_EIG_JACOBI). 'auto' is accepted with the
                                substitution stated: cuML maps 'auto' to
                                COV_EIG_DQ and we run the Jacobi arm
        svd_solver    honored   'full' is a SECOND ARM as of 2026-09-01, and
                                it is a different algorithm rather than a
                                second name: an R-SVD of the CENTERED data,
                                which is what scikit-learn's 'full' means
                                (_pca.py:539-540 sends it to _fit_full).
                                Householder QR of X_c, then a one-sided
                                Jacobi SVD of the small R; the covariance is
                                never formed and the condition number is
                                never squared, which is why scikit-learn
                                keeps this name beside 'covariance_eigh'.
                                DEVIATIONS 586-593 in
                                core/householder_qr.mojo and
                                decomposition/impl/linalg/detail/svd_full.mojo;
                                gated by
                                decomposition/checks/svd_full_check.mojo,
                                whose ill-conditioning gate MEASURES the
                                accuracy claim rather than asserting it.
                                IDENTICAL supports n_samples < n_features
                                through transposed QR and explicit right-basis
                                reconstruction (DEVIATION 593). Other numeric
                                modes retain the wide-shape refusal.
        svd_solver    refused   'randomized' and 'arpack' are NOT YET
                                IMPLEMENTED. The tall-skinny QR that used to
                                block both of them EXISTS now
                                (core/householder_qr.mojo), so the reason has
                                changed and is narrower: 'randomized' still
                                needs the explicit thin Q (RAFT's rsvd calls
                                qrGetQ at rsvd.cuh:198, :218 and :241, and
                                the 'full' arm above deliberately never forms
                                Q), a Gaussian sketch, and a seeded
                                random_state on this surface, which is a
                                reproducibility contract and not just a
                                parameter. 'arpack' is Lanczos and is a third
                                algorithm. decomposition/NOT_IMPLEMENTED.tsv
                                carries both routes. Accepting either name
                                and running an arm that is not it would be a
                                silent substitution and is refused for that
                                reason
        tol, iterated_power, random_state, n_oversamples -- not on this
                                surface: both Jacobis run RAFT's own
                                defaults (tol 1e-7, 15 sweeps; see
                                decomposition/impl/linalg/detail/pca.mojo)
        n_features > 128 refused UNDER NUMERIC_IDENTICAL ONLY, and only on
                                the COVARIANCE arm: the limit is the pinned
                                split-K Gram kernel's capacity
                                (IDENTITY_PATHS row 27; the refusal names
                                the shape). FAST runs any width through
                                the vendor matmul. svd_solver='full' builds
                                no Gram and so does not meet that limit,
                                which is UNRUN on every column and is
                                recorded as owed rather than claimed

    Whitening uses the fitted sample count, never the query batch count.
    For ordinary singular values s, components are scaled by
    sqrt(n_samples_-1)/s, equivalent in real arithmetic to
    1/sqrt(explained_variance_). The order is pinned FP32 multiply, then
    divide, with FTZ seams; it is not a promise to match another spelling's
    rounded bits. Following the existing cuML-derived skip-zero contract,
    s < float32(1e-10) skips the singular division (inverse skips its
    multiplication), retaining the sample-count scale. Degenerate columns
    therefore do not promise unit variance. Nonfinite data/model arrays and
    negative singular values or explained variances are refused on this
    whitening surface. Overflowed output is refused. These additive binding
    exports require a rebuilt extension; source availability is not new
    cross-vendor or installed-wheel qualification.

    Incremental fitting and sparse input are not implemented.
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    #: The three names for the COVARIANCE arm. 'covariance_eigh' is
    #: scikit-learn's name for it, 'jacobi' is cuML's (pca.pyx:398 ->
    #: Solver.COV_EIG_JACOBI), and 'auto' is accepted with the substitution
    #: stated in the class docstring: cuML's 'auto' is COV_EIG_DQ, ours is
    #: the Jacobi arm.
    _COV_SOLVERS = ("auto", "covariance_eigh", "jacobi")

    #: The dense arm. ONE name and not a synonym set, because it is a
    #: different algorithm from the three above and not a second word for
    #: them. cuML maps 'full' onto COV_EIG_DQ, which is why this class does
    #: not: scikit-learn's meaning is the one this signature copies.
    _DENSE_SOLVERS = ("full",)

    _SOLVERS = _COV_SOLVERS + _DENSE_SOLVERS

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
        if not all(callable(getattr(binding, name, None)) for name in
                   ("pca_whiten_transform", "pca_whiten_inverse_transform")):
            raise NotImplementedError(
                "mojolearn PCA: whiten=True needs the whitened transform pair "
                "and this build of _mojolearn_estimators does not export it. "
                "The kernel, the host surfaces and the gates are present "
                "(decomposition/impl/linalg/detail/pca.mojo::whiten_components, "
                "decomposition/estimator.mojo::pca_whiten_transform_host and "
                "pca_whiten_inverse_transform_host, "
                "decomposition/checks/pca_check.mojo::check_whiten_*); what is "
                "needed is a rebuild of this additive ABI via "
                "bindings/build_estimators.sh"
            )
        return binding

    def _validate_whiten_state(self):
        for name, shape in (("components_", (self.n_components_, self.n_features_in_)),
                            ("mean_", (self.n_features_in_,)),
                            ("singular_values_", (self.n_components_,)),
                            ("explained_variance_", (self.n_components_,))):
            value = getattr(self, name, None)
            if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                    or value.shape != shape or not value.flags.c_contiguous
                    or not np.isfinite(value).all()):
                raise ValueError(f"mojolearn PCA whitening requires finite C-order float32 {name} with shape {shape}")
        if (self.n_samples_ < 2 or np.any(self.singular_values_ < 0)
                or np.any(self.explained_variance_ < 0)):
            raise ValueError("mojolearn PCA whitening requires fit rows >=2 and nonnegative singular values/variance")

    def _dense_binding(self):
        """The binding, checked for the dense arm.

        `pca_fit_full` is an ADDITIVE export beside `pca_fit`, so an older
        `_mojolearn_estimators` that predates it is a build that simply does
        not carry the arm. That is a build state, not a user error, and it
        says so rather than failing with an attribute error. Same shape and
        same reason as `_whiten_binding`.
        """
        binding = self._bind("_mojolearn_estimators")
        if not hasattr(binding, "pca_fit_full"):
            raise NotImplementedError(
                "mojolearn PCA: svd_solver='full' needs the dense arm and "
                "this build of _mojolearn_estimators does not export it. The "
                "kernels, the host surface and the gates are present "
                "(core/householder_qr.mojo, "
                "decomposition/impl/linalg/detail/svd_full.mojo, "
                "decomposition/estimator.mojo::pca_fit_full_host, "
                "decomposition/checks/svd_full_check.mojo). The additive "
                "export is in bindings/_mojolearn_estimators.mojo; this "
                "installed extension needs a rebuild via "
                "bindings/build_estimators.sh"
            )
        return binding

    def fit(self, X, y=None):
        if self.svd_solver not in self._SOLVERS:
            raise NotImplementedError(
                f"mojolearn PCA: svd_solver={self.svd_solver!r} is not "
                "implemented. This class runs two arms: the covariance plus "
                "the device Jacobi, named 'auto' / 'covariance_eigh' / "
                "'jacobi', and the R-SVD of the centered data, named 'full'. "
                "'randomized' and 'arpack' are different algorithms again. "
                "The tall-skinny QR that used to block both of them exists "
                "now (core/householder_qr.mojo); what 'randomized' still "
                "waits on is the explicit thin Q, a Gaussian sketch and a "
                "seeded random_state on this surface, and 'arpack' is "
                "Lanczos. decomposition/NOT_IMPLEMENTED.tsv carries both "
                "routes. Accepting the name and running an arm that is not "
                "it would be a silent substitution, which is why this raises "
                "instead"
            )
        if self.whiten:
            self._whiten_binding()
        dense = self.svd_solver in self._DENSE_SOLVERS
        if dense:
            binding = self._dense_binding()
        else:
            binding = self._bind("_mojolearn_estimators")
        x, self.input_copied_ = as_f32_c(X, "X")
        if self.whiten and not np.isfinite(x).all():
            raise ValueError("mojolearn PCA whitening requires finite X")
        if x.shape[0] < 2 or x.shape[1] < 2:
            raise ValueError("mojolearn PCA requires at least 2 rows and 2 features")
        if dense and x.shape[0] < x.shape[1] and self.numeric_mode_used() != "identical":
            # DEVIATION 593, raised HERE as well as in the Mojo validator so
            # legacy modes retain their previous refusal. IDENTICAL uses
            # the transposed QR route and validates k against min(shape).
            raise NotImplementedError(
                f"mojolearn PCA: svd_solver='full' needs at least as many "
                f"samples as features and got {x.shape[0]} x {x.shape[1]}. "
                "The dense route is an R-SVD, which needs a tall matrix; the "
                "portable route for a wide one is an LQ factorization of the "
                "transpose and it is not written "
                "(decomposition/NOT_IMPLEMENTED.tsv, DEVIATION 593). "
                "svd_solver='covariance_eigh' handles this shape. Silently "
                "substituting it here would be the substitution this class "
                "refuses to make for the solver name itself"
            )
        nc = _component_count(self.n_components, x.shape)
        if dense and nc > min(x.shape):
            raise ValueError("full SVD n_components cannot exceed min(n_samples, n_features)")
        self.components_ = np.empty((nc, x.shape[1]), dtype=np.float32)
        self.mean_ = np.empty(x.shape[1], dtype=np.float32)
        self.explained_variance_ = np.empty(nc, dtype=np.float32)
        self.explained_variance_ratio_ = np.empty(nc, dtype=np.float32)
        self.singular_values_ = np.empty(nc, dtype=np.float32)
        fit_fn = binding.pca_fit_full if dense else binding.pca_fit
        self.noise_variance_ = float(fit_fn(
            _addr_ro(x), _addr(self.components_), _addr(self.mean_),
            _addr(self.explained_variance_), _addr(self.explained_variance_ratio_),
            _addr(self.singular_values_), [x.shape[0], x.shape[1], nc],
        ))
        self.n_components_ = nc
        self.n_features_in_ = x.shape[1]
        self.n_samples_ = x.shape[0]
        if self.whiten:
            self._validate_whiten_state()
        return self

    def transform(self, X):
        if not hasattr(self, "components_"):
            raise ValueError("mojolearn PCA: call fit before transform")
        x, _ = as_f32_c(X, "X")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("mojolearn PCA feature count differs from fit")
        out = np.empty((x.shape[0], self.n_components_), dtype=np.float32)
        if self.whiten:
            self._validate_whiten_state()
            if not np.isfinite(x).all():
                raise ValueError("mojolearn PCA whitening requires finite X")
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
            if not np.isfinite(out).all():
                raise ValueError("mojolearn PCA whitening produced nonfinite output")
            return out
        self._bind("_mojolearn_estimators").pca_transform(
            _addr_ro(x), _addr_ro(self.mean_), _addr_ro(self.components_),
            _addr(out), [x.shape[0], x.shape[1], self.n_components_],
        )
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

    def inverse_transform(self, X):
        if self.whiten and not hasattr(self, "components_"):
            raise ValueError("mojolearn PCA: call fit before inverse_transform")
        z, _ = as_f32_c(X, "X")
        if z.shape[1] != self.n_components_:
            raise ValueError("mojolearn PCA component count differs from fit")
        out = np.empty((z.shape[0], self.n_features_in_), dtype=np.float32)
        if self.whiten:
            self._validate_whiten_state()
            if not np.isfinite(z).all():
                raise ValueError("mojolearn PCA whitening requires finite scores")
            self._whiten_binding().pca_whiten_inverse_transform(
                _addr_ro(z), _addr_ro(self.components_),
                _addr_ro(self.singular_values_), _addr_ro(self.mean_),
                _addr(out),
                [z.shape[0], self.n_features_in_, self.n_components_,
                 self.n_samples_],
            )
            if not np.isfinite(out).all():
                raise ValueError("mojolearn PCA whitening produced nonfinite output")
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
                                eigJacobi). THE MISSING PRIMITIVE IS NO LONGER
                                THE QR: core/householder_qr.mojo landed
                                2026-09-01 with the R-SVD arm of PCA. Three
                                pieces remain, all named: the explicit thin Q
                                (qrGetQ, rsvd.cuh:198/:218/:241, which the
                                PCA arm deliberately never forms), a Gaussian
                                sketch over core/philox.mojo, and a seeded
                                random_state on this surface, which is a
                                reproducibility contract rather than one more
                                parameter. NOTE the default is NOT
                                scikit-learn's 'randomized', because
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
                "written down in decomposition/NOT_IMPLEMENTED.tsv. It no "
                "longer waits on the tall-skinny QR, which landed 2026-09-01 "
                "in core/householder_qr.mojo; it waits on the explicit thin "
                "Q, a Gaussian sketch and a seeded random_state on this "
                "surface. Accepting the name and running this arm would be a "
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
