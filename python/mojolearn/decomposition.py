# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU dimensionality reduction."""

from . import _backend, _mojolearn_estimators, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, all_finite, as_f32_c, empty
from ._mode import NumericModeMixin

#: The model file formats (the classical host inference lane, 2026-09-13).
#: `save` writes the fitted arrays as fitted, raw bytes and exact dtypes,
#: through `_serialize.write_npz`; `load` refuses a cast. What `transform`
#: reads plus the fitted spectrum a user reads back.
_PCA_FORMAT = "mojolearn-pca-1"
_TSVD_FORMAT = "mojolearn-tsvd-1"


def _saved_mode(est):
    """The tier `transform` would run on now, persisted as GradientBoosting
    persists it, so a loaded model never silently changes tier."""
    mode = getattr(est, "numeric_mode", None) or _backend.default_mode()
    if not isinstance(mode, str) or mode.strip().lower() not in (
        "fast", "deterministic", "identical"
    ):
        raise ValueError(f"mojolearn: cannot save invalid numeric_mode {mode!r}")
    return mode.strip().lower()


def _restore_mode(obj, arrays):
    mode = _serialize.scalar_str(arrays, "numeric_mode")
    if mode not in ("fast", "deterministic", "identical"):
        raise ValueError(f"mojolearn: invalid saved numeric_mode {mode!r}")
    obj.numeric_mode = mode


def _check_saved_by(arrays, path, cls):
    """The file's `estimator` must be `cls` or a base of it, so the host
    subclasses of `_classical_host.py` load the plain class's file."""
    saved_as = _serialize.scalar_str(arrays, "estimator")
    if saved_as not in (c.__name__ for c in cls.__mro__):  # glue: checks the saved class name
        raise ValueError(
            f"mojolearn: {path!r} was saved by {saved_as}, not {cls.__name__}"
        )


def _check_components(path, components, nc, nf):
    if components.ndim != 2 or tuple(components.shape) != (nc, nf):
        raise ValueError(
            f"mojolearn: {path!r} components shape {tuple(components.shape)} is "
            f"not ({nc}, {nf})"
        )


def _as_array_view(value):
    """`value` as an `Array` WITHOUT copying: itself when it already is
    one, a zero-copy `Array.from_buffer` view over any other buffer-protocol
    object (a NumPy array a caller assigned to a model attribute), None
    when it is neither. DEVIATION 2368."""
    if isinstance(value, Array):
        return value
    try:
        return Array.from_buffer(value)
    except (TypeError, ValueError):
        return None


def _dense(X):
    """A scipy.sparse input (duck-typed: toarray, nnz, format; nothing is
    imported) densified exactly, as scikit-learn's PCA (arpack /
    covariance_eigh) and TruncatedSVD accept it (lane/algos-decomp,
    2026-09-27); anything else unchanged."""
    if hasattr(X, "toarray") and hasattr(X, "nnz") and hasattr(X, "format"):
        return X.toarray()
    return X


def _component_count(n_components, shape):
    value = min(shape) if n_components is None else int(n_components)  # glue: smaller of two shape dims
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
    the divide-and-conquer `eigDC` (syevd), which is NOT implemented
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
        svd_solver    honored   'randomized' (lane/algos-decomp): scikit-
                                learn's `_fit_truncated` through
                                mojolearn.randomized_svd (a Philox Gaussian
                                sketch, orthonormalized power iterations, the
                                exact small SVD); iterated_power,
                                n_oversamples and random_state are honored
        svd_solver    refused   'arpack' (scikit-learn's implicitly
                                restarted Lanczos) is a third algorithm, NOT
                                IMPLEMENTED; accepting the name and running
                                an exact arm would be a silent substitution
                                (decomposition/NOT_IMPLEMENTED.tsv)
        n_components  honored   a float in (0, 1) (the variance fraction) and
                                'mle' (Minka's MLE, `_infer_dimension`)
        copy          accepted (the input is never written). Both Jacobis run RAFT's own
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
    rounded bits. Following the existing skip-zero contract (the same threshold as cuML),
    s < float32(1e-10) skips the singular division (inverse skips its
    multiplication), retaining the sample-count scale. Degenerate columns
    therefore do not promise unit variance. Nonfinite data/model arrays and
    negative singular values or explained variances are refused on this
    whitening surface. Overflowed output is refused. These additive binding
    exports require a rebuilt extension; source availability is not new
    cross-vendor or installed-wheel qualification.

    A scipy.sparse X is densified exactly (every arm). Incremental fitting
    is IncrementalPCA's.
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

    #: The randomized arm (lane/algos-decomp, 2026-09-27): scikit-learn's
    #: `_fit_truncated` through `mojolearn.randomized_svd` (the decomp lane's
    #: identical cells: a Philox Gaussian sketch, MGS2 power iterations, the
    #: exact small SVD; IDENTITY_PATHS rows 130, 133, 134).
    _RANDOM_SOLVERS = ("randomized",)

    #: 'arpack' is NOT one of them: scikit-learn's implicitly restarted
    #: Lanczos is a third algorithm, and running an exact arm under its name
    #: would be a silent substitution (decomposition/NOT_IMPLEMENTED.tsv).
    _SOLVERS = _COV_SOLVERS + _DENSE_SOLVERS + _RANDOM_SOLVERS

    def __init__(self, n_components=None, *, copy=True, whiten=False, svd_solver="auto", tol=0.0,
                 iterated_power="auto", n_oversamples=10, power_iteration_normalizer="auto", random_state=None):
        self.n_components = n_components
        self.copy = copy
        # scikit-learn's tol is ARPACK's convergence tolerance, which only
        # svd_solver='arpack' reads; that solver is refused, so tol must stay
        # its default 0.0 (a nonzero tol would ask for an arm that is not here).
        self.tol = tol
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state

    def _whiten_binding(self):
        """The binding, checked for the whitened pair.

        `pca_whiten_transform` / `pca_whiten_inverse_transform` are ADDITIVE
        exports beside the unwhitened ones, so an older `_mojolearn_estimators`
        that predates them is a build that simply does not carry the pair.
        That is a build state, not a user error, and it says so rather than
        failing with an arity message from the extension.
        """
        binding = self._bind("_mojolearn_estimators")
        if not all(callable(getattr(binding, name, None)) for name in  # glue: checks two binding entry names
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
        """DEVIATION 2368: the four model arrays are inspected through the
        buffer protocol, so the `Array` fit stores and any other float32
        C-contiguous buffer a caller assigned (the surface tests assign
        NumPy arrays) both pass; anything else is refused by name. The
        finiteness scan is `_buffer.all_finite` (native when the helper is
        loadable), the sign test is `Array.min()`."""
        views = {}
        for name, shape in (("components_", (self.n_components_, self.n_features_in_)),  # glue: loops over saved attribute names
                            ("mean_", (self.n_features_in_,)),
                            ("singular_values_", (self.n_components_,)),
                            ("explained_variance_", (self.n_components_,))):
            a = _as_array_view(getattr(self, name, None))
            if (a is None or a.dtype != "<f4" or tuple(a.shape) != shape
                    or not a.flags["C_CONTIGUOUS"] or not all_finite(a)):
                raise ValueError(f"mojolearn PCA whitening requires finite C-order float32 {name} with shape {shape}")
            views[name] = a
        if (self.n_samples_ < 2 or views["singular_values_"].min() < 0
                or views["explained_variance_"].min() < 0):
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
                f"implemented; this class runs {self._SOLVERS}. 'arpack' is "
                "scikit-learn's implicitly restarted Lanczos, a third "
                "algorithm; accepting the name and running an exact arm "
                "would be a silent substitution, which is why this raises "
                "(decomposition/NOT_IMPLEMENTED.tsv)"
            )
        if float(self.tol) != 0.0:
            raise NotImplementedError(
                "mojolearn PCA: tol is ARPACK's convergence tolerance and "
                "svd_solver='arpack' is not implemented; leave tol=0.0"
            )
        if self.whiten:
            self._whiten_binding()
        if self.svd_solver in self._RANDOM_SOLVERS:
            return self._fit_randomized(X)
        dense = self.svd_solver in self._DENSE_SOLVERS
        if dense:
            binding = self._dense_binding()
        else:
            binding = self._bind("_mojolearn_estimators")
        x, self.input_copied_ = as_f32_c(_dense(X), ndim=2, name="X")
        if self.whiten and not all_finite(x):
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
        frac = None
        mle = isinstance(self.n_components, str) and self.n_components == "mle"
        if mle:
            # scikit-learn's Minka MLE (lane/algos-decomp, 2026-09-27): fit
            # every component, keep the rank `_infer_dimension` picks from the
            # full explained-variance spectrum.
            if x.shape[0] < x.shape[1]:
                raise ValueError("n_components='mle' is only supported if n_samples >= n_features")
            nc = min(x.shape)  # glue: smaller of two shape dims
        elif isinstance(self.n_components, float) and not isinstance(self.n_components, bool):
            # scikit-learn's variance fraction (lane/algos-decomp, 2026-09-27):
            # fit every component, keep the fewest whose cumulative explained
            # variance ratio EXCEEDS the fraction (searchsorted side='right'),
            # the cumulative sum in float64 on the host (sequential IEEE adds).
            if not 0.0 < self.n_components < 1.0:
                raise ValueError("mojolearn PCA: a float n_components must be in (0, 1)")
            frac = float(self.n_components)
            nc = min(x.shape)  # glue: smaller of two shape dims
        else:
            nc = _component_count(self.n_components, x.shape)
        if dense and nc > min(x.shape):  # glue: smaller of two shape dims
            raise ValueError("full SVD n_components cannot exceed min(n_samples, n_features)")
        self.components_ = empty((nc, x.shape[1]), "<f4")
        self.mean_ = empty((x.shape[1],), "<f4")
        self.explained_variance_ = empty((nc,), "<f4")
        self.explained_variance_ratio_ = empty((nc,), "<f4")
        self.singular_values_ = empty((nc,), "<f4")
        fit_fn = binding.pca_fit_full if dense else binding.pca_fit
        self.noise_variance_ = float(fit_fn(
            addr_ro(x, name="x"), addr(self.components_, name="components_"), addr(self.mean_, name="mean_"),
            addr(self.explained_variance_, name="explained_variance_"), addr(self.explained_variance_ratio_, name="explained_variance_ratio_"),
            addr(self.singular_values_, name="singular_values_"), [x.shape[0], x.shape[1], nc],
        ))
        if frac is not None or mle:
            if mle:
                from ._expansion_decomp import _pca_mle_rank
                keep = _pca_mle_rank(list(self.explained_variance_), x.shape[0], self.numeric_mode_used())
            else:
                ratios = list(self.explained_variance_ratio_)
                cum, keep = 0.0, len(ratios)
                for i, r in enumerate(ratios):
                    cum += float(r)
                    if cum > frac:
                        keep = i + 1
                        break
            keep = min(keep, nc)
            ev = [float(v) for v in self.explained_variance_]
            rest = ev[keep:]
            tail = 0.0
            for v in rest:
                tail += v
            import array as _arr
            from ._buffer import frombytes
            self.noise_variance_ = float(_arr.array("f", [tail / len(rest)])[0]) if rest else 0.0
            d = x.shape[1]
            comp = _arr.array("f")
            comp.frombytes(self.components_.tobytes()[:4 * keep * d])
            self.components_ = frombytes(comp.tobytes(), "<f4", (keep, d))
            for name in ("explained_variance_", "explained_variance_ratio_", "singular_values_"):  # glue: loops over three attribute names
                setattr(self, name, frombytes(getattr(self, name).tobytes()[:4 * keep], "<f4", (keep,)))
            nc = keep
        self.n_components_ = nc
        self.n_features_in_ = x.shape[1]
        self.n_samples_ = x.shape[0]
        if self.whiten:
            self._validate_whiten_state()
        return self

    def _fit_randomized(self, X):
        """svd_solver='randomized' (scikit-learn `_pca.py::_fit_truncated`):
        the centered data through `mojolearn.randomized_svd`, then
        svd_flip on Vt; explained variance S^2 / (n - 1), the ratio against
        the total sample variance, the noise variance the mean of what is
        left. random_state None means seed 0 (the Philox stream)."""
        from ._expansion_decomp import _randomized_decompose
        x, self.input_copied_ = as_f32_c(_dense(X), ndim=2, name="X")
        if x.shape[0] < 2 or x.shape[1] < 2:
            raise ValueError("mojolearn PCA requires at least 2 rows and 2 features")
        nc = _component_count(self.n_components, x.shape)
        if nc >= min(x.shape):  # glue: smaller of two shape dims
            raise ValueError("svd_solver='randomized' needs n_components < min(n_samples, n_features)")
        n_iter = self.iterated_power
        if n_iter == "auto":
            n_iter = 7 if nc < 0.1 * min(x.shape) else 4  # glue: smaller of two shape dims
        got = _randomized_decompose(
            x, nc, center=True, n_oversamples=self.n_oversamples, n_iter=n_iter,
            power_iteration_normalizer=self.power_iteration_normalizer, random_state=self.random_state,
            numeric_mode=self.numeric_mode_used(),
            binding=_backend.binding("_mojolearn_x_decomp", self.numeric_mode_used()))
        self.components_ = got["components"]
        self.mean_ = got["mean"]
        self.explained_variance_ = got["explained_variance"]
        self.explained_variance_ratio_ = got["explained_variance_ratio"]
        self.singular_values_ = got["singular_values"]
        self.noise_variance_ = float(got["noise_variance"])
        self.n_components_ = nc
        self.n_features_in_ = x.shape[1]
        self.n_samples_ = x.shape[0]
        if self.whiten:
            self._validate_whiten_state()
        return self

    def transform(self, X):
        if not hasattr(self, "components_"):
            raise ValueError("mojolearn PCA: call fit before transform")
        x, _ = as_f32_c(_dense(X), ndim=2, name="X")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("mojolearn PCA feature count differs from fit")
        out = empty((x.shape[0], self.n_components_), "<f4")
        if self.whiten:
            self._validate_whiten_state()
            if not all_finite(x):
                raise ValueError("mojolearn PCA whitening requires finite X")
            # `n_samples_` and NOT `x.shape[0]`. DEVIATION 580: cuML's dense
            # path scales by the row count of THIS CALL (pca.pyx:770), so
            # their `transform(X[:100])` disagrees with `transform(X)[:100]`.
            # Their own sparse path uses the fit's count and so does
            # scikit-learn, and so does this.
            self._whiten_binding().pca_whiten_transform(
                addr_ro(x, name="x"), addr_ro(self.mean_, name="mean_"), addr_ro(self.components_, name="components_"),
                addr_ro(self.singular_values_, name="singular_values_"), addr(out, name="out"),
                [x.shape[0], x.shape[1], self.n_components_, self.n_samples_],
            )
            if not all_finite(out):
                raise ValueError("mojolearn PCA whitening produced nonfinite output")
            return out
        self._bind("_mojolearn_estimators").pca_transform(
            addr_ro(x, name="x"), addr_ro(self.mean_, name="mean_"), addr_ro(self.components_, name="components_"),
            addr(out, name="out"), [x.shape[0], x.shape[1], self.n_components_],
        )
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

    def inverse_transform(self, X):
        if self.whiten and not hasattr(self, "components_"):
            raise ValueError("mojolearn PCA: call fit before inverse_transform")
        z, _ = as_f32_c(_dense(X), ndim=2, name="X")
        if z.shape[1] != self.n_components_:
            raise ValueError("mojolearn PCA component count differs from fit")
        out = empty((z.shape[0], self.n_features_in_), "<f4")
        if self.whiten:
            self._validate_whiten_state()
            if not all_finite(z):
                raise ValueError("mojolearn PCA whitening requires finite scores")
            self._whiten_binding().pca_whiten_inverse_transform(
                addr_ro(z, name="z"), addr_ro(self.components_, name="components_"),
                addr_ro(self.singular_values_, name="singular_values_"), addr_ro(self.mean_, name="mean_"),
                addr(out, name="out"),
                [z.shape[0], self.n_features_in_, self.n_components_,
                 self.n_samples_],
            )
            if not all_finite(out):
                raise ValueError("mojolearn PCA whitening produced nonfinite output")
            return out
        self._bind("_mojolearn_estimators").inverse_transform(
            addr_ro(z, name="z"), addr_ro(self.components_, name="components_"), addr_ro(self.mean_, name="mean_"),
            addr(out, name="out"), [z.shape[0], self.n_features_in_, self.n_components_, 1],
        )
        return out

    def save(self, path):
        """Write the fitted model to `path` as an npz: `components_`,
        `mean_`, `singular_values_`, `explained_variance_` and
        `explained_variance_ratio_` as fitted (float32), `noise_variance_`
        as float64, `meta` `<i8` [n_components_, n_features_in_,
        n_samples_, whiten] and `svd_solver` (the classical host inference
        lane, 2026-09-13). `mojolearn.host_model(path)` transforms from it
        on a CPU with no GPU, whitened or not (the whitened pair joined the
        host binding in the kde svc host lane, 2026-09-14)."""
        if not hasattr(self, "components_"):
            raise RuntimeError("this estimator is not fitted yet")
        arrays = {
            "format": _PCA_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "svd_solver": str(self.svd_solver),
            "components": self.components_,
            "mean": self.mean_,
            "singular_values": self.singular_values_,
            "explained_variance": self.explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio_,
            "noise_variance": Array.from_list([float(self.noise_variance_)], "<f8"),
            "meta": Array.from_list(
                [int(self.n_components_), int(self.n_features_in_),
                 int(self.n_samples_), 1 if self.whiten else 0],
                "<i8",
            ),
        }
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result transforms; it does not
        refit."""
        arrays = _serialize.read_npz(path, _PCA_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 4:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 4 are needed")
        nc, nf = int(meta[0]), int(meta[1])
        obj = cls(n_components=nc, whiten=bool(int(meta[3])),
                  svd_solver=_serialize.scalar_str(arrays, "svd_solver"))
        _restore_mode(obj, arrays)
        obj.components_ = _serialize.exact(arrays, "components", "<f4")
        _check_components(path, obj.components_, nc, nf)
        obj.mean_ = _serialize.exact(arrays, "mean", "<f4")
        obj.singular_values_ = _serialize.exact(arrays, "singular_values", "<f4")
        obj.explained_variance_ = _serialize.exact(arrays, "explained_variance", "<f4")
        obj.explained_variance_ratio_ = _serialize.exact(arrays, "explained_variance_ratio", "<f4")
        for name in ("mean_",):  # glue: loops over saved attribute names
            if getattr(obj, name).size != nf:
                raise ValueError(f"mojolearn: {path!r} {name} does not match n_features_in_")
        for name in ("singular_values_", "explained_variance_", "explained_variance_ratio_"):  # glue: loops over saved attribute names
            if getattr(obj, name).size != nc:
                raise ValueError(f"mojolearn: {path!r} {name} does not match n_components_")
        noise = _serialize.exact(arrays, "noise_variance", "<f8")
        if noise.size != 1:
            raise ValueError(f"mojolearn: {path!r} noise_variance must hold one value")
        obj.noise_variance_ = float(noise[0])
        obj.n_components_ = nc
        obj.n_features_in_ = nf
        obj.n_samples_ = int(meta[2])
        return obj


class TruncatedSVD(NumericModeMixin):
    """Uncentered truncated SVD on the GPU. Reference: cuML's `tsvdFit`
    (Gram matrix + the same device Jacobi eigensolver PCA uses).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY (measured row by row by
    `tools/e2u_matrix_fit.py`):

        n_components  honored   1..n_features, the eigenpairs kept
        algorithm     honored   'covariance_eigh' and 'jacobi', both naming
                                the one arm that runs: the Gram matrix plus
                                the device Jacobi, which is what cuML's
                                `tsvdFit` takes
        algorithm     honored   'randomized' through mojolearn.randomized_svd
                                (n_iter, n_oversamples, random_state
                                honored). NOTE the default is NOT
                                scikit-learn's 'randomized'
        algorithm     refused   'arpack' (scipy's svds, implicitly restarted
                                Lanczos) is NOT IMPLEMENTED: running the Gram
                                arm under its name would be a silent
                                substitution (decomposition/
                                NOT_IMPLEMENTED.tsv)
        tol           refused unless 0.0 (it is ARPACK's tolerance)
        n_features > 128 refused UNDER NUMERIC_IDENTICAL ONLY, as for PCA
                                (IDENTITY_PATHS row 27)

    Components, singular values, `explained_variance_` and
    `explained_variance_ratio_` are exposed; the two variances are
    scikit-learn's (and tsvd.cuh's) transformed-data definition, computed
    in this class's binding (`tsvd_explained`).
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    #: The two names for the ONE exact arm this class runs (see
    #: `PCA._SOLVERS`) and 'randomized'. 'arpack' is refused by name.
    _ALGORITHMS = ("covariance_eigh", "jacobi", "randomized")

    def __init__(self, n_components=2, *, algorithm="covariance_eigh", n_iter=5, n_oversamples=10,
                 power_iteration_normalizer="auto", random_state=None, tol=0.0):
        self.n_components = n_components
        self.tol = tol
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state

    def fit(self, X, y=None):
        if float(self.tol) != 0.0:
            raise NotImplementedError(
                "mojolearn TruncatedSVD: tol is ARPACK's convergence tolerance "
                "and algorithm='arpack' is not implemented; leave tol=0.0"
            )
        if self.algorithm == "randomized":
            # scikit-learn `_truncated_svd.py` algorithm='randomized' through
            # `mojolearn.randomized_svd` (lane/algos-decomp, 2026-09-27)
            from ._expansion_decomp import _randomized_decompose
            x, self.input_copied_ = as_f32_c(_dense(X), ndim=2, name="X")
            if x.shape[0] < 2 or x.shape[1] < 2:
                raise ValueError("mojolearn TruncatedSVD requires at least 2 rows and 2 features")
            nc = _component_count(self.n_components, x.shape)
            got = _randomized_decompose(
                x, nc, center=False, n_oversamples=self.n_oversamples, n_iter=self.n_iter,
                power_iteration_normalizer=self.power_iteration_normalizer, random_state=self.random_state,
                numeric_mode=self.numeric_mode_used(),
                binding=_backend.binding("_mojolearn_x_decomp", self.numeric_mode_used()))
            self.components_ = got["components"]
            self.singular_values_ = got["singular_values"]
            self.explained_variance_ = got["explained_variance"]
            self.explained_variance_ratio_ = got["explained_variance_ratio"]
            self.n_components_ = nc
            self.n_features_in_ = x.shape[1]
            return self
        if self.algorithm not in self._ALGORITHMS:
            raise NotImplementedError(
                f"mojolearn TruncatedSVD: algorithm={self.algorithm!r} is not "
                f"implemented; this class runs {self._ALGORITHMS}. 'arpack' is "
                "scipy's svds (implicitly restarted Lanczos), a different "
                "algorithm; accepting the name and running the Gram arm would "
                "be a silent substitution, which is why this raises "
                "(decomposition/NOT_IMPLEMENTED.tsv)"
            )
        x, self.input_copied_ = as_f32_c(_dense(X), ndim=2, name="X")
        if x.shape[0] < 2 or x.shape[1] < 2:
            raise ValueError("mojolearn TruncatedSVD requires at least 2 rows and 2 features")
        nc = _component_count(self.n_components, x.shape)
        self.components_ = empty((nc, x.shape[1]), "<f4")
        self.singular_values_ = empty((nc,), "<f4")
        self._bind("_mojolearn_estimators").tsvd_fit(
            addr_ro(x, name="x"), addr(self.components_, name="components_"), addr(self.singular_values_, name="singular_values_"),
            [x.shape[0], x.shape[1], nc],
        )
        # scikit-learn's explained_variance_ / _ratio_ (np.var of X V^T per
        # column, ddof 0, against the summed column variances of X), in this
        # class's own binding (`tsvd_explained`, decomposition/estimator.mojo
        # and its host twin).
        self.explained_variance_ = empty((nc,), "<f4")
        self.explained_variance_ratio_ = empty((nc,), "<f4")
        self._bind("_mojolearn_estimators").tsvd_explained(
            addr_ro(x, name="x"), addr_ro(self.components_, name="components_"),
            addr(self.explained_variance_, name="explained_variance_"),
            addr(self.explained_variance_ratio_, name="explained_variance_ratio_"),
            [x.shape[0], x.shape[1], nc],
        )
        self.n_components_ = nc
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "components_"):
            raise ValueError("mojolearn TruncatedSVD: call fit before transform")
        x, _ = as_f32_c(_dense(X), ndim=2, name="X")
        if x.shape[1] != self.n_features_in_:
            raise ValueError("mojolearn TruncatedSVD feature count differs from fit")
        out = empty((x.shape[0], self.n_components_), "<f4")
        self._bind("_mojolearn_estimators").tsvd_transform(
            addr_ro(x, name="x"), addr_ro(self.components_, name="components_"), addr(out, name="out"),
            [x.shape[0], x.shape[1], self.n_components_],
        )
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

    def inverse_transform(self, X):
        z, _ = as_f32_c(_dense(X), ndim=2, name="X")
        if z.shape[1] != self.n_components_:
            raise ValueError("mojolearn TruncatedSVD component count differs from fit")
        out = empty((z.shape[0], self.n_features_in_), "<f4")
        # The mean pointer is unused when add_mean is false; components is a
        # valid non-null float32 address for the boundary contract.
        self._bind("_mojolearn_estimators").inverse_transform(
            addr_ro(z, name="z"), addr_ro(self.components_, name="components_"), addr_ro(self.components_, name="components_"),
            addr(out, name="out"), [z.shape[0], self.n_features_in_, self.n_components_, 0],
        )
        return out

    def save(self, path):
        """Write the fitted model to `path` as an npz: `components_` and
        `singular_values_` as fitted (float32), `meta` `<i8` [n_components_,
        n_features_in_] and `algorithm` (the classical host inference lane,
        2026-09-13). `mojolearn.host_model(path)` transforms from it on a
        CPU with no GPU."""
        if not hasattr(self, "components_"):
            raise RuntimeError("this estimator is not fitted yet")
        arrays = {
            "format": _TSVD_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "algorithm": str(self.algorithm),
            "components": self.components_,
            "singular_values": self.singular_values_,
            "meta": Array.from_list(
                [int(self.n_components_), int(self.n_features_in_)], "<i8"
            ),
        }
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result transforms; it does not
        refit."""
        arrays = _serialize.read_npz(path, _TSVD_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 2:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 2 are needed")
        nc, nf = int(meta[0]), int(meta[1])
        obj = cls(n_components=nc, algorithm=_serialize.scalar_str(arrays, "algorithm"))
        _restore_mode(obj, arrays)
        obj.components_ = _serialize.exact(arrays, "components", "<f4")
        _check_components(path, obj.components_, nc, nf)
        obj.singular_values_ = _serialize.exact(arrays, "singular_values", "<f4")
        if obj.singular_values_.size != nc:
            raise ValueError(f"mojolearn: {path!r} singular_values does not match n_components_")
        obj.n_components_ = nc
        obj.n_features_in_ = nf
        return obj
