# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU dimensionality reduction."""

from . import _backend, _mojolearn_estimators, _native_archive_bridge, _native_pca_bridge, _native_tsvd_bridge, _serialize
from ._array import Array
from ._mode import NumericModeMixin

#: The model file formats (the classical host inference lane, 2026-09-13).
#: `save` writes the fitted arrays as fitted, raw bytes and exact dtypes,
#: through `_serialize.write_npz`; `load` refuses a cast. What `transform`
#: reads plus the fitted spectrum a user reads back.
_PCA_FORMAT = "mojolearn-pca-1"
_TSVD_FORMAT = "mojolearn-tsvd-1"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered below each class; host_archive_native.mojo
# builds the loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()


def _saved_mode(est):
    """The tier `transform` would run on now, persisted as GradientBoosting
    persists it, so a loaded model never silently changes tier."""
    mode = getattr(est, "numeric_mode", None) or _backend.default_mode()
    if not isinstance(mode, str) or mode.strip().lower() not in (
        "fast", "identical"
    ):
        raise ValueError(f"mojolearn: cannot save invalid numeric_mode {mode!r}")
    return mode.strip().lower()


def _restore_mode(obj, arrays):
    mode = _serialize.scalar_str(arrays, "numeric_mode")
    if mode not in ("fast", "identical"):
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


# Maintainer notes (moved from the docstring).
# Two arms as of 2026-09-01. The covariance arm is cuML's pcaFit route
# (covariance, then an eigensolver) with cuML's JACOBI eigensolver
# (svd_solver='jacobi', cuSOLVER syevj there, a device Jacobi here). cuML's
# 'auto' reaches the divide-and-conquer eigDC (syevd), which is NOT implemented
# (decomposition/NOT_IMPLEMENTED.tsv); 'auto' runs the Jacobi arm. 'jacobi' is
# cuML's name (pca.pyx:398 maps it to COV_EIG_JACOBI).
# Wide IDENTICAL full fits use transposed QR, at most min(X.shape) components,
# and a 64-sweep/1e-6 Jacobi budget; tall fits keep the RAFT 15-sweep/1e-7
# budget (decomposition/impl/linalg/detail/pca.mojo). Nonconvergence raises.
# svd_solver='full' is scikit-learn's meaning (_pca.py:539-540 -> _fit_full):
# Householder QR of X_c, then a one-sided Jacobi SVD of the small R; the
# covariance is never formed, so the condition number is never squared. cuML
# collapses the two names; we deliberately do not. DEVIATIONS 586-593 in
# core/householder_qr.mojo and decomposition/impl/linalg/detail/svd_full.mojo;
# gated by decomposition/checks/svd_full_check.mojo (its ill-conditioning gate
# measures the accuracy claim). IDENTICAL supports n_samples < n_features via
# transposed QR and explicit right-basis reconstruction (DEVIATION 593); other
# modes keep the wide-shape refusal.
# 'randomized' (lane/algos-decomp): scikit-learn's _fit_truncated through
# mojolearn.randomized_svd (Philox Gaussian sketch, orthonormalized power
# iterations, exact small SVD).
# n_components: the eigendecomposition is of the FULL covariance on every
# setting; noise_variance_ is the mean of the discarded eigenvalues (cuML's
# truncCompExpVars).
# Whitening: cuML's rescale of a COPY of the components (pca.cuh:292-302 and
# :232-243), all three numeric modes, pinned at the seam through identical_mul
# / identical_div / ftz. DEVIATIONS 580-585 in
# decomposition/impl/linalg/detail/pca.mojo; gated by
# decomposition/checks/pca_check.mojo (unit variance, round trip, row-subset
# agreement, planted-bit edges). Components are scaled by sqrt(n_samples_-1)/s
# (pinned FP32 multiply, then divide, FTZ seams); s < float32(1e-10) skips the
# division (cuML's threshold), keeping the sample-count scale. The whitened
# pair is an additive binding export: an older extension needs a rebuild.
# Options measured row by row by tools/e2u_matrix_fit.py.
class PCA(NumericModeMixin):
    """Principal component analysis.

    Centers the data and projects it onto its leading principal axes. Input
    is converted to C-order float32; a scipy.sparse matrix is densified. Use
    :class:`IncrementalPCA` for fitting in batches.

    The default ``svd_solver="auto"`` always runs the covariance
    eigendecomposition (covariance matrix, then a Jacobi eigensolver), unlike
    scikit-learn, where ``"auto"`` chooses a solver from the data shape.
    ``svd_solver="full"`` is a different algorithm, an SVD of the centered
    data that never forms the covariance and so is more accurate on
    ill-conditioned data.

    Parameters
    ----------
    n_components : int, float, "mle" or None, default None
        Number of components to keep. An int in ``[1, n_features]``; None
        keeps ``min(n_samples, n_features)``; a float in (0, 1) keeps the
        fewest components whose cumulative explained variance ratio exceeds
        it; ``"mle"`` uses Minka's MLE (requires ``n_samples >= n_features``).
    copy : bool, default True
        Accepted for compatibility; the input is never modified.
    whiten : bool, default False
        Scale the projected columns to unit sample variance (using the
        fitted sample count, never the size of the ``transform`` batch).
        ``inverse_transform`` undoes the scaling. Columns whose singular
        value is below 1e-10 are not divided by it, so they do not reach unit
        variance.
    svd_solver : {"auto", "covariance_eigh", "jacobi", "full", "randomized"}, default "auto"
        ``"auto"``, ``"covariance_eigh"`` and ``"jacobi"`` all run the
        covariance eigendecomposition. ``"full"`` runs a QR-based SVD of the
        centered data; outside the ``"identical"`` numeric mode it needs
        ``n_samples >= n_features`` and raises ``NotImplementedError``
        otherwise. ``"randomized"`` runs :func:`randomized_svd` and needs
        ``n_components < min(n_samples, n_features)``. ``"arpack"`` raises
        ``NotImplementedError``.
    tol : float, default 0.0
        Only 0.0 is accepted (it is ARPACK's tolerance); any other value
        raises ``NotImplementedError``.
    iterated_power : int or "auto", default "auto"
        Power iterations for ``"randomized"``; ``"auto"`` uses 7 when
        ``n_components < 0.1 * min(n_samples, n_features)``, else 4.
    n_oversamples : int, default 10
        Extra random vectors for ``"randomized"``.
    power_iteration_normalizer : str, default "auto"
        Normalizer between power iterations for ``"randomized"``.
    random_state : int or None, default None
        Seed for ``"randomized"``; None means seed 0.

    Attributes
    ----------
    components_ : Array of shape (``n_components_``, n_features), float32
        Principal axes, sorted by decreasing explained variance.
    mean_ : Array of shape (n_features,), float32
        Per-feature mean of the training data.
    explained_variance_ : Array of shape (``n_components_``,), float32
        Variance explained by each component.
    explained_variance_ratio_ : Array of shape (``n_components_``,), float32
        Fraction of total variance explained by each component.
    singular_values_ : Array of shape (``n_components_``,), float32
        Singular values of the centered data for each component.
    noise_variance_ : float
        Mean of the discarded eigenvalues of the covariance.
    ``n_components_`` : int
        Number of components kept.
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_samples_ : int
        Number of training rows.
    input_copied_ : bool
        Whether ``X`` had to be copied to C-order float32.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> pca = mojolearn.PCA(n_components=3).fit(X)
    >>> Z = pca.transform(X)
    >>> X_back = pca.inverse_transform(Z)
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

    #: Every public operation below is one call into the compiled PCA owner
    #: (bindings/pca_native.mojo), connected once at import. It validates the
    #: arguments and inputs, chooses the solver and numeric-mode binding, runs
    #: the one compiled fit (projecting inside it for fit_transform) and
    #: stores the fitted attributes; NumericModeMixin leaves these unwrapped.
    _mojolearn_native_public = True

    def __init__(self, n_components=None, *, copy=True, whiten=False, svd_solver="auto", tol=0.0,
                 iterated_power="auto", n_oversamples=10, power_iteration_normalizer="auto", random_state=None,
                 numeric_mode=None):
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
        self.numeric_mode = numeric_mode

    def fit(self, X, y=None):
        """Fit the model to ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data; at least 2 rows. Must be finite.
        y : ignored

        Returns
        -------
        self : PCA
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` is too small or ``n_components`` is out of range.
        NotImplementedError
            For an unsupported ``svd_solver``, a nonzero ``tol``, or
            ``svd_solver="full"`` on a wide matrix outside ``"identical"``
            mode.
        """
        return _PCA_ENTRY(self, "fit", (X, y), _PCA_CONTEXT)

    def transform(self, X):
        """Project ``X`` onto the principal components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with the training feature count.

        Returns
        -------
        Z : ndarray of shape (n_samples, ``n_components_``), float32
            The projected (and, with ``whiten=True``, whitened) data.

        Raises
        ------
        NotFittedError
            If the model is not fitted.
        ValueError
            If the feature count differs.
        """
        return _PCA_ENTRY(self, "transform", (X,), _PCA_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit to ``X`` and return its projection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        Z : ndarray of shape (n_samples, ``n_components_``), float32
            The projected data.
        """
        return _PCA_ENTRY(self, "fit_transform", (X, y), _PCA_CONTEXT)

    def inverse_transform(self, X):
        """Map projected data back to the original feature space.

        Parameters
        ----------
        X : array-like of shape (n_samples, ``n_components_``)
            Projected data, as returned by ``transform``.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features), float32
            The reconstruction (undoing whitening when ``whiten=True``).

        Raises
        ------
        NotFittedError
            If the model is not fitted.
        ValueError
            If the component count differs from the fit.
        """
        return _PCA_ENTRY(self, "inverse_transform", (X,), _PCA_CONTEXT)

    # Saved layout (classical host inference lane, 2026-09-13): components_,
    # mean_, singular_values_, explained_variance_, explained_variance_ratio_
    # as fitted (float32), noise_variance_ float64, meta <i8 [n_components_,
    # n_features_in_, n_samples_, whiten], svd_solver. The whitened pair joined
    # the host binding in the kde svc host lane, 2026-09-14.
    def save(self, path):
        """Save the fitted model to ``path`` as an ``.npz`` file.

        A model restored with ``PCA.load`` or ``mojolearn.host_model(path)``
        transforms (whitened or not), including on a CPU-only install.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model written by ``save``.

        The loaded model transforms; it does not refit.

        Parameters
        ----------
        path : str or path-like
            File written by ``PCA.save``.

        Returns
        -------
        model : PCA
            The restored model.

        Raises
        ------
        ValueError
            If the file is not a PCA model or its arrays are inconsistent.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


# Maintainer notes (moved from the docstring). Reference: cuML's tsvdFit
# (Gram matrix + the same device Jacobi eigensolver PCA uses). 'covariance_eigh'
# and 'jacobi' both name that one exact arm. 'arpack' (scipy's svds,
# implicitly restarted Lanczos) is not implemented: running the Gram arm under
# its name would be a silent substitution (decomposition/NOT_IMPLEMENTED.tsv).
# explained_variance_ and _ratio_ are scikit-learn's (and tsvd.cuh's)
# transformed-data definition, computed in this class's binding
# (`tsvd_explained`). Options measured row by row by tools/e2u_matrix_fit.py.
class TruncatedSVD(NumericModeMixin):
    """Dimensionality reduction by truncated singular value decomposition.

    Unlike :class:`PCA`, the data is not centered, so this works on raw term
    counts or TF-IDF matrices (a scipy.sparse input is densified). Input is
    converted to C-order float32. The default ``algorithm`` is the exact
    ``"covariance_eigh"`` solver, not scikit-learn's ``"randomized"``.

    Parameters
    ----------
    n_components : int, default 2
        Number of components, in ``[1, n_features]``.
    algorithm : {"covariance_eigh", "jacobi", "randomized"}, default "covariance_eigh"
        ``"covariance_eigh"`` and ``"jacobi"`` run the exact solver (an
        eigendecomposition of the Gram matrix). ``"randomized"`` runs
        :func:`randomized_svd`. ``"arpack"`` and any other value raise
        ``NotImplementedError``.
    n_iter : int, default 5
        Power iterations for ``"randomized"``.
    n_oversamples : int, default 10
        Extra random vectors for ``"randomized"``.
    power_iteration_normalizer : str, default "auto"
        Normalizer between power iterations for ``"randomized"``.
    random_state : int or None, default None
        Seed for ``"randomized"``.
    tol : float, default 0.0
        Only 0.0 is accepted (it is ARPACK's tolerance); any other value
        raises ``NotImplementedError``.
    numeric_mode : {'fast', 'identical'} or None, default None
        Numeric mode for this estimator; None uses the package default.
        ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    components_ : Array of shape (n_components, n_features), float32
        Right singular vectors.
    singular_values_ : Array of shape (n_components,), float32
        Singular values.
    explained_variance_ : Array of shape (n_components,), float32
        Variance of each transformed column (scikit-learn's definition).
    explained_variance_ratio_ : Array of shape (n_components,), float32
        ``explained_variance_`` divided by the total variance of ``X``.
    ``n_components_`` : int
        Number of components.
    n_features_in_ : int
        Number of features seen during ``fit``.
    input_copied_ : bool
        Whether ``X`` had to be copied to C-order float32.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> svd = mojolearn.TruncatedSVD(n_components=3).fit(X)
    >>> Z = svd.transform(X)
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_estimators"

    #: The two names for the ONE exact arm this class runs (see
    #: `PCA._SOLVERS`) and 'randomized'. 'arpack' is refused by name.
    _ALGORITHMS = ("covariance_eigh", "jacobi", "randomized")

    #: Every public operation is one call into the native owner
    #: (bindings/tsvd_native.mojo tsvd_c), which runs the whole lifecycle
    #: (fitted check, X and width checks, fit bookkeeping, NumPy results);
    #: the `_mode` class-creation wrappers leave this class unwrapped.
    _mojolearn_native_public = True

    def __init__(self, n_components=2, *, algorithm="covariance_eigh", n_iter=5, n_oversamples=10,
                 power_iteration_normalizer="auto", random_state=None, tol=0.0, numeric_mode=None):
        self.n_components = n_components
        self.numeric_mode = numeric_mode
        self.tol = tol
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model to ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data; at least 2 rows and 2 features.
        y : ignored

        Returns
        -------
        self : TruncatedSVD
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` is too small or ``n_components`` is out of range.
        NotImplementedError
            For an unsupported ``algorithm`` or a nonzero ``tol``.
        """
        return _TSVD_ENTRY(self, "fit", (X, y), _TSVD_CONTEXT)

    def transform(self, X):
        """Project ``X`` onto the components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with the training feature count.

        Returns
        -------
        Z : Array of shape (n_samples, n_components), float32
            The reduced data.

        Raises
        ------
        ValueError
            If the model is not fitted or the feature count differs.
        """
        return _TSVD_ENTRY(self, "transform", (X,), _TSVD_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit to ``X`` and return its projection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        Z : Array of shape (n_samples, n_components), float32
            The reduced data.
        """
        return _TSVD_ENTRY(self, "fit_transform", (X, y), _TSVD_CONTEXT)

    def inverse_transform(self, X):
        """Map reduced data back to the original feature space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Reduced data, as returned by ``transform``.

        Returns
        -------
        X_original : Array of shape (n_samples, n_features), float32
            The reconstruction ``X @ components_``.

        Raises
        ------
        ValueError
            If the component count differs from the fit.
        """
        return _TSVD_ENTRY(self, "inverse_transform", (X,), _TSVD_CONTEXT)

    # Saved layout (classical host inference lane, 2026-09-13): components_ and
    # singular_values_ as fitted (float32), meta <i8 [n_components_,
    # n_features_in_], algorithm.
    def save(self, path):
        """Save the fitted model to ``path`` as an ``.npz`` file.

        A model restored with ``TruncatedSVD.load`` or
        ``mojolearn.host_model(path)`` transforms, including on a CPU-only
        install.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model written by ``save``.

        The loaded model transforms; it does not refit.

        Parameters
        ----------
        path : str or path-like
            File written by ``TruncatedSVD.save``.

        Returns
        -------
        model : TruncatedSVD
            The restored model.

        Raises
        ------
        ValueError
            If the file is not a TruncatedSVD model or its arrays are inconsistent.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


_native_archive_bridge.register(PCA, _PCA_FORMAT, (
    _native_archive_bridge.fitted("components_"),
    *_native_archive_bridge.head(_PCA_FORMAT),
    ("str", "svd_solver", "svd_solver"),
    ("array", "components", "components_"),
    ("array", "mean", "mean_"),
    ("array", "singular_values", "singular_values_"),
    ("array", "explained_variance", "explained_variance_"),
    ("array", "explained_variance_ratio", "explained_variance_ratio_"),
    ("f8", "noise_variance", (("float", "noise_variance_"),)),
    ("i8", "meta", (("int", "n_components_"), ("int", "n_features_in_"),
                    ("int", "n_samples_"), ("bool", "whiten"))),
))
_native_archive_bridge.register(TruncatedSVD, _TSVD_FORMAT, (
    _native_archive_bridge.fitted("components_"),
    *_native_archive_bridge.head(_TSVD_FORMAT),
    ("str", "algorithm", "algorithm"),
    ("array", "components", "components_"),
    ("array", "singular_values", "singular_values_"),
    ("i8", "meta", (("int", "n_components_"), ("int", "n_features_in_"))),
))


# scikit-learn keeps more estimators in this module than it defines; they resolve from
# the package at first use (`_sklearn_modules`), so `from mojolearn.decomposition import ...`
# takes the same names as scikit-learn's module.
from ._sklearn_modules import LAYOUT as _SKLEARN_LAYOUT, lazy_exports as _lazy_exports  # noqa: E402
__getattr__ = _lazy_exports("decomposition")


# The compiled PCA owner, connected once at import: every public PCA
# operation (fit, fit_transform, transform, inverse_transform) is one call
# into it (bindings/pca_native.mojo).
_PCA_ENTRY, _PCA_CONTEXT = _native_pca_bridge.connect_public(PCA)

# The compiled TruncatedSVD owner, connected once at import: fit,
# fit_transform, transform and inverse_transform are one call each into it
# (bindings/tsvd_native.mojo).
_TSVD_ENTRY, _TSVD_CONTEXT = _native_tsvd_bridge.connect_public(TruncatedSVD)
