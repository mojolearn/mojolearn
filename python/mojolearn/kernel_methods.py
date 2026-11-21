# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Kernel methods: kernel ridge regression and kernel feature approximations.

`KernelRidge`, `Nystroem` and `RBFSampler` follow scikit-learn's estimators of
the same names: every parameter means what scikit-learn's parameter of that
name means, and an invalid value raises an error that names the parameter.
Inputs are converted to float32; sparse input is densified.

`KernelRidge` and `Nystroem` support the kernels 'linear', 'poly' (or
'polynomial'), 'rbf', 'sigmoid', 'laplacian', 'cosine', 'chi2' and
'additive_chi2'. `KernelRidge` also accepts 'precomputed'; `Nystroem` refuses
it with `ValueError`. `RBFSampler` approximates the RBF kernel only.
`gamma=None` is scikit-learn's `1 / n_features` (1.0 for 'chi2').

A fitted model keeps every array it computed, so it can be inspected, saved
with `save` and loaded with `load`; `mojolearn.host_model(path)` predicts or
transforms from a saved model on a CPU without a GPU.
"""

# Maintainer notes (moved from the module docstring):
# The Python door of `kernel_methods/estimator.mojo` through
# `bindings/_mojolearn_kernel_methods.mojo` (workstream D, 2026-09-14).
# `kernel_methods/README.md` carries the parameter mapping to scikit-learn;
# every refusal is raised by name, here for the shapes and the spellings, on the
# Mojo host for the values (`km_validate_matrix`, `km_validate_kernel_params`,
# the alpha, gamma, degree and n_components refusals, DEVIATION 1686).
# THE MODEL IS ITS ARRAYS. Fit returns every field `kernel_methods/
# estimator.mojo`'s structs carry, so a fitted object can be read, saved and
# hashed, and predict/transform send the same arrays back down to rebuild the
# struct. `Nystroem` keeps its eigenvalues, eigenvectors and Jacobi sweep count,
# which scikit-learn discards, because the lane's identity argument is which of
# those moved (the estimator's header, point 3).
# The kernels are `svm/impl/svm_parameter.mojo`'s codes plus the lane's
# laplacian (and cosine, chi2, additive_chi2). `gamma=None` is scikit-learn's
# `1 / n_features`, one Python division, correctly rounded on every host.
# No speed claim: the lane has no published number and this door adds none.

from . import _native_archive_bridge, _native_family_bridge
from ._mode import NumericModeMixin as _NumericModeMixin
from ._native_robust_bridge import native_method as _native_method

#: The saved-model formats (lane/inference-linear-svm, 2026-09-15):
#: `mojolearn.host_model(path)` predicts or transforms from each on a CPU.
_KERNEL_RIDGE_FORMAT = "mojolearn-kernel-ridge-1"
_NYSTROEM_FORMAT = "mojolearn-nystroem-1"
_RBF_SAMPLER_FORMAT = "mojolearn-rbf-sampler-1"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native_door.mojo
# writes the schema registered after each class; host_archive_native.mojo
# builds the loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()
# The compiled family owner (bindings/kernel_methods_native_door.mojo), connected
# once at import: KernelRidge `fit`/`predict` and Nystroem `fit`/`transform`/
# `fit_transform` are one native call each. The native lifecycle selects the
# binding and owns fitted checks, feature metadata and finished public results;
# the fixed tables it reads are connected below `_MODE_CODE`/`_KERNELS`.
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()


#: `checks/numerics.mojo` codes, duplicated on purpose (the GP's reason):
#: the read-back must not share a table with the thing it checks.
_MODE_CODE = {"fast": 0, "identical": 1}

#: `kernel_methods/checks/kernel_matrix.mojo`'s KM_KERNEL_* codes:
#: `svm/impl/svm_parameter.mojo`'s enumeration (0 to 4) plus the lane's
#: laplacian at 5. A renumbering on either side is a WRONG KERNEL, not a
#: failure, which is why the surface test plants a fixture whose answer
#: moves if the codes move.
_KERNEL_LINEAR = 0
_KERNEL_POLYNOMIAL = 1
_KERNEL_RBF = 2
_KERNEL_SIGMOID = 3
_KERNEL_LAPLACIAN = 5
_KERNEL_COSINE = 6
_KERNEL_CHI2 = 7
_KERNEL_ADDITIVE_CHI2 = 8

_KERNELS = {
    "linear": _KERNEL_LINEAR,
    "poly": _KERNEL_POLYNOMIAL,
    "polynomial": _KERNEL_POLYNOMIAL,
    "rbf": _KERNEL_RBF,
    "sigmoid": _KERNEL_SIGMOID,
    "laplacian": _KERNEL_LAPLACIAN,
    "cosine": _KERNEL_COSINE,
    "chi2": _KERNEL_CHI2,
    "additive_chi2": _KERNEL_ADDITIVE_CHI2,
}


_native_family_bridge.tables(family_km_kernels=dict(_KERNELS), family_km_mode_codes=dict(_MODE_CODE))


class _KernelMethodBase(_NumericModeMixin):
    _BINDING = "_mojolearn_kernel_methods"
    # The door takes the kernel methods module and the metrics module from
    # the estimator's binding route and asks the handed module for its
    # resident entries (a host module has none: the per-call entries run):
    # device="cpu" runs beside a GPU set.
    _mojolearn_device = True
    _WHERE = "kernel method"


# Maintainer notes: `alpha` IS the Cholesky ridge (DEVIATION 1660); a kernel
# matrix that does not factor is refused by name, never solved by a
# least-squares fallback (DEVIATION 1662). Negative or NaN alpha and
# non-positive gamma are refused on the Mojo host. The cosine and chi2 kernels
# are each cell one pinned chain. `info_` is LAPACK's info from the
# factorization.
class KernelRidge(_KernelMethodBase):
    """Kernel ridge regression.

    Follows `sklearn.kernel_ridge.KernelRidge` (and cuML's). Fitting forms
    the kernel matrix `K` of the training rows, adds `alpha` to its
    diagonal, factors it with a Cholesky factorization and solves for the
    dual coefficients; prediction is `K(X, X_fit_) @ dual_coef_`. The
    training matrix is kept on the model because prediction needs it.

    If `K + alpha * I` cannot be factored at the given `alpha`, `fit`
    raises an error rather than falling back to a least-squares solve;
    increase `alpha` in that case. Inputs and outputs are float32. Supports
    the `'identical'` (default) and `'fast'` numeric modes; `'fast'` is
    available on Apple GPUs only.

    Parameters
    ----------
    alpha : float, default 1.0
        Regularization strength, added to the kernel matrix diagonal. Must
        be >= 0; zero is accepted, negative or NaN is refused.
    kernel : str, default 'linear'
        One of 'linear', 'poly', 'rbf', 'sigmoid', 'laplacian', 'cosine',
        'chi2', 'additive_chi2' or 'precomputed'.
        'cosine', 'chi2' and 'additive_chi2' are
        scikit-learn's pairwise kernels; both chi2 kernels refuse negative
        values in `X` with `ValueError`, as scikit-learn's do. With
        'precomputed', `X` is the n x n kernel matrix at fit and the
        q x n cross-kernel against the training rows at predict, used as
        given (symmetry is not checked).
    gamma : float or None, default None
        Kernel coefficient for 'poly', 'rbf', 'sigmoid', 'laplacian' and
        the chi2 kernels. None means `1 / n_features` (1.0 for 'chi2', its
        scikit-learn default). Must be positive where the kernel reads it.
    degree : int, default 3
        Degree of the 'poly' kernel.
    coef0 : float, default 1.0
        Independent term of the 'poly' and 'sigmoid' kernels.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator; None uses the process default.
        ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    dual_coef_ : Array of shape (n_samples,) or (n_samples, n_targets), float32
        Dual coefficients, 1-D when `y` was 1-D.
    X_fit_ : Array of shape (n_samples, n_features), float32
        The training data, used by `predict`.
    info_ : int
        Status of the factorization; always 0 on a fitted model.
    n_features_in_ : int
        Number of features seen during `fit`.
    n_targets_ : int
        Number of regression targets.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X.sum(axis=1)
    >>> model = mojolearn.KernelRidge(alpha=1.0, kernel="rbf").fit(X, y)
    >>> pred = model.predict(X[:5])
    """

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "regressor"

    _WHERE = "KernelRidge"

    # Member ABI family 7 (KernelRidge), kind 0: core/kernel_ridge_member.mojo.
    _mojolearn_member = (7, 0)

    def __init__(self, alpha=1.0, kernel="linear", gamma=None, degree=3, coef0=1.0):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    # Maintainer notes: `kernel_ridge_fit_host`. `sample_weight` is
    # scikit-learn's `_solve_cholesky_kernel` weighted arm: `sw = sqrt(w)`
    # (binary64, rounded once to float32), `y * sw`, `K * outer(sw, sw)`,
    # then `dual_coef * sw` (float32 on both columns, DEVIATION 1688).
    @_native_method
    def fit(self, X, y, sample_weight=None):
        """Fit the kernel ridge model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32. With
            `kernel='precomputed'`, the square n x n kernel matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        sample_weight : float or array-like of shape (n_samples,), default None
            Per-sample weights, finite and >= 0, applied as scikit-learn
            does. None weights every sample equally.

        Returns
        -------
        self : KernelRidge
            The fitted estimator.
        """
        return _NATIVE(self, "family_km_krr_fit", (None, X, y, sample_weight), _NATIVE_CONTEXT)

    # Maintainer note: `K(X, X_fit_) . dual_coef_` through the identical GEMM
    # (DEVIATION 1680).
    @_native_method
    def predict(self, X):
        """Predict with the fitted model: `K(X, X_fit_) @ dual_coef_`.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query data. With `kernel='precomputed'`, the kernel between the
            queries and the training rows, shape (n_queries, n_samples).

        Returns
        -------
        Array of shape (n_queries,) or (n_queries, n_targets), float32
            Predictions, 1-D when the fitted `y` was 1-D.
        """
        return _NATIVE(self, "family_km_krr_predict", (None, X), _NATIVE_CONTEXT)

    @_native_method
    def score(self, X, y, sample_weight=None):
        """Coefficient of determination R^2 of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True targets.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            R^2, averaged uniformly over outputs; 1.0 is a perfect fit.
        """
        return _NATIVE(self, "family_km_krr_score",
                       (None, X, y, sample_weight),
                       _NATIVE_CONTEXT)

    # The fitted X and dual live on the device after the first predict,
    # under `_resident_holder` (bindings/kernel_methods_native_door.mojo). A
    # pickle or a deepcopy carries no device handle; the copy uploads its
    # own at its first predict. Dropping an instance releases the handle
    # through the native finalizer connected below the classes.
    def __getstate__(self):
        return _NATIVE(self, "family_km_getstate", (), _NATIVE_CONTEXT)

    def __setstate__(self, state):
        _NATIVE(self, "family_km_setstate", (state,), _NATIVE_CONTEXT)

    # Maintainer note, file layout: `X_fit_` and `dual_coef_` (flat,
    # `n * n_targets`) as fitted, `hyper` `<f8` [gamma, coef0, alpha] as the fit
    # resolved them, `meta` `<i8` [n, n_features, n_targets, kernel code, degree,
    # info, 1-D target] and the numeric mode.
    def save(self, path):
        """Save the fitted model to `path` as an `.npz` file.

        The file holds the training data, the dual coefficients, the kernel
        parameters as resolved at fit and the numeric mode. Load it with
        `KernelRidge.load`, or predict from it on a CPU without a GPU with
        `mojolearn.host_model(path)`.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    @_native_method  # the native archive loader publishes the fitted state: no Python _loader wrapper
    def load(cls, path):
        """Load a model written by `save`.

        Parameters
        ----------
        path : str or path-like
            File written by `KernelRidge.save`.

        Returns
        -------
        KernelRidge
            A fitted model, ready to predict, with the saved numeric mode.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


# Maintainer notes: `random_state` seeds `km_basis_indices`, the device
# permutation. `normalization_` is NOT bitwise symmetric (DEVIATION 1674);
# transform uses its transpose, scikit-learn's arm. `eigenvalues_` are clipped
# at 1e-12 (DEVIATION 1670). n_components > n_samples is refused on the Mojo
# host. The eigenvalues, eigenvectors and sweep count are kept (scikit-learn
# discards them) because they are part of the identity argument.
class Nystroem(_KernelMethodBase):
    """Approximate a kernel map with the Nystroem method.

    Follows `sklearn.kernel_approximation.Nystroem`. `fit` picks
    `n_components` training rows at random as the basis, forms their
    kernel matrix and computes its inverse square root with a Jacobi
    eigensolver on the device; `transform` maps rows to
    `K(X, components_) @ normalization_.T`, a feature space whose inner
    products approximate the kernel. Inputs and outputs are float32.
    Supports the `'identical'` (default) and `'fast'` numeric modes;
    `'fast'` is available on Apple GPUs only.

    Parameters
    ----------
    kernel : str, default 'rbf'
        One of 'linear', 'poly', 'rbf', 'sigmoid', 'laplacian', 'cosine',
        'chi2' or 'additive_chi2'. 'precomputed' is refused with
        `ValueError`. Both chi2 kernels refuse negative values in `X`.
    gamma : float or None, default None
        Kernel coefficient. None means `1 / n_features` (1.0 for 'chi2').
    degree : int, default 3
        Degree of the 'poly' kernel.
    coef0 : float, default 1.0
        Independent term of the 'poly' and 'sigmoid' kernels.
    n_components : int, default 100
        Number of basis rows. Must be positive; a value above `n_samples`
        warns and uses `n_samples` (the full kernel).
    random_state : int, default 0
        Seed of the basis selection.
    n_jobs : int or None, default None
        Accepted for scikit-learn compatibility; ignored. The work runs on the
        device, not in host worker processes.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator; None uses the process default.
        ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    components_ : Array of shape (n_components, n_features), float32
        The basis rows.
    component_indices_ : Array of shape (n_components,), int32
        Indices of the basis rows in the training data.
    normalization_ : Array of shape (n_components, n_components), float32
        Inverse square root of the basis kernel matrix.
    eigenvalues_ : Array of shape (n_components,), float32
        Absolute eigenvalues of the basis kernel matrix (its singular
        values), descending. A negative eigenvalue keeps its sign in the
        right factor of `normalization_`, as scikit-learn's SVD does.
    eigenvectors_ : Array of shape (n_components, n_components), float32
        Eigenvectors of the basis kernel matrix.
    sweeps_ : int
        Number of Jacobi sweeps the eigensolver ran.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> features = mojolearn.Nystroem(n_components=64).fit_transform(X)
    """

    _WHERE = "Nystroem"

    # Member ABI family 8 (KernelApprox), kind 0: core/kernel_approx_member.mojo.
    _mojolearn_member = (8, 0)

    # scikit-learn's no-op keywords, at the class too: a model the archive door loads
    # (`cls.__new__`, bindings/host_archive_native_door.mojo) never ran __init__, and its
    # get_params reads these defaults (as `device` does).
    n_jobs = None

    def __init__(self, kernel="rbf", gamma=None, degree=3, coef0=1.0, n_components=100, random_state=0, *,
                 n_jobs=None):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs

    @_native_method
    def fit(self, X, y=None):
        """Choose the basis rows and compute the normalization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : Nystroem
            The fitted transformer.
        """
        return _NATIVE(self, "family_km_nys_fit", (None, X, y), _NATIVE_CONTEXT)

    @_native_method
    def transform(self, X):
        """Map `X` to the approximate kernel feature space.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_features)
            Data to transform.

        Returns
        -------
        Array of shape (n_rows, n_components), float32
            `K(X, components_) @ normalization_.T`.
        """
        return _NATIVE(self, "family_km_nys_transform", (None, X), _NATIVE_CONTEXT)

    @_native_method
    def fit_transform(self, X, y=None):
        """Fit to `X`, then transform `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        Array of shape (n_samples, n_components), float32
            The transformed training data.
        """
        return _NATIVE(self, "family_km_nys_fit_transform", (None, X, y), _NATIVE_CONTEXT)

    # Maintainer note, file layout: every model array as fitted, `hyper` `<f8`
    # [gamma, coef0] as the fit resolved them, `meta` `<i8` [n_components,
    # n_features, kernel code, degree, seed, sweeps] and the numeric mode.
    def save(self, path):
        """Save the fitted transformer to `path` as an `.npz` file.

        Load it with `Nystroem.load`, or transform from it on a CPU without
        a GPU with `mojolearn.host_model(path)`.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the transformer has not been fitted.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    @_native_method  # the native archive loader publishes the fitted state: no Python _loader wrapper
    def load(cls, path):
        """Load a transformer written by `save`.

        Parameters
        ----------
        path : str or path-like
            File written by `Nystroem.save`.

        Returns
        -------
        Nystroem
            A fitted transformer, ready to transform, with the saved
            numeric mode.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


# Maintainer notes: gamma <= 0 is refused on the Mojo host. gamma='scale' uses
# the EXACT variance of X's float32 cells, the reciprocal rounded once
# (`_scale_gamma.scale_gamma`). `sigma_` and `scale_` are computed once on the
# host (DEVIATION 1678).
class RBFSampler(_KernelMethodBase):
    """Approximate the RBF kernel with random Fourier features.

    Follows `sklearn.kernel_approximation.RBFSampler`. `fit` draws random
    weights and offsets from `random_state`; `transform` maps rows to
    `scale_ * cos(X @ random_weights_ + random_offset_)`, a feature space
    whose inner products approximate `exp(-gamma * ||x - y||^2)`. `fit`
    reads only the number of features of `X` (and its variance when
    `gamma='scale'`). Inputs and outputs are float32. Supports the
    `'identical'` (default) and `'fast'` numeric modes; `'fast'` is
    available on Apple GPUs only.

    Parameters
    ----------
    gamma : float or 'scale', default 1.0
        RBF kernel coefficient; must be positive. 'scale' is scikit-learn's
        `1 / (n_features * X.var())` (1.0 when the variance is zero),
        computed from the exact variance of the float32 data.
    n_components : int, default 100
        Number of random features. Must be positive.
    random_state : int, default 0
        Seed of the random draws.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator; None uses the process default.
        ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    random_weights_ : Array of shape (n_features, n_components), float32
        The random projection weights.
    random_offset_ : Array of shape (n_components,), float32
        The random phase offsets.
    sigma_ : float
        `sqrt(2 * gamma)`, the standard deviation of the weights.
    scale_ : float
        `sqrt(2 / n_components)`, the output scale.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> features = mojolearn.RBFSampler(gamma=0.5, n_components=256).fit_transform(X)
    """

    _WHERE = "RBFSampler"

    # Member ABI family 8 (KernelApprox), kind 1: core/kernel_approx_member.mojo.
    _mojolearn_member = (8, 1)

    def __init__(self, gamma=1.0, n_components=100, random_state=0):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    @_native_method
    def fit(self, X, y=None):
        """Draw the random weights and offsets.

        Reads only `X.shape[1]` (and, for `gamma='scale'`, the variance of
        `X`), as scikit-learn does.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : RBFSampler
            The fitted transformer.
        """
        return _NATIVE(self, "family_km_rbf_fit", (None, X, y), _NATIVE_CONTEXT)

    @_native_method
    def transform(self, X):
        """Map `X` to the random Fourier feature space.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_features)
            Data to transform.

        Returns
        -------
        Array of shape (n_rows, n_components), float32
            `scale_ * cos(X @ random_weights_ + random_offset_)`.
        """
        return _NATIVE(self, "family_km_rbf_transform", (None, X), _NATIVE_CONTEXT)

    @_native_method
    def fit_transform(self, X, y=None):
        """Fit to `X`, then transform `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        Array of shape (n_samples, n_components), float32
            The transformed training data.
        """
        return _NATIVE(self, "family_km_rbf_fit_transform", (None, X, y), _NATIVE_CONTEXT)

    # Maintainer note, file layout: the weights and offsets as drawn, `hyper`
    # `<f8` [gamma, sigma_, scale_], `meta` `<i8` [n_features, n_components,
    # seed] and the numeric mode.
    def save(self, path):
        """Save the fitted sampler to `path` as an `.npz` file.

        Load it with `RBFSampler.load`, or transform from it on a CPU
        without a GPU with `mojolearn.host_model(path)`.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the sampler has not been fitted.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    @_native_method  # the native archive loader publishes the fitted state: no Python _loader wrapper
    def load(cls, path):
        """Load a sampler written by `save`.

        Parameters
        ----------
        path : str or path-like
            File written by `RBFSampler.save`.

        Returns
        -------
        RBFSampler
            A fitted sampler with the saved weights (nothing is drawn
            again), ready to transform, with the saved numeric mode.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


_bridge = _native_archive_bridge
# Every save writes its arrays, then the format, the class name and the
# numeric mode (the 0.8.55 writer's refusal order; members are written sorted).
_bridge.register(KernelRidge, _KERNEL_RIDGE_FORMAT, (
    _bridge.fitted("dual_coef_"),
    ("array", "x_fit", "X_fit_"),
    ("flat", "dual", "dual_coef_"),
    ("f8", "hyper", (("float", ("_kernel_params", 2)), ("float", ("_kernel_params", 3)),
                     ("float", ("_kernel_params", 4)))),
    ("i8", "meta", (("dim", "X_fit_", 0), ("int", "n_features_in_"), ("int", "n_targets_"),
                    ("int", ("_kernel_params", 0)), ("int", ("_kernel_params", 1)),
                    ("int", "info_"), ("bool", "_squeeze"))),
    *_bridge.head(_KERNEL_RIDGE_FORMAT),
))
_bridge.register(Nystroem, _NYSTROEM_FORMAT, (
    _bridge.fitted("components_"),
    ("array", "components", "components_"),
    ("array", "component_indices", "component_indices_"),
    ("array", "normalization", "normalization_"),
    ("array", "eigenvalues", "eigenvalues_"),
    ("array", "eigenvectors", "eigenvectors_"),
    ("f8", "hyper", (("float", ("_kernel_params", 2)), ("float", ("_kernel_params", 3)))),
    ("i8", "meta", (("dim", "components_", 0), ("dim", "components_", 1),
                    ("int", ("_kernel_params", 0)), ("int", ("_kernel_params", 1)),
                    ("int", ("_kernel_params", 4)), ("int", "sweeps_"))),
    *_bridge.head(_NYSTROEM_FORMAT),
))
_bridge.register(RBFSampler, _RBF_SAMPLER_FORMAT, (
    _bridge.fitted("random_weights_"),
    ("array", "random_weights", "random_weights_"),
    ("array", "random_offset", "random_offset_"),
    ("f8", "hyper", (("float", ("_params", 0)), ("float", "sigma_"), ("float", "scale_"))),
    ("i8", "meta", (("dim", "random_weights_", 0), ("dim", "random_weights_", 1),
                    ("int", ("_params", 1)))),
    *_bridge.head(_RBF_SAMPLER_FORMAT),
))
del _bridge


# Dropping a fitted KernelRidge releases its resident X and dual through the
# minting binding's `kernel_ridge_resident_release` (bindings/
# host_archive_lifetime.mojo "entry" plans), held in `_resident_holder["index"]`.
_native_archive_bridge.lifetime(KernelRidge, "kernel_ridge_resident_release", "_resident_holder")

# The classes the multi-GPU kernel-method drivers admit
# (bindings/kernel_methods_native_door.mojo `family_km_par_*`).
_native_family_bridge.tables(family_km_par_classes=(KernelRidge, Nystroem))


__all__ = ["KernelRidge", "Nystroem", "RBFSampler"]

# Resolve optional mode images at import; public methods enter native admission once.
_native_family_bridge.connect_bindings("_mojolearn_kernel_methods", "_mojolearn_metrics")

# Import-time schema for the native kernel lifecycle; no runtime reflection.
_native_family_bridge.tables(family_km_public_parameters={
    KernelRidge: ("alpha", "kernel", "gamma", "degree", "coef0", "numeric_mode"),
    Nystroem: ("kernel", "gamma", "degree", "coef0", "n_components", "random_state", "numeric_mode"),
    RBFSampler: ("gamma", "n_components", "random_state", "numeric_mode"),
})
