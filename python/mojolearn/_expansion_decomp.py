# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE DECOMP LANE'S PUBLIC DOOR.

Owned by the `decomp` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`; a name
that is already public is refused at import.

Every public operation here is one call into its compiled native owner
(bindings/*_native.mojo), which admits the inputs, runs the x_decomp cells
through the image's native table and packs the result. This file holds the
estimator classes, their parameters and docstrings, and the import-time
connection of those owners.
"""
from . import _native_family_bridge, _native_runtime_manifold_bridge
from ._mode import SklearnTagsMixin, available_if, _PARAMS_METADATA, _PARAMS_METADATA_CONTEXT
from ._native_declared_params import ENTRY as _DECLARED_PARAMS_SET, CONTEXT as _DECLARED_PARAMS_CONTEXT
from ._mode import RegressorScoreMixin


__all__ = ["IncrementalPCA", "GaussianRandomProjection", "SparseRandomProjection", "johnson_lindenstrauss_min_dim",
           "NMF", "FastICA", "FactorAnalysis",
           "lu_factor", "lu_solve", "solve", "lstsq", "randomized_svd",
           "PLSRegression", "PLSCanonical", "CCA",
           "DictionaryLearning", "MiniBatchDictionaryLearning", "SparsePCA", "MiniBatchSparsePCA", "sparse_encode", "SparseCoder",
           "LatentDirichletAllocation", "Isomap", "MDS", "ClassicalMDS", "LocallyLinearEmbedding",
           "MinCovDet", "EllipticEnvelope", "AlternatingLeastSquares"]

_BINDING = "_mojolearn_x_decomp"


# TOMBSTONE: the Python decomp kit (`_M`, `_Kit`, the host buffer pool, `_vstack`/`_hstack`,
# `_dev_gather`, `_host_kit`, `_OP`, `_f32`, `_is_sparse`, `_mode`, `_norm`, `_tsqr_lstsq_core`)
# deleted 2026-11-11 by lane pyout-decomp-linalg: no product caller remained (every public
# operation is one call into its native owner; the native twins are bindings/decomp_kit_native.mojo,
# mani_kit.mojo and linalg_ops_kit.mojo). Code recoverable at 6f82d0255d.


# A device-resident matrix id of the GPU binding: lu_factor's resident pair
# (`_LUPair._dev`) holds its A, LU and piv as these, assembled natively by
# bindings/linalg_decomp_native.mojo, and frees them through the binding.
class _DevBuf:
    """A device-resident matrix of the GPU binding (x_decomp/resident.mojo):
    a pooled buffer id, returned to the pool when this object dies."""
    __slots__ = ("b", "id", "n")

    def __init__(self, b, n):
        self.b, self.n = b, n
        self.id = b.x_decomp_dev_alloc(max(n, 1))

    def __del__(self):
        try:
            self.b.x_decomp_dev_free(self.id)
        except Exception:     # interpreter shutdown: the pool dies with the process
            pass


class _Base(SklearnTagsMixin):
    _parameters = ()

    def get_params(self, deep=True):
        """Return this estimator's constructor parameters.

        Parameters
        ----------
        deep : bool, default True
            Accepted for scikit-learn compatibility; it has no effect.

        Returns
        -------
        dict
            Parameter name to current value.
        """
        # The declared decomposition schema has always ignored `deep`.
        return _PARAMS_METADATA(self, False, _PARAMS_METADATA_CONTEXT)

    def set_params(self, **params):
        """Set constructor parameters by keyword.

        Raises
        ------
        ValueError
            If a keyword is not a parameter of this estimator.

        Returns
        -------
        self
        """
        return _DECLARED_PARAMS_SET(self, params, _DECLARED_PARAMS_CONTEXT)

    # Only a class with a `transform` has `fit_transform` (scikit-learn's
    # TransformerMixin): a classifier or detector on this base has neither.
    @available_if(lambda self: hasattr(self, "transform"))
    def fit_transform(self, X, y=None):
        """Fit to X, then return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return self.fit(X, y).transform(X)

    def _check(self, attr="components_"):
        if not hasattr(self, attr):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit first")


# ================================================================ IncrementalPCA
# Every public IncrementalPCA operation is one call into the compiled owner
# (bindings/ipca_native.mojo, drivers x_decomp/ipca_owner.mojo).
# Reference: scikit-learn `decomposition/_incremental_pca.py`, `partial_fit`;
# the running mean and variance are `utils/extmath.py::_incremental_mean_and_var`.
# Each batch's SVD is the thin SVD of the stacked matrix sklearn builds,
# computed as the eigendecomposition of its d x d Gram matrix (the Jacobi
# eigh of decomposition/, device or host), then sklearn's `svd_flip` with
# `u_based_decision=False`.
class IncrementalPCA(_Base):
    """Incremental principal component analysis, fitted batch by batch.

    Follows scikit-learn's ``IncrementalPCA``: the mean, variance and
    components are updated one batch at a time, so a dataset larger than
    memory can be fitted through ``partial_fit``. Component signs follow
    scikit-learn's ``svd_flip`` on the rows of the components. A
    ``scipy.sparse`` input is densified one batch at a time in ``fit`` and
    whole in ``partial_fit`` and ``transform``.

    Parameters
    ----------
    n_components : int or None, default None
        Number of components to keep. None keeps ``n_features`` (or, in a
        later ``partial_fit``, the number already fitted). Must be between 1
        and ``n_features`` and at most the number of rows in each batch.
    whiten : bool, default False
        Divide the projected data by the square root of each component's
        explained variance.
    copy : bool, default True
        Accepted for scikit-learn compatibility; the input is never modified.
    batch_size : int or None, default None
        Rows per batch in ``fit``. None uses ``5 * n_features``.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes, sorted by decreasing explained variance.
    explained_variance_ : ndarray of shape (n_components,)
    explained_variance_ratio_ : ndarray of shape (n_components,)
    singular_values_ : ndarray of shape (n_components,)
    mean_ : ndarray of shape (n_features,)
    var_ : ndarray of shape (n_features,)
    noise_variance_ : float
    n_components_ : int
    n_samples_seen_ : int
    n_features_in_ : int
    batch_size_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.IncrementalPCA(n_components=3, batch_size=200).fit(X)
    >>> model.transform(X).shape
    (1000, 3)
    """
    # One native call per public method; its owner runs the whole lifecycle
    # (bindings/decomp_public_lifecycle_unpack.mojo), so `_mode` installs no wrappers.
    _mojolearn_native_public = True

    _parameters = ("n_components", "whiten", "copy", "batch_size", "numeric_mode")

    def __init__(self, n_components=None, *, whiten=False, copy=True, batch_size=None, numeric_mode=None):
        self.n_components, self.whiten, self.copy = n_components, whiten, copy
        self.batch_size, self.numeric_mode = batch_size, numeric_mode

    def fit(self, X, y=None):
        """Fit the model on X in batches of ``batch_size`` rows.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data. Any earlier fit is discarded.
        y : ignored

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is empty or ``n_components`` is out of range.
        """
        return _IPCA_CALL(self, "fit", (X,), _IPCA_CONTEXT)

    def partial_fit(self, X, y=None):
        """Update the model with one more batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The batch. ``n_features`` must match earlier batches and
            ``n_samples`` must be at least ``n_components``.
        y : ignored

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of features changed or ``n_components`` is out of
            range for this batch.
        """
        return _IPCA_CALL(self, "partial_fit", (X,), _IPCA_CONTEXT)

    def transform(self, X):
        """Project X onto the fitted components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If X has a different number of features than the fitted data.
        """
        return _IPCA_CALL(self, "transform", (X,), _IPCA_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit to X, then return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _IPCA_CALL(self, "fit_transform", (X,), _IPCA_CONTEXT)

    def inverse_transform(self, X):
        """Map projected data back to the original feature space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32
        """
        return _IPCA_CALL(self, "inverse_transform", (X,), _IPCA_CONTEXT)


# ================================================================ random projection
def johnson_lindenstrauss_min_dim(n_samples, *, eps=0.1):
    """Minimum dimension for a random projection that keeps distances within eps.

    Follows scikit-learn's ``johnson_lindenstrauss_min_dim`` for a single
    ``n_samples`` and ``eps`` (scalars only): ``4 log(n_samples) /
    (eps**2 / 2 - eps**3 / 3)``, truncated to an int.

    Parameters
    ----------
    n_samples : int
        Number of samples; must be greater than zero.
    eps : float, default 0.1
        Maximum distortion, strictly between 0 and 1.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If ``eps`` is not in (0, 1) or ``n_samples`` is not positive.

    Examples
    --------
    >>> import mojolearn
    >>> d = mojolearn.johnson_lindenstrauss_min_dim(1000, eps=0.5)
    """
    # DEVIATION 6900: eps ** 2, eps ** 3 correctly rounded and the pinned log,
    # not the platform pow / log (a host's last bit can move the floor), in
    # the compiled owner (bindings/jl_native_pack.mojo)
    return _JL_CALL(None, "min_dim", (n_samples, eps), _JL_CONTEXT)


# TOMBSTONE: _seed_of (helper of the deleted _rsvd_core) deleted 2026-11-08 by bddde2ea2e; code recoverable at 41ffcec665.
# Restore: git apply experiments/removed/dead_python_loops.patch


class _RandomProjection(_Base):
    """sklearn `random_projection.py::BaseRandomProjection`. The matrix is
    drawn from the lane's counter-based Philox stream (x_decomp/cells.mojo
    `rand_cell`), not numpy's generator: the same `random_state` gives the
    same matrix on every box, and a different one than sklearn's."""
    # One native call per public method; its owner runs the whole lifecycle
    # (bindings/decomp_public_lifecycle_unpack.mojo), so `_mode` installs no wrappers.
    _mojolearn_native_public = True


    # The fitted matrix lives in the native owner's `_rp_components` field
    # (bindings/rp_native_door.mojo); reading, assigning and pickling it are each
    # one native call.
    @property
    def components_(self):
        """The random projection matrix, shape (n_components, n_features)."""
        return _DECOMP_CALL(self, "components", (), _DECOMP_CONTEXT)

    @components_.setter
    def components_(self, v):
        return _DECOMP_CALL(self, "set_components", (v,), _DECOMP_CONTEXT)

    def __getstate__(self):
        return _DECOMP_CALL(self, "getstate", (), _DECOMP_CONTEXT)

    def fit(self, X, y=None):
        """Draw the projection matrix for X's shape.

        Only X's shape is used (and X must be finite); the values are not.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is empty or non-finite, ``n_components`` is not positive, or
            ``n_components="auto"`` gives a dimension larger than
            ``n_features``.
        TypeError
            If ``random_state`` is not None or an int.
        """
        return _DECOMP_CALL(self, "fit", (X,), _DECOMP_CONTEXT)

    def transform(self, X):
        """Project X: ``X @ components_.T``.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
            Always a dense array.

        Raises
        ------
        ValueError
            If X has the wrong number of features or is non-finite.
        """
        return _DECOMP_CALL(self, "transform", (X,), _DECOMP_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit to X, then return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "fit_transform", (X,), _DECOMP_CONTEXT)

    def inverse_transform(self, X):
        """Map projected data back with the pseudo-inverse of the components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32

        Raises
        ------
        ValueError
            If the model was fitted without ``compute_inverse_components=True``.
        """
        return _DECOMP_CALL(self, "inverse_transform", (X,), _DECOMP_CONTEXT)


# Box-Muller on the Philox stream.
class GaussianRandomProjection(_RandomProjection):
    """Reduce dimension with a Gaussian random matrix.

    Follows scikit-learn's ``GaussianRandomProjection``: each component entry
    is drawn from N(0, 1 / n_components).

    Parameters
    ----------
    n_components : int or "auto", default "auto"
        Target dimension. ``"auto"`` uses
        ``johnson_lindenstrauss_min_dim(n_samples, eps=eps)``.
    eps : float, default 0.1
        Distortion bound used by ``n_components="auto"`` through
        :func:`johnson_lindenstrauss_min_dim`.
    compute_inverse_components : bool, default False
        Compute ``inverse_components_`` (the pseudo-inverse of the
        components) in ``fit``, needed by ``inverse_transform``. Raises
        ValueError if the components are rank deficient.
    random_state : int or None, default None
        Seed of mojolearn's counter-based random generator; None means 0.
        The same seed gives the same matrix on every device, but not the
        matrix scikit-learn draws for that seed.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
    inverse_components_ : ndarray of shape (n_features, n_components)
        Only with ``compute_inverse_components=True``.
    n_components_ : int
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 64), dtype=np.float32)
    >>> model = mojolearn.GaussianRandomProjection(n_components=8, random_state=0).fit(X)
    >>> model.transform(X).shape
    (1000, 8)
    """
    _parameters = ("n_components", "eps", "compute_inverse_components", "random_state", "numeric_mode")

    def __init__(self, n_components="auto", *, eps=0.1, compute_inverse_components=False, random_state=None,
                 numeric_mode=None):
        self.n_components, self.eps = n_components, eps
        self.compute_inverse_components, self.random_state = compute_inverse_components, random_state
        self.numeric_mode = numeric_mode


# Achlioptas / Li et al.
class SparseRandomProjection(_RandomProjection):
    """Reduce dimension with a sparse random matrix.

    Follows scikit-learn's ``SparseRandomProjection``: each entry is
    ``+sqrt(1/density)/sqrt(n_components)`` or its negative with probability
    ``density/2`` each, and 0 otherwise. The components are stored as a dense
    array, and ``transform`` always returns a dense array.

    Parameters
    ----------
    n_components : int or "auto", default "auto"
        Target dimension. ``"auto"`` uses
        ``johnson_lindenstrauss_min_dim(n_samples, eps=eps)``.
    density : float or "auto", default "auto"
        Fraction of nonzero entries, in (0, 1]. ``"auto"`` is
        ``1 / sqrt(n_features)``. Out-of-range values raise ValueError.
    dense_output : bool, default False
        Accepted for scikit-learn compatibility; the output is always dense.
    eps : float, default 0.1
        Distortion bound used by ``n_components="auto"`` through
        :func:`johnson_lindenstrauss_min_dim`.
    compute_inverse_components : bool, default False
        Compute ``inverse_components_`` (the pseudo-inverse of the
        components) in ``fit``, needed by ``inverse_transform``. Raises
        ValueError if the components are rank deficient.
    random_state : int or None, default None
        Seed of mojolearn's counter-based random generator; None means 0.
        The same seed gives the same matrix on every device, but not the
        matrix scikit-learn draws for that seed.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
    inverse_components_ : ndarray of shape (n_features, n_components)
        Only with ``compute_inverse_components=True``.
    density_ : float
    n_components_ : int
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 64), dtype=np.float32)
    >>> model = mojolearn.SparseRandomProjection(n_components=8, random_state=0).fit(X)
    >>> model.transform(X).shape
    (1000, 8)
    """
    _parameters = ("n_components", "density", "eps", "dense_output", "compute_inverse_components",
                   "random_state", "numeric_mode")

    def __init__(self, n_components="auto", *, density="auto", eps=0.1, dense_output=False,
                 compute_inverse_components=False, random_state=None, numeric_mode=None):
        self.n_components, self.density, self.eps, self.dense_output = n_components, density, eps, dense_output
        self.compute_inverse_components, self.random_state = compute_inverse_components, random_state
        self.numeric_mode = numeric_mode


# ================================================================ NMF
# Reference: scikit-learn `decomposition/_nmf.py`: `_initialize_nmf`,
# `_fit_multiplicative_update` with `_multiplicative_update_w/_h` for
# beta_loss='frobenius', and `_fit_coordinate_descent` with
# `_cdnmf_fast.pyx::_update_cdnmf_fast`. The NNDSVD SVD is exact (through
# the Gram eigh) where sklearn calls randomized_svd. 'random' init and
# shuffle use the lane's Philox stream, not numpy's. A general float beta
# is refused by name.
class NMF(_Base):
    """Non-negative matrix factorization ``X ~ W @ H``.

    Follows scikit-learn's ``NMF``: finds non-negative ``W`` (n_samples,
    n_components) and ``H`` (n_components, n_features) minimizing the chosen
    beta divergence plus optional L1/L2 penalties. The NNDSVD initializations
    use an exact SVD rather than scikit-learn's randomized one, and random
    draws come from mojolearn's own generator, so results differ slightly
    from scikit-learn's for the same seed.

    Parameters
    ----------
    n_components : int, "auto" or None, default "auto"
        Number of components. ``"auto"`` or None uses ``n_features`` (with
        ``init="custom"``, the number of rows of the given ``H``).
    init : {"random", "nndsvd", "nndsvda", "nndsvdar", "custom"} or None, default None
        Initialization. None uses ``"nndsvda"`` when ``n_components <=
        min(n_samples, n_features)``, else ``"random"``. The NNDSVD variants
        need ``n_components <= min(n_samples, n_features)``. ``"custom"``
        takes ``W`` and ``H`` from ``fit_transform``.
    solver : {"cd", "mu"}, default "cd"
        Coordinate descent or multiplicative update.
    beta_loss : {"frobenius", "kullback-leibler", "itakura-saito"}, default "frobenius"
        Divergence to minimize; the last two need ``solver="mu"``. The
        numbers 2, 1 and 0 are accepted for the same three; any other float
        beta raises ValueError.
    tol : float, default 1e-4
        Stopping tolerance.
    max_iter : int, default 200
        Maximum number of iterations.
    random_state : int or None, default None
        Seed for ``"random"`` and ``"nndsvdar"`` initialization and for
        ``shuffle``; None means 0. Not scikit-learn's random stream.
    alpha_W : float, default 0.0
        Regularization strength on ``W``.
    alpha_H : float or "same", default "same"
        Regularization strength on ``H``; ``"same"`` uses ``alpha_W``.
    l1_ratio : float, default 0.0
        Mix of L1 (1.0) and L2 (0.0) regularization.
    verbose : int, default 0
        Accepted for scikit-learn compatibility; nothing is printed.
    shuffle : bool, default False
        Visit coordinates in random order in each ``"cd"`` sweep.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The factor ``H``.
    n_components_ : int
    n_iter_ : int
    reconstruction_err_ : float
        Beta divergence between X and ``W @ H`` (square-rooted, as in
        scikit-learn).
    n_features_in_ : int

    Raises
    ------
    ValueError
        If X has negative values, or an unsupported solver, init or
        beta_loss is requested.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.NMF(n_components=3, random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
    """
    # One native call per public method; its owner runs the whole lifecycle
    # (bindings/decomp_public_lifecycle_unpack.mojo), so `_mode` installs no wrappers.
    _mojolearn_native_public = True

    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"positive_only": True}
    _parameters = ("n_components", "init", "solver", "beta_loss", "tol", "max_iter", "random_state",
                   "alpha_W", "alpha_H", "l1_ratio", "verbose", "shuffle", "numeric_mode")

    def __init__(self, n_components="auto", *, init=None, solver="cd", beta_loss="frobenius", tol=1e-4,
                 max_iter=200, random_state=None, alpha_W=0.0, alpha_H="same", l1_ratio=0.0, verbose=0,
                 shuffle=False, numeric_mode=None):
        self.n_components, self.init, self.solver, self.beta_loss = n_components, init, solver, beta_loss
        self.tol, self.max_iter, self.random_state = tol, max_iter, random_state
        self.alpha_W, self.alpha_H, self.l1_ratio = alpha_W, alpha_H, l1_ratio
        self.verbose, self.shuffle, self.numeric_mode = verbose, shuffle, numeric_mode

    # ---- setup
    # ---- solvers
    def fit_transform(self, X, y=None, W=None, H=None):
        """Fit the model and return ``W``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Non-negative data.
        y : ignored
        W : array-like of shape (n_samples, n_components), optional
            Starting ``W``; used only with ``init="custom"``.
        H : array-like of shape (n_components, n_features), optional
            Starting ``H``; used only with ``init="custom"``.

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "fit_transform", (X, W, H), _DECOMP_CONTEXT)

    def fit(self, X, y=None, **params):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Non-negative data.
        y : ignored
        **params
            ``W`` and ``H`` for ``init="custom"``, as in ``fit_transform``.

        Returns
        -------
        self
        """
        return _DECOMP_CALL(self, "fit", (X, params), _DECOMP_CONTEXT)

    def transform(self, X):
        """Solve for ``W`` with the fitted ``H`` held fixed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Non-negative data.

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "transform", (X,), _DECOMP_CONTEXT)

    def inverse_transform(self, X):
        """Return ``X @ components_``, the data reconstructed from ``W``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            A ``W`` matrix.

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32
        """
        return _DECOMP_CALL(self, "inverse_transform", (X,), _DECOMP_CONTEXT)


# ================================================================ FastICA
# Reference: scikit-learn `decomposition/_fastica.py`: `_fit_transform`,
# `_ica_par`, `_ica_def`, `_sym_decorrelation`, `_gs_decorrelation`,
# `_logcosh`, `_exp`, `_cube`. The whitening SVD is the eigendecomposition
# of X^T X (both of sklearn's whiten_solver values take this route); a
# singular value below 10 float32 eps is clamped there, as sklearn's 'eigh'
# solver clamps it. w_init defaults to a draw from the lane's Philox
# stream, not numpy's. A callable `fun` is refused by name.
class FastICA(_Base):
    """Independent component analysis with the FastICA algorithm.

    Follows scikit-learn's ``FastICA``. Whitening always uses the
    eigendecomposition of ``X.T @ X`` (``whiten_solver`` is accepted but both
    values take this route), and the default starting matrix is drawn from
    mojolearn's own generator, so results can differ from scikit-learn's for
    the same seed.

    Parameters
    ----------
    n_components : int or None, default None
        Number of components; None, or a value above ``min(n_samples,
        n_features)``, uses ``min(n_samples, n_features)``. Ignored when
        ``whiten=False``.
    algorithm : {"parallel", "deflation"}, default "parallel"
        Estimate all components together, or one at a time.
    whiten : {"unit-variance", "arbitrary-variance"} or False, default "unit-variance"
        Whitening strategy; False assumes X is already whitened.
    fun : {"logcosh", "exp", "cube"}, default "logcosh"
        Contrast function. A callable is not supported.
    fun_args : dict or None, default None
        For ``"logcosh"``, ``{"alpha": a}`` with ``1 <= a <= 2`` (default 1.0).
    max_iter : int, default 200
        Maximum number of iterations.
    tol : float, default 1e-4
        Convergence tolerance.
    w_init : array-like of shape (n_components, n_components) or None, default None
        Starting unmixing matrix; None draws one from ``random_state``.
    whiten_solver : {"svd", "eigh"}, default "svd"
        Accepted for scikit-learn compatibility; see above.
    random_state : int or None, default None
        Seed for the default ``w_init``; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Unmixing matrix applied by ``transform``.
    mixing_ : ndarray of shape (n_features, n_components)
        Pseudo-inverse of ``components_``.
    mean_ : ndarray of shape (n_features,)
        Only when whitening.
    whitening_ : ndarray of shape (n_components, n_features)
        Only when whitening.
    n_iter_ : int
    n_features_in_ : int

    Raises
    ------
    ValueError
        For an unsupported ``algorithm``, ``whiten`` or ``fun``, ``alpha``
        outside [1, 2], or a ``w_init`` of the wrong shape.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> S = mojolearn.FastICA(n_components=3, random_state=0).fit_transform(X)
    """
    # One native call per public method; its owner runs the whole lifecycle
    # (bindings/decomp_public_lifecycle_unpack.mojo), so `_mode` installs no wrappers.
    _mojolearn_native_public = True

    _parameters = ("n_components", "algorithm", "whiten", "fun", "fun_args", "max_iter", "tol", "w_init",
                   "whiten_solver", "random_state", "numeric_mode")

    def __init__(self, n_components=None, *, algorithm="parallel", whiten="unit-variance", fun="logcosh",
                 fun_args=None, max_iter=200, tol=1e-4, w_init=None, whiten_solver="svd", random_state=None,
                 numeric_mode=None):
        self.n_components, self.algorithm, self.whiten, self.fun = n_components, algorithm, whiten, fun
        self.fun_args, self.max_iter, self.tol, self.w_init = fun_args, max_iter, tol, w_init
        self.whiten_solver, self.random_state, self.numeric_mode = whiten_solver, random_state, numeric_mode

    def fit_transform(self, X, y=None):
        """Fit the model and return the estimated sources.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "fit_transform", (X,), _DECOMP_CONTEXT)

    def fit(self, X, y=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _DECOMP_CALL(self, "fit", (X,), _DECOMP_CONTEXT)

    def transform(self, X, copy=True):
        """Recover the sources from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        copy : bool, default True
            Accepted for scikit-learn compatibility; X is never modified.

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "transform", (X,), _DECOMP_CONTEXT)

    def inverse_transform(self, X, copy=True):
        """Map sources back to the feature space with ``mixing_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
        copy : bool, default True
            Accepted for scikit-learn compatibility; X is never modified.

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32
        """
        return _DECOMP_CALL(self, "inverse_transform", (X,), _DECOMP_CONTEXT)


# ================================================================ FactorAnalysis
# Every public FactorAnalysis operation is one call into the compiled owner
# (bindings/fa_native_unpack.mojo, driver x_decomp/fa_owner.mojo).
# Reference: scikit-learn `decomposition/_factor_analysis.py`: `fit`,
# `transform`, `get_covariance`, `get_precision`, `score_samples`,
# `_ortho_rotation`. Each iteration's SVD of the scaled data is the QR +
# one-sided Jacobi SVD of decomposition/ (the eigh of the d x d Gram when
# n < d). svd_method='randomized' takes the same exact route (sklearn's
# randomized SVD is an approximation of it).
class FactorAnalysis(_Base):
    """Factor analysis: a linear Gaussian latent variable model.

    Follows scikit-learn's ``FactorAnalysis``, fitted by EM. Both
    ``svd_method`` values use an exact SVD at every iteration (scikit-learn's
    ``"randomized"`` is an approximation of it), so ``iterated_power`` and
    ``random_state`` have no effect.

    Parameters
    ----------
    n_components : int or None, default None
        Dimension of the latent space; None uses ``n_features``.
    tol : float, default 1e-2
        Stop when the log-likelihood gain falls below this.
    copy : bool, default True
        Accepted for scikit-learn compatibility; X is never modified.
    max_iter : int, default 1000
        Maximum number of EM iterations.
    noise_variance_init : array-like of shape (n_features,) or None, default None
        Starting noise variances; None uses ones.
    svd_method : {"lapack", "randomized"}, default "randomized"
        Accepted for scikit-learn compatibility; both run the exact SVD.
    iterated_power : int, default 3
        Accepted for scikit-learn compatibility; unused.
    rotation : {"varimax", "quartimax"} or None, default None
        Rotation applied to the fitted components.
    random_state : int or None, default 0
        Accepted for scikit-learn compatibility; unused.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Factor loadings.
    noise_variance_ : ndarray of shape (n_features,)
    mean_ : ndarray of shape (n_features,)
    loglike_ : list of float
        Log-likelihood at each iteration.
    n_iter_ : int
    n_features_in_ : int

    Raises
    ------
    ValueError
        For an unsupported ``svd_method`` or ``rotation``, or a
        ``noise_variance_init`` of the wrong length.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> fa = mojolearn.FactorAnalysis(n_components=2).fit(X)
    >>> Z = fa.transform(X)
    """
    # One native call per public method; its owner runs the whole lifecycle
    # (bindings/decomp_public_lifecycle_unpack.mojo), so `_mode` installs no wrappers.
    _mojolearn_native_public = True

    _parameters = ("n_components", "tol", "copy", "max_iter", "noise_variance_init", "svd_method",
                   "iterated_power", "rotation", "random_state", "numeric_mode")

    def __init__(self, n_components=None, *, tol=1e-2, copy=True, max_iter=1000, noise_variance_init=None,
                 svd_method="randomized", iterated_power=3, rotation=None, random_state=0, numeric_mode=None):
        self.n_components, self.tol, self.copy, self.max_iter = n_components, tol, copy, max_iter
        self.noise_variance_init, self.svd_method, self.iterated_power = noise_variance_init, svd_method, iterated_power
        self.rotation, self.random_state, self.numeric_mode = rotation, random_state, numeric_mode

    def fit(self, X, y=None):
        """Fit the model with EM.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _DECOMP_CALL(self, "fit", (X,), _DECOMP_CONTEXT)

    def transform(self, X):
        """Return the expected latent variables for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "transform", (X,), _DECOMP_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit to X, then return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "fit_transform", (X,), _DECOMP_CONTEXT)

    def get_covariance(self):
        """Return the model covariance ``components_.T @ components_ + diag(noise_variance_)``.

        Returns
        -------
        ndarray of shape (n_features, n_features), float32
        """
        return _DECOMP_CALL(self, "get_covariance", (), _DECOMP_CONTEXT)

    def get_precision(self):
        """Return the inverse of the model covariance.

        Returns
        -------
        ndarray of shape (n_features, n_features), float32

        Raises
        ------
        ValueError
            If the covariance is singular.
        """
        return _DECOMP_CALL(self, "get_precision", (), _DECOMP_CONTEXT)

    def score_samples(self, X):
        """Return the log-likelihood of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,), float32
        """
        return _DECOMP_CALL(self, "score_samples", (X,), _DECOMP_CONTEXT)

    def score(self, X, y=None):
        """Return the average log-likelihood of the samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        float
        """
        return _DECOMP_CALL(self, "score", (X,), _DECOMP_CONTEXT)


# ================================================================ TruncatedSVD explained variance
# ================================================================ PCA n_components='mle'
_LOG2, _LOGPI, _LOG2PI_D = 0.6931471805599453, 1.1447298858494002, 1.8378770664093453


# ================================================================ LU
# LAPACK getrf semantics. Partial pivoting on the largest |a[i, k]|, ties
# broken by the lowest row index. A zero pivot is kept (getrf's info > 0).
def lu_factor(a, *, numeric_mode=None):
    """LU factorization of a square matrix with partial pivoting.

    Follows ``scipy.linalg.lu_factor``: returns ``(lu, piv)``, where ``lu``
    holds the unit lower-triangular L (diagonal not stored) and U in one
    float32 matrix, and ``piv`` holds the 0-based row interchanges, applied
    in order. Pivoting picks the largest absolute value in the column; ties
    go to the lowest row. A zero pivot is kept and a ``RuntimeWarning`` is
    issued, as scipy does.

    Parameters
    ----------
    a : array-like of shape (n, n)
        The matrix to factor; must be finite.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this call. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Returns
    -------
    lu : ndarray of shape (n, n), float32
    piv : ndarray of shape (n,), int32
        Pass the pair to :func:`lu_solve`.

    Raises
    ------
    ValueError
        If ``a`` is not square, is empty, or contains NaN or infinity.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> A = np.random.default_rng(0).random((8, 8), dtype=np.float32) + 8 * np.eye(8, dtype=np.float32)
    >>> lu, piv = mojolearn.lu_factor(A)
    >>> x = mojolearn.lu_solve((lu, piv), np.ones(8, dtype=np.float32))
    """
    return _LINALG_CALL(None, "lu_factor", (a, numeric_mode), _LINALG_CONTEXT)


class _LUPair(tuple):
    """`lu_factor`'s (lu, piv), unpacked and indexed as the plain tuple
    scipy returns, plus `_qfix_a`: the float32 copy of A the factor came
    from (lane/apple-fast-q-linalg LU_QFIX). `lu_solve` uses it for one
    step of iterative refinement, x += LU^-1 (B - A x), the residual in
    float-float on the device. #: audit 2026-10-04: lu-solve / lu-factor
    synthetic relative_residual 3.26e-06 vs numpy 3.26e-08, torch-gpu
    8.23e-07; -D MOJOLEARN_LU_QOLD restores the unrefined solve. KEPT for
    quality 2026-10-04 (rab5-lu: 2.59e-6 -> 3.26e-8, +14-15% time)."""
    # bindings/linalg_decomp_native.mojo (`_lu_pair`) assembles the pair
    # lu_factor returns from its slots: `_qfix_a` an (n, n) float32 Array,
    # `_dev` the three `_DevBuf`s and the lu / piv arrays, or None.

    def __new__(cls, pair, a_m, dev=None):
        obj = super().__new__(cls, pair)
        obj._qfix_a = a_m
        # LU_FAST_RESIDENT: (device A, device LU, device piv, the lu and piv
        # arrays returned); `lu_solve` uses them only for this same lu / piv
        obj._dev = dev
        return obj


# LAPACK getrs semantics.
def lu_solve(lu_and_piv, b, *, trans=0, numeric_mode=None):
    """Solve a linear system from an LU factorization.

    Follows ``scipy.linalg.lu_solve``: solves ``A x = b`` (``trans=0``) or
    ``A.T x = b`` (``trans=1`` or ``2``; the same for a real matrix) with the
    pair returned by :func:`lu_factor`. A zero pivot gives 0 in that
    component of x, never infinity or NaN.

    Parameters
    ----------
    lu_and_piv : tuple (lu, piv)
        The output of :func:`lu_factor`.
    b : array-like of shape (n,) or (n, nrhs)
        Right-hand side.
    trans : {0, 1, 2}, default 0
        Which system to solve.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this call. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Returns
    -------
    x : ndarray of shape (n,) or (n, nrhs), float32
        Same shape as ``b``.

    Raises
    ------
    ValueError
        If ``trans`` is not 0, 1 or 2, ``piv`` holds an index outside
        [0, n), ``b`` has the wrong number of rows, or ``b`` is not finite.
    """
    return _LINALG_CALL(None, "lu_solve", (lu_and_piv, b, trans, numeric_mode), _LINALG_CONTEXT)


# lu_factor + lu_solve (gesv). Lane idn-dense-linalg: one binding entry
# (`x_decomp_lu_gesv`) when the binding routes there (`x_decomp_idn_flags`
# bit 1): the same launches with the factor and the pivots resident, so
# only A and B go up and X comes down; the same words, warning and refusals.
def solve(a, b, *, numeric_mode=None):
    """Solve the square linear system ``a @ x = b``.

    Follows ``numpy.linalg.solve``, computed by LU factorization with partial
    pivoting (as :func:`lu_factor` and :func:`lu_solve`). A singular matrix
    gives a ``RuntimeWarning`` rather than an exception.

    Parameters
    ----------
    a : array-like of shape (n, n)
    b : array-like of shape (n,) or (n, nrhs)
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this call. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Returns
    -------
    x : ndarray of shape (n,) or (n, nrhs), float32

    Raises
    ------
    ValueError
        If ``a`` is not square or ``b`` has the wrong number of rows.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> A = np.random.default_rng(0).random((8, 8), dtype=np.float32) + 8 * np.eye(8, dtype=np.float32)
    >>> x = mojolearn.solve(A, np.ones(8, dtype=np.float32))
    """
    return _LINALG_CALL(None, "solve", (a, b, numeric_mode), _LINALG_CONTEXT)


# TOMBSTONE: _orthonormal_cols (helper of the deleted _rsvd_core) deleted 2026-11-08 by bddde2ea2e; code recoverable at 41ffcec665.
# Restore: git apply experiments/removed/dead_python_loops.patch


# TOMBSTONE: _flip_u (orphan helper: the native linalg door flips the signs) deleted 2026-11-08 by bddde2ea2e; code recoverable at 41ffcec665.
# Restore: git apply experiments/removed/dead_python_loops.patch


# Halko et al.; RAFT `linalg/rsvd.cuh` is the same scheme. The Gaussian
# test matrix is the lane's Philox stream. Every power iteration
# re-orthonormalizes by the two-pass Householder-R solve whatever
# `power_iteration_normalizer` says: 'LU', 'QR' and 'none' span the same
# subspace, only the rounding of the basis differs. The small SVD of
# Q^T M is exact (Gram eigh).
def randomized_svd(M, n_components, *, n_oversamples=10, n_iter="auto", power_iteration_normalizer="auto",
                   transpose="auto", flip_sign=True, random_state=None, numeric_mode=None):
    """Truncated SVD by randomized range finding.

    Follows scikit-learn's ``utils.extmath.randomized_svd`` (Halko et al.).
    Every power iteration re-orthonormalizes with a QR step whatever
    ``power_iteration_normalizer`` says (all choices span the same
    subspace), and the random test matrix comes from mojolearn's own
    generator, so results differ slightly from scikit-learn's for the same
    seed.

    Parameters
    ----------
    M : array-like or scipy.sparse matrix of shape (n_samples, n_features)
    n_components : int
        Number of singular values and vectors; between 1 and
        ``min(n_samples, n_features)``.
    n_oversamples : int, default 10
        Extra random vectors used to sample the range.
    n_iter : int or "auto", default "auto"
        Power iterations. ``"auto"`` uses 7 when ``n_components < 0.1 *
        min(M.shape)``, else 4.
    power_iteration_normalizer : {"auto", "QR", "LU", "none"}, default "auto"
        Accepted for scikit-learn compatibility; see above.
    transpose : bool or "auto", default "auto"
        Work on ``M.T``; ``"auto"`` does so when M has more columns than rows.
    flip_sign : bool, default True
        Make the largest-magnitude entry of each column of U positive, as
        scikit-learn's ``svd_flip``.
    random_state : int or None, default None
        Seed of the random test matrix; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this call. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Returns
    -------
    U : ndarray of shape (n_samples, n_components), float32
    s : ndarray of shape (n_components,), float32
        Singular values, descending.
    Vt : ndarray of shape (n_components, n_features), float32

    Raises
    ------
    ValueError
        If ``n_components`` is out of range, ``power_iteration_normalizer`` is
        unknown, or M is numerically zero.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 50), dtype=np.float32)
    >>> U, s, Vt = mojolearn.randomized_svd(X, 5, random_state=0)
    """
    return _LINALG_CALL(None, "randomized_svd", (M, n_components, n_oversamples, n_iter, power_iteration_normalizer,
                                                    transpose, flip_sign, random_state, numeric_mode), _LINALG_CONTEXT)


# TOMBSTONE: _rsvd_core (dead Python power-iteration loop: randomized_svd routes to the native linalg door) deleted 2026-11-08 by bddde2ea2e; code recoverable at 41ffcec665.
# Restore: git apply experiments/removed/dead_python_loops.patch


# A tall a with N + nrhs <= 512 takes the blocked TSQR of [a | b] and the
# SVD of its small R (`_tsqr_lstsq_core`); any other shape the QR +
# one-sided Jacobi SVD of a (or of a^T).
def lstsq(a, b, rcond=None, *, numeric_mode=None):
    """Least-squares solution of ``a @ x = b``.

    Follows ``numpy.linalg.lstsq``, solved through the SVD of ``a``.
    Singular values at or below ``rcond * s.max()`` are treated as zero.

    Parameters
    ----------
    a : array-like of shape (M, N)
    b : array-like of shape (M,) or (M, K)
    rcond : float or None, default None
        Relative cutoff for small singular values. None uses float32
        machine epsilon times ``max(M, N)``.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this call. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Returns
    -------
    x : ndarray of shape (N,) or (N, K), float32
    residuals : ndarray of shape (K,) or (0,), float32
        Squared column norms of ``b - a @ x`` when ``rank == N < M``;
        otherwise empty.
    rank : int
        Effective rank of ``a``.
    s : ndarray of shape (min(M, N),), float32
        Singular values of ``a``, descending.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different numbers of rows.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> A = rng.random((100, 4), dtype=np.float32)
    >>> x, res, rank, s = mojolearn.lstsq(A, rng.random(100, dtype=np.float32))
    """
    return _LINALG_CALL(None, "lstsq", (a, b, rcond, numeric_mode), _LINALG_CONTEXT)


class _PLS(RegressorScoreMixin, _Base):
    """sklearn `cross_decomposition/_pls.py::_PLS` (NIPALS): `fit`,
    `_get_first_singular_vectors_power_method`, `_center_scale_xy`,
    `_get_first_singular_vectors_svd` (PLSCanonical algorithm='svd'),
    `_svd_flip_1d`, `transform`, `inverse_transform`, `predict`."""
    # The native PLS envelope preserves _mode/_sk_validate lifecycle and
    # public result handling; no Python decorators surround these operations.
    _mojolearn_native_public = True
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "regressor"
    _sklearn_target_tags = {"multi_output": True}
    _parameters = ("n_components", "scale", "max_iter", "tol", "copy", "numeric_mode")
    _deflation = "regression"
    _pmode = "A"

    # fit, fit_transform, transform, inverse_transform and predict are one
    # call each into the native PLS owner (bindings/pls_native.mojo), which
    # reads `_deflation` / `_pmode` from the class.
    def __init__(self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True, numeric_mode=None):
        self.n_components, self.scale, self.max_iter, self.tol, self.copy = n_components, scale, max_iter, tol, copy
        self.numeric_mode = numeric_mode

    def __sklearn_tags__(self):
        # scikit-learn's _PLS tags: a projection model, not scored as a
        # regressor by its estimator checks (poor_score)
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags

    def fit(self, X, Y):
        """Fit the model to X and Y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples,) or (n_samples, n_targets)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If Y is missing or ``n_components`` is out of range.
        """
        return _PLS_CALL(self, "fit", (X, Y, None), _PLS_CONTEXT)

    def transform(self, X, y=None, Y=None, copy=True):
        """Project X (and optionally Y) onto the fitted rotations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y, Y : array-like of shape (n_samples, n_targets), optional
            Targets to project as well (either name).
        copy : bool, default True
            Accepted for scikit-learn compatibility; inputs are never modified.

        Returns
        -------
        x_scores : ndarray of shape (n_samples, n_components), float32
        y_scores : ndarray of shape (n_samples, n_components), float32
            Returned only when Y is given.
        """
        return _PLS_CALL(self, "transform", (X, y, Y), _PLS_CONTEXT)

    def inverse_transform(self, X, y=None, Y=None):
        """Map scores back to the original X (and optionally Y) space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            X scores.
        y, Y : array-like of shape (n_samples, n_components), optional
            Y scores (either name).

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features), float32
        Y_reconstructed : ndarray of shape (n_samples, n_targets), float32
            Returned only when Y scores are given.
        """
        return _PLS_CALL(self, "inverse_transform", (X, y, Y), _PLS_CONTEXT)

    def predict(self, X, copy=True):
        """Predict targets for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        copy : bool, default True
            Accepted for scikit-learn compatibility; X is never modified.

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_targets), float32
            One-dimensional when the model was fitted with a 1-D Y.
        """
        return _PLS_CALL(self, "predict", (X,), _PLS_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit the model, then return ``transform(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_targets)

        Returns
        -------
        x_scores, y_scores : ndarray of shape (n_samples, n_components), float32
        """
        return _PLS_CALL(self, "fit_transform", (X, y, None), _PLS_CONTEXT)

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
        return _PLS_CALL(self, "score",
                         (X, y, sample_weight), _PLS_CONTEXT)


# NIPALS, mode A, regression deflation.
class PLSRegression(_PLS):
    """Partial least squares regression.

    Follows scikit-learn's ``PLSRegression`` (NIPALS algorithm). If Y's
    residual becomes constant before ``n_components`` are found, a warning
    is issued and fewer components are kept.

    Parameters
    ----------
    n_components : int, default 2
        Number of components to keep, between 1 and ``n_features``.
    scale : bool, default True
        Scale X and Y to unit variance.
    max_iter : int, default 500
        Maximum NIPALS iterations per component.
    tol : float, default 1e-06
        NIPALS convergence tolerance.
    copy : bool, default True
        Accepted for scikit-learn compatibility; inputs are never modified.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    x_weights_, y_weights_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    x_loadings_, y_loadings_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    x_rotations_, y_rotations_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    coef_ : ndarray of shape (n_targets, n_features)
    intercept_ : ndarray of shape (n_targets,)
    n_iter_ : list of int
        NIPALS iterations for each component.
    n_features_in_ : int
    x_scores_, y_scores_ : ndarray of shape (n_samples, n_components)
        Scores of the training data.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ rng.random(8, dtype=np.float32)
    >>> pred = mojolearn.PLSRegression(n_components=2).fit(X, y).predict(X)
    """
    _deflation, _pmode = "regression", "A"

    def fit(self, X, y=None, Y=None):
        """Fit the model to X and the targets (``y`` or ``Y``).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y, Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Targets (either name).

        Returns
        -------
        self
        """
        return _PLS_CALL(self, "fit", (X, y, Y), _PLS_CONTEXT)


# Mode A, canonical deflation.
class PLSCanonical(_PLS):
    """Partial least squares in canonical form.

    Follows scikit-learn's ``PLSCanonical``: finds paired directions in X
    and Y with maximal covariance, deflating both blocks symmetrically.

    Parameters
    ----------
    n_components : int, default 2
        Number of components to keep, between 1 and
        ``min(n_samples, n_features, n_targets)``.
    scale : bool, default True
        Scale X and Y to unit variance.
    algorithm : {"nipals", "svd"}, default "nipals"
        How the first singular vectors of each step are found. Any other
        value raises ValueError at construction.
    max_iter : int, default 500
        Maximum NIPALS iterations per component.
    tol : float, default 1e-06
        NIPALS convergence tolerance.
    copy : bool, default True
        Accepted for scikit-learn compatibility; inputs are never modified.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    x_weights_, y_weights_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    x_loadings_, y_loadings_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    x_rotations_, y_rotations_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    coef_ : ndarray of shape (n_targets, n_features)
    intercept_ : ndarray of shape (n_targets,)
    n_iter_ : list of int
        NIPALS iterations for each component.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X, Y = rng.random((1000, 8), dtype=np.float32), rng.random((1000, 3), dtype=np.float32)
    >>> xs, ys = mojolearn.PLSCanonical(n_components=2).fit(X, Y).transform(X, Y)
    """
    _deflation, _pmode = "canonical", "A"
    _parameters = ("n_components", "scale", "algorithm", "max_iter", "tol", "copy", "numeric_mode")

    def __init__(self, n_components=2, *, scale=True, algorithm="nipals", max_iter=500, tol=1e-06, copy=True,
                 numeric_mode=None):
        self.algorithm = algorithm
        super().__init__(n_components, scale=scale, max_iter=max_iter, tol=tol, copy=copy, numeric_mode=numeric_mode)


# NIPALS, mode B (the pseudo-inverses of X and Y), canonical deflation.
class CCA(_PLS):
    """Canonical correlation analysis.

    Follows scikit-learn's ``CCA`` (NIPALS algorithm): finds paired
    directions in X and Y with maximal correlation.

    Parameters
    ----------
    n_components : int, default 2
        Number of components to keep, between 1 and
        ``min(n_samples, n_features, n_targets)``.
    scale : bool, default True
        Scale X and Y to unit variance.
    max_iter : int, default 500
        Maximum NIPALS iterations per component.
    tol : float, default 1e-06
        NIPALS convergence tolerance.
    copy : bool, default True
        Accepted for scikit-learn compatibility; inputs are never modified.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    x_weights_, y_weights_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    x_loadings_, y_loadings_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    x_rotations_, y_rotations_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    coef_ : ndarray of shape (n_targets, n_features)
    intercept_ : ndarray of shape (n_targets,)
    n_iter_ : list of int
        NIPALS iterations for each component.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X, Y = rng.random((1000, 8), dtype=np.float32), rng.random((1000, 3), dtype=np.float32)
    >>> xs, ys = mojolearn.CCA(n_components=2).fit(X, Y).transform(X, Y)
    """
    _deflation, _pmode = "canonical", "B"


# ================================================================ dictionary learning / sparse PCA
def sparse_encode(X, dictionary, *, algorithm="lasso_lars", n_nonzero_coefs=None, alpha=None, init=None,
                  max_iter=1000, positive=False, numeric_mode=None, **_ignored):
    """Sparse-code each row of X against a fixed dictionary.

    Follows scikit-learn's ``sparse_encode``. ``"lasso_lars"`` and
    ``"lasso_cd"`` solve the same Lasso problem (both by coordinate descent;
    the optimum does not depend on the path). Other scikit-learn keywords
    (such as ``n_jobs`` or ``check_input``) are accepted and ignored.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    dictionary : array-like of shape (n_components, n_features)
        Dictionary atoms, one per row.
    algorithm : {"lasso_lars", "lasso_cd", "lars", "omp", "threshold"}, default "lasso_lars"
        Coding method.
    n_nonzero_coefs : int or None, default None
        Nonzero coefficients per row for ``"lars"`` and ``"omp"``. None uses
        ``min(max(n_features / 10, 1), n_components)``.
    alpha : float or None, default None
        Penalty for the Lasso methods, or threshold for ``"threshold"``.
        None uses 1.0.
    init : array-like of shape (n_samples, n_components) or None, default None
        Starting code for ``"lasso_cd"``.
    max_iter : int, default 1000
        Maximum iterations for the Lasso methods.
    positive : bool, default False
        Force the code to be non-negative (Lasso methods and ``"threshold"``).
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this call. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Returns
    -------
    ndarray of shape (n_samples, n_components), float32

    Raises
    ------
    ValueError
        For an unknown ``algorithm``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> D = rng.random((5, 8), dtype=np.float32)
    >>> code = mojolearn.sparse_encode(rng.random((100, 8), dtype=np.float32), D, algorithm="omp", n_nonzero_coefs=2)
    """
    return _DL_CALL(None, "sparse_encode", (X, dictionary, algorithm, n_nonzero_coefs, alpha, init, max_iter,
                                            positive, numeric_mode), _DL_CONTEXT)


class _SparseCoding(_Base):
    # One native call per public method; its owner runs the whole lifecycle
    # (bindings/decomp_public_lifecycle_unpack.mojo), so `_mode` installs no wrappers.
    _mojolearn_native_public = True

    def transform(self, X):
        """Encode X as a sparse combination of the dictionary atoms.

        Uses ``transform_algorithm`` and its settings.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
            Twice as many columns when ``split_sign=True``.
        """
        return _DL_CALL(self, "coding_transform", (X,), _DL_CONTEXT)

    def inverse_transform(self, X):
        """Rebuild data from a code: ``code @ components_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            A code, as returned by ``transform`` (with ``2 * n_components``
            columns when ``split_sign=True``).

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32
        """
        return _DL_CALL(self, "coding_inverse", (X,), _DL_CONTEXT)


# Reference: scikit-learn `decomposition/_dict_learning.py`:
# `_dict_learning`, `_update_dict`, `_sparse_encode_precomputed`,
# `_BaseSparseCoding._transform`. The initial SVD is exact (QR + one-sided
# Jacobi); a resampled unused atom is drawn from the Philox stream.
# fit_algorithm 'lars' solves its Lasso by coordinate descent (the same
# optimum). callback is refused by name.
class DictionaryLearning(_SparseCoding):
    """Learn a dictionary in which the data has a sparse representation.

    Follows scikit-learn's ``DictionaryLearning``: alternates sparse coding
    of X with updates of the dictionary atoms. The starting dictionary comes
    from an exact SVD of X, and both ``fit_algorithm`` values solve the same
    Lasso problem by coordinate descent.

    Parameters
    ----------
    n_components : int or None, default None
        Number of atoms; None uses ``n_features``.
    alpha : float, default 1
        Sparsity penalty used while fitting.
    max_iter : int, default 1000
        Maximum number of iterations.
    tol : float, default 1e-8
        Convergence tolerance.
    fit_algorithm : {"lars", "cd"}, default "lars"
        Lasso solver name used while fitting (both give the same optimum).
    transform_algorithm : {"omp", "lasso_lars", "lasso_cd", "lars", "threshold"}, default "omp"
        Method used by ``transform``; see :func:`sparse_encode`.
    transform_n_nonzero_coefs : int or None, default None
        Nonzero coefficients for ``"omp"`` and ``"lars"`` in ``transform``.
    transform_alpha : float or None, default None
        Penalty or threshold used by ``transform``; None uses ``alpha``.
    n_jobs : ignored
    code_init : array-like of shape (n_samples, n_components) or None, default None
        Starting code; used only together with ``dict_init``.
    dict_init : array-like of shape (n_components, n_features) or None, default None
        Starting dictionary; used only together with ``code_init``.
    callback : None
        Must be None; a callable raises NotImplementedError.
    verbose : ignored
    split_sign : bool, default False
        Return the positive and negative parts of the code as separate
        columns.
    random_state : int or None, default None
        Seed used to resample unused atoms; None means 0.
    positive_code : bool, default False
        Force the code to be non-negative.
    positive_dict : bool, default False
        Force the dictionary to be non-negative.
    transform_max_iter : int, default 1000
        Maximum iterations of the Lasso methods during fit and transform.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The dictionary atoms.
    error_ : list of float
        Objective value at each iteration.
    n_iter_ : int
    n_features_in_ : int

    Raises
    ------
    ValueError
        For an unknown ``fit_algorithm`` or ``transform_algorithm``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 8), dtype=np.float32)
    >>> dl = mojolearn.DictionaryLearning(n_components=5, max_iter=20, random_state=0)
    >>> code = dl.fit_transform(X)
    """
    _parameters = ("n_components", "alpha", "max_iter", "tol", "fit_algorithm", "transform_algorithm",
                   "transform_n_nonzero_coefs", "transform_alpha", "split_sign", "random_state",
                   "positive_code", "positive_dict", "transform_max_iter", "numeric_mode",
                   "callback", "code_init", "dict_init", "n_jobs", "verbose")

    def __init__(self, n_components=None, *, alpha=1, max_iter=1000, tol=1e-8, fit_algorithm="lars",
                 transform_algorithm="omp", transform_n_nonzero_coefs=None, transform_alpha=None, n_jobs=None,
                 code_init=None, dict_init=None, callback=None, verbose=False, split_sign=False, random_state=None,
                 positive_code=False, positive_dict=False, transform_max_iter=1000, numeric_mode=None):
        self.n_components, self.alpha, self.max_iter, self.tol = n_components, alpha, max_iter, tol
        self.fit_algorithm, self.transform_algorithm = fit_algorithm, transform_algorithm
        self.transform_n_nonzero_coefs, self.transform_alpha = transform_n_nonzero_coefs, transform_alpha
        self.n_jobs, self.code_init, self.dict_init, self.callback = n_jobs, code_init, dict_init, callback
        self.verbose, self.split_sign, self.random_state = verbose, split_sign, random_state
        self.positive_code, self.positive_dict = positive_code, positive_dict
        self.transform_max_iter, self.numeric_mode = transform_max_iter, numeric_mode

    def fit(self, X, y=None):
        """Fit the dictionary to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _DL_CALL(self, "dl_fit", (X,), _DL_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit the dictionary and return the code found while fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
            Twice as many columns when ``split_sign=True``.
        """
        return _DL_CALL(self, "dl_fit_transform", (X,), _DL_CONTEXT)


# Reference: `_dict_learning.py::MiniBatchDictionaryLearning`:
# `_initialize_dict`, `_minibatch_step`, `_update_inner_stats`,
# `_check_convergence`. The shuffle is a Philox permutation (sort of
# counter draws, ties to the lower index) and the initial dictionary the
# exact SVD (sklearn: randomized).
class MiniBatchDictionaryLearning(_SparseCoding):
    """Learn a sparse-coding dictionary from mini-batches.

    Follows scikit-learn's ``MiniBatchDictionaryLearning``. The starting
    dictionary comes from an exact SVD (scikit-learn uses a randomized one)
    and the shuffle uses mojolearn's own generator, so results differ
    slightly from scikit-learn's for the same seed.

    Parameters
    ----------
    n_components : int or None, default None
        Number of atoms; None uses ``n_features``.
    alpha : float, default 1
        Sparsity penalty used while fitting.
    max_iter : int, default 1000
        Maximum passes over the data.
    fit_algorithm : {"lars", "cd"}, default "lars"
        Lasso solver name used while fitting.
    n_jobs : ignored
    batch_size : int, default 256
        Rows per mini-batch (capped at ``n_samples``).
    shuffle : bool, default True
        Shuffle the rows once before fitting.
    dict_init : array-like of shape (n_components, n_features) or None, default None
        Starting dictionary.
    transform_algorithm : {"omp", "lasso_lars", "lasso_cd", "lars", "threshold"}, default "omp"
        Method used by ``transform``; see :func:`sparse_encode`.
    transform_n_nonzero_coefs : int or None, default None
        Nonzero coefficients for ``"omp"`` and ``"lars"`` in ``transform``.
    transform_alpha : float or None, default None
        Penalty or threshold used by ``transform``; None uses ``alpha``.
    verbose : ignored
    split_sign : bool, default False
        Return the positive and negative parts of the code as separate
        columns.
    random_state : int or None, default None
        Seed for the shuffle; None means 0.
    positive_code : bool, default False
        Force the code to be non-negative.
    positive_dict : bool, default False
        Force the dictionary to be non-negative.
    transform_max_iter : int, default 1000
        Maximum iterations of the Lasso methods during fit and transform.
    callback : None
        Must be None; a callable raises NotImplementedError.
    tol : float, default 1e-3
        Stop when the dictionary changes less than this between steps.
    max_no_improvement : int or None, default 10
        Stop after this many mini-batches without improvement of the
        smoothed cost; None disables this test.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The dictionary atoms.
    n_iter_ : int
        Passes over the data.
    n_steps_ : int
        Mini-batches processed.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> dl = mojolearn.MiniBatchDictionaryLearning(n_components=5, max_iter=5, random_state=0).fit(X)
    >>> code = dl.transform(X)
    """
    _parameters = ("n_components", "alpha", "max_iter", "fit_algorithm", "batch_size", "shuffle",
                   "transform_algorithm", "transform_n_nonzero_coefs", "transform_alpha", "split_sign",
                   "random_state", "positive_code", "positive_dict", "transform_max_iter", "tol",
                   "max_no_improvement", "numeric_mode",
                   "callback", "dict_init", "n_jobs", "verbose")

    def __init__(self, n_components=None, *, alpha=1, max_iter=1000, fit_algorithm="lars", n_jobs=None,
                 batch_size=256, shuffle=True, dict_init=None, transform_algorithm="omp",
                 transform_n_nonzero_coefs=None, transform_alpha=None, verbose=False, split_sign=False,
                 random_state=None, positive_code=False, positive_dict=False, transform_max_iter=1000,
                 callback=None, tol=1e-3, max_no_improvement=10, numeric_mode=None):
        self.n_components, self.alpha, self.max_iter, self.fit_algorithm = n_components, alpha, max_iter, fit_algorithm
        self.n_jobs, self.batch_size, self.shuffle, self.dict_init = n_jobs, batch_size, shuffle, dict_init
        self.transform_algorithm, self.transform_n_nonzero_coefs = transform_algorithm, transform_n_nonzero_coefs
        self.transform_alpha, self.verbose, self.split_sign = transform_alpha, verbose, split_sign
        self.random_state, self.positive_code, self.positive_dict = random_state, positive_code, positive_dict
        self.transform_max_iter, self.callback, self.tol = transform_max_iter, callback, tol
        self.max_no_improvement, self.numeric_mode = max_no_improvement, numeric_mode

    def fit(self, X, y=None):
        """Fit the dictionary to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _DL_CALL(self, "mbdl_fit", (X,), _DL_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit the dictionary, then return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DL_CALL(self, "mbdl_fit_transform", (X,), _DL_CONTEXT)


class _BaseSparsePCA(_Base):
    # One native call per public method; its owner runs the whole lifecycle
    # (bindings/decomp_public_lifecycle_unpack.mojo), so `_mode` installs no wrappers.
    _mojolearn_native_public = True

    # fit, fit_transform, transform and inverse_transform are one call each
    # into the native dictionary-learning owner (bindings/dict_learning_native.mojo),
    # which tells SparsePCA and MiniBatchSparsePCA apart by class.
    def fit_transform(self, X, y=None):
        """Fit the sparse components, then return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DL_CALL(self, "spca_fit_transform", (X,), _DL_CONTEXT)

    # sklearn `_BaseSparsePCA.transform`: ridge_regression(components_.T,
    # (X - mean_).T, ridge_alpha, solver='cholesky').
    def transform(self, X):
        """Project X onto the sparse components by ridge regression.

        Solves a least-squares problem with penalty ``ridge_alpha`` for each
        centered row, as scikit-learn does.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DL_CALL(self, "spca_transform", (X,), _DL_CONTEXT)

    def inverse_transform(self, X):
        """Map projected data back: ``X @ components_ + mean_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32
        """
        return _DL_CALL(self, "spca_inverse", (X,), _DL_CONTEXT)


# Reference: scikit-learn `decomposition/_sparse_pca.py`: dictionary
# learning on the centered X^T, the code's transpose the components.
class SparsePCA(_BaseSparsePCA):
    """Sparse principal component analysis.

    Follows scikit-learn's ``SparsePCA``: finds components with few nonzero
    loadings by dictionary learning on the centered, transposed data. Each
    component is scaled to unit norm.

    Parameters
    ----------
    n_components : int or None, default None
        Number of components; None uses ``n_features``.
    alpha : float, default 1
        Sparsity penalty; larger values give sparser components.
    ridge_alpha : float, default 0.01
        Ridge penalty used by ``transform``.
    max_iter : int, default 1000
        Maximum number of iterations.
    tol : float, default 1e-8
        Convergence tolerance.
    method : {"lars", "cd"}, default "lars"
        Lasso solver name (both give the same optimum).
    n_jobs : ignored
    U_init : array-like of shape (n_samples, n_components) or None, default None
        Starting sample loadings; used only together with ``V_init``.
    V_init : array-like of shape (n_components, n_features) or None, default None
        Starting components; used only together with ``U_init``.
    verbose : ignored
    random_state : int or None, default None
        Seed used to resample unused atoms; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Sparse components, each scaled to unit norm.
    n_components_ : int
    mean_ : ndarray of shape (n_features,)
    n_iter_ : int
    n_features_in_ : int
    error_ : list of float
        Objective value at each iteration.

    Raises
    ------
    ValueError
        If ``method`` is not ``"lars"`` or ``"cd"``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 8), dtype=np.float32)
    >>> spca = mojolearn.SparsePCA(n_components=3, max_iter=20, random_state=0).fit(X)
    >>> Z = spca.transform(X)
    """
    _parameters = ("n_components", "alpha", "ridge_alpha", "max_iter", "tol", "method", "random_state",
                   "numeric_mode",
                   "U_init", "V_init", "n_jobs", "verbose")

    def __init__(self, n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-8, method="lars",
                 n_jobs=None, U_init=None, V_init=None, verbose=False, random_state=None, numeric_mode=None):
        self.n_components, self.alpha, self.ridge_alpha = n_components, alpha, ridge_alpha
        self.max_iter, self.tol, self.method, self.n_jobs = max_iter, tol, method, n_jobs
        self.U_init, self.V_init, self.verbose, self.random_state = U_init, V_init, verbose, random_state
        self.numeric_mode = numeric_mode

    def fit(self, X, y=None):
        """Fit the sparse components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _DL_CALL(self, "spca_fit", (X,), _DL_CONTEXT)


# MiniBatchDictionaryLearning on the centered X^T, the components its
# lasso_lars transform of X^T. callback is not forwarded (never called).
class MiniBatchSparsePCA(_BaseSparsePCA):
    """Sparse principal component analysis fitted on mini-batches.

    Follows scikit-learn's ``MiniBatchSparsePCA``: runs
    :class:`MiniBatchDictionaryLearning` on the centered, transposed data.
    Each component is scaled to unit norm.

    Parameters
    ----------
    n_components : int or None, default None
        Number of components; None uses ``n_features``.
    alpha : float, default 1
        Sparsity penalty.
    ridge_alpha : float, default 0.01
        Ridge penalty used by ``transform``.
    max_iter : int, default 1000
        Maximum passes over the data.
    callback : ignored
    batch_size : int, default 3
        Features per mini-batch (the data is transposed).
    verbose : ignored
    shuffle : bool, default True
        Shuffle once before fitting.
    n_jobs : ignored
    method : {"lars", "cd"}, default "lars"
        Lasso solver name.
    random_state : int or None, default None
        Seed for the shuffle; None means 0.
    tol : float, default 1e-3
        Convergence tolerance on the dictionary change.
    max_no_improvement : int or None, default 10
        Stop after this many mini-batches without improvement; None disables
        this test.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Sparse components, each scaled to unit norm.
    n_components_ : int
    mean_ : ndarray of shape (n_features,)
    n_iter_ : int
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 8), dtype=np.float32)
    >>> spca = mojolearn.MiniBatchSparsePCA(n_components=3, max_iter=5, random_state=0).fit(X)
    """
    _parameters = ("n_components", "alpha", "ridge_alpha", "max_iter", "batch_size", "shuffle", "method",
                   "random_state", "tol", "max_no_improvement", "numeric_mode",
                   "callback", "n_jobs", "verbose")

    def __init__(self, n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, callback=None,
                 batch_size=3, verbose=False, shuffle=True, n_jobs=None, method="lars", random_state=None,
                 tol=1e-3, max_no_improvement=10, numeric_mode=None):
        self.n_components, self.alpha, self.ridge_alpha, self.max_iter = n_components, alpha, ridge_alpha, max_iter
        self.callback, self.batch_size, self.verbose, self.shuffle = callback, batch_size, verbose, shuffle
        self.n_jobs, self.method, self.random_state, self.tol = n_jobs, method, random_state, tol
        self.max_no_improvement, self.numeric_mode = max_no_improvement, numeric_mode

    def fit(self, X, y=None):
        """Fit the sparse components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _DL_CALL(self, "spca_fit", (X,), _DL_CONTEXT)


# ================================================================ LatentDirichletAllocation


# Reference: scikit-learn `decomposition/_lda.py` with
# `_online_lda_fast.pyx`: `_init_latent_vars`, `_e_step`,
# `_update_doc_distribution`, `_em_step`, `_approx_bound`, `transform`,
# `perplexity`, `score`. Variational EM in a fixed order: every document's
# inner loop is one thread, the sufficient statistics one gemm. The
# Gamma(100, 1/100) draws are Marsaglia-Tsang on the Philox stream. Online
# mini-batches go in row order, as sklearn with no shuffle does. A
# scipy.sparse count matrix is densified (exact); the per-document cell
# already skips zero counts.
class LatentDirichletAllocation(_Base):
    """Latent Dirichlet allocation topic model, fitted by variational EM.

    Follows scikit-learn's ``LatentDirichletAllocation``. X is a
    document-term count matrix (non-negative; a ``scipy.sparse`` matrix is
    densified). Random draws come from mojolearn's own generator, so topics
    differ from scikit-learn's for the same seed. Online learning takes the
    mini-batches in row order.

    Parameters
    ----------
    n_components : int, default 10
        Number of topics.
    doc_topic_prior : float or None, default None
        Prior on the document-topic distribution; None uses
        ``1 / n_components``.
    topic_word_prior : float or None, default None
        Prior on the topic-word distribution; None uses ``1 / n_components``.
    learning_method : {"batch", "online"}, default "batch"
        Use all data in each update, or mini-batches.
    learning_decay : float, default 0.7
        Online learning rate decay.
    learning_offset : float, default 10.0
        Online learning rate offset (down-weights early iterations).
    max_iter : int, default 10
        Maximum passes over the data.
    batch_size : int, default 128
        Documents per mini-batch for online learning; must be a positive int.
    evaluate_every : int, default -1
        Compute perplexity every this many iterations and stop once it
        improves by less than ``perp_tol``; 0 or negative never evaluates.
    total_samples : float, default 1e6
        Total number of documents, used by ``partial_fit`` and
        ``perplexity(sub_sampling=True)``.
    perp_tol : float, default 0.1
        Perplexity tolerance for ``evaluate_every``.
    mean_change_tol : float, default 1e-3
        Stopping tolerance of the per-document E-step.
    max_doc_update_iter : int, default 100
        Maximum E-step iterations per document.
    n_jobs : ignored
    verbose : ignored
    random_state : int or None, default None
        Seed; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Variational topic-word parameters.
    exp_dirichlet_component_ : ndarray of shape (n_components, n_features)
    doc_topic_prior_ : float
    topic_word_prior_ : float
    bound_ : float
        Final perplexity on the training data.
    n_iter_ : int
    n_batch_iter_ : int
    n_features_in_ : int

    Raises
    ------
    ValueError
        If X has negative values, or ``learning_method`` or ``batch_size``
        is invalid.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).integers(0, 5, (200, 30)).astype(np.float32)
    >>> lda = mojolearn.LatentDirichletAllocation(n_components=4, random_state=0).fit(X)
    >>> doc_topics = lda.transform(X)
    """
    # One native call per public method; its owner runs the whole lifecycle
    # (bindings/decomp_public_lifecycle_unpack.mojo), so `_mode` installs no wrappers.
    _mojolearn_native_public = True

    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"positive_only": True}
    _parameters = ("n_components", "doc_topic_prior", "topic_word_prior", "learning_method", "learning_decay",
                   "learning_offset", "max_iter", "batch_size", "evaluate_every", "total_samples", "perp_tol",
                   "mean_change_tol", "max_doc_update_iter", "random_state", "numeric_mode",
                   "n_jobs", "verbose")

    def __init__(self, n_components=10, *, doc_topic_prior=None, topic_word_prior=None, learning_method="batch",
                 learning_decay=0.7, learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=-1,
                 total_samples=1e6, perp_tol=1e-1, mean_change_tol=1e-3, max_doc_update_iter=100, n_jobs=None,
                 verbose=0, random_state=None, numeric_mode=None):
        self.n_components, self.doc_topic_prior, self.topic_word_prior = n_components, doc_topic_prior, topic_word_prior
        self.learning_method, self.learning_decay, self.learning_offset = learning_method, learning_decay, learning_offset
        self.max_iter, self.batch_size, self.evaluate_every = max_iter, batch_size, evaluate_every
        self.total_samples, self.perp_tol, self.mean_change_tol = total_samples, perp_tol, mean_change_tol
        self.max_doc_update_iter, self.n_jobs, self.verbose = max_doc_update_iter, n_jobs, verbose
        self.random_state, self.numeric_mode = random_state, numeric_mode

    def fit(self, X, y=None):
        """Fit the topic model to a document-term matrix.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Non-negative word counts.
        y : ignored

        Returns
        -------
        self
        """
        return _DECOMP_CALL(self, "fit", (X,), _DECOMP_CONTEXT)

    def partial_fit(self, X, y=None):
        """Update the model with one batch of documents (online learning).

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Non-negative word counts, split into mini-batches of
            ``batch_size``.
        y : ignored

        Returns
        -------
        self
        """
        return _DECOMP_CALL(self, "partial_fit", (X,), _DECOMP_CONTEXT)

    def transform(self, X, *, normalize=True):
        """Return the topic distribution of each document.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Non-negative word counts.
        normalize : bool, default True
            Normalize each row to sum to 1.

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "transform", (X, normalize), _DECOMP_CONTEXT)

    def fit_transform(self, X, y=None, *, normalize=True):
        """Fit the model, then return ``transform(X, normalize=normalize)``.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
        y : ignored
        normalize : bool, default True

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "fit_transform", (X, normalize), _DECOMP_CONTEXT)

    def score(self, X, y=None):
        """Return the approximate log-likelihood (variational bound) of X.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        float
        """
        return _DECOMP_CALL(self, "score", (X,), _DECOMP_CONTEXT)

    def perplexity(self, X, sub_sampling=False):
        """Return the approximate perplexity of X.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
        sub_sampling : bool, default False
            Scale the bound as if X were a sample of ``total_samples``
            documents.

        Returns
        -------
        float
            Infinity when X has no words.
        """
        return _DECOMP_CALL(self, "perplexity", (X, sub_sampling), _DECOMP_CONTEXT)


# ================================================================ manifold: Isomap, MDS, LLE
_PD_KIND = {"manhattan": 1, "cityblock": 1, "l1": 1, "chebyshev": 2, "infinity": 2, "cosine": 4}


# The manifold estimators' metric rule (`_metric_spec`), distances, kNN
# matrices, kernel centering and top eigenpairs run in their compiled owner
# (bindings/mani_decomp_native.mojo); `_PD_KIND` is its table.


# TOMBSTONE: _lanczos_batch, _lanczos_top (dead Python Lanczos driver: the top-eigenpair route runs in bindings/mani_decomp_native.mojo and x_decomp/kpca_owner.mojo) deleted 2026-11-08 by bddde2ea2e; code recoverable at 41ffcec665.
# Restore: git apply experiments/removed/dead_python_loops.patch


# Reference: scikit-learn `manifold/_isomap.py` with
# `KernelPCA(kernel='precomputed')` and `utils/graph.py`: the kNN distance
# graph (ties to the lower index), all-pairs shortest paths by Dijkstra
# (one source per thread), the kernel -0.5 * D^2 centered, its top
# eigenpairs (Jacobi eigh). path_method 'FW' runs the same Dijkstra
# (x_decomp/NOT_IMPLEMENTED.tsv: deliberately divergent). Metrics per
# DEVIATION 5319. A callable or any other metric is refused by name.
class Isomap(_Base):
    """Isomap embedding: kernel PCA on geodesic distances.

    Follows scikit-learn's ``Isomap``: builds a nearest-neighbor (or radius)
    graph, computes shortest-path distances on it, and embeds the centered
    kernel ``-0.5 * D**2`` by its top eigenvectors. If the graph has more
    than one connected component, a warning is issued and the components are
    joined. Shortest paths always use Dijkstra's algorithm, and the
    eigenproblem is always solved exactly.

    Parameters
    ----------
    n_neighbors : int or None, default 5
        Neighbors per point. Set to None to use ``radius`` instead.
    radius : float or None, default None
        Connect every point within this distance (inclusive). Only one of
        ``n_neighbors`` and ``radius`` may be set.
    n_components : int, default 2
        Embedding dimension.
    eigen_solver : {"auto", "dense"}, default "auto"
        Both solve exactly. ``"arpack"`` raises NotImplementedError.
    tol : ignored
    max_iter : ignored
    path_method : {"auto", "FW", "D"}, default "auto"
        Accepted for scikit-learn compatibility; every value runs Dijkstra.
    neighbors_algorithm : ignored
    n_jobs : ignored
    metric : str, default "minkowski"
        Distance metric: ``"euclidean"``, ``"l2"``, ``"minkowski"`` (with ``p``), ``"manhattan"``
        (``"cityblock"``, ``"l1"``), ``"chebyshev"`` (``"infinity"``) or
        ``"cosine"``; a callable or any other name raises
        NotImplementedError.
    p : float, default 2
        Minkowski power (must be positive).
    metric_params : dict or None, default None
        Only ``{"p": ...}`` is supported; other keys raise
        NotImplementedError.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
    dist_matrix_ : ndarray of shape (n_samples, n_samples)
        Geodesic distances between training points.
    n_connected_components_ : int
    n_features_in_ : int

    Raises
    ------
    ValueError
        If both or neither of ``n_neighbors`` and ``radius`` are set,
        ``radius`` is negative, or ``path_method`` or ``eigen_solver`` is
        unknown.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 8), dtype=np.float32)
    >>> Y = mojolearn.Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    """
    _parameters = ("n_neighbors", "radius", "n_components", "eigen_solver", "path_method", "metric", "p", "numeric_mode",
                   "max_iter", "metric_params", "n_jobs", "neighbors_algorithm", "tol")

    def __init__(self, *, n_neighbors=5, radius=None, n_components=2, eigen_solver="auto", tol=0, max_iter=None,
                 path_method="auto", neighbors_algorithm="auto", n_jobs=None, metric="minkowski", p=2,
                 metric_params=None, numeric_mode=None):
        self.n_neighbors, self.radius, self.n_components = n_neighbors, radius, n_components
        self.eigen_solver, self.tol, self.max_iter, self.path_method = eigen_solver, tol, max_iter, path_method
        self.neighbors_algorithm, self.n_jobs, self.metric, self.p = neighbors_algorithm, n_jobs, metric, p
        self.metric_params, self.numeric_mode = metric_params, numeric_mode

    @_native_runtime_manifold_bridge.native_lifecycle
    def fit(self, X, y=None):
        """Compute the embedding of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _NATIVE(self, "family_mani_iso_fit",
                       (None, None, X, y), _NATIVE_CONTEXT)

    @_native_runtime_manifold_bridge.native_lifecycle
    def fit_transform(self, X, y=None):
        """Fit the model and return ``embedding_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _NATIVE(self, "family_mani_iso_fit_transform",
                       (None, None, X, y), _NATIVE_CONTEXT)

    @_native_runtime_manifold_bridge.native_lifecycle
    def transform(self, X):
        """Embed new points.

        Each query's geodesic distance to the training points goes through
        its nearest training neighbors (or, with ``radius``, every training
        point within the radius), then is projected as in kernel PCA.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        ndarray of shape (n_queries, n_components), float32

        Raises
        ------
        ValueError
            With ``radius``, if a query has no training point within it.
        """
        return _NATIVE(self, "family_mani_iso_transform",
                       (None, None, X), _NATIVE_CONTEXT)

    def reconstruction_error(self):
        """Return the reconstruction error of the embedding.

        Computed as in scikit-learn: the Frobenius norm of the difference
        between the centered geodesic kernel and its rank-``n_components``
        approximation, divided by ``n_samples``.

        Returns
        -------
        float
        """
        return _NATIVE(self, "family_mani_iso_reconstruction_error",
                       (None, None), _NATIVE_CONTEXT)
# Reference: scikit-learn `manifold/_classical_mds.py`.
class ClassicalMDS(_Base):
    """Classical (Torgerson) multidimensional scaling.

    Follows scikit-learn's ``ClassicalMDS``: double-centers ``-0.5 * D**2``
    for the pairwise distances D and embeds by its top eigenvectors scaled
    by the square roots of the eigenvalues.

    Parameters
    ----------
    n_components : int, default 2
        Embedding dimension.
    metric : str, default "euclidean"
        ``"precomputed"`` (X is a distance matrix) or a distance metric:
        ``"euclidean"``, ``"l2"``, ``"minkowski"`` (``p`` from ``metric_params``), ``"manhattan"``
        (``"cityblock"``, ``"l1"``), ``"chebyshev"`` (``"infinity"``) or
        ``"cosine"``; a callable or any other name raises
        NotImplementedError.
    metric_params : dict or None, default None
        Only ``{"p": ...}`` is supported.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
    eigenvalues_ : ndarray of shape (n_components,)
    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples)
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 8), dtype=np.float32)
    >>> Y = mojolearn.ClassicalMDS(n_components=2).fit_transform(X)
    """
    _parameters = ("n_components", "metric", "numeric_mode",
                   "metric_params")

    def __init__(self, n_components=2, *, metric="euclidean", metric_params=None, numeric_mode=None):
        self.n_components, self.metric, self.metric_params, self.numeric_mode = n_components, metric, metric_params, numeric_mode

    @_native_runtime_manifold_bridge.native_lifecycle
    def fit(self, X, y=None):
        """Compute the embedding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Data, or a distance matrix with ``metric="precomputed"``.
        y : ignored

        Returns
        -------
        self
        """
        return _NATIVE(self, "family_mani_cmds_fit",
                       (None, None, X, y), _NATIVE_CONTEXT)

    @_native_runtime_manifold_bridge.native_lifecycle
    def fit_transform(self, X, y=None):
        """Compute the embedding and return ``embedding_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _NATIVE(self, "family_mani_cmds_fit_transform",
                       (None, None, X, y), _NATIVE_CONTEXT)
# Reference: scikit-learn `manifold/_mds.py`, `_smacof_single`, `smacof`.
# Random starts come from the Philox stream. init='random' is the default
# sklearn's FutureWarning names. Non-metric: the disparities are the
# isotonic regression of the distances on the dissimilarities (the linear
# lane's IsotonicRegression), rescaled.
class MDS(_Base):
    """Multidimensional scaling by SMACOF.

    Follows scikit-learn's ``MDS``: minimizes stress with the Guttman
    transform from ``n_init`` starts and keeps the start with the lowest
    stress. ``metric_mds=False`` runs Kruskal's non-metric SMACOF. Random
    starts come from mojolearn's own generator.

    Parameters
    ----------
    n_components : int, default 2
        Embedding dimension.
    metric_mds : bool, default True
        Metric (True) or non-metric (False) MDS.
    n_init : int, default 1
        Number of random starts (with ``init="random"``).
    init : {"random", "classical_mds"} or array-like, default "random"
        Starting configuration: random, the :class:`ClassicalMDS`
        embedding, or an array of shape (n_samples, n_components).
    max_iter : int, default 300
        Maximum SMACOF iterations per start.
    verbose : ignored
    eps : float, default 1e-6
        Convergence tolerance on the stress.
    n_jobs : ignored
    random_state : int or None, default None
        Seed for random starts; None means 0.
    metric : str, default "euclidean"
        ``"precomputed"`` (X is a dissimilarity matrix) or a distance
        metric: ``"euclidean"``, ``"l2"``, ``"minkowski"`` (``p`` from ``metric_params``), ``"manhattan"``
        (``"cityblock"``, ``"l1"``), ``"chebyshev"`` (``"infinity"``) or
        ``"cosine"``; a callable or any other name raises
        NotImplementedError.
    metric_params : dict or None, default None
        Only ``{"p": ...}`` is supported.
    normalized_stress : bool or "auto", default "auto"
        Report normalized stress (Kruskal's stress-1). ``"auto"`` is True
        for non-metric MDS and False otherwise.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
    stress_ : float
    n_iter_ : int
    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples)
    n_features_in_ : int

    Raises
    ------
    ValueError
        For an unknown ``init``, or with ``metric_mds=False`` when
        ``n_samples**2`` exceeds ``2**31 - 1``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((300, 8), dtype=np.float32)
    >>> Y = mojolearn.MDS(n_components=2, random_state=0).fit_transform(X)
    """
    _parameters = ("n_components", "metric_mds", "n_init", "init", "max_iter", "eps", "random_state", "metric",
                   "normalized_stress", "numeric_mode",
                   "metric_params", "n_jobs", "verbose")

    def __init__(self, n_components=2, *, metric_mds=True, n_init=1, init="random", max_iter=300, verbose=0,
                 eps=1e-6, n_jobs=None, random_state=None, metric="euclidean", metric_params=None,
                 normalized_stress="auto", numeric_mode=None):
        self.n_components, self.metric_mds, self.n_init, self.init = n_components, metric_mds, n_init, init
        self.max_iter, self.verbose, self.eps, self.n_jobs = max_iter, verbose, eps, n_jobs
        self.random_state, self.metric, self.metric_params = random_state, metric, metric_params
        self.normalized_stress, self.numeric_mode = normalized_stress, numeric_mode

    @_native_runtime_manifold_bridge.native_lifecycle
    def fit_transform(self, X, y=None, init=None):
        """Compute the embedding and return it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Data, or dissimilarities with ``metric="precomputed"``.
        y : ignored
        init : array-like of shape (n_samples, n_components) or None, default None
            Starting configuration; overrides the ``init`` parameter.

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _NATIVE(self, "family_mani_mds_fit_transform",
                       (None, None, X, init, y), _NATIVE_CONTEXT)

    @_native_runtime_manifold_bridge.native_lifecycle
    def fit(self, X, y=None, init=None):
        """Compute the embedding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
        y : ignored
        init : array-like of shape (n_samples, n_components) or None, default None
            Starting configuration; overrides the ``init`` parameter.

        Returns
        -------
        self
        """
        return _NATIVE(self, "family_mani_mds_fit",
                       (None, None, X, init, y), _NATIVE_CONTEXT)
# LocallyLinearEmbedding's shift-invert route (`_lle_smallest`) and its
# policy run in the compiled owner (bindings/mani_decomp_native.mojo).


# Reference: scikit-learn `manifold/_locally_linear.py`:
# `barycenter_kneighbors_graph`, `barycenter_weights`, `null_space` (here
# the smallest right singular vectors of I - W, which are M's smallest
# eigenvectors), `locally_linear_embedding`, `transform`. M = (I - W)^T
# (I - W), its n_components + 1 smallest eigenvectors, the first dropped.
# method='ltsa', 'hessian' and 'modified' build sklearn's M as a stacked
# factor (M = B^T B) and take the same SVD route. eigen_solver='auto' with
# method='standard' past 200 rows (sklearn's ARPACK policy) takes
# `_lle_smallest`, shift-invert subspace iteration on the deflated factor;
# 'dense' (and every other case) the full one-sided Jacobi SVD.
class LocallyLinearEmbedding(_Base):
    """Locally linear embedding.

    Follows scikit-learn's ``LocallyLinearEmbedding``: reconstructs each
    point from its Euclidean nearest neighbors and finds the low-dimensional
    embedding that best preserves those reconstruction weights. Supports the
    standard, LTSA, Hessian and modified variants.

    Parameters
    ----------
    n_neighbors : int, default 5
        Neighbors per point; must be less than ``n_samples``.
    n_components : int, default 2
        Embedding dimension.
    reg : float, default 1e-3
        Regularization of the local weight solves.
    eigen_solver : {"auto", "dense"}, default "auto"
        ``"dense"`` always runs the full exact solve. ``"auto"`` with
        ``method="standard"``, more than 200 samples and ``n_components + 1 <
        10`` uses an iterative solver for the smallest eigenvectors;
        otherwise it runs the full solve. ``"arpack"`` raises
        NotImplementedError.
    tol : ignored
    max_iter : int, default 100
        Iteration limit of the iterative solver.
    method : {"standard", "ltsa", "hessian", "modified"}, default "standard"
        LLE variant. ``"hessian"`` needs ``n_neighbors > n_components *
        (n_components + 3) / 2``; ``"modified"`` needs ``n_neighbors >=
        n_components``.
    hessian_tol : float, default 1e-4
        Tolerance for ``method="hessian"``.
    modified_tol : float, default 1e-12
        Tolerance for ``method="modified"``.
    neighbors_algorithm : ignored
    random_state : int or None, default None
        Seed of the iterative solver's start; None means 0.
    n_jobs : ignored
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
    reconstruction_error_ : float
    n_features_in_ : int

    Raises
    ------
    ValueError
        For an unknown ``method`` or ``eigen_solver``, or ``n_neighbors``
        out of range for the method.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 8), dtype=np.float32)
    >>> Y = mojolearn.LocallyLinearEmbedding(n_neighbors=10, n_components=2).fit_transform(X)
    """
    _parameters = ("n_neighbors", "n_components", "reg", "eigen_solver", "method", "random_state", "numeric_mode",
                   "hessian_tol", "max_iter", "modified_tol", "n_jobs", "neighbors_algorithm", "tol")

    def __init__(self, *, n_neighbors=5, n_components=2, reg=1e-3, eigen_solver="auto", tol=1e-6, max_iter=100,
                 method="standard", hessian_tol=1e-4, modified_tol=1e-12, neighbors_algorithm="auto",
                 random_state=None, n_jobs=None, numeric_mode=None):
        self.n_neighbors, self.n_components, self.reg, self.eigen_solver = n_neighbors, n_components, reg, eigen_solver
        self.tol, self.max_iter, self.method, self.hessian_tol = tol, max_iter, method, hessian_tol
        self.modified_tol, self.neighbors_algorithm, self.random_state = modified_tol, neighbors_algorithm, random_state
        self.n_jobs, self.numeric_mode = n_jobs, numeric_mode

    @_native_runtime_manifold_bridge.native_lifecycle
    def fit(self, X, y=None):
        """Compute the embedding of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _NATIVE(self, "family_mani_lle_fit",
                       (None, None, X, y), _NATIVE_CONTEXT)

    @_native_runtime_manifold_bridge.native_lifecycle
    def fit_transform(self, X, y=None):
        """Compute the embedding and return ``embedding_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _NATIVE(self, "family_mani_lle_fit_transform",
                       (None, None, X, y), _NATIVE_CONTEXT)

    @_native_runtime_manifold_bridge.native_lifecycle
    def transform(self, X):
        """Embed new points.

        Each query is written as a weighted combination of its nearest
        training points, and the same weights are applied to their
        embeddings.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        ndarray of shape (n_queries, n_components), float32
        """
        return _NATIVE(self, "family_mani_lle_transform",
                       (None, None, X), _NATIVE_CONTEXT)


# The compiled family owner, connected once at import: every public method
# of Isomap, ClassicalMDS, MDS and LocallyLinearEmbedding is one native call
# (bindings/mani_decomp_native.mojo). Its table: the metric kinds (the native
# kit, bindings/mani_kit.mojo, never reads the Python _M / _DevBuf classes).
_NATIVE, _NATIVE_CONTEXT = _native_runtime_manifold_bridge.connect()
_native_family_bridge.tables(family_mani_pd_kind=dict(_PD_KIND))


# ================================================================ robust covariance
# Reference: scikit-learn `covariance/_robust_covariance.py`:
# `c_step`/`_c_step`, `select_candidates`, `fast_mcd`, `MinCovDet.fit`,
# `correct_covariance`, `reweight_covariance`. The random subsets are
# Philox permutations (a sort of counter draws, ties to the lower index)
# and every argsort breaks ties by index. chi2 quantiles by bisection of
# the incomplete gamma; one feature takes sklearn's 1-D shortcut.
class MinCovDet(_Base):
    """Minimum covariance determinant: a robust covariance estimator.

    Follows scikit-learn's ``MinCovDet`` (FastMCD), including the
    consistency correction and the reweighting step. Random subsets come
    from mojolearn's own generator and ties are broken by row index, so
    results can differ from scikit-learn's for the same seed.

    Parameters
    ----------
    store_precision : bool, default True
        Store ``precision_``; when False it is None.
    assume_centered : bool, default False
        Treat the data as centered (location fixed at zero).
    support_fraction : float or None, default None
        Fraction of points in the raw support. None uses
        ``ceil((n_samples + n_features + 1) / 2)`` points.
    random_state : int or None, default None
        Seed for the random subsets; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Robust location (reweighted).
    covariance_ : ndarray of shape (n_features, n_features)
        Robust covariance (reweighted).
    precision_ : ndarray of shape (n_features, n_features) or None
    support_ : ndarray of shape (n_samples,), uint8
        1 for points used in the reweighted estimate.
    raw_location_ : ndarray of shape (n_features,)
    raw_covariance_ : ndarray of shape (n_features, n_features)
    raw_support_ : ndarray of shape (n_samples,), uint8
    dist_ : ndarray of shape (n_samples,)
        Squared Mahalanobis distances of the training points.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    >>> mcd = mojolearn.MinCovDet(random_state=0).fit(X)
    >>> d = mcd.mahalanobis(X)
    """
    # One native call per public method; its owner runs the whole lifecycle
    # (bindings/decomp_public_lifecycle_unpack.mojo), so `_mode` installs no wrappers.
    _mojolearn_native_public = True

    _parameters = ("store_precision", "assume_centered", "support_fraction", "random_state", "numeric_mode")

    def __init__(self, *, store_precision=True, assume_centered=False, support_fraction=None, random_state=None,
                 numeric_mode=None):
        self.store_precision, self.assume_centered = store_precision, assume_centered
        self.support_fraction, self.random_state, self.numeric_mode = support_fraction, random_state, numeric_mode

    # Every operation is one call into the native MinCovDet owner
    # (bindings/mcd_native.mojo). A covariance estimator has no transform,
    # so it has no fit_transform either (scikit-learn's TransformerMixin).
    @available_if(lambda self: hasattr(self, "transform"))
    def fit_transform(self, X, y=None):
        """Not available: this estimator has no ``transform``."""
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'fit_transform'")

    def fit(self, X, y=None):
        """Fit the robust location and covariance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _MCD_CALL(self, "fit", (X,), _MCD_CONTEXT)

    def mahalanobis(self, X):
        """Return squared Mahalanobis distances under the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,), float32

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _MCD_CALL(self, "mahalanobis", (X,), _MCD_CONTEXT)

    def get_precision(self):
        """Return the precision matrix (pseudo-inverse of ``covariance_``).

        Available even when ``store_precision=False``.

        Returns
        -------
        ndarray of shape (n_features, n_features), float32
        """
        return _MCD_CALL(self, "get_precision", (), _MCD_CONTEXT)

    def score(self, X, y=None):
        """Return the Gaussian log-likelihood of X under the fitted model.

        As scikit-learn's ``EmpiricalCovariance.score``: uses X's empirical
        covariance around ``location_`` and the fitted precision.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        float
        """
        return _MCD_CALL(self, "score", (X,), _MCD_CONTEXT)


class EllipticEnvelope(MinCovDet):
    """Outlier detection for Gaussian-distributed data.

    Follows scikit-learn's ``EllipticEnvelope``: fits a :class:`MinCovDet`
    and sets ``offset_`` at the ``contamination`` percentile of the negated
    training distances, so that about that fraction of training points is
    labeled an outlier.

    Parameters
    ----------
    store_precision : bool, default True
        Store ``precision_``.
    assume_centered : bool, default False
        Treat the data as centered.
    support_fraction : float or None, default None
        Fraction of points in the raw support; see :class:`MinCovDet`.
    contamination : float, default 0.1
        Expected fraction of outliers, in (0, 0.5].
    random_state : int or None, default None
        Seed for the random subsets; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    offset_ : float
        Threshold on the decision function.
    location_, covariance_, precision_, support_, dist_
        As in :class:`MinCovDet`.

    Raises
    ------
    ValueError
        If ``contamination`` is not in (0, 0.5].

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    >>> labels = mojolearn.EllipticEnvelope(contamination=0.05, random_state=0).fit(X).predict(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "outlier_detector"
    _parameters = ("store_precision", "assume_centered", "support_fraction", "contamination", "random_state",
                   "numeric_mode")

    def __init__(self, *, store_precision=True, assume_centered=False, support_fraction=None, contamination=0.1,
                 random_state=None, numeric_mode=None):
        super().__init__(store_precision=store_precision, assume_centered=assume_centered,
                         support_fraction=support_fraction, random_state=random_state, numeric_mode=numeric_mode)
        self.contamination = contamination

    def fit(self, X, y=None):
        """Fit the robust model and set ``offset_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _MCD_CALL(self, "fit", (X,), _MCD_CONTEXT)

    def score_samples(self, X):
        """Return the negated squared Mahalanobis distance of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,), float64
        """
        return _MCD_CALL(self, "score_samples", (X,), _MCD_CONTEXT)

    def decision_function(self, X):
        """Return ``score_samples(X) - offset_``; negative means outlier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,), float64
        """
        return _MCD_CALL(self, "decision_function", (X,), _MCD_CONTEXT)

    def predict(self, X):
        """Label each sample 1 (inlier) or -1 (outlier).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,), int32
        """
        return _MCD_CALL(self, "predict", (X,), _MCD_CONTEXT)

    def fit_predict(self, X, y=None):
        """Fit the model, then return ``predict(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples,), int32
        """
        return _MCD_CALL(self, "fit_predict", (X,), _MCD_CONTEXT)

    # Lane cpu4-python: the x_metrics device program (`accuracy_fraction`:
    # exact match counts, the weight total its PairSum, the ratio binary64
    # on the device; the host column runs the same units) replaces the host
    # loop `x_decomp_accuracy`.
    def score(self, X, y, sample_weight=None):
        """Return the accuracy of ``predict(X)`` against labels y.

        As scikit-learn's ``EllipticEnvelope.score``: the (weighted) fraction
        of exact label matches.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            True labels (1 inlier, -1 outlier).
        sample_weight : array-like of shape (n_samples,) or None, default None

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If y or sample_weight has a different length than X.
        ZeroDivisionError
            If the weights sum to zero.
        """
        return _MCD_CALL(self, "elliptic_score", (X, y, sample_weight), _MCD_CONTEXT)


# ================================================================ implicit ALS
# Reference: the `implicit` library's `als.py` and
# `cpu/_als.pyx::least_squares` (exact solver) and `_least_squares_cg`
# (use_cg=True; x_decomp/cells.mojo `als_cg_row`, DEVIATION 5321). Factors
# start as Philox uniform draws * 0.01; each iteration solves every user row
# then every item row as one batched Cholesky per row (one thread per row,
# sums in a fixed order). The default stays use_cg=False, the exact solve,
# so existing fits keep their bits.
class AlternatingLeastSquares(_Base):
    """Implicit-feedback matrix factorization by alternating least squares.

    The method of Hu, Koren and Volinsky (2008), following the ``implicit``
    library: confidence is ``alpha * user_items``, and each iteration solves
    for every user's factors and then every item's. Unlike ``implicit``,
    ``use_cg`` defaults to False (the exact solve), and ``user_items`` is
    held dense (0 means no interaction; a ``scipy.sparse`` matrix is
    densified).

    Parameters
    ----------
    factors : int, default 100
        Number of latent factors.
    regularization : float, default 0.01
        L2 penalty on the factors.
    alpha : float, default 1.0
        Confidence scale applied to ``user_items``.
    iterations : int, default 15
        Number of alternating sweeps.
    use_cg : bool, default False
        Use ``implicit``'s conjugate-gradient solver (``cg_steps`` steps per
        row per half-sweep) instead of the exact Cholesky solve.
    calculate_training_loss : bool, default False
        Record ``training_loss_`` after every iteration (``implicit``'s loss).
    random_state : int or None, default None
        Seed of the starting factors; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.
    cg_steps : int, default 3
        Conjugate-gradient steps per row with ``use_cg=True``; must be at
        least 1.

    Attributes
    ----------
    user_factors, user_factors_ : ndarray of shape (n_users, factors)
    item_factors, item_factors_ : ndarray of shape (n_items, factors)
    training_loss_ : list of float
        Only with ``calculate_training_loss=True``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> R = (np.random.default_rng(0).random((200, 50)) < 0.05).astype(np.float32)
    >>> als = mojolearn.AlternatingLeastSquares(factors=8, iterations=5, random_state=0).fit(R)
    >>> ids, scores = als.recommend(0, R, N=5)
    """
    # One native call per public method; its owner runs the whole lifecycle
    # (bindings/decomp_public_lifecycle_unpack.mojo), so `_mode` installs no wrappers.
    _mojolearn_native_public = True

    _parameters = ("factors", "regularization", "alpha", "iterations", "use_cg", "cg_steps",
                   "calculate_training_loss", "random_state", "numeric_mode")

    def __init__(self, factors=100, regularization=0.01, alpha=1.0, iterations=15, use_cg=False,
                 calculate_training_loss=False, random_state=None, numeric_mode=None, cg_steps=3):
        self.factors, self.regularization, self.alpha, self.iterations = factors, regularization, alpha, iterations
        self.use_cg, self.calculate_training_loss = use_cg, calculate_training_loss
        self.random_state, self.numeric_mode, self.cg_steps = random_state, numeric_mode, cg_steps

    # fit, recommend and similar_items are one call each into the native
    # AlternatingLeastSquares owner (bindings/als_native.mojo).

    def fit(self, user_items, show_progress=False):
        """Fit user and item factors.

        Parameters
        ----------
        user_items : array-like of shape (n_users, n_items)
            Interaction strengths; 0 means no interaction. A
            ``scipy.sparse`` matrix is densified.
        show_progress : bool, default False
            Accepted for ``implicit`` compatibility; nothing is shown.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``use_cg=True`` and ``cg_steps < 1``.
        """
        return _ALS_CALL(self, "fit", (user_items,), _ALS_CONTEXT)

    # A recommender has no transform, so it has no fit_transform either.
    @available_if(lambda self: hasattr(self, "transform"))
    def fit_transform(self, X, y=None):
        """Not available: this estimator has no ``transform``."""
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'fit_transform'")

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True):
        """Return the top N items for one user.

        Items are ranked by the dot product of user and item factors, ties
        going to the lower item id.

        Parameters
        ----------
        userid : int
            Row of the user.
        user_items : array-like of shape (n_users, n_items) or (1, n_items), or None
            Interactions used to filter items the user already has: row
            ``userid``, or the single row given.
        N : int, default 10
            Number of items to return; must be non-negative.
        filter_already_liked_items : bool, default True
            Skip items with a nonzero entry in the user's row.

        Returns
        -------
        ids : ndarray of shape (<= N,), int32
        scores : ndarray of shape (<= N,), float32

        Raises
        ------
        ValueError
            If ``N`` is negative or ``user_items`` has no matching row.
        """
        return _ALS_CALL(self, "recommend", (userid, user_items, N, filter_already_liked_items), _ALS_CONTEXT)

    def similar_items(self, itemid, N=10):
        """Return the N items most similar to one item.

        Similarity is the cosine of the item factors. The item itself is
        included, as ``implicit`` returns it.

        Parameters
        ----------
        itemid : int
        N : int, default 10
            Must be non-negative.

        Returns
        -------
        ids : ndarray of shape (<= N,), int32
        scores : ndarray of shape (<= N,), float32
        """
        return _ALS_CALL(self, "similar_items", (itemid, N), _ALS_CONTEXT)


class SparseCoder(_SparseCoding):
    """Sparse coding against a fixed, precomputed dictionary.

    Follows scikit-learn's ``SparseCoder``; ``transform`` encodes each row
    with :func:`sparse_encode`. Nothing is learned, so ``fit`` does nothing.

    Parameters
    ----------
    dictionary : array-like of shape (n_components, n_features)
        Dictionary atoms, one per row.
    transform_algorithm : {"omp", "lasso_lars", "lasso_cd", "lars", "threshold"}, default "omp"
        Coding method. An unknown value raises ValueError at ``fit`` or
        ``transform``.
    transform_n_nonzero_coefs : int or None, default None
        Nonzero coefficients for ``"omp"`` and ``"lars"``.
    transform_alpha : float or None, default None
        Penalty or threshold; None uses 1.0.
    split_sign : bool, default False
        Return the positive and negative parts of the code as separate
        columns.
    n_jobs : ignored
    positive_code : bool, default False
        Force the code to be non-negative.
    transform_max_iter : int, default 1000
        Maximum iterations of the Lasso methods.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
    n_components_ : int
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> coder = mojolearn.SparseCoder(rng.random((5, 8), dtype=np.float32), transform_n_nonzero_coefs=2)
    >>> code = coder.transform(rng.random((100, 8), dtype=np.float32))
    """
    # stateless: the dictionary is a constructor parameter, so transform needs no fit
    _sklearn_requires_fit = False
    _parameters = ("dictionary", "transform_algorithm", "transform_n_nonzero_coefs", "transform_alpha",
                   "split_sign", "positive_code", "transform_max_iter", "numeric_mode",
                   "n_jobs")

    def __init__(self, dictionary, *, transform_algorithm="omp", transform_n_nonzero_coefs=None, transform_alpha=None,
                 split_sign=False, n_jobs=None, positive_code=False, transform_max_iter=1000, numeric_mode=None):
        self.dictionary, self.transform_algorithm = dictionary, transform_algorithm
        self.transform_n_nonzero_coefs, self.transform_alpha = transform_n_nonzero_coefs, transform_alpha
        self.split_sign, self.n_jobs, self.positive_code = split_sign, n_jobs, positive_code
        self.transform_max_iter, self.numeric_mode = transform_max_iter, numeric_mode

    # Nothing is learned: `transform` works without `fit` (scikit-learn tags
    # SparseCoder requires_fit=False). `__init__` stores the parameters
    # verbatim; every operation checks them and reads the dictionary, in
    # the native dictionary-learning owner.
    def transform(self, X):
        """Encode X as a sparse combination of the dictionary atoms.

        Uses ``transform_algorithm`` and its settings.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
            Twice as many columns when ``split_sign=True``.
        """
        return _DL_CALL(self, "coder_transform", (X,), _DL_CONTEXT)

    def inverse_transform(self, X):
        """Rebuild data from a code: ``code @ dictionary``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            A code, as returned by ``transform`` (with ``2 * n_components``
            columns when ``split_sign=True``).

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32
        """
        return _DL_CALL(self, "coder_inverse", (X,), _DL_CONTEXT)

    def fit(self, X, y=None):
        """Check the parameters, record the dictionary's shape and return self
        (the dictionary is fixed; nothing is learned).

        Returns
        -------
        self
        """
        return _DL_CALL(self, "coder_fit", (X,), _DL_CONTEXT)

    def fit_transform(self, X, y=None):
        """Return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DL_CALL(self, "coder_fit_transform", (X,), _DL_CONTEXT)


# TOMBSTONE: _randomized_decompose (dead Python randomized PCA / TruncatedSVD arm: the owners in x_decomp/pca_randomized.mojo and x_decomp/tsvd_owner.mojo run it) deleted 2026-11-08 by bddde2ea2e; code recoverable at 41ffcec665.
# Restore: git apply experiments/removed/dead_python_loops.patch


# The compiled IncrementalPCA owner, connected once at import.
from . import _native_ipca_bridge  # noqa: E402
_IPCA_CALL, _IPCA_CONTEXT = _native_ipca_bridge.connect_public(IncrementalPCA)
# The compiled johnson_lindenstrauss_min_dim owner.
from . import _native_jl_bridge  # noqa: E402
_JL_CALL, _JL_CONTEXT = _native_jl_bridge.connect_public()


# The native dictionary-learning owner (bindings/dict_learning_native.mojo):
# DictionaryLearning, MiniBatchDictionaryLearning, SparseCoder, SparsePCA,
# MiniBatchSparsePCA and sparse_encode, one call per public operation.
from ._native_decomp_kit_bridge import connect_public as _connect_decomp_kit  # noqa: E402
_DL_CALL, _DL_CONTEXT = _connect_decomp_kit("dict_learning_native_call", minibatch_sparse_pca_class=MiniBatchSparsePCA)


# The native PLS owner (bindings/pls_native.mojo): PLSRegression, PLSCanonical, CCA.
# `score` runs R^2 through the native metrics owner (family_mcls_r2) inside
# the same call. Import connects the metrics tables; the native PLS owner
# selects and checks the current binding after entering compiled code.
from . import _metrics_impl as _pls_metrics  # noqa: E402,F401
_native_family_bridge.connect_bindings("_mojolearn_metrics")
_PLS_CALL, _PLS_CONTEXT = _connect_decomp_kit("pls_native_call", pls_regression_class=PLSRegression,
                                              pls_canonical_class=PLSCanonical, pls_cca_class=CCA,
                                              pls_metric_owner=_native_family_bridge.connect())


# The native MinCovDet owner (bindings/mcd_native.mojo): MinCovDet, EllipticEnvelope.
_MCD_CALL, _MCD_CONTEXT = _connect_decomp_kit("mcd_native_call", elliptic_envelope_class=EllipticEnvelope)


# The native AlternatingLeastSquares owner (bindings/als_native.mojo).
_ALS_CALL, _ALS_CONTEXT = _connect_decomp_kit("als_native_call")


# The native linear-algebra owner (bindings/linalg_decomp_native.mojo):
# randomized_svd, solve, lstsq, lu_factor and lu_solve. lu_factor returns an
# `_LUPair` whose device buffers are `_DevBuf`s.
_LINALG_CALL, _LINALG_CONTEXT = _connect_decomp_kit("linalg_decomp_native_call", lu_pair_class=_LUPair,
                                                    lu_device_buffer=_DevBuf)


# The compiled owner of the decomposition estimators whose public operations
# are one native call each (bindings/decomp_native.mojo): FastICA, ...
from . import _native_decomp_bridge  # noqa: E402
_DECOMP_CALL, _DECOMP_CONTEXT = _native_decomp_bridge.connect_public({
    "FastICA": FastICA, "NMF": NMF, "FactorAnalysis": FactorAnalysis,
    "LatentDirichletAllocation": LatentDirichletAllocation,
    "GaussianRandomProjection": GaussianRandomProjection, "SparseRandomProjection": SparseRandomProjection})
