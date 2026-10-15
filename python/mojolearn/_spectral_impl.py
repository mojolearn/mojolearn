# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Spectral clustering on the GPU. Reference: cuML's `SpectralClustering`.

The estimator is `SpectralClustering`. It is backed by `spectral/`, which
implements cuML 26.08's `ML::SpectralClustering::fit_predict` down through
cuVS's `cluster::spectral::detail::fit_predict` (the kNN connectivity
graph, the normalized graph Laplacian, RAFT's thick-restart Lanczos, and
`cluster/`'s already-implemented k-means) with every closed vendor library along
that path replaced by a named, numbered stand-in (DEVIATIONS 770-781).

**THIS LANE HAS RUN ON ONE APPLE M4 AND NOWHERE ELSE.** Read that before
reading anything else here. `spectral/IDENTICAL_SPECTRAL_CONTRACT.md`
section 10 says it in the lane's own words -- "no cross-vendor result of
any kind (nothing has run anywhere but one M4)" -- and the artifacts agree:
`spectral` is not one of the lanes `tools/e1_bootstrap.sh` phase 8 runs, it
has no card in either leg-11 vendor directory, and
`tools/e3_round_judge.sh` section 7 does not name it. The class docstring
below says what the profile PINS; a pinned convention is not a measured
result, and nothing in this module claims one.

The binding lives in `bindings/_mojolearn_metrics.mojo` alongside the
metrics functions and is loaded through `_metrics_impl._get_binding`.
"""

from . import _backend, _serialize
from ._mode import ParamsMixin
from ._array import Array
from ._buffer import (
    _native, addr, addr_ro, all_finite, as_f32_c, as_f64_c, as_i32_c, empty,
)
from ._expansion_decomp import _Kit, _M
from ._metrics_impl import _get_binding
from .density import _check_queries
from .linear_model import _check_saved_by, _restore_mode, _saved_mode, _shape_of

__all__ = ["SpectralClustering", "SpectralEmbedding", "spectral_embedding"]

#: `SpectralClustering.save`'s format tag (lane/spectral-predict,
#: 2026-09-15): the prediction data of a `prediction_data=True` fit.
_SPECTRAL_FORMAT = "mojolearn-spectral-1"

#: The prediction data a `prediction_data=True` fit keeps.
_PREDICTION_ATTRS = ("_pd_eigenvalues", "_pd_eigenvectors", "_pd_diag", "_pd_centroids", "_fit_X")

_AFFINITIES = ("nearest_neighbors", "precomputed")
#: SpectralEmbedding's affinities: the two above, scikit-learn's 'rbf' and
#: 'precomputed_nearest_neighbors' (lane/algos-decomp, 2026-09-27).
_EMBED_AFFINITIES = _AFFINITIES + ("rbf", "precomputed_nearest_neighbors")
#: SpectralClustering's: the same four (lane/algos-cluster option parity,
#: 2026-09-27); 'rbf' is scikit-learn's default, 'nearest_neighbors' cuML's
#: and this class's.
_CLUSTER_AFFINITIES = _EMBED_AFFINITIES
#: SpectralClustering's label assignments: cuVS's k-means, and scikit-learn's
#: two others on the same embedding (x_cluster/spectral_assign.mojo).
_ASSIGN_LABELS = ("kmeans", "discretize", "cluster_qr")
#: `x_cluster/entries.mojo::ENTRY_SPECTRAL_ASSIGN` and its method codes.
_E_SPECTRAL_ASSIGN = 12
_ASSIGN_CODES = {"discretize": 0, "cluster_qr": 1}

#: cuVS's own struct default for the eigensolver tolerance
#: (`cuvs/preprocessing/spectral_embedding.hpp:59`, `tolerance{1e-5f}`).
DEFAULT_EIGEN_TOL = 1e-5

#: cuVS's own struct default seed (`cuvs/cluster/spectral.hpp`,
#: `raft::random::RngState rng_state{0}`).
DEFAULT_SEED = 0


def _coo_triples(A, who="SpectralClustering"):
    """A precomputed affinity matrix as `(rows, cols, vals, n)` int32/float32.

    Same behavior as cuML's `spectral_clustering.pyx:306-312`, which accepts scipy or
    cupy sparse in COO/CSR/CSC and calls `sp.coo_matrix(X)` on a dense
    input. Converting a dense matrix through COO DROPS EXACT ZEROS, which is
    theirs and is also what makes a dense input usable at all.

    A sparse input is recognized by DUCK TYPE, not by importing SciPy: any
    object with a `tocoo()` whose result carries `shape`, `row`, `col` and
    `data` is taken as sparse, and those three arrays are read through the
    buffer protocol like every other input (DEVIATION 2489). SciPy, CuPy
    (after `.get()`), and anything else shaped like them qualify; nothing
    is imported to ask.
    """
    tocoo = getattr(A, "tocoo", None)
    if callable(tocoo):
        coo = tocoo()
        shape = tuple(coo.shape)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                f"mojolearn {who}: a precomputed affinity matrix "
                f"must be square, got shape {shape}"
            )
        rows, _ = as_i32_c(coo.row, ndim=1, name="rows")
        cols, _ = as_i32_c(coo.col, ndim=1, name="cols")
        vals, _ = as_f32_c(coo.data, ndim=1, name="vals")
        return rows, cols, vals, int(shape[0])
    shape = _shape_of(A)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            f"mojolearn {who}: with affinity='precomputed', X "
            "must be a square affinity matrix or a sparse matrix with "
            f"`tocoo()`, got shape {shape}"
        )
    # The nonzero test is made in float64, the widest dtype the boundary
    # carries, so an entry that is nonzero in a float64 input and rounds to
    # 0.0f is KEPT as an explicit zero, exactly as `np.nonzero` on the
    # source dtype kept it; the values are narrowed to float32 afterwards,
    # one rounding each. -0.0 is a zero, NaN is not (it compares unequal).
    dense, _ = as_f64_c(A, ndim=2, name="X")
    n = int(shape[0])
    # DEVIATION 2489: the scan in Mojo, two calls, count then fill into
    # buffers this side allocates. The Python loop this replaced
    # (DEVIATION 2373) lives on only as the oracle in
    # tests/test_native_nonzero.py, which holds the two to byte equality.
    nnz = int(_native("nonzero_f64_count")(addr_ro(dense, name="X"), dense.size)) if dense.size else 0  # cpu-route: CPU-only host binding COO route (dense device entry absent)
    rows = empty((nnz,), "<i4")
    cols = empty((nnz,), "<i4")
    vals = empty((nnz,), "<f4")
    if nnz == 0:
        # Nothing to write, and an empty Array has no address to hand the
        # binding (its pointer helpers refuse a null by design).
        return rows, cols, vals, n
    wrote = int(_native("nonzero_f64_fill")(  # cpu-route: CPU-only host binding COO route (dense device entry absent)
        addr_ro(dense, name="X"), n, n,
        [addr(rows, name="rows"), addr(cols, name="cols"),
         addr(vals, name="vals")],
        nnz,
    ))
    if wrote != nnz:
        raise RuntimeError(
            f"mojolearn {who}: nonzero_f64_fill wrote "
            f"{wrote} of {nnz} entries"
        )
    return rows, cols, vals, n


def _dense_entry(name):
    """The metrics binding's dense-affinity entry `name` (lane
    cpu2-l9-neighbors, 2026-10-04: `spectral_fit_predict_dense`,
    `spectral_embedding_dense`), or None on a binding without it, the
    CPU-only host binding, whose route is the COO scan (`_coo_triples`,
    `_DenseCOO`). On the GPU route the dense matrix is uploaded once and its
    COO compacted, checked and turned into the Laplacian on the device
    (spectral/impl/sparse/linalg/detail/dense_graph.mojo): the same COO, so
    the same bits, with no n^2 host scan."""
    try:
        return getattr(_get_binding(), name)
    except (AttributeError, ImportError):
        return None


def _dense_square(X, who):
    """A dense precomputed affinity as a C-order float32 square Array (the
    dense entries' input), with `_coo_triples`' shape refusal. The nonzero
    test is made on the float32 values on the device route: a float64
    magnitude that narrows to 0.0f is a zero there."""
    shape = _shape_of(X)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            f"mojolearn {who}: with affinity='precomputed', X "
            "must be a square affinity matrix or a sparse matrix with "
            f"`tocoo()`, got shape {shape}"
        )
    return as_f32_c(X, ndim=2, name="X")[0]


#: The dense entries' refusal codes (bindings/_mojolearn_metrics.mojo
#: `spectral_fit_predict_dense_binding`), in the COO route's check order.
_DENSE_EMPTY, _DENSE_NONFINITE, _DENSE_NEGATIVE = -1, -2, -3


# Maintainer notes (moved from the docstring). Reference: cuML 26.08's
# SpectralClustering; pipeline as in cuVS. The identity profile is
# `spectral/IDENTICAL_SPECTRAL_CONTRACT.md` (mojolearn.identical.spectral.fp32.v1).
#
# Sign rule (DEVIATION 770): after the projected eigenproblem is solved and its
# eigenvalues sorted, every column of the ncv x ncv PROJECTED eigenvector matrix
# is negated iff its first nonzero component (x != 0.0, so a leading +-0.0 is
# skipped) in ascending row index is negative; an all-zero column is left
# alone. Applied there and nowhere else: there is no sign pin on the returned
# embedding. Its signs follow from the pinned start vector (DEVIATION 772, a
# splitmix64 hashed uniform that is an exact function of (seed, n)). Adding a
# second rule would be a different profile. RAFT pins neither convention.
#
# Ordering and ties (DEVIATION 778): the projected solver returns eigenvalues
# ascending (cuSOLVER syevd's convention, which RAFT's slicing assumes). This
# path uses which=LA on the NEGATED Laplacian and takes the LAST k, so index
# k-1 is the Laplacian's smallest eigenvalue; the reversed gather puts the
# second-smallest in embedding column 0. Ties break by original index: the
# ascending sort is an insertion sort with the strict comparison
# d[order[j]] > d[key], so equal eigenvalues keep their Jacobi-diagonal order,
# a pure function of the input bits and the pinned sweep schedule. +0.0 and
# -0.0 compare equal, deliberately. LM and SM are refused rather than
# implemented: theirs sorts by magnitude with thrust::sort, which is not
# stable, so implementing it would mean inventing a tie-break.
#
# Degeneracy: same input bits and profile give the same eigenvector bits,
# including inside a degenerate subspace (the Jacobi sweep is a fixed
# sequence). The basis inside a degenerate subspace is NOT stable under
# perturbation: a one-ulp change can rotate it arbitrarily.
#
# Open coverage gaps: the device SIGN_FLIP sabotage bites only on the `hashed`
# fixture (inert on path, ring, hashed_unnorm, blobs; archive/plans/CARD_GAPS.md:78,
# contract section 9); two host instruments also cover the rule.
# spectral.ritz.vectors is recorded AFTER the sign pin, so a whole-column sign
# disagreement between vendors would be re-canonicalized before hashing
# (CARD_GAPS.md, "Three more cards that hash on the far side of a washer").
# Seam J4: the fused ROTATE (DEVIATION 781) lives in a solver shared by the
# device arm and the oracle; the hash-against-a-literal check is owed
# (spectral/NOT_IMPLEMENTED.tsv, "this lane's own gaps").
#
# DEVIATION 780, so nobody re-inflates it: of the five constants once claimed
# as ours (ncv = min(n - k, max(2k+1, 20)), max_iterations = 10 * n_samples,
# the plumbed tolerance, the Jacobi sweep cap, the ncv admissibility bound),
# three were struck on 2026-08-23: they are identical to cuVS 26.08, down to
# the RAFT_EXPECTS message (cuVS 25.08 spells them as literals). Ours: the host
# Jacobi's 60-sweep cap and the ncv admissibility guard.
#
# eigen_tol='auto' refused (DEVIATION 890): cuML 26.08's Python maps 'auto' to
# 0.0 (spectral_clustering.pyx:346-347), passed through as the tolerance; that
# disables the convergence test (res > tol), runs to 10 * n_samples iterations
# and drives an exactly converged problem into DEVIATION 774's
# restart-breakdown refusal. Default 1e-5 is cuVS's struct default
# (spectral_embedding.hpp:59).
# random_state=None means seed 0 (DEVIATION 891): cuVS's struct default is
# rng_state{0}, and the no-seed Lanczos start vector is refused (DEVIATION 772;
# std::random_device is not reproducible).
# n_init=10 is what cluster/detail/spectral.cuh passes (KMeans defaults to 1).
# Normalization: cluster/detail/spectral.cuh:35-36 hard-codes
# norm_laplacian = true and drop_first = false on the clustering path.
# assign_labels 'discretize'/'cluster_qr': x_cluster/spectral_assign.mojo, host
# float64, a Jacobi SVD for LAPACK's (DEVIATION 5119); discretize draws its
# start row from the splitmix64 stream seeded by random_state.
# predict: DEVIATION 2860 (see the notes above `predict`).
class SpectralClustering(ParamsMixin):
    """Spectral clustering on a nearest-neighbor or precomputed affinity graph.

    Builds an affinity graph, forms the normalized graph Laplacian, takes its
    ``n_components`` smallest eigenvectors with a thick-restart Lanczos solver
    and clusters the resulting embedding. Defaults follow cuML's
    ``SpectralClustering``, which differ from scikit-learn's: the default
    affinity is ``"nearest_neighbors"`` with ``n_neighbors=10``, and
    ``n_components`` defaults to ``n_clusters``. The Laplacian is always
    normalized and the first (nearly constant) eigenvector is kept, so the
    clustering runs on all ``n_components`` columns; ``embedding_`` is
    therefore not what :class:`SpectralEmbedding` returns.

    Eigenvectors are only defined up to sign, and a basis inside a repeated
    eigenvalue is arbitrary. Results are reproducible for the same input and
    seed, but a graph with ``c`` connected components has eigenvalue 0 with
    multiplicity ``c``, and an embedding with ``n_components <= c`` can change
    a lot under tiny input changes.

    Parameters
    ----------
    n_clusters : int, default 8
        Number of clusters, in ``[1, n_samples]``.
    n_components : int or None, default None
        Number of eigenvectors; None uses ``n_clusters``. Must satisfy
        ``1 <= n_components < n_samples``.
    random_state : int or None, default None
        Seed in ``[0, 2**32)``. None means seed 0, so results are
        deterministic (unlike scikit-learn and cuML, where None draws a fresh
        seed).
    n_neighbors : int, default 10
        Neighbors per row in the nearest-neighbor graph; ignored with
        ``affinity="precomputed"``.
    n_init : int, default 10
        Number of k-means restarts on the embedding (cuVS's spectral default;
        :class:`KMeans` itself defaults to 1).
    eigen_tol : float, default 1e-5
        Convergence tolerance of the eigensolver; must be a positive float.
        The string ``"auto"`` raises ``NotImplementedError``: cuML maps it to
        0.0, which disables the convergence test.
    affinity : str, default "nearest_neighbors"
        ``"nearest_neighbors"`` (symmetrized k-NN connectivity of ``X``),
        ``"precomputed"`` (``X`` is a square affinity matrix, dense or sparse),
        ``"rbf"`` (``exp(-gamma * ||x_i - x_j||**2)``, scikit-learn's kernel)
        or ``"precomputed_nearest_neighbors"`` (k-NN connectivity of a
        precomputed distance matrix, symmetrized as ``0.5 * (C + C.T)``). A
        callable or any other name raises ``ValueError``.
    assign_labels : {"kmeans", "discretize", "cluster_qr"}, default "kmeans"
        How labels are assigned from the embedding: k-means, or
        scikit-learn's ``"discretize"`` and ``"cluster_qr"``.
    eigen_solver : None
        Only None is accepted; any other value raises ``NotImplementedError``.
    gamma : float or None, default None
        RBF kernel coefficient, used only with ``affinity="rbf"``; None means
        1.0 (scikit-learn's default). Setting it with another affinity raises
        ``NotImplementedError``.
    degree, coef0, kernel_params : None
        Not supported; any value raises ``NotImplementedError``.
    n_jobs : None
        Not supported; any value raises ``NotImplementedError``.
    verbose : bool, default False
        Not supported; True raises ``NotImplementedError``.
    prediction_data : bool, default False
        Keep the eigenpairs, degree scaling and k-means centroids (and, for
        ``"nearest_neighbors"``, the training rows) so that ``predict`` and
        ``save`` work. It does not change the fitted labels or embedding.
        Supported with ``assign_labels="kmeans"`` and the
        ``"nearest_neighbors"`` or ``"precomputed"`` affinities only;
        otherwise ``NotImplementedError``.

    Attributes
    ----------
    labels_ : Array of shape (n_samples,), int32
        Cluster label of each training row, in ``[0, n_clusters)``.
    embedding_ : Array of shape (n_samples, n_components), float32
        The spectral embedding that was clustered.
    n_components_ : int
        Number of eigenvectors used.
    affinity_matrix_ : array
        The affinity matrix used by the fit: ``X`` itself for
        ``"precomputed"``, the dense matrix for ``"rbf"`` and
        ``"precomputed_nearest_neighbors"``. Not set for
        ``"nearest_neighbors"``.
    n_features_in_ : int
        Number of features of ``X``; set for ``"nearest_neighbors"`` and
        ``"rbf"``.
    n_iter_assign_ : int
        Iterations of the ``"discretize"`` label assignment; set only for it.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.SpectralClustering(n_clusters=4, random_state=0).fit(X)
    >>> labels = model.labels_
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "clusterer"

    def __init__(
        self,
        n_clusters=8,
        *,
        n_components=None,
        random_state=None,
        n_neighbors=10,
        n_init=10,
        eigen_tol=DEFAULT_EIGEN_TOL,
        affinity="nearest_neighbors",
        assign_labels="kmeans",
        eigen_solver=None,
        gamma=None,
        degree=None,
        coef0=None,
        kernel_params=None,
        n_jobs=None,
        verbose=False,
        prediction_data=False,
    ):
        # Stored verbatim, as scikit-learn requires; `_validate_params`
        # checks them when `fit` (or `load`) runs.
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_init = n_init
        self.eigen_tol = eigen_tol
        self.affinity = affinity
        self.gamma = gamma
        self.assign_labels = assign_labels
        self.prediction_data = prediction_data
        self.eigen_solver = eigen_solver
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _validate_params(self):
        """The constructor checks, run first in `fit` and `load`; the
        converted values go to private attributes, which the fit reads."""
        n_clusters, n_components, random_state = self.n_clusters, self.n_components, self.random_state
        n_neighbors, n_init, eigen_tol, affinity = self.n_neighbors, self.n_init, self.eigen_tol, self.affinity
        assign_labels, eigen_solver, gamma, degree = self.assign_labels, self.eigen_solver, self.gamma, self.degree
        coef0, kernel_params, n_jobs, verbose = self.coef0, self.kernel_params, self.n_jobs, self.verbose
        prediction_data = self.prediction_data
        if affinity not in _CLUSTER_AFFINITIES:
            raise ValueError(
                f"mojolearn SpectralClustering: affinity={affinity!r} is "
                f"refused; it must be one of {list(_CLUSTER_AFFINITIES)}. A "
                "callable or another kernel name builds the affinity outside "
                "every identity column."
            )
        if assign_labels not in _ASSIGN_LABELS:
            raise ValueError(
                f"mojolearn SpectralClustering: assign_labels="
                f"{assign_labels!r} must be one of {list(_ASSIGN_LABELS)}"
            )
        if prediction_data and assign_labels != "kmeans":
            raise NotImplementedError(
                f"mojolearn SpectralClustering: prediction_data with "
                f"assign_labels={assign_labels!r} is refused; the Nystrom "
                "predict (DEVIATION 2860) assigns by the k-means centroids")
        if eigen_solver is not None:
            raise NotImplementedError(
                f"mojolearn SpectralClustering: eigen_solver={eigen_solver!r} "
                "is refused; there is one solver here, RAFT's thick-restart "
                "Lanczos, and it is the one the identity profile pins"
            )
        if gamma is not None and affinity != "rbf":
            raise NotImplementedError(
                "mojolearn SpectralClustering: gamma is refused; it "
                "parameterizes the RBF affinity, and affinity is not 'rbf'")
        if prediction_data and affinity in ("rbf", "precomputed_nearest_neighbors"):
            raise NotImplementedError(
                f"mojolearn SpectralClustering: prediction_data with affinity={affinity!r} "
                "is refused; the Nystrom predict (DEVIATION 2860) carries the "
                "'nearest_neighbors' and 'precomputed' affinities only")
        for name, value in (  # glue: checks keyword argument values
            ("degree", degree),
            ("coef0", coef0),
            ("kernel_params", kernel_params),
        ):
            if value is not None:
                raise NotImplementedError(
                    f"mojolearn SpectralClustering: {name} is refused; it "
                    "parameterizes an RBF or polynomial affinity, and "
                    "affinity is restricted to 'nearest_neighbors' and "
                    "'precomputed'"
                )
        if n_jobs is not None:
            raise NotImplementedError(
                "mojolearn SpectralClustering: n_jobs is refused; the work "
                "is on the GPU and there is no host thread pool to size"
            )
        if verbose:
            raise NotImplementedError(
                "mojolearn SpectralClustering: verbose is refused; nothing "
                "in this path prints. Set MOJOLEARN_IDENTITY_TRACE=<path> "
                "for the identity card instead"
            )
        if isinstance(eigen_tol, str):
            raise NotImplementedError(
                f"mojolearn SpectralClustering: eigen_tol={eigen_tol!r} is "
                "refused (DEVIATION 890). cuML 26.08's Python maps 'auto' to "
                "0.0, which is passed straight through as the eigensolver "
                "tolerance and disables the convergence test (the loop "
                "condition is res > tol), so the solve runs to "
                "max_iterations = 10 * n_samples and an exactly converged "
                "problem hits the restart-breakdown refusal. Pass a positive "
                f"float; {DEFAULT_EIGEN_TOL} is cuVS's own struct default "
                "and is this class's default."
            )
        eigen_tol = float(eigen_tol)
        if not (eigen_tol > 0.0) or eigen_tol != eigen_tol:
            raise ValueError(
                "mojolearn SpectralClustering: eigen_tol must be positive, "
                f"got {eigen_tol!r} (DEVIATION 890; see the 'auto' note)"
            )
        if random_state is None:
            seed = DEFAULT_SEED
        else:
            seed = int(random_state)
            if seed < 0 or seed >= 2**32:
                raise ValueError(
                    "mojolearn SpectralClustering: random_state must satisfy "
                    f"0 <= random_state < 2**32, got {random_state!r} (cuML's "
                    "check_random_seed refuses the same range)"
                )
        if int(n_clusters) < 1:
            raise ValueError(
                "mojolearn SpectralClustering: n_clusters must be at least 1"
            )
        if int(n_init) < 1:
            raise ValueError(
                "mojolearn SpectralClustering: n_init must be at least 1"
            )
        if int(n_neighbors) < 1:
            raise ValueError(
                "mojolearn SpectralClustering: n_neighbors must be at least 1"
            )
        if n_components is not None and int(n_components) < 1:
            raise ValueError(
                "mojolearn SpectralClustering: n_components must be at least "
                "1, or None to follow n_clusters"
            )
        self._n_clusters = int(n_clusters)
        self._n_neighbors = int(n_neighbors)
        self._n_init = int(n_init)
        self._eigen_tol = eigen_tol
        #: scikit-learn's SpectralClustering default gamma is 1.0 (its
        #: SpectralEmbedding's is 1 / n_features).
        self._gamma = (1.0 if gamma is None else float(gamma)) if affinity == "rbf" else None
        self._seed = seed

    #: The binding `predict` asks; a host subclass answers the metrics host
    #: binding, which ships in the inference wheel.
    _BINDING = "_mojolearn_metrics"

    def _bind(self, name=None):
        return _get_binding(getattr(self, "numeric_mode", None))

    def _state_arrays(self, n, k):
        return (empty((k,), "<f4"), empty((n, k), "<f4"), empty((n,), "<f4"),
                empty((self._n_clusters, k), "<f4"))

    def _n_components(self):
        """cuML's rule: `n_components` defaults to `n_clusters` when None
        (`spectral_clustering.pyx`)."""
        return self._n_clusters if self.n_components is None else int(
            self.n_components
        )

    def fit(self, X, y=None):
        """Cluster ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Training data, or with ``affinity="precomputed"`` a square
            nonnegative affinity matrix (dense or sparse); with
            ``"precomputed_nearest_neighbors"`` a square distance matrix.
            Values must be finite.
        y : ignored

        Returns
        -------
        self : SpectralClustering
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` holds NaN or infinity, a precomputed affinity has no
            nonzero entries, ``n_clusters`` exceeds ``n_samples`` or
            ``n_components`` is not below ``n_samples``.
        TypeError
            If ``prediction_data`` is not a bool.
        """
        self._validate_params()
        if not isinstance(self.prediction_data, bool):
            raise TypeError(
                "mojolearn SpectralClustering: prediction_data must be a bool, "
                f"got {type(self.prediction_data).__name__}"
            )
        for name in _PREDICTION_ATTRS:  # glue: drops cached attribute names
            self.__dict__.pop(name, None)
        k = self._n_components()
        labels_out = None
        state = None
        self.__dict__.pop("affinity_matrix_", None)
        dense_fn = None
        if self.affinity in ("precomputed", "rbf", "precomputed_nearest_neighbors"):
            dense_fn = _dense_entry("spectral_fit_predict_dense")
        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
            if dense_fn is not None and not callable(getattr(X, "tocoo", None)):
                X = _dense_square(X, "SpectralClustering")
        elif self.affinity == "rbf":
            # scikit-learn's affinity='rbf' (cluster/_spectral.py:
            # pairwise_kernels(X, metric='rbf', gamma=self._gamma)), which
            # cuML does not carry: the dense exp(-gamma ||x_i - x_j||^2) by
            # the decomp lane's identical cells, SpectralEmbedding's rbf
            # route exactly (x_decomp/cells.mojo `sqdist_cell`, `exp_c`),
            # then the precomputed route below unchanged.
            xm = _M.from_input(X)
            mode = _backend.default_mode()
            kit = _Kit(mode, _backend.binding("_mojolearn_x_decomp", mode))
            aff = kit.ew("exp", kit.ew("scale", kit.sqdist(xm, xm), s=-self._gamma))
            self.affinity_matrix_ = aff.out()
            self.n_features_in_ = xm.c
            X = aff if dense_fn is not None else _DenseCOO(aff)
        elif self.affinity == "precomputed_nearest_neighbors":
            # scikit-learn's kneighbors_graph of a precomputed distance
            # matrix, symmetrized 0.5 (C + C^T): SpectralEmbedding's helper
            # (comparisons and the exact values 0, 0.5, 1 only).
            aff = SpectralEmbedding._precomputed_knn_affinity(self, X)
            self.affinity_matrix_ = aff.out()
            X = aff if dense_fn is not None else _DenseCOO(aff)
        if dense_fn is not None and (isinstance(X, _M) or isinstance(X, Array)):
            # the dense matrix (rbf / kNN affinity host store, or the
            # precomputed float32 input) straight to the device entry
            if isinstance(X, _M):
                n, dense_addr = X.r, X.addr
            else:
                n, dense_addr = int(X.shape[0]), addr_ro(X, name="X")
            self._check_shape(n, k)
            labels = empty((n,), "<i4")
            embedding = empty((n, k), "<f4")
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::
            # spectral_fit_predict_dense_binding.
            params = [n, self._n_clusters, k, self._n_init, self._n_neighbors,
                      self._eigen_tol, self._seed]
            addrs = [dense_addr, addr(labels, name="labels"), addr(embedding, name="embedding")]
            if self.prediction_data:
                state = self._state_arrays(n, k)
                addrs += [addr(a, name="prediction data") for a in state]  # glue: addresses of prediction-state buffers
            n_out = int(dense_fn(addrs, params))
            if n_out == _DENSE_EMPTY:
                raise ValueError(
                    "mojolearn SpectralClustering: the precomputed affinity "
                    "matrix has no nonzero entries"
                )
            if n_out == _DENSE_NONFINITE:
                raise ValueError(
                    "mojolearn SpectralClustering: the precomputed affinity "
                    "matrix has a non-finite entry (the implemented path refuses "
                    "it by name, because sqrt of a non-finite degree is a "
                    "NaN and no NaN may reach a recorded value)"
                )
            if n_out == _DENSE_NEGATIVE:
                raise ValueError(
                    "mojolearn SpectralClustering: the precomputed affinity "
                    "matrix has a negative entry (refused by name: sqrt of a "
                    "negative degree is a NaN in cuVS too)"
                )
            labels_out = labels
        elif self.affinity in ("precomputed", "rbf", "precomputed_nearest_neighbors"):
            # the COO route: a sparse precomputed input, or a binding without
            # the dense entry (the CPU-only host binding)
            rows, cols, vals, n = _coo_triples(X)
            if vals.size == 0:
                raise ValueError(
                    "mojolearn SpectralClustering: the precomputed affinity "
                    "matrix has no nonzero entries"
                )
            if not all_finite(vals):
                raise ValueError(
                    "mojolearn SpectralClustering: the precomputed affinity "
                    "matrix has a non-finite entry (the implemented path refuses "
                    "it by name, because sqrt of a non-finite degree is a "
                    "NaN and no NaN may reach a recorded value)"
                )
            if vals.min() < 0:
                raise ValueError(
                    "mojolearn SpectralClustering: the precomputed affinity "
                    "matrix has a negative entry (refused by name: sqrt of a "
                    "negative degree is a NaN in cuVS too)"
                )
            self._check_shape(n, k)
            labels = empty((n,), "<i4")
            embedding = empty((n, k), "<f4")
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::
            # spectral_fit_predict_graph_binding.
            # n_samples, nnz, n_clusters, n_components, n_init, n_neighbors,
            # eigen_tol, seed
            params = [
                n,
                int(vals.shape[0]),
                self._n_clusters,
                k,
                self._n_init,
                self._n_neighbors,
                self._eigen_tol,
                self._seed,
            ]
            if self.prediction_data:
                state = self._state_arrays(n, k)
                # spectral_fit_predict_graph_state_binding: the same call,
                # plus copies of the prediction data.
                n_out = int(_get_binding().spectral_fit_predict_graph_state(
                    [addr_ro(rows, name="rows"), addr_ro(cols, name="cols"),
                     addr_ro(vals, name="vals"), addr(labels, name="labels"),
                     addr(embedding, name="embedding")]
                    + [addr(a, name="prediction data") for a in state],  # glue: addresses of prediction-state buffers
                    params,
                ))
            else:
                n_out = int(
                    _get_binding().spectral_fit_predict_graph(
                        addr_ro(rows, name="rows"),
                        addr_ro(cols, name="cols"),
                        addr_ro(vals, name="vals"),
                        addr(labels, name="labels"),
                        addr(embedding, name="embedding"),
                        params,
                    )
                )
            labels_out = labels
        else:
            x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
            if not all_finite(x):
                raise ValueError(
                    "mojolearn SpectralClustering: X contains NaN or "
                    "infinity (the implemented path refuses it by name before any "
                    "recorded stage)"
                )
            n = int(x.shape[0])
            self._check_shape(n, k)
            if self._n_neighbors > n:
                raise ValueError(
                    f"mojolearn SpectralClustering: n_neighbors="
                    f"{self._n_neighbors} exceeds n_samples={n}"
                )
            labels = empty((n,), "<i4")
            embedding = empty((n, k), "<f4")
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::
            # spectral_fit_predict_dataset_binding.
            # n_samples, n_features, n_clusters, n_components, n_init,
            # n_neighbors, eigen_tol, seed
            params = [
                n,
                int(x.shape[1]),
                self._n_clusters,
                k,
                self._n_init,
                self._n_neighbors,
                self._eigen_tol,
                self._seed,
            ]
            if self.prediction_data:
                state = self._state_arrays(n, k)
                n_out = int(_get_binding().spectral_fit_predict_dataset_state(
                    [addr_ro(x, name="x"), addr(labels, name="labels"),
                     addr(embedding, name="embedding")]
                    + [addr(a, name="prediction data") for a in state],  # glue: addresses of prediction-state buffers
                    params,
                ))
                # A borrowed input is copied so a caller's later write cannot
                # move what predict reads.
                self._fit_X = x if self.input_copied_ else x.copy()
            else:
                n_out = int(
                    _get_binding().spectral_fit_predict_dataset(
                        addr_ro(x, name="x"),
                        addr(labels, name="labels"),
                        addr(embedding, name="embedding"),
                        params,
                    )
                )
            labels_out = labels
            self.n_features_in_ = int(x.shape[1])
        if n_out != k:
            # Cannot fire today: the clustering path hard-codes
            # `drop_first = false`, so `n_out == n_components`. The binding
            # checks the same thing before it writes anything, and this is
            # the reader-facing half of that guard.
            raise RuntimeError(
                f"mojolearn SpectralClustering: the kernel returned {n_out} "
                f"embedding columns where {k} were expected"
            )
        self.embedding_ = embedding
        self.n_components_ = k
        self.__dict__.pop("n_iter_assign_", None)
        if self.assign_labels != "kmeans":
            labels_out = self._assign(embedding, n, k)
        self.labels_ = labels_out
        if state is not None:
            (self._pd_eigenvalues, self._pd_eigenvectors, self._pd_diag,
             self._pd_centroids) = state
        return self

    def _assign(self, embedding, n, k):
        """scikit-learn's discretize / cluster_qr on the fit's embedding, the
        x_cluster binding's host code (ENTRY_SPECTRAL_ASSIGN)."""
        from . import _expansion_cluster
        b = _backend.binding(_expansion_cluster._XCluster._BINDING, getattr(self, "numeric_mode", None))
        ip = [n, k, _ASSIGN_CODES[self.assign_labels], int(self._seed), 30, 20]
        _f, i, sc = b.x_cluster_call(_E_SPECTRAL_ASSIGN, addr_ro(embedding, name="embedding"),
                                     embedding.size, 0, 0, ip, [0.0])
        if self.assign_labels == "discretize":
            self.n_iter_assign_ = int(sc[0])
        return Array._from_flat(i[0], (n,), "<i4")

    def _resolved_neighbors(self, n):
        """`SpectralEmbedding._precomputed_knn_affinity` reads this."""
        return self._n_neighbors

    def _check_shape(self, n, k):
        if self._n_clusters > n:
            raise ValueError(
                f"mojolearn SpectralClustering: n_clusters={self._n_clusters} "
                f"exceeds n_samples={n}"
            )
        if k >= n:
            raise ValueError(
                f"mojolearn SpectralClustering: n_components={k} must be "
                f"below n_samples={n}; the Lanczos requires "
                "1 <= n_components < n_samples and n_samples - n_components "
                "> 0 (cuVS's own RAFT_EXPECTS)"
            )

    def fit_predict(self, X, y=None):
        """Fit to ``X`` and return the training labels.

        Parameters
        ----------
        X : array-like
            As in ``fit``.
        y : ignored

        Returns
        -------
        labels : Array of shape (n_samples,), int32
            Same as ``labels_``.
        """
        return self.fit(X, y=y).labels_

    # predict, maintainer notes (DEVIATION 2860). Nystrom extension of Bengio,
    # Paiement, Vincent, Delalleau, Le Roux and Ouimet (NIPS 2003), as Fowlkes
    # et al. (TPAMI 2004) use it, then the fit's own k-means assignment. For a
    # new row x and column c:
    #     e_c(x) = (1 / mu_c) * sum_i Ktilde(x, x_i) u_c[i] / sqrt(d(x))
    #     Ktilde(x, x_i) = K(x, x_i) / (sqrt(d(x)) * sqrt(d_i))
    # u_c is the fit's unit eigenvector before its degree division,
    # mu_c = 1 + theta_c the normalized affinity's eigenvalue (theta_c the Ritz
    # value of the negated normalized Laplacian), d_i the fit's degree and
    # d(x) = sum_i K(x, x_i); the last division by sqrt(d(x)) is the fit's own
    # row scaling. A 'nearest_neighbors' query edge weighs 0.5: the fit's
    # symmetrization 0.5 * (a + b) of an edge with no reverse edge. Every fold
    # (degree, each projection) runs over training rows in ascending index,
    # seeded +0.0, one query at a time, so labels do not depend on the batch
    # and the GPU binding and the CPU host binding compute the same bytes.
    # Nothing is dropped (the clustering fit keeps the trivial column); a
    # column with |mu_c| < 1e-3 is refused by name. How often
    # predict(X_train) == labels_ holds is measured in
    # bench/results/identity_break/2026-09-15_spectral-predict/.
    def predict(self, X):
        r"""Assign new rows to the fitted clusters.

        Neither scikit-learn nor cuML provides ``predict`` for spectral
        clustering. This one uses the Nystrom out-of-sample extension (Bengio
        et al., NIPS 2003): each new row is projected onto the fitted
        eigenvectors through its affinity to the training rows, then assigned
        to the nearest fitted k-means centroid (ties go to the lowest
        centroid index). A row's label does not depend on the other rows in
        the call.

        Requires a fit with ``prediction_data=True``.

        The affinity of a new row:

        - ``"nearest_neighbors"``: its ``n_neighbors`` nearest training rows
          (Euclidean), each with weight 0.5. Training degrees are unchanged.
        - ``"precomputed"``: ``X`` is the affinity between the new rows and
          the training rows, shape ``(n_new, n_train)``, finite and
          nonnegative. A sparse matrix is densified.

        Training rows passed to ``predict`` are not guaranteed to get their
        ``labels_``: as a query, a row sees itself with weight 0.5 rather
        than 1.0 and has no reverse edges.

        Parameters
        ----------
        X : array-like of shape (n_new, n_features) or (n_new, n_train)
            New rows, or their affinity to the training rows.

        Returns
        -------
        labels : Array of shape (n_new,), int32
            Cluster labels, numbered as in ``labels_``.

        Raises
        ------
        ValueError
            If the model is not fitted, was fitted without
            ``prediction_data=True``, ``X`` has the wrong shape or no rows,
            or a precomputed affinity is non-finite or negative.

        Notes
        -----
        Every embedding column is kept. If an eigenvalue of the normalized
        affinity has magnitude below 1e-3 (the projection would amplify that
        column more than 1000x), the prediction is refused with an error
        naming the column.
        """
        return self._predict_embedding(X)[0]

    def _predict_embedding(self, X):
        """`predict`'s labels and the extended embedding (n_new x
        n_components float32)."""
        if not hasattr(self, "labels_"):
            raise ValueError(
                "mojolearn SpectralClustering.predict: this instance is not "
                "fitted yet; call fit first"
            )
        if getattr(self, "_pd_eigenvalues", None) is None:
            raise ValueError(
                "mojolearn SpectralClustering.predict: prediction data was not "
                "stored. Fit with SpectralClustering(prediction_data=True), which "
                "keeps the eigenpairs, degrees and centroids the Nystrom extension "
                "needs (DEVIATION 2860)"
            )
        if self.affinity not in _AFFINITIES:
            raise ValueError(
                f"mojolearn SpectralClustering.predict: affinity={self.affinity!r} "
                "has no out-of-sample rule; it must be one of "
                f"{list(_AFFINITIES)}"
            )
        k = int(self.n_components_)
        n_train = int(self._pd_diag.shape[0])
        if self.affinity == "precomputed":
            A = X.toarray() if callable(getattr(X, "toarray", None)) else X
            shape = _shape_of(A)
            if len(shape) != 2 or shape[1] != n_train:
                raise ValueError(
                    "mojolearn SpectralClustering.predict: with affinity="
                    "'precomputed', X must be the affinity between the new rows "
                    f"and the {n_train} training rows, shape (n_new, {n_train}); "
                    f"got shape {shape}"
                )
            q, _ = as_f32_c(A, ndim=2, name="X")
            if q.shape[0] < 1:
                raise ValueError("mojolearn SpectralClustering.predict: X has no rows; refused by name")
            if not all_finite(q):
                raise ValueError(
                    "mojolearn SpectralClustering.predict: the affinity has a "
                    "non-finite entry; refused by name"
                )
            if q.min() < 0:
                raise ValueError(
                    "mojolearn SpectralClustering.predict: the affinity has a "
                    "negative entry; refused by name, as the fit refuses one"
                )
            train_addr, n_features, affinity = 0, 0, 1
        else:
            q = _check_queries(X, self.n_features_in_, "SpectralClustering.predict")
            train_addr = addr_ro(self._fit_X, name="training rows")
            n_features, affinity = int(self.n_features_in_), 0
        nq = int(q.shape[0])
        labels = empty((nq,), "<i4")
        emb = empty((nq, k), "<f4")
        self._bind().spectral_predict(
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::spectral_predict_binding.
            [addr_ro(q, name="X"), train_addr,
             addr_ro(self._pd_eigenvalues, name="eigenvalues"),
             addr_ro(self._pd_eigenvectors, name="eigenvectors"),
             addr_ro(self._pd_diag, name="degree scaling"),
             addr_ro(self._pd_centroids, name="centroids"),
             addr(labels, name="labels"), addr(emb, name="embedding")],
            # n_train, n_queries, n_features, n_components, n_clusters,
            # n_neighbors, affinity
            [n_train, nq, n_features, k, int(self._n_clusters), int(self._n_neighbors), affinity],
        )
        return labels, emb

    # Saved layout: eigenvalues <f4 (k), eigenvectors <f4 (n_train, k), diag <f4
    # (n_train), centroids <f4 (n_clusters, k), labels <i4, x <f4 (training rows,
    # 'nearest_neighbors' only), affinity, numeric_mode, eigen_tol <f8, meta <i8
    # [n_clusters, n_components_, n_neighbors, n_init, seed,
    # random_state_is_none, n_train, n_features_in_ (0 for precomputed)]. On a
    # CPU-only install predict runs through _mojolearn_metrics_host.spectral_predict.
    def save(self, path):
        """Save a model fitted with ``prediction_data=True`` to ``path`` (npz).

        A model restored with ``SpectralClustering.load`` can ``predict``
        new rows, including on a CPU-only install, but cannot refit.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If it was fitted without ``prediction_data=True``.
        """
        if not hasattr(self, "labels_"):
            raise RuntimeError("this estimator is not fitted yet")
        if getattr(self, "_pd_eigenvalues", None) is None:
            raise ValueError(
                "mojolearn SpectralClustering.save: prediction data was not stored; "
                "fit with SpectralClustering(prediction_data=True). A model without "
                "it can label no new row, so there is nothing a saved file could serve"
            )
        nn = self.affinity == "nearest_neighbors"
        n_train = int(self._pd_diag.shape[0])
        arrays = {
            "format": _SPECTRAL_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "affinity": str(self.affinity),
            "eigenvalues": self._pd_eigenvalues,
            "eigenvectors": self._pd_eigenvectors,
            "diag": self._pd_diag,
            "centroids": self._pd_centroids,
            "labels": self.labels_,
            "eigen_tol": Array.from_list([float(self._eigen_tol)], "<f8"),
            "meta": Array.from_list(
                [int(self._n_clusters), int(self.n_components_), int(self._n_neighbors),
                 int(self._n_init), int(self._seed), 1 if self.random_state is None else 0,
                 n_train, int(self.n_features_in_) if nn else 0],
                "<i8",
            ),
        }
        if nn:
            arrays["x"] = self._fit_X
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model written by ``save``.

        The loaded model can ``predict``; it does not refit. Arrays are read
        at their saved dtype and never cast.

        Parameters
        ----------
        path : str or path-like
            File written by ``SpectralClustering.save``.

        Returns
        -------
        model : SpectralClustering
            The restored model, with ``labels_`` and ``n_components_`` set.

        Raises
        ------
        ValueError
            If the file is not a SpectralClustering model or its arrays have
            inconsistent shapes.
        """
        arrays = _serialize.read_npz(path, _SPECTRAL_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 8:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 8 are needed")
        n_clusters, k, n_neighbors, n_init, seed, seed_none, n_train, nf = (int(v) for v in meta.tolist())  # glue: unpacks the fixed meta vector
        tol = _serialize.exact(arrays, "eigen_tol", "<f8")
        if tol.size != 1:
            raise ValueError(f"mojolearn: {path!r} eigen_tol must hold one value")
        affinity = _serialize.scalar_str(arrays, "affinity")
        obj = cls(
            n_clusters=n_clusters, n_components=k, n_neighbors=n_neighbors, n_init=n_init,
            random_state=None if seed_none else seed, eigen_tol=float(tol[0]),
            affinity=affinity, prediction_data=True,
        )
        obj._validate_params()
        _restore_mode(obj, arrays)
        ev = _serialize.exact(arrays, "eigenvalues", "<f4")
        evec = _serialize.exact(arrays, "eigenvectors", "<f4")
        dg = _serialize.exact(arrays, "diag", "<f4")
        cent = _serialize.exact(arrays, "centroids", "<f4")
        labels = _serialize.exact(arrays, "labels", "<i4")
        if (ev.size != k or evec.ndim != 2 or tuple(evec.shape) != (n_train, k) or dg.size != n_train
                or cent.ndim != 2 or tuple(cent.shape) != (n_clusters, k) or labels.size != n_train):
            raise ValueError(
                f"mojolearn: {path!r} prediction data shapes do not match n_train={n_train}, "
                f"n_components={k}, n_clusters={n_clusters}"
            )
        obj._fit_X = None
        if affinity == "nearest_neighbors":
            x = _serialize.exact(arrays, "x", "<f4")
            if x.ndim != 2 or tuple(x.shape) != (n_train, nf):
                raise ValueError(f"mojolearn: {path!r} x shape {tuple(x.shape)} is not ({n_train}, {nf})")
            obj._fit_X = x
            obj.n_features_in_ = nf
        obj._pd_eigenvalues = ev
        obj._pd_eigenvectors = evec
        obj._pd_diag = dg
        obj._pd_centroids = cent
        obj.labels_ = labels
        obj.n_components_ = k
        return obj


class _DenseCOO:
    """A dense float32 matrix (`_expansion_decomp._M`) as the duck-typed COO
    `_coo_triples` reads: row-major order, exact zeros dropped (what
    `sp.coo_matrix(dense)` keeps), -0.0 a zero and NaN kept. The scan is the
    base binding's `nonzero_f32_count` / `nonzero_f32_fill`
    (core/dense_coo.mojo, lane/py-dn-kern 2026-09-28; its CPU route is the
    core host binding), which replaced a Python n^2 loop making the same
    `v != 0.0` test on the same float32 values (DEVIATION 2489's float32
    twin). Pure data movement."""

    def __init__(self, m):
        n_r, n_c = m.r, m.c
        src = m.addr
        nnz = int(_native("nonzero_f32_count")(src, n_r * n_c)) if n_r * n_c else 0  # cpu-route: CPU-only host binding COO route (dense device entry absent)
        self.shape = (n_r, n_c)
        self.row = empty((nnz,), "<i4")
        self.col = empty((nnz,), "<i4")
        self.data = empty((nnz,), "<f4")
        if nnz:
            wrote = int(_native("nonzero_f32_fill")(  # cpu-route: CPU-only host binding COO route (dense device entry absent)
                src, n_r, n_c,
                [addr(self.row, name="rows"), addr(self.col, name="cols"), addr(self.data, name="vals")],
                nnz))
            if wrote != nnz:
                raise RuntimeError(f"mojolearn: nonzero_f32_fill wrote {wrote} of {nnz} entries")

    def tocoo(self):
        return self


# Maintainer notes (moved from the docstring). Reference: cuML's
# manifold.SpectralEmbedding; the same stages SpectralClustering runs before
# its k-means. The dense RBF affinity is the decomp cells (lane/algos-decomp,
# 2026-09-27), as are 'precomputed_nearest_neighbors' and eigen_tol.
# random_state=None means seed 0 (DEVIATION 891). eigen_tol <= 0 disables the
# Lanczos convergence test (DEVIATION 890). eigen_solver names other than None
# are refused: 'arpack', 'lobpcg' and 'amg' are other algorithms, and running
# the Lanczos under their names would be a silent substitution
# (spectral/NOT_IMPLEMENTED.tsv).
class SpectralEmbedding(ParamsMixin):
    """Spectral embedding (Laplacian eigenmaps) for nonlinear dimensionality reduction.

    Builds an affinity graph, forms the normalized graph Laplacian and returns
    its smallest nontrivial eigenvectors, computed with a thick-restart
    Lanczos solver. The first (trivial) eigenvector is dropped. Follows cuML's
    ``manifold.SpectralEmbedding``. There is no ``transform``: the embedding
    covers the fitted rows only, as in scikit-learn.

    Parameters
    ----------
    n_components : int, default 2
        Dimension of the embedding; at least 1, and the eigenpairs needed
        must number fewer than ``n_samples``.
    affinity : str, default "nearest_neighbors"
        ``"nearest_neighbors"`` (symmetrized k-NN connectivity of ``X``),
        ``"precomputed"`` (``X`` is a square affinity matrix, dense or
        sparse), ``"rbf"`` (``exp(-gamma * ||x_i - x_j||**2)``) or
        ``"precomputed_nearest_neighbors"`` (``X`` is a distance matrix or
        sparse distance graph; the ``n_neighbors`` nearest per row,
        symmetrized as ``0.5 * (C + C.T)``). A callable or any other name
        raises ``ValueError``.
    random_state : int or None, default None
        Seed in ``[0, 2**32)``. None means seed 0, so results are
        deterministic (unlike scikit-learn, where None draws a fresh seed).
    n_neighbors : int or None, default None
        Neighbors per row; None uses ``max(n_samples // 10, 1)``.
    gamma : float or None, default None
        Positive RBF coefficient, used only with ``affinity="rbf"``; None
        means ``1 / n_features`` (scikit-learn's rule). Setting it with
        another affinity raises ``NotImplementedError``.
    eigen_solver : None
        Only None is accepted; any other value raises ``NotImplementedError``.
    eigen_tol : float, "auto" or None, default None
        Eigensolver tolerance. ``"auto"`` and None use 1e-5; otherwise a
        positive finite float.
    n_jobs : int or None, default None
        Accepted for compatibility; it does not change the result.
    verbose : bool, default False
        Accepted for compatibility; nothing is printed.

    Attributes
    ----------
    embedding_ : Array of shape (n_samples, n_components), float32
        The embedding of the training rows.
    affinity_matrix_ : array
        The dense affinity matrix built for ``"rbf"`` and
        ``"precomputed_nearest_neighbors"``.
    n_neighbors_ : int
        Number of neighbors used, for the nearest-neighbor affinities.
    gamma_ : float
        The RBF coefficient used, for ``affinity="rbf"``.
    n_features_in_ : int
        Number of features of ``X``, for ``"nearest_neighbors"`` and ``"rbf"``.

    Notes
    -----
    A precomputed affinity must be nonnegative and finite and must not repeat
    a ``(row, col)`` entry; its diagonal is ignored.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> Y = mojolearn.SpectralEmbedding(n_components=2).fit_transform(X)
    """

    _drop_first = True
    _norm_laplacian = True

    def __init__(
        self,
        n_components=2,
        *,
        affinity="nearest_neighbors",
        random_state=None,
        n_neighbors=None,
        gamma=None,
        eigen_solver=None,
        eigen_tol=None,
        n_jobs=None,
        verbose=False,
    ):
        # Stored verbatim, as scikit-learn requires; `_validate_params`
        # checks them when `fit` runs.
        self.n_components = n_components
        self.affinity = affinity
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.eigen_solver, self.eigen_tol, self.n_jobs, self.verbose = eigen_solver, eigen_tol, n_jobs, verbose

    def _validate_params(self):
        """The constructor checks, run first in `fit`; the converted values
        go to private attributes, which the fit reads."""
        n_components, affinity, random_state = self.n_components, self.affinity, self.random_state
        n_neighbors, gamma, eigen_solver, eigen_tol = self.n_neighbors, self.gamma, self.eigen_solver, self.eigen_tol
        if affinity not in _EMBED_AFFINITIES:
            raise ValueError(
                f"mojolearn SpectralEmbedding: affinity={affinity!r} is "
                f"refused; it must be one of {list(_EMBED_AFFINITIES)}. "
                "A callable affinity is Python code that no identity column "
                "can run."
            )
        if gamma is not None and affinity != "rbf":
            raise NotImplementedError(
                "mojolearn SpectralEmbedding: gamma is refused; it "
                "parameterizes an RBF affinity, and affinity is not 'rbf'"
            )
        if gamma is not None and not float(gamma) > 0:
            raise ValueError("mojolearn SpectralEmbedding: gamma must be positive")
        if eigen_solver is not None:
            raise NotImplementedError(
                f"mojolearn SpectralEmbedding: eigen_solver={eigen_solver!r} "
                "is refused; there is one solver here, RAFT's thick-restart "
                "Lanczos. scikit-learn's 'arpack', 'lobpcg' and 'amg' are "
                "other algorithms, and running the Lanczos under their names "
                "would be a silent substitution (spectral/NOT_IMPLEMENTED.tsv)"
            )
        if isinstance(eigen_tol, str):
            if eigen_tol != "auto":
                raise ValueError(
                    f"mojolearn SpectralEmbedding: eigen_tol={eigen_tol!r} "
                    "must be 'auto' or a positive float"
                )
        elif eigen_tol is not None:
            tol = float(eigen_tol)
            if not tol > 0.0 or tol != tol or tol == float("inf"):
                raise ValueError(
                    "mojolearn SpectralEmbedding: eigen_tol must be a positive "
                    f"finite float or 'auto', got {eigen_tol!r} (0 disables the "
                    "Lanczos convergence test, DEVIATION 890)"
                )
        # n_jobs sizes scikit-learn's neighbor search; the kNN graph here is
        # exact, so the value changes no result. verbose prints nothing.
        if int(n_components) < 1:
            raise ValueError(
                "mojolearn SpectralEmbedding: n_components must be at least 1"
            )
        if n_neighbors is not None and int(n_neighbors) < 1:
            raise ValueError(
                "mojolearn SpectralEmbedding: n_neighbors must be at least 1, "
                "or None for max(n_samples // 10, 1)"
            )
        if random_state is None:
            seed = DEFAULT_SEED
        else:
            seed = int(random_state)
            if seed < 0 or seed >= 2**32:
                raise ValueError(
                    "mojolearn SpectralEmbedding: random_state must satisfy "
                    f"0 <= random_state < 2**32, got {random_state!r}"
                )
        self._n_components = int(n_components)
        self._n_neighbors = None if n_neighbors is None else int(n_neighbors)
        self._seed = seed
        self._gamma = None if gamma is None else float(gamma)

    def _resolved_neighbors(self, n):
        return self._n_neighbors if self._n_neighbors is not None else max(n // 10, 1)

    def fit(self, X, y=None):
        """Compute the embedding of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Training data, or a square affinity (``"precomputed"``) or
            distance (``"precomputed_nearest_neighbors"``) matrix, dense or
            sparse. Values must be finite.
        y : ignored

        Returns
        -------
        self : SpectralEmbedding
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``X`` is not finite, a precomputed matrix is not square or has
            a negative or repeated entry, or ``n_components`` needs as many
            eigenpairs as there are samples.
        """
        self._validate_params()
        k = self._n_components
        drop_first = bool(self._drop_first)
        norm_laplacian = bool(self._norm_laplacian)
        # The one piece of arithmetic the reference keeps in Python
        # (spectral_embedding.pyx:294): the transform receives this number.
        n_lanczos = k + 1 if drop_first else k
        dense_fn = None
        if self.affinity in ("precomputed", "rbf", "precomputed_nearest_neighbors"):
            dense_fn = _dense_entry("spectral_embedding_dense")
        if self.affinity == "precomputed_nearest_neighbors":
            X = self._precomputed_knn_affinity(X)
            if dense_fn is None:
                X = _DenseCOO(X)
        if self.affinity == "precomputed" and dense_fn is not None and not callable(getattr(X, "tocoo", None)):
            X = _dense_square(X, "SpectralEmbedding")
        if self.affinity == "rbf":
            # scikit-learn's `affinity='rbf'` (manifold/_spectral_embedding.py
            # `_get_affinity_matrix`: rbf_kernel(X, gamma), gamma defaulting
            # to 1/n_features), which cuML does not carry (lane/algos-decomp,
            # 2026-09-27). The dense affinity exp(-gamma ||x_i - x_j||^2) is
            # the decomp lane's identical cells (x_decomp/cells.mojo
            # `sqdist_cell`, features ascending with one fused multiply-add
            # per term, then `exp_c`, the portable expf), and then takes the
            # precomputed route below unchanged: the dense matrix through COO,
            # exact zeros dropped, the diagonal dropped.
            xm = _M.from_input(X)
            gamma = self._gamma if self._gamma is not None else 1.0 / xm.c
            self.gamma_ = gamma
            mode = _backend.default_mode()
            kit = _Kit(mode, _backend.binding("_mojolearn_x_decomp", mode))
            aff = kit.ew("exp", kit.ew("scale", kit.sqdist(xm, xm), s=-gamma))
            self.affinity_matrix_ = aff.out()
            self.n_features_in_ = xm.c
            X = aff if dense_fn is not None else _DenseCOO(aff)
        if dense_fn is not None and (isinstance(X, _M) or isinstance(X, Array)):
            # the dense matrix straight to the device entry (see
            # `_dense_entry`); the value refusals come from the same scan
            if isinstance(X, _M):
                n, dense_addr = X.r, X.addr
            else:
                n, dense_addr = int(X.shape[0]), addr_ro(X, name="X")
            self._check_shape(n, n_lanczos)
            self.n_neighbors_ = self._resolved_neighbors(n)
            embedding = empty((n, k), "<f4")
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::
            # spectral_embedding_dense_binding.
            params = [n, n_lanczos, k, int(norm_laplacian), int(drop_first),
                      self._seed] + self._tol_param()
            n_out = int(dense_fn(dense_addr, addr(embedding, name="embedding"), params))
            if n_out == _DENSE_EMPTY:
                raise ValueError(
                    "mojolearn SpectralEmbedding: the precomputed affinity "
                    "matrix has no nonzero entries"
                )
            if n_out == _DENSE_NONFINITE:
                raise ValueError(
                    "mojolearn SpectralEmbedding: the precomputed affinity "
                    "matrix has a non-finite entry"
                )
            if n_out == _DENSE_NEGATIVE:
                raise ValueError(
                    "mojolearn SpectralEmbedding: the precomputed affinity "
                    "matrix has a negative entry (refused by name: sqrt of a "
                    "negative degree is a NaN)"
                )
        elif self.affinity in ("precomputed", "rbf", "precomputed_nearest_neighbors"):
            # the COO route: a sparse precomputed input, or a binding without
            # the dense entry (the CPU-only host binding)
            rows, cols, vals, n = _coo_triples(X, "SpectralEmbedding")
            if vals.size == 0:
                raise ValueError(
                    "mojolearn SpectralEmbedding: the precomputed affinity "
                    "matrix has no nonzero entries"
                )
            # the rbf affinity is exp of a finite value, finite and >= 0 by
            # construction (`exp_c` clamps), so its two scans are skipped
            if self.affinity in ("rbf", "precomputed_nearest_neighbors"):
                pass
            elif not all_finite(vals):
                raise ValueError(
                    "mojolearn SpectralEmbedding: the precomputed affinity "
                    "matrix has a non-finite entry"
                )
            if self.affinity == "precomputed" and vals.min() < 0:
                raise ValueError(
                    "mojolearn SpectralEmbedding: the precomputed affinity "
                    "matrix has a negative entry (refused by name: sqrt of a "
                    "negative degree is a NaN)"
                )
            self._check_shape(n, n_lanczos)
            self.n_neighbors_ = self._resolved_neighbors(n)
            embedding = empty((n, k), "<f4")
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::
            # spectral_embedding_graph_binding.
            params = [n, int(vals.shape[0]), n_lanczos, k,
                      int(norm_laplacian), int(drop_first), self._seed] + self._tol_param()
            n_out = int(_get_binding().spectral_embedding_graph(
                addr_ro(rows, name="rows"), addr_ro(cols, name="cols"),
                addr_ro(vals, name="vals"), addr(embedding, name="embedding"),
                params,
            ))
        else:
            x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
            if not all_finite(x):
                raise ValueError(
                    "mojolearn SpectralEmbedding: X contains NaN or infinity"
                )
            n = int(x.shape[0])
            self._check_shape(n, n_lanczos)
            n_neighbors = self._resolved_neighbors(n)
            if n_neighbors > n:
                raise ValueError(
                    f"mojolearn SpectralEmbedding: n_neighbors={n_neighbors} "
                    f"exceeds n_samples={n}"
                )
            self.n_neighbors_ = n_neighbors
            embedding = empty((n, k), "<f4")
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::
            # spectral_embedding_dataset_binding.
            params = [n, int(x.shape[1]), n_lanczos, k, n_neighbors,
                      int(norm_laplacian), int(drop_first), self._seed] + self._tol_param()
            n_out = int(_get_binding().spectral_embedding_dataset(
                addr_ro(x, name="x"), addr(embedding, name="embedding"), params,
            ))
            self.n_features_in_ = int(x.shape[1])
        if n_out != k:
            raise RuntimeError(
                f"mojolearn SpectralEmbedding: the kernel returned {n_out} "
                f"embedding columns where {k} were expected"
            )
        self.embedding_ = embedding
        return self

    def _tol_param(self):
        """The optional trailing `eigen_tol` of the two embedding bindings:
        absent for None / 'auto' (the legacy params list, cuVS's 1e-5), else
        the float, which the binding rounds to float32 once."""
        if self.eigen_tol is None or isinstance(self.eigen_tol, str):
            return []
        return [float(self.eigen_tol)]

    def _precomputed_knn_affinity(self, X):
        """scikit-learn's affinity='precomputed_nearest_neighbors':
        NearestNeighbors(metric='precomputed').kneighbors_graph(X,
        mode='connectivity'), symmetrized as 0.5 * (C + C^T). X is a distance
        matrix: sparse (only the STORED entries are candidates) or dense
        (every entry is). Each row keeps its n_neighbors smallest distances,
        ties to the lower column (scikit-learn's stable sort of a sorted
        CSR row); the query IS the fitted data, so a stored diagonal counts,
        as in scikit-learn. Comparisons and the exact values 0, 0.5 and 1
        only: no arithmetic."""
        tocoo = getattr(X, "tocoo", None)
        if callable(tocoo):
            coo = tocoo()
            n = int(coo.shape[0])
            if tuple(coo.shape) != (n, n):
                raise ValueError("mojolearn SpectralEmbedding: the precomputed distance graph must be square")
            rows, _ = as_i32_c(coo.row, ndim=1, name="rows")
            cols, _ = as_i32_c(coo.col, ndim=1, name="cols")
            vals, _ = as_f32_c(coo.data, ndim=1, name="data")
            if not (rows.shape[0] == cols.shape[0] == vals.shape[0]):
                raise ValueError("mojolearn SpectralEmbedding: the precomputed graph's row, col and data differ in length")
            nnz = int(vals.shape[0])
            dense = None
        else:
            dense, _ = as_f32_c(X, ndim=2, name="X")
            n = int(dense.shape[0])
            if tuple(dense.shape) != (n, n):
                raise ValueError("mojolearn SpectralEmbedding: the precomputed distance matrix must be square")
            nnz = 0
        k = self._resolved_neighbors(n)
        # core/dense_coo.mojo `knn_affinity_f32` (lane/py-dn-kern, 2026-09-28):
        # the per-row (value, column) selection, C and 0.5 (C + C^T) in one
        # host entry; it was a Python sort per row and an n^2 loop here.
        A = _M.zeros(n, n)
        if n == 0:
            self.affinity_matrix_ = A.out()
            return A
        status = empty((3,), "<i4")
        sp = dense is None
        _native("knn_affinity_f32")(
            [0 if sp else addr_ro(dense, name="X"),
             addr_ro(rows, name="rows") if sp and nnz else 0,
             addr_ro(cols, name="cols") if sp and nnz else 0,
             addr_ro(vals, name="data") if sp and nnz else 0,
             A.addr, addr(status, name="status")],
            [n, k, nnz, 1 if sp else 0])
        code, row, count = (int(v) for v in status.tolist())  # glue: unpacks the three-field status
        if code == 1:
            raise ValueError("mojolearn SpectralEmbedding: a precomputed distance is negative or NaN")
        if code == 2:
            # `count`: the failing row's candidates, from the binding's status
            raise ValueError(f"mojolearn SpectralEmbedding: row {row} of the precomputed graph has "
                             f"{count} stored distances, fewer than n_neighbors={k}")
        self.affinity_matrix_ = A.out()
        return A

    def _check_shape(self, n, n_lanczos):
        if n_lanczos >= n:
            raise ValueError(
                f"mojolearn SpectralEmbedding: n_components={self._n_components} "
                f"needs {n_lanczos} eigenpairs, which must be below "
                f"n_samples={n}"
            )

    def fit_transform(self, X, y=None):
        """Fit to ``X`` and return the embedding.

        Parameters
        ----------
        X : array-like
            As in ``fit``.
        y : ignored

        Returns
        -------
        embedding : Array of shape (n_samples, n_components), float32
            Same as ``embedding_``.
        """
        return self.fit(X, y).embedding_


def spectral_embedding(
    A,
    *,
    n_components=8,
    affinity="nearest_neighbors",
    random_state=None,
    n_neighbors=None,
    norm_laplacian=True,
    drop_first=True,
):
    """Spectral embedding of a dataset or affinity matrix, as a function.

    Equivalent to ``SpectralEmbedding(...).fit_transform(A)``, plus the two
    switches the class fixes. Follows cuML's ``manifold.spectral_embedding``;
    note that ``n_components`` defaults to 8 here, as in cuML.

    Parameters
    ----------
    A : array-like
        Data or matrix, interpreted according to ``affinity`` (see
        :class:`SpectralEmbedding`).
    n_components : int, default 8
        Dimension of the embedding.
    affinity : str, default "nearest_neighbors"
        Affinity, as in :class:`SpectralEmbedding`.
    random_state : int or None, default None
        Seed; None means seed 0.
    n_neighbors : int or None, default None
        Neighbors per row; None uses ``max(n_samples // 10, 1)``.
    norm_laplacian : bool, default True
        Use the symmetric normalized graph Laplacian.
    drop_first : bool, default True
        Drop the first (trivial) eigenvector.

    Returns
    -------
    embedding : Array of shape (n_samples, n_components), float32
        The embedding.

    Raises
    ------
    TypeError
        If ``norm_laplacian`` or ``drop_first`` is not a bool.
    """
    for name, value in (("norm_laplacian", norm_laplacian), ("drop_first", drop_first)):  # glue: checks two boolean flags
        if not isinstance(value, bool):
            raise TypeError(
                f"mojolearn spectral_embedding: {name} must be a bool, got "
                f"{type(value).__name__}"
            )
    model = SpectralEmbedding(
        n_components=n_components, affinity=affinity,
        random_state=random_state, n_neighbors=n_neighbors,
    )
    model._drop_first = drop_first
    model._norm_laplacian = norm_laplacian
    return model.fit_transform(A)
