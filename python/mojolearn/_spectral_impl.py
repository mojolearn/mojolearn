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

from . import _native_archive_bridge, _native_family_bridge
from ._mode import ParamsMixin, available_if
from ._buffer import _native, addr, empty
from . import _native_spectral_umap_bridge

__all__ = ["SpectralClustering", "SpectralEmbedding", "spectral_embedding"]

#: `SpectralClustering.save`'s format tag (lane/spectral-predict,
#: 2026-09-15): the prediction data of a `prediction_data=True` fit.
_SPECTRAL_FORMAT = "mojolearn-spectral-1"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered below the class; host_archive_native.mojo
# builds the loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()

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

# The compiled family owner (bindings/cfb_spectral.mojo): every public method
# is one native call; its owner selects among these import-time connections.
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()
_native_family_bridge.tables(family_cfb_sc_affinities=_CLUSTER_AFFINITIES,
                             family_cfb_se_affinities=_EMBED_AFFINITIES,
                             family_cfb_predict_affinities=_AFFINITIES,
                             family_cfb_assign_labels=_ASSIGN_LABELS,
                             family_cfb_assign_codes=dict(_ASSIGN_CODES),
                             family_cfb_default_eigen_tol=DEFAULT_EIGEN_TOL,
                             family_spectral_umap_bindings=_native_spectral_umap_bridge.connect())


def _coo_triples(A, who="SpectralClustering"):
    """A precomputed affinity matrix as `(rows, cols, vals, n)` int32/float32.

    Same behavior as cuML's `spectral_clustering.pyx:306-312`, which accepts
    scipy or cupy sparse in COO/CSR/CSC and calls `sp.coo_matrix(X)` on a
    dense input; converting a dense matrix through COO drops exact zeros. A
    sparse input is recognized by duck type (a `tocoo()` whose result
    carries `shape`, `row`, `col` and `data`); nothing is imported to ask.
    The scan is native (bindings/cfb_spectral.mojo, the base binding's
    `nonzero_f64_count` / `nonzero_f64_fill`).
    """
    return _NATIVE(None, "family_cfb_coo_triples", (A, who),
                   _NATIVE_CONTEXT)


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
    _mojolearn_native_public = True

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

    #: The binding `predict` asks; a host subclass answers the metrics host
    #: binding, which ships in the inference wheel.
    _BINDING = "_mojolearn_metrics"

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
        return _NATIVE(self, "family_cfb_sc_fit", (X, y), _NATIVE_CONTEXT)

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
        return _NATIVE(self, "family_cfb_sc_fit_predict", (X, y), _NATIVE_CONTEXT)

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
    # Import-only availability identity; the native 'spectral_predict' schema decides it.
    @available_if(lambda self: bool(self.prediction_data) or getattr(self, "_fit_X", None) is not None)
    def predict(self, X):
        r"""Assign new rows to the fitted clusters.

        Neither scikit-learn nor cuML provides ``predict`` for spectral
        clustering. This one uses the Nystrom out-of-sample extension (Bengio
        et al., NIPS 2003): each new row is projected onto the fitted
        eigenvectors through its affinity to the training rows, then assigned
        to the nearest fitted k-means centroid (ties go to the lowest
        centroid index). A row's label does not depend on the other rows in
        the call.

        The method exists only with ``prediction_data=True`` (or on a model
        loaded with `load`); otherwise, as in scikit-learn, there is no
        ``predict``.

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
        return _NATIVE(self, "family_cfb_sc_predict", (X,), _NATIVE_CONTEXT)

    def _predict_embedding(self, X):
        """`predict`'s labels and the extended embedding (n_new x
        n_components float32)."""
        return _NATIVE(self, "family_cfb_sc_predict_embedding", (X,), _NATIVE_CONTEXT)

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
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

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
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


_native_archive_bridge.register(SpectralClustering, _SPECTRAL_FORMAT, (
    _native_archive_bridge.fitted("labels_"),
    ("need_set", "_pd_eigenvalues", "ValueError",
     "mojolearn SpectralClustering.save: prediction data was not stored; "
     "fit with SpectralClustering(prediction_data=True). A model without "
     "it can label no new row, so there is nothing a saved file could serve"),
    *_native_archive_bridge.head(_SPECTRAL_FORMAT),
    ("str", "affinity", "affinity"),
    ("array", "eigenvalues", "_pd_eigenvalues"),
    ("array", "eigenvectors", "_pd_eigenvectors"),
    ("array", "diag", "_pd_diag"),
    ("array", "centroids", "_pd_centroids"),
    ("array", "labels", "labels_"),
    ("f8", "eigen_tol", (("float", "_eigen_tol"),)),
    ("i8", "meta", (
        ("int", "_n_clusters"), ("int", "n_components_"), ("int", "_n_neighbors"),
        ("int", "_n_init"), ("int", "_seed"), ("noneflag", "random_state"),
        ("dim", "_pd_diag", 0),
        ("if_eq", "affinity", "nearest_neighbors", ("int", "n_features_in_"), ("const", 0)),
    )),
    ("when", "affinity", "nearest_neighbors", (("array", "x", "_fit_X"),)),
), host_archive_spectral_affinities=tuple(_CLUSTER_AFFINITIES))


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
        nnz = int(_native("nonzero_f32_count")(src, n_r * n_c)) if n_r * n_c else 0  # cpu-route: float32 host matrix read once by nonzero_f32_count on a CPU-only install
        self.shape = (n_r, n_c)
        self.row = empty((nnz,), "<i4")
        self.col = empty((nnz,), "<i4")
        self.data = empty((nnz,), "<f4")
        if nnz:
            wrote = int(_native("nonzero_f32_fill")(  # cpu-route: float32 host matrix read once by nonzero_f32_fill on a CPU-only install
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

    _mojolearn_native_public = True
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
        go to private attributes, which the fit reads
        (bindings/cfb_spectral.mojo)."""
        return _NATIVE(self, "family_cfb_se_validate", (), _NATIVE_CONTEXT)

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
        return _NATIVE(self, "family_cfb_se_fit", (X, y), _NATIVE_CONTEXT)

    def _precomputed_knn_affinity(self, X):
        """scikit-learn's affinity='precomputed_nearest_neighbors' of a
        precomputed distance matrix, symmetrized 0.5 * (C + C^T), as an
        n x n float32 Array (and `affinity_matrix_`): the base binding's
        `knn_affinity_f32` (bindings/cfb_spectral.mojo)."""
        return _NATIVE(self, "family_cfb_se_knn_affinity", (X,),
                       _NATIVE_CONTEXT)

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
        return _NATIVE(self, "family_cfb_se_fit_transform", (X, y), _NATIVE_CONTEXT)


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
    return _NATIVE(None, "family_cfb_spectral_embedding", (
        SpectralEmbedding, A, n_components, affinity, random_state, n_neighbors,
        norm_laplacian, drop_first), _NATIVE_CONTEXT)


# Import-only constructor schemas; native owners preserve the public signatures.
_native_family_bridge.tables(
    spectral_clustering_parameters=("n_clusters", "n_components", "random_state", "n_neighbors", "n_init",
        "eigen_tol", "affinity", "gamma", "assign_labels", "prediction_data", "eigen_solver",
        "degree", "coef0", "kernel_params", "n_jobs", "verbose"),
    spectral_embedding_parameters=("n_components", "affinity", "random_state", "n_neighbors", "gamma",
        "eigen_solver", "eigen_tol", "n_jobs", "verbose"))
