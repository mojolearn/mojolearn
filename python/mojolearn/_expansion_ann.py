# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ANN LANE'S PUBLIC DOOR.

Owned by the `ann` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`; a name
that is already public is refused at import. Put the classes here, or import
them here from the lane's own modules. Two optional hooks route a SAVED model
to the CPU (`mojolearn.host_model`); `_classical_host.py` merges them when it is
first imported, after the package, so both may rely on every module existing:

  CLASSICAL_HOST_BASENAMES     {"_mojolearn_x_ann": "_mojolearn_x_ann_host"}
  def classical_host_formats() -> {"mojolearn-<format>-1": {"Estimator": HostEstimator}}
                               import `_classical_host` INSIDE it and subclass
                               its `_HostBound`; never import it at module level
"""
from . import _native_archive_bridge, _native_family_bridge
from ._native_robust_bridge import native_method as _native_ann_method
from ._buffer import addr, addr_ro, as_f32_c, as_i32_c, empty
from ._mode import NumericModeMixin

# The compiled saved-model owner, connected once at import: `save` of
# CagraIndex, IVFSQIndex and IVFRaBitQIndex is one native call that writes
# the schema registered at the end of this module
# (bindings/archive_schema_native.mojo).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()
# The compiled family owner, connected once at import: TSNE's `fit` and
# `fit_transform` are one native call each (bindings/mani_tsne_native.mojo).
from . import _native_runtime_manifold_bridge
_NATIVE, _NATIVE_CONTEXT = _native_runtime_manifold_bridge.connect()
# The classical family owner, connected once at import: the indexes' `fit`
# and `refine` are one native call each (bindings/neighbors_ann_native.mojo).
_FAMILY, _FAMILY_CONTEXT = _native_family_bridge.connect()
from . import _backend  # noqa: E402

__all__ = ["IVFPQIndex", "TSNE", "CagraIndex", "IVFSQIndex", "IVFRaBitQIndex", "refine"]


_ONES = {}


def _ann_mask(owner, filter, n):
    """cuVS's sample filter as one int32 per row (1 keeps the row, 0 removes
    it); None keeps every row (one all-ones array per row count, made once
    and only read: the host bindings take it where a GPU handle keeps its
    own, lane/py-dn-ann)."""
    from ._optional_numpy import require_numpy
    np = require_numpy('_expansion_ann')
    if filter is None:
        ones = _ONES.get(n)
        if ones is None:
            _ONES.clear()
            ones = _ONES[n] = np.ones(n, dtype=np.int32)
        return ones
    f = np.asarray(filter)
    if f.shape != (n,) or f.dtype != np.bool_:
        raise ValueError(f"mojolearn {owner}: filter must be a boolean array of shape ({n},), one flag per indexed row")
    return np.ascontiguousarray(f.astype(np.int32))


def _ann_int(owner, name, v):
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"mojolearn {owner}: {name} must be an int, got {type(v).__name__}")
    return int(v)


class _AnnResident:
    """THE RESIDENT INDEX (lane/py-dn-ann, 2026-09-28; x_ann/resident.mojo,
    the x_ann half of DEVIATION 1804's closure). A GPU binding keeps the
    fitted index on the device under a handle made at the first search and
    reused while the index arrays are the same objects at the same addresses
    and shapes; every later search uploads only its queries (and a caller's
    filter). A binding with no `x_ann_index_prepare` (the CPU host binding,
    which already reads the caller's arrays in place, or an older build)
    takes the per-call door. `neighbors.py::_resident_index_handle`'s rule."""

    _KIND = None

    def _resident_arrays(self):
        """(addresses in the prepare order, params) of the fitted index."""
        raise NotImplementedError

    def _resident_handle(self, native):
        try:
            prepare = native.x_ann_index_prepare
        except (ImportError, AttributeError):
            return None
        key = self._resident_key(native)
        cached = self.__dict__.get("_resident")
        if cached is not None and cached[0] == key:
            return cached[1]
        # an array still held only on the device comes to numpy first, then
        # the handle is replaced by a prepared one (lane fg-ivf A7)
        self._materialize_device_arrays()
        self._release_resident()
        addrs, params = self._resident_arrays()
        handle = int(prepare(self._KIND, addrs, params))
        self._resident = (key, handle, native)
        return handle

    def _resident_key(self, native):
        """The arrays a handle was made from, by identity, address and shape;
        an array a resident fit left on the device (lane fg-ivf A7) is the
        marker ("device",). Reads the dict, so no export happens."""
        d = self.__dict__
        arrays = tuple(d.get(a) for a, _dtype in self._SAVE_ARRAYS)  # glue: saved array attributes by name
        return (tuple(("device",) if a is None else (id(a), addr_ro(a, name=name), tuple(a.shape))
                      for a, (name, _dtype) in zip(arrays, self._SAVE_ARRAYS)), id(native))  # glue: saved array attributes by name

    def _materialize_device_arrays(self):
        """Arrays held only on the device come to numpy (none by default)."""
        return None

    def _resident_search(self, native, owner, filter, q, dist, idx, cand, params):
        """The IVF kinds' search through the handle; False where there is
        none (the caller then takes the per-call door)."""
        handle = self._resident_handle(native)
        if handle is None:
            return False
        addrs = [addr_ro(q, name="queries"), addr(dist, name="distances"), addr(idx, name="indices"),
                 addr(cand, name="n_candidates_")]
        mask = None
        if filter is not None:
            mask = _ann_mask(owner, filter, self.n_rows_)   # held: the binding reads it below
            addrs.append(addr_ro(mask, name="filter"))
        # queries, out_d, out_i, out_n[, filter]; m, k, n_probes (x_ann/resident.mojo)
        native.x_ann_index_search(handle, addrs, params)
        self.n_candidates_ = cand
        return True

    def _release_resident(self):
        cached = self.__dict__.pop("_resident", None)
        if cached is None:
            return
        try:
            cached[2].x_ann_index_release(cached[1])
        except Exception:  # noqa: BLE001
            pass

    # No Python `__del__`: dropping an instance releases its device index
    # through the native finalizer connected at the end of this module
    # (`_native_archive_bridge.lifetime`, bindings/host_archive_lifetime.mojo).

    def __getstate__(self):
        """No handle crosses a pickle or a deepcopy: it means something only
        in the process and registry that minted it."""
        return _FAMILY(self, "family_xn_ann_getstate", (), _FAMILY_CONTEXT)


class _AnnSaved:
    """save / load for the ann indexes: an npz holding the constructor's int
    parameters, the fitted ints and the index arrays with their dtype and
    shape, and the tier. A loaded index searches; it does not rebuild.
    No arithmetic: the arrays are written and read back byte for byte, and
    the native reader refuses any dtype cast."""

    _SAVE_FORMAT = None
    _SAVE_PARAMS = ()      # constructor parameters, all int
    _SAVE_FITTED = ()      # fitted int attributes
    _SAVE_ARRAYS = ()      # (attribute, typestr)

    def save(self, path):
        """Write the fitted index to `path` as an ``.npz`` file.

        The file holds the constructor parameters, the fitted sizes, the index
        arrays exactly as `fit` left them, and the numeric mode. Reload it with
        `load`; a loaded index searches without rebuilding.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Returns
        -------
        path : str or path-like
            The path written.

        Raises
        ------
        RuntimeError
            If the index has not been fitted.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    # One native call (bindings/host_archive_native.mojo `host_archive_ann`,
    # from the layout connected at the end of this module).
    @classmethod
    def load(cls, path):
        """Load an index written by `save`.

        The arrays are read back byte for byte; a file whose arrays have a
        different dtype, or that was saved by another class, is refused.

        Parameters
        ----------
        path : str or path-like
            File written by `save`.

        Returns
        -------
        index
            A fitted index of this class, ready to `search`.

        Raises
        ------
        ValueError
            If the file was saved by another class or does not hold this
            class's parameters.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


# IDENTICAL by construction: fixed-order codebook training and code sums,
# ties broken by index (x_ann/ivf_pq_core.mojo).
class IVFPQIndex(_AnnResident, _AnnSaved, NumericModeMixin):
    """Approximate nearest-neighbor index with inverted lists and product quantization (IVF-PQ).

    Follows cuVS's ``ivf_pq``. `fit` trains a k-means coarse quantizer that
    splits the rows into `n_lists` lists, then encodes each row with product
    quantization: the features are split into `pq_dim` subspaces and each
    subspace is stored as one of ``2**pq_bits`` codebook entries. `search`
    scans the `n_probes` lists nearest each query and returns approximate
    squared Euclidean distances computed from the codes. Codebook training
    and code sums run in a fixed order and ties are broken by row id, so the
    results are reproducible. On a GPU the fitted index stays on the device
    between searches; each search uploads only its queries.

    Parameters
    ----------
    n_lists : int
        Number of inverted lists (k-means clusters). Must be in
        ``[1, n_samples]``.
    n_probes : int
        Number of lists scanned per query. Required, with no default; a value
        greater than `n_lists` raises an error rather than being clamped.
    pq_dim : int, default 4
        Number of PQ subspaces. Must be in ``[1, n_features]``; the features
        are zero-padded up to a multiple of `pq_dim`.
    pq_bits : int, default 8
        Bits per code, so each subspace has ``2**pq_bits`` codebook entries.
        Must be in ``[1, 8]``.
    n_neighbors : int, default 8
        Number of neighbors `search` returns per query.
    kmeans_n_iters : int, default 20
        k-means iterations for the coarse quantizer.
    pq_kmeans_n_iters : int, default 20
        k-means iterations for each subspace codebook.
    random_state : int, default 0
        Seed for the k-means initializations.
    numeric_mode : str or None, default None
        Numeric mode for this estimator; None uses the package default set
        with `mojolearn.set_numeric_mode`.

    Attributes
    ----------
    centers_ : ndarray of shape (n_lists, n_features), float32
        Coarse quantizer centers.
    list_offsets_ : ndarray of shape (n_lists + 1,), int32
        Start of each list in `list_indices_`.
    list_indices_ : ndarray of shape (n_samples,), int32
        Original row ids, grouped by list.
    codebooks_ : ndarray of shape (pq_dim, 2**pq_bits, ``pq_len_``), float32
        One codebook per subspace.
    codes_ : ndarray of shape (n_samples, pq_dim), int32
        PQ code of every indexed row.
    n_features_in_ : int
        Number of features seen during `fit`.
    n_rows_ : int
        Number of indexed rows.
    n_lists_, pq_dim_, pq_bits_ : int
        The fitted values of `n_lists`, `pq_dim` and `pq_bits`.
    ``pq_len_`` : int
        Features per subspace, ``ceil(n_features / pq_dim)``.
    n_candidates_ : ndarray of shape (n_queries,), int32
        Set by `search`: how many indexed rows each query scanned.

    Raises
    ------
    TypeError
        If an integer parameter is not an int.
    ValueError
        If `n_lists`, `pq_dim` or `pq_bits` is out of range.

    See Also
    --------
    IVFIndex : IVF without compression (exact distances within probed lists).
    refine : Re-rank candidates with exact distances.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> index = mojolearn.IVFPQIndex(n_lists=16, n_probes=4, n_neighbors=5).fit(X)
    >>> distances, indices = index.search(X[:10])
    >>> indices.shape
    (10, 5)
    """

    _BINDING = "_mojolearn_x_ann"
    _NAME = "IVFPQIndex"
    _SAVE_FORMAT = "mojolearn-ivf-pq-1"
    _SAVE_PARAMS = ("n_lists", "n_probes", "pq_dim", "pq_bits", "n_neighbors", "kmeans_n_iters",
                    "pq_kmeans_n_iters", "random_state")
    _SAVE_FITTED = ("n_features_in_", "n_rows_", "n_lists_", "pq_dim_", "pq_bits_", "pq_len_")
    _SAVE_ARRAYS = (("centers_", "<f4"), ("list_offsets_", "<i4"), ("list_indices_", "<i4"),
                    ("codebooks_", "<f4"), ("codes_", "<i4"))

    def __init__(self, n_lists, n_probes, pq_dim=4, pq_bits=8, n_neighbors=8, kmeans_n_iters=20,
                 pq_kmeans_n_iters=20, random_state=0):
        self.n_lists = n_lists
        self.n_probes = n_probes
        self.pq_dim = pq_dim
        self.pq_bits = pq_bits
        self.n_neighbors = n_neighbors
        self.kmeans_n_iters = kmeans_n_iters
        self.pq_kmeans_n_iters = pq_kmeans_n_iters
        self.random_state = random_state

    _KIND = 0

    def _p(self, name):
        return _ann_int(self._NAME, name, getattr(self, name))

    def _resident_arrays(self):
        n, dim, pq_dim = self.n_rows_, self.n_features_in_, self.pq_dim_
        return ([addr_ro(self.centers_.reshape((self.n_lists_ * dim,)), name="centers_"),
                 addr_ro(self.list_offsets_, name="list_offsets_"), addr_ro(self.list_indices_, name="list_indices_"),
                 addr_ro(self.codebooks_.reshape((pq_dim * (1 << self.pq_bits_) * self.pq_len_,)), name="codebooks_"),
                 addr_ro(self.codes_.reshape((n * pq_dim,)), name="codes_")],
                [n, dim, self.n_lists_, pq_dim, self.pq_bits_])

    def __getattr__(self, name):
        # `codes_` of a RESIDENT fit (lane fg-ivf A7) is made on first read,
        # outside fit and search: `save`, a pickle and user code read it here
        if name == "codes_" and "_device_codes" in self.__dict__:
            self._materialize_device_arrays()
            return self.__dict__["codes_"]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _materialize_device_arrays(self):
        """Write the resident fit's codes into a new numpy array
        (`x_ann_index_export_codes`, one device copy in Mojo) and re-key the
        held handle to it: the handle keeps serving searches, since the array
        holds the words it holds."""
        d = self.__dict__
        if "_device_codes" not in d:
            return
        cached = d.get("_resident")
        if cached is None:
            raise RuntimeError("mojolearn IVFPQIndex: the resident index was released before its codes were read")
        n, pq_dim = self.n_rows_, self.pq_dim_
        codes = empty((n * pq_dim,), "<i4")
        cached[2].x_ann_index_export_codes(cached[1], addr(codes, name="codes_"), n * pq_dim)
        d["codes_"] = codes.reshape((n, pq_dim))
        del d["_device_codes"]
        d["_resident"] = (self._resident_key(cached[2]), cached[1], cached[2])

    def _is_built(self):
        d = self.__dict__
        return "codes_" in d or "_device_codes" in d

    def __getstate__(self):
        return _FAMILY(self, "family_xn_ann_getstate", (), _FAMILY_CONTEXT)

    # IVFPQIndex's save is the shared native one: a resident fit's `codes_`,
    # held only on the device, is exported into the file by the binding's
    # x_ann_index_export_codes, leaving the index unchanged.
    def save(self, path):
        """Write the fitted index to `path` as an ``.npz`` file.

        The file holds the constructor parameters, the fitted sizes, the index
        arrays exactly as `fit` left them, and the numeric mode. Reload it with
        `load`; a loaded index searches without rebuilding.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Returns
        -------
        path : str or path-like
            The path written.

        Raises
        ------
        RuntimeError
            If the index has not been fitted.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @_native_ann_method
    def fit(self, X, y=None):
        """Build the index from `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to index, converted to float32.
        y : None
            Ignored.

        Returns
        -------
        self : IVFPQIndex
            The fitted index.
        """
        return _FAMILY(self, "family_xn_ann_pq_fit", (None, X, y), _FAMILY_CONTEXT)

    @_native_ann_method
    def search(self, queries, filter=None):
        """Find the approximate nearest indexed rows of each query.

        Parameters
        ----------
        queries : array-like of shape (n_queries, n_features)
            Query rows, converted to float32.
        filter : array-like of bool, shape (n_rows_,), optional
            One flag per indexed row; a row flagged False is never returned
            (cuVS's sample filter, applied before scoring).

        Returns
        -------
        distances : ndarray of shape (n_queries, n_neighbors), float32
            Approximate squared Euclidean distances, ascending.
        indices : ndarray of shape (n_queries, n_neighbors), int32
            Original row ids of the neighbors. A query whose probed lists hold
            fewer than `n_neighbors` rows is padded with distance ``inf`` and
            id ``-1``.

        Raises
        ------
        ValueError
            If the index is not fitted, the queries have a different number of
            features, or `filter` has the wrong shape or dtype.
        """
        return _FAMILY(self, "family_xn_ann_pq_search", (None, queries, filter), _FAMILY_CONTEXT)


# IDENTICAL by construction (x_ann/tsne_core.mojo): exact k-NN affinities
# with the reference's perplexity bisection, attractive term over the sparse
# P, exact repulsion (no Barnes-Hut atomics), sklearn's gains/momentum
# optimizer. The 'pca' and 'random' starts are built in the binding
# (x_ann/tsne_init.mojo): 'pca' scales mojolearn's PCA embedding by the first
# column's standard deviation (one correctly rounded sum, so no platform's
# summation order enters) with one float32 divide and one float32 multiply per
# element; 'random' draws from the seeded Mojo stream (uniform_init_f32),
# whose integer-to-double draw is exact on every platform.
class TSNE(NumericModeMixin):
    """t-distributed Stochastic Neighbor Embedding (t-SNE) into two dimensions.

    Follows scikit-learn's ``TSNE`` (and cuML's): affinities from exact
    k-nearest neighbors with the perplexity search, then gradient descent with
    scikit-learn's gains and momentum, using early exaggeration for the first
    250 iterations. Unlike scikit-learn's default Barnes-Hut method, the
    repulsive forces are computed exactly, and the optimizer always runs
    exactly `max_iter` iterations (there is no early stop when progress
    stalls). The computation runs in a fixed order, so results are
    reproducible.

    Parameters
    ----------
    n_components : int, default 2
        Dimension of the embedding, 1 or 2.
    perplexity : float, default 30.0
        Effective number of neighbors. Must be in ``(0, n_samples)``.
    early_exaggeration : float, default 12.0
        Factor applied to the attractive forces during the early phase.
    learning_rate : float or 'auto', default 'auto'
        Optimizer step size. 'auto' is scikit-learn's
        ``max(n_samples / early_exaggeration / 4, 50)``.
    post_exaggeration_learning_rate_factor : float, default 1.0
        Multiplies the step size for the iterations after the early
        exaggeration phase, which then use
        ``learning_rate * post_exaggeration_learning_rate_factor``. The default
        1.0 keeps one step size throughout, as scikit-learn does; 2.0 doubles
        it after the early phase, as cuML's ``TSNE`` does. Must be positive.
    max_iter : int, default 1000
        Number of optimization iterations, all of which run.
    init : {'pca', 'random'} or array-like of shape (n_samples, n_components), default 'pca'
        Initial embedding. 'pca' is mojolearn's PCA embedding scaled as
        scikit-learn scales it: divided by the first column's standard
        deviation and multiplied by 1e-4. mojolearn uses an exact
        eigensolver where scikit-learn uses a randomized one. 'random' draws
        uniformly from ``(-5e-5, 5e-5)`` seeded by `random_state`. An array is
        used as given.
    random_state : int, default 0
        Seed for the 'random' initialization.
    numeric_mode : str or None, default None
        Numeric mode for this estimator; None uses the package default set
        with `mojolearn.set_numeric_mode`.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, 2), float32
        The embedding.
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.
    n_iter_ : int
        Number of iterations run (always `max_iter`).
    learning_rate_ : float
        The learning rate used, after resolving 'auto'.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> embedding = mojolearn.TSNE(perplexity=30.0).fit_transform(X)
    >>> embedding.shape
    (1000, 2)

    Doubling the step size after the early exaggeration phase:

    >>> tsne = mojolearn.TSNE(post_exaggeration_learning_rate_factor=2.0)
    >>> tsne.fit_transform(X).shape
    (1000, 2)
    """

    _BINDING = "_mojolearn_x_ann"
    _EXPLORATION_MAX_ITER = 250

    def __init__(self, n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate="auto",
                 max_iter=1000, init="pca", random_state=0, post_exaggeration_learning_rate_factor=1.0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        # release-0845: cuML's exact/barnes-hut schedule multiplies the rate by 2.0 after early
        # exaggeration (t_sne.pyx post_learning_rate); the kernels apply this factor there
        # (x_ann/tsne_device.mojo, x_ann/host/tsne_host.mojo tsne_fit_*). Last in the signature
        # so positional callers are unchanged.
        self.post_exaggeration_learning_rate_factor = post_exaggeration_learning_rate_factor
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state

    def fit(self, X, y=None):
        """Compute the embedding of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input rows, converted to float32.
        y : None
            Ignored.

        Returns
        -------
        self : TSNE
            The fitted estimator, with the result in `embedding_`.

        Raises
        ------
        ValueError
            If `n_components` is not 1 or 2, `perplexity` is out of range,
            `post_exaggeration_learning_rate_factor` is not positive, `init`
            is an unknown string, or an `init` array has the wrong shape.
        TypeError
            If `max_iter` or `random_state` is not an int.
        """
        return _NATIVE(self, "family_mani_tsne_fit", (None, None, X, y),
                       _NATIVE_CONTEXT)

    def fit_transform(self, X, y=None):
        """Compute the embedding of `X` and return it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input rows, converted to float32.
        y : None
            Ignored.

        Returns
        -------
        embedding : ndarray of shape (n_samples, n_components), float32
            The embedding, also stored in `embedding_`.
        """
        return _NATIVE(self, "family_mani_tsne_fit_transform",
                       (None, None, X, y), _NATIVE_CONTEXT)


# IDENTICAL by construction (x_ann/cagra_core.mojo). random_state r >= 0
# picks start node t as splitmix64 of (r + 1, t) mod n (cg_seed_node).
class CagraIndex(_AnnResident, _AnnSaved, NumericModeMixin):
    """Graph-based approximate nearest-neighbor index (CAGRA).

    Follows cuVS's ``cagra``. `fit` builds an exact k-nearest-neighbor graph
    of degree `intermediate_graph_degree`, prunes it to `graph_degree` with
    cuVS's rank-based detour pruning, and adds reverse edges. `search` walks
    the graph greedily from a set of start nodes, keeping the best
    `itopk_size` candidates, and returns squared Euclidean distances. Ties are
    broken by row id and the start nodes are the same for every query, so
    results are reproducible. The index keeps a copy of the dataset, as
    cuVS's does. On a GPU the fitted index stays on the device between
    searches; each search uploads only its queries.

    Parameters
    ----------
    graph_degree : int, default 32
        Out-degree of the final graph. Capped at `intermediate_graph_degree`.
    intermediate_graph_degree : int, default 64
        Degree of the k-NN graph built before pruning. Capped at
        ``n_samples - 1``.
    n_neighbors : int, default 8
        Number of neighbors `search` returns per query.
    itopk_size : int, default 64
        Number of candidates kept during the search. Larger values give
        better recall and slower searches.
    search_width : int, default 1
        Number of candidates whose neighbors are expanded per iteration.
    max_iterations : int, default 0
        Search iterations; 0 means `itopk_size`. The search runs at least
        ``2 * itopk_size + ceil(log(n_samples) / log(graph_degree / 2))``
        iterations, so the walk can converge.
    n_seeds : int, default 0
        Number of start nodes; 0 means cuVS's default,
        ``itopk_size + search_width * graph_degree``. The search uses at
        least ``262144 // n_features`` start nodes, so graphs that split into
        separate components are still reached; the count is capped at the
        number of rows.
    random_state : int or None, default None
        Seed for the start nodes, in the role of cuVS's ``rand_xor_mask``.
        None takes evenly spaced nodes; a non-negative int takes seeded
        pseudo-random nodes. Either way every query uses the same start nodes.
    numeric_mode : str or None, default None
        Numeric mode for this estimator; None uses the package default set
        with `mojolearn.set_numeric_mode`.

    Attributes
    ----------
    dataset_ : ndarray of shape (n_samples, n_features), float32
        The indexed rows.
    graph_ : ndarray of shape (n_samples, ``graph_degree_``), int32
        Neighbor ids of each row in the search graph.
    ``graph_degree_`` : int
        The graph degree used, after capping.
    n_features_in_ : int
        Number of features seen during `fit`.
    n_rows_ : int
        Number of indexed rows.

    Raises
    ------
    TypeError
        If an integer parameter is not an int.
    ValueError
        If `random_state` is negative.

    See Also
    --------
    IVFIndex, IVFPQIndex : Inverted-list indexes.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> index = mojolearn.CagraIndex(n_neighbors=5).fit(X)
    >>> distances, indices = index.search(X[:10])
    >>> indices.shape
    (10, 5)
    """

    _BINDING = "_mojolearn_x_ann"
    _SAVE_FORMAT = "mojolearn-cagra-2"
    _SAVE_PARAMS = ("graph_degree", "intermediate_graph_degree", "n_neighbors", "itopk_size", "search_width",
                    "max_iterations", "n_seeds", "random_state")
    _SAVE_FITTED = ("n_features_in_", "n_rows_", "graph_degree_")
    _SAVE_ARRAYS = (("dataset_", "<f4"), ("graph_", "<i4"))

    def __init__(self, graph_degree=32, intermediate_graph_degree=64, n_neighbors=8, itopk_size=64,
                 search_width=1, max_iterations=0, n_seeds=0, random_state=None):
        self.graph_degree = graph_degree
        self.intermediate_graph_degree = intermediate_graph_degree
        self.n_neighbors = n_neighbors
        self.itopk_size = itopk_size
        self.search_width = search_width
        self.max_iterations = max_iterations
        self.n_seeds = n_seeds
        self.random_state = random_state

    _KIND = 3

    def _p(self, name):
        return _ann_int("CagraIndex", name, getattr(self, name))

    def _rs(self):
        """The kernels' seed word: 0 for None, r + 1 for random_state r
        (reduced into the int31 range the kernel argument holds)."""
        r = self.random_state
        if r is None:
            return 0
        r = _ann_int("CagraIndex", "random_state", r)
        if r < 0:
            raise ValueError("mojolearn CagraIndex: random_state must be None or >= 0")
        return r % 2147483646 + 1

    def _resident_arrays(self):
        n, deg = self.n_rows_, self.graph_degree_
        return ([addr_ro(self.dataset_, name="dataset_"), addr_ro(self.graph_.reshape((n * deg,)), name="graph_")],
                [n, self.n_features_in_, deg])

    @_native_ann_method
    def fit(self, X, y=None):
        """Build the search graph from `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to index, converted to float32. The index keeps them.
        y : None
            Ignored.

        Returns
        -------
        self : CagraIndex
            The fitted index.
        """
        return _FAMILY(self, "family_xn_ann_cagra_fit", (None, X, y), _FAMILY_CONTEXT)

    @_native_ann_method
    def search(self, queries, filter=None):
        """Find the approximate nearest indexed rows of each query.

        Parameters
        ----------
        queries : array-like of shape (n_queries, n_features)
            Query rows, converted to float32.
        filter : array-like of bool, shape (n_rows_,), optional
            One flag per indexed row; a row flagged False is never returned.
            As in cuVS's single-CTA search, the filter is applied after the
            graph walk, over the `itopk_size` candidates: removed rows still
            guide the walk, and the answer is the first `n_neighbors` kept
            candidates in (distance, id) order. Distances are the same values
            the unfiltered search computes.

        Returns
        -------
        distances : ndarray of shape (n_queries, n_neighbors), float32
            Squared Euclidean distances, ascending.
        indices : ndarray of shape (n_queries, n_neighbors), int32
            Row ids of the neighbors. With a filter, a query with fewer kept
            candidates than `n_neighbors` is padded with distance ``inf`` and
            id ``-1``.

        Raises
        ------
        ValueError
            If the index is not fitted, the queries have a different number of
            features, or `filter` has the wrong shape or dtype.
        """
        return _FAMILY(self, "family_xn_ann_cagra_search", (None, queries, filter), _FAMILY_CONTEXT)

    def _search_k(self, queries, k):
        """The traversal, returning the first `k` itopk entries."""
        q, _ = as_f32_c(queries, ndim=2, name="queries")
        m, d = (int(s) for s in q.shape)  # glue: two shape integers of the input
        if d != self.n_features_in_:
            raise ValueError(f"mojolearn CagraIndex: queries have {d} features, the index has {self.n_features_in_}")
        L = self._p("itopk_size")
        max_iter = self._p("max_iterations") or L
        n_seeds = self._p("n_seeds") or (L + self._p("search_width") * self.graph_degree_)
        n_seeds = min(n_seeds, self.n_rows_)
        dist = empty((m * k,), "<f4")
        idx = empty((m * k,), "<i4")
        n, deg = self.n_rows_, self.graph_degree_
        native = self._bind()
        handle = self._resident_handle(native)
        if handle is not None:
            native.x_ann_index_search(
                handle,
                # queries, out_d, out_i (x_ann/resident.mojo)
                [addr_ro(q, name="queries"), addr(dist, name="distances"), addr(idx, name="indices")],
                # m, k, itopk_size, search_width, max_iterations, n_seeds, rs
                [m, k, L, self._p("search_width"), max_iter, n_seeds, self._rs()],
            )
            return dist.reshape((m, k)), idx.reshape((m, k))
        native.x_ann_cagra_search(
            # x, graph, queries, out_d, out_i
            [addr_ro(self.dataset_, name="dataset_"), addr_ro(self.graph_.reshape((n * deg,)), name="graph_"),
             addr_ro(q, name="queries"), addr(dist, name="distances"), addr(idx, name="indices")],
            # n, d, graph_degree, m, k, itopk_size, search_width, max_iterations, n_seeds, rs
            [n, d, deg, m, k, L, self._p("search_width"), max_iter, n_seeds, self._rs()],
        )
        return dist.reshape((m, k)), idx.reshape((m, k))

    def _search_filtered(self, queries, filter):
        from ._optional_numpy import require_numpy
        np = require_numpy('_expansion_ann')
        k = self._p("n_neighbors")
        native = self._bind()
        # the compaction in the binding (x_ann/filter_topk.mojo)
        L = self._p("itopk_size")
        keep32 = _ann_mask("CagraIndex", filter, self.n_rows_)
        bd, bi = self._search_k(queries, L)
        bd32, _ = as_f32_c(bd, ndim=2, name="itopk distances")
        bi32, _ = as_i32_c(bi, ndim=2, name="itopk indices")
        m = int(bd32.shape[0])
        dist = empty((m * k,), "<f4")
        idx = empty((m * k,), "<i4")
        native.x_ann_filter_topk(
            # itopk distances, itopk ids, keep, out_d, out_i
            [addr_ro(bd32, name="bd"), addr_ro(bi32, name="bi"), addr_ro(keep32, name="filter"),
             addr(dist, name="distances"), addr(idx, name="indices")],
            # m, n, itopk_size, k
            [m, self.n_rows_, int(bd32.shape[1]), k],
        )
        return dist.reshape((m, k)), idx.reshape((m, k))


# IDENTICAL by construction (x_ann/ivf_sq_core.mojo).
class IVFSQIndex(_AnnResident, _AnnSaved, NumericModeMixin):
    """Approximate nearest-neighbor index with inverted lists and 8-bit scalar quantization (IVF-SQ).

    Follows cuVS's ``ivf_sq``. `fit` trains the same k-means coarse quantizer
    as `IVFPQIndex`, then stores each row's residual (row minus its list
    center) quantized to 8 bits per feature, using cuVS's per-feature range
    with a 5% margin. `search` scans the decoded residuals of the `n_probes`
    lists nearest each query and returns approximate squared Euclidean
    distances. Results are reproducible. On a GPU the fitted index stays on
    the device between searches; each search uploads only its queries.

    Parameters
    ----------
    n_lists : int
        Number of inverted lists (k-means clusters). Must be in
        ``[1, n_samples]``.
    n_probes : int
        Number of lists scanned per query. Required, with no default.
    n_neighbors : int, default 8
        Number of neighbors `search` returns per query.
    kmeans_n_iters : int, default 20
        k-means iterations for the coarse quantizer.
    random_state : int, default 0
        Seed for the k-means initialization.
    max_train_points_per_cluster : int or None, default None
        Size of the training sample, as cuVS's ``ivf_sq`` parameter of the
        same name. With an int ``m``, the coarse k-means and the quantization
        ranges are both trained on ``min(n_samples, m * n_lists)`` evenly
        spaced rows, and every row is still indexed. None keeps the default:
        the k-means trains on an evenly spaced subset of the rows and the
        quantization ranges on every row. Must be positive.
    numeric_mode : str or None, default None
        Numeric mode for this estimator; None uses the package default set
        with `mojolearn.set_numeric_mode`.

    Attributes
    ----------
    centers_ : ndarray of shape (n_lists, n_features), float32
        Coarse quantizer centers.
    list_offsets_ : ndarray of shape (n_lists + 1,), int32
        Start of each list in `list_indices_`.
    list_indices_ : ndarray of shape (n_samples,), int32
        Original row ids, grouped by list.
    sq_vmin_ : ndarray of shape (n_features,), float32
        Lower end of each feature's quantization range.
    sq_delta_ : ndarray of shape (n_features,), float32
        Quantization step of each feature.
    codes_ : ndarray of shape (n_samples, n_features), int32
        8-bit code of every residual value.
    n_features_in_ : int
        Number of features seen during `fit`.
    n_rows_ : int
        Number of indexed rows.
    n_lists_ : int
        The fitted value of `n_lists`.
    n_candidates_ : ndarray of shape (n_queries,), int32
        Set by `search`: how many indexed rows each query scanned.

    Raises
    ------
    TypeError
        If an integer parameter is not an int.
    ValueError
        If `n_lists` is out of range or `max_train_points_per_cluster` is not
        positive.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> index = mojolearn.IVFSQIndex(n_lists=16, n_probes=4, n_neighbors=5).fit(X)
    >>> distances, indices = index.search(X[:10])
    >>> indices.shape
    (10, 5)

    Training on at most 32 rows per list:

    >>> index = mojolearn.IVFSQIndex(n_lists=16, n_probes=4, n_neighbors=5,
    ...                              max_train_points_per_cluster=32).fit(X)
    """

    _BINDING = "_mojolearn_x_ann"
    _SAVE_FORMAT = "mojolearn-ivf-sq-1"
    _SAVE_PARAMS = ("n_lists", "n_probes", "n_neighbors", "kmeans_n_iters", "random_state")
    _SAVE_FITTED = ("n_features_in_", "n_rows_", "n_lists_")
    _SAVE_ARRAYS = (("centers_", "<f4"), ("list_offsets_", "<i4"), ("list_indices_", "<i4"),
                    ("sq_vmin_", "<f4"), ("sq_delta_", "<f4"), ("codes_", "<i4"))

    def __init__(self, n_lists, n_probes, n_neighbors=8, kmeans_n_iters=20, random_state=0,
                 max_train_points_per_cluster=None):
        self.n_lists = n_lists
        self.n_probes = n_probes
        self.n_neighbors = n_neighbors
        self.kmeans_n_iters = kmeans_n_iters
        self.random_state = random_state
        # release-0845: cuVS ivf_sq's training-sample size (ivf_sq_build.cuh:461-470); the
        # binding's train_rows_per_list (x_ann/ivf_pq_device.mojo ivf_sq_build_device). Not in
        # _SAVE_PARAMS: it shapes only the build, and files saved before it still load.
        self.max_train_points_per_cluster = max_train_points_per_cluster

    _KIND = 1

    def _p(self, name):
        return _ann_int("IVFSQIndex", name, getattr(self, name))

    def _resident_arrays(self):
        n, dim = self.n_rows_, self.n_features_in_
        return ([addr_ro(self.centers_.reshape((self.n_lists_ * dim,)), name="centers_"),
                 addr_ro(self.list_offsets_, name="list_offsets_"), addr_ro(self.list_indices_, name="list_indices_"),
                 addr_ro(self.sq_vmin_, name="sq_vmin_"), addr_ro(self.sq_delta_, name="sq_delta_"),
                 addr_ro(self.codes_.reshape((n * dim,)), name="codes_")],
                [n, dim, self.n_lists_])

    @_native_ann_method
    def fit(self, X, y=None):
        """Build the index from `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to index, converted to float32.
        y : None
            Ignored.

        Returns
        -------
        self : IVFSQIndex
            The fitted index.
        """
        return _FAMILY(self, "family_xn_ann_sq_fit", (None, X, y), _FAMILY_CONTEXT)

    @_native_ann_method
    def search(self, queries, filter=None):
        """Find the approximate nearest indexed rows of each query.

        Parameters
        ----------
        queries : array-like of shape (n_queries, n_features)
            Query rows, converted to float32.
        filter : array-like of bool, shape (n_rows_,), optional
            One flag per indexed row; a row flagged False is never returned
            (cuVS's sample filter, applied before scoring).

        Returns
        -------
        distances : ndarray of shape (n_queries, n_neighbors), float32
            Approximate squared Euclidean distances, ascending.
        indices : ndarray of shape (n_queries, n_neighbors), int32
            Original row ids of the neighbors. A query whose probed lists hold
            fewer than `n_neighbors` rows is padded with distance ``inf`` and
            id ``-1``.

        Raises
        ------
        ValueError
            If the index is not fitted, the queries have a different number of
            features, or `filter` has the wrong shape or dtype.
        """
        return _FAMILY(self, "family_xn_ann_sq_search", (None, queries, filter), _FAMILY_CONTEXT)


# IDENTICAL by construction (x_ann/ivf_rabitq_core.mojo).
class IVFRaBitQIndex(_AnnResident, _AnnSaved, NumericModeMixin):
    """Approximate nearest-neighbor index with inverted lists and 1-bit RaBitQ codes (IVF-RaBitQ).

    Follows cuVS's ``ivf_rabitq`` (RaBitQ, Gao and Long). `fit` trains a
    k-means coarse quantizer, rotates each row's residual by a randomized
    Hadamard transform, and keeps its sign bits together with its norm and an
    inner-product correction factor. `search` scans the `n_probes` lists
    nearest each query and returns the RaBitQ distance estimates, the
    `n_neighbors` smallest by (estimate, id). The distances are estimates;
    pass the candidates to `refine` for exact distances. Results are
    reproducible. On a GPU the fitted index stays on the device between
    searches; each search uploads only its queries.

    Parameters
    ----------
    n_lists : int
        Number of inverted lists (k-means clusters). Must be in
        ``[1, n_samples]``.
    n_probes : int
        Number of lists scanned per query. Required, with no default.
    n_neighbors : int, default 8
        Number of neighbors `search` returns per query.
    kmeans_n_iters : int, default 20
        k-means iterations for the coarse quantizer.
    random_state : int, default 0
        Seed for the k-means initialization and the rotation's random signs.
    numeric_mode : str or None, default None
        Numeric mode for this estimator; None uses the package default set
        with `mojolearn.set_numeric_mode`.

    Attributes
    ----------
    centers_ : ndarray of shape (n_lists, n_features), float32
        Coarse quantizer centers.
    list_offsets_ : ndarray of shape (n_lists + 1,), int32
        Start of each list in `list_indices_`.
    list_indices_ : ndarray of shape (n_samples,), int32
        Original row ids, grouped by list.
    codes_ : ndarray of shape (n_samples, n_words), int32
        Sign bits of each rotated residual, packed 32 per word; the residual
        is zero-padded to the next power of two.
    norms_ : ndarray of shape (n_samples,), float32
        Norm of each residual.
    ip_factors_ : ndarray of shape (n_samples,), float32
        Inner-product correction factor of each row.
    n_features_in_ : int
        Number of features seen during `fit`.
    n_rows_ : int
        Number of indexed rows.
    n_lists_ : int
        The fitted value of `n_lists`.
    seed_ : int
        The seed used for the rotation.
    n_candidates_ : ndarray of shape (n_queries,), int32
        Set by `search`: how many indexed rows each query scanned.

    Raises
    ------
    TypeError
        If an integer parameter is not an int.
    ValueError
        If `n_lists` is out of range.

    See Also
    --------
    refine : Re-rank candidates with exact distances.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> index = mojolearn.IVFRaBitQIndex(n_lists=16, n_probes=4, n_neighbors=20).fit(X)
    >>> _, candidates = index.search(X[:10])
    >>> distances, indices = mojolearn.refine(X, X[:10], candidates, k=5)
    """

    _BINDING = "_mojolearn_x_ann"
    _SAVE_FORMAT = "mojolearn-ivf-rabitq-1"
    _SAVE_PARAMS = ("n_lists", "n_probes", "n_neighbors", "kmeans_n_iters", "random_state")
    _SAVE_FITTED = ("n_features_in_", "n_rows_", "n_lists_", "seed_")
    _SAVE_ARRAYS = (("centers_", "<f4"), ("list_offsets_", "<i4"), ("list_indices_", "<i4"),
                    ("codes_", "<i4"), ("norms_", "<f4"), ("ip_factors_", "<f4"))

    def __init__(self, n_lists, n_probes, n_neighbors=8, kmeans_n_iters=20, random_state=0):
        self.n_lists = n_lists
        self.n_probes = n_probes
        self.n_neighbors = n_neighbors
        self.kmeans_n_iters = kmeans_n_iters
        self.random_state = random_state

    _KIND = 2

    def _p(self, name):
        return _ann_int("IVFRaBitQIndex", name, getattr(self, name))

    def _resident_arrays(self):
        n, dim = self.n_rows_, self.n_features_in_
        words = int(self.codes_.shape[1])
        return ([addr_ro(self.centers_.reshape((self.n_lists_ * dim,)), name="centers_"),
                 addr_ro(self.list_offsets_, name="list_offsets_"), addr_ro(self.list_indices_, name="list_indices_"),
                 addr_ro(self.codes_.reshape((n * words,)), name="codes_"), addr_ro(self.norms_, name="norms_"),
                 addr_ro(self.ip_factors_, name="ip_factors_")],
                [n, dim, self.n_lists_, self.seed_])

    @_native_ann_method
    def fit(self, X, y=None):
        """Build the index from `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to index, converted to float32.
        y : None
            Ignored.

        Returns
        -------
        self : IVFRaBitQIndex
            The fitted index.
        """
        return _FAMILY(self, "family_xn_ann_rabitq_fit", (None, X, y), _FAMILY_CONTEXT)

    @_native_ann_method
    def search(self, queries, filter=None):
        """Find the approximate nearest indexed rows of each query.

        Parameters
        ----------
        queries : array-like of shape (n_queries, n_features)
            Query rows, converted to float32.
        filter : array-like of bool, shape (n_rows_,), optional
            One flag per indexed row; a row flagged False is never returned
            (cuVS's sample filter, applied before scoring).

        Returns
        -------
        distances : ndarray of shape (n_queries, n_neighbors), float32
            Estimated squared Euclidean distances, ascending.
        indices : ndarray of shape (n_queries, n_neighbors), int32
            Original row ids of the neighbors. A query whose probed lists hold
            fewer than `n_neighbors` rows is padded with distance ``inf`` and
            id ``-1``.

        Raises
        ------
        ValueError
            If the index is not fitted, the queries have a different number of
            features, or `filter` has the wrong shape or dtype.
        """
        return _FAMILY(self, "family_xn_ann_rabitq_search", (None, queries, filter), _FAMILY_CONTEXT)


# metric='euclidean' is cuVS L2SqrtExpanded: the selection runs on the squared
# keys and the root of each of the k selected distances is taken afterwards
# (in the binding), where cuVS takes it at the store.
def refine(dataset, queries, candidates, k, numeric_mode=None, metric="sqeuclidean"):
    """Re-rank candidate neighbors by exact distance.

    Follows cuVS's ``refine``. For each query, computes the exact distance to
    each of its candidate rows of `dataset` and keeps the `k` smallest, ties
    broken by row id. Use it after an approximate search (for example
    `IVFRaBitQIndex` or `IVFPQIndex` with a larger `n_neighbors`) to get exact
    distances and a better ordering.

    Parameters
    ----------
    dataset : array-like of shape (n_samples, n_features)
        The indexed rows, converted to float32.
    queries : array-like of shape (n_queries, n_features)
        Query rows, converted to float32.
    candidates : array-like of int, shape (n_queries, n_candidates)
        Candidate row ids for each query. Negative ids are padding and are
        skipped, as are ids outside ``[0, n_samples)``.
    k : int
        Number of neighbors to return per query.
    numeric_mode : str or None, default None
        Numeric mode to run in; None uses the package default set with
        `mojolearn.set_numeric_mode`.
    metric : {'sqeuclidean', 'euclidean'}, default 'sqeuclidean'
        'sqeuclidean' returns squared Euclidean distances (cuVS
        ``L2Expanded``); 'euclidean' returns their square roots (cuVS
        ``L2SqrtExpanded``). The ids and their order are the same under both.
        Inner product is not supported.

    Returns
    -------
    distances : ndarray of shape (n_queries, k), float32
        Exact distances, ascending.
    indices : ndarray of shape (n_queries, k), int32
        Row ids of the neighbors. A query with fewer than `k` valid
        candidates is padded with distance ``inf`` and id ``-1``.

    Raises
    ------
    ValueError
        If `metric` is not supported, the feature counts differ, `candidates`
        is not an integer array of shape (n_queries, n_candidates), or a
        candidate id does not fit in int32.
    TypeError
        If `k` is not an int.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> candidates = np.tile(np.arange(50), (10, 1))
    >>> distances, indices = mojolearn.refine(X, X[:10], candidates, k=5)
    >>> indices.shape
    (10, 5)
    """
    return _FAMILY(None, "family_xn_ann_refine",
                   (None, dataset, queries, candidates, k, metric, numeric_mode),
                   _FAMILY_CONTEXT)


def _ann_save_schema(cls, items=None, device=None):
    """The save schema of one `_AnnSaved` class, from its fixed tables:
    every array present ("call fit before save"), the head, the int `params`
    (each an int and not a bool, as `_ann_int` refused), the `fitted` ints
    and the arrays as stored, in the members' 0.8.55 order. `device` maps an
    array a resident fit may hold only on the device to (marker, export
    entry, shape items, refusal): it is exported into the file."""
    items = items or {}
    device = device or {}
    return (
        *(("need_either", a, device[a][0], "RuntimeError", "mojolearn {cls}: call fit before save") if a in device
          else ("need", a, "RuntimeError", "mojolearn {cls}: call fit before save")
          for a, _dtype in cls._SAVE_ARRAYS),  # glue: fixed saved array table, at import
        *_native_archive_bridge.head(cls._SAVE_FORMAT),
        ("i8", "params", tuple(items.get(p, ("strict", p, "TypeError", f"mojolearn {{cls}}: {p} must be an int, got {{type}}"))
                               for p in cls._SAVE_PARAMS)),  # glue: fixed saved parameter table, at import
        ("i8", "fitted", tuple(("int", a) for a in cls._SAVE_FITTED)),  # glue: fixed fitted table, at import
        *(("device_array", a.rstrip("_"), a, device[a][0], device[a][1], device[a][2], dtype, "RuntimeError", device[a][3])
          if a in device else ("array", a.rstrip("_"), a)
          for a, dtype in cls._SAVE_ARRAYS),  # glue: fixed saved array table, at import
    )


def _ann_layout(cls, none_below=(), **defaults):
    """`host_archive_ann`'s load layout of one `_AnnSaved` class: its saved
    parameter, fitted and array tables, the constructor's other defaults
    and the parameters saved as -1 for None."""
    return (tuple(cls._SAVE_PARAMS), tuple(cls._SAVE_FITTED),
            tuple((a, a.rstrip("_"), dtype) for a, dtype in cls._SAVE_ARRAYS),  # glue: fixed saved array table, at import
            dict(defaults), tuple(none_below))


# CagraIndex saves random_state as -1 for None, else r % 2147483646 (the
# kernels' seed word minus one), refusing a non-int and a negative value.
_CAGRA_SEED = ("none_or", "random_state", -1, ("mod", ("at_least", (
    "strict", "random_state", "TypeError", "mojolearn CagraIndex: random_state must be an int, got {type}"),
    0, "ValueError", "mojolearn CagraIndex: random_state must be None or >= 0"), 2147483646))

# The loaded indexes (`_AnnSaved.load`): a saved random_state below 0 is
# CagraIndex's None; IVFSQIndex's build-only max_train_points_per_cluster is
# not saved and loads as its default.
_ANN_LAYOUTS = {
    IVFPQIndex._SAVE_FORMAT: _ann_layout(IVFPQIndex),
    CagraIndex._SAVE_FORMAT: _ann_layout(CagraIndex, none_below=("random_state",)),
    IVFSQIndex._SAVE_FORMAT: _ann_layout(IVFSQIndex, max_train_points_per_cluster=None),
    IVFRaBitQIndex._SAVE_FORMAT: _ann_layout(IVFRaBitQIndex),
}
_PQ_DEVICE = {"codes_": ("_device_codes", "x_ann_index_export_codes", (("int", "n_rows_"), ("int", "pq_dim_")),
                         "mojolearn IVFPQIndex: the resident index was released before its codes were read")}

_native_archive_bridge.register(IVFPQIndex, IVFPQIndex._SAVE_FORMAT, _ann_save_schema(IVFPQIndex, device=_PQ_DEVICE),
                                host_archive_ann_layouts=_ANN_LAYOUTS)
_native_archive_bridge.register(CagraIndex, CagraIndex._SAVE_FORMAT,
                                _ann_save_schema(CagraIndex, {"random_state": _CAGRA_SEED}),
                                host_archive_ann_layouts=_ANN_LAYOUTS)
_native_archive_bridge.register(IVFSQIndex, IVFSQIndex._SAVE_FORMAT, _ann_save_schema(IVFSQIndex),
                                host_archive_ann_layouts=_ANN_LAYOUTS)
_native_archive_bridge.register(IVFRaBitQIndex, IVFRaBitQIndex._SAVE_FORMAT, _ann_save_schema(IVFRaBitQIndex),
                                host_archive_ann_layouts=_ANN_LAYOUTS)
# The device index of every x_ann kind is released by the native finalizer,
# through the minting binding's `x_ann_index_release`.
_native_archive_bridge.lifetime(_AnnResident, "x_ann_index_release")

# Import-time connection; native operations select the live configured binding.
_native_family_bridge.connect_bindings("_mojolearn_x_ann")

# Import-time native attribute ownership replaces PQ's lazy Python __getattr__.
# The native slot exports device codes on demand and publishes the completed
# Array/ndarray under the same public attribute, retaining the device owner.
from ._native_neighbor_queries import connect as _connect_ann_attributes
del IVFPQIndex.__getattr__
_ANN_ATTRIBUTE_CALL, _ANN_ATTRIBUTE_CONTEXT = _connect_ann_attributes(
    {IVFPQIndex: "ann_pq"}, _FAMILY_CONTEXT, _ARCHIVE_CONTEXT, ())

# Import-time connection for the native manifold fit lifecycle.
from ._native_runtime_manifold_bridge import connect_lifecycle as _connect_manifold_lifecycle
_connect_manifold_lifecycle(TSNE)
