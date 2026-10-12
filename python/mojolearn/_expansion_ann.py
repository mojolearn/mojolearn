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
from ._buffer import addr, addr_ro, as_f32_c, as_i32_c, empty
from ._mode import NumericModeMixin

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

    def __del__(self):
        try:
            self._release_resident()
        except Exception:  # noqa: BLE001
            pass

    def __getstate__(self):
        """No handle crosses a pickle or a deepcopy: it means something only
        in the process and registry that minted it."""
        state = self.__dict__.copy()
        state.pop("_resident", None)
        return state


class _AnnSaved:
    """save / load for the ann indexes: an npz holding the constructor's int
    parameters, the fitted ints and the index arrays with their dtype and
    shape, and the tier. A loaded index searches; it does not rebuild.
    No arithmetic: the arrays are written and read back byte for byte, and
    `_serialize.exact` refuses any dtype cast."""

    _SAVE_FORMAT = None
    _SAVE_PARAMS = ()      # constructor parameters, all int
    _SAVE_FITTED = ()      # fitted int attributes
    _SAVE_ARRAYS = ()      # (attribute, typestr)

    def _save_param(self, p):
        return _ann_int(type(self).__name__, p, getattr(self, p))

    @classmethod
    def _load_param(cls, p, v):
        return v

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
        from . import _serialize
        from ._array import Array
        from .decomposition import _saved_mode
        missing = [a for a, _ in self._SAVE_ARRAYS if not hasattr(self, a)]  # glue: saved array attributes by name
        if missing:
            raise RuntimeError(f"mojolearn {type(self).__name__}: call fit before save")
        arrays = {
            "format": self._SAVE_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "params": Array.from_list([self._save_param(p) for p in self._SAVE_PARAMS], "<i8"),  # glue: saved parameters by name
            "fitted": Array.from_list([int(getattr(self, a)) for a in self._SAVE_FITTED], "<i8"),  # glue: saved parameters by name
        }
        for attr, _dtype in self._SAVE_ARRAYS:  # glue: saved array attributes by name
            arrays[attr.rstrip("_")] = getattr(self, attr)
        return _serialize.write_npz(path, arrays)

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
        from . import _serialize
        from .decomposition import _check_saved_by, _restore_mode
        arrays = _serialize.read_npz(path, cls._SAVE_FORMAT)
        _check_saved_by(arrays, path, cls)
        params = _serialize.exact(arrays, "params", "<i8")
        fitted = _serialize.exact(arrays, "fitted", "<i8")
        if params.size != len(cls._SAVE_PARAMS) or fitted.size != len(cls._SAVE_FITTED):
            raise ValueError(f"mojolearn: {path!r} does not hold {cls.__name__}'s parameters")
        obj = cls(**{p: cls._load_param(p, int(params[i])) for i, p in enumerate(cls._SAVE_PARAMS)})  # glue: saved parameters by name
        _restore_mode(obj, arrays)
        for i, a in enumerate(cls._SAVE_FITTED):  # glue: saved parameters by name
            setattr(obj, a, int(fitted[i]))
        for attr, dtype in cls._SAVE_ARRAYS:  # glue: saved array attributes by name
            setattr(obj, attr, _serialize.exact(arrays, attr.rstrip("_"), dtype))
        return obj


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
    codebooks_ : ndarray of shape (pq_dim, 2**pq_bits, pq_len_), float32
        One codebook per subspace.
    codes_ : ndarray of shape (n_samples, pq_dim), int32
        PQ code of every indexed row.
    n_features_in_ : int
        Number of features seen during `fit`.
    n_rows_ : int
        Number of indexed rows.
    n_lists_, pq_dim_, pq_bits_ : int
        The fitted values of `n_lists`, `pq_dim` and `pq_bits`.
    pq_len_ : int
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
        self._materialize_device_arrays()
        return super().__getstate__()

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
        self.__dict__.pop("_device_codes", None)
        self._release_resident()
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, dim = (int(s) for s in x.shape)  # glue: two shape integers of the input
        n_lists, pq_dim, pq_bits = self._p("n_lists"), self._p("pq_dim"), self._p("pq_bits")
        if not 1 <= pq_dim <= dim:
            raise ValueError(f"mojolearn IVFPQIndex: pq_dim must be in [1, {dim}], got {pq_dim}")
        if not 1 <= pq_bits <= 8:
            raise ValueError(f"mojolearn IVFPQIndex: pq_bits must be in [1, 8], got {pq_bits}")
        if not 1 <= n_lists <= n:
            raise ValueError(f"mojolearn IVFPQIndex: n_lists must be in [1, {n}], got {n_lists}")
        pq_len = -(-dim // pq_dim)
        n_codes = 1 << pq_bits
        centers = empty((n_lists * dim,), "<f4")
        offsets = empty((n_lists + 1,), "<i4")
        indices = empty((n,), "<i4")
        codebooks = empty((pq_dim * n_codes * pq_len,), "<f4")
        native = self._bind()
        resident_build = getattr(native, "x_ann_ivf_pq_build_resident", None)
        if resident_build is not None:
            # lane fg-ivf A7: the index stays on the device under a handle;
            # the four small arrays are written here, `codes_` on first read
            handle = int(resident_build(
                # x, centers, offsets, list_indices, codebooks (bindings/_mojolearn_x_ann.mojo)
                [addr_ro(x, name="X"), addr(centers, name="centers_"), addr(offsets, name="list_offsets_"),
                 addr(indices, name="list_indices_"), addr(codebooks, name="codebooks_")],
                # n, dim, n_lists, kmeans_n_iters, seed, pq_dim, pq_bits, pq_kmeans_n_iters
                [n, dim, n_lists, self._p("kmeans_n_iters"), self._p("random_state"), pq_dim, pq_bits,
                 self._p("pq_kmeans_n_iters")],
            ))
            self.__dict__.pop("codes_", None)
            self.centers_ = centers.reshape((n_lists, dim))
            self.list_offsets_ = offsets
            self.list_indices_ = indices
            self.codebooks_ = codebooks.reshape((pq_dim, n_codes, pq_len))
            self.n_features_in_, self.n_rows_, self.n_lists_ = dim, n, n_lists
            self.pq_dim_, self.pq_bits_, self.pq_len_ = pq_dim, pq_bits, pq_len
            self._device_codes = True
            self._resident = (self._resident_key(native), handle, native)
            return self
        codes = empty((n * pq_dim,), "<i4")
        native.x_ann_ivf_pq_build(
            # x, centers, offsets, list_indices, codebooks, codes
            [addr_ro(x, name="X"), addr(centers, name="centers_"), addr(offsets, name="list_offsets_"),
             addr(indices, name="list_indices_"), addr(codebooks, name="codebooks_"), addr(codes, name="codes_")],
            # n, dim, n_lists, kmeans_n_iters, seed, pq_dim, pq_bits, pq_kmeans_n_iters
            [n, dim, n_lists, self._p("kmeans_n_iters"), self._p("random_state"), pq_dim, pq_bits,
             self._p("pq_kmeans_n_iters")],
        )
        self.centers_ = centers.reshape((n_lists, dim))
        self.list_offsets_ = offsets
        self.list_indices_ = indices
        self.codebooks_ = codebooks.reshape((pq_dim, n_codes, pq_len))
        self.codes_ = codes.reshape((n, pq_dim))
        self.n_features_in_, self.n_rows_, self.n_lists_ = dim, n, n_lists
        self.pq_dim_, self.pq_bits_, self.pq_len_ = pq_dim, pq_bits, pq_len
        return self

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
        if not self._is_built():
            raise ValueError("mojolearn IVFPQIndex: call fit before search")
        q, _ = as_f32_c(queries, ndim=2, name="queries")
        m, dim = (int(s) for s in q.shape)  # glue: two shape integers of the input
        if dim != self.n_features_in_:
            raise ValueError(f"mojolearn IVFPQIndex: queries have {dim} features, the index has {self.n_features_in_}")
        k, n_probes = self._p("n_neighbors"), self._p("n_probes")
        n, pq_dim = self.n_rows_, self.pq_dim_
        dist = empty((m * k,), "<f4")
        idx = empty((m * k,), "<i4")
        cand = empty((m,), "<i4")
        native = self._bind()
        if self._resident_search(native, "IVFPQIndex", filter, q, dist, idx, cand, [m, k, n_probes]):
            return dist.reshape((m, k)), idx.reshape((m, k))
        mask = _ann_mask("IVFPQIndex", filter, n)  # held: the binding reads it after this line
        native.x_ann_ivf_pq_search(
            # centers, offsets, list_indices, codebooks, codes, queries, out_d, out_i, out_n
            [addr_ro(self.centers_.reshape((self.n_lists_ * dim,)), name="centers_"),
             addr_ro(self.list_offsets_, name="list_offsets_"), addr_ro(self.list_indices_, name="list_indices_"),
             addr_ro(self.codebooks_.reshape((pq_dim * (1 << self.pq_bits_) * self.pq_len_,)), name="codebooks_"),
             addr_ro(self.codes_.reshape((n * pq_dim,)), name="codes_"), addr_ro(q, name="queries"),
             addr(dist, name="distances"), addr(idx, name="indices"), addr(cand, name="n_candidates_"),
             addr_ro(mask, name="filter")],
            # n, dim, n_lists, pq_dim, pq_bits, m, k, n_probes
            [n, dim, self.n_lists_, pq_dim, self.pq_bits_, m, k, n_probes],
        )
        self.n_candidates_ = cand
        return dist.reshape((m, k)), idx.reshape((m, k))


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
        Dimension of the embedding. Only 2 is supported.
    perplexity : float, default 30.0
        Effective number of neighbors. Must be in ``(0, n_samples)``.
    early_exaggeration : float, default 12.0
        Factor applied to the attractive forces during the early phase.
    learning_rate : float or 'auto', default 'auto'
        Optimizer step size. 'auto' is scikit-learn's
        ``max(n_samples / early_exaggeration / 4, 50)``.
    max_iter : int, default 1000
        Number of optimization iterations, all of which run.
    init : {'pca', 'random'} or array-like of shape (n_samples, 2), default 'pca'
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
    """

    _BINDING = "_mojolearn_x_ann"
    _EXPLORATION_MAX_ITER = 250

    def __init__(self, n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate="auto",
                 max_iter=1000, init="pca", random_state=0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state

    def _init(self, x, n, seed):
        """The caller's start array (n, 2) float32 ('pca' and 'random' are
        built in the binding, x_ann/tsne_init.mojo)."""
        from ._optional_numpy import require_numpy
        np = require_numpy('_expansion_ann')
        init = self.init
        if isinstance(init, str):
            raise ValueError(f"mojolearn TSNE: init must be 'pca', 'random' or an array, got {init!r}")
        y0 = np.ascontiguousarray(np.asarray(init, dtype=np.float32))
        if y0.shape != (n, 2):
            raise ValueError(f"mojolearn TSNE: an init array must have shape ({n}, 2), got {y0.shape}")
        return y0

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
            If `n_components` is not 2, `perplexity` is out of range, `init`
            is an unknown string, or an `init` array has the wrong shape.
        TypeError
            If `max_iter` or `random_state` is not an int.
        """
        from ._optional_numpy import require_numpy
        np = require_numpy('_expansion_ann')
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, d = (int(s) for s in x.shape)  # glue: two shape integers of the input
        if self.n_components != 2:
            raise ValueError("mojolearn TSNE: n_components must be 2 (the only arm implemented)")
        max_iter = _ann_int("TSNE", "max_iter", self.max_iter)
        seed = _ann_int("TSNE", "random_state", self.random_state)
        perplexity = float(self.perplexity)
        if not 0.0 < perplexity < n:
            raise ValueError(f"mojolearn TSNE: perplexity must be in (0, {n}), got {perplexity}")
        exag = float(self.early_exaggeration)
        if self.learning_rate == "auto":
            lr = max(n / exag / 4.0, 50.0)
        else:
            lr = float(self.learning_rate)
        exploration = min(self._EXPLORATION_MAX_ITER, max_iter)
        native = self._bind()
        tail = []
        init = self.init
        if isinstance(init, str) and init in ("random", "pca"):
            # the binding builds the start (x_ann/tsne_init.mojo): mode 1
            # random from random_state, mode 2 the PCA embedding scaled
            if init == "random":
                y0 = empty((2,), "<f4")
                mode = 1
            else:
                from .decomposition import PCA
                pca = PCA(n_components=2)
                mode_name = getattr(self, "numeric_mode", None)
                if mode_name is not None:
                    pca.numeric_mode = mode_name
                y0 = np.ascontiguousarray(np.asarray(pca.fit_transform(x), dtype=np.float32))
                mode = 2
            s64 = seed & ((1 << 64) - 1)
            tail = [mode, s64 & 0xFFFFFFFF, s64 >> 32]
        else:
            y0 = self._init(x, n, seed)       # the caller's array (or a refusal)
        emb = empty((n * 2,), "<f4")
        kl = empty((1,), "<f4")
        native.x_ann_tsne_fit(
            # x, y0, y_out, kl_out
            [addr_ro(x, name="X"), addr_ro(y0, name="init"), addr(emb, name="embedding_"), addr(kl, name="kl")],
            # n, d, max_iter, exploration_iters, perplexity, early_exaggeration, learning_rate
            # (+ init mode, random_state low, high: the binding builds y0)
            [n, d, max_iter, exploration, perplexity, exag, lr] + tail,
        )
        self.embedding_ = emb.reshape((n, 2))
        self.kl_divergence_ = float(np.asarray(kl)[0])
        self.n_iter_ = max_iter
        self.learning_rate_ = lr
        self.n_features_in_ = d
        return self

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
        embedding : ndarray of shape (n_samples, 2), float32
            The embedding, also stored in `embedding_`.
        """
        return self.fit(X).embedding_


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
        Search iterations; 0 means `itopk_size`.
    n_seeds : int, default 0
        Number of start nodes; 0 means cuVS's default,
        ``itopk_size + search_width * graph_degree`` (capped at the number of
        rows).
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
    graph_ : ndarray of shape (n_samples, graph_degree_), int32
        Neighbor ids of each row in the search graph.
    graph_degree_ : int
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

    def _save_param(self, p):
        if p == "random_state":
            return -1 if self.random_state is None else self._rs() - 1
        return super()._save_param(p)

    @classmethod
    def _load_param(cls, p, v):
        if p == "random_state":
            return None if v < 0 else v
        return v

    def _resident_arrays(self):
        n, deg = self.n_rows_, self.graph_degree_
        return ([addr_ro(self.dataset_, name="dataset_"), addr_ro(self.graph_.reshape((n * deg,)), name="graph_")],
                [n, self.n_features_in_, deg])

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
        self._release_resident()
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, d = (int(s) for s in x.shape)  # glue: two shape integers of the input
        kdeg = min(self._p("intermediate_graph_degree"), n - 1)
        deg = min(self._p("graph_degree"), kdeg)
        graph = empty((n * deg,), "<i4")
        self._bind().x_ann_cagra_build(
            # x, graph_out
            [addr_ro(x, name="X"), addr(graph, name="graph_")],
            # n, d, intermediate_graph_degree, graph_degree
            [n, d, kdeg, deg],
        )
        self.dataset_ = x
        self.graph_ = graph.reshape((n, deg))
        self.n_features_in_, self.n_rows_, self.graph_degree_ = d, n, deg
        return self

    # filtered: integer selection only over the itopk buffer
    # (x_ann/filter_topk.mojo); padding matches the IVF filters'.
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
        if not hasattr(self, "graph_"):
            raise ValueError("mojolearn CagraIndex: call fit before search")
        if filter is not None:
            return self._search_filtered(queries, filter)
        return self._search_k(queries, self._p("n_neighbors"))

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
        If `n_lists` is out of range.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> index = mojolearn.IVFSQIndex(n_lists=16, n_probes=4, n_neighbors=5).fit(X)
    >>> distances, indices = index.search(X[:10])
    >>> indices.shape
    (10, 5)
    """

    _BINDING = "_mojolearn_x_ann"
    _SAVE_FORMAT = "mojolearn-ivf-sq-1"
    _SAVE_PARAMS = ("n_lists", "n_probes", "n_neighbors", "kmeans_n_iters", "random_state")
    _SAVE_FITTED = ("n_features_in_", "n_rows_", "n_lists_")
    _SAVE_ARRAYS = (("centers_", "<f4"), ("list_offsets_", "<i4"), ("list_indices_", "<i4"),
                    ("sq_vmin_", "<f4"), ("sq_delta_", "<f4"), ("codes_", "<i4"))

    def __init__(self, n_lists, n_probes, n_neighbors=8, kmeans_n_iters=20, random_state=0):
        self.n_lists = n_lists
        self.n_probes = n_probes
        self.n_neighbors = n_neighbors
        self.kmeans_n_iters = kmeans_n_iters
        self.random_state = random_state

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
        self._release_resident()
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, dim = (int(s) for s in x.shape)  # glue: two shape integers of the input
        n_lists = self._p("n_lists")
        if not 1 <= n_lists <= n:
            raise ValueError(f"mojolearn IVFSQIndex: n_lists must be in [1, {n}], got {n_lists}")
        centers = empty((n_lists * dim,), "<f4")
        offsets = empty((n_lists + 1,), "<i4")
        indices = empty((n,), "<i4")
        vmin = empty((dim,), "<f4")
        delta = empty((dim,), "<f4")
        codes = empty((n * dim,), "<i4")
        self._bind().x_ann_ivf_sq_build(
            # x, centers, offsets, list_indices, vmin, delta, codes
            [addr_ro(x, name="X"), addr(centers, name="centers_"), addr(offsets, name="list_offsets_"),
             addr(indices, name="list_indices_"), addr(vmin, name="sq_vmin_"), addr(delta, name="sq_delta_"),
             addr(codes, name="codes_")],
            # n, dim, n_lists, kmeans_n_iters, seed
            [n, dim, n_lists, self._p("kmeans_n_iters"), self._p("random_state")],
        )
        self.centers_ = centers.reshape((n_lists, dim))
        self.list_offsets_, self.list_indices_ = offsets, indices
        self.sq_vmin_, self.sq_delta_ = vmin, delta
        self.codes_ = codes.reshape((n, dim))
        self.n_features_in_, self.n_rows_, self.n_lists_ = dim, n, n_lists
        return self

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
        if not hasattr(self, "codes_"):
            raise ValueError("mojolearn IVFSQIndex: call fit before search")
        q, _ = as_f32_c(queries, ndim=2, name="queries")
        m, dim = (int(s) for s in q.shape)  # glue: two shape integers of the input
        if dim != self.n_features_in_:
            raise ValueError(f"mojolearn IVFSQIndex: queries have {dim} features, the index has {self.n_features_in_}")
        k, n = self._p("n_neighbors"), self.n_rows_
        dist = empty((m * k,), "<f4")
        idx = empty((m * k,), "<i4")
        cand = empty((m,), "<i4")
        native = self._bind()
        if self._resident_search(native, "IVFSQIndex", filter, q, dist, idx, cand, [m, k, self._p("n_probes")]):
            return dist.reshape((m, k)), idx.reshape((m, k))
        mask = _ann_mask("IVFSQIndex", filter, n)  # held: the binding reads it after this line
        native.x_ann_ivf_sq_search(
            # centers, offsets, list_indices, vmin, delta, codes, mask, queries, out_d, out_i, out_n
            [addr_ro(self.centers_.reshape((self.n_lists_ * dim,)), name="centers_"),
             addr_ro(self.list_offsets_, name="list_offsets_"), addr_ro(self.list_indices_, name="list_indices_"),
             addr_ro(self.sq_vmin_, name="sq_vmin_"), addr_ro(self.sq_delta_, name="sq_delta_"),
             addr_ro(self.codes_.reshape((n * dim,)), name="codes_"),
             addr_ro(mask, name="filter"), addr_ro(q, name="queries"),
             addr(dist, name="distances"), addr(idx, name="indices"), addr(cand, name="n_candidates_")],
            # n, dim, n_lists, m, k, n_probes
            [n, dim, self.n_lists_, m, k, self._p("n_probes")],
        )
        self.n_candidates_ = cand
        return dist.reshape((m, k)), idx.reshape((m, k))


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
        self._release_resident()
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, dim = (int(s) for s in x.shape)  # glue: two shape integers of the input
        n_lists = self._p("n_lists")
        if not 1 <= n_lists <= n:
            raise ValueError(f"mojolearn IVFRaBitQIndex: n_lists must be in [1, {n}], got {n_lists}")
        D = 1
        while D < dim:
            D *= 2
        words = (D + 31) // 32
        centers = empty((n_lists * dim,), "<f4")
        offsets = empty((n_lists + 1,), "<i4")
        indices = empty((n,), "<i4")
        codes = empty((n * words,), "<i4")
        norms = empty((n,), "<f4")
        ips = empty((n,), "<f4")
        self._bind().x_ann_ivf_rabitq_build(
            # x, centers, offsets, list_indices, codes, norms, ips
            [addr_ro(x, name="X"), addr(centers, name="centers_"), addr(offsets, name="list_offsets_"),
             addr(indices, name="list_indices_"), addr(codes, name="codes_"), addr(norms, name="norms_"),
             addr(ips, name="ip_factors_")],
            # n, dim, n_lists, kmeans_n_iters, seed
            [n, dim, n_lists, self._p("kmeans_n_iters"), self._p("random_state")],
        )
        self.centers_ = centers.reshape((n_lists, dim))
        self.list_offsets_, self.list_indices_ = offsets, indices
        self.codes_ = codes.reshape((n, words))
        self.norms_, self.ip_factors_ = norms, ips
        self.n_features_in_, self.n_rows_, self.n_lists_, self.seed_ = dim, n, n_lists, self._p("random_state")
        return self

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
        if not hasattr(self, "codes_"):
            raise ValueError("mojolearn IVFRaBitQIndex: call fit before search")
        q, _ = as_f32_c(queries, ndim=2, name="queries")
        m, dim = (int(s) for s in q.shape)  # glue: two shape integers of the input
        if dim != self.n_features_in_:
            raise ValueError(f"mojolearn IVFRaBitQIndex: queries have {dim} features, the index has {self.n_features_in_}")
        k, n = self._p("n_neighbors"), self.n_rows_
        dist = empty((m * k,), "<f4")
        idx = empty((m * k,), "<i4")
        cand = empty((m,), "<i4")
        words = int(self.codes_.shape[1])
        native = self._bind()
        if self._resident_search(native, "IVFRaBitQIndex", filter, q, dist, idx, cand, [m, k, self._p("n_probes")]):
            return dist.reshape((m, k)), idx.reshape((m, k))
        mask = _ann_mask("IVFRaBitQIndex", filter, n)  # held: the binding reads it after this line
        native.x_ann_ivf_rabitq_search(
            # centers, offsets, list_indices, codes, norms, ips, mask, queries, out_d, out_i, out_n
            [addr_ro(self.centers_.reshape((self.n_lists_ * dim,)), name="centers_"),
             addr_ro(self.list_offsets_, name="list_offsets_"), addr_ro(self.list_indices_, name="list_indices_"),
             addr_ro(self.codes_.reshape((n * words,)), name="codes_"), addr_ro(self.norms_, name="norms_"),
             addr_ro(self.ip_factors_, name="ip_factors_"),
             addr_ro(mask, name="filter"), addr_ro(q, name="queries"),
             addr(dist, name="distances"), addr(idx, name="indices"), addr(cand, name="n_candidates_")],
            # n, dim, n_lists, seed, m, k, n_probes
            [n, dim, self.n_lists_, self.seed_, m, k, self._p("n_probes")],
        )
        self.n_candidates_ = cand
        return dist.reshape((m, k)), idx.reshape((m, k))


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
    from ._optional_numpy import require_numpy
    np = require_numpy('_expansion_ann')
    if metric not in ("sqeuclidean", "euclidean"):
        raise ValueError(f"mojolearn refine: metric must be 'sqeuclidean' or 'euclidean' "
                         f"(inner product is not implemented), got {metric!r}")
    from . import _backend
    x, _ = as_f32_c(dataset, ndim=2, name="dataset")
    q, _ = as_f32_c(queries, ndim=2, name="queries")
    n, d = (int(s) for s in x.shape)  # glue: two shape integers of the input
    m = int(q.shape[0])
    if int(q.shape[1]) != d:
        raise ValueError(f"mojolearn refine: queries have {q.shape[1]} features, the dataset has {d}")
    c = np.asarray(candidates)
    if c.ndim != 2 or c.shape[0] != m or not np.issubdtype(c.dtype, np.integer):
        raise ValueError(f"mojolearn refine: candidates must be an integer array of shape ({m}, k0)")
    k0 = int(c.shape[1])
    k = _ann_int("refine", "k", k)
    native = _backend.binding("_mojolearn_x_ann", numeric_mode)
    # the kernel skips every id outside [0, n) (refine_core.refine_cell); an
    # id beyond int32 is refused here (argument check; lane pyglue-numeric:
    # numpy's clamp of every id)
    if c.size and (int(c.min()) < -(1 << 31) or int(c.max()) >= (1 << 31)):
        raise ValueError("mojolearn refine: a candidate index exceeds int32")
    c32 = np.ascontiguousarray(c, dtype=np.int32)
    dist = empty((m * k,), "<f4")
    idx = empty((m * k,), "<i4")
    root = metric == "euclidean"
    native.x_ann_refine(
        # dataset, queries, candidates, out_d, out_i
        [addr_ro(x, name="dataset"), addr_ro(q, name="queries"), addr_ro(c32, name="candidates"),
         addr(dist, name="distances"), addr(idx, name="indices")],
        # n, d, m, k0, k (+ root: the binding takes the euclidean roots)
        [n, d, m, k0, k] + ([1] if root else []),
    )
    return dist.reshape((m, k)), idx.reshape((m, k))
