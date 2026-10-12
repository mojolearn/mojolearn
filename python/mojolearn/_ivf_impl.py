# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`IVFIndex` (cuVS `ivf_flat`), EXPOSED 2026-09-14.

The Python half of `bindings/_mojolearn_ivf.mojo`, the door of
`ivf/estimator.mojo` (cuVS `ivf_flat`, CSR lists). The index is RESIDENT
in the binding from the first search on (`ivf/resident.mojo`, the closure
of DEVIATION 1804, lane/py-dn-ann 2026-09-28): later searches upload only
their queries. `pixi run check-ivf` reads ALL OK at IDENTICAL on the Apple
M4, an NVIDIA H100 and an AMD MI300X with one card (1e7c1702) on all three
(bench/results/ivf_embed_km_legs_2026-09-14/README.md). `IVFFlat` is kept
as an alias of the same class for the names already written down.

BUILD, THEN SEARCH (lane/inference-embedding-ivf-cholesky, 2026-09-15).
`fit(X)` builds the index on the GPU (`ivf_flat_build`) and keeps it as five
arrays: `centers_` (n_lists, dim), `center_norms_` (n_lists,), squared,
`list_offsets_` (n_lists + 1,) int32, `list_indices_` (n,) int32 (the
ORIGINAL row ids, ascending within each list) and `list_data_` (n, dim).
`search(queries)` answers from them (`ivf_flat_search`). The two calls are
`ivf_flat_build_host` and `ivf_flat_search_host`, the halves the one-card
`ivf_flat_build_and_search` entry runs, over host lists the upload copies,
so the answers are the bytes the one-call door returned. Before this split
the index did not cross and nothing could be saved.

`save(path)` writes the index (format `mojolearn-ivf-flat-1`); `load(path)`
reads it back, admitted by the binding's `ivf_validate_index_arrays` at the
first search. On a CPU-only install `search` on a loaded index is public
inference through `_mojolearn_ivf_search_host`, which carries no build; `fit`
there refuses outside the internal reference context.

`n_probes` has no default (policy 1) and `n_probes > n_lists` raises on the
Mojo host rather than clamping (policy 2). `metric='sqeuclidean'` is cuVS's
`L2Expanded` (squared distances) and `'euclidean'` its `L2SqrtExpanded`. On
one index the ids and their order are the same under both and the distances
differ by the root; a build under each metric trains the k-means quantizer
on a different reduction, so at `n_probes < n_lists` the answers can differ
(policy 4, corrected 2026-09-14). The metric is a property of the BUILT
index: `search` refuses by name when `metric` no longer names it.

`metric='euclidean'` (L2SqrtExpanded) WAS REFUSED AT THIS DOOR from its
exposure until the fix on 2026-09-14 (fix/ivf-l2sqrt). On the Apple M4 it
returned 0.0 for every distance and ids that were not the nearest rows. The
cause: the build and the search filled the row-norm launch's sqrt flag from
`metric_is_sqrt(metric)`, so every norm was rooted and the expanded distance
`||q|| + ||y|| - 2 q.y` clamped to zero. The norms are squared under both
metrics now, the coarse step and the candidates are scored and selected
squared, and the root is taken over the `k` selected distances
(`ivf/impl/neighbors/ivf_common.mojo::postprocess_distances`), which is
where cuVS takes it. `ivf/checks/ivf_check.mojo::check_l2_sqrt_is_the_root_of_l2`
searches the metric; the identity_break lane is `ivf-euclidean`.
"""
from . import _backend, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty, frombytes
from ._mode import NumericModeMixin

_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}

METRIC_L2_EXPANDED = 0
METRIC_L2_SQRT_EXPANDED = 1
_METRICS = {
    "sqeuclidean": METRIC_L2_EXPANDED,
    "l2_expanded": METRIC_L2_EXPANDED,
    "euclidean": METRIC_L2_SQRT_EXPANDED,
    "l2_sqrt_expanded": METRIC_L2_SQRT_EXPANDED,
    "l2": METRIC_L2_SQRT_EXPANDED,
}
_METRIC_CODES = (METRIC_L2_EXPANDED, METRIC_L2_SQRT_EXPANDED)
_METRIC_NAMES = {METRIC_L2_EXPANDED: "sqeuclidean", METRIC_L2_SQRT_EXPANDED: "euclidean"}

#: The saved-index format tag (`save`, `load`, `mojolearn.host_model`).
_IVF_FORMAT = "mojolearn-ivf-flat-1"

#: The index arrays, in the binding's address order, with dtype and trailing
#: shape (`None` stands for the row count or the list count).
_INDEX_ARRAYS = (
    ("centers_", "<f4"),
    ("center_norms_", "<f4"),
    ("list_offsets_", "<i4"),
    ("list_indices_", "<i4"),
    ("list_data_", "<f4"),
)


def _int_param(name, v):
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"mojolearn IVFIndex: {name} must be an int, got {type(v).__name__}")
    return int(v)


# n_probes has no default (policy 1); n_probes > n_lists raises on the Mojo
# host rather than clamping (policy 2); a build under each metric trains the
# k-means quantizer on a different reduction (policy 4). The euclidean root is
# taken over the k selected squared distances
# (ivf/impl/neighbors/ivf_common.mojo::postprocess_distances), where cuVS
# takes it; ivf/checks/ivf_check.mojo::check_l2_sqrt_is_the_root_of_l2.
class IVFIndex(NumericModeMixin):
    """Approximate nearest-neighbor index with inverted lists (IVF-Flat).

    Follows cuVS's ``ivf_flat``. `fit` trains a k-means coarse quantizer that
    splits the rows into `n_lists` lists and stores the rows uncompressed,
    grouped by list. `search` scans the `n_probes` lists nearest each query
    and returns exact distances to the rows in those lists, so recall depends
    only on `n_probes`. Results are reproducible. On a GPU the fitted index
    stays on the device between searches; each search uploads only its
    queries. `IVFFlat` is another name for this class.

    Parameters
    ----------
    n_lists : int
        Number of inverted lists (k-means clusters). Must be at least 1.
    n_probes : int
        Number of lists scanned per query. Required, with no default; a value
        greater than `n_lists` raises an error rather than being clamped.
    n_neighbors : int, default 8
        Number of neighbors `search` returns per query.
    kmeans_n_iters : int, default 20
        k-means iterations for the coarse quantizer.
    metric : {'sqeuclidean', 'euclidean'}, default 'sqeuclidean'
        'sqeuclidean' (cuVS ``L2Expanded``) returns squared Euclidean
        distances; 'euclidean' (cuVS ``L2SqrtExpanded``) returns their square
        roots. 'l2_expanded' is accepted for 'sqeuclidean', and
        'l2_sqrt_expanded' and 'l2' for 'euclidean'. The metric is fixed when
        the index is built: on one index the ids are the same under both, but
        the k-means quantizer trains differently under each, so two indexes
        built with different metrics can return different neighbors when
        ``n_probes < n_lists``.
    random_state : int, default 0
        Seed for the k-means initialization.
    numeric_mode : str or None, default None
        Numeric mode for this estimator; None uses the package default set
        with `mojolearn.set_numeric_mode`.

    Attributes
    ----------
    centers_ : ndarray of shape (n_lists, n_features), float32
        Coarse quantizer centers.
    center_norms_ : ndarray of shape (n_lists,), float32
        Squared norm of each center.
    list_offsets_ : ndarray of shape (n_lists + 1,), int32
        Start of each list in `list_indices_` and `list_data_`.
    list_indices_ : ndarray of shape (n_rows_,), int32
        Original row ids, grouped by list and ascending within each list.
    list_data_ : ndarray of shape (n_rows_, n_features), float32
        The indexed rows, in the order of `list_indices_`.
    n_features_in_ : int
        Number of features seen during `fit`.
    n_rows_ : int
        Number of indexed rows, including rows added by `extend`.
    n_lists_ : int
        The fitted value of `n_lists`.
    metric_code_ : int
        The metric the index was built under: 0 for 'sqeuclidean', 1 for
        'euclidean'.
    n_candidates_ : ndarray of shape (n_queries,), int32
        Set by `search`: how many indexed rows each query scanned.
    extend_labels_ : ndarray of shape (n_new,), int32
        Set by `extend`: the list each new row joined.

    Raises
    ------
    TypeError
        If an integer parameter is not an int.
    ValueError
        If `n_lists` is less than 1 or `metric` is unknown.

    See Also
    --------
    IVFPQIndex, IVFSQIndex, IVFRaBitQIndex : Compressed IVF indexes.
    CagraIndex : Graph-based index.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> index = mojolearn.IVFIndex(n_lists=16, n_probes=4, n_neighbors=5).fit(X)
    >>> distances, indices = index.search(X[:10])
    >>> indices.shape
    (10, 5)
    """

    _BINDING = "_mojolearn_ivf"

    def __init__(self, n_lists, n_probes, n_neighbors=8, kmeans_n_iters=20, metric="sqeuclidean", random_state=0):
        self.n_lists = n_lists
        self.n_probes = n_probes
        self.n_neighbors = n_neighbors
        self.kmeans_n_iters = kmeans_n_iters
        self.metric = metric
        self.random_state = random_state

    def _extension(self):
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "ivf_numeric_mode", None)
        if fn is not None and int(fn()) != _MODE_CODE.get(want):
            raise RuntimeError(
                f"mojolearn IVFIndex: numeric_mode={want!r} was requested but {mod.__name__} "
                "reports another compile-time mode; rebuild it with bash bindings/build_ivf.sh"
            )
        return mod

    def _metric_code(self):
        if isinstance(self.metric, str):
            if self.metric not in _METRICS:
                raise ValueError(f"mojolearn IVFIndex: metric must be one of {sorted(_METRICS)}, got {self.metric!r}")
            return _METRICS[self.metric]
        if isinstance(self.metric, bool) or self.metric not in _METRIC_CODES:
            raise ValueError(f"mojolearn IVFIndex: the metric codes are {METRIC_L2_EXPANDED} (L2Expanded) and {METRIC_L2_SQRT_EXPANDED} (L2SqrtExpanded), got {self.metric!r}")
        return int(self.metric)

    @staticmethod
    def _entry(mod, name):
        fn = getattr(mod, name, None)
        if fn is None:
            raise ImportError(
                f"mojolearn IVFIndex: {getattr(mod, '__name__', mod)} exports no {name}; "
                "this binary predates the build and search split, rebuild it with bash bindings/build_ivf.sh"
            )
        return fn

    def _is_built(self):
        """`fit` (or `load`) has run: the list vectors are in numpy or held
        on the device by the resident build. Glue: reads two dict keys and
        never makes the vectors."""
        d = self.__dict__
        return "list_data_" in d or "_device_list_data" in d

    def __getattr__(self, name):
        # `list_data_` of a RESIDENT build (lane gap-ivf, 2026-10-08) is made
        # on first read, outside fit and search: `save`, a pickle, `extend`,
        # `_clone` and the shard split read it through here.
        if name == "list_data_" and "_device_list_data" in self.__dict__:
            self._materialize_list_data()
            return self.__dict__["list_data_"]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _materialize_list_data(self):
        """Write the resident index's list vectors into a new numpy array
        (`ivf_flat_index_export`, one device copy in Mojo) and re-key the held
        handle to it: the handle keeps serving searches, since the array holds
        the words it holds."""
        d = self.__dict__
        if "_device_list_data" not in d:
            return
        cached = d.get("_resident")
        if cached is None:
            raise RuntimeError("mojolearn IVFIndex: the resident index was released before its list data was read")
        n, dim = self.n_rows_, self.n_features_in_
        data = empty((n * dim,), "<f4")
        cached[2].ivf_flat_index_export(cached[1], addr(data, name="list_data_"), n * dim)
        d["list_data_"] = data.reshape((n, dim))
        del d["_device_list_data"]
        d["_resident"] = (self._resident_key(False) + (id(cached[2]),), cached[1], cached[2])

    # On a GPU binding with `ivf_flat_build_resident` the index stays on the
    # device: the four small arrays are written here, the list vectors stay in
    # the binding's resident handle that `search` names, and `list_data_` is
    # made on first read. On a CPU-only install fit refuses outside the
    # internal reference context.
    def fit(self, X, y=None):
        """Build the index from `X`.

        Building needs a GPU install; on a CPU-only install, load a saved
        index instead.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to index, converted to float32.
        y : None
            Ignored.

        Returns
        -------
        self : IVFIndex
            The fitted index.
        """
        self.__dict__.pop("_device_list_data", None)
        self._release_resident()
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, dim = (int(s) for s in x.shape)  # glue: the two shape dims
        n_lists = _int_param("n_lists", self.n_lists)
        iters = _int_param("kmeans_n_iters", self.kmeans_n_iters)
        seed = _int_param("random_state", self.random_state)
        metric = self._metric_code()
        if n_lists < 1:
            raise ValueError(f"mojolearn IVFIndex: n_lists must be at least 1, got {n_lists}")
        centers = empty((n_lists * dim,), "<f4")
        norms = empty((n_lists,), "<f4")
        offsets = empty((n_lists + 1,), "<i4")
        indices = empty((n,), "<i4")
        native = self._extension()
        resident_build = getattr(native, "ivf_flat_build_resident", None)
        if resident_build is not None:
            handle = int(resident_build(
                # ORDER MATCHES bindings/_mojolearn_ivf.mojo (ivf_flat_build_resident).
                # x, centers_out, center_norms_out, offsets_out, indices_out
                [addr_ro(x, name="X"), addr(centers, name="centers_"), addr(norms, name="center_norms_"),
                 addr(offsets, name="list_offsets_"), addr(indices, name="list_indices_")],
                # n, dim, n_lists, kmeans_n_iters, metric, seed
                [n, dim, n_lists, iters, metric, seed],
            ))
            self.__dict__.pop("list_data_", None)
            self.centers_ = centers.reshape((n_lists, dim))
            self.center_norms_ = norms
            self.list_offsets_ = offsets
            self.list_indices_ = indices
            self.n_features_in_ = dim
            self.n_rows_ = n
            self.n_lists_ = n_lists
            self.metric_code_ = metric
            self._device_list_data = True
            self._resident = (self._resident_key(False) + (id(native),), handle, native)
            return self
        data = empty((n * dim,), "<f4")
        self._entry(native, "ivf_flat_build")(
            # ORDER MATCHES bindings/ivf_index_arrays.mojo (ivf_flat_build).
            # x, centers_out, center_norms_out, offsets_out, indices_out, list_data_out
            [addr_ro(x, name="X"), addr(centers, name="centers_"), addr(norms, name="center_norms_"),
             addr(offsets, name="list_offsets_"), addr(indices, name="list_indices_"),
             addr(data, name="list_data_")],
            # n, dim, n_lists, kmeans_n_iters, metric, seed
            [n, dim, n_lists, iters, metric, seed],
        )
        self.centers_ = centers.reshape((n_lists, dim))
        self.center_norms_ = norms
        self.list_offsets_ = offsets
        self.list_indices_ = indices
        self.list_data_ = data.reshape((n, dim))
        self.n_features_in_ = dim
        self.n_rows_ = n
        self.n_lists_ = n_lists
        self.metric_code_ = metric
        return self

    # -- the resident index (lane/py-dn-ann, 2026-09-28; ivf/resident.mojo,
    # the closure of DEVIATION 1804) --

    def _resident_key(self, partial):
        # a resident build's list vectors are on the device ("device" marks
        # them); reading the dict, not the attribute, so no export happens
        arrays = tuple(self.__dict__.get(name) for name, _dtype in _INDEX_ARRAYS)  # glue: the five index array names
        return (tuple(("device",) if a is None else (id(a), addr_ro(a, name=name), tuple(a.shape))
                      for a, (name, _dtype) in zip(arrays, _INDEX_ARRAYS)),  # glue: addresses of five index arrays
                self.n_rows_, self.n_features_in_, self.n_lists_, self.metric_code_, bool(partial))

    def _resident_handle(self, native, partial=False):
        """The handle of this index held by `native` (admitted, copied and,
        on a GPU binding, uploaded ONCE by `ivf_flat_index_prepare`),
        reused while the five index arrays are the same objects at the same
        addresses and shapes; None where the loaded binding has no such door
        (a binary built before it), in which case the search takes the
        per-call path. `neighbors.py::_resident_index_handle`'s rule."""
        try:
            prepare = native.ivf_flat_index_prepare
        except (ImportError, AttributeError):
            return None
        key = self._resident_key(partial) + (id(native),)
        cached = self.__dict__.get("_resident")
        if cached is not None and cached[0] == key:
            return cached[1]
        # another binding or a partial-storage search: the vectors come to
        # numpy first, then the handle is replaced by a prepared one
        self._materialize_list_data()
        self._release_resident()
        n, dim = self.n_rows_, self.n_features_in_
        handle = int(prepare(
            # ORDER MATCHES bindings/ivf_index_arrays.mojo (ivf_flat_index_prepare).
            [addr_ro(self.centers_.reshape((self.n_lists_ * dim,)), name="centers_"),
             addr_ro(self.center_norms_, name="center_norms_"),
             addr_ro(self.list_offsets_, name="list_offsets_"),
             addr_ro(self.list_indices_, name="list_indices_"),
             addr_ro(self.list_data_.reshape((n * dim,)), name="list_data_")],
            # n, dim, n_lists, metric, partial_storage
            [n, dim, self.n_lists_, self.metric_code_, 1 if partial else 0],
        ))
        self._resident = (key, handle, native)
        return handle

    def _release_resident(self):
        """Drop the held index, if any. Quiet on a binding that cannot be
        reached any more (interpreter shutdown) or a handle already gone."""
        # a resident build's vectors live only in the handle: export first
        self._materialize_list_data()
        cached = self.__dict__.pop("_resident", None)
        if cached is None:
            return
        try:
            cached[2].ivf_flat_index_release(cached[1])
        except Exception:  # noqa: BLE001
            pass

    def __del__(self):
        try:
            self.__dict__.pop("_device_list_data", None)
            self._release_resident()
        except Exception:  # noqa: BLE001
            pass

    def __getstate__(self):
        """A pickle or a deepcopy carries no handle: the integer means
        something only in the process and registry that minted it. A resident
        build's list vectors are exported first, so the copy carries them."""
        self._materialize_list_data()
        state = self.__dict__.copy()
        state.pop("_resident", None)
        return state

    @staticmethod
    def _search_filter(filter, n):
        if filter is None:
            return None
        from ._optional_numpy import require_numpy
        np = require_numpy('_ivf_impl')
        f = np.asarray(filter)
        if f.shape != (n,) or f.dtype != np.bool_:
            raise ValueError(f"mojolearn IVFIndex: filter must be a boolean array of shape ({n},), "
                             "one flag per indexed row")
        return np.ascontiguousarray(f.astype(np.int32))

    # filter: cuVS's sample filter, applied to the merged candidates of the
    # probed lists before any distance (DEVIATION 5863); a short query raises
    # (DEVIATION 1794).
    def search(self, queries, filter=None):
        """Find the nearest indexed rows of each query within the probed lists.

        Parameters
        ----------
        queries : array-like of shape (n_queries, n_features)
            Query rows, converted to float32.
        filter : array-like of bool, shape (n_rows_,), optional
            One flag per indexed row, by original row id; a row flagged False
            is never scored, returned or counted in `n_candidates_` (cuVS's
            sample filter, applied before any distance is computed).

        Returns
        -------
        distances : ndarray of shape (n_queries, n_neighbors), float32
            Distances under `metric`, ascending.
        indices : ndarray of shape (n_queries, n_neighbors), int32
            Original row ids of the neighbors.

        Raises
        ------
        ValueError
            If the index is not fitted or loaded, the queries have a different
            number of features, `metric` no longer names the metric the index
            was built under, or `filter` has the wrong shape or dtype.

        Notes
        -----
        Unlike the compressed IVF indexes, a query whose probed lists hold
        (after filtering) fewer than `n_neighbors` rows raises an error rather
        than being padded. Raise `n_probes` if this happens.
        """
        if not self._is_built():
            raise ValueError("mojolearn IVFIndex: call fit (or load) before search")
        q, _ = as_f32_c(queries, ndim=2, name="queries")
        m, dim = q.shape
        if dim != self.n_features_in_:
            raise ValueError(f"mojolearn IVFIndex: queries have {dim} features, the index has {self.n_features_in_}")
        for name in ("n_lists", "n_probes", "n_neighbors", "kmeans_n_iters", "random_state"):  # glue: checks five parameter names
            _int_param(name, getattr(self, name))
        if self._metric_code() != self.metric_code_:
            raise ValueError(
                f"mojolearn IVFIndex: metric {self.metric!r} does not name the metric this index was "
                f"built under ({_METRIC_NAMES[self.metric_code_]!r}); a built index has one metric"
            )
        n, k = self.n_rows_, int(self.n_neighbors)
        dist = empty((m * k,), "<f4")
        idx = empty((m * k,), "<i4")
        cand = empty((m,), "<i4")
        native = self._extension()
        handle = self._resident_handle(native)
        if handle is not None:
            addrs = [addr_ro(q, name="queries"), addr(dist, name="distances"), addr(idx, name="indices"),
                     addr(cand, name="n_candidates_")]
            keep = self._search_filter(filter, n)   # held: the binding reads it below
            if keep is not None:
                addrs.append(addr_ro(keep, name="filter"))
            native.ivf_flat_index_search(
                # ORDER MATCHES bindings/ivf_index_arrays.mojo (ivf_flat_index_search).
                handle, addrs,
                # n, dim, n_lists, metric, m, k, n_probes, partial_storage
                [n, dim, self.n_lists_, self.metric_code_, m, k, int(self.n_probes), 0],
            )
            self.n_candidates_ = cand
            return dist.reshape((m, k)), idx.reshape((m, k))
        centers = self.centers_.reshape((self.n_lists_ * dim,))
        data = self.list_data_.reshape((n * dim,))
        addrs = [addr_ro(centers, name="centers_"), addr_ro(self.center_norms_, name="center_norms_"),
                 addr_ro(self.list_offsets_, name="list_offsets_"), addr_ro(self.list_indices_, name="list_indices_"),
                 addr_ro(data, name="list_data_"), addr_ro(q, name="queries"),
                 addr(dist, name="distances"), addr(idx, name="indices"), addr(cand, name="n_candidates_")]
        keep = self._search_filter(filter, n)   # held: the binding reads it below
        if keep is not None:
            addrs.append(addr_ro(keep, name="filter"))
        self._entry(native, "ivf_flat_search")(
            # ORDER MATCHES bindings/ivf_index_arrays.mojo (ivf_flat_search).
            # centers, center_norms, offsets, indices, list_data, queries, dist_out, idx_out, cand_out[, filter]
            addrs,
            # n, dim, n_lists, metric, m, k, n_probes
            [n, dim, self.n_lists_, self.metric_code_, m, k, int(self.n_probes)],
        )
        self.n_candidates_ = cand
        return dist.reshape((m, k)), idx.reshape((m, k))

    # -- extend (lane/inference-embedding-ivf-cholesky stage 2, 2026-09-15) --

    # Reference: cuVS `ivf_flat::extend` with `adaptive_centers = false`
    # (`ivf_flat_build.cuh:180-345`). Caller-supplied ids are not a parameter
    # because an arbitrary id would need a merge to keep every list ascending
    # in its carried ids (DEVIATION 1783). On a CPU-only install
    # `_mojolearn_ivf_search_host` carries it.
    def extend(self, X):
        """Add the rows of `X` to the built index.

        Follows cuVS's ``ivf_flat::extend`` with ``adaptive_centers=False``:
        each new row joins the list of its nearest center (an exact tie goes
        to the lower list id), and the centers do not move. The new rows take
        the ids ``n_rows_, n_rows_ + 1, ...`` in the order given; unlike cuVS,
        custom ids are not accepted. Extending by a set of rows in one call
        or in several calls, in the same order, gives the same index. Works
        on a CPU-only install, since it trains nothing.

        Parameters
        ----------
        X : array-like of shape (n_new, n_features)
            Rows to add, converted to float32.

        Returns
        -------
        self : IVFIndex
            The extended index. `extend_labels_` holds the list each new row
            joined.

        Raises
        ------
        ValueError
            If the index is not fitted or loaded, `X` has a different number
            of features, or `metric` no longer names the metric the index was
            built under.
        """
        if not self._is_built():
            raise ValueError("mojolearn IVFIndex: call fit (or load) before extend")
        x, _ = as_f32_c(X, ndim=2, name="X")
        m, dim = (int(v) for v in x.shape)  # glue: the two shape dims
        if dim != self.n_features_in_:
            raise ValueError(f"mojolearn IVFIndex: X has {dim} features, the index has {self.n_features_in_}")
        if self._metric_code() != self.metric_code_:
            raise ValueError(
                f"mojolearn IVFIndex: metric {self.metric!r} does not name the metric this index was "
                f"built under ({_METRIC_NAMES[self.metric_code_]!r}); a built index has one metric"
            )
        self._materialize_list_data()   # the extend reads the old vectors
        self._release_resident()
        n, n_lists = self.n_rows_, self.n_lists_
        total = n + m
        offsets = empty((n_lists + 1,), "<i4")
        indices = empty((total,), "<i4")
        data = empty((total * dim,), "<f4")
        labels = empty((m,), "<i4")
        centers = self.centers_.reshape((n_lists * dim,))
        old_data = self.list_data_.reshape((n * dim,))
        self._entry(self._extension(), "ivf_flat_extend")(
            # ORDER MATCHES bindings/ivf_index_arrays.mojo (ivf_flat_extend).
            # centers, center_norms, offsets, indices, list_data, new_x, offsets_out, indices_out, list_data_out, labels_out
            [addr_ro(centers, name="centers_"), addr_ro(self.center_norms_, name="center_norms_"),
             addr_ro(self.list_offsets_, name="list_offsets_"), addr_ro(self.list_indices_, name="list_indices_"),
             addr_ro(old_data, name="list_data_"), addr_ro(x, name="X"),
             addr(offsets, name="list_offsets_"), addr(indices, name="list_indices_"),
             addr(data, name="list_data_"), addr(labels, name="extend_labels_")],
            # n, dim, n_lists, metric, n_new
            [n, dim, n_lists, self.metric_code_, m],
        )
        self.list_offsets_ = offsets
        self.list_indices_ = indices
        self.list_data_ = data.reshape((total, dim))
        self.n_rows_ = total
        self.extend_labels_ = labels
        return self

    def _clone(self):
        """A new instance holding copies of this index's arrays (cuVS
        `ivf_flat::clone`'s role), so an extend on it leaves this one as it
        was."""
        if not self._is_built():
            raise ValueError("mojolearn IVFIndex: call fit (or load) before _clone")
        c = type(self)(n_lists=self.n_lists, n_probes=self.n_probes, n_neighbors=self.n_neighbors,
                       kmeans_n_iters=self.kmeans_n_iters, metric=self.metric, random_state=self.random_state)
        c.numeric_mode = getattr(self, "numeric_mode", None)
        for name, dtype in _INDEX_ARRAYS:  # glue: the five index array names
            a = getattr(self, name)
            setattr(c, name, frombytes(a.tobytes(), dtype, tuple(a.shape)))
        c.n_features_in_, c.n_rows_, c.n_lists_, c.metric_code_ = (
            self.n_features_in_, self.n_rows_, self.n_lists_, self.metric_code_)
        return c

    # -- saved indexes (lane/inference-embedding-ivf-cholesky, 2026-09-15) --

    # npz layout: the five index arrays as `fit` left them, `meta` `<i8`
    # [n_rows, dim, n_lists, metric, n_probes, n_neighbors, kmeans_n_iters,
    # random_state] and the tier; format tag `mojolearn-ivf-flat-1`.
    def save(self, path):
        """Write the built index to `path` as an ``.npz`` file.

        The file holds the index arrays, the parameters and the numeric mode.
        Reload it with `load`; a loaded index searches without rebuilding,
        including on a CPU-only install.

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
        if not self._is_built():
            raise RuntimeError("mojolearn IVFIndex: call fit before save")
        from .decomposition import _saved_mode
        arrays = {
            "format": _IVF_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "meta": Array.from_list(
                [int(self.n_rows_), int(self.n_features_in_), int(self.n_lists_), int(self.metric_code_),
                 _int_param("n_probes", self.n_probes), _int_param("n_neighbors", self.n_neighbors),
                 _int_param("kmeans_n_iters", self.kmeans_n_iters),
                 _int_param("random_state", self.random_state)],
                "<i8",
            ),
        }
        for name, _dtype in _INDEX_ARRAYS:  # glue: the five index array names
            arrays[name.rstrip("_")] = getattr(self, name)
        return _serialize.write_npz(path, arrays)

    # The layout itself is admitted by the binding
    # (`ivf_validate_index_arrays`) at the first search.
    @classmethod
    def load(cls, path):
        """Load an index written by `save`.

        Every array's dtype and shape is checked on load; the list layout is
        checked at the first search.

        Parameters
        ----------
        path : str or path-like
            File written by `save`.

        Returns
        -------
        index : IVFIndex
            A fitted index, ready to `search` or `extend`.

        Raises
        ------
        ValueError
            If the file was saved by another class or its contents do not
            match the IVF-Flat layout.
        """
        from .decomposition import _check_saved_by, _restore_mode
        arrays = _serialize.read_npz(path, _IVF_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 8:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 8 are needed")
        n, dim, n_lists, metric, n_probes, k, iters, seed = (int(meta[i]) for i in range(8))  # glue: unpacks the fixed meta vector
        if metric not in _METRIC_CODES:
            raise ValueError(f"mojolearn: {path!r} records metric code {metric}")
        shapes = {"centers": (n_lists, dim), "center_norms": (n_lists,), "list_offsets": (n_lists + 1,),
                  "list_indices": (n,), "list_data": (n, dim)}
        obj = cls(n_lists=n_lists, n_probes=n_probes, n_neighbors=k, kmeans_n_iters=iters,
                  metric=_METRIC_NAMES[metric], random_state=seed)
        _restore_mode(obj, arrays)
        for name, dtype in _INDEX_ARRAYS:  # glue: the five index array names
            key = name.rstrip("_")
            value = _serialize.exact(arrays, key, dtype)
            if tuple(value.shape) != shapes[key]:
                raise ValueError(f"mojolearn: {path!r} {key} has shape {tuple(value.shape)}, not {shapes[key]}")
            setattr(obj, name, value)
        obj.n_features_in_ = dim
        obj.n_rows_ = n
        obj.n_lists_ = n_lists
        obj.metric_code_ = metric
        return obj


IVFFlat = IVFIndex

__all__ = ["IVFIndex", "IVFFlat"]
