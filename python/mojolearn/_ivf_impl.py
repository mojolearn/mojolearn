# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`IVFIndex` (cuVS `ivf_flat`), EXPOSED 2026-09-14.

The Python half of `bindings/_mojolearn_ivf.mojo`, the door of
`ivf/estimator.mojo` (cuVS `ivf_flat`, CSR lists, host-resident index,
DEVIATION 1804). `pixi run check-ivf` reads ALL OK at IDENTICAL on the Apple
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


class IVFIndex(NumericModeMixin):
    """IVF-Flat build, then search (reference: cuVS `ivf_flat`).

    Parameters
    ----------
    n_lists : int
    n_probes : int
        Required; no default at this boundary (policy 1).
    n_neighbors : int, default 8
    kmeans_n_iters : int, default 20
    metric : {'sqeuclidean', 'euclidean'}, default 'sqeuclidean'
    random_state : int, default 0

    `fit(X)` builds the index; `search(queries)` returns `(distances,
    indices)` as float32 `(m, k)` and int32 `(m, k)`, with `n_candidates_`
    `(m,)` int32 set on the instance (how much of the index each query
    looked at).
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

    def fit(self, X, y=None):
        """Build the index. Returns `self`."""
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, dim = (int(s) for s in x.shape)
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
        data = empty((n * dim,), "<f4")
        self._entry(self._extension(), "ivf_flat_build")(
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

    def search(self, queries):
        if not hasattr(self, "list_data_"):
            raise ValueError("mojolearn IVFIndex: call fit (or load) before search")
        q, _ = as_f32_c(queries, ndim=2, name="queries")
        m, dim = q.shape
        if dim != self.n_features_in_:
            raise ValueError(f"mojolearn IVFIndex: queries have {dim} features, the index has {self.n_features_in_}")
        for name in ("n_lists", "n_probes", "n_neighbors", "kmeans_n_iters", "random_state"):
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
        centers = self.centers_.reshape((self.n_lists_ * dim,))
        data = self.list_data_.reshape((n * dim,))
        self._entry(self._extension(), "ivf_flat_search")(
            # ORDER MATCHES bindings/ivf_index_arrays.mojo (ivf_flat_search).
            # centers, center_norms, offsets, indices, list_data, queries, dist_out, idx_out, cand_out
            [addr_ro(centers, name="centers_"), addr_ro(self.center_norms_, name="center_norms_"),
             addr_ro(self.list_offsets_, name="list_offsets_"), addr_ro(self.list_indices_, name="list_indices_"),
             addr_ro(data, name="list_data_"), addr_ro(q, name="queries"),
             addr(dist, name="distances"), addr(idx, name="indices"), addr(cand, name="n_candidates_")],
            # n, dim, n_lists, metric, m, k, n_probes
            [n, dim, self.n_lists_, self.metric_code_, m, k, int(self.n_probes)],
        )
        self.n_candidates_ = cand
        return dist.reshape((m, k)), idx.reshape((m, k))

    # -- extend (lane/inference-embedding-ivf-cholesky stage 2, 2026-09-15) --

    def extend(self, X):
        """Add the rows of `X` to the built index. Returns `self`.

        Reference: cuVS `ivf_flat::extend` with `adaptive_centers = false`
        (`ivf_flat_build.cuh:180-345`). Each new row is assigned to the FIXED
        centres by the build's own assignment, with the build's tie rule (the
        lower list id wins an exact tie), and appended to its list. The new rows
        take the ids `n_rows_, n_rows_ + 1, ...` in the order given; cuVS takes
        caller-supplied ids, which are not a parameter here, because an arbitrary
        id would need a merge to keep every list ascending in its carried ids
        (DEVIATION 1783). So extending by a set of rows in one call, or in
        several calls over the same rows in the same order, gives the same index
        bytes, and a search afterwards is the same on every column. The centres
        and their norms do not move. `extend_labels_` (int32, one per new row)
        holds the list each new row went to.

        Public on a CPU-only install: assignment against saved centres trains
        nothing, and `_mojolearn_ivf_search_host` carries it."""
        if not hasattr(self, "list_data_"):
            raise ValueError("mojolearn IVFIndex: call fit (or load) before extend")
        x, _ = as_f32_c(X, ndim=2, name="X")
        m, dim = (int(v) for v in x.shape)
        if dim != self.n_features_in_:
            raise ValueError(f"mojolearn IVFIndex: X has {dim} features, the index has {self.n_features_in_}")
        if self._metric_code() != self.metric_code_:
            raise ValueError(
                f"mojolearn IVFIndex: metric {self.metric!r} does not name the metric this index was "
                f"built under ({_METRIC_NAMES[self.metric_code_]!r}); a built index has one metric"
            )
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
        if not hasattr(self, "list_data_"):
            raise ValueError("mojolearn IVFIndex: call fit (or load) before _clone")
        c = type(self)(n_lists=self.n_lists, n_probes=self.n_probes, n_neighbors=self.n_neighbors,
                       kmeans_n_iters=self.kmeans_n_iters, metric=self.metric, random_state=self.random_state)
        c.numeric_mode = getattr(self, "numeric_mode", None)
        for name, dtype in _INDEX_ARRAYS:
            a = getattr(self, name)
            setattr(c, name, frombytes(a.tobytes(), dtype, tuple(a.shape)))
        c.n_features_in_, c.n_rows_, c.n_lists_, c.metric_code_ = (
            self.n_features_in_, self.n_rows_, self.n_lists_, self.metric_code_)
        return c

    # -- saved indexes (lane/inference-embedding-ivf-cholesky, 2026-09-15) --

    def save(self, path):
        """Write the built index to `path` as an npz: the five index arrays
        as `fit` left them, `meta` `<i8` [n_rows, dim, n_lists, metric,
        n_probes, n_neighbors, kmeans_n_iters, random_state] and the tier.
        A loaded index searches; it does not rebuild."""
        if not hasattr(self, "list_data_"):
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
        for name, _dtype in _INDEX_ARRAYS:
            arrays[name.rstrip("_")] = getattr(self, name)
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load an index written by `save`. Every array's dtype and shape is
        checked here; the layout itself is admitted by the binding at the
        first search."""
        from .decomposition import _check_saved_by, _restore_mode
        arrays = _serialize.read_npz(path, _IVF_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 8:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 8 are needed")
        n, dim, n_lists, metric, n_probes, k, iters, seed = (int(meta[i]) for i in range(8))
        if metric not in _METRIC_CODES:
            raise ValueError(f"mojolearn: {path!r} records metric code {metric}")
        shapes = {"centers": (n_lists, dim), "center_norms": (n_lists,), "list_offsets": (n_lists + 1,),
                  "list_indices": (n,), "list_data": (n, dim)}
        obj = cls(n_lists=n_lists, n_probes=n_probes, n_neighbors=k, kmeans_n_iters=iters,
                  metric=_METRIC_NAMES[metric], random_state=seed)
        _restore_mode(obj, arrays)
        for name, dtype in _INDEX_ARRAYS:
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
