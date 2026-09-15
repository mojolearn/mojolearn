# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`IVFIndex` (cuVS `ivf_flat`), EXPOSED 2026-09-14.

The Python half of `bindings/_mojolearn_ivf.mojo`, the door of
`ivf/estimator.mojo::ivf_flat_build_and_search_host` (cuVS `ivf_flat`,
CSR lists, host-resident index, DEVIATION 1804). `pixi run check-ivf`
reads ALL OK at IDENTICAL on the Apple M4, an NVIDIA H100 and an AMD MI300X
with one card (1e7c1702) on all three
(bench/results/ivf_embed_km_legs_2026-09-14/README.md). `IVFFlat` is kept
as an alias of the same class for the names already written down.

ONE CALL, ONE CARD. `search(queries)` builds the index and searches it in
one device call under one identity card (policy 3); the index does not
cross, so the harness's model column is `n/a:no-save`. `n_probes` has no
default (policy 1) and `n_probes > n_lists` raises on the Mojo host rather
than clamping (policy 2). `metric='sqeuclidean'` is cuVS's `L2Expanded`
(squared distances) and `'euclidean'` its `L2SqrtExpanded`. On one index
the ids and their order are the same under both and the distances differ by
the root; a build under each metric trains the k-means quantizer on a
different reduction, so at `n_probes < n_lists` the answers can differ
(policy 4, corrected 2026-09-14).

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
from . import _backend
from ._buffer import addr, addr_ro, as_f32_c, empty
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

class IVFIndex(NumericModeMixin):
    """cuVS `ivf_flat` build plus search in one call.

    Parameters
    ----------
    n_lists : int
    n_probes : int
        Required; no default at this boundary (policy 1).
    n_neighbors : int, default 8
    kmeans_n_iters : int, default 20
    metric : {'sqeuclidean', 'euclidean'}, default 'sqeuclidean'
    random_state : int, default 0

    `fit(X)` records the training rows; `search(queries)` builds and
    searches under one card and returns `(distances, indices)` as float32
    `(m, k)` and int32 `(m, k)`, with `n_candidates_` `(m,)` int32 set on
    the instance (how much of the index each query looked at).
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

    def fit(self, X, y=None):
        x, _ = as_f32_c(X, ndim=2, name="X")
        self.X_fit_ = x
        self.n_features_in_ = x.shape[1]
        return self

    def search(self, queries):
        if not hasattr(self, "X_fit_"):
            raise ValueError("mojolearn IVFIndex: call fit before search")
        q, _ = as_f32_c(queries, ndim=2, name="queries")
        m, dim = q.shape
        if dim != self.n_features_in_:
            raise ValueError(f"mojolearn IVFIndex: queries have {dim} features, the fit had {self.n_features_in_}")
        for name in ("n_lists", "n_probes", "n_neighbors", "kmeans_n_iters", "random_state"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"mojolearn IVFIndex: {name} must be an int, got {type(v).__name__}")
        if isinstance(self.metric, str):
            if self.metric not in _METRICS:
                raise ValueError(f"mojolearn IVFIndex: metric must be one of {sorted(_METRICS)}, got {self.metric!r}")
            metric = _METRICS[self.metric]
        elif isinstance(self.metric, bool) or self.metric not in _METRIC_CODES:
            raise ValueError(f"mojolearn IVFIndex: the metric codes are {METRIC_L2_EXPANDED} (L2Expanded) and {METRIC_L2_SQRT_EXPANDED} (L2SqrtExpanded), got {self.metric!r}")
        else:
            metric = int(self.metric)
        n, k = self.X_fit_.shape[0], int(self.n_neighbors)
        dist = empty((m * k,), "<f4")
        idx = empty((m * k,), "<i4")
        cand = empty((m,), "<i4")
        self._extension().ivf_flat_build_and_search(
            # ORDER MATCHES bindings/_mojolearn_ivf.mojo::ivf_flat_build_and_search_binding.
            # x, queries, dist_out, idx_out, cand_out
            [addr_ro(self.X_fit_, name="X_fit_"), addr_ro(q, name="queries"), addr(dist, name="distances"), addr(idx, name="indices"), addr(cand, name="n_candidates_")],
            # n, dim, m, k, n_lists, n_probes, kmeans_n_iters, metric, seed
            [n, dim, m, k, int(self.n_lists), int(self.n_probes), int(self.kmeans_n_iters), metric, int(self.random_state)],
        )
        self.n_candidates_ = cand
        return dist.reshape((m, k)), idx.reshape((m, k))


IVFFlat = IVFIndex

__all__ = ["IVFIndex", "IVFFlat"]
