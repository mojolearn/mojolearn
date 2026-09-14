# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`IVFFlat`, PREPARED AND NOT EXPOSED (workstream D, 2026-09-14).

The Python half of `bindings/_mojolearn_ivf.mojo`, the door of
`ivf/estimator.mojo::ivf_flat_build_and_search_host` (cuVS `ivf_flat`,
CSR lists, host-resident index, DEVIATION 1804). The lane has Apple
evidence only and the plan exposes it after an NVIDIA and an AMD leg; until
then this class is not in `mojolearn.__all__`, `_mojolearn_ivf` is not in
`_backend._MODULES`, and `_extension()` below raises by name saying so.
The binding's header lists the exposure step.

ONE CALL, ONE CARD. `search(queries)` builds the index and searches it in
one device call under one identity card (policy 3); the index does not
cross, so the harness's model column is `n/a:no-save`. `n_probes` has no
default (policy 1) and `n_probes > n_lists` raises on the Mojo host rather
than clamping (policy 2). `metric='sqeuclidean'` is cuVS's `L2Expanded`
(squared distances) and `'euclidean'` its `L2SqrtExpanded`; the set and
the order are the same either way (policy 4).
"""
from . import _backend
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._mode import NumericModeMixin

_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}

METRIC_L2_EXPANDED = 0
METRIC_L2_SQRT_EXPANDED = 1
_METRICS = {"sqeuclidean": METRIC_L2_EXPANDED, "l2_expanded": METRIC_L2_EXPANDED,
            "euclidean": METRIC_L2_SQRT_EXPANDED, "l2_sqrt_expanded": METRIC_L2_SQRT_EXPANDED}

NOT_EXPOSED = (
    "mojolearn IVFFlat is prepared and not exposed: ivf/ has Apple evidence "
    "only, and the plan (docs/lanes/TEMP_claim_surface_plan_2026-09-14.md "
    "section 4, item 4) exposes it after an NVIDIA and an AMD leg read "
    "IDENTICAL. bindings/_mojolearn_ivf.mojo's header lists the exposure step."
)


class IVFFlat(NumericModeMixin):
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
        if "_mojolearn_ivf" not in _backend._MODULES:
            raise ImportError(NOT_EXPOSED)
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "ivf_numeric_mode", None)
        if fn is not None and int(fn()) != _MODE_CODE.get(want):
            raise RuntimeError(
                f"mojolearn IVFFlat: numeric_mode={want!r} was requested but {mod.__name__} "
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
            raise ValueError("mojolearn IVFFlat: call fit before search")
        q, _ = as_f32_c(queries, ndim=2, name="queries")
        m, dim = q.shape
        if dim != self.n_features_in_:
            raise ValueError(f"mojolearn IVFFlat: queries have {dim} features, the fit had {self.n_features_in_}")
        for name in ("n_lists", "n_probes", "n_neighbors", "kmeans_n_iters", "random_state"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"mojolearn IVFFlat: {name} must be an int, got {type(v).__name__}")
        if isinstance(self.metric, str):
            if self.metric not in _METRICS:
                raise ValueError(f"mojolearn IVFFlat: metric must be one of {sorted(_METRICS)}, got {self.metric!r}")
            metric = _METRICS[self.metric]
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


__all__ = ["IVFFlat"]
