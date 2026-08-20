"""Brute-force exact k-nearest-neighbours on the GPU."""

import numpy as np

from . import _mojolearn
from ._arrays import _addr, _addr_ro, as_f32_c

_DEFAULT_QUERY_TILE = 256


class NearestNeighbors:
    """Exact k-NN by brute force, mirroring cuVS's fused L2 kernel.

    EXACT, not approximate. There is no index to build and no recall to trade
    away: every query is compared against every index point. That is a
    deliberate scope choice rather than a missing feature -- an approximate
    index is a different algorithm with a different contract, and this library
    does not ship one.

    Parameters
    ----------
    n_neighbors : int, default 5
        scikit-learn's default.
    query_tile : int, default 256
        Queries processed per pass. **This is the value every published
        mojolearn k-NN number was measured at.** It may be lowered
        automatically when the workspace it implies would be too large; after
        `kneighbors` runs, `used_query_tile_` reports what actually ran, so a
        benchmark can record the configuration instead of assuming it.

    Attributes
    ----------
    used_query_tile_ : int
        Set by `kneighbors`. Differs from `query_tile` when the workspace cap
        fired, which means the run was not the measured configuration.
    """

    def __init__(self, n_neighbors=5, query_tile=_DEFAULT_QUERY_TILE):
        self.n_neighbors = n_neighbors
        self.query_tile = query_tile
        self._index = None
        self.used_query_tile_ = None

    def fit(self, X, y=None):
        """Store the index. There is no index structure to build.

        `y` is accepted and ignored, for scikit-learn call-shape
        compatibility.
        """
        idx, _ = as_f32_c(X, "X")
        # Held on the instance so the memory outlives this call: the Mojo side
        # borrows the address at `kneighbors` time and owns nothing.
        self._index = idx
        self.n_samples_fit_ = idx.shape[0]
        self.n_features_in_ = idx.shape[1]
        return self

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Distances and indices of the nearest neighbours, nearest first.

        Returns `(distances, indices)` when `return_distance`, else
        `indices`, matching scikit-learn's layout and its ordering.

        Distances are Euclidean, not squared. The kernel computes squared
        distances and the square root is taken on the way out, over
        `n_queries * k` values rather than `n_queries * n_index`, so it is not
        on the hot path.
        """
        if self._index is None:
            raise ValueError("mojolearn: call fit before kneighbors")
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        q, _ = as_f32_c(X, "X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn: X has {q.shape[1]} features, index has "
                f"{self.n_features_in_}"
            )
        if k < 1 or k > self.n_samples_fit_:
            raise ValueError(
                f"mojolearn: n_neighbors must be in [1, {self.n_samples_fit_}]"
                f", got {k}"
            )

        nq = q.shape[0]
        dist = np.empty((nq, k), dtype=np.float32)
        ind = np.empty((nq, k), dtype=np.uint32)

        # Every array named here stays in a local for the whole call. That is
        # the contract `_arrays` documents and the reason it is spelled out.
        idx = self._index
        self.used_query_tile_ = _mojolearn.knn_search(
            _addr_ro(idx),
            _addr_ro(q),
            _addr(dist),
            _addr(ind),
            # ORDER MATCHES bindings/_mojolearn.mojo::knn_search_binding.
            # n_index, n_queries, n_features, k, return_sqrt, query_tile
            [idx.shape[0], nq, idx.shape[1], k, 1, self.query_tile],
        )

        if return_distance:
            return dist, ind.astype(np.int64)
        return ind.astype(np.int64)
