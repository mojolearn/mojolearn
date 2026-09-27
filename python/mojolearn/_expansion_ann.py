# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ANN LANE'S PUBLIC DOOR (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

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
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._mode import NumericModeMixin

__all__ = ["IVFPQIndex"]


def _ann_int(owner, name, v):
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"mojolearn {owner}: {name} must be an int, got {type(v).__name__}")
    return int(v)


class IVFPQIndex(NumericModeMixin):
    """IVF-PQ build, then search (reference: cuVS `ivf_pq`), IDENTICAL by
    construction: fixed-order codebook training and code sums, ties broken
    by index (x_ann/ivf_pq_core.mojo).

    Parameters
    ----------
    n_lists : int
    n_probes : int
        Required; `n_probes > n_lists` raises (no clamp).
    pq_dim : int, default 4
        Subspaces; `dim` is zero-padded up to a multiple of it.
    pq_bits : int, default 8
        2**pq_bits codes per subspace.
    n_neighbors : int, default 8
    kmeans_n_iters : int, default 20
        The coarse quantizer (IVF-Flat's k-means).
    pq_kmeans_n_iters : int, default 20
    random_state : int, default 0

    `search(queries)` returns `(distances, indices)`: approximate squared L2
    distances float32 `(m, k)` and int32 `(m, k)` original row ids; a query
    whose probed lists hold fewer than k rows gets `(inf, -1)` fill.
    `n_candidates_` `(m,)` is set on the instance.
    """

    _BINDING = "_mojolearn_x_ann"
    _NAME = "IVFPQIndex"

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

    def _p(self, name):
        return _ann_int(self._NAME, name, getattr(self, name))

    def fit(self, X, y=None):
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, dim = (int(s) for s in x.shape)
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
        codes = empty((n * pq_dim,), "<i4")
        self._bind().x_ann_ivf_pq_build(
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

    def search(self, queries):
        if not hasattr(self, "codes_"):
            raise ValueError("mojolearn IVFPQIndex: call fit before search")
        q, _ = as_f32_c(queries, ndim=2, name="queries")
        m, dim = (int(s) for s in q.shape)
        if dim != self.n_features_in_:
            raise ValueError(f"mojolearn IVFPQIndex: queries have {dim} features, the index has {self.n_features_in_}")
        k, n_probes = self._p("n_neighbors"), self._p("n_probes")
        n, pq_dim = self.n_rows_, self.pq_dim_
        dist = empty((m * k,), "<f4")
        idx = empty((m * k,), "<i4")
        cand = empty((m,), "<i4")
        self._bind().x_ann_ivf_pq_search(
            # centers, offsets, list_indices, codebooks, codes, queries, out_d, out_i, out_n
            [addr_ro(self.centers_.reshape((self.n_lists_ * dim,)), name="centers_"),
             addr_ro(self.list_offsets_, name="list_offsets_"), addr_ro(self.list_indices_, name="list_indices_"),
             addr_ro(self.codebooks_.reshape((pq_dim * (1 << self.pq_bits_) * self.pq_len_,)), name="codebooks_"),
             addr_ro(self.codes_.reshape((n * pq_dim,)), name="codes_"), addr_ro(q, name="queries"),
             addr(dist, name="distances"), addr(idx, name="indices"), addr(cand, name="n_candidates_")],
            # n, dim, n_lists, pq_dim, pq_bits, m, k, n_probes
            [n, dim, self.n_lists_, pq_dim, self.pq_bits_, m, k, n_probes],
        )
        self.n_candidates_ = cand
        return dist.reshape((m, k)), idx.reshape((m, k))

