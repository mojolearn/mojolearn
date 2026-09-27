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

__all__ = ["IVFPQIndex", "TSNE", "CagraIndex", "IVFSQIndex", "IVFRaBitQIndex", "refine"]


def _ann_mask(owner, filter, n):
    """cuVS's sample filter as one int32 per row (1 keeps the row, 0 removes
    it); None keeps every row."""
    import numpy as np
    if filter is None:
        return np.ones(n, dtype=np.int32)
    f = np.asarray(filter)
    if f.shape != (n,) or f.dtype != np.bool_:
        raise ValueError(f"mojolearn {owner}: filter must be a boolean array of shape ({n},), one flag per indexed row")
    return np.ascontiguousarray(f.astype(np.int32))


def _ann_int(owner, name, v):
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"mojolearn {owner}: {name} must be an int, got {type(v).__name__}")
    return int(v)


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

    def save(self, path):
        from . import _serialize
        from ._array import Array
        from .decomposition import _saved_mode
        missing = [a for a, _ in self._SAVE_ARRAYS if not hasattr(self, a)]
        if missing:
            raise RuntimeError(f"mojolearn {type(self).__name__}: call fit before save")
        arrays = {
            "format": self._SAVE_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "params": Array.from_list([_ann_int(type(self).__name__, p, getattr(self, p)) for p in self._SAVE_PARAMS], "<i8"),
            "fitted": Array.from_list([int(getattr(self, a)) for a in self._SAVE_FITTED], "<i8"),
        }
        for attr, _dtype in self._SAVE_ARRAYS:
            arrays[attr.rstrip("_")] = getattr(self, attr)
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        from . import _serialize
        from .decomposition import _check_saved_by, _restore_mode
        arrays = _serialize.read_npz(path, cls._SAVE_FORMAT)
        _check_saved_by(arrays, path, cls)
        params = _serialize.exact(arrays, "params", "<i8")
        fitted = _serialize.exact(arrays, "fitted", "<i8")
        if params.size != len(cls._SAVE_PARAMS) or fitted.size != len(cls._SAVE_FITTED):
            raise ValueError(f"mojolearn: {path!r} does not hold {cls.__name__}'s parameters")
        obj = cls(**{p: int(params[i]) for i, p in enumerate(cls._SAVE_PARAMS)})
        _restore_mode(obj, arrays)
        for i, a in enumerate(cls._SAVE_FITTED):
            setattr(obj, a, int(fitted[i]))
        for attr, dtype in cls._SAVE_ARRAYS:
            setattr(obj, attr, _serialize.exact(arrays, attr.rstrip("_"), dtype))
        return obj


class IVFPQIndex(_AnnSaved, NumericModeMixin):
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

    def search(self, queries, filter=None):
        """`filter`: optional boolean array over the indexed rows; a False row
        is never returned (cuVS's sample filter, applied before scoring)."""
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
        mask = _ann_mask("IVFPQIndex", filter, n)  # held: the binding reads it after this line
        self._bind().x_ann_ivf_pq_search(
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


class TSNE(NumericModeMixin):
    """t-SNE (reference: scikit-learn `TSNE`, cuML `TSNE`), IDENTICAL by
    construction (x_ann/tsne_core.mojo): exact k-NN affinities with the
    reference's perplexity bisection, attractive term over the sparse P,
    EXACT repulsion (no Barnes-Hut atomics), sklearn's gains/momentum
    optimizer for exactly `max_iter` steps.

    Parameters
    ----------
    n_components : 2 (only)
    perplexity : float, default 30.0
    early_exaggeration : float, default 12.0
    learning_rate : float or 'auto', default 'auto'
        'auto' is sklearn's max(n / early_exaggeration / 4, 50).
    max_iter : int, default 1000
    init : 'pca' (default, as sklearn), 'random' or an array of shape
        (n_samples, 2).
        'pca' is mojolearn's PCA (identical on every column; the exact
        eigensolver where sklearn uses its randomized solver) scaled as
        sklearn scales it: divided by the first column's standard deviation
        and multiplied by 1e-4. The deviation is computed on the host in
        float64 with math.fsum (one correctly rounded sum, so no platform's
        summation order enters) and applied as one float32 divide and one
        float32 multiply per element.
        'random' is uniform(-5e-5, 5e-5) from numpy's default_rng(random_state),
        whose integer-to-double draw is exact on every platform.
    random_state : int, default 0
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
        """The start y0 (n, 2) float32: 'pca', 'random' or the caller's array."""
        import math
        import numpy as np
        init = self.init
        if isinstance(init, str) and init == "random":
            return ((np.random.default_rng(seed).random((n, 2)) - 0.5) * 1e-4).astype(np.float32)
        if isinstance(init, str) and init == "pca":
            from .decomposition import PCA
            pca = PCA(n_components=2)
            mode = getattr(self, "numeric_mode", None)
            if mode is not None:
                pca.numeric_mode = mode
            emb = np.ascontiguousarray(np.asarray(pca.fit_transform(x), dtype=np.float32))
            col = [float(v) for v in emb[:, 0]]
            mean = math.fsum(col) / n
            std = math.sqrt(math.fsum((v - mean) * (v - mean) for v in col) / n)
            if not std > 0.0:
                raise ValueError("mojolearn TSNE: init='pca' gave a constant first component; pass init='random'")
            return np.ascontiguousarray((emb / np.float32(std)) * np.float32(1e-4), dtype=np.float32)
        if isinstance(init, str):
            raise ValueError(f"mojolearn TSNE: init must be 'pca', 'random' or an array, got {init!r}")
        y0 = np.ascontiguousarray(np.asarray(init, dtype=np.float32))
        if y0.shape != (n, 2):
            raise ValueError(f"mojolearn TSNE: an init array must have shape ({n}, 2), got {y0.shape}")
        return y0

    def fit(self, X, y=None):
        import numpy as np
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, d = (int(s) for s in x.shape)
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
        y0 = self._init(x, n, seed)
        emb = empty((n * 2,), "<f4")
        kl = empty((1,), "<f4")
        self._bind().x_ann_tsne_fit(
            # x, y0, y_out, kl_out
            [addr_ro(x, name="X"), addr_ro(y0, name="init"), addr(emb, name="embedding_"), addr(kl, name="kl")],
            # n, d, max_iter, exploration_iters, perplexity, early_exaggeration, learning_rate
            [n, d, max_iter, exploration, perplexity, exag, lr],
        )
        self.embedding_ = emb.reshape((n, 2))
        self.kl_divergence_ = float(np.asarray(kl)[0])
        self.n_iter_ = max_iter
        self.learning_rate_ = lr
        self.n_features_in_ = d
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).embedding_


class CagraIndex(_AnnSaved, NumericModeMixin):
    """CAGRA graph index, build then search (reference: cuVS `cagra`),
    IDENTICAL by construction (x_ann/cagra_core.mojo): an exact k-NN
    intermediate graph, cuVS's rank-based detour pruning, reverse edges in
    (rank, source id) order, and a greedy itopk search with fixed seeds, an
    exact visited set and (distance, id) ties.

    Parameters
    ----------
    graph_degree : int, default 32
    intermediate_graph_degree : int, default 64
    n_neighbors : int, default 8
    itopk_size : int, default 64
    search_width : int, default 1
    max_iterations : int, default 0 (auto: itopk_size)
    n_seeds : int, default 0
        Evenly spaced start nodes; 0 is cuVS's pick-up count, itopk_size +
        search_width * graph_degree (their random seeds are refused: the
        seed set here is a function of n alone).

    `search(queries)` returns squared L2 distances float32 `(m, k)` and int32
    ids `(m, k)`; the index keeps a copy of the dataset, as cuVS's does.
    """

    _BINDING = "_mojolearn_x_ann"
    _SAVE_FORMAT = "mojolearn-cagra-1"
    _SAVE_PARAMS = ("graph_degree", "intermediate_graph_degree", "n_neighbors", "itopk_size", "search_width",
                    "max_iterations", "n_seeds")
    _SAVE_FITTED = ("n_features_in_", "n_rows_", "graph_degree_")
    _SAVE_ARRAYS = (("dataset_", "<f4"), ("graph_", "<i4"))

    def __init__(self, graph_degree=32, intermediate_graph_degree=64, n_neighbors=8, itopk_size=64,
                 search_width=1, max_iterations=0, n_seeds=0):
        self.graph_degree = graph_degree
        self.intermediate_graph_degree = intermediate_graph_degree
        self.n_neighbors = n_neighbors
        self.itopk_size = itopk_size
        self.search_width = search_width
        self.max_iterations = max_iterations
        self.n_seeds = n_seeds

    def _p(self, name):
        return _ann_int("CagraIndex", name, getattr(self, name))

    def fit(self, X, y=None):
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, d = (int(s) for s in x.shape)
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

    def search(self, queries, filter=None):
        """`filter`: optional boolean array over the indexed rows; a False row
        is never returned (cuVS's CAGRA sample filter, applied as cuVS's
        single-CTA search applies it: after the traversal, over the whole
        itopk buffer). A removed row still guides the walk. The answer is the
        first `n_neighbors` kept entries of the itopk buffer in its order
        (distance, then id); a query with fewer kept entries is padded with
        (+inf, -1), the IVF filters' padding. Integer selection only: every
        returned distance is a buffer entry the unfiltered search computes."""
        if not hasattr(self, "graph_"):
            raise ValueError("mojolearn CagraIndex: call fit before search")
        if filter is not None:
            return self._search_filtered(queries, filter)
        return self._search_k(queries, self._p("n_neighbors"))

    def _search_k(self, queries, k):
        """The traversal, returning the first `k` itopk entries."""
        q, _ = as_f32_c(queries, ndim=2, name="queries")
        m, d = (int(s) for s in q.shape)
        if d != self.n_features_in_:
            raise ValueError(f"mojolearn CagraIndex: queries have {d} features, the index has {self.n_features_in_}")
        L = self._p("itopk_size")
        max_iter = self._p("max_iterations") or L
        n_seeds = self._p("n_seeds") or (L + self._p("search_width") * self.graph_degree_)
        n_seeds = min(n_seeds, self.n_rows_)
        dist = empty((m * k,), "<f4")
        idx = empty((m * k,), "<i4")
        n, deg = self.n_rows_, self.graph_degree_
        self._bind().x_ann_cagra_search(
            # x, graph, queries, out_d, out_i
            [addr_ro(self.dataset_, name="dataset_"), addr_ro(self.graph_.reshape((n * deg,)), name="graph_"),
             addr_ro(q, name="queries"), addr(dist, name="distances"), addr(idx, name="indices")],
            # n, d, graph_degree, m, k, itopk_size, search_width, max_iterations, n_seeds
            [n, d, deg, m, k, L, self._p("search_width"), max_iter, n_seeds],
        )
        return dist.reshape((m, k)), idx.reshape((m, k))

    def _search_filtered(self, queries, filter):
        import numpy as np
        keep = _ann_mask("CagraIndex", filter, self.n_rows_) != 0
        k = self._p("n_neighbors")
        bd, bi = self._search_k(queries, self._p("itopk_size"))
        bd = np.asarray(bd)
        bi = np.asarray(bi)
        m = bd.shape[0]
        dist = np.full((m, k), np.inf, dtype=np.float32)
        idx = np.full((m, k), -1, dtype=np.int32)
        for q in range(m):
            o = 0
            for s in range(bd.shape[1]):
                if o == k:
                    break
                v = int(bi[q, s])
                if v >= 0 and keep[v]:
                    dist[q, o] = bd[q, s]
                    idx[q, o] = v
                    o += 1
        return dist, idx


class IVFSQIndex(_AnnSaved, NumericModeMixin):
    """IVF-SQ build, then search (reference: cuVS `ivf_sq`): the IVF-PQ
    coarse quantizer, residuals quantized to 8 bits per dimension (cuVS's
    per-dimension range with a 5% margin), a scan of the decoded residuals.
    IDENTICAL by construction (x_ann/ivf_sq_core.mojo).

    Parameters: n_lists, n_probes (required), n_neighbors=8,
    kmeans_n_iters=20, random_state=0. `search(queries, filter=None)` as
    `IVFPQIndex.search`."""

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

    def _p(self, name):
        return _ann_int("IVFSQIndex", name, getattr(self, name))

    def fit(self, X, y=None):
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, dim = (int(s) for s in x.shape)
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
        if not hasattr(self, "codes_"):
            raise ValueError("mojolearn IVFSQIndex: call fit before search")
        q, _ = as_f32_c(queries, ndim=2, name="queries")
        m, dim = (int(s) for s in q.shape)
        if dim != self.n_features_in_:
            raise ValueError(f"mojolearn IVFSQIndex: queries have {dim} features, the index has {self.n_features_in_}")
        k, n = self._p("n_neighbors"), self.n_rows_
        dist = empty((m * k,), "<f4")
        idx = empty((m * k,), "<i4")
        cand = empty((m,), "<i4")
        mask = _ann_mask("IVFSQIndex", filter, n)  # held: the binding reads it after this line
        self._bind().x_ann_ivf_sq_search(
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


class IVFRaBitQIndex(_AnnSaved, NumericModeMixin):
    """IVF-RaBitQ build, then search (reference: cuVS `ivf_rabitq`, RaBitQ
    of Gao & Long): the IVF coarse quantizer, each residual rotated by a
    randomized Hadamard transform and kept as sign bits plus its norm and
    <x_bar, o> factor; search returns the RaBitQ distance ESTIMATES, top-k
    under (estimate, id). IDENTICAL by construction
    (x_ann/ivf_rabitq_core.mojo). Pair with `refine` for exact distances.

    Parameters: n_lists, n_probes (required), n_neighbors=8,
    kmeans_n_iters=20, random_state=0 (the coarse init and the rotation's
    signs). `search(queries, filter=None)` as `IVFPQIndex.search`."""

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

    def _p(self, name):
        return _ann_int("IVFRaBitQIndex", name, getattr(self, name))

    def fit(self, X, y=None):
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, dim = (int(s) for s in x.shape)
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
        if not hasattr(self, "codes_"):
            raise ValueError("mojolearn IVFRaBitQIndex: call fit before search")
        q, _ = as_f32_c(queries, ndim=2, name="queries")
        m, dim = (int(s) for s in q.shape)
        if dim != self.n_features_in_:
            raise ValueError(f"mojolearn IVFRaBitQIndex: queries have {dim} features, the index has {self.n_features_in_}")
        k, n = self._p("n_neighbors"), self.n_rows_
        dist = empty((m * k,), "<f4")
        idx = empty((m * k,), "<i4")
        cand = empty((m,), "<i4")
        words = int(self.codes_.shape[1])
        mask = _ann_mask("IVFRaBitQIndex", filter, n)  # held: the binding reads it after this line
        self._bind().x_ann_ivf_rabitq_search(
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


def refine(dataset, queries, candidates, k, numeric_mode=None):
    """Exact re-ranking of candidate neighbors (reference: cuVS `refine`):
    squared L2 from each query to each of its candidate rows, the k smallest
    under (distance, id). `candidates` is int (m, k0); an id < 0 is padding.
    Returns `(distances, indices)`, float32 and int32 `(m, k)`; a query with
    fewer than k valid candidates gets `(inf, -1)` fill."""
    import numpy as np
    from . import _backend
    x, _ = as_f32_c(dataset, ndim=2, name="dataset")
    q, _ = as_f32_c(queries, ndim=2, name="queries")
    n, d = (int(s) for s in x.shape)
    m = int(q.shape[0])
    if int(q.shape[1]) != d:
        raise ValueError(f"mojolearn refine: queries have {q.shape[1]} features, the dataset has {d}")
    c = np.asarray(candidates)
    if c.ndim != 2 or c.shape[0] != m or not np.issubdtype(c.dtype, np.integer):
        raise ValueError(f"mojolearn refine: candidates must be an integer array of shape ({m}, k0)")
    k0 = int(c.shape[1])
    k = _ann_int("refine", "k", k)
    c32 = np.ascontiguousarray(np.where((c >= 0) & (c < n), c, -1).astype(np.int32))
    dist = empty((m * k,), "<f4")
    idx = empty((m * k,), "<i4")
    _backend.binding("_mojolearn_x_ann", numeric_mode).x_ann_refine(
        # dataset, queries, candidates, out_d, out_i
        [addr_ro(x, name="dataset"), addr_ro(q, name="queries"), addr_ro(c32, name="candidates"),
         addr(dist, name="distances"), addr(idx, name="indices")],
        # n, d, m, k0, k
        [n, d, m, k0, k],
    )
    return dist.reshape((m, k)), idx.reshape((m, k))
