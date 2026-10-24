# SPDX-License-Identifier: Apache-2.0
"""Frozen pre-migration Python paths of the approximate indexes' `fit` and `refine`
(lane native-neighbors-20261024, base b32103c00), used only as an independent test
oracle for the one-call native doors (bindings/neighbors_ann_native.mojo)."""
from mojolearn._buffer import addr, addr_ro, as_f32_c, empty
from mojolearn._ivf_impl import _int_param
from mojolearn._expansion_ann import _ann_int


def ivf_fit(self, X, y=None):
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


def pq_fit(self, X, y=None):
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


def sq_fit(self, X, y=None):
    self._release_resident()
    x, _ = as_f32_c(X, ndim=2, name="X")
    n, dim = (int(s) for s in x.shape)  # glue: two shape integers of the input
    n_lists = self._p("n_lists")
    if not 1 <= n_lists <= n:
        raise ValueError(f"mojolearn IVFSQIndex: n_lists must be in [1, {n}], got {n_lists}")
    train_per_list = 0          # 0: the binding's default training rows
    if self.max_train_points_per_cluster is not None:
        train_per_list = self._p("max_train_points_per_cluster")
        if train_per_list < 1:
            raise ValueError("mojolearn IVFSQIndex: max_train_points_per_cluster must be positive or None, "
                             f"got {train_per_list}")
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
        # n, dim, n_lists, kmeans_n_iters, seed, train_rows_per_list
        [n, dim, n_lists, self._p("kmeans_n_iters"), self._p("random_state"), train_per_list],
    )
    self.centers_ = centers.reshape((n_lists, dim))
    self.list_offsets_, self.list_indices_ = offsets, indices
    self.sq_vmin_, self.sq_delta_ = vmin, delta
    self.codes_ = codes.reshape((n, dim))
    self.n_features_in_, self.n_rows_, self.n_lists_ = dim, n, n_lists
    return self


def rabitq_fit(self, X, y=None):
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


def cagra_fit(self, X, y=None):
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


def refine(dataset, queries, candidates, k, numeric_mode=None, metric="sqeuclidean"):
    from mojolearn._optional_numpy import require_numpy
    np = require_numpy('_expansion_ann')
    if metric not in ("sqeuclidean", "euclidean"):
        raise ValueError(f"mojolearn refine: metric must be 'sqeuclidean' or 'euclidean' "
                         f"(inner product is not implemented), got {metric!r}")
    from mojolearn import _backend
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
