# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""cuVS `cpp/src/preprocessing/spectral/detail/spectral_embedding.cuh`
(v26.08.00, the file cuML 26.08's `ML::SpectralEmbedding::transform` reaches
through `cuvs::preprocessing::spectral_embedding::transform`), function for
function: `create_laplacian` (`:31-52`), `compute_eigenpairs` (`:54-116`),
`transform` on a COO (`:118-131`), `create_connectivity_graph` (`:133-205`)
and `transform` on a dataset (`:207-223`).

THE SEQUENCE, AND WHERE EACH PIECE LIVES
  dataset -> kNN (k = n_neighbors, L2SqrtExpanded, the point itself
             included)                        neighbors/estimator.mojo::knn_search
          -> COO (row i, col = neighbor, 1.0f)
          -> coo_symmetrize with 0.5f*(a+b)   ported/sparse/linalg/detail/symmetrize.mojo
          -> coo_sort, coo_remove_scalar(0)   ported/sparse/op/coo_ops.mojo
          -> laplacian_normalized (or the
             unnormalized one), then NEGATED  ported/sparse/linalg/detail/laplacian.mojo
          -> Lanczos, LA on -L, k pairs       ported/sparse/solver/detail/lanczos.mojo
          -> eigenvectors /= diagonal (norm)  here, `divide_rows_kernel`
          -> columns gathered in REVERSE,
             dropping the last when drop_first here, host gather

WHY `LA` ON THE NEGATED LAPLACIAN: the Laplacian's eigenvalues are in
`[0, 2]` and the embedding wants the SMALLEST; `create_laplacian` negates
every value (`:45-49`) so the largest algebraic eigenpairs of `-L` are the
smallest of `L`. The Lanczos returns them ASCENDING, so the LAST column is
the trivial (eigenvalue `~0`) vector, and the reversed gather (`:89-115`)
puts the smallest first and, with `drop_first`, leaves the trivial one out.
cuML's Python asks for `n_components + 1` when `drop_first` is set
(`spectral_embedding.pyx:294`); this function receives THAT number in
`params.n_components`, exactly as theirs does.

THE kNN SELF-NEIGHBOR: `all_neighbors::build` searches the dataset against
itself, so each row's first neighbor is the row itself at distance 0 and
the COO carries `(i, i, 1.0)`; `compute_graph_laplacian`'s degree then
counts it and the diagonal subtracts it back (see laplacian.mojo). Ours
reaches the same through `knn_search(index = queries = dataset)`. Ties in
the k-th distance are a hazard in theirs and in ours alike (which point is
the k-th neighbor is then a selection tie-break); the fixtures are hashed so
no two distances coincide.

THE OUTPUT LAYOUT: theirs writes `embedding` COLUMN-MAJOR `n x n_out`; ours
returns ROW-MAJOR `n x n_out` (`embedding[p * n_out + c]`). A layout, not
an arithmetic; `cluster/detail/spectral.mojo` wanted row-major anyway (it
transposes theirs to get it, `spectral.cuh:47-52`).
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import isfinite
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from mojo_only.numerics import ftz
from neighbors.estimator import knn_search
from spectral.mojo_only.device_io import download_f32, upload_f32
from spectral.ported.sparse.coo import CooGraph
from spectral.ported.sparse.linalg.detail.laplacian import (
    DeviceCoo,
    LAPLACIAN_TPB,
    compute_graph_laplacian,
    laplacian_normalized,
)
from spectral.ported.sparse.linalg.detail.symmetrize import coo_symmetrize
from spectral.ported.sparse.op.coo_ops import coo_remove_scalar, coo_sort
from spectral.ported.sparse.solver.detail.lanczos import (
    LANCZOS_TPB,
    SAB_MAXITER,
    SAB_NCV,
    lanczos_compute_eigenpairs,
)
from spectral.ported.sparse.solver.lanczos_types import (
    LANCZOS_LA,
    LanczosSolverConfig,
)


@fieldwise_init
struct SpectralEmbeddingParams(Copyable, Movable):
    """`cuvs::preprocessing::spectral_embedding::params`
    (`cuvs/preprocessing/spectral_embedding.hpp:28-69` of v26.08.00, VERIFIED
    against `~/CascadeProjects/upstream/cuvs-v26.08.00`): `n_components`,
    `n_neighbors`, `norm_laplacian`, `drop_first`, `tolerance{1e-5f}` (`:59`),
    `std::optional<uint64_t> seed = std::nullopt` (`:68`, as `has_seed` +
    `seed`). Field for field, defaults included."""

    var n_components: Int
    var n_neighbors: Int
    var norm_laplacian: Bool
    var drop_first: Bool
    var tolerance: Float32
    var has_seed: Bool
    var seed: UInt64

    @staticmethod
    def default_with(n_components: Int, n_neighbors: Int) -> Self:
        return Self(
            n_components=n_components,
            n_neighbors=n_neighbors,
            norm_laplacian=True,
            drop_first=True,
            tolerance=Float32(1e-5),
            has_seed=False,
            seed=UInt64(0),
        )


def negate_kernel(vals: MutPointer[Float32, MutAnyOrigin], nnz_in: Int32):
    """`raft::linalg::map(x -> -x)` over the Laplacian's values (`:45-49`).
    A sign flip moves no bits of magnitude."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(nnz_in):
        return
    vals.unsafe_store(i, -vals.unsafe_load(i))


def divide_rows_kernel(
    vecs: MutPointer[Float32, MutAnyOrigin],
    diag: MutPointer[Float32, MutAnyOrigin],
    k_in: Int32,
    n_in: Int32,
):
    """`matrix_vector_op<ALONG_COLUMNS>(eigenvectors, diagonal, elem /
    diag)` (`:80-87`): every component `p` of every eigenvector is divided
    by `diagonal[p]`. `vecs` is `k x n` row-major (vector `c` at `c * n`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    if i >= Int(k_in) * n:
        return
    var p = i % n
    vecs.unsafe_store(i, ftz(vecs.unsafe_load(i) / diag.unsafe_load(p)))


def create_connectivity_graph(
    ctx: DeviceContext,
    params: SpectralEmbeddingParams,
    dataset: List[Float32],
    n_samples: Int,
    n_features: Int,
    mut trace: IdentityTrace,
) raises -> CooGraph:
    """`create_connectivity_graph` (`:133-205`): kNN, `(i, neighbor, 1.0)`,
    symmetrize with `0.5f * (a + b)`, sort, drop zeros. Returns the SORTED
    symmetric COO (their `connectivity_graph`)."""
    var k_search = params.n_neighbors
    if n_samples <= 0 or n_features <= 0:
        raise Error("spectral: dataset must be n_samples x n_features with both positive")
    if len(dataset) != n_samples * n_features:
        raise Error("spectral: dataset length does not match n_samples x n_features")
    if k_search < 1 or k_search > n_samples:
        raise Error(
            "spectral: n_neighbors=" + String(k_search)
            + " must satisfy 1 <= n_neighbors <= n_samples (" + String(n_samples) + ")"
        )
    for i in range(len(dataset)):
        var x = dataset[i]
        if not isfinite(x):
            raise Error(
                "spectral: dataset has a non-finite value at index " + String(i)
                + " -- refused by name (a NaN may not reach a card)"
            )
    var nnz = n_samples * k_search
    var h_data = ctx.enqueue_create_host_buffer[DType.float32](n_samples * n_features)
    var h_dist = ctx.enqueue_create_host_buffer[DType.float32](nnz)
    var h_idx = ctx.enqueue_create_host_buffer[DType.uint32](nnz)
    ctx.synchronize()
    for i in range(n_samples * n_features):
        h_data.unsafe_ptr().unsafe_store(i, dataset[i])
    # brute_force (L2SqrtExpanded), dataset against itself (:150-159)
    _ = knn_search(
        ctx,
        h_data.unsafe_ptr(),
        n_samples,
        h_data.unsafe_ptr(),
        n_samples,
        n_features,
        k_search,
        h_dist.unsafe_ptr(),
        h_idx.unsafe_ptr(),
        True,
    )
    # knn_cols = indices; knn_rows = idx / k_search; vals = 1.0f  (:161-176)
    var rows = List[Int32]()
    var cols = List[Int32]()
    var vals = List[Float32]()
    for e in range(nnz):
        rows.append(Int32(e // k_search))
        cols.append(Int32(h_idx.unsafe_ptr().unsafe_load(e)))
        vals.append(Float32(1.0))
    _ = h_data^
    _ = h_dist^
    _ = h_idx^
    var knn = CooGraph(n_samples, rows^, cols^, vals^)
    trace.record_list_i32("spectral.knn.cols", knn.cols)
    # coo_symmetrize (:183-188), coo_sort (:190-197), coo_remove_scalar(0)
    # (:199-204)
    var sym_raw = coo_symmetrize(ctx, knn)
    var sym_sorted = coo_sort(sym_raw)
    var graph = coo_remove_scalar(sym_sorted, Float32(0.0))
    return graph^


def create_laplacian(
    ctx: DeviceContext,
    params: SpectralEmbeddingParams,
    graph: CooGraph,
    mut diagonal: DeviceBuffer[DType.float32],
    tpb: Int = LAPLACIAN_TPB,
) raises -> DeviceCoo:
    """`create_laplacian` (`:31-52`): normalized or plain, then negated."""
    var lap: DeviceCoo
    if params.norm_laplacian:
        lap = laplacian_normalized(ctx, graph, diagonal, tpb)
    else:
        lap = compute_graph_laplacian(ctx, graph, tpb)
    ctx.enqueue_function[negate_kernel](
        lap.vals.unsafe_ptr(),
        Int32(lap.nnz),
        grid_dim=((lap.nnz + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    return lap^


def compute_eigenpairs(
    ctx: DeviceContext,
    params: SpectralEmbeddingParams,
    n_samples: Int,
    mut laplacian: DeviceCoo,
    mut diagonal: DeviceBuffer[DType.float32],
    mut embedding: List[Float32],
    mut trace: IdentityTrace,
    lanczos_tpb: Int = LANCZOS_TPB,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = 0.0,
) raises -> Int:
    """`compute_eigenpairs` (`:54-116`). `embedding` comes back `n_samples x
    n_out` row-major with `n_out = n_components - 1` when `drop_first`.
    Returns `n_out`."""
    var k = params.n_components
    # `max_iterations = 10 * n_samples` (:64), the RAFT_EXPECTS (:65-66),
    # `ncv = min(n - n_components, max(2k + 1, 20))` (:67) and
    # `tolerance = config.tolerance` (:68) are ALL VERBATIM THEIRS, checked
    # against cuvs-v26.08.00. They were briefly recorded as DEVIATION 780
    # while no 26.08 checkout existed; that claim is STRUCK. Nothing in this
    # block is a choice of ours.
    # RAFT_EXPECTS(n_samples - n_components > 0)  (:65-66)
    if n_samples - k <= 0:
        raise Error("Please set `ncv` to a value in (0, n_samples)")
    var ncv_hi = 2 * k + 1
    if ncv_hi < 20:
        ncv_hi = 20
    var ncv = n_samples - k
    if ncv_hi < ncv:
        ncv = ncv_hi
    comptime if SAB_NCV:
        # C1 sabotage: cuVS 25.08's `min(n_samples, max(2k+1, 20))`, i.e.
        # no `n - k` clamp. The oracle recomputes `ncv` its own way, so the
        # two cards take a different number of Lanczos steps.
        ncv = ncv_hi
        if ncv > n_samples:
            ncv = n_samples
    var max_iterations = 10 * n_samples
    comptime if SAB_MAXITER:
        max_iterations = 1000
    var config = LanczosSolverConfig(
        n_components=k,
        max_iterations=max_iterations,
        ncv=ncv,
        tolerance=params.tolerance,
        which=LANCZOS_LA,
        has_seed=params.has_seed,
        seed=params.seed,
    )
    var eigenvalues = List[Float32]()
    var eigenvectors = List[Float32]()
    var no_v0 = List[Float32]()
    _ = lanczos_compute_eigenpairs(
        ctx, config, laplacian, no_v0, False, eigenvalues, eigenvectors, trace,
        lanczos_tpb, scratch_pad, scratch_poison,
    )
    trace.record_list_f32("spectral.ritz", eigenvalues)
    trace.record_list_f32("spectral.ritz.vectors", eigenvectors)
    # eigenvectors /= diagonal (norm_laplacian)  (:80-87)
    if params.norm_laplacian:
        var d_vecs = upload_f32(ctx, eigenvectors)
        ctx.enqueue_function[divide_rows_kernel](
            d_vecs.unsafe_ptr(),
            diagonal.unsafe_ptr(),
            Int32(k),
            Int32(n_samples),
            grid_dim=((k * n_samples + lanczos_tpb - 1) // lanczos_tpb, 1, 1),
            block_dim=(lanczos_tpb, 1, 1),
        )
        ctx.synchronize()
        eigenvectors = download_f32(ctx, d_vecs, k * n_samples)
        _ = d_vecs^
    # reversed gather (:89-115): embedding column c_out = eigenvector column
    # n_out - 1 - c_out; with drop_first the LAST (trivial) column is dropped.
    var n_out = k - 1 if params.drop_first else k
    embedding.clear()
    for p in range(n_samples):
        for c_out in range(n_out):
            var src = n_out - 1 - c_out
            embedding.append(eigenvectors[src * n_samples + p])
    trace.record_list_f32("spectral.embedding", embedding)
    return n_out


def transform_graph(
    ctx: DeviceContext,
    params: SpectralEmbeddingParams,
    connectivity_graph: CooGraph,
    mut embedding: List[Float32],
    mut trace: IdentityTrace,
    laplacian_tpb: Int = LAPLACIAN_TPB,
    lanczos_tpb: Int = LANCZOS_TPB,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = 0.0,
) raises -> Int:
    """`transform` on a COO (`:118-131`): `create_laplacian` then
    `compute_eigenpairs`. The values of the graph are validated first
    (finite, non-negative: a negative affinity makes `sqrt(degree)` a NaN in
    theirs and no NaN may reach a card). Returns `n_out`."""
    var n = connectivity_graph.n
    if n <= 0:
        raise Error("spectral: connectivity_graph must have n > 0")
    for i in range(connectivity_graph.nnz()):
        var v = connectivity_graph.vals[i]
        if not isfinite(v):
            raise Error(
                "spectral: connectivity_graph has a non-finite value at entry "
                + String(i) + " -- refused by name"
            )
        if v < Float32(0.0):
            raise Error(
                "spectral: connectivity_graph has a negative value at entry "
                + String(i) + " -- refused by name (sqrt of a negative degree is NaN in theirs)"
            )
    var diagonal = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()
    var lap = create_laplacian(ctx, params, connectivity_graph, diagonal, laplacian_tpb)
    trace.record_device[DType.int32](ctx, "spectral.L.indptr", lap.indptr, n + 1)
    trace.record_device[DType.int32](ctx, "spectral.L.cols", lap.cols, lap.nnz)
    trace.record_device[DType.float32](ctx, "spectral.L.vals", lap.vals, lap.nnz)
    if params.norm_laplacian:
        trace.record_device[DType.float32](ctx, "spectral.diag", diagonal, n)
    var n_out = compute_eigenpairs(
        ctx, params, n, lap, diagonal, embedding, trace, lanczos_tpb, scratch_pad, scratch_poison
    )
    _ = diagonal^
    _ = lap^
    return n_out


def transform_dataset(
    ctx: DeviceContext,
    params: SpectralEmbeddingParams,
    dataset: List[Float32],
    n_samples: Int,
    n_features: Int,
    mut embedding: List[Float32],
    mut trace: IdentityTrace,
    laplacian_tpb: Int = LAPLACIAN_TPB,
    lanczos_tpb: Int = LANCZOS_TPB,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = 0.0,
) raises -> Int:
    """`transform` on a dataset (`:207-223`): the kNN graph, then
    `transform_graph`. Records the graph as `spectral.W.*`."""
    var graph = create_connectivity_graph(ctx, params, dataset, n_samples, n_features, trace)
    trace.record_list_i32("spectral.W.rows", graph.rows)
    trace.record_list_i32("spectral.W.cols", graph.cols)
    trace.record_list_f32("spectral.W.vals", graph.vals)
    return transform_graph(
        ctx, params, graph, embedding, trace, laplacian_tpb, lanczos_tpb, scratch_pad, scratch_poison
    )
