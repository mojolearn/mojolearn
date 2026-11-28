# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The diffusion map on the device (lane sc-graph-1, 2026-11-28; Scanpy
`tl.diffmap`; the arithmetic is `spectral/diffmap_words.mojo`'s).

THE SEQUENCE, EVERY PIECE THE SPECTRAL EMBEDDING'S OWN
  W (diagonal dropped) -> compute_graph_laplacian    D - A, canonical COO, one
                                                     diagonal entry per row
                       -> coo_diagonal_kernel         q = the degrees of W
                       -> diffmap_density_kernel      K = w / (q_r q_c) in place,
                                                     the diagonal entries 0
                       -> laplacian_of_device_coo     D_K - K on the same COO
                       -> laplacian_normalize_device  L_sym(K), diag = sqrt(deg K)
                       -> negate, compute_eigenpairs_keep_device (keep: the
                          Ritz values and the UNDIVIDED Ritz vectors, the
                          eigenvectors of T = I - L_sym(K) Scanpy returns)
No host step between the upload of W and the one download of the pairs."""
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from spectral.diffmap_words import diffmap_density_value
from spectral.impl.sparse.coo import CooGraph
from spectral.impl.spectral_predict_common import SpectralPredictionState
from spectral.impl.sparse.matrix.detail.diagonal import coo_diagonal_kernel
from spectral.impl.sparse.linalg.detail.laplacian import (
    LAPLACIAN_TPB,
    compute_graph_laplacian,
    laplacian_normalize_device,
    laplacian_of_device_coo,
)
from spectral.impl.sparse.solver.detail.lanczos import LANCZOS_TPB
from spectral.impl.preprocessing.detail.spectral_embedding import (
    SpectralEmbeddingParams,
    compute_eigenpairs_keep_device,
    negate_kernel,
)


def diffmap_density_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    nnz_in: Int32,
):
    """Entry i of W's Laplacian `D - A` becomes K's entry: off the diagonal
    -(-w) / (q_r q_c) (`diffmap_density_value`; the negation is exact), the
    diagonal 0 (W had none). Reads q, writes only its own entry."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(nnz_in):
        return
    var r = Int(rows.unsafe_load(i))
    var c = Int(cols.unsafe_load(i))
    if r == c:
        vals.unsafe_store(i, Float32(0))
    else:
        vals.unsafe_store(i, diffmap_density_value(-vals.unsafe_load(i), q.unsafe_load(r), q.unsafe_load(c)))


def transform_diffmap_keep(
    ctx: DeviceContext,
    n_components: Int,
    seed: UInt64,
    tolerance: Float32,
    graph: CooGraph,
    mut state: SpectralPredictionState,
    mut trace: IdentityTrace,
    tpb: Int = LAPLACIAN_TPB,
    lanczos_tpb: Int = LANCZOS_TPB,
) raises -> Int:
    """The module sequence on `graph` (W, its diagonal already dropped):
    `state.eigenvalues` receives the k Ritz values of -L_sym(K) in
    decreasing order and `state.eigenvectors` the n x k row-major
    eigenvectors in the same column order. Returns k."""
    var n = graph.n
    if n <= 0:
        raise Error("diffusion map: the graph must have n > 0")
    var params = SpectralEmbeddingParams(
        n_components=n_components, n_neighbors=0, norm_laplacian=True, drop_first=False,
        tolerance=tolerance, has_seed=True, seed=seed,
    )
    var lap = compute_graph_laplacian(ctx, graph, tpb, True)
    var nnz = lap.nnz
    var q = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_memset(q, Float32(0.0))
    var grid = (nnz + tpb - 1) // tpb
    ctx.enqueue_function[coo_diagonal_kernel](
        lap.rows.unsafe_ptr(), lap.cols.unsafe_ptr(), lap.vals.unsafe_ptr(), q.unsafe_ptr(), Int32(nnz),
        grid_dim=(grid, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.enqueue_function[diffmap_density_kernel](
        lap.rows.unsafe_ptr(), lap.cols.unsafe_ptr(), lap.vals.unsafe_ptr(), q.unsafe_ptr(), Int32(nnz),
        grid_dim=(grid, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    _ = q^  # read by the density launch above; drained before release
    var lap_k = laplacian_of_device_coo(ctx, lap^, tpb)
    var diagonal = ctx.enqueue_create_buffer[DType.float32](n)
    lap_k = laplacian_normalize_device(ctx, lap_k^, diagonal, tpb)
    ctx.enqueue_function[negate_kernel](
        lap_k.vals.unsafe_ptr(), Int32(lap_k.nnz),
        grid_dim=((lap_k.nnz + tpb - 1) // tpb, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](ctx, "diffmap.L.vals", lap_k.vals, lap_k.nnz)
    var d_emb = ctx.enqueue_create_buffer[DType.float32](1)
    var n_out = compute_eigenpairs_keep_device(
        ctx, params, n, lap_k, diagonal, d_emb, state, True, trace, lanczos_tpb,
    )
    _ = d_emb^
    _ = diagonal^
    _ = lap_k^
    return n_out
