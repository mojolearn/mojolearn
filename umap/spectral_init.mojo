# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UMAP spectral initialization over the shipped cuVS/Lanczos path."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast

from core.identity_trace import IdentityTrace
from spectral.impl.sparse.coo import CooGraph
from spectral.impl.spectral_embedding import (
    MLSpectralEmbeddingParams,
    to_cuvs,
)
from spectral.impl.preprocessing.detail.spectral_embedding import (
    transform_device_coo_device,
    transform_graph_device,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_div, identical_mul
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

comptime UMAP_INIT_TOL_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_UMAP_INIT_TOL"]()
    and not is_defined["MOJOLEARN_UMAP_INIT_TOL_OFF"]()
)
"""FAST on Apple: the spectral INITIALIZATION solves to 1e-3 instead of
cuVS's 1e-5; the layout is only a starting point the optimizer then moves.
At 1e-4 (umap-learn's `eigsh` tolerance) taxi 100k took 37 Lanczos restarts,
at 1e-3 six, and the final embedding's trustworthiness (k = 15) moved by
-0.003 to +0.02 over four seeds and sizes, inside the seed-to-seed spread.

OFF BY DEFAULT since 2026-09-26 (lane/apple-identical-neural): a paired
check against the reference 1e-5, eight seeds per data set, 30k rows,
trustworthiness k = 15 on a fixed 3000-point subsample, found a systematic
loss: covtype 0.9672 -> 0.9590 (mean -0.0083, worst -0.0160, ALL eight
seeds worse), taxi 0.9809 -> 0.9766 (mean -0.0043, worst -0.0105); 5-NN
accuracy in the embedding within noise (-0.0004, -0.0014). FAST may move
bits but not quality, so the reference solve is the FAST default;
`-D MOJOLEARN_UMAP_INIT_TOL=1` is the trial arm."""
from umap.graph import FuzzySimplicialGraph
from umap.optimizer_identical_device import umap_dense_positive_coo_device


comptime SPECTRAL_POST_TPB = 256
"""Threads per block of the post-pass kernels (one thread per cell)."""


def spectral_post_peak_kernel(
    emb: MutPointer[Float32, MutAnyOrigin],
    ws: MutPointer[Int32, MutAnyOrigin],
    total_in: Int32,
    nc_in: Int32,
):
    """Pass 1 of the post-pass, one thread per cell of the row-major
    `n x nc` embedding. `ws` is `3 x nc` Int32, zeroed: `ws[c]` is raised to
    the column's largest magnitude BITS (non-negative floats order as their
    bits, so an integer max is exact and order-free on every vendor);
    `ws[2 nc + c]` is set to 1 on a non-finite value."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(total_in):
        return
    var nc = Int(nc_in)
    var c = i % nc
    var bits = bitcast[DType.uint32](emb.unsafe_load(i))
    if ((bits >> UInt32(23)) & UInt32(0xFF)) == UInt32(0xFF):
        _ = Atomic.max(ws.unsafe_offset(2 * nc + c), Int32(1))
        return
    _ = Atomic.max(ws.unsafe_offset(c), (bits & UInt32(0x7FFFFFFF)).cast[DType.int32]())


def spectral_post_pivot_kernel(
    emb: MutPointer[Float32, MutAnyOrigin],
    ws: MutPointer[Int32, MutAnyOrigin],
    total_in: Int32,
    nc_in: Int32,
):
    """Pass 2: the pivot is the FIRST row whose magnitude is the column's
    peak (the host walk's strict `>`). `ws[nc + c]` is raised to
    `0x7FFFFFFF - row` over the rows at the peak, so its max names the
    smallest such row (integer max: order-free)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(total_in):
        return
    var nc = Int(nc_in)
    var c = i % nc
    var row = i // nc
    var bits = bitcast[DType.uint32](emb.unsafe_load(i))
    if ((bits >> UInt32(23)) & UInt32(0xFF)) == UInt32(0xFF):
        return
    if (bits & UInt32(0x7FFFFFFF)).cast[DType.int32]() == ws.unsafe_load(c):
        _ = Atomic.max(ws.unsafe_offset(nc + c), Int32(0x7FFFFFFF) - Int32(row))


def spectral_post_scale_kernel(
    emb: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    ws: MutPointer[Int32, MutAnyOrigin],
    total_in: Int32,
    nc_in: Int32,
):
    """Pass 3: `dst[i, c] = ftz(identical_mul(emb[i, c], scale_c))` with
    `scale_c = identical_div(10, peak_c)`, negated when the pivot is
    negative (a sign flip moves no magnitude bits). Each thread forms its
    column's scale from the same two words, so every thread holds the same
    bits. `dst` is a separate buffer: the pivot's sign is read from `emb`
    while other cells are written."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(total_in):
        return
    var nc = Int(nc_in)
    var c = i % nc
    var peak = bitcast[DType.float32](ws.unsafe_load(c))
    var pivot = Int(Int32(0x7FFFFFFF) - ws.unsafe_load(nc + c))
    var sc = identical_div(Float32(10.0), peak)
    var pivot_bits = bitcast[DType.uint32](emb.unsafe_load(pivot * nc + c))
    if (pivot_bits >> UInt32(31)) != UInt32(0):
        sc = -sc
    dst.unsafe_store(i, ftz(identical_mul(emb.unsafe_load(i), sc)))


def _spectral_post_pass_device(
    ctx: DeviceContext,
    mut d_emb: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_components: Int,
) raises -> List[Float32]:
    """The ordered post-pass both entries share, on the device (lane
    cpu4-umap, 2026-10-04): the largest-magnitude entry of each column made
    positive, each column scaled to max magnitude 10. Three cell-parallel
    passes; only the `3 x n_components` peak/pivot/refusal words are read
    before the ONE download of the result. The host column is
    `umap/host/spectral_post_pass_host.mojo::host_spectral_post_pass` (the
    same seams, the same bits)."""
    var total = n_samples * n_components
    var blocks = (total + SPECTRAL_POST_TPB - 1) // SPECTRAL_POST_TPB
    var d_ws = ctx.enqueue_create_buffer[DType.int32](3 * n_components)
    ctx.enqueue_memset(d_ws, Int32(0))
    ctx.enqueue_function[spectral_post_peak_kernel](
        d_emb.unsafe_ptr(),
        d_ws.unsafe_ptr(),
        Int32(total),
        Int32(n_components),
        grid_dim=(blocks, 1, 1),
        block_dim=(SPECTRAL_POST_TPB, 1, 1),
    )
    ctx.enqueue_function[spectral_post_pivot_kernel](
        d_emb.unsafe_ptr(),
        d_ws.unsafe_ptr(),
        Int32(total),
        Int32(n_components),
        grid_dim=(blocks, 1, 1),
        block_dim=(SPECTRAL_POST_TPB, 1, 1),
    )
    var ws = List[Int32](length=3 * n_components, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=ws.unsafe_ptr(), src_buf=d_ws)
    ctx.synchronize()
    # The refusals in the host walk's order: column by column, a non-finite
    # value before a zero peak.
    for c in range(n_components):  # small-loop(n_components: refusal words, at most 32): one word per output dimension
        if ws[2 * n_components + c] != Int32(0):
            raise Error("UMAP spectral solver returned a non-finite value")
        if ws[c] == Int32(0):
            raise Error("UMAP spectral solver returned a zero component")
    var d_out = ctx.enqueue_create_buffer[DType.float32](total)
    ctx.enqueue_function[spectral_post_scale_kernel](
        d_emb.unsafe_ptr(),
        d_out.unsafe_ptr(),
        d_ws.unsafe_ptr(),
        Int32(total),
        Int32(n_components),
        grid_dim=(blocks, 1, 1),
        block_dim=(SPECTRAL_POST_TPB, 1, 1),
    )
    var out = List[Float32](length=total, fill=Float32(0.0))
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=d_out)
    ctx.synchronize()
    _ = d_ws^
    _ = d_out^
    return out^


def spectral_initialize_weights(
    ctx: DeviceContext,
    weights: List[Float32],
    n_samples: Int,
    n_components: Int,
    n_neighbors: Int,
    seed: UInt64,
) raises -> List[Float32]:
    """Return row-major 2D/3D initialization, canonically signed and scaled.

    The eigensolver and normalized Laplacian are the repository's shipped
    spectral implementation. IDENTICAL inherits its pinned reductions and
    seeded Lanczos path. FAST inherits its faster numeric kernels. The
    device post-pass is ordered in both modes: the largest-magnitude entry in each
    column is made positive and each column is scaled to max magnitude 10.
    """
    if n_components != 2 and n_components != 3:
        raise Error("UMAP spectral initialization supports only 2D or 3D")
    # The shipped Lanczos configuration uses k=n_components+1 and requires
    # ncv=n-k > k+1 at these small shapes.
    if n_samples < 2 * n_components + 4:
        raise Error("UMAP spectral initialization has too few samples")
    # the dense graph's validation and positive COO on the device
    # (`umap_dense_positive_coo_device`, lane cpu3-neighbors)
    var coo = umap_dense_positive_coo_device(ctx, weights, n_samples)
    var nnz = coo.nnz
    return spectral_initialize_device_coo(
        ctx, n_samples, nnz, coo.rows^, coo.cols^, coo.vals^,
        n_components, n_neighbors, seed,
    )


def spectral_initialize_coo(
    ctx: DeviceContext,
    var graph: CooGraph,
    n_samples: Int,
    n_components: Int,
    n_neighbors: Int,
    seed: UInt64,
) raises -> List[Float32]:
    """Shared solver/post-pass for dense-converted and CSR-origin COO.

    Caller supplies the same positive edges in row-major order. Lanczos
    uses ncv=min(n-k,max(2k+1,20)); UMAP2D/3D basis size is at most20*n,
    not n*n. This changes neither the solver nor its arithmetic.
    """
    if n_components < 1:
        raise Error("UMAP spectral initialization needs n_components >= 1")
    if n_samples < 2 * n_components + 4:
        raise Error("UMAP spectral initialization has too few samples")
    if graph.n != n_samples:
        raise Error("UMAP spectral COO shape disagrees with n_samples")
    var config = MLSpectralEmbeddingParams(
        n_components=n_components + 1,
        n_neighbors=n_neighbors,
        norm_laplacian=True,
        drop_first=True,
        has_seed=True,
        seed=seed,
    )
    # `transform_connectivity` is `transform_graph` at 1e-5 over `to_cuvs`;
    # its device-output twin leaves the embedding resident for the post-pass.
    var trace = IdentityTrace.disabled()
    var cp = to_cuvs(config)
    comptime if UMAP_INIT_TOL_FAST:
        cp.tolerance = Float32(1e-3)
    else:
        cp.tolerance = Float32(1e-5)
    var d_emb = ctx.enqueue_create_buffer[DType.float32](1)
    var n_out = transform_graph_device(ctx, cp, graph, d_emb, trace)
    if n_out != n_components or len(d_emb) != n_samples * n_components:
        raise Error("UMAP spectral solver returned the wrong shape")
    return _spectral_post_pass_device(ctx, d_emb, n_samples, n_components)


def spectral_initialize_device_coo(
    ctx: DeviceContext,
    n_samples: Int,
    nnz: Int,
    var rows: DeviceBuffer[DType.int32],
    var cols: DeviceBuffer[DType.int32],
    var vals: DeviceBuffer[DType.float32],
    n_components: Int,
    n_neighbors: Int,
    seed: UInt64,
) raises -> List[Float32]:
    """`spectral_initialize_coo` over a positive row-major COO already on the
    device (lane cpu3-neighbors, 2026-10-04): the same refusals, the same
    solver configuration and tolerance, the same post-pass; the COO is never
    built on the host."""
    if n_components < 1:
        raise Error("UMAP spectral initialization needs n_components >= 1")
    if n_samples < 2 * n_components + 4:
        raise Error("UMAP spectral initialization has too few samples")
    var config = MLSpectralEmbeddingParams(
        n_components=n_components + 1,
        n_neighbors=n_neighbors,
        norm_laplacian=True,
        drop_first=True,
        has_seed=True,
        seed=seed,
    )
    var trace = IdentityTrace.disabled()
    var cp = to_cuvs(config)
    comptime if UMAP_INIT_TOL_FAST:
        cp.tolerance = Float32(1e-3)
    else:
        cp.tolerance = Float32(1e-5)
    var d_emb = ctx.enqueue_create_buffer[DType.float32](1)
    var n_out = transform_device_coo_device(
        ctx, cp, n_samples, nnz, rows^, cols^, vals^, d_emb, trace
    )
    if n_out != n_components or len(d_emb) != n_samples * n_components:
        raise Error("UMAP spectral solver returned the wrong shape")
    return _spectral_post_pass_device(ctx, d_emb, n_samples, n_components)


def spectral_initialize(
    ctx: DeviceContext,
    graph: FuzzySimplicialGraph,
    n_components: Int,
    seed: UInt64,
) raises -> List[Float32]:
    return spectral_initialize_weights(
        ctx, graph.weights, graph.n_samples, n_components,
        graph.n_neighbors, seed,
    )
