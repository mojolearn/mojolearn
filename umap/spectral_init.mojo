# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UMAP spectral initialization over the shipped cuVS/Lanczos path."""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation

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
"""Threads per column block of the post-pass pivot reduction (a power of
two: the tree below halves it)."""

comptime SPECTRAL_POST_FLAG_NONFINITE = UInt32(1)
comptime SPECTRAL_POST_FLAG_ZERO = UInt32(2)


def spectral_post_pivot_kernel(
    emb: MutPointer[Float32, MutAnyOrigin],
    scale: MutPointer[Float32, MutAnyOrigin],
    flags: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
    nc_in: Int32,
):
    """One block per column `c` of the row-major `n x nc` embedding: the
    column's pivot, its refusal flag and its scale.

    The pivot is the FIRST row of largest magnitude (the host walk's strict
    `>`): the pair (|v| bits, row) with the largest magnitude bits, the
    smallest row among equals. Magnitudes are non-negative, so their bits
    order as the floats do, and the integer comparison is exact and
    order-free on every vendor. `flags[c]` is 1 on a non-finite value, 2 on
    an all-zero column, else 0. `scale[c] = identical_div(10, peak)`,
    negated when the pivot is negative (a sign flip moves no magnitude
    bits)."""
    # Fits gate: three 32-bit words per thread, far under every column's
    # shared limit (Apple's 32 KiB is the smallest).
    comptime assert SPECTRAL_POST_TPB * 12 <= 16384, "post-pass shared page must fit every column"
    var c = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var n = Int(n_in)
    var nc = Int(nc_in)
    var s_mag = stack_allocation[
        SPECTRAL_POST_TPB, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var s_row = stack_allocation[
        SPECTRAL_POST_TPB, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var s_bad = stack_allocation[
        SPECTRAL_POST_TPB, Scalar[DType.uint32], address_space = AddressSpace.SHARED
    ]()
    var best_mag = UInt32(0)
    var best_row = UInt32(0xFFFFFFFF)
    var bad = UInt32(0)
    var i = t
    while i < n:
        var bits = bitcast[DType.uint32](emb.unsafe_load(i * nc + c))
        if ((bits >> UInt32(23)) & UInt32(0xFF)) == UInt32(0xFF):
            bad = UInt32(1)
        var mag = bits & UInt32(0x7FFFFFFF)
        # rows rise along the stride, so the first row at a magnitude wins
        if best_row == UInt32(0xFFFFFFFF) or mag > best_mag:
            best_mag = mag
            best_row = UInt32(i)
        i += SPECTRAL_POST_TPB
    s_mag[t] = best_mag
    s_row[t] = best_row
    s_bad[t] = bad
    barrier()
    var active = SPECTRAL_POST_TPB // 2
    while active > 0:
        if t < active:
            var om = s_mag[t + active]
            var orow = s_row[t + active]
            var m = s_mag[t]
            var r = s_row[t]
            # an empty slot carries row 0xFFFFFFFF and magnitude 0: it never
            # beats a real row (a real row is smaller at equal magnitude)
            if om > m or (om == m and orow < r):
                s_mag[t] = om
                s_row[t] = orow
            if s_bad[t + active] != UInt32(0):
                s_bad[t] = UInt32(1)
        barrier()
        active //= 2
    if t != 0:
        return
    var mag_bits = s_mag[0]
    var pivot = Int(s_row[0])
    var flag = UInt32(0)
    if s_bad[0] != UInt32(0):
        flag = SPECTRAL_POST_FLAG_NONFINITE
    elif mag_bits == UInt32(0):
        flag = SPECTRAL_POST_FLAG_ZERO
    flags.unsafe_store(c, flag)
    if flag != UInt32(0):
        scale.unsafe_store(c, Float32(0.0))
        return
    var peak = bitcast[DType.float32](mag_bits)
    var sc = identical_div(Float32(10.0), peak)
    var pivot_bits = bitcast[DType.uint32](emb.unsafe_load(pivot * nc + c))
    if (pivot_bits >> UInt32(31)) != UInt32(0):
        sc = -sc
    scale.unsafe_store(c, sc)


def spectral_post_scale_kernel(
    emb: MutPointer[Float32, MutAnyOrigin],
    scale: MutPointer[Float32, MutAnyOrigin],
    total_in: Int32,
    nc_in: Int32,
):
    """`emb[i, c] = ftz(identical_mul(emb[i, c], scale[c]))`, one thread
    per cell."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(total_in):
        return
    var c = i % Int(nc_in)
    emb.unsafe_store(i, ftz(identical_mul(emb.unsafe_load(i), scale.unsafe_load(c))))


def _spectral_post_pass_device(
    ctx: DeviceContext,
    mut d_emb: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_components: Int,
) raises -> List[Float32]:
    """The ordered post-pass both entries share, on the device (lane
    cpu4-umap, 2026-10-04): the largest-magnitude entry of each column made
    positive, each column scaled to max magnitude 10. Only the per-column
    refusal flags are read before the ONE download of the result. The host
    column is `umap/host/spectral_post_pass_host.mojo::host_spectral_post_pass`
    (the same seams, the same bits)."""
    var total = n_samples * n_components
    var d_scale = ctx.enqueue_create_buffer[DType.float32](n_components)
    var d_flags = ctx.enqueue_create_buffer[DType.uint32](n_components)
    ctx.enqueue_function[spectral_post_pivot_kernel](
        d_emb.unsafe_ptr(),
        d_scale.unsafe_ptr(),
        d_flags.unsafe_ptr(),
        Int32(n_samples),
        Int32(n_components),
        grid_dim=(n_components, 1, 1),
        block_dim=(SPECTRAL_POST_TPB, 1, 1),
    )
    var flags = List[UInt32](length=n_components, fill=UInt32(0))
    ctx.enqueue_copy(dst_ptr=flags.unsafe_ptr(), src_buf=d_flags)
    ctx.synchronize()
    # The refusals in the host walk's order: column by column, a non-finite
    # value before a zero peak.
    for c in range(n_components):  # small-loop(n_components: refusal flags, at most 32): one word per output dimension
        if flags[c] == SPECTRAL_POST_FLAG_NONFINITE:
            raise Error("UMAP spectral solver returned a non-finite value")
        if flags[c] == SPECTRAL_POST_FLAG_ZERO:
            raise Error("UMAP spectral solver returned a zero component")
    ctx.enqueue_function[spectral_post_scale_kernel](
        d_emb.unsafe_ptr(),
        d_scale.unsafe_ptr(),
        Int32(total),
        Int32(n_components),
        grid_dim=((total + SPECTRAL_POST_TPB - 1) // SPECTRAL_POST_TPB, 1, 1),
        block_dim=(SPECTRAL_POST_TPB, 1, 1),
    )
    var out = List[Float32](length=total, fill=Float32(0.0))
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=d_emb)
    ctx.synchronize()
    _ = d_scale^
    _ = d_flags^
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
