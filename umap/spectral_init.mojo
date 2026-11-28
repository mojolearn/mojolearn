# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UMAP spectral initialization over the shipped cuVS/Lanczos path."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.atomic import Atomic
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast

from core.identity_trace import IdentityTrace
from spectral.impl.sparse.coo import CooGraph
from spectral.impl.spectral_embedding import (
    MLSpectralEmbeddingParams,
    to_cuvs,
)
from spectral.impl.preprocessing.detail.spectral_embedding import (
    SpectralEmbeddingParams,
    compute_eigenpairs_status_device,
    laplacian_device_coo,
    transform_graph_device,
)
from spectral.impl.spectral_predict_common import SpectralPredictionState
from spectral.impl.sparse.solver.lanczos_types import LanczosStatus
from spectral.checks.device_io import download_i32
from core.device_fold import device_exclusive_scan_total, device_exclusive_scan_total_from
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len
from std.sys import stderr
from umap.spectral_components_math import (
    SC_MAX_RESTARTS,
    SC_STREAM_LAYOUT,
    SpectralInitReport,
    sc_accept,
    sc_component_solved,
    sc_data_range,
    sc_expansion,
    sc_fallback_message,
    sc_hash,
    sc_mag_bits,
    sc_meta_coord,
    sc_nonfinite,
    sc_random_cell,
    sc_seed_hi,
    sc_seed_lo,
    sc_spectral_cell,
    sc_sq_step,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_div, identical_mul
from std.sys.compile import is_defined
from core.accelerator_target import TARGET_HAS_APPLE_GPU

comptime UMAP_INIT_TOL_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and TARGET_HAS_APPLE_GPU
    and is_defined["MOJOLEARN_UMAP_INIT_TOL"]()
)
# TOMBSTONE: MOJOLEARN_UMAP_INIT_TOL (opt-in arm, quality loss, no A/B kept) deleted 2026-11-15 by lane/expdel2-13-cluster; code recoverable at ade07601c8.
# Restore: git apply experiments/removed/MOJOLEARN_UMAP_INIT_TOL.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-classical: Apple fast is paused, not abandoned (Andrew); opt-in, default off, Apple fast only.
# TOMBSTONE: MOJOLEARN_UMAP_INIT_TOL_OFF (old path, no A/B recorded beside the default) deleted 2026-11-15 by lane/expdel2-13-cluster; code recoverable at 976872a147.
# Restore: git apply experiments/removed/MOJOLEARN_UMAP_INIT_TOL_OFF.patch; record in docs/TOMBSTONES.md.
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
    var ws = List[Int32](unsafe_uninit_length=3 * n_components)
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
    var out = List[Float32](unsafe_uninit_length=total)
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
    # shared handles (DeviceBuffer copies retain the same allocation, no data
    # copy): moving fields out of coo is refused (Mojo 1.0 partial move)
    var res = spectral_initialize_device_coo(
        ctx, n_samples, nnz, coo.rows.copy(), coo.cols.copy(), coo.vals.copy(),
        n_components, n_neighbors, seed,
    )
    _ = coo^
    return res^


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


comptime _SC_I32P = MutPointer[Int32, MutAnyOrigin]
comptime _SC_U32P = MutPointer[UInt32, MutAnyOrigin]
comptime _SC_F32P = MutPointer[Float32, MutAnyOrigin]
comptime SC_TPB = 256
"""Threads per block of the component kernels (one thread per item)."""


def _sc_grid(n: Int) -> Int:
    return (n + SC_TPB - 1) // SC_TPB if n > 0 else 1


@always_inline
def _sc_gid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def sc_value_check_kernel(vals: _SC_F32P, flag: _SC_I32P, nnz_in: Int32):
    """`flag[0] = 1` on a non-finite or negative edge weight (by bits; `-0`
    is not negative): the Laplacian's value refusal, taken once over the
    whole graph before the components split it."""
    var e = _sc_gid()
    if e >= Int(nnz_in):
        return
    var v = vals.unsafe_load(e)
    var bits = bitcast[DType.uint32](v)
    if sc_nonfinite(v) or ((bits >> UInt32(31)) != UInt32(0) and (bits & UInt32(0x7FFFFFFF)) != UInt32(0)):
        _ = Atomic.max(flag, Int32(1))


def sc_label_init_kernel(labels: _SC_I32P, n_in: Int32):
    var i = _sc_gid()
    if i >= Int(n_in):
        return
    labels.unsafe_store(i, Int32(i))


def sc_label_hook_kernel(rows: _SC_I32P, cols: _SC_I32P, labels: _SC_I32P, changed: _SC_I32P, nnz_in: Int32):
    """One round of min-label propagation over the edges: both endpoints and
    both endpoints' labels take the smaller label (integer `Atomic.min`).
    Every label stays a vertex of its own component and never rises, so the
    fixed point is unique: every vertex labelled with the SMALLEST index of
    its component, whatever the thread order."""
    var e = _sc_gid()
    if e >= Int(nnz_in):
        return
    var r = Int(rows.unsafe_load(e))
    var c = Int(cols.unsafe_load(e))
    var a = labels.unsafe_load(r)
    var b = labels.unsafe_load(c)
    if a == b:
        return
    var m = a if a < b else b
    _ = Atomic.min(labels.unsafe_offset(r), m)
    _ = Atomic.min(labels.unsafe_offset(c), m)
    _ = Atomic.min(labels.unsafe_offset(Int(a)), m)
    _ = Atomic.min(labels.unsafe_offset(Int(b)), m)
    _ = Atomic.max(changed, Int32(1))


def sc_label_jump_kernel(labels: _SC_I32P, n_in: Int32):
    """Pointer jumping: each label follows its chain to a self-labelled
    vertex (the chain strictly descends, so it ends)."""
    var i = _sc_gid()
    if i >= Int(n_in):
        return
    var v = labels.unsafe_load(i)
    var w = labels.unsafe_load(Int(v))
    while w != v:
        v = w
        w = labels.unsafe_load(Int(v))
    labels.unsafe_store(i, v)


def sc_root_flag_kernel(labels: _SC_I32P, flag: _SC_I32P, n_in: Int32):
    var i = _sc_gid()
    if i >= Int(n_in):
        return
    flag.unsafe_store(i, Int32(1) if labels.unsafe_load(i) == Int32(i) else Int32(0))


def sc_comp_kernel(labels: _SC_I32P, rank: _SC_I32P, comp: _SC_I32P, n_in: Int32):
    """`comp[i]` = the rank of `i`'s root among the roots: components are
    numbered by their smallest member."""
    var i = _sc_gid()
    if i >= Int(n_in):
        return
    comp.unsafe_store(i, rank.unsafe_load(Int(labels.unsafe_load(i))))


def sc_vertex_keys_kernel(comp: _SC_I32P, keys: _SC_U32P, values: _SC_U32P, counts: _SC_I32P, n_in: Int32):
    """Sort keys (component) and values (vertex index, ascending) of the
    vertices, and each component's vertex count (integer adds: exact)."""
    var i = _sc_gid()
    if i >= Int(n_in):
        return
    var c = comp.unsafe_load(i)
    keys.unsafe_store(i, UInt32(Int(c)))
    values.unsafe_store(i, UInt32(i))
    _ = Atomic.fetch_add(counts.unsafe_offset(Int(c)), Int32(1))


def sc_edge_keys_kernel(rows: _SC_I32P, comp: _SC_I32P, keys: _SC_U32P, values: _SC_U32P, counts: _SC_I32P, nnz_in: Int32):
    """The same for the edges: key = the component of the edge's row."""
    var e = _sc_gid()
    if e >= Int(nnz_in):
        return
    var c = comp.unsafe_load(Int(rows.unsafe_load(e)))
    keys.unsafe_store(e, UInt32(Int(c)))
    values.unsafe_store(e, UInt32(e))
    _ = Atomic.fetch_add(counts.unsafe_offset(Int(c)), Int32(1))


def sc_local_kernel(perm: _SC_U32P, comp: _SC_I32P, vstart: _SC_I32P, local: _SC_I32P, n_in: Int32):
    """`local[v]` = `v`'s index inside its component: its slot in the
    stable (component, index) order minus the component's first slot."""
    var p = _sc_gid()
    if p >= Int(n_in):
        return
    var v = Int(perm.unsafe_load(p))
    local.unsafe_store(v, Int32(p) - vstart.unsafe_load(Int(comp.unsafe_load(v))))


def sc_edge_gather_kernel(
    eperm: _SC_U32P, rows: _SC_I32P, cols: _SC_I32P, vals: _SC_F32P, local: _SC_I32P,
    srows: _SC_I32P, scols: _SC_I32P, svals: _SC_F32P, nnz_in: Int32,
):
    """The edges grouped by component, in their input (row-major) order
    within it, relabelled to local indices (a monotone relabelling, so each
    component's COO stays row-major)."""
    var q = _sc_gid()
    if q >= Int(nnz_in):
        return
    var e = Int(eperm.unsafe_load(q))
    srows.unsafe_store(q, local.unsafe_load(Int(rows.unsafe_load(e))))
    scols.unsafe_store(q, local.unsafe_load(Int(cols.unsafe_load(e))))
    svals.unsafe_store(q, vals.unsafe_load(e))


def sc_meta_kernel(meta: _SC_F32P, n_comp_in: Int32, dim_in: Int32, seed_lo: UInt32, seed_hi: UInt32):
    var i = _sc_gid()
    var dim = Int(dim_in)
    var n_comp = Int(n_comp_in)
    if i >= n_comp * dim:
        return
    meta.unsafe_store(i, sc_meta_coord(i // dim, i % dim, n_comp, dim, seed_lo, seed_hi))


def sc_range_kernel(meta: _SC_F32P, data_range: _SC_F32P, n_comp_in: Int32, dim_in: Int32):
    """One thread per component: half the smallest positive distance to the
    other centroids (`multi_component_layout`'s `data_range`), the other
    centroids in index order, the coordinates in order."""
    var c = _sc_gid()
    var n_comp = Int(n_comp_in)
    var dim = Int(dim_in)
    if c >= n_comp:
        return
    var best = Float32(-1.0)
    for j in range(n_comp):  # small-loop(n_comp: the other centroids; model structure, one thread per component)
        var acc = Float32(0.0)
        for d in range(dim):  # small-loop(dim: output coordinates, at most 32)
            acc = sc_sq_step(acc, meta.unsafe_load(c * dim + d), meta.unsafe_load(j * dim + d))
        if acc > Float32(0.0) and (best < Float32(0.0) or acc < best):
            best = acc
    data_range.unsafe_store(c, sc_data_range(best))


def sc_random_layout_kernel(
    dst: _SC_F32P, comp: _SC_I32P, meta: _SC_F32P, data_range: _SC_F32P,
    n_in: Int32, dim_in: Int32, seed_lo: UInt32, seed_hi: UInt32, single: Int32,
):
    """Every cell's uniform draw: around its component's centroid within its
    `data_range` (`single == 0`), or umap-learn's whole-graph fallback
    `uniform(-10, 10)` (`single != 0`; `comp`, `meta`, `data_range` unread)."""
    var i = _sc_gid()
    var dim = Int(dim_in)
    if i >= Int(n_in) * dim:
        return
    var v = i // dim
    var d = i - v * dim
    var h = sc_hash(seed_lo, seed_hi, SC_STREAM_LAYOUT, UInt32(v), UInt32(d))
    if single != Int32(0):
        dst.unsafe_store(i, sc_random_cell(h, Float32(10.0), Float32(0.0)))
        return
    var c = Int(comp.unsafe_load(v))
    dst.unsafe_store(i, sc_random_cell(h, data_range.unsafe_load(c), meta.unsafe_load(c * dim + d)))


def sc_peak_kernel(emb: _SC_F32P, ws: _SC_I32P, total_in: Int32):
    """`ws[0]` = the largest magnitude bits of a component's embedding,
    `ws[1] = 1` on a non-finite cell (integer maxima: order-free)."""
    var i = _sc_gid()
    if i >= Int(total_in):
        return
    var v = emb.unsafe_load(i)
    if sc_nonfinite(v):
        _ = Atomic.max(ws.unsafe_offset(1), Int32(1))
        return
    _ = Atomic.max(ws, sc_mag_bits(v))


def sc_place_kernel(
    emb: _SC_F32P, dst: _SC_F32P, perm: _SC_U32P, ws: _SC_I32P, meta: _SC_F32P, data_range: _SC_F32P,
    first_in: Int32, comp_in: Int32, size_in: Int32, dim_in: Int32,
):
    """A solved component's rows: local row `p` is vertex
    `perm[first + p]`; each cell is its eigenvector entry times
    `data_range / peak`, plus the centroid."""
    var i = _sc_gid()
    var dim = Int(dim_in)
    if i >= Int(size_in) * dim:
        return
    var p = i // dim
    var d = i - p * dim
    var c = Int(comp_in)
    var v = Int(perm.unsafe_load(Int(first_in) + p))
    var expansion = sc_expansion(data_range.unsafe_load(c), bitcast[DType.float32](ws.unsafe_load(0)))
    dst.unsafe_store(v * dim + d, sc_spectral_cell(emb.unsafe_load(i), expansion, meta.unsafe_load(c * dim + d)))


def _sc_solve(
    ctx: DeviceContext,
    cp: SpectralEmbeddingParams,
    n: Int,
    nnz: Int,
    var rows: DeviceBuffer[DType.int32],
    var cols: DeviceBuffer[DType.int32],
    var vals: DeviceBuffer[DType.float32],
    n_components: Int,
    mut d_emb: DeviceBuffer[DType.float32],
    mut restarts: Int,
) raises -> Bool:
    """One graph's spectral embedding (`n x n_components` into `d_emb`):
    the Laplacian (its input refusals RAISE), then the bounded eigen solve.
    Returns False when the solve failed: it raised (a restart breakdown, a
    projected-solve failure), returned the wrong shape, or stopped at
    `SC_MAX_RESTARTS` with a residual above `SC_ACCEPT_RESIDUAL`."""
    var trace = IdentityTrace.disabled()
    var diagonal = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()
    var lap = laplacian_device_coo(ctx, cp, n, nnz, rows^, cols^, vals^, diagonal, trace)
    var ok = False
    try:
        var status = LanczosStatus.empty()
        var state = SpectralPredictionState()
        var n_out = compute_eigenpairs_status_device(
            ctx, cp, n, lap, diagonal, d_emb, state, False, trace, SC_MAX_RESTARTS, status,
        )
        restarts = status.restarts
        ok = n_out == n_components and len(d_emb) == n * n_components and sc_accept(
            status.converged, status.residual
        )
    except:
        ok = False
    _ = diagonal^
    _ = lap^
    return ok


def _sc_params(n_components: Int, n_neighbors: Int, seed: UInt64) -> SpectralEmbeddingParams:
    """The solver configuration of every UMAP spectral solve."""
    var config = MLSpectralEmbeddingParams(
        n_components=n_components + 1,
        n_neighbors=n_neighbors,
        norm_laplacian=True,
        drop_first=True,
        has_seed=True,
        seed=seed,
    )
    var cp = to_cuvs(config)
    comptime if UMAP_INIT_TOL_FAST:
        cp.tolerance = Float32(1e-3)
    else:
        cp.tolerance = Float32(1e-5)
    return cp^


def spectral_initialize_device_coo_report(
    ctx: DeviceContext,
    n_samples: Int,
    nnz: Int,
    var rows: DeviceBuffer[DType.int32],
    var cols: DeviceBuffer[DType.int32],
    var vals: DeviceBuffer[DType.float32],
    n_components: Int,
    n_neighbors: Int,
    seed: UInt64,
) raises -> SpectralInitReport:
    """UMAP's spectral initialization over a positive row-major COO already
    on the device, umap-learn's `spectral_layout` (lane sc-ann-1-spectral,
    2026-11-27; arithmetic in `umap/spectral_components_math.mojo`).

    1. The connected components, on the device: min-label propagation with
       pointer jumping until no edge joins two labels (a driver loop over
       rounds, one status word each). Every vertex ends labelled with the
       smallest index of its component; components are numbered in that
       order.
    2. ONE component: the spectral embedding of the whole graph, as before,
       with the Lanczos bounded at `SC_MAX_RESTARTS` thick restarts.
    3. MORE THAN ONE: the vertices and edges are grouped by component with a
       stable radix sort (component, then input order) and relabelled to
       local indices; the centroids and `data_range`s are formed; every
       cell takes its uniform draw around its centroid; then each component
       big enough for a solve (`sc_component_solved`; a driver loop over
       components) gets its own spectral embedding, scaled to its
       `data_range` and centred, over its rows.
    4. The ordered post-pass (largest magnitude positive, max magnitude 10
       per column) over the whole embedding.

    FALLBACK, STATED: a failed eigen solve (see `_sc_solve`) leaves that
    component's rows at the uniform draw (the whole graph: umap-learn's
    `uniform(-10, 10)`), counted in `n_fallback` with `message` set.
    Refusals of the input (shape, too few samples, a non-finite or negative
    weight, an out-of-range index) still raise. Memory is linear in
    `n + nnz`: the component buffers are O(n + nnz) and each solve's Krylov
    basis is `ncv x n_c` with `ncv = min(n_c - k, max(2k + 1, 20))`."""
    if n_components < 1:
        raise Error("UMAP spectral initialization needs n_components >= 1")
    if n_samples < 2 * n_components + 4:
        raise Error("UMAP spectral initialization has too few samples")
    var n = n_samples
    var dim = n_components
    var cp = _sc_params(n_components, n_neighbors, seed)
    var seed_lo = sc_seed_lo(seed)
    var seed_hi = sc_seed_hi(seed)
    var g_n = _sc_grid(n)
    var g_e = _sc_grid(nnz)
    # 1. connected components
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    var changed = ctx.enqueue_create_buffer[DType.int32](1)
    var h_word = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_function[sc_label_init_kernel](
        labels.unsafe_ptr(), Int32(n), grid_dim=(g_n, 1, 1), block_dim=(SC_TPB, 1, 1),
    )
    for _ in range(n + 1):  # driver-loop: over label-propagation rounds (one status word each; at most n, each changing round lowers a label)
        if nnz == 0:
            break
        ctx.enqueue_memset(changed, Int32(0))
        ctx.enqueue_function[sc_label_hook_kernel](
            rows.unsafe_ptr(), cols.unsafe_ptr(), labels.unsafe_ptr(), changed.unsafe_ptr(), Int32(nnz),
            grid_dim=(g_e, 1, 1), block_dim=(SC_TPB, 1, 1),
        )
        ctx.enqueue_function[sc_label_jump_kernel](
            labels.unsafe_ptr(), Int32(n), grid_dim=(g_n, 1, 1), block_dim=(SC_TPB, 1, 1),
        )
        ctx.enqueue_copy(dst_ptr=h_word.unsafe_ptr(), src_buf=changed)
        ctx.synchronize()
        if h_word.unsafe_ptr().unsafe_load(0) == Int32(0):
            break
    var flag = ctx.enqueue_create_buffer[DType.int32](n)
    var rank = ctx.enqueue_create_buffer[DType.int32](n + 1)
    ctx.enqueue_function[sc_root_flag_kernel](
        labels.unsafe_ptr(), flag.unsafe_ptr(), Int32(n), grid_dim=(g_n, 1, 1), block_dim=(SC_TPB, 1, 1),
    )
    device_exclusive_scan_total_from(ctx, flag, rank, n)
    var rank_tail = rank.create_sub_buffer[DType.int32](n, 1)
    var n_comp_w = download_i32(ctx, rank_tail, 1)
    var n_comp = Int(n_comp_w[0])
    _ = rank_tail^
    _ = flag^
    _ = changed^
    _ = h_word^
    if n_comp <= 1:
        # 2. one component: the whole graph's embedding
        _ = rank^
        _ = labels^
        var d_emb1 = ctx.enqueue_create_buffer[DType.float32](1)
        var restarts1 = 0
        var ok1 = _sc_solve(ctx, cp, n, nnz, rows^, cols^, vals^, dim, d_emb1, restarts1)
        if ok1:
            try:
                var out1 = _spectral_post_pass_device(ctx, d_emb1, n, dim)
                return SpectralInitReport(out1^, 1, 1, 0, 0, restarts1, String(""))
            except:
                ok1 = False
        var d_rand = ctx.enqueue_create_buffer[DType.float32](n * dim)
        var no_comp = ctx.enqueue_create_buffer[DType.int32](1)
        var no_f = ctx.enqueue_create_buffer[DType.float32](1)
        ctx.enqueue_function[sc_random_layout_kernel](
            d_rand.unsafe_ptr(), no_comp.unsafe_ptr(), no_f.unsafe_ptr(), no_f.unsafe_ptr(),
            Int32(n), Int32(dim), seed_lo, seed_hi, Int32(1),
            grid_dim=(_sc_grid(n * dim), 1, 1), block_dim=(SC_TPB, 1, 1),
        )
        var outr = _spectral_post_pass_device(ctx, d_rand, n, dim)
        _ = no_comp^
        _ = no_f^
        _ = d_rand^
        _ = d_emb1^
        var msg1 = sc_fallback_message(1, 1)
        return SpectralInitReport(outr^, 1, 0, 0, 1, restarts1, msg1^)
    # 3. more than one component
    var bad = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(bad, Int32(0))
    ctx.enqueue_function[sc_value_check_kernel](
        vals.unsafe_ptr(), bad.unsafe_ptr(), Int32(nnz), grid_dim=(g_e, 1, 1), block_dim=(SC_TPB, 1, 1),
    )
    var bad_w = download_i32(ctx, bad, 1)
    if bad_w[0] != Int32(0):
        raise Error("spectral: connectivity_graph has a non-finite or negative value -- refused by name")
    var comp = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[sc_comp_kernel](
        labels.unsafe_ptr(), rank.unsafe_ptr(), comp.unsafe_ptr(), Int32(n),
        grid_dim=(g_n, 1, 1), block_dim=(SC_TPB, 1, 1),
    )
    # vertices grouped by component (stable: ascending index inside)
    var vstart = ctx.enqueue_create_buffer[DType.int32](n_comp + 1)
    ctx.enqueue_memset(vstart, Int32(0))
    var vkeys = ctx.enqueue_create_buffer[DType.uint32](n)
    var perm = ctx.enqueue_create_buffer[DType.uint32](n)
    var vtk = ctx.enqueue_create_buffer[DType.uint32](n)
    var vtv = ctx.enqueue_create_buffer[DType.uint32](n)
    var vcounts = ctx.enqueue_create_buffer[DType.int32](frs_counts_len(n))
    ctx.enqueue_function[sc_vertex_keys_kernel](
        comp.unsafe_ptr(), vkeys.unsafe_ptr(), perm.unsafe_ptr(), vstart.unsafe_ptr(), Int32(n),
        grid_dim=(g_n, 1, 1), block_dim=(SC_TPB, 1, 1),
    )
    device_exclusive_scan_total(ctx, vstart, n_comp)
    fast_radix_sort_pairs_u32(ctx, n, vkeys, perm, vtk, vtv, vcounts)
    var local = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[sc_local_kernel](
        perm.unsafe_ptr(), comp.unsafe_ptr(), vstart.unsafe_ptr(), local.unsafe_ptr(), Int32(n),
        grid_dim=(g_n, 1, 1), block_dim=(SC_TPB, 1, 1),
    )
    # edges grouped by component (stable: input order inside), local labels
    var ne = nnz if nnz > 0 else 1
    var estart = ctx.enqueue_create_buffer[DType.int32](n_comp + 1)
    ctx.enqueue_memset(estart, Int32(0))
    var ekeys = ctx.enqueue_create_buffer[DType.uint32](ne)
    var eperm = ctx.enqueue_create_buffer[DType.uint32](ne)
    var etk = ctx.enqueue_create_buffer[DType.uint32](ne)
    var etv = ctx.enqueue_create_buffer[DType.uint32](ne)
    var ecounts = ctx.enqueue_create_buffer[DType.int32](frs_counts_len(ne))
    var srows = ctx.enqueue_create_buffer[DType.int32](ne)
    var scols = ctx.enqueue_create_buffer[DType.int32](ne)
    var svals = ctx.enqueue_create_buffer[DType.float32](ne)
    if nnz > 0:
        ctx.enqueue_function[sc_edge_keys_kernel](
            rows.unsafe_ptr(), comp.unsafe_ptr(), ekeys.unsafe_ptr(), eperm.unsafe_ptr(), estart.unsafe_ptr(),
            Int32(nnz), grid_dim=(g_e, 1, 1), block_dim=(SC_TPB, 1, 1),
        )
    device_exclusive_scan_total(ctx, estart, n_comp)
    if nnz > 0:
        fast_radix_sort_pairs_u32(ctx, nnz, ekeys, eperm, etk, etv, ecounts)
        ctx.enqueue_function[sc_edge_gather_kernel](
            eperm.unsafe_ptr(), rows.unsafe_ptr(), cols.unsafe_ptr(), vals.unsafe_ptr(), local.unsafe_ptr(),
            srows.unsafe_ptr(), scols.unsafe_ptr(), svals.unsafe_ptr(), Int32(nnz),
            grid_dim=(g_e, 1, 1), block_dim=(SC_TPB, 1, 1),
        )
    # centroids, data ranges, every cell's uniform draw
    var meta = ctx.enqueue_create_buffer[DType.float32](n_comp * dim)
    var drange = ctx.enqueue_create_buffer[DType.float32](n_comp)
    ctx.enqueue_function[sc_meta_kernel](
        meta.unsafe_ptr(), Int32(n_comp), Int32(dim), seed_lo, seed_hi,
        grid_dim=(_sc_grid(n_comp * dim), 1, 1), block_dim=(SC_TPB, 1, 1),
    )
    ctx.enqueue_function[sc_range_kernel](
        meta.unsafe_ptr(), drange.unsafe_ptr(), Int32(n_comp), Int32(dim),
        grid_dim=(_sc_grid(n_comp), 1, 1), block_dim=(SC_TPB, 1, 1),
    )
    var d_out = ctx.enqueue_create_buffer[DType.float32](n * dim)
    ctx.enqueue_function[sc_random_layout_kernel](
        d_out.unsafe_ptr(), comp.unsafe_ptr(), meta.unsafe_ptr(), drange.unsafe_ptr(),
        Int32(n), Int32(dim), seed_lo, seed_hi, Int32(0),
        grid_dim=(_sc_grid(n * dim), 1, 1), block_dim=(SC_TPB, 1, 1),
    )
    # the component structure the driver loop walks (model structure)
    var vs = download_i32(ctx, vstart, n_comp + 1)
    var es = download_i32(ctx, estart, n_comp + 1)
    var ws = ctx.enqueue_create_buffer[DType.int32](2)
    var n_solved = 0
    var n_small = 0
    var n_fallback = 0
    var max_restarts = 0
    for c in range(n_comp):  # driver-loop: over connected components (each solved one is its own spectral solve)
        var size = Int(vs[c + 1] - vs[c])
        if not sc_component_solved(size, dim):
            n_small += 1
            continue
        var e0 = Int(es[c])
        var nnz_c = Int(es[c + 1]) - e0
        var ok = False
        var restarts_c = 0
        var d_emb = ctx.enqueue_create_buffer[DType.float32](1)
        if nnz_c > 0:
            var r_c = ctx.enqueue_create_buffer[DType.int32](nnz_c)
            var c_c = ctx.enqueue_create_buffer[DType.int32](nnz_c)
            var v_c = ctx.enqueue_create_buffer[DType.float32](nnz_c)
            var r_view = srows.create_sub_buffer[DType.int32](e0, nnz_c)
            var c_view = scols.create_sub_buffer[DType.int32](e0, nnz_c)
            var v_view = svals.create_sub_buffer[DType.float32](e0, nnz_c)
            ctx.enqueue_copy(dst_buf=r_c, src_buf=r_view)
            ctx.enqueue_copy(dst_buf=c_c, src_buf=c_view)
            ctx.enqueue_copy(dst_buf=v_c, src_buf=v_view)
            ctx.synchronize()
            _ = r_view^
            _ = c_view^
            _ = v_view^
            ok = _sc_solve(ctx, cp, size, nnz_c, r_c^, c_c^, v_c^, dim, d_emb, restarts_c)
        if restarts_c > max_restarts:
            max_restarts = restarts_c
        if ok:
            ctx.enqueue_memset(ws, Int32(0))
            ctx.enqueue_function[sc_peak_kernel](
                d_emb.unsafe_ptr(), ws.unsafe_ptr(), Int32(size * dim),
                grid_dim=(_sc_grid(size * dim), 1, 1), block_dim=(SC_TPB, 1, 1),
            )
            var peak = download_i32(ctx, ws, 2)
            ok = peak[1] == Int32(0) and peak[0] != Int32(0)
        if ok:
            ctx.enqueue_function[sc_place_kernel](
                d_emb.unsafe_ptr(), d_out.unsafe_ptr(), perm.unsafe_ptr(), ws.unsafe_ptr(), meta.unsafe_ptr(),
                drange.unsafe_ptr(), Int32(Int(vs[c])), Int32(c), Int32(size), Int32(dim),
                grid_dim=(_sc_grid(size * dim), 1, 1), block_dim=(SC_TPB, 1, 1),
            )
            ctx.synchronize()
            n_solved += 1
        else:
            n_fallback += 1
        _ = d_emb^
    # 4. the ordered post-pass over the whole embedding
    var out = _spectral_post_pass_device(ctx, d_out, n, dim)
    _ = rows^
    _ = cols^
    _ = vals^
    _ = labels^
    _ = rank^
    _ = bad^
    _ = comp^
    _ = vstart^
    _ = vkeys^
    _ = perm^
    _ = vtk^
    _ = vtv^
    _ = vcounts^
    _ = local^
    _ = estart^
    _ = ekeys^
    _ = eperm^
    _ = etk^
    _ = etv^
    _ = ecounts^
    _ = srows^
    _ = scols^
    _ = svals^
    _ = meta^
    _ = drange^
    _ = ws^
    _ = d_out^
    var msg = String("")
    if n_fallback > 0:
        msg = sc_fallback_message(n_fallback, n_comp)
    return SpectralInitReport(out^, n_comp, n_solved, n_small, n_fallback, max_restarts, msg^)


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
    """`spectral_initialize_device_coo_report`'s embedding. A fallback (a
    failed eigen solve, its rows the uniform draw) is REPORTED: its message
    goes to stderr as a warning here; a caller that can raise a Python
    warning takes the report form instead."""
    var report = spectral_initialize_device_coo_report(
        ctx, n_samples, nnz, rows^, cols^, vals^, n_components, n_neighbors, seed,
    )
    if report.n_fallback > 0:
        print("UserWarning: " + report.message, file=stderr)
    return report.embedding^


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
