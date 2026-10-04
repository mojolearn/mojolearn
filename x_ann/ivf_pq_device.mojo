# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-PQ on the device: one thread per cell of `x_ann/ivf_pq_core.mojo`.

The coarse quantizer is the same Lloyd cells over whole rows."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from x_ann.device_ctx import x_ann_ctx
from x_ann.stage_timer import AnnStages
from x_ann.switches import (
    ANN3_DIRECT_OUT, ANN3_HOST_PASSES, ANN3_PQ_SEED, ANN3_ROW_THREADS,
)
from x_ann.kpp_seed import kpp_seed
from x_ann.fast_env import FAST_IVFPQ_DEVICE_CODEBOOKS
from x_ann.pq_kmeans_device import PQK_CODES_MAX, PQK_LEN_MAX, pq_codebooks_device
from std.sys.info import has_apple_gpu_accelerator
from x_ann.ivf_scan_device import ivf_scan_search

from cluster.estimator import kmeans_fit
from cluster.impl.kmeans_params import INIT_ARRAY, INIT_KMEANS_PLUS_PLUS, METRIC_L2_EXPANDED
from ivf.estimator import ivf_flat_build_host
from ivf.impl.neighbors.ivf_flat.ivf_flat_build import ivf_trainset_rows
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_mul_add, identical_sqrt
from std.sys.compile import is_defined
from x_ann.io import upload_f32, upload_i32, download_f32, download_i32
from x_ann.refine_core import refine_cell
from x_ann.ivf_rabitq_core import rq_encode_cell, rq_pow2, rq_scale
from x_ann.ivf_sq_core import sq_encode_cell, sq_hi_takes, sq_lo_takes, sq_range_finish
from std.memory import bitcast
from x_ann.ivf_pq_core import (
    F32P, I32P, IvfPqIndex, pq_assign_cell, pq_inf, pq_insert, pq_labels_from_lists,
    pq_len_of, pq_residual_cell, pq_validate,
)
from x_ann.vsearch_fast import IVF_REFINE_TEAM

comptime TPB = 128

comptime PQ_FAST_TRAINSET = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and not is_defined["MOJOLEARN_PQ_FAST_TRAINSET_OFF"]()
"""FAST (lane ann-apple, 2026-09-28): each subspace codebook trains on at
most `PQ_FAST_ROWS_PER_CODE` rows per code, a seeded uniform sample of the
residuals (FAISS's `max_points_per_centroid` rule; the coarse quantizer
already samples the same way under FAST on Apple, `IVF_FAST_TRAINSET`). Every
row is still encoded against the trained codebooks. IDENTICAL trains on every
row. Quality: bench/speed/ann_fast_quality.py."""
comptime PQ_FAST_ROWS_PER_CODE = 256

comptime PQ_FAST_SEED = ANN3_PQ_SEED and GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
"""FAST on Apple, OPT-IN (lane ann-apple3, `-D MOJOLEARN_ANN3_PQ_SEED`): each
subspace codebook is seeded by `x_ann/kpp_seed.mojo` (host k-means++ over a
stride sample of its training rows, its own stream per subspace) and
cluster/'s k-means starts from those seeds (`INIT_ARRAY`), so it runs its
Lloyd iterations without its own seeding rounds."""


def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def residual_kernel(count: Int32, x: F32P, centers: F32P, labels: I32P, dim: Int32, rot_dim: Int32, dst: F32P):
    var e = _tid()
    if e < Int(count):
        pq_residual_cell(e, x, centers, labels, Int(dim), Int(rot_dim), dst)


def residual_rows_kernel(n: Int32, x: F32P, centers: F32P, labels: I32P, dim: Int32, rot_dim: Int32, dst: F32P):
    """`residual_kernel` with one thread per ROW (lane ann-apple3, OPT-IN
    `ANN3_ROW_THREADS`): thread i runs `pq_residual_cell` for its row's
    rot_dim cells in order. A cell reads its own inputs and writes its own
    word, so the words are the ones one thread per cell writes."""
    var i = _tid()
    if i < Int(n):
        var rd = Int(rot_dim)
        for c in range(rd):
            pq_residual_cell(i * rd + c, x, centers, labels, Int(dim), rd, dst)


def _enqueue_residual(
    ctx: DeviceContext, n: Int, dim: Int, rot_dim: Int, x: F32P, centers: F32P, labels: I32P, dst: F32P,
) raises:
    """The residual launch: one thread per row under `ANN3_ROW_THREADS`, one
    per cell otherwise."""
    comptime if ANN3_ROW_THREADS:
        ctx.enqueue_function[residual_rows_kernel](
            Int32(n), x, centers, labels, Int32(dim), Int32(rot_dim), dst, grid_dim=_grid(n), block_dim=TPB,
        )
    else:
        ctx.enqueue_function[residual_kernel](
            Int32(n * rot_dim), x, centers, labels, Int32(dim), Int32(rot_dim), dst,
            grid_dim=_grid(n * rot_dim), block_dim=TPB,
        )


def assign_kernel(count: Int32, r: F32P, cb: F32P, pq_dim: Int32, rot_dim: Int32, pq_len: Int32, n_codes: Int32, codes: I32P):
    var e = _tid()
    if e < Int(count):
        pq_assign_cell(e, r, cb, Int(pq_dim), Int(rot_dim), Int(pq_len), Int(n_codes), codes)


#: the staged encode's limits: codebook words per subspace, subspace width
comptime ASSIGN_CB_MAX = 4096
comptime ASSIGN_LEN_MAX = 16


def assign_staged_kernel(
    n: Int32, r: F32P, cb: F32P, pq_dim: Int32, rot_dim: Int32, pq_len: Int32, n_codes: Int32, codes: I32P
):
    """`pq_assign_cell` for rows block_idx.x * TPB + t of subspace j =
    block_idx.y, with subspace j's codebook staged once per threadgroup as
    ftz(cb) and the row's residual as ftz(r) in registers (lane ann-apple2):
    `pq_subdist`'s `ftz(ftz(a) - ftz(b))` and fused fold on the same words
    (ftz is idempotent), codes ascending, the strict `<`: the same code."""
    var t = Int(thread_idx.x)
    var j = Int(block_idx.y)
    var i = Int(block_idx.x) * TPB + t
    var pl = Int(pq_len)
    var nc = Int(n_codes)
    var per = nc * pl
    var tile = stack_allocation[ASSIGN_CB_MAX, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    for e in range(t, per, TPB):
        tile[e] = ftz(cb.unsafe_load(j * per + e))
    barrier()
    if i >= Int(n):
        return
    var rv = InlineArray[Float32, ASSIGN_LEN_MAX](fill=Float32(0.0))
    var off = i * Int(rot_dim) + j * pl
    for u in range(pl):
        rv[u] = ftz(r.unsafe_load(off + u))
    var best = 0
    var bd = Float32(0.0)
    for c in range(nc):
        var acc = Float32(0.0)
        for u in range(pl):
            var diff = ftz(rv[u] - tile[c * pl + u])
            acc = ftz(identical_mul_add(diff, diff, acc))
        if c == 0:
            bd = acc
        elif acc < bd:
            bd = acc
            best = c
    codes.unsafe_store(i * Int(pq_dim) + j, Int32(best))


def _enqueue_assign(
    ctx: DeviceContext, n: Int, r: F32P, cb: F32P, pq_dim: Int, rot_dim: Int, pq_len: Int, n_codes: Int,
    codes: I32P,
) raises:
    """The encode launch: the staged kernel when the subspace codebook fits
    (`-D MOJOLEARN_PQ_ASSIGN_UNSTAGED` keeps the one-thread-per-cell kernel)."""
    comptime if not is_defined["MOJOLEARN_PQ_ASSIGN_UNSTAGED"]():
        if n_codes * pq_len <= ASSIGN_CB_MAX and pq_len <= ASSIGN_LEN_MAX:
            ctx.enqueue_function[assign_staged_kernel](
                Int32(n), r, cb, Int32(pq_dim), Int32(rot_dim), Int32(pq_len), Int32(n_codes), codes,
                grid_dim=((n + TPB - 1) // TPB, pq_dim), block_dim=TPB,
            )
            return
    ctx.enqueue_function[assign_kernel](
        Int32(n * pq_dim), r, cb, Int32(pq_dim), Int32(rot_dim), Int32(pq_len), Int32(n_codes), codes,
        grid_dim=_grid(n * pq_dim), block_dim=TPB,
    )


def _dp[dt: DType](mut b: DeviceBuffer[dt]) -> MutPointer[Scalar[dt], MutAnyOrigin]:
    """A device buffer's address as the kernels' pointer type."""
    return b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _grid(count: Int) -> Int:
    return (count + TPB - 1) // TPB


def _coarse(
    x: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    mut centers: List[Float32], mut offsets: List[Int32], mut list_indices: List[Int32],
    mut labels: List[Int32],
) raises:
    """The coarse quantizer IS IVF-Flat's build (`ivf/estimator.mojo::
    ivf_flat_build_host`: cluster/'s k-means, L2Expanded, its CSR lists);
    the host twin is `ivf/host/ivf_host.mojo::host_ivf_build`."""
    var cst = AnnStages("ivf_coarse")
    var ctx = x_ann_ctx()
    # lane ann-apple3, behind `ANN3_HOST_PASSES`: the build without the
    # permuted vectors (no x_ann index reads them; `with_list_data` is read
    # only under the switch); the centres and the offsets move out of it;
    # the labels are the build's own assignment, the one its lists were laid
    # out from, so `pq_labels_from_lists` over those lists returns these words
    var flat = ivf_flat_build_host(
        ctx, x, n, dim, n_lists, kmeans_n_iters, METRIC_L2_EXPANDED, UInt64(seed), with_list_data=False
    )
    ctx.synchronize()
    cst.host("flat_build")
    comptime if ANN3_HOST_PASSES:
        swap(centers, flat.centers)
        swap(offsets, flat.list_offsets)
        list_indices = List[Int32](length=n, fill=Int32(0))
        labels = List[Int32](length=n, fill=Int32(0))
        for s in range(n):
            list_indices[s] = Int32(Int(flat.list_indices[s]))
            labels[s] = Int32(Int(flat.labels[s]))
    else:
        centers = flat.centers.copy()
        offsets = flat.list_offsets.copy()
        list_indices = List[Int32](capacity=n)
        for s in range(n):
            list_indices.append(Int32(Int(flat.list_indices[s])))
        labels = pq_labels_from_lists(offsets, list_indices, n_lists, n)
    _ = flat^
    _ = ctx^
    cst.host("convert")


def _codebooks(
    r: List[Float32], n: Int, rot_dim: Int, pq_dim: Int, pq_len: Int, n_codes: Int, pq_iters: Int, seed: Int,
) raises -> List[Float32]:
    """Per subspace, cluster/'s k-means (`cluster/estimator.mojo::kmeans_fit`,
    k-means++, L2Expanded, one restart) over that subspace's residual
    columns; the host twin is `host_kmeans_fit`."""
    var codebooks = List[Float32](capacity=pq_dim * n_codes * pq_len)
    var ctx = x_ann_ctx()
    var n_train = n
    comptime if PQ_FAST_TRAINSET:
        if n > PQ_FAST_ROWS_PER_CODE * n_codes:
            n_train = PQ_FAST_ROWS_PER_CODE * n_codes
    var rows = List[Int]()
    if n_train < n:
        rows = ivf_trainset_rows(n, n_train, UInt64(seed))
    else:
        for i in range(n):
            rows.append(i)
    var cbs = AnnStages("ivf_pq_codebooks")
    cbs.host("rows")
    for j in range(pq_dim):
        var sub = List[Float32](capacity=n_train * pq_len)
        for i in range(n_train):
            for t in range(pq_len):
                sub.append(r[rows[i] * rot_dim + j * pq_len + t])
        var cb = List[Float32](length=n_codes * pq_len, fill=Float32(0.0))
        var lab = List[UInt32](length=n_train, fill=UInt32(0))
        cbs.host("gather")
        var init_kind: Int = INIT_KMEANS_PLUS_PLUS
        comptime if PQ_FAST_SEED:
            if n_train >= n_codes:
                kpp_seed(
                    sub, n_train, pq_len, n_codes,
                    UInt64(seed) ^ (UInt64(j + 1) * UInt64(0x9E3779B97F4A7C15)), 16, cb,
                )
                init_kind = INIT_ARRAY
                cbs.host("seed")
        _ = kmeans_fit(
            ctx, sub.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), n_train, pq_len, n_codes,
            cb.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            lab.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            sub.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), 0,
            pq_iters, Float64(1e-4), UInt64(seed), 1, init_kind, METRIC_L2_EXPANDED, 0.0, Float64(2.0),
        )
        ctx.synchronize()
        cbs.host("kmeans_fit")
        for e in range(n_codes * pq_len):
            codebooks.append(cb[e])
        _ = sub^
        _ = lab^
    _ = ctx^
    return codebooks^


def ivf_pq_build_device(
    x: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    pq_dim: Int, pq_bits: Int, pq_iters: Int, codes_addr: Int = 0,
) raises -> IvfPqIndex:
    """`codes_addr` (lane ann-apple3, read under `ANN3_DIRECT_OUT` only): the
    address of the caller's n x pq_dim int32 array; the codes are downloaded
    straight into it and the returned index's `codes` is EMPTY. 0: the codes
    come back in the index, as before."""
    pq_validate(n, dim, n_lists, pq_dim, pq_bits, pq_iters)
    var pq_len = pq_len_of(dim, pq_dim)
    var rot_dim = pq_len * pq_dim
    var n_codes = 1 << pq_bits
    var centers = List[Float32]()
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    var labels = List[Int32]()
    var st = AnnStages("ivf_pq_build")
    _coarse(x, n, dim, n_lists, kmeans_n_iters, seed, centers, offsets, list_indices, labels)
    st.host("coarse")
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    var dc = upload_f32(ctx, centers)
    var dl = upload_i32(ctx, labels)
    var dr = ctx.enqueue_create_buffer[DType.float32](n * rot_dim)
    _enqueue_residual(ctx, n, dim, rot_dim, _dp(dx), _dp(dc), _dp(dl), _dp(dr))
    ctx.synchronize()
    # FAST on Apple, default (off: `-D MOJOLEARN_IVFPQ_FAST_DEVICE_CODEBOOKS_OFF`; lane/apple-
    # fast-ann, 2026-10-02; x_ann/pq_kmeans_device.mojo): the codebooks of
    # every subspace from one batched device Lloyd loop over the residuals
    # already in `dr`, on the same sample size as `_codebooks` (a stride
    # sample). Cause: `_codebooks` downloads the n x rot_dim residuals,
    # gathers each subspace's sample on the host and runs cluster/'s
    # `kmeans_fit` once per subspace in series (55 host-driven fits on
    # Istella), each with its seeding rounds and `synchronize`s. The
    # device codebooks are downloaded once for the index; the encode reads
    # them where they are. Moves FAST bits: paired recall check.
    var dev_cb = False
    comptime if FAST_IVFPQ_DEVICE_CODEBOOKS:
        dev_cb = pq_len <= PQK_LEN_MAX and n_codes <= PQK_CODES_MAX
    var codebooks = List[Float32]()
    var dcb: DeviceBuffer[DType.float32]
    if dev_cb:
        var n_train = n
        comptime if PQ_FAST_TRAINSET:
            if n > PQ_FAST_ROWS_PER_CODE * n_codes:
                n_train = PQ_FAST_ROWS_PER_CODE * n_codes
        dcb = pq_codebooks_device(ctx, dr, n, n_train, rot_dim, pq_dim, pq_len, n_codes, pq_iters, seed)
        codebooks = download_f32(ctx, dcb, pq_dim * n_codes * pq_len)
        st.host("codebooks")
    else:
        var r = download_f32(ctx, dr, n * rot_dim)
        st.host("residuals")
        codebooks = _codebooks(r, n, rot_dim, pq_dim, pq_len, n_codes, pq_iters, seed)
        st.host("codebooks")
        dcb = upload_f32(ctx, codebooks)
    var dcodes = ctx.enqueue_create_buffer[DType.int32](n * pq_dim)
    _enqueue_assign(ctx, n, _dp(dr), _dp(dcb), pq_dim, rot_dim, pq_len, n_codes, _dp(dcodes))
    ctx.synchronize()
    var codes = List[Int32]()
    var direct = False
    comptime if ANN3_DIRECT_OUT:
        direct = codes_addr != 0
    if direct:
        ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=codes_addr), src_buf=dcodes)
        ctx.synchronize()
    else:
        codes = download_i32(ctx, dcodes, n * pq_dim)
    st.host("encode")
    _ = dcodes^
    _ = dcb^
    _ = dr^
    _ = dl^
    _ = dc^
    _ = dx^
    _ = ctx^
    return IvfPqIndex(n_lists, dim, n, pq_dim, pq_len, n_codes, centers^, offsets^, list_indices^, codebooks^, codes^)


def ivf_pq_search_device(
    centers: List[Float32], offsets: List[Int32], list_indices: List[Int32], codebooks: List[Float32],
    codes: List[Int32], mask: List[Int32], n_lists: Int, dim: Int, pq_dim: Int, pq_bits: Int,
    queries: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
) raises:
    var sst = AnnStages("ivf_search")
    var ctx = x_ann_ctx()
    var dc = upload_f32(ctx, centers)
    var doff = upload_i32(ctx, offsets)
    var dli = upload_i32(ctx, list_indices)
    var dcodes = upload_i32(ctx, codes)
    var dcb = upload_f32(ctx, codebooks)
    var dmask = upload_i32(ctx, mask)
    sst.mark(ctx, "upload")
    ivf_pq_search_on(
        ctx, _dp(dc), _dp(doff), _dp(dli), _dp(dcb), _dp(dcodes), _dp(dmask), offsets, n_lists, dim,
        pq_dim, pq_bits, queries, m, k, n_probes, out_d, out_i, out_n,
        False, _dp(dcodes), _dp(dc), _dp(dc), False,
    )
    _ = dmask^
    _ = dcb^
    _ = dcodes^
    _ = dli^
    _ = doff^
    _ = dc^
    _ = ctx^


def ivf_pq_search_on(
    ctx: DeviceContext, dc: F32P, doff: I32P, dli: I32P, dcb: F32P, dcodes: I32P, dmask: I32P,
    offsets: List[Int32], n_lists: Int, dim: Int, pq_dim: Int, pq_bits: Int,
    queries: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
    have_pre: Bool, pre_codes: I32P, pre_a: F32P, pre_b: F32P, mask_pre: Bool,
) raises:
    """The search over an index already on the device (`ivf_pq_search_device`
    uploads it first; `x_ann/resident.mojo` holds it): the queries up, the
    scan, the three outputs down. `have_pre` ... `mask_pre`: the resident
    index's list-order arrays (`ivf_scan_search`); `mask_pre` says dmask is
    the all-ones filter, which is its own list-order copy."""
    var pq_len = pq_len_of(dim, pq_dim)
    var n_codes = 1 << pq_bits
    var dq = upload_f32(ctx, queries)
    var dd = ctx.enqueue_create_buffer[DType.float32](m * k)
    var di = ctx.enqueue_create_buffer[DType.int32](m * k)
    var dn = ctx.enqueue_create_buffer[DType.int32](m)
    # lane ann-apple: the split scan (x_ann/ivf_scan_device.mojo), the same
    # bits as the old one thread per query running `pq_search_cell`
    ivf_scan_search[0](
        ctx, _dp(dq), dc, doff, dli, dcodes, dmask, dcb, dcb,
        offsets, n_lists, dim, m, k, n_probes, pq_dim, pq_len, n_codes, 1, 1, 0, Float32(1.0), _dp(dd), _dp(di), _dp(dn),
        have_pre, pre_codes, pre_a, pre_b, mask_pre, dmask,
    )
    out_d = download_f32(ctx, dd, m * k)
    out_i = download_i32(ctx, di, m * k)
    out_n = download_i32(ctx, dn, m)
    _ = dn^
    _ = di^
    _ = dd^
    _ = dq^


#: rows per partial of the SQ range (lane ann-apple)
comptime SQ_RANGE_ROWS = 1024


@always_inline
def _is_nan(v: Float32) -> Bool:
    var b = bitcast[DType.uint32](v)
    return (b & UInt32(0x7F800000)) == UInt32(0x7F800000) and (b & UInt32(0x007FFFFF)) != UInt32(0)


def sq_range_part_kernel(count: Int32, r: F32P, n: Int32, dim: Int32, lo_p: F32P, hi_p: F32P, has_p: I32P):
    """One thread per (row chunk g, column c), e = g * dim + c: the chunk's
    minimum and maximum over its non-NaN values under `sq_lo_takes` /
    `sq_hi_takes` in row order (the first of equal values kept)."""
    var e = _tid()
    if e < Int(count):
        var d = Int(dim)
        var g = e // d
        var c = e % d
        var i0 = g * SQ_RANGE_ROWS
        var i1 = i0 + SQ_RANGE_ROWS
        if i1 > Int(n):
            i1 = Int(n)
        var have = False
        var lo = Float32(0.0)
        var hi = Float32(0.0)
        for i in range(i0, i1):
            var v = ftz(r.unsafe_load(i * d + c))
            if _is_nan(v):
                continue
            if not have:
                lo = v
                hi = v
                have = True
            else:
                if sq_lo_takes(v, lo):
                    lo = v
                if sq_hi_takes(v, hi):
                    hi = v
        lo_p.unsafe_store(e, lo)
        hi_p.unsafe_store(e, hi)
        has_p.unsafe_store(e, Int32(1) if have else Int32(0))


def sq_range_join_kernel(
    dim: Int32, r: F32P, n_parts: Int32, lo_p: F32P, hi_p: F32P, has_p: I32P, vmin: F32P, delta: F32P
):
    """Column c: `sq_range_cell`'s result from the partials in row order. The
    cell starts from row 0 and never lets a NaN replace a number, so a NaN in
    row 0 stays, and otherwise the running values are the non-NaN minimum and
    maximum, the first of equal values kept, which is the partials joined in
    order under the same comparisons. The same words."""
    var c = _tid()
    if c < Int(dim):
        var d = Int(dim)
        var lo = ftz(r.unsafe_load(c))
        var hi = lo
        if not _is_nan(lo):
            for g in range(Int(n_parts)):
                if has_p.unsafe_load(g * d + c) != 0:
                    var pl = lo_p.unsafe_load(g * d + c)
                    var ph = hi_p.unsafe_load(g * d + c)
                    if sq_lo_takes(pl, lo):
                        lo = pl
                    if sq_hi_takes(ph, hi):
                        hi = ph
        sq_range_finish(c, lo, hi, vmin, delta)


def _sq_range_enqueue(
    ctx: DeviceContext, mut dr: DeviceBuffer[DType.float32], n: Int, dim: Int,
    mut dvmin: DeviceBuffer[DType.float32], mut ddelta: DeviceBuffer[DType.float32],
) raises:
    """`sq_range_cell` for every column as chunk partials, then the join."""
    var parts = (n + SQ_RANGE_ROWS - 1) // SQ_RANGE_ROWS
    var lo_p = ctx.enqueue_create_buffer[DType.float32](parts * dim)
    var hi_p = ctx.enqueue_create_buffer[DType.float32](parts * dim)
    var has_p = ctx.enqueue_create_buffer[DType.int32](parts * dim)
    ctx.enqueue_function[sq_range_part_kernel](
        Int32(parts * dim), dr.unsafe_ptr(), Int32(n), Int32(dim), lo_p.unsafe_ptr(), hi_p.unsafe_ptr(),
        has_p.unsafe_ptr(), grid_dim=_grid(parts * dim), block_dim=TPB,
    )
    ctx.enqueue_function[sq_range_join_kernel](
        Int32(dim), dr.unsafe_ptr(), Int32(parts), lo_p.unsafe_ptr(), hi_p.unsafe_ptr(), has_p.unsafe_ptr(),
        dvmin.unsafe_ptr(), ddelta.unsafe_ptr(), grid_dim=_grid(dim), block_dim=TPB,
    )
    ctx.synchronize()
    _ = has_p^
    _ = hi_p^
    _ = lo_p^


def sq_encode_kernel(count: Int32, r: F32P, dim: Int32, vmin: F32P, delta: F32P, codes: I32P):
    var e = _tid()
    if e < Int(count):
        sq_encode_cell(e, r, Int(dim), vmin, delta, codes)


def sq_encode_rows_kernel(n: Int32, r: F32P, dim: Int32, vmin: F32P, delta: F32P, codes: I32P):
    """`sq_encode_kernel` with one thread per ROW (lane ann-apple3, OPT-IN
    `ANN3_ROW_THREADS`): the row's dim cells in order, each `sq_encode_cell`."""
    var i = _tid()
    if i < Int(n):
        var d = Int(dim)
        for c in range(d):
            sq_encode_cell(i * d + c, r, d, vmin, delta, codes)


def ivf_sq_build_device(
    x: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    mut centers: List[Float32], mut offsets: List[Int32], mut list_indices: List[Int32],
    mut vmin: List[Float32], mut delta: List[Float32], mut codes: List[Int32], codes_addr: Int = 0,
) raises:
    """`codes_addr`: `ivf_pq_build_device`'s, for the n x dim int32 codes
    (`codes` is then left EMPTY)."""
    pq_validate(n, dim, n_lists, 1, 1, 1)
    var labels = List[Int32]()
    var st = AnnStages("ivf_sq_build")
    _coarse(x, n, dim, n_lists, kmeans_n_iters, seed, centers, offsets, list_indices, labels)
    st.host("coarse")
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    var dc = upload_f32(ctx, centers)
    var dl = upload_i32(ctx, labels)
    var dr = ctx.enqueue_create_buffer[DType.float32](n * dim)
    var dvmin = ctx.enqueue_create_buffer[DType.float32](dim)
    var ddelta = ctx.enqueue_create_buffer[DType.float32](dim)
    var dcodes = ctx.enqueue_create_buffer[DType.int32](n * dim)
    _enqueue_residual(ctx, n, dim, dim, _dp(dx), _dp(dc), _dp(dl), _dp(dr))
    st.mark(ctx, "upload_residuals")
    _sq_range_enqueue(ctx, dr, n, dim, dvmin, ddelta)
    st.mark(ctx, "range")
    comptime if ANN3_ROW_THREADS:
        ctx.enqueue_function[sq_encode_rows_kernel](Int32(n), dr.unsafe_ptr(), Int32(dim), dvmin.unsafe_ptr(),
                                                    ddelta.unsafe_ptr(), dcodes.unsafe_ptr(), grid_dim=_grid(n),
                                                    block_dim=TPB)
    else:
        ctx.enqueue_function[sq_encode_kernel](Int32(n * dim), dr.unsafe_ptr(), Int32(dim), dvmin.unsafe_ptr(),
                                               ddelta.unsafe_ptr(), dcodes.unsafe_ptr(), grid_dim=_grid(n * dim),
                                               block_dim=TPB)
    ctx.synchronize()
    vmin = download_f32(ctx, dvmin, dim)
    delta = download_f32(ctx, ddelta, dim)
    var direct = False
    comptime if ANN3_DIRECT_OUT:
        direct = codes_addr != 0
    if direct:
        codes = List[Int32]()
        ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=codes_addr), src_buf=dcodes)
        ctx.synchronize()
    else:
        codes = download_i32(ctx, dcodes, n * dim)
    st.host("encode")
    _ = dcodes^
    _ = ddelta^
    _ = dvmin^
    _ = dr^
    _ = dl^
    _ = dc^
    _ = dx^
    _ = ctx^


def ivf_sq_search_device(
    centers: List[Float32], offsets: List[Int32], list_indices: List[Int32], vmin: List[Float32],
    delta: List[Float32], codes: List[Int32], mask: List[Int32], n_lists: Int, dim: Int,
    queries: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
) raises:
    var sst = AnnStages("ivf_search")
    var ctx = x_ann_ctx()
    var dc = upload_f32(ctx, centers)
    var doff = upload_i32(ctx, offsets)
    var dli = upload_i32(ctx, list_indices)
    var dcodes = upload_i32(ctx, codes)
    var dvmin = upload_f32(ctx, vmin)
    var ddelta = upload_f32(ctx, delta)
    var dmask = upload_i32(ctx, mask)
    sst.mark(ctx, "upload")
    ivf_sq_search_on(
        ctx, _dp(dc), _dp(doff), _dp(dli), _dp(dvmin), _dp(ddelta), _dp(dcodes), _dp(dmask), offsets,
        n_lists, dim, queries, m, k, n_probes, out_d, out_i, out_n,
        False, _dp(dcodes), _dp(dc), _dp(dc), False,
    )
    _ = dmask^
    _ = ddelta^
    _ = dvmin^
    _ = dcodes^
    _ = dli^
    _ = doff^
    _ = dc^
    _ = ctx^


def ivf_sq_search_on(
    ctx: DeviceContext, dc: F32P, doff: I32P, dli: I32P, dvmin: F32P, ddelta: F32P, dcodes: I32P,
    dmask: I32P, offsets: List[Int32], n_lists: Int, dim: Int,
    queries: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
    have_pre: Bool, pre_codes: I32P, pre_a: F32P, pre_b: F32P, mask_pre: Bool,
) raises:
    """`ivf_pq_search_on`'s SQ twin: the index already on the device."""
    var dq = upload_f32(ctx, queries)
    var dd = ctx.enqueue_create_buffer[DType.float32](m * k)
    var di = ctx.enqueue_create_buffer[DType.int32](m * k)
    var dn = ctx.enqueue_create_buffer[DType.int32](m)
    ivf_scan_search[1](
        ctx, _dp(dq), dc, doff, dli, dcodes, dmask, dvmin, ddelta,
        offsets, n_lists, dim, m, k, n_probes, 1, 1, 1, 1, 1, 0, Float32(1.0), _dp(dd), _dp(di), _dp(dn),
        have_pre, pre_codes, pre_a, pre_b, mask_pre, dmask,
    )
    out_d = download_f32(ctx, dd, m * k)
    out_i = download_i32(ctx, di, m * k)
    out_n = download_i32(ctx, dn, m)
    _ = dn^
    _ = di^
    _ = dd^
    _ = dq^


def refine_kernel(m: Int32, x: F32P, n: Int32, d: Int32, queries: F32P, cand: I32P, k0: Int32, k: Int32,
                  out_d: F32P, out_i: I32P, root: Int32):
    var q = _tid()
    if q < Int(m):
        refine_cell(q, x, Int(n), Int(d), queries, cand, Int(k0), Int(k), out_d, out_i, root != 0)


#: the refine team (lane af-vsearch): threads per query (candidates at most),
#: the widest query row staged in threadgroup memory
comptime REFINE_T = 128
comptime REFINE_DIM_MAX = 512


def refine_team_kernel(
    x: F32P, n: Int32, d: Int32, queries: F32P, cand: I32P, k0: Int32, k: Int32, out_d: F32P, out_i: I32P,
    root: Int32,
):
    """FAST on Apple, OPT-IN (lane af-vsearch, `IVF_REFINE_TEAM`): `refine_cell`
    with one threadgroup of REFINE_T per query (k0 <= REFINE_T, d <=
    REFINE_DIM_MAX; the launch checks). The query row is staged in
    threadgroup memory as ftz(q); thread t < k0 scores candidate t by the
    cell's statements (a padding id or a repeat of an earlier slot is skipped;
    the ascending fused square sum of `ftz(ftz(q) - ftz(x))`, the same words)
    into threadgroup memory; thread 0 runs the cell's `pq_insert` over the
    scored slots in slot order. The same words in the same insertion order:
    the same result."""
    var qi = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var dd = Int(d)
    var kk0 = Int(k0)
    var kk = Int(k)
    var sq = stack_allocation[REFINE_DIM_MAX, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var sd = stack_allocation[REFINE_T, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var sv = stack_allocation[REFINE_T, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    for c in range(t, dd, REFINE_T):
        sq[c] = ftz(queries.unsafe_load(qi * dd + c))
    barrier()
    if t < kk0:
        var v = Int(cand.unsafe_load(qi * kk0 + t))
        var keep = v >= 0 and v < Int(n)
        if keep:
            for u in range(t):
                if Int(cand.unsafe_load(qi * kk0 + u)) == v:
                    keep = False
                    break
        var acc = Float32(0.0)
        if keep:
            for c in range(dd):
                var diff = ftz(sq[c] - ftz(x.unsafe_load(v * dd + c)))
                acc = ftz(identical_mul_add(diff, diff, acc))
        sd[t] = acc
        sv[t] = Int32(v) if keep else Int32(-1)
    barrier()
    if t == 0:
        var base = qi * kk
        for s in range(kk):
            out_d.unsafe_store(base + s, pq_inf())
            out_i.unsafe_store(base + s, Int32(-1))
        for u in range(kk0):
            var id = sv[u]
            if id >= 0:
                pq_insert(kk, base, sd[u], id, out_d, out_i)
        if root != 0:  # refine_cell's `root` rule (main, lane apple-fast-py2mojo-cluster)
            for s in range(kk):
                out_d.unsafe_store(base + s, identical_sqrt(out_d.unsafe_load(base + s)))


def refine_device_team(
    x_addr: Int, n: Int, d: Int, queries: List[Float32], m: Int, cand: List[Int32], k0: Int, k: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], root: Bool = False,
) raises:
    """FAST on Apple, OPT-IN (lane af-vsearch, `IVF_REFINE_TEAM`): `refine_device`
    with the dataset uploaded straight from the caller's n x d float32 array
    at `x_addr` (no host copy into a List first; the binding holds the array
    under the released GIL for the whole call) and `refine_team_kernel` when
    it applies (k0 <= REFINE_T, d <= REFINE_DIM_MAX), `refine_kernel`
    otherwise. The same words either way."""
    var ctx = x_ann_ctx()
    var dx = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=F32P(unsafe_from_address=x_addr))
    ctx.synchronize()
    var dq = upload_f32(ctx, queries)
    var dcand = upload_i32(ctx, cand)
    var dd = ctx.enqueue_create_buffer[DType.float32](m * k)
    var di = ctx.enqueue_create_buffer[DType.int32](m * k)
    if k0 <= REFINE_T and d <= REFINE_DIM_MAX:
        ctx.enqueue_function[refine_team_kernel](
            dx.unsafe_ptr(), Int32(n), Int32(d), dq.unsafe_ptr(), dcand.unsafe_ptr(), Int32(k0), Int32(k),
            dd.unsafe_ptr(), di.unsafe_ptr(), Int32(1) if root else Int32(0), grid_dim=m, block_dim=REFINE_T,
        )
    else:
        ctx.enqueue_function[refine_kernel](Int32(m), dx.unsafe_ptr(), Int32(n), Int32(d), dq.unsafe_ptr(),
                                            dcand.unsafe_ptr(), Int32(k0), Int32(k), dd.unsafe_ptr(), di.unsafe_ptr(),
                                            Int32(1) if root else Int32(0), grid_dim=_grid(m), block_dim=TPB)
    ctx.synchronize()
    out_d = download_f32(ctx, dd, m * k)
    out_i = download_i32(ctx, di, m * k)
    _ = di^
    _ = dd^
    _ = dcand^
    _ = dq^
    _ = dx^
    _ = ctx^


def refine_device(
    x: List[Float32], n: Int, d: Int, queries: List[Float32], m: Int, cand: List[Int32], k0: Int, k: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], root: Bool = False,
) raises:
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    var dq = upload_f32(ctx, queries)
    var dcand = upload_i32(ctx, cand)
    var dd = ctx.enqueue_create_buffer[DType.float32](m * k)
    var di = ctx.enqueue_create_buffer[DType.int32](m * k)
    ctx.enqueue_function[refine_kernel](Int32(m), dx.unsafe_ptr(), Int32(n), Int32(d), dq.unsafe_ptr(),
                                        dcand.unsafe_ptr(), Int32(k0), Int32(k), dd.unsafe_ptr(), di.unsafe_ptr(),
                                        Int32(1) if root else Int32(0), grid_dim=_grid(m), block_dim=TPB)
    ctx.synchronize()
    out_d = download_f32(ctx, dd, m * k)
    out_i = download_i32(ctx, di, m * k)
    _ = di^
    _ = dd^
    _ = dcand^
    _ = dq^
    _ = dx^
    _ = ctx^


def rq_encode_kernel(
    n: Int32, x: F32P, centers: F32P, labels: I32P, dim: Int32, D: Int32, seed: Int32, scale: Float32,
    ws: F32P, words: Int32, codes: I32P, norms: F32P, ips: F32P,
):
    var i = _tid()
    if i < Int(n):
        rq_encode_cell(i, x, centers, labels, Int(dim), Int(D), Int(seed), scale, ws, Int(words), codes, norms, ips)


def ivf_rabitq_build_device(
    x: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    mut centers: List[Float32], mut offsets: List[Int32], mut list_indices: List[Int32],
    mut codes: List[Int32], mut norms: List[Float32], mut ips: List[Float32],
) raises:
    pq_validate(n, dim, n_lists, 1, 1, 1)
    var D = rq_pow2(dim)
    var words = (D + 31) // 32
    var scale = rq_scale(D)
    var labels = List[Int32]()
    var st = AnnStages("ivf_rabitq_build")
    _coarse(x, n, dim, n_lists, kmeans_n_iters, seed, centers, offsets, list_indices, labels)
    st.host("coarse")
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    var dc = upload_f32(ctx, centers)
    var dl = upload_i32(ctx, labels)
    var dws = ctx.enqueue_create_buffer[DType.float32](n * D)
    var dcodes = ctx.enqueue_create_buffer[DType.int32](n * words)
    var dnorm = ctx.enqueue_create_buffer[DType.float32](n)
    var dip = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_function[rq_encode_kernel](
        Int32(n), dx.unsafe_ptr(), dc.unsafe_ptr(), dl.unsafe_ptr(), Int32(dim), Int32(D), Int32(seed), scale,
        dws.unsafe_ptr(), Int32(words), dcodes.unsafe_ptr(), dnorm.unsafe_ptr(), dip.unsafe_ptr(),
        grid_dim=_grid(n), block_dim=TPB,
    )
    ctx.synchronize()
    codes = download_i32(ctx, dcodes, n * words)
    norms = download_f32(ctx, dnorm, n)
    ips = download_f32(ctx, dip, n)
    st.host("encode")
    _ = dip^
    _ = dnorm^
    _ = dcodes^
    _ = dws^
    _ = dl^
    _ = dc^
    _ = dx^
    _ = ctx^


def ivf_rabitq_search_device(
    centers: List[Float32], offsets: List[Int32], list_indices: List[Int32], codes: List[Int32],
    norms: List[Float32], ips: List[Float32], mask: List[Int32], n_lists: Int, dim: Int, seed: Int,
    queries: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
) raises:
    var sst = AnnStages("ivf_search")
    var ctx = x_ann_ctx()
    var dc = upload_f32(ctx, centers)
    var doff = upload_i32(ctx, offsets)
    var dli = upload_i32(ctx, list_indices)
    var dcodes = upload_i32(ctx, codes)
    var dnorm = upload_f32(ctx, norms)
    var dip = upload_f32(ctx, ips)
    var dmask = upload_i32(ctx, mask)
    sst.mark(ctx, "upload")
    ivf_rabitq_search_on(
        ctx, _dp(dc), _dp(doff), _dp(dli), _dp(dcodes), _dp(dnorm), _dp(dip), _dp(dmask), offsets,
        n_lists, dim, seed, queries, m, k, n_probes, out_d, out_i, out_n,
        False, _dp(dcodes), _dp(dnorm), _dp(dip), False,
    )
    _ = dmask^
    _ = dip^
    _ = dnorm^
    _ = dcodes^
    _ = dli^
    _ = doff^
    _ = dc^
    _ = ctx^


def ivf_rabitq_search_on(
    ctx: DeviceContext, dc: F32P, doff: I32P, dli: I32P, dcodes: I32P, dnorm: F32P, dip: F32P,
    dmask: I32P, offsets: List[Int32], n_lists: Int, dim: Int, seed: Int,
    queries: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
    have_pre: Bool, pre_codes: I32P, pre_a: F32P, pre_b: F32P, mask_pre: Bool,
) raises:
    """`ivf_pq_search_on`'s RaBitQ twin: the index already on the device."""
    var D = rq_pow2(dim)
    var words = (D + 31) // 32
    var scale = rq_scale(D)
    var dq = upload_f32(ctx, queries)
    var dd = ctx.enqueue_create_buffer[DType.float32](m * k)
    var di = ctx.enqueue_create_buffer[DType.int32](m * k)
    var dn = ctx.enqueue_create_buffer[DType.int32](m)
    ivf_scan_search[2](
        ctx, _dp(dq), dc, doff, dli, dcodes, dmask, dnorm, dip,
        offsets, n_lists, dim, m, k, n_probes, 1, 1, 1, D, words, seed, scale, _dp(dd), _dp(di), _dp(dn),
        have_pre, pre_codes, pre_a, pre_b, mask_pre, dmask,
    )
    out_d = download_f32(ctx, dd, m * k)
    out_i = download_i32(ctx, di, m * k)
    out_n = download_i32(ctx, dn, m)
    _ = dn^
    _ = di^
    _ = dd^
    _ = dq^


def pq_encode_device(
    r: List[Float32], cb: List[Float32], n: Int, pq_dim: Int, pq_len: Int, n_codes: Int,
) raises -> List[Int32]:
    """The encoding launch alone over given residuals and codebooks (the
    DEVIATION 5801 check plants duplicate codewords through it)."""
    var rot_dim = pq_dim * pq_len
    var ctx = x_ann_ctx()
    var dr = upload_f32(ctx, r)
    var dcb = upload_f32(ctx, cb)
    var dcodes = ctx.enqueue_create_buffer[DType.int32](n * pq_dim)
    _enqueue_assign(ctx, n, _dp(dr), _dp(dcb), pq_dim, rot_dim, pq_len, n_codes, _dp(dcodes))
    ctx.synchronize()
    var codes = download_i32(ctx, dcodes, n * pq_dim)
    _ = dcodes^
    _ = dcb^
    _ = dr^
    _ = ctx^
    return codes^


def sq_range_device(r: List[Float32], n: Int, dim: Int, mut vmin: List[Float32], mut delta: List[Float32]) raises:
    """The SQ range launch alone over given residuals (the 5830 check)."""
    var ctx = x_ann_ctx()
    var dr = upload_f32(ctx, r)
    var dv = ctx.enqueue_create_buffer[DType.float32](dim)
    var dd = ctx.enqueue_create_buffer[DType.float32](dim)
    _sq_range_enqueue(ctx, dr, n, dim, dv, dd)
    ctx.synchronize()
    vmin = download_f32(ctx, dv, dim)
    delta = download_f32(ctx, dd, dim)
    _ = dd^
    _ = dv^
    _ = dr^
    _ = ctx^


def sq_encode_given_device(
    r: List[Float32], n: Int, dim: Int, vmin: List[Float32], delta: List[Float32],
) raises -> List[Int32]:
    """The SQ encoding launch alone with a given range (the 5831 check)."""
    var ctx = x_ann_ctx()
    var dr = upload_f32(ctx, r)
    var dv = upload_f32(ctx, vmin)
    var dd = upload_f32(ctx, delta)
    var dc = ctx.enqueue_create_buffer[DType.int32](n * dim)
    ctx.enqueue_function[sq_encode_kernel](Int32(n * dim), dr.unsafe_ptr(), Int32(dim), dv.unsafe_ptr(),
                                           dd.unsafe_ptr(), dc.unsafe_ptr(), grid_dim=_grid(n * dim), block_dim=TPB)
    ctx.synchronize()
    var codes = download_i32(ctx, dc, n * dim)
    _ = dc^
    _ = dd^
    _ = dv^
    _ = dr^
    _ = ctx^
    return codes^


def rq_encode_given_device(
    x: List[Float32], n: Int, dim: Int, centers: List[Float32], labels: List[Int32], seed: Int,
    mut codes: List[Int32], mut norms: List[Float32], mut ips: List[Float32],
) raises:
    """The RaBitQ encoding launch alone over given centres and labels (the
    5840/5841 checks plant a row equal to its centre)."""
    var D = rq_pow2(dim)
    var words = (D + 31) // 32
    var scale = rq_scale(D)
    var ctx = x_ann_ctx()
    var dx = upload_f32(ctx, x)
    var dc = upload_f32(ctx, centers)
    var dl = upload_i32(ctx, labels)
    var dws = ctx.enqueue_create_buffer[DType.float32](n * D)
    var dcodes = ctx.enqueue_create_buffer[DType.int32](n * words)
    var dnorm = ctx.enqueue_create_buffer[DType.float32](n)
    var dip = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_function[rq_encode_kernel](
        Int32(n), dx.unsafe_ptr(), dc.unsafe_ptr(), dl.unsafe_ptr(), Int32(dim), Int32(D), Int32(seed), scale,
        dws.unsafe_ptr(), Int32(words), dcodes.unsafe_ptr(), dnorm.unsafe_ptr(), dip.unsafe_ptr(),
        grid_dim=_grid(n), block_dim=TPB,
    )
    ctx.synchronize()
    codes = download_i32(ctx, dcodes, n * words)
    norms = download_f32(ctx, dnorm, n)
    ips = download_f32(ctx, dip, n)
    _ = dip^
    _ = dnorm^
    _ = dcodes^
    _ = dws^
    _ = dl^
    _ = dc^
    _ = dx^
    _ = ctx^
