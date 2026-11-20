# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-PQ on the device: one thread per cell of `x_ann/ivf_pq_core.mojo`.

The coarse quantizer is the same Lloyd cells over whole rows."""

from max.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation, unsafe_memcpy
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from x_ann.device_ctx import x_ann_ctx
from x_ann.stage_timer import AnnStages
from x_ann.switches import (
    ANN3_DIRECT_OUT, ANN3_HOST_PASSES, ANN3_PQ_SEED, ANN3_ROW_THREADS, IDN_PQ_DEVICE_CODEBOOKS,
    IDN_PQ_TRAINSET_STRIDED, IDN_PQ_TRAIN_ROWS_PER_CODE,
    IVF_PQ_ONE_UPLOAD,
)
from x_ann.kpp_seed import kpp_seed
from x_ann.fast_env import FAST_IVFPQ_DEVICE_CODEBOOKS
from x_ann.pq_kmeans_device import pq_codebooks_device
from x_ann.ivf_pq_core import PQK_CODES_MAX, PQK_LEN_MAX
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from x_ann.ivf_scan_device import ivf_scan_search

from cluster.estimator import kmeans_fit_rows
from cluster.impl.kmeans_params import INIT_ARRAY, INIT_KMEANS_PLUS_PLUS, METRIC_L2_EXPANDED
from ivf.estimator import ivf_flat_build_host, ivf_flat_build_resident_host
from ivf.impl.neighbors.ivf_flat.ivf_flat_build import IvfFlatBuildDevice, ivf_trainset_rows
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import ivf_explicit_train_rows, ivf_explicit_train_stride
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_mul_add, identical_sqrt
from x_ann.io import upload_f32, upload_i32, download_f32, download_i32, enqueue_query_status
from x_ann.io import upload_f32_ptr, upload_i32_ptr, download_f32_into, download_i32_into
from x_ann.refine_core import refine_cell
from x_ann.ivf_rabitq_core import rq_encode_cell, rq_pow2, rq_scale
from x_ann.ivf_sq_core import sq_encode_cell, sq_hi_takes, sq_lo_takes, sq_range_finish
from std.memory import bitcast
from x_ann.ivf_pq_core import (
    F32P, I32P, IvfPqIndex, pq_assign_cell, pq_inf, pq_insert, pq_labels_from_lists,
    pq_len_of, pq_residual_cell, pq_validate,
)

comptime TPB = 128

#: A caller's host array (kept alive by the caller for the whole call).
comptime HF32 = MutPointer[Float32, MutUntrackedOrigin]
comptime HI32 = MutPointer[Int32, MutUntrackedOrigin]

# TOMBSTONE: MOJOLEARN_PQ_FAST_TRAINSET_OFF (old path, no A/B recorded beside the fast default) deleted 2026-11-15 by lane/expdel2-13-cluster; code recoverable at 1657e37f94.
# Restore: git apply experiments/removed/MOJOLEARN_PQ_FAST_TRAINSET_OFF.patch; record in docs/TOMBSTONES.md.
comptime PQ_FAST_TRAINSET = GLOBAL_NUMERIC_MODE == NUMERIC_FAST
"""FAST (lane ann-apple, 2026-09-28): each subspace codebook trains on at
most `PQ_FAST_ROWS_PER_CODE` rows per code, a seeded uniform sample of the
residuals (FAISS's `max_points_per_centroid` rule; the coarse quantizer
already samples the same way under FAST on Apple, `IVF_FAST_TRAINSET`). Every
row is still encoded against the trained codebooks. IDENTICAL trains on every
row. Quality: bench/speed/ann_fast_quality.py."""
comptime PQ_FAST_ROWS_PER_CODE = 256

comptime PQ_FAST_SEED = ANN3_PQ_SEED and GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU
"""FAST on Apple, OPT-IN (lane ann-apple3, `-D MOJOLEARN_ANN3_PQ_SEED`): each
subspace codebook is seeded by `x_ann/kpp_seed.mojo` (host k-means++ over a
stride sample of its training rows, its own stream per subspace) and
cluster/'s k-means starts from those seeds (`INIT_ARRAY`), so it runs its
Lloyd iterations without its own seeding rounds."""
# (Sweep 2026-11-15: comptime PQ_FAST_SEED (always False: ANN3_PQ_SEED) and its kpp_seed branch removed.)


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
    var rv = Array[Float32, ASSIGN_LEN_MAX](fill=Float32(0.0))
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
    """The encode launch: the staged kernel when the subspace codebook fits,
    the one-thread-per-cell kernel otherwise."""
    # TOMBSTONE: MOJOLEARN_PQ_ASSIGN_UNSTAGED (unmeasured opt-in arm) deleted 2026-11-15 by lane/expdel2-13-cluster; code recoverable at 5243764241.
    # Restore: git apply experiments/removed/MOJOLEARN_PQ_ASSIGN_UNSTAGED.patch; record in docs/TOMBSTONES.md.
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
    mut labels: List[Int32], train_rows_per_list: Int = 0,
) raises:
    """The coarse quantizer IS IVF-Flat's build (`ivf/estimator.mojo::
    ivf_flat_build_host`: cluster/'s k-means, L2Expanded, its CSR lists);
    the host twin is `ivf/host/ivf_host.mojo::host_ivf_build`.
    `train_rows_per_list > 0`: the caller-set training rows (release-0845)."""
    var cst = AnnStages("ivf_coarse")
    var ctx = x_ann_ctx()
    # lane ann-apple3, behind `ANN3_HOST_PASSES`: the build without the
    # permuted vectors (no x_ann index reads them; `with_list_data` is read
    # only under the switch); the centres and the offsets move out of it;
    # the labels are the build's own assignment, the one its lists were laid
    # out from, so `pq_labels_from_lists` over those lists returns these words
    var flat = ivf_flat_build_host(
        ctx, x, n, dim, n_lists, kmeans_n_iters, METRIC_L2_EXPANDED, UInt64(seed), with_list_data=False,
        train_rows_per_list=train_rows_per_list,
    )
    ctx.synchronize()
    cst.host("flat_build")
    # cpu3-neighbors: the ids move over as words (each below n < 2^31, so a
    # uint32 word IS the int32 word), with no per-row host conversion walk;
    # the old `ANN3_HOST_PASSES`-off arm (always on) is gone.
    swap(centers, flat.centers)
    swap(offsets, flat.list_offsets)
    list_indices = List[Int32](length=n, fill=Int32(0))
    labels = List[Int32](length=n, fill=Int32(0))
    if n > 0:
        unsafe_memcpy(dest=list_indices.unsafe_ptr(), src=flat.list_indices.unsafe_ptr().bitcast[Int32](), count=n)
        unsafe_memcpy(dest=labels.unsafe_ptr(), src=flat.labels.unsafe_ptr().bitcast[Int32](), count=n)
    _ = flat^
    _ = ctx^
    cst.host("convert")


def pq_sub_gather_kernel(
    count: Int32, r: F32P, rows: I32P, use_rows: Int32, rot_dim: Int32, j: Int32, pq_len: Int32, dst: F32P,
):
    """cpu3-neighbors: subspace j's training sample, gathered on the device
    from the resident residuals: dst[i, t] = r[row(i), j pq_len + t] with
    row(i) = rows[i] (a sampled train set) or i (every row). A copy, so the
    words are the host gather's."""
    var e = _tid()
    if e < Int(count):
        var pl = Int(pq_len)
        var i = e // pl
        var t = e - i * pl
        var src = i
        if use_rows != Int32(0):
            src = Int(rows.unsafe_load(i))
        dst.unsafe_store(e, r.unsafe_load(src * Int(rot_dim) + Int(j) * pl + t))


def _codebooks(
    ctx: DeviceContext, mut dr: DeviceBuffer[DType.float32], n: Int, rot_dim: Int, pq_dim: Int, pq_len: Int,
    n_codes: Int, pq_iters: Int, seed: Int,
) raises -> List[Float32]:
    """Per subspace, cluster/'s k-means (`cluster/estimator.mojo::
    kmeans_fit_rows`, the same words as `kmeans_fit` with unit weights:
    k-means++, L2Expanded, one restart) over that subspace's residual
    columns; the host twin is `host_kmeans_fit`.

    cpu3-neighbors: the residuals stay on the device. Each subspace's sample
    is gathered by `pq_sub_gather_kernel` from `dr` (no n x rot_dim download,
    no host gather), the fit reads that device copy, and each fit writes its
    centroids straight into its slot of the returned codebooks. Same words
    as before: the gathered values and their order are the host gather's."""
    var cb_len = n_codes * pq_len
    var codebooks = List[Float32](length=pq_dim * cb_len, fill=Float32(0.0))
    var n_train = n
    comptime if PQ_FAST_TRAINSET:
        if n > PQ_FAST_ROWS_PER_CODE * n_codes:
            n_train = PQ_FAST_ROWS_PER_CODE * n_codes
    # `kmeans_fit_rows` reads only the length of its row list (the gathered
    # copy is already on the device); the sample's row ids go up once
    var rows = List[Int](length=n_train, fill=0)
    var use_rows = n_train < n
    var drows = ctx.enqueue_create_buffer[DType.int32](n_train if use_rows else 1)
    if use_rows:
        rows = ivf_trainset_rows(n, n_train, UInt64(seed))
        var rows32 = List[Int32](length=n_train, fill=Int32(0))
        for i in range(n_train):  # small-loop(n_train: at most PQ_FAST_ROWS_PER_CODE x n_codes sample ids, FAST only): narrows the FAST sample's row ids for one upload
            rows32[i] = Int32(rows[i])
        ctx.enqueue_copy(dst_buf=drows, src_ptr=rows32.unsafe_ptr())
        ctx.synchronize()
        _ = rows32^
    var cbs = AnnStages("ivf_pq_codebooks")
    cbs.host("rows")
    var dsub = ctx.enqueue_create_buffer[DType.float32](n_train * pq_len)
    var lab = List[UInt32](length=n_train, fill=UInt32(0))
    for j in range(pq_dim):
        ctx.enqueue_function[pq_sub_gather_kernel](
            Int32(n_train * pq_len), _dp(dr), _dp(drows), Int32(1 if use_rows else 0), Int32(rot_dim),
            Int32(j), Int32(pq_len), _dp(dsub),
            grid_dim=_grid(n_train * pq_len), block_dim=TPB,
        )
        cbs.host(String("gather.") + String(j))
        var cb_ptr = codebooks.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]().unsafe_offset(j * cb_len)
        var init_kind: Int = INIT_KMEANS_PLUS_PLUS
        comptime if PQ_FAST_SEED:
            # FAST on Apple, OPT-IN (`-D MOJOLEARN_ANN3_PQ_SEED`): the host
            # k-means++ seed reads the sample, so this opt-in arm alone
            # downloads it
            if n_train >= n_codes:
                var sub = download_f32(ctx, dsub, n_train * pq_len)
                var cb = List[Float32](length=cb_len, fill=Float32(0.0))
                kpp_seed(
                    sub, n_train, pq_len, n_codes,
                    UInt64(seed) ^ (UInt64(j + 1) * UInt64(0x9E3779B97F4A7C15)), 16, cb,
                )
                unsafe_memcpy(dest=cb_ptr, src=cb.unsafe_ptr(), count=cb_len)
                init_kind = INIT_ARRAY
                cbs.host("seed")
        _ = kmeans_fit_rows(
            ctx, dsub, cb_ptr, rows, pq_len, n_codes, cb_ptr,
            lab.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            pq_iters, Float64(1e-4), UInt64(seed), 1, init_kind, METRIC_L2_EXPANDED, Float64(2.0),
            lazy_shift=True,  # KMEANS_LAZY_SHIFT (FAST on Apple default, 2026-10-04): IVF-PQ codebooks
        )
        ctx.synchronize()
        # lane fg-ivf: the subspace in the stage name (MOJOLEARN_ANN_STAGES=1;
        # MOJOLEARN_KMEANS_STAGES=1 adds cluster/'s `KM` lines inside the fit)
        cbs.host(String("kmeans_fit.") + String(j))
    _ = lab^
    _ = dsub^
    _ = drows^
    return codebooks^


def _coarse_resident(
    ctx: DeviceContext, x: List[Float32], x_addr: Int, n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int,
    seed: Int, mut centers: List[Float32], mut offsets: List[Int32], mut list_indices: List[Int32],
    mut coarse_dev: IvfFlatBuildDevice, train_rows_per_list: Int = 0,
) raises:
    """lane fg-ivf A3 (`IVF_PQ_ONE_UPLOAD`): `_coarse` through IVF-Flat's
    RESIDENT build with `keep_rows`: the same parameters and statements
    (`ivf_flat_build_resident_host` is `ivf_flat_build_host`'s twin), the
    centres, offsets and carried ids on the host as before, and the uploaded
    rows (`coarse_dev.dx`), the centres (`coarse_dev.dcenters`) and the final
    assignment (`coarse_dev.dlabels`) LEFT ON THE DEVICE for the residuals.
    `x` may be empty when the build reads the rows from `x_addr`
    (`IVF_BUILD_FROM_POINTER`). `train_rows_per_list`: `_coarse`'s."""
    var cst = AnnStages("ivf_coarse")
    coarse_dev.keep_rows = True
    var flat = ivf_flat_build_resident_host(
        ctx, x, x_addr, n, dim, n_lists, kmeans_n_iters, METRIC_L2_EXPANDED, UInt64(seed), coarse_dev,
        train_rows_per_list,
    )
    ctx.synchronize()
    cst.host("flat_build_resident")
    swap(centers, flat.centers)
    swap(offsets, flat.list_offsets)
    list_indices = List[Int32](length=n, fill=Int32(0))
    if n > 0:
        unsafe_memcpy(dest=list_indices.unsafe_ptr(), src=flat.list_indices.unsafe_ptr().bitcast[Int32](), count=n)
    _ = flat^
    cst.host("convert")


def _coarse_rows(
    ctx: DeviceContext, x: List[Float32], x_addr: Int, n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int,
    seed: Int, train_rows_per_list: Int, mut centers: List[Float32], mut offsets: List[Int32],
    mut list_indices: List[Int32], mut coarse_dev: IvfFlatBuildDevice,
    mut dx: DeviceBuffer[DType.float32], mut dc: DeviceBuffer[DType.float32], mut dl: DeviceBuffer[DType.int32],
) raises -> Bool:
    """The coarse step of the IVF-SQ and IVF-RaBitQ builds with what the
    encode reads LEFT ON THE DEVICE (lane fx-ann-fit-nocopy, 2026-11-20):
    the rows in `dx`, the centres in `dc`, and the final assignment in
    `coarse_dev.dlabels` (returns True; uint32 words below n_lists < 2^31,
    the same words as int32) or in `dl` (returns False).

    Under `IVF_PQ_ONE_UPLOAD` this is IVF-PQ's coarse step
    (`_coarse_resident`): the rows go up once, read from `x_addr` when `x`
    is empty (`IVF_BUILD_FROM_POINTER`). Before, both builds ran `_coarse`
    over a list copy of the rows, then uploaded the rows a second time and
    sent the centres and the n labels back up from host lists. The same
    build statements on the same rows: no bit moves. Without the switch
    (FAST): `_coarse` over the list and the three uploads, as before."""
    var one_upload = False
    comptime if IVF_PQ_ONE_UPLOAD:
        one_upload = True
    if one_upload:
        _coarse_resident(
            ctx, x, x_addr, n, dim, n_lists, kmeans_n_iters, seed, centers, offsets, list_indices, coarse_dev,
            train_rows_per_list,
        )
        swap(dx, coarse_dev.dx)
        swap(dc, coarse_dev.dcenters)
    else:
        if len(x) != n * dim:
            # the bindings copy the rows in whenever this arm is compiled
            raise Error("x_ann build: the rows must be passed as a list with IVF_PQ_ONE_UPLOAD off")
        var labels = List[Int32]()
        _coarse(x, n, dim, n_lists, kmeans_n_iters, seed, centers, offsets, list_indices, labels, train_rows_per_list)
        dx = upload_f32(ctx, x)
        dc = upload_f32(ctx, centers)
        dl = upload_i32(ctx, labels)
        _ = labels^
    return one_upload


def _coarse_out(
    centers: List[Float32], offsets: List[Int32], list_indices: List[Int32], n: Int, dim: Int, n_lists: Int,
    centers_p: HF32, offsets_p: HI32, indices_p: HI32,
) raises:
    """The coarse step's three host lists into the caller's arrays: one
    memcpy each (n_lists x dim, n_lists + 1 and n words)."""
    if len(centers) != n_lists * dim or len(offsets) != n_lists + 1 or len(list_indices) != n:
        raise Error("x_ann build: the coarse lists disagree with n, dim and n_lists")
    unsafe_memcpy(dest=centers_p, src=centers.unsafe_ptr(), count=n_lists * dim)
    unsafe_memcpy(dest=offsets_p, src=offsets.unsafe_ptr(), count=n_lists + 1)
    unsafe_memcpy(dest=indices_p, src=list_indices.unsafe_ptr(), count=n)


struct IvfPqDevice(Movable):
    """lane fg-ivf A7 (`IVF_PQ_RESIDENT_FIT`): the device side a RESIDENT
    IVF-PQ build leaves behind for `x_ann/resident.mojo`: the coarse centres,
    the codebooks and the row-order codes (n x pq_dim), the buffers the
    build already holds. `IvfPqDevice(ctx)` is the placeholder a
    non-resident build passes (one-word buffers, never read)."""

    var dcenters: DeviceBuffer[DType.float32]
    var dcb: DeviceBuffer[DType.float32]
    var dcodes: DeviceBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext) raises:
        self.dcenters = ctx.enqueue_create_buffer[DType.float32](1)
        self.dcb = ctx.enqueue_create_buffer[DType.float32](1)
        self.dcodes = ctx.enqueue_create_buffer[DType.int32](1)


def ivf_pq_build_device(
    x: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    pq_dim: Int, pq_bits: Int, pq_iters: Int, codes_addr: Int = 0, x_addr: Int = 0,
) raises -> IvfPqIndex:
    """`codes_addr`: the address of the caller's n x pq_dim int32 array; the
    codes are downloaded straight into it and the returned index's `codes`
    is EMPTY (lane fx-ann-fit-nocopy, 2026-11-20: through the pooled pinned
    stages, `download_i32_into`; under the opt-in `ANN3_DIRECT_OUT`, lane
    ann-apple3, one raw copy into the pageable array). 0: the codes come
    back in the index (the checks). `x_addr` (lane fg-ivf A3): the
    caller's n x dim float32 rows, read in place by the coarse build when `x`
    is empty (`IVF_PQ_ONE_UPLOAD` with `IVF_BUILD_FROM_POINTER`)."""
    var ctx = x_ann_ctx()
    var dev = IvfPqDevice(ctx)
    var index = _ivf_pq_build_impl[False](
        ctx, x, x_addr, n, dim, n_lists, kmeans_n_iters, seed, pq_dim, pq_bits, pq_iters, codes_addr, dev
    )
    _ = dev^
    _ = ctx^
    return index^


def ivf_pq_build_device_resident(
    x: List[Float32], x_addr: Int, n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    pq_dim: Int, pq_bits: Int, pq_iters: Int, mut dev: IvfPqDevice,
) raises -> IvfPqIndex:
    """lane fg-ivf A7: `ivf_pq_build_device` with the codes, centres and
    codebooks LEFT ON THE DEVICE in `dev` (`x_ann/resident.mojo::
    x_ann_register_pq_built` makes the handle). The returned host index
    carries the centres, offsets, ids and codebooks and an EMPTY `codes`."""
    var ctx = x_ann_ctx()
    var index = _ivf_pq_build_impl[True](
        ctx, x, x_addr, n, dim, n_lists, kmeans_n_iters, seed, pq_dim, pq_bits, pq_iters, 0, dev
    )
    _ = ctx^
    return index^


def _ivf_pq_build_impl[resident: Bool](
    ctx: DeviceContext, x: List[Float32], x_addr: Int, n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int,
    seed: Int, pq_dim: Int, pq_bits: Int, pq_iters: Int, codes_addr: Int, mut out_dev: IvfPqDevice,
) raises -> IvfPqIndex:
    pq_validate(n, dim, n_lists, pq_dim, pq_bits, pq_iters)
    var pq_len = pq_len_of(dim, pq_dim)
    var rot_dim = pq_len * pq_dim
    var n_codes = 1 << pq_bits
    var centers = List[Float32]()
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    var labels = List[Int32]()
    var st = AnnStages("ivf_pq_build")
    # lane fg-ivf A3 (`IVF_PQ_ONE_UPLOAD`, default on outside FAST; `_OFF`
    # restores the second upload): the coarse build's own device rows,
    # centres and labels feed the residuals
    var coarse_dev = IvfFlatBuildDevice(ctx)
    var one_upload = False
    comptime if IVF_PQ_ONE_UPLOAD:
        one_upload = True
    if one_upload:
        _coarse_resident(
            ctx, x, x_addr, n, dim, n_lists, kmeans_n_iters, seed, centers, offsets, list_indices, coarse_dev
        )
    else:
        if len(x) != n * dim:
            # the bindings copy the rows in whenever this arm is compiled
            raise Error("ivf_pq_build_device: the rows must be passed as a list with IVF_PQ_ONE_UPLOAD off")
        _coarse(x, n, dim, n_lists, kmeans_n_iters, seed, centers, offsets, list_indices, labels)
    st.host("coarse")
    var dx: DeviceBuffer[DType.float32]
    var dc: DeviceBuffer[DType.float32]
    var dl: DeviceBuffer[DType.int32]
    if one_upload:
        dx = ctx.enqueue_create_buffer[DType.float32](1)
        dc = ctx.enqueue_create_buffer[DType.float32](1)
        dl = ctx.enqueue_create_buffer[DType.int32](1)
        swap(dx, coarse_dev.dx)
        swap(dc, coarse_dev.dcenters)
    else:
        dx = upload_f32(ctx, x)
        dc = upload_f32(ctx, centers)
        dl = upload_i32(ctx, labels)
    var dr = ctx.enqueue_create_buffer[DType.float32](n * rot_dim)
    if one_upload:
        # the build's uint32 labels are below n_lists < 2^31: the same words as int32
        _enqueue_residual(
            ctx, n, dim, rot_dim, _dp(dx), _dp(dc), _dp(coarse_dev.dlabels).bitcast[Int32](), _dp(dr)
        )
    else:
        _enqueue_residual(ctx, n, dim, rot_dim, _dp(dx), _dp(dc), _dp(dl), _dp(dr))
    ctx.synchronize()
    st.host("residual")
    # FAST on Apple, default (lane/apple-fast-ann, 2026-10-02; x_ann/pq_kmeans_device.mojo): the codebooks of
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
    # lane fg-ivf A1 (`IDN_PQ_DEVICE_CODEBOOKS`, IDENTICAL default since lane
    # postmerge-act-1, `_OFF` restores the per-subspace fits; bits
    # change on every column; x_ann/switches.mojo): the same batched loop in
    # IDENTICAL on NVIDIA and AMD, with the pinned fold (`PQK_PINNED`); the
    # host twin is `_codebooks_host_batched` under the same gate and limits.
    comptime if IDN_PQ_DEVICE_CODEBOOKS:
        dev_cb = pq_len <= PQK_LEN_MAX and n_codes <= PQK_CODES_MAX
    var codebooks = List[Float32]()
    var dcb: DeviceBuffer[DType.float32]
    if dev_cb:
        var n_train = n
        comptime if PQ_FAST_TRAINSET:
            if n > PQ_FAST_ROWS_PER_CODE * n_codes:
                n_train = PQ_FAST_ROWS_PER_CODE * n_codes
        # lane hdb-ivf D8(a), IDN_PQ_TRAINSET_STRIDED (x_ann/switches.mojo);
        # the host twin reads the same n_train
        comptime if IDN_PQ_TRAINSET_STRIDED:
            if n > IDN_PQ_TRAIN_ROWS_PER_CODE * n_codes:
                n_train = IDN_PQ_TRAIN_ROWS_PER_CODE * n_codes
        dcb = pq_codebooks_device(ctx, dr, n, n_train, rot_dim, pq_dim, pq_len, n_codes, pq_iters, seed)
        codebooks = download_f32(ctx, dcb, pq_dim * n_codes * pq_len)
        st.host("codebooks")
    else:
        codebooks = _codebooks(ctx, dr, n, rot_dim, pq_dim, pq_len, n_codes, pq_iters, seed)
        st.host("codebooks")
        dcb = upload_f32(ctx, codebooks)
    var dcodes = ctx.enqueue_create_buffer[DType.int32](n * pq_dim)
    _enqueue_assign(ctx, n, _dp(dr), _dp(dcb), pq_dim, rot_dim, pq_len, n_codes, _dp(dcodes))
    ctx.synchronize()
    var codes = List[Int32]()
    comptime if resident:
        # lane fg-ivf A7: the codes, centres and codebooks stay on the device
        swap(out_dev.dcodes, dcodes)
        swap(out_dev.dcb, dcb)
        swap(out_dev.dcenters, dc)
        st.host("encode_resident")
    else:
        if codes_addr != 0:
            # lane fx-ann-fit-nocopy (2026-11-20): into the caller's array.
            # Gone: a result list the size of the codes and the copy out of it.
            comptime if ANN3_DIRECT_OUT:
                ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=codes_addr), src_buf=dcodes)
                ctx.synchronize()
            else:
                download_i32_into(ctx, dcodes, n * pq_dim, HI32(unsafe_from_address=codes_addr))
        else:
            codes = download_i32(ctx, dcodes, n * pq_dim)
        st.host("encode")
    _ = dcodes^
    _ = dcb^
    _ = dr^
    _ = dl^
    _ = dc^
    _ = dx^
    _ = coarse_dev^
    return IvfPqIndex(n_lists, dim, n, pq_dim, pq_len, n_codes, centers^, offsets^, list_indices^, codebooks^, codes^)


def ivf_pq_search_device(
    centers: List[Float32], offsets: List[Int32], list_indices: List[Int32], codebooks: List[Float32],
    codes: List[Int32], mask: List[Int32], n_lists: Int, dim: Int, pq_dim: Int, pq_bits: Int,
    queries: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
    status_addr: Int = 0,
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
    if len(queries) < m * dim:
        raise Error("x_ann search: the query list is shorter than m x dim")
    ivf_pq_search_on(
        ctx, _dp(dc), _dp(doff), _dp(dli), _dp(dcb), _dp(dcodes), _dp(dmask), offsets, n_lists, dim,
        pq_dim, pq_bits,
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(queries.unsafe_ptr())),
        m, k, n_probes, out_d, out_i, out_n,
        False, _dp(dcodes), _dp(dc), _dp(dc), False, status_addr,
    )
    # the query list is named here, after the drains inside the call above
    _ = len(queries)
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
    qp: MutPointer[Float32, MutUntrackedOrigin], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
    have_pre: Bool, pre_codes: I32P, pre_a: F32P, pre_b: F32P, mask_pre: Bool,
    status_addr: Int = 0,
) raises:
    """The search over an index already on the device (`ivf_pq_search_device`
    uploads it first; `x_ann/resident.mojo` holds it): the queries up, the
    scan, the three outputs down. `have_pre` ... `mask_pre`: the resident
    index's list-order arrays (`ivf_scan_search`); `mask_pre` says dmask is
    the all-ones filter, which is its own list-order copy."""
    var pq_len = pq_len_of(dim, pq_dim)
    var n_codes = 1 << pq_bits
    # The queries are read where the caller holds them (`qp`, m x dim words
    # kept alive by the caller for the whole call; lane fx-board-1p2,
    # 2026-11-20): the resident search copied them into a new list first.
    var dq = upload_f32_ptr(ctx, qp, m * dim)
    # Lane a8b-device-x-neighbors-rest: the queries' finite status behind
    # their upload, read back by the downloads below (no new sync).
    var dst = enqueue_query_status(ctx, dq, m * dim, status_addr)
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
    _ = dst^


#: rows per partial of the SQ range (lane ann-apple)
comptime SQ_RANGE_ROWS = 1024


@always_inline
def _is_nan(v: Float32) -> Bool:
    var b = bitcast[DType.uint32](v)
    return (b & UInt32(0x7F800000)) == UInt32(0x7F800000) and (b & UInt32(0x007FFFFF)) != UInt32(0)


def sq_range_part_kernel(
    count: Int32, r: F32P, n: Int32, dim: Int32, lo_p: F32P, hi_p: F32P, has_p: I32P, row_stride: Int32
):
    """One thread per (row chunk g, column c), e = g * dim + c: the chunk's
    minimum and maximum over its non-NaN values under `sq_lo_takes` /
    `sq_hi_takes` in row order (the first of equal values kept). Row i is
    row i * row_stride of `r` (release-0845: 1 unless IVFSQIndex sets
    max_train_points_per_cluster)."""
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
            var v = ftz(r.unsafe_load(i * Int(row_stride) * d + c))
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
    mut dvmin: DeviceBuffer[DType.float32], mut ddelta: DeviceBuffer[DType.float32], row_stride: Int = 1,
) raises:
    """`sq_range_cell` for every column as chunk partials, then the join, over
    rows 0, s, ..., (n - 1) s of `dr` (s = `row_stride`; the join reads row 0,
    which is always one of them)."""
    var parts = (n + SQ_RANGE_ROWS - 1) // SQ_RANGE_ROWS
    var lo_p = ctx.enqueue_create_buffer[DType.float32](parts * dim)
    var hi_p = ctx.enqueue_create_buffer[DType.float32](parts * dim)
    var has_p = ctx.enqueue_create_buffer[DType.int32](parts * dim)
    ctx.enqueue_function[sq_range_part_kernel](
        Int32(parts * dim), dr.unsafe_ptr(), Int32(n), Int32(dim), lo_p.unsafe_ptr(), hi_p.unsafe_ptr(),
        has_p.unsafe_ptr(), Int32(row_stride), grid_dim=_grid(parts * dim), block_dim=TPB,
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
    mut vmin: List[Float32], mut delta: List[Float32], mut codes: List[Int32],
    train_rows_per_list: Int = 0,
) raises:
    """The list form (the checks'): `ivf_sq_build_device_ptr` over the rows
    in `x`, the results in six new lists."""
    pq_validate(n, dim, n_lists, 1, 1, 1)
    if len(x) != n * dim:
        raise Error("ivf_sq_build_device: the row list is not n x dim")
    var c = List[Float32](unsafe_uninit_length=n_lists * dim)
    var off = List[Int32](unsafe_uninit_length=n_lists + 1)
    var li = List[Int32](unsafe_uninit_length=n)
    var vm = List[Float32](unsafe_uninit_length=dim)
    var de = List[Float32](unsafe_uninit_length=dim)
    var co = List[Int32](unsafe_uninit_length=n * dim)
    ivf_sq_build_device_ptr(
        x, 0, n, dim, n_lists, kmeans_n_iters, seed,
        HF32(unsafe_from_address=Int(c.unsafe_ptr())), HI32(unsafe_from_address=Int(off.unsafe_ptr())),
        HI32(unsafe_from_address=Int(li.unsafe_ptr())), HF32(unsafe_from_address=Int(vm.unsafe_ptr())),
        HF32(unsafe_from_address=Int(de.unsafe_ptr())), HI32(unsafe_from_address=Int(co.unsafe_ptr())),
        train_rows_per_list,
    )
    # the rows are named here, after the drains inside the call above
    _ = len(x)
    centers = c^
    offsets = off^
    list_indices = li^
    vmin = vm^
    delta = de^
    codes = co^


def ivf_sq_build_device_ptr(
    x: List[Float32], x_addr: Int, n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    centers_p: HF32, offsets_p: HI32, indices_p: HI32, vmin_p: HF32, delta_p: HF32, codes_p: HI32,
    train_rows_per_list: Int,
) raises:
    """The IVF-SQ build on the CALLER's arrays (lane fx-ann-fit-nocopy,
    2026-11-20). The rows are read from `x_addr` when `x` is empty
    (`_coarse_rows`) and uploaded once; the centres (n_lists x dim), the
    offsets (n_lists + 1), the carried ids (n), the range (dim and dim) and
    the n x dim int32 codes are written where the caller wants them. Before,
    the binding copied the rows into a list, the build uploaded them twice,
    and the codes came down into a list and were copied out of it: for
    400,000 x 220 rows three 352 MB host arrays touched for the first time
    inside the call (half its wall as system time). The same words from the
    same kernels; the caller keeps every array alive for the whole call
    (each copy is drained before the return).

    `train_rows_per_list` (release-0845; IVFSQIndex max_train_points_per_cluster,
    cuVS ivf_sq's): 0 trains the coarse k-means on the IVF rule's rows and the
    SQ range on every residual (the default, unchanged); `m > 0` trains both on
    the same `ivf_explicit_train_rows(n, n_lists, m)` strided rows, as cuVS
    trains both on its training subset (ivf_sq_build.cuh:457-525). Every row
    is still assigned and encoded."""
    pq_validate(n, dim, n_lists, 1, 1, 1)
    var st = AnnStages("ivf_sq_build")
    var ctx = x_ann_ctx()
    var centers = List[Float32]()
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    var coarse_dev = IvfFlatBuildDevice(ctx)
    var dx = ctx.enqueue_create_buffer[DType.float32](1)
    var dc = ctx.enqueue_create_buffer[DType.float32](1)
    var dl = ctx.enqueue_create_buffer[DType.int32](1)
    var device_labels = _coarse_rows(
        ctx, x, x_addr, n, dim, n_lists, kmeans_n_iters, seed, train_rows_per_list, centers, offsets,
        list_indices, coarse_dev, dx, dc, dl,
    )
    st.host("coarse")
    _coarse_out(centers, offsets, list_indices, n, dim, n_lists, centers_p, offsets_p, indices_p)
    var dr = ctx.enqueue_create_buffer[DType.float32](n * dim)
    var dvmin = ctx.enqueue_create_buffer[DType.float32](dim)
    var ddelta = ctx.enqueue_create_buffer[DType.float32](dim)
    var dcodes = ctx.enqueue_create_buffer[DType.int32](n * dim)
    if device_labels:
        _enqueue_residual(
            ctx, n, dim, dim, _dp(dx), _dp(dc), _dp(coarse_dev.dlabels).bitcast[Int32](), _dp(dr)
        )
    else:
        _enqueue_residual(ctx, n, dim, dim, _dp(dx), _dp(dc), _dp(dl), _dp(dr))
    st.mark(ctx, "upload_residuals")
    if train_rows_per_list > 0:
        _sq_range_enqueue(
            ctx, dr, ivf_explicit_train_rows(n, n_lists, train_rows_per_list), dim, dvmin, ddelta,
            ivf_explicit_train_stride(n, n_lists, train_rows_per_list),
        )
    else:
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
    download_f32_into(ctx, dvmin, dim, vmin_p)
    download_f32_into(ctx, ddelta, dim, delta_p)
    # The codes go into the caller's array through the pooled pinned stages
    # (`download_i32_into`). The opt-in `ANN3_DIRECT_OUT` (lane ann-apple3,
    # Apple FAST, paused and unmeasured) keeps its one raw copy into the
    # pageable array as the arm.
    comptime if ANN3_DIRECT_OUT:
        ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=Int(codes_p)), src_buf=dcodes)
        ctx.synchronize()
    else:
        download_i32_into(ctx, dcodes, n * dim, codes_p)
    st.host("encode")
    # the rows are named here, after the drains above
    _ = len(x)
    _ = dcodes^
    _ = ddelta^
    _ = dvmin^
    _ = dr^
    _ = dl^
    _ = dc^
    _ = dx^
    _ = coarse_dev^
    _ = ctx^


def ivf_sq_search_device(
    centers: List[Float32], offsets: List[Int32], list_indices: List[Int32], vmin: List[Float32],
    delta: List[Float32], codes: List[Int32], mask: List[Int32], n_lists: Int, dim: Int,
    queries: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
    status_addr: Int = 0,
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
    if len(queries) < m * dim:
        raise Error("x_ann search: the query list is shorter than m x dim")
    ivf_sq_search_on(
        ctx, _dp(dc), _dp(doff), _dp(dli), _dp(dvmin), _dp(ddelta), _dp(dcodes), _dp(dmask), offsets,
        n_lists, dim,
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(queries.unsafe_ptr())),
        m, k, n_probes, out_d, out_i, out_n,
        False, _dp(dcodes), _dp(dc), _dp(dc), False, status_addr,
    )
    # the query list is named here, after the drains inside the call above
    _ = len(queries)
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
    qp: MutPointer[Float32, MutUntrackedOrigin], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
    have_pre: Bool, pre_codes: I32P, pre_a: F32P, pre_b: F32P, mask_pre: Bool,
    status_addr: Int = 0,
) raises:
    """`ivf_pq_search_on`'s SQ twin: the index already on the device."""
    # The queries are read where the caller holds them (`qp`, m x dim words
    # kept alive by the caller for the whole call; lane fx-board-1p2,
    # 2026-11-20): the resident search copied them into a new list first.
    var dq = upload_f32_ptr(ctx, qp, m * dim)
    # Lane a8b-device-x-neighbors-rest: the queries' finite status behind
    # their upload, read back by the downloads below (no new sync).
    var dst = enqueue_query_status(ctx, dq, m * dim, status_addr)
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
    _ = dst^


def refine_kernel(m: Int32, x: F32P, n: Int32, d: Int32, queries: F32P, cand: I32P, k0: Int32, k: Int32,
                  out_d: F32P, out_i: I32P, root: Int32):
    var q = _tid()
    if q < Int(m):
        refine_cell(q, x, Int(n), Int(d), queries, cand, Int(k0), Int(k), out_d, out_i, root != 0)


# TOMBSTONE: MOJOLEARN_IVF_REFINE_TEAM (DROPPED-noise) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: ivf refine with one threadgroup per query and the dataset uploaded from the caller's array (refine_team_kernel, refine_device_team); ivf-refine taxi 1,208.3 -> 1,201.8 ms (-0.5%, noise).
# Restore: git apply experiments/removed/MOJOLEARN_IVF_REFINE_TEAM.patch; record in docs/TOMBSTONES.md.


def refine_device(
    x: List[Float32], n: Int, d: Int, queries: List[Float32], m: Int, cand: List[Int32], k0: Int, k: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], root: Bool = False, status_addr: Int = 0,
) raises:
    """The list form (the checks'): `refine_device_ptr` on the lists' own
    words, the results in two new lists."""
    if len(x) < n * d or len(queries) < m * d or len(cand) < m * k0:
        raise Error("refine: an input list is shorter than its shape")
    var rd = List[Float32](unsafe_uninit_length=m * k)
    var ri = List[Int32](unsafe_uninit_length=m * k)
    refine_device_ptr(
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr())), n, d,
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(queries.unsafe_ptr())), m,
        MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(cand.unsafe_ptr())), k0, k,
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(rd.unsafe_ptr())),
        MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(ri.unsafe_ptr())),
        root, status_addr,
    )
    # the inputs are named here, after the drains inside the call above
    _ = len(x)
    _ = len(queries)
    _ = len(cand)
    out_d = rd^
    out_i = ri^


def refine_device_ptr(
    xp: MutPointer[Float32, MutUntrackedOrigin], n: Int, d: Int,
    qp: MutPointer[Float32, MutUntrackedOrigin], m: Int,
    candp: MutPointer[Int32, MutUntrackedOrigin], k0: Int, k: Int,
    out_dp: MutPointer[Float32, MutUntrackedOrigin], out_ip: MutPointer[Int32, MutUntrackedOrigin],
    root: Bool, status_addr: Int,
) raises:
    """`refine` on the CALLER's arrays (lane fx-board-modelsel-metrics,
    2026-11-19): the dataset (n x d), the queries (m x d) and the candidate
    ids (m x k0) are uploaded from where they are, and the m x k distances
    and ids are written where the caller wants them. Before, the binding
    copied all three inputs into new lists first (the whole dataset: 880 MB
    of first-touch pages for 1,000,000 x 220, two thirds of the call's wall
    as system time) and the results into lists and out again. The same
    words go to the same kernel; the caller keeps every array alive for the
    whole call (each copy is drained before the return)."""
    var ctx = x_ann_ctx()
    var dx = upload_f32_ptr(ctx, xp, n * d)
    var dq = upload_f32_ptr(ctx, qp, m * d)
    # Lane a8b-device-x-neighbors-rest: the dataset's and the queries'
    # finite status behind their uploads, read back by the downloads below.
    var dst = enqueue_query_status(ctx, dx, n * d, dq, m * d, status_addr)
    var dcand = upload_i32_ptr(ctx, candp, m * k0)
    var dd = ctx.enqueue_create_buffer[DType.float32](m * k)
    var di = ctx.enqueue_create_buffer[DType.int32](m * k)
    ctx.enqueue_function[refine_kernel](Int32(m), dx.unsafe_ptr(), Int32(n), Int32(d), dq.unsafe_ptr(),
                                        dcand.unsafe_ptr(), Int32(k0), Int32(k), dd.unsafe_ptr(), di.unsafe_ptr(),
                                        Int32(1) if root else Int32(0), grid_dim=_grid(m), block_dim=TPB)
    ctx.synchronize()
    download_f32_into(ctx, dd, m * k, out_dp)
    download_i32_into(ctx, di, m * k, out_ip)
    _ = di^
    _ = dd^
    _ = dcand^
    _ = dq^
    _ = dx^
    _ = dst^
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
    """The list form (the checks'): `ivf_rabitq_build_device_ptr` over the
    rows in `x`, the results in six new lists."""
    pq_validate(n, dim, n_lists, 1, 1, 1)
    if len(x) != n * dim:
        raise Error("ivf_rabitq_build_device: the row list is not n x dim")
    var words = (rq_pow2(dim) + 31) // 32
    var c = List[Float32](unsafe_uninit_length=n_lists * dim)
    var off = List[Int32](unsafe_uninit_length=n_lists + 1)
    var li = List[Int32](unsafe_uninit_length=n)
    var co = List[Int32](unsafe_uninit_length=n * words)
    var no = List[Float32](unsafe_uninit_length=n)
    var ip = List[Float32](unsafe_uninit_length=n)
    ivf_rabitq_build_device_ptr(
        x, 0, n, dim, n_lists, kmeans_n_iters, seed,
        HF32(unsafe_from_address=Int(c.unsafe_ptr())), HI32(unsafe_from_address=Int(off.unsafe_ptr())),
        HI32(unsafe_from_address=Int(li.unsafe_ptr())), HI32(unsafe_from_address=Int(co.unsafe_ptr())),
        HF32(unsafe_from_address=Int(no.unsafe_ptr())), HF32(unsafe_from_address=Int(ip.unsafe_ptr())),
    )
    # the rows are named here, after the drains inside the call above
    _ = len(x)
    centers = c^
    offsets = off^
    list_indices = li^
    codes = co^
    norms = no^
    ips = ip^


def ivf_rabitq_build_device_ptr(
    x: List[Float32], x_addr: Int, n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    centers_p: HF32, offsets_p: HI32, indices_p: HI32, codes_p: HI32, norms_p: HF32, ips_p: HF32,
) raises:
    """The IVF-RaBitQ build on the CALLER's arrays (lane fx-ann-fit-nocopy,
    2026-11-20; `ivf_sq_build_device_ptr`'s form): the rows up once, read
    from `x_addr` when `x` is empty (`_coarse_rows`); the centres, offsets
    and ids, the n x words int32 codes and the n norms and n factors written
    where the caller wants them. Gone: the list copy of the rows, their
    second upload, the centres' and labels' way back up, and three result
    lists with a copy out of each. The same words from the same kernel."""
    pq_validate(n, dim, n_lists, 1, 1, 1)
    var D = rq_pow2(dim)
    var words = (D + 31) // 32
    var scale = rq_scale(D)
    var st = AnnStages("ivf_rabitq_build")
    var ctx = x_ann_ctx()
    var centers = List[Float32]()
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    var coarse_dev = IvfFlatBuildDevice(ctx)
    var dx = ctx.enqueue_create_buffer[DType.float32](1)
    var dc = ctx.enqueue_create_buffer[DType.float32](1)
    var dl = ctx.enqueue_create_buffer[DType.int32](1)
    var device_labels = _coarse_rows(
        ctx, x, x_addr, n, dim, n_lists, kmeans_n_iters, seed, 0, centers, offsets, list_indices, coarse_dev,
        dx, dc, dl,
    )
    st.host("coarse")
    _coarse_out(centers, offsets, list_indices, n, dim, n_lists, centers_p, offsets_p, indices_p)
    var dws = ctx.enqueue_create_buffer[DType.float32](n * D)
    var dcodes = ctx.enqueue_create_buffer[DType.int32](n * words)
    var dnorm = ctx.enqueue_create_buffer[DType.float32](n)
    var dip = ctx.enqueue_create_buffer[DType.float32](n)
    if device_labels:
        ctx.enqueue_function[rq_encode_kernel](
            Int32(n), _dp(dx), _dp(dc), _dp(coarse_dev.dlabels).bitcast[Int32](), Int32(dim), Int32(D),
            Int32(seed), scale, _dp(dws), Int32(words), _dp(dcodes), _dp(dnorm), _dp(dip),
            grid_dim=_grid(n), block_dim=TPB,
        )
    else:
        ctx.enqueue_function[rq_encode_kernel](
            Int32(n), _dp(dx), _dp(dc), _dp(dl), Int32(dim), Int32(D), Int32(seed), scale,
            _dp(dws), Int32(words), _dp(dcodes), _dp(dnorm), _dp(dip),
            grid_dim=_grid(n), block_dim=TPB,
        )
    ctx.synchronize()
    download_i32_into(ctx, dcodes, n * words, codes_p)
    download_f32_into(ctx, dnorm, n, norms_p)
    download_f32_into(ctx, dip, n, ips_p)
    st.host("encode")
    # the rows are named here, after the drains above
    _ = len(x)
    _ = dip^
    _ = dnorm^
    _ = dcodes^
    _ = dws^
    _ = dl^
    _ = dc^
    _ = dx^
    _ = coarse_dev^
    _ = ctx^


def ivf_rabitq_search_device(
    centers: List[Float32], offsets: List[Int32], list_indices: List[Int32], codes: List[Int32],
    norms: List[Float32], ips: List[Float32], mask: List[Int32], n_lists: Int, dim: Int, seed: Int,
    queries: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
    status_addr: Int = 0,
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
    if len(queries) < m * dim:
        raise Error("x_ann search: the query list is shorter than m x dim")
    ivf_rabitq_search_on(
        ctx, _dp(dc), _dp(doff), _dp(dli), _dp(dcodes), _dp(dnorm), _dp(dip), _dp(dmask), offsets,
        n_lists, dim, seed,
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(queries.unsafe_ptr())),
        m, k, n_probes, out_d, out_i, out_n,
        False, _dp(dcodes), _dp(dnorm), _dp(dip), False, status_addr,
    )
    # the query list is named here, after the drains inside the call above
    _ = len(queries)
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
    qp: MutPointer[Float32, MutUntrackedOrigin], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
    have_pre: Bool, pre_codes: I32P, pre_a: F32P, pre_b: F32P, mask_pre: Bool,
    status_addr: Int = 0,
) raises:
    """`ivf_pq_search_on`'s RaBitQ twin: the index already on the device."""
    var D = rq_pow2(dim)
    var words = (D + 31) // 32
    var scale = rq_scale(D)
    # The queries are read where the caller holds them (`qp`, m x dim words
    # kept alive by the caller for the whole call; lane fx-board-1p2,
    # 2026-11-20): the resident search copied them into a new list first.
    var dq = upload_f32_ptr(ctx, qp, m * dim)
    # Lane a8b-device-x-neighbors-rest: the queries' finite status behind
    # their upload, read back by the downloads below (no new sync).
    var dst = enqueue_query_status(ctx, dq, m * dim, status_addr)
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
    _ = dst^


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
