"""MeanShift, IDENTICAL on every vendor: every shift of every seed on a grid
of (seed, row chunk) blocks (K6, lane ml-cluster-nbrs 2026-10-04). Switch:
`IDN_MEANSHIFT_GRID` (x_cluster/bodies.mojo), taken by
`x_cluster/device_ops.mojo` `DeviceOps.meanshift`.

Cause: `_meanshift_team_kernel` (x_cluster/device_ops.mojo) takes ONE block
per seed and walks all n rows of every shift inside it (bin seeding: tens of
seeds, so tens of blocks busy for the whole fit).

The fold is `bodies.meanshift_seed_blocked`'s, word for word:

  * `_msi_part_kernel`: block (s, c) tests the MSI_T rows of chunk c against
    seed s's center (the `sq_dist_rows` chain), then thread f folds feature f
    over the chunk's rows within the bandwidth, ascending from zero; the
    partials go to `part[s, c, :]`, the integer count to `cntp[s, c]`;
  * `_msi_finish_kernel`: one block per seed; thread f folds the chunk
    partials ascending from zero, thread 0 sums the integer counts, runs the
    quotients, the shift and the stop test in feature order and marks the
    seed done.

The host enqueues MSI_GROUP shifts between reads of the done flags (one
synchronize per group); the stop decision itself is made on the device, and
a done seed's blocks return at once. Shared pages are gated by
`lib_smem_page_fits_for`; past MSI_MAX_D features (or a page that does not
fit) the caller runs the one-thread `_meanshift_kernel`, whose body is the
same blocked fold.
"""
from experiments.classical_identical_ideas.graph_controls import C40_SEED_TILES, C40_ACTIVE_SEEDS
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for
from checks.numerics import ftz, identical_div, identical_mul, identical_sqrt
from x_cluster.bodies import FPtr, IPtr, IDN_MEANSHIFT_GRID, MSI_T

comptime MSI_TPB = 256
comptime MSI_MAX_D = 1024
comptime MSI_GROUP = 16
"""Shifts enqueued between two reads of the done flags."""
comptime MSI_PART_BYTES = (MSI_MAX_D + MSI_T + MSI_TPB) * 4
comptime MSI_FINISH_BYTES = (MSI_MAX_D + MSI_TPB) * 4
comptime MSI_FITS = lib_smem_page_fits_for[TARGET_COLUMN, MSI_PART_BYTES]() and lib_smem_page_fits_for[
    TARGET_COLUMN, MSI_FINISH_BYTES
]()
comptime MSI_ON = IDN_MEANSHIFT_GRID and MSI_FITS


# Stable active descriptors preserve seed IDs and per-seed stopping state.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
def _c40_active_kernel(done: IPtr, ns: Int32, active: IPtr):
    if thread_idx.x == 0:
        var count = 0
        for seed in range(Int(ns)):
            if done[seed] == 0:
                active[count+1] = Int32(seed)
                count += 1
        active[0] = Int32(count)


def _msi_part_kernel(
    x: FPtr, n: Int32, d: Int32, bw: Float32, centers: FPtr, done: IPtr, nch: Int32, part: FPtr, cntp: IPtr, active: IPtr,
):
    var b = Int(block_idx.x)
    var NC = Int(nch)
    var s = b // NC
    var c = b - s * NC
    var tid = Int(thread_idx.x)
    comptime if C40_ACTIVE_SEEDS:
        if s >= Int(active[0]):
            return
        s = Int(active[s+1])
    if done[s] != Int32(0):
        return
    var N = Int(n)
    var D = Int(d)
    var cen = stack_allocation[MSI_MAX_D, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var flag = stack_allocation[MSI_T, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var cnts = stack_allocation[MSI_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    for f in range(tid, D, MSI_TPB):
        cen[f] = centers[s * D + f]
    barrier()
    var t0 = c * MSI_T
    var cnt = N - t0
    if cnt > MSI_T:
        cnt = MSI_T
    var mine = 0
    for r in range(tid, cnt, MSI_TPB):
        var p = t0 + r
        var acc = Float32(0)
        for f in range(D):
            var t = ftz(ftz(cen[f]) - ftz(x[p * D + f]))
            acc = ftz(acc + ftz(identical_mul(t, t)))
        var w = identical_sqrt(acc) <= bw
        flag[r] = Int32(1) if w else Int32(0)
        if w:
            mine += 1
    cnts[tid] = Int32(mine)
    barrier()
    var o = (s * NC + c) * D
    for f in range(tid, D, MSI_TPB):
        var a = Float32(0)
        for r in range(cnt):
            if flag[r] != Int32(0):
                a = ftz(a + ftz(x[(t0 + r) * D + f]))
        part[o + f] = a
    if tid == 0:
        var within = Int32(0)
        for u in range(MSI_TPB):
            within += cnts[u]
        cntp[s * NC + c] = within


def _c40_seed_pair_kernel(
    x: FPtr, n: Int32, d: Int32, bw: Float32, centers: FPtr, done: IPtr,
    nch: Int32, part: FPtr, cntp: IPtr, ns: Int32, active: IPtr,
):
    # Two independent seeds share input-row loads. Shared storage is
    # (1024 + 2*MSI_T) float32 words; larger rows keep the existing path.
    var chunk = Int(block_idx.x) % Int(nch)
    var first = (Int(block_idx.x) // Int(nch))*2
    var seed_ids = InlineArray[Int,2](fill=Int(ns))
    comptime for lane in range(2):
        comptime if C40_ACTIVE_SEEDS:
            if first+lane < Int(active[0]):
                seed_ids[lane] = Int(active[first+lane+1])
        else:
            seed_ids[lane] = first+lane
    var tid = Int(thread_idx.x)
    var D = Int(d)
    var cen = stack_allocation[1024, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var flag = stack_allocation[2*MSI_T, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    comptime for seed in range(2):
        if seed_ids[seed] < Int(ns) and done[seed_ids[seed]] == 0:
            for f in range(tid,D,MSI_TPB):
                cen[seed*512+f] = centers[(seed_ids[seed])*D+f]
    barrier()
    var count = min(MSI_T,Int(n)-chunk*MSI_T)
    for r in range(tid,count,MSI_TPB):
        var acc = InlineArray[Float32,2](fill=Float32(0))
        for f in range(D):
            var value = ftz(x[(chunk*MSI_T+r)*D+f])
            comptime for seed in range(2):
                if seed_ids[seed] < Int(ns) and done[seed_ids[seed]] == 0:
                    var delta = ftz(ftz(cen[seed*512+f])-value)
                    acc[seed] = ftz(acc[seed]+ftz(identical_mul(delta,delta)))
        comptime for seed in range(2):
            flag[seed*MSI_T+r] = Int32(1) if seed_ids[seed] < Int(ns) and done[seed_ids[seed]] == 0 and identical_sqrt(acc[seed]) <= bw else Int32(0)
    barrier()
    comptime for seed in range(2):
        if seed_ids[seed] < Int(ns) and done[seed_ids[seed]] == 0:
            for f in range(tid,D,MSI_TPB):
                var value = Float32(0)
                for r in range(count):
                    if flag[seed*MSI_T+r] != 0:
                        value = ftz(value+ftz(x[(chunk*MSI_T+r)*D+f]))
                part[((seed_ids[seed])*Int(nch)+chunk)*D+f] = value
            if tid == 0:
                var cnt = Int32(0)
                for r in range(count):
                    cnt += flag[seed*MSI_T+r]
                cntp[(seed_ids[seed])*Int(nch)+chunk] = cnt


def _msi_finish_kernel(
    d: Int32, nch: Int32, part: FPtr, cntp: IPtr, centers: FPtr, done: IPtr, intensity: IPtr, iters: IPtr,
    stop: Float32, max_iter: Int32,
):
    var s = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    if done[s] != Int32(0):
        return
    var D = Int(d)
    var NC = Int(nch)
    var sums = stack_allocation[MSI_MAX_D, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var cnts = stack_allocation[MSI_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    for f in range(tid, D, MSI_TPB):
        var a = Float32(0)
        for c in range(NC):
            a = ftz(a + part[(s * NC + c) * D + f])
        sums[f] = a
    var mine = Int32(0)
    for c in range(tid, NC, MSI_TPB):
        mine += cntp[s * NC + c]
    cnts[tid] = mine
    barrier()
    if tid == 0:
        var within = 0
        for u in range(MSI_TPB):
            within += Int(cnts[u])
        intensity[s] = Int32(within)
        if within == 0:
            done[s] = Int32(1)
        else:
            var shift2 = Float32(0)
            var cntf = Float32(within)
            for f in range(D):
                var m = ftz(identical_div(sums[f], cntf))
                var t = ftz(m - centers[s * D + f])
                shift2 = ftz(shift2 + ftz(identical_mul(t, t)))
                centers[s * D + f] = m
            if identical_sqrt(shift2) <= stop or iters[s] == max_iter:
                done[s] = Int32(1)
            else:
                iters[s] = iters[s] + Int32(1)


def _msi_pending_kernel(done: IPtr, ns: Int32, pending: IPtr):
    """pending[0] = 1 when any seed's done flag is still 0 (one block; every
    writer stores the same word, so the result is order free). The host reads
    this one word instead of the ns flags."""
    var tid = Int(thread_idx.x)
    for s in range(tid, Int(ns), MSI_TPB):
        if done[s] == Int32(0):
            pending[0] = Int32(1)
            break


def meanshift_idn_grid(
    ctx: DeviceContext, x: FPtr, n: Int, d: Int, bw: Float32, stop: Float32, max_iter: Int,
    centers: FPtr, ns: Int, intensity: IPtr, iters: IPtr,
) raises -> Bool:
    """The grid shifts of every seed; `centers` (ns x d) in and out,
    `intensity` / `iters` (ns, zero on entry) as `bodies.meanshift_seed`
    leaves them under K6. False (nothing done) when MSI_ON is off or past
    MSI_MAX_D features: the caller then runs `_meanshift_kernel`, whose body
    is the same blocked fold."""
    comptime if MSI_ON:
        if d > MSI_MAX_D or d <= 0 or ns <= 0 or n <= 0:
            return False
        var nch = (n + MSI_T - 1) // MSI_T
        var part = ctx.enqueue_create_buffer[DType.float32](ns * nch * d)
        var cntb = ctx.enqueue_create_buffer[DType.int32](ns * nch)
        var active = ctx.enqueue_create_buffer[DType.int32](ns+1 if C40_ACTIVE_SEEDS else 1)
        var ap = active.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var done = ctx.enqueue_create_buffer[DType.int32](ns)
        ctx.enqueue_memset(done, Int32(0))
        # `iters` counts the completed shifts and must start at zero
        var pp = part.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var cp = cntb.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var dp = done.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pend = ctx.enqueue_create_buffer[DType.int32](1)
        var pdp = pend.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var hd = List[Int32](length=1, fill=Int32(0))
        var it = 0
        # at most max_iter + 1 shifts: the stop test runs after the shift
        while it <= max_iter:
            for _g in range(MSI_GROUP):
                if it > max_iter:
                    break
                comptime if C40_ACTIVE_SEEDS:
                    ctx.enqueue_function[_c40_active_kernel](dp,Int32(ns),ap,grid_dim=1,block_dim=1)
                if C40_SEED_TILES and d <= 512:
                    ctx.enqueue_function[_c40_seed_pair_kernel](
                        x,Int32(n),Int32(d),bw,centers,dp,Int32(nch),pp,cp,Int32(ns),ap,
                        grid_dim=((ns+1)//2)*nch,block_dim=MSI_TPB,
                    )
                else:
                    ctx.enqueue_function[_msi_part_kernel](
                        x, Int32(n), Int32(d), bw, centers, dp, Int32(nch), pp, cp, ap,
                        grid_dim=ns * nch, block_dim=MSI_TPB,
                    )
                ctx.enqueue_function[_msi_finish_kernel](
                    Int32(d), Int32(nch), pp, cp, centers, dp, intensity, iters, stop, Int32(max_iter),
                    grid_dim=ns, block_dim=MSI_TPB,
                )
                it += 1
            # the done test runs on the device; the host reads one word
            ctx.enqueue_memset(pend, Int32(0))
            ctx.enqueue_function[_msi_pending_kernel](dp, Int32(ns), pdp, grid_dim=1, block_dim=MSI_TPB)
            ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=pend)
            ctx.synchronize()
            if hd[0] == Int32(0):
                break
        # the buffers outlive every launch that holds their pointers
        _ = part^
        _ = cntb^
        _ = done^
        _ = active^
        _ = pend^
        _ = hd^
        return True
    return False
