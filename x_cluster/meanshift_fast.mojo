# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MeanShift, FAST on Apple: every shift of every seed on the whole GPU
(lane/apple-fast-cluster, 2026-10-02). Switch: `MEANSHIFT_FAST_GRID` below,
taken by `x_cluster/device_ops.mojo` `DeviceOps.meanshift`: on by default
in FAST on Apple; `-D MOJOLEARN_X_CLUSTER_FAST_MEANSHIFT_OFF=1` turns it
off (the old `-D MOJOLEARN_X_CLUSTER_FAST_MEANSHIFT=1` is harmless). No
build reads the environment.

Cause: `_meanshift_team_kernel` (x_cluster/device_ops.mojo:169) takes ONE
block per seed and walks all n rows of every shift inside it, so a fit with
bin seeding (Istella 10k x 220: 18 seeds) keeps 18 blocks of the M3 Ultra
busy for the whole fit (board: meanshift Istella 13x scikit-learn). The
`MOJOLEARN_MEANSHIFT_BLOCK` opt-in (device_ops.mojo:265) is the same
one-block-per-seed shape, d <= 16 only.

Here one shift is TWO launches over a grid of (seed, row chunk) blocks:

  * `_msg_part_kernel`: block (s, c) tests its MSG_T rows against seed s's
    center (the `sq_dist_rows` chain), then thread f folds feature f over
    the rows within the bandwidth; the chunk's feature sums and its count
    go to `part[s, c]`;
  * `_msg_finish_kernel`: one block per seed folds the chunks in order,
    thread 0 runs the quotients, the shift and the stop test in feature
    order (`bodies.meanshift_seed`'s words) and marks the seed done.

The host enqueues MSG_GROUP shifts between reads of the done flags (one
synchronize per group); a done seed's blocks return at once. The addends
are the team kernel's; the order of the sum is not (bits move under FAST;
the paired quality check holds). IDENTICAL compiles none of this.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_div, identical_mul, identical_sqrt
from x_cluster.bodies import FPtr, IPtr

# Default in FAST on Apple since the M3 A/B (lane/apple-fast-cluster aaef7b261,
# n=1, Istella): meanshift 52.1 -> 34.9 ms, n_clusters 12 and silhouette
# .4035 unchanged. `-D MOJOLEARN_X_CLUSTER_FAST_MEANSHIFT_OFF=1` turns it off.
comptime MEANSHIFT_FAST_GRID = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_X_CLUSTER_FAST_MEANSHIFT_OFF"]()
)
comptime MSG_TPB = 256
comptime MSG_T = 256
"""Rows per chunk (one row per thread in the bandwidth test)."""
comptime MSG_MAX_D = 1024
comptime MSG_GROUP = 16
"""Shifts enqueued between two reads of the done flags."""


def _msg_part_kernel(
    x: FPtr, n: Int32, d: Int32, bw: Float32, centers: FPtr, done: IPtr, nch: Int32, part: FPtr,
):
    var b = Int(block_idx.x)
    var NC = Int(nch)
    var s = b // NC
    var c = b - s * NC
    var tid = Int(thread_idx.x)
    if done[s] != Int32(0):
        return
    var N = Int(n)
    var D = Int(d)
    var cen = stack_allocation[MSG_MAX_D, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var flag = stack_allocation[MSG_T, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var cnts = stack_allocation[MSG_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    for f in range(tid, D, MSG_TPB):
        cen[f] = centers[s * D + f]
    barrier()
    var t0 = c * MSG_T
    var cnt = N - t0
    if cnt > MSG_T:
        cnt = MSG_T
    var mine = 0
    for r in range(tid, cnt, MSG_TPB):
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
    var o = (s * NC + c) * (D + 1)
    for f in range(tid, D, MSG_TPB):
        var a = Float32(0)
        for r in range(cnt):
            if flag[r] != Int32(0):
                a = ftz(a + ftz(x[(t0 + r) * D + f]))
        part[o + f] = a
    if tid == 0:
        var within = 0
        for u in range(MSG_TPB):
            within += Int(cnts[u])
        part[o + D] = Float32(within)


def _msg_finish_kernel(
    d: Int32, nch: Int32, part: FPtr, centers: FPtr, done: IPtr, intensity: IPtr, iters: IPtr,
    stop: Float32, max_iter: Int32,
):
    var s = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    if done[s] != Int32(0):
        return
    var D = Int(d)
    var NC = Int(nch)
    var sums = stack_allocation[MSG_MAX_D + 1, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    for f in range(tid, D + 1, MSG_TPB):
        var a = Float32(0)
        for c in range(NC):
            a = ftz(a + part[(s * NC + c) * (D + 1) + f])
        sums[f] = a
    barrier()
    if tid == 0:
        var within = Int(sums[D])
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


def _msg_pending_kernel(done: IPtr, ns: Int32, pending: IPtr):
    """pending[0] = 1 when any seed's done flag is still 0 (one block; every
    writer stores the same word, so the result is order free). The host reads
    this one word instead of the ns flags."""
    var tid = Int(thread_idx.x)
    for s in range(tid, Int(ns), MSG_TPB):
        if done[s] == Int32(0):
            pending[0] = Int32(1)
            break


def meanshift_fast_grid(
    ctx: DeviceContext, x: FPtr, n: Int, d: Int, bw: Float32, stop: Float32, max_iter: Int,
    centers: FPtr, ns: Int, intensity: IPtr, iters: IPtr,
) raises -> Bool:
    """The grid shifts of every seed; `centers` (ns x d) in and out,
    `intensity` / `iters` (ns, zero on entry) as `bodies.meanshift_seed`
    leaves them. False (nothing done) outside FAST on Apple or past
    MSG_MAX_D features."""
    comptime if MEANSHIFT_FAST_GRID:
        if d > MSG_MAX_D or ns <= 0 or n <= 0:
            return False
        var nch = (n + MSG_T - 1) // MSG_T
        var part = ctx.enqueue_create_buffer[DType.float32](ns * nch * (d + 1))
        var done = ctx.enqueue_create_buffer[DType.int32](ns)
        ctx.enqueue_memset(done, Int32(0))
        var pp = part.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var dp = done.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pend = ctx.enqueue_create_buffer[DType.int32](1)
        var pdp = pend.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var hd = List[Int32](length=1, fill=Int32(0))
        var it = 0
        # at most max_iter + 1 shifts: the stop test runs after the shift
        while it <= max_iter:
            for _g in range(MSG_GROUP):
                if it > max_iter:
                    break
                ctx.enqueue_function[_msg_part_kernel](
                    x, Int32(n), Int32(d), bw, centers, dp, Int32(nch), pp,
                    grid_dim=ns * nch, block_dim=MSG_TPB,
                )
                ctx.enqueue_function[_msg_finish_kernel](
                    Int32(d), Int32(nch), pp, centers, dp, intensity, iters, stop, Int32(max_iter),
                    grid_dim=ns, block_dim=MSG_TPB,
                )
                it += 1
            # the done test runs on the device; the host reads one word
            ctx.enqueue_memset(pend, Int32(0))
            ctx.enqueue_function[_msg_pending_kernel](dp, Int32(ns), pdp, grid_dim=1, block_dim=MSG_TPB)
            ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=pend)
            ctx.synchronize()
            if hd[0] == Int32(0):
                break
        # the buffers outlive every launch that holds their pointers
        _ = part^
        _ = done^
        _ = pend^
        _ = hd^
        return True
    return False
