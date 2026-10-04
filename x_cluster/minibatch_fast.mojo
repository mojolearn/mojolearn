# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MiniBatchKMeans, FAST on Apple: the mini-batch steps resident on the
device (lane/apple-fast-cluster, 2026-10-02). Switch: `MINIBATCH_FAST_DEV`
below, taken by `x_cluster/minibatch.mojo` `minibatch_fit`: on by default
in FAST on Apple; `-D MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_OFF=1` turns it
off (the old `-D MOJOLEARN_X_CLUSTER_FAST_MINIBATCH=1` is harmless). No
build reads the environment.

Cause: `minibatch_fit`'s step loop (x_cluster/minibatch.mojo, `for step in
range(n_steps)`) gathers the batch on the host, uploads it (`ops.set`),
runs the one `nearest` kernel, READS the labels and distances back
(`ops.get_if`: a synchronize per step, ~4 ms on Metal) and folds the k
center updates on the host over the whole batch. Istella (100k x 220,
batch 4096, max_iter 100: up to 2,441 steps) spends the fit in those round
trips (board: minibatch-kmeans Istella 5.5x scikit-learn).

Here X stays on the device (the fit's `xs` slot), MBF_GROUP steps' batch
indices are drawn on the host (the fit's stream, in step order) and
uploaded at once, and each step is four launches with no synchronize:

  * `_mbf_assign_kernel`: a thread per batch row, the nearest center
    (centers staged in threadgroup memory), the row's label and squared
    distance, the block's distance sum (the batch inertia's partials);
  * `_mbf_sum_kernel`: a block per (center, 256-row chunk) folds the
    chunk's rows of that center per feature, with the count;
  * `_mbf_finish_kernel`: a thread per (center, feature) cell:
    `c * w`, `+ sum`, `w += count`, `* (1 / w)` (sklearn
    `update_center_dense`'s steps), into the NEXT history slot; the inertia;
  * `_mbf_reassign_kernel`: one block, sklearn `_random_reassign` on the
    updated counts (O(k) on the lead thread; the picked rows copied by the
    block). Its draws come from a SECOND splitmix64 stream seeded from the
    fit's seed, because the batch draws of the group precede them.

The host reads the group's MBF_GROUP batch inertias (one synchronize),
runs `_mini_batch_convergence` exactly as the step loop does, and on a
stop takes that step's centers and counts from the history. FAST promises
quality, not bits: the summation order and the reassignment stream
differ from the step loop's. IDENTICAL compiles none of this.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_div, identical_mul, identical_mul64
from x_cluster.bodies import FPtr, IPtr, SplitMix64

# Default in FAST on Apple since the M3 A/B (lane/apple-fast-cluster aaef7b261,
# n=1, Istella): minibatch-kmeans 388 -> 352 ms, silhouette .1167 -> .1182.
# `-D MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_OFF=1` turns it off.
comptime MINIBATCH_FAST_DEV = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_OFF"]()
)
comptime MBF_TPB = 256
comptime MBF_CH = 256
"""Batch rows per chunk of `_mbf_sum_kernel`."""
comptime MBF_MAX_KD = 4096
"""Center cells staged in threadgroup memory (16 KB)."""
comptime MBF_MAX_K = 256
comptime MBF_MAX_BATCH = 4096
"""The reassignment's pool lives in threadgroup memory (16 KB)."""
comptime MBF_GROUP = 32
"""Steps enqueued between two reads of the batch inertias."""

# lane/apple-fast-gap-cls2 (2026-10-03): MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_POOL
# (x_cluster/minibatch_ptr.mojo): X's device buffer from a pool kept between
# fits, not a fresh 880 MB allocation per fit at Istella's shape. FAST + Apple
# default since the M3 A/B (n=1, quality identical): minibatch-kmeans istella
# 256.7 -> 170.7 ms, taxi 42.5 -> 38.1 ms. -D MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_POOL_OFF
# turns it off; the old -D name stays harmless.
comptime MBK_CLS2_POOL = MINIBATCH_FAST_DEV and not is_defined["MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_POOL_OFF"]()

comptime UPtr = MutPointer[UInt64, MutAnyOrigin]

# lane/apple-fast-gap-clus3 (2026-10-03): -D MOJOLEARN_X_CLUSTER_FAST_CLS3_MBK_ROWGRP
# (now the FAST + Apple default, see below). Cause: `_mbf_assign_kernel` is a thread per
# batch row, so a 4096-row batch is 16 blocks on an 80-core M3 Ultra, and each
# thread walks its own 880-byte row k times (uncoalesced across the
# simdgroup). With the switch a batch row is a 32-thread group (8 rows a
# block, 512 blocks): lane l takes features l, l + 32, ... (a row's reads
# coalesced), keeps the k partial distances in registers, and the group folds
# them lane by lane in threadgroup memory. k <= MBF_RG_MAXK, else the old
# kernel. FAST: the distance's summation order changes (quality, not bits).
# FAST + Apple default since the M3 A/B clus3-mbk-rowgrp-istella (n=1):
# minibatch-kmeans istella 172 -> 148 ms (-14.0%), silhouette .1182 identical;
# -D MOJOLEARN_X_CLUSTER_FAST_CLS3_MBK_ROWGRP_OFF turns it off.
comptime MBK_CLS3_ROWGRP = MINIBATCH_FAST_DEV and not is_defined["MOJOLEARN_X_CLUSTER_FAST_CLS3_MBK_ROWGRP_OFF"]()
comptime MBF_RG_W = 32
"""Threads per batch row."""
comptime MBF_RG_ROWS = MBF_TPB // MBF_RG_W
comptime MBF_RG_MAXK = 16

# lane/apple-fast-w2-clres (2026-10-04), OPEN, opt-in, FAST + Apple only.
# -D MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP: `_mbf_sum_kernel` walks all 256
# rows of its chunk for every feature, testing a flag and re-reading the row's
# index from device memory for each of the ~1/k rows that belong to its center
# (256 dependent iterations per thread, per step). With the switch the block
# first compacts its center's rows (a 256-wide prefix scan of the flags, the
# row offsets kept in threadgroup memory) and each feature thread sums only
# those rows, in the SAME ascending row order: the same float sums, the same
# centers, same bits as main.
comptime MBK_W2_SUMCMP = MINIBATCH_FAST_DEV and is_defined["MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP"]()
# -D MOJOLEARN_X_CLUSTER_FAST_W2_MBK_LABRG: the fit's last pass (labels and
# distances of all n rows, x_cluster/minibatch_ptr.mojo) is
# `DeviceOps.nearest`, a thread per row that walks its 880-byte Istella row k
# times (uncoalesced, 1M rows). With the switch it is `_mbf_label_rg_kernel`,
# the CLS3_ROWGRP batch assignment over every row: a 32-thread group per row,
# coalesced reads, one pass over X. k <= MBF_RG_MAXK, else the old pass.
# FAST: the distance's summation order changes (labels may flip only at
# near-ties; quality checked against main).
comptime MBK_W2_LABRG = MINIBATCH_FAST_DEV and is_defined["MOJOLEARN_X_CLUSTER_FAST_W2_MBK_LABRG"]()


@always_inline
def _sm_next(st: UPtr) -> UInt64:
    """`bodies.SplitMix64.next` on a device-resident state."""
    var s = st[0] + UInt64(0x9E3779B97F4A7C15)
    st[0] = s
    var z = s
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _mbf_assign_kernel(
    x: FPtr, idx: IPtr, batch: Int32, c: FPtr, k: Int32, d: Int32, lab: IPtr, dist: FPtr, ipart: FPtr,
):
    var tid = Int(thread_idx.x)
    var t = Int(block_idx.x) * MBF_TPB + tid
    var K = Int(k)
    var D = Int(d)
    var cs = stack_allocation[MBF_MAX_KD, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var red = stack_allocation[MBF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    for q in range(tid, K * D, MBF_TPB):
        cs[q] = c[q]
    barrier()
    var v = Float32(0)
    if t < Int(batch):
        var p = Int(idx[t])
        var best = Float32(0)
        var bi = 0
        for j in range(K):
            var acc = Float32(0)
            for f in range(D):
                var tt = ftz(ftz(cs[j * D + f]) - ftz(x[p * D + f]))
                acc = ftz(acc + ftz(identical_mul(tt, tt)))
            if j == 0 or acc < best:
                best = acc
                bi = j
        lab[t] = Int32(bi)
        dist[t] = best
        v = best
    red[tid] = v
    barrier()
    var off = MBF_TPB // 2
    while off > 0:
        if tid < off:
            red[tid] = red[tid] + red[tid + off]
        barrier()
        off //= 2
    if tid == 0:
        ipart[Int(block_idx.x)] = red[0]


def _mbf_assign_rg_kernel(
    x: FPtr, idx: IPtr, batch: Int32, c: FPtr, k: Int32, d: Int32, lab: IPtr, dist: FPtr, ipart: FPtr,
):
    """MBK_CLS3_ROWGRP: `_mbf_assign_kernel` with a 32-thread group per row
    (see the switch). Ties to the lower center, as there."""
    var tid = Int(thread_idx.x)
    var g = tid // MBF_RG_W
    var l = tid - g * MBF_RG_W
    var t = Int(block_idx.x) * MBF_RG_ROWS + g
    var K = Int(k)
    var D = Int(d)
    var red = stack_allocation[MBF_TPB * MBF_RG_MAXK, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var rd = stack_allocation[MBF_RG_ROWS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = SIMD[DType.float32, MBF_RG_MAXK](0)
    var live = t < Int(batch)
    if live:
        var p = Int(idx[t])
        for f in range(l, D, MBF_RG_W):
            var xv = ftz(x[p * D + f])
            comptime for j in range(MBF_RG_MAXK):
                if j < K:
                    var tt = ftz(ftz(c[j * D + f]) - xv)
                    acc[j] = ftz(acc[j] + ftz(identical_mul(tt, tt)))
    comptime for j in range(MBF_RG_MAXK):
        red[tid * MBF_RG_MAXK + j] = acc[j]
    barrier()
    # lane j < K folds center j's 32 partials in lane order
    if l < K:
        var s = Float32(0)
        for q in range(MBF_RG_W):
            s = ftz(s + red[(g * MBF_RG_W + q) * MBF_RG_MAXK + l])
        red[(g * MBF_RG_W) * MBF_RG_MAXK + MBF_RG_MAXK * MBF_RG_W // 2 + l] = s
    barrier()
    if l == 0:
        var v = Float32(0)
        if live:
            var base = (g * MBF_RG_W) * MBF_RG_MAXK + MBF_RG_MAXK * MBF_RG_W // 2
            var best = red[base]
            var bi = 0
            for j in range(1, K):
                var a = red[base + j]
                if a < best:
                    best = a
                    bi = j
            lab[t] = Int32(bi)
            dist[t] = best
            v = best
        rd[g] = v
    barrier()
    if tid == 0:
        var a = Float32(0)
        for q in range(MBF_RG_ROWS):
            a = a + rd[q]
        ipart[Int(block_idx.x)] = a


def _mbf_sum_kernel(
    x: FPtr, idx: IPtr, batch: Int32, lab: IPtr, k: Int32, d: Int32, nchunk: Int32, part: FPtr, cpart: IPtr,
):
    var b = Int(block_idx.x)
    var NC = Int(nchunk)
    var j = b // NC
    var c = b - j * NC
    var tid = Int(thread_idx.x)
    var D = Int(d)
    var flag = stack_allocation[MBF_CH, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var cnts = stack_allocation[MBF_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var t0 = c * MBF_CH
    var cnt = Int(batch) - t0
    if cnt > MBF_CH:
        cnt = MBF_CH
    var mine = 0
    for r in range(tid, cnt, MBF_TPB):
        var w = Int(lab[t0 + r]) == j
        flag[r] = Int32(1) if w else Int32(0)
        if w:
            mine += 1
    cnts[tid] = Int32(mine)
    barrier()
    var o = (j * NC + c) * D
    for f in range(tid, D, MBF_TPB):
        var a = Float32(0)
        for r in range(cnt):
            if flag[r] != Int32(0):
                a = ftz(a + ftz(x[Int(idx[t0 + r]) * D + f]))
        part[o + f] = a
    if tid == 0:
        var within = 0
        for u in range(MBF_TPB):
            within += Int(cnts[u])
        cpart[j * NC + c] = Int32(within)


def _mbf_sum_cmp_kernel(
    x: FPtr, idx: IPtr, batch: Int32, lab: IPtr, k: Int32, d: Int32, nchunk: Int32, part: FPtr, cpart: IPtr,
):
    """MBK_W2_SUMCMP: `_mbf_sum_kernel` over the compacted rows of center
    `j` in chunk `c` (ascending row order, so the same sums). MBF_CH ==
    MBF_TPB: thread `tid` owns chunk row `tid`."""
    comptime assert MBF_CH == MBF_TPB, "the compacted sum gives each thread one chunk row"
    var b = Int(block_idx.x)
    var NC = Int(nchunk)
    var j = b // NC
    var c = b - j * NC
    var tid = Int(thread_idx.x)
    var D = Int(d)
    var scan = stack_allocation[MBF_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var rows = stack_allocation[MBF_CH, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var t0 = c * MBF_CH
    var cnt = Int(batch) - t0
    if cnt > MBF_CH:
        cnt = MBF_CH
    var mine = Int32(0)
    if tid < cnt and Int(lab[t0 + tid]) == j:
        mine = Int32(1)
    scan[tid] = mine
    barrier()
    # inclusive Hillis-Steele scan of the 256 flags
    var off = 1
    while off < MBF_TPB:
        var v = scan[tid]
        if tid >= off:
            v = v + scan[tid - off]
        barrier()
        scan[tid] = v
        barrier()
        off *= 2
    var m = Int(scan[MBF_TPB - 1])
    if mine != Int32(0):
        rows[Int(scan[tid]) - 1] = idx[t0 + tid]
    barrier()
    var o = (j * NC + c) * D
    for f in range(tid, D, MBF_TPB):
        var a = Float32(0)
        for q in range(m):
            a = ftz(a + ftz(x[Int(rows[q]) * D + f]))
        part[o + f] = a
    if tid == 0:
        cpart[j * NC + c] = Int32(m)


def _mbf_label_rg_kernel(x: FPtr, n: Int32, c: FPtr, k: Int32, d: Int32, lab: IPtr, dist: FPtr):
    """MBK_W2_LABRG: `_mbf_assign_rg_kernel` over rows 0..n-1 (no index
    list, no inertia partials): a 32-thread group per row, lane l takes
    features l, l + 32, ...; ties to the lower center."""
    var tid = Int(thread_idx.x)
    var g = tid // MBF_RG_W
    var l = tid - g * MBF_RG_W
    var t = Int(block_idx.x) * MBF_RG_ROWS + g
    var K = Int(k)
    var D = Int(d)
    var red = stack_allocation[MBF_TPB * MBF_RG_MAXK, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = SIMD[DType.float32, MBF_RG_MAXK](0)
    var live = t < Int(n)
    if live:
        for f in range(l, D, MBF_RG_W):
            var xv = ftz(x[t * D + f])
            comptime for j in range(MBF_RG_MAXK):
                if j < K:
                    var tt = ftz(ftz(c[j * D + f]) - xv)
                    acc[j] = ftz(acc[j] + ftz(identical_mul(tt, tt)))
    comptime for j in range(MBF_RG_MAXK):
        red[tid * MBF_RG_MAXK + j] = acc[j]
    barrier()
    if l < K:
        var s = Float32(0)
        for q in range(MBF_RG_W):
            s = ftz(s + red[(g * MBF_RG_W + q) * MBF_RG_MAXK + l])
        red[(g * MBF_RG_W) * MBF_RG_MAXK + MBF_RG_MAXK * MBF_RG_W // 2 + l] = s
    barrier()
    if l == 0 and live:
        var base = (g * MBF_RG_W) * MBF_RG_MAXK + MBF_RG_MAXK * MBF_RG_W // 2
        var best = red[base]
        var bi = 0
        for j in range(1, K):
            var a = red[base + j]
            if a < best:
                best = a
                bi = j
        lab[t] = Int32(bi)
        dist[t] = best


def mbk_labels_rg(ctx: DeviceContext, x: FPtr, n: Int, c: FPtr, k: Int, d: Int, lab: IPtr, dist: FPtr) raises -> Bool:
    """MBK_W2_LABRG: enqueue the all-rows labelling (`lab`, `dist`, n each).
    False (nothing enqueued) outside the switch or for k > MBF_RG_MAXK."""
    comptime if MBK_W2_LABRG:
        if k < 1 or k > MBF_RG_MAXK or n < 1 or d < 1:
            return False
        ctx.enqueue_function[_mbf_label_rg_kernel](
            x, Int32(n), c, Int32(k), Int32(d), lab, dist,
            grid_dim=(n + MBF_RG_ROWS - 1) // MBF_RG_ROWS, block_dim=MBF_TPB,
        )
        return True
    return False


def _mbf_finish_kernel(
    c_in: FPtr, w_in: FPtr, part: FPtr, cpart: IPtr, nchunk: Int32, k: Int32, d: Int32,
    c_out: FPtr, w_out: FPtr, ipart: FPtr, nblk: Int32, inertia: FPtr,
):
    var cell = Int(block_idx.x) * MBF_TPB + Int(thread_idx.x)
    var D = Int(d)
    var NC = Int(nchunk)
    if cell == 0:
        var acc = Float32(0)
        for b in range(Int(nblk)):
            acc = acc + ipart[b]
        inertia[0] = acc
    if cell < Int(k) * D:
        var j = cell // D
        var f = cell - j * D
        var wsum = Float32(0)
        for c in range(NC):
            wsum = ftz(wsum + Float32(cpart[j * NC + c]))
        if wsum > Float32(0):
            var s = Float32(0)
            for c in range(NC):
                s = ftz(s + part[(j * NC + c) * D + f])
            var v = ftz(identical_mul(c_in[cell], w_in[j]))
            v = ftz(v + s)
            var wn = ftz(w_in[j] + wsum)
            var alpha = ftz(identical_div(Float32(1), wn))
            c_out[cell] = ftz(identical_mul(v, alpha))
            if f == 0:
                w_out[j] = wn
        else:
            c_out[cell] = c_in[cell]
            if f == 0:
                w_out[j] = w_in[j]


def _mbf_reassign_kernel(
    x: FPtr, idx: IPtr, batch: Int32, c: FPtr, w_in: FPtr, w: FPtr, k: Int32, d: Int32, ratio: Float32,
    rng: UPtr, since: IPtr,
):
    """sklearn `_random_reassign` + the reassignment of `_mini_batch_step`
    (x_cluster/minibatch.mojo `minibatch_step`, `if reassign and ratio > 0`)
    on the updated counts `w` (the previous step's counts `w_in` decide
    `any_empty`); `c` is the updated centers, in place."""
    var tid = Int(thread_idx.x)
    var K = Int(k)
    var D = Int(d)
    var B = Int(batch)
    var pool = stack_allocation[MBF_MAX_BATCH, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var to = stack_allocation[MBF_MAX_K, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var picked = stack_allocation[MBF_MAX_K, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var st = stack_allocation[2, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    for t in range(tid, B, MBF_TPB):
        pool[t] = Int32(t)
    for j in range(tid, K, MBF_TPB):
        to[j] = Int32(0)
    barrier()
    if tid == 0:
        var sc = Int(since[0]) + B
        var any_empty = False
        for j in range(K):
            if w_in[j] == Float32(0):
                any_empty = True
        var reassign = any_empty or sc >= 10 * K
        if reassign:
            sc = 0
        since[0] = Int32(sc)
        var nre = 0
        if reassign and ratio > Float32(0):
            var wmax = w[0]
            for j in range(1, K):
                if w[j] > wmax:
                    wmax = w[j]
            var thr = ratio * wmax
            for j in range(K):
                if w[j] < thr:
                    to[j] = Int32(1)
                    nre += 1
            var half = B // 2
            if 2 * nre > B:
                # np.argsort(weight_sums)[half:] stay: a stable ascending sort
                # by weight (`picked` holds the order until the picks)
                for j in range(K):
                    picked[j] = Int32(j)
                for a in range(1, K):
                    var b = a
                    while b > 0 and w[Int(picked[b - 1])] > w[Int(picked[b])]:
                        var tmp = picked[b - 1]
                        picked[b - 1] = picked[b]
                        picked[b] = tmp
                        b -= 1
                for q in range(half, K):
                    to[Int(picked[q])] = Int32(0)
                nre = 0
                for j in range(K):
                    if to[j] != Int32(0):
                        nre += 1
            if nre > 0:
                # choice(batch, nre, replace=False): a partial Fisher-Yates
                for q in range(nre):
                    var r = q + Int(_sm_next(rng) % UInt64(B - q))
                    var tmp = pool[q]
                    pool[q] = pool[r]
                    pool[r] = tmp
                    picked[q] = pool[q]
                var wmin = Float32(0)
                var have = False
                for j in range(K):
                    if to[j] == Int32(0) and (not have or w[j] < wmin):
                        wmin = w[j]
                        have = True
                st[1] = wmin
        st[0] = Float32(nre)
    barrier()
    var nre = Int(st[0])
    if nre > 0:
        var q = 0
        for j in range(K):
            if to[j] != Int32(0):
                var p = Int(idx[Int(picked[q])])
                for f in range(tid, D, MBF_TPB):
                    c[j * D + f] = x[p * D + f]
                if tid == 0:
                    w[j] = st[1]
                q += 1


def minibatch_fast_steps(
    ctx: DeviceContext, x: FPtr, n: Int, d: Int, k: Int, batch: Int, n_steps: Int, max_no_improvement: Int,
    ratio: Float64, seed: UInt64, mut rng: SplitMix64, mut c: List[Float32], mut w: List[Float32],
    mut steps_done: Int,
) raises -> Bool:
    """The step loop of `minibatch_fit` (unit weights, tol <= 0) on the
    device: `c` (k x d) and `w` (k) in and out, `steps_done` the steps run.
    False (nothing done) outside FAST on Apple or past the shape caps."""
    comptime if MINIBATCH_FAST_DEV:
        if k * d > MBF_MAX_KD or k > MBF_MAX_K or batch > MBF_MAX_BATCH or batch < 1 or n_steps < 1 or k < 1:
            return False
        var G = MBF_GROUP
        var kd = k * d
        var rg = False
        comptime if MBK_CLS3_ROWGRP:
            rg = k <= MBF_RG_MAXK
        var nblk = (batch + MBF_TPB - 1) // MBF_TPB
        if rg:
            nblk = (batch + MBF_RG_ROWS - 1) // MBF_RG_ROWS
        var nchunk = (batch + MBF_CH - 1) // MBF_CH
        var ncell_blk = (kd + MBF_TPB - 1) // MBF_TPB
        var d_idx = ctx.enqueue_create_buffer[DType.int32](G * batch)
        var d_lab = ctx.enqueue_create_buffer[DType.int32](batch)
        var d_dist = ctx.enqueue_create_buffer[DType.float32](batch)
        var d_ipart = ctx.enqueue_create_buffer[DType.float32](nblk)
        var d_part = ctx.enqueue_create_buffer[DType.float32](k * nchunk * d)
        var d_cpart = ctx.enqueue_create_buffer[DType.int32](k * nchunk)
        var d_c = ctx.enqueue_create_buffer[DType.float32]((G + 1) * kd)
        var d_w = ctx.enqueue_create_buffer[DType.float32]((G + 1) * k)
        var d_in = ctx.enqueue_create_buffer[DType.float32](G)
        var d_rng = ctx.enqueue_create_buffer[DType.uint64](1)
        var d_since = ctx.enqueue_create_buffer[DType.int32](1)
        ctx.enqueue_memset(d_since, Int32(0))
        var h_rng = List[UInt64](length=1, fill=seed ^ UInt64(0x5851F42D4C957F2D))
        var c0 = d_c.create_sub_buffer[DType.float32](0, kd)
        var w0 = d_w.create_sub_buffer[DType.float32](0, k)
        ctx.enqueue_copy(dst_buf=c0, src_ptr=c.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=w0, src_ptr=w.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_rng, src_ptr=h_rng.unsafe_ptr())
        ctx.synchronize()
        var p_idx = d_idx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_lab = d_lab.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_dist = d_dist.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_ipart = d_ipart.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_part = d_part.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_cpart = d_cpart.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_c = d_c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_w = d_w.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_in = d_in.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_rng = d_rng.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_since = d_since.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var h_in = List[Float32](length=G, fill=Float32(0))
        # _mini_batch_convergence's state, as the step loop keeps it
        var ewa = Float64(0)
        var have_ewa = False
        var ewa_min = Float64(0)
        var have_min = False
        var no_improvement = 0
        var step0 = 0
        var in0 = 0
        var final_slot = 0
        var stopped = False
        while step0 < n_steps and not stopped:
            var gsz = n_steps - step0
            if gsz > G:
                gsz = G
            var h_idx = List[Int32](capacity=gsz * batch)
            for _s in range(gsz):
                for _t in range(batch):
                    h_idx.append(Int32(rng.below(n)))
            var iv = d_idx.create_sub_buffer[DType.int32](0, gsz * batch)
            ctx.enqueue_copy(dst_buf=iv, src_ptr=h_idx.unsafe_ptr())
            for g in range(gsz):
                var slot_in = in0 if g == 0 else g
                var slot_out = g + 1
                var pi = p_idx + g * batch
                var ci = p_c + slot_in * kd
                var wi = p_w + slot_in * k
                var co = p_c + slot_out * kd
                var wo = p_w + slot_out * k
                if rg:
                    ctx.enqueue_function[_mbf_assign_rg_kernel](
                        x, pi, Int32(batch), ci, Int32(k), Int32(d), p_lab, p_dist, p_ipart,
                        grid_dim=nblk, block_dim=MBF_TPB,
                    )
                else:
                    ctx.enqueue_function[_mbf_assign_kernel](
                        x, pi, Int32(batch), ci, Int32(k), Int32(d), p_lab, p_dist, p_ipart,
                        grid_dim=nblk, block_dim=MBF_TPB,
                    )
                comptime if MBK_W2_SUMCMP:
                    ctx.enqueue_function[_mbf_sum_cmp_kernel](
                        x, pi, Int32(batch), p_lab, Int32(k), Int32(d), Int32(nchunk), p_part, p_cpart,
                        grid_dim=k * nchunk, block_dim=MBF_TPB,
                    )
                else:
                    ctx.enqueue_function[_mbf_sum_kernel](
                        x, pi, Int32(batch), p_lab, Int32(k), Int32(d), Int32(nchunk), p_part, p_cpart,
                        grid_dim=k * nchunk, block_dim=MBF_TPB,
                    )
                ctx.enqueue_function[_mbf_finish_kernel](
                    ci, wi, p_part, p_cpart, Int32(nchunk), Int32(k), Int32(d), co, wo, p_ipart, Int32(nblk),
                    p_in + g, grid_dim=ncell_blk, block_dim=MBF_TPB,
                )
                ctx.enqueue_function[_mbf_reassign_kernel](
                    x, pi, Int32(batch), co, wi, wo, Int32(k), Int32(d), Float32(ratio), p_rng, p_since,
                    grid_dim=1, block_dim=MBF_TPB,
                )
            ctx.enqueue_copy(dst_ptr=h_in.unsafe_ptr(), src_buf=d_in)
            ctx.synchronize()
            _ = h_idx^
            final_slot = gsz
            for g in range(gsz):
                var step = step0 + g
                var bi = Float64(h_in[g]) / Float64(batch)
                if step == 0:
                    continue
                if not have_ewa:
                    ewa = bi
                    have_ewa = True
                else:
                    var alpha = identical_mul64(Float64(batch), 2.0) / Float64(n + 1)
                    if alpha > 1:
                        alpha = 1
                    ewa = identical_mul64(ewa, 1 - alpha) + identical_mul64(bi, alpha)
                if not have_min or ewa < ewa_min:
                    no_improvement = 0
                    ewa_min = ewa
                    have_min = True
                else:
                    no_improvement += 1
                if max_no_improvement >= 0 and no_improvement >= max_no_improvement:
                    stopped = True
                    final_slot = g + 1
                    step0 = step + 1
                    break
            if not stopped:
                step0 += gsz
                in0 = G
        steps_done = step0
        var cf = d_c.create_sub_buffer[DType.float32](final_slot * kd, kd)
        var wf = d_w.create_sub_buffer[DType.float32](final_slot * k, k)
        ctx.enqueue_copy(dst_ptr=c.unsafe_ptr(), src_buf=cf)
        ctx.enqueue_copy(dst_ptr=w.unsafe_ptr(), src_buf=wf)
        ctx.synchronize()
        # every buffer outlives the launches that hold its pointer
        _ = d_idx^
        _ = d_lab^
        _ = d_dist^
        _ = d_ipart^
        _ = d_part^
        _ = d_cpart^
        _ = d_c^
        _ = d_w^
        _ = d_in^
        _ = d_rng^
        _ = d_since^
        _ = h_rng^
        _ = h_in^
        return True
    return False
