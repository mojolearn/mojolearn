# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CLUSTER LANE'S DEVICE COLUMN (lane/algos-cluster): `ClusterOps` on a
GPU. Each primitive is one kernel whose thread `t` calls the `x_cluster/
bodies.mojo` body for index `t`; nothing is folded across threads, so no
launch shape can move a bit. Only the GPU binding imports this file."""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.os import getenv
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from std.ffi import _Global
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL, ftz, identical_div, identical_mul, identical_sqrt

from x_cluster.bodies import (
    FPtr,
    sq_dist_rows,
    IPtr,
    cov_cell,
    chain_add,
    cov_final,
    cov_term,
    argmax_row,
    exp_cell,
    mean_final,
    nk_final,
    xk_term,
    gauss_q_cell,
    nk_cell,
    pdist_cell,
    resp_row,
    xk_cell,
    ap_availability_col,
    ap_exemplar_cell,
    ap_noise_cell,
    ap_r_update,
    meanshift_seed,
    nearest_row,
    sqdist_cell,
    sqrt_cell,
    tree_descend,
    ward_cell,
    lance_williams,
    LINK_WARD,
)
from cluster.estimator import kmeans_fit, kmeans_fit_rows
from cluster.impl.kmeans_params import METRIC_L2_EXPANDED
from gemm.checks.gemm_identical import identical_gemm_into, identical_gemm_workspace_max_floats
from gemm.contract import OP_TN
from mixture.checks.mstep import center_scale_kernel, cov_finish_kernel, means_divide_kernel
from x_cluster.ops import ClusterOps
from x_cluster.device_post import (
    PTPB,
    RTPB,
    SCAN_PER,
    UPtr,
    agc_comp_kernel,
    agc_edges_kernel,
    agc_join_kernel,
    agc_members_kernel,
    agc_prop_kernel,
    agc_root_kernel,
    agc_rowbest_kernel,
    agc_start_kernel,
    ap_acc_kernel,
    ap_best_kernel,
    ap_c_kernel,
    ap_conv_kernel,
    ap_equal_kernel,
    ap_exsc_kernel,
    ap_inv_kernel,
    ap_isc_kernel,
    ap_label_kernel,
    bin_count_kernel,
    bin_key_kernel,
    bin_scatter_kernel,
    compact_rows_kernel,
    count_neg_kernel,
    ff_chunk_kernel,
    fill_i_kernel,
    first_equal_kernel,
    kpp_search_kernel,
    kpp_take_kernel,
    lowest_part_kernel,
    max_part_kernel,
    ms_last_kernel,
    ms_mark_kernel,
    ms_noise_kernel,
    ms_rank_kernel,
    negate_kernel,
    nonneg_kernel,
    onehot_kernel,
    optics_flag_kernel,
    optics_init_kernel,
    optics_label_kernel,
    optics_part_kernel,
    optics_step_kernel,
    pgrid,
    rand_resp_kernel,
    scan_out_kernel,
    scan_part_kernel,
    set_diag_kernel,
    sign_side_kernel,
)
from x_cluster.post_bodies import FM_FF, FM_MIN, FM_PROD, FM_VAL, FM_WMIN, FOLD_CHUNK, ff_of_f64
from std.gpu import WARP_SIZE
from std.gpu.primitives.warp import shuffle_idx
from x_cluster.minibatch_cells import mb_center_wsum, mb_center_word
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for

comptime TPB = 128


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _sqdist_kernel(a: FPtr, na: Int32, b: FPtr, nb: Int32, d: Int32, dst: FPtr):
    var t = _tid()
    if t < Int(na) * Int(nb):
        sqdist_cell(a, Int(na), b, Int(nb), Int(d), dst, t)



def _gather_rows_kernel(src: FPtr, d: Int32, idx: IPtr, m: Int32, dst: FPtr):
    """Block t copies row idx[t] (no division per element)."""
    var t = Int(block_idx.x)
    var dd = Int(d)
    if t < Int(m):
        var s0 = Int(idx[t]) * dd
        var d0 = t * dd
        var f = Int(thread_idx.x)
        while f < dd:
            dst[d0 + f] = src[s0 + f]
            f += Int(block_dim.x)


# lane/neural-pass112 (2026-10-02): the pairwise squared distances in tiles.
# `_sqdist_kernel` is one thread per cell, each reading its two rows from
# device memory (M4, istella 10K x 220 self-distances for MeanShift's
# bandwidth: 0.63 s). Here a block owns SQT_B x SQT_B cells and stages
# SQT_K features of its SQT_B rows of a and of b in threadgroup memory;
# each thread folds SQT_R x SQT_R cells, every cell its own chain over the
# features ascending with `sq_dist_rows`' statements (the staged words are
# already flushed: ftz is idempotent), so every word is the same.
# `MOJOLEARN_XC_SQDIST_TILED=0` restores the per-cell kernel.
comptime SQT_R = 4
comptime SQT_TD = 16
comptime SQT_B = SQT_R * SQT_TD
comptime SQT_K = 16
comptime SQT_BYTES = 2 * SQT_B * SQT_K * 4
comptime SQDIST_TILED = lib_smem_page_fits_for[TARGET_COLUMN, SQT_BYTES]()


def _sqdist_tiled_kernel(a: FPtr, na: Int32, b: FPtr, nb: Int32, d: Int32, dst: FPtr):
    var NA = Int(na)
    var NB = Int(nb)
    var D = Int(d)
    var tid = Int(thread_idx.x)
    var ti = tid // SQT_TD
    var tj = tid % SQT_TD
    var i0 = Int(block_idx.y) * SQT_B
    var j0 = Int(block_idx.x) * SQT_B
    var sa = stack_allocation[SQT_B * SQT_K, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sb = stack_allocation[SQT_B * SQT_K, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = SIMD[DType.float32, SQT_R * SQT_R](0)
    var f0 = 0
    while f0 < D:
        var kc = min(SQT_K, D - f0)
        barrier()
        for u in range(tid, SQT_B * SQT_K, SQT_TD * SQT_TD):
            var r = u // SQT_K
            var k = u % SQT_K
            var ra = i0 + r
            var rb = j0 + r
            var ok = k < kc
            sa[u] = ftz(a[min(ra, NA - 1) * D + f0 + min(k, kc - 1)]) if (ok and ra < NA) else Float32(0)
            sb[u] = ftz(b[min(rb, NB - 1) * D + f0 + min(k, kc - 1)]) if (ok and rb < NB) else Float32(0)
        barrier()
        for k in range(kc):
            var av = SIMD[DType.float32, SQT_R]()
            var bv = SIMD[DType.float32, SQT_R]()
            comptime for q in range(SQT_R):
                av[q] = sa[(ti + q * SQT_TD) * SQT_K + k]
                bv[q] = sb[(tj + q * SQT_TD) * SQT_K + k]
            comptime for qi in range(SQT_R):
                comptime for qj in range(SQT_R):
                    var t = ftz(av[qi] - bv[qj])
                    acc[qi * SQT_R + qj] = ftz(acc[qi * SQT_R + qj] + ftz(identical_mul(t, t)))
        f0 += kc
    comptime for qi in range(SQT_R):
        comptime for qj in range(SQT_R):
            var i = i0 + ti + qi * SQT_TD
            var j = j0 + tj + qj * SQT_TD
            if i < NA and j < NB:
                dst[i * NB + j] = acc[qi * SQT_R + qj]


def _nearest_kernel(a: FPtr, na: Int32, b: FPtr, nb: Int32, d: Int32, labels: IPtr, dist: FPtr):
    var t = _tid()
    if t < Int(na):
        nearest_row(a, b, Int(nb), Int(d), labels, dist, t)


def _sqrt_kernel(x: FPtr, n: Int32):
    var t = _tid()
    if t < Int(n):
        sqrt_cell(x, t)


comptime KTH_TPB = 256


# DEVIATION 5120 (the device order statistic by a four-pass radix select on
# the float bits, one block per row). Row 200; kth_check (5120 arm).
def _kth_kernel(m: FPtr, n_rows: Int32, n_cols: Int32, k: Int32, dst: FPtr):
    """`bodies.kth_smallest_row` for row `block_idx.x`, the SAME value by a
    different exact route: the k-th smallest masked bit pattern found one
    byte at a time, most significant first (256-bin shared histograms of
    integer counts: every interleaving of the atomics gives the same counts,
    so no launch shape can move a bit). The bisection answers the smallest
    `v` in [0, +inf] with `count(bits <= v) >= k`, which is the k-th
    smallest pattern clamped to +inf (a NaN pattern, or `k` past the row,
    reads +inf; `k <= 0` reads +0). The old one-thread-per-row walk made 31
    passes over the row on ONE thread, the single-row median of
    AffinityPropagation 31 serial passes over n^2 values."""
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nc = Int(n_cols)
    var hist = stack_allocation[256, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var state = stack_allocation[3, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    if tid == 0:
        state[0] = Int32(0)  # the prefix found so far (bits above `shift`)
        state[1] = Int32(k)  # the rank still wanted inside that prefix
        state[2] = Int32(0)  # 1: the row holds fewer than k values
    barrier()
    for step in range(4):
        var shift = 24 - 8 * step
        for b in range(tid, 256, KTH_TPB):
            hist[b] = Int32(0)
        barrier()
        var prefix = UInt32(state[0])
        for j in range(tid, nc, KTH_TPB):
            var bits = bitcast[DType.uint32](m[row * nc + j]) & UInt32(0x7FFFFFFF)
            var above = UInt32(0) if step == 0 else (bits >> UInt32(shift + 8)) << UInt32(shift + 8)
            if above == prefix:
                _ = Atomic.fetch_add(hist.unsafe_offset(Int((bits >> UInt32(shift)) & UInt32(0xFF))), Int32(1))
        barrier()
        if tid == 0:
            var rem = Int(state[1])
            var acc = 0
            var chosen = -1
            for b in range(256):
                var c = Int(hist[b])
                if acc + c >= rem:
                    chosen = b
                    break
                acc += c
            if chosen < 0:
                state[2] = Int32(1)
                chosen = 255
            state[0] = Int32(prefix | (UInt32(chosen) << UInt32(shift)))
            state[1] = Int32(rem - acc)
        barrier()
    if tid == 0:
        var r = UInt32(state[0])
        if state[2] != Int32(0) or r > UInt32(0x7F800000):
            r = UInt32(0x7F800000)
        dst[row] = bitcast[DType.float32](r)


def _meanshift_kernel(
    x: FPtr, n: Int32, d: Int32, bw: Float32, stop: Float32, max_iter: Int32,
    centers: FPtr, ns: Int32, scratch: FPtr, intensity: IPtr, iters: IPtr,
):
    var t = _tid()
    if t < Int(ns):
        meanshift_seed(x, Int(n), Int(d), bw, stop, Int(max_iter), centers, scratch, intensity, iters, t)


# lane/neural-pass111 (2026-10-02): one BLOCK per seed, IDENTICAL bits.
# `_meanshift_kernel` walks all n rows of every shift on one thread per
# seed: istella 10K x 220 with bin seeding is 18 seeds, so 18 threads ran
# the whole fit (M4: 10.1 s of the 10.9 s fit; sklearn 1.7 s). Here a
# block takes a seed: per tile of MST_T rows every thread tests its rows
# against the bandwidth (`sq_dist_rows`' chain on the center, which lives
# in threadgroup memory), then thread f folds feature f over the tile's
# rows within the bandwidth, ascending, as `meanshift_seed` does; thread 0
# runs the quotients, the shift and the stop test in feature order. Every
# word is `meanshift_seed`'s. d > MST_MAX_D (or a page that does not fit)
# runs `_meanshift_kernel`; `MOJOLEARN_MEANSHIFT_TEAM=0` restores it.
comptime MST_TPB = 256
comptime MST_T = 1024
comptime MST_MAX_D = 1024
comptime MST_U = 16
comptime MST_BYTES = (2 * MST_MAX_D + MST_T + MST_TPB + 4) * 4
comptime MEANSHIFT_TEAM = lib_smem_page_fits_for[TARGET_COLUMN, MST_BYTES]()


def _meanshift_team_kernel(
    x: FPtr, n: Int32, d: Int32, bw: Float32, stop: Float32, max_iter: Int32,
    centers: FPtr, intensity: IPtr, iters: IPtr,
):
    var s = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var N = Int(n)
    var D = Int(d)
    var cen = stack_allocation[MST_MAX_D, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sums = stack_allocation[MST_MAX_D, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var flag = stack_allocation[MST_T, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var cnts = stack_allocation[MST_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var state = stack_allocation[4, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    for f in range(tid, D, MST_TPB):
        cen[f] = centers[s * D + f]
    if tid == 0:
        state[0] = Int32(0)  # 1: the seed is done
        state[1] = Int32(0)  # completed shifts
        state[2] = Int32(0)  # rows within the bandwidth at the last shift
    barrier()
    while True:
        for f in range(tid, D, MST_TPB):
            sums[f] = Float32(0)
        var mine = 0
        var t0 = 0
        while t0 < N:
            var cnt = min(MST_T, N - t0)
            barrier()
            for r in range(tid, cnt, MST_TPB):
                var p = t0 + r
                var acc = Float32(0)
                for f in range(D):
                    var t = ftz(ftz(cen[f]) - ftz(x[p * D + f]))
                    acc = ftz(acc + ftz(identical_mul(t, t)))
                var w = identical_sqrt(acc) <= bw
                flag[r] = Int32(1) if w else Int32(0)
                if w:
                    mine += 1
            barrier()
            for f in range(tid, D, MST_TPB):
                var a = sums[f]
                var r = 0
                while r + MST_U <= cnt:
                    var bv = SIMD[DType.float32, MST_U]()
                    var bf = SIMD[DType.int32, MST_U]()
                    comptime for u in range(MST_U):
                        bf[u] = flag[r + u]
                        bv[u] = x[(t0 + r + u) * D + f]
                    comptime for u in range(MST_U):
                        if bf[u] != Int32(0):
                            a = ftz(a + ftz(bv[u]))
                    r += MST_U
                while r < cnt:
                    if flag[r] != Int32(0):
                        a = ftz(a + ftz(x[(t0 + r) * D + f]))
                    r += 1
                sums[f] = a
            t0 += cnt
        cnts[tid] = Int32(mine)
        barrier()
        if tid == 0:
            var within = 0
            for u in range(MST_TPB):
                within += Int(cnts[u])
            state[2] = Int32(within)
            if within == 0:
                state[0] = Int32(1)
            else:
                var shift2 = Float32(0)
                var c = Float32(within)
                for f in range(D):
                    var m = ftz(identical_div(sums[f], c))
                    var t = ftz(m - cen[f])
                    shift2 = ftz(shift2 + ftz(identical_mul(t, t)))
                    cen[f] = m
                if identical_sqrt(shift2) <= stop or Int(state[1]) == Int(max_iter):
                    state[0] = Int32(1)
                else:
                    state[1] = state[1] + Int32(1)
        barrier()
        if state[0] != Int32(0):
            break
    for f in range(tid, D, MST_TPB):
        centers[s * D + f] = cen[f]
    if tid == 0:
        intensity[s] = state[2]
        iters[s] = state[1]


# FAST ONLY (lane cluster-apple3), d <= MSB_MAX_D, OPT-IN
# `-D MOJOLEARN_MEANSHIFT_BLOCK=1`: one BLOCK per seed. `_meanshift_kernel`
# walks all n rows of every shift on one thread per seed, so a fit with a few
# hundred seeds keeps a few blocks busy. Here the block's threads each fold
# their stride of the rows and the partial sums fold in threadgroup memory.
# The addends, the quotient and the stop test are `bodies.meanshift_seed`'s;
# the order of the sum is not (bits move; the paired quality check).
comptime MEANSHIFT_BLOCK = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and is_defined["MOJOLEARN_MEANSHIFT_BLOCK"]()
comptime MSB_TPB = 256
comptime MSB_MAX_D = 16
comptime MSB_W = MSB_MAX_D + 1  # the feature sums, then the count


def _meanshift_block_kernel(
    x: FPtr, n: Int32, d: Int32, bw: Float32, stop: Float32, max_iter: Int32,
    centers: FPtr, intensity: IPtr, iters: IPtr,
):
    var s = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var N = Int(n)
    var D = Int(d)
    var red = stack_allocation[MSB_TPB * MSB_W, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    # the seed's center lives in threadgroup memory between the shifts (a
    # barrier orders threadgroup memory; on Apple it does not order the device's)
    var cen = stack_allocation[MSB_MAX_D, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var state = stack_allocation[3, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    if tid < D:
        cen[tid] = centers[s * D + tid]
    if tid == 0:
        state[0] = Int32(0)  # 1: the seed is done
        state[1] = Int32(0)  # completed shifts
        state[2] = Int32(0)  # rows within the bandwidth at the last shift
    barrier()
    var o = tid * MSB_W
    while True:
        for f in range(D + 1):
            red[o + f] = Float32(0)
        for p in range(tid, N, MSB_TPB):
            var acc = Float32(0)
            for f in range(D):
                var t = ftz(ftz(cen[f]) - ftz(x[p * D + f]))
                acc = ftz(acc + ftz(identical_mul(t, t)))
            if identical_sqrt(acc) <= bw:
                for f in range(D):
                    red[o + f] = ftz(red[o + f] + ftz(x[p * D + f]))
                red[o + D] = red[o + D] + Float32(1)
        barrier()
        var off = MSB_TPB // 2
        while off > 0:
            if tid < off:
                for f in range(D + 1):
                    red[o + f] = ftz(red[o + f] + red[(tid + off) * MSB_W + f])
            barrier()
            off //= 2
        if tid == 0:
            var cnt = red[D]
            state[2] = Int32(Int(cnt))
            if cnt == Float32(0):
                state[0] = Int32(1)
            else:
                var shift2 = Float32(0)
                for f in range(D):
                    var m = ftz(identical_div(red[f], cnt))
                    var t = ftz(m - cen[f])
                    shift2 = ftz(shift2 + ftz(identical_mul(t, t)))
                    cen[f] = m
                if identical_sqrt(shift2) <= stop or state[1] == max_iter:
                    state[0] = Int32(1)
                else:
                    state[1] = state[1] + Int32(1)
        barrier()
        if state[0] != Int32(0):
            break
    if tid < D:
        centers[s * D + tid] = cen[tid]
    if tid == 0:
        intensity[s] = state[2]
        iters[s] = state[1]


comptime AP_TPB = 256


@always_inline
def _ap_key(v: Float32, k: Int) -> UInt64:
    """An integer whose order is `ap_responsibility_row`'s pick: the float's
    order in the high word (-0.0 folded onto +0.0, which `>` treats as
    equal), the LOWER index winning in the low word. The row's max and its
    lowest index come from one integer max, never a float compare."""
    var b = bitcast[DType.uint32](v)
    if b == UInt32(0x80000000):
        b = UInt32(0)
    var ok = (b ^ UInt32(0x80000000)) if (b & UInt32(0x80000000)) == UInt32(0) else ~b
    return (UInt64(ok) << 32) | UInt64(UInt32(0xFFFFFFFF) - UInt32(k))


@always_inline
def _ap_key_index(key: UInt64) -> Int:
    return Int(UInt32(0xFFFFFFFF) - UInt32(key & UInt64(0xFFFFFFFF)))


def _ap_r_kernel(s: FPtr, a: FPtr, r: FPtr, n: Int32, damping: Float32):
    """`ap_responsibility_row` for row `block_idx.x` on one block: the max
    of `ftz(A + S)` with its lowest index, then the second max over the
    other columns with ITS lowest index (each an integer max of `_ap_key`,
    so exactly the row loop's picks, whatever the block's fold shape), then
    every cell's `ap_r_update`. Coalesced reads where the row-per-thread
    kernel strode by `n`."""
    var i = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var N = Int(n)
    var red = stack_allocation[AP_TPB, Scalar[DType.uint64], address_space = AddressSpace.SHARED]()
    var mine = UInt64(0)
    for k in range(tid, N, AP_TPB):
        mine = max(mine, _ap_key(ftz(a[i * N + k] + s[i * N + k]), k))
    red[tid] = mine
    barrier()
    var off = AP_TPB // 2
    while off > 0:
        if tid < off:
            red[tid] = max(red[tid], red[tid + off])
        barrier()
        off //= 2
    var arg = _ap_key_index(red[0])
    barrier()
    var mine2 = UInt64(0)
    for k in range(tid, N, AP_TPB):
        if k != arg:
            mine2 = max(mine2, _ap_key(ftz(a[i * N + k] + s[i * N + k]), k))
    red[tid] = mine2
    barrier()
    off = AP_TPB // 2
    while off > 0:
        if tid < off:
            red[tid] = max(red[tid], red[tid + off])
        barrier()
        off //= 2
    var top2 = red[0]
    var first = ftz(a[i * N + arg] + s[i * N + arg])
    var second = Float32(-3.4028234663852886e38)
    if N > 1:
        var a2 = _ap_key_index(top2)
        second = ftz(a[i * N + a2] + s[i * N + a2])
    var one_minus = ftz(Float32(1) - damping)
    for k in range(tid, N, AP_TPB):
        ap_r_update(s, r, N, damping, one_minus, i, k, first, second, arg)


# Lane cluster-apple3, FAST, OPT-IN `-D MOJOLEARN_AP_EXACT=1`: `_ap_r_kernel`
# with the row's max and second max taken in ONE walk of the row. The keys
# are distinct integers (the column is their low word), so the two largest
# of a row are the same two whatever the fold's shape: the same picks, the
# same R, one read of A + S less per iteration.
comptime AP_R_TOP2 = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and is_defined["MOJOLEARN_AP_EXACT"]()


def _ap_r_top2_kernel(s: FPtr, a: FPtr, r: FPtr, n: Int32, damping: Float32):
    var i = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var N = Int(n)
    var red1 = stack_allocation[AP_TPB, Scalar[DType.uint64], address_space = AddressSpace.SHARED]()
    var red2 = stack_allocation[AP_TPB, Scalar[DType.uint64], address_space = AddressSpace.SHARED]()
    var m1 = UInt64(0)
    var m2 = UInt64(0)
    for k in range(tid, N, AP_TPB):
        var key = _ap_key(ftz(a[i * N + k] + s[i * N + k]), k)
        if key > m1:
            m2 = m1
            m1 = key
        elif key > m2:
            m2 = key
    red1[tid] = m1
    red2[tid] = m2
    barrier()
    var off = AP_TPB // 2
    while off > 0:
        if tid < off:
            var a1 = red1[tid]
            var a2 = red2[tid]
            var b1 = red1[tid + off]
            var b2 = red2[tid + off]
            red1[tid] = max(a1, b1)
            red2[tid] = max(min(a1, b1), max(a2, b2))
        barrier()
        off //= 2
    var arg = _ap_key_index(red1[0])
    var top2 = red2[0]
    var first = ftz(a[i * N + arg] + s[i * N + arg])
    var second = Float32(-3.4028234663852886e38)
    if N > 1:
        var a2i = _ap_key_index(top2)
        second = ftz(a[i * N + a2i] + s[i * N + a2i])
    var one_minus = ftz(Float32(1) - damping)
    for k in range(tid, N, AP_TPB):
        ap_r_update(s, r, N, damping, one_minus, i, k, first, second, arg)


def _ap_a_kernel(r: FPtr, a: FPtr, n: Int32, damping: Float32):
    var t = _tid()
    if t < Int(n):
        ap_availability_col(r, a, Int(n), damping, t)


def _ap_noise_kernel(s: FPtr, m: Int64, seed: UInt64):
    var t = _tid()
    if t < Int(m):
        ap_noise_cell(s, seed, t)


def _ap_e_kernel(a: FPtr, r: FPtr, n: Int32, e: IPtr):
    var t = _tid()
    if t < Int(n):
        ap_exemplar_cell(a, r, Int(n), e, t)


def _gauss_q_kernel(x: FPtr, n: Int32, d: Int32, means: FPtr, pchol: FPtr, kc: Int32, dst: FPtr):
    var t = _tid()
    if t < Int(n) * Int(kc):
        gauss_q_cell(x, Int(d), means, pchol, Int(kc), dst, t)


def _resp_kernel(q: FPtr, c: FPtr, n: Int32, kc: Int32, lpn: FPtr):
    var t = _tid()
    if t < Int(n):
        resp_row(q, c, Int(kc), lpn, t)


def _exp_kernel(src: FPtr, dst: FPtr, n: Int32):
    var t = _tid()
    if t < Int(n):
        exp_cell(src, dst, t)


def _argmax_kernel(src: FPtr, n: Int32, kc: Int32, dst: IPtr):
    var t = _tid()
    if t < Int(n):
        argmax_row(src, Int(kc), dst, t)


def _nk_kernel(resp: FPtr, n: Int32, kc: Int32, dst: FPtr):
    var t = _tid()
    if t < Int(kc):
        nk_cell(resp, Int(n), Int(kc), dst, t)


def _xk_kernel(resp: FPtr, x: FPtr, n: Int32, d: Int32, kc: Int32, nk: FPtr, dst: FPtr):
    var t = _tid()
    if t < Int(kc) * Int(d):
        xk_cell(resp, x, Int(n), Int(d), Int(kc), nk, dst, t)


def _cov_kernel(resp: FPtr, x: FPtr, n: Int32, d: Int32, kc: Int32, means: FPtr, nk: FPtr, reg: Float32, dst: FPtr):
    var t = _tid()
    if t < Int(kc) * Int(d) * Int(d):
        cov_cell(resp, x, Int(n), Int(d), Int(kc), means, nk, reg, dst, t)


comptime MOM_TPB = 256
comptime MOM_SMEM = 4096  # floats: 16 KB, inside Apple's 32 KB threadgroup memory
comptime MOM_MAX_D = 64
comptime MOM_ROWS = 256  # rows per tile at most
comptime MOM_COV_CPB = 16  # covariance chains per block
comptime MOM_UNROLL = 8  # addends read ahead of the (still ascending) adds


# DEVIATION 5121 (the device M-step moments: the addends of a row tile formed
# in parallel into shared memory, then every fold one thread's register chain
# over them in ascending row order). Row 201; moments_check (5121 arm).
def _moments_pass_kernel(
    resp: FPtr, x: FPtr, n: Int32, d: Int32, kc: Int32, reg: Float32, nk: FPtr, means: FPtr, cov: FPtr,
    cov_pass: Int32,
):
    """The SAME folds as `bodies.nk_cell`, `xk_cell` and `cov_cell` (their
    `*_term`, `chain_add` and `*_final` functions, rows ascending), split
    between the threads that FORM the addends and the one thread per chain
    that ADDS them. The addend of a row does not depend on the chain, so
    forming a tile of them first, by every thread of the block, moves no
    bit; the adds stay one register chain per output in row order.

    Pass 1 (`cov_pass` 0; one block per component k): chains 0..d-1 are the
    mean sums of feature a, chain d the nk sum; the means divide by the
    FINAL nk, as `xk_cell` does. Pass 2 (`cov_pass` 1; blocks k * G + g):
    MOM_COV_CPB covariance chains per block against pass 1's means.

    The one-thread-per-cell kernels this replaces ran each chain straight
    from global memory, one dependent load and a dozen dependent ops per row
    on a handful of warps (31 ms of a 37 ms BayesianGaussianMixture
    iteration at 100,000 x 8, 8 components, H100)."""
    var tid = Int(thread_idx.x)
    var N = Int(n)
    var D = Int(d)
    var K = Int(kc)
    var nch: Int
    var cpb: Int
    var k: Int
    var c0: Int
    if cov_pass == Int32(0):
        k = Int(block_idx.x)
        nch = D + 1
        cpb = D + 1
        c0 = 0
    else:
        var g_per_k = (D * D + MOM_COV_CPB - 1) // MOM_COV_CPB
        k = Int(block_idx.x) // g_per_k
        nch = D * D
        cpb = MOM_COV_CPB
        c0 = (Int(block_idx.x) - k * g_per_k) * MOM_COV_CPB
    var T = MOM_SMEM // cpb
    if T > MOM_ROWS:
        T = MOM_ROWS
    var terms = stack_allocation[MOM_SMEM, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var fin = stack_allocation[1, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = Float32(0)
    var mine = tid < cpb and c0 + tid < nch
    var r0 = 0
    while r0 < N:
        var m = N - r0
        if m > T:
            m = T
        for e in range(tid, m * cpb, MOM_TPB):
            var j = e // cpb
            var q = e - j * cpb
            var c = c0 + q
            var i = r0 + j
            var r = resp[i * K + k]
            var t = Float32(0)
            if c < nch:
                if cov_pass == Int32(0):
                    t = r if c == D else xk_term(r, x[i * D + c])
                else:
                    var a = c // D
                    var b = c - a * D
                    t = cov_term(r, x[i * D + a], x[i * D + b], means[k * D + a], means[k * D + b])
            terms[e] = t
        barrier()
        if mine:
            var j0 = 0
            while j0 + MOM_UNROLL <= m:
                var v = SIMD[DType.float32, MOM_UNROLL]()
                comptime for u in range(MOM_UNROLL):
                    v[u] = terms[(j0 + u) * cpb + tid]
                comptime for u in range(MOM_UNROLL):
                    acc = chain_add(acc, v[u])
                j0 += MOM_UNROLL
            for j in range(j0, m):
                acc = chain_add(acc, terms[j * cpb + tid])
        barrier()
        r0 += m
    var c = c0 + tid
    if cov_pass == Int32(0):
        if mine and c == D:
            var v = nk_final(acc)
            fin[0] = v
            nk[k] = v
        barrier()
        if mine and c < D:
            means[k * D + c] = mean_final(acc, fin[0])
    elif mine:
        var a = c // D
        var b = c - a * D
        cov[k * nch + c] = cov_final(acc, nk[k], reg, a == b)


# FAST ONLY (lane/cluster-apple): the moments split over row slices. Each
# block (k, slice) sums its rows' addends per chain in a fixed thread layout
# and writes one partial per chain; a second kernel adds the partials over
# the slices. The same addends and finals as 5110/5121, another summation
# order: bits move, quality is the paired check's.
comptime MOMF_ROWS = 2048  # rows per slice


@always_inline
def _momf_term(resp: FPtr, x: FPtr, i: Int, D: Int, K: Int, k: Int, c: Int, cov_pass: Int32, means: FPtr) -> Float32:
    var r = resp[i * K + k]
    if cov_pass == Int32(0):
        return r if c == D else xk_term(r, x[i * D + c])
    var a = c // D
    var b = c - a * D
    return cov_term(r, x[i * D + a], x[i * D + b], means[k * D + a], means[k * D + b])


def _momf_partial_kernel(
    resp: FPtr, x: FPtr, n: Int32, d: Int32, kc: Int32, means: FPtr, part: FPtr, n_slices: Int32, cov_pass: Int32,
):
    var tid = Int(thread_idx.x)
    var N = Int(n)
    var D = Int(d)
    var K = Int(kc)
    var S = Int(n_slices)
    var k = Int(block_idx.x) // S
    var sl = Int(block_idx.x) - k * S
    var nch = D + 1 if cov_pass == Int32(0) else D * D
    var r0 = sl * MOMF_ROWS
    var r1 = r0 + MOMF_ROWS
    if r1 > N:
        r1 = N
    var red = stack_allocation[MOM_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var c0 = 0
    while c0 < nch:
        var cc = nch - c0
        if cc > MOM_TPB:
            cc = MOM_TPB
        var groups = MOM_TPB // cc
        var mine = tid < groups * cc
        var g = tid // cc
        var c = c0 + (tid - g * cc)
        var acc = Float32(0)
        if mine:
            var i = r0 + g
            while i < r1:
                acc = acc + _momf_term(resp, x, i, D, K, k, c, cov_pass, means)
                i += groups
        red[tid] = acc
        barrier()
        if mine and g == 0:
            var tot = Float32(0)
            for gg in range(groups):
                tot = tot + red[gg * cc + tid]
            part[(k * S + sl) * nch + c] = tot
        barrier()
        c0 += cc


def _momf_final_kernel(
    part: FPtr, n_slices: Int32, d: Int32, kc: Int32, reg: Float32, nk: FPtr, means: FPtr, cov: FPtr, cov_pass: Int32,
):
    var t = _tid()
    var D = Int(d)
    var K = Int(kc)
    var S = Int(n_slices)
    var nch = D + 1 if cov_pass == Int32(0) else D * D
    if t >= K * nch:
        return
    var k = t // nch
    var c = t - k * nch
    var acc = Float32(0)
    for sl in range(S):
        acc = acc + part[(k * S + sl) * nch + c]
    if cov_pass == Int32(0):
        var nkacc = Float32(0)
        for sl in range(S):
            nkacc = nkacc + part[(k * S + sl) * nch + D]
        var nkv = nk_final(nkacc)
        if c == D:
            nk[k] = nkv
        else:
            means[k * D + c] = mean_final(acc, nkv)
    else:
        var a = c // D
        var b = c - a * D
        cov[k * nch + c] = cov_final(acc, nk[k], reg, a == b)


def _estep_row_kernel(
    x: FPtr, n: Int32, d: Int32, means: FPtr, pchol: FPtr, c: FPtr, kc: Int32, q: FPtr, r: FPtr, lpn: FPtr,
):
    """Lane cluster-apple3: row i's whole E-step on its one thread, the three
    bodies in the order the three kernels ran them."""
    var i = _tid()
    if i >= Int(n):
        return
    var K = Int(kc)
    for k in range(K):
        gauss_q_cell(x, Int(d), means, pchol, K, q, i * K + k)
    resp_row(q, c, K, lpn, i)
    for k in range(K):
        exp_cell(q, r, i * K + k)


def _dot_groups_kernel(a: FPtr, b: FPtr, n: Int32, g: Int32, parts: FPtr):
    """FAST (lane cluster-apple3): thread q sums a * b over its run of `g`
    consecutive cells."""
    var q = _tid()
    var N = Int(n)
    var G = Int(g)
    var t0 = q * G
    if t0 >= N:
        return
    var t1 = t0 + G
    if t1 > N:
        t1 = N
    var acc = Float32(0)
    for t in range(t0, t1):
        acc = acc + a[t] * b[t]
    parts[q] = acc


# FAST ONLY (lane cluster-apple3), d <= MOMS_MAX_D, OPT-IN
# `-D MOJOLEARN_MOMENTS_ROWS=1`: the moments with every ROW read once per
# component. Thread (k, g) owns MOMS_ROWS consecutive rows and folds all of
# its chains over them (the means' d + 1, then the covariance's upper
# triangle) in its own slice of threadgroup memory; a second kernel adds the
# threads' partials. `_momf_partial_kernel` read every row once per CHAIN.
# The same addends and finals as 5110/5121, another order: bits move.
comptime MOMS = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and is_defined["MOJOLEARN_MOMENTS_ROWS"]()
comptime MOMS_MAX_D = 8
comptime MOMS_ROWS = 64
comptime MOMS_TPB = 128
comptime MOMS_W = MOMS_MAX_D * (MOMS_MAX_D + 1) // 2  # chains a thread holds at most


def _moms_part_kernel(
    resp: FPtr, x: FPtr, n: Int32, d: Int32, kc: Int32, means: FPtr, part: FPtr, n_groups: Int32, cov_pass: Int32,
):
    var tid = Int(thread_idx.x)
    var t = Int(block_idx.x) * MOMS_TPB + tid
    var N = Int(n)
    var D = Int(d)
    var K = Int(kc)
    var G = Int(n_groups)
    var acc = stack_allocation[MOMS_TPB * MOMS_W, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    if t >= K * G:
        return
    var k = t // G
    var g = t - k * G
    var r0 = g * MOMS_ROWS
    var r1 = r0 + MOMS_ROWS
    if r1 > N:
        r1 = N
    var o = tid * MOMS_W
    var nch = D + 1 if cov_pass == Int32(0) else (D * (D + 1)) // 2
    for c in range(nch):
        acc[o + c] = Float32(0)
    if cov_pass == Int32(0):
        for i in range(r0, r1):
            var r = resp[i * K + k]
            for a in range(D):
                acc[o + a] = acc[o + a] + xk_term(r, x[i * D + a])
            acc[o + D] = acc[o + D] + r
    else:
        for i in range(r0, r1):
            var r = resp[i * K + k]
            var c = 0
            for a in range(D):
                var xa = x[i * D + a]
                var ma = means[k * D + a]
                for b in range(a, D):
                    acc[o + c] = acc[o + c] + cov_term(r, xa, x[i * D + b], ma, means[k * D + b])
                    c += 1
    for c in range(nch):
        part[(k * G + g) * MOMS_W + c] = acc[o + c]


def _moms_final_kernel(
    part: FPtr, n_groups: Int32, d: Int32, kc: Int32, reg: Float32, nk: FPtr, means: FPtr, cov: FPtr, cov_pass: Int32,
):
    var t = _tid()
    var D = Int(d)
    var K = Int(kc)
    var G = Int(n_groups)
    var nout = D + 1 if cov_pass == Int32(0) else D * D
    if t >= K * nout:
        return
    var k = t // nout
    var c = t - k * nout
    if cov_pass == Int32(0):
        var acc = Float32(0)
        var nkacc = Float32(0)
        for g in range(G):
            acc = acc + part[(k * G + g) * MOMS_W + c]
            nkacc = nkacc + part[(k * G + g) * MOMS_W + D]
        var nkv = nk_final(nkacc)
        if c == D:
            nk[k] = nkv
        else:
            means[k * D + c] = mean_final(acc, nkv)
    else:
        var a = c // D
        var b = c - a * D
        var lo = a if a < b else b
        var hi = b if a < b else a
        # the upper triangle's chain of (lo, hi), rows of the triangle ascending
        var q = lo * D - (lo * (lo - 1)) // 2 + (hi - lo)
        var acc = Float32(0)
        for g in range(G):
            acc = acc + part[(k * G + g) * MOMS_W + q]
        cov[k * D * D + c] = cov_final(acc, nk[k], reg, a == b)


def _pdist_kernel(a: FPtr, na: Int32, b: FPtr, nb: Int32, d: Int32, metric: Int32, p: Float32, dst: FPtr):
    var t = _tid()
    if t < Int(na) * Int(nb):
        pdist_cell(a, Int(na), b, Int(nb), Int(d), Int(metric), p, dst, t)


def _descend_kernel(x: FPtr, n: Int32, d: Int32, centers: FPtr, nodes: IPtr, labels: IPtr):
    var t = _tid()
    if t < Int(n):
        tree_descend(x, Int(d), centers, nodes, labels, t)


comptime KF_TPB = 256  # threads of a block, and the bins of a byte
comptime KF_VPT = 64  # values per thread


# Lane cluster-apple3: the radix select of `_kth_kernel` for ONE long row,
# over every block of the grid. The same integer counts, so the same value.
def _kthf_hist_kernel(m: FPtr, n: Int32, prefix: Int32, shift: Int32, first: Int32, part: IPtr):
    """Block `b` counts its KF_TPB * KF_VPT values whose bits above `shift +
    8` equal `prefix` by their byte at `shift`, in threadgroup memory
    (integer atomics: every interleaving gives the same counts), then writes
    its 256 counts to `part[b]`."""
    var blk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var N = Int(n)
    var hist = stack_allocation[256, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    hist[tid] = Int32(0)
    barrier()
    var pre = UInt32(prefix)
    var sh = UInt32(shift)
    var base = blk * (KF_TPB * KF_VPT)
    for j in range(KF_VPT):
        var t = base + j * KF_TPB + tid
        if t < N:
            var bits = bitcast[DType.uint32](m[t]) & UInt32(0x7FFFFFFF)
            var above = UInt32(0)
            if first == Int32(0):
                above = (bits >> (sh + UInt32(8))) << (sh + UInt32(8))
            if above == pre:
                _ = Atomic.fetch_add(hist.unsafe_offset(Int((bits >> sh) & UInt32(0xFF))), Int32(1))
    barrier()
    part[blk * 256 + tid] = hist[tid]


def _kthf_sum_kernel(part: IPtr, n_blocks: Int32, hist: IPtr):
    """Thread `b` of the one block: bin b summed over the blocks."""
    var b = Int(thread_idx.x)
    var acc = Int32(0)
    for q in range(Int(n_blocks)):
        acc += part[q * 256 + b]
    hist[b] = acc


def _diag_kernel(src: FPtr, n: Int32, dst: FPtr):
    var t = _tid()
    if t < Int(n):
        dst[t] = src[t * Int(n) + t]


comptime APF_TPB = 256
comptime APF_ROWS = 64  # rows per slice


# FAST ONLY (lane cluster-apple3): the availability update with the column
# sums folded over row slices on every block of the grid, then every cell
# its own thread. The addends and the cell update are `ap_availability_col`'s;
# the column sum's order is not (bits move; the paired quality check).
def _apf_part_kernel(r: FPtr, n: Int32, n_tiles: Int32, part: FPtr):
    """Block (slice s, tile c), thread t: column c * APF_TPB + t summed over
    the slice's rows (`Rp`: the positive part off the diagonal, the value
    on it). Adjacent threads read adjacent cells of each row."""
    var N = Int(n)
    var T = Int(n_tiles)
    var s = Int(block_idx.x) // T
    var c = Int(block_idx.x) - s * T
    var k = c * APF_TPB + Int(thread_idx.x)
    if k >= N:
        return
    var i0 = s * APF_ROWS
    var i1 = i0 + APF_ROWS
    if i1 > N:
        i1 = N
    var acc = Float32(0)
    for i in range(i0, i1):
        var v = r[i * N + k]
        if i == k or v > Float32(0):
            acc = acc + v
    part[s * N + k] = acc


def _apf_sum_kernel(part: FPtr, n: Int32, n_slices: Int32, colsum: FPtr):
    var k = _tid()
    var N = Int(n)
    if k >= N:
        return
    var acc = Float32(0)
    for s in range(Int(n_slices)):
        acc = acc + part[s * N + k]
    colsum[k] = acc


def _apf_update_kernel(r: FPtr, a: FPtr, colsum: FPtr, n: Int32, n_tiles: Int32, damping: Float32):
    """Block (row i, tile c), thread t: cell (i, c * APF_TPB + t)."""
    var N = Int(n)
    var T = Int(n_tiles)
    var i = Int(block_idx.x) // T
    var c = Int(block_idx.x) - i * T
    var k = c * APF_TPB + Int(thread_idx.x)
    if k >= N:
        return
    var one_minus = ftz(Float32(1) - damping)
    var v = r[i * N + k]
    var rp = v if (i == k or v > Float32(0)) else Float32(0)
    var nw = ftz(colsum[k] - rp)
    if i != k and nw > Float32(0):
        nw = Float32(0)
    var old = a[i * N + k]
    a[i * N + k] = ftz(ftz(identical_mul(old, damping)) + ftz(identical_mul(nw, one_minus)))


comptime WNN_TPB = 256


# FAST ONLY (lane cluster-apple3): one round of the ward tree's reciprocal
# nearest neighbours. Not an IDENTICAL path.
def _ward_nn_kernel(c: FPtr, sz: FPtr, l: Int32, d: Int32, nn: IPtr, md: FPtr):
    """Block `p`: the cluster q != p at the lowest `bodies.ward_cell`, the
    lowest q on a tie, by an integer min of `(float bits, q)` keys (the
    values are >= +0, so their bits order as they do). Every thread reads
    its own stride of the centroids; the keys fold in threadgroup memory."""
    var p = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var L = Int(l)
    var D = Int(d)
    var red = stack_allocation[WNN_TPB, Scalar[DType.uint64], address_space = AddressSpace.SHARED]()
    var mine = UInt64(0xFFFFFFFFFFFFFFFF)
    for q in range(tid, L, WNN_TPB):
        if q != p:
            var v = ward_cell(c, sz, D, p, q)
            mine = min(mine, (UInt64(bitcast[DType.uint32](v)) << 32) | UInt64(UInt32(q)))
    red[tid] = mine
    barrier()
    var off = WNN_TPB // 2
    while off > 0:
        if tid < off:
            red[tid] = min(red[tid], red[tid + off])
        barrier()
        off //= 2
    if tid == 0:
        var key = red[0]
        nn[p] = Int32(Int(UInt32(key & UInt64(0xFFFFFFFF))))
        md[p] = bitcast[DType.float32](UInt32(key >> 32))


@always_inline
def _grid(n: Int) -> Int:
    return (n + TPB - 1) // TPB if n > 0 else 1


# ---------------------------------------------------------------------------
# THE AGGLOMERATIVE MERGE LOOP ON THE DEVICE (lane hr2-mds-agglo, 2026-10-02).
# `agglo.agglo_tree`'s unconstrained loop, every step as parallel kernels on
# the resident n x n matrix: no host step between two merges. Every pick is a
# min of `_agg_key` (the value's order in the high word, the index in the low
# word), so the reduction's shape cannot move it: the lowest value, the
# lowest index on a tie, exactly the host loop's strict `<` scans.
#   state ints:   [0] a  [1] b  [2] rescan list length  [3] error
#   state floats: [0] dab  [1] size a  [2] size b
comptime AGG_TPB = 256
comptime AGG_PER = AGG_TPB * 4
comptime AGG_RESCAN_BLOCKS = 128
comptime AGG_NONE = UInt64(0xFFFFFFFFFFFFFFFF)


@always_inline
def _agg_key(v: Float32, k: Int) -> UInt64:
    """Ascending in v (-0.0 folded onto +0.0, which `<` treats as equal),
    then ascending in k."""
    var b = bitcast[DType.uint32](v)
    if b == UInt32(0x80000000):
        b = UInt32(0)
    var o = (b ^ UInt32(0x80000000)) if (b & UInt32(0x80000000)) == UInt32(0) else ~b
    return (UInt64(o) << 32) | UInt64(UInt32(k))


@always_inline
def _agg_block_min(red: UnsafePointer[UInt64, MutUntrackedOrigin, address_space=AddressSpace.SHARED], mine: UInt64) -> UInt64:
    var tid = Int(thread_idx.x)
    red[tid] = mine
    barrier()
    var off = AGG_TPB // 2
    while off > 0:
        if tid < off:
            red[tid] = min(red[tid], red[tid + off])
        barrier()
        off //= 2
    var r = red[0]
    barrier()
    return r


def _agg_init_kernel(live: IPtr, nn: IPtr, md: FPtr, sz: FPtr, node: IPtr, lst: IPtr, st: IPtr, n: Int32):
    var i = _tid()
    if i < Int(n):
        live[i] = 1
        nn[i] = -1
        md[i] = Float32.MAX * Float32(2)
        sz[i] = Float32(1)
        node[i] = Int32(i)
        lst[i] = Int32(i)
    if i == 0:
        st[0] = 0
        st[1] = 0
        st[2] = n


def _agg_mirror_kernel(x: FPtr, dm: FPtr, n: Int32, bad: IPtr):
    var t = _tid()
    var N = Int(n)
    if t < N * N:
        var i = t // N
        var j = t % N
        if i == j:
            dm[t] = Float32(0)
        else:
            var lo = i if i < j else j
            var hi = j if i < j else i
            var v = x[lo * N + hi]
            if i < j and (not (v >= Float32(0)) or v == Float32.MAX * Float32(2)):
                bad[0] = 1
            dm[t] = v


def _agg_argmin_part_kernel(md: FPtr, nn: IPtr, live: IPtr, n: Int32, part: MutPointer[UInt64, MutAnyOrigin], st: IPtr):
    """part[block] = the min key of (md[i], i) over the block's live rows
    with a partner."""
    var red = stack_allocation[AGG_TPB, UInt64, address_space = AddressSpace.SHARED]()
    var mine = AGG_NONE
    var base = Int(block_idx.x) * AGG_PER
    var end = min(base + AGG_PER, Int(n))
    for i in range(base + Int(thread_idx.x), end, AGG_TPB):
        if live[i] != 0 and nn[i] >= 0:
            mine = min(mine, _agg_key(md[i], i))
    var r = _agg_block_min(red, mine)
    if thread_idx.x == 0:
        part[Int(block_idx.x)] = r


@always_inline
def _agg_pick(part: MutPointer[UInt64, MutAnyOrigin], nb: Int) -> UInt64:
    """The step's pick from the argmin partials (a few per thousand rows),
    read by every thread that needs it."""
    var r = AGG_NONE
    for q in range(nb):
        r = min(r, part[q])
    return r


def _agg_lw_kernel(
    part: MutPointer[UInt64, MutAnyOrigin], nb: Int32, dm: FPtr, nn: IPtr, live: IPtr, sz: FPtr, n: Int32,
    linkage: Int32, st: IPtr, stf: FPtr, adj: IPtr, con: Int32,
):
    """The merge (a, b = nn[a]) from the partials, then row and column a of
    the matrix: the Lance-Williams value of (a u b) to every live k other
    than a and b. Thread 0 records a, b, dab and the two sizes for the
    kernels after it."""
    var k = _tid()
    var N = Int(n)
    if st[3] != 0:
        return
    var r = _agg_pick(part, Int(nb))
    if r == AGG_NONE:
        if k == 0:
            st[3] = 1
        return
    var a = Int(UInt32(r & UInt64(0xFFFFFFFF)))
    var b = Int(nn[a])
    var dab = dm[a * N + b]
    var na = sz[a]
    var nbs = sz[b]
    if k == 0:
        st[0] = Int32(a)
        st[1] = Int32(b)
        st[2] = 0
        stf[0] = dab
        stf[1] = na
        stf[2] = nbs
    if k >= N or k == a or k == b or live[k] == 0:
        return
    var ha = True
    var hb = True
    if con != 0:
        ha = adj[a * N + k] != 0
        hb = adj[b * N + k] != 0
    if Int(linkage) == LINK_WARD or ha or hb:
        var v = lance_williams(Int(linkage), dm[a * N + k], dm[b * N + k], dab, na, nbs, sz[k], ha, hb)
        dm[a * N + k] = v
        dm[k * N + a] = v
    if con != 0 and (ha or hb):
        adj[a * N + k] = 1
        adj[k * N + a] = 1


def _agg_flag_kernel(
    dm: FPtr, live: IPtr, nn: IPtr, md: FPtr, sz: FPtr, node: IPtr, n: Int32, step: Int32, linkage: Int32,
    ch: IPtr, dist: FPtr, st: IPtr, stf: FPtr, lst: IPtr, adj: IPtr, con: Int32,
):
    """Thread a books the merge (the children pair and value; b dies; a
    takes the merged size and node id n + step) and joins the rescan list
    with every live row below b whose partner was a or b; a live row below
    a otherwise takes a when it is nearer (or as near with a lower index)."""
    var i = _tid()
    var N = Int(n)
    if i >= N or st[3] != 0:
        return
    var a = Int(st[0])
    var b = Int(st[1])
    var go = False
    if i == a:
        go = True
        var dab = stf[0]
        var x = node[a]
        var y = node[b]
        var s = Int(step)
        ch[2 * s] = x if x < y else y
        ch[2 * s + 1] = y if x < y else x
        dist[s] = identical_sqrt(dab) if Int(linkage) == LINK_WARD else dab
        live[b] = 0
        nn[b] = -1
        sz[a] = ftz(stf[1] + stf[2])
        node[a] = Int32(N + s)
    elif i < b and live[i] != 0:
        var q = Int(nn[i])
        if q == a or q == b:
            go = True
        elif i < a and (con == 0 or adj[i * N + a] != 0):
            var v = dm[i * N + a]
            if q < 0 or v < md[i] or (v == md[i] and a < q):
                nn[i] = Int32(a)
                md[i] = v
    if go:
        var at = Atomic.fetch_add(st.unsafe_offset(2), Int32(1))
        lst[Int(at)] = Int32(i)


def _agg_rescan_kernel(dm: FPtr, live: IPtr, nn: IPtr, md: FPtr, n: Int32, st: IPtr, lst: IPtr, adj: IPtr, con: Int32):
    """nn[i], md[i] for each listed row: the live column j > i at the lowest
    value, the lowest j on a tie (-1, +inf when none). One block a row."""
    var red = stack_allocation[AGG_TPB, UInt64, address_space = AddressSpace.SHARED]()
    var N = Int(n)
    var cnt = Int(st[2])
    if st[3] != 0:
        cnt = 0
    var e = Int(block_idx.x)
    while e < cnt:
        var i = Int(lst[e])
        var mine = AGG_NONE
        for j in range(i + 1 + Int(thread_idx.x), N, AGG_TPB):
            if live[j] != 0 and (con == 0 or adj[i * N + j] != 0):
                mine = min(mine, _agg_key(dm[i * N + j], j))
        var r = _agg_block_min(red, mine)
        if thread_idx.x == 0:
            if r == AGG_NONE:
                nn[i] = -1
                md[i] = Float32.MAX * Float32(2)
            else:
                var j = Int(UInt32(r & UInt64(0xFFFFFFFF)))
                nn[i] = Int32(j)
                md[i] = dm[i * N + j]
        e += Int(grid_dim.x)



struct _ClusterContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext for every x_cluster entry (the
    x_cnn `_Global` pattern: one DeviceContext per process). A context per call hung the SECOND
    `x_cluster_call` in a process on an RTX 4090 (futex wait): the context
    was a field declared BEFORE the call's buffers, so it was torn down
    while they still held its allocations; on Metal a context per call also
    exhausts the per-process command queues. The slot keeps a reference for
    the life of the process, so every call's buffers die inside it. One slot
    per numeric tier, so a FAST and an IDENTICAL .so never share it."""
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojoXClusterContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXClusterContextFast"
comptime X_CLUSTER_CONTEXT = _Global[StorageType=_ClusterContext, name=_CTX_NAME, init_fn=_ClusterContext.__init__]


def x_cluster_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_CLUSTER_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


# An upload of at least this many values synchronizes instead of keeping a
# host copy alive (the one-off upload of X at a fit's start); smaller uploads
# and every `zeros` (a device memset) enqueue without a synchronize.
comptime _PUT_SYNC_MIN = 1 << 20


struct DeviceOps(ClusterOps):
    """Kernels and uploads are only ENQUEUED: the stream runs them in order,
    and the one synchronize is where the host reads (`get`, `get_i`, `gets`,
    `get_if`). An upload's source is a COPY held in `pend_f` / `pend_i` until
    that synchronize, so the caller's List may change or die at once; `zeros`
    is a device memset. A sync per call cost a host round trip each (on Metal
    the dominant cost: about 4 ms per sync with pending work on the M4).
    Scheduling only: every kernel sees the same bytes in the same order."""
    var ctx: DeviceContext
    var f: List[DeviceBuffer[DType.float32]]
    var i: List[DeviceBuffer[DType.int32]]
    var pend_f: List[List[Float32]]
    var pend_i: List[List[Int32]]
    var mpart: DeviceBuffer[DType.float32]
    """FAST moments' per-slice partials, grown once per fit."""
    var mpart_n: Int
    var ph_on: Bool
    """MOJOLEARN_XC_PHASES=1 (a diagnostic, lane cluster-apple3): every
    primitive drains the stream when it returns and its wall time is added to
    its name; the time between two primitives is the driver's (`host`). The
    table prints when the fit's ops die. Off, nothing changes."""
    var ph_t: Int
    var ph_names: List[String]
    var ph_ns: List[Int]
    var ph_calls: List[Int]
    var u: List[DeviceBuffer[DType.uint64]]
    """The post-processing primitives' 64-bit key buffers (lane cgr2-cluster)."""

    def __init__(out self) raises:
        self.ctx = x_cluster_ctx()
        self.f = List[DeviceBuffer[DType.float32]]()
        self.i = List[DeviceBuffer[DType.int32]]()
        self.pend_f = List[List[Float32]]()
        self.pend_i = List[List[Int32]]()
        self.mpart = self.ctx.enqueue_create_buffer[DType.float32](1)
        self.mpart_n = 1
        self.ph_on = getenv("MOJOLEARN_XC_PHASES") == "1"
        self.ph_t = Int(perf_counter_ns())
        self.ph_names = List[String]()
        self.ph_ns = List[Int]()
        self.ph_calls = List[Int]()
        self.u = List[DeviceBuffer[DType.uint64]]()

    def __del__(deinit self):
        # the buffers and the pending sources die with this value: drain first
        try:
            self.ctx.synchronize()
        except:
            pass
        if self.ph_on:
            var now = Int(perf_counter_ns())
            print("XCPHASE host_tail " + String(Float64(now - self.ph_t) / 1.0e6) + " ms")
            for q in range(len(self.ph_names)):
                print(
                    "XCPHASE " + self.ph_names[q] + " " + String(Float64(self.ph_ns[q]) / 1.0e6) + " ms calls="
                    + String(self.ph_calls[q])
                )

    def _ph_add(mut self, name: String, ns: Int):
        for q in range(len(self.ph_names)):
            if self.ph_names[q] == name:
                self.ph_ns[q] += ns
                self.ph_calls[q] += 1
                return
        self.ph_names.append(name)
        self.ph_ns.append(ns)
        self.ph_calls.append(1)

    def _ph0(mut self):
        """A primitive starts: the time since the last mark was the driver's."""
        if self.ph_on:
            var now = Int(perf_counter_ns())
            self._ph_add("host", now - self.ph_t)
            self.ph_t = now

    def _ph1(mut self, name: String) raises:
        """A primitive returns: drain, and the time since `_ph0` is its own."""
        if self.ph_on:
            self.ctx.synchronize()
            var now = Int(perf_counter_ns())
            self._ph_add(name, now - self.ph_t)
            self.ph_t = now

    def _sync(mut self) raises:
        self.ctx.synchronize()
        self.pend_f = List[List[Float32]]()
        self.pend_i = List[List[Int32]]()

    def _fp(mut self, slot: Int) -> FPtr:
        return self.f[slot].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    def _ip(mut self, slot: Int) -> IPtr:
        return self.i[slot].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    def _upload(mut self, buf: DeviceBuffer[DType.float32], v: List[Float32]) raises:
        var n = len(v)
        if n == 0:
            return
        if n >= _PUT_SYNC_MIN:
            self.ctx.enqueue_copy(dst_buf=buf, src_ptr=v.unsafe_ptr())
            self._sync()
            return
        self.pend_f.append(v.copy())
        self.ctx.enqueue_copy(dst_buf=buf, src_ptr=self.pend_f[len(self.pend_f) - 1].unsafe_ptr())

    def _upload_i(mut self, buf: DeviceBuffer[DType.int32], v: List[Int32]) raises:
        var n = len(v)
        if n == 0:
            return
        if n >= _PUT_SYNC_MIN:
            self.ctx.enqueue_copy(dst_buf=buf, src_ptr=v.unsafe_ptr())
            self._sync()
            return
        self.pend_i.append(v.copy())
        self.ctx.enqueue_copy(dst_buf=buf, src_ptr=self.pend_i[len(self.pend_i) - 1].unsafe_ptr())

    def put(mut self, v: List[Float32]) raises -> Int:
        self._ph0()
        var n = len(v)
        var buf = self.ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        self._upload(buf, v)
        self.f.append(buf^)
        self._ph1("put")
        return len(self.f) - 1

    def put_i(mut self, v: List[Int32]) raises -> Int:
        self._ph0()
        var n = len(v)
        var buf = self.ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
        self._upload_i(buf, v)
        self.i.append(buf^)
        self._ph1("put_i")
        return len(self.i) - 1

    def zeros(mut self, n: Int) raises -> Int:
        self._ph0()
        var buf = self.ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        self.ctx.enqueue_memset(buf, Float32(0))
        self.f.append(buf^)
        self._ph1("zeros")
        return len(self.f) - 1

    def zeros_i(mut self, n: Int) raises -> Int:
        self._ph0()
        var buf = self.ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
        self.ctx.enqueue_memset(buf, Int32(0))
        self.i.append(buf^)
        self._ph1("zeros_i")
        return len(self.i) - 1

    def _enq_get(mut self, slot: Int, n: Int, mut out: List[Float32]) raises:
        out = List[Float32](length=n, fill=Float32(0))
        if n > 0:
            var view = self.f[slot].create_sub_buffer[DType.float32](0, n)
            self.ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=view)

    def _enq_get_i(mut self, slot: Int, n: Int, mut out: List[Int32]) raises:
        out = List[Int32](length=n, fill=Int32(0))
        if n > 0:
            var view = self.i[slot].create_sub_buffer[DType.int32](0, n)
            self.ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=view)

    def get(mut self, slot: Int, n: Int) raises -> List[Float32]:
        self._ph0()
        var out = List[Float32]()
        self._enq_get(slot, n, out)
        self._sync()
        self._ph1("get")
        return out^

    def get_i(mut self, slot: Int, n: Int) raises -> List[Int32]:
        self._ph0()
        var out = List[Int32]()
        self._enq_get_i(slot, n, out)
        self._sync()
        self._ph1("get_i")
        return out^

    def gets(mut self, slots: List[Int], ns: List[Int]) raises -> List[List[Float32]]:
        self._ph0()
        var outs = List[List[Float32]](capacity=len(slots))
        for q in range(len(slots)):
            outs.append(List[Float32](length=ns[q], fill=Float32(0)))
        for q in range(len(slots)):
            if ns[q] > 0:
                var view = self.f[slots[q]].create_sub_buffer[DType.float32](0, ns[q])
                self.ctx.enqueue_copy(dst_ptr=outs[q].unsafe_ptr(), src_buf=view)
        self._sync()
        self._ph1("gets")
        return outs^

    def get_if(
        mut self, islot: Int, ni: Int, fslot: Int, nf: Int, mut oi: List[Int32], mut of: List[Float32]
    ) raises:
        self._ph0()
        self._enq_get_i(islot, ni, oi)
        self._enq_get(fslot, nf, of)
        self._sync()
        self._ph1("get_if")

    def set(mut self, slot: Int, v: List[Float32]) raises:
        self._ph0()
        var n = len(v)
        if n == 0:
            self._ph1("set")
            return
        var view = self.f[slot].create_sub_buffer[DType.float32](0, n)
        if n >= _PUT_SYNC_MIN:
            self.ctx.enqueue_copy(dst_buf=view, src_ptr=v.unsafe_ptr())
            self._sync()
            self._ph1("set")
            return
        self.pend_f.append(v.copy())
        self.ctx.enqueue_copy(dst_buf=view, src_ptr=self.pend_f[len(self.pend_f) - 1].unsafe_ptr())
        self._ph1("set")

    def sqdist(mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, dst: Int) raises:
        self._ph0()
        comptime if SQDIST_TILED:
            if na > 0 and nb > 0 and d > 0 and getenv("MOJOLEARN_XC_SQDIST_TILED") != "0":
                self.ctx.enqueue_function[_sqdist_tiled_kernel](
                    self._fp(a), Int32(na), self._fp(b), Int32(nb), Int32(d), self._fp(dst),
                    grid_dim=((nb + SQT_B - 1) // SQT_B, (na + SQT_B - 1) // SQT_B), block_dim=SQT_TD * SQT_TD,
                )
                self._ph1("sqdist")
                return
        self.ctx.enqueue_function[_sqdist_kernel](
            self._fp(a), Int32(na), self._fp(b), Int32(nb), Int32(d), self._fp(dst),
            grid_dim=_grid(na * nb), block_dim=TPB,
        )
        self._ph1("sqdist")

    def nearest(mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, labels: Int, dist: Int) raises:
        self._ph0()
        self.ctx.enqueue_function[_nearest_kernel](
            self._fp(a), Int32(na), self._fp(b), Int32(nb), Int32(d), self._ip(labels), self._fp(dist),
            grid_dim=_grid(na), block_dim=TPB,
        )
        self._ph1("nearest")

    def sqrt(mut self, x: Int, n: Int) raises:
        self._ph0()
        self.ctx.enqueue_function[_sqrt_kernel](
            self._fp(x), Int32(n), grid_dim=_grid(n), block_dim=TPB,
        )
        self._ph1("sqrt")

    def kth(mut self, m: Int, n_rows: Int, n_cols: Int, k: Int, dst: Int) raises:
        self._ph0()
        if n_rows <= 0:
            self._ph1("kth")
            return
        self.ctx.enqueue_function[_kth_kernel](
            self._fp(m), Int32(n_rows), Int32(n_cols), Int32(k), self._fp(dst),
            grid_dim=n_rows if n_rows > 0 else 1, block_dim=KTH_TPB,
        )
        self._ph1("kth")

    def meanshift(
        mut self, x: Int, n: Int, d: Int, bw: Float32, stop: Float32, max_iter: Int,
        centers: Int, ns: Int, scratch: Int, intensity: Int, iters: Int,
    ) raises:
        self._ph0()
        comptime if MEANSHIFT_BLOCK:
            if d <= MSB_MAX_D and ns > 0:
                self.ctx.enqueue_function[_meanshift_block_kernel](
                    self._fp(x), Int32(n), Int32(d), bw, stop, Int32(max_iter),
                    self._fp(centers), self._ip(intensity), self._ip(iters),
                    grid_dim=ns, block_dim=MSB_TPB,
                )
                self._ph1("meanshift")
                return
        comptime if MEANSHIFT_TEAM:
            if d <= MST_MAX_D and ns > 0 and getenv("MOJOLEARN_MEANSHIFT_TEAM") != "0":
                self.ctx.enqueue_function[_meanshift_team_kernel](
                    self._fp(x), Int32(n), Int32(d), bw, stop, Int32(max_iter),
                    self._fp(centers), self._ip(intensity), self._ip(iters),
                    grid_dim=ns, block_dim=MST_TPB,
                )
                self._ph1("meanshift")
                return
        self.ctx.enqueue_function[_meanshift_kernel](
            self._fp(x), Int32(n), Int32(d), bw, stop, Int32(max_iter),
            self._fp(centers), Int32(ns), self._fp(scratch), self._ip(intensity), self._ip(iters),
            grid_dim=_grid(ns), block_dim=TPB,
        )
        self._ph1("meanshift")

    def ap_r(mut self, s: Int, a: Int, r: Int, n: Int, damping: Float32) raises:
        self._ph0()
        comptime if AP_R_TOP2:
            self.ctx.enqueue_function[_ap_r_top2_kernel](
                self._fp(s), self._fp(a), self._fp(r), Int32(n), damping, grid_dim=n if n > 0 else 1,
                block_dim=AP_TPB,
            )
        else:
            self.ctx.enqueue_function[_ap_r_kernel](
                self._fp(s), self._fp(a), self._fp(r), Int32(n), damping, grid_dim=n if n > 0 else 1,
                block_dim=AP_TPB,
            )
        self._ph1("ap_r")

    def ap_a(mut self, r: Int, a: Int, n: Int, damping: Float32) raises:
        self._ph0()
        self.ctx.enqueue_function[_ap_a_kernel](
            self._fp(r), self._fp(a), Int32(n), damping, grid_dim=_grid(n), block_dim=TPB,
        )
        self._ph1("ap_a")

    def ap_noise(mut self, s: Int, m: Int, seed: UInt64) raises:
        self._ph0()
        if m <= 0:
            self._ph1("ap_noise")
            return
        self.ctx.enqueue_function[_ap_noise_kernel](self._fp(s), Int64(m), seed, grid_dim=_grid(m), block_dim=TPB)
        self._ph1("ap_noise")

    def ap_e(mut self, a: Int, r: Int, n: Int, e: Int) raises:
        self._ph0()
        self.ctx.enqueue_function[_ap_e_kernel](
            self._fp(a), self._fp(r), Int32(n), self._ip(e), grid_dim=_grid(n), block_dim=TPB,
        )
        self._ph1("ap_e")

    def descend(mut self, x: Int, n: Int, d: Int, centers: Int, nodes: Int, labels: Int) raises:
        self._ph0()
        self.ctx.enqueue_function[_descend_kernel](
            self._fp(x), Int32(n), Int32(d), self._fp(centers), self._ip(nodes), self._ip(labels),
            grid_dim=_grid(n), block_dim=TPB,
        )
        self._ph1("descend")

    def kmeans(
        mut self, x: List[Float32], n: Int, d: Int, k: Int, max_iter: Int, tol: Float64,
        seed: UInt64, n_init: Int, init: Int, mut centers: List[Float32], mut labels: List[Int32],
        weights: List[Float32] = List[Float32](),
    ) raises -> Float64:
        self._ph0()
        var xc = x.copy()
        centers = List[Float32](length=k * d, fill=Float32(0))
        var lab = List[UInt32](length=n, fill=UInt32(0))
        var w = weights.copy() if len(weights) > 0 else List[Float32](length=1, fill=Float32(1))
        var r = kmeans_fit(
            self.ctx, xc.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), n, d, k,
            centers.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            lab.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            w.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), len(weights),
            max_iter=max_iter, tol=tol, seed=seed, n_init=n_init, init=init, metric=METRIC_L2_EXPANDED,
        )
        # KEEP THE TWO INPUTS ALIVE THROUGH THE CALL: Mojo ends a value's life
        # at its last use, which for `xc` and `w` is `unsafe_ptr()`, so without
        # these lines the fit uploads freed memory (measured 2026-09-27: run-to-
        # run drift and CUDA_ERROR_ILLEGAL_ADDRESS on the bisecting lane).
        _ = xc^
        _ = w^
        labels = List[Int32](capacity=n)
        for t in range(n):
            labels.append(Int32(lab[t]))
        self._ph1("kmeans")
        return r.inertia

    def gather_rows(mut self, src: Int, d: Int, idx: Int, m: Int, dst: Int) raises:
        self._ph0()
        # a block a row, as wide as the row (lane/neural-pass133: 4096 blocks
        # of TPB threads for 15-word taxi rows cost the minibatch step more
        # than its assignment)
        self.ctx.enqueue_function[_gather_rows_kernel](
            self._fp(src), Int32(d), self._ip(idx), Int32(m), self._fp(dst),
            grid_dim=max(m, 1), block_dim=min(TPB, max(32, (d + 31) // 32 * 32)),
        )
        self._ph1("gather_rows")

    def kmeans_rows(
        mut self, sub: Int, x: List[Float32], rows: List[Int], d: Int, k: Int, max_iter: Int,
        tol: Float64, seed: UInt64, n_init: Int, init: Int, mut centers: List[Float32],
        mut labels: List[Int32],
    ) raises -> Float64:
        self._ph0()
        var n = len(rows)
        centers = List[Float32](length=k * d, fill=Float32(0))
        var lab = List[UInt32](length=n, fill=UInt32(0))
        var r = kmeans_fit_rows(
            self.ctx, self.f[sub], MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr())), rows, d, k,
            centers.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            lab.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            max_iter=max_iter, tol=tol, seed=seed, n_init=n_init, init=init, metric=METRIC_L2_EXPANDED,
        )
        _ = x[0]
        _ = rows[0]
        labels = List[Int32](capacity=n)
        for t in range(n):
            labels.append(Int32(lab[t]))
        self._ph1("kmeans")
        return r.inertia

    def shrink(mut self, slot: Int) raises:
        self.f[slot] = self.ctx.enqueue_create_buffer[DType.float32](1)

    def empty(mut self, n: Int) raises -> Int:
        self.f.append(self.ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1))
        return len(self.f) - 1

    def gauss_q(mut self, x: Int, n: Int, d: Int, means: Int, pchol: Int, kc: Int, dst: Int) raises:
        self._ph0()
        self.ctx.enqueue_function[_gauss_q_kernel](
            self._fp(x), Int32(n), Int32(d), self._fp(means), self._fp(pchol), Int32(kc), self._fp(dst),
            grid_dim=_grid(n * kc), block_dim=TPB,
        )
        self._ph1("gauss_q")

    def resp(mut self, q: Int, c: Int, n: Int, kc: Int, lpn: Int) raises:
        self._ph0()
        self.ctx.enqueue_function[_resp_kernel](
            self._fp(q), self._fp(c), Int32(n), Int32(kc), self._fp(lpn), grid_dim=_grid(n), block_dim=TPB,
        )
        self._ph1("resp")

    def exp(mut self, src: Int, dst: Int, n: Int) raises:
        self._ph0()
        self.ctx.enqueue_function[_exp_kernel](
            self._fp(src), self._fp(dst), Int32(n), grid_dim=_grid(n), block_dim=TPB,
        )
        self._ph1("exp")

    def argmax_rows(mut self, src: Int, n: Int, kc: Int, labels: Int) raises:
        self._ph0()
        if n > 0 and kc > 0:
            self.ctx.enqueue_function[_argmax_kernel](
                self._fp(src), Int32(n), Int32(kc), self._ip(labels), grid_dim=_grid(n), block_dim=TPB,
            )
        self._ph1("argmax_rows")

    def moments(
        mut self, resp: Int, x: Int, n: Int, d: Int, kc: Int, reg: Float32, nk: Int, means: Int, cov: Int
    ) raises:
        self._ph0()
        if kc <= 0:
            self._ph1("moments")
            return
        # DEVIATION 5110 (revised 2026-09-29): under IDENTICAL the means and
        # covariances fold the sample axis through the identical GEMM, as
        # `mixture/` does (x_cluster/host/moments_gemm.mojo says why)
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            self._moments_gemm(resp, x, n, d, kc, reg, nk, means, cov)
            self._ph1("moments")
            return
        comptime if MOMS:
            if d <= MOMS_MAX_D and n > 0:
                var G = (n + MOMS_ROWS - 1) // MOMS_ROWS
                var need = kc * G * MOMS_W
                if need > self.mpart_n:
                    self.mpart = self.ctx.enqueue_create_buffer[DType.float32](need)
                    self.mpart_n = need
                var pp = self.mpart.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
                for cp in range(2):
                    var nout = d + 1 if cp == 0 else d * d
                    self.ctx.enqueue_function[_moms_part_kernel](
                        self._fp(resp), self._fp(x), Int32(n), Int32(d), Int32(kc), self._fp(means), pp,
                        Int32(G), Int32(cp), grid_dim=(kc * G + MOMS_TPB - 1) // MOMS_TPB, block_dim=MOMS_TPB,
                    )
                    self.ctx.enqueue_function[_moms_final_kernel](
                        pp, Int32(G), Int32(d), Int32(kc), reg, self._fp(nk), self._fp(means), self._fp(cov),
                        Int32(cp), grid_dim=_grid(kc * nout), block_dim=TPB,
                    )
                self._ph1("moments")
                return
        comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            if d <= MOM_MAX_D and n > 0:
                var S = (n + MOMF_ROWS - 1) // MOMF_ROWS
                var need = kc * S * (d * d if d * d > d + 1 else d + 1)
                if need > self.mpart_n:
                    self.mpart = self.ctx.enqueue_create_buffer[DType.float32](need)
                    self.mpart_n = need
                var pp = self.mpart.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
                for cp in range(2):
                    var nch = d + 1 if cp == 0 else d * d
                    self.ctx.enqueue_function[_momf_partial_kernel](
                        self._fp(resp), self._fp(x), Int32(n), Int32(d), Int32(kc), self._fp(means), pp,
                        Int32(S), Int32(cp), grid_dim=kc * S, block_dim=MOM_TPB,
                    )
                    self.ctx.enqueue_function[_momf_final_kernel](
                        pp, Int32(S), Int32(d), Int32(kc), reg, self._fp(nk), self._fp(means), self._fp(cov),
                        Int32(cp), grid_dim=_grid(kc * nch), block_dim=TPB,
                    )
                self._ph1("moments")
                return
        if d <= MOM_MAX_D:
            self.ctx.enqueue_function[_moments_pass_kernel](
                self._fp(resp), self._fp(x), Int32(n), Int32(d), Int32(kc), reg, self._fp(nk), self._fp(means),
                self._fp(cov), Int32(0), grid_dim=kc, block_dim=MOM_TPB,
            )
            var g_per_k = (d * d + MOM_COV_CPB - 1) // MOM_COV_CPB
            self.ctx.enqueue_function[_moments_pass_kernel](
                self._fp(resp), self._fp(x), Int32(n), Int32(d), Int32(kc), reg, self._fp(nk), self._fp(means),
                self._fp(cov), Int32(1), grid_dim=kc * g_per_k, block_dim=MOM_TPB,
            )
            self._ph1("moments")
            return
        self.ctx.enqueue_function[_nk_kernel](
            self._fp(resp), Int32(n), Int32(kc), self._fp(nk), grid_dim=_grid(kc), block_dim=TPB,
        )
        self.ctx.enqueue_function[_xk_kernel](
            self._fp(resp), self._fp(x), Int32(n), Int32(d), Int32(kc), self._fp(nk), self._fp(means),
            grid_dim=_grid(kc * d), block_dim=TPB,
        )
        self.ctx.enqueue_function[_cov_kernel](
            self._fp(resp), self._fp(x), Int32(n), Int32(d), Int32(kc), self._fp(means), self._fp(nk), reg,
            self._fp(cov), grid_dim=_grid(kc * d * d), block_dim=TPB,
        )
        self._ph1("moments")

    def _moments_gemm(
        mut self, resp: Int, x: Int, n: Int, d: Int, kc: Int, reg: Float32, nk: Int, means: Int, cov: Int
    ) raises:
        """nk by its row-ascending chain; means = (resp^T . X) / nk and, per
        component, cov = ((resp * diff)^T . diff) / nk + reg on the diagonal,
        both products through `identical_gemm_into` at OP_TN
        (`mixture/checks/mstep.mojo`'s kernels around them)."""
        self.ctx.enqueue_function[_nk_kernel](
            self._fp(resp), Int32(n), Int32(kc), self._fp(nk), grid_dim=_grid(kc), block_dim=TPB,
        )
        var wsn = identical_gemm_workspace_max_floats(kc, d, n)
        var w2 = identical_gemm_workspace_max_floats(d, d, n)
        if w2 > wsn:
            wsn = w2
        if wsn < 1:
            wsn = 1
        var rawn = kc * d if kc * d > d * d else d * d
        var nd = n * d if n * d > 0 else 1
        var raw = self.ctx.enqueue_create_buffer[DType.float32](rawn)
        var ws = self.ctx.enqueue_create_buffer[DType.float32](wsn)
        var diff = self.ctx.enqueue_create_buffer[DType.float32](nd)
        var scaled = self.ctx.enqueue_create_buffer[DType.float32](nd)
        # handle copies: the same device memory as the slots
        var rb = self.f[resp].copy()
        var xb = self.f[x].copy()
        identical_gemm_into(self.ctx, raw, rb, xb, ws, kc, d, n, OP_TN)
        self.ctx.enqueue_function[means_divide_kernel](
            raw.unsafe_ptr(), self._fp(nk), self._fp(means), Int32(kc), Int32(d),
            grid_dim=_grid(kc * d), block_dim=TPB,
        )
        for k in range(kc):
            self.ctx.enqueue_function[center_scale_kernel](
                self._fp(x), self._fp(means), self._fp(resp), diff.unsafe_ptr(), scaled.unsafe_ptr(),
                Int32(n), Int32(d), Int32(k), Int32(kc), grid_dim=_grid(n * d), block_dim=TPB,
            )
            identical_gemm_into(self.ctx, raw, scaled, diff, ws, d, d, n, OP_TN)
            self.ctx.enqueue_function[cov_finish_kernel](
                raw.unsafe_ptr(), self._fp(nk), self._fp(cov), Int32(d), Int32(k), reg, Int32(1),
                grid_dim=_grid(d * d), block_dim=TPB,
            )
        # the scratch buffers die with this call: drain first
        self.ctx.synchronize()
        _ = raw^
        _ = ws^
        _ = diff^
        _ = scaled^
        _ = rb^
        _ = xb^

    def pdist(
        mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, metric: Int, p: Float32, dst: Int
    ) raises:
        self._ph0()
        self.ctx.enqueue_function[_pdist_kernel](
            self._fp(a), Int32(na), self._fp(b), Int32(nb), Int32(d), Int32(metric), p, self._fp(dst),
            grid_dim=_grid(na * nb), block_dim=TPB,
        )
        self._ph1("pdist")

    def fast_device(self) -> Bool:
        return GLOBAL_NUMERIC_MODE == NUMERIC_FAST

    def ward_nn(mut self, c: Int, sz: Int, l: Int, d: Int, nn: Int, md: Int) raises:
        self._ph0()
        if l < 2:
            self._ph1("ward_nn")
            return
        self.ctx.enqueue_function[_ward_nn_kernel](
            self._fp(c), self._fp(sz), Int32(l), Int32(d), self._ip(nn), self._fp(md),
            grid_dim=l, block_dim=WNN_TPB,
        )
        self._ph1("ward_nn")

    def kth_flat(mut self, m: Int, n: Int, k: Int) raises -> Float32:
        self._ph0()
        var per = KF_TPB * KF_VPT
        var n_blocks = (n + per - 1) // per
        if n_blocks < 1:
            n_blocks = 1
        var part = self.ctx.enqueue_create_buffer[DType.int32](n_blocks * 256)
        var hist = self.ctx.enqueue_create_buffer[DType.int32](256)
        var pp = part.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var hp = hist.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var prefix = UInt32(0)
        var rem = k
        var short = False
        for step in range(4):
            var shift = 24 - 8 * step
            self.ctx.enqueue_function[_kthf_hist_kernel](
                self._fp(m), Int32(n), Int32(Int(prefix)), Int32(shift), Int32(1 if step == 0 else 0), pp,
                grid_dim=n_blocks, block_dim=KF_TPB,
            )
            self.ctx.enqueue_function[_kthf_sum_kernel](pp, Int32(n_blocks), hp, grid_dim=1, block_dim=256)
            var h = List[Int32](length=256, fill=Int32(0))
            self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=hist)
            self._sync()
            var acc = 0
            var chosen = -1
            for b in range(256):
                var c = Int(h[b])
                if acc + c >= rem:
                    chosen = b
                    break
                acc += c
            if chosen < 0:
                short = True
                chosen = 255
            prefix = prefix | (UInt32(chosen) << UInt32(shift))
            rem = rem - acc
        # the two buffers outlive every launch that holds their pointers
        _ = part^
        _ = hist^
        var r = prefix
        if short or r > UInt32(0x7F800000):
            r = UInt32(0x7F800000)
        self._ph1("kth_flat")
        return bitcast[DType.float32](r)

    def get_diag(mut self, slot: Int, n: Int) raises -> List[Float32]:
        self._ph0()
        var dbuf = self.ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        var dp = dbuf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var out = List[Float32](length=n, fill=Float32(0))
        if n > 0:
            self.ctx.enqueue_function[_diag_kernel](self._fp(slot), Int32(n), dp, grid_dim=_grid(n), block_dim=TPB)
            self.ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=dbuf)
        self._sync()
        _ = dbuf^
        self._ph1("get_diag")
        return out^

    def ap_a_split(mut self, r: Int, a: Int, n: Int, damping: Float32) raises:
        self._ph0()
        var n_tiles = (n + APF_TPB - 1) // APF_TPB
        var n_slices = (n + APF_ROWS - 1) // APF_ROWS
        var need = n_slices * n + n
        if need > self.mpart_n:
            self.mpart = self.ctx.enqueue_create_buffer[DType.float32](need)
            self.mpart_n = need
        # the slices' partial sums, then the n column sums, in the one scratch buffer
        var pp = self.mpart.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var cp = pp + n_slices * n
        self.ctx.enqueue_function[_apf_part_kernel](
            self._fp(r), Int32(n), Int32(n_tiles), pp, grid_dim=n_slices * n_tiles, block_dim=APF_TPB,
        )
        self.ctx.enqueue_function[_apf_sum_kernel](pp, Int32(n), Int32(n_slices), cp, grid_dim=_grid(n), block_dim=TPB)
        self.ctx.enqueue_function[_apf_update_kernel](
            self._fp(r), self._fp(a), cp, Int32(n), Int32(n_tiles), damping, grid_dim=n * n_tiles, block_dim=APF_TPB,
        )
        self._ph1("ap_a_split")

    def dot_groups(mut self, a: Int, b: Int, n: Int, g: Int, parts: Int) raises:
        self._ph0()
        if n > 0 and g > 0:
            self.ctx.enqueue_function[_dot_groups_kernel](
                self._fp(a), self._fp(b), Int32(n), Int32(g), self._fp(parts),
                grid_dim=_grid((n + g - 1) // g), block_dim=TPB,
            )
        self._ph1("dot_groups")

    def agglo_on_device(self) -> Bool:
        return True

    def agglo_mirror(mut self, x: Int, n: Int, dst: Int) raises:
        if n * n > 2147483647:
            raise Error("AgglomerativeClustering: n * n exceeds the Int32 index bound")
        self._ph0()
        var bad = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.ctx.enqueue_memset(bad, Int32(0))
        var pbad = bad.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        self.ctx.enqueue_function[_agg_mirror_kernel](
            self._fp(x), self._fp(dst), Int32(n), pbad, grid_dim=_grid(n * n), block_dim=TPB,
        )
        var h = List[Int32](length=1, fill=Int32(0))
        self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=bad)
        self._sync()
        _ = bad^
        self._ph1("agglo_mirror")
        if h[0] != 0:
            raise Error("AgglomerativeClustering: a precomputed distance matrix must be finite and non-negative")

    def agglo_merge(
        mut self, dm: Int, adj: Int, n: Int, linkage: Int, n_merges: Int, mut children: List[Int32],
        mut dist: List[Float32],
    ) raises:
        if n * n > 2147483647:
            raise Error("AgglomerativeClustering: n * n exceeds the Int32 index bound")
        self._ph0()
        var ctx = self.ctx.copy()
        var live = ctx.enqueue_create_buffer[DType.int32](n)
        var nn = ctx.enqueue_create_buffer[DType.int32](n)
        var node = ctx.enqueue_create_buffer[DType.int32](n)
        var lst = ctx.enqueue_create_buffer[DType.int32](n)
        var st = ctx.enqueue_create_buffer[DType.int32](4)
        ctx.enqueue_memset(st, Int32(0))
        var md = ctx.enqueue_create_buffer[DType.float32](n)
        var sz = ctx.enqueue_create_buffer[DType.float32](n)
        var stf = ctx.enqueue_create_buffer[DType.float32](3)
        var nch = max(2 * n_merges, 1)
        var ndv = max(n_merges, 1)
        var ch = ctx.enqueue_create_buffer[DType.int32](nch)
        var dv = ctx.enqueue_create_buffer[DType.float32](ndv)
        var nb = (n + AGG_PER - 1) // AGG_PER
        var part = ctx.enqueue_create_buffer[DType.uint64](nb)
        var p_live = live.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_nn = nn.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_node = node.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_lst = lst.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_st = st.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_md = md.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_sz = sz.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_stf = stf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_ch = ch.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_dv = dv.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_part = part.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var p_dm = self._fp(dm)
        var con = Int32(1) if adj >= 0 else Int32(0)
        var p_adj = self._ip(adj) if adj >= 0 else p_lst
        var rb = n if n < AGG_RESCAN_BLOCKS else AGG_RESCAN_BLOCKS
        ctx.enqueue_function[_agg_init_kernel](
            p_live, p_nn, p_md, p_sz, p_node, p_lst, p_st, Int32(n), grid_dim=_grid(n), block_dim=TPB,
        )
        # every row's first nearest partner: the rescan of the full list
        ctx.enqueue_function[_agg_rescan_kernel](
            p_dm, p_live, p_nn, p_md, Int32(n), p_st, p_lst, p_adj, con, grid_dim=rb, block_dim=AGG_TPB,
        )
        for step in range(n_merges):
            ctx.enqueue_function[_agg_argmin_part_kernel](
                p_md, p_nn, p_live, Int32(n), p_part, p_st, grid_dim=nb, block_dim=AGG_TPB,
            )
            ctx.enqueue_function[_agg_lw_kernel](
                p_part, Int32(nb), p_dm, p_nn, p_live, p_sz, Int32(n), Int32(linkage), p_st, p_stf, p_adj, con,
                grid_dim=_grid(n), block_dim=TPB,
            )
            ctx.enqueue_function[_agg_flag_kernel](
                p_dm, p_live, p_nn, p_md, p_sz, p_node, Int32(n), Int32(step), Int32(linkage), p_ch, p_dv,
                p_st, p_stf, p_lst, p_adj, con, grid_dim=_grid(n), block_dim=TPB,
            )
            ctx.enqueue_function[_agg_rescan_kernel](
                p_dm, p_live, p_nn, p_md, Int32(n), p_st, p_lst, p_adj, con, grid_dim=rb, block_dim=AGG_TPB,
            )
        var h_st = List[Int32](length=4, fill=Int32(0))
        children = List[Int32](length=nch, fill=Int32(0))
        dist = List[Float32](length=ndv, fill=Float32(0))
        ctx.enqueue_copy(dst_ptr=h_st.unsafe_ptr(), src_buf=st)
        ctx.enqueue_copy(dst_ptr=children.unsafe_ptr(), src_buf=ch)
        ctx.enqueue_copy(dst_ptr=dist.unsafe_ptr(), src_buf=dv)
        self._sync()
        _ = live^
        _ = nn^
        _ = node^
        _ = lst^
        _ = st^
        _ = md^
        _ = sz^
        _ = stf^
        _ = ch^
        _ = dv^
        _ = part^
        self._ph1("agglo_merge")
        if h_st[3] != 0:
            raise Error("AgglomerativeClustering: no connected pair is left to merge")
        children.resize(2 * n_merges, Int32(0))
        dist.resize(n_merges, Float32(0))

    def alloc(mut self, n: Int) raises -> Int:
        self._ph0()
        var buf = self.ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        self.f.append(buf^)
        self._ph1("alloc")
        return len(self.f) - 1

    def estep(
        mut self, x: Int, n: Int, d: Int, means: Int, pchol: Int, c: Int, kc: Int, q: Int, r: Int, lpn: Int
    ) raises:
        self._ph0()
        self.ctx.enqueue_function[_estep_row_kernel](
            self._fp(x), Int32(n), Int32(d), self._fp(means), self._fp(pchol), self._fp(c), Int32(kc),
            self._fp(q), self._fp(r), self._fp(lpn), grid_dim=_grid(n), block_dim=TPB,
        )
        self._ph1("estep")

    def set_i(mut self, slot: Int, v: List[Int32]) raises:
        self._ph0()
        var n = len(v)
        if n == 0:
            self._ph1("set_i")
            return
        var view = self.i[slot].create_sub_buffer[DType.int32](0, n)
        self.pend_i.append(v.copy())
        self.ctx.enqueue_copy(dst_buf=view, src_ptr=self.pend_i[len(self.pend_i) - 1].unsafe_ptr())
        self._ph1("set_i")

    def mb_assign(mut self, src: Int, d: Int, idx: Int, m: Int, c: Int, k: Int, labels: Int, dist: Int, dst: Int) raises:
        self._ph0()
        self.ctx.enqueue_function[_mb_assign_kernel](
            self._fp(src), Int32(d), self._ip(idx), Int32(m), self._fp(c), Int32(k), self._ip(labels),
            self._fp(dist), self._fp(dst), grid_dim=_grid(m), block_dim=TPB,
        )
        self._ph1("mb_assign")

    def mb_update(mut self, b: Int, batch: Int, labels: Int, c: Int, w: Int, k: Int, d: Int) raises:
        self._ph0()
        comptime if MB_BLOCK_FITS:
            self.ctx.enqueue_function[_mb_centers_block_kernel](
                self._fp(b), Int32(batch), self._ip(labels), self._fp(c), self._fp(w), Int32(d),
                grid_dim=k, block_dim=MB_TPB,
            )
        else:
            self.ctx.enqueue_function[_mb_centers_kernel](
                self._fp(b), Int32(batch), self._ip(labels), self._fp(c), self._fp(w), Int32(k), Int32(d),
                grid_dim=_grid(k * d), block_dim=TPB,
            )
            self.ctx.enqueue_function[_mb_counts_kernel](
                Int32(batch), self._ip(labels), self._fp(w), Int32(k), grid_dim=_grid(k), block_dim=TPB,
            )
        self._ph1("mb_update")

    # ------------------------------------------------------------------
    # lane cgr2-cluster: the post-processing primitives on the device
    def _keys(mut self, n: Int) raises -> UPtr:
        """A new uninitialized 64-bit key buffer that lives as long as the ops."""
        self.u.append(self.ctx.enqueue_create_buffer[DType.uint64](n if n > 0 else 1))
        return self.u[len(self.u) - 1].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    def _int1(mut self, slot: Int) raises -> Int:
        """Read word 0 of an int slot (the device waits)."""
        var h = List[Int32](length=1, fill=Int32(0))
        self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=self.i[slot])
        self._sync()
        return Int(h[0])

    def _scan(mut self, flags: IPtr, n: Int) raises -> Tuple[Int, Int]:
        """The exclusive integer scan of n flags: (int slot of the prefix
        counts, int slot whose word 0 is the total). Enqueued only."""
        var nb = (n + SCAN_PER - 1) // SCAN_PER
        if nb < 1:
            nb = 1
        var part = self.zeros_i(nb)
        var out = self.zeros_i(n)
        var tot = self.zeros_i(1)
        self.ctx.enqueue_function[scan_part_kernel](flags, Int32(n), self._ip(part), grid_dim=nb, block_dim=RTPB)
        self.ctx.enqueue_function[scan_out_kernel](
            flags, Int32(n), self._ip(part), Int32(nb), self._ip(out), self._ip(tot), grid_dim=nb, block_dim=RTPB,
        )
        return (out, tot)

    def _flag_true(mut self, bad: Int) raises -> Bool:
        return self._int1(bad) != 0

    def _fold(mut self, mode: Int, pa: FPtr, pb: FPtr, pc: FPtr, n: Int, oh: FPtr, ol: FPtr, idx: Int) raises:
        """The float-float fold of n elements into oh/ol[idx], enqueued."""
        if n <= 0:
            var z = self.zeros(1)
            self.ctx.enqueue_function[ff_chunk_kernel](
                Int32(FM_VAL), self._fp(z), self._fp(z), self._fp(z), Int32(0), oh, ol, Int32(idx),
                grid_dim=1, block_dim=RTPB,
            )
            return
        var nch = (n + FOLD_CHUNK - 1) // FOLD_CHUNK
        var md = mode
        var a = pa
        var b = pb
        var c = pc
        var cnt = n
        while True:
            if nch == 1:
                self.ctx.enqueue_function[ff_chunk_kernel](
                    Int32(md), a, b, c, Int32(cnt), oh, ol, Int32(idx), grid_dim=1, block_dim=RTPB,
                )
                return
            var th = self.alloc(nch)
            var tl = self.alloc(nch)
            self.ctx.enqueue_function[ff_chunk_kernel](
                Int32(md), a, b, c, Int32(cnt), self._fp(th), self._fp(tl), Int32(0), grid_dim=nch, block_dim=RTPB,
            )
            md = FM_FF
            a = self._fp(th)
            b = self._fp(tl)
            c = self._fp(th)
            cnt = nch
            nch = (nch + FOLD_CHUNK - 1) // FOLD_CHUNK

    def agglo_connect(mut self, edges: Int, n_edges: Int, n: Int, dm: Int, linkage: Int, adj: Int) raises -> Int:
        self._ph0()
        var bad = self.zeros_i(1)
        var pa = self._ip(adj)
        if n_edges > 0:
            self.ctx.enqueue_function[agc_edges_kernel](
                self._fp(edges), Int32(n_edges), Int32(n), pa, self._ip(bad), grid_dim=pgrid(n_edges), block_dim=PTPB,
            )
        if self._flag_true(bad):
            raise Error("AgglomerativeClustering: a connectivity edge is outside [0, n)")
        # the components: min-label propagation to the fixed point
        var lab_h = List[Int32](capacity=n)
        for v in range(n):
            lab_h.append(Int32(v))
        var lab = self.put_i(lab_h)
        var changed = self.zeros_i(1)
        while True:
            self.ctx.enqueue_function[agc_prop_kernel](
                pa, Int32(n), self._ip(lab), self._ip(changed), grid_dim=pgrid(n), block_dim=PTPB,
            )
            self.ctx.enqueue_memset(self.i[changed], Int32(0))
            self.ctx.enqueue_function[agc_prop_kernel](
                pa, Int32(n), self._ip(lab), self._ip(changed), grid_dim=pgrid(n), block_dim=PTPB,
            )
            if not self._flag_true(changed):
                break
            self.ctx.enqueue_memset(self.i[changed], Int32(0))
        var isroot = self.zeros_i(n)
        self.ctx.enqueue_function[agc_root_kernel](self._ip(lab), Int32(n), self._ip(isroot), grid_dim=pgrid(n), block_dim=PTPB)
        var sc = self._scan(self._ip(isroot), n)
        var cid = sc[0]
        var C = self._int1(sc[1])
        if C <= 1:
            self._ph1("agglo_connect")
            return 1
        var comp = self.zeros_i(n)
        var start = self.zeros_i(C)
        var members = self.zeros_i(n)
        self.ctx.enqueue_function[agc_comp_kernel](
            self._ip(lab), self._ip(cid), Int32(n), self._ip(comp), grid_dim=pgrid(n), block_dim=PTPB,
        )
        self.ctx.enqueue_function[agc_start_kernel](
            self._ip(isroot), self._ip(cid), self._ip(comp), Int32(n), self._ip(start), grid_dim=pgrid(n), block_dim=PTPB,
        )
        self.ctx.enqueue_function[agc_members_kernel](
            self._ip(comp), self._ip(start), Int32(n), self._ip(members), grid_dim=pgrid(n), block_dim=PTPB,
        )
        var cc = (1 << 24) // n
        if cc < 1:
            cc = 1
        if cc > C:
            cc = C
        var rowbest = self._keys(n * cc)
        var ward = Int32(1) if linkage == LINK_WARD else Int32(0)
        var c0 = 0
        while c0 < C:
            var k = min(cc, C - c0)
            self.ctx.enqueue_function[agc_rowbest_kernel](
                self._fp(dm), self._ip(comp), self._ip(start), self._ip(members), Int32(n), Int32(C), Int32(c0),
                Int32(k), ward, rowbest, grid_dim=pgrid(n * k), block_dim=PTPB,
            )
            self.ctx.enqueue_function[agc_join_kernel](
                rowbest, self._ip(comp), self._ip(start), self._ip(members), Int32(n), Int32(C), Int32(c0), Int32(k),
                pa, grid_dim=pgrid(C * k), block_dim=PTPB,
            )
            c0 += k
        self._ph1("agglo_connect")
        return C

    def check_nonneg(mut self, x: Int, n: Int) raises -> Bool:
        self._ph0()
        var bad = self.zeros_i(1)
        if n > 0:
            self.ctx.enqueue_function[nonneg_kernel](self._fp(x), Int32(n), self._ip(bad), grid_dim=pgrid(n), block_dim=PTPB)
        var r = not self._flag_true(bad)
        self._ph1("check_nonneg")
        return r

    def optics_order(
        mut self, dm: Int, core: Int, n: Int, max_eps: Float32, ordering: Int, reach: Int, pred: Int
    ) raises:
        self._ph0()
        var done = self.zeros_i(n)
        var nb = (n + SCAN_PER - 1) // SCAN_PER
        var part = self._keys(nb)
        self.ctx.enqueue_function[optics_init_kernel](
            self._fp(core), Int32(n), max_eps, self._fp(reach), self._ip(pred), self._ip(done),
            grid_dim=pgrid(n), block_dim=PTPB,
        )
        for step in range(n):
            self.ctx.enqueue_function[optics_part_kernel](
                self._fp(reach), self._ip(done), Int32(n), part, grid_dim=nb, block_dim=RTPB,
            )
            self.ctx.enqueue_function[optics_step_kernel](
                part, Int32(nb), self._fp(dm), self._fp(core), Int32(n), max_eps, self._ip(done), self._fp(reach),
                self._ip(pred), self._ip(ordering), Int32(step), grid_dim=pgrid(n), block_dim=PTPB,
            )
        self._ph1("optics_order")

    def optics_dbscan(mut self, ordering: Int, reach: Int, core: Int, n: Int, eps: Float32, labels: Int) raises:
        self._ph0()
        var flag = self.zeros_i(n)
        self.ctx.enqueue_function[optics_flag_kernel](
            self._ip(ordering), self._fp(reach), self._fp(core), Int32(n), eps, self._ip(flag),
            grid_dim=pgrid(n), block_dim=PTPB,
        )
        var sc = self._scan(self._ip(flag), n)
        self.ctx.enqueue_function[optics_label_kernel](
            self._ip(ordering), self._fp(reach), self._fp(core), Int32(n), eps, self._ip(flag), self._ip(sc[0]),
            self._ip(labels), grid_dim=pgrid(n), block_dim=PTPB,
        )
        self._ph1("optics_dbscan")

    def sum_ff(mut self, a: Int, b: Int, c: Int, n: Int, mode: Int) raises -> Float64:
        self._ph0()
        var o = self.zeros(2)
        var po = self._fp(o)
        self._fold(mode, self._fp(a), self._fp(b if b >= 0 else a), self._fp(c if c >= 0 else a), n, po, po + 1, 0)
        var h = self.get(o, 2)
        self._ph1("sum_ff")
        return Float64(h[0]) + Float64(h[1])

    def bin_seeds(mut self, x: Int, n: Int, d: Int, bin_size: Float32, min_bin_freq: Int, dst: Int) raises -> Int:
        self._ph0()
        var keys = self.alloc(n * d)
        var rep = self.zeros_i(n)
        var kept = self.zeros_i(n)
        self.ctx.enqueue_function[bin_key_kernel](
            self._fp(x), Int32(n * d), bin_size, self._fp(keys), grid_dim=pgrid(n * d), block_dim=PTPB,
        )
        self.ctx.enqueue_function[first_equal_kernel](
            self._fp(keys), self._ip(rep), Int32(0), Int32(d), Int32(n), self._ip(rep), grid_dim=pgrid(n), block_dim=PTPB,
        )
        self.ctx.enqueue_function[bin_count_kernel](
            self._ip(rep), Int32(n), Int32(min_bin_freq), self._ip(kept), grid_dim=pgrid(n), block_dim=PTPB,
        )
        var sc = self._scan(self._ip(kept), n)
        self.ctx.enqueue_function[bin_scatter_kernel](
            self._fp(keys), self._ip(kept), self._ip(sc[0]), Int32(n), Int32(d), bin_size, self._fp(dst),
            grid_dim=pgrid(n), block_dim=PTPB,
        )
        var k = self._int1(sc[1])
        self._ph1("bin_seeds")
        return k

    def ms_unique(
        mut self, centers: Int, inten: Int, iters: Int, ns: Int, d: Int, dst: Int, mut n_iter: Int
    ) raises -> Int:
        self._ph0()
        var nb = (ns + RTPB - 1) // RTPB
        if nb < 1:
            nb = 1
        var part = self.zeros_i(nb)
        self.ctx.enqueue_function[max_part_kernel](self._ip(iters), Int32(ns), self._ip(part), grid_dim=nb, block_dim=RTPB)
        var rep = self.zeros_i(ns)
        var rv = self.zeros_i(ns)
        var cnt = self.zeros_i(1)
        self.ctx.enqueue_function[first_equal_kernel](
            self._fp(centers), self._ip(inten), Int32(1), Int32(d), Int32(ns), self._ip(rep), grid_dim=pgrid(ns), block_dim=PTPB,
        )
        self.ctx.enqueue_function[ms_last_kernel](
            self._ip(rep), self._ip(inten), Int32(ns), self._ip(rv), grid_dim=pgrid(ns), block_dim=PTPB,
        )
        self.ctx.enqueue_function[ms_rank_kernel](
            self._fp(centers), self._ip(rep), self._ip(rv), Int32(ns), Int32(d), self._fp(dst), self._ip(cnt),
            grid_dim=pgrid(ns), block_dim=PTPB,
        )
        var parts = self.get_i(part, nb)
        n_iter = 0
        for q in range(nb):
            if Int(parts[q]) > n_iter:
                n_iter = Int(parts[q])
        var m = self._int1(cnt)
        self._ph1("ms_unique")
        return m

    def ms_suppress(mut self, sorted: Int, dd: Int, m: Int, d: Int, bw: Float32, dst: Int) raises -> Int:
        self._ph0()
        var und = self.zeros_i(m)
        var uni = self.zeros_i(m)
        self.ctx.enqueue_function[fill_i_kernel](self._ip(und), Int32(m), Int32(1), grid_dim=pgrid(m), block_dim=PTPB)
        var nb = (m + SCAN_PER - 1) // SCAN_PER
        if nb < 1:
            nb = 1
        var part = self.zeros_i(nb)
        # rounds: the lowest undecided center is kept and drops every
        # undecided one within the bandwidth; the batch ends with a check
        while True:
            for _r in range(32):
                self.ctx.enqueue_function[lowest_part_kernel](
                    self._ip(und), Int32(m), self._ip(part), grid_dim=nb, block_dim=RTPB,
                )
                self.ctx.enqueue_function[ms_mark_kernel](
                    self._ip(part), Int32(nb), self._fp(dd), Int32(m), bw, self._ip(und), self._ip(uni),
                    grid_dim=pgrid(m), block_dim=PTPB,
                )
            self.ctx.enqueue_function[lowest_part_kernel](
                self._ip(und), Int32(m), self._ip(part), grid_dim=nb, block_dim=RTPB,
            )
            var parts = self.get_i(part, nb)
            var left = False
            for q in range(nb):
                if Int(parts[q]) < m:
                    left = True
            if not left:
                break
        var sc = self._scan(self._ip(uni), m)
        self.ctx.enqueue_function[compact_rows_kernel](
            self._fp(sorted), self._ip(uni), self._ip(sc[0]), Int32(m), Int32(d), self._fp(dst),
            grid_dim=pgrid(m), block_dim=PTPB,
        )
        var kc = self._int1(sc[1])
        self._ph1("ms_suppress")
        return kc

    def ms_noise(mut self, labels: Int, dist: Int, n: Int, bw: Float32) raises:
        self._ph0()
        self.ctx.enqueue_function[ms_noise_kernel](
            self._ip(labels), self._fp(dist), Int32(n), bw, grid_dim=pgrid(n), block_dim=PTPB,
        )
        self._ph1("ms_noise")

    def negate(mut self, src: Int, dst: Int, n: Int) raises:
        self._ph0()
        self.ctx.enqueue_function[negate_kernel](self._fp(src), self._fp(dst), Int32(n), grid_dim=pgrid(n), block_dim=PTPB)
        self._ph1("negate")

    def count_neg(mut self, x: Int, n: Int) raises -> Int:
        self._ph0()
        var cnt = self.zeros_i(1)
        self.ctx.enqueue_function[count_neg_kernel](self._fp(x), Int32(n), self._ip(cnt), grid_dim=pgrid(n), block_dim=PTPB)
        var c = self._int1(cnt)
        self._ph1("count_neg")
        return c

    def sign_side(mut self, src: Int, n: Int, neg: Bool, dst: Int) raises:
        self._ph0()
        self.ctx.enqueue_function[sign_side_kernel](
            self._fp(src), Int32(n), Int32(1) if neg else Int32(0), self._fp(dst), grid_dim=pgrid(n), block_dim=PTPB,
        )
        self._ph1("sign_side")

    def ap_equal(mut self, s: Int, pref: Int, n: Int) raises -> Bool:
        self._ph0()
        var bad = self.zeros_i(1)
        self.ctx.enqueue_function[ap_equal_kernel](
            self._fp(s), self._fp(pref), Int32(n), self._ip(bad), grid_dim=pgrid(n * n), block_dim=PTPB,
        )
        var r = not self._flag_true(bad)
        self._ph1("ap_equal")
        return r

    def set_diag(mut self, s: Int, v: Int, n: Int) raises:
        self._ph0()
        self.ctx.enqueue_function[set_diag_kernel](self._fp(s), self._fp(v), Int32(n), grid_dim=pgrid(n), block_dim=PTPB)
        self._ph1("set_diag")

    def ap_conv(mut self, e: Int, ring: Int, n: Int, conv_iter: Int, it: Int) raises -> Bool:
        self._ph0()
        var st = self.zeros_i(2)
        self.ctx.enqueue_function[ap_conv_kernel](
            self._ip(e), self._ip(ring), Int32(n), Int32(conv_iter), Int32(it), self._ip(st),
            grid_dim=pgrid(n), block_dim=PTPB,
        )
        var h = self.get_i(st, 2)
        self._ph1("ap_conv")
        return it >= conv_iter and h[1] == 0 and h[0] > 0

    def ap_exemplars(mut self, s: Int, e: Int, n: Int, centers: Int, labels: Int) raises -> Int:
        self._ph0()
        var sc = self._scan(self._ip(e), n)
        var K = self._int1(sc[1])
        if K == 0:
            self.ctx.enqueue_function[fill_i_kernel](self._ip(labels), Int32(n), Int32(-1), grid_dim=pgrid(n), block_dim=PTPB)
            self._ph1("ap_exemplars")
            return 0
        var ex = self.zeros_i(K)
        var inv = self.zeros_i(n)
        var c = self.zeros_i(n)
        var acc = self.alloc(n)
        self.ctx.enqueue_function[ap_exsc_kernel](
            self._ip(e), self._ip(sc[0]), Int32(n), self._ip(ex), grid_dim=pgrid(n), block_dim=PTPB,
        )
        for rnd in range(2):
            self.ctx.enqueue_function[fill_i_kernel](self._ip(inv), Int32(n), Int32(-1), grid_dim=pgrid(n), block_dim=PTPB)
            self.ctx.enqueue_function[ap_inv_kernel](self._ip(ex), Int32(K), self._ip(inv), grid_dim=pgrid(K), block_dim=PTPB)
            self.ctx.enqueue_function[ap_c_kernel](
                self._fp(s), self._ip(ex), Int32(K), Int32(n), self._ip(inv), self._ip(c), grid_dim=pgrid(n), block_dim=PTPB,
            )
            if rnd == 0:
                self.ctx.enqueue_function[ap_acc_kernel](
                    self._fp(s), self._ip(c), Int32(n), self._fp(acc), grid_dim=pgrid(n), block_dim=PTPB,
                )
                self.ctx.enqueue_function[ap_best_kernel](
                    self._fp(acc), self._ip(c), Int32(n), self._ip(ex), grid_dim=K, block_dim=RTPB,
                )
        var isc = self.zeros_i(n)
        self.ctx.enqueue_function[ap_isc_kernel](self._ip(ex), self._ip(c), Int32(n), self._ip(isc), grid_dim=pgrid(n), block_dim=PTPB)
        var sc2 = self._scan(self._ip(isc), n)
        self.ctx.enqueue_function[ap_label_kernel](
            self._ip(ex), self._ip(c), self._ip(isc), self._ip(sc2[0]), Int32(n), self._ip(centers), self._ip(labels),
            grid_dim=pgrid(n), block_dim=PTPB,
        )
        var nc = self._int1(sc2[1])
        self._ph1("ap_exemplars")
        return nc

    def onehot(mut self, idx: Int, m: Int, kc: Int, by_row: Bool, dst: Int) raises:
        self._ph0()
        self.ctx.enqueue_function[onehot_kernel](
            self._ip(idx), Int32(m), Int32(kc), Int32(1) if by_row else Int32(0), self._fp(dst),
            grid_dim=pgrid(m), block_dim=PTPB,
        )
        self._ph1("onehot")

    def rand_resp(mut self, dst: Int, n: Int, kc: Int, state: UInt64) raises:
        self._ph0()
        self.ctx.enqueue_function[rand_resp_kernel](self._fp(dst), Int32(n), Int32(kc), state, grid_dim=pgrid(n), block_dim=PTPB)
        self._ph1("rand_resp")

    def kpp_search(mut self, closest: Int, w: Int, m: Int, vs: List[Float64], ids: Int) raises:
        self._ph0()
        var mode = FM_PROD if w >= 0 else FM_VAL
        var nch = (m + FOLD_CHUNK - 1) // FOLD_CHUNK
        var th = self.alloc(nch)
        var tl = self.alloc(nch)
        var pc = self._fp(closest)
        var pw = self._fp(w if w >= 0 else closest)
        self.ctx.enqueue_function[ff_chunk_kernel](
            Int32(mode), pc, pw, pc, Int32(m), self._fp(th), self._fp(tl), Int32(0), grid_dim=nch, block_dim=RTPB,
        )
        var vh = List[Float32](capacity=2 * len(vs))
        for t in range(len(vs)):
            var v = ff_of_f64(vs[t])
            vh.append(v.hi)
            vh.append(v.lo)
        var vslot = self.put(vh)
        self.ctx.enqueue_function[kpp_search_kernel](
            Int32(mode), pc, pw, self._fp(th), self._fp(tl), Int32(m), self._fp(vslot), Int32(len(vs)), self._ip(ids),
            grid_dim=pgrid(len(vs)), block_dim=PTPB,
        )
        self._ph1("kpp_search")

    def kpp_pots(mut self, dc: Int, closest: Int, w: Int, nt: Int, m: Int) raises -> List[Float64]:
        self._ph0()
        var o = self.zeros(2 * nt)
        var po = self._fp(o)
        var pc = self._fp(closest)
        var pw = self._fp(w if w >= 0 else closest)
        for t in range(nt):
            self._fold(FM_WMIN if w >= 0 else FM_MIN, self._fp(dc) + t * m, pc, pw, m, po, po + nt, t)
        var h = self.get(o, 2 * nt)
        var out = List[Float64](capacity=nt)
        for t in range(nt):
            out.append(Float64(h[t]) + Float64(h[nt + t]))
        self._ph1("kpp_pots")
        return out^

    def kpp_take(mut self, dc: Int, closest: Int, best: Int, m: Int) raises:
        self._ph0()
        self.ctx.enqueue_function[kpp_take_kernel](
            self._fp(dc) + best * m, self._fp(closest), Int32(m), grid_dim=pgrid(m), block_dim=PTPB,
        )
        self._ph1("kpp_take")





def _mb_centers_kernel(b: FPtr, batch: Int32, labels: IPtr, c: FPtr, w: FPtr, k: Int32, d: Int32):
    """One thread a center word (lane/neural-pass133): `mb_center_word`; the
    counts are left to `_mb_counts_kernel` (launched after, so every thread
    reads the batch's starting count)."""
    var q = _tid()
    var dd = Int(d)
    if q < Int(k) * dd:
        var j = q // dd
        var f = q - j * dd
        var wsum = mb_center_wsum(labels, Int(batch), j)
        if wsum > Float32(0):
            c[q] = mb_center_word(b, Int(batch), labels, c[q], w[j], wsum, j, f, dd)


def _mb_counts_kernel(batch: Int32, labels: IPtr, w: FPtr, k: Int32):
    var j = _tid()
    if j < Int(k):
        var wsum = mb_center_wsum(labels, Int(batch), j)
        if wsum > Float32(0):
            w[j] = ftz(w[j] + wsum)


#: The block form (one block a center, `_mb_centers_block_kernel`): each
#: chunk of the batch compacted to the center's rows by a block scan in
#: threadgroup memory, each thread a feature's chain over them in batch order;
#: the count an integer sum (exact, so the host's `+ 1` chain's word below
#: 2^24 rows). Behind its fits gate; the per-word kernels above otherwise.
comptime MB_TPB = 256
comptime MB_PF = 8
comptime MB_BLOCK_FITS = lib_smem_page_fits_for[TARGET_COLUMN, 2 * MB_TPB * 4]()


def _mb_centers_block_kernel(b: FPtr, batch: Int32, labels: IPtr, c: FPtr, w: FPtr, d: Int32):
    """One block a center j: per chunk of MB_TPB batch rows, a block scan
    compacts the chunk's rows of j in batch order into `sel`, then each
    thread chains its feature over them; the count is the sum of the chunk
    counts (exact). Thread 0 writes the count last: every thread read it
    into `wj` before the first barrier."""
    var j = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nb = Int(batch)
    var dd = Int(d)
    var scan = stack_allocation[MB_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var sel = stack_allocation[MB_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var wj = w[j]
    var total = 0
    var f0 = 0
    while f0 < dd:
        var f = f0 + tid
        var acc = Float32(0)
        if f < dd:
            acc = ftz(identical_mul(c[j * dd + f], wj))
        var tot = 0
        var t0 = 0
        while t0 < nb:
            var mine = 1 if (t0 + tid < nb and Int(labels[t0 + tid]) == j) else 0
            # inclusive scan: within the warp by shuffles, then the warp
            # totals (integers: any order gives the same counts)
            var lane = tid % WARP_SIZE
            var wid = tid // WARP_SIZE
            var v = Int32(mine)
            var off = 1
            while off < WARP_SIZE:
                var u = shuffle_idx(v, UInt32(max(lane - off, 0)))
                if lane >= off:
                    v += u
                off *= 2
            barrier()
            if lane == WARP_SIZE - 1:
                scan[wid] = v
            barrier()
            var base = Int32(0)
            for q in range(wid):
                base += scan[q]
            var m = 0
            for q in range(MB_TPB // WARP_SIZE):
                m += Int(scan[q])
            if mine == 1:
                sel[Int(base + v) - 1] = Int32(tid)
            barrier()
            if f < dd:
                # the loads of MB_PF rows issued before their adds (the adds
                # stay in batch order): the chain no longer waits on each load
                var q = 0
                while q + MB_PF <= m:
                    var v = SIMD[DType.float32, MB_PF]()
                    comptime for u in range(MB_PF):
                        v[u] = b[(t0 + Int(sel[q + u])) * dd + f]
                    comptime for u in range(MB_PF):
                        acc = ftz(acc + ftz(v[u]))
                    q += MB_PF
                while q < m:
                    acc = ftz(acc + ftz(b[(t0 + Int(sel[q])) * dd + f]))
                    q += 1
            tot += m
            t0 += MB_TPB
        total = tot
        if total > 0 and f < dd:
            var alpha = ftz(identical_div(Float32(1), ftz(wj + Float32(total))))
            c[j * dd + f] = ftz(identical_mul(acc, alpha))
        f0 += MB_TPB
    barrier()
    if tid == 0 and total > 0:
        w[j] = ftz(wj + Float32(total))


def _mb_assign_kernel(src: FPtr, d: Int32, idx: IPtr, m: Int32, c: FPtr, k: Int32, labels: IPtr, dist: FPtr,
                      dst: FPtr):
    """One thread a batch row t (lane/neural-pass133): copies row idx[t] of
    `src` to row t of `dst`, then `nearest_row`'s loop on it (the same
    `sq_dist_rows` words, the lowest index on a tie)."""
    var t = _tid()
    if t < Int(m):
        var dd = Int(d)
        var r = Int(idx[t])
        for f in range(dd):
            dst[t * dd + f] = src[r * dd + f]
        var best = sq_dist_rows(dst, t, c, 0, dd)
        var bi = 0
        for j in range(1, Int(k)):
            var v = sq_dist_rows(dst, t, c, j, dd)
            if v < best:
                best = v
                bi = j
        labels[t] = Int32(bi)
        dist[t] = best
