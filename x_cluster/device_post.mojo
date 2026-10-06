"""THE CLUSTER LANE'S POST-PROCESSING KERNELS (lane cgr2-cluster,
2026-10-03): the n-sized parts of agglomerative-with-connectivity, OPTICS,
MeanShift, AffinityPropagation, the mixtures and k-means++ that ran on the
host between device calls (`ClusterOps` primitives, `DeviceOps` in
`x_cluster/device_ops.mojo` enqueues them). Every pick is a min of
`post_bodies.order_key` (the value, then the index), every count an integer
(atomics or block sums: any interleaving gives the same integer), every float
sum the fixed blocked-then-tree float-float fold of `post_bodies`, so no
launch shape moves a bit and the host column's loops give the same words.
Only the GPU binding imports this file."""
from experiments.classical_identical_ideas.graph_controls import C41_FUSED_MINIMA
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_sqrt
from x_cluster.bodies import FPtr, IPtr
from x_cluster.post_bodies import (
    FOLD_CHUNK,
    FOLD_LANES,
    KEY_NONE,
    bin_key,
    bin_value,
    center_cell,
    center_greater,
    ff_lane,
    ff_mean_cell,
    ff_strided_lane,
    first_equal_cell,
    kpp_search_cell,
    optics_relax_cell,
    order_key,
    rand_resp_row,
)
from x_linear.ff import FF, ff_add

comptime UPtr = MutPointer[UInt64, MutAnyOrigin]
comptime PTPB = 128
#: threads of a reduction / scan block; a scan block covers SCAN_PER values
comptime RTPB = 256
comptime SCAN_PER = RTPB * 4


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def pgrid(n: Int) -> Int:
    return (n + PTPB - 1) // PTPB if n > 0 else 1


@always_inline
def _block_min_u64(red: UnsafePointer[UInt64, MutUntrackedOrigin, address_space=AddressSpace.SHARED], mine: UInt64) -> UInt64:
    var tid = Int(thread_idx.x)
    red[tid] = mine
    barrier()
    var off = RTPB // 2
    while off > 0:
        if tid < off:
            red[tid] = min(red[tid], red[tid + off])
        barrier()
        off //= 2
    var r = red[0]
    barrier()
    return r


# ------------------------------------------------------------- integer scan
def scan_part_kernel(flags: IPtr, n: Int32, part: IPtr):
    """part[b] = the sum of block b's SCAN_PER flags."""
    var red = stack_allocation[RTPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var base = Int(block_idx.x) * SCAN_PER + tid * 4
    var s = Int32(0)
    for q in range(4):
        if base + q < Int(n):
            s += flags[base + q]
    red[tid] = s
    barrier()
    var off = RTPB // 2
    while off > 0:
        if tid < off:
            red[tid] = red[tid] + red[tid + off]
        barrier()
        off //= 2
    if tid == 0:
        part[Int(block_idx.x)] = red[0]


def scan_out_kernel(flags: IPtr, n: Int32, part: IPtr, nb: Int32, dst: IPtr, total: IPtr):
    """dst[t] = the sum of flags[0 .. t) (exclusive); total[0] = the sum of
    all (block 0)."""
    var sh = stack_allocation[RTPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var off0 = Int32(0)
    for q in range(b):
        off0 += part[q]
    if b == 0 and tid == 0:
        var tt = Int32(0)
        for q in range(Int(nb)):
            tt += part[q]
        total[0] = tt
    var base = b * SCAN_PER + tid * 4
    var loc = Int32(0)
    for q in range(4):
        if base + q < Int(n):
            loc += flags[base + q]
    sh[tid] = loc
    barrier()
    var off = 1
    while off < RTPB:
        var v = Int32(0)
        if tid >= off:
            v = sh[tid - off]
        barrier()
        sh[tid] = sh[tid] + v
        barrier()
        off *= 2
    var run = off0 + sh[tid] - loc
    for q in range(4):
        if base + q < Int(n):
            dst[base + q] = run
            run += flags[base + q]


# ------------------------------------------------------------- float-float fold
def ff_chunk_kernel(mode: Int32, a: FPtr, b: FPtr, c: FPtr, n: Int32, oh: FPtr, ol: FPtr, off: Int32):
    """Block q: the fold of chunk q (`post_bodies.ff_lane`, then the tree)
    into oh/ol[off + q]."""
    var sh = stack_allocation[RTPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sl = stack_allocation[RTPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var base = Int(block_idx.x) * (FOLD_LANES * 4)
    var end = min(Int(n), base + FOLD_LANES * 4)
    var v = ff_lane(Int(mode), a, b, c, base, end, tid)
    sh[tid] = v.hi
    sl[tid] = v.lo
    barrier()
    var o = FOLD_LANES // 2
    while o > 0:
        if tid < o:
            var r = ff_add(FF(sh[tid], sl[tid]), FF(sh[tid + o], sl[tid + o]))
            sh[tid] = r.hi
            sl[tid] = r.lo
        barrier()
        o //= 2
    if tid == 0:
        oh[Int(off) + Int(block_idx.x)] = sh[0]
        ol[Int(off) + Int(block_idx.x)] = sl[0]


def ff_cols_chunk_kernel(
    ff_in: Int32, a: FPtr, b: FPtr, n: Int32, col_stride: Int32, row_stride: Int32,
    oh: FPtr, ol: FPtr, nch_out: Int32, ncols: Int32,
):
    """Block f + q * ncols (a 1-D grid of ncols * nch_out blocks, adjacent
    blocks on adjacent columns): the fold of chunk q of column f, whose element t is
    a[f * col_stride + t * row_stride] (`post_bodies.ff_strided_lane`, then
    `ff_chunk_kernel`'s tree), into oh/ol[f * nch_out + q] (lane
    fix-c1-cluster, `DeviceOps.center_cols`; the host twin is
    `post_bodies.ff_col_fold_host`)."""
    var sh = stack_allocation[RTPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sl = stack_allocation[RTPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var f = Int(block_idx.x) % Int(ncols)
    var q = Int(block_idx.x) // Int(ncols)
    var base = q * FOLD_CHUNK
    var end = min(Int(n), base + FOLD_CHUNK)
    var v = ff_strided_lane(
        ff_in != Int32(0), a, b, f * Int(col_stride), Int(row_stride), base, end, tid
    )
    sh[tid] = v.hi
    sl[tid] = v.lo
    barrier()
    var o = FOLD_LANES // 2
    while o > 0:
        if tid < o:
            var r = ff_add(FF(sh[tid], sl[tid]), FF(sh[tid + o], sl[tid + o]))
            sh[tid] = r.hi
            sl[tid] = r.lo
        barrier()
        o //= 2
    if tid == 0:
        oh[f * Int(nch_out) + q] = sh[0]
        ol[f * Int(nch_out) + q] = sl[0]


def ff_col_mean_kernel(th: FPtr, tl: FPtr, d: Int32, n: Int32, mean: FPtr):
    """mean[f] = `post_bodies.ff_mean_cell` of column f's folded sum."""
    var f = _tid()
    if f < Int(d):
        mean[f] = ff_mean_cell(th[f], tl[f], Int(n))


def center_cols_kernel(x: FPtr, total: Int64, d: Int32, mean: FPtr, dst: FPtr):
    """dst[i] = `post_bodies.center_cell`(x[i], mean[i % d]), i < total.
    `total` is Int64: a kernel argument cannot be Int (not DevicePassable)."""
    var i = _tid()
    if i < Int(total):
        dst[i] = center_cell(x[i], mean[i % Int(d)])


def kpp_search_kernel(mode: Int32, a: FPtr, b: FPtr, th: FPtr, tl: FPtr, n: Int32, v: FPtr, nt: Int32, ids: IPtr):
    var t = _tid()
    if t < Int(nt):
        ids[t] = Int32(kpp_search_cell(Int(mode), a, b, th, tl, Int(n), v[2 * t], v[2 * t + 1]))


def kpp_take_kernel(dc: FPtr, closest: FPtr, m: Int32):
    var j = _tid()
    if j < Int(m):
        var v = dc[j]
        if v < closest[j]:
            closest[j] = v


# ------------------------------------------------------------- small maps
def nonneg_kernel(x: FPtr, n: Int32, bad: IPtr):
    var t = _tid()
    if t < Int(n) and not (x[t] >= Float32(0)):
        bad[0] = 1


def negate_kernel(src: FPtr, dst: FPtr, n: Int32):
    var t = _tid()
    if t < Int(n):
        dst[t] = -src[t]


def count_neg_kernel(x: FPtr, n: Int32, cnt: IPtr):
    var t = _tid()
    if t < Int(n) and x[t] < Float32(0):
        _ = Atomic.fetch_add(cnt, Int32(1))


def sign_side_kernel(src: FPtr, n: Int32, neg: Int32, dst: FPtr):
    var t = _tid()
    if t < Int(n):
        var inf = Float32.MAX * Float32(2)
        var v = src[t]
        if neg != 0:
            dst[t] = -v if v < Float32(0) else inf
        else:
            dst[t] = inf if v < Float32(0) else v


def set_diag_kernel(s: FPtr, v: FPtr, n: Int32):
    var i = _tid()
    if i < Int(n):
        s[i * Int(n) + i] = v[i]


def ap_equal_kernel(s: FPtr, pref: FPtr, n: Int32, bad: IPtr):
    var t = _tid()
    var N = Int(n)
    if t < N * N:
        var i = t // N
        var j = t % N
        if N >= 2 and i != j and s[t] != s[1]:
            bad[0] = 1
    if t >= 1 and t < N and pref[t] != pref[0]:
        bad[0] = 1


def onehot_kernel(idx: IPtr, m: Int32, kc: Int32, by_row: Int32, dst: FPtr):
    var q = _tid()
    if q < Int(m):
        if by_row != 0:
            dst[q * Int(kc) + Int(idx[q])] = Float32(1)
        else:
            dst[Int(idx[q]) * Int(kc) + q] = Float32(1)


def rand_resp_kernel(dst: FPtr, n: Int32, kc: Int32, state: UInt64):
    var i = _tid()
    if i < Int(n):
        rand_resp_row(state, Int(kc), dst, i)


def ms_noise_kernel(labels: IPtr, dist: FPtr, n: Int32, bw: Float32):
    var r = _tid()
    if r < Int(n) and not (dist[r] <= bw):
        labels[r] = Int32(-1)


def max_part_kernel(v: IPtr, n: Int32, part: IPtr):
    """part[b] = the max of block b's RTPB values (0 when none)."""
    var red = stack_allocation[RTPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var t = Int(block_idx.x) * RTPB + tid
    red[tid] = v[t] if t < Int(n) else Int32(0)
    barrier()
    var off = RTPB // 2
    while off > 0:
        if tid < off:
            red[tid] = max(red[tid], red[tid + off])
        barrier()
        off //= 2
    if tid == 0:
        part[Int(block_idx.x)] = red[0]


# ------------------------------------------------------------- OPTICS
def optics_init_kernel(core: FPtr, n: Int32, max_eps: Float32, reach: FPtr, pred: IPtr, done: IPtr):
    var i = _tid()
    if i < Int(n):
        var inf = Float32.MAX * Float32(2)
        if core[i] > max_eps:
            core[i] = inf
        reach[i] = inf
        pred[i] = -1
        done[i] = 0


def optics_part_kernel(reach: FPtr, done: IPtr, n: Int32, part: UPtr):
    """part[b] = the min key of (reach[i], i) over block b's unprocessed rows."""
    var red = stack_allocation[RTPB, UInt64, address_space = AddressSpace.SHARED]()
    var mine = KEY_NONE
    var base = Int(block_idx.x) * SCAN_PER
    var end = min(base + SCAN_PER, Int(n))
    for i in range(base + Int(thread_idx.x), end, RTPB):
        if done[i] == 0:
            mine = min(mine, order_key(reach[i], i))
    var r = _block_min_u64(red, mine)
    if thread_idx.x == 0:
        part[Int(block_idx.x)] = r


def optics_step_kernel(
    part: UPtr, nb: Int32, dist: FPtr, core: FPtr, n: Int32, max_eps: Float32, done: IPtr, reach: FPtr,
    pred: IPtr, ordering: IPtr, step: Int32,
):
    """The step's point from the partials (every thread reads them), booked
    by thread 0; every row relaxes against it."""
    var o = _tid()
    var r = KEY_NONE
    for q in range(Int(nb)):
        r = min(r, part[q])
    if r == KEY_NONE:
        return
    var point = Int(UInt32(r & UInt64(0xFFFFFFFF)))
    if o == 0:
        done[point] = 1
        ordering[Int(step)] = Int32(point)
    if o >= Int(n):
        return
    var cp = core[point]
    if cp != Float32.MAX * Float32(2):
        optics_relax_cell(dist, Int(n), point, cp, max_eps, done, reach, pred, o)


#: fam-cluster (2026-10-04), IDENTICAL: OPTICS' ordering loop as ONE launch a
#: step (`optics_fused_kernel`: pick from the previous launch's partial keys,
#: relax, scan for the next step's partial keys) in place of two
#: (`optics_part_kernel` then `optics_step_kernel`): n + 2 launches a fit
#: for 2n + 1. The keys (`order_key`), the relaxation (`optics_relax_cell`)
#: and the set each step scans are the two kernels' own, so the picks, the
#: reachabilities and the predecessors are the same words.
#: `-D MOJOLEARN_IDN_OPTICS_FUSED_STEP_OFF=1` restores the two launches.
comptime IDN_OPTICS_FUSED_STEP = C41_FUSED_MINIMA or (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_OPTICS_FUSED_STEP_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)


def optics_fused_kernel(
    part: UPtr, nb: Int32, dist: FPtr, core: FPtr, n: Int32, max_eps: Float32, done: IPtr, reach: FPtr,
    pred: IPtr, ordering: IPtr, step: Int32,
):
    """Step `step` of the ordering in one launch of `nb` blocks of RTPB
    threads, block b owning rows `[b * SCAN_PER, (b + 1) * SCAN_PER)` (each
    thread a stride of RTPB through them). `step >= 0`: the step's point is
    the min of the `nb` partial keys the previous launch left in half
    `step & 1` of `part`; its owner marks it done, block 0 books it, every
    other unprocessed row relaxes against it. Then (every step, and the
    scan-only `step == -1`) the block's min key over its unprocessed rows
    goes to half `(step + 1) & 1`, which no block reads in this launch."""
    var red = stack_allocation[RTPB, UInt64, address_space = AddressSpace.SHARED]()
    var NB = Int(nb)
    var N = Int(n)
    var s = Int(step)
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var base = b * SCAN_PER
    var end = min(base + SCAN_PER, N)
    comptime if C41_FUSED_MINIMA:
        var r = KEY_NONE
        if s >= 0:
            for q in range(NB):
                r = min(r,part[(s & 1)*NB+q])
        var point = Int(UInt32(r & UInt64(0xFFFFFFFF))) if r != KEY_NONE else -1
        if b == 0 and t == 0 and point >= 0:
            ordering[s] = Int32(point)
        var mine = KEY_NONE
        for o in range(base+t,end,RTPB):
            if point == o:
                done[o] = 1
            elif point >= 0 and core[point] != Float32.MAX*Float32(2):
                optics_relax_cell(dist,N,point,core[point],max_eps,done,reach,pred,o)
            if done[o] == 0:
                mine = min(mine,order_key(reach[o],o))
        var rr = _block_min_u64(red,mine)
        if t == 0:
            part[((s+1)&1)*NB+b] = rr
        return
    if s >= 0:
        var r = KEY_NONE
        var src = (s & 1) * NB
        for q in range(NB):
            r = min(r, part[src + q])
        if r != KEY_NONE:
            var point = Int(UInt32(r & UInt64(0xFFFFFFFF)))
            if b == 0 and t == 0:
                ordering[s] = Int32(point)
            var cp = core[point]
            var live = cp != Float32.MAX * Float32(2)
            for o in range(base + t, end, RTPB):
                if o == point:
                    done[o] = 1
                elif live:
                    optics_relax_cell(dist, N, point, cp, max_eps, done, reach, pred, o)
    var mine = KEY_NONE
    for i in range(base + t, end, RTPB):
        if done[i] == 0:
            mine = min(mine, order_key(reach[i], i))
    var rr = _block_min_u64(red, mine)
    if t == 0:
        part[((s + 1) & 1) * NB + b] = rr


def optics_flag_kernel(ordering: IPtr, reach: FPtr, core: FPtr, n: Int32, eps: Float32, flag: IPtr):
    var q = _tid()
    if q < Int(n):
        var p = Int(ordering[q])
        flag[q] = 1 if (reach[p] > eps and core[p] <= eps) else 0


def optics_label_kernel(ordering: IPtr, reach: FPtr, core: FPtr, n: Int32, eps: Float32, flag: IPtr, ex: IPtr, labels: IPtr):
    var q = _tid()
    if q < Int(n):
        var p = Int(ordering[q])
        var lab = ex[q] + flag[q] - 1
        if reach[p] > eps and not (core[p] <= eps):
            lab = -1
        labels[p] = lab


# ------------------------------------------------------------- MeanShift
def bin_key_kernel(x: FPtr, nd: Int32, bin_size: Float32, keys: FPtr):
    var t = _tid()
    if t < Int(nd):
        keys[t] = bin_key(x[t], bin_size)


def first_equal_kernel(a: FPtr, mask: IPtr, use_mask: Int32, d: Int32, n: Int32, rep: IPtr):
    var s = _tid()
    if s < Int(n):
        rep[s] = first_equal_cell(a, mask, use_mask != 0, Int(d), s)


def bin_count_kernel(rep: IPtr, n: Int32, min_freq: Int32, kept: IPtr):
    var u = _tid()
    if u < Int(n):
        var k = Int32(0)
        if Int(rep[u]) == u:
            var c = 0
            for r in range(u, Int(n)):
                if Int(rep[r]) == u:
                    c += 1
            if c >= Int(min_freq):
                k = 1
        kept[u] = k


def bin_scatter_kernel(keys: FPtr, kept: IPtr, pos: IPtr, n: Int32, d: Int32, bin_size: Float32, dst: FPtr):
    var u = _tid()
    if u < Int(n) and kept[u] != 0:
        var p = Int(pos[u])
        for f in range(Int(d)):
            dst[p * Int(d) + f] = bin_value(keys[u * Int(d) + f], bin_size)


def ms_last_kernel(rep: IPtr, inten: IPtr, ns: Int32, rv: IPtr):
    """rv[u] = the intensity of the last seed whose center is u's."""
    var u = _tid()
    if u < Int(ns) and Int(rep[u]) == u:
        var s = Int(ns) - 1
        while s > u and Int(rep[s]) != u:
            s -= 1
        rv[u] = inten[s]


def ms_rank_kernel(c: FPtr, rep: IPtr, rv: IPtr, ns: Int32, d: Int32, dst: FPtr, cnt: IPtr):
    var u = _tid()
    if u < Int(ns) and Int(rep[u]) == u:
        var rank = 0
        for v in range(Int(ns)):
            if v != u and Int(rep[v]) == v and center_greater(c, rv, v, u, Int(d)):
                rank += 1
        for f in range(Int(d)):
            dst[rank * Int(d) + f] = c[u * Int(d) + f]
        _ = Atomic.fetch_add(cnt, Int32(1))


def lowest_part_kernel(und: IPtr, m: Int32, part: IPtr):
    """part[b] = the lowest undecided index of block b (m when none)."""
    var red = stack_allocation[RTPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var mine = m
    var base = Int(block_idx.x) * SCAN_PER
    var end = min(base + SCAN_PER, Int(m))
    for i in range(base + tid, end, RTPB):
        if und[i] != 0 and Int32(i) < mine:
            mine = Int32(i)
    red[tid] = mine
    barrier()
    var off = RTPB // 2
    while off > 0:
        if tid < off:
            red[tid] = min(red[tid], red[tid + off])
        barrier()
        off //= 2
    if tid == 0:
        part[Int(block_idx.x)] = red[0]


def ms_mark_kernel(part: IPtr, nb: Int32, dd: FPtr, m: Int32, bw: Float32, und: IPtr, uni: IPtr):
    var j = _tid()
    var i0 = m
    for q in range(Int(nb)):
        i0 = min(i0, part[q])
    if i0 >= m or j >= Int(m):
        return
    if j == Int(i0):
        uni[j] = 1
        und[j] = 0
    elif und[j] != 0 and dd[Int(i0) * Int(m) + j] <= bw:
        und[j] = 0


def compact_rows_kernel(src: FPtr, flag: IPtr, pos: IPtr, m: Int32, d: Int32, dst: FPtr):
    var i = _tid()
    if i < Int(m) and flag[i] != 0:
        for f in range(Int(d)):
            dst[Int(pos[i]) * Int(d) + f] = src[i * Int(d) + f]


def fill_i_kernel(dst: IPtr, n: Int32, v: Int32):
    var i = _tid()
    if i < Int(n):
        dst[i] = v


# ------------------------------------------------------------- AffinityPropagation
def ap_conv_kernel(e: IPtr, ring: IPtr, n: Int32, ci: Int32, it: Int32, st: IPtr):
    """st[0] += every e; st[1] = 1 when a row's window is mixed."""
    var i = _tid()
    if i < Int(n):
        var C = Int(ci)
        ring[i * C + Int(it) % C] = e[i]
        if e[i] != 0:
            _ = Atomic.fetch_add(st, Int32(1))
        if Int(it) >= C:
            var se = 0
            for q in range(C):
                se += Int(ring[i * C + q])
            if se != C and se != 0:
                st[1] = 1


def ap_exsc_kernel(e: IPtr, rk: IPtr, n: Int32, ex: IPtr):
    var i = _tid()
    if i < Int(n) and e[i] != 0:
        ex[Int(rk[i])] = Int32(i)


@always_inline
def _argmax_ex(s: FPtr, n: Int, i: Int, ex: IPtr, K: Int) -> Int:
    var best = 0
    var bv = s[i * n + Int(ex[0])]
    for q in range(1, K):
        var v = s[i * n + Int(ex[q])]
        if v > bv:
            bv = v
            best = q
    return best


def ap_c_kernel(s: FPtr, ex: IPtr, K: Int32, n: Int32, inv: IPtr, c: IPtr):
    """c[i] = the column of ex nearest row i; an exemplar takes its own."""
    var i = _tid()
    if i < Int(n):
        if inv[i] >= 0:
            c[i] = inv[i]
        else:
            c[i] = Int32(_argmax_ex(s, Int(n), i, ex, Int(K)))


def ap_inv_kernel(ex: IPtr, K: Int32, inv: IPtr):
    var k = _tid()
    if k < Int(K):
        inv[Int(ex[k])] = Int32(k)


def ap_acc_kernel(s: FPtr, c: IPtr, n: Int32, acc: FPtr):
    """acc[j] = the sum over the rows i of j's cluster, ascending, of S[i, j]."""
    var j = _tid()
    if j < Int(n):
        var k = c[j]
        var a = Float32(0)
        for i in range(Int(n)):
            if c[i] == k:
                a = ftz(a + s[i * Int(n) + j])
        acc[j] = a


def ap_best_kernel(acc: FPtr, c: IPtr, n: Int32, ex: IPtr):
    """Block k: ex[k] = the column of cluster k with the highest acc, the
    lowest on a tie."""
    var red = stack_allocation[RTPB, UInt64, address_space = AddressSpace.SHARED]()
    var k = Int32(block_idx.x)
    var mine = KEY_NONE
    for j in range(Int(thread_idx.x), Int(n), RTPB):
        if c[j] == k:
            mine = min(mine, order_key(-acc[j], j))
    var r = _block_min_u64(red, mine)
    if thread_idx.x == 0 and r != KEY_NONE:
        ex[Int(k)] = Int32(Int(UInt32(r & UInt64(0xFFFFFFFF))))


def ap_isc_kernel(ex: IPtr, c: IPtr, n: Int32, isc: IPtr):
    var i = _tid()
    if i < Int(n):
        isc[Int(ex[Int(c[i])])] = 1


def ap_label_kernel(ex: IPtr, c: IPtr, isc: IPtr, rk: IPtr, n: Int32, centers: IPtr, labels: IPtr):
    var i = _tid()
    if i < Int(n):
        if isc[i] != 0:
            centers[Int(rk[i])] = Int32(i)
        labels[i] = rk[Int(ex[Int(c[i])])]


# ------------------------------------------------------------- agglomerative connectivity
def agc_edges_kernel(edges: FPtr, ne: Int32, n: Int32, adj: IPtr, bad: IPtr):
    var t = _tid()
    if t < Int(ne):
        var r = Int(edges[2 * t])
        var c = Int(edges[2 * t + 1])
        var N = Int(n)
        if r < 0 or r >= N or c < 0 or c >= N:
            bad[0] = 1
            return
        if r != c:
            adj[r * N + c] = 1
            adj[c * N + r] = 1


def agc_prop_kernel(adj: IPtr, n: Int32, lab: IPtr, changed: IPtr):
    """Min-label propagation with a shortcut: labels only fall and stay
    labels of the same component, so the fixed point (every vertex labelled
    by its component's lowest vertex) is reached in any interleaving."""
    var v = _tid()
    var N = Int(n)
    if v < N:
        var m = lab[v]
        for u in range(N):
            if adj[v * N + u] != 0:
                var l = lab[u]
                if l < m:
                    m = l
        var j = lab[Int(m)]
        if j < m:
            m = j
        if m < lab[v]:
            lab[v] = m
            changed[0] = 1


def agc_root_kernel(lab: IPtr, n: Int32, isroot: IPtr):
    var v = _tid()
    if v < Int(n):
        isroot[v] = 1 if Int(lab[v]) == v else 0


def agc_comp_kernel(lab: IPtr, cid: IPtr, n: Int32, comp: IPtr):
    var v = _tid()
    if v < Int(n):
        comp[v] = cid[Int(lab[v])]


def agc_start_kernel(isroot: IPtr, cid: IPtr, comp: IPtr, n: Int32, start: IPtr):
    var v = _tid()
    if v < Int(n) and isroot[v] != 0:
        var c = cid[v]
        var s = 0
        for u in range(Int(n)):
            if comp[u] < c:
                s += 1
        start[Int(c)] = Int32(s)


def agc_members_kernel(comp: IPtr, start: IPtr, n: Int32, members: IPtr):
    var a = _tid()
    if a < Int(n):
        var c = comp[a]
        var p = 0
        for u in range(a):
            if comp[u] == c:
                p += 1
        members[Int(start[Int(c)]) + p] = Int32(a)


@always_inline
def _comp_size(start: IPtr, c: Int, C: Int, n: Int) -> Int:
    var e = Int(start[c + 1]) if c + 1 < C else n
    return e - Int(start[c])


def agc_rowbest_kernel(
    dm: FPtr, comp: IPtr, start: IPtr, members: IPtr, n: Int32, C: Int32, c0: Int32, cc: Int32, ward: Int32,
    rowbest: UPtr,
):
    """rowbest[a, j] = the min key of (dissimilarity, a * n + b) over b in
    component c0 + j, for a component below a's (else none)."""
    var t = _tid()
    var N = Int(n)
    var CC = Int(cc)
    if t >= N * CC:
        return
    var a = t // CC
    var cj = Int(c0) + t % CC
    var r = KEY_NONE
    if cj < Int(comp[a]):
        var s0 = Int(start[cj])
        var sz = _comp_size(start, cj, Int(C), N)
        for q in range(sz):
            var b = Int(members[s0 + q])
            var v = dm[a * N + b]
            if ward != 0:
                v = identical_sqrt(v)
            r = min(r, order_key(v, a * N + b))
    rowbest[t] = r


def agc_join_kernel(
    rowbest: UPtr, comp: IPtr, start: IPtr, members: IPtr, n: Int32, C: Int32, c0: Int32, cc: Int32, adj: IPtr
):
    """For component pair (ci, c0 + j), j below ci: the closest pair over the
    rows of ci becomes an edge."""
    var t = _tid()
    var N = Int(n)
    var CC = Int(cc)
    if t >= Int(C) * CC:
        return
    var ci = t // CC
    var jl = t % CC
    if Int(c0) + jl >= ci:
        return
    var s0 = Int(start[ci])
    var sz = _comp_size(start, ci, Int(C), N)
    var r = KEY_NONE
    for q in range(sz):
        var a = Int(members[s0 + q])
        r = min(r, rowbest[a * CC + jl])
    if r == KEY_NONE:
        return
    var low = Int(UInt32(r & UInt64(0xFFFFFFFF)))
    var a = low // N
    var b = low % N
    adj[a * N + b] = 1
    adj[b * N + a] = 1
