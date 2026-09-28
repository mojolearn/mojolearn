# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CLUSTER LANE'S DEVICE COLUMN (lane/algos-cluster): `ClusterOps` on a
GPU. Each primitive is one kernel whose thread `t` calls the `x_cluster/
bodies.mojo` body for index `t`; nothing is folded across threads, so no
launch shape can move a bit. Only the GPU binding imports this file."""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.ffi import _Global
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz

from x_cluster.bodies import (
    FPtr,
    IPtr,
    cov_cell,
    chain_add,
    cov_final,
    cov_term,
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
)
from cluster.estimator import kmeans_fit
from cluster.impl.kmeans_params import METRIC_L2_EXPANDED
from x_cluster.ops import ClusterOps

comptime TPB = 128


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _sqdist_kernel(a: FPtr, na: Int32, b: FPtr, nb: Int32, d: Int32, dst: FPtr):
    var t = _tid()
    if t < Int(na) * Int(nb):
        sqdist_cell(a, Int(na), b, Int(nb), Int(d), dst, t)


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


def _ap_a_kernel(r: FPtr, a: FPtr, n: Int32, damping: Float32):
    var t = _tid()
    if t < Int(n):
        ap_availability_col(r, a, Int(n), damping, t)


def _ap_noise_kernel(s: FPtr, m: Int32, seed: UInt64):
    # m as Int32: Int is not DevicePassable (the kernel did not instantiate)
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
# order: bits move, quality is the paired check's (progress file).
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


def _pdist_kernel(a: FPtr, na: Int32, b: FPtr, nb: Int32, d: Int32, metric: Int32, p: Float32, dst: FPtr):
    var t = _tid()
    if t < Int(na) * Int(nb):
        pdist_cell(a, Int(na), b, Int(nb), Int(d), Int(metric), p, dst, t)


def _descend_kernel(x: FPtr, n: Int32, d: Int32, centers: FPtr, nodes: IPtr, labels: IPtr):
    var t = _tid()
    if t < Int(n):
        tree_descend(x, Int(d), centers, nodes, labels, t)


@always_inline
def _grid(n: Int) -> Int:
    return (n + TPB - 1) // TPB if n > 0 else 1


struct _ClusterContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext for every x_cluster entry (the
    x_cnn `_Global` pattern; ALGORITHM_EXPANSION_BRIEFS.md "one
    DeviceContext per process"). A context per call hung the SECOND
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

    def __init__(out self) raises:
        self.ctx = x_cluster_ctx()
        self.f = List[DeviceBuffer[DType.float32]]()
        self.i = List[DeviceBuffer[DType.int32]]()
        self.pend_f = List[List[Float32]]()
        self.pend_i = List[List[Int32]]()
        self.mpart = self.ctx.enqueue_create_buffer[DType.float32](1)
        self.mpart_n = 1

    def __del__(deinit self):
        # the buffers and the pending sources die with this value: drain first
        try:
            self.ctx.synchronize()
        except:
            pass

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
        var n = len(v)
        var buf = self.ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        self._upload(buf, v)
        self.f.append(buf^)
        return len(self.f) - 1

    def put_i(mut self, v: List[Int32]) raises -> Int:
        var n = len(v)
        var buf = self.ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
        self._upload_i(buf, v)
        self.i.append(buf^)
        return len(self.i) - 1

    def zeros(mut self, n: Int) raises -> Int:
        var buf = self.ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        self.ctx.enqueue_memset(buf, Float32(0))
        self.f.append(buf^)
        return len(self.f) - 1

    def zeros_i(mut self, n: Int) raises -> Int:
        var buf = self.ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
        self.ctx.enqueue_memset(buf, Int32(0))
        self.i.append(buf^)
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
        var out = List[Float32]()
        self._enq_get(slot, n, out)
        self._sync()
        return out^

    def get_i(mut self, slot: Int, n: Int) raises -> List[Int32]:
        var out = List[Int32]()
        self._enq_get_i(slot, n, out)
        self._sync()
        return out^

    def gets(mut self, slots: List[Int], ns: List[Int]) raises -> List[List[Float32]]:
        var outs = List[List[Float32]](capacity=len(slots))
        for q in range(len(slots)):
            outs.append(List[Float32](length=ns[q], fill=Float32(0)))
        for q in range(len(slots)):
            if ns[q] > 0:
                var view = self.f[slots[q]].create_sub_buffer[DType.float32](0, ns[q])
                self.ctx.enqueue_copy(dst_ptr=outs[q].unsafe_ptr(), src_buf=view)
        self._sync()
        return outs^

    def get_if(
        mut self, islot: Int, ni: Int, fslot: Int, nf: Int, mut oi: List[Int32], mut of: List[Float32]
    ) raises:
        self._enq_get_i(islot, ni, oi)
        self._enq_get(fslot, nf, of)
        self._sync()

    def set(mut self, slot: Int, v: List[Float32]) raises:
        var n = len(v)
        if n == 0:
            return
        var view = self.f[slot].create_sub_buffer[DType.float32](0, n)
        if n >= _PUT_SYNC_MIN:
            self.ctx.enqueue_copy(dst_buf=view, src_ptr=v.unsafe_ptr())
            self._sync()
            return
        self.pend_f.append(v.copy())
        self.ctx.enqueue_copy(dst_buf=view, src_ptr=self.pend_f[len(self.pend_f) - 1].unsafe_ptr())

    def sqdist(mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, dst: Int) raises:
        self.ctx.enqueue_function[_sqdist_kernel](
            self._fp(a), Int32(na), self._fp(b), Int32(nb), Int32(d), self._fp(dst),
            grid_dim=_grid(na * nb), block_dim=TPB,
        )

    def nearest(mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, labels: Int, dist: Int) raises:
        self.ctx.enqueue_function[_nearest_kernel](
            self._fp(a), Int32(na), self._fp(b), Int32(nb), Int32(d), self._ip(labels), self._fp(dist),
            grid_dim=_grid(na), block_dim=TPB,
        )

    def sqrt(mut self, x: Int, n: Int) raises:
        self.ctx.enqueue_function[_sqrt_kernel](
            self._fp(x), Int32(n), grid_dim=_grid(n), block_dim=TPB,
        )

    def kth(mut self, m: Int, n_rows: Int, n_cols: Int, k: Int, dst: Int) raises:
        if n_rows <= 0:
            return
        self.ctx.enqueue_function[_kth_kernel](
            self._fp(m), Int32(n_rows), Int32(n_cols), Int32(k), self._fp(dst),
            grid_dim=n_rows if n_rows > 0 else 1, block_dim=KTH_TPB,
        )

    def meanshift(
        mut self, x: Int, n: Int, d: Int, bw: Float32, stop: Float32, max_iter: Int,
        centers: Int, ns: Int, scratch: Int, intensity: Int, iters: Int,
    ) raises:
        self.ctx.enqueue_function[_meanshift_kernel](
            self._fp(x), Int32(n), Int32(d), bw, stop, Int32(max_iter),
            self._fp(centers), Int32(ns), self._fp(scratch), self._ip(intensity), self._ip(iters),
            grid_dim=_grid(ns), block_dim=TPB,
        )

    def ap_r(mut self, s: Int, a: Int, r: Int, n: Int, damping: Float32) raises:
        self.ctx.enqueue_function[_ap_r_kernel](
            self._fp(s), self._fp(a), self._fp(r), Int32(n), damping, grid_dim=n if n > 0 else 1, block_dim=AP_TPB,
        )

    def ap_a(mut self, r: Int, a: Int, n: Int, damping: Float32) raises:
        self.ctx.enqueue_function[_ap_a_kernel](
            self._fp(r), self._fp(a), Int32(n), damping, grid_dim=_grid(n), block_dim=TPB,
        )

    def ap_noise(mut self, s: Int, m: Int, seed: UInt64) raises:
        if m <= 0:
            return
        self.ctx.enqueue_function[_ap_noise_kernel](self._fp(s), Int32(m), seed, grid_dim=_grid(m), block_dim=TPB)

    def ap_e(mut self, a: Int, r: Int, n: Int, e: Int) raises:
        self.ctx.enqueue_function[_ap_e_kernel](
            self._fp(a), self._fp(r), Int32(n), self._ip(e), grid_dim=_grid(n), block_dim=TPB,
        )

    def descend(mut self, x: Int, n: Int, d: Int, centers: Int, nodes: Int, labels: Int) raises:
        self.ctx.enqueue_function[_descend_kernel](
            self._fp(x), Int32(n), Int32(d), self._fp(centers), self._ip(nodes), self._ip(labels),
            grid_dim=_grid(n), block_dim=TPB,
        )

    def kmeans(
        mut self, x: List[Float32], n: Int, d: Int, k: Int, max_iter: Int, tol: Float64,
        seed: UInt64, n_init: Int, init: Int, mut centers: List[Float32], mut labels: List[Int32],
        weights: List[Float32] = List[Float32](),
    ) raises -> Float64:
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
        return r.inertia

    def gauss_q(mut self, x: Int, n: Int, d: Int, means: Int, pchol: Int, kc: Int, dst: Int) raises:
        self.ctx.enqueue_function[_gauss_q_kernel](
            self._fp(x), Int32(n), Int32(d), self._fp(means), self._fp(pchol), Int32(kc), self._fp(dst),
            grid_dim=_grid(n * kc), block_dim=TPB,
        )

    def resp(mut self, q: Int, c: Int, n: Int, kc: Int, lpn: Int) raises:
        self.ctx.enqueue_function[_resp_kernel](
            self._fp(q), self._fp(c), Int32(n), Int32(kc), self._fp(lpn), grid_dim=_grid(n), block_dim=TPB,
        )

    def exp(mut self, src: Int, dst: Int, n: Int) raises:
        self.ctx.enqueue_function[_exp_kernel](
            self._fp(src), self._fp(dst), Int32(n), grid_dim=_grid(n), block_dim=TPB,
        )

    def moments(
        mut self, resp: Int, x: Int, n: Int, d: Int, kc: Int, reg: Float32, nk: Int, means: Int, cov: Int
    ) raises:
        if kc <= 0:
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

    def pdist(
        mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, metric: Int, p: Float32, dst: Int
    ) raises:
        self.ctx.enqueue_function[_pdist_kernel](
            self._fp(a), Int32(na), self._fp(b), Int32(nb), Int32(d), Int32(metric), p, self._fp(dst),
            grid_dim=_grid(na * nb), block_dim=TPB,
        )
