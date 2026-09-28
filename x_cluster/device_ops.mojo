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
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

from x_cluster.bodies import (
    FPtr,
    IPtr,
    cov_cell,
    exp_cell,
    gauss_q_cell,
    nk_cell,
    pdist_cell,
    resp_row,
    xk_cell,
    ap_availability_col,
    ap_exemplar_cell,
    ap_responsibility_row,
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


def _ap_r_kernel(s: FPtr, a: FPtr, r: FPtr, n: Int32, damping: Float32):
    var t = _tid()
    if t < Int(n):
        ap_responsibility_row(s, a, r, Int(n), damping, t)


def _ap_a_kernel(r: FPtr, a: FPtr, n: Int32, damping: Float32):
    var t = _tid()
    if t < Int(n):
        ap_availability_col(r, a, Int(n), damping, t)


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


struct DeviceOps(ClusterOps):
    """Kernels are only ENQUEUED: the stream runs them in order, and the one
    synchronize is where the host reads (`get`, `get_i`) or where a host
    List is the copy's source (`put`, `set`). A sync per kernel cost a
    host round trip each (on Metal the dominant cost, 4 ms per sync)."""
    var ctx: DeviceContext
    var f: List[DeviceBuffer[DType.float32]]
    var i: List[DeviceBuffer[DType.int32]]

    def __init__(out self) raises:
        self.ctx = x_cluster_ctx()
        self.f = List[DeviceBuffer[DType.float32]]()
        self.i = List[DeviceBuffer[DType.int32]]()

    def _fp(mut self, slot: Int) -> FPtr:
        return self.f[slot].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    def _ip(mut self, slot: Int) -> IPtr:
        return self.i[slot].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    def put(mut self, v: List[Float32]) raises -> Int:
        var n = len(v)
        var buf = self.ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        if n > 0:
            self.ctx.enqueue_copy(dst_buf=buf, src_ptr=v.unsafe_ptr())
        self.ctx.synchronize()
        self.f.append(buf^)
        return len(self.f) - 1

    def put_i(mut self, v: List[Int32]) raises -> Int:
        var n = len(v)
        var buf = self.ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
        if n > 0:
            self.ctx.enqueue_copy(dst_buf=buf, src_ptr=v.unsafe_ptr())
        self.ctx.synchronize()
        self.i.append(buf^)
        return len(self.i) - 1

    def zeros(mut self, n: Int) raises -> Int:
        return self.put(List[Float32](length=n, fill=Float32(0)))

    def zeros_i(mut self, n: Int) raises -> Int:
        return self.put_i(List[Int32](length=n, fill=Int32(0)))

    def get(mut self, slot: Int, n: Int) raises -> List[Float32]:
        var out = List[Float32](length=n, fill=Float32(0))
        if n > 0:
            var view = self.f[slot].create_sub_buffer[DType.float32](0, n)
            self.ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=view)
        self.ctx.synchronize()
        return out^

    def get_i(mut self, slot: Int, n: Int) raises -> List[Int32]:
        var out = List[Int32](length=n, fill=Int32(0))
        if n > 0:
            var view = self.i[slot].create_sub_buffer[DType.int32](0, n)
            self.ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=view)
        self.ctx.synchronize()
        return out^

    def set(mut self, slot: Int, v: List[Float32]) raises:
        var n = len(v)
        if n > 0:
            var view = self.f[slot].create_sub_buffer[DType.float32](0, n)
            self.ctx.enqueue_copy(dst_buf=view, src_ptr=v.unsafe_ptr())
        self.ctx.synchronize()

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
            self._fp(s), self._fp(a), self._fp(r), Int32(n), damping, grid_dim=_grid(n), block_dim=TPB,
        )

    def ap_a(mut self, r: Int, a: Int, n: Int, damping: Float32) raises:
        self.ctx.enqueue_function[_ap_a_kernel](
            self._fp(r), self._fp(a), Int32(n), damping, grid_dim=_grid(n), block_dim=TPB,
        )

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
