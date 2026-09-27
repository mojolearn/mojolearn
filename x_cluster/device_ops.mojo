# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CLUSTER LANE'S DEVICE COLUMN (lane/algos-cluster): `ClusterOps` on a
GPU. Each primitive is one kernel whose thread `t` calls the `x_cluster/
bodies.mojo` body for index `t`; nothing is folded across threads, so no
launch shape can move a bit. Only the GPU binding imports this file."""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from x_cluster.bodies import (
    FPtr,
    IPtr,
    ap_availability_col,
    ap_exemplar_cell,
    ap_responsibility_row,
    kth_smallest_row,
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


def _kth_kernel(m: FPtr, n_rows: Int32, n_cols: Int32, k: Int32, dst: FPtr):
    var t = _tid()
    if t < Int(n_rows):
        kth_smallest_row(m, Int(n_cols), Int(k), dst, t)


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


def _descend_kernel(x: FPtr, n: Int32, d: Int32, centers: FPtr, nodes: IPtr, labels: IPtr):
    var t = _tid()
    if t < Int(n):
        tree_descend(x, Int(d), centers, nodes, labels, t)


@always_inline
def _grid(n: Int) -> Int:
    return (n + TPB - 1) // TPB if n > 0 else 1


struct DeviceOps(ClusterOps):
    var ctx: DeviceContext
    var f: List[DeviceBuffer[DType.float32]]
    var i: List[DeviceBuffer[DType.int32]]

    def __init__(out self) raises:
        self.ctx = DeviceContext()
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
        self.ctx.synchronize()

    def nearest(mut self, a: Int, na: Int, b: Int, nb: Int, d: Int, labels: Int, dist: Int) raises:
        self.ctx.enqueue_function[_nearest_kernel](
            self._fp(a), Int32(na), self._fp(b), Int32(nb), Int32(d), self._ip(labels), self._fp(dist),
            grid_dim=_grid(na), block_dim=TPB,
        )
        self.ctx.synchronize()

    def sqrt(mut self, x: Int, n: Int) raises:
        self.ctx.enqueue_function[_sqrt_kernel](
            self._fp(x), Int32(n), grid_dim=_grid(n), block_dim=TPB,
        )
        self.ctx.synchronize()

    def kth(mut self, m: Int, n_rows: Int, n_cols: Int, k: Int, dst: Int) raises:
        self.ctx.enqueue_function[_kth_kernel](
            self._fp(m), Int32(n_rows), Int32(n_cols), Int32(k), self._fp(dst),
            grid_dim=_grid(n_rows), block_dim=TPB,
        )
        self.ctx.synchronize()

    def meanshift(
        mut self, x: Int, n: Int, d: Int, bw: Float32, stop: Float32, max_iter: Int,
        centers: Int, ns: Int, scratch: Int, intensity: Int, iters: Int,
    ) raises:
        self.ctx.enqueue_function[_meanshift_kernel](
            self._fp(x), Int32(n), Int32(d), bw, stop, Int32(max_iter),
            self._fp(centers), Int32(ns), self._fp(scratch), self._ip(intensity), self._ip(iters),
            grid_dim=_grid(ns), block_dim=TPB,
        )
        self.ctx.synchronize()

    def ap_r(mut self, s: Int, a: Int, r: Int, n: Int, damping: Float32) raises:
        self.ctx.enqueue_function[_ap_r_kernel](
            self._fp(s), self._fp(a), self._fp(r), Int32(n), damping, grid_dim=_grid(n), block_dim=TPB,
        )
        self.ctx.synchronize()

    def ap_a(mut self, r: Int, a: Int, n: Int, damping: Float32) raises:
        self.ctx.enqueue_function[_ap_a_kernel](
            self._fp(r), self._fp(a), Int32(n), damping, grid_dim=_grid(n), block_dim=TPB,
        )
        self.ctx.synchronize()

    def ap_e(mut self, a: Int, r: Int, n: Int, e: Int) raises:
        self.ctx.enqueue_function[_ap_e_kernel](
            self._fp(a), self._fp(r), Int32(n), self._ip(e), grid_dim=_grid(n), block_dim=TPB,
        )
        self.ctx.synchronize()

    def descend(mut self, x: Int, n: Int, d: Int, centers: Int, nodes: Int, labels: Int) raises:
        self.ctx.enqueue_function[_descend_kernel](
            self._fp(x), Int32(n), Int32(d), self._fp(centers), self._ip(nodes), self._ip(labels),
            grid_dim=_grid(n), block_dim=TPB,
        )
        self.ctx.synchronize()

    def kmeans(
        mut self, x: List[Float32], n: Int, d: Int, k: Int, max_iter: Int, tol: Float64,
        seed: UInt64, n_init: Int, init: Int, mut centers: List[Float32], mut labels: List[Int32],
    ) raises -> Float64:
        var xc = x.copy()
        centers = List[Float32](length=k * d, fill=Float32(0))
        var lab = List[UInt32](length=n, fill=UInt32(0))
        var w = List[Float32](length=1, fill=Float32(1))
        var r = kmeans_fit(
            self.ctx, xc.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), n, d, k,
            centers.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            lab.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            w.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), 0,
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
