# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of x_neighbors/graph_par.mojo (lane hr-graph): every
stage's items in ascending t, the same driver, so its bits are the device's
by construction. CPU-only installs and the verification digests only."""
from std.memory import bitcast
from x_neighbors.items import FP, IP
from x_neighbors.host_ops import X_NEIGHBORS_HOST_SABOTAGE
from x_neighbors.graph_par import (
    GA, GExec, Lay, LP, FPU, IPU, LPU, GP_NST, gp_item, pr_drive, lv_drive, pr_drive_csr, lv_drive_csr,
)


struct GraphCpu(GExec):
    """The arenas as host lists; `a` is the caller's array."""
    var lf: List[Float32]
    var li: List[Int32]
    var ll: List[Int64]
    var llay: List[Int64]
    var a: Int

    def __init__(out self, a: Int):
        self.lf = List[Float32](length=1, fill=Float32(0))
        self.li = List[Int32](length=1, fill=Int32(0))
        self.ll = List[Int64](length=1, fill=Int64(0))
        self.llay = List[Int64](length=1, fill=Int64(0))
        self.a = a

    def alloc(mut self, lay: Lay) raises -> GA:
        self.lf = List[Float32](length=max(lay.nf, 1), fill=Float32(0))
        self.li = List[Int32](length=max(lay.ni, 1), fill=Int32(0))
        self.ll = List[Int64](length=max(lay.nl, 1), fill=Int64(0))
        self.llay = lay.off.copy()
        return GA(
            FPU(unsafe_from_address=Int(self.lf.unsafe_ptr())),
            IPU(unsafe_from_address=Int(self.li.unsafe_ptr())),
            LPU(unsafe_from_address=Int(self.ll.unsafe_ptr())),
            LPU(unsafe_from_address=Int(self.llay.unsafe_ptr())),
            FPU(unsafe_from_address=self.a if self.a != 0 else Int(self.lf.unsafe_ptr())),
            0, 0, 0, 0, 0, 0, 0, 0, Float32(0), Float32(0),
        )

    def run(mut self, stage: Int, count: Int, g: GA) raises:
        comptime for s in range(GP_NST):
            if stage == s:
                for t in range(count):
                    gp_item[s](t, g)

    def get_i(mut self, slot: Int, idx: Int) raises -> Int:
        return Int(self.li[Int(self.llay[slot]) + idx])

    def get_f(mut self, slot: Int, idx: Int) raises -> Float32:
        return self.lf[Int(self.llay[slot]) + idx]

    def get_is(mut self, slot: Int, count: Int) raises -> List[Int32]:
        var h = List[Int32](length=max(count, 1), fill=Int32(0))
        var o = Int(self.llay[slot])
        for k in range(count):
            h[k] = self.li[o + k]
        return h^

    def get_fs(mut self, slot: Int, count: Int) raises -> List[Float32]:
        var h = List[Float32](length=max(count, 1), fill=Float32(0))
        var o = Int(self.llay[slot])
        for k in range(count):
            h[k] = self.lf[o + k]
        return h^

    def up_f(mut self, slot: Int, addr: Int, count: Int) raises:
        var src = FP(unsafe_from_address=addr)
        var o = Int(self.llay[slot])
        for k in range(count):
            self.lf[o + k] = src.unsafe_load(k)

    def up_i(mut self, slot: Int, addr: Int, count: Int) raises:
        var src = IP(unsafe_from_address=addr)
        var o = Int(self.llay[slot])
        for k in range(count):
            self.li[o + k] = src.unsafe_load(k)

    def down_f(mut self, slot: Int, addr: Int, count: Int) raises:
        var dst = FP(unsafe_from_address=addr)
        var o = Int(self.llay[slot])
        for k in range(count):
            dst.unsafe_store(k, self.lf[o + k])

    def down_i(mut self, slot: Int, addr: Int, count: Int) raises:
        var dst = IP(unsafe_from_address=addr)
        var o = Int(self.llay[slot])
        for k in range(count):
            dst.unsafe_store(k, self.li[o + k])


def pr_iterate_cpu(
    a: Int, x: Int, p: Int, dw: Int, info: Int,
    n: Int, max_iter: Int, thr: Float64, binary: Int, alpha: Float32,
) raises:
    """The host column of `graph_dev.pr_iterate_gpu`."""
    var ex = GraphCpu(a)
    pr_drive(ex, n, max_iter, thr, binary, alpha, x, p, dw, info)
    _ = ex^
    comptime if X_NEIGHBORS_HOST_SABOTAGE:
        if n > 0:
            var px = FP(unsafe_from_address=x)
            px.unsafe_store(0, px.unsafe_load(0) + Float32(1e-3))


def op_louvain(a: Int, labels: Int, info: Int, n: Int, max_level: Int, resolution: Float32,
               threshold: Float32) raises:
    """The host column of `graph_dev.op_louvain`."""
    var ex = GraphCpu(a)
    lv_drive(ex, n, max_level, resolution, threshold, labels, info)
    _ = ex^
    comptime if X_NEIGHBORS_HOST_SABOTAGE:
        var pi = FP(unsafe_from_address=info)
        pi.unsafe_store(0, pi.unsafe_load(0) + Float32(1e-3))


def op_pr_csr(
    indptr: Int, indices: Int, vals: Int, x: Int, p: Int, dw: Int, info: Int,
    n: Int, nnz: Int, has_vals: Int, max_iter: Int, thr_hi: Int, thr_lo: Int, binary: Int,
    x_uniform: Int, p_uniform: Int, dw_uniform: Int, alpha: Float32,
) raises:
    """The host column of `graph_dev.op_pr_csr` (lane gap-graph)."""
    var ex = GraphCpu(0)
    pr_drive_csr(ex, n, nnz, indptr, indices, vals, has_vals, max_iter,
                 bitcast[DType.float64]((UInt64(thr_hi) << UInt64(32)) | UInt64(thr_lo)), binary, alpha,
                 x, p, dw, x_uniform, p_uniform, dw_uniform, info)
    _ = ex^
    comptime if X_NEIGHBORS_HOST_SABOTAGE:
        if n > 0:
            var px = FP(unsafe_from_address=x)
            px.unsafe_store(0, px.unsafe_load(0) + Float32(1e-3))


def op_louvain_csr(
    indptr: Int, indices: Int, vals: Int, labels: Int, info: Int,
    n: Int, nnz: Int, has_vals: Int, max_level: Int, resolution: Float32, threshold: Float32,
) raises:
    """The host column of `graph_dev.op_louvain_csr` (lane gap-graph)."""
    var ex = GraphCpu(0)
    lv_drive_csr(ex, n, nnz, indptr, indices, vals, has_vals, max_level, resolution, threshold, labels, info)
    _ = ex^
    comptime if X_NEIGHBORS_HOST_SABOTAGE:
        var pi = FP(unsafe_from_address=info)
        pi.unsafe_store(0, pi.unsafe_load(0) + Float32(1e-3))
