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
    ld_drive, ld_drive_csr, sn_drive, pg_drive,
)
from core.host_parallel import host_parallelize

comptime GP_HOST_GRAIN = 4096
"""Items per host task of one launch. A graph item is a few to a few
hundred loads (a row, a cell, a scan block), so 4096 of them outweigh a
pool hop by orders of magnitude while a launch of 10^5 items still spreads
over 25 tasks; a launch below one grain (the scalar stages, small graphs)
runs inline. A cost rule over item counts, not a dataset shape."""


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
        # Lane sc-graph-1 (2026-11-28): the items of one launch run across the
        # host's cores in GP_HOST_GRAIN chunks. graph_par's contract (no item
        # reads what another item of the same launch writes; shared counters
        # are integer atomics) is what lets the device run them at once, so
        # the bits are the ascending loop's. A launch of one chunk stays on
        # the calling thread (no pool hop for a scalar stage).
        comptime for s in range(GP_NST):
            if stage == s:
                var tasks = (count + GP_HOST_GRAIN - 1) // GP_HOST_GRAIN
                if tasks <= 1:
                    for t in range(count):
                        gp_item[s](t, g)
                else:
                    def chunk(k: Int) {imm g, imm count}:
                        var lo = k * GP_HOST_GRAIN
                        var hi = min(lo + GP_HOST_GRAIN, count)
                        for t in range(lo, hi):
                            gp_item[s](t, g)
                    host_parallelize(chunk, tasks)

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


def op_leiden(a: Int, labels: Int, info: Int, n: Int, n_iterations: Int, quality: Int, seed: Int,
              resolution: Float32) raises:
    """The host column of `graph_dev.op_leiden`."""
    var ex = GraphCpu(a)
    ld_drive(ex, n, n_iterations, quality, seed, resolution, labels, info)
    _ = ex^
    comptime if X_NEIGHBORS_HOST_SABOTAGE:
        var pi = FP(unsafe_from_address=info)
        pi.unsafe_store(0, pi.unsafe_load(0) + Float32(1e-3))


def op_leiden_csr(
    indptr: Int, indices: Int, vals: Int, labels: Int, info: Int,
    n: Int, nnz: Int, has_vals: Int, n_iterations: Int, quality: Int, seed: Int, resolution: Float32,
) raises:
    """The host column of `graph_dev.op_leiden_csr`."""
    var ex = GraphCpu(0)
    ld_drive_csr(ex, n, nnz, indptr, indices, vals, has_vals, n_iterations, quality, seed, resolution, labels, info)
    _ = ex^
    comptime if X_NEIGHBORS_HOST_SABOTAGE:
        var pi = FP(unsafe_from_address=info)
        pi.unsafe_store(0, pi.unsafe_load(0) + Float32(1e-3))


def op_snn(
    indptr: Int, indices: Int, out_indptr: Int, out_indices: Int, out_data: Int, info: Int,
    n: Int, nnz: Int, k_uniform: Int, cap: Int, prune: Float32,
) raises:
    """The host column of `graph_dev.op_snn`."""
    var ex = GraphCpu(0)
    sn_drive(ex, n, nnz, indptr, indices, k_uniform, prune, cap, out_indptr, out_indices, out_data, info)
    _ = ex^
    comptime if X_NEIGHBORS_HOST_SABOTAGE:
        var pi = IP(unsafe_from_address=info)
        if Int(pi.unsafe_load(1)) == 0 and Int(pi.unsafe_load(0)) > 0:
            var pd = FP(unsafe_from_address=out_data)
            pd.unsafe_store(0, pd.unsafe_load(0) + Float32(1e-3))


def op_paga(a: Int, labels: Int, conn: Int, tree: Int, sizes: Int, info: Int, n: Int, cap: Int) raises:
    """The host column of `graph_dev.op_paga`."""
    var ex = GraphCpu(a)
    pg_drive(ex, n, 0, False, 0, 0, labels, cap, conn, tree, sizes, info)
    _ = ex^
    comptime if X_NEIGHBORS_HOST_SABOTAGE:
        _pg_host_sabotage(conn, info)


def op_paga_csr(
    indptr: Int, indices: Int, labels: Int, conn: Int, tree: Int, sizes: Int, info: Int,
    n: Int, nnz: Int, cap: Int,
) raises:
    """The host column of `graph_dev.op_paga_csr`."""
    var ex = GraphCpu(0)
    pg_drive(ex, n, nnz, True, indptr, indices, labels, cap, conn, tree, sizes, info)
    _ = ex^
    comptime if X_NEIGHBORS_HOST_SABOTAGE:
        _pg_host_sabotage(conn, info)


def _pg_host_sabotage(conn: Int, info: Int):
    """The family's host negative control on PAGA: the first connectivity
    word moved when the result was written."""
    var pi = IP(unsafe_from_address=info)
    if Int(pi.unsafe_load(1)) == 0 and Int(pi.unsafe_load(0)) > 0:
        var pc = FP(unsafe_from_address=conn)
        pc.unsafe_store(0, pc.unsafe_load(0) + Float32(1e-3))
