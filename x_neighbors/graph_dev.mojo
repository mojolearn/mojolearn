# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device column of x_neighbors/graph_par.mojo (lane hr-graph): every
stage's items as threads of one launch, the arenas resident on the device,
launches in order on the lane's one stream. Only the driver's control
values (counts, flags, the convergence sum, the modularity) cross back.

A/B arms during measurement (deleted with the old routes):
  -D MOJOLEARN_XN_LOUVAIN_OLD_WALK  Louvain: the old sequential sparse walk
  -D MOJOLEARN_XN_PR_OLD_SCAN       PageRank: the old CSR scan (in
                                    iter_device.op_pr_iterate_sparse)
"""
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from x_neighbors.items import FP, IP
from x_neighbors.device_ops import xn_ctx, _buf, _grid, _tid, BLOCK
from x_neighbors.louvain_sparse import louvain_item_sparse
from x_neighbors.graph_par import GA, GExec, Lay, LP, GP_NST, gp_item, pr_drive, lv_drive


def gp_kernel[S: Int](
    f: FP, i: IP, l: LP, lay: LP, a: FP,
    n0: Int64, n1: Int64, n2: Int64, n3: Int64, n4: Int64, n5: Int64, n6: Int64, n7: Int64,
    x0: Float32, x1: Float32, cnt: Int64,
):
    var t = _tid()
    if t < Int(cnt):
        gp_item[S](t, GA(Int(f), Int(i), Int(l), Int(lay), Int(a), Int(n0), Int(n1), Int(n2), Int(n3), Int(n4), Int(n5), Int(n6), Int(n7), x0, x1))


struct GraphDev(GExec):
    """The arenas on the device; `a` (the dense input) uploaded once."""
    var ctx: DeviceContext
    var bf: DeviceBuffer[DType.float32]
    var bi: DeviceBuffer[DType.int32]
    var bl: DeviceBuffer[DType.int64]
    var blay: DeviceBuffer[DType.int64]
    var ba: DeviceBuffer[DType.float32]
    var off: List[Int64]

    def __init__(out self, a: Int, a_count: Int) raises:
        var ctx = xn_ctx()
        self.bf = ctx.enqueue_create_buffer[DType.float32](1)
        self.bi = ctx.enqueue_create_buffer[DType.int32](1)
        self.bl = ctx.enqueue_create_buffer[DType.int64](1)
        self.blay = ctx.enqueue_create_buffer[DType.int64](1)
        self.ba = _buf(ctx, a, a_count, a != 0)
        self.off = List[Int64]()
        self.ctx = ctx^

    def _ga(self) -> GA:
        return GA(
            Int(self.bf.unsafe_ptr()),
            Int(self.bi.unsafe_ptr()),
            Int(self.bl.unsafe_ptr()),
            Int(self.blay.unsafe_ptr()),
            Int(self.ba.unsafe_ptr()),
            0, 0, 0, 0, 0, 0, 0, 0, Float32(0), Float32(0),
        )

    def alloc(mut self, lay: Lay) raises -> GA:
        # the previous arenas may still be read by queued launches
        self.ctx.synchronize()
        self.bf = self.ctx.enqueue_create_buffer[DType.float32](max(lay.nf, 1))
        self.bi = self.ctx.enqueue_create_buffer[DType.int32](max(lay.ni, 1))
        self.bl = self.ctx.enqueue_create_buffer[DType.int64](max(lay.nl, 1))
        self.blay = self.ctx.enqueue_create_buffer[DType.int64](max(len(lay.off), 1))
        self.off = lay.off.copy()
        self.ctx.enqueue_copy(dst_buf=self.blay, src_ptr=self.off.unsafe_ptr())
        self.ctx.synchronize()
        return self._ga()

    def run(mut self, stage: Int, count: Int, g: GA) raises:
        if count <= 0:
            return
        comptime for s in range(GP_NST):
            if stage == s:
                self.ctx.enqueue_function[gp_kernel[s]](
                    FP(unsafe_from_address=g.f), IP(unsafe_from_address=g.i), LP(unsafe_from_address=g.l),
                    LP(unsafe_from_address=g.lay), FP(unsafe_from_address=g.a),
                    Int64(g.n0), Int64(g.n1), Int64(g.n2), Int64(g.n3), Int64(g.n4), Int64(g.n5), Int64(g.n6),
                    Int64(g.n7), g.x0, g.x1, Int64(count),
                    grid_dim=_grid(count), block_dim=(BLOCK if count > 1 else 1),
                )

    def get_i(mut self, slot: Int, idx: Int) raises -> Int:
        var h = List[Int32](length=1, fill=Int32(0))
        var sub = self.bi.create_sub_buffer[DType.int32](Int(self.off[slot]) + idx, 1)
        self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=sub)
        self.ctx.synchronize()
        _ = sub^
        return Int(h[0])

    def get_f(mut self, slot: Int, idx: Int) raises -> Float32:
        var h = List[Float32](length=1, fill=Float32(0))
        var sub = self.bf.create_sub_buffer[DType.float32](Int(self.off[slot]) + idx, 1)
        self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=sub)
        self.ctx.synchronize()
        _ = sub^
        return h[0]

    def get_is(mut self, slot: Int, count: Int) raises -> List[Int32]:
        var h = List[Int32](length=max(count, 1), fill=Int32(0))
        if count > 0:
            var sub = self.bi.create_sub_buffer[DType.int32](Int(self.off[slot]), count)
            self.ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=sub)
            self.ctx.synchronize()
            _ = sub^
        return h^

    def up_f(mut self, slot: Int, addr: Int, count: Int) raises:
        if count <= 0:
            return
        var sub = self.bf.create_sub_buffer[DType.float32](Int(self.off[slot]), count)
        self.ctx.enqueue_copy(dst_buf=sub, src_ptr=FP(unsafe_from_address=addr))
        self.ctx.synchronize()
        _ = sub^

    def down_f(mut self, slot: Int, addr: Int, count: Int) raises:
        if count <= 0:
            return
        var sub = self.bf.create_sub_buffer[DType.float32](Int(self.off[slot]), count)
        self.ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=addr), src_buf=sub)
        self.ctx.synchronize()
        _ = sub^

    def down_i(mut self, slot: Int, addr: Int, count: Int) raises:
        if count <= 0:
            return
        var sub = self.bi.create_sub_buffer[DType.int32](Int(self.off[slot]), count)
        self.ctx.enqueue_copy(dst_ptr=IP(unsafe_from_address=addr), src_buf=sub)
        self.ctx.synchronize()
        _ = sub^


def pr_iterate_gpu(
    a: Int, x: Int, p: Int, dw: Int, info: Int,
    n: Int, max_iter: Int, thr: Float64, binary: Int, alpha: Float32,
) raises:
    """PageRank on the device: the column lists built from the dense
    adjacency and the iteration, all on the device (graph_par.pr_drive)."""
    var ex = GraphDev(a, n * n)
    pr_drive(ex, n, max_iter, thr, binary, alpha, x, p, dw, info)
    _ = ex^


def op_louvain(a: Int, labels: Int, info: Int, n: Int, max_level: Int, resolution: Float32,
               threshold: Float32) raises:
    """Louvain on the device (graph_par.lv_drive)."""
    comptime if is_defined["MOJOLEARN_XN_LOUVAIN_OLD_WALK"]():
        louvain_item_sparse(FP(unsafe_from_address=a), IP(unsafe_from_address=labels), FP(unsafe_from_address=info),
                            n, max_level, resolution, threshold)
        return
    var ex = GraphDev(a, n * n)
    lv_drive(ex, n, max_level, resolution, threshold, labels, info)
    _ = ex^
