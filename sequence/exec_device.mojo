# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`DeviceExec`: the lane's operations on the GPU, one thread per element,
over `sequence/ops.mojo::apply`, the body `HostExec` loops over on the CPU."""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from sequence.exec import Exec
from sequence.ops import FP, Args, apply

comptime TPB = 128


def seq_kernel[OP: Int](
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    i0: Int64, i1: Int64, i2: Int64, i3: Int64, i4: Int64, i5: Int64,
    i6: Int64, i7: Int64, i8: Int64, i9: Int64, i10: Int64, i11: Int64,
    f0: Float32, f1: Float32, f2: Float32, f3: Float32,
    f4: Float32, f5: Float32, f6: Float32, f7: Float32,
    n: Int64,
):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n):
        var a = Args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11,
                     Int(i0), Int(i1), Int(i2), Int(i3), Int(i4), Int(i5),
                     Int(i6), Int(i7), Int(i8), Int(i9), Int(i10), Int(i11),
                     f0, f1, f2, f3, f4, f5, f6, f7)
        apply[OP](t, a)


struct DeviceExec(Exec):
    var ctx: DeviceContext
    var bufs: List[DeviceBuffer[DType.float32]]
    var base: List[Int]
    var size: List[Int]

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.base = List[Int]()
        self.size = List[Int]()

    def alloc(mut self, n: Int) raises -> FP:
        var count = n if n > 0 else 1
        var buf = self.ctx.enqueue_create_buffer[DType.float32](count)
        buf.enqueue_fill(Float32(0.0))
        var p = FP(unsafe_from_address=Int(buf.unsafe_ptr()))
        self.base.append(Int(p))
        self.size.append(count)
        self.bufs.append(buf^)
        return p

    def _find(self, p: FP, n: Int) raises -> Tuple[Int, Int]:
        var addr = Int(p)
        for i in range(len(self.base)):
            var off = (addr - self.base[i]) // 4
            if addr >= self.base[i] and off + n <= self.size[i]:
                return (i, off)
        raise Error("sequence: a copy names no buffer this device Exec allocated")

    def upload(mut self, dst: FP, src: FP, n: Int) raises:
        if n <= 0:
            return
        var found = self._find(dst, n)
        var host = self.ctx.enqueue_create_host_buffer[DType.float32](n)
        for i in range(n):
            host.unsafe_ptr().unsafe_store(i, src.unsafe_load(i))
        var view = self.bufs[found[0]].create_sub_buffer[DType.float32](found[1], n)
        self.ctx.enqueue_copy(dst_buf=view, src_ptr=host.unsafe_ptr())
        self.ctx.synchronize()
        _ = view^
        _ = host^

    def download(mut self, dst: FP, src: FP, n: Int) raises:
        if n <= 0:
            return
        var found = self._find(src, n)
        var host = self.ctx.enqueue_create_host_buffer[DType.float32](n)
        var view = self.bufs[found[0]].create_sub_buffer[DType.float32](found[1], n)
        self.ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
        self.ctx.synchronize()
        for i in range(n):
            dst.unsafe_store(i, host.unsafe_ptr().unsafe_load(i))
        _ = view^
        _ = host^

    def launch[OP: Int](mut self, a: Args, n: Int) raises:
        if n <= 0:
            return
        self.ctx.enqueue_function[seq_kernel[OP]](
            a.p0, a.p1, a.p2, a.p3, a.p4, a.p5, a.p6, a.p7, a.p8, a.p9, a.p10, a.p11,
            Int64(a.i0), Int64(a.i1), Int64(a.i2), Int64(a.i3), Int64(a.i4), Int64(a.i5),
            Int64(a.i6), Int64(a.i7), Int64(a.i8), Int64(a.i9), Int64(a.i10), Int64(a.i11),
            a.f0, a.f1, a.f2, a.f3, a.f4, a.f5, a.f6, a.f7,
            Int64(n),
            grid_dim=((n + TPB - 1) // TPB, 1, 1),
            block_dim=(TPB, 1, 1),
        )

    def sync(mut self) raises:
        self.ctx.synchronize()
