# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`DeviceExec`: the lane's operations on the GPU, one thread per element,
over `sequence/ops.mojo::apply`, the body `HostExec` loops over on the CPU."""
from std.ffi import _Global
from std.memory import bitcast
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

from sequence.exec import Exec
from sequence.dispatch import apply
from sequence.ops import FP, Args

comptime TPB = 128


struct _SeqContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext for every `DeviceExec`. A context
    per binding call exhausts Metal's per-process command queues within one
    fit (memory: METAL QUEUE LIMIT IS PER-PROCESS; the cnn lane's M2 Pro
    finding). Storage is `std.ffi._Global`, one slot per numeric tier so a
    FAST and an IDENTICAL .so in one process never share it."""
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojoXSequenceContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXSequenceContextFast"
comptime X_SEQUENCE_CONTEXT = _Global[StorageType=_SeqContext, name=_CTX_NAME, init_fn=_SeqContext.__init__]


def sequence_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_SEQUENCE_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


@always_inline
def _pack_ii(lo: Int, hi: Int) -> Int64:
    """Two Int32 values in one Int64 word (lo in the low half)."""
    return Int64(Int(UInt32(Int32(lo))) | (Int(UInt32(Int32(hi))) << 32))


@always_inline
def _lo(w: Int64) -> Int:
    return Int(Int32(w & 0xFFFFFFFF))


@always_inline
def _hi(w: Int64) -> Int:
    return Int(Int32((w >> 32) & 0xFFFFFFFF))


@always_inline
def _pack_ff(lo: Float32, hi: Float32) -> Int64:
    """Two floats' bit patterns in one Int64 word (exact)."""
    return Int64(Int(bitcast[DType.uint32](lo)) | (Int(bitcast[DType.uint32](hi)) << 32))


@always_inline
def _flo(w: Int64) -> Float32:
    return bitcast[DType.float32](UInt32(w & 0xFFFFFFFF))


@always_inline
def _fhi(w: Int64) -> Float32:
    return bitcast[DType.float32](UInt32((w >> 32) & 0xFFFFFFFF))


def seq_kernel[OP: Int](
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    i01: Int64, i23: Int64, i45: Int64, i67: Int64, i89: Int64, i1011: Int64,
    f01: Int64, f23: Int64, f45: Int64, f67: Int64,
    n: Int64,
):
    """One thread per element. The twelve integers and eight floats travel
    packed two to an Int64 word (bit-exact): Metal binds every kernel
    argument to its own buffer slot and has 31, so the unpacked 33-argument
    signature failed to compile on Apple."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n):
        var a = Args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11,
                     _lo(i01), _hi(i01), _lo(i23), _hi(i23), _lo(i45), _hi(i45),
                     _lo(i67), _hi(i67), _lo(i89), _hi(i89), _lo(i1011), _hi(i1011),
                     _flo(f01), _fhi(f01), _flo(f23), _fhi(f23),
                     _flo(f45), _fhi(f45), _flo(f67), _fhi(f67))
        apply[OP](t, a)


struct DeviceExec(Exec):
    var ctx: DeviceContext
    var bufs: List[DeviceBuffer[DType.float32]]
    var base: List[Int]
    var size: List[Int]

    def __init__(out self) raises:
        self.ctx = sequence_ctx()
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

    def __deinit__(deinit self):
        # The context outlives this Exec: drain its queue before the buffers
        # go, so no queued kernel reads a freed buffer.
        try:
            self.ctx.synchronize()
        except:
            pass

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

    def bind(mut self, src: FP, n: Int) raises -> FP:
        var p = self.alloc(n)
        self.upload(p, src, n)
        return p

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
        comptime I32_MAX = 2147483647
        for v in [a.i0, a.i1, a.i2, a.i3, a.i4, a.i5, a.i6, a.i7, a.i8, a.i9, a.i10, a.i11]:
            if v > I32_MAX or v < -I32_MAX - 1:
                raise Error("sequence DeviceExec: an integer argument does not fit Int32 (" + String(v) + ")")
        self.ctx.enqueue_function[seq_kernel[OP]](
            a.p0, a.p1, a.p2, a.p3, a.p4, a.p5, a.p6, a.p7, a.p8, a.p9, a.p10, a.p11,
            _pack_ii(a.i0, a.i1), _pack_ii(a.i2, a.i3), _pack_ii(a.i4, a.i5),
            _pack_ii(a.i6, a.i7), _pack_ii(a.i8, a.i9), _pack_ii(a.i10, a.i11),
            _pack_ff(a.f0, a.f1), _pack_ff(a.f2, a.f3), _pack_ff(a.f4, a.f5), _pack_ff(a.f6, a.f7),
            Int64(n),
            grid_dim=((n + TPB - 1) // TPB, 1, 1),
            block_dim=(TPB, 1, 1),
        )

    def sync(mut self) raises:
        self.ctx.synchronize()
