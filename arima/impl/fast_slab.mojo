# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Opt-in ARIMA device slab (`-D MOJOLEARN_ARIMA_SLAB`, FAST + Apple only,
lane/apple-fast-w3-arima, 2026-10-04).

THE CAUSE. On Metal every live, separately allocated buffer is made
resident on every command encoder: the host cost of one launch grows about
0.25 us per live allocation (M4 probe 2026-09-25: 26 us with none, 270 us
with 1,000; 1,000 sub-buffer views of ONE allocation cost nothing; memory
note metal-launch-cost-scales-with-live-buffers, core/device_arena.mojo).
The AutoARIMA search round (fast_order_search.mojo) issues about eleven
small launches per order plus one filter per (rd, k) group, every round,
for every order in the group, while each order holds about 32 separate
allocations (OrderOptimizer 18, FastEvalWS x_ext + two ARIMAParams) and the
group's KalmanWorkspace / ARIMAParams / y another ~34: roughly 165 live
buffers, i.e. ~40 us extra host cost on each of ~45 launches per round.
The final fit (order_min_lbfgs, maxiter 1000) has the same shape with ~70.

THE CHANGE. Between `slab_begin` and `slab_end` the ARIMA constructors
(ARIMAParams, KalmanWorkspace, FastEvalWS, OrderOptimizer, the grouped y)
take `create_sub_buffer` views of a few process-global 64 MB chunks (one
float32 list, one int32 list) instead of fresh allocations. Stack
discipline: `slab_end` returns the cursor to the mark of its `slab_begin`.
Reused storage is zeroed by ONE kernel per chunk range at `slab_end`, and a
new chunk is zeroed when created, so every view starts as zero words, as a
fresh Metal allocation does: kernels read the same words they read before.
Everything runs on one in-order queue, so the zeroing is ordered after the
last use of the released views and before the first use of new ones.
Only storage moves; no arithmetic changes, so all bits must be unchanged.
Outside a begin/end window, and in every build without the define, each
helper is exactly `ctx.enqueue_create_buffer`.
"""
from std.ffi import _Global
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

# Candidate (opt-in), not yet measured. Expected bit-identical (storage only).
comptime ARIMA_SLAB = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_ARIMA_SLAB"]()
    and not is_defined["MOJOLEARN_ARIMA_SLAB_OFF"]()
)

#: words per chunk (64 MB of float32 / int32) and view alignment (256 B)
comptime SLAB_CHUNK = 1 << 24
comptime SLAB_ALIGN = 64
comptime SLAB_ZERO_TPB = 256


struct _Slab(Defaultable, Movable):
    var f: List[DeviceBuffer[DType.float32]]
    var i: List[DeviceBuffer[DType.int32]]
    var f_cur: Int
    var f_off: Int
    var i_cur: Int
    var i_off: Int
    var depth: Int

    def __init__(out self):
        self.f = List[DeviceBuffer[DType.float32]]()
        self.i = List[DeviceBuffer[DType.int32]]()
        self.f_cur = 0
        self.f_off = 0
        self.i_cur = 0
        self.i_off = 0
        self.depth = 0


comptime _SLAB = _Global[StorageType=_Slab, name="MojoArimaFastSlab", init_fn=_Slab.__init__]


def _slab_zero_f32_kernel(p: MutPointer[Float32, MutAnyOrigin], n_in: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n_in):
        p[t] = Float32(0.0)


def _slab_zero_i32_kernel(p: MutPointer[Int32, MutAnyOrigin], n_in: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n_in):
        p[t] = Int32(0)


def _zero_f32(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], start: Int, count: Int) raises:
    if count <= 0:
        return
    var v = buf.create_sub_buffer[DType.float32](start, count)
    ctx.enqueue_function[_slab_zero_f32_kernel](
        v.unsafe_ptr(), Int32(count),
        grid_dim=((count + SLAB_ZERO_TPB - 1) // SLAB_ZERO_TPB, 1, 1),
        block_dim=(SLAB_ZERO_TPB, 1, 1),
    )
    _ = v^


def _zero_i32(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], start: Int, count: Int) raises:
    if count <= 0:
        return
    var v = buf.create_sub_buffer[DType.int32](start, count)
    ctx.enqueue_function[_slab_zero_i32_kernel](
        v.unsafe_ptr(), Int32(count),
        grid_dim=((count + SLAB_ZERO_TPB - 1) // SLAB_ZERO_TPB, 1, 1),
        block_dim=(SLAB_ZERO_TPB, 1, 1),
    )
    _ = v^


def slab_begin() raises -> List[Int]:
    """Open a window: [f_cur, f_off, i_cur, i_off] to hand to `slab_end`,
    or an empty mark (nothing opened) in builds without ARIMA_SLAB."""
    var mark = List[Int]()
    comptime if ARIMA_SLAB:
        var p = _SLAB.get_or_create_ptr()
        mark.append(p[].f_cur)
        mark.append(p[].f_off)
        mark.append(p[].i_cur)
        mark.append(p[].i_off)
        p[].depth += 1
    return mark^


def slab_end(ctx: DeviceContext, mark: List[Int]) raises:
    """Close the window opened with `mark`: zero every word taken since,
    then return the cursor to the mark. The caller has enqueued the last
    use of every view taken in the window (the queue orders the rest)."""
    comptime if ARIMA_SLAB:
        if len(mark) != 4:
            return
        var p = _SLAB.get_or_create_ptr()
        var c = mark[0]
        while c <= p[].f_cur and c < len(p[].f):
            var start = mark[1] if c == mark[0] else 0
            var stop = p[].f_off if c == p[].f_cur else SLAB_CHUNK
            _zero_f32(ctx, p[].f[c], start, stop - start)
            c += 1
        c = mark[2]
        while c <= p[].i_cur and c < len(p[].i):
            var start = mark[3] if c == mark[2] else 0
            var stop = p[].i_off if c == p[].i_cur else SLAB_CHUNK
            _zero_i32(ctx, p[].i[c], start, stop - start)
            c += 1
        p[].f_cur = mark[0]
        p[].f_off = mark[1]
        p[].i_cur = mark[2]
        p[].i_off = mark[3]
        if p[].depth > 0:
            p[].depth -= 1


def slab_f32(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    """`n` float32 words: a zeroed slab view inside a window, else (and in
    every build without ARIMA_SLAB) `ctx.enqueue_create_buffer`."""
    comptime if ARIMA_SLAB:
        var p = _SLAB.get_or_create_ptr()
        var want = n if n > 0 else 1
        var need = ((want + SLAB_ALIGN - 1) // SLAB_ALIGN) * SLAB_ALIGN
        if p[].depth > 0 and need <= SLAB_CHUNK:
            if p[].f_off + need > SLAB_CHUNK:
                p[].f_cur += 1
                p[].f_off = 0
            if p[].f_cur >= len(p[].f):
                p[].f.append(ctx.enqueue_create_buffer[DType.float32](SLAB_CHUNK))
                _zero_f32(ctx, p[].f[len(p[].f) - 1], 0, SLAB_CHUNK)
            var view = p[].f[p[].f_cur].create_sub_buffer[DType.float32](p[].f_off, want)
            p[].f_off += need
            return view^
    return ctx.enqueue_create_buffer[DType.float32](n)


def slab_i32(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.int32]:
    """`n` int32 words, as `slab_f32`."""
    comptime if ARIMA_SLAB:
        var p = _SLAB.get_or_create_ptr()
        var want = n if n > 0 else 1
        var need = ((want + SLAB_ALIGN - 1) // SLAB_ALIGN) * SLAB_ALIGN
        if p[].depth > 0 and need <= SLAB_CHUNK:
            if p[].i_off + need > SLAB_CHUNK:
                p[].i_cur += 1
                p[].i_off = 0
            if p[].i_cur >= len(p[].i):
                p[].i.append(ctx.enqueue_create_buffer[DType.int32](SLAB_CHUNK))
                _zero_i32(ctx, p[].i[len(p[].i) - 1], 0, SLAB_CHUNK)
            var view = p[].i[p[].i_cur].create_sub_buffer[DType.int32](p[].i_off, want)
            p[].i_off += need
            return view^
    return ctx.enqueue_create_buffer[DType.int32](n)
