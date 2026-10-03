# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Per-row probability glue moved out of Python (lane/apple-fast-py2mojo-linear).

One door, `py2mojo_rows(mode, src, dst, params)`, on the GPU bindings (a grid
kernel, one thread per row, grid-stride) and on the CPU bindings (the same
per-row function in a host loop). Every value is a binary64 word of
`checks/soft_f64.mojo` (the Apple GPU has no float64; the words are IEEE's,
so NVIDIA, AMD, Apple and the host agree), each operation the one the
Python line did, in its order, so the outputs keep their bits:

  ROWS_LOG (1): `_log_or_inf` over n float64 (LogisticRegression.
      predict_log_proba): log(p) for p > 0 (`sf64_log`, `_portable_math.log`
      statement for statement), -inf at +-0, NaN otherwise.
  ROWS_SGD_PROBA (2): SGDClassifier.predict_proba from the float32 sigmoids
      (n, k): k == 1 gives [1 - p, p]; k > 1 divides each row by its sum
      (`_portable_math.nsum`, CPython 3.12+'s `sum`), 1 / k when the sum is
      not positive. float32 out (n, max(k, 2)).
  ROWS_LRCV_PROBA (3): LogisticRegressionCV.predict_proba from the float32
      scores (n, k): k == 1 the two-branch sigmoid over the pinned exp; k > 1
      exp(v - max) over the row divided by their nsum. float32 out.
  ROWS_HUBER_OUT (4): HuberRegressor.outliers_, |y - pred| > thr in
      binary64 (y, pred float32; thr = scale_ * epsilon, a float64), uint8
      out (n). params [n, 1, pred address, thr].
  ROWS_SQRT_F32 (5): `fl32(sqrt(w))` over n float32 (LinearRegression's
      sample-weight roots, cuML olsFit's sqrt(w) row scale): one correctly
      rounded binary32 square root (`portable_sqrtf`, subnormals flushed
      to zero). float32 out (n). Lane pyglue-sweep (Oct 3): this was a
      Python comprehension over the rows.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceContext

from checks.numerics import portable_sqrtf
from checks.soft_f64 import (
    SF64_NAN,
    SF64_ONE,
    SF64_SIGN,
    SF64_ZERO,
    sf64_add,
    sf64_div,
    sf64_exp,
    sf64_from_f32,
    sf64_from_int,
    sf64_gt,
    sf64_is_nan,
    sf64_log,
    sf64_lt,
    sf64_neg,
    sf64_sub,
    sf64_to_f32,
)

comptime ROWS_LOG = 1
comptime ROWS_SGD_PROBA = 2
comptime ROWS_LRCV_PROBA = 3
comptime ROWS_HUBER_OUT = 4
comptime ROWS_SQRT_F32 = 5

comptime ROWS_TPB = 256
comptime ROWS_MAX_BLOCKS = 4096

comptime _F32P = MutPointer[Float32, MutAnyOrigin]
comptime _U64P = MutPointer[UInt64, MutAnyOrigin]
comptime _U8P = MutPointer[UInt8, MutAnyOrigin]
comptime _NEG_INF = UInt64(0xFFF0000000000000)


@always_inline
def _abs(a: UInt64) -> UInt64:
    return a & ~SF64_SIGN


@always_inline
def _finite(a: UInt64) -> Bool:
    return ((a >> 52) & 0x7FF) != 0x7FF


@always_inline
def _ge(a: UInt64, b: UInt64) -> Bool:
    """Python's `a >= b` (False when either is NaN)."""
    if sf64_is_nan(a) or sf64_is_nan(b):
        return False
    return not sf64_lt(a, b)


@always_inline
def _is_zero(a: UInt64) -> Bool:
    return _abs(a) == 0


@fieldwise_init
struct NSum(Copyable, Movable, TrivialRegisterPassable):
    """`_portable_math.nsum`'s state: `0 + x0`, then Neumaier's compensated
    fold in order, the compensation added once at the end when nonzero and
    finite."""
    var total: UInt64
    var c: UInt64
    var started: Bool

    @always_inline
    def add(mut self, x: UInt64):
        if not self.started:
            self.total = sf64_add(SF64_ZERO, x)
            self.started = True
            return
        var t = sf64_add(self.total, x)
        if _ge(_abs(self.total), _abs(x)):
            self.c = sf64_add(self.c, sf64_add(sf64_sub(self.total, t), x))
        else:
            self.c = sf64_add(self.c, sf64_add(sf64_sub(x, t), self.total))
        self.total = t

    @always_inline
    def result(self) -> UInt64:
        if not self.started:
            return SF64_ZERO
        if not _is_zero(self.c) and _finite(self.c):
            return sf64_add(self.total, self.c)
        return self.total


@always_inline
def log_or_inf(p: UInt64) -> UInt64:
    if sf64_gt(p, SF64_ZERO) and not sf64_is_nan(p):
        return sf64_log(p)
    if _is_zero(p):
        return _NEG_INF
    return SF64_NAN


@always_inline
def rows_one(mode: Int, src32: _F32P, src64: _U64P, k: Int, i: Int, dst32: _F32P, dst64: _U64P,
             src2: _F32P, scal: UInt64, dst8: _U8P):
    """Row (or element, for ROWS_LOG, ROWS_HUBER_OUT and ROWS_SQRT_F32) i."""
    if mode == ROWS_SQRT_F32:
        dst32[i] = portable_sqrtf(src32[i])
        return
    if mode == ROWS_HUBER_OUT:
        var r = _abs(sf64_sub(sf64_from_f32(src32[i]), sf64_from_f32(src2[i])))
        var out_ = sf64_gt(r, scal) and not sf64_is_nan(r) and not sf64_is_nan(scal)
        dst8[i] = UInt8(1) if out_ else UInt8(0)
        return
    if mode == ROWS_LOG:
        dst64[i] = log_or_inf(src64[i])
        return
    if mode == ROWS_SGD_PROBA:
        if k == 1:
            var p = src32[i]
            dst32[2 * i] = sf64_to_f32(sf64_sub(SF64_ONE, sf64_from_f32(p)))
            dst32[2 * i + 1] = p
            return
        var acc = NSum(SF64_ZERO, SF64_ZERO, False)
        for c in range(k):
            acc.add(sf64_from_f32(src32[i * k + c]))
        var s = acc.result()
        if sf64_gt(s, SF64_ZERO) and not sf64_is_nan(s):
            for c in range(k):
                dst32[i * k + c] = sf64_to_f32(sf64_div(sf64_from_f32(src32[i * k + c]), s))
        else:
            var u = sf64_to_f32(sf64_div(SF64_ONE, sf64_from_int(k)))
            for c in range(k):
                dst32[i * k + c] = u
        return
    # ROWS_LRCV_PROBA
    if k == 1:
        var z = sf64_from_f32(src32[i])
        var nonneg = _ge(z, SF64_ZERO)
        var e = sf64_exp(sf64_neg(z) if nonneg else z)
        var den = sf64_add(SF64_ONE, e)
        var p = sf64_div(SF64_ONE, den) if nonneg else sf64_div(e, den)
        dst32[2 * i] = sf64_to_f32(sf64_sub(SF64_ONE, p))
        dst32[2 * i + 1] = sf64_to_f32(p)
        return
    var m = sf64_from_f32(src32[i * k])
    for c in range(1, k):
        var v = sf64_from_f32(src32[i * k + c])
        if sf64_gt(v, m) and not sf64_is_nan(v) and not sf64_is_nan(m):
            m = v
    var acc = NSum(SF64_ZERO, SF64_ZERO, False)
    for c in range(k):
        acc.add(sf64_exp(sf64_sub(sf64_from_f32(src32[i * k + c]), m)))
    var s = acc.result()
    for c in range(k):
        var e = sf64_exp(sf64_sub(sf64_from_f32(src32[i * k + c]), m))
        dst32[i * k + c] = sf64_to_f32(sf64_div(e, s))


def rows_kernel(mode: Int32, src32: _F32P, src64: _U64P, k: Int32, n: Int64, dst32: _F32P, dst64: _U64P,
                src2: _F32P, scal: UInt64, dst8: _U8P):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        rows_one(Int(mode), src32, src64, Int(k), i, dst32, dst64, src2, scal, dst8)
        i += stride


@fieldwise_init
struct RowsCall(Copyable, Movable):
    var mode: Int
    var n: Int
    var k: Int
    var src: Int
    var dst: Int
    var src2: Int
    var scal: UInt64

    def src_count(self) -> Int:
        if self.mode == ROWS_LOG or self.mode == ROWS_HUBER_OUT or self.mode == ROWS_SQRT_F32:
            return self.n
        return self.n * self.k

    def dst_count(self) -> Int:
        if self.mode == ROWS_LOG or self.mode == ROWS_HUBER_OUT or self.mode == ROWS_SQRT_F32:
            return self.n
        return self.n * (2 if self.k == 1 else self.k)


def rows_call_from_python(mode: PythonObject, src_addr: PythonObject, dst_addr: PythonObject,
                          params: PythonObject) raises -> RowsCall:
    var md = Int(py=mode)
    if md < ROWS_LOG or md > ROWS_SQRT_F32:
        raise Error("py2mojo_rows: unknown mode")
    var want = 4 if md == ROWS_HUBER_OUT else 2
    if len(params) != want:
        raise Error("py2mojo_rows: params must contain n, k (and, for HUBER_OUT, the pred address and thr)")
    var src2 = 0
    var scal = UInt64(0)
    if md == ROWS_HUBER_OUT:
        src2 = Int(py=params[2])
        scal = bitcast[DType.uint64](Float64(py=params[3]))
    var call = RowsCall(md, Int(py=params[0]), Int(py=params[1]), Int(py=src_addr), Int(py=dst_addr), src2, scal)
    if call.n < 0 or call.k < 1:
        raise Error("py2mojo_rows: n >= 0 and k >= 1 required")
    if call.n > 0 and (call.src == 0 or call.dst == 0 or (md == ROWS_HUBER_OUT and call.src2 == 0)):
        raise Error("py2mojo_rows: null buffer")
    return call^


def rows_device(ctx: DeviceContext, call: RowsCall) raises:
    if call.n == 0:
        return
    var nsrc = call.src_count()
    var ndst = call.dst_count()
    var wide = call.mode == ROWS_LOG
    var huber = call.mode == ROWS_HUBER_OUT
    var d_s32 = ctx.enqueue_create_buffer[DType.float32](1 if wide else nsrc)
    var d_s64 = ctx.enqueue_create_buffer[DType.uint64](nsrc if wide else 1)
    var d_d32 = ctx.enqueue_create_buffer[DType.float32](1 if wide or huber else ndst)
    var d_d64 = ctx.enqueue_create_buffer[DType.uint64](ndst if wide else 1)
    var d_s2 = ctx.enqueue_create_buffer[DType.float32](nsrc if huber else 1)
    var d_d8 = ctx.enqueue_create_buffer[DType.uint8](ndst if huber else 1)
    if huber:
        ctx.enqueue_copy(dst_buf=d_s2, src_ptr=_F32P(unsafe_from_address=call.src2))
    if wide:
        ctx.enqueue_copy(dst_buf=d_s64, src_ptr=_U64P(unsafe_from_address=call.src))
    else:
        ctx.enqueue_copy(dst_buf=d_s32, src_ptr=_F32P(unsafe_from_address=call.src))
    var blocks = max(1, min(ROWS_MAX_BLOCKS, (call.n + ROWS_TPB - 1) // ROWS_TPB))
    ctx.enqueue_function[rows_kernel](
        Int32(call.mode), d_s32.unsafe_ptr(), d_s64.unsafe_ptr(), Int32(call.k), Int64(call.n),
        d_d32.unsafe_ptr(), d_d64.unsafe_ptr(), d_s2.unsafe_ptr(), call.scal, d_d8.unsafe_ptr(),
        grid_dim=blocks, block_dim=ROWS_TPB,
    )
    if huber:
        ctx.enqueue_copy(dst_ptr=_U8P(unsafe_from_address=call.dst), src_buf=d_d8)
    elif wide:
        ctx.enqueue_copy(dst_ptr=_U64P(unsafe_from_address=call.dst), src_buf=d_d64)
    else:
        ctx.enqueue_copy(dst_ptr=_F32P(unsafe_from_address=call.dst), src_buf=d_d32)
    ctx.synchronize()
    _ = d_s32^
    _ = d_s64^
    _ = d_d32^
    _ = d_d64^
    _ = d_s2^
    _ = d_d8^


def py2mojo_rows_device_binding(
    ctx: DeviceContext, mode: PythonObject, src_addr: PythonObject, dst_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The GPU bindings' `py2mojo_rows`; returns n."""
    var call = rows_call_from_python(mode, src_addr, dst_addr, params)
    with GILReleased(Python()):
        rows_device(ctx, call)
    return PythonObject(call.n)


def rows_host(call: RowsCall):
    var s32 = _F32P(unsafe_from_address=call.src)
    var s64 = _U64P(unsafe_from_address=call.src)
    var d32 = _F32P(unsafe_from_address=call.dst)
    var d64 = _U64P(unsafe_from_address=call.dst)
    var s2 = _F32P(unsafe_from_address=call.src2 if call.src2 != 0 else call.src)
    var d8 = _U8P(unsafe_from_address=call.dst)
    for i in range(call.n):
        rows_one(call.mode, s32, s64, call.k, i, d32, d64, s2, call.scal, d8)


def py2mojo_rows_host_binding(
    mode: PythonObject, src_addr: PythonObject, dst_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """The CPU bindings' `py2mojo_rows` (the same per-row words); returns n."""
    var call = rows_call_from_python(mode, src_addr, dst_addr, params)
    if call.n > 0:
        with GILReleased(Python()):
            rows_host(call)
    return PythonObject(call.n)
