# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`SimpleVec` / `SimpleDenseMat`: the vector operations the QN solver is written in.

Reference: `cuml/cpp/src/glm/qn/simple_mat/dense.hpp` (cuML `00094f7`):
`ax`, `axpy`, `dot`, `squaredNorm`, `nrmMax`, `nrm2`, `copy_async`, `fill`.
Partial (`assign_gemm` is `core/gemm.mojo`'s business and is called from
`glm_base.mojo` directly; the sparse twin `sparse.hpp` is not implemented). Do
not improve.

THEIR REDUCTIONS ARE FLOAT ATOMICS, AND THAT IS THE WHOLE IDENTITY STORY
------------------------------------------------------------------------
`dot` is `raft::linalg::mapThenSumReduce` (`dense.hpp:288-296`), whose
kernel folds each 256-thread block with `cub::BlockReduce` and then
**`raft::myAtomicAdd`s the block partials into `out`**
(`raft/linalg/detail/map_then_reduce.cuh:33-38`). The order in which
blocks arrive at that atomic is the scheduler's, so `dot(grad, drt)` -- the
number the line search compares against zero and against the Armijo bound
-- is a different float run to run ON ONE GPU, and every branch in
`qn_linesearch.cuh` and `qn_util.cuh` downstream of it is a data-dependent
branch on a non-reproducible bit. cuML accepts that (one backend, no
identity claim). We do not: IDENTITY_PATHS' rule has three moves and a
float atomic is the textbook case of REPLACE.

DEVIATION 547: every reduction here is ONE BLOCK of `STATS_TPB` threads
striding the vector, each partial through `identical_mul_add`, folded by
`core/pinned_reduce.pinned_block_sum` -- `block.sum` under FAST, a
lane-width-independent halving tree under IDENTICAL -- and written once by
thread 0. No atomic, no second block, so the sum is a pure function of the
vector's bits and of `STATS_TPB`, which `lib_block_bounds_a_float_fold`
pins to one value on every column. The vectors these fold are
`n_param = D + fit_intercept` long, so one block is also the right size.
`nrmMax` is a selection and needs no fold pin (`pinned_block_max`, row 30's
reasoning); it is here for the same reason as the others, to be one block
and to read back through one path.

`axpy`'s `a * x + y` is row 9's contraction exactly (`dense.hpp:179`, a
device lambda nvcc contracts by default): `identical_mul_add`. `ax`'s
`a * x` is one rounding. The stores are seams the next kernel reads: `ftz`.

IN-PLACE VARIANTS EXIST BECAUSE MOJO REFUSES ALIASED LAUNCH ARGUMENTS.
`drt.ax(ys / yy, drt)` and `drt.axpy(-alpha, yj, drt)` pass one buffer as
both operand and result; `enqueue_function` rejects the same origin twice,
so `ax_inplace_kernel` / `axpy_inplace_kernel` are the same arithmetic with
one pointer. Same for `squaredNorm = dot(u, u)`.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast

from core.column_stats import STATS_TPB
from core.pinned_reduce import pinned_block_max, pinned_block_sum
from checks.numerics import ftz, identical_mul_add


#: Elementwise launch width for `ax`/`axpy`. SCHEDULING: one thread, one cell.
comptime VEC_ELEM_TPB = 256


def _grid(n: Int) -> Int:
    return (n + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB


def ax_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    a: Float32,
):
    """`this = a * x` (`dense.hpp:162-168`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        out_v.unsafe_store(i, ftz(a * x.unsafe_load(i)))


def ax_inplace_kernel(
    x: MutPointer[Float32, MutAnyOrigin], n_in: Int32, a: Float32
):
    """`x = a * x`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        x.unsafe_store(i, ftz(a * x.unsafe_load(i)))


def axpy_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    a: Float32,
):
    """`this = a * x + y` (`dense.hpp:171-181`), one rounding under IDENTICAL."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        out_v.unsafe_store(
            i, ftz(identical_mul_add(a, x.unsafe_load(i), y.unsafe_load(i)))
        )


def axpy_inplace_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    a: Float32,
):
    """`y = a * x + y`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        y.unsafe_store(
            i, ftz(identical_mul_add(a, x.unsafe_load(i), y.unsafe_load(i)))
        )


def dot_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`dot(u, v)`: ONE block, strided partials, pinned fold. See the module
    docstring for what this replaces. Launch with `grid = 1, block =
    STATS_TPB` and nothing else: the fold's contract is that every thread
    of the block arrives."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var acc = Float32(0.0)
    var i = tid
    while i < n:
        acc = identical_mul_add(u.unsafe_load(i), v.unsafe_load(i), acc)
        i += STATS_TPB
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        out_v.unsafe_store(0, s0)


def dot_self_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`squaredNorm(u) = dot(u, u)`, character for character the kernel
    above with one pointer."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var acc = Float32(0.0)
    var i = tid
    while i < n:
        var x = u.unsafe_load(i)
        acc = identical_mul_add(x, x, acc)
        i += STATS_TPB
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        out_v.unsafe_store(0, s0)


def nrm_max_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`nrmMax(u)`: `max(|u_i|)` seeded at 0 (`dense.hpp:306-316`)."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var acc = Float32(0.0)
    var i = tid
    while i < n:
        var x = abs(u.unsafe_load(i))
        if x > acc:
            acc = x
        i += STATS_TPB
    var m = pinned_block_max[STATS_TPB](acc)
    if tid == 0:
        out_v.unsafe_store(0, m)


# ---------------------------------------------------------------------------
# host wrappers: launch, read the one scalar back, return it
# ---------------------------------------------------------------------------


def read_scalars(
    ctx: DeviceContext,
    mut scalar: DeviceBuffer[DType.float32],
    mut stage: HostBuffer[DType.float32],
    count: Int,
) raises:
    """lane/linear-apple: the first `count` words of `scalar` into `stage`
    behind ONE synchronize. Several reductions enqueued back to back into
    distinct words of one scalar buffer come home together, so a host step
    that needs k device scalars pays one round trip instead of k. The
    values are the same words `_read_scalar` would return one at a time
    (nothing between the launches reads them); only the number of syncs
    changes, which on Metal is the cost (~4 ms per sync with pending work
    on the M4)."""
    var sub = scalar.create_sub_buffer[DType.float32](0, count)
    ctx.enqueue_copy(dst_ptr=stage.unsafe_ptr(), src_buf=sub)
    ctx.synchronize()
    _ = sub^


def _read_scalar(
    ctx: DeviceContext, mut scalar: DeviceBuffer[DType.float32]
) raises -> Float32:
    var h = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=scalar)
    ctx.synchronize()
    var v = h.unsafe_ptr().unsafe_load(0)
    _ = h^
    return v


def dot(
    ctx: DeviceContext,
    mut u: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32],
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
) raises -> Float32:
    """`dot(u, v, tmp_dev, stream)`, `dense.hpp:288`."""
    ctx.enqueue_function[dot_kernel](
        scalar.unsafe_ptr(), u.unsafe_ptr(), v.unsafe_ptr(), Int32(n),
        grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    return _read_scalar(ctx, scalar)


def squared_norm(
    ctx: DeviceContext,
    mut u: DeviceBuffer[DType.float32],
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
) raises -> Float32:
    """`squaredNorm(u) = dot(u, u)`, `dense.hpp:300`."""
    ctx.enqueue_function[dot_self_kernel](
        scalar.unsafe_ptr(), u.unsafe_ptr(), Int32(n),
        grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    return _read_scalar(ctx, scalar)


def nrm_max(
    ctx: DeviceContext,
    mut u: DeviceBuffer[DType.float32],
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
) raises -> Float32:
    """`nrmMax(u)`, `dense.hpp:306`."""
    ctx.enqueue_function[nrm_max_kernel](
        scalar.unsafe_ptr(), u.unsafe_ptr(), Int32(n),
        grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    return _read_scalar(ctx, scalar)


def nrm2(
    ctx: DeviceContext,
    mut u: DeviceBuffer[DType.float32],
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
) raises -> Float32:
    """`nrm2(u) = raft::mySqrt(squaredNorm(u))`, `dense.hpp:318`. The sqrt
    is on the HOST in theirs and here: a Float32 IEEE sqrt, correctly
    rounded on every host (`sqrtss` / `fsqrt`), no libm."""
    from std.math import sqrt

    return sqrt(squared_norm(ctx, u, n, scalar))


def ax(
    ctx: DeviceContext,
    mut out_v: DeviceBuffer[DType.float32],
    a: Float32,
    mut x: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    ctx.enqueue_function[ax_kernel](
        out_v.unsafe_ptr(), x.unsafe_ptr(), Int32(n), a,
        grid_dim=(_grid(n), 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
    )


def ax_inplace(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], a: Float32, n: Int
) raises:
    ctx.enqueue_function[ax_inplace_kernel](
        x.unsafe_ptr(), Int32(n), a,
        grid_dim=(_grid(n), 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
    )


def axpy(
    ctx: DeviceContext,
    mut out_v: DeviceBuffer[DType.float32],
    a: Float32,
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """`out = a * x + y`."""
    ctx.enqueue_function[axpy_kernel](
        out_v.unsafe_ptr(), x.unsafe_ptr(), y.unsafe_ptr(), Int32(n), a,
        grid_dim=(_grid(n), 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
    )


def axpy_inplace(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    a: Float32,
    mut x: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """`y = a * x + y`."""
    ctx.enqueue_function[axpy_inplace_kernel](
        y.unsafe_ptr(), x.unsafe_ptr(), Int32(n), a,
        grid_dim=(_grid(n), 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
    )


def copy_vec(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
) raises:
    """`copy_async`. Buffer to buffer, whole length."""
    ctx.enqueue_copy(dst_buf=dst, src_buf=src)


# ---------------------------------------------------------------------------
# HOST IEEE SCALAR ARITHMETIC ON THE DEVICE (lane/linear-apple, 2026-09-28)
# ---------------------------------------------------------------------------
#
# `qn_util.lbfgs_search_dir` used to bring every two-loop dot home, divide
# and subtract on the HOST, and pass the result back as a launch argument:
# 2 + 2 * min(m, n_vec) synchronizes per L-BFGS iteration. The fused
# two-loop kernel does that scalar arithmetic in the block instead, so it
# must produce the host's IEEE words exactly. Two operations are needed:
#
#   `ieee_div_f32(a, b)`   a / b, a and b finite and NOT subnormal (both are
#                          ftz'd pinned reductions, or 0 for a)
#   `ieee_sub_f32(a, b)`   a - b, any finite a, b
#
# Normal results come from ONE hardware operation, correctly rounded on
# every column (DEVIATION 740's measurement for `/`; `-` is a basic IEEE
# operation everywhere). The only place a GPU can differ from the host is
# a SUBNORMAL result or operand, which a flushing ALU turns into a signed
# zero or reads as one. Those cases are detected BY BITS (an integer test,
# no float compare on a reduction value: the Apple compiler pitfall) and
# recomputed in integers in units of 2^-149, the subnormal quantum, then
# rounded to nearest even exactly as the host rounds.


@always_inline
def _bits32(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


@always_inline
def _expf32(x: Float32) -> Int:
    return Int((_bits32(x) >> 23) & UInt32(0xFF))


@always_inline
def _quanta149(x: Float32) -> UInt64:
    """|x| as an integer multiple of 2^-149, for |x| < 2^-120 (exponent
    field below 30): at most 2^53, exact."""
    var b = _bits32(x)
    var e = Int((b >> 23) & UInt32(0xFF))
    var f = (b & UInt32(0x7FFFFF)).cast[DType.uint64]()
    if e == 0:
        return f
    return (f | UInt64(0x800000)) << UInt64(e - 1)


def _from_quanta(sign: UInt32, mag: UInt64) -> Float32:
    """`sign * mag * 2^-149` rounded to nearest even into a float32 (mag
    below 2^60). mag < 2^24 is exact and its bits ARE the encoding (the
    implicit bit at 2^23 lands in the exponent field)."""
    if mag < UInt64(0x1000000):
        return bitcast[DType.float32](sign | mag.cast[DType.uint32]())
    var shift = 0
    var t = mag
    while t >= UInt64(0x1000000):
        t >>= 1
        shift += 1
    var mant = mag >> UInt64(shift)
    var rem = mag - (mant << UInt64(shift))
    var half = UInt64(1) << UInt64(shift - 1)
    if rem > half or (rem == half and (mant & UInt64(1)) == UInt64(1)):
        mant += 1
    # mant in [2^23, 2^24]; adding shift << 23 carries a 2^24 mant into the
    # next exponent, which is the right word.
    return bitcast[DType.float32](sign | ((UInt32(shift) << 23) + mant.cast[DType.uint32]()))


def ieee_div_f32(a: Float32, b: Float32) -> Float32:
    """The host's `a / b` for finite, non-subnormal `a` and `b` (b != 0)."""
    var q = a / b
    if _expf32(q) != 0 or (_bits32(a) & UInt32(0x7FFFFFFF)) == UInt32(0):
        return q
    # |a / b| < 2^-126 (or flushed to zero): R = round(ma * 2^s / mb)
    var ba = _bits32(a)
    var bb = _bits32(b)
    var sign = (ba ^ bb) & UInt32(0x80000000)
    var ma = ((ba & UInt32(0x7FFFFF)) | UInt32(0x800000)).cast[DType.uint64]()
    var mb = ((bb & UInt32(0x7FFFFF)) | UInt32(0x800000)).cast[DType.uint64]()
    var s = (_expf32(a) - 150) - (_expf32(b) - 150) + 149
    var num = ma
    var den = mb
    if s >= 0:
        if s > 39:
            return q  # not a tiny quotient; unreachable for a tiny q
        num = ma << UInt64(s)
    else:
        if -s > 38:
            return bitcast[DType.float32](sign)  # below half a quantum
        den = mb << UInt64(-s)
    var r = num // den
    var rem = num - r * den
    if rem * 2 > den or (rem * 2 == den and (r & UInt64(1)) == UInt64(1)):
        r += 1
    return bitcast[DType.float32](sign | r.cast[DType.uint32]())


def ieee_sub_f32(a: Float32, b: Float32) -> Float32:
    """The host's `a - b` for finite `a`, `b`."""
    var r = a - b
    var ea = _expf32(a)
    var eb = _expf32(b)
    if not (_expf32(r) == 0 or ea == 0 or eb == 0):
        return r
    # Both below 2^-120: exact in quanta (the difference of two multiples
    # of 2^-149). With one operand at or above 2^-120 the other, being
    # subnormal or zero, is under half an ulp of every neighbour of the big
    # one, so the correctly rounded hardware result stands.
    if ea >= 30 or eb >= 30:
        return r
    var qa = _quanta149(a)
    var qb = _quanta149(b)
    # a - b = a + (-b), as signed magnitudes in quanta
    var sa = (_bits32(a) & UInt32(0x80000000)) != UInt32(0)
    var sb = (_bits32(b) & UInt32(0x80000000)) == UInt32(0)
    var mag: UInt64
    var neg: Bool
    if sa == sb:
        mag = qa + qb
        neg = sa
    elif qa >= qb:
        mag = qa - qb
        neg = sa
    else:
        mag = qb - qa
        neg = sb
    if mag == UInt64(0):
        return r  # an exact zero: the hardware's signed zero is the host's
    return _from_quanta(UInt32(0x80000000) if neg else UInt32(0), mag)
