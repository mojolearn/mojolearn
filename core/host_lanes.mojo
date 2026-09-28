# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the CPU host bindings; product code.
"""Lane-wise respellings of scalar seams for the CPU host paths.

Moved verbatim out of `training/byte_lm_host_kernels.mojo` (lane neural-cpu,
2026-09-28) so the transformer, Mamba and GEMM host paths can use them
without importing the byte LM (whose kernels import the transformer oracle).
`training/byte_lm_host_kernels.mojo` re-exports every name, so its callers and
its checks are unchanged: `training/checks/byte_lm_host_exp_check.mojo`
compares `expf_lanes` and `silu_lanes` with the scalar seams over all 2^32
Float32 bit patterns, and `training/checks/byte_lm_host_kernels_check.mojo`
holds `fmax_fold_span` to the scalar fold.

IDENTICAL ONLY, like their first home: each function is the IDENTICAL seam's
arithmetic. A caller compiled in another tier must keep the scalar seam
(`lanes_are_identical` says which).
"""

from std.math import floor, fma, max, min
from std.memory import bitcast
from std.sys.info import simd_width_of

from std.sys import llvm_intrinsic

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_exp,
    identical_fmax,
    identical_mul,
    identical_silu,
)


comptime HOST_FW = simd_width_of[DType.float32]()
comptime F32V = SIMD[DType.float32, HOST_FW]
comptime U32V = SIMD[DType.uint32, HOST_FW]

#: True when the lanes below equal this build's scalar seams (`ftz`,
#: `identical_exp`, `identical_silu`, `identical_fmax`) bit for bit.
comptime lanes_are_identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


@always_inline
def ftz_lanes(x: F32V) -> F32V:
    """`ftz` on every lane, without a branch.

    Under IDENTICAL, `ftz(x)` is the sign bit alone when the exponent field is
    zero and the mantissa is not, and `x` otherwise. When the exponent field
    and the mantissa are both zero, `x` already IS its sign bit. Selecting the
    sign bit whenever the exponent field is zero is therefore the same map on
    all 2^32 bit patterns."""
    var bits = bitcast[DType.uint32](x)
    var subnormal = (bits & U32V(0x7F800000)).eq(U32V(0))
    return bitcast[DType.float32](subnormal.select(bits & U32V(0x80000000), bits))


@always_inline
def _nan_bits(x: F32V) -> SIMD[DType.bool, HOST_FW]:
    """NaN lanes by bits: magnitude above the infinity's."""
    return (bitcast[DType.uint32](x) & U32V(0x7FFFFFFF)).gt(U32V(0x7F800000))


@always_inline
def expf_lanes(x: F32V) -> F32V:
    """`checks/numerics.mojo::portable_expf` on every lane: the same
    operations on the same constants in the same order, with the scalar's
    three early returns (NaN as is, `+inf` above 88.722835, `+0.0` below
    -87.33655) applied last as masks in the scalar's priority. Special lanes
    run the arithmetic on `+0.0`, so no lane converts a NaN or an out-of-range
    value to an integer.

    The NaN lanes are passed through by an INTEGER select on the input's bits.
    Selected as floats, a signaling NaN came back quieted (0x7f800001 as
    0x7fc00001, measured on the M4), where the scalar returns it untouched.
    `training/checks/byte_lm_host_exp_check.mojo` compares this with the
    scalar over all 2^32 bit patterns."""
    var nan = _nan_bits(x)
    var over = x.gt(F32V(88.722835))
    var under = x.lt(F32V(-87.33655))
    var xs = (nan | over | under).select(F32V(0.0), x)
    # ONE rounding, as the default build fused it and as portable_expf spells it (lane/pinned-mul-contract-free)
    var t = fma(xs, F32V(1.4426950408889634), F32V(0.5))
    var zf = floor(t)
    var r = fma(zf, F32V(-0.693359375), xs)
    r = fma(zf, F32V(2.12194440e-4), r)
    var q = F32V(1.9875691500e-4)
    q = fma(q, r, F32V(1.3981999507e-3))
    q = fma(q, r, F32V(8.3334519073e-3))
    q = fma(q, r, F32V(4.1665795894e-2))
    q = fma(q, r, F32V(1.6666665459e-1))
    q = fma(q, r, F32V(5.0000001201e-1))
    var r2 = r * r
    var y = fma(q, r2, r)
    y = y + F32V(1.0)
    var k = zf.cast[DType.int32]()
    var k1 = k >> SIMD[DType.int32, HOST_FW](1)
    var k2 = k - k1
    var bias = SIMD[DType.int32, HOST_FW](127)
    var shift = SIMD[DType.int32, HOST_FW](23)
    y = y * bitcast[DType.float32](((k1 + bias) << shift).cast[DType.uint32]())
    y = y * bitcast[DType.float32](((k2 + bias) << shift).cast[DType.uint32]())
    y = y.lt(F32V(1.1754943508222875e-38)).select(F32V(0.0), y)
    y = under.select(F32V(0.0), y)
    y = over.select(bitcast[DType.float32](U32V(0x7F800000)), y)
    return bitcast[DType.float32](nan.select(bitcast[DType.uint32](x), bitcast[DType.uint32](y)))


@always_inline
def silu_lanes(x: F32V) -> F32V:
    """`checks/numerics.mojo::portable_siluf` on every lane: NaN as is (by
    an integer select, as in `expf_lanes`), else
    `portable_divf(x, portable_expf(-x) + 1.0)`, whose flushes are the
    unconditional `_ftz_always`, which is `ftz_lanes` on every bit pattern.
    Compared with the scalar over all 2^32 bit patterns by
    `training/checks/byte_lm_host_exp_check.mojo`."""
    var nan = _nan_bits(x)
    var d = expf_lanes(-x) + F32V(1.0)
    var quotient = ftz_lanes(ftz_lanes(x) / ftz_lanes(d))
    return bitcast[DType.float32](nan.select(bitcast[DType.uint32](x), bitcast[DType.uint32](quotient)))


@always_inline
def _order_key_scalar(v: Float32) -> UInt32:
    """`checks/numerics.mojo::_total_order_key`: negative values map to
    `~bits`, others to `bits | 0x80000000`, so integer order is float order."""
    var b = bitcast[DType.uint32](v)
    if (b & UInt32(0x80000000)) != UInt32(0):
        return b ^ UInt32(0xFFFFFFFF)
    return b | UInt32(0x80000000)


@always_inline
def _order_keys(v: F32V) -> U32V:
    """`_order_key_scalar` on every lane."""
    var b = bitcast[DType.uint32](v)
    var negative = (b & U32V(0x80000000)).ne(U32V(0))
    return negative.select(b ^ U32V(0xFFFFFFFF), b | U32V(0x80000000))


def fmax_fold_span(values: List[Float32], base: Int, count: Int) -> Float32:
    """What `identical_fmax` folds to over `values[base : base + count]`, in
    ANY fold shape, for `count >= 1` operands of at least one `fmax`.

    `portable_fmaxf(a, b)` (DEVIATION 825) returns the canonical quiet NaN
    0x7FC00000 when either operand is a NaN, and otherwise the flushed operand
    of the larger `_total_order_key`, the first on equal keys. Equal keys are
    equal bits, so it is exactly commutative and associative over all of
    Float32, which is why the transformer contract (S14) and the loss
    contract (L1) both name this fold's shape free. Any fold of it is
    therefore the canonical NaN if any operand is a NaN, and otherwise the
    flushed operand of the largest key, found here as lanes with no branch on
    the data. A fold of one value that performs no `fmax` is the caller's."""
    var p = values.unsafe_ptr()
    var magnitude = UInt32(0)
    var best = UInt32(0)
    var i = 0
    while i + HOST_FW <= count:
        var v = p.unsafe_load[width=HOST_FW](base + i)
        magnitude = max(magnitude, (bitcast[DType.uint32](v) & U32V(0x7FFFFFFF)).reduce_max())
        best = max(best, _order_keys(ftz_lanes(v)).reduce_max())
        i += HOST_FW
    while i < count:
        var x = p.unsafe_load(base + i)
        magnitude = max(magnitude, bitcast[DType.uint32](x) & UInt32(0x7FFFFFFF))
        best = max(best, _order_key_scalar(ftz(x)))
        i += 1
    if magnitude > UInt32(0x7F800000):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    if (best & UInt32(0x80000000)) != UInt32(0):
        return bitcast[DType.float32](best & UInt32(0x7FFFFFFF))
    return bitcast[DType.float32](best ^ UInt32(0xFFFFFFFF))


# ===========================================================================
# SPAN HELPERS (lane neural-cpu, 2026-09-28): one oracle statement over a
# contiguous span, as lanes under IDENTICAL and as the scalar statement in any
# other tier. Each names the scalar statement it equals.
# ===========================================================================

@always_inline
def pinned_mul_lanes(a: F32V, b: F32V) -> F32V:
    """`pinned_mul_f32` on every lane (the host spelling: the product behind
    an arithmetic fence, so no following add can fuse with it)."""
    return llvm_intrinsic["llvm.arithmetic.fence", F32V, has_side_effect=False](a * b)


def span_scale(src: List[Float32], sb: Int, n: Int, scale: Float32, mut dst: List[Float32], db: Int):
    """`dst[db + j] = ftz(identical_mul(ftz(src[sb + j]), scale))` for j < n."""
    var sp = src.unsafe_ptr()
    var dp = dst.unsafe_ptr()
    var j = 0
    comptime if lanes_are_identical:
        var sv = F32V(scale)
        while j + HOST_FW <= n:
            dp.unsafe_store(db + j, ftz_lanes(pinned_mul_lanes(ftz_lanes(sp.unsafe_load[width=HOST_FW](sb + j)), sv)))
            j += HOST_FW
    while j < n:
        dp.unsafe_store(db + j, ftz(identical_mul(ftz(sp.unsafe_load(sb + j)), scale)))
        j += 1


def span_exp_shift(src: List[Float32], sb: Int, n: Int, mx: Float32, mut dst: List[Float32], db: Int):
    """`dst[db + j] = ftz(identical_exp(ftz(ftz(src[sb + j]) - ftz(mx))))`."""
    var sp = src.unsafe_ptr()
    var dp = dst.unsafe_ptr()
    var fm = ftz(mx)
    var j = 0
    comptime if lanes_are_identical:
        var mv = F32V(fm)
        while j + HOST_FW <= n:
            dp.unsafe_store(db + j, ftz_lanes(expf_lanes(ftz_lanes(ftz_lanes(sp.unsafe_load[width=HOST_FW](sb + j)) - mv))))
            j += HOST_FW
    while j < n:
        dp.unsafe_store(db + j, ftz(identical_exp(ftz(ftz(sp.unsafe_load(sb + j)) - fm))))
        j += 1


def span_div(src: List[Float32], sb: Int, n: Int, den: Float32, mut dst: List[Float32], db: Int):
    """`dst[db + j] = ftz(identical_div(ftz(src[sb + j]), den))`. Under
    IDENTICAL `identical_div` is `portable_divf`: both operands flushed, ONE
    division, the result flushed."""
    var sp = src.unsafe_ptr()
    var dp = dst.unsafe_ptr()
    var j = 0
    comptime if lanes_are_identical:
        var dv = ftz_lanes(F32V(den))
        while j + HOST_FW <= n:
            dp.unsafe_store(db + j, ftz_lanes(ftz_lanes(sp.unsafe_load[width=HOST_FW](sb + j)) / dv))
            j += HOST_FW
    while j < n:
        dp.unsafe_store(db + j, ftz(identical_div(ftz(sp.unsafe_load(sb + j)), den)))
        j += 1


def span_fmax_fold(values: List[Float32], base: Int, count: Int) -> Float32:
    """The oracle's S14 statement for `count >= 1`:
    `mx = ftz(v[base]); for j in 1..count: mx = identical_fmax(mx, ftz(v[base+j]))`.
    Under IDENTICAL the fold's shape is free (`fmax_fold_span`)."""
    comptime if lanes_are_identical:
        if count >= 2:
            return fmax_fold_span(values, base, count)
    var mx = ftz(values[base])
    for j in range(1, count):
        mx = identical_fmax(mx, ftz(values[base + j]))
    return mx


def span_silu(src: List[Float32], sb: Int, n: Int, mut dst: List[Float32], db: Int):
    """`dst[db + j] = ftz(identical_silu(ftz(src[sb + j])))`. Under IDENTICAL
    `identical_silu` is `portable_siluf`, which `silu_lanes` equals on every
    bit pattern."""
    var sp = src.unsafe_ptr()
    var dp = dst.unsafe_ptr()
    var j = 0
    comptime if lanes_are_identical:
        while j + HOST_FW <= n:
            dp.unsafe_store(db + j, ftz_lanes(silu_lanes(ftz_lanes(sp.unsafe_load[width=HOST_FW](sb + j)))))
            j += HOST_FW
    while j < n:
        dp.unsafe_store(db + j, ftz(identical_silu(ftz(sp.unsafe_load(sb + j)))))
        j += 1


def span_mul(a: List[Float32], ab: Int, b: List[Float32], bb: Int, n: Int, mut dst: List[Float32], db: Int):
    """`dst[db + j] = ftz(identical_mul(ftz(a[ab + j]), ftz(b[bb + j])))`."""
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var dp = dst.unsafe_ptr()
    var j = 0
    comptime if lanes_are_identical:
        while j + HOST_FW <= n:
            dp.unsafe_store(db + j, ftz_lanes(pinned_mul_lanes(
                ftz_lanes(ap.unsafe_load[width=HOST_FW](ab + j)), ftz_lanes(bp.unsafe_load[width=HOST_FW](bb + j)))))
            j += HOST_FW
    while j < n:
        dp.unsafe_store(db + j, ftz(identical_mul(ftz(ap.unsafe_load(ab + j)), ftz(bp.unsafe_load(bb + j)))))
        j += 1


def span_add(a: List[Float32], ab: Int, b: List[Float32], bb: Int, n: Int, mut dst: List[Float32], db: Int):
    """`dst[db + j] = ftz(ftz(a[ab + j]) + ftz(b[bb + j]))`."""
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var dp = dst.unsafe_ptr()
    var j = 0
    comptime if lanes_are_identical:
        while j + HOST_FW <= n:
            dp.unsafe_store(db + j, ftz_lanes(
                ftz_lanes(ap.unsafe_load[width=HOST_FW](ab + j)) + ftz_lanes(bp.unsafe_load[width=HOST_FW](bb + j))))
            j += HOST_FW
    while j < n:
        dp.unsafe_store(db + j, ftz(ftz(ap.unsafe_load(ab + j)) + ftz(bp.unsafe_load(bb + j))))
        j += 1
