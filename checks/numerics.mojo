# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What may vary between backends, and what may not."""

from std.sys.compile import is_defined



comptime GLOBAL_NUMERIC_MODE = (
    NUMERIC_IDENTICAL if is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]()
    else (
        NUMERIC_DETERMINISTIC
        if is_defined["MOJOLEARN_NUMERIC_DETERMINISTIC"]()
        else NUMERIC_FAST
    )
)


comptime PIN_DETERMINISM = (
    GLOBAL_NUMERIC_MODE == NUMERIC_DETERMINISTIC
    or GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
)

comptime PIN_CROSS_VENDOR = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


def numeric_mode_name() -> String:
    """The build's tier, for every banner that prints one."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_DETERMINISTIC:
        return String("DETERMINISTIC")
    return String("FAST")


comptime NUMERIC_FAST = 0

comptime NUMERIC_IDENTICAL = 1


comptime NUMERIC_DETERMINISTIC = 2


@fieldwise_init
struct NumericMode(Copyable, Movable):
    """The mode, plus the resolved value of every numeric row."""

    var mode: Int

    @staticmethod
    def default() -> Self:
        """`FAST`."""
        return Self(NUMERIC_FAST)

    def deterministic_flush(self) -> Bool:
        """NUMERIC ROW."""
        return PIN_DETERMINISM

    def fixed_replication(self) -> Bool:
        """NUMERIC ROW, and it LOOKS like a scheduling row, which is the whole reason this file exists."""
        return self.mode == NUMERIC_IDENTICAL

    def free_block_shape(self) -> Bool:
        """SCHEDULING ROW."""
        return True




from std.memory import bitcast
from std.sys import llvm_intrinsic
from std.sys.info import is_amd_gpu, is_apple_gpu, is_nvidia_gpu


def ftz(x: Float32) -> Float32:
    """IDENTITY_PATHS row 10's construction: the denormal policy.

    lane/amd-step-time (2026-09-24): in code compiled FOR an AMD GPU the
    same function is spelled as one `v_cmp_class_f32` (mask 0x90, the two
    subnormal classes) and a select of the signed zero, the spelling the GEMM
    seam already ships on AMD (`gemm_identical._ftz_class`, 2026-09-18). It
    returns the same word for every input: a subnormal becomes its signed
    zero, everything else is returned unchanged. Host code and every other
    target keep the integer spelling. MEASURED on an MI300X at the T3 shape:
    lean B4 step 0.896 -> 0.819 s, optimizer step 57.5 -> 52.6 s, the replay
    of steps 101..103 PASS against the H100 chain and 10 identity lanes
    IDENTICAL. `-D MOJOLEARN_FTZ_NO_CLASS=1` is the revert arm."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_FTZ_NO_CLASS"]() and is_amd_gpu():
        var subnormal = llvm_intrinsic[
            "llvm.amdgcn.class.f32", Bool, has_side_effect=False
        ](x, Int32(0x90))
        var zero = bitcast[DType.float32](bitcast[DType.uint32](x) & UInt32(0x80000000))
        return zero if subnormal else x
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var b = bitcast[DType.uint32](x)
        if (b & UInt32(0x7F800000)) == UInt32(0) and (
            b & UInt32(0x007FFFFF)
        ) != UInt32(0):
            return bitcast[DType.float32](b & UInt32(0x80000000))
    return x


def identical_mul_add(a: Float32, b: Float32, c: Float32) -> Float32:
    """IDENTITY_PATHS row 9's construction: the contraction pin."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return _fma_f32(a, b, c)
    return a * b + c


def _fma_f32(a: Float32, b: Float32, c: Float32) -> Float32:
    from std.math import fma

    return fma(a, b, c)


def identical_mul(a: Float32, b: Float32) -> Float32:
    """DEVIATION 826 (2026-08-24): the OTHER half of row 9's contraction pin, and it belongs beside `identical_mul_add` rather than in three other directories.

    Under IDENTICAL it is `pinned_mul_f32`. It used to be `fma(a, b, -0.0)`,
    which LLVM folds into a plain contractable product that a following add
    then fused with (lane/pinned-mul-contract-free, 2026-09-26)."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return pinned_mul_f32(a, b)
    return a * b


def identical_mul64(a: Float64, b: Float64) -> Float64:
    """`identical_mul`'s float64 twin: `pinned_mul_f64` under IDENTICAL, `a * b` otherwise."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return pinned_mul_f64(a, b)
    return a * b


# THE PINNED PRODUCT (IDENTITY_PATHS row 9, lane/pinned-mul-contract-free,
# 2026-09-26). `pinned_mul_f32(a, b)` / `pinned_mul_f64(a, b)` are the
# correctly rounded product `a*b`, bit for bit, as a value no code generator
# can fuse into a neighboring add or subtract, whatever contraction mode the
# build uses. `fma(a, b, -0.0)` is NOT that: LLVM folds it to `fmul contract`
# and the backend then fuses that fmul with the add it feeds. The spelling is
# per target, each one checked in that target's own output (probes in
# tools/contraction/):
#   host CPU (arm64, x86-64) and AMD GPU: `llvm.arithmetic.fence(a * b)`; the
#     fence is a no-op in the machine code and a wall to contraction (arm64
#     `fmul`+`fadd`, gfx942 `v_mul_f32`+`v_add_f32`).
#   NVIDIA: `llvm.nvvm.mul.rn.{f,d}`, i.e. PTX `mul.rn`. A fenced `fmul`
#     reaches PTX as a plain `mul.f32`, which ptxas may still fuse with an
#     `add.f32`; the `.rn` modifier forbids that (PTX ISA, `mul`).
#   Apple GPU: `llvm.fma(a, b, -0.0)` called WITHOUT fast-math flags; LLVM
#     folds it to an fmul without `contract`, which the Metal compiler does
#     not fuse (measured on an M4: 0 of 65536 fused, `a*b + c` 65506). The
#     Metal compiler service crashes on `llvm.arithmetic.fence`.
@always_inline
def pinned_mul_f32(a: Float32, b: Float32) -> Float32:
    """`a*b`, correctly rounded, never fused into a neighbor. See above."""
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.mul.rn.f", Float32, has_side_effect=False](a, b)
    elif is_apple_gpu():
        return llvm_intrinsic["llvm.fma.f32", Float32, has_side_effect=False](
            a, b, Float32(-0.0)
        )
    else:
        return llvm_intrinsic[
            "llvm.arithmetic.fence.f32", Float32, has_side_effect=False
        ](a * b)


@always_inline
def pinned_mul_f64(a: Float64, b: Float64) -> Float64:
    """`pinned_mul_f32`'s float64 twin (no Apple GPU arm: Metal has no float64)."""
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.mul.rn.d", Float64, has_side_effect=False](a, b)
    elif is_apple_gpu():
        return llvm_intrinsic["llvm.fma.f64", Float64, has_side_effect=False](
            a, b, Float64(-0.0)
        )
    else:
        return llvm_intrinsic[
            "llvm.arithmetic.fence.f64", Float64, has_side_effect=False
        ](a * b)




def portable_expf(x: Float32) -> Float32:
    """`expf` as one arithmetic."""
    from std.math import floor
    from std.memory import bitcast

    if x != x:  # NaN propagates untouched
        return x
    if x > Float32(88.722835):
        return bitcast[DType.float32](UInt32(0x7F800000))  # +inf
    if x < Float32(-87.33655):
        return Float32(0.0)

    # ONE rounding, written out: `x * log2(e) + 0.5` is what every default
    # (contract=fast) build fused this pair into, so the explicit fma keeps
    # those bits and makes them independent of the contraction mode
    # (lane/explicit-fma-contract-proof, 2026-09-26).
    var t = _fma_f32(x, Float32(1.4426950408889634), Float32(0.5))
    var zf = floor(t)  # exact
    var k = Int(zf)
    var r = _fma_f32(zf, Float32(-0.693359375), x)
    r = _fma_f32(zf, Float32(2.12194440e-4), r)

    var q = Float32(1.9875691500e-4)
    q = _fma_f32(q, r, Float32(1.3981999507e-3))
    q = _fma_f32(q, r, Float32(8.3334519073e-3))
    q = _fma_f32(q, r, Float32(4.1665795894e-2))
    q = _fma_f32(q, r, Float32(1.6666665459e-1))
    q = _fma_f32(q, r, Float32(5.0000001201e-1))
    var r2 = r * r
    var y = _fma_f32(q, r2, r)
    y = y + Float32(1.0)

    var k1 = k >> 1
    var k2 = k - k1
    y = y * bitcast[DType.float32](UInt32((k1 + 127) << 23))
    # Pinned: every `portable_expf(..) + 1` caller (sigmoid, silu, tanh,
    # softplus) had this exact power-of-two scaling fused into its add by the
    # default build (lane/pinned-mul-contract-free, 2026-09-26).
    y = pinned_mul_f32(y, bitcast[DType.float32](UInt32((k2 + 127) << 23)))
    if y < Float32(1.1754943508222875e-38):
        return Float32(0.0)
    return y


def portable_logf(x_in: Float32) -> Float32:
    """`logf` as one arithmetic."""
    from std.memory import bitcast

    var x = x_in
    if x != x:
        return x
    if abs(x) < Float32(1.1754943508222875e-38):
        x = Float32(0.0)
    if x == Float32(0.0):
        return bitcast[DType.float32](UInt32(0xFF800000))  # -inf
    if x < Float32(0.0):
        return bitcast[DType.float32](UInt32(0x7FC00000))  # quiet NaN
    if x == bitcast[DType.float32](UInt32(0x7F800000)):
        return x  # +inf

    var bits = rebind[UInt32](x.to_bits())
    var e = Int((bits >> 23) & UInt32(0xFF)) - 126
    var m = bitcast[DType.float32](
        (bits & UInt32(0x007FFFFF)) | UInt32(0x3F000000)
    )
    if m < Float32(0.7071067811865476):
        e -= 1
        m = m + m
    var t = m - Float32(1.0)
    return _cephes_logf_core(t, e)


def _cephes_logf_core(t: Float32, e: Int) -> Float32:
    """The Cephes logf mantissa polynomial and exponent re-entry, ONE source for two callers: `portable_logf` (t = m - 1 on [sqrt(1/2), sqrt(2)), e the unbiased exponent) and `portable_log1pf`'s small-x branch (t = x itself, e = 0, DEVIATION 742)."""
    var z = t * t

    var p = Float32(7.0376836292e-2)
    p = _fma_f32(p, t, Float32(-1.1514610310e-1))
    p = _fma_f32(p, t, Float32(1.1676998740e-1))
    p = _fma_f32(p, t, Float32(-1.2420140846e-1))
    p = _fma_f32(p, t, Float32(1.4249322787e-1))
    p = _fma_f32(p, t, Float32(-1.6668057665e-1))
    p = _fma_f32(p, t, Float32(2.0000714765e-1))
    p = _fma_f32(p, t, Float32(-2.4999993993e-1))
    p = _fma_f32(p, t, Float32(3.3333331174e-1))

    var tz = t * z
    var y = tz * p
    var ef = Float32(e)  # exact, |e| <= 150
    y = _fma_f32(ef, Float32(-2.12194440e-4), y)
    y = _fma_f32(Float32(-0.5), z, y)
    var r = t + y
    r = _fma_f32(ef, Float32(0.693359375), r)
    return r


def identical_exp(x: Float32) -> Float32:
    """Row 12's seam call."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_expf(x)
    from std.math import exp

    return exp(x)


def identical_log(x: Float32) -> Float32:
    """Row 12's seam call, `log` side."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_logf(x)
    from std.math import log

    return log(x)




def portable_sqrtf(x_in: Float32) -> Float32:
    """`sqrtf` as one arithmetic, correctly rounded."""
    from std.memory import bitcast

    var x = x_in
    if x != x:
        return x
    if abs(x) < Float32(1.1754943508222875e-38):
        return Float32(0.0)  # +-0 and subnormals, either sign
    if x < Float32(0.0):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    if x == bitcast[DType.float32](UInt32(0x7F800000)):
        return x
    var scaled_down = False
    if x < bitcast[DType.float32](UInt32(0x0F800000)):  # 2^-96
        x = x * bitcast[DType.float32](UInt32(0x5F800000))  # 2^64
        scaled_down = True
    var bits = rebind[UInt32](x.to_bits())
    var y = bitcast[DType.float32]((bits >> 1) + UInt32(0x1FBD1DF5))
    # Pinned halvings: the default build fused each `0.5 * s` into the next
    # step's `y + x / y` (exact in this range, so the same bits, but a
    # contraction choice all the same; lane/pinned-mul-contract-free).
    y = pinned_mul_f32(Float32(0.5), y + x / y)
    y = pinned_mul_f32(Float32(0.5), y + x / y)
    y = pinned_mul_f32(Float32(0.5), y + x / y)
    var yb = rebind[UInt32](y.to_bits())
    var best = y
    var r_best = abs(_fma_f32(-y, y, x))
    for k in range(4):
        var cb: UInt32
        if k == 0:
            cb = yb - UInt32(1)
        elif k == 1:
            cb = yb + UInt32(1)
        elif k == 2:
            cb = yb - UInt32(2)
        else:
            cb = yb + UInt32(2)
        var c = bitcast[DType.float32](cb)
        var r = abs(_fma_f32(-c, c, x))
        if r < r_best:
            best = c
            r_best = r
    if scaled_down:
        best = best * bitcast[DType.float32](UInt32(0x2F800000))  # 2^-32
    return best


def portable_cosf(x_in: Float32) -> Float32:
    """`cosf` as one arithmetic: Cephes single-precision cosf over the Cody-Waite reduction's domain |x| < 8192."""
    from std.memory import bitcast

    if x_in != x_in or abs(x_in) == bitcast[DType.float32](UInt32(0x7F800000)):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    return _cephes_sincosf_core(abs(x_in), Float32(1.0), True)


def _cephes_sincosf_core(
    x_abs: Float32, sign_in: Float32, is_cos: Bool
) -> Float32:
    """Cephes single-precision `sinf`/`cosf` from `single/sinf.c`: the octant computation, the three-part Cody-Waite pi/4 reduction and BOTH polynomials, ONE source for two callers (`portable_cosf`, and `portable_sinf` from DEVIATION 820)."""
    from std.math import floor

    var x = x_abs
    var j = Int(floor(x * Float32(1.27323954473516)))
    var yj = Float32(j)
    if (j & 1) != 0:
        j += 1
        yj = yj + Float32(1.0)
    j = j & 7
    x = _fma_f32(yj, Float32(-0.78515625), x)
    x = _fma_f32(yj, Float32(-2.4187564849853515625e-4), x)
    x = _fma_f32(yj, Float32(-3.77489497744594108e-8), x)
    var sign = sign_in
    if j > 3:
        j -= 4
        sign = -sign
    if is_cos and j > 1:
        sign = -sign
    var z = x * x
    var use_sin_poly = j == 1 or j == 2
    if not is_cos:
        use_sin_poly = not use_sin_poly
    var y: Float32
    if use_sin_poly:
        var p = Float32(-1.9515295891e-4)
        p = _fma_f32(p, z, Float32(8.3321608736e-3))
        p = _fma_f32(p, z, Float32(-1.6666654611e-1))
        y = _fma_f32(x * z, p, x)
    else:
        var p = Float32(2.443315711809948e-5)
        p = _fma_f32(p, z, Float32(-1.388731625493765e-3))
        p = _fma_f32(p, z, Float32(4.166664568298827e-2))
        y = _fma_f32(z * z, p, _fma_f32(Float32(-0.5), z, Float32(1.0)))
    return sign * y


def portable_powf(x: Float32, p: Float32) -> Float32:
    """`powf(x, p)` for x > 0 as `exp(p * log(x))` through the two row-12 functions -- the Bayesian bootstrap's `tmp ** bagging_temperature` (`gbdt/gpu_util/kernel/bootstrap.mojo`) is the consumer, and its `tmp` is `-log(u + 1e-20) > 0`."""
    from std.memory import bitcast

    if p == Float32(0.0):
        return Float32(1.0)
    if x != x or p != p:
        return bitcast[DType.float32](UInt32(0x7FC00000))
    if x < Float32(0.0):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    if x == Float32(0.0):
        if p > Float32(0.0):
            return Float32(0.0)
        return bitcast[DType.float32](UInt32(0x7F800000))
    return portable_expf(p * portable_logf(x))


def portable_exp64(x: Float64) -> Float64:
    """HOST-ONLY (no float64 on device, `mojolearn-hardware-limits`): `exp` for Float64 as one arithmetic, for the PROBABILITY LINKS -- the sigmoid/softmax CatBoost computes in double (`eval_processing.h:186-226`) and this implementation's `one_vs_all_probabilities` / `multiclass_probabilities` / the Python wrapper's Logloss sigmoid mirror."""
    from std.math import floor, fma
    from std.memory import bitcast

    if x != x:
        return x
    if x > 709.782712893384:
        return bitcast[DType.float64](UInt64(0x7FF0000000000000))  # +inf
    if x < -708.3964185322641:
        return Float64(0.0)
    # one rounding, as the default contract=fast build fused it (portable_expf's note)
    var k = floor(fma(x, 1.4426950408889634, 0.5))
    var r = fma(k, -6.93145751953125e-1, x)
    r = fma(k, -1.42860682030941723212e-6, r)
    var xx = r * r
    var px = fma(1.26177193074810590878e-4, xx, 3.02994407707441961300e-2)
    px = fma(px, xx, 9.99999999999999999910e-1)
    px = px * r
    var qx = fma(3.00198505138664455042e-6, xx, 2.52448340349684104192e-3)
    qx = fma(qx, xx, 2.27265548208155028766e-1)
    qx = fma(qx, xx, 2.00000000000000000009e0)
    var y = px / (qx - px)
    y = fma(2.0, y, 1.0)
    var ki = Int(k)
    var k1 = ki >> 1
    var k2 = ki - k1
    y = y * bitcast[DType.float64](UInt64(k1 + 1023) << 52)
    y = y * bitcast[DType.float64](UInt64(k2 + 1023) << 52)
    return y


def identical_exp64(x: Float64) -> Float64:
    """The probability links' seam call (host): IDENTICAL routes through `portable_exp64`, FAST is the host stdlib verbatim."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_exp64(x)
    from std.math import exp

    return exp(x)


def portable_log64(x_in: Float64) -> Float64:
    """HOST-ONLY (no float64 on device): `log` for Float64 as one arithmetic, the sibling of `portable_exp64` for the same reason (each host libm rounds double log differently in the last bit)."""
    from std.math import fma
    from std.memory import bitcast

    var x = x_in
    if x != x:
        return x
    if x == Float64(0.0):
        return bitcast[DType.float64](UInt64(0xFFF0000000000000))  # -inf
    if x < Float64(0.0):
        return bitcast[DType.float64](UInt64(0x7FF8000000000000))  # nan
    var bits = bitcast[DType.uint64](x)
    if bits == UInt64(0x7FF0000000000000):
        return x  # +inf
    var e = 0
    if (bits >> 52) == UInt64(0):
        x = x * 18014398509481984.0
        bits = bitcast[DType.uint64](x)
        e = -54
    e += Int((bits >> 52) & UInt64(0x7FF)) - 1022
    var m = bitcast[DType.float64]((bits & UInt64(0x000FFFFFFFFFFFFF)) | UInt64(0x3FE0000000000000))
    comptime SQRTH = 0.70710678118654752440
    var z: Float64
    var y: Float64
    var xm: Float64
    if e > 2 or e < -2:
        if m < SQRTH:
            e -= 1
            z = m - 0.5
            y = fma(0.5, z, 0.5)
        else:
            z = m - 0.5
            z = z - 0.5
            y = fma(0.5, m, 0.5)
        xm = z / y
        z = xm * xm
        var r = fma(-7.89580278884799154124e-1, z, 1.63866645699558079767e1)
        r = fma(r, z, -6.41409952958715622951e1)
        var sq = z + -3.56722798256324312549e1
        sq = fma(sq, z, 3.12093766372244180303e2)
        sq = fma(sq, z, -7.69691943550460008604e2)
        z = xm * (z * r / sq)
        var ye = Float64(e)
        z = fma(ye, -2.121944400546905827679e-4, z)
        z = z + xm
        z = fma(ye, 0.693359375, z)
        return z
    if m < SQRTH:
        e -= 1
        xm = fma(2.0, m, -1.0)
    else:
        xm = m - 1.0
    z = xm * xm
    var pp = fma(1.01875663804580931796e-4, xm, 4.97494994976747001425e-1)
    pp = fma(pp, xm, 4.70579119878881725854e0)
    pp = fma(pp, xm, 1.44989225341610930846e1)
    pp = fma(pp, xm, 1.79368678507819816313e1)
    pp = fma(pp, xm, 7.70838733755885391666e0)
    var qq = xm + 1.12873587189167450590e1
    qq = fma(qq, xm, 4.52279145837532221105e1)
    qq = fma(qq, xm, 8.29875266912776603211e1)
    qq = fma(qq, xm, 7.11544750618563894466e1)
    qq = fma(qq, xm, 2.31251620126765340583e1)
    y = xm * (z * pp / qq)
    var ye2 = Float64(e)
    y = fma(ye2, -2.121944400546905827679e-4, y)
    y = fma(z, -0.5, y)
    z = xm + y
    z = fma(ye2, 0.693359375, z)
    return z


def identical_log64(x: Float64) -> Float64:
    """The host double-log seam call: IDENTICAL routes through `portable_log64`, FAST is the host stdlib verbatim (the `identical_exp64` contract)."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_log64(x)
    from std.math import log

    return log(x)


def portable_log2_64(x_in: Float64) -> Float64:
    """HOST-ONLY (no float64 on device): `log2` for Float64 as one arithmetic. DEVIATION 2260 (2026-09-08). `portable_log64`'s argument reduction and BOTH of its Cephes rational approximations VERBATIM (same fma spellings, same association), with `log2.c`'s re-entry in place of `log.c`'s: the fraction's log is scaled by `LOG2EA = log2(e) - 1` and re-added to itself, so the exponent enters as an exact integer and `portable_log2_64(2^k) == k` bit for bit -- the property RF's `max_features='log2'` truncation (`randomforest.mojo::compute_max_features_log2`, `builder.cuh:240`) stands on. Replaces the host libm `log2` there, the one foreign call that IDENTITY_PATHS row 18 could not pin. Accuracy claim: within 2 ulp of the host libm on [2, 4096] (gated in `ensemble/checks/predict_check.mojo` beside the truncation rule); the full-range 2^18-hashed-input gate is `check-portable-log2-64`, including all 2098 representable powers of two, branch-boundary neighbors and special values. Its recorded output hash is host-agreement evidence only after independent host runs."""
    from std.math import fma
    from std.memory import bitcast

    var x = x_in
    if x != x:
        return x
    if x == Float64(0.0):
        return bitcast[DType.float64](UInt64(0xFFF0000000000000))  # -inf
    if x < Float64(0.0):
        return bitcast[DType.float64](UInt64(0x7FF8000000000000))  # nan
    var bits = bitcast[DType.uint64](x)
    if bits == UInt64(0x7FF0000000000000):
        return x  # +inf
    var e = 0
    if (bits >> 52) == UInt64(0):
        x = x * 18014398509481984.0
        bits = bitcast[DType.uint64](x)
        e = -54
    e += Int((bits >> 52) & UInt64(0x7FF)) - 1022
    var m = bitcast[DType.float64]((bits & UInt64(0x000FFFFFFFFFFFFF)) | UInt64(0x3FE0000000000000))
    comptime SQRTH = 0.70710678118654752440
    # `log2.c`'s LOG2EA = log2(e) - 1, split off so the exponent's integer
    # part is added exactly and only the fraction's log is scaled.
    comptime LOG2EA = 0.44269504088896340735992
    var z: Float64
    var y: Float64
    var xm: Float64
    if e > 2 or e < -2:
        # `log2.c:154-186` -- log(x) = z + z^3 R(z)/S(z), z = 2(m-1)/(m+1)
        if m < SQRTH:
            e -= 1
            z = m - 0.5
            y = fma(0.5, z, 0.5)
        else:
            z = m - 0.5
            z = z - 0.5
            y = fma(0.5, m, 0.5)
        xm = z / y
        z = xm * xm
        var r = fma(-7.89580278884799154124e-1, z, 1.63866645699558079767e1)
        r = fma(r, z, -6.41409952958715622951e1)
        var sq = z + -3.56722798256324312549e1
        sq = fma(sq, z, 3.12093766372244180303e2)
        sq = fma(sq, z, -7.69691943550460008604e2)
        y = xm * (z * r / sq)
    else:
        # `log2.c:190-215` -- log(1+xm) = xm - xm^2/2 + xm^3 P(xm)/Q(xm)
        if m < SQRTH:
            e -= 1
            xm = fma(2.0, m, -1.0)
        else:
            xm = m - 1.0
        z = xm * xm
        var pp = fma(1.01875663804580931796e-4, xm, 4.97494994976747001425e-1)
        pp = fma(pp, xm, 4.70579119878881725854e0)
        pp = fma(pp, xm, 1.44989225341610930846e1)
        pp = fma(pp, xm, 1.79368678507819816313e1)
        pp = fma(pp, xm, 7.70838733755885391666e0)
        var qq = xm + 1.12873587189167450590e1
        qq = fma(qq, xm, 4.52279145837532221105e1)
        qq = fma(qq, xm, 8.29875266912776603211e1)
        qq = fma(qq, xm, 7.11544750618563894466e1)
        qq = fma(qq, xm, 2.31251620126765340583e1)
        y = xm * (z * pp / qq)
        y = fma(z, -0.5, y)
    # `log2.c:217-224` -- multiply the fraction's log by log2(e) as
    # (1 + LOG2EA), then add the base-2 exponent as an exact integer:
    #     z = y*LOG2EA; z += x*LOG2EA; z += y; z += x; z += e;
    # At a power of two xm == y == 0, so the result is exactly `e`.
    var out = y * LOG2EA
    out = fma(xm, LOG2EA, out)
    out = out + y
    out = out + xm
    out = out + Float64(e)
    return out


def identical_log2_64(x: Float64) -> Float64:
    """Host log2 seam: only IDENTICAL selects the portable arithmetic."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_log2_64(x)
    from std.math import log2

    return log2(x)


def portable_pow64(x: Float64, p: Float64) -> Float64:
    """HOST-ONLY binary64 real power with pinned portable log/exp arithmetic.

    This is an exp(p*log(abs(x))) approximation, NOT a correctly rounded or
    tight-ULP pow implementation. Its finite accuracy is measured separately
    against libm; cross-host identity requires matching independent captures.
    NaNs are canonicalized except x**0 and 1**p, which return 1. Negative
    finite bases require integer exponents; parity is determined from bits,
    never a potentially overflowing integer conversion. Signed zero and
    infinities follow real pow special cases. Subnormal inputs are retained;
    subnormal outputs use a LOCAL scaled-exp reconstruction, leaving the
    existing portable_exp64 flush contract unchanged for all other consumers.
    """
    from std.math import fma
    from std.memory import bitcast

    comptime SIGN = UInt64(0x8000000000000000)
    comptime ABS = UInt64(0x7FFFFFFFFFFFFFFF)
    comptime INF = UInt64(0x7FF0000000000000)
    comptime ONE = UInt64(0x3FF0000000000000)
    comptime NAN = UInt64(0x7FF8000000000000)
    var xb = bitcast[DType.uint64](x)
    var pb = bitcast[DType.uint64](p)
    var axb = xb & ABS
    var apb = pb & ABS
    if apb == 0 or xb == ONE:
        return Float64(1)
    if axb > INF or apb > INF:
        return bitcast[DType.float64](NAN)
    if apb == INF:
        if axb == ONE:
            return Float64(1)
        if (axb > ONE) == (p > Float64(0)):
            return bitcast[DType.float64](INF)
        return Float64(0)

    # Finite exponent classification, including |p| >= 2**53 (all even).
    var integral = False
    var odd = False
    var pe = Int((apb >> 52) & UInt64(0x7FF)) - 1023
    if pe >= 0:
        if pe > 52:
            integral = True
        else:
            var fraction = 52 - pe
            var significand = (apb & UInt64(0x000FFFFFFFFFFFFF)) | (UInt64(1) << 52)
            integral = (significand & ((UInt64(1) << UInt64(fraction)) - UInt64(1))) == 0
            odd = integral and ((significand >> UInt64(fraction)) & UInt64(1)) != 0
    var sign = SIGN if (xb & SIGN) != 0 and odd else UInt64(0)
    if axb == 0:
        return bitcast[DType.float64](sign | (INF if p < Float64(0) else UInt64(0)))
    if axb == INF:
        return bitcast[DType.float64](sign | (INF if p > Float64(0) else UInt64(0)))
    if (xb & SIGN) != 0 and not integral:
        return bitcast[DType.float64](NAN)
    if p == Float64(1):
        return x
    if p == Float64(-1):
        return Float64(1) / x
    if p == Float64(2):
        return x * x

    # Integer powers of binary powers are exact, including overflow and
    # subnormal ties. This also avoids exp(log(2)*1024) rounding below inf.
    var binary_power = (axb & UInt64(0x000FFFFFFFFFFFFF)) == 0
    var xe = Int(axb >> 52) - 1023
    if axb < (UInt64(1) << 52) and (axb & (axb - UInt64(1))) == 0:
        binary_power = True
        var shifted_bits = axb
        xe = -1074
        while shifted_bits > 1:
            shifted_bits >>= 1
            xe += 1
    if integral and binary_power:
        var exponent = p * Float64(xe)
        if exponent > Float64(1023):
            return bitcast[DType.float64](sign | INF)
        if exponent < Float64(-1074):
            return bitcast[DType.float64](sign)
        var e = Int(exponent)
        var result_bits = UInt64(e + 1023) << 52 if e >= -1022 else UInt64(1) << UInt64(e + 1074)
        return bitcast[DType.float64](sign | result_bits)

    var ax = bitcast[DType.float64](axb)
    var power = p * portable_log64(ax)
    var magnitude: Float64
    if power < Float64(-708):
        if power < Float64(-746):
            magnitude = Float64(0)
        else:
            # exp(power + 512*ln(2)) * 2**-512. Both exp's argument
            # and result are normal; only the final IEEE multiply may be
            # subnormal. Split ln(2) preserves the existing exp reduction.
            var shifted = fma(Float64(512), Float64(6.93145751953125e-1), power)
            shifted = fma(Float64(512), Float64(1.42860682030941723212e-6), shifted)
            magnitude = portable_exp64(shifted) * bitcast[DType.float64](UInt64(511) << 52)
    else:
        magnitude = portable_exp64(power)
    return bitcast[DType.float64](bitcast[DType.uint64](magnitude) | sign)


def identical_pow64(x: Float64, p: Float64) -> Float64:
    """Host pow seam: IDENTICAL portable; other modes keep std.math.pow."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_pow64(x, p)
    from std.math import pow

    return pow(x, p)


def identical_sqrt(x: Float32) -> Float32:
    """Row 10's sqrt seam call: IDENTICAL routes through `portable_sqrtf` (one arithmetic, correctly rounded, the same bits on the approximate- sqrt column too); FAST is the stdlib's device path verbatim."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_sqrtf(x)
    from std.math import sqrt

    return sqrt(x)


def identical_cos(x: Float32) -> Float32:
    """Row 12's `cos` seam call (Box-Muller). Same two-arm contract."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_cosf(x)
    from std.math import cos

    return cos(x)


def identical_pow(x: Float32, p: Float32) -> Float32:
    """Row 12's `pow` seam call (Bayesian bootstrap temperature)."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_powf(x, p)
    return x**p




def ftz_simd[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """`ftz`, lane by lane."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var out = x
        comptime for i in range(w):
            out[i] = ftz(x[i])
        return out
    return x


def identical_mul_add_simd[
    w: Int
](
    a: SIMD[DType.float32, w],
    b: SIMD[DType.float32, w],
    c: SIMD[DType.float32, w],
) -> SIMD[DType.float32, w]:
    """`identical_mul_add`, lane by lane: ONE rounding under IDENTICAL, the naive chain under FAST."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return _fma_f32_simd[w](a, b, c)
    return a * b + c


def _fma_f32_simd[
    w: Int
](
    a: SIMD[DType.float32, w],
    b: SIMD[DType.float32, w],
    c: SIMD[DType.float32, w],
) -> SIMD[DType.float32, w]:
    from std.math import fma

    return fma(a, b, c)




def _ftz_always(x: Float32) -> Float32:
    """Row 10's flush, UNCONDITIONAL and BY BITS: the portable_* functions below bake the policy in so their bits do not depend on the build mode (the same choice `portable_expf`/`portable_logf` made inline)."""
    from std.memory import bitcast

    var b = rebind[UInt32](x.to_bits())
    if (b & UInt32(0x7F800000)) == UInt32(0):
        return bitcast[DType.float32](b & UInt32(0x80000000))
    return x


def portable_divf(a: Float32, b: Float32) -> Float32:
    """DEVIATION 740: `a / b` under row 10's flush model -- operands flushed to signed zero, ONE hardware division (correctly rounded on every column measured: Apple normal class 0 wrong over 2^20 pairs in `check-division`; the H100/MI325X `check-ieee-arith` div arms 0 wrong), result flushed."""
    return _ftz_always(_ftz_always(a) / _ftz_always(b))


def portable_rsqrtf(x_in: Float32) -> Float32:
    """DEVIATION 741, pin (a): `1 / portable_sqrtf(x)`."""
    from std.memory import bitcast

    var x = _ftz_always(x_in)  # BEFORE the sign test: Metal compares flush
    if x != x:
        return x
    if x < Float32(0.0):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    var s = portable_sqrtf(x)
    if s == Float32(0.0):
        return bitcast[DType.float32](UInt32(0x7F800000))  # +inf
    if s == bitcast[DType.float32](UInt32(0x7F800000)):
        return Float32(0.0)
    return portable_divf(Float32(1.0), s)


def portable_log1pf(x_in: Float32) -> Float32:
    """DEVIATION 742: `log1pf` as one arithmetic."""
    from std.memory import bitcast

    var x = _ftz_always(x_in)
    if x != x:
        return x
    if x == Float32(0.0):
        return x
    if x == bitcast[DType.float32](UInt32(0x7F800000)):
        return x
    if x < Float32(-1.0):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    if x == Float32(-1.0):
        return bitcast[DType.float32](UInt32(0xFF800000))  # -inf
    if x >= bitcast[DType.float32](UInt32(0xBE95F61A)) and x <= bitcast[
        DType.float32
    ](UInt32(0x3ED413CD)):
        return _cephes_logf_core(x, 0)
    return portable_logf(x + Float32(1.0))


def portable_sigmoidf(x: Float32) -> Float32:
    """DEVIATION 743: `1 / (1 + portable_expf(-x))` through `portable_divf`."""
    if x != x:
        return x
    var e = portable_expf(-x)
    var d = e + Float32(1.0)
    return portable_divf(Float32(1.0), d)


def portable_siluf(x: Float32) -> Float32:
    """DEVIATION 744: the reference's `x / (1 + portable_expf(-x))`, one division through `portable_divf` (NOT x * sigmoid(x))."""
    if x != x:
        return x
    var e = portable_expf(-x)
    var d = e + Float32(1.0)
    return portable_divf(x, d)


def portable_softplusf(x: Float32) -> Float32:
    """DEVIATION 745: `x <= 20 ?"""
    if x <= Float32(20.0):
        return portable_log1pf(portable_expf(x))
    return x


def identical_div(a: Float32, b: Float32) -> Float32:
    """Row 49's seam: IDENTICAL routes through `portable_divf` (the flush model around one correctly rounded division); FAST is plain `/`."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_divf(a, b)
    return a / b


def identical_rsqrt(x: Float32) -> Float32:
    """Row 50's seam: IDENTICAL is `portable_rsqrtf`; FAST is the reference's own spelling `1 / sqrt(x)` through the stdlib sqrt (the vendor's sqrt -- approximate on NVIDIA, DEVIATION 258), never the rsqrt intrinsic."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_rsqrtf(x)
    from std.math import sqrt

    return Float32(1.0) / sqrt(x)


def identical_log1p(x: Float32) -> Float32:
    """Row 51's seam: IDENTICAL is `portable_log1pf`; FAST is `log(1 + x)` through the stdlib log -- Triton 3's own spelling of the same thing (mamba_ssm/ops/triton/softplus.py:11) -- and NOT `std.math.log1p`, because that one lowers the Float32 case THROUGH FLOAT64 (`air.convert.f.f64.f.f32` ..."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_log1pf(x)
    from std.math import log

    return log(Float32(1.0) + x)


def identical_sigmoid(x: Float32) -> Float32:
    """Row 52's seam: IDENTICAL is `portable_sigmoidf`; FAST is `1 / (1 + exp(-x))` through the stdlib exp."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_sigmoidf(x)
    from std.math import exp

    return Float32(1.0) / (Float32(1.0) + exp(-x))


def identical_silu(x: Float32) -> Float32:
    """Row 53's seam: IDENTICAL is `portable_siluf`; FAST is the reference's `x / (1 + exp(-x))` through the stdlib exp."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_siluf(x)
    from std.math import exp

    return x / (Float32(1.0) + exp(-x))


def identical_softplus(x: Float32) -> Float32:
    """Row 54's seam: IDENTICAL is `portable_softplusf`; FAST is the reference's guard around Triton 3's `log(exp(x) + 1)` through the stdlib (not `log1p`: see `identical_log1p` for why it cannot be)."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_softplusf(x)
    from std.math import exp, log

    if x <= Float32(20.0):
        return log(exp(x) + Float32(1.0))
    return x




comptime GELU_SQRT2_BITS: UInt32 = 0x3FB504F3

comptime GELU_TANH_SCALE_BITS: UInt32 = 0x3F4C422A

comptime GELU_TANH_COEF_BITS: UInt32 = 0x3D372713

comptime TANH_SAT_BITS: UInt32 = 0x42317218

comptime NEG_MAXLOGF_BITS: UInt32 = 0xC2B17218


def portable_sinf(x_in: Float32) -> Float32:
    """DEVIATION 820: `sinf` as one arithmetic, Cephes single-precision `sinf` (`cephes/single/sinf.c`) through `_cephes_sincosf_core` -- the SAME reduction and the SAME two polynomials `portable_cosf` already uses and is already certified on."""
    from std.memory import bitcast

    if x_in != x_in or abs(x_in) == bitcast[DType.float32](UInt32(0x7F800000)):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    var x = _ftz_always(x_in)
    var sign = Float32(1.0)
    if (rebind[UInt32](x.to_bits()) & UInt32(0x80000000)) != UInt32(0):
        sign = Float32(-1.0)
    return _cephes_sincosf_core(abs(x), sign, False)


def portable_tanhf(x_in: Float32) -> Float32:
    """DEVIATION 821: `tanhf` as one arithmetic, Cephes single-precision `tanhf` (`cephes/single/tanhf.c`) verbatim in structure."""
    from std.memory import bitcast

    var xx = _ftz_always(x_in)
    if xx != xx:
        return xx
    if xx == Float32(0.0):
        return xx  # +-0 -> +-0; Cephes loses the sign here, see above
    var x = abs(xx)
    if x > bitcast[DType.float32](TANH_SAT_BITS):
        if xx < Float32(0.0):
            return Float32(-1.0)
        return Float32(1.0)
    if x >= Float32(0.625):
        var e = portable_expf(x + x)
        var d = e + Float32(1.0)
        var q = portable_divf(Float32(2.0), d)
        var z = Float32(1.0) - q
        if xx < Float32(0.0):
            return -z
        return z
    var zs = x * x
    var p = Float32(-5.70498872745e-3)
    p = _fma_f32(p, zs, Float32(2.06390887954e-2))
    p = _fma_f32(p, zs, Float32(-5.37397155531e-2))
    p = _fma_f32(p, zs, Float32(1.33314422036e-1))
    p = _fma_f32(p, zs, Float32(-3.33332819422e-1))
    var pz = p * zs
    return _fma_f32(pz, xx, xx)


def _cephes_erfcf_ge1(a: Float32) -> Float32:
    """The Cephes `erfcf` body (`cephes/single/ndtrf.c`) for |a| >= 1 ONLY."""
    from std.memory import bitcast

    var x = abs(a)
    var aa = x * x
    var z = -aa
    if z < bitcast[DType.float32](NEG_MAXLOGF_BITS):
        if a < Float32(0.0):
            return Float32(2.0)
        return Float32(0.0)
    z = portable_expf(z)
    var q = portable_divf(Float32(1.0), x)
    var y = q * q
    var p: Float32
    if x < Float32(2.0):
        p = Float32(2.326819970068386e-2)
        p = _fma_f32(p, y, Float32(-1.387039388740657e-1))
        p = _fma_f32(p, y, Float32(3.687424674597105e-1))
        p = _fma_f32(p, y, Float32(-5.824733027278666e-1))
        p = _fma_f32(p, y, Float32(6.210004621745983e-1))
        p = _fma_f32(p, y, Float32(-4.944515323274145e-1))
        p = _fma_f32(p, y, Float32(3.404879937665872e-1))
        p = _fma_f32(p, y, Float32(-2.741127028184656e-1))
        p = _fma_f32(p, y, Float32(5.638259427386472e-1))
    else:
        p = Float32(-10.47766399936249)
        p = _fma_f32(p, y, Float32(12.97719955372516))
        p = _fma_f32(p, y, Float32(-7.495518717768503))
        p = _fma_f32(p, y, Float32(2.921019019210786))
        p = _fma_f32(p, y, Float32(-1.015265279202700))
        p = _fma_f32(p, y, Float32(4.218463358204948e-1))
        p = _fma_f32(p, y, Float32(-2.820767439740514e-1))
        p = _fma_f32(p, y, Float32(5.641895067754075e-1))
    var t = _ftz_always(z * q)
    var r = _ftz_always(t * p)
    if a < Float32(0.0):
        r = Float32(2.0) - r
    return r


def portable_erff(x_in: Float32) -> Float32:
    """DEVIATION 822: `erff` as one arithmetic, Cephes single-precision `erff` (`cephes/single/ndtrf.c`) verbatim in structure."""
    var x = _ftz_always(x_in)
    if x != x:
        return x
    if abs(x) > Float32(1.0):
        return Float32(1.0) - _cephes_erfcf_ge1(x)
    var z = x * x
    var p = Float32(7.853861353153693e-5)
    p = _fma_f32(p, z, Float32(-8.010193625184903e-4))
    p = _fma_f32(p, z, Float32(5.188327685732524e-3))
    p = _fma_f32(p, z, Float32(-2.685381193529856e-2))
    p = _fma_f32(p, z, Float32(1.128358514861418e-1))
    p = _fma_f32(p, z, Float32(-3.761262582423300e-1))
    p = _fma_f32(p, z, Float32(1.128379165726710))
    # Pinned: a caller's `1 + erf` (the exact gelu) sits across this branch's
    # join from the product; no build fuses it today (the PTX of every gelu
    # kernel keeps `mul.f32` then `add.f32`), and the pin keeps it that way
    # for ptxas and for any future inlining (lane/pinned-mul-contract-free).
    return pinned_mul_f32(x, p)


def portable_gelu_erf(x_in: Float32) -> Float32:
    """DEVIATION 823: the EXACT gelu, `input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))`, which is `GELUActivation._gelu_python` at activations.py:85-86 verbatim."""
    from std.memory import bitcast

    var x = _ftz_always(x_in)
    if x != x:
        return x
    var s2 = bitcast[DType.float32](GELU_SQRT2_BITS)
    var e = portable_erff(portable_divf(x, s2))
    var s = Float32(1.0) + e
    var h = _ftz_always(x * Float32(0.5))
    return _ftz_always(h * s)


def portable_gelu_tanh(x_in: Float32) -> Float32:
    """DEVIATION 824: the TANH gelu, `input * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))`, which is `GELUTanh._gelu_tanh_python` at activations.py:47-48 verbatim, and the same arithmetic as `NewGELUActivation.forward` at :65 (the factor is written on the other side of `input`, which is the same bits)."""
    from std.memory import bitcast

    var x = _ftz_always(x_in)
    if x != x:
        return x
    var c3 = bitcast[DType.float32](GELU_TANH_COEF_BITS)
    var kk = bitcast[DType.float32](GELU_TANH_SCALE_BITS)
    var x2 = _ftz_always(x * x)
    var x3 = _ftz_always(x2 * x)
    var cx = _ftz_always(c3 * x3)
    var inner = _ftz_always(x + cx)
    var arg = _ftz_always(kk * inner)
    var t = portable_tanhf(arg)
    var s = Float32(1.0) + t
    var h = _ftz_always(x * Float32(0.5))
    return _ftz_always(h * s)


def _total_order_key(v: Float32) -> UInt32:
    """DEVIATION 204's map, the SECOND copy in this tree and deliberately not a second DESIGN: a float as an unsigned key whose INTEGER order is the float's order."""
    var b = rebind[UInt32](v.to_bits())
    if (b & UInt32(0x80000000)) != UInt32(0):
        return ~b
    return b | UInt32(0x80000000)


def portable_fmaxf(a_in: Float32, b_in: Float32) -> Float32:
    """DEVIATION 825: an elementwise maximum whose result does not depend on the argument order, the vendor, or the fold shape."""
    from std.memory import bitcast

    if a_in != a_in or b_in != b_in:
        return bitcast[DType.float32](UInt32(0x7FC00000))
    var a = _ftz_always(a_in)
    var b = _ftz_always(b_in)
    if _total_order_key(b) > _total_order_key(a):
        return b
    return a


def identical_sin(x: Float32) -> Float32:
    """Row 12's `sin` seam call (RoPE)."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_sinf(x)
    from std.math import sin

    return sin(x)


def identical_tanh(x: Float32) -> Float32:
    """DEVIATION 821's seam: IDENTICAL is `portable_tanhf`; FAST is `std.math.tanh`, the stdlib's device path verbatim."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_tanhf(x)
    from std.math import tanh

    return tanh(x)


def identical_erf(x: Float32) -> Float32:
    """DEVIATION 822's seam: IDENTICAL is `portable_erff`; FAST is `std.math.erf`, the stdlib's device path verbatim."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_erff(x)
    from std.math import erf

    return erf(x)


def identical_gelu_erf(x: Float32) -> Float32:
    """DEVIATION 823's seam, the EXACT gelu: IDENTICAL is `portable_gelu_erf`; FAST is the reference's own spelling `x * 0.5 * (1 + erf(x / sqrt(2)))` through the stdlib erf, with the same pinned float32 divisor."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_gelu_erf(x)
    from std.math import erf
    from std.memory import bitcast

    var s2 = bitcast[DType.float32](GELU_SQRT2_BITS)
    return x * Float32(0.5) * (Float32(1.0) + erf(x / s2))


def identical_gelu_tanh(x: Float32) -> Float32:
    """DEVIATION 824's seam, the TANH gelu: IDENTICAL is `portable_gelu_tanh`; FAST is the reference's own spelling `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` through the stdlib tanh, with the same pinned float32 constants and the same unfused association."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_gelu_tanh(x)
    from std.math import tanh
    from std.memory import bitcast

    var c3 = bitcast[DType.float32](GELU_TANH_COEF_BITS)
    var kk = bitcast[DType.float32](GELU_TANH_SCALE_BITS)
    return (
        x
        * Float32(0.5)
        * (Float32(1.0) + tanh(kk * (x + c3 * (x * x * x))))
    )


def identical_fmax(a: Float32, b: Float32) -> Float32:
    """DEVIATION 825's seam (softmax's row maximum): IDENTICAL is `portable_fmaxf` (the total-order selection, NaN canonicalized, operands and result flushed); FAST is the stdlib `max`, whose signed-zero and NaN answers are the vendor's own choice."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_fmaxf(a, b)
    from std.math import max

    return max(a, b)


def portable_clampf(x_in: Float32, lo_in: Float32, hi_in: Float32) -> Float32:
    """DEVIATION 788 (the mamba2 lane's dt seam, `mamba/IDENTICAL_MAMBA2_CONTRACT.md` seam S9): a clamp whose result does not depend on the vendor, the argument order of a hardware min/max, or a compiler's constant-fold of one -- the ONE new primitive that profile needs."""
    from std.memory import bitcast

    if x_in != x_in or lo_in != lo_in or hi_in != hi_in:
        return bitcast[DType.float32](UInt32(0x7FC00000))
    var x = _ftz_always(x_in)
    var lo = _ftz_always(lo_in)
    var hi = _ftz_always(hi_in)
    var v = x
    if _total_order_key(lo) > _total_order_key(v):
        v = lo
    if _total_order_key(hi) < _total_order_key(v):
        v = hi
    return v


def identical_clamp(x: Float32, lo: Float32, hi: Float32) -> Float32:
    """DEVIATION 788's seam (mamba2 S9, `dt = clamp(softplus(dt_raw + bias), lo, hi)`): IDENTICAL is `portable_clampf` (total-order value-pick, NaN canonicalized, operands flushed); FAST is the reference's own spelling `min(max(x, lo), hi)` through the stdlib min/max, whose signed-zero and NaN answers are the vendor's own choice -- FAST's contract is..."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_clampf(x, lo, hi)
    from std.math import max, min

    return min(max(x, lo), hi)


# ===========================================================================
# LOW-BIT STORAGE SEAMS (lane/identical-lowbit-inference, 2026-09-17)
# ===========================================================================
# DEVIATION 2900 to 2905. The bf16 and int8 profiles of
# `gemm/IDENTICAL_LOWBIT_CONTRACT.md` are built from these six helpers and
# from nothing else, and the oracle in `gemm/host/gemm_lowbit_oracle.mojo`
# imports them rather than spelling them a second time. Every one is a
# bit-level construction: no conversion here trusts a backend's `cast`,
# because a cast's rounding at a tie is a codegen decision on Metal and a
# documented one on CUDA, and the contract needs it to be neither.


def bf16_bits_to_f32(bits: UInt16) -> Float32:
    """DEVIATION 2900: the exact widening. A bf16 is the top half of a
    float32, so this is a shift and cannot round. Contract L-1."""
    return bitcast[DType.float32](UInt32(bits) << UInt32(16))


def f32_to_bf16_bits_rne(x_in: Float32) -> UInt16:
    """DEVIATION 2901: the narrowing seam, round to nearest even on the
    discarded 16 bits, after the flush. Contract L-2.

    A NaN keeps its sign and top payload bits and is forced quiet, so a NaN
    never narrows to an infinity. An overflow rounds up into the infinity
    encoding, which is what round-to-nearest means at the top of the range.
    """
    var x = ftz(x_in)
    var b = bitcast[DType.uint32](x)
    if (b & UInt32(0x7F800000)) == UInt32(0x7F800000) and (
        b & UInt32(0x007FFFFF)
    ) != UInt32(0):
        return UInt16((b >> UInt32(16)) | UInt32(0x0040))
    var lsb = (b >> UInt32(16)) & UInt32(1)
    var rounded = b + UInt32(0x7FFF) + lsb
    return UInt16(rounded >> UInt32(16))


#: `1.5 * 2^23`. Adding and subtracting it rounds a float32 of magnitude at
#: most `2^22` to an integer under the default rounding mode, and nothing
#: else: no multiply, so contraction has nothing to fuse, and no operation
#: below the smallest normal.
comptime RNE_MAGIC_F32 = Float32(12582912.0)


def f32_round_half_even(x: Float32) -> Float32:
    """DEVIATION 2902: round to the nearest integer, ties to even, for
    `|x| <= 2^22`. Contract L-4. Spelled with the magic constant rather than
    a library `round`, whose tie rule is the library's."""
    if x >= Float32(0.0):
        return (x + RNE_MAGIC_F32) - RNE_MAGIC_F32
    return (x - RNE_MAGIC_F32) + RNE_MAGIC_F32


def pow2_f32(e: Int) -> Float32:
    """DEVIATION 2903: `2^e` as a float32 built from its exponent field.
    Above 127 it is the infinity; below -126 it is `+0.0`, because the
    contract has no subnormal scales (contract L-3)."""
    if e > 127:
        return bitcast[DType.float32](UInt32(0x7F800000))
    if e < -126:
        return Float32(0.0)
    return bitcast[DType.float32](UInt32(e + 127) << UInt32(23))


def f32_exponent(x: Float32) -> Int:
    """`floor(log2 |x|)` for a normal `x`, read off the exponent field.
    Zero and every subnormal answer -127."""
    var b = bitcast[DType.uint32](x)
    return Int((b >> UInt32(23)) & UInt32(0xFF)) - 127


def i32_to_f32_pinned(v: Int32) -> Float32:
    """DEVIATION 2904: the integer accumulator as a float32, contract L-5.

    Exact below `2^24` and rounded to nearest even above it, spelled as two
    exact conversions (a 20-bit high part and a 12-bit low part, each below
    `2^24`), one exact scaling by `2^12`, and ONE float32 addition, so the
    only rounding is the addition's and that one is IEEE's. A backend's own
    int-to-float instruction rounds the same way on every vendor this tree
    ships to, and this spelling is what makes that a construction rather
    than a survey."""
    var neg = v < Int32(0)
    var wide = Int64(v)
    if neg:
        wide = -wide
    var mag = UInt32(wide)
    var hi = Float32(Int(mag >> UInt32(12)))
    var lo = Float32(Int(mag & UInt32(0xFFF)))
    var r = hi * Float32(4096.0) + lo
    if neg:
        return -r
    return r


#: The quantizer scales a row so its largest magnitude lands in `[64, 128)`:
#: one exponent below the int8 ceiling, which keeps `rne(x * 2^-e)` at most
#: 128 and the clamp below a one-sided event.
comptime INT8_TARGET_EXPONENT = 6


def int8_row_exponent(absmax: Float32) -> Int:
    """DEVIATION 2905: the power-of-two scale of a row, contract L-3.
    `e = floor(log2 absmax) - 6`, so `absmax * 2^-e` is in `[64, 128)`.
    An all-zero row (absmax `0`, after the flush) takes `e = 0`."""
    var a = ftz(absmax)
    if a == Float32(0.0) or a != a:
        return 0
    return f32_exponent(a) - INT8_TARGET_EXPONENT


def quantize_int8_value(x: Float32, e: Int) -> Int8:
    """DEVIATION 2905: `q = clamp(rne(x * 2^-e), -127, 127)`, contract L-4.
    The scaling is one multiply by a power of two, exact unless it lands
    below the smallest normal, where the flush makes it zero; the rounding
    is `f32_round_half_even`; the clamp is symmetric so `-q` is always
    representable. A NaN quantizes to zero: the caller refuses NaN weights
    before it ever gets here, and this is the one answer that cannot
    corrupt a neighbor."""
    if x != x:
        return Int8(0)
    var s = ftz(identical_mul(ftz(x), pow2_f32(-e)))
    var r = f32_round_half_even(s)
    if r > Float32(127.0):
        r = Float32(127.0)
    if r < Float32(-127.0):
        r = Float32(-127.0)
    return Int8(Int(r))


def dequant_int8_pinned(acc: Int32, e_sum: Int) -> Float32:
    """DEVIATION 2904: the dequantization seam, contract L-5 and L-6.
    `ftz(acc_f32 * 2^(ea + eb))`: one multiply by a power of two, exact
    unless it overflows or lands below the smallest normal, then the flush.
    The multiply is `identical_mul` so that FAST and IDENTICAL builds spell
    it the same way and only the pins differ."""
    return ftz(identical_mul(i32_to_f32_pinned(acc), pow2_f32(e_sum)))
