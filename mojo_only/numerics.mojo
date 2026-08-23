"""What may vary between backends, and what may not.

NOT a port. CatBoost has no equivalent and does not need one: it ships one
GPU backend and accepts a non-deterministic answer. We ship Metal, CUDA and
HIP from one source, so "the same fit gives the same model" is a property we
have to design for rather than inherit.

THE AXIS, and it is the thing that is easy to get wrong
-------------------------------------------------------
The tempting split is "numeric rows" against "scheduling rows", vary the
scheduling per vendor and pin the numerics. **That split is false**, because
some scheduling decisions ARE numeric decisions. A block count is a summation
order. A partial-sum tree shape is a summation order. How many private copies
a histogram is replicated into decides how many partial sums get combined and
in what sequence.

So a row is classified by ONE question:

    Does it change the SEQUENCE or the PRECISION of the arithmetic?

If yes it is a NUMERIC row and `NumericMode.IDENTICAL` pins it, whatever it
looks like. If no it is a SCHEDULING row and every backend picks freely, in
both modes. Grid shape, threads per block, occupancy targets, launch batching
and prefetch distance are scheduling. Accumulator width, flush order,
replication factor, block count and contraction are numeric.

WHAT A MODE CANNOT DO
---------------------
Two things escape the table and must be handled in source rather than
configured:

- **Floating-point atomics are order-nondeterministic run to run**, not just
  backend to backend. No row pins them. `IDENTICAL` has to REPLACE the
  accumulator, not configure it.
- **FMA contraction is a codegen decision.** The compiler decides whether
  `a*b+c` becomes one rounding or two. A runtime row cannot reach it; it is
  fixed at the source, per kernel.

THE COST IS A MEASUREMENT, NOT AN ARGUMENT
------------------------------------------
Because the modes differ by a named, enumerable set of rows, "what does
determinism cost" has an answer in seconds rather than in opinions. Run the
two modes interleaved and report the ratio.
"""

from std.sys.compile import is_defined



#: **THE DEFAULT.** Every row is read from the device's own column. Full
#: per-vendor speed, and histograms flush through floating-point atomics.
#:
#: **What that costs, stated because a user must not discover it:** float
#: atomics have no ordering guarantee, so the last bits move between two runs
#: of the SAME fit on the SAME device, not only between vendors. Model files
#: will not be byte-comparable, a fit is not reproducible from its seed
#: alone, and a regression test cannot assert on exact predictions. This is
#: CatBoost's shipped behavior, so it is a defensible default rather than a
#: novel one.
#: **THE SWITCH. One line, the whole build.**
#:
#: It did not exist until 2026-08-21 and the mode was therefore NOT REACHABLE:
#: five kernel files each declared their own `comptime BUILD_MODE =
#: NUMERIC_FAST`, so selecting `IDENTICAL` meant editing five files and
#: knowing which five. A toggle a user cannot flip is a toggle that does not
#: exist, and every claim made about "the IDENTICAL mode" was a claim about a
#: configuration nobody had ever built.
#:
#: Found by Andrew asking whether the toggle actually works. It did not.
#:
#: Every site that used to hardcode `NUMERIC_FAST` now reads this, so the
#: default is unchanged bit for bit and the other mode rebuilds the whole
#: tree. Comptime, because the two modes are different code (a float atomic
#: and a fixed-point accumulator are not one configured value), which is the
#: same reason `TARGET_COLUMN` is a build rather than a flag.
#:
#: **A BUILD DEFINE SINCE 2026-08-23, NOT AN EDITED LINE.** It used to read
#: `= NUMERIC_FAST` and every identity gate flipped it with sed and reverted
#: on exit; two sessions sharing one checkout then lost an edit made inside a
#: flip window (restored from a backup copy), and a flip left behind
#: mislabels every number the next session measures. Now, exactly like the
#: column (`kernel_matrix.mojo`'s `-D MOJOLEARN_COLUMN_*`):
#:
#:     mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . <check>.mojo
#:     MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_gbdt.sh
#:
#: `tools/with_identical_mode.sh <cmd>` injects the define into a `mojo
#: run`/`mojo build` command line (no file is touched); the build scripts
#: land an identical binary set under `python/mojolearn/identical/`, and
#: `python/mojolearn/_backend.py` loads that set when the env var
#: `MOJOLEARN_NUMERIC_MODE=identical` is set at import. Default is FAST, in
#: every spelling. The source line below never changes again.
comptime GLOBAL_NUMERIC_MODE = (
    NUMERIC_IDENTICAL if is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]() else NUMERIC_FAST
)


comptime NUMERIC_FAST = 0

#: Opt in when reproducibility is needed: numeric rows are read from
#: `COLUMN_BIT_IDENTICAL` instead of the device's column, so the same fit
#: gives the same model on Apple, NVIDIA and AMD and on repeated runs.
#:
#: The cost is a measurement, not an argument: the two modes differ by a
#: named set of rows, so run them interleaved and report the ratio.
comptime NUMERIC_IDENTICAL = 1


@fieldwise_init
struct NumericMode(Copyable, Movable):
    """The mode, plus the resolved value of every numeric row.

    The rows are resolved ONCE, here, rather than each kernel calling
    `getenv` and reaching its own conclusion. A kernel that decides for itself
    is how two kernels come to disagree about which mode they are in.
    """

    var mode: Int

    @staticmethod
    def default() -> Self:
        """`FAST`. Per-vendor speed everywhere, and results that move in the
        last bits run to run. See `NUMERIC_FAST` for the full consequence."""
        return Self(NUMERIC_FAST)

    def deterministic_flush(self) -> Bool:
        """NUMERIC ROW. Whether the shared histogram flushes to global memory
        through a fixed-point integer accumulator instead of `atomicAdd` on
        `float`.

        Integer addition is associative, so a fixed-point flush is
        order-independent and therefore reproducible; a float atomic is not,
        and no ordering guarantee is available to make it so. This is the row
        that cannot be a toggle over one implementation: the two modes run
        different code.
        """
        return self.mode == NUMERIC_IDENTICAL

    def fixed_replication(self) -> Bool:
        """NUMERIC ROW, and it LOOKS like a scheduling row, which is the whole
        reason this file exists.

        How many private copies the shared histogram is replicated into
        decides how many partial sums are combined and in what order. Under
        `IDENTICAL` the replication factor is pinned to a value every backend
        can meet, so the reduction tree has the same shape everywhere. Under
        `FAST` each backend picks the factor its threadgroup budget allows,
        and Apple's 32 KB cap already forces a different one from CUDA's 48 KB.
        """
        return self.mode == NUMERIC_IDENTICAL

    def free_block_shape(self) -> Bool:
        """SCHEDULING ROW. Threads per block, grid shape, how many blocks are
        launched per partition to fill the machine.

        Free in BOTH modes. It changes which thread does which work and not
        what is added to what, PROVIDED the replication factor and the
        reduction width are pinned separately, which is what
        `fixed_replication` is for. Splitting these two apart is the point of
        the classification: they are usually chosen together and only one of
        them is numeric.
        """
        return True


# ============ THE TWO SOURCE-LEVEL CONSTRUCTIONS ==========================
# The header above names two things no mode row can reach: float atomics
# (REPLACED by the fixed-point accumulator elsewhere) and the pair below,
# which exist so that IDENTICAL's arithmetic is ONE arithmetic on every
# backend. Both are comptime no-ops under FAST.


def ftz(x: Float32) -> Float32:
    """IDENTITY_PATHS row 10's construction: the denormal policy.

    MEASURED, NOT DESIGNED. `check-ieee-arith` (2^20 hashed patterns)
    found Metal through MAX correctly rounded on every normal input with
    FLUSH-TO-ZERO on denormal operands, intermediates and results -- and
    this exact model (flush operands to SIGNED zero, correctly-rounded
    op, flush result) reproduced ALL 53,041 observed divergences bit for
    bit. CUDA's default honors denormals, so without this the same fit
    diverges across vendors on any pathway a denormal can reach.

    Under IDENTICAL, apply at every float SEAM a kernel writes for
    another kernel or the host to read (the row-10 checklist in
    IDENTITY_PATHS.md). On an FTZ backend the flush is bitwise a no-op
    -- flushing what hardware already flushed -- so Apple's IDENTICAL
    bits do not move; on a denormal-honoring backend it aligns them to
    the FTZ ones. Under FAST it compiles away entirely.

    Intermediates INSIDE an expression cannot be reached this way on a
    non-FTZ backend; row 10's checklist therefore also requires that
    pinned expressions be written with their intermediates stored
    through `ftz` (one extra local per step), which is exactly how the
    check's model computes.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if abs(x) < Float32(1.1754943508222875e-38) and x != Float32(0.0):
            if x < Float32(0.0):
                return Float32(-0.0)
            return Float32(0.0)
    return x


def identical_mul_add(a: Float32, b: Float32, c: Float32) -> Float32:
    """IDENTITY_PATHS row 9's construction: the contraction pin.

    `a*b + c` is one rounding or two at the CODEGEN'S whim, and the whim
    is per backend and per context. CUDA's compiler contracts by default,
    and Mojo has been seen contracting ACROSS expressions where clang
    would not (the mojotrees plateau-tie incident). No flag pins this
    from a mode table.

    **METAL THROUGH MAX CONTRACTS TOO, CORRECTED 2026-08-23.** The
    sentence that stood here said "Metal measured UNFUSED on 2^20
    patterns (`check-ieee-arith`, fused 0 / unfused 1,046,394)". That
    verdict came off 2^20 HASHED patterns of which ZERO separate a fused
    `a*b+c` from an unfused one -- random exponents put the product and
    the addend so far apart that both spellings round the same way -- and
    the tie arm was written `if got == unfused` FIRST, so every tie was
    counted as evidence of UNFUSED. A backend that contracts everything
    scored exactly the same. `check-ieee-arith` now carries a
    BUILT-TO-SEPARATE arm and Metal reports FUSED on 1,629 of 1,629
    separating patterns. See IDENTITY_PATHS row 9 for the full
    correction, including the independent corroboration from the GBDT
    lane the same day.

    Under IDENTICAL every enumerated multiply-add seam calls this and
    gets `fma` -- ONE rounding, identical on every backend that
    implements IEEE fma, which Metal, PTX and AMDGPU all do. Under FAST
    it is the naive chain and the backend does whatever it measures
    fastest.

    CONSEQUENCE OF THE CORRECTION, and it is the thing to carry away:
    this helper is BIT-INERT ON APPLE, exactly as `ftz` is on an FTZ
    backend, so IDENTICAL and FAST agree at these seams on this column.
    "Apple's bits did not move" is therefore NOT evidence that a pin is
    unreached, and any reasoning anywhere that rests on "Apple is
    unfused" is unsound and must be re-derived. The pin's value is the
    backend that does NOT contract by default, where an unpinned
    `acc += x * y` rounds twice against Metal's once.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return _fma_f32(a, b, c)
    return a * b + c


def _fma_f32(a: Float32, b: Float32, c: Float32) -> Float32:
    from std.math import fma

    return fma(a, b, c)


# ============ ROW 12: THE PORTABLE TRANSCENDENTALS ========================
# `std.math.exp`/`log` lower to whatever each target ships (PTX fast paths,
# OCML on AMD, Metal's own), so exp(x)'s last bit is a VENDOR CHOICE even
# with every reduction pinned. E1 measured it: with byte-identical inputs
# and tree 0 identical through winners, Apple<->AMD Logloss first diverged
# at `tree000.perm0.leaves.estimated` (2026-08-22), and NVIDIA produced a
# THIRD set of bits (2026-08-23). The closure is the pair below: ONE
# polynomial from ONE source, evaluated through fma and basic ops only --
# each correctly rounded on every backend (`check-ieee-arith`) -- so the
# same bits come out everywhere the same bits go in.
#
# The polynomials are Cephes single-precision expf/logf (Moshier's tables,
# the ancestor of half the world's float32 libms): degree-5 exp on
# [-ln2/2, ln2/2] after 2^k reduction, degree-8 log-mantissa on
# [sqrt(1/2), sqrt(2)). Accuracy is a measured <= 2 ulp against a float64
# reference (`check-portable-translog`), which is libm-class; the property
# purchased is not accuracy but SAMENESS. Both functions flush subnormal
# inputs and never produce a subnormal output (row 10's policy is baked in
# UNCONDITIONALLY, so the bits do not depend on the build mode) -- exp
# underflows straight to zero at the FLT_MIN boundary and log treats a
# flushed input as zero.


def portable_expf(x: Float32) -> Float32:
    """`expf` as one arithmetic. Out-of-range saturates: above 88.722835
    returns +inf, below -87.33655 returns 0.0 (everything below is
    subnormal, which row 10 flushes anyway), and a subnormal result at
    the boundary itself is flushed explicitly so the underflow edge is
    the same bit on FTZ and denormal-honoring backends alike."""
    from std.math import floor
    from std.memory import bitcast

    if x != x:  # NaN propagates untouched
        return x
    if x > Float32(88.722835):
        return bitcast[DType.float32](UInt32(0x7F800000))  # +inf
    if x < Float32(-87.33655):
        return Float32(0.0)

    # k = round(x / ln2); r = x - k*ln2 via the split constant, each step
    # one rounding
    var t = x * Float32(1.4426950408889634)
    t = t + Float32(0.5)
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

    # scale by 2^k in two exact power-of-two multiplies so k = 128 and
    # k = -126 both stay inside the exponent field
    var k1 = k >> 1
    var k2 = k - k1
    y = y * bitcast[DType.float32](UInt32((k1 + 127) << 23))
    y = y * bitcast[DType.float32](UInt32((k2 + 127) << 23))
    # the underflow edge: a denormal-honoring backend can produce a
    # gradual-underflow subnormal here where Metal already flushed; one
    # explicit flush makes the edge one bit everywhere
    if y < Float32(1.1754943508222875e-38):
        return Float32(0.0)
    return y


def portable_logf(x_in: Float32) -> Float32:
    """`logf` as one arithmetic. log(0) is -inf, log(negative) is NaN,
    log(+inf) is +inf; a subnormal input is flushed to zero FIRST (row
    10's policy), so it returns -inf like the zero it is on an FTZ
    backend."""
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

    # frexp by bits: m in [0.5, 1), then fold to [sqrt(1/2), sqrt(2))
    var bits = rebind[UInt32](x.to_bits())
    var e = Int((bits >> 23) & UInt32(0xFF)) - 126
    var m = bitcast[DType.float32](
        (bits & UInt32(0x007FFFFF)) | UInt32(0x3F000000)
    )
    if m < Float32(0.7071067811865476):
        e -= 1
        m = m + m
    var t = m - Float32(1.0)
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
    """Row 12's seam call. IDENTICAL routes through `portable_expf` (one
    arithmetic everywhere); FAST is the stdlib's device path verbatim, so
    Apple FAST bits do not move."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_expf(x)
    from std.math import exp

    return exp(x)


def identical_log(x: Float32) -> Float32:
    """Row 12's seam call, `log` side. Same two-arm contract as
    `identical_exp`."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_logf(x)
    from std.math import log

    return log(x)


# ============ DEVIATION 258 (2026-08-23): ROW 10 ON NVIDIA -- sqrt IS NOT
# ============ CORRECTLY ROUNDED; ROW 12's REMAINING DEVICE TRANSCENDENTALS,
# ============ cos AND pow. Consumers routed in the same commit:
# ============ compute_scores.mojo (symmetric Cosine, 3 sites),
# ============ pointwise_scores.mojo (pointwise Cosine 2 sites + the
# ============ log(weight+1) score terms), random_gen.mojo (Box-Muller
# ============ sqrt/log/cos, next_poisson's log and sqrt),
# ============ bootstrap.mojo (Bayesian -log(u) and tmp**temperature).
# ============ Gate: check-portable-sqrtcos. E2 round 1 measured the
# ============ consequence at c7a3493: Apple<->NVIDIA 43 GBDT cells
# ============ OUTPUT-ONLY@winners.scores and 15 DIVERGENT (argmax flips on
# ============ tied Quantile-class scores, Bayesian/Lq first histograms),
# ============ every ET/RF cell IDENTICAL.
# `check-ieee-arith` on the H100 (E2, 2026-08-23): sqrt 180,714 of 2^20
# hashed patterns off by one ulp -- 176,577 of them NORMAL, so this is not
# the denormal policy -- while div is 0 wrong and fma exact; the Cosine
# score shape was 141,895 wrong. Mojo's `std.math.sqrt` lowers to an
# APPROXIMATE PTX sqrt there, where Metal and HIP are correctly rounded,
# so every `score / sqrt(denum_sqr)` differs on the NVIDIA column only:
# that is E1's unexplained `tree001.winners.scores` NVIDIA divergence
# (E1_RESULTS.md), named. Row 10's sentence "IEEE-correct on normals
# everywhere measured" was true of two vendors and false of the third.
#
# The closure has row 12's shape: ONE arithmetic from the basic ops that
# ARE correctly rounded on every column measured (add/mul/div/fma), so the
# bits are the same everywhere the same bits go in. `portable_sqrtf` is
# additionally CORRECTLY ROUNDED by construction (candidate selection on
# the exact-enough fma residual), measured 0 mismatches against a float64
# reference over 2^20 patterns (`check-portable-translog`), so under
# IDENTICAL the Metal/HIP bits do not move either. `portable_cosf` is
# Cephes cosf (Cody-Waite three-part pi/4 reduction, the sin/cos
# polynomials) for the ONE consumer on the tree paths -- Box-Muller's
# `cos(2*pi*u)` in `gbdt/gpu_util/kernel/random_gen.mojo`, |x| < 2*pi.
# `portable_powf` is `exp(p * log(x))` through the two row-12 functions,
# for the Bayesian bootstrap's `tmp ** bagging_temperature` when the
# temperature is not 1. Every step below is an explicit fma or a single
# basic op: the reductions are written as fma ON PURPOSE so no backend's
# codegen can contract (row 9) them differently.


def portable_sqrtf(x_in: Float32) -> Float32:
    """`sqrtf` as one arithmetic, correctly rounded. NaN and negative ->
    quiet NaN; +-0 -> itself; +inf -> +inf; a subnormal input is flushed
    to +0 first (row 10's policy) and returns +0."""
    from std.memory import bitcast

    var x = x_in
    if x != x:
        return x
    if x < Float32(0.0):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    if abs(x) < Float32(1.1754943508222875e-38):
        return Float32(0.0)  # +-0 and subnormals
    if x == bitcast[DType.float32](UInt32(0x7F800000)):
        return x
    # TINY INPUTS ARE SCALED UP FIRST: below 2^-96 the selection residual
    # x - c*c (~2^-23 x) would fall under the normal range and round or
    # flush to zero, blinding the pick (measured: 968 of 2^20 lanes wrong
    # before this, every one with x < 2^-100). 2^64 in and 2^-32 out are
    # exact.
    var scaled_down = False
    if x < bitcast[DType.float32](UInt32(0x0F800000)):  # 2^-96
        x = x * bitcast[DType.float32](UInt32(0x5F800000))  # 2^64
        scaled_down = True
    # exponent-halving seed (~3.5% error), then three Heron steps through
    # the correctly-rounded divide: 3.5% -> 6e-4 -> 2e-7 -> below an ulp
    var bits = rebind[UInt32](x.to_bits())
    var y = bitcast[DType.float32]((bits >> 1) + UInt32(0x1FBD1DF5))
    y = Float32(0.5) * (y + x / y)
    y = Float32(0.5) * (y + x / y)
    y = Float32(0.5) * (y + x / y)
    # candidate selection: the root is within two ulps of y after three
    # rounded Heron steps; pick the neighbour in y-2..y+2 with the
    # smallest |x - c*c|, each residual one fma (a single rounding of a
    # quantity ~2^-23 smaller than the gap between neighbours' residuals,
    # so the order is exact and the pick is the correctly rounded root)
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
    """`cosf` as one arithmetic: Cephes single-precision cosf over the
    Cody-Waite reduction's domain |x| < 8192. Measured: <= 2 ulp on the
    tree paths' one consumer range (Box-Muller's `cos(2*pi*u)`, |x| <
    2*pi) and <= 8 ulp across the domain (the ulp count grows near the
    zeros of cos, where the result is tiny -- the ABSOLUTE error stays at
    the 1e-7 level). NaN and +-inf return quiet NaN."""
    from std.math import floor
    from std.memory import bitcast

    if x_in != x_in or abs(x_in) == bitcast[DType.float32](UInt32(0x7F800000)):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    var x = abs(x_in)
    # j = floor(x * 4/pi), made even, mod 8
    var j = Int(floor(x * Float32(1.27323954473516)))
    var yj = Float32(j)
    if (j & 1) != 0:
        j += 1
        yj = yj + Float32(1.0)
    j = j & 7
    # three-part pi/4 reduction, each step one fma (one rounding)
    x = _fma_f32(yj, Float32(-0.78515625), x)
    x = _fma_f32(yj, Float32(-2.4187564849853515625e-4), x)
    x = _fma_f32(yj, Float32(-3.77489497744594108e-8), x)
    var sign = Float32(1.0)
    if j > 3:
        j -= 4
        sign = -sign
    if j > 1:
        sign = -sign
    var z = x * x
    var y: Float32
    if j == 1 or j == 2:
        # sin polynomial: x + x*z*P(z)
        var p = Float32(-1.9515295891e-4)
        p = _fma_f32(p, z, Float32(8.3321608736e-3))
        p = _fma_f32(p, z, Float32(-1.6666654611e-1))
        y = _fma_f32(x * z, p, x)
    else:
        # cos polynomial: 1 - z/2 + z*z*P(z)
        var p = Float32(2.443315711809948e-5)
        p = _fma_f32(p, z, Float32(-1.388731625493765e-3))
        p = _fma_f32(p, z, Float32(4.166664568298827e-2))
        y = _fma_f32(z * z, p, _fma_f32(Float32(-0.5), z, Float32(1.0)))
    return sign * y


def portable_powf(x: Float32, p: Float32) -> Float32:
    """`powf(x, p)` for x > 0 as `exp(p * log(x))` through the two row-12
    functions -- the Bayesian bootstrap's `tmp ** bagging_temperature`
    (`gbdt/gpu_util/kernel/bootstrap.mojo`) is the consumer, and its
    `tmp` is `-log(u + 1e-20) > 0`. x == 0 returns 0 for p > 0 (and +inf
    for p < 0, 1 for p == 0); negative x is the consumer's bug and
    returns NaN. The bits are the same everywhere; accuracy is the
    composition's (a few ulp)."""
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
    """HOST-ONLY (no float64 on device, `mojolearn-hardware-limits`):
    `exp` for Float64 as one arithmetic, for the PROBABILITY LINKS --
    the sigmoid/softmax CatBoost computes in double
    (`eval_processing.h:186-226`) and this port's
    `one_vs_all_probabilities` / `multiclass_probabilities` / the Python
    wrapper's Logloss sigmoid mirror. Each host libm rounds double exp
    differently in the last bit (macOS Accelerate vs glibc), so a
    probability computed through `std.math.exp` carried the host's bit
    even when every raw margin matched (E2 round 1: gbdt_logloss
    DIVERGENT@proba, cards identical). Cephes `exp` for double: 2^k
    reduction with the split ln2, the (P2, Q3) rational on r^2, one fma
    per step; ~1 ulp of double, which after the cast to float32 is the
    correctly-rounded float32 probability in all but double-rounding
    corner cases."""
    from std.math import floor, fma
    from std.memory import bitcast

    if x != x:
        return x
    if x > 709.782712893384:
        return bitcast[DType.float64](UInt64(0x7FF0000000000000))  # +inf
    if x < -708.3964185322641:
        return Float64(0.0)
    var k = floor(x * 1.4426950408889634 + 0.5)
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
    # scale by 2^k in two exact steps so |k| up to 1074 stays in range
    var ki = Int(k)
    var k1 = ki >> 1
    var k2 = ki - k1
    y = y * bitcast[DType.float64](UInt64(k1 + 1023) << 52)
    y = y * bitcast[DType.float64](UInt64(k2 + 1023) << 52)
    return y


def identical_exp64(x: Float64) -> Float64:
    """The probability links' seam call (host): IDENTICAL routes through
    `portable_exp64`, FAST is the host stdlib verbatim."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_exp64(x)
    from std.math import exp

    return exp(x)


def portable_log64(x_in: Float64) -> Float64:
    """HOST-ONLY (no float64 on device): `log` for Float64 as one
    arithmetic, the sibling of `portable_exp64` for the same reason (each
    host libm rounds double log differently in the last bit). Asked for
    by the metrics lane (DEVIATION 651, RAFT's double-precision seams) and
    the KDE lane (DEVIATION 601's normalization constant), 2026-08-23.

    Cephes `log` for double, both arms: `frexp` to m in [0.5, 1), then for
    |e| <= 2 the (P5, Q5) rational in x = m - 1 (or 2m - 1 below sqrt(1/2)),
    else the (R2, S3) rational in x = (m - 1)/(m + 1) (z = x^2); the
    exponent re-enters as e * ln2 in two pieces (0.693359375 and
    -2.121944400546905827679e-4) so the large-|e| arm keeps its bits. One
    `fma` per Horner step, every other op one rounding. ~1 ulp of double.
    Domain: NaN -> NaN; +inf -> +inf; 0 -> -inf; negative -> NaN; a
    denormal input is scaled by 2^54 first."""
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
        # denormal: scale up by 2^54, remember it
        x = x * 18014398509481984.0
        bits = bitcast[DType.uint64](x)
        e = -54
    # frexp: m in [0.5, 1), x = m * 2^e
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
        # R(z) / S(z)
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
    """The host double-log seam call: IDENTICAL routes through
    `portable_log64`, FAST is the host stdlib verbatim (the `identical_exp64`
    contract)."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_log64(x)
    from std.math import log

    return log(x)


def identical_sqrt(x: Float32) -> Float32:
    """Row 10's sqrt seam call: IDENTICAL routes through `portable_sqrtf`
    (one arithmetic, correctly rounded, the same bits on the approximate-
    sqrt column too); FAST is the stdlib's device path verbatim."""
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
    """Row 12's `pow` seam call (Bayesian bootstrap temperature). Same
    two-arm contract; FAST is the `**` the site used."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_powf(x, p)
    return x**p


# ============ THE SIMD SPELLINGS OF THE SAME TWO CONSTRUCTIONS ===========
# Added 2026-08-23 by the UNSUPERVISED IDENTITY lane (IDENTITY_PATHS rows
# 19-24). They are not a new idea and must never become one: each is the
# scalar function above applied lane by lane, and any divergence between a
# scalar site and a SIMD site would be a second arithmetic wearing the same
# name.
#
# They exist because the distance kernels this repository actually ships --
# `cluster/.../fused_distance_nn/simt_kernel.mojo`,
# `neighbors/.../fused_l2_knn.mojo`, `dbscan/.../epsilon_neighborhood.mojo`
# -- carry their accumulators as `SIMD[DType.float32, AccColsPerTh]`,
# because that is what RAFT's `Policy4x4` register tile IS. Writing
# `identical_mul_add` at those seams one lane at a time would either
# scalarize the register tile or need a loop the codegen has to re-vectorize.


def ftz_simd[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """`ftz`, lane by lane. Comptime no-op under FAST.

    Same model the `check-ieee-arith` ftz arm reproduced all 53,041 Metal
    divergences with: a denormal operand or result flushes to a zero
    CARRYING ITS SIGN, everything else is untouched.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        # LANE BY LANE THROUGH THE SCALAR FUNCTION, not a re-derivation of
        # it in SIMD predicates. The invariant this file needs is that a
        # scalar seam and a SIMD seam flush IDENTICALLY, and the cheapest
        # way to guarantee that is for one of them to BE the other. The
        # loop is comptime, so it unrolls to `w` copies of the same three
        # instructions.
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
    """`identical_mul_add`, lane by lane: ONE rounding under IDENTICAL, the
    naive chain under FAST."""
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
