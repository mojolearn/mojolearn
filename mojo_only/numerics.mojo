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
#:
#: (This paragraph documented `NUMERIC_FAST` but sat above
#: `GLOBAL_NUMERIC_MODE`, glued to the switch's own block with no blank line
#: between them, so the doc tool attached it to the switch and
#: `NUMERIC_FAST` carried no documentation at all. Moved 2026-08-24 by the
#: transformer-numerics lane; no code line changed.)
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


def identical_mul(a: Float32, b: Float32) -> Float32:
    """DEVIATION 826 (2026-08-24): the OTHER half of row 9's contraction
    pin, and it belongs beside `identical_mul_add` rather than in three
    other directories.

    A multiply no codegen may contract into a neighboring add, spelled
    `identical_mul_add(a, b, -0.0)`. Under IDENTICAL that is
    `fma(a, b, -0.0)`, which is bit-equal to the correctly rounded
    product at EVERY input including both zero signs (`p + (-0.0) == p`
    for a p of either zero sign under round-to-nearest) and which
    presents no syntactic multiply for a compiler to contract. Under
    FAST it is `a * b + (-0.0)`, the plain product.

    THE `-0.0` ADDEND IS LOAD-BEARING AND A `+0.0` WOULD BE A BUG:
    `(-0.0) + (+0.0)` is `+0.0` under round-to-nearest, so a `+0.0`
    addend LAUNDERS a negative zero product into a positive one. That is
    the gemm lane's fixture F6a lesson. The zero sign is not cosmetic
    here -- it propagates through every downstream product and the mamba
    corpus's `adv_signed_zeros` case compares 585 zero cells BY SIGN BIT
    precisely because a tolerance check cannot see it.

    WHY `var p = a * b` FOLLOWED BY `p + c` IS NOT ENOUGH: the gemm
    README's F3 scar records that spelling being CONTRACTED ACROSS
    STATEMENTS on this host. A separate statement is not a barrier.

    NOT A NEW CONSTRUCTION. DEVIATION 720 introduced it and three files
    carry identical copies under the name `pinned_mul`:
    `mamba/mojo_only/mamba_oracle.mojo:41`,
    `mamba/ported/mamba_ssm/ops/selective_scan_interface.mojo:263` and
    `mamba/ported/transformers/models/mamba/modeling_mamba.mojo:207`.
    All three agree bit for bit with this one, which is the good case and
    also the fragile one -- four copies of an arithmetic have four
    chances to drift. This is the canonical home; the three may later be
    pointed here, and that is their owners' call, not this file's."""
    return identical_mul_add(a, b, Float32(-0.0))


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
# purchased is not accuracy but SAMENESS. Neither ever produces a subnormal
# output (row 10's policy is baked in UNCONDITIONALLY, so the bits do not
# depend on the build mode): exp underflows straight to zero at the FLT_MIN
# boundary. `logf` FLUSHES ITS SUBNORMAL INPUT and then treats it as the
# zero it flushed to, returning -inf.
#
# THE SENTENCE HERE USED TO SAY "Both functions flush subnormal inputs",
# which is false of `expf` -- it has no input flush and needs none.
# Corrected 2026-08-24 by the transformer-numerics lane, WITH the reason,
# because "needs none" is the interesting half: a subnormal argument is
# absorbed by `t + 0.5` (so `floor` gives 0 and k = 0), by the `q` chain
# (whose last coefficient is 0.5) and finally by `y + 1.0`, so exp returns
# exactly 1.0 on an FTZ column and on a denormal-honoring one alike. The
# new functions below are NOT all like that -- `sinf`, `tanhf` and `erff`
# each return a value proportional to their argument near zero, so a
# subnormal input survives to the output there and each one DOES flush
# (DEVIATIONS 820-822).


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
    return _cephes_logf_core(t, e)


def _cephes_logf_core(t: Float32, e: Int) -> Float32:
    """The Cephes logf mantissa polynomial and exponent re-entry, ONE
    source for two callers: `portable_logf` (t = m - 1 on [sqrt(1/2),
    sqrt(2)), e the unbiased exponent) and `portable_log1pf`'s small-x
    branch (t = x itself, e = 0, DEVIATION 742). Factored out 2026-08-23
    with the op sequence UNCHANGED -- `check-portable-translog`'s device
    hash 8705486125800438413 is the proof that `portable_logf`'s bits did
    not move."""
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
# polynomials); its first consumer on the tree paths was Box-Muller's
# `cos(2*pi*u)` in `gbdt/gpu_util/kernel/random_gen.mojo`, |x| < 2*pi. It
# is no longer the only one and it no longer holds the arithmetic: the
# reduction, the octant bookkeeping and BOTH polynomials moved into
# `_cephes_sincosf_core` on 2026-08-24 so that `portable_sinf` (DEVIATION
# 820, RoPE) shares them instead of carrying a second copy. The op
# sequence for the cos side is UNCHANGED -- see that function's docstring
# for the argument and for what re-running the gate proves.
# `portable_powf` is `exp(p * log(x))` through the two row-12 functions,
# for the Bayesian bootstrap's `tmp ** bagging_temperature` when the
# temperature is not 1. Every step below is an explicit fma or a single
# basic op: the reductions are written as fma ON PURPOSE so no backend's
# codegen can contract (row 9) them differently.


def portable_sqrtf(x_in: Float32) -> Float32:
    """`sqrtf` as one arithmetic, correctly rounded. NaN and negative
    normals -> quiet NaN; +inf -> +inf; +0, -0 AND every subnormal OF
    EITHER SIGN return +0 (the flush branch below catches them all -- so
    sqrt(-0) is +0 here where IEEE says -0; this sentence used to read
    "+-0 -> itself", corrected 2026-08-23 by the numerics-NN lane, which
    plants it through `portable_rsqrtf(-0) = +inf`)."""
    from std.memory import bitcast

    var x = x_in
    if x != x:
        return x
    # THE FLUSH BRANCH BEFORE THE SIGN TEST (reordered 2026-08-23 by the
    # numerics-NN lane): Metal flushes COMPARE operands, so with the sign
    # test first a NEGATIVE SUBNORMAL took `x < 0` on the host (NaN) and
    # not on the device (+0) -- 2,046 of 2^20 raw-pattern lanes in
    # check-portable-nn's rsqrt sweep, the first sweep to feed this
    # function negative subnormals. Flush first and both columns return
    # +0. No other input's bits move (check-portable-sqrtcos hash
    # 12295913102197186379 unchanged).
    if abs(x) < Float32(1.1754943508222875e-38):
        return Float32(0.0)  # +-0 and subnormals, either sign
    if x < Float32(0.0):
        return bitcast[DType.float32](UInt32(0x7FC00000))
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
    tree paths' first consumer range (Box-Muller's `cos(2*pi*u)`, |x| <
    2*pi) and <= 8 ulp across the domain (the ulp count grows near the
    zeros of cos, where the result is tiny -- the ABSOLUTE error stays at
    the 1e-7 level). NaN and +-inf return quiet NaN.

    THE BODY MOVED INTO `_cephes_sincosf_core` ON 2026-08-24 and the op
    sequence is UNCHANGED, exactly as `_cephes_logf_core` was factored out
    of `portable_logf` on 2026-08-23. This function is already certified
    and other lanes depend on its bits, so the argument that they cannot
    have moved is written out rather than asserted:

    - Every arithmetic step in the core is an explicit `_fma_f32` or ONE
      basic op. There is no `a*b + c` expression left for any codegen to
      contract or not contract, so the factoring cannot change a rounding
      count the way an inlining decision could.
    - The core is entered with `sign_in = 1.0` and `is_cos = True`, which
      makes `if is_cos and j > 1` the old unconditional `if j > 1` and
      `use_sin_poly` the old `j == 1 or j == 2`. Same branches, same
      order, same operands.
    - Everything that differs between the sin and cos entries is Int or
      Bool bookkeeping. No float value is computed differently and no
      float value is computed in a different order.
    - The argument is CHECKABLE, not merely stated: re-run
      `pixi run check-portable-sqrtcos` and its printed
      `sqrtcos device hash` must still be 12295913102197186379. That is
      the same proof `_cephes_logf_core` gave with the translog hash
      8705486125800438413.

    Its subnormal input is NOT flushed, and that is deliberate rather than
    an oversight: cos of any subnormal is exactly 1.0 on both an FTZ and a
    denormal-honoring column (`x*x` underflows to +0 either way, so the
    polynomial reduces to `fma(0, p, fma(-0.5, 0, 1.0))`), so a flush here
    would be bit-inert and would still be a change to a certified
    function for no gain. `portable_sinf` is the opposite case and DOES
    flush -- see DEVIATION 820."""
    from std.memory import bitcast

    if x_in != x_in or abs(x_in) == bitcast[DType.float32](UInt32(0x7F800000)):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    return _cephes_sincosf_core(abs(x_in), Float32(1.0), True)


def _cephes_sincosf_core(
    x_abs: Float32, sign_in: Float32, is_cos: Bool
) -> Float32:
    """Cephes single-precision `sinf`/`cosf` from `single/sinf.c`: the
    octant computation, the three-part Cody-Waite pi/4 reduction and BOTH
    polynomials, ONE source for two callers (`portable_cosf`, and
    `portable_sinf` from DEVIATION 820). Domain |x_abs| < 8192, which is
    Cephes's own `lossth`; above it the reduction loses the argument and
    this function does not guard, exactly as it did not before the
    factoring.

    Cephes ships sin and cos as two functions that differ in THREE places
    and agree everywhere else, which is why one core with two flags is the
    honest shape rather than a clever one:

      (1) cos starts from `sign = +1` and takes |x|; sin carries the
          argument's own sign into `sign`. That is `sign_in`.
      (2) cos flips the sign a second time on `j > 1`; sin does not.
          That is `is_cos` at the second flip.
      (3) the octant selects the OTHER polynomial: on j in {1, 2} cos
          evaluates the SIN polynomial and sin evaluates the COS one.
          That is `is_cos` at `use_sin_poly`.

    Two DEVIATIONS from Cephes as written, both inherited from the
    already-certified cos path and both deliberate:

      - the reduction is three `fma`s (three roundings) where Cephes
        writes `((x - y*DP1) - y*DP2) - y*DP3` (six). Row 9: a pinned seam
        is written as an explicit `fma` so no backend's codegen can decide
        the contraction for us.
      - the Horner steps and the two polynomial tails are `fma` for the
        same reason. The ASSOCIATION is Cephes's own: the sin tail is
        `p*(z*x) + x`, which is Cephes's `y *= z*x; y += x`, fused.

    ACCURACY IS NOT THE POINT HERE, AGREEMENT IS. Cephes measures its own
    sinf at 3.8e-9 theoretical relative error on [-pi/4, pi/4] and its
    cosf at 8.3e-8 peak; the fma spelling is at least that good but has
    only been MEASURED for the cos side (<= 2 ulp on |x| < 2*pi, <= 8 ulp
    across the domain). The sin side's ulp bound is NOT MEASURED YET and
    is not guessed here -- `check-portable-trig` owes it."""
    from std.math import floor

    var x = x_abs
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


# ============ DEVIATIONS 740-746 (2026-08-23): THE NEURAL-BLOCK PRIMITIVES
# ============ -- division CHARACTERIZED and named, rsqrt, log1p, sigmoid,
# ============ silu, softplus. Built for the Mamba-1 identity block
# ============ (IDENTICAL_SSM_NOTES.md "build order" item 1, promoted by
# ============ Andrew 2026-08-23); the mamba lane imports these, nothing
# ============ else in the tree calls them yet. Gates: check-division (row
# ============ 49) and check-portable-nn (rows 50-54). No performance work.
#
# Every function here is built the way `portable_expf`/`portable_sqrtf`
# are: one spelling from one source, every step an explicit fma or one
# basic op, row 10's flush policy baked in UNCONDITIONALLY (the bits of a
# portable_* function never depend on the build mode), special values
# PLANTED (row 39) and asserted by the gate, identical bits host and
# device, a printed certificate hash the H100/MI325X legs re-print. Each
# has an `identical_*` seam: portable under IDENTICAL, the stdlib/hardware
# spelling verbatim under FAST.
#
# The reference for the spellings is mamba_ssm at e9594ce
# (/Users/andrewhendel/CascadeProjects/upstream/mamba):
#   softplus  csrc/selective_scan/selective_scan_fwd_kernel.cuh:160
#             `delta <= 20.f ? log1pf(expf(delta)) : delta`
#   silu      csrc/selective_scan/selective_scan_fwd_kernel.cuh:298
#             `z_val / (1 + expf(-z_val))`  (ONE division, no multiply --
#             the same spelling ATen's silu kernel uses, which is what
#             `selective_scan_ref`'s `F.silu(z)` reaches)
#   sigmoid   mamba_ssm/ops/triton/k_activations.py:42 `tl.sigmoid(x)`
#             (= 1 / (1 + exp(-x)))
#   rsqrt     mamba_ssm/ops/triton/layer_norm.py:120 (reference) and :272
#             (Triton kernel): `1 / sqrt(var + eps)` -- the reference
#             never calls an rsqrt intrinsic, so neither do we
#   log1p     the `log1pf` above; Triton 3 spells it `log(exp(dt) + 1)`
#             (mamba_ssm/ops/triton/softplus.py:11), we do not
#
# DEVIATION 740 -- DIVISION, row 10's open clause, CHARACTERIZED on Apple
# by `check-division` (2^20 hashed (a, b) pairs in 13 classes: normals,
# each-operand-subnormal, both-subnormal, zero dividend, zero divisor,
# inf, NaN, quotient overflow, quotient underflow to a subnormal, to
# zero, exact power-of-two quotients, raw patterns). Apple M4 through MAX:
# the NORMAL class is correctly rounded (0 wrong), every subnormal-touching
# class is reproduced bit for bit by row 10's flush model (operands flushed
# to signed zero, one correctly rounded division, result flushed), zero
# lanes match neither. So on Apple `a / b` IS correctly rounded on every
# class once the flush is the reference, and `portable_divf` below is
# plain `/` wrapped in that flush -- bit-inert on Apple, aligning on a
# denormal-honoring column. NOT claimed for NVIDIA or AMD until a leg
# re-prints the certificate hash: `check-ieee-arith` measured div 0 wrong
# on the H100 and MI325X over ITS patterns, but "correctly rounded
# everywhere measured" was false once already (sqrt, DEVIATION 258), so
# the named seam exists so a vendor that is wrong has one place to be
# fixed (a Newton-refined reciprocal with an fma correction step would go
# here, under IDENTICAL only).
#
# DEVIATION 741 -- rsqrt PIN (a): `1 / portable_sqrtf(x)`, two correctly
# rounded operations, NOT the correctly rounded rsqrt: measured on Apple
# against the float64 1/sqrt rounded once, max 1 ulp, off on 134,858 of
# 520,133 positive-normal lanes in check-portable-nn (187,068 of 720,414
# in check-division's D arm). Chosen over (b) a correctly-rounded-by-
# construction rsqrt because the reference's spelling IS `1 / sqrt(...)`
# (layer_norm.py:120, :272), the contract is "one fixed function of the
# input bits" and not "correctly rounded", and (b) is NOT cheap: the
# residual `x*c*c - 1` needs a split product (two fmas round the product
# once too often to order the neighbours), a second candidate loop for no
# identity gain. RECORDED (DEVIATION 746) and it is worth knowing: Metal's
# `rsqrt` INTRINSIC matched the float64 1/sqrt on 720,414 of 720,414
# positive normals (0 wrong, max 0 ulp) -- Apple's intrinsic is the
# correctly rounded one and the pin is the one that is not. It is still
# never called on a pinned path: an intrinsic is per-vendor by definition
# (PTX `rsqrt.approx` is ~2 ulp), and the pin's property is sameness.
#
# DEVIATION 742 -- log1p: the Cephes logf polynomial applied to t = x
# directly (e = 0, no 1+x formed, no cancellation) on the polynomial's own
# design domain x in [sqrt(1/2)-1, sqrt(2)-1] = [-0.29289..., 0.41421...],
# and `portable_logf(1 + x)` (one rounding on 1+x, then the certified log)
# outside it. The branch points are those two constants as Float32
# (0xBE95F61A and 0x3ED413CD), compared `<=` on both sides. Cephes single
# ships no log1pf; this IS its log kernel, one source, not a new table.
#
# DEVIATION 743 -- sigmoid: `1 / (1 + portable_expf(-x))` through
# `portable_divf`. Large negative x: exp overflows to +inf, 1/+inf = +0.0
# (planted, x <= -88.72). The band x in (-88.72, -87.34]: exp(-x) is
# finite in [2^126, 2^128), 1/(1+e) is a SUBNORMAL quotient -- the flush
# in `portable_divf` returns +0.0 on every column where a denormal-
# honoring backend would keep 2^-127 (planted at x = -88). Large positive
# x: exp(-x) underflows or is absorbed, 1 + 0 = 1, result exactly 1.0
# (planted at x >= 17 and +inf).
#
# DEVIATION 744 -- silu: the REFERENCE'S spelling `x / (1 + portable_expf(-x))`,
# one division through `portable_divf`, NOT `x * sigmoid(x)`: the two
# differ in the last bit (one rounding against two), the reference kernel
# and ATen both spell it as the single quotient, and Andrew's order is "do
# what the reference does". Consequences planted: silu(-inf) = -inf/+inf =
# NaN (the reference's too); x <= -88.72 gives x/+inf = -0.0 (SIGNED);
# silu(+-0) = +-0; silu(+inf) = +inf; a subnormal x is flushed by the
# division's operand flush and returns a signed zero.
#
# DEVIATION 745 -- softplus: `x <= 20 ? log1p(exp(x)) : x`, the reference
# kernel's guard verbatim (selective_scan_fwd_kernel.cuh:160). The guard
# boundary is planted both sides: x = 20.0 takes the log1p(exp) arm, the
# next float above 20 returns itself; NaN fails `<=` and returns itself;
# -inf -> exp 0 -> log1p(0) = +0; x < -87.34 -> +0.
#
# DEVIATION 746 -- the hardware `rsqrt` intrinsic (`std.math.rsqrt`) is
# RECORDED by check-division (mismatch count and max ulp against the
# float64 1/sqrt, and against the pin), never used. Apple M4: 0 of
# 720,414 off the float64 answer; off the pin on 187,068; hash
# 8605842820492558534 (per-vendor, not a certificate).
#
# ALSO FOUND BUILDING THESE (2026-08-23), both about row 10 on Metal:
#   (i) METAL FLUSHES COMPARE OPERANDS: `-subnormal < 0.0` is false there,
#       so `portable_sqrtf`'s sign test ran BEFORE its flush branch and a
#       negative subnormal returned NaN on the host and +0 on the device
#       (2,046 of 2^20 lanes, the first sweep to feed it one); the two
#       branches are reordered (flush first, both columns +0; the sqrtcos
#       certificate hash did not move) and `_ftz_always` flushes BY BITS.
#  (ii) `std.math.log1p` lowers Float32 through FLOAT64 and Metal cannot
#       compile a kernel that calls it; the FAST arms spell `log(1 + x)`.


def _ftz_always(x: Float32) -> Float32:
    """Row 10's flush, UNCONDITIONAL and BY BITS: the portable_* functions
    below bake the policy in so their bits do not depend on the build mode
    (the same choice `portable_expf`/`portable_logf` made inline). A
    subnormal becomes a zero carrying its sign; zero, normals, infinities
    and NaN pass through untouched. `ftz` above is the mode-gated seam
    spelling; this one is for inside a portable function only.

    BY BITS, NOT BY COMPARE, because METAL FLUSHES COMPARE OPERANDS TOO
    (measured 2026-08-23, check-portable-nn's first device run): on Metal
    `-subnormal < 0.0` is FALSE and `subnormal != 0.0` is FALSE, so a
    compare-written flush returns the subnormal bit pattern unchanged and
    a sign test before the flush takes the wrong branch (2,046 of 2^20
    rsqrt lanes: host NaN, device +inf, every one a negative subnormal).
    Integer ops do not flush. The mode-gated `ftz` above is compare-
    written and therefore bit-inert on Metal for a PASS-THROUGH subnormal
    (one loaded from memory and stored with no arithmetic in between);
    that is named in IDENTITY_PATHS row 49 and left to its owner."""
    from std.memory import bitcast

    var b = rebind[UInt32](x.to_bits())
    if (b & UInt32(0x7F800000)) == UInt32(0):
        return bitcast[DType.float32](b & UInt32(0x80000000))
    return x


def portable_divf(a: Float32, b: Float32) -> Float32:
    """DEVIATION 740: `a / b` under row 10's flush model -- operands
    flushed to signed zero, ONE hardware division (correctly rounded on
    every column measured: Apple normal class 0 wrong over 2^20 pairs in
    `check-division`; the H100/MI325X `check-ieee-arith` div arms 0
    wrong), result flushed. On an FTZ column this is bitwise `a / b`; on a
    denormal-honoring column it aligns the subnormal classes to the FTZ
    ones. Planted: 1/+0 = +inf, 1/-0 = -inf, +-0/x = +-0 (sign rule),
    x/inf = signed 0, inf/inf and 0/0 = NaN, a subnormal operand divides
    as the signed zero it flushes to (so subnormal/normal = signed 0 and
    normal/subnormal = signed inf), a subnormal quotient returns signed 0."""
    return _ftz_always(_ftz_always(a) / _ftz_always(b))


def portable_rsqrtf(x_in: Float32) -> Float32:
    """DEVIATION 741, pin (a): `1 / portable_sqrtf(x)`. NaN -> NaN;
    negative -> quiet NaN; +0, -0 and every subnormal -> +inf (the sqrt
    flushes all three to +0 -- so rsqrt(-0) is +inf here where IEEE says
    -inf; planted); +inf -> +0. Never a subnormal result (1/sqrt of any
    normal is in [2^-64, 2^63])."""
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
    """DEVIATION 742: `log1pf` as one arithmetic. NaN -> NaN; +-0 -> +-0
    (IEEE's sign rule, planted); a subnormal input is flushed first and
    returns the signed zero it flushes to; +inf -> +inf; x == -1 -> -inf;
    x < -1 -> quiet NaN. On [-0.29289..., 0.41421...] the Cephes logf
    polynomial is evaluated on t = x with e = 0 (no 1+x, no
    cancellation); outside, `portable_logf(x + 1)`. Measured <= 2 ulp
    against libm's double log1p over 2^20 hashed inputs
    (`check-portable-nn`)."""
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
    # the branch points, sqrt(1/2)-1 and sqrt(2)-1 as Float32:
    # 0xBE95F61A = -0.29289323, 0x3ED413CD = 0.41421357
    if x >= bitcast[DType.float32](UInt32(0xBE95F61A)) and x <= bitcast[
        DType.float32
    ](UInt32(0x3ED413CD)):
        return _cephes_logf_core(x, 0)
    return portable_logf(x + Float32(1.0))


def portable_sigmoidf(x: Float32) -> Float32:
    """DEVIATION 743: `1 / (1 + portable_expf(-x))` through `portable_divf`.
    NaN -> NaN; +-0 -> 0.5; x <= -88.72 -> +0.0 (exp overflow, 1/+inf);
    x in (-88.72, -87.34] -> +0.0 (a subnormal quotient, flushed);
    x >= ~17 and +inf -> exactly 1.0; -inf -> +0.0."""
    if x != x:
        return x
    var e = portable_expf(-x)
    var d = e + Float32(1.0)
    return portable_divf(Float32(1.0), d)


def portable_siluf(x: Float32) -> Float32:
    """DEVIATION 744: the reference's `x / (1 + portable_expf(-x))`, one
    division through `portable_divf` (NOT x * sigmoid(x)). NaN -> NaN;
    +-0 -> +-0; +inf -> +inf; -inf -> NaN (-inf/+inf, the reference's
    answer too); x <= -88.72 -> -0.0 (x/+inf, signed); a subnormal x ->
    its signed zero (operand flush)."""
    if x != x:
        return x
    var e = portable_expf(-x)
    var d = e + Float32(1.0)
    return portable_divf(x, d)


def portable_softplusf(x: Float32) -> Float32:
    """DEVIATION 745: `x <= 20 ? log1p(exp(x)) : x`, selective_scan_fwd_
    kernel.cuh:160 verbatim through the portable pair. NaN -> NaN (fails
    the `<=`); -inf -> +0; x < -87.34 -> +0 (exp underflow); x = 20.0
    takes the log1p(exp) arm; the next float above 20 returns itself;
    +inf -> +inf."""
    if x <= Float32(20.0):
        return portable_log1pf(portable_expf(x))
    return x


def identical_div(a: Float32, b: Float32) -> Float32:
    """Row 49's seam: IDENTICAL routes through `portable_divf` (the flush
    model around one correctly rounded division); FAST is plain `/`.
    Bit-inert on Apple in both modes; the value of the pin is a denormal-
    honoring column. NOT cross-vendor certified until a leg re-prints
    `check-division`'s certificate hash (the DEVIATION 258 lesson)."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_divf(a, b)
    return a / b


def identical_rsqrt(x: Float32) -> Float32:
    """Row 50's seam: IDENTICAL is `portable_rsqrtf`; FAST is the
    reference's own spelling `1 / sqrt(x)` through the stdlib sqrt (the
    vendor's sqrt -- approximate on NVIDIA, DEVIATION 258), never the
    rsqrt intrinsic."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_rsqrtf(x)
    from std.math import sqrt

    return Float32(1.0) / sqrt(x)


def identical_log1p(x: Float32) -> Float32:
    """Row 51's seam: IDENTICAL is `portable_log1pf`; FAST is `log(1 + x)`
    through the stdlib log -- Triton 3's own spelling of the same thing
    (mamba_ssm/ops/triton/softplus.py:11) -- and NOT `std.math.log1p`,
    because that one lowers the Float32 case THROUGH FLOAT64
    (`air.convert.f.f64.f.f32` ... `llvm.fma.f64`) and Metal refuses to
    compile a kernel that calls it (measured 2026-08-23 building
    check-portable-nn's first sabotage arm). A stdlib function that does
    not exist on one column is not a FAST arm."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_log1pf(x)
    from std.math import log

    return log(Float32(1.0) + x)


def identical_sigmoid(x: Float32) -> Float32:
    """Row 52's seam: IDENTICAL is `portable_sigmoidf`; FAST is
    `1 / (1 + exp(-x))` through the stdlib exp."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_sigmoidf(x)
    from std.math import exp

    return Float32(1.0) / (Float32(1.0) + exp(-x))


def identical_silu(x: Float32) -> Float32:
    """Row 53's seam: IDENTICAL is `portable_siluf`; FAST is the reference's
    `x / (1 + exp(-x))` through the stdlib exp."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_siluf(x)
    from std.math import exp

    return x / (Float32(1.0) + exp(-x))


def identical_softplus(x: Float32) -> Float32:
    """Row 54's seam: IDENTICAL is `portable_softplusf`; FAST is the
    reference's guard around Triton 3's `log(exp(x) + 1)` through the
    stdlib (not `log1p`: see `identical_log1p` for why it cannot be)."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_softplusf(x)
    from std.math import exp, log

    if x <= Float32(20.0):
        return log(exp(x) + Float32(1.0))
    return x


# ============ DEVIATIONS 820-825 (2026-08-24): THE TRANSFORMER-BLOCK
# ============ PRIMITIVES -- sin, tanh, erf, the TWO gelus, and a
# ============ deterministic elementwise max. DEVIATION 826 is the
# ============ pinned product `identical_mul` and lives with
# ============ `identical_mul_add` at the top of this file rather than
# ============ here, because that is where a reader looks for it;
# ============ 827-829 are RESERVED for this lane and unused. Built for
# ============ the Llama-shaped identity block (RMSNorm, RoPE, softmax
# ============ attention, SwiGLU MLP); the
# ============ transformer lane imports these, nothing else in the tree
# ============ calls them yet. Gates: check-portable-trig,
# ============ check-portable-gelu and check-portable-fmax, SPECIFIED at
# ============ the foot of this file and NOT YET WRITTEN -- until they
# ============ exist and a second vendor re-prints their hashes, nothing
# ============ here is certified and it must not be described as such.
# ============ No performance work.
#
# Everything the transformer block needs that this file ALREADY had is
# reused unchanged: `identical_mul_add` and `ftz` (row 9 and row 10),
# `portable_expf` (softmax), `portable_rsqrtf` (RMSNorm), `portable_divf`
# (every quotient), `portable_siluf` (SwiGLU), `portable_cosf` (RoPE's
# cosine half). This block adds only what an audit found MISSING.
#
# Every function here is built the way `portable_expf`/`portable_sqrtf`
# and the 740-746 family are: one spelling from one source, every step an
# explicit fma or one basic op, row 10's flush policy baked in
# UNCONDITIONALLY (the bits of a portable_* function never depend on the
# build mode), special values PLANTED (row 39) and asserted by the gate,
# identical bits host and device, a printed certificate hash the H100 and
# MI325X legs re-print. Each has an `identical_*` seam: portable under
# IDENTICAL, the stdlib/vendor spelling verbatim under FAST.
#
# TWO REFERENCES, AND WHICH ONE WINS WHERE. This is the DEVIATION 744
# rule applied to five more functions, so it is worth restating: the
# reference for the MODEL is HuggingFace transformers at d56c55b
# (/Users/andrewhendel/CascadeProjects/upstream/transformers) and the
# reference for the MATH is Cephes single precision (Moshier, netlib
# `cephes/single`, Release 2.2 June 1992), the same tables
# `portable_expf`/`portable_logf`/`portable_cosf` already carry. Where
# they disagree:
#
#   * the MODEL decides the OPERATION ORDER and the constants. If
#     HuggingFace rounds a product and then adds, we round a product and
#     then add -- we do NOT fuse it into an fma just because the fma is
#     more accurate, because the property being bought is agreement with
#     a spelling somebody wrote down, not accuracy.
#   * the MATH SOURCE decides the POLYNOMIAL. HuggingFace calls
#     `torch.erf` and `torch.tanh` and never says what is inside them, so
#     there is nothing to mirror; Cephes is a named table with a
#     published error bound and it is what the rest of this file already
#     uses.
#
# The model's spellings, quoted:
#   gelu (exact)  activations.py:85-86, `GELUActivation._gelu_python`
#                 `input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))`
#   gelu (tanh)   activations.py:47-48, `GELUTanh._gelu_tanh_python`
#                 `input * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi)
#                  * (input + 0.044715 * torch.pow(input, 3.0))))`
#                 and activations.py:65, `NewGELUActivation.forward`
#                 `0.5 * input * (1.0 + torch.tanh(math.sqrt(2 / math.pi)
#                  * (input + 0.044715 * torch.pow(input, 3.0))))`
#   Both gelus are registered and both ship: ACT2CLS maps "gelu" and
#   "gelu_python" to the erf form and "gelu_new" / "gelu_pytorch_tanh" /
#   "gelu_python_tanh" to the tanh form (activations.py:325-331).
#
# THE TWO GELUS ARE NOT THE SAME FUNCTION AND MUST NEVER BE CONFLATED.
# HuggingFace's own docstring says so in the file: the tanh form "is not
# an exact numerical match due to rounding errors" (activations.py:36-37).
# They disagree by ~1e-3 absolute around |x| = 2, which is four orders of
# magnitude above float32 epsilon -- far too large for a tolerance check
# to absorb and far too small for a wrong answer to look wrong. A config
# says which one a checkpoint wants; picking the other one silently is
# exactly the class of defect this repository's per-stage cards exist to
# catch, so they are two functions with two DEVIATION numbers and two
# seams.
#
# CONSTANTS ARE PINNED BY BITS. `String(float)` does not round-trip in
# Mojo (`mojo-string-float-roundtrip`), and all three model constants are
# computed in Python DOUBLE and then rounded to float32 by torch's scalar
# promotion, so a decimal literal in this file is a transcription and not
# a pin. The four below are written as bit patterns with the decimal in
# the comment, which is the spelling `portable_log1pf`'s branch points
# already use.


#: `math.sqrt(2.0)` = 1.4142135623730951 rounded to Float32 = 1.4142135
#: (`torch` promotes the Python double scalar to the tensor's dtype, so
#: the division HuggingFace performs is by this float32 and not by the
#: double). The divisor in `GELUActivation._gelu_python`.
comptime GELU_SQRT2_BITS: UInt32 = 0x3FB504F3

#: `math.sqrt(2.0 / math.pi)` = 0.7978845608028654 rounded to Float32 =
#: 0.79788458. The outer scale in the tanh gelu.
comptime GELU_TANH_SCALE_BITS: UInt32 = 0x3F4C422A

#: `0.044715` rounded to Float32 = 0.044714998. The cubic coefficient in
#: the tanh gelu. NOT 0.044715 exactly -- that is the point of pinning it.
comptime GELU_TANH_COEF_BITS: UInt32 = 0x3D372713

#: Cephes `0.5 * MAXLOGF`, tanh's saturation threshold. MAXLOGF is the
#: IEEE arm of `cephes/single/constf.c` (88.72283905206835, itself
#: 88.72283935546875 as a float32), and C evaluates `0.5 * MAXLOGF` on
#: the float32 value, giving 44.361419677734375 EXACTLY -- a power of two
#: times a float32 is exact, so this threshold has no rounding question.
comptime TANH_SAT_BITS: UInt32 = 0x42317218

#: `-MAXLOGF` = -88.72283935546875, the erfc underflow guard from
#: `cephes/single/ndtrf.c`.
comptime NEG_MAXLOGF_BITS: UInt32 = 0xC2B17218


def portable_sinf(x_in: Float32) -> Float32:
    """DEVIATION 820: `sinf` as one arithmetic, Cephes single-precision
    `sinf` (`cephes/single/sinf.c`) through `_cephes_sincosf_core` --
    the SAME reduction and the SAME two polynomials `portable_cosf`
    already uses and is already certified on. Needed for RoPE, whose
    rotation wants sin and cos of the same angle; two independently
    written trig functions would be two arithmetics wearing one name,
    which is the failure mode `ftz_simd`'s comment names for the SIMD
    seams and the reason this is a refactor and not a new file.

    Domain |x| < 8192 (Cephes's `lossth`, the reduction's own limit).
    NaN and +-inf return quiet NaN, the same contract `portable_cosf`
    has. RoPE's angles are `position * inv_freq` with `inv_freq` in
    (0, 1], so a Llama context of 8192 lands exactly at the domain edge
    and a longer one leaves it: THE TRANSFORMER LANE MUST EITHER BOUND
    THE ANGLE OR ASK FOR A WIDER REDUCTION, and this sentence exists so
    that the choice is made rather than discovered.

    THE INPUT IS FLUSHED AND `portable_cosf`'S IS NOT, which looks
    inconsistent and is not. Near zero sin returns its argument, so a
    subnormal argument survives all the way to the output: on a
    denormal-honoring column `sin(1e-40)` is `1e-40` and on Metal it is
    a signed zero, and that is a real cross-vendor divergence. Cos
    returns 1.0 for every subnormal on both columns, so its flush would
    be bit-inert and its bits are certified. Same policy, different
    reachability.

    THE SIGN IS TAKEN BY BITS, NOT BY `x_in < 0`, and this is the
    DEVIATION 746 (i) lesson rather than a style choice: METAL FLUSHES
    COMPARE OPERANDS, so `-subnormal < 0.0` is FALSE there and TRUE on
    the host. Cephes writes `if (xx < 0)`; writing that here would
    reproduce, exactly, the negative-subnormal split that
    `portable_sqrtf` was reordered to fix (2,046 of 2^20 lanes). Flush
    first, then read the sign bit with integer ops, which do not flush.

    ONE KNOWING DEVIATION FROM CEPHES. `sinf(-0.0)` is `-0.0` here and
    `+0.0` in Cephes, because Cephes's `if (xx < 0)` is false for a
    negative zero and the sign is then never applied. IEEE 754, torch
    and numpy all give `-0.0`, `portable_log1pf` and `portable_siluf` in
    this file already chose the IEEE sign rule for the same case, and
    the bit-taken sign gives it for free. Recorded rather than silent
    because the standing rule is to fix the reference's bugs, not port
    them. It is INERT at the one consumer -- RoPE's angle is
    `position * inv_freq` with position >= 0 -- so no shipped path can
    tell the difference; the gate plants it anyway.

    ACCURACY IS NOT THE GOAL, AGREEMENT IS. Cephes states 3.8e-9
    theoretical relative error for this polynomial on [-pi/4, pi/4]. The
    ulp bound of THIS spelling of it, over the whole reduction domain,
    IS NOT MEASURED and is deliberately not guessed here;
    `check-portable-trig` owes the number."""
    from std.memory import bitcast

    if x_in != x_in or abs(x_in) == bitcast[DType.float32](UInt32(0x7F800000)):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    var x = _ftz_always(x_in)
    var sign = Float32(1.0)
    if (rebind[UInt32](x.to_bits()) & UInt32(0x80000000)) != UInt32(0):
        sign = Float32(-1.0)
    return _cephes_sincosf_core(abs(x), sign, False)


def portable_tanhf(x_in: Float32) -> Float32:
    """DEVIATION 821: `tanhf` as one arithmetic, Cephes single-precision
    `tanhf` (`cephes/single/tanhf.c`) verbatim in structure. Needed for
    the tanh approximation of GELU (DEVIATION 824); HuggingFace calls
    `torch.tanh` and says nothing about what is inside it, so the math
    source decides the polynomial and there is no model spelling to
    mirror.

    THIS IS THE SECOND PORTABLE TANH IN THE TREE AND THE FIRST ONE WAS
    CONSIDERED FIRST. `arima/ported/timeSeries/jones_transform.mojo`
    (DEVIATION 675) already builds `tanh(x/2) = (e^x - 1)/(e^x + 1)` on
    top of `identical_exp`, with `+-1` returned outright for |x| > 80.
    It was NOT adopted here, and the reason is a number that lane
    already measured rather than a preference: `(e^x - 1)` CANCELS for
    small |x|, and DEVIATION 675 records its round trip at 2.73e-6
    relative, about 23 float32 ulp, with its own OWED item asking
    whether `expm1` is worth a numbered replacement. GELU's argument
    passes through zero on nearly every token -- that is the region the
    activation is FOR -- so the identity's worst region is this
    consumer's hottest one. Cephes's polynomial arm exists precisely to
    avoid that cancellation: below 0.625 it evaluates in `z = x*x` and
    adds the correction ONTO x, so nothing is subtracted from anything
    near it.

    That is a judgment about accuracy, and accuracy is not what this
    file buys -- so the honest statement is narrower: BOTH spellings
    would be bit-identical across vendors, which is the property being
    purchased; the Cephes one is additionally not visibly wrong near
    zero, and a 23-ulp activation would be. Two spellings of tanh now
    exist in the tree and that is one too many; retiring DEVIATION 675
    onto this function would also answer its OWED item, but
    `jones_transform.mojo` is another lane's file and this note is a
    referral, not a change. Note also that ARIMA wants `tanh(x/2)` and
    the halving is exact, so `portable_tanhf(x * 0.5)` is a drop-in.

    THE TWO BRANCHES, both Cephes's:

      |x| >= 0.625   `1 - 2/(exp(2x) + 1)` through `portable_expf` and
                     `portable_divf`. Three roundings after the exp: the
                     `+ 1.0`, the division, and the `1.0 -`. NOT fused
                     anywhere -- there is no multiply-add seam in this
                     arm to fuse.
      |x| <  0.625   the degree-4 polynomial in `z = x*x`, Horner
                     through `_fma_f32` (row 9: an explicit fma so no
                     backend's codegen picks the contraction), then
                     Cephes's own tail `P*z*xx + xx`. The ASSOCIATION is
                     Cephes's: `(P*z)` is rounded on its own and then
                     `fma(P*z, xx, xx)` finishes it, so the product with
                     the signed argument and the final add are ONE
                     rounding where Cephes has two. The tail multiplies
                     by `xx` and not by `|x|`, which is what makes the
                     polynomial arm odd with no sign fixup.

    |x| > 0.5*MAXLOGF (44.361419677734375, `TANH_SAT_BITS`) saturates to
    +-1.0, Cephes's own guard.

    ONE PORTED BUG, FIXED RATHER THAN INHERITED: `tanhf(-0.0)` is
    `+0.0` in Cephes, and IEEE 754, torch and numpy all say `-0.0`. It
    is not an obvious defect -- it survives BECAUSE the polynomial arm
    is odd. At xx = -0.0 the Horner value is -0.33332819422, `P*z` is
    `-0.33332819422 * (+0.0)` = `-0.0`, the tail's `* xx` is
    `(-0.0) * (-0.0)` = `+0.0`, and `(+0.0) + (-0.0)` is `+0.0` under
    round-to-nearest, so the argument's sign is cancelled twice and
    lost. Verified by evaluating Cephes's own expression in float32
    rather than assumed. An explicit `x == 0` guard returns the
    argument, which is IEEE's answer and torch's, and is also the answer
    `identical_tanh`'s FAST arm will give -- leaving the Cephes value in
    would have made IDENTICAL and FAST disagree on the sign of a zero,
    which is a row-39 defect the gate must plant for.

    THE INPUT IS FLUSHED, for `portable_sinf`'s reason: near zero tanh
    returns its argument, so a subnormal would survive to the output and
    split the columns. After the flush no subnormal remains, so the
    `xx < 0.0` sign tests below are safe to write as compares -- the
    Metal compare-flush hazard (DEVIATION 746 (i)) needs a subnormal
    operand to bite and there is none left.

    Planted: NaN -> NaN; +inf -> 1.0; -inf -> -1.0; +-0 -> +-0; a
    subnormal -> its signed zero; x = 44.361419677734375 exactly takes
    the exp arm (the quotient `2/(exp(2x)+1)` is a SUBNORMAL there,
    which `portable_divf` flushes to +0, so the arm returns exactly 1.0
    on every column -- a denormal-honoring column that kept the
    subnormal would round `1 - 5.9e-39` to 1.0 anyway, so the flush is
    bit-inert at this edge and is stated only so nobody has to re-derive
    it); the next float above it returns 1.0 through the saturation
    guard instead.

    ACCURACY IS NOT THE GOAL, AGREEMENT IS. Cephes measures 1.3e-7 peak
    relative error on [-2, 2] over 100,000 trials and 7.2e-8 on its own
    test interval [-0.625, 0.625]. The ulp bound of THIS spelling IS NOT
    MEASURED; `check-portable-gelu` owes it."""
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
    """The Cephes `erfcf` body (`cephes/single/ndtrf.c`) for |a| >= 1
    ONLY. Private, and the restriction is the point: Cephes's `erff` and
    `erfcf` call each other (erf defers to erfc above 1, erfc defers to
    erf below 1), which is fine in C and is a mutual recursion here. The
    `x < 1.0` arm is the one `portable_erff` owns, so it is dropped and
    this function is entered only from above it.

    `erfc(x) = exp(-x^2) * (1/x) * P(1/x^2)`, with a nine-term P on
    1 <= x < 2 and an eight-term R on x >= 2. NOTE THAT THE COMMENT
    ABOVE P IN CEPHES SAYS `P(1/x)` AND THE CODE PASSES `1/x^2`; the
    code is right and the comment is loose -- both branches have the
    same shape, and the constant term of each table is 1/sqrt(pi)
    (0.5638259 and 0.5641895), which is the correct x -> inf asymptote
    only for the shape the code computes. Checked at x = 1, where the
    table sums to 0.4275837 and `exp(-1) * 1 * 0.4275837` = 0.157299
    against the true erfc(1) = 0.15729921.

    Rounding, seam by seam: `x*x` one product then an exact negation
    (Cephes writes `z = -a*a`, which is `(-a)*a`, the same magnitude and
    the same sign, so the bits do not depend on which spelling is used);
    `portable_expf` for the exponential; `portable_divf` for `1/x`;
    `q*q` one product; Horner through `_fma_f32`; then `(z*q)` and
    `(z*q)*p` as TWO separate roundings, which is Cephes's own
    left-to-right `y = z * q * p` and is NOT fused -- an fma here would
    be a different function.

    `z*q` and `(z*q)*p` are each stored through `_ftz_always` because
    they REACH the subnormal range: at x near 9.4 the exponential is
    ~1e-38 and the product with q ~ 0.107 is ~1e-39. At the one consumer
    that is bit-inert (`portable_erff` returns `1 - r`, and 1 minus
    anything subnormal is 1.0 on both columns), but the seam is flushed
    anyway because a value that leaves this function must not depend on
    the column's denormal policy.

    Cephes's second `if (y == 0.0) goto under` is NOT ported. It is
    numerically a no-op: it re-enters the underflow label, which returns
    0.0 for a >= 0 (which y already is) and is unreachable for a < 0
    (where y has just been set to 2.0). All it does in Cephes is raise
    `mtherr`, and this file has no errno. `UTHRESH` is defined in
    ndtrf.c and never referenced by any of its three functions; it is
    not ported either.

    Precondition: |a| >= 1, so no subnormal can reach here and the
    compares below are safe."""
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
        # erfc(x) = exp(-x^2) 1/x P(1/x^2), 1 < x < 2
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
        # erfc(x) = exp(-x^2) 1/x R(1/x^2), 2 < x < 14
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
    """DEVIATION 822: `erff` as one arithmetic, Cephes single-precision
    `erff` (`cephes/single/ndtrf.c`) verbatim in structure. Needed for
    the EXACT gelu (DEVIATION 823); HuggingFace calls `torch.erf` and
    says nothing about what is inside it, so the math source decides the
    polynomial and there is no model spelling to mirror.

    `|x| <= 1`: `erf(x) = x * T(x^2)`, the seven-term T table, Horner
    through `_fma_f32` (row 9) and then ONE product with x. `|x| > 1`:
    `1 - erfc(x)` through `_cephes_erfcf_ge1`, one rounding on the
    subtraction. The branch is `fabsf(x) > 1.0`, so x = 1.0 exactly
    takes the polynomial arm, which is Cephes's own boundary.

    THE INPUT IS FLUSHED, for `portable_sinf`'s reason: near zero
    `erf(x) = x * 1.1283792`, so a subnormal argument produces a
    subnormal output on a denormal-honoring column and a signed zero on
    Metal.

    Planted: NaN -> NaN; +-0 -> +-0 (the polynomial's constant term is
    positive, so the product carries the argument's sign); a subnormal
    -> its signed zero; +inf -> 1.0 and -inf -> -1.0 (both fall out of
    Cephes's own underflow guard, `erfc(+inf) = 0` and `erfc(-inf) = 2`,
    with no special case needed); |x| > 9.4193 saturates to +-1.0 the
    same way.

    ACCURACY IS NOT THE GOAL, AGREEMENT IS. Cephes measures 1.7e-7 peak
    relative error and 2.8e-8 rms for erff over [-9.3, 9.3] on 50,000
    trials, which is a shade over one float32 ulp; its erfcf arm is
    weaker (3.9e-6 peak), and since erff routes through erfcf above 1
    the peak relative error there is erfcf's -- but ABSOLUTE error stays
    at the 1e-7 level because erf is within an ulp of +-1 out there.
    Both are Cephes's numbers for Cephes's spelling. The ulp bound of
    THIS spelling IS NOT MEASURED; `check-portable-gelu` owes it."""
    var x = _ftz_always(x_in)
    if x != x:
        return x
    if abs(x) > Float32(1.0):
        return Float32(1.0) - _cephes_erfcf_ge1(x)
    # erf(x) = x P(x^2), 0 < x < 1
    var z = x * x
    var p = Float32(7.853861353153693e-5)
    p = _fma_f32(p, z, Float32(-8.010193625184903e-4))
    p = _fma_f32(p, z, Float32(5.188327685732524e-3))
    p = _fma_f32(p, z, Float32(-2.685381193529856e-2))
    p = _fma_f32(p, z, Float32(1.128358514861418e-1))
    p = _fma_f32(p, z, Float32(-3.761262582423300e-1))
    p = _fma_f32(p, z, Float32(1.128379165726710))
    return x * p


def portable_gelu_erf(x_in: Float32) -> Float32:
    """DEVIATION 823: the EXACT gelu,
    `input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))`, which is
    `GELUActivation._gelu_python` at activations.py:85-86 verbatim.
    ACT2CLS reaches it as `"gelu"` and `"gelu_python"`
    (activations.py:325, :329). This is NOT `portable_gelu_tanh` and the
    two must never be substituted for one another -- see the block
    comment above.

    THE MODEL'S ORDER, SEAM BY SEAM, all four roundings named:

      1. `input / math.sqrt(2.0)` -- ONE division through
         `portable_divf`, by the float32 `GELU_SQRT2_BITS`. torch
         promotes the Python double scalar to the tensor's dtype before
         the op, so the divisor is the float32 1.4142135 and not the
         double; dividing by the double first and rounding after would
         be a different function.
      2. `torch.erf(...)` -- `portable_erff`.
      3. `1.0 + erf` -- one add.
      4. `input * 0.5` -- EXACT, a power of two, so it is not a rounding
         at all; and `x * 0.5` and `0.5 * x` are the same bits, which is
         why `GELUActivation` and `NewGELUActivation` writing the factor
         on opposite sides is a difference in text and not in arithmetic.
      5. the final product -- one rounding.

    NOTHING IS FUSED. There is no multiply-add seam in this expression:
    torch evaluates it as separate elementwise kernels, so every product
    and every sum rounds on its own, and an `identical_mul_add` anywhere
    in here would make our gelu disagree with the reference's by design.
    This is DEVIATION 744's rule (the reference's spelling wins for
    operation order) applied to a longer expression.

    `x * 0.5` IS STORED THROUGH `_ftz_always` and so is the result:
    halving a normal at the bottom of the range produces a SUBNORMAL
    (x = FLT_MIN gives 5.9e-39), which Metal flushes and a
    denormal-honoring column keeps. The input is flushed first for
    `portable_erff`'s reason.

    Planted: NaN -> NaN; +-0 -> +-0; +inf -> +inf; a subnormal -> its
    signed zero. **-inf -> NaN**, and that is the reference's answer
    too, not a defect of ours: `-inf * 0.5` is `-inf`, `erf(-inf)` is
    `-1.0`, `1.0 + (-1.0)` is `+0.0`, and `-inf * +0.0` is NaN. Recorded
    because it is exactly the kind of edge a reader assumes we got
    wrong."""
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
    """DEVIATION 824: the TANH gelu,
    `input * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input +
    0.044715 * torch.pow(input, 3.0))))`, which is
    `GELUTanh._gelu_tanh_python` at activations.py:47-48 verbatim, and
    the same arithmetic as `NewGELUActivation.forward` at :65 (the
    factor is written on the other side of `input`, which is the same
    bits). ACT2CLS reaches it as `"gelu_new"`, `"gelu_pytorch_tanh"` and
    `"gelu_python_tanh"` (activations.py:328, :330, :331). This is NOT
    `portable_gelu_erf`; HuggingFace's own docstring says the two are
    "not an exact numerical match" (activations.py:36-37).

    THE MODEL'S ORDER, SEAM BY SEAM, every rounding named and NONE of
    them fused:

      1. `torch.pow(input, 3.0)` -- `(x*x)*x`, TWO roundings. See the
         judgment call below.
      2. `0.044715 * x^3` -- one product, by the float32
         `GELU_TANH_COEF_BITS` (0.044714998, not 0.044715).
      3. `input + ...` -- one add. NOT an fma. torch runs the multiply
         and the add as separate elementwise kernels and therefore
         rounds twice; folding them into one `identical_mul_add` would
         be more accurate and would be a different function. DEVIATION
         744's rule.
      4. `math.sqrt(2/pi) * (...)` -- one product, by the float32
         `GELU_TANH_SCALE_BITS`.
      5. `torch.tanh(...)` -- `portable_tanhf`.
      6. `1.0 + tanh` -- one add.
      7. `input * 0.5` -- exact.
      8. the final product -- one rounding.

    JUDGMENT CALL, STATED PLAINLY BECAUSE IT IS NOT VERIFIED HERE:
    `torch.pow(tensor, 3.0)` is written as `(x*x)*x` above on the
    understanding that ATen's `pow_tensor_scalar` specializes small
    integral exponents into repeated multiplication rather than calling
    `std::pow`. That is the behavior this lane expects and it is what
    the reader of the HuggingFace line expects too, but it was NOT read
    out of the PyTorch source (no checkout was available) and it was NOT
    measured. If it is wrong, it is wrong in the last bit of one
    intermediate and `check-portable-gelu`'s torch corpus arm is what
    will say so. The association `(x*x)*x` rather than `x*(x*x)` is the
    same choice and carries the same caveat.

    Intermediates are stored through `_ftz_always` at every step, which
    is row 10's checklist for a pinned expression rather than a
    precaution: `x*x` underflows to a subnormal and then to zero for
    |x| < ~1.1e-19, and `x * 0.5` is subnormal at the bottom of the
    normal range.

    Planted: NaN -> NaN; +-0 -> +-0 (the cube and the products all carry
    the sign, and `+-0 + +-0` is the matching signed zero); a subnormal
    -> its signed zero; +inf -> +inf. **-inf -> NaN**, for
    `portable_gelu_erf`'s reason: tanh(-inf) is -1.0, `1.0 + (-1.0)` is
    +0.0, and `-inf * +0.0` is NaN. NO OVERFLOW EDGE: for a large finite
    x the cube overflows to +-inf and the argument to tanh with it, but
    tanh saturates to +-1.0 and the result is `x * 0.5 * 2.0` = x, which
    is the correct answer and torch's."""
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
    """DEVIATION 204's map, the SECOND copy in this tree and deliberately
    not a second DESIGN: a float as an unsigned key whose INTEGER order
    is the float's order. Non-negative -> set the sign bit; negative ->
    invert every bit. Monotone on all of IEEE-754 away from NaN, exactly
    invertible, nothing approximated.

    The original is `range_key` in
    `extratrees/ported/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:616`,
    where DEVIATION 204 introduced it to make a cross-block float
    min/max fold lock-free and DEVIATION 452 (IDENTITY_PATHS row 13)
    then used it to CLOSE the signed-zero hazard: `+0.0` and `-0.0`
    compare EQUAL, so under a plain float max which one survives is
    decided by the fold order, and the sign reached the model through a
    `threshold == max -> min` guard.

    IT IS COPIED RATHER THAN IMPORTED, and the reason is a dependency
    direction rather than laziness. `numerics.mojo` is the floating-point
    contract every other directory imports; importing a ported extratrees
    kernel from it would point the arrow backwards and put a decision
    tree's build in the path of every transformer. The two spellings are
    the same two lines and `check-portable-fmax` asserts they agree, so
    a divergence is caught rather than assumed away."""
    var b = rebind[UInt32](v.to_bits())
    if (b & UInt32(0x80000000)) != UInt32(0):
        return ~b
    return b | UInt32(0x80000000)


def portable_fmaxf(a_in: Float32, b_in: Float32) -> Float32:
    """DEVIATION 825: an elementwise maximum whose result does not depend
    on the argument order, the vendor, or the fold shape. Needed by
    softmax, whose row maximum is a REDUCTION -- and a reduction over a
    max that is not commutative and associative has an answer that
    depends on the tree the scheduler happened to build, which is
    IDENTITY_PATHS row 8 and row 20 all over again.

    THREE HAZARDS, and only the first is the famous one:

      (a) SIGNED ZERO. `+0.0 == -0.0` is true, so a plain `a > b ? a : b`
          returns whichever operand it was handed first and a fold
          returns whichever the topology reached last. Attention scores
          reach `-0.0` easily (a masked lane, a flushed subnormal at a
          row-10 seam) and `+0.0` just as easily, and the two live in one
          row. THE KEY MAP MAKES THEM ORDERED: `key(+0)` is 0x80000000
          and `key(-0)` is 0x7FFFFFFF, so `+0.0` wins every time in
          every order.
      (b) NaN. IEEE `maxNum` returns the non-NaN operand, C `fmax` agrees,
          torch's `max` PROPAGATES, and a hardware max instruction does
          whatever its vendor chose. Worse, the key map alone does not
          settle it: a positive quiet NaN keys ABOVE `+inf` and wins
          every max while a negative one keys BELOW `-inf` and loses
          every max, so the answer would depend on a payload's sign bit.
          Settled here BEFORE the map, by returning the canonical quiet
          NaN 0x7FC00000 if either operand is NaN -- order-independent,
          payload-independent, and it propagates the way torch does.
      (c) DENORMALS. Both operands are flushed first, so a subnormal
          compares as the signed zero it flushes to on Metal and on a
          denormal-honoring column alike, and the value RETURNED carries
          the flush (row 10's model: flush operands, one operation,
          flush result).

    The result is therefore commutative and associative over all of
    Float32 including both zeros and NaN, which is precisely the
    property a reduction needs: `check-portable-fmax` asserts it by
    folding one planted row in several orders and requiring one answer.

    NO HARDWARE `max` INSTRUCTION AND NO FLOAT COMPARE ANYWHERE. The
    selection is integer, which matters three times: integer ops do not
    flush on Metal (DEVIATION 746 (i)); a vendor max instruction is a
    per-vendor choice by definition (the same reason DEVIATION 746 bans
    the `rsqrt` intrinsic even though Apple's is the correctly rounded
    one); and the zero split is not hypothetical -- IDENTITY_PATHS row
    39 MEASURED `max(+0.0, -0.0)` as `-0.0` ON APPLE (the second
    operand) and `+0.0` on NVIDIA and AMD, which is quoted in
    `holtwinters/ported/holtwinters/internal/hw_utils.mojo` (DEVIATION
    663). That file is also the place to learn why a compare chain and
    not a `max` even against a CONSTANT: its `HW_MAX_CLAMP` sabotage came
    back NULL ON APPLE because LLVM folded `maxnum(0.0, v)` into a
    compare-select whose tie answer is `+0.0`, which it is allowed to do
    since maxnum's zero tie is unspecified. The spelling's answer
    depended on whether a constant got folded. An integer total order
    has no tie to answer, so it satisfies that lesson a fortiori.

    THIS IS THE SCALAR PRIMITIVE. It is not, and does not replace,
    either of the two BLOCK-LEVEL reductions already in the tree, and a
    transformer attention kernel has to choose between three things
    rather than two:

      `core/pinned_reduce.mojo:159` `pinned_block_max` -- a halving fold
        through threadgroup memory with no lane primitive. Its own block
        comment (lines 145-155) says it is safe ONLY because its single
        caller folds `abs(...)`, so no `-0.0` is ever compared, and it
        requires a future caller that cannot say the same to state why
        first. **AN ATTENTION KERNEL CANNOT SAY THE SAME.** Logits reach
        `-0.0` easily -- a masked lane, a flushed subnormal at a row-10
        seam, a product with a negative zero -- so softmax's row maximum
        must NOT use it as it stands. The clean fix is to give that fold
        this function as its combine step, which is the block's owner's
        call and not this file's.
      `svm/mojo_only/pinned_argreduce.mojo:109` `pinned_block_argmax` --
        genuinely `-0.0`-safe, over a total order of (value, key) with
        ties to the SMALLER key (DEVIATIONS 633/635). That is the right
        answer when there IS a key and the winner's identity is what the
        caller wants. Softmax's row maximum has no key and wants only
        the value, so paying for one would be inventing a tie-break
        nobody asked for.
      this function -- the scalar combine. A block fold built out of it
        is order-independent because the combine is commutative and
        associative, which is the property `check_fmax_fold_invariant`
        exists to assert."""
    from std.memory import bitcast

    if a_in != a_in or b_in != b_in:
        return bitcast[DType.float32](UInt32(0x7FC00000))
    var a = _ftz_always(a_in)
    var b = _ftz_always(b_in)
    if _total_order_key(b) > _total_order_key(a):
        return b
    return a


def identical_sin(x: Float32) -> Float32:
    """Row 12's `sin` seam call (RoPE). IDENTICAL routes through
    `portable_sinf` (one arithmetic everywhere, sharing the certified
    reduction with `identical_cos`); FAST is the stdlib's device path
    verbatim, so FAST bits do not move."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_sinf(x)
    from std.math import sin

    return sin(x)


def identical_tanh(x: Float32) -> Float32:
    """DEVIATION 821's seam: IDENTICAL is `portable_tanhf`; FAST is
    `std.math.tanh`, the stdlib's device path verbatim.

    ONE RISK ON THE FAST ARM, NAMED SO IT IS NOT DISCOVERED THE HARD WAY:
    `identical_log1p`'s FAST arm CANNOT be `std.math.log1p`, because that
    one lowers the Float32 case through FLOAT64 and Metal refuses to
    compile a kernel that calls it (DEVIATION 746 (ii)). `std.math.tanh`
    has not been checked for the same defect by this lane -- it is used
    elsewhere in the tree (`arima/ported/timeSeries/jones_transform.mojo`)
    but only on a host path. If Metal refuses this arm, the fix is one
    line and it is Cephes's own large-x identity:
    `1 - 2/(exp(x + x) + 1)` through `std.math.exp`. Do not "fix" it by
    routing FAST to the portable function: FAST's contract is the
    vendor's own spelling."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_tanhf(x)
    from std.math import tanh

    return tanh(x)


def identical_erf(x: Float32) -> Float32:
    """DEVIATION 822's seam: IDENTICAL is `portable_erff`; FAST is
    `std.math.erf`, the stdlib's device path verbatim. Carries
    `identical_tanh`'s float64-lowering risk for the same reason and has
    not been checked either; if Metal refuses this arm the FAST spelling
    must become an explicit float32 composition, and the portable
    function is the one to copy the shape of."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_erff(x)
    from std.math import erf

    return erf(x)


def identical_gelu_erf(x: Float32) -> Float32:
    """DEVIATION 823's seam, the EXACT gelu: IDENTICAL is
    `portable_gelu_erf`; FAST is the reference's own spelling
    `x * 0.5 * (1 + erf(x / sqrt(2)))` through the stdlib erf, with the
    same pinned float32 divisor. NOT interchangeable with
    `identical_gelu_tanh` -- a checkpoint's config says which one it
    wants."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_gelu_erf(x)
    from std.math import erf
    from std.memory import bitcast

    var s2 = bitcast[DType.float32](GELU_SQRT2_BITS)
    return x * Float32(0.5) * (Float32(1.0) + erf(x / s2))


def identical_gelu_tanh(x: Float32) -> Float32:
    """DEVIATION 824's seam, the TANH gelu: IDENTICAL is
    `portable_gelu_tanh`; FAST is the reference's own spelling
    `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` through the
    stdlib tanh, with the same pinned float32 constants and the same
    unfused association. NOT interchangeable with `identical_gelu_erf`."""
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
    """DEVIATION 825's seam (softmax's row maximum): IDENTICAL is
    `portable_fmaxf` (the total-order selection, NaN canonicalized,
    operands and result flushed); FAST is the stdlib `max`, whose
    signed-zero and NaN answers are the vendor's own choice."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return portable_fmaxf(a, b)
    from std.math import max

    return max(a, b)


# ============ THE GATES DEVIATIONS 820-825 OWE ============================
# Written here rather than in a plan file because the thing most likely to
# go wrong with a new portable_* function is that its check exists, passes,
# and is BLIND (NOVELTY note 13/14: output-only gates are blind; a gate is
# likelier VACUOUS than wrong). NONE of these exist yet. The three check
# files below are the transformer lane's to write; this block is the
# specification, and the numbers in it are the ones a run must print.
#
# The shape is the one `check-portable-nn` (`mojo_only/portable_nn_check.mojo`)
# already has, three arms per file over 2^20 hashed inputs PER FUNCTION,
# and it is copied rather than reinvented:
#
#   ARM 1  ACCURACY, HOST. Each function against a float64 reference
#          reached through FFI (libm's double `sin`, `tanh`, `erf`), ulp
#          distance on the finite domain, plus EVERY planted special value
#          asserted BY BITS. Bounds are NOT stated here because they are
#          NOT MEASURED; the first run RECORDS them and a later run
#          asserts them. Writing a guessed bound into the check first is
#          how a check comes to assert what the code does rather than what
#          it should do.
#   ARM 2  DEVICE == HOST. One kernel evaluates every function of the file
#          on the same inputs; the device bits must equal the host bits on
#          every lane (NaN compared as NaN). The printed
#          `<name> device hash` is the cross-vendor certificate line, and
#          per-function hashes are printed too so a leg can name which
#          function moved. NaN canonicalized to 0x7FC00000 before hashing.
#   ARM 3  REACH (`sabotage` argv). The device arm computes the STDLIB
#          spelling instead, the host arm stays portable, and the compare
#          MUST then fail on many lanes per function. Zero mismatches for
#          any function means that function's compare is blind.
#
# `mojo_only/portable_trig_check.mojo`, `pixi run check-portable-trig`
#   FUNCTIONS: `portable_sinf`, and `portable_cosf` AGAIN.
#   `check_cos_bits_did_not_move` IS THE POINT OF THIS FILE and must run
#   first: `portable_cosf` was refactored onto `_cephes_sincosf_core` and
#   is already certified, so this file exists to prove the bits did not
#   move before it proves anything about sin. The proof is not new work --
#   re-run `pixi run check-portable-sqrtcos` and its printed
#   `sqrtcos device hash` must still read 12295913102197186379. Do that
#   BEFORE trusting any number below it.
#   `check_sin_cos_share_one_reduction`: for a scattered set of x, assert
#   `portable_sinf(x)` and `portable_cosf(x)` agree with the core called
#   directly, so that a future edit that gives one of them its own
#   reduction fails here rather than in a model.
#   `check_sin_planted`: sin(+0) = +0 AND sin(-0) = -0 BY SIGN BIT (an
#   `assert_equal` on floats passes on either, which is the trap
#   `adv_signed_zeros` found in the mamba corpus); sin(NaN), sin(+-inf)
#   quiet NaN; sin(subnormal) = its signed zero on BOTH host and device,
#   which is the one that separates this implementation from a compare-
#   written one.
#   `check_sin_odd`: `portable_sinf(-x) == -portable_sinf(x)` BY BITS over
#   the hashed sweep. The bit-taken sign makes this exact, and it is the
#   cheapest single assertion that catches a wrong octant flip.
#   `check_sin_quadrants`: sin at k*pi/2 for k in -16..16 and at
#   k*pi/4 -- the octant table has eight entries and a sweep of uniform
#   randoms enters them unevenly. `uniform-test-data-hides-permutation`
#   applies: plant the angles, do not sample them.
#   `check_rope_domain_edge`: |x| just under and just over 8192, asserting
#   that the reduction's documented domain is where the docstring says it
#   is. This one is expected to FAIL above the edge and the check records
#   the failure rather than hiding it -- it is the evidence the
#   transformer lane needs to decide whether RoPE must bound its angle.
#   SABOTAGE ARMS (in source, one build each, `-D` selected):
#     TRIG_SAB_NO_SECOND_FLIP    -- drop `if is_cos and j > 1`. Must move
#                                   cos and NOT sin.
#     TRIG_SAB_POLY_NOT_SWAPPED  -- drop the `if not is_cos` flip of
#                                   `use_sin_poly`. Must move sin and NOT
#                                   cos. Together these two are what
#                                   proves the shared core did not quietly
#                                   make sin into cos.
#     TRIG_SAB_SIGN_BY_COMPARE   -- take sin's sign with `x_in < 0.0`
#                                   instead of the sign bit. BIT-INERT ON
#                                   THE HOST BY CONSTRUCTION: it can only
#                                   bite on a device that flushes compare
#                                   operands, so the fixture MUST carry
#                                   negative subnormals and the arm MUST
#                                   be run on Metal. If it reports zero
#                                   mismatches on Apple the arm is broken,
#                                   not the code.
#     TRIG_SAB_NO_INPUT_FLUSH    -- drop `_ftz_always` in `portable_sinf`.
#                                   Same caveat: it separates host from
#                                   device, so the compare that must fail
#                                   is ARM 2's, not ARM 1's.
#
# `mojo_only/portable_gelu_check.mojo`, `pixi run check-portable-gelu`
#   FUNCTIONS: `portable_tanhf`, `portable_erff`, `portable_gelu_erf`,
#   `portable_gelu_tanh`.
#   `check_two_gelus_disagree` IS THE POINT OF THIS FILE. It must assert
#   that `portable_gelu_erf` and `portable_gelu_tanh` DIFFER, and by how
#   much: record the max absolute difference over the sweep and assert it
#   is above 1e-4 (they part company around |x| = 2 by ~1e-3). Without
#   this a copy-paste that routed both seams to one implementation would
#   pass every other check in this file. It is a NEGATIVE control and it
#   is load-bearing.
#   `check_tanh_branch_boundary`: |x| = 0.625 exactly takes the exp arm
#   and the float below it takes the polynomial arm, asserted on both
#   sides and on both signs; and |x| = 44.361419677734375 exactly takes
#   the exp arm while the next float above takes the saturation guard.
#   Both boundaries planted, in the `portable_softplusf` guard's style.
#   `check_tanh_planted`: tanh(-0) = -0 BY SIGN BIT. This is the assertion
#   that the Cephes bug stayed fixed, and an `assert_equal` cannot see it.
#   Plus tanh(NaN), tanh(+-inf) = +-1.0, tanh(subnormal) = its signed
#   zero.
#   `check_erf_planted`: erf(+-0) = +-0 BY SIGN BIT; erf(+-inf) = +-1.0;
#   erf(NaN); erf at |x| = 1.0 exactly (the polynomial arm, both signs);
#   erf(|x| > 9.4193) = +-1.0; erf(2.0) exactly (the P/R table boundary,
#   both sides).
#   `check_erf_odd`: `portable_erff(-x) == -portable_erff(x)` BY BITS. It
#   holds for the polynomial arm by construction; ABOVE |x| = 1 IT DOES
#   NOT, because `1 - erfc(x)` and `1 - (2 - erfc(|x|))` are different
#   roundings. So this check must assert oddness only on |x| <= 1 and must
#   RECORD the asymmetry above it rather than assert it away -- if it
#   asserts oddness everywhere it will fail, and the failure is the
#   reference's shape, not a defect.
#   `check_gelu_planted`: for BOTH gelus -- +-0 -> +-0 BY SIGN BIT;
#   +inf -> +inf; **-inf -> NaN** (both of them, and the check must assert
#   NaN rather than -0.0, because a reader "fixing" that edge would break
#   agreement with torch); NaN -> NaN; a subnormal -> its signed zero;
#   x = FLT_MIN, where `x * 0.5` is subnormal and the flush is what makes
#   the two columns agree.
#   `check_gelu_constants`: the three model constants read back BY BITS --
#   0x3FB504F3, 0x3F4C422A, 0x3D372713 -- against `Float32(<decimal>)`
#   built from the decimal in the comment. This is the
#   `mojo-string-float-roundtrip` rule made checkable rather than trusted.
#   TORCH CORPUS ARM, and it is the only arm here that compares against
#   something we did not write: generate `gelu(x)` for both forms from
#   HuggingFace's `_gelu_python` and `_gelu_tanh_python` in float32,
#   dumped in the `mamba/corpus/` style with a hashed input spec and the
#   inputs compared BITWISE first. It is what settles the one open
#   judgment call in DEVIATION 824 -- whether `torch.pow(x, 3.0)` is
#   `(x*x)*x`. Calibrate the tolerance PER CASE from the corpus's own
#   `--self-test`, which is the `adv_gate_saturation` lesson.
#   SABOTAGE ARMS:
#     GELU_SAB_ONE_GELU      -- route `portable_gelu_tanh` to
#                               `portable_gelu_erf`. Must fail
#                               `check_two_gelus_disagree` and nothing
#                               else needs to catch it.
#     GELU_SAB_FUSED_INNER   -- spell the tanh gelu's
#                               `x + 0.044715*x^3` as one
#                               `identical_mul_add`. This is the arm that
#                               proves the UNFUSED decision is reached;
#                               it is a last-bit change, so it needs the
#                               bitwise compare and will be invisible to
#                               any tolerance.
#     GELU_SAB_DOUBLE_CONST  -- divide by `Float32(1.4142135623730951)`
#                               computed as a double first. Expected to
#                               be BIT-INERT (the literal is correctly
#                               rounded to the same float32), and it is
#                               listed so that its inertness is RECORDED
#                               rather than mistaken for coverage.
#     GELU_SAB_CEPHES_ZERO   -- drop `portable_tanhf`'s `x == 0` guard,
#                               restoring the Cephes sign loss. Must fail
#                               `check_tanh_planted` on the sign bit and
#                               NOTHING ELSE -- if it fails an ulp check
#                               too, that check is comparing zeros
#                               wrongly.
#
# `mojo_only/portable_fmax_check.mojo`, `pixi run check-portable-fmax`
#   `check_fmax_order_invariant`: `portable_fmaxf(a, b) == portable_fmaxf(b, a)`
#   BY BITS over the hashed sweep AND over a planted list that includes
#   both zeros, both infinities, quiet NaN of BOTH SIGNS, and subnormals
#   of both signs.
#   `check_fmax_fold_invariant` IS THE POINT OF THIS FILE, because the
#   consumer is a REDUCTION and not a binary op: take one planted row,
#   fold it left to right, right to left, and as a balanced binary tree,
#   and assert all three give the same bits. Row 13 was a fold-order
#   defect and a pairwise check cannot see one.
#   `check_fmax_signed_zero`: `max(+0, -0)` and `max(-0, +0)` both return
#   +0.0 BY SIGN BIT, and a row containing both zeros folds to +0.0 in
#   every order.
#   `check_fmax_nan`: any NaN operand, either position, either sign of
#   payload, returns 0x7FC00000 exactly. THE SIGN OF THE PAYLOAD IS THE
#   ASSERTION THAT MATTERS: the key map alone sends a positive NaN above
#   +inf and a negative one below -inf, so a negative quiet NaN is the
#   input that separates the canonicalization from the map.
#   `check_fmax_agrees_with_range_key`: assert `_total_order_key` returns
#   the same UInt32 as `range_key` in
#   `extratrees/ported/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo`
#   over a scattered fixture including negatives, both zeros and both
#   infinities. The two are one map deliberately copied; this is what
#   makes the copy checkable instead of a claim.
#   ARM 2 (device == host) applies here as much as anywhere: integer ops
#   do not flush, so the device and host answers should be identical on
#   every lane INCLUDING the subnormal ones, and that is the property to
#   assert.
#   SABOTAGE ARMS:
#     FMAX_SAB_HARDWARE_MAX  -- use `std.math.max`. Must fail the signed-
#                               zero and NaN checks. Row 39 already
#                               measured the split it should reproduce
#                               (`max(+0.0, -0.0)` = `-0.0` on Apple,
#                               `+0.0` on NVIDIA and AMD), so the arm has
#                               a PREDICTED answer per column rather than
#                               just "must differ". If it comes back null
#                               on Apple, suspect the fold DEVIATION 663
#                               found -- both operands must be RUNTIME
#                               values, because `maxnum` against a
#                               compile-time zero gets folded to a
#                               compare-select and the arm goes inert.
#                               And run it on the other two columns before
#                               concluding anything, the DEVIATION 258
#                               lesson.
#     FMAX_SAB_SIGN_UNFLIPPED -- set the sign bit unconditionally instead
#                               of inverting a negative's bits, which is
#                               `RANGE_SAB_SIGN_UNFLIPPED` from DEVIATION
#                               204 verbatim. Must move exactly the
#                               comparisons involving a negative operand,
#                               so the fixture MUST carry negatives.
#     FMAX_SAB_NO_NAN_CANON  -- drop the NaN branch and let the key map
#                               decide. Must fail `check_fmax_nan` on the
#                               negative-payload lanes specifically.
#
# `identical_mul` (DEVIATION 826) needs no check file of its own, and
# saying why is cheaper than writing a blind one. It is one expression
# over `identical_mul_add`, which row 9's `check-ieee-arith` already
# gates, and DEVIATION 720's copies are already gated by the mamba
# lane's `adv_signed_zeros` corpus case (585 zero cells compared BY SIGN
# BIT, 125 negative zeros surviving seams S3, S4 and S12). What IS owed
# is one assertion, and it belongs wherever this file is next gated:
# `identical_mul(-1.0, 0.0)` and `identical_mul(1.0, -0.0)` must return
# `-0.0` BY SIGN BIT, and `identical_mul` must agree bit for bit with
# `pinned_mul` in all three of the mamba files over a scattered fixture.
# That is the assertion that catches the fourth copy drifting.
