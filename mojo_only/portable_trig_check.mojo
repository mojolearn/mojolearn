"""IDENTITY_PATHS: the RoPE trig pair. DEVIATION 820's `portable_sinf` and
the already-certified `portable_cosf` that it was refactored to share
`_cephes_sincosf_core` with. The fourth gate of portable device
arithmetic, sibling of `check-portable-translog`, `check-portable-sqrtcos`
and `check-portable-nn`, and its shape is copied from
`mojo_only/portable_nn_check.mojo` rather than reinvented.

THE FIRST THING THIS FILE OWES IS NOT ABOUT SIN. `portable_cosf` is
certified and other lanes depend on its bits; DEVIATION 820 moved its body
into a shared core. So `check_cos_bits_did_not_move` runs FIRST, and the
authoritative proof is not in this file at all -- it is
`pixi run check-portable-sqrtcos`, whose printed `sqrtcos device hash`
must still read 12295913102197186379. RUN THAT BEFORE TRUSTING ANY NUMBER
BELOW IT. What this file adds is a second, independent witness over the
same cos inputs (`trig cos-fixture hash`), which is a NEW number and not
the sqrtcos one.

THREE ARMS, over 2^20 hashed and planted inputs:

1. ACCURACY, HOST -- `portable_sinf` and `portable_cosf` against libm's
   double `sin`/`cos` through FFI, ulp distance on the finite in-domain
   lanes, every planted special value asserted BY BITS (signed zeros
   compared by SIGN BIT, because `assert_equal` on floats passes on
   either). NO ULP BOUND IS ASSERTED. The bound of this fma spelling of
   Cephes has never been measured; this run RECORDS it and a later run
   turns the recorded number into an assertion. A guessed bound is how a
   check comes to assert what the code does rather than what it should do.
2. DEVICE == HOST -- one kernel evaluates both functions on the same
   inputs; the device bits must equal the host bits on every lane (NaN
   compared as NaN). `trig device hash` is the cross-vendor certificate
   line, per-function hashes beside it so a leg can name which function
   moved. NaN canonicalized to 0x7FC00000 before hashing.
3. REACH -- the sabotage arms below. Each is a named compile-time
   alternative spelling of ONE decision, selected by a `-D` define.

DEVIATION 940: THE SABOTAGE SPELLINGS LIVE HERE, NOT IN `numerics.mojo`.
The specification at the foot of `mojo_only/numerics.mojo` asks for the
arms "in source". They are not, and the reason is ownership: this lane
does not own `numerics.mojo`, and a gate that has to edit the thing it
gates is not an independent gate. Instead each arm is a LOCAL re-spelling
of `portable_sinf` / `_cephes_sincosf_core` with one decision changed
(`_sab_sinf`, `_sab_cosf`, `_sab_sincos_core` below). With no define
armed the local spelling is a faithful copy and MUST agree with the real
function bit for bit on every lane -- which is itself an always-on
assertion that the copy has not drifted. With an arm armed it must
disagree, in the predicted place and nowhere else.

DEVIATION 941: THE ARMS ARE SELF-JUDGING. Four counters are printed per
function, not one, because two of the four arms have a PER-COLUMN
prediction and a single mismatch count cannot tell "inert on this column"
from "blind":

    dev(portable) vs host(portable)   arm 2. MUST be 0 in every build.
    host(portable) vs host(sab)       host reach.
    dev(portable)  vs dev(sab)        device reach.
    dev(sab)       vs host(sab)       arm 2 UNDER SABOTAGE -- this is the
                                      compare the specification means when
                                      it says a flush arm "separates host
                                      from device".

The run also probes, at runtime, whether THIS device flushes subnormals
(`x * one` with `one` read from a buffer so no fold can answer it early --
the DEVIATION 663 lesson), and uses the answer to judge the two arms whose
prediction is per-column.

SABOTAGE ARMS, and what each must do:

  -D MOJOLEARN_TRIG_SABOTAGE_NO_SECOND_FLIP=1
        Drop `if is_cos and j > 1` from the core. MUST move cos and MUST
        NOT move sin. Host reach.
  -D MOJOLEARN_TRIG_SABOTAGE_POLY_NOT_SWAPPED=1
        Drop the `if not is_cos` flip of `use_sin_poly`. MUST move sin and
        MUST NOT move cos. Host reach. Together with the arm above this is
        what proves the shared core did not quietly turn sin into cos.
  -D MOJOLEARN_TRIG_SABOTAGE_SIGN_BY_COMPARE=1
        Take sin's sign with `x_in < 0.0` instead of the sign bit. THIS
        ARM PINS TWO SEPARATE DECISIONS AND ITS CLAUSE IS AN EXACT COUNT,
        NOT A "MUST DIFFER" (DEVIATION 949; the first version of this file
        asserted the arm was host-inert BY CONSTRUCTION, which was FALSE,
        and the run that proved it wrong is the reason this paragraph is
        this long):

          (i) `-0.0` SEPARATES THE TWO SPELLINGS ON EVERY COLUMN, with no
              flushing involved at all, because `-0.0 < 0.0` is FALSE
              while `-0.0`'s SIGN BIT IS SET. Sign-by-bits gives
              `sin(-0.0) = -0.0` and sign-by-compare gives `+0.0`. That
              is DEVIATION 820's KNOWING DEVIATION FROM CEPHES -- IEEE,
              torch and numpy all say `-0.0` and Cephes's own
              `if (xx < 0)` loses it -- so this arm is precisely the
              sabotage that undoes that decision, and it MUST move those
              lanes on the host too. HOST REACH IS EXACTLY THE NUMBER OF
              `-0.0` INPUTS IN THE FIXTURE (one), and the run asserts the
              count AND that the moved lane's input is `0x80000000`.
          (ii) NEGATIVE SUBNORMALS separate them only where the column
              flushes compare operands (DEVIATION 746 (i)): on Metal
              `-subnormal < 0.0` is FALSE and on the host it is TRUE. So
              DEVICE REACH IS EXACTLY `n(-0.0) + n(negative subnormal)`
              on a flushing column and EXACTLY `n(-0.0)` on one that
              honors denormals. The fixture carries 2^17 subnormal lanes
              of both signs so that half of the clause has something to
              count, and the run judges it against the MEASURED flush
              probe rather than against an assumption about the vendor.
  -D MOJOLEARN_TRIG_SABOTAGE_NO_INPUT_FLUSH=1
        Drop `_ftz_always` from `portable_sinf`. Host reach MUST be
        non-zero (the host keeps the subnormal, the portable function
        returns its signed zero); and on a flushing column the
        dev(sab)-vs-host(sab) counter MUST also be non-zero, which is the
        arm 2 failure the specification predicts.

Run:
    pixi run check-portable-trig
    pixi run mojo run -D MOJOLEARN_TRIG_SABOTAGE_NO_SECOND_FLIP=1 \\
        -I . mojo_only/portable_trig_check.mojo
and the same for the other three names. A misspelled define is silently
ignored by the compiler, so the run PRINTS the arm it was built with
(`trig sabotage <name>`, `tools/gemm_ladder.sh:71`'s scar); read that line
before believing a clean result.
"""

from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import floor, sin
from std.memory import bitcast
from std.sys import argv
from std.sys.compile import is_defined
from std.ffi import external_call

from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    _cephes_sincosf_core,
    _fma_f32,
    _ftz_always,
    portable_cosf,
    portable_sinf,
    numeric_mode_name,
)

comptime N = 1 << 20
comptime BLOCK = 256

# ---- the four arms (DEVIATION 940: local spellings, not source edits) ----

comptime SAB_NO_SECOND_FLIP = is_defined[
    "MOJOLEARN_TRIG_SABOTAGE_NO_SECOND_FLIP"
]()
"""Drop the core's `if is_cos and j > 1` second sign flip. Cephes's cosf
flips twice and its sinf once; this is difference (2) of the three the core
carries, so it must move cos alone."""

comptime SAB_POLY_NOT_SWAPPED = is_defined[
    "MOJOLEARN_TRIG_SABOTAGE_POLY_NOT_SWAPPED"
]()
"""Drop the core's `if not is_cos: use_sin_poly = not use_sin_poly`. That
is difference (3): on j in {1, 2} cos evaluates the SIN polynomial and sin
evaluates the COS one. Dropping it gives sin cos's octant table, so it must
move sin alone."""

comptime SAB_SIGN_BY_COMPARE = is_defined[
    "MOJOLEARN_TRIG_SABOTAGE_SIGN_BY_COMPARE"
]()
"""Cephes's own `if (xx < 0)`, which DEVIATION 820 refused, and it is
wrong for TWO independent reasons rather than one. `-0.0 < 0.0` is FALSE
on EVERY column while `-0.0`'s sign bit is SET, so `sin(-0.0)` comes back
`+0.0` here and `-0.0` from the real function -- that half needs no
flushing and bites on the host. And on Metal `-subnormal < 0.0` is FALSE
too (DEVIATION 746 (i)), so every negative subnormal comes back `+0.0` as
well -- that half is per-column. NOT bit-inert on the host: the first
version of this file said it was, and DEVIATION 949 is the correction."""

comptime SAB_NO_INPUT_FLUSH = is_defined[
    "MOJOLEARN_TRIG_SABOTAGE_NO_INPUT_FLUSH"
]()
"""Drop `_ftz_always` from `portable_sinf`. Near zero sin returns its
argument, so an unflushed subnormal survives to the output on a
denormal-honoring column and is a signed zero on Metal -- the divergence
DEVIATION 820's flush exists to close."""

comptime ANY_SABOTAGE = (
    SAB_NO_SECOND_FLIP
    or SAB_POLY_NOT_SWAPPED
    or SAB_SIGN_BY_COMPARE
    or SAB_NO_INPUT_FLUSH
)


def trig_sabotage_name() -> String:
    """The armed arm, printed so a `-D` that was misspelled and silently
    dropped cannot be read as a clean run."""
    comptime if SAB_NO_SECOND_FLIP:
        return String("NO_SECOND_FLIP")
    comptime if SAB_POLY_NOT_SWAPPED:
        return String("POLY_NOT_SWAPPED")
    comptime if SAB_SIGN_BY_COMPARE:
        return String("SIGN_BY_COMPARE")
    comptime if SAB_NO_INPUT_FLUSH:
        return String("NO_INPUT_FLUSH")
    return String("none")


# ---- the local re-spellings ---------------------------------------------
def _sab_sincos_core(
    x_abs: Float32, sign_in: Float32, is_cos: Bool
) -> Float32:
    """A faithful copy of `_cephes_sincosf_core` (numerics.mojo) with the
    two structural arms folded in. With no arm armed it must be bit-equal
    to the original on every lane, and `check_sabotage_arms` asserts
    exactly that -- the copy is validated by the same run that uses it."""
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
    comptime if not SAB_NO_SECOND_FLIP:
        if is_cos and j > 1:
            sign = -sign
    var z = x * x
    var use_sin_poly = j == 1 or j == 2
    comptime if not SAB_POLY_NOT_SWAPPED:
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


def _sab_sinf(x_in: Float32) -> Float32:
    """A faithful copy of `portable_sinf` carrying the two sin-only arms."""
    if x_in != x_in or abs(x_in) == bitcast[DType.float32](UInt32(0x7F800000)):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    var x = x_in
    comptime if not SAB_NO_INPUT_FLUSH:
        x = _ftz_always(x_in)
    var sign = Float32(1.0)
    comptime if SAB_SIGN_BY_COMPARE:
        # Cephes's own spelling, and it loses the sign in two places.
        # `-0.0 < 0.0` is FALSE on every column, so this misses a
        # negative zero everywhere; and the compare operand is the
        # UNFLUSHED argument on purpose, which is where Metal's compare
        # flush additionally loses every negative subnormal.
        if x_in < Float32(0.0):
            sign = Float32(-1.0)
    else:
        if (rebind[UInt32](x.to_bits()) & UInt32(0x80000000)) != UInt32(0):
            sign = Float32(-1.0)
    return _sab_sincos_core(abs(x), sign, False)


def _sab_cosf(x_in: Float32) -> Float32:
    """A faithful copy of `portable_cosf`. It carries no arm of its own;
    the two structural arms reach it through `_sab_sincos_core`, which is
    the point -- they are the core's decisions, not cos's."""
    if x_in != x_in or abs(x_in) == bitcast[DType.float32](UInt32(0x7F800000)):
        return bitcast[DType.float32](UInt32(0x7FC00000))
    return _sab_sincos_core(abs(x_in), Float32(1.0), True)


def trig_kernel(
    xs: MutPointer[Float32, MutAnyOrigin],
    ones: MutPointer[Float32, MutAnyOrigin],
    out_sin: MutPointer[Float32, MutAnyOrigin],
    out_cos: MutPointer[Float32, MutAnyOrigin],
    out_sin_sab: MutPointer[Float32, MutAnyOrigin],
    out_cos_sab: MutPointer[Float32, MutAnyOrigin],
    out_vendor: MutPointer[Float32, MutAnyOrigin],
    out_probe: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var x = xs.unsafe_load(i)
        out_sin.unsafe_store(i, portable_sinf(x))
        out_cos.unsafe_store(i, portable_cosf(x))
        out_sin_sab.unsafe_store(i, _sab_sinf(x))
        out_cos_sab.unsafe_store(i, _sab_cosf(x))
        # `identical_sin`'s FAST arm, compiled and launched here whatever
        # the build mode: RECORDED, never asserted, and the only reason it
        # exists is that a FAST arm nobody has ever put on a device is a
        # claim rather than a path.
        out_vendor.unsafe_store(i, sin(x))
        # the flush probe. `ones` is a runtime buffer so nothing can fold
        # the multiply away (DEVIATION 663's scar).
        out_probe.unsafe_store(i, x * ones.unsafe_load(0))


def _splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _bits(x: Float32) -> UInt32:
    return rebind[UInt32](x.to_bits())


def _f(b: UInt32) -> Float32:
    return bitcast[DType.float32](b)


def _is_nan(b: UInt32) -> Bool:
    return (b & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000)


def _canon(b: UInt32) -> UInt32:
    if _is_nan(b):
        return UInt32(0x7FC00000)
    return b


def _same(a: UInt32, b: UInt32) -> Bool:
    return _canon(a) == _canon(b)


def _ord(b: UInt32) -> Int64:
    if b & UInt32(0x80000000):
        return Int64(0xFFFFFFFF) - Int64(b & UInt32(0x7FFFFFFF))
    return Int64(0x100000000) + Int64(b)


def _ulp_dist(a: Float32, b: Float32) -> Int64:
    var d = _ord(_bits(a)) - _ord(_bits(b))
    if d < Int64(0):
        return -d
    return d


def _finite(b: UInt32) -> Bool:
    return (b & UInt32(0x7F800000)) != UInt32(0x7F800000)


def _normal(b: UInt32) -> Bool:
    var e = b & UInt32(0x7F800000)
    return e != UInt32(0) and e != UInt32(0x7F800000)


def _sin64(x: Float64) -> Float64:
    return external_call["sin", Float64](x)


def _cos64(x: Float64) -> Float64:
    return external_call["cos", Float64](x)


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _expect(name: String, got: Float32, want_bits: UInt32) raises:
    """NaN-tolerant: a payload is not a mismatch."""
    if not _same(_bits(got), want_bits):
        raise Error(
            name + ": planted value wrong, got " + hex(_bits(got))
            + " want " + hex(want_bits)
        )


def _expect_raw(name: String, got: Float32, want_bits: UInt32) raises:
    """BY SIGN BIT. `assert_equal` on floats passes on either zero, which
    is the trap `adv_signed_zeros` found in the mamba corpus, so every
    signed-zero contract in this file goes through here and not through
    `_expect`."""
    if _bits(got) != want_bits:
        raise Error(
            name + ": wrong BITS, got " + hex(_bits(got))
            + " want " + hex(want_bits)
        )


# ---- the fixture ---------------------------------------------------------
# 48 planted lanes, then hashed. NO RAW 32-BIT PATTERNS ABOVE THE DOMAIN:
# `_cephes_sincosf_core` computes `Int(floor(x * 4/pi))`, and for a raw
# pattern like 1e30 that integer conversion is out of range and its result
# is not defined by the language. A gate must not compare undefined against
# undefined, so the sweep stays inside |x| < 8192 and
# `check_rope_domain_edge` probes above the edge at PLANTED values only,
# where the answer can be read.
comptime NPLANT = 48


def _trig_x(i: Int) -> Float32:
    if i == 0:
        return _f(UInt32(0x00000000))  # +0
    if i == 1:
        return _f(UInt32(0x80000000))  # -0
    if i == 2:
        return _f(UInt32(0x7FC00000))  # quiet NaN
    if i == 3:
        return _f(UInt32(0x7FFFFFFF))  # NaN, other payload
    if i == 4:
        return _f(UInt32(0xFFC00000))  # NaN, negative payload
    if i == 5:
        return _f(UInt32(0x7F800000))  # +inf
    if i == 6:
        return _f(UInt32(0xFF800000))  # -inf
    if i == 7:
        return _f(UInt32(0x00000001))  # smallest subnormal
    if i == 8:
        return _f(UInt32(0x80000001))
    if i == 9:
        return _f(UInt32(0x00400000))
    if i == 10:
        return _f(UInt32(0x80400000))
    if i == 11:
        return _f(UInt32(0x007FFFFF))  # largest subnormal
    if i == 12:
        return _f(UInt32(0x807FFFFF))
    if i == 13:
        return _f(UInt32(0x00800000))  # FLT_MIN
    if i == 14:
        return _f(UInt32(0x80800000))
    if i >= 15 and i < 31:
        # k*pi/4, k = -8..7 -- the octant boundaries, PLANTED. Row
        # `uniform-test-data-hides-permutation`: a uniform sweep enters the
        # octant table unevenly, so the boundaries are placed by hand.
        return Float32(Float64(i - 23) * 0.7853981633974483)
    if i >= 31 and i < 39:
        return Float32(Float64(i - 35) * 1.5707963267948966)
    if i == 39:
        return Float32(1.0)
    if i == 40:
        return Float32(-1.0)
    if i == 41:
        return Float32(8191.0)
    if i == 42:
        return Float32(-8191.0)
    if i == 43:
        return Float32(0.5)
    if i == 44:
        return Float32(-0.5)
    if i == 45:
        return Float32(1e-10)
    if i == 46:
        return Float32(-1e-10)
    if i == 47:
        return Float32(1e-20)
    var h = _splitmix(UInt64(7 * i + 3))
    var sel = (h >> 40) & UInt64(7)
    var u = Float64(h & UInt64(0xFFFFFFFF)) / Float64(4294967296.0)
    if sel == UInt64(0):
        # 1/8 of the sweep is a subnormal of a hashed sign and mantissa.
        # The two flush arms below need these at density; 2^17 lanes is
        # enough that a per-lane split cannot hide in a rounding.
        var mant = UInt32(h & UInt64(0x007FFFFF))
        if mant == UInt32(0):
            mant = UInt32(1)
        var sgn = UInt32(0)
        if ((h >> 32) & UInt64(1)) == UInt64(1):
            sgn = UInt32(0x80000000)
        return _f(mant | sgn)
    if sel <= UInt64(4):
        # half the sweep in RoPE's own busy range
        return Float32(-6.283185307179586 + 12.566370614359172 * u)
    # the rest across the reduction's domain
    return Float32(-8191.0 + 16382.0 * u)


# ---- the sqrtcos cos fixture, reproduced -------------------------------
def _sqrtcos_cos_x(i: Int) -> Float32:
    """`mojo_only/portable_sqrtcos_check.mojo`'s cos column, lane for lane.
    Hashing `portable_cosf` over exactly the inputs the certified gate uses
    gives a witness that moves if and only if the refactored cos moved on
    that gate's own fixture. It is NOT the sqrtcos hash (that one
    interleaves sqrt and pow) and must not be reported as one."""
    if i == 0:
        return Float32(0.0)
    if i == 1:
        return Float32(1.5707963267948966)
    if i == 2:
        return Float32(3.141592653589793)
    if i == 3:
        return Float32(6.283185307179586)
    if i == 4:
        return Float32(-0.7853981633974483)
    if i == 5:
        return Float32(2.356194490192345)
    if i == 6:
        return Float32(0.5)
    if i == 7:
        return Float32(8191.0)
    var h2 = _splitmix(UInt64(3 * i + 2))
    var u = Float64(h2 & UInt64(0xFFFFFFFF)) / Float64(4294967296.0)
    if (h2 >> 40) & UInt64(1) == UInt64(0):
        return Float32(6.283185307179586 * u)
    return Float32(-8192.0 + 16384.0 * u)


# ==========================================================================
# THE CHECKS
# ==========================================================================
def check_cos_bits_did_not_move() raises:
    """RUNS FIRST, and it is the point of this file: `portable_cosf` is
    already certified and DEVIATION 820 moved its body. Three things are
    asserted and one is printed.

    (a) `portable_cosf(x)` is EXACTLY `_cephes_sincosf_core(|x|, +1, True)`
        on a scattered fixture. That is the factoring's own claim, written
        as an assertion instead of as a paragraph.
    (b) cos's special-value contract, by bits.
    (c) cos does NOT flush its input -- cos of every subnormal is exactly
        1.0 on both columns, which is why DEVIATION 820 left the certified
        function alone. Asserted here so a later "consistency" edit that
        adds the flush is caught as the change to a certified function that
        it would be.

    Printed: `trig cos-fixture hash`, cos over the sqrtcos gate's own cos
    inputs. The AUTHORITATIVE proof is still
    `pixi run check-portable-sqrtcos` printing 12295913102197186379."""
    for i in range(NPLANT):
        var x = _trig_x(i)
        var b = _bits(x)
        if _is_nan(b) or (b & UInt32(0x7FFFFFFF)) == UInt32(0x7F800000):
            continue
        var got = portable_cosf(x)
        var want = _cephes_sincosf_core(abs(x), Float32(1.0), True)
        if _bits(got) != _bits(want):
            raise Error(
                "portable_cosf no longer equals the shared core at lane "
                + String(i) + ": " + hex(_bits(got)) + " vs "
                + hex(_bits(want))
            )
    for i in range(NPLANT, NPLANT + 4096):
        var x = _trig_x(i)
        var got = portable_cosf(x)
        var want = _cephes_sincosf_core(abs(x), Float32(1.0), True)
        if _bits(got) != _bits(want):
            raise Error(
                "portable_cosf no longer equals the shared core at sweep lane "
                + String(i)
            )
    comptime PINF = UInt32(0x7F800000)
    comptime NINF = UInt32(0xFF800000)
    comptime QNAN = UInt32(0x7FC00000)
    comptime ONE = UInt32(0x3F800000)
    _expect("cos(NaN)", portable_cosf(_f(QNAN)), QNAN)
    _expect("cos(+inf)", portable_cosf(_f(PINF)), QNAN)
    _expect("cos(-inf)", portable_cosf(_f(NINF)), QNAN)
    _expect_raw("cos(+0)", portable_cosf(_f(UInt32(0))), ONE)
    _expect_raw("cos(-0)", portable_cosf(_f(UInt32(0x80000000))), ONE)
    # (c) the no-flush clause, both signs and both ends of the subnormal band
    _expect_raw("cos(subnormal)", portable_cosf(_f(UInt32(0x00000001))), ONE)
    _expect_raw("cos(-subnormal)", portable_cosf(_f(UInt32(0x80000001))), ONE)
    _expect_raw("cos(largest subnormal)", portable_cosf(_f(UInt32(0x007FFFFF))), ONE)
    _expect_raw("cos(-largest subnormal)", portable_cosf(_f(UInt32(0x807FFFFF))), ONE)
    var h = UInt64(0xCBF29CE484222325)
    for i in range(N):
        var c = UInt64(_canon(_bits(portable_cosf(_sqrtcos_cos_x(i)))))
        h = (h ^ c) * UInt64(0x100000001B3)
    print("check_cos_bits_did_not_move: cos == the shared core on every")
    print("  planted and swept lane; the no-flush clause holds")
    print("  trig cos-fixture hash =", h)
    print("  (NOT the sqrtcos hash. Run `pixi run check-portable-sqrtcos`")
    print("   and confirm `sqrtcos device hash` is still 12295913102197186379.)")


def check_sin_cos_share_one_reduction() raises:
    """A future edit that gives sin its own reduction must fail HERE and
    not in a model. Both entries are asserted against the core called
    directly, with sin's two pre-steps (flush, then the sign taken by BITS)
    spelled out so that a change to either is visible."""
    var bad = 0
    for i in range(NPLANT + 65536):
        var x = _trig_x(i)
        var b = _bits(x)
        if _is_nan(b) or (b & UInt32(0x7FFFFFFF)) == UInt32(0x7F800000):
            continue
        var xf = _ftz_always(x)
        var sgn = Float32(1.0)
        if (_bits(xf) & UInt32(0x80000000)) != UInt32(0):
            sgn = Float32(-1.0)
        var want_sin = _cephes_sincosf_core(abs(xf), sgn, False)
        var want_cos = _cephes_sincosf_core(abs(x), Float32(1.0), True)
        if _bits(portable_sinf(x)) != _bits(want_sin):
            bad += 1
        if _bits(portable_cosf(x)) != _bits(want_cos):
            bad += 1
    if bad != 0:
        raise Error(
            "sin and cos no longer share ONE reduction: " + String(bad)
            + " lanes disagree with `_cephes_sincosf_core` called directly"
        )
    print("check_sin_cos_share_one_reduction: both entries are the shared core")


def check_sin_planted() raises:
    """Row 39, BY SIGN BIT. `sin(-0.0)` is the one that matters: it is
    `-0.0` here and `+0.0` in Cephes (whose `if (xx < 0)` is false for a
    negative zero), it is what IEEE, torch and numpy say, and an
    `assert_equal` on floats cannot see the difference."""
    comptime PZ = UInt32(0x00000000)
    comptime NZ = UInt32(0x80000000)
    comptime PINF = UInt32(0x7F800000)
    comptime NINF = UInt32(0xFF800000)
    comptime QNAN = UInt32(0x7FC00000)
    _expect_raw("sin(+0)", portable_sinf(_f(PZ)), PZ)
    _expect_raw("sin(-0)", portable_sinf(_f(NZ)), NZ)
    _expect("sin(NaN)", portable_sinf(_f(QNAN)), QNAN)
    _expect("sin(NaN, other payload)", portable_sinf(_f(UInt32(0x7FFFFFFF))), QNAN)
    _expect("sin(NaN, negative payload)", portable_sinf(_f(UInt32(0xFFC00000))), QNAN)
    _expect("sin(+inf)", portable_sinf(_f(PINF)), QNAN)
    _expect("sin(-inf)", portable_sinf(_f(NINF)), QNAN)
    # the subnormal contract, BOTH signs and BOTH ends of the band. This is
    # the one that separates a bit-taken sign from a compare-written one.
    _expect_raw("sin(smallest subnormal)", portable_sinf(_f(UInt32(0x00000001))), PZ)
    _expect_raw("sin(-smallest subnormal)", portable_sinf(_f(UInt32(0x80000001))), NZ)
    _expect_raw("sin(subnormal)", portable_sinf(_f(UInt32(0x00400000))), PZ)
    _expect_raw("sin(-subnormal)", portable_sinf(_f(UInt32(0x80400000))), NZ)
    _expect_raw("sin(largest subnormal)", portable_sinf(_f(UInt32(0x007FFFFF))), PZ)
    _expect_raw("sin(-largest subnormal)", portable_sinf(_f(UInt32(0x807FFFFF))), NZ)
    print("check_sin_planted: every special value by BITS, both zeros by SIGN")


def check_sin_odd() raises:
    """`portable_sinf(-x) == -portable_sinf(x)` BY BITS over the sweep.
    The bit-taken sign makes this exact rather than approximate, and it is
    the cheapest single assertion that catches a wrong octant flip: an
    octant table that is off by one is even somewhere."""
    var bad = 0
    var first = Float32(0.0)
    for i in range(N):
        var x = _trig_x(i)
        var b = _bits(x)
        if _is_nan(b) or (b & UInt32(0x7FFFFFFF)) == UInt32(0x7F800000):
            continue
        var lhs = portable_sinf(-x)
        var rhs = -portable_sinf(x)
        if _bits(lhs) != _bits(rhs):
            if bad == 0:
                first = x
            bad += 1
    if bad != 0:
        raise Error(
            "portable_sinf is not odd on " + String(bad) + " lanes, first x = "
            + String(first)
        )
    print("check_sin_odd: sin(-x) == -sin(x) BY BITS on every finite lane")


def check_sin_quadrants() raises:
    """sin and cos at k*pi/4 and k*pi/2 for k in -16..16, PLANTED rather
    than sampled, plus a tally of which octant state each one reached.

    A DISAGREEMENT WITH THE SPECIFICATION, RECORDED RATHER THAN
    IMPLEMENTED AS WRITTEN. The specification asks for "all eight octants".
    There are not eight reachable states. Cephes evenizes the octant index
    (`if (j & 1) { j += 1; y += 1; }`) BEFORE the `j &= 7`, so j is always
    even and only {0, 2, 4, 6} can occur; {1, 3, 5, 7} are unreachable by
    construction. This check therefore requires all FOUR reachable states
    and records the other four as structurally dead, which is the honest
    version of the same coverage claim. Requiring eight would have been a
    clause nobody can satisfy."""
    var seen0 = 0
    var seen2 = 0
    var seen4 = 0
    var seen6 = 0
    var odd_seen = 0
    var max_ulp_sin = Int64(0)
    var max_ulp_cos = Int64(0)
    var worst = Float32(0.0)
    for kk in range(-16, 17):
        for which in range(2):
            var step = Float64(0.7853981633974483)
            if which == 1:
                step = Float64(1.5707963267948966)
            var x = Float32(Float64(kk) * step)
            # the octant state, computed exactly as the core computes it
            var j = Int(floor(abs(x) * Float32(1.27323954473516)))
            if (j & 1) != 0:
                j += 1
            j = j & 7
            if j == 0:
                seen0 += 1
            elif j == 2:
                seen2 += 1
            elif j == 4:
                seen4 += 1
            elif j == 6:
                seen6 += 1
            else:
                odd_seen += 1
            var ds = _ulp_dist(portable_sinf(x), Float32(_sin64(Float64(x))))
            var dc = _ulp_dist(portable_cosf(x), Float32(_cos64(Float64(x))))
            if ds > max_ulp_sin:
                max_ulp_sin = ds
                worst = x
            if dc > max_ulp_cos:
                max_ulp_cos = dc
    if seen0 == 0 or seen2 == 0 or seen4 == 0 or seen6 == 0:
        raise Error(
            "the planted quadrant angles do not reach every octant state: "
            + String(seen0) + "/" + String(seen2) + "/" + String(seen4)
            + "/" + String(seen6)
        )
    if odd_seen != 0:
        raise Error(
            "an ODD octant index was reached (" + String(odd_seen)
            + " times), which the evenize step makes impossible -- the"
            + " reduction is not the one this check models"
        )
    print("check_sin_quadrants: octant states reached j=0:", seen0, " j=2:", seen2,
          " j=4:", seen4, " j=6:", seen6, "; odd j unreachable by construction")
    print("  RECORDED at the planted quadrant angles: sin max", max_ulp_sin,
          "ulp (worst x =", worst, "), cos max", max_ulp_cos, "ulp")


comptime NEDGE = 12


def _edge_x(t: Int) -> Float64:
    """The domain ladder, as a pure function rather than a container: a
    `List` is not implicitly copyable in this Mojo and a gate has no
    business owning one for twelve constants."""
    if t == 0:
        return Float64(1.0)
    if t == 1:
        return Float64(10.0)
    if t == 2:
        return Float64(100.0)
    if t == 3:
        return Float64(1000.0)
    if t == 4:
        return Float64(4096.0)
    if t == 5:
        return Float64(8191.0)
    if t == 6:
        return Float64(8191.9995)
    if t == 7:
        return Float64(8192.0)
    if t == 8:
        return Float64(8192.5)
    if t == 9:
        return Float64(16384.0)
    if t == 10:
        return Float64(65536.0)
    return Float64(1000000.0)


def check_rope_domain_edge() raises:
    """DEVIATION 943: THIS ONE PRODUCES A NUMBER, NOT A VERDICT.

    `portable_sinf`'s reduction is documented valid for |x| < 8192, and a
    Llama context length of 8192 lands exactly on the edge (RoPE's angle is
    `position * inv_freq` with `inv_freq` in (0, 1], so position 8191 gives
    at most 8191). The transformer lane has to decide whether to bound the
    angle, and a pass/fail at 8192 would not help it decide: the Cody-Waite
    error does not JUMP at 8192, it GROWS with |x| (the reduced argument
    carries about `ulp(x)` of error, so the absolute error in sin is about
    `x * 2^-24`). 8192 is where Cephes stopped calling it acceptable, not
    where it breaks.

    So this prints the ladder. What the lane needs is the error AT ITS OWN
    CONTEXT LENGTH, and that is the column below. NOTHING HERE IS
    ASSERTED except that the answer stays finite, because a NaN or an
    infinity above the edge would be a different failure mode entirely and
    the caller must not be told "wrong by 5e-4" when the truth is "not a
    number"."""
    print("check_rope_domain_edge: RECORDED, not asserted.")
    print("     x            sin abs err vs libm    cos abs err vs libm")
    for t in range(NEDGE):
        var xd = _edge_x(t)
        var x = Float32(xd)
        var ys = portable_sinf(x)
        var yc = portable_cosf(x)
        if not _finite(_bits(ys)) or not _finite(_bits(yc)):
            raise Error(
                "portable trig returned a non-finite value at x = "
                + String(xd) + ": that is a different failure from a lost"
                + " reduction and the caller must not be told otherwise"
            )
        var es = abs(Float64(ys) - _sin64(Float64(x)))
        var ec = abs(Float64(yc) - _cos64(Float64(x)))
        print("  ", x, "   ", es, "   ", ec)
    print("  READ IT AS: the reduction degrades smoothly; 8192 is Cephes's")
    print("  `lossth`, not a cliff. RoPE at context 8192 uses |x| <= 8191.")


def check_sabotage_arms(
    host_sin: MutPointer[Float32, MutUntrackedOrigin],
    host_cos: MutPointer[Float32, MutUntrackedOrigin],
    host_sin_sab: MutPointer[Float32, MutUntrackedOrigin],
    host_cos_sab: MutPointer[Float32, MutUntrackedOrigin],
    dev_sin: MutPointer[Float32, MutUntrackedOrigin],
    dev_cos: MutPointer[Float32, MutUntrackedOrigin],
    dev_sin_sab: MutPointer[Float32, MutUntrackedOrigin],
    dev_cos_sab: MutPointer[Float32, MutUntrackedOrigin],
    xs: MutPointer[Float32, MutUntrackedOrigin],
    device_flushes: Bool,
) raises:
    """DEVIATION 941's four counters, and the per-arm judgment.

    `xs` is here so that a moved lane can be described BY ITS INPUT rather
    than only counted. DEVIATION 949: the first version of this file
    asserted SIGN_BY_COMPARE was host-inert, the run found one moved host
    lane, and nobody could tell from the output WHICH lane it was. A
    counter that cannot name what it counted costs a round."""
    var host_reach_sin = 0
    var host_reach_cos = 0
    var dev_reach_sin = 0
    var dev_reach_cos = 0
    var arm2_sab_sin = 0
    var arm2_sab_cos = 0
    # the fixture census the SIGN_BY_COMPARE clause is an exact count over
    var n_neg_zero = 0
    var n_neg_sub = 0
    # the first three moved host lanes, described rather than tallied
    var moved_n = 0
    var moved_i0 = -1
    var moved_i1 = -1
    var moved_i2 = -1
    for i in range(N):
        var xb = _bits(xs.unsafe_load(i))
        if xb == UInt32(0x80000000):
            n_neg_zero += 1
        elif (
            (xb & UInt32(0x80000000)) != UInt32(0)
            and (xb & UInt32(0x7F800000)) == UInt32(0)
            and (xb & UInt32(0x007FFFFF)) != UInt32(0)
        ):
            n_neg_sub += 1
        var hs = _bits(host_sin.unsafe_load(i))
        var hc = _bits(host_cos.unsafe_load(i))
        var hss = _bits(host_sin_sab.unsafe_load(i))
        var hcs = _bits(host_cos_sab.unsafe_load(i))
        var ds = _bits(dev_sin.unsafe_load(i))
        var dc = _bits(dev_cos.unsafe_load(i))
        var dss = _bits(dev_sin_sab.unsafe_load(i))
        var dcs = _bits(dev_cos_sab.unsafe_load(i))
        if not _same(hs, hss):
            host_reach_sin += 1
            if moved_n == 0:
                moved_i0 = i
            elif moved_n == 1:
                moved_i1 = i
            elif moved_n == 2:
                moved_i2 = i
            moved_n += 1
        if not _same(hc, hcs):
            host_reach_cos += 1
        if not _same(ds, dss):
            dev_reach_sin += 1
        if not _same(dc, dcs):
            dev_reach_cos += 1
        if not _same(dss, hss):
            arm2_sab_sin += 1
        if not _same(dcs, hcs):
            arm2_sab_cos += 1
    print("reach counters (portable vs the local spelling):")
    print("  host reach   sin", host_reach_sin, " cos", host_reach_cos)
    print("  device reach sin", dev_reach_sin, " cos", dev_reach_cos)
    print("  arm 2 under sabotage (dev(sab) vs host(sab)) sin", arm2_sab_sin,
          " cos", arm2_sab_cos)
    print("  fixture census:", n_neg_zero, "inputs are exactly -0.0,",
          n_neg_sub, "are negative subnormals")
    # EVERY moved host lane is DESCRIBED, not just counted, so the next
    # reader does not have to re-derive which input separated the two
    # spellings (DEVIATION 949).
    if moved_n != 0:
        print("  the first", 3 if moved_n > 3 else moved_n,
              "moved host SIN lane(s), described by their INPUT:")
        for t in range(3):
            var mi = moved_i0
            if t == 1:
                mi = moved_i1
            elif t == 2:
                mi = moved_i2
            if mi < 0:
                continue
            print("    lane", mi, " x =", hex(_bits(xs.unsafe_load(mi))),
                  " portable sin =", hex(_bits(host_sin.unsafe_load(mi))),
                  " sabotaged sin =", hex(_bits(host_sin_sab.unsafe_load(mi))))

    comptime if not ANY_SABOTAGE:
        if (host_reach_sin != 0 or host_reach_cos != 0
                or dev_reach_sin != 0 or dev_reach_cos != 0):
            raise Error(
                "NO arm is armed and the local spelling still disagrees with"
                + " the real function: the copy in this file has DRIFTED and"
                + " every sabotage result it produces is worthless"
            )
        print("  clean build: the local spelling is bit-equal to the real one")
    comptime if SAB_NO_SECOND_FLIP:
        if host_reach_cos == 0:
            raise Error(
                "NO_SECOND_FLIP moved nothing in cos: the second sign flip is"
                + " not reached, so `check_cos_bits_did_not_move` is blind"
            )
        if host_reach_sin != 0:
            raise Error(
                "NO_SECOND_FLIP moved SIN on " + String(host_reach_sin)
                + " lanes. It is cos's flip alone; if it moves sin then sin"
                + " is taking cos's path through the core"
            )
        print("  NO_SECOND_FLIP: cos moved, sin did not -- as predicted")
    comptime if SAB_POLY_NOT_SWAPPED:
        if host_reach_sin == 0:
            raise Error(
                "POLY_NOT_SWAPPED moved nothing in sin: the polynomial swap"
                + " is not reached and sin's octant table is ungated"
            )
        if host_reach_cos != 0:
            raise Error(
                "POLY_NOT_SWAPPED moved COS on " + String(host_reach_cos)
                + " lanes, which it cannot do -- the swap is guarded by"
                + " `if not is_cos`"
            )
        print("  POLY_NOT_SWAPPED: sin moved, cos did not -- as predicted")
    comptime if SAB_SIGN_BY_COMPARE:
        # DEVIATION 949. THIS CLAUSE PINS TWO SEPARATE PROPERTIES AND BOTH
        # ARE EXACT COUNTS. The first version asserted host inertness "by
        # construction" and was WRONG: `-0.0 < 0.0` is FALSE while
        # `-0.0`'s sign bit is SET, so a negative zero separates the two
        # spellings on EVERY column with no flushing involved. That is
        # DEVIATION 820's knowing deviation from Cephes, and this arm is
        # exactly the sabotage that undoes it.
        if host_reach_cos != 0:
            raise Error(
                "SIGN_BY_COMPARE moved COS on " + String(host_reach_cos)
                + " lanes. Cos takes no sign from its argument at all, so"
                + " it has no such decision to undo"
            )
        # (i) the -0.0 half, true on every column
        if host_reach_sin != n_neg_zero:
            raise Error(
                "SIGN_BY_COMPARE moved " + String(host_reach_sin)
                + " HOST lanes but the fixture carries exactly "
                + String(n_neg_zero)
                + " inputs of -0.0, and -0.0 is the ONLY input that"
                + " separates a bit-taken sign from a compare-taken one"
                + " without a flush. Read the moved-lane list printed"
                + " above: if the extra lanes are not -0.0 then this arm"
                + " is reaching a decision nobody has modelled, and THAT"
                + " is the finding"
            )
        if moved_i0 >= 0:
            var mb = _bits(xs.unsafe_load(moved_i0))
            if mb != UInt32(0x80000000):
                raise Error(
                    "SIGN_BY_COMPARE's moved host lane has input " + hex(mb)
                    + ", not -0.0 (0x80000000). The count matched by"
                    + " coincidence and the clause is pinning the wrong"
                    + " thing"
                )
            if _bits(host_sin.unsafe_load(moved_i0)) != UInt32(0x80000000):
                raise Error(
                    "portable_sinf(-0.0) is not -0.0 in this very run, so"
                    + " DEVIATION 820's IEEE sign rule is not live and this"
                    + " arm has nothing to undo"
                )
            if _bits(host_sin_sab.unsafe_load(moved_i0)) != UInt32(0x00000000):
                raise Error(
                    "the compare-taken sign did not give Cephes's +0.0 at"
                    + " -0.0; the arm is not the arm it claims to be"
                )
        print("  SIGN_BY_COMPARE (i): host reach", host_reach_sin,
              "= the fixture's", n_neg_zero, "input(s) of -0.0.")
        print("    portable sin(-0.0) = 0x80000000, sabotaged = 0x00000000.")
        print("    DEVIATION 820's departure from Cephes is LIVE on the host.")
        # (ii) the negative-subnormal half, per column
        if device_flushes:
            if dev_reach_sin != n_neg_zero + n_neg_sub:
                raise Error(
                    "SIGN_BY_COMPARE device reach is " + String(dev_reach_sin)
                    + " but this column WAS MEASURED to flush and the"
                    + " fixture carries " + String(n_neg_zero) + " -0.0 plus "
                    + String(n_neg_sub)
                    + " negative subnormals, every one of which must move."
                    + " A shortfall means the compare flush did not reach"
                    + " every lane; an excess means the arm moves something"
                    + " that is neither"
                )
            print("  SIGN_BY_COMPARE (ii): flushing column, device reach",
                  dev_reach_sin, "=", n_neg_zero, "+", n_neg_sub,
                  "-- every negative subnormal moved and nothing else did")
        else:
            if dev_reach_sin != n_neg_zero:
                raise Error(
                    "SIGN_BY_COMPARE device reach is " + String(dev_reach_sin)
                    + " on a column measured NOT to flush, where only the "
                    + String(n_neg_zero) + " -0.0 input(s) can move"
                )
            print("  SIGN_BY_COMPARE (ii): this column HONORS denormals, so")
            print("    only the -0.0 lane moves and device reach is",
                  dev_reach_sin, "as it must be. The subnormal half of this")
            print("    arm is EXPECTED INERT here; it needs Metal.")
    comptime if SAB_NO_INPUT_FLUSH:
        if host_reach_sin == 0:
            raise Error(
                "NO_INPUT_FLUSH moved nothing on the host, which it must:"
                + " without the flush the host returns the subnormal itself"
                + " where `portable_sinf` returns its signed zero"
            )
        if host_reach_cos != 0:
            raise Error(
                "NO_INPUT_FLUSH moved COS, which has no flush to drop"
            )
        print("  NO_INPUT_FLUSH: host reach", host_reach_sin,
              "-- the flush is reached")
        if device_flushes:
            if arm2_sab_sin == 0:
                raise Error(
                    "NO_INPUT_FLUSH: the unflushed spelling agrees between"
                    + " host and device on a column that flushes. That is the"
                    + " divergence DEVIATION 820 exists to close and this run"
                    + " cannot see it"
                )
            print("    and it separates host from device:", arm2_sab_sin,
                  "lanes -- the arm 2 failure the specification predicts")
        else:
            print("    device does not flush, so arm 2 stays clean here;")
            print("    the cross-column split needs Metal.")


def main() raises:
    for a in argv():
        if a == "sabotage":
            print("NOTE: this gate takes its reach arms as `-D` defines, not")
            print("      as an argv word. See the module docstring.")

    print("portable trig check (DEVIATION 820); build mode", _mode_name(),
          "(the portable_* functions do not depend on it)")
    print("trig sabotage", trig_sabotage_name())

    check_cos_bits_did_not_move()
    check_sin_cos_share_one_reduction()
    check_sin_planted()
    check_sin_odd()
    check_sin_quadrants()
    check_rope_domain_edge()

    var ctx = DeviceContext()

    # ---- the fixture --------------------------------------------------
    var h_x = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_one = ctx.enqueue_create_host_buffer[DType.float32](1)
    var px = h_x.unsafe_ptr()
    h_one.unsafe_ptr().unsafe_store(0, Float32(1.0))
    var n_sub = 0
    var n_neg_sub = 0
    for i in range(N):
        var x = _trig_x(i)
        px.unsafe_store(i, x)
        var b = _bits(x)
        if (b & UInt32(0x7F800000)) == UInt32(0) and (b & UInt32(0x007FFFFF)) != UInt32(0):
            n_sub += 1
            if (b & UInt32(0x80000000)) != UInt32(0):
                n_neg_sub += 1
    print("fixture:", N, "lanes,", n_sub, "subnormal (", n_neg_sub, "negative)")
    if n_neg_sub < 1000:
        raise Error(
            "the fixture carries only " + String(n_neg_sub)
            + " negative subnormals: the two flush arms cannot be judged"
        )

    # ---- arm 1: accuracy on the host, RECORDED -------------------------
    var h_sin = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_cos = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_sin_sab = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_cos_sab = ctx.enqueue_create_host_buffer[DType.float32](N)
    var ps = h_sin.unsafe_ptr()
    var pc = h_cos.unsafe_ptr()
    var pss = h_sin_sab.unsafe_ptr()
    var pcs = h_cos_sab.unsafe_ptr()
    var max_ulp_sin = Int64(0)
    var max_ulp_sin_rope = Int64(0)
    var max_ulp_cos = Int64(0)
    var n_meas = 0
    var worst_sin = Float32(0.0)
    for i in range(N):
        var x = px.unsafe_load(i)
        var ys = portable_sinf(x)
        var yc = portable_cosf(x)
        ps.unsafe_store(i, ys)
        pc.unsafe_store(i, yc)
        pss.unsafe_store(i, _sab_sinf(x))
        pcs.unsafe_store(i, _sab_cosf(x))
        if _normal(_bits(x)):
            n_meas += 1
            var ds = _ulp_dist(ys, Float32(_sin64(Float64(x))))
            var dc = _ulp_dist(yc, Float32(_cos64(Float64(x))))
            if ds > max_ulp_sin:
                max_ulp_sin = ds
                worst_sin = x
            if abs(x) < Float32(6.2831855) and ds > max_ulp_sin_rope:
                max_ulp_sin_rope = ds
            if dc > max_ulp_cos:
                max_ulp_cos = dc
    print("portable_sinf vs libm double sin (", n_meas, "normal lanes ):")
    print("  max", max_ulp_sin, "ulp across |x| < 8192 (worst x =", worst_sin, ")")
    print("  max", max_ulp_sin_rope, "ulp on |x| < 2*pi")
    print("portable_cosf vs libm double cos: max", max_ulp_cos, "ulp")
    print("  RECORDED, NOT ASSERTED (DEVIATION 820 says the sin bound was")
    print("  never measured). Freeze these numbers into assertions on the")
    print("  SECOND run, once three columns have printed the same ones.")

    # ---- arm 2 + arm 3: the device ------------------------------------
    var d_x = ctx.enqueue_create_buffer[DType.float32](N)
    var d_one = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_buf=d_x, src_ptr=h_x.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_one, src_ptr=h_one.unsafe_ptr())
    var o_sin = ctx.enqueue_create_buffer[DType.float32](N)
    var o_cos = ctx.enqueue_create_buffer[DType.float32](N)
    var o_sin_sab = ctx.enqueue_create_buffer[DType.float32](N)
    var o_cos_sab = ctx.enqueue_create_buffer[DType.float32](N)
    var o_vendor = ctx.enqueue_create_buffer[DType.float32](N)
    var o_probe = ctx.enqueue_create_buffer[DType.float32](N)

    ctx.enqueue_function[trig_kernel](
        d_x.unsafe_ptr(), d_one.unsafe_ptr(),
        o_sin.unsafe_ptr(), o_cos.unsafe_ptr(),
        o_sin_sab.unsafe_ptr(), o_cos_sab.unsafe_ptr(),
        o_vendor.unsafe_ptr(), o_probe.unsafe_ptr(),
        Int32(N),
        grid_dim=(N + BLOCK - 1) // BLOCK,
        block_dim=(BLOCK, 1, 1),
    )

    var r_sin = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_cos = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_sin_sab = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_cos_sab = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_vendor = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_probe = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_ptr=r_sin.unsafe_ptr(), src_buf=o_sin)
    ctx.enqueue_copy(dst_ptr=r_cos.unsafe_ptr(), src_buf=o_cos)
    ctx.enqueue_copy(dst_ptr=r_sin_sab.unsafe_ptr(), src_buf=o_sin_sab)
    ctx.enqueue_copy(dst_ptr=r_cos_sab.unsafe_ptr(), src_buf=o_cos_sab)
    ctx.enqueue_copy(dst_ptr=r_vendor.unsafe_ptr(), src_buf=o_vendor)
    ctx.enqueue_copy(dst_ptr=r_probe.unsafe_ptr(), src_buf=o_probe)
    ctx.synchronize()

    var qs = r_sin.unsafe_ptr()
    var qc = r_cos.unsafe_ptr()
    var qss = r_sin_sab.unsafe_ptr()
    var qcs = r_cos_sab.unsafe_ptr()
    var qv = r_vendor.unsafe_ptr()
    var qp = r_probe.unsafe_ptr()

    # does THIS column flush subnormals? Lane 9 is 0x00400000 and lane 10
    # is its negative; the probe multiplied each by a runtime 1.0.
    var device_flushes = _bits(qp.unsafe_load(9)) == UInt32(0)
    var probe_neg = _bits(qp.unsafe_load(10))
    print("flush probe: subnormal * 1.0 came back", hex(_bits(qp.unsafe_load(9))),
          "and", hex(probe_neg))
    if device_flushes:
        print("  this column FLUSHES subnormals")
        if probe_neg != UInt32(0x80000000) and probe_neg != UInt32(0):
            print("  NOTE: the negative probe did not flush to a zero;")
            print("  read the two arms below with that in mind.")
    else:
        print("  this column HONORS subnormals")

    var mm_sin = 0
    var mm_cos = 0
    var h_all = UInt64(0xCBF29CE484222325)
    var h_s = UInt64(0xCBF29CE484222325)
    var h_c = UInt64(0xCBF29CE484222325)
    var shown = 0
    var max_ulp_vendor = Int64(0)
    var vendor_exact = 0
    var vendor_n = 0
    for i in range(N):
        var db = _bits(qs.unsafe_load(i))
        var hb = _bits(ps.unsafe_load(i))
        if not _same(db, hb):
            mm_sin += 1
            if shown < 3:
                print("  sin mismatch lane", i, "device", hex(db), "host", hex(hb))
                shown += 1
        var dbc = _bits(qc.unsafe_load(i))
        var hbc = _bits(pc.unsafe_load(i))
        if not _same(dbc, hbc):
            mm_cos += 1
        var cs = UInt64(_canon(db))
        var cc = UInt64(_canon(dbc))
        h_s = (h_s ^ cs) * UInt64(0x100000001B3)
        h_c = (h_c ^ cc) * UInt64(0x100000001B3)
        h_all = (h_all ^ cs) * UInt64(0x100000001B3)
        h_all = (h_all ^ cc) * UInt64(0x100000001B3)
        # the vendor witness, RECORDED
        var x = px.unsafe_load(i)
        if _normal(_bits(x)) and abs(x) < Float32(1000.0):
            vendor_n += 1
            var d = _ulp_dist(qv.unsafe_load(i), ps.unsafe_load(i))
            if d == Int64(0):
                vendor_exact += 1
            if d > max_ulp_vendor:
                max_ulp_vendor = d

    print("trig device hash =", h_all)
    print("  per function: sin", h_s, " cos", h_c)
    print("  device/host mismatches: sin", mm_sin, " cos", mm_cos)
    print("VENDOR WITNESS, RECORDED (this is `identical_sin`'s FAST arm,")
    print("  `std.math.sin`, launched on the device whatever the build mode --")
    print("  a FAST arm nobody has run on a device is a claim, not a path):")
    print("  vs portable_sinf on", vendor_n, "normal lanes with |x| < 1000:")
    print("  max", max_ulp_vendor, "ulp, exact on", vendor_exact)

    if mm_sin != 0 or mm_cos != 0:
        raise Error(
            "device bits differ from host bits on " + String(mm_sin + mm_cos)
            + " lanes: this backend broke the basic-ops premise"
        )
    print("device == host on every lane")

    # EVERY pointer handed on carries an explicit, IDENTICAL origin cast.
    # Without it the argument origins do not unify at the call site and
    # the parameter's origin cannot be inferred; this is the spelling
    # `ensemble/mojo_only/sampled_cols_check.mojo` already uses.
    check_sabotage_arms(
        ps.unsafe_origin_cast[MutUntrackedOrigin](),
        pc.unsafe_origin_cast[MutUntrackedOrigin](),
        pss.unsafe_origin_cast[MutUntrackedOrigin](),
        pcs.unsafe_origin_cast[MutUntrackedOrigin](),
        qs.unsafe_origin_cast[MutUntrackedOrigin](),
        qc.unsafe_origin_cast[MutUntrackedOrigin](),
        qss.unsafe_origin_cast[MutUntrackedOrigin](),
        qcs.unsafe_origin_cast[MutUntrackedOrigin](),
        px.unsafe_origin_cast[MutUntrackedOrigin](),
        device_flushes,
    )

    comptime if ANY_SABOTAGE:
        print("SABOTAGED BUILD (" + trig_sabotage_name()
              + "): the arm behaved as predicted above.")
    else:
        print("all arms OK")

    _ = h_x.unsafe_ptr()
    _ = h_one.unsafe_ptr()
    _ = h_sin.unsafe_ptr()
    _ = h_cos.unsafe_ptr()
    _ = h_sin_sab.unsafe_ptr()
    _ = h_cos_sab.unsafe_ptr()
    _ = r_sin.unsafe_ptr()
    _ = r_cos.unsafe_ptr()
    _ = r_sin_sab.unsafe_ptr()
    _ = r_cos_sab.unsafe_ptr()
    _ = r_vendor.unsafe_ptr()
    _ = r_probe.unsafe_ptr()
