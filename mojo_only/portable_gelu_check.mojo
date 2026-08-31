# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTITY_PATHS: the transformer activation family. DEVIATIONS 821-824 --
`portable_tanhf`, `portable_erff`, and the TWO gelus that are not the same
function. The fifth gate of portable device arithmetic, shaped like
`mojo_only/portable_nn_check.mojo` and not reinvented.

THE POINT OF THIS FILE IS A NEGATIVE CONTROL. `check_two_gelus_disagree`
runs first and asserts that `portable_gelu_erf` and `portable_gelu_tanh`
ARE DIFFERENT FUNCTIONS. HuggingFace ships both, a checkpoint's config
picks one, and its own docstring says the tanh form "is not an exact
numerical match" (activations.py:36-37). A copy-paste that routed both
seams to one implementation would pass every other check in this file, so
this one is load-bearing and it is not a formality.

THREE ARMS, over 2^20 hashed and planted inputs per function:

1. ACCURACY, HOST -- each function against a float64 reference reached
   through FFI (libm's double `tanh` and `erf`), ulp distance on the finite
   lanes, every planted special value asserted BY BITS with signed zeros
   compared BY SIGN BIT. NO ULP BOUND IS ASSERTED. Cephes's published
   numbers are for Cephes's spelling, not for this fma spelling of it, and
   the difference has never been measured. This run RECORDS the numbers;
   a later run, once three columns have printed the same ones, turns them
   into assertions.
2. DEVICE == HOST -- one kernel evaluates all four on the same inputs;
   `gelu device hash` is the cross-vendor certificate line, per-function
   hashes beside it. NaN canonicalized to 0x7FC00000 before hashing.
3. REACH -- the sabotage arms below, `-D` selected.

DEVIATION 942: THE FAST ARMS ARE LAUNCHED ON THE DEVICE, UNCONDITIONALLY.
`identical_tanh` and `identical_erf` route to `std.math.tanh` and
`std.math.erf` under FAST, and DEVIATION 821 records that NEITHER HAS EVER
BEEN CHECKED for the float64-lowering defect that made Metal refuse
`std.math.log1p` (DEVIATION 746 (ii)). Under an IDENTICAL build those arms
are `comptime if`-ed away and never compiled, so the question stays open
forever. This kernel therefore calls both DIRECTLY, in every build mode, so
that the compiler has to answer it.

    IF METAL REFUSES TO COMPILE THIS FILE, READ THE ERROR BEFORE ASSUMING
    A DEFECT. A refusal naming a float64 operation IS THE ANSWER TO
    DEVIATION 821's open question, not a bug in this gate. The fallback is
    the one DEVIATION 821 already wrote down: build with
    `-D MOJOLEARN_GELU_NO_VENDOR_WITNESS=1` to drop the two vendor calls
    and get the rest of the gate, then hand the refusal to the transformer
    lane, whose one-line fix is Cephes's large-x identity
    `1 - 2/(exp(x+x) + 1)` through `std.math.exp`.

The vendor answers are RECORDED, never asserted. They are the vendor's own
spelling and they are allowed to differ.

SABOTAGE ARMS. DEVIATION 940 applies: the arms are LOCAL re-spellings in
this file rather than edits to `numerics.mojo`, which this lane does not
own. With no define armed each local spelling must be bit-equal to the real
function on every lane, which is an always-on assertion that the copy has
not drifted; with an arm armed it must differ in the predicted place.

  -D MOJOLEARN_GELU_SABOTAGE_ONE_GELU=1
        Route the tanh gelu to the erf gelu. MUST fail
        `check_two_gelus_disagree`, and nothing else needs to catch it.
  -D MOJOLEARN_GELU_SABOTAGE_FUSED_INNER=1
        Spell the tanh gelu's `x + 0.044715*x^3` as one
        `identical_mul_add`. This is the arm that proves the UNFUSED
        decision (DEVIATION 744's rule: the reference's spelling wins for
        operation order) is REACHED. A last-bit change, invisible to any
        tolerance, so it is judged by the bitwise compare alone.
  -D MOJOLEARN_GELU_SABOTAGE_DOUBLE_CONST=1
        EXPECTED INERT, AND THAT IS WHY IT IS HERE. Divide by
        `Float32(1.4142135623730951)` written as a literal instead of by
        `GELU_SQRT2_BITS`. The literal is correctly rounded to the same
        float32, so the arm should move NOTHING. DEVIATION 946 records the
        inertness rather than deleting the arm: an inert arm that is
        written down is a measured hole, an inert arm that is deleted is a
        hole nobody knows about. The run PRINTS the verdict; a non-zero
        reach here would mean Mojo's literal does not round-trip, which is
        a finding about `mojo-string-float-roundtrip` and not about gelu.
  -D MOJOLEARN_GELU_SABOTAGE_CEPHES_ZERO=1
        Drop `portable_tanhf`'s `x == 0` guard, restoring the Cephes sign
        loss (at xx = -0.0 the odd polynomial cancels the sign TWICE and
        returns +0.0). MUST fail `check_tanh_planted` ON THE SIGN BIT and
        MUST move ONLY the negative-zero lanes. If it moves an ulp count
        too, that ulp count is comparing zeros wrongly.

Run:
    pixi run check-portable-gelu
    pixi run mojo run -D MOJOLEARN_GELU_SABOTAGE_ONE_GELU=1 \\
        -I . mojo_only/portable_gelu_check.mojo
and the same for the other three names. A misspelled define is silently
ignored, so the run PRINTS the arm it was built with (`gelu sabotage
<name>`, `tools/gemm_ladder.sh:71`'s scar); read that line first.
"""

from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import erf, tanh
from std.memory import bitcast
from std.sys import argv
from std.sys.compile import is_defined
from std.ffi import external_call

from mojo_only.numerics import (
    GELU_SQRT2_BITS,
    GELU_TANH_COEF_BITS,
    GELU_TANH_SCALE_BITS,
    GLOBAL_NUMERIC_MODE,
    NEG_MAXLOGF_BITS,
    NUMERIC_IDENTICAL,
    TANH_SAT_BITS,
    _cephes_erfcf_ge1,
    _fma_f32,
    _ftz_always,
    identical_mul_add,
    portable_divf,
    portable_erff,
    portable_expf,
    portable_gelu_erf,
    portable_gelu_tanh,
    portable_tanhf,
    numeric_mode_name,
)

comptime N = 1 << 20
comptime BLOCK = 256

comptime SAB_ONE_GELU = is_defined["MOJOLEARN_GELU_SABOTAGE_ONE_GELU"]()
"""Route `portable_gelu_tanh` to `portable_gelu_erf`. The two gelus part
company by about 1e-3 around |x| = 2, four orders of magnitude above
float32 epsilon: too large for a tolerance to absorb, too small for a wrong
answer to look wrong."""

comptime SAB_FUSED_INNER = is_defined["MOJOLEARN_GELU_SABOTAGE_FUSED_INNER"]()
"""`identical_mul_add(c3, x3, x)` for `x + 0.044715*x^3`. torch runs the
multiply and the add as two elementwise kernels and rounds twice; fusing
would be more ACCURATE and would be a DIFFERENT FUNCTION."""

comptime SAB_DOUBLE_CONST = is_defined["MOJOLEARN_GELU_SABOTAGE_DOUBLE_CONST"]()
"""EXPECTED INERT. See DEVIATION 946 in the module docstring."""

comptime SAB_CEPHES_ZERO = is_defined["MOJOLEARN_GELU_SABOTAGE_CEPHES_ZERO"]()
"""Drop tanh's `x == 0` guard. The Cephes bug, un-fixed."""

comptime NO_VENDOR_WITNESS = is_defined["MOJOLEARN_GELU_NO_VENDOR_WITNESS"]()
"""Drop the `std.math.tanh` / `std.math.erf` calls from the kernel. Only
for a column that REFUSES to compile them -- and that refusal is itself the
answer DEVIATION 821 asked for, so record it before setting this."""

comptime ANY_SABOTAGE = (
    SAB_ONE_GELU or SAB_FUSED_INNER or SAB_DOUBLE_CONST or SAB_CEPHES_ZERO
)


def gelu_sabotage_name() -> String:
    comptime if SAB_ONE_GELU:
        return String("ONE_GELU")
    comptime if SAB_FUSED_INNER:
        return String("FUSED_INNER")
    comptime if SAB_DOUBLE_CONST:
        return String("DOUBLE_CONST")
    comptime if SAB_CEPHES_ZERO:
        return String("CEPHES_ZERO")
    return String("none")


# ---- the local re-spellings (DEVIATION 940) -----------------------------
def _sab_tanhf(x_in: Float32) -> Float32:
    """A faithful copy of `portable_tanhf` carrying the CEPHES_ZERO arm."""
    var xx = _ftz_always(x_in)
    if xx != xx:
        return xx
    comptime if not SAB_CEPHES_ZERO:
        if xx == Float32(0.0):
            return xx
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


def _sab_gelu_erf(x_in: Float32) -> Float32:
    """A faithful copy of `portable_gelu_erf` carrying the DOUBLE_CONST
    arm."""
    var x = _ftz_always(x_in)
    if x != x:
        return x
    var s2 = bitcast[DType.float32](GELU_SQRT2_BITS)
    comptime if SAB_DOUBLE_CONST:
        s2 = Float32(1.4142135623730951)
    var e = portable_erff(portable_divf(x, s2))
    var s = Float32(1.0) + e
    var h = _ftz_always(x * Float32(0.5))
    return _ftz_always(h * s)


def _sab_gelu_tanh(x_in: Float32) -> Float32:
    """A faithful copy of `portable_gelu_tanh` carrying the ONE_GELU and
    FUSED_INNER arms. It calls `portable_tanhf`, NOT `_sab_tanhf`, on
    purpose: CEPHES_ZERO is tanh's arm and must be judged on tanh alone,
    which is what "MUST move nothing else" means."""
    comptime if SAB_ONE_GELU:
        return portable_gelu_erf(x_in)
    var x = _ftz_always(x_in)
    if x != x:
        return x
    var c3 = bitcast[DType.float32](GELU_TANH_COEF_BITS)
    var kk = bitcast[DType.float32](GELU_TANH_SCALE_BITS)
    var x2 = _ftz_always(x * x)
    var x3 = _ftz_always(x2 * x)
    var inner: Float32
    comptime if SAB_FUSED_INNER:
        inner = _ftz_always(identical_mul_add(c3, x3, x))
    else:
        var cx = _ftz_always(c3 * x3)
        inner = _ftz_always(x + cx)
    var arg = _ftz_always(kk * inner)
    var t = portable_tanhf(arg)
    var s = Float32(1.0) + t
    var h = _ftz_always(x * Float32(0.5))
    return _ftz_always(h * s)


def _gelu_tanh_assoc_right(x_in: Float32) -> Float32:
    """DEVIATION 945's probe, not a sabotage arm: the tanh gelu with the
    cube spelled `x*(x*x)` instead of `(x*x)*x`. See
    `check_torch_pow_association_is_owed`."""
    var x = _ftz_always(x_in)
    if x != x:
        return x
    var c3 = bitcast[DType.float32](GELU_TANH_COEF_BITS)
    var kk = bitcast[DType.float32](GELU_TANH_SCALE_BITS)
    var x2 = _ftz_always(x * x)
    var x3 = _ftz_always(x * x2)
    var cx = _ftz_always(c3 * x3)
    var inner = _ftz_always(x + cx)
    var arg = _ftz_always(kk * inner)
    var t = portable_tanhf(arg)
    var s = Float32(1.0) + t
    var h = _ftz_always(x * Float32(0.5))
    return _ftz_always(h * s)


# ---- the two tanh arms, spelled separately so the branch can be pinned --
def _tanh_exp_arm(xx: Float32) -> Float32:
    var x = abs(xx)
    var e = portable_expf(x + x)
    var d = e + Float32(1.0)
    var q = portable_divf(Float32(2.0), d)
    var z = Float32(1.0) - q
    if xx < Float32(0.0):
        return -z
    return z


def _tanh_poly_arm(xx: Float32) -> Float32:
    var x = abs(xx)
    var zs = x * x
    var p = Float32(-5.70498872745e-3)
    p = _fma_f32(p, zs, Float32(2.06390887954e-2))
    p = _fma_f32(p, zs, Float32(-5.37397155531e-2))
    p = _fma_f32(p, zs, Float32(1.33314422036e-1))
    p = _fma_f32(p, zs, Float32(-3.33332819422e-1))
    var pz = p * zs
    return _fma_f32(pz, xx, xx)


# ---- the two erfc tables, likewise -------------------------------------
def _erfc_table(a: Float32, use_p: Bool) -> Float32:
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
    if use_p:
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


def gelu_kernel(
    xs: MutPointer[Float32, MutAnyOrigin],
    ones: MutPointer[Float32, MutAnyOrigin],
    out_tanh: MutPointer[Float32, MutAnyOrigin],
    out_erf: MutPointer[Float32, MutAnyOrigin],
    out_ge: MutPointer[Float32, MutAnyOrigin],
    out_gt: MutPointer[Float32, MutAnyOrigin],
    out_tanh_sab: MutPointer[Float32, MutAnyOrigin],
    out_ge_sab: MutPointer[Float32, MutAnyOrigin],
    out_gt_sab: MutPointer[Float32, MutAnyOrigin],
    out_vtanh: MutPointer[Float32, MutAnyOrigin],
    out_verf: MutPointer[Float32, MutAnyOrigin],
    out_probe: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var x = xs.unsafe_load(i)
        out_tanh.unsafe_store(i, portable_tanhf(x))
        out_erf.unsafe_store(i, portable_erff(x))
        out_ge.unsafe_store(i, portable_gelu_erf(x))
        out_gt.unsafe_store(i, portable_gelu_tanh(x))
        out_tanh_sab.unsafe_store(i, _sab_tanhf(x))
        out_ge_sab.unsafe_store(i, _sab_gelu_erf(x))
        out_gt_sab.unsafe_store(i, _sab_gelu_tanh(x))
        # DEVIATION 942: the FAST arms, on the device, in every build mode.
        comptime if NO_VENDOR_WITNESS:
            out_vtanh.unsafe_store(i, Float32(0.0))
            out_verf.unsafe_store(i, Float32(0.0))
        else:
            out_vtanh.unsafe_store(i, tanh(x))
            out_verf.unsafe_store(i, erf(x))
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


def _mag_band(x: Float32) -> Int:
    """Six magnitude bands of |x|, for the vendor witness's band table.
    A defect that lives in one band is invisible in an aggregate maximum
    and obvious in this one."""
    var a = abs(x)
    if a < Float32(1e-30):
        return 0
    if a < Float32(1e-20):
        return 1
    if a < Float32(1e-10):
        return 2
    if a < Float32(1e-3):
        return 3
    if a < Float32(1.0):
        return 4
    return 5


def _tanh64(x: Float64) -> Float64:
    return external_call["tanh", Float64](x)


def _erf64(x: Float64) -> Float64:
    return external_call["erf", Float64](x)


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
    if not _same(_bits(got), want_bits):
        raise Error(
            name + ": planted value wrong, got " + hex(_bits(got))
            + " want " + hex(want_bits)
        )


def _expect_raw(name: String, got: Float32, want_bits: UInt32) raises:
    """BY SIGN BIT: `-0.0 == 0.0` is TRUE, so a float comparison passes on
    either and every signed-zero contract here goes through this."""
    if _bits(got) != want_bits:
        raise Error(
            name + ": wrong BITS, got " + hex(_bits(got))
            + " want " + hex(want_bits)
        )


# ---- the fixture --------------------------------------------------------
# Unlike the trig gate, RAW 32-bit patterns are safe here: no function in
# this family converts a float to an Int, so there is no out-of-range
# conversion to walk into. Raw patterns give NaN, both infinities, both
# zeros, subnormals and huge magnitudes at their natural density.
comptime NPLANT = 40


def _gelu_x(i: Int) -> Float32:
    if i == 0:
        return _f(UInt32(0x00000000))  # +0
    if i == 1:
        return _f(UInt32(0x80000000))  # -0
    if i == 2:
        return _f(UInt32(0x7FC00000))
    if i == 3:
        return _f(UInt32(0xFFC00000))
    if i == 4:
        return _f(UInt32(0x7F800000))  # +inf
    if i == 5:
        return _f(UInt32(0xFF800000))  # -inf
    if i == 6:
        return _f(UInt32(0x00000001))
    if i == 7:
        return _f(UInt32(0x80000001))
    if i == 8:
        return _f(UInt32(0x00400000))
    if i == 9:
        return _f(UInt32(0x80400000))
    if i == 10:
        return _f(UInt32(0x007FFFFF))
    if i == 11:
        return _f(UInt32(0x807FFFFF))
    if i == 12:
        return _f(UInt32(0x00800000))  # FLT_MIN: x*0.5 is SUBNORMAL
    if i == 13:
        return _f(UInt32(0x80800000))
    if i == 14:
        return Float32(0.625)  # tanh's branch point, exp arm
    if i == 15:
        return _f(_bits(Float32(0.625)) - UInt32(1))  # polynomial arm
    if i == 16:
        return Float32(-0.625)
    if i == 17:
        return -_f(_bits(Float32(0.625)) - UInt32(1))
    if i == 18:
        return bitcast[DType.float32](TANH_SAT_BITS)
    if i == 19:
        return _f(TANH_SAT_BITS + UInt32(1))
    if i == 20:
        return -bitcast[DType.float32](TANH_SAT_BITS)
    if i == 21:
        return Float32(1.0)  # erf's branch point, polynomial arm
    if i == 22:
        return Float32(-1.0)
    if i == 23:
        return _f(_bits(Float32(1.0)) + UInt32(1))  # erfc arm
    if i == 24:
        return Float32(2.0)  # the erfc P/R table boundary, R side
    if i == 25:
        return _f(_bits(Float32(2.0)) - UInt32(1))  # P side
    if i == 26:
        return Float32(-2.0)
    if i == 27:
        return Float32(9.4193)
    if i == 28:
        return Float32(-9.4193)
    if i == 29:
        return Float32(10.0)
    if i == 30:
        return Float32(-10.0)
    if i == 31:
        return Float32(0.5)
    if i == 32:
        return Float32(-0.5)
    if i == 33:
        return Float32(1e-20)
    if i == 34:
        return Float32(-1e-20)
    if i == 35:
        return Float32(1e30)
    if i == 36:
        return Float32(-1e30)
    if i == 37:
        return _f(UInt32(0x7F7FFFFF))  # FLT_MAX
    if i == 38:
        return _f(UInt32(0xFF7FFFFF))
    if i == 39:
        return Float32(0.044715)
    var h = _splitmix(UInt64(11 * i + 5))
    var sel = (h >> 40) & UInt64(3)
    if sel == UInt64(0):
        return _f(UInt32(h & UInt64(0xFFFFFFFF)))
    var u = Float64(h & UInt64(0xFFFFFFFF)) / Float64(4294967296.0)
    if sel == UInt64(1):
        # the region where the two gelus part company
        return Float32(-4.0 + 8.0 * u)
    return Float32(-16.0 + 32.0 * u)


# ==========================================================================
# THE CHECKS
# ==========================================================================
def check_two_gelus_disagree() raises:
    """THE POINT OF THIS FILE, and a NEGATIVE control. The two gelus must
    be measurably different functions. HuggingFace registers both -- "gelu"
    and "gelu_python" reach the erf form, "gelu_new", "gelu_pytorch_tanh"
    and "gelu_python_tanh" reach the tanh form -- and picking the other one
    silently is exactly the class of defect a per-stage card exists to
    catch.

    The comparison is against `_sab_gelu_tanh`, not against
    `portable_gelu_tanh`, so that the ONE_GELU arm reaches it. With no arm
    armed the two are bit-equal (asserted separately in
    `check_sabotage_arms`).

    The 1e-4 threshold is the reference's own claim (they part company by
    about 1e-3 around |x| = 2), not a number this lane invented."""
    var max_abs = Float64(0.0)
    var worst = Float32(0.0)
    var n_diff = 0
    var n_cmp = 0
    for i in range(N):
        var x = _gelu_x(i)
        if not _normal(_bits(x)) or abs(x) > Float32(30.0):
            continue
        var a = portable_gelu_erf(x)
        var b = _sab_gelu_tanh(x)
        if not _finite(_bits(a)) or not _finite(_bits(b)):
            continue
        n_cmp += 1
        if _bits(a) != _bits(b):
            n_diff += 1
        var d = abs(Float64(a) - Float64(b))
        if d > max_abs:
            max_abs = d
            worst = x
    print("check_two_gelus_disagree:", n_diff, "of", n_cmp,
          "lanes differ BY BITS; max |erf gelu - tanh gelu| =", max_abs,
          "at x =", worst)
    comptime if SAB_ONE_GELU:
        # SELF-JUDGING, like every other arm in these three gates: a
        # sabotaged build is CORRECT when the negative control fires, so
        # the run reports rather than exiting non-zero, and it exits
        # non-zero only when the arm found nothing.
        if max_abs > 1e-4:
            raise Error(
                "ONE_GELU is armed and the two gelus STILL differ by "
                + String(max_abs)
                + ". The define did not reach the build, or"
                + " `_sab_gelu_tanh` is not the spelling under test"
            )
        print("  ONE_GELU: THE NEGATIVE CONTROL FIRED. The two seams routed")
        print("  to one implementation are measured as one function, which")
        print("  is exactly what this check exists to refuse.")
        return
    if max_abs <= 1e-4:
        raise Error(
            "the two gelus agree to within 1e-4 over the whole sweep. They"
            + " are DIFFERENT FUNCTIONS (HuggingFace activations.py:36-37"
            + " says so) and this gate has just measured them as one:"
            + " a seam has been copy-pasted"
        )
    if n_diff == 0:
        raise Error(
            "the two gelus are BIT-EQUAL on every lane while differing"
            + " numerically, which is impossible -- the compare is blind"
        )


def check_tanh_branch_boundary() raises:
    """|x| = 0.625 exactly takes the EXP arm and the float below it takes
    the POLYNOMIAL arm, both signs, both sides. Each arm is spelled
    separately above so the branch can be pinned by BITS rather than by
    "the answer looks right".

    AND A HOLE, RECORDED. `portable_tanhf`'s saturation guard at
    |x| = 44.361419677734375 is BIT-INERT ON EVERY INPUT, not only at the
    boundary. The exp arm already returns exactly 1.0 for every x above
    about 9.01 (`2/(exp(2x)+1)` falls below half an ulp of 1 there), and at
    +inf `portable_expf` saturates so the arm still returns 1.0. So the
    guard cannot be falsified by any output. It is REACHED BUT INERT
    (`reached-but-inert`), it is Cephes's own guard and there is nothing to
    fix, but nothing in this gate or any other can tell whether it is
    present. That is stated here rather than hidden behind a passing
    assertion."""
    var b625 = Float32(0.625)
    var below = _f(_bits(b625) - UInt32(1))
    for s in range(2):
        var sgn = Float32(1.0)
        var nm = String("+")
        if s == 1:
            sgn = Float32(-1.0)
            nm = String("-")
        var at = sgn * b625
        var lo = sgn * below
        if _bits(portable_tanhf(at)) != _bits(_tanh_exp_arm(at)):
            raise Error("tanh(" + nm + "0.625) did not take the EXP arm")
        if _bits(portable_tanhf(lo)) != _bits(_tanh_poly_arm(lo)):
            raise Error(
                "tanh(the float below " + nm + "0.625) did not take the"
                + " POLYNOMIAL arm"
            )
        if _bits(_tanh_exp_arm(at)) == _bits(_tanh_poly_arm(at)):
            print("  NOTE: at " + nm + "0.625 the two arms agree bit for bit,")
            print("  so the boundary is unobservable from the output there.")
    print("check_tanh_branch_boundary: 0.625 takes exp, the float below")
    print("  takes the polynomial, both signs, pinned by BITS")
    # the saturation guard, measured rather than assumed
    var sat = bitcast[DType.float32](TANH_SAT_BITS)
    var above = _f(TANH_SAT_BITS + UInt32(1))
    var big = Float32(1e30)
    var inf = _f(UInt32(0x7F800000))
    _expect_raw("tanh(0.5*MAXLOGF)", portable_tanhf(sat), UInt32(0x3F800000))
    _expect_raw("tanh(next above)", portable_tanhf(above), UInt32(0x3F800000))
    var inert = (
        _bits(_tanh_exp_arm(above)) == UInt32(0x3F800000)
        and _bits(_tanh_exp_arm(big)) == UInt32(0x3F800000)
        and _bits(_tanh_exp_arm(inf)) == UInt32(0x3F800000)
    )
    if inert:
        print("  RECORDED HOLE: the saturation guard is BIT-INERT on every")
        print("  input (the exp arm returns exactly 1.0 above x ~ 9.01 and")
        print("  at +inf too). Nothing can falsify its presence. It is")
        print("  Cephes's own guard; there is nothing to fix, only to know.")
    else:
        print("  the saturation guard IS observable -- the exp arm does not")
        print("  return 1.0 above the threshold on this column. That is new;")
        print("  it means a fixture straddling 44.3614 has teeth after all.")


def check_tanh_planted() raises:
    """`tanh(-0.0)` is `-0.0`. That is the assertion that the Cephes bug
    stayed fixed, and an equality on floats cannot see it: Cephes returns
    `+0.0` there because the odd polynomial cancels the argument's sign
    TWICE (`P*z` is `-0.33332819 * +0.0` = `-0.0`, the tail's `* xx` is
    `(-0.0) * (-0.0)` = `+0.0`, and `(+0.0) + (-0.0)` is `+0.0`), and
    `-0.0 == 0.0` is TRUE. Every zero here is compared BY SIGN BIT."""
    comptime PZ = UInt32(0x00000000)
    comptime NZ = UInt32(0x80000000)
    comptime ONE = UInt32(0x3F800000)
    comptime NONE = UInt32(0xBF800000)
    comptime QNAN = UInt32(0x7FC00000)
    _expect_raw("tanh(+0)", portable_tanhf(_f(PZ)), PZ)
    _expect_raw("tanh(-0)", portable_tanhf(_f(NZ)), NZ)
    _expect("tanh(NaN)", portable_tanhf(_f(QNAN)), QNAN)
    _expect("tanh(NaN, negative payload)", portable_tanhf(_f(UInt32(0xFFC00000))), QNAN)
    _expect_raw("tanh(+inf)", portable_tanhf(_f(UInt32(0x7F800000))), ONE)
    _expect_raw("tanh(-inf)", portable_tanhf(_f(UInt32(0xFF800000))), NONE)
    _expect_raw("tanh(subnormal)", portable_tanhf(_f(UInt32(0x00400000))), PZ)
    _expect_raw("tanh(-subnormal)", portable_tanhf(_f(UInt32(0x80400000))), NZ)
    _expect_raw("tanh(smallest subnormal)", portable_tanhf(_f(UInt32(0x00000001))), PZ)
    _expect_raw("tanh(-smallest subnormal)", portable_tanhf(_f(UInt32(0x80000001))), NZ)
    _expect_raw("tanh(largest subnormal)", portable_tanhf(_f(UInt32(0x007FFFFF))), PZ)
    _expect_raw("tanh(-largest subnormal)", portable_tanhf(_f(UInt32(0x807FFFFF))), NZ)
    # the un-guarded spelling's answer, printed whether or not the arm is
    # armed, so the bug being fixed is VISIBLE and not merely absent
    print("check_tanh_planted: tanh(-0) =", hex(_bits(portable_tanhf(_f(NZ)))),
          "(IEEE, torch and numpy all say 0x80000000)")
    print("  the local spelling under this build gives",
          hex(_bits(_sab_tanhf(_f(NZ)))),
          "(Cephes un-guarded would be 0x00000000)")


def check_erf_planted() raises:
    """erf's special values and BOTH table boundaries. `|x| > 1.0` is the
    branch, so x = 1.0 exactly takes the polynomial arm (Cephes's own
    boundary), and inside `_cephes_erfcf_ge1` the P table serves
    1 <= x < 2 and the R table x >= 2, so x = 2.0 exactly takes R."""
    comptime PZ = UInt32(0x00000000)
    comptime NZ = UInt32(0x80000000)
    comptime ONE = UInt32(0x3F800000)
    comptime NONE = UInt32(0xBF800000)
    comptime QNAN = UInt32(0x7FC00000)
    _expect_raw("erf(+0)", portable_erff(_f(PZ)), PZ)
    _expect_raw("erf(-0)", portable_erff(_f(NZ)), NZ)
    _expect("erf(NaN)", portable_erff(_f(QNAN)), QNAN)
    _expect("erf(NaN, negative payload)", portable_erff(_f(UInt32(0xFFC00000))), QNAN)
    _expect_raw("erf(+inf)", portable_erff(_f(UInt32(0x7F800000))), ONE)
    _expect_raw("erf(-inf)", portable_erff(_f(UInt32(0xFF800000))), NONE)
    _expect_raw("erf(subnormal)", portable_erff(_f(UInt32(0x00400000))), PZ)
    _expect_raw("erf(-subnormal)", portable_erff(_f(UInt32(0x80400000))), NZ)
    _expect_raw("erf(10)", portable_erff(Float32(10.0)), ONE)
    _expect_raw("erf(-10)", portable_erff(Float32(-10.0)), NONE)
    _expect_raw("erf(9.5)", portable_erff(Float32(9.5)), ONE)
    # x = 1.0 exactly is the POLYNOMIAL arm
    var one = Float32(1.0)
    var z = one * one
    var p = Float32(7.853861353153693e-5)
    p = _fma_f32(p, z, Float32(-8.010193625184903e-4))
    p = _fma_f32(p, z, Float32(5.188327685732524e-3))
    p = _fma_f32(p, z, Float32(-2.685381193529856e-2))
    p = _fma_f32(p, z, Float32(1.128358514861418e-1))
    p = _fma_f32(p, z, Float32(-3.761262582423300e-1))
    p = _fma_f32(p, z, Float32(1.128379165726710))
    if _bits(portable_erff(one)) != _bits(one * p):
        raise Error("erf(1.0) did not take the POLYNOMIAL arm")
    if _bits(portable_erff(-one)) != _bits(-one * p):
        raise Error("erf(-1.0) did not take the POLYNOMIAL arm")
    var just_above = _f(_bits(one) + UInt32(1))
    if _bits(portable_erff(just_above)) != _bits(
        Float32(1.0) - _cephes_erfcf_ge1(just_above)
    ):
        raise Error("the float above 1.0 did not take the erfc arm")
    # x = 2.0 exactly is the R table, the float below it the P table
    var two = Float32(2.0)
    var below2 = _f(_bits(two) - UInt32(1))
    if _bits(_cephes_erfcf_ge1(two)) != _bits(_erfc_table(two, False)):
        raise Error("erfc(2.0) did not take the R table")
    if _bits(_cephes_erfcf_ge1(below2)) != _bits(_erfc_table(below2, True)):
        raise Error("erfc(the float below 2.0) did not take the P table")
    if _bits(_erfc_table(two, True)) == _bits(_erfc_table(two, False)):
        print("  NOTE: at x = 2.0 the P and R tables agree bit for bit, so")
        print("  that boundary is unobservable from the output.")
    print("check_erf_planted: specials by BITS; the 1.0 and 2.0 boundaries")
    print("  pinned to the arm each is supposed to take")


def check_erf_odd() raises:
    """DEVIATION 944: ODDNESS IS ASSERTED ONLY WHERE IT HOLDS.

    `erf(-x) == -erf(x)` BY BITS is exact on |x| <= 1, where the answer is
    `x * T(x^2)` and the sign rides on one product. ABOVE |x| = 1 IT IS NOT
    TRUE and it is not a defect: erf(x) is `1 - erfc(x)` and erf(-x) is
    `1 - (2 - erfc(|x|))`, which are different roundings of the same real
    number. Asserting oddness everywhere would fail, and the failure would
    be the reference's shape rather than ours. So the asymmetry above 1 is
    MEASURED and printed instead."""
    var bad_low = 0
    var n_low = 0
    var n_high = 0
    var asym_high = 0
    var max_asym = Int64(0)
    var worst = Float32(0.0)
    for i in range(N):
        var x = _gelu_x(i)
        if not _finite(_bits(x)) or _bits(x) & UInt32(0x7FFFFFFF) == UInt32(0):
            continue
        var lhs = portable_erff(-x)
        var rhs = -portable_erff(x)
        if abs(x) <= Float32(1.0):
            n_low += 1
            if _bits(lhs) != _bits(rhs):
                if bad_low == 0:
                    worst = x
                bad_low += 1
        else:
            n_high += 1
            if _bits(lhs) != _bits(rhs):
                asym_high += 1
                var d = _ulp_dist(lhs, rhs)
                if d > max_asym:
                    max_asym = d
    if bad_low != 0:
        raise Error(
            "portable_erff is not odd on " + String(bad_low) + " of "
            + String(n_low) + " lanes with |x| <= 1, first x = "
            + String(worst) + ". On that arm oddness is exact by"
            + " construction, so this is a real defect"
        )
    print("check_erf_odd: exact on all", n_low, "lanes with |x| <= 1")
    print("  RECORDED above |x| = 1:", asym_high, "of", n_high,
          "lanes asymmetric, max", max_asym, "ulp. That is `1 - erfc(x)`")
    print("  versus `1 - (2 - erfc(|x|))`, the reference's shape, not ours.")


def check_gelu_planted() raises:
    """Both gelus, row 39, BY SIGN BIT -- and `-inf -> NaN` is ASSERTED
    rather than treated as a bug.

    `-inf * 0.5` is `-inf`, `erf(-inf)` is `-1.0`, `1.0 + (-1.0)` is
    `+0.0`, and `-inf * (+0.0)` is NaN. That is the reference's answer too.
    A reader "fixing" that edge to `-0.0` would break agreement with torch,
    so the gate pins it in the direction that keeps agreement."""
    comptime PZ = UInt32(0x00000000)
    comptime NZ = UInt32(0x80000000)
    comptime PINF = UInt32(0x7F800000)
    comptime NINF = UInt32(0xFF800000)
    comptime QNAN = UInt32(0x7FC00000)
    for g in range(2):
        var nm = String("gelu_erf")
        if g == 1:
            nm = String("gelu_tanh")
        for k in range(9):
            var xb = PZ
            var want = PZ
            var lab = String("")
            if k == 0:
                xb = PZ
                want = PZ
                lab = String("(+0)")
            elif k == 1:
                xb = NZ
                want = NZ
                lab = String("(-0)")
            elif k == 2:
                xb = PINF
                want = PINF
                lab = String("(+inf)")
            elif k == 3:
                xb = NINF
                want = QNAN
                lab = String("(-inf) -- NaN IS THE REFERENCE'S ANSWER")
            elif k == 4:
                xb = QNAN
                want = QNAN
                lab = String("(NaN)")
            elif k == 5:
                xb = UInt32(0x00400000)
                want = PZ
                lab = String("(subnormal)")
            elif k == 6:
                xb = UInt32(0x80400000)
                want = NZ
                lab = String("(-subnormal)")
            elif k == 7:
                xb = UInt32(0x00800000)
                want = PZ
                lab = String("(FLT_MIN, where x*0.5 is subnormal)")
            else:
                xb = UInt32(0x80800000)
                want = NZ
                lab = String("(-FLT_MIN)")
            var got = portable_gelu_erf(_f(xb))
            if g == 1:
                got = portable_gelu_tanh(_f(xb))
            if want == QNAN:
                _expect(nm + lab, got, want)
            else:
                _expect_raw(nm + lab, got, want)
    print("check_gelu_planted: both gelus, every special value BY BITS,")
    print("  -inf -> NaN ASSERTED (it is torch's answer, not a defect)")


def check_gelu_constants() raises:
    """`String(float)` does not round-trip in Mojo and all three model
    constants are Python doubles that torch's scalar promotion rounds to
    float32, so a decimal literal in `numerics.mojo` is a TRANSCRIPTION and
    not a pin. This reads the five pinned bit patterns back and checks each
    against the decimal in its comment. That is the
    `mojo-string-float-roundtrip` rule made checkable rather than
    trusted."""
    var s2 = bitcast[DType.float32](GELU_SQRT2_BITS)
    var kk = bitcast[DType.float32](GELU_TANH_SCALE_BITS)
    var c3 = bitcast[DType.float32](GELU_TANH_COEF_BITS)
    var sat = bitcast[DType.float32](TANH_SAT_BITS)
    var nml = bitcast[DType.float32](NEG_MAXLOGF_BITS)
    if _bits(s2) != UInt32(0x3FB504F3):
        raise Error("GELU_SQRT2_BITS is not 0x3FB504F3")
    if _bits(kk) != UInt32(0x3F4C422A):
        raise Error("GELU_TANH_SCALE_BITS is not 0x3F4C422A")
    if _bits(c3) != UInt32(0x3D372713):
        raise Error("GELU_TANH_COEF_BITS is not 0x3D372713")
    if _bits(sat) != UInt32(0x42317218):
        raise Error("TANH_SAT_BITS is not 0x42317218")
    if _bits(nml) != UInt32(0xC2B17218):
        raise Error("NEG_MAXLOGF_BITS is not 0xC2B17218")
    _expect_raw("sqrt(2) pin", Float32(1.4142135623730951), UInt32(0x3FB504F3))
    _expect_raw("sqrt(2/pi) pin", Float32(0.7978845608028654), UInt32(0x3F4C422A))
    _expect_raw("0.044715 pin", Float32(0.044715), UInt32(0x3D372713))
    _expect_raw("0.5*MAXLOGF pin", Float32(44.361419677734375), UInt32(0x42317218))
    _expect_raw("-MAXLOGF pin", Float32(-88.72283905206835), UInt32(0xC2B17218))
    # 0.5 * MAXLOGF is EXACT (a power of two times a float32), so the
    # threshold has no rounding question and this says so by computing it
    if _bits(Float32(0.5) * Float32(88.72283905206835)) != UInt32(0x42317218):
        raise Error(
            "0.5 * MAXLOGF is not TANH_SAT_BITS: the halving was supposed to"
            + " be exact"
        )
    print("check_gelu_constants: five pins read back BY BITS and each one")
    print("  agrees with the decimal in its comment")
    print("  sqrt2", s2, hex(_bits(s2)), " sqrt2pi", kk, hex(_bits(kk)))
    print("  coef", c3, hex(_bits(c3)), " tanh sat", sat, hex(_bits(sat)))


def check_torch_pow_association_is_owed() raises:
    """DEVIATION 945: THE TORCH CORPUS ARM IS OWED AND IS NOT FAKED HERE.

    DEVIATION 824 writes `torch.pow(input, 3.0)` as `(x*x)*x` on the
    understanding that ATen specializes small integral exponents into
    repeated multiplication. That was NOT read out of the PyTorch source --
    no checkout exists under `/Users/andrewhendel/CascadeProjects/upstream/`
    -- and it was NOT measured. This gate does not have a torch corpus and
    WILL NOT INVENT ONE. Guessing what PyTorch does and then asserting the
    guess would produce a gate that certifies the guess.

    What it CAN do, and does, is price the open question: it measures how
    much the association choice is worth in the final gelu. If the answer
    is "nothing on any lane" the question is moot; if it is "one ulp on
    some lanes" then the corpus arm is worth building and this number says
    how big a corpus tolerance would have to be to hide the answer.

    WHAT WOULD SETTLE IT, for whoever lands the corpus: dump
    `_gelu_python` and `_gelu_tanh_python` from HuggingFace transformers at
    d56c55b over a hashed float32 input spec, in the `mamba/corpus/` style,
    compare the INPUTS bitwise first, and calibrate the per-case tolerance
    from the corpus's own `--self-test` (the `adv_gate_saturation` lesson).
    Then this function's two numbers become the discriminator: if the
    corpus matches `(x*x)*x` and not `x*(x*x)` on the lanes counted here,
    DEVIATION 824's judgment call is settled in its favor."""
    var n_diff = 0
    var n_cmp = 0
    var max_ulp = Int64(0)
    var max_abs = Float64(0.0)
    var worst = Float32(0.0)
    for i in range(N):
        var x = _gelu_x(i)
        if not _normal(_bits(x)) or abs(x) > Float32(30.0):
            continue
        n_cmp += 1
        var a = portable_gelu_tanh(x)
        var b = _gelu_tanh_assoc_right(x)
        if _bits(a) != _bits(b):
            n_diff += 1
            var d = _ulp_dist(a, b)
            if d > max_ulp:
                max_ulp = d
                worst = x
            var da = abs(Float64(a) - Float64(b))
            if da > max_abs:
                max_abs = da
    print("check_torch_pow_association_is_owed: OWED, not answered.")
    print("  `(x*x)*x` vs `x*(x*x)` in the tanh gelu:", n_diff, "of", n_cmp,
          "lanes differ, max", max_ulp, "ulp, max abs", max_abs,
          "at x =", worst)
    if n_diff == 0:
        print("  The association is BIT-INERT here, so DEVIATION 824's open")
        print("  judgment call costs nothing and the corpus arm is optional.")
    else:
        print("  The association IS worth bits. Until a torch corpus lands,")
        print("  DEVIATION 824's `(x*x)*x` is an UNVERIFIED CHOICE and this")
        print("  gate certifies our spelling of it, not agreement with torch.")


def check_sabotage_arms(
    host_tanh: MutPointer[Float32, MutUntrackedOrigin],
    host_ge: MutPointer[Float32, MutUntrackedOrigin],
    host_gt: MutPointer[Float32, MutUntrackedOrigin],
    host_tanh_sab: MutPointer[Float32, MutUntrackedOrigin],
    host_ge_sab: MutPointer[Float32, MutUntrackedOrigin],
    host_gt_sab: MutPointer[Float32, MutUntrackedOrigin],
    xs: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """DEVIATION 941: one counter per function plus a SHAPE check on which
    lanes moved, because "it moved" is a weaker statement than "it moved
    exactly where the clause lives"."""
    var reach_tanh = 0
    var reach_ge = 0
    var reach_gt = 0
    var tanh_moved_nonzero = 0
    var tanh_moved_neg_zero = 0
    for i in range(N):
        var ht = _bits(host_tanh.unsafe_load(i))
        var hts = _bits(host_tanh_sab.unsafe_load(i))
        if not _same(ht, hts):
            reach_tanh += 1
            var xb = _bits(_ftz_always(xs.unsafe_load(i)))
            if xb == UInt32(0x80000000):
                tanh_moved_neg_zero += 1
            else:
                tanh_moved_nonzero += 1
        if not _same(_bits(host_ge.unsafe_load(i)), _bits(host_ge_sab.unsafe_load(i))):
            reach_ge += 1
        if not _same(_bits(host_gt.unsafe_load(i)), _bits(host_gt_sab.unsafe_load(i))):
            reach_gt += 1
    print("reach counters (portable vs the local spelling, host):")
    print("  tanh", reach_tanh, " gelu_erf", reach_ge, " gelu_tanh", reach_gt)

    comptime if not ANY_SABOTAGE:
        if reach_tanh != 0 or reach_ge != 0 or reach_gt != 0:
            raise Error(
                "NO arm is armed and the local spellings still disagree with"
                + " the real functions: the copies in this file have DRIFTED"
                + " and every sabotage result they produce is worthless"
            )
        print("  clean build: the local spellings are bit-equal to the real ones")
    comptime if SAB_ONE_GELU:
        if reach_gt == 0:
            raise Error(
                "ONE_GELU moved nothing: the arm did not build at all"
            )
        if reach_ge != 0:
            raise Error(
                "ONE_GELU moved the ERF gelu, which it does not touch"
            )
        print("  ONE_GELU: gelu_tanh moved on", reach_gt,
              "lanes, gelu_erf on none -- and the negative control above")
        print("  fired, which is the demonstration")
    comptime if SAB_FUSED_INNER:
        if reach_gt == 0:
            raise Error(
                "FUSED_INNER moved nothing: the UNFUSED decision in"
                + " DEVIATION 824 step 3 is not reached, so nothing gates it"
            )
        if reach_ge != 0:
            raise Error(
                "FUSED_INNER moved the ERF gelu, which has no such seam"
            )
        print("  FUSED_INNER: gelu_tanh moved on", reach_gt,
              "lanes -- the unfused association is reached")
    comptime if SAB_DOUBLE_CONST:
        if reach_ge == 0:
            print("  DOUBLE_CONST: INERT, AS PREDICTED (DEVIATION 946).")
            print("    `Float32(1.4142135623730951)` is the same float32 as")
            print("    GELU_SQRT2_BITS, so there is nothing here to move.")
            print("    This arm buys NO coverage and is kept only so that")
            print("    its inertness is a recorded measurement.")
        else:
            raise Error(
                "DOUBLE_CONST moved " + String(reach_ge)
                + " lanes, which it was predicted not to. Either Mojo's"
                + " decimal literal does not round-trip to 0x3FB504F3 -- a"
                + " finding about `mojo-string-float-roundtrip`, not about"
                + " gelu -- or the constant in numerics.mojo is not what its"
                + " comment says"
            )
    comptime if SAB_CEPHES_ZERO:
        if reach_tanh == 0:
            raise Error(
                "CEPHES_ZERO moved nothing: `check_tanh_planted`'s"
                + " sign-bit assertion is BLIND and the fixed bug is not"
                + " actually gated"
            )
        if tanh_moved_nonzero != 0:
            raise Error(
                "CEPHES_ZERO moved " + String(tanh_moved_nonzero)
                + " lanes whose flushed input is NOT a negative zero. It is"
                + " supposed to move the negative zeros and NOTHING ELSE; a"
                + " wider blast radius means the guard is doing more than"
                + " restoring a sign"
            )
        print("  CEPHES_ZERO: moved", tanh_moved_neg_zero,
              "lanes, every one of them a negative zero -- and nothing else")


def main() raises:
    for a in argv():
        if a == "sabotage":
            print("NOTE: this gate takes its reach arms as `-D` defines, not")
            print("      as an argv word. See the module docstring.")

    print("portable gelu check (DEVIATIONS 821-824); build mode", _mode_name(),
          "(the portable_* functions do not depend on it)")
    print("gelu sabotage", gelu_sabotage_name())
    comptime if NO_VENDOR_WITNESS:
        print("VENDOR WITNESS DISABLED. DEVIATION 821's open question about")
        print("  std.math.tanh / std.math.erf lowering through float64 is")
        print("  NOT answered by this run.")

    check_gelu_constants()
    check_two_gelus_disagree()
    check_tanh_branch_boundary()
    check_tanh_planted()
    check_erf_planted()
    check_erf_odd()
    check_gelu_planted()
    check_torch_pow_association_is_owed()

    var ctx = DeviceContext()

    var h_x = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_one = ctx.enqueue_create_host_buffer[DType.float32](1)
    var px = h_x.unsafe_ptr()
    h_one.unsafe_ptr().unsafe_store(0, Float32(1.0))
    var n_sub = 0
    var n_nan = 0
    for i in range(N):
        var x = _gelu_x(i)
        px.unsafe_store(i, x)
        var b = _bits(x)
        if (b & UInt32(0x7F800000)) == UInt32(0) and (b & UInt32(0x007FFFFF)) != UInt32(0):
            n_sub += 1
        if _is_nan(b):
            n_nan += 1
    print("fixture:", N, "lanes,", n_sub, "subnormal,", n_nan, "NaN")

    # ---- arm 1: accuracy on the host, RECORDED -------------------------
    var h_tanh = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_erf = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_ge = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_gt = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_tanh_sab = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_ge_sab = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_gt_sab = ctx.enqueue_create_host_buffer[DType.float32](N)
    var pt = h_tanh.unsafe_ptr()
    var pe = h_erf.unsafe_ptr()
    var pge = h_ge.unsafe_ptr()
    var pgt = h_gt.unsafe_ptr()
    var pts = h_tanh_sab.unsafe_ptr()
    var pges = h_ge_sab.unsafe_ptr()
    var pgts = h_gt_sab.unsafe_ptr()
    var max_ulp_t = Int64(0)
    var max_ulp_t_poly = Int64(0)
    var max_ulp_e = Int64(0)
    var max_ulp_e_poly = Int64(0)
    var max_ulp_ge = Int64(0)
    var max_ulp_gt = Int64(0)
    var n_t = 0
    var n_e = 0
    var worst_t = Float32(0.0)
    var worst_e = Float32(0.0)
    for i in range(N):
        var x = px.unsafe_load(i)
        var yt = portable_tanhf(x)
        var ye = portable_erff(x)
        var yge = portable_gelu_erf(x)
        var ygt = portable_gelu_tanh(x)
        pt.unsafe_store(i, yt)
        pe.unsafe_store(i, ye)
        pge.unsafe_store(i, yge)
        pgt.unsafe_store(i, ygt)
        pts.unsafe_store(i, _sab_tanhf(x))
        pges.unsafe_store(i, _sab_gelu_erf(x))
        pgts.unsafe_store(i, _sab_gelu_tanh(x))
        if not _normal(_bits(x)):
            continue
        var x64 = Float64(x)
        n_t += 1
        var rt = Float32(_tanh64(x64))
        var dt = _ulp_dist(yt, rt)
        if yt == Float32(0.0) and rt == Float32(0.0):
            dt = Int64(0)
        if dt > max_ulp_t:
            max_ulp_t = dt
            worst_t = x
        if abs(x) < Float32(0.625) and dt > max_ulp_t_poly:
            max_ulp_t_poly = dt
        n_e += 1
        var re = Float32(_erf64(x64))
        var de = _ulp_dist(ye, re)
        if ye == Float32(0.0) and re == Float32(0.0):
            de = Int64(0)
        if de > max_ulp_e:
            max_ulp_e = de
            worst_e = x
        if abs(x) <= Float32(1.0) and de > max_ulp_e_poly:
            max_ulp_e_poly = de
        # the gelus, against the float64 composition of the float64
        # references. A REFERENCE FOR DISTANCE, not an oracle: torch's own
        # answer is the float32 composition and it is what we agree with.
        if abs(x) < Float32(30.0):
            var rge = Float32(x64 * 0.5 * (1.0 + _erf64(x64 / 1.4142135623730951)))
            var rgt = Float32(
                x64 * 0.5
                * (1.0 + _tanh64(0.7978845608028654 * (x64 + 0.044715 * x64 * x64 * x64)))
            )
            var dge = _ulp_dist(yge, rge)
            var dgt = _ulp_dist(ygt, rgt)
            if yge == Float32(0.0) and rge == Float32(0.0):
                dge = Int64(0)
            if ygt == Float32(0.0) and rgt == Float32(0.0):
                dgt = Int64(0)
            if dge > max_ulp_ge:
                max_ulp_ge = dge
            if dgt > max_ulp_gt:
                max_ulp_gt = dgt

    print("ACCURACY, RECORDED, NOT ASSERTED (DEVIATIONS 821-824 all say the")
    print("  bound of THIS spelling was never measured; a guessed bound is")
    print("  how a check comes to assert what the code does):")
    print("  portable_tanhf vs libm double tanh (", n_t, "normal lanes ): max",
          max_ulp_t, "ulp (worst x =", worst_t, "); polynomial arm max",
          max_ulp_t_poly, "ulp")
    print("  portable_erff vs libm double erf (", n_e, "normal lanes ): max",
          max_ulp_e, "ulp (worst x =", worst_e, "); polynomial arm max",
          max_ulp_e_poly, "ulp")
    print("  portable_gelu_erf vs the float64 composition: max", max_ulp_ge, "ulp")
    print("  portable_gelu_tanh vs the float64 composition: max", max_ulp_gt, "ulp")
    print("  Freeze these into assertions on the SECOND run, once three")
    print("  columns have printed the same ones.")

    # ---- arm 2 + arm 3: the device ------------------------------------
    var d_x = ctx.enqueue_create_buffer[DType.float32](N)
    var d_one = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_buf=d_x, src_ptr=h_x.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_one, src_ptr=h_one.unsafe_ptr())
    var o_t = ctx.enqueue_create_buffer[DType.float32](N)
    var o_e = ctx.enqueue_create_buffer[DType.float32](N)
    var o_ge = ctx.enqueue_create_buffer[DType.float32](N)
    var o_gt = ctx.enqueue_create_buffer[DType.float32](N)
    var o_ts = ctx.enqueue_create_buffer[DType.float32](N)
    var o_ges = ctx.enqueue_create_buffer[DType.float32](N)
    var o_gts = ctx.enqueue_create_buffer[DType.float32](N)
    var o_vt = ctx.enqueue_create_buffer[DType.float32](N)
    var o_ve = ctx.enqueue_create_buffer[DType.float32](N)
    var o_probe = ctx.enqueue_create_buffer[DType.float32](N)

    ctx.enqueue_function[gelu_kernel](
        d_x.unsafe_ptr(), d_one.unsafe_ptr(),
        o_t.unsafe_ptr(), o_e.unsafe_ptr(), o_ge.unsafe_ptr(), o_gt.unsafe_ptr(),
        o_ts.unsafe_ptr(), o_ges.unsafe_ptr(), o_gts.unsafe_ptr(),
        o_vt.unsafe_ptr(), o_ve.unsafe_ptr(), o_probe.unsafe_ptr(),
        Int32(N),
        grid_dim=(N + BLOCK - 1) // BLOCK,
        block_dim=(BLOCK, 1, 1),
    )

    var r_t = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_e = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_ge = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_gt = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_vt = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_ve = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_probe = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_ptr=r_t.unsafe_ptr(), src_buf=o_t)
    ctx.enqueue_copy(dst_ptr=r_e.unsafe_ptr(), src_buf=o_e)
    ctx.enqueue_copy(dst_ptr=r_ge.unsafe_ptr(), src_buf=o_ge)
    ctx.enqueue_copy(dst_ptr=r_gt.unsafe_ptr(), src_buf=o_gt)
    ctx.enqueue_copy(dst_ptr=r_vt.unsafe_ptr(), src_buf=o_vt)
    ctx.enqueue_copy(dst_ptr=r_ve.unsafe_ptr(), src_buf=o_ve)
    ctx.enqueue_copy(dst_ptr=r_probe.unsafe_ptr(), src_buf=o_probe)
    ctx.synchronize()

    var qt = r_t.unsafe_ptr()
    var qe = r_e.unsafe_ptr()
    var qge = r_ge.unsafe_ptr()
    var qgt = r_gt.unsafe_ptr()
    var qvt = r_vt.unsafe_ptr()
    var qve = r_ve.unsafe_ptr()
    var qp = r_probe.unsafe_ptr()

    var device_flushes = _bits(qp.unsafe_load(8)) == UInt32(0)
    if device_flushes:
        print("flush probe: this column FLUSHES subnormals")
    else:
        print("flush probe: this column HONORS subnormals")

    var mm0 = 0
    var mm1 = 0
    var mm2 = 0
    var mm3 = 0
    var hash0 = UInt64(0)
    var hash1 = UInt64(0)
    var hash2 = UInt64(0)
    var hash3 = UInt64(0)
    var fnv = UInt64(0xCBF29CE484222325)
    var shown: Int
    for k in range(4):
        var dp = qt
        var hp = pt
        if k == 1:
            dp = qe
            hp = pe
        elif k == 2:
            dp = qge
            hp = pge
        elif k == 3:
            dp = qgt
            hp = pgt
        var bad = 0
        var hk = UInt64(0xCBF29CE484222325)
        shown = 0
        for i in range(N):
            var db = _bits(dp.unsafe_load(i))
            var hb = _bits(hp.unsafe_load(i))
            if not _same(db, hb):
                bad += 1
                if shown < 3:
                    print("  mismatch fn", k, "lane", i, "device", hex(db),
                          "host", hex(hb))
                    shown += 1
            var c = UInt64(_canon(db))
            hk = (hk ^ c) * UInt64(0x100000001B3)
            fnv = (fnv ^ c) * UInt64(0x100000001B3)
        if k == 0:
            mm0 = bad
            hash0 = hk
        elif k == 1:
            mm1 = bad
            hash1 = hk
        elif k == 2:
            mm2 = bad
            hash2 = hk
        else:
            mm3 = bad
            hash3 = hk

    print("gelu device hash =", fnv)
    print("  per function: tanh", hash0, " erf", hash1,
          " gelu_erf", hash2, " gelu_tanh", hash3)
    print("  device/host mismatches: tanh", mm0, " erf", mm1,
          " gelu_erf", mm2, " gelu_tanh", mm3)

    # ---- DEVIATION 942: the vendor witness, RECORDED -------------------
    # DEVIATION 949's second half. The first run of this gate printed
    # `vendor tanh vs portable_tanhf: max 72109828 ulp`. Seventy-two
    # million ulp is not a rounding disagreement: tanh's RANGE IS [-1, 1],
    # so two correct answers cannot be more than about 2^25 ulp apart
    # anywhere in it, and 72,109,828 is 0x044C4F04, which read as a float32
    # bit pattern is a NORMAL of exponent field 8, about 2.4e-36. That is
    # the signature of one side being a tiny normal while the other is
    # zero, and this block exists to say which side, at which input, in
    # which band -- because "max 72109828" is a number nobody can act on.
    #
    # NOTHING HERE IS ASSERTED. The FAST arm is the vendor's own spelling
    # and is allowed to differ; this gate's job is to CHARACTERIZE, and
    # anything it finds belongs in a report to Modular, not in a failure.
    comptime if not NO_VENDOR_WITNESS:
        var vt_max = Int64(0)
        var ve_max = Int64(0)
        var vt_exact = 0
        var ve_exact = 0
        var vn = 0
        var vt_absmax = Float64(0.0)
        # the worst vendor-tanh lane and the two previous record-holders,
        # kept by INDEX so they can be DESCRIBED afterwards rather than
        # merely counted. Three rather than one because a single outlier
        # and a whole broken band read identically as one number.
        var wt0 = -1
        var wt1 = -1
        var wt2 = -1
        var we0 = -1
        # the census that separates "last bit differs" from "structurally
        # wrong". tanh and erf both map into [-1, 1]; a vendor answer
        # outside it, or non-finite, is a defect on its face.
        var vt_out_of_range = 0
        var ve_out_of_range = 0
        var vt_nonfinite = 0
        var ve_nonfinite = 0
        var vt_zeroed = 0     # vendor is +-0 where portable is not
        var vt_unzeroed = 0   # portable is +-0 where vendor is not
        var vt_signflip = 0   # both non-zero, opposite signs
        # max ulp per magnitude band of |x|, which localizes a defect that
        # lives in one band instead of averaging it away
        var bmax0 = Int64(0)
        var bmax1 = Int64(0)
        var bmax2 = Int64(0)
        var bmax3 = Int64(0)
        var bmax4 = Int64(0)
        var bmax5 = Int64(0)
        var bn0 = 0
        var bn1 = 0
        var bn2 = 0
        var bn3 = 0
        var bn4 = 0
        var bn5 = 0
        for i in range(N):
            var x = px.unsafe_load(i)
            if not _normal(_bits(x)) or abs(x) > Float32(8.0):
                continue
            vn += 1
            var vt = qvt.unsafe_load(i)
            var ve = qve.unsafe_load(i)
            var ht = pt.unsafe_load(i)
            var he = pe.unsafe_load(i)
            var dt = _ulp_dist(vt, ht)
            var de = _ulp_dist(ve, he)
            if dt == Int64(0):
                vt_exact += 1
            if de == Int64(0):
                ve_exact += 1
            if dt > vt_max:
                vt_max = dt
                wt2 = wt1
                wt1 = wt0
                wt0 = i
            if de > ve_max:
                ve_max = de
                we0 = i
            if _finite(_bits(vt)):
                if abs(vt) > Float32(1.0):
                    vt_out_of_range += 1
                var da = abs(Float64(vt) - Float64(ht))
                if da > vt_absmax:
                    vt_absmax = da
            else:
                vt_nonfinite += 1
            if _finite(_bits(ve)):
                if abs(ve) > Float32(1.0):
                    ve_out_of_range += 1
            else:
                ve_nonfinite += 1
            if vt == Float32(0.0) and ht != Float32(0.0):
                vt_zeroed += 1
            if ht == Float32(0.0) and vt != Float32(0.0):
                vt_unzeroed += 1
            if vt != Float32(0.0) and ht != Float32(0.0):
                if (_bits(vt) & UInt32(0x80000000)) != (
                    _bits(ht) & UInt32(0x80000000)
                ):
                    vt_signflip += 1
            var b = _mag_band(x)
            if b == 0:
                bn0 += 1
                if dt > bmax0:
                    bmax0 = dt
            elif b == 1:
                bn1 += 1
                if dt > bmax1:
                    bmax1 = dt
            elif b == 2:
                bn2 += 1
                if dt > bmax2:
                    bmax2 = dt
            elif b == 3:
                bn3 += 1
                if dt > bmax3:
                    bmax3 = dt
            elif b == 4:
                bn4 += 1
                if dt > bmax4:
                    bmax4 = dt
            else:
                bn5 += 1
                if dt > bmax5:
                    bmax5 = dt
        print("VENDOR WITNESS, RECORDED (DEVIATION 942). `std.math.tanh` and")
        print("  `std.math.erf` COMPILED AND RAN on this device -- that alone")
        print("  answers DEVIATION 821's open float64-lowering question for")
        print("  this column. They are `identical_tanh`/`identical_erf`'s FAST")
        print("  arms and are ALLOWED to differ; nothing here is asserted.")
        print("  over", vn, "normal lanes with |x| < 8:")
        print("  vendor tanh vs portable_tanhf: max", vt_max, "ulp, exact on",
              vt_exact)
        print("  vendor erf  vs portable_erff : max", ve_max, "ulp, exact on",
              ve_exact)
        print("  max ABSOLUTE difference, vendor tanh:", vt_absmax)
        print("    (tanh maps into [-1, 1], so anything near or above 2 here")
        print("     is structural and not a rounding disagreement)")
        print("  vendor range census: tanh |v| > 1 on", vt_out_of_range,
              "lanes, non-finite on", vt_nonfinite)
        print("                       erf  |v| > 1 on", ve_out_of_range,
              "lanes, non-finite on", ve_nonfinite)
        print("  vendor tanh returned +-0 where portable did not:", vt_zeroed)
        print("  portable returned +-0 where vendor tanh did not:", vt_unzeroed)
        print("  opposite signs, both non-zero:", vt_signflip)
        print("  max ulp by |x| band (lane count, max ulp):")
        print("    |x| < 1e-30        ", bn0, bmax0)
        print("    1e-30 .. 1e-20     ", bn1, bmax1)
        print("    1e-20 .. 1e-10     ", bn2, bmax2)
        print("    1e-10 .. 1e-3      ", bn3, bmax3)
        print("    1e-3  .. 1         ", bn4, bmax4)
        print("    1     .. 8         ", bn5, bmax5)
        print("  THE WORST VENDOR TANH LANE AND THE TWO PREVIOUS RECORDS:")
        for t in range(3):
            var wi = wt0
            if t == 1:
                wi = wt1
            elif t == 2:
                wi = wt2
            if wi < 0:
                continue
            var xw = px.unsafe_load(wi)
            var hw = pt.unsafe_load(wi)
            var vw = qvt.unsafe_load(wi)
            print("    lane", wi)
            print("      x        =", xw, hex(_bits(xw)))
            print("      portable =", hw, hex(_bits(hw)))
            print("      vendor   =", vw, hex(_bits(vw)))
            print("      ulp dist =", _ulp_dist(vw, hw),
                  " abs diff =", abs(Float64(vw) - Float64(hw)))
        if we0 >= 0:
            var xe = px.unsafe_load(we0)
            print("  THE WORST VENDOR ERF LANE:")
            print("    lane", we0)
            print("      x        =", xe, hex(_bits(xe)))
            print("      portable =", pe.unsafe_load(we0),
                  hex(_bits(pe.unsafe_load(we0))))
            print("      vendor   =", qve.unsafe_load(we0),
                  hex(_bits(qve.unsafe_load(we0))))
        print("  READ IT AS: if the out-of-range and non-finite counts are 0")
        print("  and the large band maxes sit only in the tiny-|x| bands,")
        print("  the vendor is losing a tiny argument that tanh must return")
        print("  UNCHANGED (tanh(x) = x below about 1e-4), which is an")
        print("  underflow in its own reduction and not a rounding choice.")
        print("  If instead |v| > 1 or non-finite is non-zero, the lowering")
        print("  itself is broken and the band table says where.")

    if mm0 + mm1 + mm2 + mm3 != 0:
        raise Error(
            "device bits differ from host bits on "
            + String(mm0 + mm1 + mm2 + mm3)
            + " lanes: this backend broke the basic-ops premise"
        )
    print("device == host on every lane")

    # EVERY pointer handed on carries an explicit, IDENTICAL origin cast,
    # the spelling `ensemble/mojo_only/sampled_cols_check.mojo` uses.
    check_sabotage_arms(
        pt.unsafe_origin_cast[MutUntrackedOrigin](),
        pge.unsafe_origin_cast[MutUntrackedOrigin](),
        pgt.unsafe_origin_cast[MutUntrackedOrigin](),
        pts.unsafe_origin_cast[MutUntrackedOrigin](),
        pges.unsafe_origin_cast[MutUntrackedOrigin](),
        pgts.unsafe_origin_cast[MutUntrackedOrigin](),
        px.unsafe_origin_cast[MutUntrackedOrigin](),
    )

    comptime if ANY_SABOTAGE:
        print("SABOTAGED BUILD (" + gelu_sabotage_name()
              + "): the arm behaved as predicted above.")
    else:
        print("all arms OK")

    _ = h_x.unsafe_ptr()
    _ = h_one.unsafe_ptr()
    _ = h_tanh.unsafe_ptr()
    _ = h_erf.unsafe_ptr()
    _ = h_ge.unsafe_ptr()
    _ = h_gt.unsafe_ptr()
    _ = h_tanh_sab.unsafe_ptr()
    _ = h_ge_sab.unsafe_ptr()
    _ = h_gt_sab.unsafe_ptr()
    _ = r_t.unsafe_ptr()
    _ = r_e.unsafe_ptr()
    _ = r_ge.unsafe_ptr()
    _ = r_gt.unsafe_ptr()
    _ = r_vt.unsafe_ptr()
    _ = r_ve.unsafe_ptr()
    _ = r_probe.unsafe_ptr()
