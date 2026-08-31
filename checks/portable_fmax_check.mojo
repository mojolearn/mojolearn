# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTITY_PATHS: softmax's row maximum. DEVIATION 825's `portable_fmaxf`,
plus the one assertion DEVIATION 826's `identical_mul` was owed. The sixth
gate of portable device arithmetic, shaped like
`checks/portable_nn_check.mojo` and not reinvented.

THE POINT OF THIS FILE IS THAT THE CONSUMER IS A REDUCTION AND NOT A BINARY
OP. `check_fmax_fold_invariant` folds one planted row LEFT TO RIGHT, RIGHT
TO LEFT and as a BALANCED BINARY TREE and requires one answer from all
three. A pairwise check structurally cannot see the defect this function
exists to prevent: IDENTITY_PATHS row 13 was a fold-order defect, and
`fmax(a, b) == fmax(b, a)` holds for a plain float max on every input that
is not a zero pair, so a commutativity check alone would have passed
through it.

THREE ARMS, over 2^20 hashed and planted operand PAIRS:

1. HOST -- order invariance, fold invariance in three orders, the signed
   zero contract BY SIGN BIT, the NaN contract with BOTH PAYLOAD SIGNS, and
   `_total_order_key` against `range_key`, its original in
   `extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo`.
   No accuracy arm and no ulp figure: nothing here approximates anything.
   The map is exactly invertible and the selection is integer, so every
   assertion in this file is exact or it is not made.
2. DEVICE == HOST -- one kernel evaluates the same pairs and the same three
   fold orders; the device bits must equal the host bits on EVERY lane
   INCLUDING the subnormal ones. That is a stronger claim than the other
   portable gates make and DEVIATION 825 is why it is available: integer
   ops do not flush, the operands are flushed explicitly first, so there is
   no column-dependent step left. `fmax device hash` is the certificate
   line.
3. REACH -- the sabotage arms below, `-D` selected.

DEVIATION 940 applies: the arms are LOCAL re-spellings in this file rather
than edits to `numerics.mojo`, which this lane does not own. With no define
armed the local spelling must be bit-equal to the real function on every
lane, which is an always-on assertion that the copy has not drifted.

SABOTAGE ARMS:

  -D MOJOLEARN_FMAX_SABOTAGE_HARDWARE_MAX=1
        Use `std.math.max`. MUST fail the signed-zero and NaN checks.
        IT HAS A PREDICTED ANSWER PER COLUMN, not merely "must differ":
        IDENTITY_PATHS row 39 MEASURED `max(+0.0, -0.0)` as `-0.0` on Apple
        and `+0.0` on NVIDIA and AMD, so the run PRINTS what this column
        actually returned for both operand orders.
        BOTH OPERANDS ARE RUNTIME VALUES READ FROM A BUFFER, on purpose.
        DEVIATION 663's `HW_MAX_CLAMP` arm came back NULL ON APPLE because
        LLVM folded `maxnum(0.0, v)` into a compare-select whose tie answer
        is `+0.0` -- the spelling's answer depended on whether a constant
        got folded. A compile-time zero here would repeat that and the arm
        would go inert for the wrong reason.
  -D MOJOLEARN_FMAX_SABOTAGE_SIGN_UNFLIPPED=1
        Set the sign bit unconditionally instead of inverting a negative's
        bits. `RANGE_SAB_SIGN_UNFLIPPED` from DEVIATION 204, verbatim. For
        a non-negative float the two spellings are the SAME expression, so
        this must move EXACTLY the comparisons carrying a negative operand
        and the run asserts that SHAPE, not just a count.
  -D MOJOLEARN_FMAX_SABOTAGE_NO_NAN_CANON=1
        Drop the NaN branch and let the key map decide. MUST fail on the
        NEGATIVE-PAYLOAD NaN lanes specifically: a positive quiet NaN keys
        ABOVE +inf and wins every max while a negative one keys BELOW -inf
        and loses every max, so a negative payload is the input that
        separates the canonicalization from the map. The run asserts that
        at least one moved lane carries a negative-payload NaN.

Run:
    pixi run check-portable-fmax
    pixi run mojo run -D MOJOLEARN_FMAX_SABOTAGE_HARDWARE_MAX=1 \\
        -I . checks/portable_fmax_check.mojo
and the same for the other two names. A misspelled define is silently
ignored, so the run PRINTS the arm it was built with (`fmax sabotage
<name>`, `tools/gemm_ladder.sh:71`'s scar); read that line first.
"""

from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import max
from std.memory import bitcast
from std.sys import argv
from std.sys.compile import is_defined

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    _ftz_always,
    _total_order_key,
    identical_mul,
    identical_mul_add,
    portable_fmaxf,
    numeric_mode_name,
)

# DEVIATION 947's cross-file arm. These four imports are the only reason
# this gate depends on anything outside `checks/`, and they are here
# because DEVIATION 826 names four copies of one arithmetic as "the good
# case and also the fragile one". `range_key` is DEVIATION 204's original,
# which `_total_order_key` is a deliberate second copy of; the three
# `pinned_mul`s are DEVIATION 720's copies of `identical_mul`.
#
# IF THE MAMBA OR EXTRATREES TREES ARE MID-EDIT AND THIS GATE WILL NOT
# BUILD, these five imports and the two checks that use them are the first
# thing to lift out -- they are drift detectors for other lanes' files, not
# claims about `numerics.mojo` itself.
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    range_key,
)
from mamba.checks.mamba_oracle import pinned_mul as pinned_mul_oracle
from mamba.impl.mamba_ssm.ops.selective_scan_interface import (
    pinned_mul as pinned_mul_scan,
)
from mamba.impl.transformers.models.mamba.modeling_mamba import (
    pinned_mul as pinned_mul_block,
)

comptime N = 1 << 20
comptime BLOCK = 256
comptime GROUP = 16  # lanes per fold group in the device fold arm
comptime NGROUP = N // GROUP

comptime SAB_HARDWARE_MAX = is_defined["MOJOLEARN_FMAX_SABOTAGE_HARDWARE_MAX"]()
comptime SAB_SIGN_UNFLIPPED = is_defined["MOJOLEARN_FMAX_SABOTAGE_SIGN_UNFLIPPED"]()
comptime SAB_NO_NAN_CANON = is_defined["MOJOLEARN_FMAX_SABOTAGE_NO_NAN_CANON"]()

comptime ANY_SABOTAGE = (
    SAB_HARDWARE_MAX or SAB_SIGN_UNFLIPPED or SAB_NO_NAN_CANON
)


def fmax_sabotage_name() -> String:
    comptime if SAB_HARDWARE_MAX:
        return String("HARDWARE_MAX")
    comptime if SAB_SIGN_UNFLIPPED:
        return String("SIGN_UNFLIPPED")
    comptime if SAB_NO_NAN_CANON:
        return String("NO_NAN_CANON")
    return String("none")


# ---- the local re-spellings (DEVIATION 940) -----------------------------
def _sab_key(v: Float32) -> UInt32:
    """A faithful copy of `_total_order_key` carrying the SIGN_UNFLIPPED
    arm."""
    var b = rebind[UInt32](v.to_bits())
    comptime if SAB_SIGN_UNFLIPPED:
        return b | UInt32(0x80000000)
    if (b & UInt32(0x80000000)) != UInt32(0):
        return ~b
    return b | UInt32(0x80000000)


def _sab_fmax(a_in: Float32, b_in: Float32) -> Float32:
    """A faithful copy of `portable_fmaxf` carrying all three arms."""
    comptime if SAB_HARDWARE_MAX:
        # No flush, no canonicalization, no key: the vendor's own
        # instruction, which is the whole point of the arm.
        return max(a_in, b_in)
    comptime if not SAB_NO_NAN_CANON:
        if a_in != a_in or b_in != b_in:
            return bitcast[DType.float32](UInt32(0x7FC00000))
    var a = _ftz_always(a_in)
    var b = _ftz_always(b_in)
    if _sab_key(b) > _sab_key(a):
        return b
    return a


def fmax_kernel(
    xa: MutPointer[Float32, MutAnyOrigin],
    xb: MutPointer[Float32, MutAnyOrigin],
    out_ab: MutPointer[Float32, MutAnyOrigin],
    out_ba: MutPointer[Float32, MutAnyOrigin],
    out_sab: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var a = xa.unsafe_load(i)
        var b = xb.unsafe_load(i)
        out_ab.unsafe_store(i, portable_fmaxf(a, b))
        out_ba.unsafe_store(i, portable_fmaxf(b, a))
        out_sab.unsafe_store(i, _sab_fmax(a, b))


def fold_kernel(
    xa: MutPointer[Float32, MutAnyOrigin],
    out_lr: MutPointer[Float32, MutAnyOrigin],
    out_rl: MutPointer[Float32, MutAnyOrigin],
    out_tree: MutPointer[Float32, MutAnyOrigin],
    ngroup_in: Int32,
):
    """One thread folds one GROUP-long row three ways. The three orders are
    the same three the host folds, so a fold-order dependence that only
    appears under a device's rounding mode has somewhere to show up."""
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if g < Int(ngroup_in):
        var base = g * GROUP
        var acc = xa.unsafe_load(base)
        for k in range(1, GROUP):
            acc = portable_fmaxf(acc, xa.unsafe_load(base + k))
        out_lr.unsafe_store(g, acc)
        var acc2 = xa.unsafe_load(base + GROUP - 1)
        for k in range(1, GROUP):
            acc2 = portable_fmaxf(acc2, xa.unsafe_load(base + GROUP - 1 - k))
        out_rl.unsafe_store(g, acc2)
        # balanced binary tree, in place over a small stack window
        var t0 = portable_fmaxf(xa.unsafe_load(base + 0), xa.unsafe_load(base + 1))
        var t1 = portable_fmaxf(xa.unsafe_load(base + 2), xa.unsafe_load(base + 3))
        var t2 = portable_fmaxf(xa.unsafe_load(base + 4), xa.unsafe_load(base + 5))
        var t3 = portable_fmaxf(xa.unsafe_load(base + 6), xa.unsafe_load(base + 7))
        var t4 = portable_fmaxf(xa.unsafe_load(base + 8), xa.unsafe_load(base + 9))
        var t5 = portable_fmaxf(xa.unsafe_load(base + 10), xa.unsafe_load(base + 11))
        var t6 = portable_fmaxf(xa.unsafe_load(base + 12), xa.unsafe_load(base + 13))
        var t7 = portable_fmaxf(xa.unsafe_load(base + 14), xa.unsafe_load(base + 15))
        var u0 = portable_fmaxf(t0, t1)
        var u1 = portable_fmaxf(t2, t3)
        var u2 = portable_fmaxf(t4, t5)
        var u3 = portable_fmaxf(t6, t7)
        var v0 = portable_fmaxf(u0, u1)
        var v1 = portable_fmaxf(u2, u3)
        out_tree.unsafe_store(g, portable_fmaxf(v0, v1))


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


def _neg_nan(b: UInt32) -> Bool:
    return _is_nan(b) and (b & UInt32(0x80000000)) != UInt32(0)


def _canon(b: UInt32) -> UInt32:
    if _is_nan(b):
        return UInt32(0x7FC00000)
    return b


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


# ---- the fixture --------------------------------------------------------
# Raw 32-bit patterns are the right fixture here: `portable_fmaxf` has no
# domain, and raw patterns give NaN of BOTH payload signs, both zeros, both
# infinities and subnormals of both signs at their natural density. The
# planted head guarantees the classes the arms need rather than leaving
# them to the hash.
comptime NPLANT = 32


def _class_value(k: Int) -> UInt32:
    if k == 0:
        return UInt32(0x00000000)  # +0
    if k == 1:
        return UInt32(0x80000000)  # -0
    if k == 2:
        return UInt32(0x7F800000)  # +inf
    if k == 3:
        return UInt32(0xFF800000)  # -inf
    if k == 4:
        return UInt32(0x7FC00000)  # +qNaN
    if k == 5:
        return UInt32(0xFFC00000)  # -qNaN, THE separating input
    if k == 6:
        return UInt32(0x7FFFFFFF)  # +NaN, other payload
    if k == 7:
        return UInt32(0xFFFFFFFF)  # -NaN, other payload
    if k == 8:
        return UInt32(0x00400000)  # subnormal
    if k == 9:
        return UInt32(0x80400000)  # -subnormal
    if k == 10:
        return UInt32(0x3F800000)  # 1.0
    if k == 11:
        return UInt32(0xBF800000)  # -1.0
    if k == 12:
        return UInt32(0x7F7FFFFF)  # FLT_MAX
    if k == 13:
        return UInt32(0xFF7FFFFF)  # -FLT_MAX
    if k == 14:
        return UInt32(0x00800000)  # FLT_MIN
    return UInt32(0x80800000)  # -FLT_MIN


comptime NCLASS = 16


def _fmax_a(i: Int) -> Float32:
    if i < NPLANT:
        return _f(_class_value(i % NCLASS))
    var h = _splitmix(UInt64(13 * i + 7))
    if ((h >> 40) & UInt64(3)) == UInt64(0):
        return _f(_class_value(Int((h >> 8) % UInt64(NCLASS))))
    return _f(UInt32(h & UInt64(0xFFFFFFFF)))


def _fmax_b(i: Int) -> Float32:
    if i < NPLANT:
        return _f(_class_value((i // NCLASS + i) % NCLASS))
    var h = _splitmix(UInt64(13 * i + 11))
    if ((h >> 40) & UInt64(3)) == UInt64(0):
        return _f(_class_value(Int((h >> 8) % UInt64(NCLASS))))
    return _f(UInt32(h & UInt64(0xFFFFFFFF)))


# ---- the folds and the planted rows, WITHOUT A CONTAINER ----------------
# Every row is read through a pure function of (kind, k) and every fold is
# written over that function, so no `List` is built, returned or passed.
# The first cut of this file used `List[UInt32]` rows and did not compile:
# a `List` is not implicitly copyable, so returning one needs `^` and
# handing the same one to three folds needs `.copy()` each time. A pure
# indexing function has neither problem and is the spelling the rest of
# this directory already uses.

comptime ROWLEN = 8


def _row_value(kind: Int, k: Int) -> UInt32:
    """The planted fold rows, one cell at a time. Row 0 is the signed-zero
    row and it is the one IDENTITY_PATHS row 13's defect lived in."""
    if kind == 0:
        if k == 0:
            return UInt32(0x80000000)
        if k == 1:
            return UInt32(0x00000000)
        if k == 2:
            return UInt32(0x80000000)
        if k == 3:
            return UInt32(0x00000000)
        if k == 4:
            return UInt32(0x80000000)
        if k == 5:
            return UInt32(0x80000000)
        if k == 6:
            return UInt32(0x00000000)
        return UInt32(0x80000000)
    if kind == 1:
        # both zeros, both infinities, subnormals of both signs, normals
        if k == 0:
            return UInt32(0xBF800000)
        if k == 1:
            return UInt32(0x00000000)
        if k == 2:
            return UInt32(0xFF800000)
        if k == 3:
            return UInt32(0x80400000)
        if k == 4:
            return UInt32(0x80000000)
        if k == 5:
            return UInt32(0x00400000)
        if k == 6:
            return UInt32(0xC0000000)
        return UInt32(0x80800000)
    if kind == 2:
        # a negative NaN payload buried in the middle: under the key map
        # alone it keys BELOW -inf and loses every comparison, so a fold
        # would return a finite number for a row containing a NaN
        if k == 0:
            return UInt32(0xBF800000)
        if k == 1:
            return UInt32(0xFFC00000)
        if k == 2:
            return UInt32(0xFF800000)
        if k == 3:
            return UInt32(0xC0000000)
        if k == 4:
            return UInt32(0xBF000000)
        if k == 5:
            return UInt32(0xC1200000)
        if k == 6:
            return UInt32(0xBE800000)
        return UInt32(0xC2C80000)
    if kind == 3:
        # a POSITIVE NaN payload, which the key map alone would let win
        if k == 0:
            return UInt32(0x3F800000)
        if k == 1:
            return UInt32(0x7FC00000)
        if k == 2:
            return UInt32(0x7F800000)
        if k == 3:
            return UInt32(0x40000000)
        if k == 4:
            return UInt32(0x00000000)
        if k == 5:
            return UInt32(0x80000000)
        if k == 6:
            return UInt32(0x41200000)
        return UInt32(0x42C80000)
    # all negative, which is where SIGN_UNFLIPPED lives
    if k == 0:
        return UInt32(0xBF800000)
    if k == 1:
        return UInt32(0xC0000000)
    if k == 2:
        return UInt32(0xBE800000)
    if k == 3:
        return UInt32(0xC1200000)
    if k == 4:
        return UInt32(0xBF000000)
    if k == 5:
        return UInt32(0xC2C80000)
    if k == 6:
        return UInt32(0xBD800000)
    return UInt32(0xC0490FDB)


def _plant_fold_lr(kind: Int) -> Float32:
    var acc = _f(_row_value(kind, 0))
    for k in range(1, ROWLEN):
        acc = portable_fmaxf(acc, _f(_row_value(kind, k)))
    return acc


def _plant_fold_rl(kind: Int) -> Float32:
    var acc = _f(_row_value(kind, ROWLEN - 1))
    for k in range(1, ROWLEN):
        acc = portable_fmaxf(acc, _f(_row_value(kind, ROWLEN - 1 - k)))
    return acc


def _plant_fold_tree(kind: Int) -> Float32:
    """A balanced binary tree over the eight cells, written out. The shape
    is the third fold order and it is the one a block reduction actually
    builds."""
    var t0 = portable_fmaxf(_f(_row_value(kind, 0)), _f(_row_value(kind, 1)))
    var t1 = portable_fmaxf(_f(_row_value(kind, 2)), _f(_row_value(kind, 3)))
    var t2 = portable_fmaxf(_f(_row_value(kind, 4)), _f(_row_value(kind, 5)))
    var t3 = portable_fmaxf(_f(_row_value(kind, 6)), _f(_row_value(kind, 7)))
    var u0 = portable_fmaxf(t0, t1)
    var u1 = portable_fmaxf(t2, t3)
    return portable_fmaxf(u0, u1)


def _plant_plain_fold_lr(kind: Int) -> Float32:
    """`a > b ? a : b`, the spelling `portable_fmaxf` exists to replace.
    Used only to SHOW that the defect this gate guards against is real on
    this column rather than hypothetical."""
    var acc = _f(_row_value(kind, 0))
    for k in range(1, ROWLEN):
        var v = _f(_row_value(kind, k))
        if v > acc:
            acc = v
    return acc


def _plant_plain_fold_rl(kind: Int) -> Float32:
    var acc = _f(_row_value(kind, ROWLEN - 1))
    for k in range(1, ROWLEN):
        var v = _f(_row_value(kind, ROWLEN - 1 - k))
        if v > acc:
            acc = v
    return acc


# ---- the same three orders over a GROUP-long slice of the sweep ---------
# `_fmax_a(i)` is a pure function of the lane index and it is what fills
# the `a` buffer, so the host folds below and the device `fold_kernel`
# read the same numbers without either one needing a pointer handed to it.
def _sweep_fold_lr(base: Int) -> Float32:
    var acc = _fmax_a(base)
    for k in range(1, GROUP):
        acc = portable_fmaxf(acc, _fmax_a(base + k))
    return acc


def _sweep_fold_rl(base: Int) -> Float32:
    var acc = _fmax_a(base + GROUP - 1)
    for k in range(1, GROUP):
        acc = portable_fmaxf(acc, _fmax_a(base + GROUP - 1 - k))
    return acc


def _sweep_fold_tree(base: Int) -> Float32:
    var t0 = portable_fmaxf(_fmax_a(base + 0), _fmax_a(base + 1))
    var t1 = portable_fmaxf(_fmax_a(base + 2), _fmax_a(base + 3))
    var t2 = portable_fmaxf(_fmax_a(base + 4), _fmax_a(base + 5))
    var t3 = portable_fmaxf(_fmax_a(base + 6), _fmax_a(base + 7))
    var t4 = portable_fmaxf(_fmax_a(base + 8), _fmax_a(base + 9))
    var t5 = portable_fmaxf(_fmax_a(base + 10), _fmax_a(base + 11))
    var t6 = portable_fmaxf(_fmax_a(base + 12), _fmax_a(base + 13))
    var t7 = portable_fmaxf(_fmax_a(base + 14), _fmax_a(base + 15))
    var u0 = portable_fmaxf(t0, t1)
    var u1 = portable_fmaxf(t2, t3)
    var u2 = portable_fmaxf(t4, t5)
    var u3 = portable_fmaxf(t6, t7)
    var v0 = portable_fmaxf(u0, u1)
    var v1 = portable_fmaxf(u2, u3)
    return portable_fmaxf(v0, v1)


def _nan_pattern(t: Int) -> UInt32:
    """Six quiet NaN payloads, three positive and three negative. A
    SIGNALING pattern would add a second variable -- whether the column
    quiets it on load -- to a check that is about the PAYLOAD SIGN, and
    the softmax consumer never produces one anyway."""
    if t == 0:
        return UInt32(0x7FC00000)
    if t == 1:
        return UInt32(0xFFC00000)
    if t == 2:
        return UInt32(0x7FC00001)
    if t == 3:
        return UInt32(0xFFC00001)
    if t == 4:
        return UInt32(0x7FFFFFFF)
    return UInt32(0xFFFFFFFF)


comptime NNAN = 6
comptime NROW = 5


# ==========================================================================
# THE CHECKS
# ==========================================================================
def check_fmax_order_invariant() raises:
    """`portable_fmaxf(a, b) == portable_fmaxf(b, a)` BY BITS over the
    hashed sweep and over the planted class cross-product, which includes
    both zeros, both infinities, quiet NaN of BOTH SIGNS and subnormals of
    both signs."""
    var bad = 0
    for ka in range(NCLASS):
        for kb in range(NCLASS):
            var a = _f(_class_value(ka))
            var b = _f(_class_value(kb))
            if _bits(portable_fmaxf(a, b)) != _bits(portable_fmaxf(b, a)):
                bad += 1
                if bad <= 4:
                    print("  order-dependent at", hex(_class_value(ka)), "/",
                          hex(_class_value(kb)), ":",
                          hex(_bits(portable_fmaxf(a, b))), "vs",
                          hex(_bits(portable_fmaxf(b, a))))
    for i in range(N):
        var a = _fmax_a(i)
        var b = _fmax_b(i)
        if _bits(portable_fmaxf(a, b)) != _bits(portable_fmaxf(b, a)):
            bad += 1
    if bad != 0:
        raise Error(
            "portable_fmaxf is not commutative on " + String(bad)
            + " operand pairs: a reduction built on it has an answer that"
            + " depends on the tree the scheduler happened to build"
        )
    print("check_fmax_order_invariant: commutative on the 16x16 planted")
    print("  class cross-product and on all", N, "hashed pairs")


def check_fmax_fold_invariant() raises:
    """THE POINT OF THIS FILE. Three fold orders, one answer.

    A pairwise check cannot see a fold-order defect: `max(a, b)` and
    `max(b, a)` agree for a plain float max on every input that is not a
    zero pair, so commutativity alone would pass straight through
    IDENTITY_PATHS row 13. Only a FOLD in more than one order can fail.

    The plain `a > b ? a : b` fold is run beside it and its two orders are
    PRINTED, so that the defect being guarded against is demonstrated on
    this column rather than asserted from a memo."""
    for kind in range(NROW):
        var lr = _plant_fold_lr(kind)
        var rl = _plant_fold_rl(kind)
        var tr = _plant_fold_tree(kind)
        if _bits(lr) != _bits(rl) or _bits(lr) != _bits(tr):
            raise Error(
                "planted row " + String(kind)
                + " folds to different BITS in different orders: L->R "
                + hex(_bits(lr)) + ", R->L " + hex(_bits(rl)) + ", tree "
                + hex(_bits(tr))
            )
    print("check_fmax_fold_invariant: all", NROW,
          "planted rows fold to one answer in three orders")
    var pl = _plant_plain_fold_lr(0)
    var pr = _plant_plain_fold_rl(0)
    print("  the plain `a > b ? a : b` fold of the signed-zero row:")
    print("    L->R", hex(_bits(pl)), "  R->L", hex(_bits(pr)))
    if _bits(pl) == _bits(pr):
        print("    they AGREE on this column, so this particular row does not")
        print("    demonstrate the hazard here. It is still real -- row 39")
        print("    measured the split across vendors -- but do not report")
        print("    this line as evidence on this column.")
    else:
        print("    they DISAGREE: the hazard is live on this column and the")
        print("    total-order fold is what closes it.")
    # hashed rows too, so the planted five are not the whole coverage
    var bad = 0
    for g in range(4096):
        var lr = _sweep_fold_lr(g * GROUP)
        var rl = _sweep_fold_rl(g * GROUP)
        var tr = _sweep_fold_tree(g * GROUP)
        if _canon(_bits(lr)) != _canon(_bits(rl)) or _canon(_bits(lr)) != _canon(_bits(tr)):
            bad += 1
    if bad != 0:
        raise Error(
            String(bad) + " hashed rows of " + String(GROUP)
            + " fold to different answers in different orders"
        )
    print("  and 4096 hashed rows of", GROUP, "likewise")


def check_fmax_signed_zero() raises:
    """`max(+0, -0)` and `max(-0, +0)` both return `+0.0` BY SIGN BIT.
    `key(+0)` is 0x80000000 and `key(-0)` is 0x7FFFFFFF, so `+0.0` wins in
    every order. A float equality here would pass on either answer, which
    is exactly the trap."""
    comptime PZ = UInt32(0x00000000)
    comptime NZ = UInt32(0x80000000)
    if _bits(portable_fmaxf(_f(PZ), _f(NZ))) != PZ:
        raise Error("fmax(+0, -0) is not +0.0 BY SIGN BIT")
    if _bits(portable_fmaxf(_f(NZ), _f(PZ))) != PZ:
        raise Error("fmax(-0, +0) is not +0.0 BY SIGN BIT")
    if _bits(portable_fmaxf(_f(NZ), _f(NZ))) != NZ:
        raise Error("fmax(-0, -0) is not -0.0: the map must be an identity here")
    # a subnormal flushes to its signed zero BEFORE the comparison
    if _bits(portable_fmaxf(_f(UInt32(0x80400000)), _f(NZ))) != NZ:
        raise Error("fmax(-subnormal, -0) is not -0.0")
    if _bits(portable_fmaxf(_f(UInt32(0x00400000)), _f(NZ))) != PZ:
        raise Error("fmax(subnormal, -0) is not +0.0")
    if _total_order_key(_f(PZ)) != UInt32(0x80000000):
        raise Error("key(+0) is not 0x80000000")
    if _total_order_key(_f(NZ)) != UInt32(0x7FFFFFFF):
        raise Error("key(-0) is not 0x7FFFFFFF")
    if (
        _bits(_plant_fold_lr(0)) != PZ
        or _bits(_plant_fold_rl(0)) != PZ
        or _bits(_plant_fold_tree(0)) != PZ
    ):
        raise Error("a row containing both zeros does not fold to +0.0 in every order")
    print("check_fmax_signed_zero: +0 wins in both orders and in every fold")


def check_fmax_nan() raises:
    """Any NaN operand, either position, either payload sign, returns
    exactly 0x7FC00000.

    THE SIGN OF THE PAYLOAD IS THE ASSERTION THAT MATTERS. The key map
    ALONE does not settle NaN: a positive quiet NaN keys ABOVE +inf and
    wins every max, a negative one keys BELOW -inf and loses every max, so
    without the canonicalization the answer would depend on a payload's
    sign bit. The two key values are printed so the claim is visible and
    not merely stated."""
    comptime QNAN = UInt32(0x7FC00000)
    for t in range(NNAN):
        var nb = _nan_pattern(t)
        var nan = _f(nb)
        for k in range(NCLASS):
            var other = _f(_class_value(k))
            if _bits(portable_fmaxf(nan, other)) != QNAN:
                raise Error(
                    "fmax(NaN " + hex(nb) + ", " + hex(_class_value(k))
                    + ") is not the canonical 0x7FC00000"
                )
            if _bits(portable_fmaxf(other, nan)) != QNAN:
                raise Error(
                    "fmax(" + hex(_class_value(k)) + ", NaN " + hex(nb)
                    + ") is not the canonical 0x7FC00000"
                )
    print("check_fmax_nan: 6 payloads x 16 classes x 2 positions all return")
    print("  0x7FC00000 exactly")
    print("  why the map alone is not enough: key(+qNaN) =",
          hex(_total_order_key(_f(UInt32(0x7FC00000)))), "> key(+inf) =",
          hex(_total_order_key(_f(UInt32(0x7F800000)))))
    print("  and key(-qNaN) =", hex(_total_order_key(_f(UInt32(0xFFC00000)))),
          "< key(-inf) =", hex(_total_order_key(_f(UInt32(0xFF800000)))))
    if _total_order_key(_f(UInt32(0xFFC00000))) >= _total_order_key(_f(UInt32(0xFF800000))):
        raise Error(
            "a negative quiet NaN does NOT key below -inf, so the input this"
            + " check calls the separating one does not separate anything"
            + " and the NaN arm is untested"
        )


def check_fmax_agrees_with_range_key() raises:
    """`_total_order_key` (numerics.mojo) against `range_key`
    (DEVIATION 204's original, in extratrees). They are one map deliberately
    copied rather than imported -- `numerics.mojo` is the contract every
    other directory imports and pointing it at a decision tree's kernel
    would reverse the arrow -- so this is what makes the copy CHECKABLE
    instead of a claim."""
    var bad = 0
    for k in range(NCLASS):
        var v = _f(_class_value(k))
        if _is_nan(_bits(v)):
            continue  # DEVIATION 163: NaN never reaches range_key
        if _total_order_key(v) != range_key(v):
            bad += 1
    for i in range(1 << 16):
        var h = _splitmix(UInt64(29 * i + 3))
        var v = _f(UInt32(h & UInt64(0xFFFFFFFF)))
        if _is_nan(_bits(v)):
            continue
        if _total_order_key(v) != range_key(v):
            bad += 1
    if bad != 0:
        raise Error(
            "_total_order_key and range_key disagree on " + String(bad)
            + " values: the two copies of DEVIATION 204's map have DRIFTED"
        )
    print("check_fmax_agrees_with_range_key: the two copies of DEVIATION")
    print("  204's map agree on every planted class and 65,536 hashed values")


def check_identical_mul_signed_zero() raises:
    """DEVIATION 947: THE ONE ASSERTION DEVIATION 826 WAS OWED, LANDED HERE.

    `numerics.mojo`'s own note says `identical_mul` needs no check file of
    its own -- it is one expression over `identical_mul_add`, which row 9's
    `check-ieee-arith` already gates -- but that one assertion belongs
    "wherever this file is next gated". This is the next gate, so here it
    is, in the file that already compares things by sign bit.

    THREE THINGS, and the first is not in the specification:

    (a) `Float32(-0.0)` MUST HAVE THE BITS 0x80000000. `identical_mul`'s
        addend is written as a decimal literal and the whole clause is that
        a `+0.0` addend would LAUNDER a negative zero product (the gemm
        lane's F6a lesson). If Mojo's literal were a positive zero the
        clause would be silently broken and every other assertion here
        would still pass. `mojo-string-float-roundtrip` says not to trust a
        literal; this is that rule applied to the one literal that matters.
    (b) `identical_mul(-1.0, 0.0)` and `identical_mul(1.0, -0.0)` are
        `-0.0` BY SIGN BIT.
    (c) `identical_mul` agrees BIT FOR BIT with all three `pinned_mul`
        copies (DEVIATION 720) over a scattered fixture. Four copies of one
        arithmetic have four chances to drift, and this is the assertion
        that catches the fourth."""
    comptime PZ = UInt32(0x00000000)
    comptime NZ = UInt32(0x80000000)
    if _bits(Float32(-0.0)) != NZ:
        raise Error(
            "Float32(-0.0) has the bits " + hex(_bits(Float32(-0.0)))
            + ", not 0x80000000. `identical_mul`'s addend is that literal"
            + " and its whole clause is that a +0.0 addend launders a"
            + " negative zero product: the clause is broken"
        )
    if _bits(identical_mul_add(Float32(1.0), Float32(-0.0), Float32(-0.0))) != NZ:
        raise Error("(-0.0) + (-0.0) did not stay negative through the fma")
    if _bits(identical_mul(Float32(-1.0), _f(PZ))) != NZ:
        raise Error("identical_mul(-1.0, +0.0) is not -0.0 BY SIGN BIT")
    if _bits(identical_mul(Float32(1.0), _f(NZ))) != NZ:
        raise Error("identical_mul(1.0, -0.0) is not -0.0 BY SIGN BIT")
    if _bits(identical_mul(Float32(-1.0), _f(NZ))) != PZ:
        raise Error("identical_mul(-1.0, -0.0) is not +0.0 BY SIGN BIT")
    if _bits(identical_mul(_f(PZ), _f(NZ))) != NZ:
        raise Error("identical_mul(+0.0, -0.0) is not -0.0 BY SIGN BIT")
    var drift_o = 0
    var drift_s = 0
    var drift_b = 0
    var vs_product = 0
    for i in range(1 << 16):
        var h1 = _splitmix(UInt64(31 * i + 5))
        var h2 = _splitmix(UInt64(31 * i + 17))
        var a = _f(UInt32(h1 & UInt64(0xFFFFFFFF)))
        var b = _f(UInt32(h2 & UInt64(0xFFFFFFFF)))
        var m = identical_mul(a, b)
        if _canon(_bits(m)) != _canon(_bits(pinned_mul_oracle(a, b))):
            drift_o += 1
        if _canon(_bits(m)) != _canon(_bits(pinned_mul_scan(a, b))):
            drift_s += 1
        if _canon(_bits(m)) != _canon(_bits(pinned_mul_block(a, b))):
            drift_b += 1
        if _canon(_bits(m)) != _canon(_bits(a * b)):
            vs_product += 1
    if drift_o != 0 or drift_s != 0 or drift_b != 0:
        raise Error(
            "identical_mul has DRIFTED from its DEVIATION 720 copies:"
            + " mamba_oracle " + String(drift_o)
            + ", selective_scan_interface " + String(drift_s)
            + ", modeling_mamba " + String(drift_b)
        )
    print("check_identical_mul_signed_zero: Float32(-0.0) is 0x80000000;")
    print("  the signed-zero products are right BY SIGN BIT; and all three")
    print("  DEVIATION 720 `pinned_mul` copies agree bit for bit over 65,536")
    print("  hashed pairs")
    print("  RECORDED:", vs_product, "of 65536 pairs where identical_mul")
    print("  differs from a plain `a * b` (the pin is supposed to be the")
    print("  correctly rounded product, so 0 is the expected number)")


def check_sabotage_arms(
    host_ab: MutPointer[Float32, MutUntrackedOrigin],
    host_sab: MutPointer[Float32, MutUntrackedOrigin],
    dev_ab: MutPointer[Float32, MutUntrackedOrigin],
    dev_sab: MutPointer[Float32, MutUntrackedOrigin],
    xa: MutPointer[Float32, MutUntrackedOrigin],
    xb: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """DEVIATION 941's counters, plus the SHAPE of which lanes moved --
    "it moved" is a weaker statement than "it moved exactly where the
    clause lives"."""
    var host_reach = 0
    var dev_reach = 0
    var moved_with_negative = 0
    var moved_without_negative = 0
    var moved_with_neg_nan = 0
    var moved_zero_pair = 0
    for i in range(N):
        var h = _bits(host_ab.unsafe_load(i))
        var hs = _bits(host_sab.unsafe_load(i))
        if _canon(h) != _canon(hs):
            host_reach += 1
            var ba = _bits(xa.unsafe_load(i))
            var bb = _bits(xb.unsafe_load(i))
            if _neg_nan(ba) or _neg_nan(bb):
                moved_with_neg_nan += 1
            if (ba & UInt32(0x80000000)) != UInt32(0) or (bb & UInt32(0x80000000)) != UInt32(0):
                moved_with_negative += 1
            else:
                moved_without_negative += 1
            if (ba & UInt32(0x7FFFFFFF)) == UInt32(0) and (bb & UInt32(0x7FFFFFFF)) == UInt32(0):
                moved_zero_pair += 1
        if _canon(_bits(dev_ab.unsafe_load(i))) != _canon(_bits(dev_sab.unsafe_load(i))):
            dev_reach += 1
    print("reach counters (portable vs the local spelling):")
    print("  host", host_reach, " device", dev_reach)
    print("  of the moved lanes:", moved_with_negative, "carry a negative")
    print("  operand,", moved_without_negative, "carry none,",
          moved_with_neg_nan, "carry a NEGATIVE-payload NaN,",
          moved_zero_pair, "are a zero pair")

    comptime if not ANY_SABOTAGE:
        if host_reach != 0 or dev_reach != 0:
            raise Error(
                "NO arm is armed and the local spelling still disagrees with"
                + " `portable_fmaxf`: the copy in this file has DRIFTED and"
                + " every sabotage result it produces is worthless"
            )
        print("  clean build: the local spelling is bit-equal to the real one")
    comptime if SAB_HARDWARE_MAX:
        # The row-39 prediction, per column, read off the operands as
        # RUNTIME values so that no constant fold can answer it early.
        var pz = xa.unsafe_load(0)
        var nz = xb.unsafe_load(0)
        print("  HARDWARE_MAX, row 39's probe. THESE TWO LINES ARE THE HOST;")
        print("  the DEVICE answer is the `device` reach count above and the")
        print("  device/host split is what row 39 actually measured.")
        print("    operands", hex(_bits(pz)), "and", hex(_bits(nz)))
        print("    host max(a, b) =", hex(_bits(max(pz, nz))),
              "  host max(b, a) =", hex(_bits(max(nz, pz))))
        print("    DEVICE max(+0.0, -0.0) =",
              hex(_bits(dev_sab.unsafe_load(0))),
              " <- THE row 39 NUMBER FOR THIS COLUMN")
        print("    row 39 measured -0.0 (the second operand) on Apple's")
        print("    DEVICE and +0.0 on NVIDIA's and AMD's.")
        print("    `portable_fmaxf` returns 0x00000000 everywhere; the")
        print("    device's own portable answer at lane 0 is",
              hex(_bits(dev_ab.unsafe_load(0))))
        # The two contracts the specification says this arm must break,
        # asserted against the SABOTAGED spelling directly. Every operand
        # is a runtime buffer load for DEVIATION 663's reason.
        var nan_op = xa.unsafe_load(4)  # a positive quiet NaN
        var one_op = xa.unsafe_load(10)  # 1.0
        var sz_ok = (
            _bits(_sab_fmax(pz, nz)) == UInt32(0)
            and _bits(_sab_fmax(nz, pz)) == UInt32(0)
        )
        var nan_ok = (
            _bits(_sab_fmax(nan_op, one_op)) == UInt32(0x7FC00000)
            and _bits(_sab_fmax(one_op, nan_op)) == UInt32(0x7FC00000)
        )
        print("    vendor max on NaN: max(NaN, 1.0) =",
              hex(_bits(_sab_fmax(nan_op, one_op))), " max(1.0, NaN) =",
              hex(_bits(_sab_fmax(one_op, nan_op))))
        if sz_ok and nan_ok:
            raise Error(
                "HARDWARE_MAX satisfies BOTH the signed-zero and the NaN"
                + " contract, so `check_fmax_signed_zero` and"
                + " `check_fmax_nan` have nothing to catch on this column"
                + " and are VACUOUS here. Suspect the DEVIATION 663 fold:"
                + " check that both operands reached the instruction as"
                + " runtime values, and run the arm on the other two"
                + " columns before concluding anything (DEVIATION 258)"
            )
        if sz_ok:
            print("    the SIGNED-ZERO contract survived the vendor max on")
            print("    this column (row 39 says NVIDIA and AMD do return")
            print("    +0.0); the NaN contract is what broke here.")
        if nan_ok:
            print("    the NaN contract survived the vendor max on this")
            print("    column; the signed-zero contract is what broke.")
        if host_reach == 0:
            raise Error(
                "HARDWARE_MAX found ZERO mismatches over the whole sweep"
                + " even though one contract broke on the probes, which is"
                + " a contradiction -- the sweep compare is blind"
            )
        print("  HARDWARE_MAX: moved", host_reach, "lanes on the host and",
              dev_reach, "on the device")
    comptime if SAB_SIGN_UNFLIPPED:
        if host_reach == 0:
            raise Error(
                "SIGN_UNFLIPPED moved nothing: the negative branch of the"
                + " key map is not reached, which is the whole content of"
                + " DEVIATION 204's map"
            )
        if moved_without_negative != 0:
            raise Error(
                "SIGN_UNFLIPPED moved " + String(moved_without_negative)
                + " lanes carrying NO negative operand. For a non-negative"
                + " float the two spellings are the SAME expression, so a"
                + " move there means this arm is not the arm it claims to"
                + " be -- the first attempt at DEVIATION 204's version of"
                + " this arm had exactly that shape and the check refused it"
            )
        print("  SIGN_UNFLIPPED: moved", host_reach,
              "lanes, every one carrying a negative operand")
    comptime if SAB_NO_NAN_CANON:
        if host_reach == 0:
            raise Error(
                "NO_NAN_CANON moved nothing: `check_fmax_nan` is blind and"
                + " the canonicalization is ungated"
            )
        if moved_with_neg_nan == 0:
            raise Error(
                "NO_NAN_CANON moved " + String(host_reach)
                + " lanes but NOT ONE of them carries a negative-payload"
                + " NaN. That is the input that separates the"
                + " canonicalization from the key map, so this arm has not"
                + " demonstrated the thing it exists to demonstrate"
            )
        print("  NO_NAN_CANON: moved", host_reach, "lanes,",
              moved_with_neg_nan, "of them carrying a negative-payload NaN")


def main() raises:
    for a in argv():
        if a == "sabotage":
            print("NOTE: this gate takes its reach arms as `-D` defines, not")
            print("      as an argv word. See the module docstring.")

    print("portable fmax check (DEVIATION 825); build mode", _mode_name(),
          "(the portable_* functions do not depend on it)")
    print("fmax sabotage", fmax_sabotage_name())

    check_identical_mul_signed_zero()
    check_fmax_agrees_with_range_key()
    check_fmax_signed_zero()
    check_fmax_nan()
    check_fmax_order_invariant()
    check_fmax_fold_invariant()

    var ctx = DeviceContext()

    var h_a = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_b = ctx.enqueue_create_host_buffer[DType.float32](N)
    var pa = h_a.unsafe_ptr()
    var pb = h_b.unsafe_ptr()
    # lane 0 is the row 39 probe pair, (+0.0, -0.0), and it lives in a
    # BUFFER so the HARDWARE_MAX arm reads it as two runtime values
    pa.unsafe_store(0, _f(UInt32(0x00000000)))
    pb.unsafe_store(0, _f(UInt32(0x80000000)))
    var n_nan = 0
    var n_neg_nan = 0
    var n_sub = 0
    var n_zero_pair = 0
    for i in range(1, N):
        pa.unsafe_store(i, _fmax_a(i))
        pb.unsafe_store(i, _fmax_b(i))
    for i in range(N):
        var ba = _bits(pa.unsafe_load(i))
        var bb = _bits(pb.unsafe_load(i))
        if _is_nan(ba) or _is_nan(bb):
            n_nan += 1
        if _neg_nan(ba) or _neg_nan(bb):
            n_neg_nan += 1
        if (ba & UInt32(0x7F800000)) == UInt32(0) and (ba & UInt32(0x007FFFFF)) != UInt32(0):
            n_sub += 1
        if (ba & UInt32(0x7FFFFFFF)) == UInt32(0) and (bb & UInt32(0x7FFFFFFF)) == UInt32(0):
            n_zero_pair += 1
    print("fixture:", N, "pairs;", n_nan, "carry a NaN,", n_neg_nan,
          "a NEGATIVE-payload NaN,", n_sub, "a subnormal a,",
          n_zero_pair, "are a zero pair")
    if n_neg_nan < 100:
        raise Error(
            "the fixture carries only " + String(n_neg_nan)
            + " negative-payload NaN pairs: NO_NAN_CANON cannot be judged"
        )
    if n_zero_pair < 100:
        raise Error(
            "the fixture carries only " + String(n_zero_pair)
            + " zero pairs: the signed-zero hazard cannot be judged"
        )

    # ---- host answers --------------------------------------------------
    var h_ab = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_sab = ctx.enqueue_create_host_buffer[DType.float32](N)
    var pab = h_ab.unsafe_ptr()
    var psab = h_sab.unsafe_ptr()
    for i in range(N):
        var a = pa.unsafe_load(i)
        var b = pb.unsafe_load(i)
        pab.unsafe_store(i, portable_fmaxf(a, b))
        psab.unsafe_store(i, _sab_fmax(a, b))

    var h_flr = ctx.enqueue_create_host_buffer[DType.float32](NGROUP)
    var h_frl = ctx.enqueue_create_host_buffer[DType.float32](NGROUP)
    var h_ftr = ctx.enqueue_create_host_buffer[DType.float32](NGROUP)
    var pflr = h_flr.unsafe_ptr()
    var pfrl = h_frl.unsafe_ptr()
    var pftr = h_ftr.unsafe_ptr()
    for g in range(NGROUP):
        # `_fmax_a(i)` is the pure function that FILLED `pa`, so the host
        # folds read exactly the numbers the device kernel reads out of
        # `d_a`. Lane 0 is the one cell main overwrote and `_fmax_a(0)` is
        # the same +0.0 it was overwritten with.
        pflr.unsafe_store(g, _sweep_fold_lr(g * GROUP))
        pfrl.unsafe_store(g, _sweep_fold_rl(g * GROUP))
        pftr.unsafe_store(g, _sweep_fold_tree(g * GROUP))

    # ---- the device ----------------------------------------------------
    var d_a = ctx.enqueue_create_buffer[DType.float32](N)
    var d_b = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_buf=d_a, src_ptr=h_a.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_b, src_ptr=h_b.unsafe_ptr())
    var o_ab = ctx.enqueue_create_buffer[DType.float32](N)
    var o_ba = ctx.enqueue_create_buffer[DType.float32](N)
    var o_sab = ctx.enqueue_create_buffer[DType.float32](N)
    var o_flr = ctx.enqueue_create_buffer[DType.float32](NGROUP)
    var o_frl = ctx.enqueue_create_buffer[DType.float32](NGROUP)
    var o_ftr = ctx.enqueue_create_buffer[DType.float32](NGROUP)

    ctx.enqueue_function[fmax_kernel](
        d_a.unsafe_ptr(), d_b.unsafe_ptr(),
        o_ab.unsafe_ptr(), o_ba.unsafe_ptr(), o_sab.unsafe_ptr(),
        Int32(N),
        grid_dim=(N + BLOCK - 1) // BLOCK,
        block_dim=(BLOCK, 1, 1),
    )
    ctx.enqueue_function[fold_kernel](
        d_a.unsafe_ptr(),
        o_flr.unsafe_ptr(), o_frl.unsafe_ptr(), o_ftr.unsafe_ptr(),
        Int32(NGROUP),
        grid_dim=(NGROUP + BLOCK - 1) // BLOCK,
        block_dim=(BLOCK, 1, 1),
    )

    var r_ab = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_ba = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_sab = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_flr = ctx.enqueue_create_host_buffer[DType.float32](NGROUP)
    var r_frl = ctx.enqueue_create_host_buffer[DType.float32](NGROUP)
    var r_ftr = ctx.enqueue_create_host_buffer[DType.float32](NGROUP)
    ctx.enqueue_copy(dst_ptr=r_ab.unsafe_ptr(), src_buf=o_ab)
    ctx.enqueue_copy(dst_ptr=r_ba.unsafe_ptr(), src_buf=o_ba)
    ctx.enqueue_copy(dst_ptr=r_sab.unsafe_ptr(), src_buf=o_sab)
    ctx.enqueue_copy(dst_ptr=r_flr.unsafe_ptr(), src_buf=o_flr)
    ctx.enqueue_copy(dst_ptr=r_frl.unsafe_ptr(), src_buf=o_frl)
    ctx.enqueue_copy(dst_ptr=r_ftr.unsafe_ptr(), src_buf=o_ftr)
    ctx.synchronize()

    var qab = r_ab.unsafe_ptr()
    var qba = r_ba.unsafe_ptr()
    var qsab = r_sab.unsafe_ptr()
    var qflr = r_flr.unsafe_ptr()
    var qfrl = r_frl.unsafe_ptr()
    var qftr = r_ftr.unsafe_ptr()

    var mm = 0
    var mm_sub = 0
    var mm_order = 0
    var fnv = UInt64(0xCBF29CE484222325)
    var shown = 0
    for i in range(N):
        var db = _bits(qab.unsafe_load(i))
        var hb = _bits(pab.unsafe_load(i))
        if _canon(db) != _canon(hb):
            mm += 1
            var ba = _bits(pa.unsafe_load(i))
            var bb = _bits(pb.unsafe_load(i))
            var suba = (ba & UInt32(0x7F800000)) == UInt32(0) and (ba & UInt32(0x007FFFFF)) != UInt32(0)
            var subb = (bb & UInt32(0x7F800000)) == UInt32(0) and (bb & UInt32(0x007FFFFF)) != UInt32(0)
            if suba or subb:
                mm_sub += 1
            if shown < 3:
                print("  mismatch lane", i, "a", hex(ba), "b", hex(bb),
                      "device", hex(db), "host", hex(hb))
                shown += 1
        if _canon(db) != _canon(_bits(qba.unsafe_load(i))):
            mm_order += 1
        fnv = (fnv ^ UInt64(_canon(db))) * UInt64(0x100000001B3)

    var mm_fold = 0
    for g in range(NGROUP):
        var l = _canon(_bits(qflr.unsafe_load(g)))
        var r = _canon(_bits(qfrl.unsafe_load(g)))
        var t = _canon(_bits(qftr.unsafe_load(g)))
        if l != r or l != t:
            mm_fold += 1
        if l != _canon(_bits(pflr.unsafe_load(g))):
            mm_fold += 1
        if r != _canon(_bits(pfrl.unsafe_load(g))):
            mm_fold += 1
        if t != _canon(_bits(pftr.unsafe_load(g))):
            mm_fold += 1
        fnv = (fnv ^ UInt64(l)) * UInt64(0x100000001B3)

    print("fmax device hash =", fnv)
    print("  device/host mismatches:", mm, "( ", mm_sub,
          "of them involving a subnormal operand )")
    print("  device order-dependence: ", mm_order, "lanes")
    print("  device fold disagreements (three orders, and vs the host):",
          mm_fold, "of", NGROUP, "rows")

    if mm != 0:
        raise Error(
            "device bits differ from host bits on " + String(mm)
            + " lanes. DEVIATION 825's claim is stronger here than for the"
            + " other portable gates -- integer ops do not flush and the"
            + " operands are flushed explicitly, so there should be NO"
            + " column-dependent step left, subnormal lanes included"
        )
    if mm_order != 0:
        raise Error(
            "portable_fmaxf is not commutative ON THE DEVICE on "
            + String(mm_order) + " lanes"
        )
    if mm_fold != 0:
        raise Error(
            "the device fold is order-dependent or disagrees with the host"
            + " on " + String(mm_fold) + " rows: softmax's row maximum would"
            + " depend on the tree the scheduler built"
        )
    print("device == host on every lane, both orders, all three folds")

    # EVERY pointer handed on carries an explicit, IDENTICAL origin cast,
    # the spelling `ensemble/checks/sampled_cols_check.mojo` uses.
    check_sabotage_arms(
        pab.unsafe_origin_cast[MutUntrackedOrigin](),
        psab.unsafe_origin_cast[MutUntrackedOrigin](),
        qab.unsafe_origin_cast[MutUntrackedOrigin](),
        qsab.unsafe_origin_cast[MutUntrackedOrigin](),
        pa.unsafe_origin_cast[MutUntrackedOrigin](),
        pb.unsafe_origin_cast[MutUntrackedOrigin](),
    )

    comptime if ANY_SABOTAGE:
        print("SABOTAGED BUILD (" + fmax_sabotage_name()
              + "): the arm behaved as predicted above.")
    else:
        print("all arms OK")

    _ = h_a.unsafe_ptr()
    _ = h_b.unsafe_ptr()
    _ = h_ab.unsafe_ptr()
    _ = h_sab.unsafe_ptr()
    _ = h_flr.unsafe_ptr()
    _ = h_frl.unsafe_ptr()
    _ = h_ftr.unsafe_ptr()
    _ = r_ab.unsafe_ptr()
    _ = r_ba.unsafe_ptr()
    _ = r_sab.unsafe_ptr()
    _ = r_flr.unsafe_ptr()
    _ = r_frl.unsafe_ptr()
    _ = r_ftr.unsafe_ptr()
