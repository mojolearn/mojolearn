# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""cuML `cpp/src/holtwinters/internal/hw_utils.cuh` (v26.08.00).

The constants, the launch-geometry helpers and the three device helpers
(`abs_device`, `bound_device`, `max3`). This file also carries the lane's
sabotage defines (a no-op in every build that does not name them), so every
sabotage has one switchboard.

============ DEVIATION 663 (2026-08-23): `bound_device` IS A COMPARE CHAIN,
============ NOT `fminf(fmaxf(val, 0), 1)` ==================================
WHAT THEIRS DOES (`hw_utils.cuh:63-71`): `fminf(fmaxf(val, .0), 1.)` on the
float instantiation. On NVIDIA `fmaxf(-0.0, +0.0)` is `+0.0` (IEEE maximum)
and `fmaxf(NaN, 0)` is `0` (the non-NaN operand).
WHAT OURS DOES: `if v > 0: r = v else r = +0.0; if r > 1: r = 1`. No
hardware `max`/`min` anywhere: a pure compare chain, the same answer on
every vendor, and it is exactly NVIDIA's answer on the two hazards of
IDENTITY_PATHS row 39: `-0.0 -> +0.0` and `NaN -> +0.0`.
WHY: row 39 measured `max(+0.0, -0.0)` as `-0.0` on Apple (second operand)
and `+0.0` on NVIDIA/AMD; `min` mirrors it. The bounded `alpha`, `beta`,
`gamma` are RECORDED stages (`hw.params`) and feed every downstream bit (a
`-0.0` alpha makes `alpha * x` a `-0.0` where NVIDIA makes `+0.0`), so the
clamp may not be a hardware max. A COMPUTED NaN in the optimizer (their
known instability, cuml#888; DEVIATION 662 closes the one legal-input
route) reaches this clamp and must come out the same bits everywhere.
MEASURED: `hw_check::check_hw_signed_zero_clamp` plants `-0.0` and NaN at
the clamp on the host helper, as the `alpha/beta/gamma` of a direct eval
launch (device and oracle), and at the RECORDED clamp (`hw.params` on the
zero series). TWO sabotage arms, and only one bites:
  CLAMP_GE      the lower test loosened to `val >= lo`, so `-0.0` survives.
                FAILS on Apple at the recorded clamp -- this is DEVIATION
                663's gate.
  HW_MAX_CLAMP  the hardware `min(max(0.0, v), 1.0)`. NULL ON APPLE, and
                the reason is worth keeping: row 39 measured `max(+0.0,
                -0.0)` on two RUNTIME operands, but the clamp's zero is a
                compile-time constant and `llvm.maxnum(0.0, v)` is folded
                to a compare-select returning `+0.0` on the tie, which LLVM
                may do because maxnum's zero-tie answer is unspecified. The
                arm is kept for a vendor that does not fold it, and it has
                still never been built off Apple. The AMD leg of 2026-08-31
                ran the clean gate, not the arms.
That null is exactly why the clamp is a compare chain and not a `max`: the
spelling's answer here depends on whether a constant got folded, which is
not a property anyone should be relying on for a recorded stage.
============================================================================
"""

from std.memory import bitcast
from std.sys.compile import is_defined

# ---------------------------------------------------------------------------
# Sabotage switchboard (all no-ops unless the define is named on the command
# line: `-D MOJOLEARN_HW_SABOTAGE_<NAME>=1`). README's sabotage table.
# ---------------------------------------------------------------------------
#: conv1d's filter sum starts at `block_idx.x % filter_size` and wraps (the
#: order made a function of launch geometry). MUST FAIL device-vs-oracle and
#: launch invariance.
comptime SAB_ROTATE_CONV = is_defined["MOJOLEARN_HW_SABOTAGE_ROTATE_CONV"]()
#: the BFGS step-size `sqrt` through `std.math.sqrt` instead of
#: `identical_sqrt` (row 10: approximate on NVIDIA, correctly rounded on
#: Apple/AMD). MEASURED 2026-08-23: NULL ON APPLE -- the whole gate is
#: byte-identical to the clean build, because on Metal both spellings are
#: the correctly-rounded hardware sqrt. It is the NVIDIA arm of the seam.
#: The AMD leg of 2026-08-31 does NOT discharge it. That leg ran the clean
#: gate, not the sabotage arms, so this arm has still never been built on a
#: non-Apple box. RECORDED, not a failing arm on this box.
comptime SAB_STD_SQRT = is_defined["MOJOLEARN_HW_SABOTAGE_STD_SQRT"]()
#: `ftz` dropped at the eval recurrence's stored intermediates. RECORD.
comptime SAB_NO_FTZ = is_defined["MOJOLEARN_HW_SABOTAGE_NO_FTZ"]()
#: DEVIATION 662 off: a zero search direction takes their `0.866 / sqrt(0)`
#: step. MUST FAIL the zero-series gate (NaN params, canonicalized).
comptime SAB_NO_ZERO_DIR_GUARD = is_defined["MOJOLEARN_HW_SABOTAGE_NO_ZERO_DIR_GUARD"]()
#: DEVIATION 663 off, arm 1: `bound_device` as `min(max(0.0, v), 1.0)`
#: (hardware max, zero FIRST -- the orientation IDENTITY_PATHS row 39 warns
#: about). MEASURED 2026-08-23: NULL ON APPLE. Row 39's `max(+0.0, -0.0) =
#: second operand` was measured on a GPU expression with two RUNTIME
#: operands; here one operand is the compile-time constant `+0.0`, and both
#: the host and Metal fold `maxnum(0.0, v)` to a compare-select whose tie
#: answer is `+0.0`. LLVM is allowed to: `llvm.maxnum(+0, -0)` may return
#: EITHER operand, so the fold is legal and the sabotage never bites. It
#: stays as the arm that would bite on a vendor whose `max` is not folded.
comptime SAB_HW_MAX_CLAMP = is_defined["MOJOLEARN_HW_SABOTAGE_HW_MAX_CLAMP"]()
#: DEVIATION 663 off, arm 2 -- THE ARM THAT BITES. The compare chain's
#: LOWER test loosened from `val > lo` to `val >= lo`, so a `-0.0` compares
#: equal to `+0.0`, takes the value branch and SURVIVES the clamp. One
#: character, and it is the exact mis-spelling a reader would write. MUST
#: FAIL `check_hw_signed_zero_clamp` at the RECORDED clamp.
comptime SAB_CLAMP_GE = is_defined["MOJOLEARN_HW_SABOTAGE_CLAMP_GE"]()
#: the eval's level update with the OTHER product fused (`fma((1-a), lt,
#: a*x)` instead of `fma(a, x, (1-a)*lt)`): a different contraction
#: spelling. MUST FAIL device-vs-oracle (the oracle keeps the pinned one).
comptime SAB_SWAP_FMA = is_defined["MOJOLEARN_HW_SABOTAGE_SWAP_FMA"]()
#: THE OPTIMIZER'S TIE-BREAK, arm 1: the line-search acceptance test
#: loosened from `loss > loss_ref + step * cauchy` to `>=`. On an EXACT tie
#: -- and a tie is not exotic here: a flat or exactly-recovered series has
#: `loss == loss_ref` and `cauchy == 0` -- theirs accepts the step and ours
#: would take another halving, so `step_size`, `nx` and every bit after
#: them move. MUST FAIL.
comptime SAB_LS_TIE = is_defined["MOJOLEARN_HW_SABOTAGE_LS_TIE"]()
#: THE OPTIMIZER'S TIE-BREAK, arm 2: the two stop criteria tested in the
#: OTHER order (`min_error_diff` before `min_param_diff`). Their order
#: (`hw_optim.cuh:524-526`) decides which criterion is REPORTED on an
#: iteration where BOTH fire, and `hw.opt.criterion` is a recorded stage.
#: Pure tie-break: no arithmetic changes, only the label -- and, because
#: the criterion is what the fit returns, the answer a caller reads.
comptime SAB_CRIT_ORDER = is_defined["MOJOLEARN_HW_SABOTAGE_CRIT_ORDER"]()


def hw_sabotage_name() -> String:
    var s = String("")
    comptime if SAB_ROTATE_CONV:
        s += "ROTATE_CONV "
    comptime if SAB_STD_SQRT:
        s += "STD_SQRT "
    comptime if SAB_NO_FTZ:
        s += "NO_FTZ "
    comptime if SAB_NO_ZERO_DIR_GUARD:
        s += "NO_ZERO_DIR_GUARD "
    comptime if SAB_HW_MAX_CLAMP:
        s += "HW_MAX_CLAMP "
    comptime if SAB_CLAMP_GE:
        s += "CLAMP_GE "
    comptime if SAB_SWAP_FMA:
        s += "SWAP_FMA "
    comptime if SAB_LS_TIE:
        s += "LS_TIE "
    comptime if SAB_CRIT_ORDER:
        s += "CRIT_ORDER "
    if s == "":
        return String("none")
    return s


# #define STMP_EPS (1e-6)   -- a double literal converted to Dtype at use
comptime STMP_EPS = Float32(Float64(1e-6))
# #define GOLD 0.38196601125010515...  (parabolic path; UNPORTED, kept for the record)
comptime GOLD = Float32(Float64(0.381966011250105151795413165634361882279690820194237137864551377294739537181097550292792795810608862515245))
# #define PG_EPS 1e-10
comptime PG_EPS = Float32(Float64(1e-10))
comptime MAX_BLOCKS_PER_DIM = 65535

# The seasonal fit path's fixed optimizer launch width (`hw_optim.cuh:862-863`).
comptime HW_OPTIM_TPB = 128


def get_threads_per_block(n: Int, max_threads: Int = 512) -> Int:
    """`GET_THREADS_PER_BLOCK(n, max_threads)`: 32 / 128 / 512 by n. A
    SCHEDULING choice (one thread per series, no cross-thread arithmetic);
    the gates vary it freely."""
    var ret: Int
    if n <= 128:
        ret = 32
    elif n <= 1024:
        ret = 128
    else:
        ret = 512
    return max_threads if ret > max_threads else ret


def get_num_blocks(n: Int, max_threads: Int = 512, max_blocks: Int = MAX_BLOCKS_PER_DIM) -> Int:
    """`GET_NUM_BLOCKS(n, max_threads, max_blocks)`.

    NOT ON THE LAUNCH PATH, AND DELIBERATELY SO. Every kernel launcher in
    this lane computes `(n + tpb - 1) // tpb` directly instead of calling
    this, because THEIR 65535 cap silently under-covers: past the cap the
    grid stops growing and the tail of the batch is never written. That is
    a bug, and not launching with the cap is the fix.

    It is kept because it is their spelling and DERIVATION_MAP.tsv cites it,
    not because anything calls it. It is also UNREACHABLE in effect: the
    cap binds only above `65535 * GET_THREADS_PER_BLOCK(batch_size)`
    series, and `GET_THREADS_PER_BLOCK` returns 512 for any batch past
    1024, so it needs more than 33.5 million series -- at this lane's
    n = 48 that is over 6 GB of input before a single kernel runs. No
    shape this lane will ever run approaches it. NOT_IMPLEMENTED.tsv records it
    as UNPORTED-IN-EFFECT rather than pretending it is gated."""
    var ret = (n - 1) // get_threads_per_block(n, max_threads) + 1
    return max_blocks if ret > max_blocks else ret


@always_inline
def abs_device(val: Float32) -> Float32:
    """`fabsf`: exact, a sign-bit clear."""
    return abs(val)


@always_inline
def bound_device(val: Float32, lo: Float32 = Float32(0.0), hi: Float32 = Float32(1.0)) -> Float32:
    """`fminf(fmaxf(val, min), max)` as DEVIATION 663's compare chain (see
    the file header): `v > lo ? v : lo`, then `r > hi ? hi : r`. `-lo` /
    NaN come out as `lo` (+0.0 at the default), `+inf` as `hi`."""
    comptime if SAB_HW_MAX_CLAMP:
        # the documented sabotage: hardware max with the ZERO FIRST, so on
        # Apple (second-operand tie rule, row 39) a -0.0 val SURVIVES; the
        # val-first spelling is accidentally right here and was inert.
        return min(max(lo, val), hi)
    var r: Float32
    comptime if SAB_CLAMP_GE:
        # the documented sabotage: `>=` lets a -0.0 through the lower test.
        if val >= lo:
            r = val
        else:
            r = lo
    else:
        if val > lo:
            r = val
        else:
            r = lo
    if r > hi:
        r = hi
    return r


@always_inline
def max3(a: Float32, b: Float32, c: Float32) -> Float32:
    """`a > b ? (a > c ? a : c) : (b > c ? b : c)` -- strict compares, so
    on a tie (both zeros, or NaN) the later operand is returned. Its only
    consumers are `min_* > max` compares, which are sign- and
    payload-blind; it is never recorded."""
    if a > b:
        return a if a > c else c
    return b if b > c else c


def hw_float32_inf() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F800000))


def hw_is_finite(v: Float32) -> Bool:
    return v == v and abs(v) != hw_float32_inf()
