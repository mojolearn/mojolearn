"""cuML `cpp/src/holtwinters/internal/hw_utils.cuh` (v26.08.00).

The constants, the launch-geometry helpers and the three device helpers
(`abs_device`, `bound_device`, `max3`). This file also carries the lane's
sabotage defines (a no-op in every build that does not name them), so every
sabotage has one switchboard.

============ DEVIATION 663 (2026-08-23): `bound_device` IS A COMPARE CHAIN,
============ NOT `fminf(fmaxf(val, 0), 1)` ==================================
WHAT THEIRS DOES (`hw_utils.cuh:58-66`): `fminf(fmaxf(val, .0), 1.)` on the
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
the clamp on the host helper and as the `alpha/beta/gamma` of a direct eval
launch (device and oracle); the sabotage `MOJOLEARN_HW_SABOTAGE_HW_MAX_CLAMP`
spells it `min(max(0.0, v), 1.0)` (the hardware max, zero first) and FAILS
on Apple (`-0.0` survives). README carries the lines.
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
#: Apple/AMD). REPORT/FAIL as measured.
comptime SAB_STD_SQRT = is_defined["MOJOLEARN_HW_SABOTAGE_STD_SQRT"]()
#: `ftz` dropped at the eval recurrence's stored intermediates. RECORD.
comptime SAB_NO_FTZ = is_defined["MOJOLEARN_HW_SABOTAGE_NO_FTZ"]()
#: DEVIATION 662 off: a zero search direction takes their `0.866 / sqrt(0)`
#: step. MUST FAIL the zero-series gate (NaN params, canonicalized).
comptime SAB_NO_ZERO_DIR_GUARD = is_defined["MOJOLEARN_HW_SABOTAGE_NO_ZERO_DIR_GUARD"]()
#: DEVIATION 663 off: `bound_device` as `min(max(0.0, v), 1.0)` (hardware
#: max, zero FIRST). MUST FAIL the -0.0 clamp gate on Apple.
comptime SAB_HW_MAX_CLAMP = is_defined["MOJOLEARN_HW_SABOTAGE_HW_MAX_CLAMP"]()
#: the eval's level update with the OTHER product fused (`fma((1-a), lt,
#: a*x)` instead of `fma(a, x, (1-a)*lt)`): a different contraction
#: spelling. MUST FAIL device-vs-oracle (the oracle keeps the pinned one).
comptime SAB_SWAP_FMA = is_defined["MOJOLEARN_HW_SABOTAGE_SWAP_FMA"]()


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
    comptime if SAB_SWAP_FMA:
        s += "SWAP_FMA "
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

# The seasonal fit path's fixed optimizer launch width (`hw_optim.cuh:867-868`).
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
    """`GET_NUM_BLOCKS(n, max_threads, max_blocks)`."""
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
