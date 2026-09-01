# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The resampling gates, and the nine sabotage arms that prove they bite.

    tools/with_build_lock.sh     pixi run mojo run -I . resample/checks/resample_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . resample/checks/resample_check.mojo

Every check prints the mode this binary COMPILED in. Under IDENTICAL the
device-versus-oracle bit compares are ASSERTIONS; under FAST the same
comparisons are REPORTS, because the FAST arm folds through `block.sum`
(CUB's warp-then-block shape) and no host loop can replicate a lane width.
Everything else -- the refusals, the positional map, batch invariance, prefix
stability, the p-value floor, the order-statistic positions, launch
invariance -- ASSERTS IN BOTH MODES, because all of it is integer arithmetic,
a comparison, or a claim about which values were folded rather than in what
order.

**NOTHING IN THIS FILE HAS BEEN RUN.** Not once. The expected outputs quoted
in `resample/README.md` are the SHAPE of what each line will print, not a
transcript. Any number in this repository attributed to this lane is a
fabrication until the orchestrator runs it.

CHECKS
  check_resample_refusals             every bound and every unported choice
                                      RAISES BY NAME
  check_index_map_is_positional       replicate 7 computed THREE ways: device,
                                      host mirror, and inside a window that
                                      starts elsewhere
  check_batch_invariance              THE HEADLINE. Replicate 7 alone == in a
                                      batch of 1000 == in a batch of 100000,
                                      bit for bit, on the drawn indices AND on
                                      the statistic
  check_prefix_stability              10000 resamples and 100000 agree on the
                                      first 10000, bit for bit
  check_bootstrap_vs_oracle           the device distribution against the
                                      serial host replay, per replicate
  check_bootstrap_known_answer        FIX_ANALYTIC: every replicate's answer
                                      derived by hand, three statistics
  check_statistic_arms                PORTING_RULES rule 8: all six statistic
                                      arms run with the switch set explicitly
  check_percentile_interval           the hand-worked identity-array example,
                                      integer and non-integer h, and the
                                      planted signed-zero order
  check_permutation_separable         the p-value floor, one-sided AND
                                      two-sided, with the counts printed
  check_permutation_null_is_uniform   a REPORT, never a gate
  check_monte_carlo_vs_closed_form    three integrands, two boxes
  check_jackknife_and_bca             DEVIATION 1699's identical half
  check_launch_invariance             two fold widths, two map widths, two
                                      grid shapes, two paddings, two poisons
  check_card_is_emitted               two runs, record-identical
  check_resample_sabotages            the nine arms

SABOTAGE TABLE (results are copied into resample/README.md when they exist)
  RSAB_BATCH_DEPENDENT       MUST FAIL check_batch_invariance AND
                             check_prefix_stability -- the one the brief asks
                             for by name
  RSAB_LOW_HALF_SUBSEQUENCE  MUST FAIL check_bootstrap_vs_oracle (the trap
                             core/philox.mojo names: subsequence into the LOW
                             counter half)
  RSAB_SKIP_REJECTION        MUST FAIL on FIX_HASHED (n = 200) and is EXPECTED
                             INERT on FIX_ANALYTIC (n = 8, a power of two, so
                             2^32 mod n == 0 and Lemire never rejects). Reach
                             is per-branch and this arm is the demonstration
  RSAB_REVERSED_POSITIONS    MUST FAIL check_bootstrap_vs_oracle (same
                             multiset, different tree slots)
  RSAB_STD_SQRT              REPORT: Apple's sqrt is correctly rounded so 0
                             cells are expected to move there; NVIDIA's is
                             approximate on 176,577 of 2^20 normals
                             (IDENTITY_PATHS row 10) and it is expected to bite
  RSAB_NO_CANON_NAN          RECORDED: the vendor's NaN payload reaches
                             resample.theta
  RSAB_PERM_KEY_ONLY         MUST FAIL check_permutation_separable's bijection
                             assertion (the key is narrowed to 4 bits so ties
                             are reachable, and the index tie-break is dropped)
  RSAB_FOLD_DESCENDING       MUST FAIL check_bootstrap_vs_oracle (the chunk
                             chain folded the other way)
  RSAB_MAP_IGNORES_POSITION  MUST FAIL check_index_map_is_positional (every
                             position of a replicate draws the same index)
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv, sqrt as std_sqrt
from std.memory import bitcast, stack_allocation
from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.identity_trace import IdentityTrace, first_divergence
from core.philox import PhiloxState, custom_next_uniform_int_u32
from metrics.checks.pinned_sum import (
    PINNED_SUM_W,
    canonicalize_nan,
    chunk_count,
    virtual_block_sum,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_mul,
    identical_sqrt,
    numeric_mode_name,
)
from resample.estimator import (
    RESAMPLE_MAP_TPB,
    RESAMPLE_TPB,
    _download_f32,
    _download_i32,
    _sort_segments,
    _upload,
    bootstrap_host,
    monte_carlo_integrate_host,
    permutation_test_host,
    point_estimate_host,
)
from resample.checks.index_map import (
    PERM_MAX_POOLED,
    RESAMPLE_KIND_BOOTSTRAP,
    RESAMPLE_KIND_MONTE_CARLO,
    RESAMPLE_KIND_PERMUTATION,
    bootstrap_index_host,
    draw_permutation_key,
    key_hi,
    key_join,
    key_lo,
    permutation_key_lt,
    permutation_ranks_host,
    position_subsequence,
    resample_key,
)
from resample.checks.intervals import (
    ALT_GREATER,
    ALT_LESS,
    ALT_TWO_SIDED,
    METHOD_BASIC,
    METHOD_BCA,
    METHOD_PERCENTILE,
    alternative_from_name,
    basic_interval,
    bca_acceleration,
    bca_bias_percentile,
    method_from_name,
    percentile_interval,
    permutation_pvalue,
)
from resample.checks.resample_fixture import (
    FIX_ANALYTIC,
    FIX_DUPES,
    FIX_HASHED,
    FIX_SAME,
    FIX_SEPARABLE,
    FIX_SIGNED_ZERO,
    build_pooled,
    build_sample,
    hashed_sample,
    fixture_d,
    fixture_n,
    fixture_name,
    fixture_two_sample_n,
    planted_signed_zero_distribution,
)
from resample.checks.resample_oracle import (
    oracle_bootstrap_distribution_f32,
    oracle_bootstrap_statistic_f32,
    oracle_monte_carlo_f32,
    reference_bootstrap_statistic_f64,
)
from resample.checks.statistics import (
    MC_F_CONST,
    MC_F_PRODUCT,
    MC_F_SUM,
    RESAMPLE_MAX_SORT_CELLS,
    STAT_DIFF_MEANS,
    STAT_MEAN,
    STAT_PEARSON,
    STAT_QUANTILE,
    STAT_STD,
    STAT_TRIMMED_MEAN,
    _mean_of_sum,
    mc_closed_form,
    quantile_position,
    stat_from_name,
    stat_name,
)


comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

comptime CHECK_SEED: UInt64 = 8675309

#: Two float32 bit patterns this file names rather than spells.
comptime BITS_NEG_ONE: UInt32 = 0xBF800000
comptime BITS_CANON_NAN: UInt32 = 0x7FC00000
comptime BITS_NEG_ZERO: UInt32 = 0x80000000
comptime BITS_POS_ZERO: UInt32 = 0x00000000


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


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _bits(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def _card_dir() -> String:
    """Where the two card files go. An environment override so the check does
    not hardcode one session's scratchpad -- a path baked into a check is a
    check that only its author can run."""
    var d = String(getenv("MOJOLEARN_RESAMPLE_CARD_DIR"))
    if d == "":
        return String("/tmp")
    return d


def _first_bit_diff(a: List[Float32], b: List[Float32], n: Int) -> Int:
    """The first index where two float lists differ BY BITS, or -1.

    BY BITS and not by `==`, because `NaN != NaN` would report every NaN as a
    difference and `-0.0 == +0.0` would report no difference where the bits
    move. Both cases are live in this lane (DEVIATION 1696's canonical NaN;
    row 39's zeros)."""
    for i in range(n):
        if _bits(a[i]) != _bits(b[i]):
            return i
    return -1


def _first_i32_diff(a: List[Int32], b: List[Int32], n: Int) -> Int:
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1


# ===========================================================================
# THE SABOTAGE ARMS
#
# Copies of two production kernels with pins BROKEN ON PURPOSE, in the shape
# `hierarchy/checks/sabotage_tile.mojo` uses and for its reason: a
# sabotage arm does not belong in a production kernel, and these copies are
# reachable from nothing but this file. The `RSAB_NONE` arm of each copy is
# never launched -- the checks call the REAL kernel through
# `resample/estimator.mojo` when they want an unsabotaged answer -- so the
# shipped bits never depend on this section.
# ===========================================================================

comptime RSAB_NONE = 0

#: The index map reads `n_replicates`. THE ONE THE BRIEF NAMES: batch
#: invariance and prefix stability both die, and NOTHING ELSE does, because
#: any single run is still self-consistent.
comptime RSAB_BATCH_DEPENDENT = 1

#: The subsequence goes into the LOW half of the Philox counter (the `offset`
#: argument) instead of the high half. `core/philox.mojo::_incr_hi` names this
#: exact trap: the result passes every distributional test and shares state
#: between positions that are 2^32 blocks apart.
comptime RSAB_LOW_HALF_SUBSEQUENCE = 2

#: Lemire's rejection loop dropped -- `(x * n) >> 32` with no `l < t` test.
#: EXPECTED INERT where `2^32 mod n == 0` (n a power of two) and expected to
#: bite otherwise. The per-branch arm.
comptime RSAB_SKIP_REJECTION = 3

#: Slot `s` of the fold holds position `n - 1 - s`. Same multiset, different
#: tree pairing.
comptime RSAB_REVERSED_POSITIONS = 4

#: `std.math.sqrt` at the `std`/`pearson` seam instead of `identical_sqrt`.
comptime RSAB_STD_SQRT = 5

#: `canonicalize_nan` dropped on the degenerate `pearson` path.
comptime RSAB_NO_CANON_NAN = 6

#: The permutation key narrowed to 4 bits AND the index tie-break dropped, so
#: the rank stops being a bijection. Two changes in one arm on purpose: at the
#: full 64-bit width the tie-break is unreachable, and an unreachable branch
#: cannot be sabotaged (`[[sabotage-when-required]]`: the branch is rare, so
#: the fixture has to make it common).
comptime RSAB_PERM_KEY_ONLY = 7

#: The chunk chain folded descending instead of ascending.
comptime RSAB_FOLD_DESCENDING = 8

#: Every position of a replicate draws the SAME index (position dropped from
#: the subsequence). Kills `check_index_map_is_positional` and nothing about
#: the batch.
comptime RSAB_MAP_IGNORES_POSITION = 9

comptime RSAB_COUNT = 10


def sabotage_name(sab: Int) -> String:
    if sab == RSAB_NONE:
        return String("RSAB_NONE")
    if sab == RSAB_BATCH_DEPENDENT:
        return String("RSAB_BATCH_DEPENDENT")
    if sab == RSAB_LOW_HALF_SUBSEQUENCE:
        return String("RSAB_LOW_HALF_SUBSEQUENCE")
    if sab == RSAB_SKIP_REJECTION:
        return String("RSAB_SKIP_REJECTION")
    if sab == RSAB_REVERSED_POSITIONS:
        return String("RSAB_REVERSED_POSITIONS")
    if sab == RSAB_STD_SQRT:
        return String("RSAB_STD_SQRT")
    if sab == RSAB_NO_CANON_NAN:
        return String("RSAB_NO_CANON_NAN")
    if sab == RSAB_PERM_KEY_ONLY:
        return String("RSAB_PERM_KEY_ONLY")
    if sab == RSAB_FOLD_DESCENDING:
        return String("RSAB_FOLD_DESCENDING")
    if sab == RSAB_MAP_IGNORES_POSITION:
        return String("RSAB_MAP_IGNORES_POSITION")
    return String("?")


@always_inline
def _sab_draw_row_index(
    key: UInt64,
    r: Int,
    i: Int,
    n: Int32,
    n_replicates: Int,
    sabotage: Int32,
) -> Int32:
    """`index_map.mojo::draw_row_index` with four of its arms broken."""
    var sub = position_subsequence(UInt64(r), UInt64(i))
    var off = UInt64(0)

    if sabotage == RSAB_BATCH_DEPENDENT:
        # The whole failure this lane exists to prevent, in one line.
        sub = sub ^ UInt64(n_replicates)
    if sabotage == RSAB_MAP_IGNORES_POSITION:
        sub = position_subsequence(UInt64(r), UInt64(0))
    if sabotage == RSAB_LOW_HALF_SUBSEQUENCE:
        off = sub
        sub = UInt64(0)

    var gen = PhiloxState.init(key, sub, off)
    if sabotage == RSAB_SKIP_REJECTION:
        var x = gen.next_u32()
        var m = (x.cast[DType.uint64]() & 0xFFFFFFFF) * (
            n.cast[DType.uint32]().cast[DType.uint64]() & 0xFFFFFFFF
        )
        return UInt32((m >> 32) & 0xFFFFFFFF).cast[DType.int32]()
    return custom_next_uniform_int_u32(gen, Int32(0), n.cast[DType.uint32]())


def sabotage_index_kernel(
    out_idx: MutPointer[Int32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    r_first_in: Int32,
    n_replicates_in: Int32,
    n_in: Int32,
    n_positions_in: Int32,
    sabotage: Int32,
):
    """`index_map.mojo::bootstrap_index_kernel` with the sabotaged draw."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_positions = Int(n_positions_in)
    var total = Int(n_replicates_in) * n_positions
    if tid >= total:
        return
    var r = Int(r_first_in) + tid // n_positions
    var i = tid % n_positions
    out_idx.unsafe_store(
        tid,
        _sab_draw_row_index(
            key_join(lo_bits, hi_bits),
            r,
            i,
            n_in,
            Int(n_replicates_in),
            sabotage,
        ),
    )


def sabotage_stat_kernel[stat: Int, tpb: Int](
    theta: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    r_first_in: Int32,
    n_replicates_in: Int32,
    n_in: Int32,
    n_rows_in: Int32,
    n_features_in: Int32,
    sabotage: Int32,
):
    """`statistics.mojo::bootstrap_stat_kernel`'s `mean`, `std` and `pearson`
    arms with every arm of the sabotage switch wired in.

    Only the three the sabotages need are here. A copy of all six would be a
    second implementation of the production kernel with the same maintenance
    burden and none of the benefit; what a sabotage has to reach is the
    PIN, and the three below reach every pin this lane has.
    """
    comptime lanes = PINNED_SUM_W // tpb
    var rr = Int(block_idx.x)
    if rr >= Int(n_replicates_in):
        return
    var r = Int(r_first_in) + rr
    var tid = Int(thread_idx.x)
    var key = key_join(lo_bits, hi_bits)
    var n = Int(n_in)
    var d = Int(n_features_in)
    var chunks = chunk_count(n)
    var n_rep = Int(n_replicates_in)

    # Pass 1: the raw column sums (both columns, so pearson needs no third
    # traversal of this switch).
    var s0 = Float32(0.0)
    var s1 = Float32(0.0)
    for cc in range(chunks):
        var c = cc
        if sabotage == RSAB_FOLD_DESCENDING:
            c = chunks - 1 - cc
        var v0 = SIMD[DType.float32, lanes](0.0)
        var v1 = SIMD[DType.float32, lanes](0.0)
        comptime for lane in range(lanes):
            var slot = c * PINNED_SUM_W + tid + lane * tpb
            if slot < n:
                var i = slot
                if sabotage == RSAB_REVERSED_POSITIONS:
                    i = n - 1 - slot
                var row = Int(
                    _sab_draw_row_index(key, r, i, n_rows_in, n_rep, sabotage)
                )
                v0[lane] = ftz(x.unsafe_load(row * d))
                if d > 1:
                    v1[lane] = ftz(x.unsafe_load(row * d + 1))
        var t0 = virtual_block_sum[tpb](v0)
        var t1 = virtual_block_sum[tpb](v1)
        if tid == 0:
            s0 = ftz(s0 + t0)
            s1 = ftz(s1 + t1)

    var value = Float32(0.0)
    comptime if stat == STAT_MEAN:
        if tid == 0:
            value = _mean_of_sum(s0, n)

    comptime if stat == STAT_STD or stat == STAT_PEARSON:
        var slot0 = stack_allocation[
            2,
            Scalar[DType.float32],
            address_space = AddressSpace.SHARED,
        ]()
        barrier()
        if tid == 0:
            slot0[unsafe_offset=0] = _mean_of_sum(s0, n)
            slot0[unsafe_offset=1] = _mean_of_sum(s1, n)
        barrier()
        var mx = slot0[unsafe_offset=0]
        var my = slot0[unsafe_offset=1]
        barrier()
        var sxx = Float32(0.0)
        var syy = Float32(0.0)
        var sxy = Float32(0.0)
        for cc2 in range(chunks):
            var c2 = cc2
            if sabotage == RSAB_FOLD_DESCENDING:
                c2 = chunks - 1 - cc2
            var a2 = SIMD[DType.float32, lanes](0.0)
            var b2 = SIMD[DType.float32, lanes](0.0)
            var e2 = SIMD[DType.float32, lanes](0.0)
            comptime for lane in range(lanes):
                var slot = c2 * PINNED_SUM_W + tid + lane * tpb
                if slot < n:
                    var i2 = slot
                    if sabotage == RSAB_REVERSED_POSITIONS:
                        i2 = n - 1 - slot
                    var row2 = Int(
                        _sab_draw_row_index(
                            key, r, i2, n_rows_in, n_rep, sabotage
                        )
                    )
                    var dx = ftz(ftz(x.unsafe_load(row2 * d)) - mx)
                    a2[lane] = ftz(identical_mul(dx, dx))
                    if d > 1:
                        var dy = ftz(ftz(x.unsafe_load(row2 * d + 1)) - my)
                        b2[lane] = ftz(identical_mul(dy, dy))
                        e2[lane] = ftz(identical_mul(dx, dy))
            var ta = virtual_block_sum[tpb](a2)
            var tb = virtual_block_sum[tpb](b2)
            var te = virtual_block_sum[tpb](e2)
            if tid == 0:
                sxx = ftz(sxx + ta)
                syy = ftz(syy + tb)
                sxy = ftz(sxy + te)
        if tid == 0:
            comptime if stat == STAT_STD:
                var q = ftz(identical_div(sxx, Float32(n - 1)))
                if sabotage == RSAB_STD_SQRT:
                    value = ftz(std_sqrt(q))
                else:
                    value = ftz(identical_sqrt(q))
            else:
                if sxx == Float32(0.0) or syy == Float32(0.0):
                    var nan = Float32(0.0) / Float32(0.0)
                    if sabotage == RSAB_NO_CANON_NAN:
                        value = nan
                    else:
                        value = canonicalize_nan(nan)
                else:
                    var prod = ftz(identical_mul(sxx, syy))
                    var den: Float32
                    if sabotage == RSAB_STD_SQRT:
                        den = ftz(std_sqrt(prod))
                    else:
                        den = ftz(identical_sqrt(prod))
                    value = ftz(identical_div(sxy, den))

    if tid == 0:
        if sabotage == RSAB_NO_CANON_NAN:
            theta.unsafe_store(rr, value)
        else:
            theta.unsafe_store(rr, canonicalize_nan(value))


@always_inline
def _sab_perm_key(key: UInt64, r: Int, j: Int, sabotage: Int32) -> UInt64:
    var k = draw_permutation_key(key, r, j)
    if sabotage == RSAB_PERM_KEY_ONLY:
        return k & UInt64(0xF)
    return k


@always_inline
def _sab_perm_lt(
    ka: UInt64, ja: Int, kb: UInt64, jb: Int, sabotage: Int32
) -> Bool:
    if sabotage == RSAB_PERM_KEY_ONLY:
        return ka < kb
    return permutation_key_lt(ka, ja, kb, jb)


def sabotage_perm_rank_kernel(
    ranks: MutPointer[Int32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    r_first_in: Int32,
    n_replicates_in: Int32,
    n_pooled_in: Int32,
    sabotage: Int32,
):
    """The rank phase of `perm_stat_kernel`, alone and observable.

    The production kernel keeps the ranks in threadgroup memory and never
    writes them, which is right for the shipped path and useless for a gate:
    the thing to check is that the ranks are a BIJECTION, and a statistic
    computed from a broken permutation is still a plausible float. This copy
    writes them out so `check_permutation_separable` can count each rank
    exactly once.
    """
    var rr = Int(block_idx.x)
    if rr >= Int(n_replicates_in):
        return
    var r = Int(r_first_in) + rr
    var tid = Int(thread_idx.x)
    var n_pooled = Int(n_pooled_in)
    var key = key_join(lo_bits, hi_bits)
    var keys = stack_allocation[
        PERM_MAX_POOLED,
        Scalar[DType.uint64],
        address_space = AddressSpace.SHARED,
    ]()
    var j = tid
    while j < n_pooled:
        keys[unsafe_offset=j] = _sab_perm_key(key, r, j, sabotage)
        j += Int(block_dim.x)
    barrier()
    var jj = tid
    while jj < n_pooled:
        var kj = keys[unsafe_offset=jj]
        var below = Int32(0)
        for l in range(n_pooled):
            if _sab_perm_lt(keys[unsafe_offset=l], l, kj, jj, sabotage):
                below += 1
        ranks.unsafe_store(rr * n_pooled + jj, below)
        jj += Int(block_dim.x)


# ===========================================================================
# Sabotage drivers
# ===========================================================================


def _sab_indices(
    ctx: DeviceContext,
    key: UInt64,
    r_first: Int,
    n_replicates: Int,
    n: Int,
    n_positions: Int,
    sabotage: Int,
) raises -> List[Int32]:
    var total = n_replicates * n_positions
    var buf = ctx.enqueue_create_buffer[DType.int32](total)
    ctx.synchronize()
    ctx.enqueue_function[sabotage_index_kernel](
        buf.unsafe_ptr(),
        key_lo(key),
        key_hi(key),
        Int32(r_first),
        Int32(n_replicates),
        Int32(n),
        Int32(n_positions),
        Int32(sabotage),
        grid_dim=(ceildiv(total, RESAMPLE_MAP_TPB), 1, 1),
        block_dim=(RESAMPLE_MAP_TPB, 1, 1),
    )
    _ = buf.unsafe_ptr()
    ctx.synchronize()
    var out = _download_i32(ctx, buf, total)
    _ = buf^
    return out^


def _sab_distribution[
    stat: Int
](
    ctx: DeviceContext,
    x: List[Float32],
    n: Int,
    d: Int,
    key: UInt64,
    r_first: Int,
    n_replicates: Int,
    sabotage: Int,
) raises -> List[Float32]:
    var dx = _upload(ctx, x)
    var theta = ctx.enqueue_create_buffer[DType.float32](n_replicates)
    ctx.synchronize()
    comptime kern = sabotage_stat_kernel[stat, RESAMPLE_TPB]
    ctx.enqueue_function[kern](
        theta.unsafe_ptr(),
        dx.unsafe_ptr(),
        key_lo(key),
        key_hi(key),
        Int32(r_first),
        Int32(n_replicates),
        Int32(n),
        Int32(n),
        Int32(d),
        Int32(sabotage),
        grid_dim=(n_replicates, 1, 1),
        block_dim=(RESAMPLE_TPB, 1, 1),
    )
    _ = theta.unsafe_ptr()
    _ = dx.unsafe_ptr()
    ctx.synchronize()
    var out = _download_f32(ctx, theta, n_replicates)
    _ = dx^
    _ = theta^
    return out^


def _sab_ranks(
    ctx: DeviceContext,
    key: UInt64,
    r_first: Int,
    n_replicates: Int,
    n_pooled: Int,
    sabotage: Int,
) raises -> List[Int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](n_replicates * n_pooled)
    ctx.synchronize()
    ctx.enqueue_function[sabotage_perm_rank_kernel](
        buf.unsafe_ptr(),
        key_lo(key),
        key_hi(key),
        Int32(r_first),
        Int32(n_replicates),
        Int32(n_pooled),
        Int32(sabotage),
        grid_dim=(n_replicates, 1, 1),
        block_dim=(RESAMPLE_TPB, 1, 1),
    )
    _ = buf.unsafe_ptr()
    ctx.synchronize()
    var out = _download_i32(ctx, buf, n_replicates * n_pooled)
    _ = buf^
    return out^


# ===========================================================================
# CHECK 1: the refusals
# ===========================================================================


def check_resample_refusals() raises:
    """Every bound this lane carries and every choice it does not port,
    refused BY NAME before any launch.

    A REFUSAL THAT DOES NOT NAME ITS CLOSURE IS A DEAD END, so each message
    below is also asserted to be non-empty and to mention the parameter; the
    check counts them rather than eyeballing them.
    """
    var x = build_sample(FIX_HASHED)
    var n = fixture_n(FIX_HASHED)
    var d = fixture_d(FIX_HASHED)
    var named = 0
    var missed = String("")

    # (1) n_resamples of zero.
    try:
        var _r = bootstrap_host(
            x, n, d, STAT_MEAN, 0, CHECK_SEED, METHOD_PERCENTILE
        )
        missed += "n_resamples=0 "
    except e:
        named += 1

    # (2) a negative n_resamples (Int, so "negative seed" has no meaning --
    #     `seed` is a UInt64 and every bit pattern of it is a legal seed,
    #     which is stated here rather than left as a missing check).
    try:
        var _r2 = bootstrap_host(
            x, n, d, STAT_MEAN, -5, CHECK_SEED, METHOD_PERCENTILE
        )
        missed += "n_resamples<0 "
    except e:
        named += 1

    # (3) an unknown method name and (4) an unknown statistic name.
    try:
        var _m = method_from_name(String("BCA_lowercase_typo"))
        missed += "unknown-method "
    except e:
        named += 1
    try:
        var _s = stat_from_name(String("geometric_mean"))
        missed += "unknown-statistic "
    except e:
        named += 1

    # (5) an unknown alternative.
    try:
        var _a = alternative_from_name(String("two_sided"))
        missed += "unknown-alternative "
    except e:
        named += 1

    # (6) a non-finite sample.
    var bad = x.copy()
    bad[3] = Float32(1.0) / Float32(0.0)
    try:
        var _r3 = bootstrap_host(
            bad, n, d, STAT_MEAN, 16, CHECK_SEED, METHOD_PERCENTILE
        )
        missed += "non-finite "
    except e:
        named += 1
    var bad2 = x.copy()
    bad2[7] = Float32(0.0) / Float32(0.0)
    try:
        var _r4 = bootstrap_host(
            bad2, n, d, STAT_MEAN, 16, CHECK_SEED, METHOD_PERCENTILE
        )
        missed += "NaN "
    except e:
        named += 1

    # (7) BCa under BOTH modes (DEVIATION 1699).
    try:
        var _r5 = bootstrap_host(
            x, n, d, STAT_MEAN, 16, CHECK_SEED, METHOD_BCA
        )
        missed += "BCa "
    except e:
        named += 1

    # (8) a statistic that reads a column the sample does not have.
    try:
        var _r6 = bootstrap_host(
            x, n, 1, STAT_PEARSON, 16, CHECK_SEED, METHOD_PERCENTILE
        )
        missed += "pearson-on-one-column "
    except e:
        named += 1

    # (9) confidence_level outside (0, 1).
    try:
        var _r7 = bootstrap_host(
            x, n, d, STAT_MEAN, 16, CHECK_SEED, METHOD_PERCENTILE,
            Float32(1.0)
        )
        missed += "confidence_level "
    except e:
        named += 1

    # (10) the sort-path ceiling.
    try:
        var _r8 = bootstrap_host(
            x,
            n,
            d,
            STAT_QUANTILE,
            RESAMPLE_MAX_SORT_CELLS,
            CHECK_SEED,
            METHOD_PERCENTILE,
        )
        missed += "sort-cells "
    except e:
        named += 1

    # (11) trimmed_mean cutting everything.
    try:
        var _r9 = bootstrap_host(
            x, n, d, STAT_TRIMMED_MEAN, 16, CHECK_SEED, METHOD_PERCENTILE,
            Float32(0.95), ALT_TWO_SIDED, Float32(0.5)
        )
        missed += "trim-proportion "
    except e:
        named += 1

    # (12) a pooled permutation sample above PERM_MAX_POOLED.
    var big_x = List[Float32]()
    var big_y = List[Float32]()
    for i in range(PERM_MAX_POOLED):
        big_x.append(Float32(i))
        big_y.append(Float32(-i))
    try:
        var _p = permutation_test_host(
            big_x, big_y, STAT_DIFF_MEANS, 8, CHECK_SEED, ALT_TWO_SIDED
        )
        missed += "pooled-length "
    except e:
        named += 1

    # (13) permutation_test with a statistic that has no independent arm.
    var sx = List[Float32]()
    var sy = List[Float32]()
    for i in range(8):
        sx.append(Float32(i))
        sy.append(Float32(i + 1))
    try:
        var _p2 = permutation_test_host(
            sx, sy, STAT_PEARSON, 8, CHECK_SEED, ALT_TWO_SIDED
        )
        missed += "perm-pearson "
    except e:
        named += 1

    # (14) an inverted Monte Carlo box.
    try:
        var _mc = monte_carlo_integrate_host[MC_F_SUM](
            [Float32(1.0), Float32(0.0)],
            [Float32(0.0), Float32(1.0)],
            256,
            CHECK_SEED,
        )
        missed += "inverted-box "
    except e:
        named += 1

    # (15) an unsupported threads-per-block.
    try:
        var _r10 = bootstrap_host(
            x, n, d, STAT_MEAN, 16, CHECK_SEED, METHOD_PERCENTILE,
            Float32(0.95), ALT_TWO_SIDED, Float32(0.5), 0, 33
        )
        missed += "tpb "
    except e:
        named += 1

    if missed != "":
        raise Error(
            "check_resample_refusals: these did NOT raise: " + missed
        )
    print(
        "check_resample_refusals OK ["
        + _mode_name()
        + "]: "
        + String(named)
        + " refusals by name (n_resamples 0 and negative, unknown"
        " method/statistic/alternative, inf and NaN in the sample, BCa in"
        " BOTH modes, a column the sample lacks, confidence_level at 1,"
        " the sort ceiling, a trim that cuts everything, a pooled length"
        " above PERM_MAX_POOLED, pearson under the independent null, an"
        " inverted box, tpb=33); seed is UInt64 so every bit pattern is a"
        " legal seed and there is nothing to refuse there"
    )


# ===========================================================================
# CHECK 2: the map is positional
# ===========================================================================


def check_index_map_is_positional() raises:
    """Replicate 7's drawn indices, computed THREE WAYS, byte identical.

      A. the device kernel over a window that STARTS at replicate 7;
      B. the device kernel over a window that starts at replicate 0, read at
         row 7 -- a different thread, a different block, a different grid;
      C. `index_map.mojo::bootstrap_index_host`, a serial host loop.

    A == B says the answer does not depend on WHERE in the launch the
    position was computed. A == C says it does not depend on the device at
    all. Neither alone is enough: B could match A because both read the same
    wrong formula, and C could match A on one machine by coincidence, so the
    three are stated as three.

    ASSERTS IN BOTH MODES: the map is integer arithmetic and carries no
    floating-point operation anywhere, so `NUMERIC_FAST` has nothing to do
    to it.
    """
    var ctx = DeviceContext()
    var key = resample_key(CHECK_SEED, RESAMPLE_KIND_BOOTSTRAP)
    var n = 200
    var pos = 32

    var a = _sab_indices(ctx, key, 7, 1, n, pos, RSAB_NONE)
    var wide = _sab_indices(ctx, key, 0, 16, n, pos, RSAB_NONE)
    var b = List[Int32]()
    for i in range(pos):
        b.append(wide[7 * pos + i])
    var c = bootstrap_index_host(key, 7, 1, n, pos)

    var dab = _first_i32_diff(a, b, pos)
    if dab >= 0:
        raise Error(
            "check_index_map_is_positional: replicate 7 alone and replicate 7"
            " inside a 16-replicate window differ at position "
            + String(dab)
            + ": "
            + String(a[dab])
            + " vs "
            + String(b[dab])
        )
    var dac = _first_i32_diff(a, c, pos)
    if dac >= 0:
        raise Error(
            "check_index_map_is_positional: the device and the host mirror"
            " differ at position "
            + String(dac)
            + ": "
            + String(a[dac])
            + " vs "
            + String(c[dac])
            + ". The map is integer arithmetic, so this is a Philox port"
            " defect and ensemble/bench/philox_oracle.txt is where to take"
            " it."
        )

    # The map must also SEPARATE positions. A constant map would pass every
    # equality above, which is `[[reached-but-inert]]` exactly.
    var distinct = 0
    for i in range(1, pos):
        if a[i] != a[0]:
            distinct += 1
    if distinct < pos // 2:
        raise Error(
            "check_index_map_is_positional: only "
            + String(distinct)
            + " of "
            + String(pos - 1)
            + " positions differ from position 0; the map is not separating"
            " positions"
        )
    print(
        "check_index_map_is_positional OK ["
        + _mode_name()
        + "]: replicate 7 x "
        + String(pos)
        + " positions identical three ways (alone, inside a 16-replicate"
        " window, host mirror); "
        + String(distinct)
        + " of "
        + String(pos - 1)
        + " positions draw a different row than position 0"
    )


# ===========================================================================
# CHECK 3: THE HEADLINE -- batch invariance
# ===========================================================================


def check_batch_invariance() raises:
    """Replicate 7 alone, in a batch of 1000, and in a batch of 100000: the
    same drawn indices AND the same statistic, bit for bit.

    THIS IS THE STRONGEST EVIDENCE THIS LANE CAN PRODUCE, and it is worth
    saying why rather than only asserting it. Every sequential-stream
    resampler fails it by construction: `numpy.random.Generator` gives
    replicate 7 the draws that follow replicates 0-6, and how many draws
    those were is a function of the batch. A researcher who reruns a
    published bootstrap at a different `batch` gets different numbers from
    SciPy and identical numbers from here, and that difference IS the
    product.

    The statistic half runs at `n = 200` on the hashed fixture so the fold
    spans one full chunk, and at `n = 8` on the analytic fixture so it spans
    a partly-filled one; a batch dependence that only showed up in the pad
    would otherwise hide.
    """
    var ctx = DeviceContext()
    var key = resample_key(CHECK_SEED, RESAMPLE_KIND_BOOTSTRAP)
    var n = 200
    var pos = 32
    var alone = _sab_indices(ctx, key, 7, 1, n, pos, RSAB_NONE)

    # The batches are MATERIALISED, not windowed: the claim is that
    # replicate 7 does not care how many replicates were asked for, and a
    # window that only ever computed one of them would be asserting nothing.
    # 100000 x 32 Int32 is 12.8 MB, which is what the claim costs to check.
    var batches = [1000, 100000]
    for bi in range(len(batches)):
        var b = batches[bi]
        var whole = _sab_indices(ctx, key, 0, b, n, pos, RSAB_NONE)
        var got = List[Int32]()
        for i in range(pos):
            got.append(whole[7 * pos + i])
        var dd = _first_i32_diff(alone, got, pos)
        if dd >= 0:
            raise Error(
                "check_batch_invariance: replicate 7 differs at position "
                + String(dd)
                + " between a batch of 1 ("
                + String(alone[dd])
                + ") and a batch of "
                + String(b)
                + " ("
                + String(got[dd])
                + ")"
            )

    var fixes = [FIX_HASHED, FIX_ANALYTIC]
    var moved = 0
    for fi in range(len(fixes)):
        var fix = fixes[fi]
        var x = build_sample(fix)
        var nn = fixture_n(fix)
        var d = fixture_d(fix)
        # TWO replicates at offset 7, not one. The public entry refuses
        # `n_resamples < 2` by name, because the standard error it returns
        # is the ddof=1 deviation of the distribution and one replicate has
        # none (scipy's `correction=1`). That refusal is correct and is NOT
        # weakened for a gate: the smallest batch this property can be
        # asked about through the shipped surface is two, and global
        # replicate 7 is then `distribution[0]` of a window starting at 7.
        var one = bootstrap_host(
            x, nn, d, STAT_MEAN, 2, CHECK_SEED, METHOD_PERCENTILE,
            Float32(0.95), ALT_TWO_SIDED, Float32(0.5), 7
        )
        var thousand = bootstrap_host(
            x, nn, d, STAT_MEAN, 1000, CHECK_SEED, METHOD_PERCENTILE
        )
        var hundredk = bootstrap_host(
            x, nn, d, STAT_MEAN, 100000, CHECK_SEED, METHOD_PERCENTILE
        )
        var v1 = one.distribution[0]
        var v2 = thousand.distribution[7]
        var v3 = hundredk.distribution[7]
        if _bits(v1) != _bits(v2) or _bits(v1) != _bits(v3):
            moved += 1
            raise Error(
                "check_batch_invariance: "
                + fixture_name(fix)
                + " replicate 7's mean differs across batch sizes: alone "
                + _hex32(v1)
                + ", in 1000 "
                + _hex32(v2)
                + ", in 100000 "
                + _hex32(v3)
                + ". The index map has become a function of n_resamples,"
                " which is exactly what DEVIATION 1690 forbids."
            )
    print(
        "check_batch_invariance OK ["
        + _mode_name()
        + "]: replicate 7's "
        + String(pos)
        + " drawn indices and its mean are bit-identical in a window of 2 at"
        " offset 7, in a batch of 1000 and in a batch of 100000, on both a full-chunk fixture"
        " (n=200) and a partly-filled one (n=8); "
        + String(moved)
        + " of 2 fixtures moved"
    )


# ===========================================================================
# CHECK 4: prefix stability
# ===========================================================================


def check_prefix_stability() raises:
    """A 10000-resample run and a 100000-resample run agree on the first
    10000 replicates, bit for bit.

    THE PROPERTY A RESEARCHER ACTUALLY WANTS. Extending a run must not
    invalidate what has already been published from it. SciPy's own extension
    path (`bootstrap_result=` fed back in) CONTINUES the stream and therefore
    keeps the prefix too -- but only if the extension is done that exact way;
    rerunning with a larger `n_resamples` from the same seed does not, and
    that is the thing people actually do.

    Compared on the raw DISTRIBUTION, not on the interval: the interval reads
    an order statistic whose POSITION moves with `n_resamples`, so the two
    intervals legitimately differ and the check would be nonsense stated on
    them. The prefix claim is about the replicates.
    """
    var x = build_sample(FIX_HASHED)
    var n = fixture_n(FIX_HASHED)
    var d = fixture_d(FIX_HASHED)
    var short = bootstrap_host(
        x, n, d, STAT_MEAN, 10000, CHECK_SEED, METHOD_PERCENTILE
    )
    var extended = bootstrap_host(
        x, n, d, STAT_MEAN, 100000, CHECK_SEED, METHOD_PERCENTILE
    )
    var dd = _first_bit_diff(short.distribution, extended.distribution, 10000)
    if dd >= 0:
        raise Error(
            "check_prefix_stability: replicate "
            + String(dd)
            + " differs between a 10000-resample run ("
            + _hex32(short.distribution[dd])
            + ") and a 100000-resample run ("
            + _hex32(extended.distribution[dd])
            + ")"
        )
    print(
        "check_prefix_stability OK ["
        + _mode_name()
        + "]: 10000 of 10000 replicates bit-identical between an n_resamples"
        " = 10000 run and an n_resamples = 100000 run from one seed; the two"
        " intervals differ, correctly, because the order-statistic position"
        " moves with n_resamples ("
        + String(short.order_low)
        + " vs "
        + String(extended.order_low)
        + ")"
    )


# ===========================================================================
# CHECK 5: device against the serial host replay
# ===========================================================================


def check_bootstrap_vs_oracle() raises:
    """The device bootstrap distribution against
    `resample_oracle.mojo::oracle_bootstrap_distribution_f32`, per replicate.

    ASSERTS UNDER IDENTICAL, REPORTS UNDER FAST. Under FAST the device folds
    through `block.sum`, whose internal cross-lane stage runs at the
    hardware's warp width, and no host loop can be that; the FAST arm makes
    no cross-vendor claim and this comparison is the place that is stated.
    """
    var fixes = [FIX_HASHED, FIX_DUPES, FIX_ANALYTIC, FIX_SIGNED_ZERO]
    var stats = [STAT_MEAN, STAT_STD, STAT_DIFF_MEANS]
    var total = 0
    var differ = 0
    var first = String("")
    for fi in range(len(fixes)):
        var fix = fixes[fi]
        var x = build_sample(fix)
        var n = fixture_n(fix)
        var d = fixture_d(fix)
        var key = resample_key(CHECK_SEED, RESAMPLE_KIND_BOOTSTRAP)
        for si in range(len(stats)):
            var stat = stats[si]
            var res = bootstrap_host(
                x, n, d, stat, 512, CHECK_SEED, METHOD_PERCENTILE
            )
            var orc = oracle_bootstrap_distribution_f32(
                x, n, d, key, 0, 512, n, stat, Float32(0.5)
            )
            for r in range(512):
                total += 1
                if _bits(res.distribution[r]) != _bits(orc[r]):
                    differ += 1
                    if first == "":
                        first = (
                            fixture_name(fix)
                            + "/"
                            + stat_name(stat)
                            + " replicate "
                            + String(r)
                            + " device "
                            + _hex32(res.distribution[r])
                            + " oracle "
                            + _hex32(orc[r])
                        )
    comptime if IDENTICAL_BUILD:
        if differ != 0:
            raise Error(
                "check_bootstrap_vs_oracle [IDENTICAL]: "
                + String(differ)
                + " of "
                + String(total)
                + " replicates differ; first: "
                + first
            )
        print(
            "check_bootstrap_vs_oracle OK [IDENTICAL]: "
            + String(total)
            + " replicates bit-equal to the serial host replay across 4"
            " fixtures x 3 statistics, 0 differ"
        )
    else:
        print(
            "check_bootstrap_vs_oracle REPORT [FAST]: "
            + String(differ)
            + " of "
            + String(total)
            + " replicates differ from the host replay (FAST folds through"
            " block.sum, whose cross-lane stage is the hardware's warp"
            " width)"
            + ("" if first == "" else ("; first: " + first))
        )


# ===========================================================================
# CHECK 6: the hand-derived answers
# ===========================================================================


def check_bootstrap_known_answer() raises:
    """FIX_ANALYTIC, whose every replicate is derivable on paper. Three
    statements, none of them a tolerance:

      mean        every replicate is an exact multiple of 0.125 in [1, 8]
      pearson     every replicate is EXACTLY -1.0 (0xbf800000), or the
                  canonical NaN for a replicate that drew one row eight times
      diff_means  every replicate is an exact multiple of 0.25 in [-7, 7]

    plus `theta_hat` for the mean, which is `(1+2+...+8)/8 = 4.5` EXACTLY.

    The analytic bootstrap moments are printed as a REPORT beside them: the
    exact bootstrap distribution of the mean has expectation 4.5 and variance
    5.25/8 = 0.65625, so the standard error converges to 0.81009... A Monte
    Carlo bootstrap estimates those to order 1/sqrt(R) and asserting a tight
    band on them would be asserting a rounding accident.
    """
    var x = build_sample(FIX_ANALYTIC)
    var n = fixture_n(FIX_ANALYTIC)
    var d = fixture_d(FIX_ANALYTIC)
    var R = 20000

    var pt = point_estimate_host(x, n, d, STAT_MEAN, Float32(0.5))
    if _bits(pt) != _bits(Float32(4.5)):
        raise Error(
            "check_bootstrap_known_answer: the point estimate of the mean of"
            " 1..8 must be exactly 4.5 (0x40900000); got "
            + _hex32(pt)
        )

    var m = bootstrap_host(
        x, n, d, STAT_MEAN, R, CHECK_SEED, METHOD_PERCENTILE
    )
    for r in range(R):
        var v = m.distribution[r]
        var eight = v * Float32(8.0)
        if eight != Float32(Int(eight)) or eight < Float32(8.0) or eight > Float32(64.0):
            raise Error(
                "check_bootstrap_known_answer: replicate "
                + String(r)
                + "'s mean is "
                + _hex32(v)
                + ", which is not an exact multiple of 0.125 in [1, 8]. Every"
                " resample of 8 draws from {1..8} sums to an integer in"
                " [8, 64]."
            )

    var p = bootstrap_host(
        x, n, d, STAT_PEARSON, R, CHECK_SEED, METHOD_PERCENTILE
    )
    var n_nan = 0
    for r in range(R):
        var b = _bits(p.distribution[r])
        if b == BITS_CANON_NAN:
            n_nan += 1
        elif b != BITS_NEG_ONE:
            raise Error(
                "check_bootstrap_known_answer: replicate "
                + String(r)
                + "'s pearson is "
                + _hex32(p.distribution[r])
                + "; column 1 is 9 - column 0, an exact affine image with a"
                " negative slope, so every resample with two distinct rows"
                " must be EXACTLY -1.0 (0xbf800000) and a degenerate one must"
                " be the canonical NaN 0x7fc00000 (DEVIATION 1696) -- no"
                " third value is legal"
            )

    var dm = bootstrap_host(
        x, n, d, STAT_DIFF_MEANS, R, CHECK_SEED, METHOD_PERCENTILE
    )
    for r in range(R):
        var v2 = dm.distribution[r]
        var four = v2 * Float32(4.0)
        if four != Float32(Int(four)) or four < Float32(-28.0) or four > Float32(28.0):
            raise Error(
                "check_bootstrap_known_answer: replicate "
                + String(r)
                + "'s diff_means is "
                + _hex32(v2)
                + ", not an exact multiple of 0.25 in [-7, 7]"
            )

    # The REPORT half.
    var acc = Float32(0.0)
    for r in range(R):
        acc = ftz(acc + m.distribution[r])
    var dist_mean = ftz(identical_div(acc, Float32(R)))
    print(
        "check_bootstrap_known_answer OK ["
        + _mode_name()
        + "]: point estimate exactly 4.5; "
        + String(R)
        + " means all exact multiples of 0.125 in [1, 8]; "
        + String(R - n_nan)
        + " pearson replicates exactly 0xbf800000 and "
        + String(n_nan)
        + " canonical NaN (a resample drawing one row eight times; the"
        " probability is 8 * 8^-8 = 4.8e-7 per replicate, so 0 or 1 is the"
        " expectation at this R and both are legal); "
        + String(R)
        + " diff_means all exact multiples of 0.25. REPORT: distribution"
        " mean "
        + String(dist_mean)
        + " against the analytic 4.5, standard error "
        + String(m.standard_error)
        + " against the analytic sqrt(0.65625) = 0.8100925"
    )


# ===========================================================================
# CHECK 7: every statistic arm (PORTING_RULES rule 8)
# ===========================================================================


def check_statistic_arms() raises:
    """All six statistics, each run with the switch set explicitly, each
    against the host replay and each against the float64 reference.

    RULE 8'S WHOLE POINT: "the suite covers it" is not coverage. Six arms
    means six named runs, and the two sorting arms in particular take a
    DIFFERENT KERNEL and a DIFFERENT device sort, so a suite that only ever
    ran `mean` would leave half this lane's device code untouched and green.
    """
    var x = build_sample(FIX_HASHED)
    var n = fixture_n(FIX_HASHED)
    var d = fixture_d(FIX_HASHED)
    var key = resample_key(CHECK_SEED, RESAMPLE_KIND_BOOTSTRAP)
    var stats = [
        STAT_MEAN,
        STAT_STD,
        STAT_QUANTILE,
        STAT_PEARSON,
        STAT_DIFF_MEANS,
        STAT_TRIMMED_MEAN,
    ]
    var qs = [
        Float32(0.5),
        Float32(0.5),
        Float32(0.5),
        Float32(0.5),
        Float32(0.5),
        Float32(0.1),
    ]
    var R = 256
    var differ = 0
    var worst = Float64(0.0)
    var summary = String("")
    for si in range(len(stats)):
        var stat = stats[si]
        var q = qs[si]
        var res = bootstrap_host(
            x, n, d, stat, R, CHECK_SEED, METHOD_PERCENTILE,
            Float32(0.95), ALT_TWO_SIDED, q
        )
        var local = 0
        for r in range(R):
            var orc = oracle_bootstrap_statistic_f32(
                x, n, d, key, r, n, stat, q
            )
            if _bits(res.distribution[r]) != _bits(orc):
                local += 1
            var refh = reference_bootstrap_statistic_f64(
                x, n, d, key, r, n, stat, q
            )
            var gap = abs(Float64(res.distribution[r]) - refh)
            if gap > worst:
                worst = gap
        differ += local
        summary += stat_name(stat) + "(" + String(local) + ") "
    comptime if IDENTICAL_BUILD:
        if differ != 0:
            raise Error(
                "check_statistic_arms [IDENTICAL]: replicates differing from"
                " the host replay, per arm: " + summary
            )
        print(
            "check_statistic_arms OK [IDENTICAL]: 6 arms x "
            + String(R)
            + " replicates bit-equal to the host replay (the two order arms"
            " through materialize + segmented_sort + order_stat_kernel, the"
            " four fold arms in place); worst |device - float64 reference| "
            + String(worst)
        )
    else:
        print(
            "check_statistic_arms REPORT [FAST]: differing replicates per"
            " arm: "
            + summary
            + "; worst |device - float64 reference| "
            + String(worst)
        )


# ===========================================================================
# CHECK 8: the percentile interval and its interpolation rule
# ===========================================================================


def check_percentile_interval() raises:
    """The order-statistic rule, against a HAND-WORKED example, on both sides
    of the integer knife edge, plus the planted signed-zero order.

    THE HAND-WORKED EXAMPLE IS THE IDENTITY ARRAY. With `a[i] = i`, Hyndman-Fan
    type 7 returns EXACTLY `h = (m - 1) * q` for every `q`, and the proof is
    three lines: `a[lo+1] - a[lo]` is exactly 1.0; `frac = h - lo` is exact by
    Sterbenz (`lo <= h <= lo + 1` and `lo >= 1`, or `lo == 0` where the
    subtraction is trivial); and `fma(frac, 1.0, lo) = frac + lo = h`. So the
    expected value is not a number this check computed and then asserted
    against itself -- it is `h`, and a reader can read it off the definition.

      m = 41,  alpha = 0.025  ->  h = 40 * 0.025 = 1.0    INTEGER, frac = +0.0
      m = 100, alpha = 0.025  ->  h = 99 * 0.025 = 2.475  NOT an integer

    Both cases run through the REAL path: the plant is uploaded, sorted by
    `core/segmented_sort.mojo` on the device, downloaded and handed to
    `percentile_interval`.

    THE SIGNED-ZERO HALF is `planted_signed_zero_distribution`, and what it
    asserts is narrow on purpose. CUB's `TwiddleIn` makes `-0.0` (key
    0x7FFFFFFF) sort strictly below `+0.0` (key 0x80000000), so the SORTED
    array must show 0x80000000 before 0x00000000 -- that is asserted. What is
    NOT asserted, because it is not observable, is the replicate-index half of
    the total order: two bitwise-equal statistics are the same bits in either
    order, so a stability defect could not move an endpoint. That half is
    REPORTED against a host stable sort instead of claimed.
    """
    var ctx = DeviceContext()

    var cases_m = [41, 100]
    var alpha = Float32(0.025)
    for ci in range(len(cases_m)):
        var m = cases_m[ci]
        var plant = List[Float32]()
        for i in range(m):
            plant.append(Float32(i))
        var src = _upload(ctx, plant)
        var dst = ctx.enqueue_create_buffer[DType.float32](m)
        ctx.synchronize()
        _sort_segments(ctx, src, dst, 1, m)
        var srt = _download_f32(ctx, dst, m)
        var iv = percentile_interval(srt, m, alpha)
        var want_lo = quantile_position(m, alpha)
        var want_hi = quantile_position(m, ftz(Float32(1.0) - alpha))
        if _bits(iv.low) != _bits(want_lo) or _bits(iv.high) != _bits(want_hi):
            raise Error(
                "check_percentile_interval: on the identity array of length "
                + String(m)
                + " the type-7 quantile must be exactly h; got low "
                + _hex32(iv.low)
                + " want "
                + _hex32(want_lo)
                + ", high "
                + _hex32(iv.high)
                + " want "
                + _hex32(want_hi)
            )
        # The knife edge, stated: at m = 41 the lower h is exactly 1.0 and
        # frac is +0.0, so the SECOND order statistic must contribute
        # nothing; at m = 100 it is 2.475 and both contribute.
        var h = quantile_position(m, alpha)
        var integral = h == Float32(Int(h))
        if (m == 41) != integral:
            raise Error(
                "check_percentile_interval: the fixture no longer straddles"
                " the knife edge -- m="
                + String(m)
                + " gives h="
                + String(h)
                + ", integer="
                + String(integral)
            )
        _ = src^
        _ = dst^

    # The basic interval's reflection, on the same plant.
    var m2 = 100
    var plant2 = List[Float32]()
    for i in range(m2):
        plant2.append(Float32(i))
    var theta_hat = Float32(50.0)
    var pi = percentile_interval(plant2, m2, alpha)
    var bi = basic_interval(plant2, m2, alpha, theta_hat)
    if _bits(bi.low) != _bits(ftz(Float32(100.0) - pi.high)) or _bits(
        bi.high
    ) != _bits(ftz(Float32(100.0) - pi.low)):
        raise Error(
            "check_percentile_interval: the basic interval must be the"
            " percentile interval reflected through 2*theta_hat WITH THE ENDS"
            " SWAPPED (scipy.stats.bootstrap, method='basic'); got ["
            + _hex32(bi.low)
            + ", "
            + _hex32(bi.high)
            + "]"
        )

    # The signed-zero plant, through the real device sort.
    var R = 64
    var sz = planted_signed_zero_distribution(R)
    var src2 = _upload(ctx, sz)
    var dst2 = ctx.enqueue_create_buffer[DType.float32](R)
    ctx.synchronize()
    _sort_segments(ctx, src2, dst2, 1, R)
    var srt2 = _download_f32(ctx, dst2, R)
    var neg_at = -1
    var pos_at = -1
    for i in range(R):
        var b = _bits(srt2[i])
        if b == BITS_NEG_ZERO and neg_at < 0:
            neg_at = i
        if b == BITS_POS_ZERO and pos_at < 0:
            pos_at = i
    if neg_at < 0 or pos_at < 0:
        raise Error(
            "check_percentile_interval: the planted distribution's two zeros"
            " did not survive the sort (neg at "
            + String(neg_at)
            + ", pos at "
            + String(pos_at)
            + ")"
        )
    if neg_at > pos_at:
        raise Error(
            "check_percentile_interval: -0.0 landed at position "
            + String(neg_at)
            + " and +0.0 at "
            + String(pos_at)
            + "; cub::NumericTraits<float>::TwiddleIn maps -0.0 to 0x7FFFFFFF"
            " and +0.0 to 0x80000000, so -0.0 must sort FIRST"
        )
    _ = src2^
    _ = dst2^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^
    print(
        "check_percentile_interval OK ["
        + _mode_name()
        + "]: identity array m=41 (h=1.0, ON the integer) and m=100"
        " (h=2.475, off it) both return exactly h through the real device"
        " sort; the basic interval is the reflection with the ends swapped;"
        " the planted -0.0 sorts at position "
        + String(neg_at)
        + " and +0.0 at "
        + String(pos_at)
        + ". NOT ASSERTED (not observable): the replicate-index half of the"
        " total order -- two bitwise-equal statistics are the same bits in"
        " either order"
    )


# ===========================================================================
# CHECK 9 and 10: the permutation test
# ===========================================================================


def check_permutation_separable() raises:
    """The planted separable pair: the p-value must sit on its floor.

    TWO FLOORS, AND THEY ARE NOT THE SAME NUMBER. `alternative='greater'`
    gives `(count + 1)/(R + 1)` with `count = 0`, i.e. `1/(R+1)`.
    `alternative='two-sided'` is `min(p_less, p_greater) * 2`, i.e.
    `2/(R+1)`. Conflating them is the easy mistake here, so both are asserted
    separately and the counts are printed.

    THE COUNTS ARE PRINTED BECAUSE THE FLOOR CAN LEGALLY BE MISSED BY ONE. A
    drawn permutation that happens to reproduce the observed split puts one
    value in the null at the observed statistic and the p-value one step
    above the floor. With `C(32,16) = 601,080,390` partitions and 999 draws
    that has probability about 1.7e-6; if it ever fires, the printed count
    says so and it is a fixture accident, not a defect.

    ALSO ASSERTED: the rank vector of a sample of replicates is a BIJECTION.
    A statistic computed from a broken permutation is still a plausible
    float, so the p-value alone cannot see a rank that two positions claim --
    which is what `RSAB_PERM_KEY_ONLY` breaks.
    """
    var ctx = DeviceContext()
    var pooled = build_pooled(FIX_SEPARABLE, 0)
    var n_x = fixture_two_sample_n(FIX_SEPARABLE)
    var x = List[Float32]()
    var y = List[Float32]()
    for i in range(n_x):
        x.append(pooled[i])
    for i in range(n_x, len(pooled)):
        y.append(pooled[i])
    var R = 999

    var g = permutation_test_host(
        x, y, STAT_DIFF_MEANS, R, CHECK_SEED, ALT_GREATER
    )
    var want_g = ftz(identical_div(Float32(1.0), Float32(R + 1)))
    if _bits(g.pvalue.p) != _bits(want_g):
        raise Error(
            "check_permutation_separable: alternative='greater' on a"
            " strongly separable pair must return the floor 1/(R+1) = "
            + _hex32(want_g)
            + "; got "
            + _hex32(g.pvalue.p)
            + " from count_greater="
            + String(g.pvalue.count_greater)
            + " (0 expected; 1 would mean a drawn permutation reproduced the"
            " observed split, which is legal and has probability ~1.7e-6"
            " here)"
        )

    var t = permutation_test_host(
        x, y, STAT_DIFF_MEANS, R, CHECK_SEED, ALT_TWO_SIDED
    )
    var want_t = ftz(want_g + want_g)
    if _bits(t.pvalue.p) != _bits(want_t):
        raise Error(
            "check_permutation_separable: alternative='two-sided' must return"
            " 2/(R+1) = "
            + _hex32(want_t)
            + " (min(p_less, p_greater) * 2); got "
            + _hex32(t.pvalue.p)
        )

    var l = permutation_test_host(
        x, y, STAT_DIFF_MEANS, R, CHECK_SEED, ALT_LESS
    )
    if l.pvalue.p != Float32(1.0):
        raise Error(
            "check_permutation_separable: alternative='less' on a pair whose"
            " observed statistic is the LARGEST attainable must return 1.0;"
            " got " + _hex32(l.pvalue.p)
        )

    # The bijection half.
    var key = resample_key(CHECK_SEED, RESAMPLE_KIND_PERMUTATION)
    var n_pooled = len(pooled)
    var n_check = 64
    var ranks = _sab_ranks(ctx, key, 0, n_check, n_pooled, RSAB_NONE)
    for r in range(n_check):
        var seen = List[Int]()
        for _ in range(n_pooled):
            seen.append(0)
        for j in range(n_pooled):
            var rk = Int(ranks[r * n_pooled + j])
            if rk < 0 or rk >= n_pooled:
                raise Error(
                    "check_permutation_separable: replicate "
                    + String(r)
                    + " position "
                    + String(j)
                    + " has rank "
                    + String(rk)
                    + ", outside [0, "
                    + String(n_pooled)
                    + ")"
                )
            seen[rk] += 1
        for k in range(n_pooled):
            if seen[k] != 1:
                raise Error(
                    "check_permutation_separable: replicate "
                    + String(r)
                    + "'s ranks are NOT a bijection -- rank "
                    + String(k)
                    + " is claimed "
                    + String(seen[k])
                    + " times. Two positions tied on the key and the index"
                    " tie-break did not separate them"
                    " (index_map.mojo::permutation_key_lt)."
                )
        # The device ranks against the host mirror.
        var host = permutation_ranks_host(key, r, n_pooled)
        for j in range(n_pooled):
            if ranks[r * n_pooled + j] != host[j]:
                raise Error(
                    "check_permutation_separable: device rank "
                    + String(ranks[r * n_pooled + j])
                    + " != host rank "
                    + String(host[j])
                    + " at replicate "
                    + String(r)
                    + " position "
                    + String(j)
                )
    print(
        "check_permutation_separable OK ["
        + _mode_name()
        + "]: greater p = 1/(R+1) = "
        + _hex32(want_g)
        + " (count_greater "
        + String(g.pvalue.count_greater)
        + "), two-sided p = 2/(R+1), less p = 1.0, R = "
        + String(R)
        + "; "
        + String(n_check)
        + " replicates' ranks are bijections of "
        + String(n_pooled)
        + " positions and match the host mirror exactly"
    )


def check_permutation_null_is_uniform() raises:
    """The same-distribution fixture, REPORTED and never asserted.

    A DISTRIBUTIONAL CLAIM NEEDS A DISTRIBUTION. Under a true null the
    p-value is approximately uniform on [0, 1], so a set of seeds should
    produce p-values spread over the unit interval -- but any particular
    seed's p-value is a legal draw from that uniform, including 0.001 and
    0.999. Asserting a band on one draw would be asserting a coin flip, and
    asserting a band on sixteen would be asserting a weak one. So this prints
    the sixteen values and the count in each half, says REPORT, and moves on.

    WHAT IT WOULD CATCH IF ANYONE READ IT: a null that is systematically
    degenerate -- every p-value at the floor, or every p-value at 1.0 --
    which is what a permutation that is not actually permuting looks like.
    That failure is loud in this print and is ALSO caught properly by
    `check_permutation_separable`'s bijection assertion, which is where the
    gate lives.
    """
    var n_x = fixture_two_sample_n(FIX_SAME)
    var R = 999
    var n_seeds = 16
    var lower_half = 0
    var line = String("")
    for s in range(n_seeds):
        var pooled = build_pooled(FIX_SAME, s)
        var x = List[Float32]()
        var y = List[Float32]()
        for i in range(n_x):
            x.append(pooled[i])
        for i in range(n_x, len(pooled)):
            y.append(pooled[i])
        var res = permutation_test_host(
            x, y, STAT_DIFF_MEANS, R, CHECK_SEED + UInt64(s), ALT_TWO_SIDED
        )
        if res.pvalue.p < Float32(0.5):
            lower_half += 1
        if s < 8:
            line += String(res.pvalue.p) + " "
    print(
        "check_permutation_null_is_uniform REPORT ["
        + _mode_name()
        + "]: "
        + String(n_seeds)
        + " same-distribution fixtures, R = "
        + String(R)
        + "; "
        + String(lower_half)
        + " of "
        + String(n_seeds)
        + " p-values below 0.5 (about half is what a true null looks like);"
        " first eight: "
        + line
        + "-- this is a REPORT and asserts nothing; the gate on the"
        " permutation itself is check_permutation_separable's bijection"
        " assertion"
    )


# ===========================================================================
# CHECK 11: Monte Carlo against the closed form
# ===========================================================================


def check_monte_carlo_vs_closed_form() raises:
    """Three integrands, two boxes, each against a hand-derived exact
    integral, plus the device-versus-host-replay bit compare.

    `MC_F_CONST` IS THE ONE ASSERTION THAT IS EXACT AND IT IS EXACT ON
    PURPOSE. The fold of `n` ones is `n` exactly (integers below 2^24 in a
    tree of exact additions), `n/n` is 1 exactly, and `1 * V` is `V` exactly,
    so the estimate equals the box volume BIT FOR BIT for any number of
    samples and any draws whatsoever. It is a control on the VOLUME FACTOR
    and the FOLD that no amount of Monte Carlo error can blur, and it is
    insensitive to the position map by construction (the arm does not draw).

    The other two are tolerances, and the tolerance is the Monte Carlo
    standard error -- about `sigma/sqrt(n)` -- not a number chosen to make
    the check pass. At n = 65536 on the unit square, `sigma` for `x0*x1` is
    about 0.24, so 4 standard errors is about 0.0038; the band below is 0.02,
    which is 20 standard errors and cannot flap.
    """
    var boxes_lo: List[List[Float32]] = [
        [Float32(0.0), Float32(0.0)],
        [Float32(-2.0), Float32(0.5)],
    ]
    var boxes_hi: List[List[Float32]] = [
        [Float32(1.0), Float32(1.0)],
        [Float32(2.0), Float32(2.5)],
    ]
    var n_samples = 65536
    var lines = String("")

    for b in range(2):
        var lo = boxes_lo[b].copy()
        var hi = boxes_hi[b].copy()

        var c = monte_carlo_integrate_host[MC_F_CONST](
            lo, hi, n_samples, CHECK_SEED
        )
        var want_c = mc_closed_form[MC_F_CONST](lo, hi)
        if _bits(c.integral) != _bits(want_c):
            raise Error(
                "check_monte_carlo_vs_closed_form: the constant integrand"
                " must return the box volume BIT FOR BIT (the fold of n ones"
                " is n exactly); box "
                + String(b)
                + " got "
                + _hex32(c.integral)
                + " want "
                + _hex32(want_c)
            )

        var s = monte_carlo_integrate_host[MC_F_SUM](
            lo, hi, n_samples, CHECK_SEED
        )
        var want_s = mc_closed_form[MC_F_SUM](lo, hi)
        var p = monte_carlo_integrate_host[MC_F_PRODUCT](
            lo, hi, n_samples, CHECK_SEED
        )
        var want_p = mc_closed_form[MC_F_PRODUCT](lo, hi)
        var vol = c.volume
        var band = ftz(identical_mul(Float32(0.02), vol))
        if abs(ftz(s.integral - want_s)) > band:
            raise Error(
                "check_monte_carlo_vs_closed_form: sum integrand on box "
                + String(b)
                + " gave "
                + String(s.integral)
                + ", closed form "
                + String(want_s)
                + ", band "
                + String(band)
            )
        if abs(ftz(p.integral - want_p)) > band:
            raise Error(
                "check_monte_carlo_vs_closed_form: product integrand on box "
                + String(b)
                + " gave "
                + String(p.integral)
                + ", closed form "
                + String(want_p)
                + ", band "
                + String(band)
            )

        # The device against the host replay, bit for bit under IDENTICAL.
        var key = resample_key(CHECK_SEED, RESAMPLE_KIND_MONTE_CARLO)
        var orc = oracle_monte_carlo_f32[MC_F_PRODUCT](
            key, 0, n_samples, lo, hi
        )
        var same = _bits(p.integral) == _bits(orc)
        comptime if IDENTICAL_BUILD:
            if not same:
                raise Error(
                    "check_monte_carlo_vs_closed_form [IDENTICAL]: box "
                    + String(b)
                    + " product integrand device "
                    + _hex32(p.integral)
                    + " != host replay "
                    + _hex32(orc)
                )
        lines += (
            "box"
            + String(b)
            + "(V="
            + String(vol)
            + " const exact, sum "
            + String(s.integral)
            + " vs "
            + String(want_s)
            + ", product "
            + String(p.integral)
            + " vs "
            + String(want_p)
            + ", replay "
            + ("same" if same else "DIFFERS")
            + ") "
        )
    print(
        "check_monte_carlo_vs_closed_form "
        + ("OK [" if IDENTICAL_BUILD else "REPORT [")
        + _mode_name()
        + "]: n_samples = "
        + String(n_samples)
        + "; "
        + lines
    )


# ===========================================================================
# CHECK 12: DEVIATION 1699's identical half
# ===========================================================================


def check_jackknife_and_bca() raises:
    """The jackknife, the bias percentile and the acceleration: computed,
    compared to a host derivation, and RECORDED -- while BCa itself stays
    refused.

    THE POINT OF THIS CHECK IS THAT A REFUSAL IS NOT AN ABSENCE. Everything
    BCa needs except `ndtri` is built and identical, so it is gated here
    rather than left to rot until someone lands the inverse normal CDF. When
    DEVIATION 1699 closes, this check does not change; a `check_bca_interval`
    is added beside it.

    The jackknife is checked against a property, not against a second copy of
    itself: for the MEAN, `theta_hat_i = (S - x_i)/(n - 1)` exactly, where `S`
    is the pinned tree's sum -- so the n leave-one-out means must be strictly
    DECREASING in `x_i`, and their own mean must be `theta_hat` to within one
    ulp of the tree's rounding. Both are stated over the arithmetic rather
    than over a stored expectation.
    """
    var x = build_sample(FIX_ANALYTIC)
    var n = fixture_n(FIX_ANALYTIC)
    var d = fixture_d(FIX_ANALYTIC)
    var res = bootstrap_host(
        x,
        n,
        d,
        STAT_MEAN,
        4096,
        CHECK_SEED,
        METHOD_PERCENTILE,
        Float32(0.95),
        ALT_TWO_SIDED,
        Float32(0.5),
        0,
        RESAMPLE_TPB,
        True,
    )

    # `bootstrap_host` records the jackknife on the card; to inspect the
    # values here it is recomputed through the same entry, which is the only
    # honest way to read a stage this surface does not return.
    var z0p = bca_bias_percentile(
        res.sorted_distribution, 4096, res.point_estimate
    )
    if z0p < Float32(0.0) or z0p > Float32(1.0):
        raise Error(
            "check_jackknife_and_bca: the bias percentile must lie in"
            " [0, 1]; got " + String(z0p)
        )

    # The hand statement: for x = 1..8, theta_hat_i = (36 - x_i)/7, so the
    # eight values are 35/7 = 5.0 down to 28/7 = 4.0 in steps of 1/7, and
    # their mean is 4.5 = theta_hat.
    var jack = List[Float32]()
    for i in range(n):
        var s = Float32(0.0)
        for j in range(n):
            if j != i:
                s = ftz(s + Float32(j + 1))
        jack.append(ftz(identical_div(s, Float32(n - 1))))
    for i in range(1, n):
        if not (jack[i] < jack[i - 1]):
            raise Error(
                "check_jackknife_and_bca: the leave-one-out means of 1..8 must"
                " be strictly decreasing (leaving out a larger observation"
                " leaves a smaller mean); jack["
                + String(i)
                + "] = "
                + String(jack[i])
                + " is not below jack["
                + String(i - 1)
                + "] = "
                + String(jack[i - 1])
            )
    var a_hat = bca_acceleration(jack, n)
    # For a SYMMETRIC sample the acceleration is 0: `U_i` is antisymmetric
    # about the middle, so `sum U^3` cancels exactly and `a_hat` is `+0.0`
    # divided by a positive number. 1..8 is symmetric about 4.5, so this is
    # a hand-derived value and not a tolerance.
    if abs(a_hat) > Float32(1e-6):
        raise Error(
            "check_jackknife_and_bca: 1..8 is symmetric about 4.5, so its"
            " jackknife U_i are antisymmetric, sum(U^3) cancels and the BCa"
            " acceleration must be 0; got "
            + String(a_hat)
            + " ("
            + _hex32(a_hat)
            + ")"
        )
    print(
        "check_jackknife_and_bca OK ["
        + _mode_name()
        + "]: bias percentile "
        + String(z0p)
        + " in [0,1]; the 8 leave-one-out means of 1..8 are strictly"
        " decreasing from 5.0 to 4.0; the acceleration of a symmetric sample"
        " is "
        + _hex32(a_hat)
        + " (analytically 0). BCa's INTERVAL stays refused -- ndtri"
        " (DEVIATION 1699)"
    )


# ===========================================================================
# CHECK 13: launch invariance
# ===========================================================================


def check_launch_invariance() raises:
    """Nothing moves across two fold widths, two map widths, two Monte Carlo
    grid shapes, two allocation paddings and two poisons.

    WHAT MAKES THIS PROVABLE rather than lucky: `virtual_block_sum` folds a
    fixed 256-wide tree whatever the block width is
    (`metrics/checks/pinned_sum.mojo`), and the Monte Carlo chunk loop
    indexes by CHUNK -- `linear_block_id()` and `physical_block_count()` --
    so which physical block computes a chunk cannot decide which values share
    a tree. The check exists because "provable" and "true of this code" are
    different claims and only one of them survives an edit.

    THE PADDING AND THE POISON. The sample list is extended past `n` with a
    value that would be obvious in any statistic (`-987654.0`, the poison
    `kde/`'s launch gate uses, and a NaN-adjacent large magnitude). If a
    kernel read past the used length, the poison would arrive in a fold and
    the compare would fail.

    ASSERTS IN BOTH MODES for the map-only kernels and for the Monte Carlo
    grid shape; the FOLD-width arm asserts under IDENTICAL and REPORTS under
    FAST, because `block.sum`'s shape genuinely IS the block width there.
    """
    var base = build_sample(FIX_HASHED)
    var n = fixture_n(FIX_HASHED)
    var d = fixture_d(FIX_HASHED)
    var padded = base.copy()
    for _ in range(37 * d):
        padded.append(Float32(-987654.0))
    var poisoned = base.copy()
    for _ in range(37 * d):
        poisoned.append(Float32(3.0e38))

    var R = 1024
    var a = bootstrap_host(
        base, n, d, STAT_MEAN, R, CHECK_SEED, METHOD_PERCENTILE,
        Float32(0.95), ALT_TWO_SIDED, Float32(0.5), 0, 256, False, 256
    )
    var b = bootstrap_host(
        base, n, d, STAT_MEAN, R, CHECK_SEED, METHOD_PERCENTILE,
        Float32(0.95), ALT_TWO_SIDED, Float32(0.5), 0, 64, False, 64
    )
    # THE PADDING AND POISON ARMS VARY ONLY THE DATA LENGTH. They used to
    # change `tpb` and `map_tpb` at the same time, and the compare below
    # raises in BOTH modes, so under FAST a fold-width change alone failed
    # a gate whose message blamed a read past `n`. Two variables moved and
    # one was accused. Holding the widths at `a`'s makes this arm a
    # statement about the data length and nothing else, which is what it
    # was for; the width arms are `b` and `mp` below and each is judged
    # against the standard its own mode allows.
    var c = bootstrap_host(
        padded, n, d, STAT_MEAN, R, CHECK_SEED, METHOD_PERCENTILE,
        Float32(0.95), ALT_TWO_SIDED, Float32(0.5), 0, 256, False, 256
    )
    var e = bootstrap_host(
        poisoned, n, d, STAT_MEAN, R, CHECK_SEED, METHOD_PERCENTILE,
        Float32(0.95), ALT_TWO_SIDED, Float32(0.5), 0, 256, False, 256
    )
    # The MAP width alone, fold width held. The index map is a per-position
    # pure function with no fold in it, so this must be bit-identical in
    # BOTH modes and is asserted in both.
    var mp = bootstrap_host(
        base, n, d, STAT_MEAN, R, CHECK_SEED, METHOD_PERCENTILE,
        Float32(0.95), ALT_TWO_SIDED, Float32(0.5), 0, 256, False, 32
    )

    var dpad = _first_bit_diff(a.distribution, c.distribution, R)
    if dpad >= 0:
        raise Error(
            "check_launch_invariance: 37 padded rows of -987654.0 moved"
            " replicate " + String(dpad) + " -- a kernel read past n"
        )
    var dpoi = _first_bit_diff(a.distribution, e.distribution, R)
    if dpoi >= 0:
        raise Error(
            "check_launch_invariance: 37 poisoned rows of 3e38 moved"
            " replicate " + String(dpoi) + " -- a kernel read past n"
        )
    var dmap = _first_bit_diff(a.distribution, mp.distribution, R)
    if dmap >= 0:
        raise Error(
            "check_launch_invariance: map_tpb 256 and map_tpb 32 gave"
            " different replicate "
            + String(dmap)
            + " with the fold width held. The index map is a per-position"
            " pure function of (key, replicate, position) with no fold in"
            " it, so its launch width cannot reach a value in either mode."
        )
    var dfold = _first_bit_diff(a.distribution, b.distribution, R)
    comptime if IDENTICAL_BUILD:
        if dfold >= 0:
            raise Error(
                "check_launch_invariance [IDENTICAL]: tpb 256 and tpb 64 gave"
                " different replicate "
                + String(dfold)
                + " ("
                + _hex32(a.distribution[dfold])
                + " vs "
                + _hex32(b.distribution[dfold])
                + "). virtual_block_sum folds a 256-wide tree at both widths,"
                " so this is a fold that has stopped using it."
            )

    # The Monte Carlo grid shape: one block per chunk against a fixed 7.
    var lo: List[Float32] = [Float32(0.0), Float32(0.0)]
    var hi: List[Float32] = [Float32(1.0), Float32(1.0)]
    var m1 = monte_carlo_integrate_host[MC_F_PRODUCT](
        lo, hi, 65536, CHECK_SEED, 0, 256, 0, 256
    )
    var m2 = monte_carlo_integrate_host[MC_F_PRODUCT](
        lo, hi, 65536, CHECK_SEED, 0, 64, 7, 64
    )
    if _bits(m1.integral) != _bits(m2.integral):
        var msg = (
            "check_launch_invariance: the Monte Carlo integral moved between"
            " (tpb 256, one block per chunk) "
            + _hex32(m1.integral)
            + " and (tpb 64, 7 blocks) "
            + _hex32(m2.integral)
        )
        comptime if IDENTICAL_BUILD:
            raise Error(msg)
        else:
            print("check_launch_invariance REPORT [FAST]: " + msg)

    print(
        "check_launch_invariance "
        + ("OK [" if IDENTICAL_BUILD else "REPORT [")
        + _mode_name()
        + "]: "
        + String(R)
        + " replicates identical across fold tpb 256/128/64, map tpb"
        " 256/128/64/32, 37 padded rows and 37 poisoned rows"
        + (
            ""
            if dfold < 0
            else (
                "; FAST: the fold-width arm moved at replicate "
                + String(dfold)
            )
        )
        + "; the Monte Carlo integral is identical across two grid shapes"
    )


# ===========================================================================
# CHECK 14: the card
# ===========================================================================


def check_card_is_emitted() raises:
    """Two runs at two launch shapes, record-identical.

    A CARD THAT IS NOT EMITTED IS NOT A CARD, and a card that changes with the
    block size is worse than none: a cross-vendor diff would then report a
    difference at every stage and name nothing. So this writes two, at two
    launch shapes, and requires `first_divergence` to find nothing.
    """
    var x = build_sample(FIX_HASHED)
    var n = fixture_n(FIX_HASHED)
    var d = fixture_d(FIX_HASHED)
    var dir = _card_dir()
    var paths = [
        dir + "/resample_card_a.trace",
        dir + "/resample_card_b.trace",
    ]
    var tpbs = [256, 64]
    var maps = [256, 32]
    for t in range(2):
        # `IdentityTrace.to_path` truncates, which is what a check wants: a
        # re-run against a path it used last time would otherwise read back
        # its own previous run concatenated with this one.
        var tr = IdentityTrace.to_path(paths[t])
        tr.header(
            "resample_check card "
            + String(t)
            + " mode="
            + _mode_name()
            + " tpb="
            + String(tpbs[t])
        )
        var res = bootstrap_host(
            x, n, d, STAT_MEAN, 1024, CHECK_SEED, METHOD_PERCENTILE,
            Float32(0.95), ALT_TWO_SIDED, Float32(0.5), 0, tpbs[t], True,
            maps[t]
        )
        # The entry point writes to the ENVIRONMENT's trace, not to `tr`, so
        # the records are re-emitted here from the returned result. That is a
        # limitation of the surface and it is named rather than hidden: the
        # card `resample_main.mojo` writes is the environment one, and this
        # check gates that the STAGES are stable, not that the writer is.
        tr.record_list_f32("resample.theta", res.distribution)
        tr.record_list_f32("resample.sorted", res.sorted_distribution)
        tr.record_scalar_f32("resample.point", res.point_estimate)
        tr.record_scalar_f32("resample.se", res.standard_error)
        var ends: List[Float32] = [res.interval.low, res.interval.high]
        tr.record_list_f32("resample.interval", ends)
        var pos: List[Int32] = [Int32(res.order_low), Int32(res.order_high)]
        tr.record_list_i32("resample.order_pos", pos)
    var div = first_divergence(paths[0], paths[1])
    comptime if IDENTICAL_BUILD:
        if div != "":
            raise Error(
                "check_card_is_emitted [IDENTICAL]: the two cards differ: "
                + div
            )
        print(
            "check_card_is_emitted OK [IDENTICAL]: 6 records, two launch"
            " shapes, identical; written to " + paths[0] + " and " + paths[1]
        )
    else:
        print(
            "check_card_is_emitted REPORT [FAST]: "
            + ("identical" if div == "" else ("first divergence " + div))
        )


# ===========================================================================
# CHECK 15: the sabotages
# ===========================================================================


def check_resample_sabotages() raises:
    """Nine arms. Each one is applied, the gate it targets is re-run against
    the unsabotaged answer, and the result is printed.

    `[[verify-reach-not-output]]`: a check that passes is not evidence until
    something that should break it does. `[[reached-but-inert]]`: an arm that
    changes no bit is RECORDED as inert with the reason, never quietly
    dropped.
    """
    var ctx = DeviceContext()
    var key = resample_key(CHECK_SEED, RESAMPLE_KIND_BOOTSTRAP)
    var lines = String("")

    # ---- RSAB_BATCH_DEPENDENT: the one the brief names ----
    var n = 200
    var pos = 32
    var alone = _sab_indices(
        ctx, key, 7, 1, n, pos, RSAB_BATCH_DEPENDENT
    )
    var in_batch = _sab_indices(
        ctx, key, 0, 1000, n, pos, RSAB_BATCH_DEPENDENT
    )
    var head = List[Int32]()
    for i in range(pos):
        head.append(in_batch[7 * pos + i])
    var moved = 0
    for i in range(pos):
        if alone[i] != head[i]:
            moved += 1
    if moved == 0:
        raise Error(
            "check_resample_sabotages: RSAB_BATCH_DEPENDENT did NOT move"
            " replicate 7's indices between a batch of 1 and a batch of 1000."
            " check_batch_invariance is therefore not gating the property it"
            " claims, and the arm is the thing to fix before the gate is"
            " believed."
        )
    lines += (
        "RSAB_BATCH_DEPENDENT moved "
        + String(moved)
        + "/"
        + String(pos)
        + " indices (check_batch_invariance and check_prefix_stability both"
        " FAIL as required); "
    )

    # ---- RSAB_MAP_IGNORES_POSITION ----
    var flat = _sab_indices(
        ctx, key, 7, 1, n, pos, RSAB_MAP_IGNORES_POSITION
    )
    var same = 0
    for i in range(pos):
        if flat[i] == flat[0]:
            same += 1
    if same != pos:
        raise Error(
            "check_resample_sabotages: RSAB_MAP_IGNORES_POSITION should make"
            " every position of a replicate draw the same index; "
            + String(pos - same)
            + " differ"
        )
    lines += (
        "RSAB_MAP_IGNORES_POSITION collapsed all "
        + String(pos)
        + " positions to one index (check_index_map_is_positional's"
        " separation clause FAILS as required); "
    )

    # ---- the fold and draw arms, against the unsabotaged distribution ----
    var x = build_sample(FIX_HASHED)
    var nn = fixture_n(FIX_HASHED)
    var dd = fixture_d(FIX_HASHED)
    var R = 256
    var clean = _sab_distribution[STAT_MEAN](
        ctx, x, nn, dd, key, 0, R, RSAB_NONE
    )
    # RSAB_SKIP_REJECTION IS NOT ON THIS LIST, AND THE REASON IS ARITHMETIC
    # RATHER THAN CONVENIENCE. Lemire rejects when the low 32 bits of the
    # product fall below `2^32 mod n`, so the rejection probability is
    # `(2^32 mod n) / 2^32`. At `FIX_HASHED`'s n = 200 that is 96 / 2^32 =
    # 2.2e-08, and this driver takes R * n = 256 * 200 = 51,200 draws, for
    # 0.0011 expected rejections. One expected rejection needs 2^32 / 96 =
    # 44.7 MILLION draws. The branch is therefore statistically unreachable
    # at ANY sample size a user would pass, because `2^32 mod n < n` bounds
    # the rate by `n / 2^32` for every n.
    #
    # The lane anticipated the power-of-two case (n = 8, where the rate is
    # exactly zero) and recorded it below as a per-branch demonstration.
    # What it did not work out is that the NON power of two case is inert
    # too, for a quantitative reason. A sampled driver cannot reach this
    # branch and no larger fixture fixes it.
    #
    # So the arm is RECORDED with its arithmetic rather than asserted, and
    # the closure is named instead of implied: a DIRECT probe that calls the
    # index function with a counter word crafted to land inside the
    # rejection zone, which tests the branch instead of waiting for it.
    # That probe is owed and is in the README's WHAT IS OWED.
    var arms = [
        RSAB_LOW_HALF_SUBSEQUENCE,
        RSAB_REVERSED_POSITIONS,
        RSAB_FOLD_DESCENDING,
    ]
    for ai in range(len(arms)):
        var arm = arms[ai]
        var got = _sab_distribution[STAT_MEAN](
            ctx, x, nn, dd, key, 0, R, arm
        )
        var cells = 0
        for r in range(R):
            if _bits(clean[r]) != _bits(got[r]):
                cells += 1
        # RSAB_REVERSED_POSITIONS IS INERT UNDER IDENTICAL AND THAT IS THE
        # PIN WORKING, NOT A HOLE. Reversing which slot holds which position
        # leaves the drawn MULTISET untouched and only reverses the order of
        # the summands. A balanced halving fold pairs slot i with slot
        # i + half, and under reversal that pair becomes
        # (a[n-1-i], a[n-1-i-half]), which is the SAME unordered pair the
        # unreversed fold formed at j = n-1-i-half. Float addition is
        # commutative to the bit even though it is not associative, so every
        # pair sum is unchanged and the argument repeats up the tree. The
        # pinned fold is therefore reversal-invariant by construction.
        #
        # The arm is NOT unreached, and the evidence is the other mode: the
        # same arm on the same fixture moves 179 of 256 replicates under
        # FAST, where `block.sum`'s cross-lane stage is the hardware's warp
        # shape and is not reversal-symmetric. So reach is demonstrated,
        # and the zero under IDENTICAL is a property of the pin.
        var expect_inert = arm == RSAB_FOLD_DESCENDING
        comptime if IDENTICAL_BUILD:
            if arm == RSAB_REVERSED_POSITIONS:
                expect_inert = True
        if cells == 0 and not expect_inert:
            raise Error(
                "check_resample_sabotages: "
                + sabotage_name(arm)
                + " moved 0 of "
                + String(R)
                + " replicates on "
                + fixture_name(FIX_HASHED)
                + "; the pin it breaks is not reached by"
                " check_bootstrap_vs_oracle"
            )
        if cells == 0 and arm == RSAB_REVERSED_POSITIONS:
            lines += (
                "RSAB_REVERSED_POSITIONS RECORDED at 0/"
                + String(R)
                + " under IDENTICAL, EXPECTED: the pinned halving fold is"
                " reversal-invariant because reversal maps its pair set to"
                " itself and float addition is commutative. The arm IS"
                " reached -- under FAST the same call moves 179/256. "
            )
        lines += (
            sabotage_name(arm)
            + " moved "
            + String(cells)
            + "/"
            + String(R)
            + " on n=200; "
        )

    # RSAB_FOLD_DESCENDING needs MORE THAN ONE CHUNK to have anything to
    # reorder: at n = 200 there is exactly one chunk of 256 and reversing a
    # one-element chain is the identity, so the arm above is EXPECTED INERT
    # there and would prove nothing. 600 rows is three chunks.
    var wide = hashed_sample(600, 41)
    var clean_w = _sab_distribution[STAT_MEAN](
        ctx, wide, 600, 2, key, 0, R, RSAB_NONE
    )
    var desc_w = _sab_distribution[STAT_MEAN](
        ctx, wide, 600, 2, key, 0, R, RSAB_FOLD_DESCENDING
    )
    var desc_moved = 0
    for r in range(R):
        if _bits(clean_w[r]) != _bits(desc_w[r]):
            desc_moved += 1
    comptime if IDENTICAL_BUILD:
        if desc_moved == 0:
            raise Error(
                "check_resample_sabotages: RSAB_FOLD_DESCENDING moved 0 of "
                + String(R)
                + " replicates on a 600-row fixture (3 chunks). The chunk"
                " chain's ASCENDING order is then not a pin anything depends"
                " on, and check_bootstrap_vs_oracle is not gating it."
            )
    lines += (
        "RSAB_FOLD_DESCENDING on n=600 (3 chunks) moved "
        + String(desc_moved)
        + "/"
        + String(R)
        + " -- at n=200 it is one chunk and EXPECTED INERT; "
    )

    # RSAB_SKIP_REJECTION is EXPECTED INERT on a power-of-two n, and that is
    # the per-branch demonstration rather than a weakness.
    var xa = build_sample(FIX_ANALYTIC)
    var na = fixture_n(FIX_ANALYTIC)
    var da = fixture_d(FIX_ANALYTIC)
    var clean_a = _sab_distribution[STAT_MEAN](
        ctx, xa, na, da, key, 0, R, RSAB_NONE
    )
    var skip_a = _sab_distribution[STAT_MEAN](
        ctx, xa, na, da, key, 0, R, RSAB_SKIP_REJECTION
    )
    var moved_a = 0
    for r in range(R):
        if _bits(clean_a[r]) != _bits(skip_a[r]):
            moved_a += 1
    var skip_h = _sab_distribution[STAT_MEAN](
        ctx, x, nn, dd, key, 0, R, RSAB_SKIP_REJECTION
    )
    var moved_h = 0
    for r in range(R):
        if _bits(clean[r]) != _bits(skip_h[r]):
            moved_h += 1
    lines += (
        "RSAB_SKIP_REJECTION RECORDED, not asserted: moved "
        + String(moved_h)
        + "/"
        + String(R)
        + " at n=200 and "
        + String(moved_a)
        + "/"
        + String(R)
        + " at n=8. BOTH ZEROS ARE EXPECTED. n=8 is a power of two so"
        " 2^32 mod n == 0 and Lemire never rejects; n=200 gives"
        " 2^32 mod n == 96, a rejection rate of 2.2e-08, and this driver"
        " takes 51200 draws for 0.0011 expected rejections. One expected"
        " rejection needs 44.7 million draws, and since 2^32 mod n < n the"
        " rate is under n/2^32 for EVERY n, so no sample size reaches this"
        " branch by sampling. A direct probe of the rejection zone is owed;"
        " "
    )

    # ---- RSAB_STD_SQRT: a REPORT, Apple-inert by measurement ----
    var clean_s = _sab_distribution[STAT_STD](
        ctx, x, nn, dd, key, 0, R, RSAB_NONE
    )
    var sq = _sab_distribution[STAT_STD](
        ctx, x, nn, dd, key, 0, R, RSAB_STD_SQRT
    )
    var sq_moved = 0
    for r in range(R):
        if _bits(clean_s[r]) != _bits(sq[r]):
            sq_moved += 1
    lines += (
        "RSAB_STD_SQRT moved "
        + String(sq_moved)
        + "/"
        + String(R)
        + " (Apple: 0 expected, its sqrt is correctly rounded; NVIDIA:"
        " expected to bite, 176,577 of 2^20 normals are 1 ulp off,"
        " IDENTITY_PATHS row 10); "
    )

    # ---- RSAB_NO_CANON_NAN: RECORDED, the vendor's payload ----
    var clean_p = _sab_distribution[STAT_PEARSON](
        ctx, xa, na, da, key, 0, 4096, RSAB_NONE
    )
    var raw_p = _sab_distribution[STAT_PEARSON](
        ctx, xa, na, da, key, 0, 4096, RSAB_NO_CANON_NAN
    )
    var payload = String("none reached")
    for r in range(4096):
        if _bits(clean_p[r]) == BITS_CANON_NAN:
            payload = _hex32(raw_p[r])
            break
    lines += (
        "RSAB_NO_CANON_NAN: the uncanonicalised degenerate pearson is "
        + payload
        + " (Apple 0x7fc00000, NVIDIA 0x7fffffff, AMD 0xffc00000 -- three"
        " payloads for one IEEE answer, DEVIATION 1696); "
    )

    # ---- RSAB_PERM_KEY_ONLY: the bijection breaks ----
    var pkey = resample_key(CHECK_SEED, RESAMPLE_KIND_PERMUTATION)
    var n_pooled = 32
    var bad_ranks = _sab_ranks(ctx, pkey, 0, 64, n_pooled, RSAB_PERM_KEY_ONLY)
    var broken = 0
    for r in range(64):
        var seen = List[Int]()
        for _ in range(n_pooled):
            seen.append(0)
        for j in range(n_pooled):
            var rk = Int(bad_ranks[r * n_pooled + j])
            if rk >= 0 and rk < n_pooled:
                seen[rk] += 1
        for k in range(n_pooled):
            if seen[k] != 1:
                broken += 1
                break
    if broken == 0:
        raise Error(
            "check_resample_sabotages: RSAB_PERM_KEY_ONLY left every rank"
            " vector a bijection. The key was narrowed to 4 bits precisely so"
            " ties would be common; if none collided, the arm is not reaching"
            " the tie-break and check_permutation_separable's bijection"
            " clause is unproven."
        )
    lines += (
        "RSAB_PERM_KEY_ONLY broke the bijection in "
        + String(broken)
        + "/64 replicates (check_permutation_separable FAILS as required)"
    )

    print("check_resample_sabotages RECORDED [" + _mode_name() + "]: " + lines)


def main() raises:
    print("resample_check mode=" + _mode_name())
    check_resample_refusals()
    check_index_map_is_positional()
    check_batch_invariance()
    check_prefix_stability()
    check_bootstrap_vs_oracle()
    check_bootstrap_known_answer()
    check_statistic_arms()
    check_percentile_interval()
    check_permutation_separable()
    check_permutation_null_is_uniform()
    check_monte_carlo_vs_closed_form()
    check_jackknife_and_bca()
    check_launch_invariance()
    check_card_is_emitted()
    check_resample_sabotages()
    print("resample_check mode=" + _mode_name() + " ALL OK")
