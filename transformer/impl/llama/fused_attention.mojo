# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fused attention for profile `mojolearn.identical.transformer.fp32.v1`:
the SAME arithmetic as the eager seams S11-S19 (forward) and stages 17-24
(backward), in the SAME fold order, with nothing materialized in HBM.

THE DESIGN CONSTRAINT IS THAT NO BIT MOVES. The eager path records seven
`[B, n_heads, L, S]` stages per forward and four per backward; at the Samba
shape (d_model 1024, 16 heads, window 2048, seq 4096, batch 4) each one is
1 GB and the eager forward spends its time streaming them. This file
removes the traffic by RECOMPUTATION and not by rescaling: there is no
online softmax here, because an online softmax changes the fold and the
fold is the contract.

    pass 1 (max)     the row maximum, `identical_fmax` over the row's
                     VISIBLE cells (contract 5.1: the fold shape is free)
    pass 2 (denom)   scores recomputed, `exp(s - m)`, the SERIAL ASCENDING
                     chain from `+0.0` (contract 5.3)
    pass 3 (ctx)     scores recomputed, `e / denom` ONE division each, the
                     SERIAL ASCENDING fma chain from `+0.0` (contract 7.2)

Each pass is one sweep over the row's visible key range with the key and
value tiles staged through threadgroup memory; the score itself is the
gemm profile's one-leaf chain (`head_dim <= CONTRACT_K_LEAF_MIN`, so
`P == 1` and the fold is the ascending fma chain seeded `+0.0`, contract
6 and 7.3), spelled with the same per-step seam the tuned GEMM plans use
(NVIDIA rounds the FMA then flushes via hardware multiply-by-one; other
columns use the software seam).

WHY THE MASKED CELLS MAY BE SKIPPED, AND WHEN THEY MAY NOT. A masked cell
is `ftz(s + (-FLT_MAX))`, which is exactly `-FLT_MAX` whenever
`|s| < 2^102`; its `exp` is exactly `+0.0`, its weight is exactly `+0.0`,
and every chain step it contributes is `acc + (+-0.0)`, which is `acc`
unless `acc` is `-0.0`. So the fused kernels skip the masked cells under
two conditions the host and the kernel check rather than assume:

  1. THE REGIME (host, before the launch): every operand finite and
     `head_dim * max|q| * max|k| < 2^100`, so no score can reach the
     magnitude at which `s + (-FLT_MAX)` stops being `-FLT_MAX`; for the
     backward also `head_dim * max|dctx| * max|v| < 2^100`. Outside it the
     caller runs the eager path, which computes every cell.
  2. THE CORNER (kernel, per chain): a chain that holds `-0.0` when its
     visible run ends could be laundered to `+0.0` by a masked tail whose
     products are `+0.0` (a flushed subnormal product is how a chain
     reaches `-0.0`). The kernel does not reason about the tail; it sets a
     flag and the caller runs the eager path for the whole call.

Both fallbacks are EXACT by construction (the eager path is the profile)
and both are counted by the launcher's status, which the fused check
asserts on: a case built to hit the corner must report it.

WHAT IS NOT HERE. Plants (`transformer_fixture.ScorePlant`) are an eager
feature; a planted call takes the eager path. `head_dim` outside
`fused_supported_head_dim` takes the eager path (the leaf tree at
`head_dim > 128` is not spelled here). Every sabotage build takes the
eager path, because the sabotage arms test the eager spelling.

THE OPT-IN ARMS (DEVIATIONS 2525 to 2527, 2026-09-11, brief
`docs/lanes/BRIEF_attention_step_2026-09-11.md`) relax "nothing
materialized in HBM" where the shape makes it cheap: at the LM target
shape (batch 1, 12 heads, L 2048) a `[B, n_heads, L, S]` scratch is 201 MB
and the recomputation it replaces is 61 percent of the training step.
They keep every chain's terms and order and change only what is
recomputed and which thread holds which chain. The composed arm
`stash_tiled` is the shipped default since the 2026-09-11 H100 leg (a
bit-equal 1.47x on the lean target step on both corpora); the kernels
above remain the path for every head dim other than 64 and the reference
every arm is gated against. See the arm hook below. DEVIATION 2528 (the
second round, brief section 12) is a trial-build arm on top of
stash_tiled; the shipped build compiles none of its kernels.

`[[ALWAYS GPU-agnostic]]`: one source; the only vendor row read is
`lib_hardware_ftz_fma_for`, through the kernel matrix.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast, stack_allocation
from std.os import getenv
from std.sys import llvm_intrinsic
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.device_scan import (
    NONFINITE_NONE,
    device_first_nonfinite,
    nonfinite_partial_kernel,
)
from checks.kernel_matrix import (
    TARGET_COLUMN,
    attn_zdot_rows_per_block_for,
    lib_hardware_ftz_fma_for,
    lib_smem_page_fits_for,
)
from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_fmax,
    identical_mul_add,
)


# ===========================================================================
# STATUS CODES AND LIMITS
# ===========================================================================

comptime FUSED_RAN = 0
"""The fused kernels produced the output."""
comptime FUSED_REFUSED_REGIME = 1
"""The regime bound failed (or the head_dim is unsupported); nothing was
written and the caller must run the eager path."""
comptime FUSED_CORNER = 2
"""A chain ended its visible run holding `-0.0`; the caller must run the
eager path."""

comptime FUSED_THREADS = 256
"""Threads per block for the row-tiled kernels: `TQ * head_dim`."""

comptime NEG_ZERO_BITS: UInt32 = 0x80000000

comptime REGIME_BOUND: Float64 = 1267650600228229401496703205376.0
"""`2^100`. `head_dim * max|a| * max|b|` below this keeps every dot below
`2^102`, where `x + (-FLT_MAX)` is still exactly `-FLT_MAX` (the spacing of
Float32 at `FLT_MAX` is `2^104`)."""

comptime FUSED_HW_FTZ_FMA = lib_hardware_ftz_fma_for[TARGET_COLUMN]()
"""The tier term is gone: this lane builds IDENTICAL only (2026-09-10)."""


# ===========================================================================
# THE ATTENTION ARM HOOK (DEVIATIONS 2525 to 2527, 2026-09-11; brief
# docs/lanes/BRIEF_attention_step_2026-09-11.md).
#
# `-D MOJOLEARN_ATTN_ARM_TRIAL=1` (never on a shipped build) compiles every
# candidate arm of the fused forward and backward, clean and sabotage, and
# lets the HOST pick one
# per call from the environment, the `MOJOLEARN_KNN_SELECT_TRIAL` pattern:
# `fused_attention_arm_from_env` is read once per launcher call (the
# callers' signatures are fixed, so the launcher reads it itself; a getenv
# per layer is microseconds) and the harness passes an arm explicitly
# through `fused_forward_launch_arm` / `fused_backward_launch_arm` so two
# arms alternate inside one process.
#
#   MOJOLEARN_ATTN_ARM=baseline        the shipped kernels
#   MOJOLEARN_ATTN_ARM=bwd_stash       DEVIATION 2525: the backward
#                                      materializes y and dy once (zdot),
#                                      ds once (dq), and dq/dk/dv fold from
#                                      the stash; the same TQ = 4 / BJ = 4
#                                      block shapes as the shipped kernels
#   MOJOLEARN_ATTN_ARM=fwd_sstash      DEVIATION 2526: the forward keeps the
#                                      masked score from pass 1 and the exp
#                                      from pass 2 in a scratch, so passes
#                                      2 and 3 do not recompute the dot
#   MOJOLEARN_ATTN_ARM=bwd_stash_tiled DEVIATION 2527: bwd_stash's zdot, then
#                                      register-blocked 64 x 64 folds for
#                                      dq and for dk/dv over the stash
#   MOJOLEARN_ATTN_ARM=stash           fwd_sstash + bwd_stash
#   MOJOLEARN_ATTN_ARM=stash_tiled     fwd_sstash + bwd_stash_tiled
#   MOJOLEARN_ATTN_ARM=stash_tiled_ztiled
#                                      DEVIATION 2528 (second round): the
#                                      zdot becomes a register-blocked y/dy
#                                      kernel plus a row z fold, rows per
#                                      block from the kernel-matrix row
#                                      `attn_zdot_rows_per_block_for`;
#                                      `_r32` / `_r64` force the geometry
#                                      (bwd_stash_tiled_ztiled likewise)
#   unset or empty                     the build default, ATTN_ARM_DEFAULT
#   anything else                      RAISES; the harness relies on it
#   MOJOLEARN_ATTN_ARM_SABOTAGE=1      the chosen arm's SABOTAGE
#                                      instantiation (reach proof; each
#                                      candidate kernel flips one ulp of a
#                                      value only that kernel produces)
#   MOJOLEARN_ATTN_ARM_SABOTAGE=new    the second-round kernels' sabotage
#                                      only (ATTN_ARM_SABOTAGE_NEW)
# The name grammar and `fused_attention_arm_parse`, its inverse, are in
# brief section 12.1.
#
# ATTN_ARM_DEFAULT is what the shipped build runs; it reads no environment.
# It was `baseline` until the 2026-09-11 H100 leg flipped it to stash_tiled
# (see its docstring); a shipped build compiles only the default arm's
# clean kernels (ATTN_ARM_COMPILED). The arms instantiate at head_dim 64 (the
# target shape); any other head dim takes the shipped kernels under every
# arm, and the status line of the launchers says nothing about it, so the
# harness asserts the arm's own witness (the sabotage flip) rather than
# the status.
#
# `-D MOJOLEARN_ATTN_PHASE_TIMERS=1` compiles per-kernel timers into both
# launchers; they print under the SAME run-time switch as the block timers
# (`MOJOLEARN_TRANSFORMER_TIMING=1`), one `timing attn.<kernel> <ms> ms`
# line per launch on fd 1, so the LM step's `--component-timing` step folds
# them into result.json with no harness change (the probe already excludes
# the `attn.` prefix from its total: these are sub-phases of `attn.core`
# and `bwd.attention`). A timer build synchronizes around every launch:
# its numbers are a breakdown, never a request price.
# ===========================================================================
comptime ATTN_ARM_TRIAL = is_defined["MOJOLEARN_ATTN_ARM_TRIAL"]()
comptime ATTN_PHASE_TIMERS = is_defined["MOJOLEARN_ATTN_PHASE_TIMERS"]()

comptime ATTN_ARM_BASELINE = 0
comptime ATTN_ARM_FWD_SSTASH = 1
"""Bit: the forward score/exp stash (DEVIATION 2526)."""
comptime ATTN_ARM_BWD_STASH = 2
"""Bit: the backward y/dy/ds stash with the shipped block shapes (DEVIATION 2525)."""
comptime ATTN_ARM_BWD_TILED = 4
"""Bit: register-blocked dq and dk/dv folds over the stash (DEVIATION 2527); implies ATTN_ARM_BWD_STASH."""
comptime ATTN_ARM_SABOTAGE = 16
"""OR'd into the arm value; the launchers strip it. Flips only the
first-round kernels (DEVIATIONS 2525 to 2527)."""
comptime ATTN_ARM_BWD_ZTILED = 8
"""Bit: DEVIATION 2528 (trial builds only), zdot as the register-blocked
y/dy kernel plus the row z fold; needs ATTN_ARM_BWD_STASH and
ATTN_ARM_BWD_TILED (dq and dk/dv stay the 2527 tiled folds)."""
# Bits 32 (FWD_QRES, DEVIATION 2530) and 64 (FWD_GRID, DEVIATION 2531) are
# reserved by brief section 11.8 and have no kernel in this file yet.
comptime ATTN_ARM_SABOTAGE_NEW = 128
"""OR'd into the arm value: the SECOND-round kernels' sabotage only, so
reach on top of stash_tiled names the new kernel and nothing else."""
comptime ATTN_ARM_ZROWS32 = 256
"""Trial geometry knob: DEVIATION 2528 at 32 query rows per block, on any
column (arm-name token `_r32`)."""
comptime ATTN_ARM_ZROWS64 = 512
"""Trial geometry knob: DEVIATION 2528 at 64 query rows per block, on any
column (arm-name token `_r64`)."""
comptime ATTN_ARM_BASE_BITS = ATTN_ARM_FWD_SSTASH | ATTN_ARM_BWD_STASH | ATTN_ARM_BWD_TILED
comptime ATTN_ARM_NEW_BITS = ATTN_ARM_BWD_ZTILED
"""The second-round kernel bits; an arm carrying any of them proves reach
with ATTN_ARM_SABOTAGE_NEW."""
comptime ATTN_ARM_NEW_FWD_BITS = 0
"""The second-round bits that touch the forward (none built yet); a
sabotage_new run of an arm without them must leave ctx, amax and denom
alone."""
comptime ATTN_ARM_DEFAULT = ATTN_ARM_FWD_SSTASH | ATTN_ARM_BWD_STASH | ATTN_ARM_BWD_TILED
"""FLIPPED 2026-09-11 to stash_tiled (DEVIATIONS 2525 to 2527): H100 leg
bench/results/e1g/2026-09-11_113013-nvidia-h100-attention-step, bit-equal
to the shipped kernels and to the eager oracle on real activations from
both corpora, lean target step 0.562/0.559 s -> 0.383/0.380 s
(shakespeare/cpython) with every step witness equal. A later leg that
flips again edits this line and the brief."""
comptime ATTN_ARM_COMPILED = ATTN_ARM_TRIAL or (ATTN_ARM_DEFAULT != ATTN_ARM_BASELINE)
"""Whether the launchers compile the arm kernels at all: on a trial build
(every arm, clean and sabotage) or when the build default is an arm (that
arm's clean kernels only; the sabotage instantiations stay trial-only)."""

comptime ATTN_STASH_HD = 64
"""The only head dim the candidate arms instantiate (the target shape)."""


def _attn_arm_base_from_name(base: String, full: String) raises -> Int:
    """The first-round base of an arm name (section 12.1)."""
    if base == "baseline":
        return ATTN_ARM_BASELINE
    if base == "bwd_stash":
        return ATTN_ARM_BWD_STASH
    if base == "fwd_sstash":
        return ATTN_ARM_FWD_SSTASH
    if base == "bwd_stash_tiled":
        return ATTN_ARM_BWD_STASH | ATTN_ARM_BWD_TILED
    if base == "stash":
        return ATTN_ARM_FWD_SSTASH | ATTN_ARM_BWD_STASH
    if base == "stash_tiled":
        return ATTN_ARM_FWD_SSTASH | ATTN_ARM_BWD_STASH | ATTN_ARM_BWD_TILED
    raise Error(
        "attention arm '" + full + "' is not an attention arm: the base '"
        + base + "' is not one of baseline, bwd_stash, fwd_sstash,"
        + " bwd_stash_tiled, stash, stash_tiled (then optionally _ztiled,"
        + " then _r32 or _r64, then +sabotage and/or +sabotage_new)"
    )


def fused_attention_arm_parse(name: String) raises -> Int:
    """The arm value an arm NAME spells; the exact inverse of
    `fused_attention_arm_name` over the valid arms (brief section 12.1).

    Grammar: base, then `_ztiled` (DEVIATION 2528), then `_r32` or `_r64`
    (the 2528 rows per block, forcing the kernel-matrix row), then
    `+sabotage` (first-round kernels) and `+sabotage_new` (second-round
    kernels). Raises on an unknown base, `_ztiled` without the tiled
    stash backward, or a rows token without `_ztiled`."""
    var rest = String(name)
    var arm = 0
    if rest.endswith("+sabotage_new"):
        arm = arm | ATTN_ARM_SABOTAGE_NEW
        rest = String(rest.removesuffix("+sabotage_new"))
    if rest.endswith("+sabotage"):
        arm = arm | ATTN_ARM_SABOTAGE
        rest = String(rest.removesuffix("+sabotage"))
    if rest.endswith("_r64"):
        arm = arm | ATTN_ARM_ZROWS64
        rest = String(rest.removesuffix("_r64"))
    elif rest.endswith("_r32"):
        arm = arm | ATTN_ARM_ZROWS32
        rest = String(rest.removesuffix("_r32"))
    if rest.endswith("_ztiled"):
        arm = arm | ATTN_ARM_BWD_ZTILED
        rest = String(rest.removesuffix("_ztiled"))
    arm = arm | _attn_arm_base_from_name(rest, name)
    comptime tiled_stash = ATTN_ARM_BWD_STASH | ATTN_ARM_BWD_TILED
    if (arm & ATTN_ARM_BWD_ZTILED) != 0 and (arm & tiled_stash) != tiled_stash:
        raise Error(
            "attention arm '" + name + "': _ztiled (DEVIATION 2528) needs the"
            + " tiled stash backward (bwd_stash_tiled or stash_tiled)"
        )
    if (arm & (ATTN_ARM_ZROWS32 | ATTN_ARM_ZROWS64)) != 0 and (arm & ATTN_ARM_BWD_ZTILED) == 0:
        raise Error(
            "attention arm '" + name + "': _r32 / _r64 set the rows of"
            + " DEVIATION 2528 and need _ztiled"
        )
    return arm


def fused_attention_arm_from_env() raises -> Int:
    """The attention arm for THIS call, read on the host.

    Trial builds read `MOJOLEARN_ATTN_ARM` (any name
    `fused_attention_arm_parse` accepts; unset = the build default,
    anything else raises) and `MOJOLEARN_ATTN_ARM_SABOTAGE` (exactly "1"
    sets ATTN_ARM_SABOTAGE, exactly "new" sets ATTN_ARM_SABOTAGE_NEW).
    Every other build returns ATTN_ARM_DEFAULT without reading the
    environment at all."""
    comptime if not ATTN_ARM_TRIAL:
        return ATTN_ARM_DEFAULT
    var name = String(getenv("MOJOLEARN_ATTN_ARM"))
    var arm: Int
    if name == "":
        arm = ATTN_ARM_DEFAULT
    else:
        arm = fused_attention_arm_parse(name)
    var sab = String(getenv("MOJOLEARN_ATTN_ARM_SABOTAGE"))
    if sab == "1":
        arm = arm | ATTN_ARM_SABOTAGE
    elif sab == "new":
        arm = arm | ATTN_ARM_SABOTAGE_NEW
    return arm


def fused_attention_arm_name(arm: Int) -> String:
    """The name `fused_attention_arm_parse` reads back as `arm` (section
    12.1). A bit this function does not know is spelled `_bits<N>`, and
    an invalid first-round combination `arm<N>`; the parser refuses both,
    so a name printed beside a timing is never a different arm's name."""
    var base = arm & ATTN_ARM_BASE_BITS
    var name = String("baseline")
    if base == ATTN_ARM_BWD_STASH:
        name = String("bwd_stash")
    elif base == ATTN_ARM_FWD_SSTASH:
        name = String("fwd_sstash")
    elif base == (ATTN_ARM_BWD_STASH | ATTN_ARM_BWD_TILED):
        name = String("bwd_stash_tiled")
    elif base == (ATTN_ARM_FWD_SSTASH | ATTN_ARM_BWD_STASH):
        name = String("stash")
    elif base == (ATTN_ARM_FWD_SSTASH | ATTN_ARM_BWD_STASH | ATTN_ARM_BWD_TILED):
        name = String("stash_tiled")
    elif base != ATTN_ARM_BASELINE:
        name = String("arm") + String(base)
    if (arm & ATTN_ARM_BWD_ZTILED) != 0:
        name += "_ztiled"
    if (arm & ATTN_ARM_ZROWS32) != 0:
        name += "_r32"
    if (arm & ATTN_ARM_ZROWS64) != 0:
        name += "_r64"
    var known = (
        ATTN_ARM_BASE_BITS | ATTN_ARM_BWD_ZTILED | ATTN_ARM_ZROWS32
        | ATTN_ARM_ZROWS64 | ATTN_ARM_SABOTAGE | ATTN_ARM_SABOTAGE_NEW
    )
    var other = arm - (arm & known)
    if other != 0:
        name += "_bits" + String(other)
    if (arm & ATTN_ARM_SABOTAGE) != 0:
        name += "+sabotage"
    if (arm & ATTN_ARM_SABOTAGE_NEW) != 0:
        name += "+sabotage_new"
    return name


def fused_attention_arm_reach_bit(arm: Int) -> Int:
    """The sabotage bit that proves reach for `arm`: ATTN_ARM_SABOTAGE_NEW
    when the arm carries a second-round kernel bit (so the proof names the
    new kernel, not the stash_tiled kernels under it), else
    ATTN_ARM_SABOTAGE."""
    if (arm & ATTN_ARM_NEW_BITS) != 0:
        return ATTN_ARM_SABOTAGE_NEW
    return ATTN_ARM_SABOTAGE


comptime ATTN_ZT_BK = 64
"""Keys per block iteration of `fused_bwd_ydy_tiled_kernel` (DEVIATION 2528)."""
comptime ATTN_ZT_KS = 16
"""The p-window width of the same kernel (the hd-64 forward's KS)."""


def _ydy_tiled_page_bytes(rows: Int) -> Int:
    """`fused_bwd_ydy_tiled_kernel`'s one shared page: Q and dctx for
    `rows` query rows, K and V for ATTN_ZT_BK keys, at stride KS + 4."""
    return (2 * rows + 2 * ATTN_ZT_BK) * (ATTN_ZT_KS + 4) * 4


comptime ATTN_ZT_FITS_32 = lib_smem_page_fits_for[TARGET_COLUMN, _ydy_tiled_page_bytes(32)]()
comptime ATTN_ZT_FITS_64 = lib_smem_page_fits_for[TARGET_COLUMN, _ydy_tiled_page_bytes(64)]()
comptime ATTN_ZDOT_ROWS_COLUMN = attn_zdot_rows_per_block_for[TARGET_COLUMN]()
"""The column's SCHEDULING row for DEVIATION 2528 (kernel matrix)."""


def fused_attention_zdot_rows(arm: Int) -> Int:
    """Query rows per block DEVIATION 2528 runs for `arm`: the `_r32` /
    `_r64` knob when the name forces one, else the column's row; 0 when
    that page does not fit the column (the launcher then runs the
    first-round stash kernels, and the harness prints the 0)."""
    var rows = ATTN_ZDOT_ROWS_COLUMN
    if (arm & ATTN_ARM_ZROWS32) != 0:
        rows = 32
    elif (arm & ATTN_ARM_ZROWS64) != 0:
        rows = 64
    if rows == 32 and ATTN_ZT_FITS_32:
        return 32
    if rows == 64 and ATTN_ZT_FITS_64:
        return 64
    return 0


@always_inline
def _attn_timer_on() -> Bool:
    """The per-kernel timers: compiled by `MOJOLEARN_ATTN_PHASE_TIMERS`,
    switched on by the block timers' run-time switch."""
    comptime if ATTN_PHASE_TIMERS:
        return String(getenv("MOJOLEARN_TRANSFORMER_TIMING")) != ""
    return False


def _attn_tick(ctx: DeviceContext, on: Bool, mut t: Int, name: String) raises:
    """Synchronize, print `timing attn.<name> <ms> ms`, advance `t`. The
    same line shape as `modeling_llama.timing_tick` (which this file cannot
    import: that module imports this one)."""
    if not on:
        return
    ctx.synchronize()
    var now = Int(perf_counter_ns())
    print(
        "timing attn." + name + " " + String(Float64(now - t) / 1000000.0)
        + " ms"
    )
    t = now


comptime ATTN_OPERAND_DUMP = is_defined["MOJOLEARN_ATTN_OPERAND_DUMP"]()
"""`-D MOJOLEARN_ATTN_OPERAND_DUMP=1` (never on a shipped build): the
backward launcher writes the operands of its FIRST call in the process
(the last layer of the first step: `q_rope`, `k_cache`, `v_cache`,
`d_attn_ctx`, as the launcher receives them, plus `meta.txt` with the
call's geometry) into `MOJOLEARN_ATTN_OPERAND_DUMP_DIR`, so the attention
microbenchmark can price the arms on activations the real training path
produced on a real corpus (ENGINEERING_RULES section 9). Nothing is
written when the directory is unset or `meta.txt` already exists there;
the dump is outside every timed region the probe reports (it precedes
the regime scans, and only the one call pays it)."""


def _file_exists(path: String) -> Bool:
    try:
        var f = open(path, "r")
        f.close()
        return True
    except:
        return False


def _dump_f32(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int, path: String) raises:
    """`n` floats of a device buffer as little-endian bytes, `.bin`."""
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var raw = host.unsafe_ptr().bitcast[UInt8]()
    var bytes = List[UInt8]()
    for i in range(n * 4):
        bytes.append(raw.unsafe_load(i))
    with open(path, "w") as fh:
        fh.write_bytes(Span(bytes))
    _ = host^


def _maybe_dump_operands(
    ctx: DeviceContext,
    mut q_rope: DeviceBuffer[DType.float32],
    mut dctx: DeviceBuffer[DType.float32],
    mut k_cache: DeviceBuffer[DType.float32],
    mut v_cache: DeviceBuffer[DType.float32],
    b: Int, l: Int, nh: Int, nkv: Int, hd: Int, s: Int, pos0: Int,
    key_lo: Int, window: Int, scale: Float32,
) raises:
    """See `ATTN_OPERAND_DUMP`. `meta.txt` is one line of space-separated
    integers `b l nh nkv hd s pos0 key_lo window` followed by the scale's
    hex bits, then one line per file."""
    var d = String(getenv("MOJOLEARN_ATTN_OPERAND_DUMP_DIR"))
    if d == "":
        return
    var meta = d + "/meta.txt"
    if _file_exists(meta):
        return
    _dump_f32(ctx, q_rope, b * l * nh * hd, d + "/q.bin")
    _dump_f32(ctx, dctx, b * l * nh * hd, d + "/dctx.bin")
    _dump_f32(ctx, k_cache, b * nkv * s * hd, d + "/k.bin")
    _dump_f32(ctx, v_cache, b * nkv * s * hd, d + "/v.bin")
    var bits = bitcast[DType.uint32](scale)
    var line = (
        String(b) + " " + String(l) + " " + String(nh) + " " + String(nkv)
        + " " + String(hd) + " " + String(s) + " " + String(pos0) + " "
        + String(key_lo) + " " + String(window) + " " + String(Int(bits)) + "\n"
        + "q.bin [B*L][nh*hd] float32 little-endian\n"
        + "dctx.bin [B*L][nh*hd] float32 little-endian\n"
        + "k.bin [B][nkv][S][hd] float32 little-endian\n"
        + "v.bin [B][nkv][S][hd] float32 little-endian\n"
    )
    with open(meta, "w") as fh:
        fh.write(line)


@always_inline
def _flip_ulp(x: Float32) -> Float32:
    """The sabotage perturbation: one ulp of a nonzero value (sign and
    exponent untouched); a zero (an underflowed weight, common) becomes
    `2^-100`, a normal, so the flip cannot be laundered by the next `ftz`
    the way a one-ulp flip of `+0.0` (the smallest subnormal) would be."""
    var b = bitcast[DType.uint32](x)
    if (b & UInt32(0x7FFFFFFF)) == UInt32(0):
        return bitcast[DType.float32](UInt32(0x0D800000))
    return bitcast[DType.float32](b ^ UInt32(1))


def _fused_page_bytes(hd: Int) -> Int:
    """The largest shared page any of the four fused kernels claims per
    block at `hd`, in bytes: the kernels' `stack_allocation` sizes spelled
    once, host-side, so the column's shared limit can be asked BEFORE a
    pipeline is created. Forward: `ks BK*(HD+1) + es TQ*(BK+1) + mp, mv
    TQ*HD each + rs TQ`; zdot and dq: `ks, vs BKb*(HD+1) each + ys, dys
    TQ*(BKb+1) each`; dkdv: the same with `TT == BKb` and `BJ == TQ`. At
    `hd == 128` that is 35,600 (forward) and 33,552 (backward) bytes, over
    a 32 KB column; at 64 it is 19,744."""
    var tq = FUSED_THREADS // hd
    var bk_f = hd if hd <= 64 else 64
    var fwd = bk_f * (hd + 1) + tq * (bk_f + 1) + 2 * tq * hd + tq
    var half = hd // 2
    var bk_b = half if half <= 32 else 32
    var bwd = 2 * bk_b * (hd + 1) + 2 * tq * (bk_b + 1)
    var m = fwd
    if bwd > m:
        m = bwd
    return m * 4


comptime FUSED_FITS_16 = lib_smem_page_fits_for[TARGET_COLUMN, _fused_page_bytes(16)]()
comptime FUSED_FITS_24 = lib_smem_page_fits_for[TARGET_COLUMN, _fused_page_bytes(24)]()
comptime FUSED_FITS_64 = lib_smem_page_fits_for[TARGET_COLUMN, _fused_page_bytes(64)]()
comptime FUSED_FITS_128 = lib_smem_page_fits_for[TARGET_COLUMN, _fused_page_bytes(128)]()


def fused_supported_head_dim(hd: Int) -> Bool:
    """The head dims this file instantiates kernels for, AND whose shared
    page fits the column (`lib_smem_page_fits_for`, kernel matrix). All are
    at or below `CONTRACT_K_LEAF_MIN` (128), so the score is the one-leaf
    chain. A head dim that does not fit takes the eager path, which is the
    same bits by construction; found by the Apple RUN OWED 2026-09-09
    (`hd128_win20_l70`: Metal refused the 35,600-byte forward page)."""
    return (
        (hd == 16 and FUSED_FITS_16)
        or (hd == 24 and FUSED_FITS_24)
        or (hd == 64 and FUSED_FITS_64)
        or (hd == 128 and FUSED_FITS_128)
    )


def fused_forward_supported_head_dim(hd: Int) -> Bool:
    """Forward hd128 uses a smaller register-blocked page. Backward keeps
    its original support predicate and takes eager when its page cannot
    fit; forward and backward independently decide their exact fallback."""
    if hd == 128:
        return lib_smem_page_fits_for[TARGET_COLUMN, 18624]()
    return fused_supported_head_dim(hd)


def fused_rows_per_block(hd: Int) -> Int:
    return FUSED_THREADS // hd


@always_inline
def _step(a: Float32, b: Float32, acc: Float32) -> Float32:
    """`ftz(fma(ftz(a), ftz(b), acc))`, the per-term seam of every chain in
    the profile (gemm contract 4 + 5c; S19; the backward chains).
    NVIDIA rounds FMA without FTZ, then flushes the rounded result via
    hardware multiply-by-one. A single hardware FTZ FMA instead flushes
    before rounding at some smallest-normal boundaries. This matches
    NVIDIA software; other columns' FMA boundary behavior is a separate
    numerical audit.
    """
    comptime if FUSED_HW_FTZ_FMA:
        var rounded = llvm_intrinsic[
            "llvm.nvvm.fma.rn.f", Float32, has_side_effect=False
        ](ftz(a), ftz(b), acc)
        return llvm_intrinsic[
            "llvm.nvvm.mul.rn.ftz.f", Float32, has_side_effect=False
        ](rounded, Float32(1.0))
    return ftz(identical_mul_add(ftz(a), ftz(b), acc))


@always_inline
def _step_preflushed(a: Float32, b: Float32, acc: Float32) -> Float32:
    """Ascending RN-then-flush chain on operands already flushed on staging."""
    comptime if FUSED_HW_FTZ_FMA:
        var rounded = llvm_intrinsic[
            "llvm.nvvm.fma.rn.f", Float32, has_side_effect=False
        ](a, b, acc)
        return llvm_intrinsic[
            "llvm.nvvm.mul.rn.ftz.f", Float32, has_side_effect=False
        ](rounded, Float32(1.0))
    return ftz(identical_mul_add(a, b, acc))


@always_inline
def _pmul(a: Float32, b: Float32) -> Float32:
    """`ftz(pinned_mul(ftz(a), ftz(b)))`: one rounding, `-0.0` addend."""
    return _step(a, b, Float32(-0.0))


@always_inline
def _row_range(
    t: Int, pos0: Int, key_lo: Int, window: Int, s: Int
) -> Tuple[Int, Int]:
    """The packed key indices `[lo, hi]` the query at row `t` sees: exactly
    `attn_mask_kernel`'s predicate, solved for `j`."""
    var p_q = pos0 + t
    var hi = p_q - key_lo
    if hi > s - 1:
        hi = s - 1
    var lo = 0
    if window > 0:
        lo = p_q - window + 1 - key_lo
        if lo < 0:
            lo = 0
    return (lo, hi)


@always_inline
def _key_query_range(
    j: Int, pos0: Int, key_lo: Int, window: Int, l: Int
) -> Tuple[Int, Int]:
    """The query rows `[lo, hi]` that see packed key `j` (may be empty,
    `lo > hi`)."""
    var p_k = key_lo + j
    var lo = p_k - pos0
    if lo < 0:
        lo = 0
    var hi = l - 1
    if window > 0:
        var w_hi = p_k - pos0 + window - 1
        if w_hi < hi:
            hi = w_hi
    return (lo, hi)


# ===========================================================================
# THE REGIME BOUND: max |x| over a buffer, NaN as +inf
# ===========================================================================

comptime ABSMAX_TPB = 256
comptime ABSMAX_BLOCKS = 512


def absmax_partial_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    buf: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """One partial per block: the plain `max` of `|x|` with NaN read as
    `+inf`. A BOUND, not a profile value: nothing here reaches the card."""
    var n = Int(n_in)
    var red = stack_allocation[
        ABSMAX_TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    var stride = Int(grid_dim.x) * ABSMAX_TPB
    var i = Int(block_idx.x) * ABSMAX_TPB + tid
    var m = Float32(0.0)
    while i < n:
        var v = buf.unsafe_load(i)
        if v != v:
            v = bitcast[DType.float32](UInt32(0x7F800000))
        if v < Float32(0.0):
            v = -v
        if v > m:
            m = v
        i += stride
    red.unsafe_store(tid, m)
    barrier()
    var active = ABSMAX_TPB // 2
    while active > 0:
        if tid < active:
            var o = red.unsafe_load(tid + active)
            if o > red.unsafe_load(tid):
                red.unsafe_store(tid, o)
        barrier()
        active = active // 2
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), red.unsafe_load(0))


def device_absmax(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> Float64:
    """`max |buf[0:n]|` as a Float64, `+inf` if any element is not finite."""
    if n <= 0:
        return 0.0
    var blocks = (n + ABSMAX_TPB - 1) // ABSMAX_TPB
    if blocks > ABSMAX_BLOCKS:
        blocks = ABSMAX_BLOCKS
    var part = ctx.enqueue_create_buffer[DType.float32](blocks)
    ctx.synchronize()
    ctx.enqueue_function[absmax_partial_kernel](
        part.unsafe_ptr(),
        buf.unsafe_ptr(),
        Int32(n),
        grid_dim=(blocks, 1, 1),
        block_dim=(ABSMAX_TPB, 1, 1),
    )
    ctx.synchronize()
    var host = ctx.enqueue_create_host_buffer[DType.float32](blocks)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=part)
    ctx.synchronize()
    var m = Float32(0.0)
    for i in range(blocks):
        var v = host.unsafe_ptr().unsafe_load(i)
        if v > m:
            m = v
    _ = host^
    _ = part^
    return Float64(m)


# DEVIATION 2514 step 1 (2026-09-11): `NONFINITE_NONE`,
# `nonfinite_partial_kernel` and `device_first_nonfinite` MOVED to
# `core/device_scan.mojo` so the training lane can scan without importing
# this file. They are re-imported at the top of this module, so every
# caller that spelled `from transformer.impl.llama.fused_attention import
# device_first_nonfinite` still resolves; the kernel, geometry and fold are
# unchanged.


def regime_product_ok(hd: Int, a_max: Float64, b_max: Float64) -> Bool:
    """`hd * a_max * b_max < 2^100`, both finite."""
    var inf = Float64(bitcast[DType.float32](UInt32(0x7F800000)))
    if not (a_max < inf) or not (b_max < inf):
        return False
    return Float64(hd) * a_max * b_max < REGIME_BOUND


def regime_finite(x_max: Float64) -> Bool:
    var inf = Float64(bitcast[DType.float32](UInt32(0x7F800000)))
    return x_max < inf


# ===========================================================================
# THE FUSED FORWARD
# ===========================================================================


def fused_attn_forward_regblocked_kernel[HD: Int](
    ctxv: MutPointer[Float32, MutAnyOrigin],
    amax: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    corner: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    v_cache: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, l_in: Int32, nh_in: Int32, nkv_in: Int32,
    s_in: Int32, pos0_in: Int32, key_lo_in: Int32, window_in: Int32,
    scale_in: Float32,
):
    """Register tiles preserve each ascending HD-term dot and all three
    passes. hd64 uses 64 query rows and KS16; hd128 uses 16 rows and KS64
    to reduce live accumulators and score-window barriers. Staging is
    reused for V: 17,152 shared bytes at hd64, 18,624 at hd128."""
    comptime TQ = 16 if HD == 128 else 64
    comptime RPT = TQ // 16
    comptime BK = 32
    comptime KS = 64 if HD == 128 else 16
    comptime STRIDE = KS + 4
    var stg = stack_allocation[BK * HD, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tile = stack_allocation[TQ * 33, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var stats = stack_allocation[2 * TQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var tr = tid // 16
    var tc = tid % 16
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)
    var ntb = (l + TQ - 1) // TQ
    var raw = Int(block_idx.x)
    var t0 = (raw % ntb) * TQ
    var h = (raw // ntb) % nh
    var bb = raw // ntb // nh
    var kvbase = (bb * nkv + h // (nh // nkv)) * s * HD
    var t1 = min(t0 + TQ - 1, l - 1)
    var r0 = _row_range(t0, pos0, key_lo, window, s)
    var r1 = _row_range(t1, pos0, key_lo, window, s)
    var kb_lo = r0[0] // BK
    var kb_hi = r1[1] // BK
    var negmax = bitcast[DType.float32](UInt32(0xFF7FFFFF))
    var mpart = SIMD[DType.float32, RPT](negmax)
    var dacc = Float32(0.0)
    var cacc = SIMD[DType.float32, RPT * (HD // 16)](0.0)
    comptime for phase in range(3):
        for kb in range(kb_lo, kb_hi + 1):
            var dots = SIMD[DType.float32, RPT * 2](0.0)
            comptime for pw in range(HD // KS):
                # The two operand pages are padded along the contracted
                # dimension; loads serve RPT*2 independent fma chains.
                comptime for si in range((TQ + BK) * KS // 256):
                    var i = tid + si * 256
                    var r = i // KS
                    var p = i % KS
                    var x = Float32(0.0)
                    if r < TQ:
                        var t = t0 + r
                        if t < l:
                            x = ftz(q_rope.unsafe_load((bb * l + t) * nh * HD + h * HD + pw * KS + p))
                    else:
                        var j = kb * BK + r - TQ
                        if j < s:
                            x = ftz(k_cache.unsafe_load(kvbase + j * HD + pw * KS + p))
                    stg.unsafe_store(r * STRIDE + p, x)
                barrier()
                comptime for p in range(KS):
                    var qa = SIMD[DType.float32, RPT](0.0)
                    var ka = SIMD[DType.float32, 2](0.0)
                    comptime for u in range(RPT):
                        qa[u] = stg.unsafe_load((tr + u * 16) * STRIDE + p)
                    comptime for v in range(2):
                        ka[v] = stg.unsafe_load((TQ + tc + v * 16) * STRIDE + p)
                    comptime for u in range(RPT):
                        comptime for v in range(2):
                            dots[u * 2 + v] = _step_preflushed(qa[u], ka[v], dots[u * 2 + v])
                barrier()
            comptime for u in range(RPT):
                var r = tr + u * 16
                var t = t0 + r
                var rr = _row_range(t, pos0, key_lo, window, s)
                comptime for v in range(2):
                    var jj = tc + v * 16
                    var j = kb * BK + jj
                    if t < l and j >= rr[0] and j <= rr[1]:
                        var masked = ftz(_pmul(dots[u * 2 + v], scale_in) + Float32(0.0))
                        comptime if phase == 0:
                            mpart[u] = identical_fmax(mpart[u], masked)
                        else:
                            var e = ftz(identical_exp(ftz(ftz(masked) - ftz(stats.unsafe_load(r)))))
                            comptime if phase == 2:
                                e = ftz(identical_div(ftz(e), ftz(stats.unsafe_load(TQ + r))))
                            tile.unsafe_store(r * 33 + jj, e)
            barrier()
            comptime if phase == 1:
                if tid < TQ and t0 + tid < l:
                    var rr = _row_range(t0 + tid, pos0, key_lo, window, s)
                    comptime for jj in range(BK):
                        var j = kb * BK + jj
                        if j >= rr[0] and j <= rr[1]:
                            dacc = ftz(ftz(dacc) + ftz(tile.unsafe_load(tid * 33 + jj)))
            elif phase == 2:
                comptime for si in range(BK * HD // 256):
                    var i = tid + si * 256
                    var j = kb * BK + i // HD
                    var x = Float32(0.0)
                    if j < s:
                        x = ftz(v_cache.unsafe_load(kvbase + j * HD + i % HD))
                    stg.unsafe_store(i, x)
                barrier()
                comptime for jj in range(BK):
                    var j = kb * BK + jj
                    var va = SIMD[DType.float32, HD // 16](0.0)
                    comptime for v in range(HD // 16):
                        va[v] = stg.unsafe_load(jj * HD + tc + v * 16)
                    comptime for u in range(RPT):
                        var r = tr + u * 16
                        var rr = _row_range(t0 + r, pos0, key_lo, window, s)
                        if t0 + r < l and j >= rr[0] and j <= rr[1]:
                            var w = tile.unsafe_load(r * 33 + jj)
                            comptime for v in range(HD // 16):
                                cacc[u * (HD // 16) + v] = _step(w, va[v], cacc[u * (HD // 16) + v])
            barrier()
        comptime if phase == 0:
            comptime for u in range(RPT):
                tile.unsafe_store((tr + u * 16) * 16 + tc, mpart[u])
            barrier()
            if tid < TQ and t0 + tid < l:
                var m = negmax
                comptime for c in range(16):
                    m = identical_fmax(m, tile.unsafe_load(tid * 16 + c))
                stats.unsafe_store(tid, m)
                amax.unsafe_store((bb * nh + h) * l + t0 + tid, m)
        elif phase == 1:
            if tid < TQ and t0 + tid < l:
                stats.unsafe_store(TQ + tid, ftz(dacc))
                denom.unsafe_store((bb * nh + h) * l + t0 + tid, ftz(dacc))
        barrier()
    comptime for u in range(RPT):
        var t = t0 + tr + u * 16
        var rr = _row_range(t, pos0, key_lo, window, s)
        if t < l:
            comptime for v in range(HD // 16):
                var x = cacc[u * (HD // 16) + v]
                if bitcast[DType.uint32](x) == NEG_ZERO_BITS and rr[1] < s - 1:
                    corner.unsafe_store(0, Float32(1.0))
                ctxv.unsafe_store((bb * l + t) * nh * HD + h * HD + tc + v * 16, x)


# ---------------------------------------------------------------------------
# DEVIATION 2526 (arm `fwd_sstash`): the forward with the score kept.
#
# THE READING. `fused_attn_forward_regblocked_kernel` sweeps the row tile's
# visible key range THREE times and recomputes the 64-term score chain in
# every sweep: per visible cell it spends three dots (each 64 RN-FMA +
# 64 FTZ-multiply seams), two `identical_exp`, one `identical_div` and the
# 64-term context chain, and its staging reloads the SAME 64 x 64 Q tile
# from global memory on every (key block, 16-wide p window) of every pass
# (two thirds of the staged floats are Q). The three passes are the
# contract (max, then the serial denominator, then the serial context
# chain, no online rescaling); the recomputation is not.
#
# THE MECHANISM. Pass 1 computes each visible cell's masked score exactly
# as the shipped kernel does and ALSO stores it to a `[B, n_heads, L, S]`
# scratch the launcher owns for the call (201 MB at the target shape).
# Pass 2 reads the score back, computes the exp with the same row maximum,
# stores the exp over the score (its own cell: no race) and folds the
# denominator as before. Pass 3 reads the exp back and divides. Passes 2
# and 3 stage nothing for the dots and run no dot; per visible cell the
# work is one dot, one exp, one div and the context chain.
#
# WHY NO BIT MOVES. The masked score of a cell is `ftz(pmul(dot, scale)
# + 0.0)` from one dot chain over p ascending; the shipped pass 2 and pass 3
# recompute that same chain from the same staged operands and land on the
# same bits, so reading the pass-1 bits back is the same value. The exp of
# pass 3 in the shipped kernel is `ftz(exp(ftz(ftz(masked) - ftz(m))))`
# with the pass-1 `m` (`stats[r]` is written once at the end of pass 1 and
# never again), which is the bits pass 2 stored. The folds (`dacc` over
# keys ascending, `cacc` over keys ascending with `_step`) read the same
# tile the shipped kernel reads, in the same order. `amax` and `denom` are
# produced by the same code. Only cells the shipped kernel computes are
# stored or read (the visibility test is copied), so a masked cell is
# never touched and the corner rule is unchanged.
#
# SABOTAGE (reach): the pass-1 store flips one ulp of the score, so the
# denominator, the weights and the context move while `amax` (from the
# register value) does not.
# ---------------------------------------------------------------------------
def fused_attn_forward_regblocked_sstash_kernel[HD: Int, SABOTAGE: Bool](
    ctxv: MutPointer[Float32, MutAnyOrigin],
    amax: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    corner: MutPointer[Float32, MutAnyOrigin],
    sstash: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    v_cache: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, l_in: Int32, nh_in: Int32, nkv_in: Int32,
    s_in: Int32, pos0_in: Int32, key_lo_in: Int32, window_in: Int32,
    scale_in: Float32,
):
    """`fused_attn_forward_regblocked_kernel` with the score of pass 1 and
    the exp of pass 2 kept in `sstash` (`[B, n_heads, L, S]`, the call's
    packed key stride); see the comment above."""
    comptime TQ = 16 if HD == 128 else 64
    comptime RPT = TQ // 16
    comptime BK = 32
    comptime KS = 64 if HD == 128 else 16
    comptime STRIDE = KS + 4
    var stg = stack_allocation[BK * HD, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tile = stack_allocation[TQ * 33, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var stats = stack_allocation[2 * TQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var tr = tid // 16
    var tc = tid % 16
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)
    var ntb = (l + TQ - 1) // TQ
    var raw = Int(block_idx.x)
    var t0 = (raw % ntb) * TQ
    var h = (raw // ntb) % nh
    var bb = raw // ntb // nh
    var kvbase = (bb * nkv + h // (nh // nkv)) * s * HD
    var stbase = (bb * nh + h) * l * s
    var t1 = min(t0 + TQ - 1, l - 1)
    var r0 = _row_range(t0, pos0, key_lo, window, s)
    var r1 = _row_range(t1, pos0, key_lo, window, s)
    var kb_lo = r0[0] // BK
    var kb_hi = r1[1] // BK
    var negmax = bitcast[DType.float32](UInt32(0xFF7FFFFF))
    var mpart = SIMD[DType.float32, RPT](negmax)
    var dacc = Float32(0.0)
    var cacc = SIMD[DType.float32, RPT * (HD // 16)](0.0)
    comptime for phase in range(3):
        for kb in range(kb_lo, kb_hi + 1):
            comptime if phase == 0:
                var dots = SIMD[DType.float32, RPT * 2](0.0)
                comptime for pw in range(HD // KS):
                    comptime for si in range((TQ + BK) * KS // 256):
                        var i = tid + si * 256
                        var r = i // KS
                        var p = i % KS
                        var x = Float32(0.0)
                        if r < TQ:
                            var t = t0 + r
                            if t < l:
                                x = ftz(q_rope.unsafe_load((bb * l + t) * nh * HD + h * HD + pw * KS + p))
                        else:
                            var j = kb * BK + r - TQ
                            if j < s:
                                x = ftz(k_cache.unsafe_load(kvbase + j * HD + pw * KS + p))
                        stg.unsafe_store(r * STRIDE + p, x)
                    barrier()
                    comptime for p in range(KS):
                        var qa = SIMD[DType.float32, RPT](0.0)
                        var ka = SIMD[DType.float32, 2](0.0)
                        comptime for u in range(RPT):
                            qa[u] = stg.unsafe_load((tr + u * 16) * STRIDE + p)
                        comptime for v in range(2):
                            ka[v] = stg.unsafe_load((TQ + tc + v * 16) * STRIDE + p)
                        comptime for u in range(RPT):
                            comptime for v in range(2):
                                dots[u * 2 + v] = _step_preflushed(qa[u], ka[v], dots[u * 2 + v])
                    barrier()
                comptime for u in range(RPT):
                    var r = tr + u * 16
                    var t = t0 + r
                    var rr = _row_range(t, pos0, key_lo, window, s)
                    comptime for v in range(2):
                        var jj = tc + v * 16
                        var j = kb * BK + jj
                        if t < l and j >= rr[0] and j <= rr[1]:
                            var masked = ftz(_pmul(dots[u * 2 + v], scale_in) + Float32(0.0))
                            mpart[u] = identical_fmax(mpart[u], masked)
                            comptime if SABOTAGE:
                                sstash.unsafe_store(stbase + t * s + j, _flip_ulp(masked))
                            else:
                                sstash.unsafe_store(stbase + t * s + j, masked)
            else:
                comptime for u in range(RPT):
                    var r = tr + u * 16
                    var t = t0 + r
                    var rr = _row_range(t, pos0, key_lo, window, s)
                    comptime for v in range(2):
                        var jj = tc + v * 16
                        var j = kb * BK + jj
                        if t < l and j >= rr[0] and j <= rr[1]:
                            var cell = stbase + t * s + j
                            comptime if phase == 1:
                                var masked = sstash.unsafe_load(cell)
                                var e = ftz(identical_exp(ftz(ftz(masked) - ftz(stats.unsafe_load(r)))))
                                sstash.unsafe_store(cell, e)
                                tile.unsafe_store(r * 33 + jj, e)
                            else:
                                var e = sstash.unsafe_load(cell)
                                tile.unsafe_store(r * 33 + jj, ftz(identical_div(ftz(e), ftz(stats.unsafe_load(TQ + r)))))
            barrier()
            comptime if phase == 1:
                if tid < TQ and t0 + tid < l:
                    var rr = _row_range(t0 + tid, pos0, key_lo, window, s)
                    comptime for jj in range(BK):
                        var j = kb * BK + jj
                        if j >= rr[0] and j <= rr[1]:
                            dacc = ftz(ftz(dacc) + ftz(tile.unsafe_load(tid * 33 + jj)))
            elif phase == 2:
                comptime for si in range(BK * HD // 256):
                    var i = tid + si * 256
                    var j = kb * BK + i // HD
                    var x = Float32(0.0)
                    if j < s:
                        x = ftz(v_cache.unsafe_load(kvbase + j * HD + i % HD))
                    stg.unsafe_store(i, x)
                barrier()
                comptime for jj in range(BK):
                    var j = kb * BK + jj
                    var va = SIMD[DType.float32, HD // 16](0.0)
                    comptime for v in range(HD // 16):
                        va[v] = stg.unsafe_load(jj * HD + tc + v * 16)
                    comptime for u in range(RPT):
                        var r = tr + u * 16
                        var rr = _row_range(t0 + r, pos0, key_lo, window, s)
                        if t0 + r < l and j >= rr[0] and j <= rr[1]:
                            var w = tile.unsafe_load(r * 33 + jj)
                            comptime for v in range(HD // 16):
                                cacc[u * (HD // 16) + v] = _step(w, va[v], cacc[u * (HD // 16) + v])
            barrier()
        comptime if phase == 0:
            comptime for u in range(RPT):
                tile.unsafe_store((tr + u * 16) * 16 + tc, mpart[u])
            barrier()
            if tid < TQ and t0 + tid < l:
                var m = negmax
                comptime for c in range(16):
                    m = identical_fmax(m, tile.unsafe_load(tid * 16 + c))
                stats.unsafe_store(tid, m)
                amax.unsafe_store((bb * nh + h) * l + t0 + tid, m)
        elif phase == 1:
            if tid < TQ and t0 + tid < l:
                stats.unsafe_store(TQ + tid, ftz(dacc))
                denom.unsafe_store((bb * nh + h) * l + t0 + tid, ftz(dacc))
        barrier()
    comptime for u in range(RPT):
        var t = t0 + tr + u * 16
        var rr = _row_range(t, pos0, key_lo, window, s)
        if t < l:
            comptime for v in range(HD // 16):
                var x = cacc[u * (HD // 16) + v]
                if bitcast[DType.uint32](x) == NEG_ZERO_BITS and rr[1] < s - 1:
                    corner.unsafe_store(0, Float32(1.0))
                ctxv.unsafe_store((bb * l + t) * nh * HD + h * HD + tc + v * 16, x)


def fused_attn_forward_kernel[HD: Int, TQ: Int](
    ctxv: MutPointer[Float32, MutAnyOrigin],
    amax: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    corner: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    v_cache: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
    key_lo_in: Int32,
    window_in: Int32,
    scale_in: Float32,
):
    """One block owns `TQ` consecutive query rows of one `(batch, head)`;
    thread `(row, lane)`. In the score passes lane `e < BK` owns key
    `kb * BK + e` of the current key block; in the context pass lane `d`
    owns output column `d`. Every thread reaches every `barrier()`: the key
    block loop bounds are block-uniform and the only early return is
    block-uniform and precedes every barrier.

    Seams, in the eager kernels' spelling: S11 `_step` chain over `p`
    seeded `+0.0` (the one-leaf gemm cell); S12 `_pmul(dot, scale)`; S13
    `ftz(sc + 0.0)`; S14 `identical_fmax`, any order; S15/S16
    `ftz(identical_exp(ftz(masked - m)))`; S17 `ftz(ftz(acc) + e)`
    ascending; S18 `ftz(identical_div(e, denom))`; S19 `_step(w, v, acc)`
    ascending."""
    comptime BK = HD if HD <= 64 else 64
    comptime NT = TQ * HD
    comptime KSTRIDE = HD + 1
    comptime ESTRIDE = BK + 1
    comptime SLOTS = (BK * HD + NT - 1) // NT

    var ks = stack_allocation[
        BK * KSTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var es = stack_allocation[
        TQ * ESTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var mp = stack_allocation[
        TQ * HD,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var mv = stack_allocation[
        TQ * HD,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var rs = stack_allocation[
        TQ,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)
    var n_rep = nh // nkv

    var ntb = (l + TQ - 1) // TQ
    var raw = Int(block_idx.x)
    var tb = raw % ntb
    var rest = raw // ntb
    var h = rest % nh
    var bb = rest // nh
    if bb >= b:
        return
    var kvh = h // n_rep

    var tid = Int(thread_idx.x)
    var tr = tid // HD
    var lane = tid - tr * HD
    var t = tb * TQ + tr
    var valid = t < l
    var tt = t
    if not valid:
        tt = l - 1
    var rr = _row_range(tt, pos0, key_lo, window, s)
    var j_lo = rr[0]
    var j_hi = rr[1]
    var t0 = tb * TQ
    var t1 = t0 + TQ - 1
    if t1 > l - 1:
        t1 = l - 1
    var r0 = _row_range(t0, pos0, key_lo, window, s)
    var r1 = _row_range(t1, pos0, key_lo, window, s)
    var kb_lo = r0[0] // BK
    var kb_hi = r1[1] // BK

    var kvbase = (bb * nkv + kvh) * s * HD
    var qbase = (bb * l + tt) * nh * HD + h * HD
    var row = (bb * nh + h) * l + tt

    var q = stack_allocation[HD, Scalar[DType.float32]]()
    comptime for p in range(HD):
        q.unsafe_store(p, ftz(q_rope.unsafe_load(qbase + p)))

    # ---- pass 1: the row maximum over the VISIBLE cells ---------------
    var mpart = Float32(0.0)
    var mvalid = False
    for kb in range(kb_lo, kb_hi + 1):
        comptime for si in range(SLOTS):
            var i = tid + si * NT
            if i < BK * HD:
                var r = i // HD
                var c = i - r * HD
                var j = kb * BK + r
                var v = Float32(0.0)
                if j < s:
                    v = ftz(k_cache.unsafe_load(kvbase + j * HD + c))
                ks.unsafe_store(r * KSTRIDE + c, v)
        barrier()
        if valid and lane < BK:
            var j = kb * BK + lane
            if j >= j_lo and j <= j_hi:
                var dot = Float32(0.0)
                comptime for p in range(HD):
                    dot = _step(q.unsafe_load(p), ks.unsafe_load(lane * KSTRIDE + p), dot)
                var sc = _pmul(dot, scale_in)
                var masked = ftz(sc + Float32(0.0))
                if mvalid:
                    mpart = identical_fmax(mpart, masked)
                else:
                    mpart = masked
                    mvalid = True
        barrier()
    mp.unsafe_store(tr * HD + lane, mpart)
    if mvalid:
        mv.unsafe_store(tr * HD + lane, Float32(1.0))
    else:
        mv.unsafe_store(tr * HD + lane, Float32(0.0))
    barrier()
    if valid and lane == 0:
        var m = Float32(0.0)
        var have = False
        for e in range(HD):
            if mv.unsafe_load(tr * HD + e) != Float32(0.0):
                var v = mp.unsafe_load(tr * HD + e)
                if have:
                    m = identical_fmax(m, v)
                else:
                    m = v
                    have = True
        amax.unsafe_store(row, m)
        rs.unsafe_store(tr, m)
    barrier()
    var m_row = rs.unsafe_load(tr)
    barrier()

    # ---- pass 2: the denominator, SERIAL ASCENDING from +0.0 -----------
    var dacc = Float32(0.0)
    for kb in range(kb_lo, kb_hi + 1):
        comptime for si in range(SLOTS):
            var i = tid + si * NT
            if i < BK * HD:
                var r = i // HD
                var c = i - r * HD
                var j = kb * BK + r
                var v = Float32(0.0)
                if j < s:
                    v = ftz(k_cache.unsafe_load(kvbase + j * HD + c))
                ks.unsafe_store(r * KSTRIDE + c, v)
        barrier()
        if valid and lane < BK:
            var j = kb * BK + lane
            if j >= j_lo and j <= j_hi:
                var dot = Float32(0.0)
                comptime for p in range(HD):
                    dot = _step(q.unsafe_load(p), ks.unsafe_load(lane * KSTRIDE + p), dot)
                var sc = _pmul(dot, scale_in)
                var masked = ftz(sc + Float32(0.0))
                var d = ftz(ftz(masked) - ftz(m_row))
                es.unsafe_store(tr * ESTRIDE + lane, ftz(identical_exp(d)))
        barrier()
        if valid and lane == 0:
            for jj in range(BK):
                var j = kb * BK + jj
                if j >= j_lo and j <= j_hi:
                    dacc = ftz(
                        ftz(dacc) + ftz(es.unsafe_load(tr * ESTRIDE + jj))
                    )
        barrier()
    if valid and lane == 0:
        denom.unsafe_store(row, ftz(dacc))
        rs.unsafe_store(tr, ftz(dacc))
    barrier()
    var d_row = rs.unsafe_load(tr)
    barrier()

    # ---- pass 3: weights and the context chain, SERIAL ASCENDING -------
    var cacc = Float32(0.0)
    for kb in range(kb_lo, kb_hi + 1):
        comptime for si in range(SLOTS):
            var i = tid + si * NT
            if i < BK * HD:
                var r = i // HD
                var c = i - r * HD
                var j = kb * BK + r
                var v = Float32(0.0)
                if j < s:
                    v = ftz(k_cache.unsafe_load(kvbase + j * HD + c))
                ks.unsafe_store(r * KSTRIDE + c, v)
        barrier()
        if valid and lane < BK:
            var j = kb * BK + lane
            if j >= j_lo and j <= j_hi:
                var dot = Float32(0.0)
                comptime for p in range(HD):
                    dot = _step(q.unsafe_load(p), ks.unsafe_load(lane * KSTRIDE + p), dot)
                var sc = _pmul(dot, scale_in)
                var masked = ftz(sc + Float32(0.0))
                var d = ftz(ftz(masked) - ftz(m_row))
                var e = ftz(identical_exp(d))
                es.unsafe_store(
                    tr * ESTRIDE + lane,
                    ftz(identical_div(ftz(e), ftz(d_row))),
                )
        barrier()
        comptime for si in range(SLOTS):
            var i = tid + si * NT
            if i < BK * HD:
                var r = i // HD
                var c = i - r * HD
                var j = kb * BK + r
                var v = Float32(0.0)
                if j < s:
                    v = ftz(v_cache.unsafe_load(kvbase + j * HD + c))
                ks.unsafe_store(r * KSTRIDE + c, v)
        barrier()
        if valid:
            for jj in range(BK):
                var j = kb * BK + jj
                if j >= j_lo and j <= j_hi:
                    cacc = _step(
                        es.unsafe_load(tr * ESTRIDE + jj),
                        ks.unsafe_load(jj * KSTRIDE + lane),
                        cacc,
                    )
        barrier()
    if valid:
        if bitcast[DType.uint32](cacc) == NEG_ZERO_BITS and j_hi < s - 1:
            corner.unsafe_store(0, Float32(1.0))
        ctxv.unsafe_store((bb * l + t) * nh * HD + h * HD + lane, cacc)


# ===========================================================================
# THE FUSED BACKWARD: z, then dq, then dk and dv
# ===========================================================================


def fused_bwd_zdot_kernel[HD: Int, TQ: Int](
    zdot: MutPointer[Float32, MutAnyOrigin],
    corner: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    dctx: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    v_cache: MutPointer[Float32, MutAnyOrigin],
    amax: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
    key_lo_in: Int32,
    window_in: Int32,
    scale_in: Float32,
):
    """`z = sum_j dy_j * y_j`, `bwd_softmax_zdot_kernel`'s chain, with `y`
    recomputed from the score and `dy` recomputed as the one-leaf gemm cell
    `dctx[t] . v[j]` (stage 17's routed OP_NT at `k = head_dim`). Lanes
    below `HD/2` hold `q[t]` and produce `y`; lanes from `HD/2` hold
    `dctx[t]` and produce `dy`; `BK` keys per block iteration."""
    comptime HALF = HD // 2
    comptime BK = HALF if HALF <= 32 else 32
    comptime NT = TQ * HD
    comptime KSTRIDE = HD + 1
    comptime ESTRIDE = BK + 1
    comptime SLOTS = (BK * HD + NT - 1) // NT

    var ks = stack_allocation[
        BK * KSTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var vs = stack_allocation[
        BK * KSTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var ys = stack_allocation[
        TQ * ESTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var dys = stack_allocation[
        TQ * ESTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)
    var n_rep = nh // nkv

    var ntb = (l + TQ - 1) // TQ
    var raw = Int(block_idx.x)
    var tb = raw % ntb
    var rest = raw // ntb
    var h = rest % nh
    var bb = rest // nh
    if bb >= b:
        return
    var kvh = h // n_rep

    var tid = Int(thread_idx.x)
    var tr = tid // HD
    var lane = tid - tr * HD
    var is_y = lane < HALF
    var kj = lane
    if not is_y:
        kj = lane - HALF
    var active = kj < BK
    var t = tb * TQ + tr
    var valid = t < l
    var tt = t
    if not valid:
        tt = l - 1
    var rr = _row_range(tt, pos0, key_lo, window, s)
    var j_lo = rr[0]
    var j_hi = rr[1]
    var t0 = tb * TQ
    var t1 = t0 + TQ - 1
    if t1 > l - 1:
        t1 = l - 1
    var r0 = _row_range(t0, pos0, key_lo, window, s)
    var r1 = _row_range(t1, pos0, key_lo, window, s)
    var kb_lo = r0[0] // BK
    var kb_hi = r1[1] // BK

    var kvbase = (bb * nkv + kvh) * s * HD
    var rowbase = (bb * l + tt) * nh * HD + h * HD
    var row = (bb * nh + h) * l + tt
    var m_row = ftz(amax.unsafe_load(row))
    var d_row = ftz(denom.unsafe_load(row))

    var vec = stack_allocation[HD, Scalar[DType.float32]]()
    if is_y:
        comptime for p in range(HD):
            vec.unsafe_store(p, ftz(q_rope.unsafe_load(rowbase + p)))
    else:
        comptime for p in range(HD):
            vec.unsafe_store(p, ftz(dctx.unsafe_load(rowbase + p)))

    var z = Float32(0.0)
    for kb in range(kb_lo, kb_hi + 1):
        comptime for si in range(SLOTS):
            var i = tid + si * NT
            if i < BK * HD:
                var r = i // HD
                var c = i - r * HD
                var j = kb * BK + r
                var kv = Float32(0.0)
                var vv = Float32(0.0)
                if j < s:
                    kv = ftz(k_cache.unsafe_load(kvbase + j * HD + c))
                    vv = ftz(v_cache.unsafe_load(kvbase + j * HD + c))
                ks.unsafe_store(r * KSTRIDE + c, kv)
                vs.unsafe_store(r * KSTRIDE + c, vv)
        barrier()
        if valid and active:
            var j = kb * BK + kj
            if j >= j_lo and j <= j_hi:
                if is_y:
                    var dot = Float32(0.0)
                    comptime for p in range(HD):
                        dot = _step(vec.unsafe_load(p), ks.unsafe_load(kj * KSTRIDE + p), dot)
                    var sc = _pmul(dot, scale_in)
                    var masked = ftz(sc + Float32(0.0))
                    var e = ftz(identical_exp(ftz(ftz(masked) - m_row)))
                    ys.unsafe_store(
                        tr * ESTRIDE + kj, ftz(identical_div(ftz(e), d_row))
                    )
                else:
                    var dy = Float32(0.0)
                    comptime for p in range(HD):
                        dy = _step(vec.unsafe_load(p), vs.unsafe_load(kj * KSTRIDE + p), dy)
                    dys.unsafe_store(tr * ESTRIDE + kj, ftz(dy))
        barrier()
        if valid and lane == 0:
            for jj in range(BK):
                var j = kb * BK + jj
                if j >= j_lo and j <= j_hi:
                    z = _step(
                        dys.unsafe_load(tr * ESTRIDE + jj),
                        ys.unsafe_load(tr * ESTRIDE + jj),
                        z,
                    )
        barrier()
    if valid and lane == 0:
        var zf = ftz(z)
        if bitcast[DType.uint32](zf) == NEG_ZERO_BITS and j_hi < s - 1:
            corner.unsafe_store(0, Float32(1.0))
        zdot.unsafe_store(row, zf)


def fused_bwd_dq_kernel[HD: Int, TQ: Int](
    dq: MutPointer[Float32, MutAnyOrigin],
    corner: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    dctx: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    v_cache: MutPointer[Float32, MutAnyOrigin],
    amax: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    zdot: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
    key_lo_in: Int32,
    window_in: Int32,
    scale_in: Float32,
):
    """`dq[t, d]`, `bwd_dq_kernel`'s chain over the key axis, with the
    per-cell gradient recomputed: `y` and `dy` as in the z kernel, then
    stage 19 `ds = pmul(y, ftz(dy - z))`, stage 20 the identity, stage 21
    `dcell = pmul(ds, scale)`, then `_step(dcell, k[j, d], acc)` ascending."""
    comptime HALF = HD // 2
    comptime BK = HALF if HALF <= 32 else 32
    comptime NT = TQ * HD
    comptime KSTRIDE = HD + 1
    comptime ESTRIDE = BK + 1
    comptime SLOTS = (BK * HD + NT - 1) // NT

    var ks = stack_allocation[
        BK * KSTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var vs = stack_allocation[
        BK * KSTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var ys = stack_allocation[
        TQ * ESTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var dys = stack_allocation[
        TQ * ESTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)
    var n_rep = nh // nkv

    var ntb = (l + TQ - 1) // TQ
    var raw = Int(block_idx.x)
    var tb = raw % ntb
    var rest = raw // ntb
    var h = rest % nh
    var bb = rest // nh
    if bb >= b:
        return
    var kvh = h // n_rep

    var tid = Int(thread_idx.x)
    var tr = tid // HD
    var lane = tid - tr * HD
    var is_y = lane < HALF
    var kj = lane
    if not is_y:
        kj = lane - HALF
    var active = kj < BK
    var t = tb * TQ + tr
    var valid = t < l
    var tt = t
    if not valid:
        tt = l - 1
    var rr = _row_range(tt, pos0, key_lo, window, s)
    var j_lo = rr[0]
    var j_hi = rr[1]
    var t0 = tb * TQ
    var t1 = t0 + TQ - 1
    if t1 > l - 1:
        t1 = l - 1
    var r0 = _row_range(t0, pos0, key_lo, window, s)
    var r1 = _row_range(t1, pos0, key_lo, window, s)
    var kb_lo = r0[0] // BK
    var kb_hi = r1[1] // BK

    var kvbase = (bb * nkv + kvh) * s * HD
    var rowbase = (bb * l + tt) * nh * HD + h * HD
    var row = (bb * nh + h) * l + tt
    var m_row = ftz(amax.unsafe_load(row))
    var d_row = ftz(denom.unsafe_load(row))
    var z_row = ftz(zdot.unsafe_load(row))

    var vec = stack_allocation[HD, Scalar[DType.float32]]()
    if is_y:
        comptime for p in range(HD):
            vec.unsafe_store(p, ftz(q_rope.unsafe_load(rowbase + p)))
    else:
        comptime for p in range(HD):
            vec.unsafe_store(p, ftz(dctx.unsafe_load(rowbase + p)))

    var acc = Float32(0.0)
    for kb in range(kb_lo, kb_hi + 1):
        comptime for si in range(SLOTS):
            var i = tid + si * NT
            if i < BK * HD:
                var r = i // HD
                var c = i - r * HD
                var j = kb * BK + r
                var kv = Float32(0.0)
                var vv = Float32(0.0)
                if j < s:
                    kv = ftz(k_cache.unsafe_load(kvbase + j * HD + c))
                    vv = ftz(v_cache.unsafe_load(kvbase + j * HD + c))
                ks.unsafe_store(r * KSTRIDE + c, kv)
                vs.unsafe_store(r * KSTRIDE + c, vv)
        barrier()
        if valid and active:
            var j = kb * BK + kj
            if j >= j_lo and j <= j_hi:
                if is_y:
                    var dot = Float32(0.0)
                    comptime for p in range(HD):
                        dot = _step(vec.unsafe_load(p), ks.unsafe_load(kj * KSTRIDE + p), dot)
                    var sc = _pmul(dot, scale_in)
                    var masked = ftz(sc + Float32(0.0))
                    var e = ftz(identical_exp(ftz(ftz(masked) - m_row)))
                    ys.unsafe_store(
                        tr * ESTRIDE + kj, ftz(identical_div(ftz(e), d_row))
                    )
                else:
                    var dy = Float32(0.0)
                    comptime for p in range(HD):
                        dy = _step(vec.unsafe_load(p), vs.unsafe_load(kj * KSTRIDE + p), dy)
                    dys.unsafe_store(tr * ESTRIDE + kj, ftz(dy))
        barrier()
        # Stages 19-21 for this block's keys, into the `y` slot the same
        # thread just wrote (own slot: no race).
        if valid and lane < BK:
            var j = kb * BK + lane
            if j >= j_lo and j <= j_hi:
                var yv = ftz(ys.unsafe_load(tr * ESTRIDE + lane))
                var dv = ftz(dys.unsafe_load(tr * ESTRIDE + lane))
                var ds = _pmul(yv, ftz(ftz(dv) - ftz(z_row)))
                ys.unsafe_store(tr * ESTRIDE + lane, _pmul(ftz(ds), scale_in))
        barrier()
        if valid:
            for jj in range(BK):
                var j = kb * BK + jj
                if j >= j_lo and j <= j_hi:
                    acc = _step(
                        ys.unsafe_load(tr * ESTRIDE + jj),
                        ks.unsafe_load(jj * KSTRIDE + lane),
                        acc,
                    )
        barrier()
    if valid:
        if bitcast[DType.uint32](acc) == NEG_ZERO_BITS and j_hi < s - 1:
            corner.unsafe_store(0, Float32(1.0))
        dq.unsafe_store((bb * l + t) * nh * HD + h * HD + lane, acc)


def fused_bwd_dkdv_kernel[HD: Int, BJ: Int](
    dk: MutPointer[Float32, MutAnyOrigin],
    dv: MutPointer[Float32, MutAnyOrigin],
    corner: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    dctx: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    v_cache: MutPointer[Float32, MutAnyOrigin],
    amax: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    zdot: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
    key_lo_in: Int32,
    window_in: Int32,
    scale_in: Float32,
):
    """`dk[j, d]` and `dv[j, d]`, `bwd_dk_kernel`'s and `bwd_dv_kernel`'s
    chains over `(head in the kv group ASCENDING, query ASCENDING)`, one
    block per `BJ` consecutive keys of one `(batch, kv head)`; thread
    `(key, lane)`. Lanes below `HD/2` hold `k[j]` and produce `y` for
    query `t0 + lane`; lanes from `HD/2` hold `v[j]` and produce `dy` for
    query `t0 + lane - HD/2`; then every lane `d` folds its two chains.
    `TT` queries per block iteration."""
    comptime HALF = HD // 2
    comptime TT = HALF if HALF <= 32 else 32
    comptime NT = BJ * HD
    comptime KSTRIDE = HD + 1
    comptime ESTRIDE = TT + 1
    comptime SLOTS = (TT * HD + NT - 1) // NT

    var qs = stack_allocation[
        TT * KSTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var dcs = stack_allocation[
        TT * KSTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var ys = stack_allocation[
        BJ * ESTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var dys = stack_allocation[
        BJ * ESTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)
    var n_rep = nh // nkv

    var njb = (s + BJ - 1) // BJ
    var raw = Int(block_idx.x)
    var jb = raw % njb
    var rest = raw // njb
    var kvh = rest % nkv
    var bb = rest // nkv
    if bb >= b:
        return

    var tid = Int(thread_idx.x)
    var jr = tid // HD
    var lane = tid - jr * HD
    var is_y = lane < HALF
    var kt = lane
    if not is_y:
        kt = lane - HALF
    var active = kt < TT
    var j = jb * BJ + jr
    var valid = j < s
    var jj = j
    if not valid:
        jj = s - 1
    var qr = _key_query_range(jj, pos0, key_lo, window, l)
    var t_lo = qr[0]
    var t_hi = qr[1]
    var j0 = jb * BJ
    var j1 = j0 + BJ - 1
    if j1 > s - 1:
        j1 = s - 1
    var q0 = _key_query_range(j0, pos0, key_lo, window, l)
    var q1 = _key_query_range(j1, pos0, key_lo, window, l)
    var tb_lo = q0[0] // TT
    var tb_hi = q1[1] // TT
    if q1[1] < q0[0]:
        # No query in this call sees any key of this block: every chain
        # is the empty chain, `+0.0`, and the block stores it.
        tb_hi = tb_lo - 1

    var kvbase = (bb * nkv + kvh) * s * HD
    var vec = stack_allocation[HD, Scalar[DType.float32]]()
    if is_y:
        comptime for p in range(HD):
            vec.unsafe_store(p, ftz(k_cache.unsafe_load(kvbase + jj * HD + p)))
    else:
        comptime for p in range(HD):
            vec.unsafe_store(p, ftz(v_cache.unsafe_load(kvbase + jj * HD + p)))

    var dk_acc = Float32(0.0)
    var dv_acc = Float32(0.0)
    var hit = False
    for hh in range(n_rep):
        var h = kvh * n_rep + hh
        for tb in range(tb_lo, tb_hi + 1):
            comptime for si in range(SLOTS):
                var i = tid + si * NT
                if i < TT * HD:
                    var r = i // HD
                    var c = i - r * HD
                    var t = tb * TT + r
                    var qv = Float32(0.0)
                    var dcv = Float32(0.0)
                    if t < l:
                        var off = (bb * l + t) * nh * HD + h * HD + c
                        qv = ftz(q_rope.unsafe_load(off))
                        dcv = ftz(dctx.unsafe_load(off))
                    qs.unsafe_store(r * KSTRIDE + c, qv)
                    dcs.unsafe_store(r * KSTRIDE + c, dcv)
            barrier()
            if valid and active:
                var t = tb * TT + kt
                if t < l and t >= t_lo and t <= t_hi:
                    var row = (bb * nh + h) * l + t
                    if is_y:
                        var dot = Float32(0.0)
                        comptime for p in range(HD):
                            dot = _step(qs.unsafe_load(kt * KSTRIDE + p), vec.unsafe_load(p), dot)
                        var sc = _pmul(dot, scale_in)
                        var masked = ftz(sc + Float32(0.0))
                        var m_row = ftz(amax.unsafe_load(row))
                        var d_row = ftz(denom.unsafe_load(row))
                        var e = ftz(identical_exp(ftz(ftz(masked) - m_row)))
                        ys.unsafe_store(
                            jr * ESTRIDE + kt,
                            ftz(identical_div(ftz(e), d_row)),
                        )
                    else:
                        var dy = Float32(0.0)
                        comptime for p in range(HD):
                            dy = _step(dcs.unsafe_load(kt * KSTRIDE + p), vec.unsafe_load(p), dy)
                        dys.unsafe_store(jr * ESTRIDE + kt, ftz(dy))
            barrier()
            if valid and lane < TT:
                var t = tb * TT + lane
                if t < l and t >= t_lo and t <= t_hi:
                    var row = (bb * nh + h) * l + t
                    var z_row = ftz(zdot.unsafe_load(row))
                    var yv = ftz(ys.unsafe_load(jr * ESTRIDE + lane))
                    var dv_ = ftz(dys.unsafe_load(jr * ESTRIDE + lane))
                    var ds = _pmul(yv, ftz(ftz(dv_) - ftz(z_row)))
                    dys.unsafe_store(jr * ESTRIDE + lane, _pmul(ftz(ds), scale_in))
            barrier()
            if valid:
                for tk in range(TT):
                    var t = tb * TT + tk
                    if t < l and t >= t_lo and t <= t_hi:
                        dk_acc = _step(
                            dys.unsafe_load(jr * ESTRIDE + tk),
                            qs.unsafe_load(tk * KSTRIDE + lane),
                            dk_acc,
                        )
                        dv_acc = _step(
                            ys.unsafe_load(jr * ESTRIDE + tk),
                            dcs.unsafe_load(tk * KSTRIDE + lane),
                            dv_acc,
                        )
            barrier()
        # The end of this head's visible run for key `j`: a `-0.0` here
        # could be laundered by the masked cells that follow in the chain.
        if bitcast[DType.uint32](dk_acc) == NEG_ZERO_BITS:
            hit = True
        if bitcast[DType.uint32](dv_acc) == NEG_ZERO_BITS:
            hit = True
    if valid:
        if hit:
            corner.unsafe_store(0, Float32(1.0))
        dk.unsafe_store(kvbase + j * HD + lane, dk_acc)
        dv.unsafe_store(kvbase + j * HD + lane, dv_acc)


# ===========================================================================
# DEVIATION 2525 (arm `bwd_stash`): the backward with y, dy and ds kept.
#
# THE READING. The shipped backward recomputes every cell's `y` (a 64-term
# score chain, exp, div) and `dy` (a 64-term chain over dctx and v) THREE
# times: once in `fused_bwd_zdot_kernel` (for z), once in
# `fused_bwd_dq_kernel` (for ds and the dq chain) and once in
# `fused_bwd_dkdv_kernel` (for the dk and dv chains), and the dkdv kernel
# also gathers `amax`, `denom` and `zdot` per cell from global memory. Per
# visible cell that is six dots of 64 RN-FMA + FTZ-multiply seams, three
# exp, three div, three 64-term folds; the folds are the contract, the
# recomputation is not. The block shapes are TQ = 4 query rows (zdot, dq)
# and BJ = 4 keys (dkdv) per 256 threads, so every block stages the whole
# key (or query) range for four rows, and the z chain runs on four threads
# of 256 between barriers.
#
# THE MECHANISM, in three kernels that keep the shipped block shapes and
# fold loops:
#   zdot_stash   the shipped zdot, plus a store of each visible cell's `y`
#                and `dy` (the bits it already holds) to two `[B, n_heads,
#                L, S]` scratches the launcher owns for the call (2 x 201 MB
#                at the target shape);
#   dq_stash     no dots: reads `y` and `dy` back, computes stage 19-21
#                (`ds = pmul(y, ftz(dy - z))`, `dcell = pmul(ds, scale)`)
#                once, stores `dcell` OVER `dy` (own cell) for the dkdv
#                kernel, and folds dq from the staged K tile as before;
#   dkdv_stash   no dots, no gathers: stages the `y` and `dcell` tiles for
#                its four keys beside the Q and dctx tiles and folds dk and
#                dv as before.
# Per visible cell: two dots, one exp, one div, three folds.
#
# WHY NO BIT MOVES. `y[t, j]` and `dy[t, j]` are pure functions of the
# inputs computed by one spelling (the score chain over p ascending, the
# pinned scale, the mask add, the exp against the row maximum, the
# division by the row denominator; the dctx.v chain over p ascending), and
# the three shipped kernels compute them from the same staged operands and
# land on the same bits; storing the bits and reading them back is the
# same value. `dcell` is computed by the same three operations the shipped
# dq and dkdv kernels each apply to the same `y`, `dy`, `z` and scale.
# The folds (z over keys ascending, dq over keys ascending, dk and dv over
# (head ascending, query ascending)) are the shipped loops reading the
# same tiles in the same order. Only visible cells are stored or read (the
# visibility tests are copied); the corner rule is unchanged.
#
# SABOTAGE (reach): dq_stash flips one ulp of the `dcell` it stages for
# its own fold (dq moves, dk does not); dkdv_stash flips one ulp of the `y`
# tile it stages (dv moves, dk does not).
# ===========================================================================


def fused_bwd_zdot_stash_kernel[HD: Int, TQ: Int](
    zdot: MutPointer[Float32, MutAnyOrigin],
    corner: MutPointer[Float32, MutAnyOrigin],
    y_st: MutPointer[Float32, MutAnyOrigin],
    dy_st: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    dctx: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    v_cache: MutPointer[Float32, MutAnyOrigin],
    amax: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
    key_lo_in: Int32,
    window_in: Int32,
    scale_in: Float32,
):
    """`fused_bwd_zdot_kernel` plus the `y` and `dy` stores; see above."""
    comptime HALF = HD // 2
    comptime BK = HALF if HALF <= 32 else 32
    comptime NT = TQ * HD
    comptime KSTRIDE = HD + 1
    comptime ESTRIDE = BK + 1
    comptime SLOTS = (BK * HD + NT - 1) // NT

    var ks = stack_allocation[
        BK * KSTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var vs = stack_allocation[
        BK * KSTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var ys = stack_allocation[
        TQ * ESTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var dys = stack_allocation[
        TQ * ESTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)
    var n_rep = nh // nkv

    var ntb = (l + TQ - 1) // TQ
    var raw = Int(block_idx.x)
    var tb = raw % ntb
    var rest = raw // ntb
    var h = rest % nh
    var bb = rest // nh
    if bb >= b:
        return
    var kvh = h // n_rep

    var tid = Int(thread_idx.x)
    var tr = tid // HD
    var lane = tid - tr * HD
    var is_y = lane < HALF
    var kj = lane
    if not is_y:
        kj = lane - HALF
    var active = kj < BK
    var t = tb * TQ + tr
    var valid = t < l
    var tt = t
    if not valid:
        tt = l - 1
    var rr = _row_range(tt, pos0, key_lo, window, s)
    var j_lo = rr[0]
    var j_hi = rr[1]
    var t0 = tb * TQ
    var t1 = t0 + TQ - 1
    if t1 > l - 1:
        t1 = l - 1
    var r0 = _row_range(t0, pos0, key_lo, window, s)
    var r1 = _row_range(t1, pos0, key_lo, window, s)
    var kb_lo = r0[0] // BK
    var kb_hi = r1[1] // BK

    var kvbase = (bb * nkv + kvh) * s * HD
    var rowbase = (bb * l + tt) * nh * HD + h * HD
    var row = (bb * nh + h) * l + tt
    var stbase = row * s
    var m_row = ftz(amax.unsafe_load(row))
    var d_row = ftz(denom.unsafe_load(row))

    var vec = stack_allocation[HD, Scalar[DType.float32]]()
    if is_y:
        comptime for p in range(HD):
            vec.unsafe_store(p, ftz(q_rope.unsafe_load(rowbase + p)))
    else:
        comptime for p in range(HD):
            vec.unsafe_store(p, ftz(dctx.unsafe_load(rowbase + p)))

    var z = Float32(0.0)
    for kb in range(kb_lo, kb_hi + 1):
        comptime for si in range(SLOTS):
            var i = tid + si * NT
            if i < BK * HD:
                var r = i // HD
                var c = i - r * HD
                var j = kb * BK + r
                var kv = Float32(0.0)
                var vv = Float32(0.0)
                if j < s:
                    kv = ftz(k_cache.unsafe_load(kvbase + j * HD + c))
                    vv = ftz(v_cache.unsafe_load(kvbase + j * HD + c))
                ks.unsafe_store(r * KSTRIDE + c, kv)
                vs.unsafe_store(r * KSTRIDE + c, vv)
        barrier()
        if valid and active:
            var j = kb * BK + kj
            if j >= j_lo and j <= j_hi:
                if is_y:
                    var dot = Float32(0.0)
                    comptime for p in range(HD):
                        dot = _step(vec.unsafe_load(p), ks.unsafe_load(kj * KSTRIDE + p), dot)
                    var sc = _pmul(dot, scale_in)
                    var masked = ftz(sc + Float32(0.0))
                    var e = ftz(identical_exp(ftz(ftz(masked) - m_row)))
                    var yv = ftz(identical_div(ftz(e), d_row))
                    ys.unsafe_store(tr * ESTRIDE + kj, yv)
                    y_st.unsafe_store(stbase + j, yv)
                else:
                    var dy = Float32(0.0)
                    comptime for p in range(HD):
                        dy = _step(vec.unsafe_load(p), vs.unsafe_load(kj * KSTRIDE + p), dy)
                    var dyv = ftz(dy)
                    dys.unsafe_store(tr * ESTRIDE + kj, dyv)
                    dy_st.unsafe_store(stbase + j, dyv)
        barrier()
        if valid and lane == 0:
            for jj in range(BK):
                var j = kb * BK + jj
                if j >= j_lo and j <= j_hi:
                    z = _step(
                        dys.unsafe_load(tr * ESTRIDE + jj),
                        ys.unsafe_load(tr * ESTRIDE + jj),
                        z,
                    )
        barrier()
    if valid and lane == 0:
        var zf = ftz(z)
        if bitcast[DType.uint32](zf) == NEG_ZERO_BITS and j_hi < s - 1:
            corner.unsafe_store(0, Float32(1.0))
        zdot.unsafe_store(row, zf)


def fused_bwd_dq_stash_kernel[HD: Int, TQ: Int, SABOTAGE: Bool](
    dq: MutPointer[Float32, MutAnyOrigin],
    corner: MutPointer[Float32, MutAnyOrigin],
    y_st: MutPointer[Float32, MutAnyOrigin],
    dy_st: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    zdot: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
    key_lo_in: Int32,
    window_in: Int32,
    scale_in: Float32,
):
    """`fused_bwd_dq_kernel`'s fold, with `y` and `dy` read from the stash
    and `dcell` (stage 21) written over `dy` for the dkdv kernel; see
    above."""
    comptime HALF = HD // 2
    comptime BK = HALF if HALF <= 32 else 32
    comptime NT = TQ * HD
    comptime KSTRIDE = HD + 1
    comptime ESTRIDE = BK + 1
    comptime SLOTS = (BK * HD + NT - 1) // NT

    var ks = stack_allocation[
        BK * KSTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var ys = stack_allocation[
        TQ * ESTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)
    var n_rep = nh // nkv

    var ntb = (l + TQ - 1) // TQ
    var raw = Int(block_idx.x)
    var tb = raw % ntb
    var rest = raw // ntb
    var h = rest % nh
    var bb = rest // nh
    if bb >= b:
        return
    var kvh = h // n_rep

    var tid = Int(thread_idx.x)
    var tr = tid // HD
    var lane = tid - tr * HD
    var t = tb * TQ + tr
    var valid = t < l
    var tt = t
    if not valid:
        tt = l - 1
    var rr = _row_range(tt, pos0, key_lo, window, s)
    var j_lo = rr[0]
    var j_hi = rr[1]
    var t0 = tb * TQ
    var t1 = t0 + TQ - 1
    if t1 > l - 1:
        t1 = l - 1
    var r0 = _row_range(t0, pos0, key_lo, window, s)
    var r1 = _row_range(t1, pos0, key_lo, window, s)
    var kb_lo = r0[0] // BK
    var kb_hi = r1[1] // BK

    var kvbase = (bb * nkv + kvh) * s * HD
    var row = (bb * nh + h) * l + tt
    var stbase = row * s
    var z_row = ftz(zdot.unsafe_load(row))

    var acc = Float32(0.0)
    for kb in range(kb_lo, kb_hi + 1):
        comptime for si in range(SLOTS):
            var i = tid + si * NT
            if i < BK * HD:
                var r = i // HD
                var c = i - r * HD
                var j = kb * BK + r
                var kv = Float32(0.0)
                if j < s:
                    kv = ftz(k_cache.unsafe_load(kvbase + j * HD + c))
                ks.unsafe_store(r * KSTRIDE + c, kv)
        # Stages 19-21 for this block's keys from the stash, once, into
        # the fold's slot AND over `dy` for the dkdv kernel (own cell).
        if valid and lane < BK:
            var j = kb * BK + lane
            if j >= j_lo and j <= j_hi:
                var yv = ftz(y_st.unsafe_load(stbase + j))
                var dv = ftz(dy_st.unsafe_load(stbase + j))
                var ds = _pmul(yv, ftz(ftz(dv) - ftz(z_row)))
                var dcell = _pmul(ftz(ds), scale_in)
                dy_st.unsafe_store(stbase + j, dcell)
                comptime if SABOTAGE:
                    ys.unsafe_store(tr * ESTRIDE + lane, _flip_ulp(dcell))
                else:
                    ys.unsafe_store(tr * ESTRIDE + lane, dcell)
        barrier()
        if valid:
            for jj in range(BK):
                var j = kb * BK + jj
                if j >= j_lo and j <= j_hi:
                    acc = _step(
                        ys.unsafe_load(tr * ESTRIDE + jj),
                        ks.unsafe_load(jj * KSTRIDE + lane),
                        acc,
                    )
        barrier()
    if valid:
        if bitcast[DType.uint32](acc) == NEG_ZERO_BITS and j_hi < s - 1:
            corner.unsafe_store(0, Float32(1.0))
        dq.unsafe_store((bb * l + t) * nh * HD + h * HD + lane, acc)


def fused_bwd_dkdv_stash_kernel[HD: Int, BJ: Int, SABOTAGE: Bool](
    dk: MutPointer[Float32, MutAnyOrigin],
    dv: MutPointer[Float32, MutAnyOrigin],
    corner: MutPointer[Float32, MutAnyOrigin],
    y_st: MutPointer[Float32, MutAnyOrigin],
    ds_st: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    dctx: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
    key_lo_in: Int32,
    window_in: Int32,
):
    """`fused_bwd_dkdv_kernel`'s folds, with the `y` and `dcell` tiles
    staged from the stash instead of recomputed; see above. `ds_st` is the
    `dy` scratch after `fused_bwd_dq_stash_kernel` wrote `dcell` over it."""
    comptime HALF = HD // 2
    comptime TT = HALF if HALF <= 32 else 32
    comptime NT = BJ * HD
    comptime KSTRIDE = HD + 1
    comptime ESTRIDE = TT + 1
    comptime SLOTS = (TT * HD + NT - 1) // NT
    comptime CELLS = TT * BJ
    comptime CSLOTS = (CELLS + NT - 1) // NT

    var qs = stack_allocation[
        TT * KSTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var dcs = stack_allocation[
        TT * KSTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var ys = stack_allocation[
        BJ * ESTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var dys = stack_allocation[
        BJ * ESTRIDE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)
    var n_rep = nh // nkv

    var njb = (s + BJ - 1) // BJ
    var raw = Int(block_idx.x)
    var jb = raw % njb
    var rest = raw // njb
    var kvh = rest % nkv
    var bb = rest // nkv
    if bb >= b:
        return

    var tid = Int(thread_idx.x)
    var jr = tid // HD
    var lane = tid - jr * HD
    var j = jb * BJ + jr
    var valid = j < s
    var jj = j
    if not valid:
        jj = s - 1
    var qr = _key_query_range(jj, pos0, key_lo, window, l)
    var t_lo = qr[0]
    var t_hi = qr[1]
    var j0 = jb * BJ
    var j1 = j0 + BJ - 1
    if j1 > s - 1:
        j1 = s - 1
    var q0 = _key_query_range(j0, pos0, key_lo, window, l)
    var q1 = _key_query_range(j1, pos0, key_lo, window, l)
    var tb_lo = q0[0] // TT
    var tb_hi = q1[1] // TT
    if q1[1] < q0[0]:
        tb_hi = tb_lo - 1

    var kvbase = (bb * nkv + kvh) * s * HD

    var dk_acc = Float32(0.0)
    var dv_acc = Float32(0.0)
    var hit = False
    for hh in range(n_rep):
        var h = kvh * n_rep + hh
        var hbase = (bb * nh + h) * l
        for tb in range(tb_lo, tb_hi + 1):
            comptime for si in range(SLOTS):
                var i = tid + si * NT
                if i < TT * HD:
                    var r = i // HD
                    var c = i - r * HD
                    var t = tb * TT + r
                    var qv = Float32(0.0)
                    var dcv = Float32(0.0)
                    if t < l:
                        var off = (bb * l + t) * nh * HD + h * HD + c
                        qv = ftz(q_rope.unsafe_load(off))
                        dcv = ftz(dctx.unsafe_load(off))
                    qs.unsafe_store(r * KSTRIDE + c, qv)
                    dcs.unsafe_store(r * KSTRIDE + c, dcv)
            # The y and dcell tiles for this block's keys: cell (query r,
            # key c), `BJ` consecutive keys per query row of the stash.
            comptime for si in range(CSLOTS):
                var i = tid + si * NT
                if i < CELLS:
                    var r = i // BJ
                    var c = i - r * BJ
                    var t = tb * TT + r
                    var jc = j0 + c
                    var yv = Float32(0.0)
                    var dsv = Float32(0.0)
                    if t < l and jc < s:
                        var cr = _key_query_range(jc, pos0, key_lo, window, l)
                        if t >= cr[0] and t <= cr[1]:
                            var cell = (hbase + t) * s + jc
                            yv = y_st.unsafe_load(cell)
                            dsv = ds_st.unsafe_load(cell)
                    comptime if SABOTAGE:
                        ys.unsafe_store(c * ESTRIDE + r, _flip_ulp(yv))
                    else:
                        ys.unsafe_store(c * ESTRIDE + r, yv)
                    dys.unsafe_store(c * ESTRIDE + r, dsv)
            barrier()
            if valid:
                for tk in range(TT):
                    var t = tb * TT + tk
                    if t < l and t >= t_lo and t <= t_hi:
                        dk_acc = _step(
                            dys.unsafe_load(jr * ESTRIDE + tk),
                            qs.unsafe_load(tk * KSTRIDE + lane),
                            dk_acc,
                        )
                        dv_acc = _step(
                            ys.unsafe_load(jr * ESTRIDE + tk),
                            dcs.unsafe_load(tk * KSTRIDE + lane),
                            dv_acc,
                        )
            barrier()
        if bitcast[DType.uint32](dk_acc) == NEG_ZERO_BITS:
            hit = True
        if bitcast[DType.uint32](dv_acc) == NEG_ZERO_BITS:
            hit = True
    if valid:
        if hit:
            corner.unsafe_store(0, Float32(1.0))
        dk.unsafe_store(kvbase + j * HD + lane, dk_acc)
        dv.unsafe_store(kvbase + j * HD + lane, dv_acc)


# ===========================================================================
# DEVIATION 2527 (arm `bwd_stash_tiled`): register-blocked folds over the
# stash.
#
# THE READING. With y, dy and dcell materialized (DEVIATION 2525), dq, dk
# and dv are three K-serial contractions: `dq[t, d]` over keys ascending
# of `dcell[t, j] * k[j, d]`, `dk[j, d]` over queries ascending of
# `dcell[t, j] * q[t, d]`, `dv[j, d]` over queries ascending of
# `y[t, j] * dctx[t, d]`. The 2525 kernels fold them with the shipped
# shapes: four rows (or keys) per 256-thread block, one accumulator per
# thread, two shared loads per RN-FMA + FTZ-multiply pair, and a full
# re-staging of the contracted operand for every four outputs. That is the
# register-blocking problem the GEMM lane solved: an output tile of
# 64 x 64 per block, 4 x 4 outputs per thread, so one staged operand
# element feeds four accumulators and the shared-load to FMA ratio drops
# from 2 to 0.5 (dq) or 0.25 (dk and dv, which share their staged tiles).
#
# THE MECHANISM.
#   dq_tiled     block = 64 query rows x 64 output columns of one (batch,
#                head); thread (tr, tc) owns rows `tr + 16u` and columns
#                `tc + 16v`, 16 accumulators; per 16-key tile it stages the
#                K tile [16][64] and the dcell tile [64][16] (computed from
#                the y and dy stash by the staging thread and written back
#                over dy for the dkdv kernel, as 2525's dq does), then for
#                each key of the tile in ascending order applies `_step`
#                to every (row, column) the cell is visible for.
#   dkdv_tiled   block = 64 keys x 64 columns of one (batch, kv head);
#                thread (tr, tc) owns keys `tr + 16u` and columns
#                `tc + 16v`, 16 + 16 accumulators; per 16-query tile of
#                each head of the group it stages Q [16][64], dctx [16][64],
#                y [16][64 keys] and dcell [16][64 keys], then for each
#                query ascending applies the two `_step`s where visible.
#
# WHY NO BIT MOVES. Every accumulator is one output cell's chain from
# `+0.0`, stepped by the same `_step` on the same two operands the shipped
# kernel steps it with, in the same order: dq over j ascending (key tiles
# ascending, keys within a tile ascending), dk and dv over (head in the kv
# group ascending, query tiles ascending, queries within a tile ascending).
# Masked cells are skipped by the copied visibility tests, so the chain
# contains exactly the shipped terms. Which thread holds which chain, how
# many chains a thread holds, and how the operands reach shared memory
# are execution-plan choices the contract does not read. `dcell` is the
# same three operations as before. The corner rule is applied per chain.
#
# SABOTAGE (reach): dq_tiled flips one ulp of the dcell it stages for its
# own fold (dq moves, dk does not); dkdv_tiled flips one ulp of the staged
# y tile (dv moves, dk does not).
# ===========================================================================

comptime TILED_TK = 16
"""Keys per staged tile of `fused_bwd_dq_tiled_kernel`."""
comptime TILED_TT = 16
"""Queries per staged tile of `fused_bwd_dkdv_tiled_kernel`."""


def fused_bwd_dq_tiled_kernel[HD: Int, SABOTAGE: Bool](
    dq: MutPointer[Float32, MutAnyOrigin],
    corner: MutPointer[Float32, MutAnyOrigin],
    y_st: MutPointer[Float32, MutAnyOrigin],
    dy_st: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    zdot: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
    key_lo_in: Int32,
    window_in: Int32,
    scale_in: Float32,
):
    """See the comment above. 256 threads; `TQ = 64` rows per block."""
    comptime TQ = 64
    comptime RPT = TQ // 16
    comptime CPT = HD // 16
    comptime TK = TILED_TK
    var kst = stack_allocation[TK * HD, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dst = stack_allocation[TQ * TK, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var zs = stack_allocation[TQ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()

    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)
    var n_rep = nh // nkv

    var ntb = (l + TQ - 1) // TQ
    var raw = Int(block_idx.x)
    var tb = raw % ntb
    var rest = raw // ntb
    var h = rest % nh
    var bb = rest // nh
    if bb >= b:
        return
    var kvh = h // n_rep

    var tid = Int(thread_idx.x)
    var tr = tid // 16
    var tc = tid % 16
    var t0 = tb * TQ
    var t1 = t0 + TQ - 1
    if t1 > l - 1:
        t1 = l - 1
    var r0 = _row_range(t0, pos0, key_lo, window, s)
    var r1 = _row_range(t1, pos0, key_lo, window, s)
    var kb_lo = r0[0] // TK
    var kb_hi = r1[1] // TK

    var kvbase = (bb * nkv + kvh) * s * HD
    var hbase = (bb * nh + h) * l

    # The rows this thread folds: their visible ranges, once.
    var lo = SIMD[DType.int32, RPT](0)
    var hi = SIMD[DType.int32, RPT](-1)
    comptime for u in range(RPT):
        var t = t0 + tr + u * 16
        if t < l:
            var rr = _row_range(t, pos0, key_lo, window, s)
            lo[u] = Int32(rr[0])
            hi[u] = Int32(rr[1])
    if tid < TQ:
        var zv = Float32(0.0)
        if t0 + tid < l:
            zv = ftz(zdot.unsafe_load(hbase + t0 + tid))
        zs.unsafe_store(tid, zv)
    barrier()

    var acc = SIMD[DType.float32, RPT * CPT](0.0)
    for kb in range(kb_lo, kb_hi + 1):
        var j0 = kb * TK
        # K tile [TK][HD], coalesced along d.
        comptime for si in range(TK * HD // 256):
            var i = tid + si * 256
            var r = i // HD
            var c = i - r * HD
            var jc = j0 + r
            var kv = Float32(0.0)
            if jc < s:
                kv = ftz(k_cache.unsafe_load(kvbase + jc * HD + c))
            kst.unsafe_store(i, kv)
        # dcell tile [TQ][TK]: stages 19-21 from the stash, written back
        # over dy for the dkdv kernel (own cell).
        comptime for si in range(TQ * TK // 256):
            var i = tid + si * 256
            var r = i // TK
            var c = i - r * TK
            var t = t0 + r
            var jc = j0 + c
            var dcell = Float32(0.0)
            if t < l and jc < s:
                var rr = _row_range(t, pos0, key_lo, window, s)
                if jc >= rr[0] and jc <= rr[1]:
                    var cell = (hbase + t) * s + jc
                    var yv = ftz(y_st.unsafe_load(cell))
                    var dv = ftz(dy_st.unsafe_load(cell))
                    var ds = _pmul(yv, ftz(ftz(dv) - ftz(zs.unsafe_load(r))))
                    dcell = _pmul(ftz(ds), scale_in)
                    dy_st.unsafe_store(cell, dcell)
            comptime if SABOTAGE:
                dst.unsafe_store(i, _flip_ulp(dcell))
            else:
                dst.unsafe_store(i, dcell)
        barrier()
        comptime for jk in range(TK):
            var jc = j0 + jk
            var ka = SIMD[DType.float32, CPT](0.0)
            comptime for v in range(CPT):
                ka[v] = kst.unsafe_load(jk * HD + tc + v * 16)
            comptime for u in range(RPT):
                if Int32(jc) >= lo[u] and Int32(jc) <= hi[u]:
                    var dcell = dst.unsafe_load((tr + u * 16) * TK + jk)
                    comptime for v in range(CPT):
                        acc[u * CPT + v] = _step(dcell, ka[v], acc[u * CPT + v])
        barrier()
    comptime for u in range(RPT):
        var t = t0 + tr + u * 16
        if t < l:
            comptime for v in range(CPT):
                var x = acc[u * CPT + v]
                if bitcast[DType.uint32](x) == NEG_ZERO_BITS and Int(hi[u]) < s - 1:
                    corner.unsafe_store(0, Float32(1.0))
                dq.unsafe_store((bb * l + t) * nh * HD + h * HD + tc + v * 16, x)


def fused_bwd_dkdv_tiled_kernel[HD: Int, SABOTAGE: Bool](
    dk: MutPointer[Float32, MutAnyOrigin],
    dv: MutPointer[Float32, MutAnyOrigin],
    corner: MutPointer[Float32, MutAnyOrigin],
    y_st: MutPointer[Float32, MutAnyOrigin],
    ds_st: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    dctx: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
    key_lo_in: Int32,
    window_in: Int32,
):
    """See the comment above. 256 threads; `BJ = 64` keys per block."""
    comptime BJ = 64
    comptime RPT = BJ // 16
    comptime CPT = HD // 16
    comptime TT = TILED_TT
    var qs = stack_allocation[TT * HD, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dcs = stack_allocation[TT * HD, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ys = stack_allocation[TT * BJ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dss = stack_allocation[TT * BJ, Scalar[DType.float32], address_space = AddressSpace.SHARED]()

    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)
    var n_rep = nh // nkv

    var njb = (s + BJ - 1) // BJ
    var raw = Int(block_idx.x)
    var jb = raw % njb
    var rest = raw // njb
    var kvh = rest % nkv
    var bb = rest // nkv
    if bb >= b:
        return

    var tid = Int(thread_idx.x)
    var tr = tid // 16
    var tc = tid % 16
    var j0 = jb * BJ
    var j1 = j0 + BJ - 1
    if j1 > s - 1:
        j1 = s - 1
    var q0 = _key_query_range(j0, pos0, key_lo, window, l)
    var q1 = _key_query_range(j1, pos0, key_lo, window, l)
    var tb_lo = q0[0] // TT
    var tb_hi = q1[1] // TT
    if q1[1] < q0[0]:
        tb_hi = tb_lo - 1

    var kvbase = (bb * nkv + kvh) * s * HD

    # The keys this thread folds: their visible query ranges, once.
    var lo = SIMD[DType.int32, RPT](0)
    var hi = SIMD[DType.int32, RPT](-1)
    comptime for u in range(RPT):
        var jc = j0 + tr + u * 16
        if jc < s:
            var qr = _key_query_range(jc, pos0, key_lo, window, l)
            lo[u] = Int32(qr[0])
            hi[u] = Int32(qr[1])

    var dk_acc = SIMD[DType.float32, RPT * CPT](0.0)
    var dv_acc = SIMD[DType.float32, RPT * CPT](0.0)
    var hit = False
    for hh in range(n_rep):
        var h = kvh * n_rep + hh
        var hbase = (bb * nh + h) * l
        for tb in range(tb_lo, tb_hi + 1):
            var tq0 = tb * TT
            # Q and dctx tiles [TT][HD], coalesced along d.
            comptime for si in range(TT * HD // 256):
                var i = tid + si * 256
                var r = i // HD
                var c = i - r * HD
                var t = tq0 + r
                var qv = Float32(0.0)
                var dcv = Float32(0.0)
                if t < l:
                    var off = (bb * l + t) * nh * HD + h * HD + c
                    qv = ftz(q_rope.unsafe_load(off))
                    dcv = ftz(dctx.unsafe_load(off))
                qs.unsafe_store(i, qv)
                dcs.unsafe_store(i, dcv)
            # y and dcell tiles [TT][BJ], coalesced along j.
            comptime for si in range(TT * BJ // 256):
                var i = tid + si * 256
                var r = i // BJ
                var c = i - r * BJ
                var t = tq0 + r
                var jc = j0 + c
                var yv = Float32(0.0)
                var dsv = Float32(0.0)
                if t < l and jc < s:
                    var cr = _key_query_range(jc, pos0, key_lo, window, l)
                    if t >= cr[0] and t <= cr[1]:
                        var cell = (hbase + t) * s + jc
                        yv = y_st.unsafe_load(cell)
                        dsv = ds_st.unsafe_load(cell)
                comptime if SABOTAGE:
                    ys.unsafe_store(i, _flip_ulp(yv))
                else:
                    ys.unsafe_store(i, yv)
                dss.unsafe_store(i, dsv)
            barrier()
            comptime for tk in range(TT):
                var t = tq0 + tk
                var qa = SIMD[DType.float32, CPT](0.0)
                var da = SIMD[DType.float32, CPT](0.0)
                comptime for v in range(CPT):
                    qa[v] = qs.unsafe_load(tk * HD + tc + v * 16)
                    da[v] = dcs.unsafe_load(tk * HD + tc + v * 16)
                comptime for u in range(RPT):
                    if t < l and Int32(t) >= lo[u] and Int32(t) <= hi[u]:
                        var dcell = dss.unsafe_load(tk * BJ + tr + u * 16)
                        var yv = ys.unsafe_load(tk * BJ + tr + u * 16)
                        comptime for v in range(CPT):
                            dk_acc[u * CPT + v] = _step(dcell, qa[v], dk_acc[u * CPT + v])
                            dv_acc[u * CPT + v] = _step(yv, da[v], dv_acc[u * CPT + v])
            barrier()
        # The end of this head's visible run for every key this thread
        # holds: a `-0.0` here could be laundered by the masked tail.
        comptime for i in range(RPT * CPT):
            if bitcast[DType.uint32](dk_acc[i]) == NEG_ZERO_BITS:
                hit = True
            if bitcast[DType.uint32](dv_acc[i]) == NEG_ZERO_BITS:
                hit = True
    comptime for u in range(RPT):
        var jc = j0 + tr + u * 16
        if jc < s:
            if hit:
                corner.unsafe_store(0, Float32(1.0))
            comptime for v in range(CPT):
                dk.unsafe_store(kvbase + jc * HD + tc + v * 16, dk_acc[u * CPT + v])
                dv.unsafe_store(kvbase + jc * HD + tc + v * 16, dv_acc[u * CPT + v])


# ===========================================================================
# DEVIATION 2528 (arm token `_ztiled`, TRIAL BUILDS ONLY; brief
# docs/lanes/BRIEF_attention_step_2026-09-11.md section 12.2): zdot as a
# register-blocked y/dy kernel plus a row z fold.
#
# THE READING. Under stash_tiled the backward's floor is
# `fused_bwd_zdot_stash_kernel` (89.4 of 125.5 ms on the H100 leg, brief
# section 10): four query rows per 256-thread block, 32-key staging
# round trips of K and V for those four rows, one 64-term dot per thread,
# and the z chain folded on lane 0 of each row while 252 threads wait.
#
# THE MECHANISM.
#   ydy_tiled  (kernel A) block = TQ query rows (64 or 32, the kernel-matrix
#              row `attn_zdot_rows_per_block_for` or the arm's `_r32` /
#              `_r64`) of one (batch, head); thread (tr, tc) holds rows
#              `tr + 16u` and keys `tc + 16v` of a 64-key block iteration,
#              so 4 * TQ / 16 y dots and as many dy dots; per 16-wide p
#              window it stages Q and dctx (TQ rows) and K and V (64 keys)
#              through `ftz` on ONE page at stride 20, then runs 16 p steps
#              of `_step_preflushed`; after the four windows it finishes
#              every visible cell's y against the row's `ftz(amax)` and
#              `ftz(denom)` (registers, loaded once) and stores y and
#              `ftz(dy)` to the stashes.
#   zfold      (kernel B) one row per thread, 256 rows per block, no shared
#              memory and no barrier: `z = _step(dy, y, z)` over the row's
#              visible keys ascending, read from the stashes, then `ftz`,
#              the corner test and the zdot store.
# dq and dk/dv are then the 2527 tiled folds, unchanged instantiations.
#
# WHY NO BIT MOVES (brief 12.2 in full). `ftz` maps only subnormals, so it
# is idempotent and `_step(a, b, acc) == _step_preflushed(ftz(a), ftz(b),
# acc)` on every column; kernel A's chains are the shipped chains on the
# same flushed operands in the same p order (q then k, dctx then v), and
# the cell arithmetic after the dot is the shipped spelling term for term.
# Kernel B's chain is the shipped z chain on the stored bits, keys
# ascending, masked keys skipped by the copied `_row_range` test. Which
# thread holds a chain and how operands reach shared memory are plan
# choices the contract does not read.
#
# SABOTAGE (reach, ATTN_ARM_SABOTAGE_NEW): kernel A flips one ulp of every
# y it stores; zdot, dq, dk and dv move, the forward buffers do not.
# ===========================================================================


def fused_bwd_ydy_tiled_kernel[HD: Int, TQ: Int, SABOTAGE: Bool](
    y_st: MutPointer[Float32, MutAnyOrigin],
    dy_st: MutPointer[Float32, MutAnyOrigin],
    q_rope: MutPointer[Float32, MutAnyOrigin],
    dctx: MutPointer[Float32, MutAnyOrigin],
    k_cache: MutPointer[Float32, MutAnyOrigin],
    v_cache: MutPointer[Float32, MutAnyOrigin],
    amax: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    nkv_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
    key_lo_in: Int32,
    window_in: Int32,
    scale_in: Float32,
):
    """Kernel A of DEVIATION 2528; see the comment above. 256 threads,
    `TQ` query rows per block, ATTN_ZT_BK keys per block iteration, one
    shared page of `(2 * TQ + 2 * BK) * (KS + 4)` floats (20,480 B at
    TQ 64, 15,360 B at TQ 32)."""
    comptime RPT = TQ // 16
    comptime BK = ATTN_ZT_BK
    comptime KPT = BK // 16
    comptime KS = ATTN_ZT_KS
    comptime STRIDE = KS + 4
    comptime NR = 2 * TQ + 2 * BK
    var stg = stack_allocation[NR * STRIDE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()

    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var nkv = Int(nkv_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)
    var n_rep = nh // nkv

    var ntb = (l + TQ - 1) // TQ
    var raw = Int(block_idx.x)
    var tb = raw % ntb
    var rest = raw // ntb
    var h = rest % nh
    var bb = rest // nh
    if bb >= b:
        return
    var kvh = h // n_rep

    var tid = Int(thread_idx.x)
    var tr = tid // 16
    var tc = tid % 16
    var t0 = tb * TQ
    var t1 = t0 + TQ - 1
    if t1 > l - 1:
        t1 = l - 1
    var r0 = _row_range(t0, pos0, key_lo, window, s)
    var r1 = _row_range(t1, pos0, key_lo, window, s)
    var kb_lo = r0[0] // BK
    var kb_hi = r1[1] // BK

    var kvbase = (bb * nkv + kvh) * s * HD
    var hbase = (bb * nh + h) * l

    # The rows this thread holds: visible ranges and the row scalars, once.
    var lo = SIMD[DType.int32, RPT](0)
    var hi = SIMD[DType.int32, RPT](-1)
    var mrow = SIMD[DType.float32, RPT](0.0)
    var drow = SIMD[DType.float32, RPT](0.0)
    comptime for u in range(RPT):
        var t = t0 + tr + u * 16
        if t < l:
            var rr = _row_range(t, pos0, key_lo, window, s)
            lo[u] = Int32(rr[0])
            hi[u] = Int32(rr[1])
            mrow[u] = ftz(amax.unsafe_load(hbase + t))
            drow[u] = ftz(denom.unsafe_load(hbase + t))

    for kb in range(kb_lo, kb_hi + 1):
        # Every y and dy chain starts from +0.0 at its key's block iteration.
        var ydots = SIMD[DType.float32, RPT * KPT](0.0)
        var ddots = SIMD[DType.float32, RPT * KPT](0.0)
        comptime for pw in range(HD // KS):
            # One page per window: Q rows [0, TQ), dctx rows [TQ, 2 TQ), K
            # keys [2 TQ, 2 TQ + BK), V keys [2 TQ + BK, NR), each through
            # `ftz`, KS values per row at stride KS + 4.
            comptime for si in range(NR * KS // 256):
                var i = tid + si * 256
                var r = i // KS
                var p = i % KS
                var x = Float32(0.0)
                if r < TQ:
                    var t = t0 + r
                    if t < l:
                        x = ftz(q_rope.unsafe_load((bb * l + t) * nh * HD + h * HD + pw * KS + p))
                elif r < 2 * TQ:
                    var t = t0 + r - TQ
                    if t < l:
                        x = ftz(dctx.unsafe_load((bb * l + t) * nh * HD + h * HD + pw * KS + p))
                elif r < 2 * TQ + BK:
                    var j = kb * BK + r - 2 * TQ
                    if j < s:
                        x = ftz(k_cache.unsafe_load(kvbase + j * HD + pw * KS + p))
                else:
                    var j = kb * BK + r - 2 * TQ - BK
                    if j < s:
                        x = ftz(v_cache.unsafe_load(kvbase + j * HD + pw * KS + p))
                stg.unsafe_store(r * STRIDE + p, x)
            barrier()
            comptime for p in range(KS):
                var qa = SIMD[DType.float32, RPT](0.0)
                var da = SIMD[DType.float32, RPT](0.0)
                var ka = SIMD[DType.float32, KPT](0.0)
                var va = SIMD[DType.float32, KPT](0.0)
                comptime for u in range(RPT):
                    qa[u] = stg.unsafe_load((tr + u * 16) * STRIDE + p)
                    da[u] = stg.unsafe_load((TQ + tr + u * 16) * STRIDE + p)
                comptime for v in range(KPT):
                    ka[v] = stg.unsafe_load((2 * TQ + tc + v * 16) * STRIDE + p)
                    va[v] = stg.unsafe_load((2 * TQ + BK + tc + v * 16) * STRIDE + p)
                comptime for u in range(RPT):
                    comptime for v in range(KPT):
                        ydots[u * KPT + v] = _step_preflushed(qa[u], ka[v], ydots[u * KPT + v])
                        ddots[u * KPT + v] = _step_preflushed(da[u], va[v], ddots[u * KPT + v])
            barrier()
        # The shipped cell arithmetic, visible cells only, into the stashes.
        comptime for u in range(RPT):
            var t = t0 + tr + u * 16
            comptime for v in range(KPT):
                var j = kb * BK + tc + v * 16
                if t < l and Int32(j) >= lo[u] and Int32(j) <= hi[u]:
                    var cell = (hbase + t) * s + j
                    var masked = ftz(_pmul(ydots[u * KPT + v], scale_in) + Float32(0.0))
                    var e = ftz(identical_exp(ftz(ftz(masked) - mrow[u])))
                    var yv = ftz(identical_div(ftz(e), drow[u]))
                    comptime if SABOTAGE:
                        y_st.unsafe_store(cell, _flip_ulp(yv))
                    else:
                        y_st.unsafe_store(cell, yv)
                    dy_st.unsafe_store(cell, ftz(ddots[u * KPT + v]))


def fused_bwd_zfold_kernel[TZ: Int](
    zdot: MutPointer[Float32, MutAnyOrigin],
    corner: MutPointer[Float32, MutAnyOrigin],
    y_st: MutPointer[Float32, MutAnyOrigin],
    dy_st: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    nh_in: Int32,
    s_in: Int32,
    pos0_in: Int32,
    key_lo_in: Int32,
    window_in: Int32,
):
    """Kernel B of DEVIATION 2528; see the comment above. `TZ` query rows
    per block, one per thread (launched with `TZ` threads per block). The
    key loop runs the BLOCK's key range so every thread's trip count is the
    same; each step is taken only where the key is in the thread's row
    range, which is the shipped chain's term set."""
    var b = Int(b_in)
    var l = Int(l_in)
    var nh = Int(nh_in)
    var s = Int(s_in)
    var pos0 = Int(pos0_in)
    var key_lo = Int(key_lo_in)
    var window = Int(window_in)

    var ntb = (l + TZ - 1) // TZ
    var raw = Int(block_idx.x)
    var tb = raw % ntb
    var rest = raw // ntb
    var h = rest % nh
    var bb = rest // nh
    if bb >= b:
        return

    var tid = Int(thread_idx.x)
    var t0 = tb * TZ
    var t1 = t0 + TZ - 1
    if t1 > l - 1:
        t1 = l - 1
    var r0 = _row_range(t0, pos0, key_lo, window, s)
    var r1 = _row_range(t1, pos0, key_lo, window, s)
    var t = t0 + tid
    var valid = t < l
    var tt = t
    if not valid:
        tt = l - 1
    var rr = _row_range(tt, pos0, key_lo, window, s)
    var j_lo = rr[0]
    var j_hi = rr[1]
    var row = (bb * nh + h) * l + tt
    var stbase = row * s

    var z = Float32(0.0)
    for j in range(r0[0], r1[1] + 1):
        if valid and j >= j_lo and j <= j_hi:
            z = _step(dy_st.unsafe_load(stbase + j), y_st.unsafe_load(stbase + j), z)
    if valid:
        var zf = ftz(z)
        if bitcast[DType.uint32](zf) == NEG_ZERO_BITS and j_hi < s - 1:
            corner.unsafe_store(0, Float32(1.0))
        zdot.unsafe_store(row, zf)


# ===========================================================================
# THE LAUNCHERS. Raw buffers in, so that neither the forward's nor the
# backward's stage struct has to be imported here (both import this file).
# ===========================================================================


def _read_flag(
    ctx: DeviceContext, mut flag: DeviceBuffer[DType.float32]
) raises -> Bool:
    var host = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=flag)
    ctx.synchronize()
    var v = host.unsafe_ptr().unsafe_load(0)
    _ = host^
    return v != Float32(0.0)


def _zero_flag(ctx: DeviceContext) raises -> DeviceBuffer[DType.float32]:
    var f = ctx.enqueue_create_buffer[DType.float32](1)
    f.enqueue_fill(Float32(0.0))
    ctx.synchronize()
    return f^


def fused_forward_launch(
    ctx: DeviceContext,
    mut ctxv: DeviceBuffer[DType.float32],
    mut amax: DeviceBuffer[DType.float32],
    mut denom: DeviceBuffer[DType.float32],
    mut q_rope: DeviceBuffer[DType.float32],
    mut k_cache: DeviceBuffer[DType.float32],
    mut v_cache: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    nh: Int,
    nkv: Int,
    hd: Int,
    s: Int,
    pos0: Int,
    key_lo: Int,
    window: Int,
    scale: Float32,
) raises -> Int:
    """The fused forward: `ctxv`, `amax` and `denom` are written on
    `FUSED_RAN`; on any other status nothing the caller reads is defined
    and the eager path must run. `k_cache`/`v_cache` are the PACKED span
    `[B, n_kv, S, head_dim]` the eager path reads. The arm is the build
    default, or the environment's on a trial build
    (`fused_attention_arm_from_env`)."""
    return fused_forward_launch_arm(
        ctx, ctxv, amax, denom, q_rope, k_cache, v_cache, b, l, nh, nkv,
        hd, s, pos0, key_lo, window, scale, fused_attention_arm_from_env(),
    )


def fused_forward_launch_arm(
    ctx: DeviceContext,
    mut ctxv: DeviceBuffer[DType.float32],
    mut amax: DeviceBuffer[DType.float32],
    mut denom: DeviceBuffer[DType.float32],
    mut q_rope: DeviceBuffer[DType.float32],
    mut k_cache: DeviceBuffer[DType.float32],
    mut v_cache: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    nh: Int,
    nkv: Int,
    hd: Int,
    s: Int,
    pos0: Int,
    key_lo: Int,
    window: Int,
    scale: Float32,
    arm: Int,
) raises -> Int:
    """`fused_forward_launch` with the arm given (the harness alternates
    arms inside one process). A candidate arm's kernels exist on a
    `-D MOJOLEARN_ATTN_ARM_TRIAL=1` build, or when it is ATTN_ARM_DEFAULT
    (clean kernels only); otherwise the arm value runs the shipped kernels,
    so a harness must prove reach by sabotage rather than trust the arm it
    asked for."""
    if not fused_forward_supported_head_dim(hd):
        return FUSED_REFUSED_REGIME
    var ton = _attn_timer_on()
    var tk = Int(perf_counter_ns())
    var qmax = device_absmax(ctx, q_rope, b * l * nh * hd)
    var kmax = device_absmax(ctx, k_cache, b * nkv * s * hd)
    var vmax = device_absmax(ctx, v_cache, b * nkv * s * hd)
    if not regime_product_ok(hd, qmax, kmax) or not regime_finite(vmax):
        return FUSED_REFUSED_REGIME
    _attn_tick(ctx, ton, tk, "fwd_regime_scan")
    var corner = _zero_flag(ctx)
    var tq = fused_rows_per_block(hd)
    var blocks = b * nh * ((l + tq - 1) // tq)
    var nt = hd * tq
    var ran_arm = False
    comptime if ATTN_ARM_COMPILED:
        var sabotage = (arm & ATTN_ARM_SABOTAGE) != 0
        var want_sstash = (arm & ATTN_ARM_FWD_SSTASH) != 0 and hd == ATTN_STASH_HD
        if want_sstash:
            # DEVIATION 2526: the score/exp scratch, `[B, n_heads, L, S]`
            # at the call's packed stride, owned for this call.
            var sstash = ctx.enqueue_create_buffer[DType.float32](b * nh * l * s)
            ctx.synchronize()
            _attn_tick(ctx, ton, tk, "fwd_scratch_alloc")
            blocks = b * nh * ((l + 63) // 64)
            nt = FUSED_THREADS
            if sabotage:
                comptime if ATTN_ARM_TRIAL:
                    comptime ks = fused_attn_forward_regblocked_sstash_kernel[ATTN_STASH_HD, True]
                    ctx.enqueue_function[ks](
                        ctxv.unsafe_ptr(), amax.unsafe_ptr(), denom.unsafe_ptr(),
                        corner.unsafe_ptr(), sstash.unsafe_ptr(), q_rope.unsafe_ptr(),
                        k_cache.unsafe_ptr(), v_cache.unsafe_ptr(), Int32(b), Int32(l),
                        Int32(nh), Int32(nkv), Int32(s), Int32(pos0), Int32(key_lo),
                        Int32(window), scale,
                        grid_dim=(blocks, 1, 1), block_dim=(nt, 1, 1),
                    )
            else:
                comptime kc = fused_attn_forward_regblocked_sstash_kernel[ATTN_STASH_HD, False]
                ctx.enqueue_function[kc](
                    ctxv.unsafe_ptr(), amax.unsafe_ptr(), denom.unsafe_ptr(),
                    corner.unsafe_ptr(), sstash.unsafe_ptr(), q_rope.unsafe_ptr(),
                    k_cache.unsafe_ptr(), v_cache.unsafe_ptr(), Int32(b), Int32(l),
                    Int32(nh), Int32(nkv), Int32(s), Int32(pos0), Int32(key_lo),
                    Int32(window), scale,
                    grid_dim=(blocks, 1, 1), block_dim=(nt, 1, 1),
                )
            ctx.synchronize()
            _attn_tick(ctx, ton, tk, "fwd_sstash_kernel")
            _ = sstash^
            ran_arm = True
    if not ran_arm:
        if hd == 16:
            comptime k16 = fused_attn_forward_kernel[16, FUSED_THREADS // 16]
            ctx.enqueue_function[k16](
                ctxv.unsafe_ptr(), amax.unsafe_ptr(), denom.unsafe_ptr(),
                corner.unsafe_ptr(), q_rope.unsafe_ptr(), k_cache.unsafe_ptr(),
                v_cache.unsafe_ptr(), Int32(b), Int32(l), Int32(nh), Int32(nkv),
                Int32(s), Int32(pos0), Int32(key_lo), Int32(window), scale,
                grid_dim=(blocks, 1, 1), block_dim=(nt, 1, 1),
            )
        elif hd == 24:
            comptime k24 = fused_attn_forward_kernel[24, FUSED_THREADS // 24]
            ctx.enqueue_function[k24](
                ctxv.unsafe_ptr(), amax.unsafe_ptr(), denom.unsafe_ptr(),
                corner.unsafe_ptr(), q_rope.unsafe_ptr(), k_cache.unsafe_ptr(),
                v_cache.unsafe_ptr(), Int32(b), Int32(l), Int32(nh), Int32(nkv),
                Int32(s), Int32(pos0), Int32(key_lo), Int32(window), scale,
                grid_dim=(blocks, 1, 1), block_dim=(nt, 1, 1),
            )
        elif hd == 64:
            comptime k64 = fused_attn_forward_regblocked_kernel[64]
            blocks = b * nh * ((l + 63) // 64)
            nt = FUSED_THREADS
            ctx.enqueue_function[k64](
                ctxv.unsafe_ptr(), amax.unsafe_ptr(), denom.unsafe_ptr(),
                corner.unsafe_ptr(), q_rope.unsafe_ptr(), k_cache.unsafe_ptr(),
                v_cache.unsafe_ptr(), Int32(b), Int32(l), Int32(nh), Int32(nkv),
                Int32(s), Int32(pos0), Int32(key_lo), Int32(window), scale,
                grid_dim=(blocks, 1, 1), block_dim=(nt, 1, 1),
            )
        else:
            comptime k128 = fused_attn_forward_regblocked_kernel[128]
            blocks = b * nh * ((l + 15) // 16)
            nt = FUSED_THREADS
            ctx.enqueue_function[k128](
                ctxv.unsafe_ptr(), amax.unsafe_ptr(), denom.unsafe_ptr(),
                corner.unsafe_ptr(), q_rope.unsafe_ptr(), k_cache.unsafe_ptr(),
                v_cache.unsafe_ptr(), Int32(b), Int32(l), Int32(nh), Int32(nkv),
                Int32(s), Int32(pos0), Int32(key_lo), Int32(window), scale,
                grid_dim=(blocks, 1, 1), block_dim=(nt, 1, 1),
            )
        _attn_tick(ctx, ton, tk, "fwd_kernel")
    ctx.synchronize()
    var hit = _read_flag(ctx, corner)
    _ = corner^
    _attn_tick(ctx, ton, tk, "fwd_corner_flag")
    if hit:
        return FUSED_CORNER
    return FUSED_RAN


def fused_backward_launch(
    ctx: DeviceContext,
    mut zdot: DeviceBuffer[DType.float32],
    mut dq: DeviceBuffer[DType.float32],
    mut dk: DeviceBuffer[DType.float32],
    mut dv: DeviceBuffer[DType.float32],
    mut q_rope: DeviceBuffer[DType.float32],
    mut dctx: DeviceBuffer[DType.float32],
    mut k_cache: DeviceBuffer[DType.float32],
    mut v_cache: DeviceBuffer[DType.float32],
    mut amax: DeviceBuffer[DType.float32],
    mut denom: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    nh: Int,
    nkv: Int,
    hd: Int,
    s: Int,
    pos0: Int,
    key_lo: Int,
    window: Int,
    scale: Float32,
) raises -> Int:
    """The fused backward: `zdot` (stage 18), `dq` (22, `[M, nh*hd]`),
    `dk` and `dv` (23-24, `[B, n_kv, S, hd]`) on `FUSED_RAN`. `amax` and
    `denom` are the forward's row scalars (either path writes them). The
    arm is the build default, or the environment's on a trial build."""
    return fused_backward_launch_arm(
        ctx, zdot, dq, dk, dv, q_rope, dctx, k_cache, v_cache, amax, denom,
        b, l, nh, nkv, hd, s, pos0, key_lo, window, scale,
        fused_attention_arm_from_env(),
    )


def _launch_bwd_shipped[HD: Int](
    ctx: DeviceContext,
    on: Bool,
    mut tk: Int,
    mut zdot: DeviceBuffer[DType.float32],
    mut dq: DeviceBuffer[DType.float32],
    mut dk: DeviceBuffer[DType.float32],
    mut dv: DeviceBuffer[DType.float32],
    mut corner: DeviceBuffer[DType.float32],
    mut q_rope: DeviceBuffer[DType.float32],
    mut dctx: DeviceBuffer[DType.float32],
    mut k_cache: DeviceBuffer[DType.float32],
    mut v_cache: DeviceBuffer[DType.float32],
    mut amax: DeviceBuffer[DType.float32],
    mut denom: DeviceBuffer[DType.float32],
    b: Int, l: Int, nh: Int, nkv: Int, s: Int, pos0: Int, key_lo: Int,
    window: Int, scale: Float32, row_blocks: Int, key_blocks: Int, nt: Int,
) raises:
    """The three shipped backward kernels at `HD`, exactly as launched
    before the arm hook (zdot, a wait, then dq and dkdv in stream order),
    with the per-kernel ticks between them."""
    comptime TQ = FUSED_THREADS // HD
    comptime z = fused_bwd_zdot_kernel[HD, TQ]
    comptime q = fused_bwd_dq_kernel[HD, TQ]
    comptime kv = fused_bwd_dkdv_kernel[HD, TQ]
    ctx.enqueue_function[z](
        zdot.unsafe_ptr(), corner.unsafe_ptr(), q_rope.unsafe_ptr(),
        dctx.unsafe_ptr(), k_cache.unsafe_ptr(), v_cache.unsafe_ptr(),
        amax.unsafe_ptr(), denom.unsafe_ptr(), Int32(b), Int32(l),
        Int32(nh), Int32(nkv), Int32(s), Int32(pos0), Int32(key_lo),
        Int32(window), scale,
        grid_dim=(row_blocks, 1, 1), block_dim=(nt, 1, 1),
    )
    ctx.synchronize()
    _attn_tick(ctx, on, tk, "bwd_zdot")
    ctx.enqueue_function[q](
        dq.unsafe_ptr(), corner.unsafe_ptr(), q_rope.unsafe_ptr(),
        dctx.unsafe_ptr(), k_cache.unsafe_ptr(), v_cache.unsafe_ptr(),
        amax.unsafe_ptr(), denom.unsafe_ptr(), zdot.unsafe_ptr(),
        Int32(b), Int32(l), Int32(nh), Int32(nkv), Int32(s), Int32(pos0),
        Int32(key_lo), Int32(window), scale,
        grid_dim=(row_blocks, 1, 1), block_dim=(nt, 1, 1),
    )
    _attn_tick(ctx, on, tk, "bwd_dq")
    ctx.enqueue_function[kv](
        dk.unsafe_ptr(), dv.unsafe_ptr(), corner.unsafe_ptr(),
        q_rope.unsafe_ptr(), dctx.unsafe_ptr(), k_cache.unsafe_ptr(),
        v_cache.unsafe_ptr(), amax.unsafe_ptr(), denom.unsafe_ptr(),
        zdot.unsafe_ptr(), Int32(b), Int32(l), Int32(nh), Int32(nkv),
        Int32(s), Int32(pos0), Int32(key_lo), Int32(window), scale,
        grid_dim=(key_blocks, 1, 1), block_dim=(nt, 1, 1),
    )
    _attn_tick(ctx, on, tk, "bwd_dkdv")


def _launch_bwd_ztiled[HD: Int, TQZ: Int, ZSAB: Bool](
    ctx: DeviceContext,
    on: Bool,
    mut tk: Int,
    mut zdot: DeviceBuffer[DType.float32],
    mut dq: DeviceBuffer[DType.float32],
    mut dk: DeviceBuffer[DType.float32],
    mut dv: DeviceBuffer[DType.float32],
    mut corner: DeviceBuffer[DType.float32],
    mut q_rope: DeviceBuffer[DType.float32],
    mut dctx: DeviceBuffer[DType.float32],
    mut k_cache: DeviceBuffer[DType.float32],
    mut v_cache: DeviceBuffer[DType.float32],
    mut amax: DeviceBuffer[DType.float32],
    mut denom: DeviceBuffer[DType.float32],
    b: Int, l: Int, nh: Int, nkv: Int, s: Int, pos0: Int, key_lo: Int,
    window: Int, scale: Float32, fold_sabotage: Bool,
) raises:
    """DEVIATION 2528's backward (trial builds only): the two stashes,
    kernel A at `TQZ` rows per block (`ZSAB` its sabotage), kernel B, then
    the 2527 tiled dq and dk/dv folds launched exactly as the stash_tiled
    branch launches them (`fold_sabotage` picks their first-round sabotage
    instantiations). Generic, so a build that never calls it (every shipped
    build) instantiates none of the new kernels."""
    var cells = b * nh * l * s
    var y_st = ctx.enqueue_create_buffer[DType.float32](cells)
    var dy_st = ctx.enqueue_create_buffer[DType.float32](cells)
    ctx.synchronize()
    _attn_tick(ctx, on, tk, "bwd_scratch_alloc")
    comptime ka = fused_bwd_ydy_tiled_kernel[HD, TQZ, ZSAB]
    ctx.enqueue_function[ka](
        y_st.unsafe_ptr(), dy_st.unsafe_ptr(), q_rope.unsafe_ptr(),
        dctx.unsafe_ptr(), k_cache.unsafe_ptr(), v_cache.unsafe_ptr(),
        amax.unsafe_ptr(), denom.unsafe_ptr(), Int32(b), Int32(l),
        Int32(nh), Int32(nkv), Int32(s), Int32(pos0), Int32(key_lo),
        Int32(window), scale,
        grid_dim=(b * nh * ((l + TQZ - 1) // TQZ), 1, 1),
        block_dim=(FUSED_THREADS, 1, 1),
    )
    _attn_tick(ctx, on, tk, "bwd_ydy_tiled")
    comptime kz = fused_bwd_zfold_kernel[FUSED_THREADS]
    ctx.enqueue_function[kz](
        zdot.unsafe_ptr(), corner.unsafe_ptr(), y_st.unsafe_ptr(),
        dy_st.unsafe_ptr(), Int32(b), Int32(l), Int32(nh), Int32(s),
        Int32(pos0), Int32(key_lo), Int32(window),
        grid_dim=(b * nh * ((l + FUSED_THREADS - 1) // FUSED_THREADS), 1, 1),
        block_dim=(FUSED_THREADS, 1, 1),
    )
    ctx.synchronize()
    _attn_tick(ctx, on, tk, "bwd_zfold")
    var dq_blocks = b * nh * ((l + 63) // 64)
    var kv_blocks = b * nkv * ((s + 63) // 64)
    if fold_sabotage:
        comptime qs = fused_bwd_dq_tiled_kernel[HD, True]
        ctx.enqueue_function[qs](
            dq.unsafe_ptr(), corner.unsafe_ptr(), y_st.unsafe_ptr(),
            dy_st.unsafe_ptr(), k_cache.unsafe_ptr(), zdot.unsafe_ptr(),
            Int32(b), Int32(l), Int32(nh), Int32(nkv), Int32(s),
            Int32(pos0), Int32(key_lo), Int32(window), scale,
            grid_dim=(dq_blocks, 1, 1), block_dim=(FUSED_THREADS, 1, 1),
        )
    else:
        comptime qc = fused_bwd_dq_tiled_kernel[HD, False]
        ctx.enqueue_function[qc](
            dq.unsafe_ptr(), corner.unsafe_ptr(), y_st.unsafe_ptr(),
            dy_st.unsafe_ptr(), k_cache.unsafe_ptr(), zdot.unsafe_ptr(),
            Int32(b), Int32(l), Int32(nh), Int32(nkv), Int32(s),
            Int32(pos0), Int32(key_lo), Int32(window), scale,
            grid_dim=(dq_blocks, 1, 1), block_dim=(FUSED_THREADS, 1, 1),
        )
    _attn_tick(ctx, on, tk, "bwd_dq_tiled")
    if fold_sabotage:
        comptime kvs = fused_bwd_dkdv_tiled_kernel[HD, True]
        ctx.enqueue_function[kvs](
            dk.unsafe_ptr(), dv.unsafe_ptr(), corner.unsafe_ptr(),
            y_st.unsafe_ptr(), dy_st.unsafe_ptr(), q_rope.unsafe_ptr(),
            dctx.unsafe_ptr(), Int32(b), Int32(l), Int32(nh), Int32(nkv),
            Int32(s), Int32(pos0), Int32(key_lo), Int32(window),
            grid_dim=(kv_blocks, 1, 1), block_dim=(FUSED_THREADS, 1, 1),
        )
    else:
        comptime kvc = fused_bwd_dkdv_tiled_kernel[HD, False]
        ctx.enqueue_function[kvc](
            dk.unsafe_ptr(), dv.unsafe_ptr(), corner.unsafe_ptr(),
            y_st.unsafe_ptr(), dy_st.unsafe_ptr(), q_rope.unsafe_ptr(),
            dctx.unsafe_ptr(), Int32(b), Int32(l), Int32(nh), Int32(nkv),
            Int32(s), Int32(pos0), Int32(key_lo), Int32(window),
            grid_dim=(kv_blocks, 1, 1), block_dim=(FUSED_THREADS, 1, 1),
        )
    _attn_tick(ctx, on, tk, "bwd_dkdv_tiled")
    # The stashes must outlive the enqueued kernels: synchronize, then the
    # explicit last use (a buffer is freed at its last use).
    ctx.synchronize()
    _ = y_st^
    _ = dy_st^


def fused_backward_launch_arm(
    ctx: DeviceContext,
    mut zdot: DeviceBuffer[DType.float32],
    mut dq: DeviceBuffer[DType.float32],
    mut dk: DeviceBuffer[DType.float32],
    mut dv: DeviceBuffer[DType.float32],
    mut q_rope: DeviceBuffer[DType.float32],
    mut dctx: DeviceBuffer[DType.float32],
    mut k_cache: DeviceBuffer[DType.float32],
    mut v_cache: DeviceBuffer[DType.float32],
    mut amax: DeviceBuffer[DType.float32],
    mut denom: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    nh: Int,
    nkv: Int,
    hd: Int,
    s: Int,
    pos0: Int,
    key_lo: Int,
    window: Int,
    scale: Float32,
    arm: Int,
) raises -> Int:
    """`fused_backward_launch` with the arm given; see
    `fused_forward_launch_arm` for what an arm value means on a build
    without the trial hook."""
    if not fused_supported_head_dim(hd):
        return FUSED_REFUSED_REGIME
    comptime if ATTN_OPERAND_DUMP:
        _maybe_dump_operands(
            ctx, q_rope, dctx, k_cache, v_cache, b, l, nh, nkv, hd, s, pos0,
            key_lo, window, scale,
        )
    var ton = _attn_timer_on()
    var tk = Int(perf_counter_ns())
    var qmax = device_absmax(ctx, q_rope, b * l * nh * hd)
    var kmax = device_absmax(ctx, k_cache, b * nkv * s * hd)
    var vmax = device_absmax(ctx, v_cache, b * nkv * s * hd)
    var dmax = device_absmax(ctx, dctx, b * l * nh * hd)
    if not regime_product_ok(hd, qmax, kmax):
        return FUSED_REFUSED_REGIME
    if not regime_product_ok(hd, dmax, vmax):
        return FUSED_REFUSED_REGIME
    _attn_tick(ctx, ton, tk, "bwd_regime_scan")
    var corner = _zero_flag(ctx)
    var tq = fused_rows_per_block(hd)
    var row_blocks = b * nh * ((l + tq - 1) // tq)
    var key_blocks = b * nkv * ((s + tq - 1) // tq)
    var nt = hd * tq
    var ran_arm = False
    comptime HD = ATTN_STASH_HD
    comptime TQ = FUSED_THREADS // ATTN_STASH_HD
    comptime if ATTN_ARM_COMPILED:
        var sabotage = (arm & ATTN_ARM_SABOTAGE) != 0
        var want_stash = (arm & ATTN_ARM_BWD_STASH) != 0 and hd == ATTN_STASH_HD
        var want_tiled = (arm & ATTN_ARM_BWD_TILED) != 0
        comptime if ATTN_ARM_TRIAL:
            # DEVIATION 2528 (brief section 12.2), trial builds only. It runs
            # before the first-round branch, which then runs only when this
            # one did not; a shipped build compiles none of it.
            var zrows = fused_attention_zdot_rows(arm)
            if (arm & ATTN_ARM_BWD_ZTILED) != 0 and want_stash and want_tiled and zrows != 0:
                var zsab = (arm & ATTN_ARM_SABOTAGE_NEW) != 0
                if zrows == 32:
                    if zsab:
                        _launch_bwd_ztiled[HD, 32, True](
                            ctx, ton, tk, zdot, dq, dk, dv, corner, q_rope, dctx,
                            k_cache, v_cache, amax, denom, b, l, nh, nkv, s, pos0,
                            key_lo, window, scale, sabotage,
                        )
                    else:
                        _launch_bwd_ztiled[HD, 32, False](
                            ctx, ton, tk, zdot, dq, dk, dv, corner, q_rope, dctx,
                            k_cache, v_cache, amax, denom, b, l, nh, nkv, s, pos0,
                            key_lo, window, scale, sabotage,
                        )
                else:
                    if zsab:
                        _launch_bwd_ztiled[HD, 64, True](
                            ctx, ton, tk, zdot, dq, dk, dv, corner, q_rope, dctx,
                            k_cache, v_cache, amax, denom, b, l, nh, nkv, s, pos0,
                            key_lo, window, scale, sabotage,
                        )
                    else:
                        _launch_bwd_ztiled[HD, 64, False](
                            ctx, ton, tk, zdot, dq, dk, dv, corner, q_rope, dctx,
                            k_cache, v_cache, amax, denom, b, l, nh, nkv, s, pos0,
                            key_lo, window, scale, sabotage,
                        )
                ran_arm = True
        if want_stash and not ran_arm:
            # DEVIATIONS 2525 and 2527: the y and dy scratches, `[B,
            # n_heads, L, S]` at the call's packed stride, owned for this
            # call; dy becomes dcell after the dq kernel.
            var cells = b * nh * l * s
            var y_st = ctx.enqueue_create_buffer[DType.float32](cells)
            var dy_st = ctx.enqueue_create_buffer[DType.float32](cells)
            ctx.synchronize()
            _attn_tick(ctx, ton, tk, "bwd_scratch_alloc")
            comptime zk = fused_bwd_zdot_stash_kernel[HD, TQ]
            ctx.enqueue_function[zk](
                zdot.unsafe_ptr(), corner.unsafe_ptr(), y_st.unsafe_ptr(),
                dy_st.unsafe_ptr(), q_rope.unsafe_ptr(), dctx.unsafe_ptr(),
                k_cache.unsafe_ptr(), v_cache.unsafe_ptr(), amax.unsafe_ptr(),
                denom.unsafe_ptr(), Int32(b), Int32(l), Int32(nh), Int32(nkv),
                Int32(s), Int32(pos0), Int32(key_lo), Int32(window), scale,
                grid_dim=(row_blocks, 1, 1), block_dim=(nt, 1, 1),
            )
            ctx.synchronize()
            _attn_tick(ctx, ton, tk, "bwd_zdot_stash")
            if want_tiled:
                var dq_blocks = b * nh * ((l + 63) // 64)
                var kv_blocks = b * nkv * ((s + 63) // 64)
                if sabotage:
                    comptime if ATTN_ARM_TRIAL:
                        comptime qs = fused_bwd_dq_tiled_kernel[HD, True]
                        ctx.enqueue_function[qs](
                            dq.unsafe_ptr(), corner.unsafe_ptr(), y_st.unsafe_ptr(),
                            dy_st.unsafe_ptr(), k_cache.unsafe_ptr(), zdot.unsafe_ptr(),
                            Int32(b), Int32(l), Int32(nh), Int32(nkv), Int32(s),
                            Int32(pos0), Int32(key_lo), Int32(window), scale,
                            grid_dim=(dq_blocks, 1, 1), block_dim=(FUSED_THREADS, 1, 1),
                        )
                else:
                    comptime qc = fused_bwd_dq_tiled_kernel[HD, False]
                    ctx.enqueue_function[qc](
                        dq.unsafe_ptr(), corner.unsafe_ptr(), y_st.unsafe_ptr(),
                        dy_st.unsafe_ptr(), k_cache.unsafe_ptr(), zdot.unsafe_ptr(),
                        Int32(b), Int32(l), Int32(nh), Int32(nkv), Int32(s),
                        Int32(pos0), Int32(key_lo), Int32(window), scale,
                        grid_dim=(dq_blocks, 1, 1), block_dim=(FUSED_THREADS, 1, 1),
                    )
                _attn_tick(ctx, ton, tk, "bwd_dq_tiled")
                if sabotage:
                    comptime if ATTN_ARM_TRIAL:
                        comptime kvs = fused_bwd_dkdv_tiled_kernel[HD, True]
                        ctx.enqueue_function[kvs](
                            dk.unsafe_ptr(), dv.unsafe_ptr(), corner.unsafe_ptr(),
                            y_st.unsafe_ptr(), dy_st.unsafe_ptr(), q_rope.unsafe_ptr(),
                            dctx.unsafe_ptr(), Int32(b), Int32(l), Int32(nh), Int32(nkv),
                            Int32(s), Int32(pos0), Int32(key_lo), Int32(window),
                            grid_dim=(kv_blocks, 1, 1), block_dim=(FUSED_THREADS, 1, 1),
                        )
                else:
                    comptime kvc = fused_bwd_dkdv_tiled_kernel[HD, False]
                    ctx.enqueue_function[kvc](
                        dk.unsafe_ptr(), dv.unsafe_ptr(), corner.unsafe_ptr(),
                        y_st.unsafe_ptr(), dy_st.unsafe_ptr(), q_rope.unsafe_ptr(),
                        dctx.unsafe_ptr(), Int32(b), Int32(l), Int32(nh), Int32(nkv),
                        Int32(s), Int32(pos0), Int32(key_lo), Int32(window),
                        grid_dim=(kv_blocks, 1, 1), block_dim=(FUSED_THREADS, 1, 1),
                    )
                _attn_tick(ctx, ton, tk, "bwd_dkdv_tiled")
            else:
                if sabotage:
                    comptime if ATTN_ARM_TRIAL:
                        comptime qs2 = fused_bwd_dq_stash_kernel[HD, TQ, True]
                        ctx.enqueue_function[qs2](
                            dq.unsafe_ptr(), corner.unsafe_ptr(), y_st.unsafe_ptr(),
                            dy_st.unsafe_ptr(), k_cache.unsafe_ptr(), zdot.unsafe_ptr(),
                            Int32(b), Int32(l), Int32(nh), Int32(nkv), Int32(s),
                            Int32(pos0), Int32(key_lo), Int32(window), scale,
                            grid_dim=(row_blocks, 1, 1), block_dim=(nt, 1, 1),
                        )
                else:
                    comptime qc2 = fused_bwd_dq_stash_kernel[HD, TQ, False]
                    ctx.enqueue_function[qc2](
                        dq.unsafe_ptr(), corner.unsafe_ptr(), y_st.unsafe_ptr(),
                        dy_st.unsafe_ptr(), k_cache.unsafe_ptr(), zdot.unsafe_ptr(),
                        Int32(b), Int32(l), Int32(nh), Int32(nkv), Int32(s),
                        Int32(pos0), Int32(key_lo), Int32(window), scale,
                        grid_dim=(row_blocks, 1, 1), block_dim=(nt, 1, 1),
                    )
                _attn_tick(ctx, ton, tk, "bwd_dq_stash")
                if sabotage:
                    comptime if ATTN_ARM_TRIAL:
                        comptime kvs2 = fused_bwd_dkdv_stash_kernel[HD, TQ, True]
                        ctx.enqueue_function[kvs2](
                            dk.unsafe_ptr(), dv.unsafe_ptr(), corner.unsafe_ptr(),
                            y_st.unsafe_ptr(), dy_st.unsafe_ptr(), q_rope.unsafe_ptr(),
                            dctx.unsafe_ptr(), Int32(b), Int32(l), Int32(nh), Int32(nkv),
                            Int32(s), Int32(pos0), Int32(key_lo), Int32(window),
                            grid_dim=(key_blocks, 1, 1), block_dim=(nt, 1, 1),
                        )
                else:
                    comptime kvc2 = fused_bwd_dkdv_stash_kernel[HD, TQ, False]
                    ctx.enqueue_function[kvc2](
                        dk.unsafe_ptr(), dv.unsafe_ptr(), corner.unsafe_ptr(),
                        y_st.unsafe_ptr(), dy_st.unsafe_ptr(), q_rope.unsafe_ptr(),
                        dctx.unsafe_ptr(), Int32(b), Int32(l), Int32(nh), Int32(nkv),
                        Int32(s), Int32(pos0), Int32(key_lo), Int32(window),
                        grid_dim=(key_blocks, 1, 1), block_dim=(nt, 1, 1),
                    )
                _attn_tick(ctx, ton, tk, "bwd_dkdv_stash")
            ctx.synchronize()
            _ = y_st^
            _ = dy_st^
            ran_arm = True
    if not ran_arm:
        if hd == 16:
            _launch_bwd_shipped[16](
                ctx, ton, tk, zdot, dq, dk, dv, corner, q_rope, dctx, k_cache,
                v_cache, amax, denom, b, l, nh, nkv, s, pos0, key_lo, window,
                scale, row_blocks, key_blocks, nt,
            )
        elif hd == 24:
            _launch_bwd_shipped[24](
                ctx, ton, tk, zdot, dq, dk, dv, corner, q_rope, dctx, k_cache,
                v_cache, amax, denom, b, l, nh, nkv, s, pos0, key_lo, window,
                scale, row_blocks, key_blocks, nt,
            )
        elif hd == 64:
            _launch_bwd_shipped[64](
                ctx, ton, tk, zdot, dq, dk, dv, corner, q_rope, dctx, k_cache,
                v_cache, amax, denom, b, l, nh, nkv, s, pos0, key_lo, window,
                scale, row_blocks, key_blocks, nt,
            )
        else:
            _launch_bwd_shipped[128](
                ctx, ton, tk, zdot, dq, dk, dv, corner, q_rope, dctx, k_cache,
                v_cache, amax, denom, b, l, nh, nkv, s, pos0, key_lo, window,
                scale, row_blocks, key_blocks, nt,
            )
    ctx.synchronize()
    var hit = _read_flag(ctx, corner)
    _ = corner^
    _attn_tick(ctx, ton, tk, "bwd_corner_flag")
    if hit:
        return FUSED_CORNER
    return FUSED_RAN
