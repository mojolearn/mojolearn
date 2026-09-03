# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Mamba-3 SISO core ON THE DEVICE -- seams S7-S20 and S22 plus the
four report pieces -- under profile `mojolearn.identical.mamba3.siso.fp32.v1`
(`mamba/IDENTICAL_MAMBA3_CONTRACT.md`). **COPY, DO NOT IMPROVE.**

Upstream: the chunked schedule SHAPE is
`mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py::mamba3_siso_fwd_kernel`
(state-spaces/mamba `e9594ce`; phase 1 :276-351, phase 2 :353-451, the
final-state pick :709-729) with the VALUES pinned to
`tests/ops/triton/test_mamba3_siso.py::mamba3_siso_fwd_ref` (:149-340) as
the contract's section 4 spells them; the angle chain's shape is
`ops/triton/mamba3/angle_dt.py` (:83-122) with its per-chunk mod placement
REFUSED (DEVIATION 829). The kernel's exp2*log2e respelling, its PTX
`cos.approx`/`sin.approx`/`tanh.approx` (utils.py:13-69), its inert
`min(., 0.0)` guard (:412) and its `tl.cumsum`/`tl.sum`/`tl.dot` trees are
NOT pinned -- `identical_exp`, DEVIATION 820's `portable_cosf`/
`portable_sinf`, `identical_tanh`/`identical_sigmoid` and the mamba2
S11/DEVIATION-782 serial spellings are inherited over them. The block
around this core is `mamba/impl/mamba_ssm/modules/mamba3.mojo`; the host
oracle `mamba/checks/mamba3_oracle.mojo` is the ANSWER bit for bit, and
this file is an independent transcription of the same order sharing only
`checks/numerics.mojo` with it.

WHY NO FLOAT CROSSES A THREAD BOUNDARY IN THIS FILE (the mamba2 rule,
unchanged): every kernel owns its output cell entirely -- the angle
recurrence is one thread per (b, h, angle-index) serial chain; the dacs
cumsum and each segsum column are one thread per serial chain; every
contraction cell (S14's QK dot, S16's s and its fold, S17's read-out,
S20's increment) is one thread computing its gemm-v1 leaf IN REGISTERS
(every core contraction has k <= 128, ONE leaf -- see DEVIATION 834); the
inter-chunk pass is one thread per (b, h, p, n) walking chunks serially.
No shared memory, no warp primitive, no atomic, no cross-block reduction.
Contract clauses 8(b) and 8(c) are properties of this SHAPE.

THE DEVIATIONS IMPLEMENTED HERE (contract section 9 + the oracle's; cited,
never renumbered): 827 (`m3_q_eff()` below is the ONE place the
CHUNK_SIZE_32 arm may rebuild Q); 828 (portable trig pair on the one
rotation spelling both paths share; interleaved (2i, 2i+1) pairs;
UNFUSED two-product rotation; pairs >= num_rope_angles are STRUCTURAL
identity -- biased values copied, no trig ever computed); 829 (per-token
serial angle recurrence, mod 2pi EVERY step; ANGLE_MOD_PER_CHUNK and
ANGLE_MOD_AT_END are this file's arms); 830 (strict-causal S16 with the
diagonal STRUCTURAL +0.0; the diagonal rides S14's gamma-scaled
pre-rotation dot into S18's one add; DIAG_INCLUDE_SUBTRACT is the arm);
831/832 (the working-sequence schedule: the serial pass hands the block
the state ENTERING THE LAST WORKING CHUNK -- the sealed boundary -- as
the carried h, and `h_last` after the final padded chunk as the report);
833 (`y = ftz(ystate + yintra)`, ystate first -- the kernel's accumulator
order, fwd:406-418); S22's resumption correction takes the NORMATIVE
reference's scalar-first association (:266-267) over the kernel's
(fwd:371), the RESUME_KERNEL_ASSOC arm keeping the kernel's.

DEVIATION 834 (NEW, this file owns it) -- FOLD_SERIAL_ZERO_SEED HAS NO
WITNESSABLE IN-CORE SITE AT Q = 64. Contract section 8f lists the arm as
an inherited survivor, but every contraction this core spells has
k <= 128 (in_proj/out_proj are the gemm lane's own code; in-core k is 128
for the n-contractions and Q = 64 for the j/row folds), and
`contract_leaf_size` makes k <= 128 ONE serial ascending leaf -- the
"serial chain replaces the balanced fold" arm is then bit-identical to
the profile, i.e. VACUOUS BY CONSTRUCTION, and an arm that cannot be
witnessed may not be credited (the contract's own 8f rule, which
outranks the survivor list). The flag is still DEFINED here so the armed
build exists and REFUSES BY NAME in the check (never passes, never
silently skips); the fold arm's real site stays the gemm lane's own
certified arms at k > 128.

THE SABOTAGE ARMS THIS FILE OWNS (contract 8f; each a compile-time
alternative spelling of ONE clause, OFF unless -D-named; the check
INVERTS its verdict when one is armed):

    MOJOLEARN_MAMBA3_SABOTAGE_SEGSUM_DESCENDING    dacs/seg folds reversed
    MOJOLEARN_MAMBA3_SABOTAGE_CHUNK_SIZE_32        Q rebuilt at 32 (827's falsifier)
    MOJOLEARN_MAMBA3_SABOTAGE_TRAP_LEFT_ONLY       S15 consumes gamma, not scale (Euler)
    MOJOLEARN_MAMBA3_SABOTAGE_ANGLE_MOD_PER_CHUNK  angle_dt.py's mod placement
    MOJOLEARN_MAMBA3_SABOTAGE_ANGLE_MOD_AT_END     fwd_ref's mod placement
    MOJOLEARN_MAMBA3_SABOTAGE_ROTATE_HALF_SPLIT    the MIMO (i, i+N/2) pairing (828's falsifier)
    MOJOLEARN_MAMBA3_SABOTAGE_DIAG_INCLUDE_SUBTRACT  fwd_ref's diagonal at S18 (830's falsifier)
    MOJOLEARN_MAMBA3_SABOTAGE_STATE_TERM_SCALE_FIRST  S17 decay-into-q before the contraction
    MOJOLEARN_MAMBA3_SABOTAGE_RESUME_KERNEL_ASSOC  S22 spelled with fwd:371's association
    MOJOLEARN_MAMBA3_SABOTAGE_FOLD_SERIAL_ZERO_SEED  DEFINED, refused VACUOUS (DEVIATION 834)

The block file owns A_FLOOR_UNCLAMPED and STEP_UPSTREAM_RECURRENCE.

TRAP_LEFT_ONLY is spelled at S15 (the CONSUMPTION site: k_scaled uses
gamma instead of scale) rather than inside S9, ON PURPOSE: the contract's
8f table requires the arm's FIRST moved stage to be `kscale.out`, and
`trap.scale` is itself a recorded stage that an S9-site arm would move
first. The armed build's trap.scale stays the profile's; only what the
fold consumes changes -- one clause, one site.

Padding (contract section 3): every intra-chunk structure is ALWAYS
length Q; a padded row's q/k/v/dt/sigma/ADT contribute exact-zero
products; padded dacs positions COPY the last real value; the LAST REAL
TOKEN's shifted operands are STRUCTURAL +0.0 (its beta' leg belongs to
the next call -- the trapezoid's seam).

EVERYTHING HERE IS RUN OWED. No kernel below has compiled or run; the
commands live in `mamba/checks/mamba3_check.mojo`'s header.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_mul_add,
    identical_sigmoid,
    identical_silu,
    identical_tanh,
    portable_cosf,
    portable_sinf,
)
from mamba.checks.mamba3_fixture import (
    M3_CHUNK_SIZE,
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
    M3_PI,
    M3_TWO_PI,
)


def pinned_mul(a: Float32, b: Float32) -> Float32:
    """DEVIATION 720's construction (the oracle's `pinned_mul`), spelled
    here so this file shares only `checks/numerics.mojo` with the host
    side."""
    return identical_mul_add(a, b, Float32(-0.0))


def m3_mod_2pi(x: Float32) -> Float32:
    """DEVIATION 829's composed mod: `ftz(x - pinned_mul(2pi,
    floor(identical_div(x, 2pi))))`, floor exact, 2pi the pinned bits
    (0x40C90FDB). This file's own transcription of the composition, the
    pinned_mul rule."""
    from std.math import floor

    return ftz(
        x - pinned_mul(M3_TWO_PI, floor(identical_div(x, M3_TWO_PI)))
    )


# ===========================================================================
# Sabotage arms (see header).
# ===========================================================================

comptime SAB3_SEGSUM_DESCENDING = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_SEGSUM_DESCENDING"
]()
comptime SAB3_CHUNK_SIZE_32 = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_CHUNK_SIZE_32"
]()
comptime SAB3_TRAP_LEFT_ONLY = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_TRAP_LEFT_ONLY"
]()
comptime SAB3_ANGLE_MOD_PER_CHUNK = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_ANGLE_MOD_PER_CHUNK"
]()
comptime SAB3_ANGLE_MOD_AT_END = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_ANGLE_MOD_AT_END"
]()
comptime SAB3_ROTATE_HALF_SPLIT = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_ROTATE_HALF_SPLIT"
]()
comptime SAB3_DIAG_INCLUDE_SUBTRACT = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_DIAG_INCLUDE_SUBTRACT"
]()
comptime SAB3_STATE_TERM_SCALE_FIRST = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_STATE_TERM_SCALE_FIRST"
]()
comptime SAB3_RESUME_KERNEL_ASSOC = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_RESUME_KERNEL_ASSOC"
]()
comptime SAB3_FOLD_SERIAL_ZERO_SEED = is_defined[
    "MOJOLEARN_MAMBA3_SABOTAGE_FOLD_SERIAL_ZERO_SEED"
]()

comptime SISO3_ANY_SABOTAGE = (
    SAB3_SEGSUM_DESCENDING
    or SAB3_CHUNK_SIZE_32
    or SAB3_TRAP_LEFT_ONLY
    or SAB3_ANGLE_MOD_PER_CHUNK
    or SAB3_ANGLE_MOD_AT_END
    or SAB3_ROTATE_HALF_SPLIT
    or SAB3_DIAG_INCLUDE_SUBTRACT
    or SAB3_STATE_TERM_SCALE_FIRST
    or SAB3_RESUME_KERNEL_ASSOC
    or SAB3_FOLD_SERIAL_ZERO_SEED
)


def siso3_sabotage_name() -> String:
    comptime if SAB3_SEGSUM_DESCENDING:
        return String("SEGSUM_DESCENDING")
    comptime if SAB3_CHUNK_SIZE_32:
        return String("CHUNK_SIZE_32")
    comptime if SAB3_TRAP_LEFT_ONLY:
        return String("TRAP_LEFT_ONLY")
    comptime if SAB3_ANGLE_MOD_PER_CHUNK:
        return String("ANGLE_MOD_PER_CHUNK")
    comptime if SAB3_ANGLE_MOD_AT_END:
        return String("ANGLE_MOD_AT_END")
    comptime if SAB3_ROTATE_HALF_SPLIT:
        return String("ROTATE_HALF_SPLIT")
    comptime if SAB3_DIAG_INCLUDE_SUBTRACT:
        return String("DIAG_INCLUDE_SUBTRACT")
    comptime if SAB3_STATE_TERM_SCALE_FIRST:
        return String("STATE_TERM_SCALE_FIRST")
    comptime if SAB3_RESUME_KERNEL_ASSOC:
        return String("RESUME_KERNEL_ASSOC")
    comptime if SAB3_FOLD_SERIAL_ZERO_SEED:
        return String("FOLD_SERIAL_ZERO_SEED")
    return String("none")


def m3_q_eff() -> Int:
    """THE chunk size the build runs. DEVIATION 827 (783's standing at
    the new value): Q = 64 is a profile constant; this function is the
    ONE site the CHUNK_SIZE_32 arm may touch, and everything downstream
    derives from its answer."""
    comptime if SAB3_CHUNK_SIZE_32:
        return 32
    return M3_CHUNK_SIZE


def m3_n_chunks(t_work: Int) -> Int:
    var qv = m3_q_eff()
    var nc = (t_work + qv - 1) // qv
    if nc < 1:
        nc = 1
    return nc


# ===========================================================================
# Launch geometry: an EXECUTION-plan quantity. No kernel reads block
# geometry in any expression that reaches a fold boundary or a seed.
# ===========================================================================

comptime MAMBA3_TPB = 128


def _grid(n: Int) -> Int:
    var g = (n + MAMBA3_TPB - 1) // MAMBA3_TPB
    if g < 1:
        return 1
    return g


# ===========================================================================
# S7 (ADT = A * dt, PRODUCT) + S8 (sigma(trap)) for the NEW working rows,
# plus the dt copy into the working array. One thread per (b, li, h).
# ===========================================================================


def m3_pre_kernel(
    adt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    sig_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    dt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    a_new: MutPointer[Float32, MutAnyOrigin],  # [M, H]
    dt_new: MutPointer[Float32, MutAnyOrigin],  # [M, H]
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [M, dip]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
    c_trap_in: Int32,
):
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var dip = Int(dip_in)
    var c_trap = Int(c_trap_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * l * nh:
        return
    var bb = cell // (l * nh)
    var rem = cell - bb * l * nh
    var li = rem // nh
    var hh = rem - li * nh
    var mm = bb * l + li
    var widx = (bb * t_work + q0 + li) * nh + hh
    var dtv = dt_new.unsafe_load(mm * nh + hh)
    dt_work.unsafe_store(widx, dtv)  # copy, not a seam
    adt_work.unsafe_store(
        widx,
        ftz(pinned_mul(ftz(a_new.unsafe_load(mm * nh + hh)), ftz(dtv))),
    )
    sig_work.unsafe_store(
        widx,
        ftz(
            identical_sigmoid(
                ftz(in_proj.unsafe_load(mm * dip + c_trap + hh))
            )
        ),
    )


# ===========================================================================
# S9 -- gamma, beta', scale over ALL working rows. The shifted operands
# past the sequence end are STRUCTURAL +0.0 (the trapezoid's seam). One
# thread per (b, t, h).
# ===========================================================================


def m3_scale_kernel(
    gamma_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    betap_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    scale_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    dt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    sig_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
):
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * t_work * nh:
        return
    var bb = cell // (t_work * nh)
    var rem = cell - bb * t_work * nh
    var t = rem // nh
    var hh = rem - t * nh
    var g = ftz(
        pinned_mul(
            ftz(dt_work.unsafe_load(cell)), ftz(sig_work.unsafe_load(cell))
        )
    )
    var bp = Float32(0.0)
    if t + 1 < t_work:
        var nxt = (bb * t_work + t + 1) * nh + hh
        bp = ftz(
            pinned_mul(
                ftz(dt_work.unsafe_load(nxt)),
                ftz(Float32(1.0) - ftz(sig_work.unsafe_load(nxt))),
            )
        )
    gamma_work.unsafe_store(cell, g)
    betap_work.unsafe_store(cell, bp)
    scale_work.unsafe_store(cell, ftz(g + bp))


# ===========================================================================
# S10 -- the angle recurrence: SERIAL per token, mod 2pi EVERY step
# (DEVIATION 829), theta_-1 = the carried angle state. One thread per
# (b, h, angle-index) walking the NEW tokens. Owns ANGLE_MOD_PER_CHUNK
# (angle_dt.py:104-117's placement) and ANGLE_MOD_AT_END (fwd_ref
# :252-257's placement) -- both differ from the profile exactly when a
# 2pi-seam crossing occurs before the last token, the witnessed property.
# ===========================================================================


def m3_angle_kernel(
    theta_out: MutPointer[Float32, MutAnyOrigin],  # [M, H, R]
    theta_state: MutPointer[Float32, MutAnyOrigin],  # [B, H, R] in/out
    theta_last: MutPointer[Float32, MutAnyOrigin],  # [B, H, R] report
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [M, dip]
    dt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
    c_ang_in: Int32,
):
    comptime r_ang = M3_NUM_ROPE_ANGLES
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var dip = Int(dip_in)
    var c_ang = Int(c_ang_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * r_ang:
        return
    var bb = cell // (nh * r_ang)
    var rem = cell - bb * nh * r_ang
    var hh = rem // r_ang
    var r = rem - hh * r_ang
    var run = ftz(theta_state.unsafe_load(cell))
    var qv = m3_q_eff()
    for li in range(l):
        var mm = bb * l + li
        var a = ftz(
            pinned_mul(
                identical_tanh(
                    ftz(in_proj.unsafe_load(mm * dip + c_ang + r))
                ),
                M3_PI,
            )
        )
        var inc = ftz(
            pinned_mul(
                a, ftz(dt_work.unsafe_load((bb * t_work + q0 + li) * nh + hh))
            )
        )
        comptime if SAB3_ANGLE_MOD_PER_CHUNK:
            # SABOTAGE (829's falsifier 1): accumulate UNMODDED within a
            # working chunk, mod the OUTPUTS per element and the running
            # state only at chunk boundaries (angle_dt.py:104-117).
            run = ftz(run + inc)
            theta_out.unsafe_store((mm * nh + hh) * r_ang + r, m3_mod_2pi(run))
            if ((q0 + li + 1) - ((q0 + li + 1) // qv) * qv) == 0:
                run = m3_mod_2pi(run)
        else:
            comptime if SAB3_ANGLE_MOD_AT_END:
                # SABOTAGE (829's falsifier 2): the whole-sequence cumsum
                # unmodded, mod once per emitted element (fwd_ref
                # :252-257); the carried state is the modded last value.
                run = ftz(run + inc)
                theta_out.unsafe_store(
                    (mm * nh + hh) * r_ang + r, m3_mod_2pi(run)
                )
            else:
                run = m3_mod_2pi(ftz(run + inc))
                theta_out.unsafe_store((mm * nh + hh) * r_ang + r, run)
    comptime if SAB3_ANGLE_MOD_PER_CHUNK:
        run = m3_mod_2pi(run)
    comptime if SAB3_ANGLE_MOD_AT_END:
        run = m3_mod_2pi(run)
    theta_state.unsafe_store(cell, run)
    theta_last.unsafe_store(cell, run)


# ===========================================================================
# S12 (bias AFTER the norm) + S11/S13 (portable trig pair, interleaved
# (2i, 2i+1) pairs, UNFUSED two-product rotation; pairs >= R structurally
# unrotated -- DEVIATION 828). One thread per (b, li, h, pair). Owns
# ROTATE_HALF_SPLIT (the MIMO (i, i+N/2) pairing, mamba3.py:360-363's own
# note names it as the OTHER permutation).
# ===========================================================================


def m3_rot_kernel(
    rotq_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    rotk_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    theta_out: MutPointer[Float32, MutAnyOrigin],  # [M, H, R]
    bcb: MutPointer[Float32, MutAnyOrigin],  # [M, N] normed B
    bcc: MutPointer[Float32, MutAnyOrigin],  # [M, N] normed C
    b_bias: MutPointer[Float32, MutAnyOrigin],  # [H, N]
    c_bias: MutPointer[Float32, MutAnyOrigin],  # [H, N]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
):
    comptime n_state = M3_D_STATE
    comptime r_ang = M3_NUM_ROPE_ANGLES
    comptime n_pairs = M3_D_STATE // 2
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * l * nh * n_pairs:
        return
    var bb = cell // (l * nh * n_pairs)
    var rem = cell - bb * l * nh * n_pairs
    var li = rem // (nh * n_pairs)
    var rem2 = rem - li * nh * n_pairs
    var hh = rem2 // n_pairs
    var j = rem2 - hh * n_pairs
    var mm = bb * l + li
    var e0 = 2 * j
    var e1 = 2 * j + 1
    var rotated = j < r_ang
    var th_idx = j
    comptime if SAB3_ROTATE_HALF_SPLIT:
        # SABOTAGE (828's falsifier): the MIMO half-split permutation --
        # element j paired with element j + N/2, angle j, for j < R;
        # elements [R, N/2) and [N/2 + R, N) unrotated.
        e0 = j
        e1 = j + n_state // 2
        rotated = j < r_ang
        th_idx = j
    var q0v = ftz(
        ftz(bcc.unsafe_load(mm * n_state + e0))
        + ftz(c_bias.unsafe_load(hh * n_state + e0))
    )
    var q1v = ftz(
        ftz(bcc.unsafe_load(mm * n_state + e1))
        + ftz(c_bias.unsafe_load(hh * n_state + e1))
    )
    var k0v = ftz(
        ftz(bcb.unsafe_load(mm * n_state + e0))
        + ftz(b_bias.unsafe_load(hh * n_state + e0))
    )
    var k1v = ftz(
        ftz(bcb.unsafe_load(mm * n_state + e1))
        + ftz(b_bias.unsafe_load(hh * n_state + e1))
    )
    var base = ((bb * t_work + q0 + li) * nh + hh) * n_state
    if rotated:
        var th = ftz(
            theta_out.unsafe_load((mm * nh + hh) * r_ang + th_idx)
        )
        var cv = ftz(portable_cosf(th))
        var sv = ftz(portable_sinf(th))
        rotq_work.unsafe_store(
            base + e0,
            ftz(ftz(pinned_mul(q0v, cv)) - ftz(pinned_mul(q1v, sv))),
        )
        rotq_work.unsafe_store(
            base + e1,
            ftz(ftz(pinned_mul(q0v, sv)) + ftz(pinned_mul(q1v, cv))),
        )
        rotk_work.unsafe_store(
            base + e0,
            ftz(ftz(pinned_mul(k0v, cv)) - ftz(pinned_mul(k1v, sv))),
        )
        rotk_work.unsafe_store(
            base + e1,
            ftz(ftz(pinned_mul(k0v, sv)) + ftz(pinned_mul(k1v, cv))),
        )
    else:
        # STRUCTURAL identity: never-computed trig (DEVIATION 828 --
        # bit-equal to the references' cos-1/sin-0 pad spellings).
        rotq_work.unsafe_store(base + e0, q0v)
        rotq_work.unsafe_store(base + e1, q1v)
        rotk_work.unsafe_store(base + e0, k0v)
        rotk_work.unsafe_store(base + e1, k1v)


# ===========================================================================
# S14 -- the pre-rotation QK dot: gemm v1 cell over n (k = 128, ONE serial
# ascending leaf), then times gamma (DEVIATION 830). One thread per
# (b, li, h); the biased operands are recomputed with S12's exact
# spelling (a pure function of the same bits).
# ===========================================================================


def m3_qkdot_kernel(
    qkdot: MutPointer[Float32, MutAnyOrigin],  # [M, H]
    bcb: MutPointer[Float32, MutAnyOrigin],  # [M, N]
    bcc: MutPointer[Float32, MutAnyOrigin],  # [M, N]
    b_bias: MutPointer[Float32, MutAnyOrigin],  # [H, N]
    c_bias: MutPointer[Float32, MutAnyOrigin],  # [H, N]
    gamma_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
):
    comptime n_state = M3_D_STATE
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * l * nh:
        return
    var bb = cell // (l * nh)
    var rem = cell - bb * l * nh
    var li = rem // nh
    var hh = rem - li * nh
    var mm = bb * l + li
    var acc = Float32(0.0)
    for n in range(n_state):
        var qv = ftz(
            ftz(bcc.unsafe_load(mm * n_state + n))
            + ftz(c_bias.unsafe_load(hh * n_state + n))
        )
        var kv = ftz(
            ftz(bcb.unsafe_load(mm * n_state + n))
            + ftz(b_bias.unsafe_load(hh * n_state + n))
        )
        acc = ftz(identical_mul_add(qv, kv, acc))
    qkdot.unsafe_store(
        mm * nh + hh,
        ftz(
            pinned_mul(
                ftz(acc),
                gamma_work.unsafe_load((bb * t_work + q0 + li) * nh + hh),
            )
        ),
    )


# ===========================================================================
# S15 -- K scaling over ALL working rows (the carried k-state and the
# k_last report are PRE-scale). One thread per element. Owns
# TRAP_LEFT_ONLY (consumes gamma instead of scale -- Euler, no trapezoid).
# ===========================================================================


def m3_kscale_kernel(
    kscale_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    rotk_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    scale_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    gamma_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    n_in: Int32,  # B * T * H * N
):
    comptime n_state = M3_D_STATE
    var n = Int(n_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= n:
        return
    var row = cell // n_state  # (b, t, h)
    var sc: Float32
    comptime if SAB3_TRAP_LEFT_ONLY:
        sc = gamma_work.unsafe_load(row)
    else:
        sc = scale_work.unsafe_load(row)
    kscale_work.unsafe_store(
        cell, ftz(pinned_mul(ftz(rotk_work.unsafe_load(cell)), sc))
    )


# ===========================================================================
# mamba2 S11 inherited -- the per-chunk cumsum of ADT, SERIAL ASCENDING;
# the chunk boundary is a hard reset; padded positions COPY the last real
# value. One thread per (b, h, c).
# ===========================================================================


def m3_dacs_kernel(
    dacs: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    adt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * nc:
        return
    var bb = cell // (nh * nc)
    var rem = cell - bb * nh * nc
    var hh = rem // nc
    var c = rem - hh * nc
    var c0 = c * qv
    var real = t_work - c0
    if real > qv:
        real = qv
    var run = Float32(0.0)
    for i in range(qv):
        if i < real:
            var v = ftz(
                adt_work.unsafe_load((bb * t_work + c0 + i) * nh + hh)
            )
            comptime if SAB3_SEGSUM_DESCENDING:
                # SABOTAGE: rebuild THIS prefix by a descending fold.
                var acc = Float32(0.0)
                var k = i
                while k >= 0:
                    acc = ftz(
                        acc
                        + ftz(
                            adt_work.unsafe_load(
                                (bb * t_work + c0 + k) * nh + hh
                            )
                        )
                    )
                    k -= 1
                run = acc
            else:
                if i == 0:
                    run = v
                else:
                    run = ftz(run + v)
        # else: COPY of the last real value, not an add.
        dacs.unsafe_store(((bb * nh + hh) * nc + c) * qv + i, run)


# ===========================================================================
# S16's decay -- L = exp(segsum), STRICT triangle: +0.0 ON and above the
# diagonal, STRUCTURAL (DEVIATIONS 782 inherited + 830's strict mask).
# Column-serial rebuild, one thread per (b, c, h, j).
# ===========================================================================


def m3_seg_l_kernel(
    seg_l: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, Q, Q]
    adt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nc * nh * qv:
        return
    var bb = cell // (nc * nh * qv)
    var rem = cell - bb * nc * nh * qv
    var c = rem // (nh * qv)
    var rem2 = rem - c * nh * qv
    var hh = rem2 // qv
    var j = rem2 - hh * qv
    var c0 = c * qv
    var real = t_work - c0
    if real > qv:
        real = qv
    var lbase = (((bb * nc + c) * nh + hh) * qv) * qv
    # On and above the diagonal: +0.0, STRUCTURAL -- written, never
    # computed (the diagonal is DEVIATION 830's, moved to S14/S18).
    for i in range(j + 1):
        seg_l.unsafe_store(lbase + i * qv + j, Float32(0.0))
    var acc = Float32(0.0)
    for i in range(j + 1, qv):
        if i < real:
            comptime if SAB3_SEGSUM_DESCENDING:
                var a2 = Float32(0.0)
                var k = i
                while k > j:
                    a2 = ftz(
                        a2
                        + ftz(
                            adt_work.unsafe_load(
                                (bb * t_work + c0 + k) * nh + hh
                            )
                        )
                    )
                    k -= 1
                acc = a2
            else:
                acc = ftz(
                    acc
                    + ftz(
                        adt_work.unsafe_load((bb * t_work + c0 + i) * nh + hh)
                    )
                )
        seg_l.unsafe_store(lbase + i * qv + j, ftz(identical_exp(acc)))


# ===========================================================================
# S22 -- the pending Input_States correction: the NORMATIVE reference's
# scalar-first association (fwd_ref :266-267), `c = pinned_mul(dt_1,
# ftz(1 - sigma_1))`, `t = pinned_mul(pinned_mul(v_st, k_st), c)`,
# `h0 = ftz(h_in + t)`. dt_1/sigma_1 are the call's FIRST token's (the
# state guard makes a pending call fresh, so q0 = 0). One thread per
# (b, h, p, n). Owns RESUME_KERNEL_ASSOC (the kernel's fwd:371
# association, ((v*k)*dt)*(1-sigma) -- refused by seam S22).
# ===========================================================================


def m3_resume_kernel(
    h_state: MutPointer[Float32, MutAnyOrigin],  # [B, H, P, N] in/out
    pend_k: MutPointer[Float32, MutAnyOrigin],  # [B, H, N]
    pend_v: MutPointer[Float32, MutAnyOrigin],  # [B, H, P]
    dt_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    sig_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
):
    comptime n_state = M3_D_STATE
    comptime p_dim = M3_HEADDIM
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var pn = p_dim * n_state
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * pn:
        return
    var bb = cell // (nh * pn)
    var rem = cell - bb * nh * pn
    var hh = rem // pn
    var i = rem - hh * pn
    var p = i // n_state
    var n = i - p * n_state
    var dt1 = ftz(dt_work.unsafe_load((bb * t_work + 0) * nh + hh))
    var om = ftz(
        Float32(1.0) - ftz(sig_work.unsafe_load((bb * t_work + 0) * nh + hh))
    )
    var vk = ftz(
        pinned_mul(
            ftz(pend_v.unsafe_load((bb * nh + hh) * p_dim + p)),
            ftz(pend_k.unsafe_load((bb * nh + hh) * n_state + n)),
        )
    )
    var tv: Float32
    comptime if SAB3_RESUME_KERNEL_ASSOC:
        # SABOTAGE: the kernel's association -- ((v*k)*dt)*(1-sigma),
        # fwd:371 -- where the profile folds the scalar FIRST.
        tv = ftz(pinned_mul(ftz(pinned_mul(vk, dt1)), om))
    else:
        var csc = ftz(pinned_mul(dt1, om))
        tv = ftz(pinned_mul(vk, csc))
    h_state.unsafe_store(cell, ftz(ftz(h_state.unsafe_load(cell)) + tv))


# ===========================================================================
# S20 -- the SERIAL inter-chunk pass: one thread per (b, h, p, n) walking
# chunks ascending. Writes pass_states (the state ENTERING each working
# chunk), h_last (after the FINAL padded chunk -- the report), and the
# carried h = the state entering the LAST working chunk (DEVIATION
# 832(i)'s sealed boundary) back into h_state. The increment is a gemm v1
# cell at k = Q = 64 (ONE leaf) over `(v ⊙ exp(da_cs_rev))^T . k_scaled`;
# the update is mamba2 S17's ONE-rounding fused fma.
# ===========================================================================


def m3_statepass_kernel(
    pass_states: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, P, N]
    h_last: MutPointer[Float32, MutAnyOrigin],  # [B, H, P, N]
    h_state: MutPointer[Float32, MutAnyOrigin],  # [B, H, P, N] in/out
    kscale_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    v_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    dacs: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    comptime n_state = M3_D_STATE
    comptime p_dim = M3_HEADDIM
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var pn = p_dim * n_state
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * pn:
        return
    var bb = cell // (nh * pn)
    var rem = cell - bb * nh * pn
    var hh = rem // pn
    var i = rem - hh * pn
    var p = i // n_state
    var n = i - p * n_state

    var h = ftz(h_state.unsafe_load(cell))
    var sealed = h
    for c in range(nc):
        pass_states.unsafe_store((((bb * nc + c) * nh + hh) * pn) + i, h)
        if c == nc - 1:
            sealed = h
        var c0 = c * qv
        var real = t_work - c0
        if real > qv:
            real = qv
        var dl = ftz(
            dacs.unsafe_load(((bb * nh + hh) * nc + c) * qv + (qv - 1))
        )
        var acc = Float32(0.0)
        for j in range(qv):
            var vsv = Float32(0.0)
            var ksv = Float32(0.0)
            if j < real:
                var drev = ftz(
                    dl
                    - ftz(
                        dacs.unsafe_load(((bb * nh + hh) * nc + c) * qv + j)
                    )
                )
                var e = ftz(identical_exp(drev))
                vsv = ftz(
                    pinned_mul(
                        ftz(
                            v_work.unsafe_load(
                                ((bb * t_work + c0 + j) * nh + hh) * p_dim
                                + p
                            )
                        ),
                        e,
                    )
                )
                ksv = ftz(
                    kscale_work.unsafe_load(
                        ((bb * t_work + c0 + j) * nh + hh) * n_state + n
                    )
                )
            # padded rows: exact zeros enter the fold (contract s3).
            acc = ftz(identical_mul_add(vsv, ksv, acc))
        var scale_c = ftz(identical_exp(dl))
        h = ftz(identical_mul_add(scale_c, ftz(h), ftz(acc)))
    h_last.unsafe_store(cell, h)
    h_state.unsafe_store(cell, sealed)


# ===========================================================================
# S16's s matrix -- s[i][j] = q_rot_i . k_scaled_j, a gemm v1 cell over n
# (k = 128, ONE leaf), computed for the STRICT triangle only; +0.0
# written elsewhere (an unrecorded working scratch; the recorded stage is
# seg.L). One thread per (b, c, h, i, j).
# ===========================================================================


def m3_qk_s_kernel(
    qk_s: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, Q, Q]
    rotq_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    kscale_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    comptime n_state = M3_D_STATE
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nc * nh * qv * qv:
        return
    var bb = cell // (nc * nh * qv * qv)
    var rem = cell - bb * nc * nh * qv * qv
    var c = rem // (nh * qv * qv)
    var rem2 = rem - c * nh * qv * qv
    var hh = rem2 // (qv * qv)
    var rem3 = rem2 - hh * qv * qv
    var i = rem3 // qv
    var j = rem3 - i * qv
    var c0 = c * qv
    var real = t_work - c0
    if real > qv:
        real = qv
    var out = Float32(0.0)
    if j < i and i < real:
        var ti = bb * t_work + c0 + i
        var tj = bb * t_work + c0 + j
        var acc = Float32(0.0)
        for n in range(n_state):
            acc = ftz(
                identical_mul_add(
                    ftz(rotq_work.unsafe_load((ti * nh + hh) * n_state + n)),
                    ftz(
                        kscale_work.unsafe_load((tj * nh + hh) * n_state + n)
                    ),
                    acc,
                )
            )
        out = ftz(acc)
    qk_s.unsafe_store(cell, out)


# ===========================================================================
# S16 -- Y_intra for the NEW rows: M = s ⊙ L (PRODUCT, structural +0.0 on
# and above the diagonal), folded against v over the chunk positions --
# a gemm v1 cell at k = Q = 64 (ONE leaf). One thread per (b, li, h, p).
# ===========================================================================


def m3_yintra_kernel(
    yintra: MutPointer[Float32, MutAnyOrigin],  # [M, H, P]
    qk_s: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, Q, Q]
    seg_l: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, Q, Q]
    v_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    comptime p_dim = M3_HEADDIM
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * l * nh * p_dim:
        return
    var bb = cell // (l * nh * p_dim)
    var rem = cell - bb * l * nh * p_dim
    var li = rem // (nh * p_dim)
    var rem2 = rem - li * nh * p_dim
    var hh = rem2 // p_dim
    var p = rem2 - hh * p_dim
    var t = q0 + li
    var c = t // qv
    var ii = t - c * qv
    var sbase = (((bb * nc + c) * nh + hh) * qv + ii) * qv
    var acc = Float32(0.0)
    for jj in range(qv):
        var m_ij = Float32(0.0)
        if jj < ii:
            m_ij = ftz(
                pinned_mul(
                    ftz(qk_s.unsafe_load(sbase + jj)),
                    ftz(seg_l.unsafe_load(sbase + jj)),
                )
            )
        var xv = Float32(0.0)
        if c * qv + jj < t_work:
            xv = ftz(
                v_work.unsafe_load(
                    ((bb * t_work + c * qv + jj) * nh + hh) * p_dim + p
                )
            )
        acc = ftz(identical_mul_add(m_ij, xv, acc))
    yintra.unsafe_store(cell, ftz(acc))


# ===========================================================================
# S17 -- Y_state for the NEW rows: (q_rot . h_entering) over n (gemm v1
# cell, k = 128, ONE leaf), then * exp(da_cs_i), da_cs INCLUSIVE of token
# i (mamba2 S11/S18 inherited). One thread per (b, li, h, p). Owns
# STATE_TERM_SCALE_FIRST (decay folded into q BEFORE the contraction).
# ===========================================================================


def m3_ystate_kernel(
    ystate: MutPointer[Float32, MutAnyOrigin],  # [M, H, P]
    rotq_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    pass_states: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, P, N]
    dacs: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    comptime n_state = M3_D_STATE
    comptime p_dim = M3_HEADDIM
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * l * nh * p_dim:
        return
    var bb = cell // (l * nh * p_dim)
    var rem = cell - bb * l * nh * p_dim
    var li = rem // (nh * p_dim)
    var rem2 = rem - li * nh * p_dim
    var hh = rem2 // p_dim
    var p = rem2 - hh * p_dim
    var t = q0 + li
    var c = t // qv
    var ii = t - c * qv
    var e_i = ftz(
        identical_exp(
            ftz(dacs.unsafe_load(((bb * nh + hh) * nc + c) * qv + ii))
        )
    )
    var pbase = ((((bb * nc + c) * nh + hh) * p_dim) + p) * n_state
    var acc = Float32(0.0)
    for n in range(n_state):
        var qv2 = ftz(
            rotq_work.unsafe_load(((bb * t_work + t) * nh + hh) * n_state + n)
        )
        var hv = ftz(pass_states.unsafe_load(pbase + n))
        comptime if SAB3_STATE_TERM_SCALE_FIRST:
            # SABOTAGE: decay into q BEFORE the contraction (contract 8f).
            acc = ftz(
                identical_mul_add(ftz(pinned_mul(qv2, e_i)), hv, acc)
            )
        else:
            acc = ftz(identical_mul_add(qv2, hv, acc))
    comptime if SAB3_STATE_TERM_SCALE_FIRST:
        ystate.unsafe_store(cell, ftz(acc))
    else:
        ystate.unsafe_store(cell, ftz(pinned_mul(ftz(acc), e_i)))


# ===========================================================================
# S18 (diagonal + D skip, ONE add on top of DEVIATION 833's Y-combine) +
# S19 (Z gate, the ONE-division silu). One thread per (b, li, h, p). Owns
# DIAG_INCLUDE_SUBTRACT (fwd_ref's diagonal: full-scale rotated dot minus
# beta' times the pre-rotation dot, :282/:297/:309 -- equal in exact
# arithmetic, different bits; DEVIATION 830's required-RED arm, spelled at
# THIS site so it moves skip.out first and nothing earlier).
# ===========================================================================


def m3_skip_gate_kernel(
    skip_out: MutPointer[Float32, MutAnyOrigin],  # [M, H, P]
    gate_out: MutPointer[Float32, MutAnyOrigin],  # [M, H, P]
    yintra: MutPointer[Float32, MutAnyOrigin],  # [M, H, P]
    ystate: MutPointer[Float32, MutAnyOrigin],  # [M, H, P]
    qkdot: MutPointer[Float32, MutAnyOrigin],  # [M, H]
    v_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    in_proj: MutPointer[Float32, MutAnyOrigin],  # [M, dip]
    d_skip: MutPointer[Float32, MutAnyOrigin],  # [H]
    rotq_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N] (DIAG arm)
    rotk_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N] (DIAG arm)
    bcb: MutPointer[Float32, MutAnyOrigin],  # [M, N] (DIAG arm)
    bcc: MutPointer[Float32, MutAnyOrigin],  # [M, N] (DIAG arm)
    b_bias: MutPointer[Float32, MutAnyOrigin],  # [H, N] (DIAG arm)
    c_bias: MutPointer[Float32, MutAnyOrigin],  # [H, N] (DIAG arm)
    scale_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H] (DIAG arm)
    betap_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H] (DIAG arm)
    b_in: Int32,
    l_in: Int32,
    q0_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
    c_z_in: Int32,
):
    comptime n_state = M3_D_STATE
    comptime p_dim = M3_HEADDIM
    var b = Int(b_in)
    var l = Int(l_in)
    var q0 = Int(q0_in)
    var nh = Int(nh_in)
    var dip = Int(dip_in)
    var c_z = Int(c_z_in)
    var t_work = q0 + l
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * l * nh * p_dim:
        return
    var bb = cell // (l * nh * p_dim)
    var rem = cell - bb * l * nh * p_dim
    var li = rem // (nh * p_dim)
    var rem2 = rem - li * nh * p_dim
    var hh = rem2 // p_dim
    var p = rem2 - hh * p_dim
    var mm = bb * l + li
    var t = q0 + li
    # DEVIATION 833: y = ftz(ystate + yintra), ystate the left operand.
    var y0 = ftz(
        ftz(ystate.unsafe_load(cell)) + ftz(yintra.unsafe_load(cell))
    )
    var tval: Float32
    comptime if SAB3_DIAG_INCLUDE_SUBTRACT:
        # SABOTAGE (830's falsifier): the fwd_ref diagonal -- full scale
        # on the ROTATED dot, minus beta' times the PRE-rotation dot.
        # Witnessed by any nonzero-angle fixture: rotation ROUNDING
        # separates q_rot.k_rot from q.k even though exact arithmetic
        # does not.
        var dr = Float32(0.0)
        var dp = Float32(0.0)
        for n in range(n_state):
            dr = ftz(
                identical_mul_add(
                    ftz(
                        rotq_work.unsafe_load(
                            ((bb * t_work + t) * nh + hh) * n_state + n
                        )
                    ),
                    ftz(
                        rotk_work.unsafe_load(
                            ((bb * t_work + t) * nh + hh) * n_state + n
                        )
                    ),
                    dr,
                )
            )
            var qv = ftz(
                ftz(bcc.unsafe_load(mm * n_state + n))
                + ftz(c_bias.unsafe_load(hh * n_state + n))
            )
            var kv = ftz(
                ftz(bcb.unsafe_load(mm * n_state + n))
                + ftz(b_bias.unsafe_load(hh * n_state + n))
            )
            dp = ftz(identical_mul_add(qv, kv, dp))
        var widx = (bb * t_work + t) * nh + hh
        var d_ref = ftz(
            ftz(pinned_mul(ftz(dr), scale_work.unsafe_load(widx)))
            - ftz(pinned_mul(ftz(dp), betap_work.unsafe_load(widx)))
        )
        tval = ftz(ftz(d_skip.unsafe_load(hh)) + d_ref)
    else:
        tval = ftz(
            ftz(d_skip.unsafe_load(hh))
            + ftz(qkdot.unsafe_load(mm * nh + hh))
        )
    var pv = ftz(
        pinned_mul(
            tval,
            ftz(
                v_work.unsafe_load(((bb * t_work + t) * nh + hh) * p_dim + p)
            ),
        )
    )
    var sk = ftz(y0 + pv)
    skip_out.unsafe_store(cell, sk)
    var zv = ftz(in_proj.unsafe_load(mm * dip + c_z + hh * p_dim + p))
    gate_out.unsafe_store(
        cell, ftz(pinned_mul(ftz(sk), ftz(identical_silu(zv))))
    )


# ===========================================================================
# The k_last / v_last reports: post-bias post-rotation PRE-scale k and
# raw v of the LAST REAL working row (fwd wrapper :709-729). Copies.
# ===========================================================================


def m3_reports_kernel(
    k_last: MutPointer[Float32, MutAnyOrigin],  # [B, H, N]
    v_last: MutPointer[Float32, MutAnyOrigin],  # [B, H, P]
    rotk_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, N]
    v_work: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
):
    comptime n_state = M3_D_STATE
    comptime p_dim = M3_HEADDIM
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * (n_state + p_dim):
        return
    var bb = cell // (nh * (n_state + p_dim))
    var rem = cell - bb * nh * (n_state + p_dim)
    var hh = rem // (n_state + p_dim)
    var d = rem - hh * (n_state + p_dim)
    var last = bb * t_work + (t_work - 1)
    if d < n_state:
        k_last.unsafe_store(
            (bb * nh + hh) * n_state + d,
            rotk_work.unsafe_load((last * nh + hh) * n_state + d),
        )
    else:
        var p = d - n_state
        v_last.unsafe_store(
            (bb * nh + hh) * p_dim + p,
            v_work.unsafe_load((last * nh + hh) * p_dim + p),
        )


# ===========================================================================
# THE LAUNCHER: seams S7-S20, S22 + the four reports over the working
# sequence. All buffers are the CALLER's fields (the block's
# stages/state structs); this function only launches and synchronizes.
# Stage recording is the block's (one place, one prefix, contract section
# 7's order). `[[mojo-buffer-freed-at-last-use]]`: no scratch is
# allocated here.
# ===========================================================================


def m3_siso_forward(
    ctx: DeviceContext,
    mut adt_work: DeviceBuffer[DType.float32],
    mut sig_work: DeviceBuffer[DType.float32],
    mut dt_work: DeviceBuffer[DType.float32],
    mut gamma_work: DeviceBuffer[DType.float32],
    mut betap_work: DeviceBuffer[DType.float32],
    mut scale_work: DeviceBuffer[DType.float32],
    mut theta_out: DeviceBuffer[DType.float32],
    mut theta_last: DeviceBuffer[DType.float32],
    mut rotq_work: DeviceBuffer[DType.float32],
    mut rotk_work: DeviceBuffer[DType.float32],
    mut qkdot: DeviceBuffer[DType.float32],
    mut kscale_work: DeviceBuffer[DType.float32],
    mut v_work: DeviceBuffer[DType.float32],
    mut dacs: DeviceBuffer[DType.float32],
    mut seg_l: DeviceBuffer[DType.float32],
    mut qk_s: DeviceBuffer[DType.float32],
    mut pass_states: DeviceBuffer[DType.float32],
    mut yintra: DeviceBuffer[DType.float32],
    mut ystate: DeviceBuffer[DType.float32],
    mut skip_out: DeviceBuffer[DType.float32],
    mut gate_out: DeviceBuffer[DType.float32],
    mut h_last: DeviceBuffer[DType.float32],
    mut k_last: DeviceBuffer[DType.float32],
    mut v_last: DeviceBuffer[DType.float32],
    mut a_new: DeviceBuffer[DType.float32],
    mut dt_new: DeviceBuffer[DType.float32],
    mut in_proj: DeviceBuffer[DType.float32],
    mut bcb: DeviceBuffer[DType.float32],
    mut bcc: DeviceBuffer[DType.float32],
    mut b_bias: DeviceBuffer[DType.float32],
    mut c_bias: DeviceBuffer[DType.float32],
    mut d_skip: DeviceBuffer[DType.float32],
    mut theta_state: DeviceBuffer[DType.float32],
    mut h_state: DeviceBuffer[DType.float32],
    mut pend_k: DeviceBuffer[DType.float32],
    mut pend_v: DeviceBuffer[DType.float32],
    apply_resume: Bool,
    b: Int,
    l: Int,
    q0: Int,
    nh: Int,
    dip: Int,
    c_z: Int,
    c_trap: Int,
    c_ang: Int,
) raises:
    """One SISO core pass over the working sequence (q0 buffered rows,
    whose q/k/v/dt/sigma/ADT slots of the working arrays the block
    pre-filled from its state buffer, ++ l new rows). `h_state` enters as
    the carried boundary (S22-corrected here when `apply_resume`) and
    leaves as the state ENTERING THE LAST WORKING CHUNK (DEVIATION
    832(i)); `theta_state` advances through the new tokens; `h_last` /
    `k_last` / `v_last` / `theta_last` are the reports. SYNCHRONIZES
    before returning."""
    var t_work = q0 + l
    var qv = m3_q_eff()
    var nc = m3_n_chunks(t_work)
    comptime n_state = M3_D_STATE
    comptime p_dim = M3_HEADDIM
    comptime r_ang = M3_NUM_ROPE_ANGLES

    ctx.enqueue_function[m3_pre_kernel](
        adt_work.unsafe_ptr(),
        sig_work.unsafe_ptr(),
        dt_work.unsafe_ptr(),
        a_new.unsafe_ptr(),
        dt_new.unsafe_ptr(),
        in_proj.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(q0),
        Int32(nh),
        Int32(dip),
        Int32(c_trap),
        grid_dim=(_grid(b * l * nh), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.enqueue_function[m3_scale_kernel](
        gamma_work.unsafe_ptr(),
        betap_work.unsafe_ptr(),
        scale_work.unsafe_ptr(),
        dt_work.unsafe_ptr(),
        sig_work.unsafe_ptr(),
        Int32(b),
        Int32(t_work),
        Int32(nh),
        grid_dim=(_grid(b * t_work * nh), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.enqueue_function[m3_angle_kernel](
        theta_out.unsafe_ptr(),
        theta_state.unsafe_ptr(),
        theta_last.unsafe_ptr(),
        in_proj.unsafe_ptr(),
        dt_work.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(q0),
        Int32(nh),
        Int32(dip),
        Int32(c_ang),
        grid_dim=(_grid(b * nh * r_ang), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.enqueue_function[m3_rot_kernel](
        rotq_work.unsafe_ptr(),
        rotk_work.unsafe_ptr(),
        theta_out.unsafe_ptr(),
        bcb.unsafe_ptr(),
        bcc.unsafe_ptr(),
        b_bias.unsafe_ptr(),
        c_bias.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(q0),
        Int32(nh),
        grid_dim=(_grid(b * l * nh * (n_state // 2)), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.enqueue_function[m3_qkdot_kernel](
        qkdot.unsafe_ptr(),
        bcb.unsafe_ptr(),
        bcc.unsafe_ptr(),
        b_bias.unsafe_ptr(),
        c_bias.unsafe_ptr(),
        gamma_work.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(q0),
        Int32(nh),
        grid_dim=(_grid(b * l * nh), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.enqueue_function[m3_kscale_kernel](
        kscale_work.unsafe_ptr(),
        rotk_work.unsafe_ptr(),
        scale_work.unsafe_ptr(),
        gamma_work.unsafe_ptr(),
        Int32(b * t_work * nh * n_state),
        grid_dim=(_grid(b * t_work * nh * n_state), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.enqueue_function[m3_dacs_kernel](
        dacs.unsafe_ptr(),
        adt_work.unsafe_ptr(),
        Int32(b),
        Int32(t_work),
        Int32(nh),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(b * nh * nc), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.enqueue_function[m3_seg_l_kernel](
        seg_l.unsafe_ptr(),
        adt_work.unsafe_ptr(),
        Int32(b),
        Int32(t_work),
        Int32(nh),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(b * nc * nh * qv), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    if apply_resume:
        ctx.enqueue_function[m3_resume_kernel](
            h_state.unsafe_ptr(),
            pend_k.unsafe_ptr(),
            pend_v.unsafe_ptr(),
            dt_work.unsafe_ptr(),
            sig_work.unsafe_ptr(),
            Int32(b),
            Int32(t_work),
            Int32(nh),
            grid_dim=(_grid(b * nh * p_dim * n_state), 1, 1),
            block_dim=(MAMBA3_TPB, 1, 1),
        )
    ctx.enqueue_function[m3_statepass_kernel](
        pass_states.unsafe_ptr(),
        h_last.unsafe_ptr(),
        h_state.unsafe_ptr(),
        kscale_work.unsafe_ptr(),
        v_work.unsafe_ptr(),
        dacs.unsafe_ptr(),
        Int32(b),
        Int32(t_work),
        Int32(nh),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(b * nh * p_dim * n_state), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.enqueue_function[m3_qk_s_kernel](
        qk_s.unsafe_ptr(),
        rotq_work.unsafe_ptr(),
        kscale_work.unsafe_ptr(),
        Int32(b),
        Int32(t_work),
        Int32(nh),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(b * nc * nh * qv * qv), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.enqueue_function[m3_yintra_kernel](
        yintra.unsafe_ptr(),
        qk_s.unsafe_ptr(),
        seg_l.unsafe_ptr(),
        v_work.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(q0),
        Int32(nh),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(b * l * nh * p_dim), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.enqueue_function[m3_ystate_kernel](
        ystate.unsafe_ptr(),
        rotq_work.unsafe_ptr(),
        pass_states.unsafe_ptr(),
        dacs.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(q0),
        Int32(nh),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(b * l * nh * p_dim), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.enqueue_function[m3_skip_gate_kernel](
        skip_out.unsafe_ptr(),
        gate_out.unsafe_ptr(),
        yintra.unsafe_ptr(),
        ystate.unsafe_ptr(),
        qkdot.unsafe_ptr(),
        v_work.unsafe_ptr(),
        in_proj.unsafe_ptr(),
        d_skip.unsafe_ptr(),
        rotq_work.unsafe_ptr(),
        rotk_work.unsafe_ptr(),
        bcb.unsafe_ptr(),
        bcc.unsafe_ptr(),
        b_bias.unsafe_ptr(),
        c_bias.unsafe_ptr(),
        scale_work.unsafe_ptr(),
        betap_work.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(q0),
        Int32(nh),
        Int32(dip),
        Int32(c_z),
        grid_dim=(_grid(b * l * nh * p_dim), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.enqueue_function[m3_reports_kernel](
        k_last.unsafe_ptr(),
        v_last.unsafe_ptr(),
        rotk_work.unsafe_ptr(),
        v_work.unsafe_ptr(),
        Int32(b),
        Int32(t_work),
        Int32(nh),
        grid_dim=(_grid(b * nh * (n_state + p_dim)), 1, 1),
        block_dim=(MAMBA3_TPB, 1, 1),
    )
    ctx.synchronize()
