# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fused attention path against the eager stage kernels, BIT FOR BIT,
at shapes the fixture gate does not reach, plus the two refusals it must
take.

    tools/with_identical_mode.sh pixi run check-transformer-fused

WHAT THE FIXTURE GATE ALREADY COVERS. `check-transformer` records
`attn.ctx` from the fused output on every unplanted fixture (17 cases,
L <= 64, one row tile, one key block) and `check-transformer-backward`
records stages 22-24 from the fused backward. Those cards are the profile's
gate and they stay byte-equal to the three-vendor references; this file
is the SHAPE coverage they cannot give: many row tiles and key blocks, a
window that starts inside a key block, a decode step whose packed span
starts past position 0, a ring gather (`key_lo > 0`) with several rows,
`head_dim` 16/24/128 beside 64, and `n_rep` 1 and 2.

WHAT IT ASSERTS. For every case: the eager kernels' `ctx`, `amax`, `denom`
(forward) and `zdot`, `dq`, `dk`, `dv` (backward) equal the fused kernels'
bit for bit, whenever the fused launcher reports `FUSED_RAN`; and the
launcher's STATUS matches the case's expectation, so that
  * the `underflow` case, built so every visible product flushes to `-0.0`
    and the masked tail would launder it to `+0.0`, REPORTS THE CORNER
    (`FUSED_CORNER`) and the wrapper's fallback then equals eager;
  * the `regime` cases, built past the `2^100` bound, are REFUSED
    (`FUSED_REFUSED_REGIME`) before any kernel runs.
A fused path that silently skipped the corner would pass every ordinary
case here and differ from eager on `underflow` -- which is the case that
exists to say so ([[reached-but-inert]]).

WHICH ARM RAN (DEVIATION 2534, brief
docs/lanes/BRIEF_attention_step_2026-09-11.md section 15). The shipped
attention arm is a kernel-matrix row per column (`attn_default_arm_for`:
NVIDIA `stash_tiled_fgrid_r32_qres_pf`, AMD
`stash_tiled_fgrid_r32_qres_pf_kvgrid_r32` (brief section 18), Apple and the
rest `stash_tiled`). The direct launches here go through
`fused_forward_launch_ran` / `fused_backward_launch_ran`, which report the
arm word of the kernels that actually launched, set inside the branch that
launched them. The check prints a `DEFAULT` line (column, arm, word), proves
the default word names itself and runs as named at head_dim 64 on this
build, and FAILS when a launch at head_dim 64 ran anything but the resolved
half of the arm (or when a launch at another head dim, or a refused launch,
ran anything but the shipped kernels). On a build without
`-D MOJOLEARN_ATTN_ARM_TRIAL=1` the arm is the column's default, so a green
run on a column is "the default ran there and is bit-identical to eager".
`-D MOJOLEARN_ATTN_DEFAULT_R3_EVERY_COLUMN=1` gives every column NVIDIA's
default, so the same no-trial check reaches the shipped round 3 branch on a
Mac.

WHICH DK/DV KERNEL RAN (brief section 18). A `DKDV` line names the dk/dv
kernel the default's backward launches at head_dim 64 and the one this run's
arm launches, with `shipped_kv_branch=` (whether this build compiles the
column default's DEVIATION 2597 dk/dv instantiation) and `default_kv_keys=`.
The check FAILS when the arm's resolved backward carries a DEVIATION 2596 /
2597 token and no direct backward launch reported it, or when a launch
reported one the arm does not resolve to. `-D
MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN=1` gives every column AMD's
default (`stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`), so the same no-trial
check reaches the shipped 2597 dk/dv branch on a Mac.

The forward wrapper (`eager_attention_forward`, trace off, path auto) is
what the surface calls, so its output is compared too: it must equal the
eager core whatever status it took.

THE KEPT EXP STASH (DEVIATIONS 2650 to 2652, brief section 20). The direct
launches go through `fused_forward_launch_estash_ran` and
`fused_backward_launch_estash_ran` with a kept buffer of this check's own,
so an `_estash` arm named by MOJOLEARN_ATTN_ARM on a trial build runs its
backward on the kept stash and reports the estash bits in `ran`; every
other arm, and every shipped build, takes the plain launchers through them
unchanged. A `WRAPPER` line prints the wrapper's kept cell count
(`stages.attn_estash_cells`): 0 on a shipped build and for every arm
without `_estash`.
"""

from std.memory import bitcast
from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from checks.kernel_matrix import TARGET_COLUMN, column_name
from checks.numerics import numeric_mode_name
from transformer.checks.transformer_fixture import bits32_hex, fixture_tensor
from transformer.checks.transformer_backward import (
    LlamaBackwardStages,
    bwd_attention_eager_stages,
)
from transformer.impl.llama.fused_attention import (
    ATTN_ARM_BASELINE,
    ATTN_ARM_BWD_KVGRID,
    ATTN_ARM_BWD_KVRECOMPUTE,
    ATTN_ARM_BWD_KVSPLIT,
    ATTN_ARM_BWD_STASH,
    ATTN_ARM_BWD_TILED,
    ATTN_ARM_DEFAULT,
    ATTN_ARM_KV_BITS,
    ATTN_ARM_KVROWS32,
    ATTN_ARM_PREFLUSH,
    ATTN_ARM_TRIAL,
    ATTN_DEFAULT_KV_KEYS,
    ATTN_SHIPPED_BWD_KV,
    ATTN_STASH_HD,
    FUSED_CORNER,
    FUSED_RAN,
    FUSED_REFUSED_REGIME,
    fused_attention_arm_backward_resolved,
    fused_attention_arm_forward_resolved,
    fused_attention_arm_from_env,
    fused_attention_arm_name,
    fused_attention_arm_parse,
    fused_backward_launch_estash_ran,
    fused_forward_launch_estash_ran,
    fused_supported_head_dim,
    fused_forward_supported_head_dim,
)
from transformer.impl.llama.modeling_llama import (
    LlamaDeviceStages,
    LlamaDims,
    PLANT_AT_NONE,
    _download,
    _upload,
    attention_eager_core,
    eager_attention_forward,
    llama_attention_scale,
    llama_key_lo,
    llama_key_span,
)


comptime SEED: UInt64 = 0x4675736564417474
"""'FusedAtt', distinct from every fixture seed base."""

comptime EXPECT_ANY = -1


@fieldwise_init
struct FusedCase(Copyable, Movable):
    var name: String
    var b: Int
    var l: Int
    var nh: Int
    var nkv: Int
    var hd: Int
    var window: Int
    var pos0: Int
    var v_lo: Float64
    var v_hi: Float64
    var v_flip_at: Int
    """Keys at or past this packed index take `-v` (the underflow tail);
    -1 for none."""
    var q_scale: Float64
    var dctx_scale: Float64
    var expect_fwd: Int
    var expect_bwd: Int


def cases() -> List[FusedCase]:
    var out = List[FusedCase]()
    out.append(FusedCase("win45_hd64_l150", 1, 150, 4, 2, 64, 45, 0, -1.0, 1.0, -1, 1.0, 1.0, FUSED_RAN, FUSED_RAN))
    out.append(FusedCase("base_hd64_l300", 2, 300, 4, 2, 64, 0, 0, -1.0, 1.0, -1, 1.0, 1.0, FUSED_RAN, FUSED_RAN))
    out.append(FusedCase("win96_hd64_l300", 2, 300, 4, 2, 64, 96, 0, -1.0, 1.0, -1, 1.0, 1.0, FUSED_RAN, FUSED_RAN))
    out.append(FusedCase("win2048_hd64_l700_nrep4", 1, 700, 4, 1, 64, 2048, 0, -1.0, 1.0, -1, 1.0, 1.0, FUSED_RAN, FUSED_RAN))
    out.append(FusedCase("decode_hd64_pos200", 2, 1, 4, 2, 64, 0, 200, -1.0, 1.0, -1, 1.0, 1.0, FUSED_RAN, FUSED_RAN))
    out.append(FusedCase("decode_win64_pos200", 2, 1, 4, 2, 64, 64, 200, -1.0, 1.0, -1, 1.0, 1.0, FUSED_RAN, FUSED_RAN))
    out.append(FusedCase("split_win50_pos100_l37", 1, 37, 2, 1, 64, 50, 100, -1.0, 1.0, -1, 1.0, 1.0, FUSED_RAN, FUSED_RAN))
    out.append(FusedCase("hd16_win7_l40", 2, 40, 2, 1, 16, 7, 0, -1.0, 1.0, -1, 1.0, 1.0, FUSED_RAN, FUSED_RAN))
    out.append(FusedCase("hd24_l33", 1, 33, 2, 1, 24, 0, 0, -1.0, 1.0, -1, 1.0, 1.0, FUSED_RAN, FUSED_RAN))
    # Forward hd128 now uses 18,624 shared bytes; the backward still uses
    # its legacy page and may independently refuse on a 32 KB column.
    var hd128 = FUSED_RAN if fused_supported_head_dim(128) else FUSED_REFUSED_REGIME
    var f128 = FUSED_RAN if fused_forward_supported_head_dim(128) else FUSED_REFUSED_REGIME
    out.append(FusedCase("hd128_win20_l70", 1, 70, 2, 2, 128, 20, 0, -1.0, 1.0, -1, 1.0, 1.0, f128, hd128))
    out.append(FusedCase("hd128_win45_l150", 1, 150, 4, 1, 128, 45, 0, -1.0, 1.0, -1, 1.0, 1.0, f128, hd128))
    # Every visible product `w * v` flushes to `-0.0` (|v| ~ 1e-37, w >= 1/48),
    # and the keys from 40 on carry `+v`, so the eager tail launders rows
    # t < 40 to `+0.0`. The fused chain must report the corner.
    var c128 = FUSED_CORNER if f128 == FUSED_RAN else FUSED_REFUSED_REGIME
    out.append(FusedCase("underflow_hd128_l48", 1, 48, 2, 1, 128, 0, 0, -1.2e-37, -0.8e-37, 40, 1.0, 1.0, c128, EXPECT_ANY))
    out.append(FusedCase("underflow_hd64_l48", 1, 48, 2, 1, 64, 0, 0, -1.2e-37, -0.8e-37, 40, 1.0, 1.0, FUSED_CORNER, EXPECT_ANY))
    out.append(FusedCase("regime_q1e30", 1, 32, 2, 1, 64, 0, 0, -1.0, 1.0, -1, 1e30, 1.0, FUSED_REFUSED_REGIME, FUSED_REFUSED_REGIME))
    out.append(FusedCase("regime_dctx1e30", 1, 32, 2, 1, 64, 0, 0, -1.0, 1.0, -1, 1.0, 1e30, FUSED_RAN, FUSED_REFUSED_REGIME))
    return out^


def status_name(st: Int) -> String:
    if st == FUSED_RAN:
        return String("RAN")
    if st == FUSED_REFUSED_REGIME:
        return String("REFUSED_REGIME")
    if st == FUSED_CORNER:
        return String("CORNER")
    return String("status ") + String(st)


def hexbits(v: Float32) -> String:
    return bits32_hex(v)


def compare(name: String, what: String, a: List[Float32], b: List[Float32]) raises -> Int:
    """Cells that differ BY BITS; the first one is named."""
    if len(a) != len(b):
        raise Error(
            name + ": " + what + " lengths differ, " + String(len(a)) + " vs "
            + String(len(b))
        )
    var moved = 0
    var first = -1
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            if first < 0:
                first = i
            moved += 1
    if moved > 0:
        print(
            "    " + what + ": " + String(moved) + " of " + String(len(a))
            + " cells MOVED; first at " + String(first) + " eager "
            + hexbits(a[first]) + " fused " + hexbits(b[first])
        )
    else:
        print("    " + what + ": " + String(len(a)) + " cells bit-identical")
    return moved


def scaled(values: List[Float32], f: Float64) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(values)):
        out.append(Float32(Float64(values[i]) * f))
    return out^


def expected_ran(hd: Int, st: Int, resolved: Int) -> Int:
    """The arm word a launch must report: nothing but the shipped kernels
    (word 0) when it refused before any kernel, or at a head dim other than
    the arms' 64; otherwise the arm's resolved half for this build."""
    if st == FUSED_REFUSED_REGIME or hd != ATTN_STASH_HD:
        return ATTN_ARM_BASELINE
    return resolved


def require_ran(name: String, direction: String, got: Int, want: Int) raises:
    if got != want:
        raise Error(
            name + ": the fused " + direction + " RAN "
            + fused_attention_arm_name(got) + " (word " + String(got)
            + ") and this build resolves " + fused_attention_arm_name(want)
            + " (word " + String(want) + ")"
        )


def dkdv_kernel_name(word: Int) -> String:
    """The dk/dv launch a BACKWARD arm word names at head_dim 64 (brief
    sections 4, 12, 14, 16 and 18): the shipped recompute kernel for word 0,
    the 2525 stash kernel, the 2527 tiled fold or its 2533 preflushed copy,
    the 2596 recompute kernel after the stashes are freed, the 2597 joint
    kernel at its keys per block, or the 2597 single fold launched twice.
    Read from a `ran` or resolved word, so it names what the launcher
    reported, never a copy of the dispatch."""
    if (word & ATTN_ARM_BWD_STASH) == 0:
        return String("fused_bwd_dkdv_kernel[64,4]")
    if (word & ATTN_ARM_BWD_TILED) == 0:
        return String("fused_bwd_dkdv_stash_kernel[64,4]")
    if (word & ATTN_ARM_BWD_KVRECOMPUTE) != 0:
        return String("fused_bwd_dkdv_kernel[64,4]_stashes_freed")
    var keys = 64
    if (word & ATTN_ARM_KVROWS32) != 0:
        keys = 32
    if (word & ATTN_ARM_BWD_KVSPLIT) != 0:
        return String("fused_bwd_kvfold_r2_kernel[64,") + String(keys) + String("]x2")
    if (word & ATTN_ARM_BWD_KVGRID) != 0:
        return String("fused_bwd_dkdv_r2_kernel[64,") + String(keys) + String("]")
    if (word & ATTN_ARM_PREFLUSH) != 0:
        return String("fused_bwd_dkdv_tiled_pf_kernel[64]")
    return String("fused_bwd_dkdv_tiled_kernel[64]")


def run_case(
    ctx: DeviceContext,
    c: FusedCase,
    arm: Int,
    mut arm_launches: Int,
    mut bwd_launches: Int,
    mut kv_launches: Int,
) raises -> Int:
    """Returns the number of moved cells across every compared buffer;
    `arm_launches` counts the direct launches that ran an arm (word not 0),
    `bwd_launches` the backward ones among them, and `kv_launches` the
    backward launches that reported a DEVIATION 2596 / 2597 dk/dv token."""
    var dm = c.nh * c.hd
    var dims = LlamaDims(dm, c.nh, c.nkv, c.hd, 4 * dm)
    dims.validate()
    var b = c.b
    var l = c.l
    var pos0 = c.pos0
    var window = c.window
    var key_lo = llama_key_lo(pos0, window)
    var s = llama_key_span(pos0, l, window)
    var s_max = pos0 + l
    var scale = llama_attention_scale(c.hd)
    var qn = b * l * c.nh * c.hd
    var kn = b * c.nkv * s * c.hd
    print(
        "  case " + c.name + "  B=" + String(b) + " L=" + String(l) + " S="
        + String(s) + " key_lo=" + String(key_lo) + " win=" + String(window)
        + " pos0=" + String(pos0) + " nh=" + String(c.nh) + " nkv="
        + String(c.nkv) + " hd=" + String(c.hd)
    )

    var stages = LlamaDeviceStages(ctx, b, l, s_max, dims, window)
    var q = fixture_tensor(SEED, 1, qn, -1.0, 1.0)
    if c.q_scale != 1.0:
        q = scaled(q, c.q_scale)
    var k = fixture_tensor(SEED, 2, kn, -1.0, 1.0)
    var v = fixture_tensor(SEED, 3, kn, c.v_lo, c.v_hi)
    if c.v_flip_at >= 0:
        # Flat index (b, kvh, j, d): flip the sign of every key at or past
        # the flip point.
        for i in range(kn):
            var j = (i // c.hd) % s
            if j >= c.v_flip_at:
                v[i] = -v[i]
    stages.q_rope = _upload(ctx, q)
    stages.k_cache = _upload(ctx, k)
    stages.v_cache = _upload(ctx, v)

    var off = IdentityTrace.disabled()
    var empty_i = List[Int]()
    var empty_b = List[UInt32]()
    attention_eager_core(
        ctx, stages, b, l, s, pos0, key_lo, window, dims, PLANT_AT_NONE,
        empty_i, empty_b, off, String(""),
    )
    var e_ctx = _download(ctx, stages.ctxv, qn)
    var e_max = _download(ctx, stages.amax, b * c.nh * l)
    var e_den = _download(ctx, stages.denom, b * c.nh * l)

    var moved = 0
    # DEVIATION 2652: this check's own kept exp stash for the direct launches
    # (grown by the forward launcher under an `_estash` arm).
    var kept = _upload(ctx, List[Float32](length=1, fill=Float32(0.0)))
    var kept_cells = 0
    # ---- the fused forward, directly, with the arm it ran ------------------
    stages.ctxv = _upload(ctx, List[Float32](length=qn, fill=Float32(0.0)))
    var ran_f = -1
    var st = fused_forward_launch_estash_ran(
        ctx, stages.ctxv, stages.amax, stages.denom, stages.q_rope,
        stages.k_cache, stages.v_cache, kept, b, l, c.nh, c.nkv, c.hd, s, pos0,
        key_lo, window, scale, arm, ran_f, kept_cells,
    )
    print("    fused forward status: " + status_name(st) + "  ran " + fused_attention_arm_name(ran_f))
    if c.expect_fwd != EXPECT_ANY and st != c.expect_fwd:
        raise Error(
            c.name + ": the fused forward reported " + status_name(st)
            + " and the case expects " + status_name(c.expect_fwd)
        )
    require_ran(c.name, "forward", ran_f, expected_ran(c.hd, st, fused_attention_arm_forward_resolved(arm)))
    if ran_f != ATTN_ARM_BASELINE:
        arm_launches += 1
    if st == FUSED_RAN:
        moved += compare(c.name, "fwd ctx (direct)", e_ctx, _download(ctx, stages.ctxv, qn))
        moved += compare(c.name, "fwd amax", e_max, _download(ctx, stages.amax, b * c.nh * l))
        moved += compare(c.name, "fwd denom", e_den, _download(ctx, stages.denom, b * c.nh * l))
    # ---- the wrapper the surface calls: equal to eager whatever it took --
    stages.ctxv = _upload(ctx, List[Float32](length=qn, fill=Float32(0.0)))
    var off2 = IdentityTrace.disabled()
    var wst = eager_attention_forward(
        ctx, stages, b, l, s, pos0, key_lo, window, dims, PLANT_AT_NONE,
        empty_i, empty_b, off2, String(""), False,
    )
    print("    wrapper status: " + status_name(wst))
    print("    WRAPPER status=" + status_name(wst) + " estash_cells=" + String(stages.attn_estash_cells) + " cells=" + String(b * c.nh * l * s))
    moved += compare(c.name, "fwd ctx (wrapper)", e_ctx, _download(ctx, stages.ctxv, qn))
    moved += compare(c.name, "fwd amax (wrapper)", e_max, _download(ctx, stages.amax, b * c.nh * l))
    moved += compare(c.name, "fwd denom (wrapper)", e_den, _download(ctx, stages.denom, b * c.nh * l))

    # ---- the backward -----------------------------------------------------
    var bst = LlamaBackwardStages(ctx, b, l, s_max, dims)
    var dctx = fixture_tensor(SEED, 4, qn, -1.0, 1.0)
    if c.dctx_scale != 1.0:
        dctx = scaled(dctx, c.dctx_scale)
    bst.d_attn_ctx = _upload(ctx, dctx)
    var offb = IdentityTrace.disabled()
    bwd_attention_eager_stages(
        ctx, bst, stages, b, l, s, pos0, key_lo, window, dims, scale, offb,
        String(""),
    )
    var e_z = _download(ctx, bst.attn_zdot, b * c.nh * l)
    var e_dq = _download(ctx, bst.d_q_rope, qn)
    var e_dk = _download(ctx, bst.d_k_cache, kn)
    var e_dv = _download(ctx, bst.d_v_cache, kn)
    bst.attn_zdot = _upload(ctx, List[Float32](length=b * c.nh * l, fill=Float32(0.0)))
    bst.d_q_rope = _upload(ctx, List[Float32](length=qn, fill=Float32(0.0)))
    bst.d_k_cache = _upload(ctx, List[Float32](length=kn, fill=Float32(0.0)))
    bst.d_v_cache = _upload(ctx, List[Float32](length=kn, fill=Float32(0.0)))
    var ran_b = -1
    var bs = fused_backward_launch_estash_ran(
        ctx, bst.attn_zdot, bst.d_q_rope, bst.d_k_cache, bst.d_v_cache,
        stages.q_rope, bst.d_attn_ctx, stages.k_cache, stages.v_cache,
        stages.amax, stages.denom, kept, kept_cells, b, l, c.nh, c.nkv, c.hd,
        s, pos0, key_lo, window, scale, arm, ran_b,
    )
    print("    fused backward status: " + status_name(bs) + "  ran " + fused_attention_arm_name(ran_b))
    if c.expect_bwd != EXPECT_ANY and bs != c.expect_bwd:
        raise Error(
            c.name + ": the fused backward reported " + status_name(bs)
            + " and the case expects " + status_name(c.expect_bwd)
        )
    require_ran(c.name, "backward", ran_b, expected_ran(c.hd, bs, fused_attention_arm_backward_resolved(arm)))
    if ran_b != ATTN_ARM_BASELINE:
        arm_launches += 1
        bwd_launches += 1
    if (ran_b & ATTN_ARM_KV_BITS) != 0:
        kv_launches += 1
    if bs == FUSED_RAN:
        moved += compare(c.name, "bwd zdot", e_z, _download(ctx, bst.attn_zdot, b * c.nh * l))
        moved += compare(c.name, "bwd dq", e_dq, _download(ctx, bst.d_q_rope, qn))
        moved += compare(c.name, "bwd dk", e_dk, _download(ctx, bst.d_k_cache, kn))
        moved += compare(c.name, "bwd dv", e_dv, _download(ctx, bst.d_v_cache, kn))
    _ = kept^
    _ = bst^
    _ = stages^
    return moved


def main() raises:
    print("=== transformer fused attention vs eager, BITWISE (mode " + numeric_mode_name() + ")")
    # The arm every launch below runs: the column's default on a build
    # without the trial define (it reads no environment), else
    # MOJOLEARN_ATTN_ARM or the default when that is unset.
    var arm = fused_attention_arm_from_env()
    var default_name = fused_attention_arm_name(ATTN_ARM_DEFAULT)
    print(
        "DEFAULT column=" + column_name(TARGET_COLUMN) + " arm=" + default_name
        + " word=" + String(ATTN_ARM_DEFAULT)
        + " source=kernel_matrix.attn_default_arm_for trial_hook=" + String(ATTN_ARM_TRIAL)
    )
    if fused_attention_arm_parse(default_name) != ATTN_ARM_DEFAULT:
        raise Error(
            "transformer_fused_check: the default arm word " + String(ATTN_ARM_DEFAULT)
            + " names itself '" + default_name + "', which parses back as another arm"
        )
    var default_resolved = (
        fused_attention_arm_forward_resolved(ATTN_ARM_DEFAULT)
        | fused_attention_arm_backward_resolved(ATTN_ARM_DEFAULT)
    )
    print("DEFAULT resolved_hd64=" + fused_attention_arm_name(default_resolved))
    if default_resolved != ATTN_ARM_DEFAULT:
        raise Error(
            "transformer_fused_check: the column's default arm " + default_name
            + " resolves to " + fused_attention_arm_name(default_resolved)
            + " at head_dim 64 on this build (a page that does not fit this"
            + " column, or a kernel this build does not compile); the kernel"
            + " matrix row names an arm the column cannot run as named"
        )
    var arm_name = fused_attention_arm_name(arm)
    var fwd_name = fused_attention_arm_name(fused_attention_arm_forward_resolved(arm))
    var bwd_name = fused_attention_arm_name(fused_attention_arm_backward_resolved(arm))
    print(
        "ARM this_run=" + arm_name + " is_default=" + String(arm == ATTN_ARM_DEFAULT)
        + " forward_hd64=" + fwd_name + " backward_hd64=" + bwd_name
    )
    # Brief section 18: which dk/dv kernel the default's and this run's
    # backward launch at head_dim 64 on this build.
    var default_bwd = fused_attention_arm_backward_resolved(ATTN_ARM_DEFAULT)
    var arm_bwd = fused_attention_arm_backward_resolved(arm)
    var arm_dkdv = dkdv_kernel_name(arm_bwd)
    print(
        "DKDV default_hd64=" + dkdv_kernel_name(default_bwd)
        + " this_run_hd64=" + arm_dkdv
        + " shipped_kv_branch=" + String(ATTN_SHIPPED_BWD_KV)
        + " default_kv_keys=" + String(ATTN_DEFAULT_KV_KEYS)
    )
    var ctx = DeviceContext()
    var all_cases = cases()
    var total_moved = 0
    var arm_launches = 0
    var bwd_launches = 0
    var kv_launches = 0
    var n = 0
    for i in range(len(all_cases)):
        var c = all_cases[i].copy()
        total_moved += run_case(ctx, c, arm, arm_launches, bwd_launches, kv_launches)
        n += 1
    if total_moved > 0:
        raise Error(
            "transformer_fused_check: " + String(total_moved)
            + " cells MOVED between the eager and the fused attention"
        )
    if arm_launches == 0 and arm != ATTN_ARM_BASELINE:
        raise Error(
            "transformer_fused_check: no direct launch ran the arm " + arm_name
            + " (every one ran the shipped kernels); the head_dim 64 cases did not reach it"
        )
    var arm_kv = (arm_bwd & ATTN_ARM_KV_BITS) != 0
    if arm_kv and kv_launches == 0:
        raise Error(
            "transformer_fused_check: the arm " + arm_name + " resolves its dk/dv"
            + " to " + arm_dkdv + " at head_dim 64 on this build and no direct"
            + " backward launch reported that kernel"
        )
    if (not arm_kv) and kv_launches > 0:
        raise Error(
            "transformer_fused_check: " + String(kv_launches) + " direct backward"
            + " launches reported a DEVIATION 2596 / 2597 dk/dv kernel and the arm "
            + arm_name + " resolves to " + arm_dkdv
        )
    print(
        "transformer_fused_check: PASS, " + String(n)
        + " cases, every compared buffer bit-identical, every status as expected;"
        + " column " + column_name(TARGET_COLUMN) + " arm " + arm_name
        + " (default " + default_name + "), " + String(arm_launches)
        + " direct launches RAN " + fwd_name + " / " + bwd_name + " at head_dim 64;"
        + " dk/dv " + arm_dkdv + " in " + String(bwd_launches) + " backward launches"
    )
    _ = ctx^
