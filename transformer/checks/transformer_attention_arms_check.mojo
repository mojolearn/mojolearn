# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fused attention ARMS (DEVIATIONS 2525 to 2528) against the eager
stage kernels, BIT FOR BIT, on the fused check's cases, plus reach by
sabotage. Additive beside `transformer_fused_check.mojo`, which gates the
shipped kernels; this file gates the arms on the same cases.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . \\
        transformer/checks/transformer_attention_arms_check.mojo

WHAT IT ASSERTS.
  * NAMES first, host only: `fused_attention_arm_parse` and
    `fused_attention_arm_name` are inverses on every valid spelling and
    on every arm below with each sabotage bit, and the parser refuses the
    invalid spellings (brief section 12.1).
  * For every case of `transformer_fused_check.cases()` and every arm
    (bwd_stash, fwd_sstash, bwd_stash_tiled, stash_tiled, the shipped
    default, and stash_tiled_ztiled_r64 and stash_tiled_ztiled_r32,
    DEVIATION 2528 at both geometries, so "new arm = eager" and
    "default = eager" hold in one run):
      - the arm's launcher reports the status the case expects (RAN, the
        regime refusal, the corner), exactly as the shipped kernels must;
      - on RAN, ctx, amax, denom (forward) and zdot, dq, dk, dv (backward)
        equal the eager kernels' bits;
      - REACH, on a head-dim-64 case, with the arm's reach bit
        (`fused_attention_arm_reach_bit`: ATTN_ARM_SABOTAGE for a
        first-round arm, ATTN_ARM_SABOTAGE_NEW for an arm with a
        second-round bit). A first-round forward bit must move the forward
        (ctx or denom) on every case whose forward RAN; a first-round
        backward bit, or DEVIATION 2528 under sabotage_new, must move the
        backward (zdot, dq, dk or dv) on every case whose backward RAN;
        and a sabotage_new run of an arm with no second-round forward bit
        must move NO forward cell, so the proof names the new kernel and
        not the stash_tiled kernels under it. Reach is per branch, never a
        sum of both directions.
      - at any other head dim every arm takes the shipped kernels by
        design, and a sabotage run must move nothing.
Without -D MOJOLEARN_ATTN_ARM_TRIAL=1 every non-default arm runs the
default kernels; the reach section then FAILS, naming the missing define,
so a green run is always a run of the arms.

This is a SMALL-SHAPE gate (L <= 700): it says the arms compute the
profile's bits where the fused check says the shipped kernels do. The
target shape, the price and the LM step are the harness's
(`bench/attention_step_price_main.mojo`) and the leg's.
"""

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from checks.numerics import numeric_mode_name
from transformer.checks.transformer_fixture import fixture_tensor
from transformer.checks.transformer_backward import (
    LlamaBackwardStages,
    bwd_attention_eager_stages,
)
from transformer.checks.transformer_fused_check import (
    EXPECT_ANY,
    FusedCase,
    SEED,
    cases,
    compare,
    scaled,
    status_name,
)
from transformer.impl.llama.fused_attention import (
    ATTN_ARM_BWD_STASH,
    ATTN_ARM_BWD_TILED,
    ATTN_ARM_BWD_ZTILED,
    ATTN_ARM_FWD_SSTASH,
    ATTN_ARM_NEW_FWD_BITS,
    ATTN_ARM_SABOTAGE,
    ATTN_ARM_SABOTAGE_NEW,
    ATTN_ARM_TRIAL,
    ATTN_ARM_ZROWS32,
    ATTN_ARM_ZROWS64,
    ATTN_STASH_HD,
    FUSED_RAN,
    fused_attention_arm_name,
    fused_attention_arm_parse,
    fused_attention_arm_reach_bit,
    fused_backward_launch_arm,
    fused_forward_launch_arm,
)
from transformer.impl.llama.modeling_llama import (
    LlamaDeviceStages,
    LlamaDims,
    PLANT_AT_NONE,
    _download,
    _upload,
    attention_eager_core,
    llama_attention_scale,
    llama_key_lo,
    llama_key_span,
)


def arms() -> List[Int]:
    comptime stash_tiled = ATTN_ARM_FWD_SSTASH | ATTN_ARM_BWD_STASH | ATTN_ARM_BWD_TILED
    var out = List[Int]()
    out.append(ATTN_ARM_BWD_STASH)
    out.append(ATTN_ARM_FWD_SSTASH)
    out.append(ATTN_ARM_BWD_STASH | ATTN_ARM_BWD_TILED)
    out.append(stash_tiled)
    out.append(stash_tiled | ATTN_ARM_BWD_ZTILED | ATTN_ARM_ZROWS64)
    out.append(stash_tiled | ATTN_ARM_BWD_ZTILED | ATTN_ARM_ZROWS32)
    return out^


def check_names(mut failures: List[String]) raises:
    """The parser and the name function are inverses (brief 12.1)."""
    var good = List[String]()
    good.append("baseline")
    good.append("bwd_stash")
    good.append("fwd_sstash")
    good.append("bwd_stash_tiled")
    good.append("stash")
    good.append("stash_tiled")
    good.append("bwd_stash_tiled_ztiled")
    good.append("stash_tiled_ztiled")
    good.append("stash_tiled_ztiled_r32")
    good.append("stash_tiled_ztiled_r64")
    good.append("stash_tiled+sabotage")
    good.append("stash_tiled_ztiled_r32+sabotage_new")
    good.append("stash_tiled_ztiled_r64+sabotage+sabotage_new")
    for i in range(len(good)):
        var n = String(good[i])
        var got = fused_attention_arm_name(fused_attention_arm_parse(n))
        if got != n:
            failures.append("NAMES: '" + n + "' parses and names back as '" + got + "'")
    var all_arms = arms()
    for i in range(len(all_arms)):
        for extra in range(4):
            var a = all_arms[i]
            if extra == 1:
                a = a | ATTN_ARM_SABOTAGE
            elif extra == 2:
                a = a | ATTN_ARM_SABOTAGE_NEW
            elif extra == 3:
                a = a | ATTN_ARM_SABOTAGE | ATTN_ARM_SABOTAGE_NEW
            var back = fused_attention_arm_parse(fused_attention_arm_name(a))
            if back != a:
                failures.append("NAMES: arm " + String(a) + " names as '" + fused_attention_arm_name(a) + "' and parses back as " + String(back))
    var bad = List[String]()
    bad.append("")
    bad.append("arm4")
    bad.append("stash_ztiled")
    bad.append("bwd_stash_ztiled")
    bad.append("stash_tiled_r32")
    bad.append("stash_tiled_ztiled_r32_r64")
    bad.append("stash_tiled_bits1024")
    bad.append("stash_tiled_ztiled+sabotage_new+sabotage")
    for i in range(len(bad)):
        var refused = False
        try:
            _ = fused_attention_arm_parse(bad[i])
        except:
            refused = True
        if not refused:
            failures.append("NAMES: '" + bad[i] + "' was accepted; the parser must refuse it")
    print(
        "  names: " + String(len(good)) + " spellings and "
        + String(4 * len(all_arms)) + " arm values round-trip, "
        + String(len(bad)) + " invalid spellings refused"
    )


def run_case(ctx: DeviceContext, c: FusedCase, mut failures: List[String]) raises:
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
    var rn = b * c.nh * l
    print(
        "  case " + c.name + "  B=" + String(b) + " L=" + String(l) + " S="
        + String(s) + " win=" + String(window) + " pos0=" + String(pos0)
        + " nh=" + String(c.nh) + " nkv=" + String(c.nkv) + " hd=" + String(c.hd)
    )

    var stages = LlamaDeviceStages(ctx, b, l, s_max, dims, window)
    var q = fixture_tensor(SEED, 1, qn, -1.0, 1.0)
    if c.q_scale != 1.0:
        q = scaled(q, c.q_scale)
    var k = fixture_tensor(SEED, 2, kn, -1.0, 1.0)
    var v = fixture_tensor(SEED, 3, kn, c.v_lo, c.v_hi)
    if c.v_flip_at >= 0:
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
    var e_max = _download(ctx, stages.amax, rn)
    var e_den = _download(ctx, stages.denom, rn)

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
    var e_z = _download(ctx, bst.attn_zdot, rn)
    var e_dq = _download(ctx, bst.d_q_rope, qn)
    var e_dk = _download(ctx, bst.d_k_cache, kn)
    var e_dv = _download(ctx, bst.d_v_cache, kn)
    # The eager path wrote amax and denom into `stages`; the fused backward
    # reads them from there, as the block does.
    var f_amax = _upload(ctx, e_max)
    var f_den = _upload(ctx, e_den)

    var all_arms = arms()
    for ai in range(len(all_arms)):
        var arm = all_arms[ai]
        var name = fused_attention_arm_name(arm)
        var reach_bit = fused_attention_arm_reach_bit(arm)
        # Which direction this arm's sabotage must reach (per branch).
        var need_fwd = (arm & ATTN_ARM_FWD_SSTASH) != 0
        var need_bwd = (arm & ATTN_ARM_BWD_STASH) != 0
        var fwd_must_hold = False
        if reach_bit == ATTN_ARM_SABOTAGE_NEW:
            need_fwd = (arm & ATTN_ARM_NEW_FWD_BITS) != 0
            need_bwd = (arm & ATTN_ARM_BWD_ZTILED) != 0
            fwd_must_hold = not need_fwd
        for sab in range(2):
            var this_arm = arm
            if sab == 1:
                this_arm = arm | reach_bit
            var label = fused_attention_arm_name(this_arm)
            # ---- forward ------------------------------------------------
            var ctxv = _upload(ctx, List[Float32](length=qn, fill=Float32(0.0)))
            var amax = _upload(ctx, List[Float32](length=rn, fill=Float32(0.0)))
            var denom = _upload(ctx, List[Float32](length=rn, fill=Float32(0.0)))
            var st = fused_forward_launch_arm(
                ctx, ctxv, amax, denom, stages.q_rope, stages.k_cache,
                stages.v_cache, b, l, c.nh, c.nkv, c.hd, s, pos0, key_lo,
                window, scale, this_arm,
            )
            print("    " + label + " forward status: " + status_name(st))
            var fwd_moved = 0
            if sab == 0:
                if c.expect_fwd != EXPECT_ANY and st != c.expect_fwd:
                    failures.append(c.name + " " + label + ": forward reported " + status_name(st) + ", expected " + status_name(c.expect_fwd))
                if st == FUSED_RAN:
                    fwd_moved += compare(c.name, label + " fwd ctx", e_ctx, _download(ctx, ctxv, qn))
                    fwd_moved += compare(c.name, label + " fwd amax", e_max, _download(ctx, amax, rn))
                    fwd_moved += compare(c.name, label + " fwd denom", e_den, _download(ctx, denom, rn))
                    if fwd_moved > 0:
                        failures.append(c.name + " " + label + ": forward differs from eager in " + String(fwd_moved) + " cells")
            elif st == FUSED_RAN:
                fwd_moved += compare(c.name, label + " fwd ctx", e_ctx, _download(ctx, ctxv, qn))
                fwd_moved += compare(c.name, label + " fwd amax", e_max, _download(ctx, amax, rn))
                fwd_moved += compare(c.name, label + " fwd denom", e_den, _download(ctx, denom, rn))
            # ---- backward -----------------------------------------------
            var zdot = _upload(ctx, List[Float32](length=rn, fill=Float32(0.0)))
            var dq = _upload(ctx, List[Float32](length=qn, fill=Float32(0.0)))
            var dk = _upload(ctx, List[Float32](length=kn, fill=Float32(0.0)))
            var dv = _upload(ctx, List[Float32](length=kn, fill=Float32(0.0)))
            var bs = fused_backward_launch_arm(
                ctx, zdot, dq, dk, dv, stages.q_rope, bst.d_attn_ctx,
                stages.k_cache, stages.v_cache, f_amax, f_den, b, l, c.nh,
                c.nkv, c.hd, s, pos0, key_lo, window, scale, this_arm,
            )
            print("    " + label + " backward status: " + status_name(bs))
            var bwd_moved = 0
            if sab == 0:
                if c.expect_bwd != EXPECT_ANY and bs != c.expect_bwd:
                    failures.append(c.name + " " + label + ": backward reported " + status_name(bs) + ", expected " + status_name(c.expect_bwd))
                if bs == FUSED_RAN:
                    bwd_moved += compare(c.name, label + " bwd zdot", e_z, _download(ctx, zdot, rn))
                    bwd_moved += compare(c.name, label + " bwd dq", e_dq, _download(ctx, dq, qn))
                    bwd_moved += compare(c.name, label + " bwd dk", e_dk, _download(ctx, dk, kn))
                    bwd_moved += compare(c.name, label + " bwd dv", e_dv, _download(ctx, dv, kn))
                    if bwd_moved > 0:
                        failures.append(c.name + " " + label + ": backward differs from eager in " + String(bwd_moved) + " cells")
            elif bs == FUSED_RAN:
                bwd_moved += compare(c.name, label + " bwd zdot", e_z, _download(ctx, zdot, rn))
                bwd_moved += compare(c.name, label + " bwd dq", e_dq, _download(ctx, dq, qn))
                bwd_moved += compare(c.name, label + " bwd dk", e_dk, _download(ctx, dk, kn))
                bwd_moved += compare(c.name, label + " bwd dv", e_dv, _download(ctx, dv, kn))
            # ---- reach, per branch ---------------------------------------
            if sab == 1:
                if c.hd == ATTN_STASH_HD:
                    if need_fwd and c.expect_fwd == FUSED_RAN:
                        print("    REACH " + label + " forward_flipped_cells=" + String(fwd_moved))
                        if fwd_moved == 0:
                            failures.append(c.name + ": FORWARD REACH NOT PROVEN for " + label + " (sabotage moved no forward cell; build lacks -D MOJOLEARN_ATTN_ARM_TRIAL=1 or the arm is not wired)")
                    if need_bwd and c.expect_bwd == FUSED_RAN:
                        print("    REACH " + label + " backward_flipped_cells=" + String(bwd_moved))
                        if bwd_moved == 0:
                            failures.append(c.name + ": BACKWARD REACH NOT PROVEN for " + label + " (sabotage moved no backward cell; build lacks -D MOJOLEARN_ATTN_ARM_TRIAL=1 or the arm is not wired)")
                    if fwd_must_hold and st == FUSED_RAN:
                        print("    REACH " + label + " forward_cells_that_must_hold_moved=" + String(fwd_moved))
                        if fwd_moved > 0:
                            failures.append(c.name + ": " + label + " moved " + String(fwd_moved) + " forward cells; a second-round backward sabotage must reach no forward buffer")
                elif fwd_moved + bwd_moved > 0:
                    failures.append(c.name + ": " + label + " moved " + String(fwd_moved + bwd_moved) + " cells at head_dim " + String(c.hd) + ", where every arm runs the shipped kernels")
            _ = ctxv^
            _ = amax^
            _ = denom^
            _ = zdot^
            _ = dq^
            _ = dk^
            _ = dv^
    _ = f_amax^
    _ = f_den^
    _ = bst^
    _ = stages^


def main() raises:
    print("=== transformer fused attention ARMS vs eager, BITWISE (mode " + numeric_mode_name() + ", trial hook " + String(ATTN_ARM_TRIAL) + ")")
    if not ATTN_ARM_TRIAL:
        print("NOTE: no -D MOJOLEARN_ATTN_ARM_TRIAL=1: every non-default arm runs the default kernels; reach will FAIL")
    var failures = List[String]()
    check_names(failures)
    var ctx = DeviceContext()
    var all_cases = cases()
    var n = 0
    for i in range(len(all_cases)):
        var c = all_cases[i].copy()
        run_case(ctx, c, failures)
        n += 1
    var n_arms = len(arms())
    if len(failures) > 0:
        for i in range(len(failures)):
            print("FAIL " + failures[i])
        raise Error(
            "transformer_attention_arms_check: " + String(len(failures))
            + " failure(s) over " + String(n) + " cases x " + String(n_arms)
            + " arms; see the FAIL lines"
        )
    print(
        "transformer_attention_arms_check: PASS, names inverse, " + String(n)
        + " cases x " + String(n_arms) + " arms, every RAN buffer bit-identical"
        + " to eager, every status as expected, reach proven per branch at"
        + " head_dim 64, nothing moved at other head dims"
    )
    _ = ctx^
