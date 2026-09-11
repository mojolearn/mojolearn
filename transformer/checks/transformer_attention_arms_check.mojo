# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fused attention ARMS (DEVIATIONS 2525 to 2528, 2530, 2531 and 2533)
against the eager stage kernels, BIT FOR BIT, on the fused check's cases,
plus reach by sabotage. Additive beside `transformer_fused_check.mojo`,
which gates the shipped kernels; this file gates the arms on the same cases.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . \\
        transformer/checks/transformer_attention_arms_check.mojo

WHAT IT ASSERTS.
  * NAMES first, host only: `fused_attention_arm_parse` and
    `fused_attention_arm_name` are inverses on every valid spelling and
    on every arm below with each sabotage bit, and the parser refuses the
    invalid spellings (brief sections 12.1 and 14.1).
  * For every case of `transformer_fused_check.cases()` and every arm
    (bwd_stash, fwd_sstash, bwd_stash_tiled, stash_tiled (the shipped
    default on every column but NVIDIA), stash_tiled_ztiled_r64 and
    stash_tiled_ztiled_r32 (DEVIATION 2528), stash_tiled_ztiled_r64_pf
    (2528 with 2533), stash_tiled_pf (2533), stash_tiled_fgrid_r64 and
    stash_tiled_fgrid_r32 (2531), stash_tiled_fgrid_r32_pf,
    stash_tiled_fgrid_r32_qres (2530) and stash_tiled_fgrid_r32_qres_pf
    (NVIDIA's shipped default, DEVIATION 2534), then five arms on that one:
    `_kvrecompute` (DEVIATION 2596) and `_kvgrid_r64`, `_kvgrid_r32`,
    `_kvsplit`, `_kvgrid_r32_kvsplit` (DEVIATION 2597), so every second-round
    forward instantiation (rows 64 and 32, Q residency, preflush) and every
    second-round backward launch runs, and "new arm = eager" and
    "default = eager" hold in one run):
      - the arm's launcher reports the status the case expects (RAN, the
        regime refusal, the corner), exactly as the shipped kernels must;
      - DEVIATION 2534: the launcher reports (`fused_forward_launch_ran`,
        `fused_backward_launch_ran`) that it ran the arm's resolved kernels
        at head_dim 64 (`fused_attention_arm_forward_resolved`,
        `fused_attention_arm_backward_resolved`), and the shipped kernels at
        every other head dim or on a refusal, clean and sabotaged alike;
      - on RAN, ctx, amax, denom (forward) and zdot, dq, dk, dv (backward)
        equal the eager kernels' bits;
      - REACH, on a head-dim-64 case, with the arm's reach bit
        (`fused_attention_arm_reach_bit`: ATTN_ARM_SABOTAGE for a
        first-round arm, ATTN_ARM_SABOTAGE_NEW for an arm with a
        second-round bit). Per branch, never a sum of both directions:
        a first-round forward bit must move the forward (ctx or denom) on
        every case whose forward RAN and a first-round backward bit the
        backward; under sabotage_new an arm with a second-round forward
        kernel (`fused_attention_arm_new_forward`) must move the forward
        and one without must move NO forward cell, and likewise for the
        backward (`fused_attention_arm_new_backward`), so the proof names
        the new kernels and not the stash_tiled kernels under them.
      - ATTRIBUTION under sabotage_new (brief section 14.5): the 2530 flip
        (every staged Q value) must move amax or denom; the 2531 flip (the
        pass-3 weight) and the 2533 forward flip (ctx columns 0 to 15)
        must hold amax and denom, and the 2533 forward flip must hold ctx
        columns 16 and up; the 2533 backward flip (the stored zdot) must
        move zdot and hold dv when 2528 is not in the arm (2528's y flip
        moves every backward buffer, so a composed arm is attributed per
        direction only).
      - DK/DV REACH for the DEVIATION 2596 / 2597 arms, a third run under
        ATTN_ARM_SABOTAGE_KV: at head_dim 64 dk and dv must move and zdot,
        dq and the forward must hold (brief section 16.4).
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

from std.memory import bitcast
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
    expected_ran,
    scaled,
    status_name,
)
from transformer.impl.llama.fused_attention import (
    ATTN_ARM_BWD_KVGRID,
    ATTN_ARM_BWD_KVRECOMPUTE,
    ATTN_ARM_BWD_KVSPLIT,
    ATTN_ARM_BWD_STASH,
    ATTN_ARM_BWD_TILED,
    ATTN_ARM_BWD_ZTILED,
    ATTN_ARM_FROWS32,
    ATTN_ARM_FROWS64,
    ATTN_ARM_FWD_GRID,
    ATTN_ARM_FWD_QRES,
    ATTN_ARM_FWD_SSTASH,
    ATTN_ARM_KVROWS32,
    ATTN_ARM_KVROWS64,
    ATTN_ARM_PREFLUSH,
    ATTN_ARM_SABOTAGE,
    ATTN_ARM_SABOTAGE_KV,
    ATTN_ARM_SABOTAGE_NEW,
    ATTN_ARM_TRIAL,
    ATTN_ARM_ZROWS32,
    ATTN_ARM_ZROWS64,
    ATTN_STASH_HD,
    FUSED_RAN,
    fused_attention_arm_kv,
    fused_attention_arm_name,
    fused_attention_arm_new_backward,
    fused_attention_arm_new_forward,
    fused_attention_arm_parse,
    fused_attention_arm_backward_resolved,
    fused_attention_arm_forward_resolved,
    fused_attention_arm_reach_bit,
    fused_backward_launch_ran,
    fused_forward_launch_ran,
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


comptime PF_FLIP_COLUMNS = 16
"""The 2533 forward sabotage flips the ctx cells of each thread's context
lane 0, which are head-dim columns 0 to 15; columns from here up must hold."""


def arms() -> List[Int]:
    comptime stash_tiled = ATTN_ARM_FWD_SSTASH | ATTN_ARM_BWD_STASH | ATTN_ARM_BWD_TILED
    comptime grid32 = ATTN_ARM_FWD_GRID | ATTN_ARM_FROWS32
    comptime grid64 = ATTN_ARM_FWD_GRID | ATTN_ARM_FROWS64
    var out = List[Int]()
    out.append(ATTN_ARM_BWD_STASH)
    out.append(ATTN_ARM_FWD_SSTASH)
    out.append(ATTN_ARM_BWD_STASH | ATTN_ARM_BWD_TILED)
    out.append(stash_tiled)
    out.append(stash_tiled | ATTN_ARM_BWD_ZTILED | ATTN_ARM_ZROWS64)
    out.append(stash_tiled | ATTN_ARM_BWD_ZTILED | ATTN_ARM_ZROWS32)
    out.append(stash_tiled | ATTN_ARM_BWD_ZTILED | ATTN_ARM_ZROWS64 | ATTN_ARM_PREFLUSH)
    out.append(stash_tiled | ATTN_ARM_PREFLUSH)
    out.append(stash_tiled | grid64)
    out.append(stash_tiled | grid32)
    out.append(stash_tiled | grid32 | ATTN_ARM_PREFLUSH)
    out.append(stash_tiled | grid32 | ATTN_ARM_FWD_QRES)
    out.append(stash_tiled | grid32 | ATTN_ARM_FWD_QRES | ATTN_ARM_PREFLUSH)
    # DEVIATIONS 2596 and 2597 (brief section 16), on the round 3 arm.
    comptime r3 = stash_tiled | grid32 | ATTN_ARM_FWD_QRES | ATTN_ARM_PREFLUSH
    out.append(r3 | ATTN_ARM_BWD_KVRECOMPUTE)
    out.append(r3 | ATTN_ARM_BWD_KVGRID | ATTN_ARM_KVROWS64)
    out.append(r3 | ATTN_ARM_BWD_KVGRID | ATTN_ARM_KVROWS32)
    out.append(r3 | ATTN_ARM_BWD_KVSPLIT)
    out.append(r3 | ATTN_ARM_BWD_KVGRID | ATTN_ARM_KVROWS32 | ATTN_ARM_BWD_KVSPLIT)
    return out^


def check_names(mut failures: List[String]) raises:
    """The parser and the name function are inverses (brief 12.1, 14.1)."""
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
    good.append("stash_tiled_pf")
    good.append("bwd_stash_tiled_pf")
    good.append("fwd_sstash_pf")
    good.append("stash_tiled_fgrid")
    good.append("stash_tiled_fgrid_r32")
    good.append("stash_tiled_fgrid_r64")
    good.append("fwd_sstash_fgrid_r32_qres")
    good.append("stash_tiled_fgrid_r32_qres_pf")
    good.append("stash_tiled_ztiled_r64_pf")
    good.append("stash_tiled_ztiled_r32_fgrid_r32_qres_pf+sabotage_new")
    good.append("bwd_stash_tiled_ztiled_pf+sabotage+sabotage_new")
    good.append("stash_tiled_pf_kvsplit")
    good.append("bwd_stash_tiled_pf_kvgrid")
    good.append("stash_tiled_pf_kvgrid_r32")
    good.append("stash_tiled_fgrid_r32_qres_pf_kvgrid_r64")
    good.append("stash_tiled_fgrid_r32_qres_pf_kvgrid_r32_kvsplit")
    good.append("stash_tiled_fgrid_r32_qres_pf_kvsplit+sabotage_kv")
    good.append("stash_tiled_pf_kvgrid_r64_kvsplit+sabotage+sabotage_new+sabotage_kv")
    good.append("stash_tiled_fgrid_r32_qres_pf_kvrecompute")
    good.append("bwd_stash_tiled_pf_kvrecompute+sabotage_kv")
    for i in range(len(good)):
        var n = String(good[i])
        var got = fused_attention_arm_name(fused_attention_arm_parse(n))
        if got != n:
            failures.append("NAMES: '" + n + "' parses and names back as '" + got + "'")
    var all_arms = arms()
    for i in range(len(all_arms)):
        for extra in range(8):
            var a = all_arms[i]
            if (extra & 1) != 0:
                a = a | ATTN_ARM_SABOTAGE
            if (extra & 2) != 0:
                a = a | ATTN_ARM_SABOTAGE_NEW
            if (extra & 4) != 0:
                a = a | ATTN_ARM_SABOTAGE_KV
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
    bad.append("stash_tiled_bits8192")
    bad.append("stash_tiled_ztiled+sabotage_new+sabotage")
    bad.append("stash_pf")
    bad.append("bwd_stash_pf")
    bad.append("baseline_pf")
    bad.append("bwd_stash_tiled_fgrid")
    bad.append("stash_tiled_qres")
    bad.append("stash_tiled_fgrid_qres")
    bad.append("stash_tiled_fgrid_r64_qres")
    bad.append("stash_tiled_pf_fgrid_r32")
    bad.append("stash_tiled_fgrid_r32_r64")
    bad.append("stash_tiled_qres_fgrid_r32")
    bad.append("stash_tiled_kvsplit")
    bad.append("fwd_sstash_pf_kvgrid")
    bad.append("stash_tiled_ztiled_r64_pf_kvsplit")
    bad.append("stash_tiled_pf_kvsplit_kvgrid")
    bad.append("stash_tiled_pf_kvgrid_r32_r64")
    bad.append("stash_tiled_kvgrid_pf")
    bad.append("stash_tiled_pf_kvsplit+sabotage_kv+sabotage_new")
    bad.append("stash_tiled_kvrecompute")
    bad.append("stash_tiled_pf_kvrecompute_kvsplit")
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
        + String(8 * len(all_arms)) + " arm values round-trip, "
        + String(len(bad)) + " invalid spellings refused"
    )


def moved_from_column(a: List[Float32], b: List[Float32], hd: Int, lo: Int) -> Int:
    """Cells that differ BY BITS at head-dim columns `lo` and up of a
    `[B*L][nh*hd]` buffer (the flat index modulo `hd` is the column)."""
    var m = len(a)
    if len(b) < m:
        m = len(b)
    var n = 0
    for i in range(m):
        if i % hd >= lo:
            if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
                n += 1
    return n


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
        var reach_bit = fused_attention_arm_reach_bit(arm)
        var second = reach_bit == ATTN_ARM_SABOTAGE_NEW
        # Which direction this arm's sabotage must reach (per branch), and
        # which direction must hold.
        var need_fwd = (arm & ATTN_ARM_FWD_SSTASH) != 0
        var need_bwd = (arm & ATTN_ARM_BWD_STASH) != 0
        var fwd_must_hold = False
        var bwd_must_hold = False
        if second:
            need_fwd = fused_attention_arm_new_forward(arm)
            need_bwd = fused_attention_arm_new_backward(arm)
            fwd_must_hold = not need_fwd
            bwd_must_hold = not need_bwd
        var qres = (arm & ATTN_ARM_FWD_QRES) != 0
        var preflush = (arm & ATTN_ARM_PREFLUSH) != 0
        var ztiled = (arm & ATTN_ARM_BWD_ZTILED) != 0
        var kv = fused_attention_arm_kv(arm)
        for sab in range(3):
            # sab 2: DEVIATIONS 2596 / 2597, the dk/dv launch's own flip
            # (ATTN_ARM_SABOTAGE_KV), for the arms that carry one.
            if sab == 2 and not kv:
                continue
            var this_arm = arm
            if sab == 1:
                this_arm = arm | reach_bit
            elif sab == 2:
                this_arm = arm | ATTN_ARM_SABOTAGE_KV
            var label = fused_attention_arm_name(this_arm)
            # ---- forward ------------------------------------------------
            var ctxv = _upload(ctx, List[Float32](length=qn, fill=Float32(0.0)))
            var amax = _upload(ctx, List[Float32](length=rn, fill=Float32(0.0)))
            var denom = _upload(ctx, List[Float32](length=rn, fill=Float32(0.0)))
            var ran_f = -1
            var st = fused_forward_launch_ran(
                ctx, ctxv, amax, denom, stages.q_rope, stages.k_cache,
                stages.v_cache, b, l, c.nh, c.nkv, c.hd, s, pos0, key_lo,
                window, scale, this_arm, ran_f,
            )
            print("    " + label + " forward status: " + status_name(st) + "  ran " + fused_attention_arm_name(ran_f))
            # DEVIATION 2534: the kernels that launched are the arm's resolved
            # forward (the shipped kernels at other head dims or on a refusal).
            var want_f = expected_ran(c.hd, st, fused_attention_arm_forward_resolved(this_arm))
            if ran_f != want_f:
                failures.append(c.name + " " + label + ": forward RAN " + fused_attention_arm_name(ran_f) + " and the build resolves " + fused_attention_arm_name(want_f))
            var m_ctx = 0
            var m_amax = 0
            var m_den = 0
            var m_ctx_hi = 0
            if st == FUSED_RAN:
                var got_ctx = _download(ctx, ctxv, qn)
                m_ctx = compare(c.name, label + " fwd ctx", e_ctx, got_ctx)
                m_amax = compare(c.name, label + " fwd amax", e_max, _download(ctx, amax, rn))
                m_den = compare(c.name, label + " fwd denom", e_den, _download(ctx, denom, rn))
                m_ctx_hi = moved_from_column(e_ctx, got_ctx, c.hd, PF_FLIP_COLUMNS)
            var fwd_moved = m_ctx + m_amax + m_den
            if sab == 0:
                if c.expect_fwd != EXPECT_ANY and st != c.expect_fwd:
                    failures.append(c.name + " " + label + ": forward reported " + status_name(st) + ", expected " + status_name(c.expect_fwd))
                if fwd_moved > 0:
                    failures.append(c.name + " " + label + ": forward differs from eager in " + String(fwd_moved) + " cells")
            # ---- backward -----------------------------------------------
            var zdot = _upload(ctx, List[Float32](length=rn, fill=Float32(0.0)))
            var dq = _upload(ctx, List[Float32](length=qn, fill=Float32(0.0)))
            var dk = _upload(ctx, List[Float32](length=kn, fill=Float32(0.0)))
            var dv = _upload(ctx, List[Float32](length=kn, fill=Float32(0.0)))
            var ran_b = -1
            var bs = fused_backward_launch_ran(
                ctx, zdot, dq, dk, dv, stages.q_rope, bst.d_attn_ctx,
                stages.k_cache, stages.v_cache, f_amax, f_den, b, l, c.nh,
                c.nkv, c.hd, s, pos0, key_lo, window, scale, this_arm, ran_b,
            )
            print("    " + label + " backward status: " + status_name(bs) + "  ran " + fused_attention_arm_name(ran_b))
            var want_b = expected_ran(c.hd, bs, fused_attention_arm_backward_resolved(this_arm))
            if ran_b != want_b:
                failures.append(c.name + " " + label + ": backward RAN " + fused_attention_arm_name(ran_b) + " and the build resolves " + fused_attention_arm_name(want_b))
            var m_z = 0
            var m_dq = 0
            var m_dk = 0
            var m_dv = 0
            if bs == FUSED_RAN:
                m_z = compare(c.name, label + " bwd zdot", e_z, _download(ctx, zdot, rn))
                m_dq = compare(c.name, label + " bwd dq", e_dq, _download(ctx, dq, qn))
                m_dk = compare(c.name, label + " bwd dk", e_dk, _download(ctx, dk, kn))
                m_dv = compare(c.name, label + " bwd dv", e_dv, _download(ctx, dv, kn))
            var bwd_moved = m_z + m_dq + m_dk + m_dv
            if sab == 0:
                if c.expect_bwd != EXPECT_ANY and bs != c.expect_bwd:
                    failures.append(c.name + " " + label + ": backward reported " + status_name(bs) + ", expected " + status_name(c.expect_bwd))
                if bwd_moved > 0:
                    failures.append(c.name + " " + label + ": backward differs from eager in " + String(bwd_moved) + " cells")
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
                            failures.append(c.name + ": " + label + " moved " + String(fwd_moved) + " forward cells; a second-round sabotage of an arm with no second-round forward kernel must reach no forward buffer")
                    if bwd_must_hold and bs == FUSED_RAN:
                        print("    REACH " + label + " backward_cells_that_must_hold_moved=" + String(bwd_moved))
                        if bwd_moved > 0:
                            failures.append(c.name + ": " + label + " moved " + String(bwd_moved) + " backward cells; a second-round sabotage of an arm with no second-round backward kernel must reach no backward buffer")
                    # Attribution: which second-round flip moved (14.5).
                    if second and need_fwd and st == FUSED_RAN:
                        print(
                            "    REACH " + label + " amax_denom_moved=" + String(m_amax + m_den)
                            + " ctx_moved=" + String(m_ctx) + " ctx_moved_columns_16_up=" + String(m_ctx_hi)
                        )
                        if qres:
                            if m_amax + m_den == 0:
                                failures.append(c.name + ": " + label + ": the DEVIATION 2530 flip (every staged Q value) moved neither amax nor denom; the Q residency instantiation did not run")
                        else:
                            if m_amax + m_den > 0:
                                failures.append(c.name + ": " + label + " moved " + String(m_amax + m_den) + " amax/denom cells; the 2531 and 2533 forward flips reach ctx only")
                            if preflush and m_ctx_hi > 0:
                                failures.append(c.name + ": " + label + " moved " + String(m_ctx_hi) + " ctx cells at columns 16 and up; the 2533 forward flip reaches columns 0 to 15 only (the preflushed instantiation did not run)")
                    if second and need_bwd and not ztiled and bs == FUSED_RAN:
                        print("    REACH " + label + " zdot_moved=" + String(m_z) + " dv_moved=" + String(m_dv))
                        if m_z == 0 or m_dv > 0:
                            failures.append(c.name + ": " + label + ": the DEVIATION 2533 backward flip (the stored zdot) must move zdot and hold dv; zdot moved " + String(m_z) + ", dv moved " + String(m_dv))
                elif fwd_moved + bwd_moved > 0:
                    failures.append(c.name + ": " + label + " moved " + String(fwd_moved + bwd_moved) + " cells at head_dim " + String(c.hd) + ", where every arm runs the shipped kernels")
            # ---- dk/dv reach (DEVIATIONS 2596 and 2597, brief 16.4) ------
            if sab == 2:
                if c.hd == ATTN_STASH_HD:
                    if bs == FUSED_RAN:
                        print(
                            "    REACH " + label + " dk_moved=" + String(m_dk) + " dv_moved=" + String(m_dv)
                            + " zdot_moved=" + String(m_z) + " dq_moved=" + String(m_dq)
                            + " forward_moved=" + String(fwd_moved)
                        )
                        if c.expect_bwd == FUSED_RAN and (m_dk == 0 or m_dv == 0):
                            failures.append(c.name + ": DK/DV REACH NOT PROVEN for " + label + " (dk moved " + String(m_dk) + ", dv moved " + String(m_dv) + "; both must move)")
                        if m_z + m_dq > 0:
                            failures.append(c.name + ": " + label + " moved " + String(m_z + m_dq) + " zdot or dq cells; the DEVIATION 2596 / 2597 flips reach dk and dv only")
                    if st == FUSED_RAN and fwd_moved > 0:
                        failures.append(c.name + ": " + label + " moved " + String(fwd_moved) + " forward cells; the DEVIATION 2596 / 2597 flips reach dk and dv only")
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
        + " head_dim 64 with the second-round flips attributed, nothing moved"
        + " at other head dims"
    )
    _ = ctx^
