# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fused attention ARMS (DEVIATIONS 2525 to 2527) against the eager
stage kernels, BIT FOR BIT, on the fused check's cases, plus reach by
sabotage. Additive beside `transformer_fused_check.mojo`, which gates the
shipped kernels; this file gates the opt-in arms on the same cases.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_ATTN_ARM_TRIAL=1 -I . \\
        transformer/checks/transformer_attention_arms_check.mojo

WHAT IT ASSERTS, for every case of `transformer_fused_check.cases()` and
every arm (bwd_stash, fwd_sstash, bwd_stash_tiled):
  * the arm's launcher reports the status the case expects (RAN, the
    regime refusal, the corner), exactly as the shipped kernels must;
  * on RAN, ctx, amax, denom (forward) and zdot, dq, dk, dv (backward)
    equal the eager kernels' bits;
  * REACH, on a head-dim-64 case that RAN: the arm's sabotage instantiation
    moves at least one cell of the outputs that arm produces (the forward
    arm moves denom or ctx; a backward arm moves dq, dk or dv), and the
    clean arm restores the eager bits. At any other head dim the arms
    take the shipped kernels by design and no reach is asserted.
Without -D MOJOLEARN_ATTN_ARM_TRIAL=1 every arm runs the shipped kernels;
the equality section then passes trivially and the reach section FAILS,
naming the missing define, so a green run is always a run of the arms.

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
    ATTN_ARM_FWD_SSTASH,
    ATTN_ARM_SABOTAGE,
    ATTN_ARM_TRIAL,
    ATTN_STASH_HD,
    FUSED_RAN,
    fused_attention_arm_name,
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
    var out = List[Int]()
    out.append(ATTN_ARM_BWD_STASH)
    out.append(ATTN_ARM_FWD_SSTASH)
    out.append(ATTN_ARM_BWD_STASH | ATTN_ARM_BWD_TILED)
    return out^


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
        var is_fwd_arm = (arm & ATTN_ARM_FWD_SSTASH) != 0
        var is_bwd_arm = (arm & ATTN_ARM_BWD_STASH) != 0
        for sab in range(2):
            var this_arm = arm
            var label = String(name)
            if sab == 1:
                this_arm = arm | ATTN_ARM_SABOTAGE
                label = name + "+sabotage"
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
                bwd_moved += compare(c.name, label + " bwd dq", e_dq, _download(ctx, dq, qn))
                bwd_moved += compare(c.name, label + " bwd dk", e_dk, _download(ctx, dk, kn))
                bwd_moved += compare(c.name, label + " bwd dv", e_dv, _download(ctx, dv, kn))
            # ---- reach: only where the arm is wired and the case RAN -----
            if sab == 1 and c.hd == ATTN_STASH_HD:
                var moved = 0
                var applies = False
                if is_fwd_arm and c.expect_fwd == FUSED_RAN:
                    applies = True
                    moved += fwd_moved
                if is_bwd_arm and c.expect_bwd == FUSED_RAN:
                    applies = True
                    moved += bwd_moved
                if applies:
                    print("    REACH " + name + " sabotage_flipped_cells=" + String(moved))
                    if moved == 0:
                        failures.append(c.name + ": REACH NOT PROVEN for " + name + " (sabotage moved nothing; build lacks -D MOJOLEARN_ATTN_ARM_TRIAL=1 or the arm is not wired)")
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
        print("NOTE: no -D MOJOLEARN_ATTN_ARM_TRIAL=1: every arm runs the shipped kernels; reach will FAIL")
    var ctx = DeviceContext()
    var all_cases = cases()
    var failures = List[String]()
    var n = 0
    for i in range(len(all_cases)):
        var c = all_cases[i].copy()
        run_case(ctx, c, failures)
        n += 1
    if len(failures) > 0:
        for i in range(len(failures)):
            print("FAIL " + failures[i])
        raise Error(
            "transformer_attention_arms_check: " + String(len(failures))
            + " failure(s) over " + String(n) + " cases; see the FAIL lines"
        )
    print(
        "transformer_attention_arms_check: PASS, " + String(n)
        + " cases x 3 arms, every RAN buffer bit-identical to eager, every"
        + " status as expected, reach proven at head_dim 64"
    )
    _ = ctx^
