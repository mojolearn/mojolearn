# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Scratch initialization must zero every field with one production host wait.

Build with -D MOJOLEARN_STEP_PHASE_TIMERS=1. This checks the actual executed
wait counter, not elapsed time on a shared machine. Also run with
-D MOJOLEARN_MAMBA_POISON=1 to exercise guard-band lifetimes.
"""
from std.sys.compile import is_defined
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from core.step_phase import step_counts_now
from core.device_scan import device_first_nonfinite
from transformer.impl.llama.modeling_llama import LlamaDims, LlamaDeviceStages
from mamba.impl.modeling.modeling_mamba import MambaDims, MambaDeviceStages, MAMBA_GUARD, mamba_download, mamba_upload


def check_zero(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int, name: String) raises:
    var values = mamba_download(ctx, buf, n)
    for i in range(n):
        if bitcast[DType.uint32](values[i]) != 0:
            raise Error(name + " is not positive zero at " + String(i))


def check_transfer(ctx: DeviceContext, n: Int) raises:
    var values = List[Float32](length=n, fill=Float32(0.0))
    var patterns: List[UInt32] = [0, 0x80000000, 1, 0x007FFFFF, 0x3F800000,
                                 0x7FC00123, 0x7F800000, 0xFF800000]
    for i in range(n):
        values[i] = bitcast[DType.float32](patterns[i % len(patterns)])
    var dev = mamba_upload(ctx, values)
    var back = mamba_download(ctx, dev, n)
    for i in range(n):
        if bitcast[DType.uint32](values[i]) != bitcast[DType.uint32](back[i]):
            raise Error("transfer changed bits at " + String(i))
    _ = dev^


def check_stages(ctx: DeviceContext, lean: Bool) raises:
    var before = step_counts_now()
    var t = LlamaDeviceStages(ctx, 2, 3, 7, LlamaDims(32, 2, 1, 16, 64), lean=lean)
    var after = step_counts_now()
    print("transformer initialization waits", after.syncs - before.syncs)
    if after.syncs - before.syncs != 1:
        raise Error("scratch initialization must take one host wait")
    check_zero(ctx, t.norm1_sumsq, len(t.norm1_sumsq), "transformer.norm1_sumsq")
    check_zero(ctx, t.norm1_out, len(t.norm1_out), "transformer.norm1_out")
    check_zero(ctx, t.q_proj, len(t.q_proj), "transformer.q_proj")
    check_zero(ctx, t.k_proj, len(t.k_proj), "transformer.k_proj")
    check_zero(ctx, t.v_proj, len(t.v_proj), "transformer.v_proj")
    check_zero(ctx, t.q_rope, len(t.q_rope), "transformer.q_rope")
    check_zero(ctx, t.k_rope, len(t.k_rope), "transformer.k_rope")
    check_zero(ctx, t.k_cache, len(t.k_cache), "transformer.k_cache")
    check_zero(ctx, t.v_cache, len(t.v_cache), "transformer.v_cache")
    check_zero(ctx, t.scores, len(t.scores), "transformer.scores")
    check_zero(ctx, t.masked, len(t.masked), "transformer.masked")
    check_zero(ctx, t.amax, len(t.amax), "transformer.amax")
    check_zero(ctx, t.aexp, len(t.aexp), "transformer.aexp")
    check_zero(ctx, t.denom, len(t.denom), "transformer.denom")
    check_zero(ctx, t.weights, len(t.weights), "transformer.weights")
    check_zero(ctx, t.ctxv, len(t.ctxv), "transformer.ctxv")
    check_zero(ctx, t.o_proj, len(t.o_proj), "transformer.o_proj")
    check_zero(ctx, t.residual1, len(t.residual1), "transformer.residual1")
    check_zero(ctx, t.norm2_sumsq, len(t.norm2_sumsq), "transformer.norm2_sumsq")
    check_zero(ctx, t.norm2_out, len(t.norm2_out), "transformer.norm2_out")
    check_zero(ctx, t.gate_proj, len(t.gate_proj), "transformer.gate_proj")
    check_zero(ctx, t.up_proj, len(t.up_proj), "transformer.up_proj")
    check_zero(ctx, t.silu_out, len(t.silu_out), "transformer.silu_out")
    check_zero(ctx, t.gated, len(t.gated), "transformer.gated")
    check_zero(ctx, t.down_proj, len(t.down_proj), "transformer.down_proj")
    check_zero(ctx, t.residual2, len(t.residual2), "transformer.residual2")
    check_zero(ctx, t.qbh, len(t.qbh), "transformer.qbh")
    check_zero(ctx, t.kbh, len(t.kbh), "transformer.kbh")
    check_zero(ctx, t.sbh, len(t.sbh), "transformer.sbh")
    before = step_counts_now()
    var m = MambaDeviceStages(ctx, 2, 3, MambaDims.of(32))
    after = step_counts_now()
    print("mamba initialization waits", after.syncs - before.syncs)
    var expected = 20 if MAMBA_GUARD > 0 else 1
    if after.syncs - before.syncs != expected:
        raise Error("mamba stage initialization wait budget exceeded")
    var b = 2
    var l = 3
    var dm = 32
    var di = 64
    var r = 2
    var xr = 34
    var tokens = b * l
    check_zero(ctx, m.norm_sumsq, tokens, "mamba1.norm_sumsq")
    check_zero(ctx, m.norm_out, tokens * dm, "mamba1.norm_out")
    check_zero(ctx, m.in_proj, tokens * 2 * di, "mamba1.in_proj")
    check_zero(ctx, m.a_out, di * 16, "mamba1.a_out")
    check_zero(ctx, m.conv_out, tokens * di, "mamba1.conv_out")
    check_zero(ctx, m.silu_out, tokens * di, "mamba1.silu_out")
    check_zero(ctx, m.conv_win, b * di * 4, "mamba1.conv_win")
    check_zero(ctx, m.x_proj, tokens * xr, "mamba1.x_proj")
    check_zero(ctx, m.dt_proj, tokens * di, "mamba1.dt_proj")
    check_zero(ctx, m.softplus_out, tokens * di, "mamba1.softplus_out")
    check_zero(ctx, m.scan_y, tokens * di, "mamba1.scan_y")
    check_zero(ctx, m.scan_h, b * di * 16, "mamba1.scan_h")
    check_zero(ctx, m.skip_out, tokens * di, "mamba1.skip_out")
    check_zero(ctx, m.gate_out, tokens * di, "mamba1.gate_out")
    check_zero(ctx, m.out_proj, tokens * dm, "mamba1.out_proj")
    check_zero(ctx, m.residual_out, tokens * dm, "mamba1.residual_out")
    check_zero(ctx, m.dt_low, tokens * r, "mamba1.dt_low")
    check_zero(ctx, m.b_mat, tokens * 16, "mamba1.b_mat")
    check_zero(ctx, m.c_mat, tokens * 16, "mamba1.c_mat")
    print("PASS all scratch fields are positive zero")
    _ = t^
    _ = m^


def main() raises:
    comptime assert is_defined["MOJOLEARN_STEP_PHASE_TIMERS"](), "enable the wait counter"
    var ctx = DeviceContext()
    var lengths: List[Int] = [0, 1, 7, 256, 513]
    for n in lengths:
        check_transfer(ctx, n)
    check_stages(ctx, False)
    check_stages(ctx, True)
    var finite_values: List[Float32] = [1.0, 2.0, 3.0]
    var finite = mamba_upload(ctx, finite_values)
    var before = step_counts_now()
    var first = device_first_nonfinite(ctx, finite, 3)
    var after = step_counts_now()
    print("nonfinite scan waits", after.syncs - before.syncs)
    if first != -1 or after.syncs - before.syncs != 1:
        raise Error("clean scan must return -1 with one host wait")
    print("PASS transfer bits and scan wait budget")
    _ = finite^
    _ = ctx^
