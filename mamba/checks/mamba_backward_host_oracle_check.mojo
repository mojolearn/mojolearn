# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HOST CHECK: the Mamba-1 host backward oracle
(`mamba/checks/mamba_backward_oracle.mojo`) against the device prefill VJP
(`mamba/impl/modeling/modeling_mamba_prefill_backward.mojo`), stage by
stage, on all 16 corpus cases, with the signed-dyadic objective
`mamba_check.mojo::objective_cotangent` spells.

The device pass runs on the host as `tools/mamba_host_gen.py` writes it
(`mamba/host/gen/`), the same pass the mamba host binding serves and the
one that reads IDENTICAL x4 against the 166-lane record's Apple, NVIDIA and
AMD columns (bench/results/identity_break/2026-09-15_cpu-mamba). No GPU.

Why it exists (2026-09-15). The oracle had been checked against a float64
tolerance only, never bitwise against the device, and it was not the device's
function: three of its free choices were the plan's REJECTED readings.
B7 fused the middle of silu' (three roundings; the plan's DEVIATION 1085 and
the device say four), B18 folded ddelta as two chains joined by an add (the
plan's DEVIATION 1083 and the device fold ONE chain, B term then A term per
n; the old reading is the device's SAB_BWD_DDELTA_TWO_FOLDS arm), and T1
folded a stored +0.0 seed where DEVIATION 1082 makes the seed an omitted
operation (the device's SAB_BWD_T1_SEED_ADD arm; it turned -0.0 contributions
into +0.0 on adv_gate_saturation). Before the fix this check read 151 differing
tensors over the 16 cases (bwd.dz and bwd.ddelta first); after it, none.

Exit 1 on any difference.

    pixi run check-mamba1-backward-oracle-host

or, built once:

    pixi run mojo build -j 1 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU \
        -I . -I bindings mamba/checks/mamba_backward_host_oracle_check.mojo -o m1_bwd_oracle_check
    ./m1_bwd_oracle_check
"""
from std.memory import bitcast
from std.sys import exit
from mamba.host.device_shim import DeviceContext, DeviceBuffer
from mamba.host.gen.identity_trace import IdentityTrace
from mamba.host.gen.mamba_backward_checks import (
    PROJ_OUT, PROJ_DT, PROJ_IN, PROJ_X, RED_CONV_BIAS, RED_CONV_W_TAP0, RED_D,
    RED_DT_BIAS, RED_NORM_W, mamba_backward_proj_a_into, mamba_backward_proj_b_into,
    mamba_backward_reduce_into, mamba_backward_workspace_max_floats,
)
from mamba.checks.mamba_fixture import D_CONV, D_STATE, MambaDims, MambaWeights, corpus_case, corpus_case_weights, corpus_case_x, CORPUS_CASE_COUNT
from mamba.host.gen.modeling_mamba import (
    MambaDeviceStages, MambaDeviceState, MambaDeviceWeights, mamba_block_forward,
    mamba_download, mamba_upload, mamba_zeros,
)
from mamba.host.gen.modeling_mamba_backward import (
    mamba_bwd_concat_xp_into, mamba_bwd_concat_p_into, mamba_bwd_ddtp_into,
    mamba_bwd_dhin_into, mamba_bwd_du_join_into, mamba_bwd_gate_into, mamba_bwd_norm_into,
)
from mamba.host.gen.selective_scan_backward import (
    bwd_da_partial_floats, bwd_dh_floats, bwd_h_checkpoint_floats, mamba_bwd_da_into,
    mamba_bwd_da_log_into, mamba_bwd_dbc_into, selective_scan_bwd_scan_into,
    selective_scan_checkpoint_fn,
)
from mamba.host.gen.modeling_mamba_prefill_backward import mamba1_prefill_backward
from mamba.checks.mamba_oracle import MambaState, mamba_block_oracle
from mamba.checks.mamba_backward_oracle import mamba_block_backward_oracle


def cot(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var numer = (i * 37 + 11) % 31 - 15
        if numer == 0:
            numer = 1
        out.append(Float32(numer) / Float32(16.0))
    return out^


def cmp(name: String, host: List[Float32], dev: List[Float32]) -> Int:
    if len(host) != len(dev):
        print("  " + name + ": LENGTH host " + String(len(host)) + " device " + String(len(dev)))
        return 1
    var n = 0
    var first = -1
    for i in range(len(host)):
        if bitcast[DType.uint32](host[i]) != bitcast[DType.uint32](dev[i]):
            n += 1
            if first < 0:
                first = i
    if n == 0:
        print("  " + name + ": same (" + String(len(host)) + ")")
        return 0
    print("  " + name + ": DIFF " + String(n) + "/" + String(len(host)) + " first " + String(first)
          + " oracle " + String(host[first]) + " device " + String(dev[first]))
    return 1


def main() raises:
    var bad = 0
    for k in range(CORPUS_CASE_COUNT):
        var c = corpus_case(k)
        var dims = MambaDims.of(c.d_model)
        var w = corpus_case_weights(k)
        var input = corpus_case_x(k)
        var b = c.b
        var l = c.l
        var m = b * l
        var di = dims.d_inner
        var grad_output = cot(m * dims.d_model)
        print("case " + String(k) + " " + String(c.name) + " B=" + String(b) + " L=" + String(l) + " dm=" + String(c.d_model))
        # ---- oracle
        var s_in = MambaState(b, dims)
        var s_fw = MambaState(b, dims)
        var st = mamba_block_oracle(w, input, b, l, s_fw)
        var bst = mamba_block_backward_oracle(w, input, grad_output, st, s_in, b, l)
        # ---- device pass on the host (the generated driver's order)
        var ctx = DeviceContext()
        var dweights = MambaDeviceWeights(ctx, w)
        var state_in = MambaDeviceState(ctx, b, dims)
        var state = MambaDeviceState(ctx, b, dims)
        var stages = MambaDeviceStages(ctx, b, l, dims)
        var x = mamba_upload(ctx, input)
        var trace = IdentityTrace.disabled()
        mamba_block_forward(ctx, stages, state, dweights, x, b, l, trace, String("diag"))
        bad += cmp("fwd.gate_out", st.gate_out, mamba_download(ctx, stages.gate_out, m * di))
        bad += cmp("fwd.skip_out", st.skip_out, mamba_download(ctx, stages.skip_out, m * di))
        bad += cmp("fwd.softplus", st.softplus_out, mamba_download(ctx, stages.softplus_out, m * di))
        bad += cmp("fwd.x_proj", st.x_proj, mamba_download(ctx, stages.x_proj, m * dims.x_proj_rows()))
        var dres = mamba_upload(ctx, grad_output)
        var workspace = mamba_zeros(ctx, mamba_backward_workspace_max_floats(dims, m))
        var dg = mamba_zeros(ctx, m * di)
        var dw_out = mamba_zeros(ctx, dims.d_model * di)
        mamba_backward_proj_a_into(ctx, dg, dres, dweights.w_out, workspace, PROJ_OUT, dims, m)
        mamba_backward_proj_b_into(ctx, dw_out, dres, stages.gate_out, workspace, PROJ_OUT, dims, m)
        bad += cmp("bwd.dg", bst.dg, mamba_download(ctx, dg, m * di))
        bad += cmp("bwd.dW_out", bst.dw_out, mamba_download(ctx, dw_out, dims.d_model * di))
        var dsk = mamba_zeros(ctx, m * di)
        var dz = mamba_zeros(ctx, m * di)
        var du_d = mamba_zeros(ctx, m * di)
        var product_d = mamba_zeros(ctx, m * di)
        mamba_bwd_gate_into(ctx, dsk, dz, du_d, product_d, dg, stages.skip_out, stages.silu_out, stages.in_proj, dweights.d_skip, m, di)
        bad += cmp("bwd.dsk", bst.dsk, mamba_download(ctx, dsk, m * di))
        bad += cmp("bwd.dz", bst.dz, mamba_download(ctx, dz, m * di))
        var h_ckpt = mamba_zeros(ctx, bwd_h_checkpoint_floats(b, l, di))
        selective_scan_checkpoint_fn(ctx, h_ckpt, state_in.h, stages.silu_out, stages.softplus_out, stages.a_out, stages.b_mat, b, l, di)
        bad += cmp("bwd.h_ckpt", bst.h_ckpt, mamba_download(ctx, h_ckpt, bwd_h_checkpoint_floats(b, l, di)))
        var dh = mamba_zeros(ctx, bwd_dh_floats(b, l, di))
        var du_s = mamba_zeros(ctx, m * di)
        var ddelta = mamba_zeros(ctx, m * di)
        var scan_w = mamba_zeros(ctx, m * di)
        selective_scan_bwd_scan_into(ctx, dh, du_s, ddelta, scan_w, dsk, stages.c_mat, stages.b_mat, stages.softplus_out, stages.a_out, stages.silu_out, h_ckpt, b, l, di)
        bad += cmp("bwd.dh", bst.dh, mamba_download(ctx, dh, bwd_dh_floats(b, l, di)))
        bad += cmp("bwd.du_s", bst.du_s, mamba_download(ctx, du_s, m * di))
        bad += cmp("bwd.ddelta", bst.ddelta, mamba_download(ctx, ddelta, m * di))
        var dcm = mamba_zeros(ctx, m * D_STATE)
        var dbm = mamba_zeros(ctx, m * D_STATE)
        mamba_bwd_dbc_into(ctx, dcm, dbm, dsk, h_ckpt, scan_w, dh, b, l, di)
        bad += cmp("bwd.dCm", bst.dcm, mamba_download(ctx, dcm, m * D_STATE))
        bad += cmp("bwd.dBm", bst.dbm, mamba_download(ctx, dbm, m * D_STATE))
        var da = mamba_zeros(ctx, di * D_STATE)
        var da_partial = mamba_zeros(ctx, bwd_da_partial_floats(b, di))
        mamba_bwd_da_into(ctx, da, da_partial, dh, stages.softplus_out, stages.a_out, stages.silu_out, stages.b_mat, h_ckpt, b, l, di)
        var da_log = mamba_zeros(ctx, di * D_STATE)
        mamba_bwd_da_log_into(ctx, da_log, da, stages.a_out, di)
        bad += cmp("bwd.dA_log", bst.da_log, mamba_download(ctx, da_log, di * D_STATE))
        var ddtp = mamba_zeros(ctx, m * di)
        mamba_bwd_ddtp_into(ctx, ddtp, ddelta, stages.dt_proj, dweights.b_dt, m, di)
        bad += cmp("bwd.ddtp", bst.ddtp, mamba_download(ctx, ddtp, m * di))
        var gr = mamba1_prefill_backward(w, input, grad_output, b, l)
        bad += cmp("pub.x", bst.dx, gr.x)
        bad += cmp("pub.norm_weight", bst.dw_norm, gr.norm_weight)
        bad += cmp("pub.in_proj", bst.dw_in, gr.in_proj_weight)
        bad += cmp("pub.conv1d_weight", bst.dcw, gr.conv1d_weight)
        bad += cmp("pub.conv1d_bias", bst.dcb, gr.conv1d_bias)
        bad += cmp("pub.x_proj", bst.dw_x, gr.x_proj_weight)
        bad += cmp("pub.dt_proj_weight", bst.dw_dt, gr.dt_proj_weight)
        bad += cmp("pub.dt_proj_bias", bst.db_dt, gr.dt_proj_bias)
        bad += cmp("pub.A_log", bst.da_log, gr.A_log)
        bad += cmp("pub.D", bst.dd_skip, gr.D)
        bad += cmp("pub.out_proj", bst.dw_out, gr.out_proj_weight)
        bad += cmp("pub.dB", bst.dbm, gr.stage_dB)
        bad += cmp("pub.dC", bst.dcm, gr.stage_dC)
        _ = dres^
        _ = workspace^
        _ = x^
    if bad != 0:
        print("FAIL: " + String(bad) + " tensors differ between the host backward oracle and the device VJP")
        exit(1)
    print("PASS: the host backward oracle equals the device VJP on every compared tensor of the 16 corpus cases")
