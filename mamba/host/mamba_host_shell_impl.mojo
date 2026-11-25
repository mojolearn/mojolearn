# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host Mamba-1, Mamba-2 and Mamba-3 entries behind `_mojolearn_mamba_host`.

The host-column runs that the Python boundary in
`bindings/_mojolearn_mamba_host.mojo` calls. Nothing here
touches Python: every entry takes buffer addresses, Ints and typed sessions.
"""

from std.memory import unsafe_memcpy
from std.sys.compile import is_defined
from std.os import abort

from core.hostptr import f32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.host.identical_gemm import GEMM_ORACLE_HOST_SABOTAGE
from mamba.checks.mamba_fixture import D_CONV, D_STATE, MambaDims, MambaWeights
from mamba.checks.mamba_oracle import MambaState, mamba_block_oracle
from mamba.checks.mamba2_fixture import (
    M2_CHUNK_SIZE,
    M2_D_CONV,
    M2_D_STATE,
    M2_HEADDIM,
    Mamba2Dims,
    Mamba2Weights,
)
from mamba.checks.mamba2_oracle import Mamba2State, mamba2_block_oracle
from mamba.checks.mamba3_fixture import (
    M3_CHUNK_SIZE,
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
    Mamba3Dims,
    Mamba3Weights,
)
from mamba.checks.mamba3_oracle import Mamba3State, mamba3_block_oracle
from mamba.host.gen.mamba2_prefill_chain import mamba2_prefill_backward
from mamba.host.gen.mamba3_prefill_chain import mamba3_prefill_backward
from mamba.host.gen.mamba1_prefill_chain import mamba1_prefill_backward
from mamba.host.device_shim import DeviceContext as M1HostContext
from mamba.host.gen.identity_trace import IdentityTrace as M1HostTrace
from mamba.host.gen.modeling_mamba import MambaDeviceState as M1HostState, MambaDeviceWeights as M1HostWeights, MambaDeviceStages as M1HostStages, mamba_upload as m1_host_upload, mamba_download as m1_host_download, mamba_block_forward as m1_host_forward


def _write(addr: Int, values: List[Float32], n: Int) raises:
    if len(values) != n:
        raise Error(
            String("mamba host: internal length mismatch, ")
            + String(len(values))
            + " values for a buffer of "
            + String(n)
        )
    if n > 0:
        unsafe_memcpy(dest=f32_ptr(addr), src=values.unsafe_ptr(), count=n)


# ===========================================================================
# Mamba-1
# ===========================================================================


def _m1_weights(a: List[Int], dm: Int) raises -> MambaWeights:
    var dims = MambaDims.of(dm)
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    var w = MambaWeights(dims)
    w.norm_w = read_f32(a[1], dm)
    w.w_in = read_f32(a[2], 2 * di * dm)
    w.conv_w = read_f32(a[3], di * D_CONV)
    w.conv_b = read_f32(a[4], di)
    w.w_x = read_f32(a[5], xr * di)
    w.w_dt = read_f32(a[6], di * r)
    w.b_dt = read_f32(a[7], di)
    w.a_log = read_f32(a[8], di * D_STATE)
    w.d_skip = read_f32(a[9], di)
    w.w_out = read_f32(a[10], dm * di)
    return w^


def _mamba1_run(a: List[Int], b: Int, l: Int, dm: Int, position: Int = 0) raises:
    if b <= 0 or l <= 0:
        raise Error("mamba1: B and L must be positive")
    var w = _m1_weights(a, dm)
    var di = w.dims.d_inner
    var state = MambaState(b, w.dims)
    state.conv_win = read_f32(a[11], b * di * D_CONV)
    state.h = read_f32(a[12], b * di * D_STATE)
    var st = mamba_block_oracle(w, read_f32(a[0], b * l * dm), b, l, state)
    _write(a[13], st.residual_out, b * l * dm)
    _write(a[11], state.conv_win, b * di * D_CONV)
    _write(a[12], state.h, b * di * D_STATE)


def _mamba1_backward_run(a: List[Int], b: Int, l: Int, dm: Int) raises:
    var w = _m1_weights(a, dm)
    var dims = w.dims.copy()
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    var gr = mamba1_prefill_backward(
        w, read_f32(a[0], b * l * dm), read_f32(a[11], b * l * dm), b, l
    )
    _write(a[12], gr.x, b * l * dm)
    _write(a[13], gr.norm_weight, dm)
    _write(a[14], gr.in_proj_weight, 2 * di * dm)
    _write(a[15], gr.conv1d_weight, di * D_CONV)
    _write(a[16], gr.conv1d_bias, di)
    _write(a[17], gr.x_proj_weight, xr * di)
    _write(a[18], gr.dt_proj_weight, di * r)
    _write(a[19], gr.dt_proj_bias, di)
    _write(a[20], gr.A_log, di * D_STATE)
    _write(a[21], gr.D, di)
    _write(a[22], gr.out_proj_weight, dm * di)


# ===========================================================================
# Mamba-2
# ===========================================================================


def _m2_weights(a: List[Int], dm: Int) raises -> Mamba2Weights:
    var dims = Mamba2Dims.of(dm)
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    var w = Mamba2Weights(dims)
    w.norm_w = read_f32(a[1], dm)
    w.w_in = read_f32(a[2], dip * dm)
    w.conv_w = read_f32(a[3], cd * M2_D_CONV)
    w.conv_b = read_f32(a[4], cd)
    w.dt_bias = read_f32(a[5], nh)
    w.a_log = read_f32(a[6], nh)
    w.d_skip = read_f32(a[7], nh)
    w.gnorm_w = read_f32(a[8], di)
    w.w_out = read_f32(a[9], dm * di)
    return w^


def _mamba2_run(a: List[Int], b: Int, l: Int, dm: Int, q0: Int, dt_lo: Float32, dt_hi: Float32) raises -> Int:
    var w = _m2_weights(a, dm)
    var dims = w.dims.copy()
    var cd = dims.conv_dim()
    var nh = dims.nheads
    if q0 < 0 or q0 >= M2_CHUNK_SIZE:
        raise Error(
            String("mamba2: buf_len must be in [0, ")
            + String(M2_CHUNK_SIZE)
            + "), got "
            + String(q0)
            + "; the open-chunk buffer holds at most CHUNK_SIZE - 1 rows"
            " between calls (contract section 5), so the two sides of this"
            " boundary disagree about the state"
        )
    if b <= 0 or l <= 0:
        raise Error("mamba2: B and L must be positive")
    var h_n = b * nh * M2_HEADDIM * M2_D_STATE
    var state = Mamba2State(b, dims)
    state.conv_win = read_f32(a[10], b * cd * M2_D_CONV)
    state.h = read_f32(a[11], h_n)
    state.buf_xbc = read_f32(a[12], b * M2_CHUNK_SIZE * cd)
    state.buf_dtraw = read_f32(a[13], b * M2_CHUNK_SIZE * nh)
    state.buf_len = q0
    var st = mamba2_block_oracle(w, read_f32(a[0], b * l * dm), b, l, dt_lo, dt_hi, state)
    _write(a[14], st.residual_out, b * l * dm)
    _write(a[15], st.h_last, h_n)
    _write(a[10], state.conv_win, b * cd * M2_D_CONV)
    _write(a[11], state.h, h_n)
    _write(a[12], state.buf_xbc, b * M2_CHUNK_SIZE * cd)
    _write(a[13], state.buf_dtraw, b * M2_CHUNK_SIZE * nh)
    return state.buf_len


def _mamba2_backward_run(a: List[Int], b: Int, l: Int, dm: Int, dt_lo: Float32, dt_hi: Float32) raises:
    var w = _m2_weights(a, dm)
    var dims = w.dims.copy()
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    var gr = mamba2_prefill_backward(
        w, read_f32(a[0], b * l * dm), read_f32(a[10], b * l * dm), b, l, dt_lo, dt_hi
    )
    _write(a[11], gr.x, b * l * dm)
    _write(a[12], gr.block_norm_weight, dm)
    _write(a[13], gr.in_proj_weight, dip * dm)
    _write(a[14], gr.conv1d_weight, cd * 4)
    _write(a[15], gr.conv1d_bias, cd)
    _write(a[16], gr.dt_bias, nh)
    _write(a[17], gr.A_log, nh)
    _write(a[18], gr.D, nh)
    _write(a[19], gr.norm_weight, di)
    _write(a[20], gr.out_proj_weight, dm * di)


# ===========================================================================
# Mamba-3
# ===========================================================================


def _m3_weights(a: List[Int], dm: Int) raises -> Mamba3Weights:
    var dims = Mamba3Dims.of(dm)
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    var w = Mamba3Weights(dims)
    w.norm_w = read_f32(a[1], dm)
    w.w_in = read_f32(a[2], dip * dm)
    w.dt_bias = read_f32(a[3], nh)
    w.bnorm_w = read_f32(a[4], M3_D_STATE)
    w.cnorm_w = read_f32(a[5], M3_D_STATE)
    w.b_bias = read_f32(a[6], nh * M3_D_STATE)
    w.c_bias = read_f32(a[7], nh * M3_D_STATE)
    w.d_skip = read_f32(a[8], nh)
    w.w_out = read_f32(a[9], dm * di)
    return w^


def _mamba3_run(a: List[Int], b: Int, l: Int, dm: Int, q0: Int, pend: Int, fresh: Bool) raises -> Int:
    var w = _m3_weights(a, dm)
    var dims = w.dims.copy()
    var nh = dims.nheads
    if q0 < 0 or q0 > M3_CHUNK_SIZE:
        raise Error(
            String("mamba3: buf_len must be in [0, ")
            + String(M3_CHUNK_SIZE)
            + "] (INCLUSIVE -- the buffer never empties, DEVIATION"
            " 832(i): r in [1, Q] after every call, 0 only before the"
            " first token), got "
            + String(q0)
            + "; the two sides of this boundary disagree about the state"
        )
    if pend != 0 and pend != 1:
        raise Error(
            "mamba3: pending must be 0 or 1, got "
            + String(pend)
            + "; the two sides of this boundary disagree about the state"
        )
    if b <= 0 or l <= 0:
        raise Error("mamba3: B and L must be positive")
    var theta_n = b * nh * M3_NUM_ROPE_ANGLES
    var h_n = b * nh * M3_HEADDIM * M3_D_STATE
    var qrow_n = b * M3_CHUNK_SIZE * nh
    var k_n = b * nh * M3_D_STATE
    var v_n = b * nh * M3_HEADDIM
    var state = Mamba3State(b, dims)
    if not fresh:
        state.buf_qrot = read_f32(a[12], qrow_n * M3_D_STATE)
        state.buf_krot = read_f32(a[13], qrow_n * M3_D_STATE)
        state.buf_v = read_f32(a[14], qrow_n * M3_HEADDIM)
        state.buf_dt = read_f32(a[15], qrow_n)
        state.buf_sig = read_f32(a[16], qrow_n)
        state.buf_adt = read_f32(a[17], qrow_n)
        state.buf_len = q0
        if pend == 1:
            state.set_input_states(
                read_f32(a[10], theta_n), read_f32(a[11], h_n),
                read_f32(a[18], k_n), read_f32(a[19], v_n),
            )
        else:
            state.theta = read_f32(a[10], theta_n)
            state.h = read_f32(a[11], h_n)
            state.pend_k = read_f32(a[18], k_n)
            state.pend_v = read_f32(a[19], v_n)
    var st = mamba3_block_oracle(w, read_f32(a[0], b * l * dm), b, l, state)
    _write(a[20], st.residual_out, b * l * dm)
    _write(a[21], st.h_last, h_n)
    _write(a[22], st.k_last, k_n)
    _write(a[23], st.v_last, v_n)
    _write(a[24], st.theta_last, theta_n)
    if not fresh:
        _write(a[10], state.theta, theta_n)
        _write(a[11], state.h, h_n)
        _write(a[12], state.buf_qrot, qrow_n * M3_D_STATE)
        _write(a[13], state.buf_krot, qrow_n * M3_D_STATE)
        _write(a[14], state.buf_v, qrow_n * M3_HEADDIM)
        _write(a[15], state.buf_dt, qrow_n)
        _write(a[16], state.buf_sig, qrow_n)
        _write(a[17], state.buf_adt, qrow_n)
        _write(a[18], state.pend_k, k_n)
        _write(a[19], state.pend_v, v_n)
    return state.buf_len


def _mamba3_backward_run(a: List[Int], b: Int, l: Int, dm: Int) raises:
    var w = _m3_weights(a, dm)
    var dims = w.dims.copy()
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    var gr = mamba3_prefill_backward(
        w, read_f32(a[0], b * l * dm), read_f32(a[10], b * l * dm), b, l
    )
    _write(a[11], gr.x, b * l * dm)
    _write(a[12], gr.block_norm_weight, dm)
    _write(a[13], gr.in_proj_weight, dip * dm)
    _write(a[14], gr.dt_bias, nh)
    _write(a[15], gr.B_norm_weight, M3_D_STATE)
    _write(a[16], gr.C_norm_weight, M3_D_STATE)
    _write(a[17], gr.B_bias, nh * M3_D_STATE)
    _write(a[18], gr.C_bias, nh * M3_D_STATE)
    _write(a[19], gr.D, nh)
    _write(a[20], gr.out_proj_weight, dm * di)


# TOMBSTONE: MOJOLEARN_NN40_OWNED_WEIGHTS (unmeasured opt-in arm: owned weight snapshots, host column) deleted 2026-11-15 by lane/expdel2-27-neural; code recoverable at 0e66cc9a52.
# Restore: git apply experiments/removed/MOJOLEARN_NN40_OWNED_WEIGHTS.patch
# TOMBSTONE: MOJOLEARN_NN40_OWNED_WEIGHTS (leftover: NN40_OWNED_WEIGHTS, M3OwnedHostSession and its install/forward/backward/export/state runs, _nn40_host_require) deleted 2026-11-25 by lane/owned-weights-deletion; code recoverable at 097a876282.
