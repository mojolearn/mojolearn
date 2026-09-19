# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_mamba` family: the Mamba-1, Mamba-2 and
Mamba-3 blocks' forward, decode step and zero-state IDENTICAL prefill
backward on the host (lane/cpu-training-mamba, 2026-09-15).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. Every entry below
carries the GPU binding's name, address list and params list
(`bindings/_mojolearn_mamba.mojo`), so `Mamba1Block`, `Mamba2Block` and
`Mamba3Block` (`python/mojolearn/_mamba_impl.py`) run unchanged on a
CPU-only install through `_backend._HOST_MODULES`:

  `mamba1_forward`, `mamba1_decode_step`
        `mamba/checks/mamba_oracle.mojo::mamba_block_oracle`, the host
        Float32 oracle the device block is gated against bitwise.
  `mamba2_forward`, `mamba2_decode_step`
        `mamba/checks/mamba2_oracle.mojo::mamba2_block_oracle` (the three
        piece state, the dt clamp, the open-chunk buffer).
  `mamba3_forward`, `mamba3_forward_fresh`, `mamba3_decode_step`
        `mamba/checks/mamba3_oracle.mojo::mamba3_block_oracle`.
  `mamba1_backward`, `mamba2_backward`, `mamba3_backward`
        the device's own zero-state prefill VJPs
        (`mamba/impl/modeling/modeling_mamba_prefill_backward.mojo`,
        `mamba/impl/modules/mamba2_prefill_backward.mojo`,
        `mamba/impl/modules/mamba3_prefill_backward.mojo`), forward stages
        included, as `tools/mamba_host_gen.py` writes them out for the host
        under `mamba/host/gen/`: every kernel a serial loop over its launch
        grid, every buffer a host allocation and the device GEMM
        `gemm_oracle` (`mamba/host/device_shim.mojo`). The Mamba-1 host
        backward oracle is NOT used: on the M4 it differs from the device
        VJP in the low bits of eight of eleven gradients.

The sabotage arm is `gemm_oracle`'s (`-D MOJOLEARN_HOST_SABOTAGE=1` walks
every GEMM leaf descending), which every block's projections reach.
"""
from std.memory import memcpy
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, read_f32
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
from mamba.host.gen.mamba2_prefill_backward import mamba2_prefill_backward
from mamba.host.gen.mamba3_prefill_backward import mamba3_prefill_backward
from mamba.host.gen.modeling_mamba_prefill_backward import mamba1_prefill_backward


def _write(addr: Int, values: List[Float32], n: Int) raises:
    if len(values) != n:
        raise Error(
            String("mamba host: internal length mismatch, ")
            + String(len(values))
            + " values for a buffer of "
            + String(n)
        )
    if n > 0:
        memcpy(dest=f32_ptr(addr), src=values.unsafe_ptr(), count=n)


def _addrs(addrs: PythonObject, n: Int, what: String) raises -> List[Int]:
    if len(addrs) != n:
        raise Error(
            what + ": addrs must contain " + String(n) + " addresses, got "
            + String(len(addrs))
        )
    var a = List[Int]()
    for i in range(n):
        var address = Int(py=addrs[i])
        if address == 0:
            raise Error(what + ": null buffer address at slot " + String(i))
        a.append(address)
    return a^


# ===========================================================================
# THE READ-BACK
# ===========================================================================


def mamba_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "mamba host: this binding restates the IDENTICAL arms only;"
        " bindings/build_host_family.sh passes -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def mamba_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def mamba_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "mamba host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_mamba_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def mamba_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every GEMM leaf descending on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(GEMM_ORACLE_HOST_SABOTAGE)


def mamba_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def mamba_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


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


def _mamba1_run(a: List[Int], b: Int, l: Int, dm: Int) raises:
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


def mamba1_forward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`bindings/_mojolearn_mamba.mojo::mamba1_forward_binding`'s contract:
    14 addresses, params B, L, d_model."""
    var a = _addrs(addrs, 14, String("mamba1_forward"))
    if len(params) != 3:
        raise Error("mamba1_forward: params must contain 3 values (B, L, d_model), got " + String(len(params)))
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    with GILReleased(Python()):
        _mamba1_run(a, b, l, dm)
    return PythonObject(0)


def mamba1_decode_step_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """The forward at L = 1 with the state carried; params B, d_model."""
    var a = _addrs(addrs, 14, String("mamba1_decode_step"))
    if len(params) != 2:
        raise Error("mamba1_decode_step: params must contain 2 values (B, d_model), got " + String(len(params)))
    var b = Int(py=params[0])
    var dm = Int(py=params[1])
    with GILReleased(Python()):
        _mamba1_run(a, b, 1, dm)
    return PythonObject(0)


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


def mamba1_backward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """Zero-state IDENTICAL VJP: 23 addresses, params B, L, d_model."""
    if len(addrs) != 23 or len(params) != 3:
        raise Error("mamba1 backward: expected 23 addresses and B, L, d_model")
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    if b <= 0 or l <= 0 or dm <= 0:
        raise Error("mamba1 backward: B, L and d_model must be positive")
    var a = _addrs(addrs, 23, String("mamba1 backward"))
    with GILReleased(Python()):
        _mamba1_backward_run(a, b, l, dm)
    return PythonObject(0)


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


def mamba2_forward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """16 addresses; params B, L, d_model, buf_len, dt_lo, dt_hi. Returns
    the post-call buf_len."""
    var a = _addrs(addrs, 16, String("mamba2_forward"))
    if len(params) != 6:
        raise Error(
            "mamba2_forward: params must contain 6 values (B, L, d_model,"
            " buf_len, dt_lo, dt_hi), got " + String(len(params))
        )
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    var q0 = Int(py=params[3])
    var dt_lo = Float32(Float64(py=params[4]))
    var dt_hi = Float32(Float64(py=params[5]))
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba2_run(a, b, l, dm, q0, dt_lo, dt_hi)
    return PythonObject(out_len)


def mamba2_decode_step_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """The forward at L = 1; params B, d_model, buf_len, dt_lo, dt_hi."""
    var a = _addrs(addrs, 16, String("mamba2_decode_step"))
    if len(params) != 5:
        raise Error(
            "mamba2_decode_step: params must contain 5 values (B, d_model,"
            " buf_len, dt_lo, dt_hi), got " + String(len(params))
        )
    var b = Int(py=params[0])
    var dm = Int(py=params[1])
    var q0 = Int(py=params[2])
    var dt_lo = Float32(Float64(py=params[3]))
    var dt_hi = Float32(Float64(py=params[4]))
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba2_run(a, b, 1, dm, q0, dt_lo, dt_hi)
    return PythonObject(out_len)


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


def mamba2_backward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """21 addresses; params B, L, d_model, dt_lo, dt_hi."""
    if len(addrs) != 21 or len(params) != 5:
        raise Error("mamba2 backward: expected 21 addresses and 5 scalars")
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    if b <= 0 or l <= 0 or dm <= 0:
        raise Error("mamba2 backward: B, L and d_model must be positive")
    var a = _addrs(addrs, 21, String("mamba2 backward"))
    var dt_lo = Float32(Float64(py=params[3]))
    var dt_hi = Float32(Float64(py=params[4]))
    with GILReleased(Python()):
        _mamba2_backward_run(a, b, l, dm, dt_lo, dt_hi)
    return PythonObject(0)


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


def mamba3_forward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """25 addresses; params B, L, d_model, buf_len, pending."""
    var a = _addrs(addrs, 25, String("mamba3_forward"))
    if len(params) != 5:
        raise Error(
            "mamba3_forward: params must contain 5 values (B, L, d_model,"
            " buf_len, pending), got " + String(len(params))
        )
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    var q0 = Int(py=params[3])
    var pend = Int(py=params[4])
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba3_run(a, b, l, dm, q0, pend, False)
    return PythonObject(out_len)


def mamba3_forward_fresh_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """15 addresses (x, nine weights, y, four reports); params B, L,
    d_model. A zero state whose resumption pieces are discarded."""
    if len(addrs) != 15 or len(params) != 3:
        raise Error("mamba3_forward_fresh: expected 15 addresses and 3 parameters (B, L, d_model)")
    var a = List[Int]()
    for i in range(10):
        a.append(Int(py=addrs[i]))
    for _ in range(10):
        a.append(0)
    for i in range(10, 15):
        a.append(Int(py=addrs[i]))
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    if b < 1 or l < 1:
        raise Error("mamba3_forward_fresh: B and L must be positive")
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba3_run(a, b, l, dm, 0, 0, True)
    return PythonObject(out_len)


def mamba3_decode_step_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """The forward at L = 1; params B, d_model, buf_len, pending."""
    var a = _addrs(addrs, 25, String("mamba3_decode_step"))
    if len(params) != 4:
        raise Error(
            "mamba3_decode_step: params must contain 4 values (B,"
            " d_model, buf_len, pending), got " + String(len(params))
        )
    var b = Int(py=params[0])
    var dm = Int(py=params[1])
    var q0 = Int(py=params[2])
    var pend = Int(py=params[3])
    var out_len = 0
    with GILReleased(Python()):
        out_len = _mamba3_run(a, b, 1, dm, q0, pend, False)
    return PythonObject(out_len)


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


def mamba3_backward_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """21 addresses; params B, L, d_model."""
    if len(addrs) != 21 or len(params) != 3:
        raise Error("mamba3 backward: expected 21 addresses and 3 scalars")
    var b = Int(py=params[0])
    var l = Int(py=params[1])
    var dm = Int(py=params[2])
    if b <= 0 or l <= 0 or dm <= 0:
        raise Error("mamba3 backward: B, L and d_model must be positive")
    var a = _addrs(addrs, 21, String("mamba3 backward"))
    with GILReleased(Python()):
        _mamba3_backward_run(a, b, l, dm)
    return PythonObject(0)


@export
def PyInit__mojolearn_mamba_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_mamba_host")
        m.def_function[mamba_host_numeric_mode_binding]("mamba_host_numeric_mode")
        m.def_function[mamba_host_vendor_binding]("mamba_host_vendor")
        m.def_function[mamba_host_column_binding]("mamba_host_column")
        m.def_function[mamba_host_sabotage_binding]("mamba_host_sabotage")
        m.def_function[mamba_vendor_binding]("mamba_vendor")
        m.def_function[mamba_numeric_mode_binding]("mamba_numeric_mode")
        m.def_function[mamba1_forward_binding]("mamba1_forward")
        m.def_function[mamba1_backward_binding]("mamba1_backward")
        m.def_function[mamba1_decode_step_binding]("mamba1_decode_step")
        m.def_function[mamba2_forward_binding]("mamba2_forward")
        m.def_function[mamba2_decode_step_binding]("mamba2_decode_step")
        m.def_function[mamba2_backward_binding]("mamba2_backward")
        m.def_function[mamba3_forward_binding]("mamba3_forward")
        m.def_function[mamba3_forward_fresh_binding]("mamba3_forward_fresh")
        m.def_function[mamba3_decode_step_binding]("mamba3_decode_step")
        m.def_function[mamba3_backward_binding]("mamba3_backward")
        return m.finalize()
    except error:
        abort(String("failed to create _mojolearn_mamba_host: ", error))
