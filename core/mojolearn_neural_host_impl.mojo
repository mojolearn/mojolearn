# SPDX-License-Identifier: Apache-2.0
"""Native CPU inference owners shared by the public binding and descriptor provider."""
from std.math import isfinite
from core.hostptr import copy_f32, f32_ptr, read_f32, read_i32
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
from training.host.mlp_oracle import host_mlp_bias_activation
from training.host.samba_ops_oracle import (
    host_samba_embedding_forward,
    host_samba_linear_forward,
    host_samba_rms_norm_forward,
)
from transformer.host.transformer_block_host import (
    transformer_host_forward,
    transformer_host_weights,
    transformer_host_weights_opts,
)
from transformer.checks.transformer_fixture import TransformerWeights
from transformer.block_options import (
    BLOCK_OPTION_ADDRS,
    BLOCK_OPTION_PARAMS,
    BlockOptions,
)


def _finite(values: List[Float32], what: String) raises:
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error(what + " is not finite")


def _opt_read(addr: Int, n: Int, on: Bool, name: String, what: String) raises -> List[Float32]:
    if on:
        if addr == 0:
            raise Error(what + ": " + name + " is required by its option and its address is null")
        return read_f32(addr, n)
    if addr != 0:
        raise Error(what + ": " + name + " was passed but its option is off; pass the option or drop the tensor")
    return List[Float32]()


def _transformer_weights_from(
    a: List[Int], dm: Int, nh: Int, nkv: Int, hd: Int, it: Int,
    rope_positions: Int, opts: BlockOptions, what: String,
) raises -> TransformerWeights:
    """`_mojolearn_transformer_host.mojo::_host_weights_from`, word for
    word: the oracle's weights from the 13 + 11 address layout."""
    var qw = nh * hd
    var kw = nkv * hd
    var w_gate = List[Float32]()
    if opts.gated():
        if a[7] == 0:
            raise Error(what + ": gate_proj.weight address is null (a gated MLP needs it)")
        w_gate = read_f32(a[7], it * dm)
    elif a[7] != 0:
        raise Error(what + ": gate_proj.weight was passed but the MLP is ungated (mlp gelu / gelu_tanh)")
    return transformer_host_weights_opts(
        dm, nh, nkv, hd, it, rope_positions, opts,
        read_f32(a[1], dm), read_f32(a[2], dm), read_f32(a[3], qw * dm),
        read_f32(a[4], kw * dm), read_f32(a[5], kw * dm), read_f32(a[6], dm * qw),
        w_gate^, read_f32(a[8], it * dm), read_f32(a[9], dm * it),
        _opt_read(a[13], qw, opts.qkv_bias, "q_proj.bias", what),
        _opt_read(a[14], kw, opts.qkv_bias, "k_proj.bias", what),
        _opt_read(a[15], kw, opts.qkv_bias, "v_proj.bias", what),
        _opt_read(a[16], dm, opts.o_bias, "o_proj.bias", what),
        _opt_read(a[17], dm, opts.norm_bias, "input_layernorm.bias", what),
        _opt_read(a[18], dm, opts.norm_bias, "post_attention_layernorm.bias", what),
        _opt_read(a[19], it, opts.mlp_bias, "up_proj.bias", what),
        _opt_read(a[20], dm, opts.mlp_bias, "down_proj.bias", what),
        _opt_read(a[21], it, opts.has_gate_bias(), "gate_proj.bias", what),
        _opt_read(a[22], hd, opts.qk_norm, "q_norm.weight", what),
        _opt_read(a[23], hd, opts.qk_norm, "k_norm.weight", what),
    )


def _write(addr: Int, values: List[Float32], n: Int) raises:
    """`_mojolearn_mamba_host`'s `_write`: a length-checked copy out, so a
    state piece that came back the wrong size is an error by name rather
    than a buffer overrun."""
    if len(values) != n:
        raise Error(
            String("neural host: internal length mismatch, ")
            + String(len(values))
            + " values for a buffer of "
            + String(n)
        )
    if n > 0:
        copy_f32(values.unsafe_ptr(), f32_ptr(addr), n)


def _run_forward(
    a: List[Int], b: Int, l: Int, dm: Int, nh: Int, nkv: Int, hd: Int,
    it: Int, smax: Int, s0: Int, window: Int, opts: BlockOptions,
    int15: Bool = False,
) raises -> Int:
    """`a` is the device binding's thirteen: x, the nine weights, k_cache,
    v_cache, y_out, then the eleven optional addresses (zeros when
    absent). The caches are read at entry and written back."""
    if b <= 0 or l <= 0:
        raise Error("transformer: B and L must be positive")
    if smax <= 0:
        raise Error("transformer: max_tokens must be positive")
    if window < 0:
        raise Error("transformer: window must be >= 0 (0 = full causal)")
    if dm <= 0 or nh <= 0 or nkv <= 0 or hd <= 0 or it <= 0:
        raise Error("transformer: d_model, n_heads, n_kv_heads, head_dim and intermediate must be positive")
    var qw = nh * hd
    var kw = nkv * hd
    var w = _transformer_weights_from(a, dm, nh, nkv, hd, it, smax, opts, String("transformer"))
    var cap = smax
    if window > 0:
        cap = window
    var cache_n = b * nkv * cap * hd
    var k_in = read_f32(a[10], cache_n)
    var v_in = read_f32(a[11], cache_n)
    var x = read_f32(a[0], b * l * dm)
    var out = transformer_host_forward(w, x, b, l, smax, s0, window, k_in, v_in, int15=int15)
    _write(a[12], out.y, b * l * dm)
    _write(a[10], out.k_cache, cache_n)
    _write(a[11], out.v_cache, cache_n)
    return out.cached_tokens


def _m1_weights(a: List[Int], dm: Int) raises -> MambaWeights:
    """`_mojolearn_mamba_host::_m1_weights`: the ten `Mamba1Block` weights
    from a[1..10], the device binding's order."""
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
    """14 addresses: x, the ten weights, conv_window, h, y_out. The two
    state pieces are read at entry and written back."""
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


def _m2_weights(a: List[Int], dm: Int) raises -> Mamba2Weights:
    """`_mojolearn_mamba_host::_m2_weights`: the nine `Mamba2Block` weights
    from a[1..9]."""
    var dims = Mamba2Dims.of(dm)
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var nh = dims.nheads
    var dip = dims.d_in_proj()
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


def _mamba2_run(
    a: List[Int], b: Int, l: Int, dm: Int, q0: Int, dt_lo: Float32, dt_hi: Float32
) raises -> Int:
    """16 addresses: x, the nine weights, conv_window, h, buffer_xbc,
    buffer_dtraw, y_out, h_last_out. Returns the post-call buf_len."""
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


def _m3_weights(a: List[Int], dm: Int) raises -> Mamba3Weights:
    """`_mojolearn_mamba_host::_m3_weights`: the nine `Mamba3Block` weights
    from a[1..9]."""
    var dims = Mamba3Dims.of(dm)
    var di = dims.d_inner
    var nh = dims.nheads
    var dip = dims.d_in_proj()
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
    """25 addresses: x, the nine weights, the ten state pieces, y_out and
    the four reports. `fresh` runs a zero state and writes no state back
    (the a[10..19] slots are then 0 and never touched)."""
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
