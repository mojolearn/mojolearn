# SPDX-License-Identifier: Apache-2.0
"""Shared buffers for the Mamba-3 output-coverage suites (lane
fix-mamba3-unfilled-rest, 2026-11-13).

The public doors allocate the Mamba-3 outputs (y and the four reports
h_last, k_last, v_last, theta_last) UNFILLED (`generation_array_unfilled`)
on the forward, the carried-state forward and step, the resident session's
step and decode and the tape prefill; `-D MOJOLEARN_MAMBA3_FORWARD_ZERO_FILL`
keeps them zero-filled. That is only sound when the native call writes every
output byte. The suites run each native route twice on the same seeded
corpus inputs (mamba/checks/mamba3_fixture.mojo, case 1: B 2, L 4, d_model
32) and the same zero start state: once into outputs prefilled with 0xFF
words (a NaN pattern) and once into zero-filled outputs. Every output and
the written-back state must agree bit for bit; a byte the call leaves
unwritten differs between the two passes.
"""
from std.memory import bitcast
from mamba.checks.mamba3_fixture import (
    Mamba3Dims, M3_D_STATE, M3_HEADDIM, M3_CHUNK_SIZE, M3_NUM_ROPE_ANGLES, m3_corpus_case, m3_case_weights,
    m3_case_x,
)

comptime M3_COV_CASE = 1
"""The corpus case (B 2, L 4, d_model 32): small, two rows, one head."""
comptime M3_COV_POISON: UInt32 = 0xFFFFFFFF
comptime M3_COV_ZERO: UInt32 = 0


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def m3_cov_filled(n: Int, word: UInt32) -> List[Float32]:
    """`n` floats whose bits are all `word`."""
    return List[Float32](length=n, fill=bitcast[DType.float32](word))


def m3_cov_weights() raises -> List[List[Float32]]:
    """The case's nine weights in the per-call ABI order (norm, in_proj,
    dt_bias, B_norm, C_norm, B_bias, C_bias, D, out_proj)."""
    var w = m3_case_weights(M3_COV_CASE)
    var out = List[List[Float32]]()
    out.append(w.norm_w.copy())
    out.append(w.w_in.copy())
    out.append(w.dt_bias.copy())
    out.append(w.bnorm_w.copy())
    out.append(w.cnorm_w.copy())
    out.append(w.b_bias.copy())
    out.append(w.c_bias.copy())
    out.append(w.d_skip.copy())
    out.append(w.w_out.copy())
    return out^


def m3_cov_shape() raises -> Tuple[Int, Int, Int]:
    """(B, L, d_model) of the case."""
    var c = m3_corpus_case(M3_COV_CASE)
    return (c.b, c.l, c.d_model)


def m3_cov_x() raises -> List[Float32]:
    """x (B, L, d_model) of the case."""
    return m3_case_x(M3_COV_CASE)


def m3_cov_token(x: List[Float32], b: Int, l: Int, dm: Int, t: Int) -> List[Float32]:
    """Token t of every row: (B, 1, d_model)."""
    var out = List[Float32](capacity=b * dm)
    for row in range(b):  # small-loop(two fixture rows)
        for j in range(dm):  # small-loop(d_model 32 fixture values)
            out.append(x[(row * l + t) * dm + j])
    return out^


def m3_cov_state_sizes(b: Int, dm: Int) raises -> List[Int]:
    """The ten carried-state pieces in the ABI order (theta, h, buf_qrot,
    buf_krot, buf_v, buf_dt, buf_sig, buf_adt, pending_k, pending_v)."""
    var nh = Mamba3Dims.of(dm).nheads
    var qrow = b * M3_CHUNK_SIZE * nh
    var sizes: List[Int] = [b * nh * M3_NUM_ROPE_ANGLES, b * nh * M3_HEADDIM * M3_D_STATE, qrow * M3_D_STATE,
                            qrow * M3_D_STATE, qrow * M3_HEADDIM, qrow, qrow, qrow, b * nh * M3_D_STATE,
                            b * nh * M3_HEADDIM]
    return sizes^


def m3_cov_zero_state(b: Int, dm: Int) raises -> List[List[Float32]]:
    """A fresh (all-zero) carried state."""
    var sizes = m3_cov_state_sizes(b, dm)
    var out = List[List[Float32]]()
    for i in range(len(sizes)):  # small-loop(the ten state pieces)
        out.append(m3_cov_filled(sizes[i], M3_COV_ZERO))
    return out^


def m3_cov_outputs(b: Int, l: Int, dm: Int, word: UInt32) raises -> List[List[Float32]]:
    """[y (B, L, d_model), h_last, k_last, v_last, theta_last], every word
    `word` (the door's allocation, poisoned or zero-filled)."""
    var nh = Mamba3Dims.of(dm).nheads
    var out = List[List[Float32]]()
    out.append(m3_cov_filled(b * l * dm, word))
    out.append(m3_cov_filled(b * nh * M3_HEADDIM * M3_D_STATE, word))
    out.append(m3_cov_filled(b * nh * M3_D_STATE, word))
    out.append(m3_cov_filled(b * nh * M3_HEADDIM, word))
    out.append(m3_cov_filled(b * nh * M3_NUM_ROPE_ANGLES, word))
    return out^


def m3_cov_addresses(pieces: List[List[Float32]]) -> List[Int]:
    var out = List[Int](capacity=len(pieces))
    for i in range(len(pieces)):  # small-loop(at most ten buffer addresses)
        out.append(Int(pieces[i].unsafe_ptr()))
    return out^


def m3_cov_same_bits(a: List[List[Float32]], b: List[List[Float32]], what: String) raises:
    """Every piece of `a` equals `b` bit for bit."""
    require(len(a) == len(b), what + ": piece counts differ")
    for k in range(len(a)):  # small-loop(at most ten output pieces)
        require(len(a[k]) == len(b[k]), what + ": piece " + String(k) + " lengths differ")
        for i in range(len(a[k])):  # small-loop(fixture piece: at most 16384 floats compared)
            if bitcast[DType.uint32](a[k][i]) != bitcast[DType.uint32](b[k][i]):
                raise Error(what + ": piece " + String(k) + " element " + String(i)
                            + " differs between the 0xFF-prefilled and the zero-filled outputs (an unwritten byte)")
