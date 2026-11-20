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
    Mamba3Dims,
    M3_D_STATE,
    M3_HEADDIM,
    M3_CHUNK_SIZE,
    M3_NUM_ROPE_ANGLES,
    m3_corpus_case,
    m3_case_weights,
    m3_case_x,
)
from mamba.host.mamba_host_shell_impl import _mamba3_run
from autograd.block_step_ops import OPS_M3_STEP
from mamba.impl.mamba_block_program import mamba_block_call, MambaBlockProgram
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime M3_COV_CASE = 1
"""The corpus case (B 2, L 4, d_model 32): small, two rows, one head."""
comptime M3_COV_POISON: UInt32 = 0xFFFFFFFF
comptime M3_COV_ZERO: UInt32 = 0


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


# ---- The host column (was mamba3_output_coverage_cpu_suite.mojo) ----
# Mamba-3 outputs are written whole on the CPU column (lane
# fix-mamba3-unfilled-rest, 2026-11-13).
#
# The host Mamba-3 body every CPU carried-state call reaches
# (`_mojolearn_mamba_host`'s state operations 4 and 5 run
# mamba/host/mamba_host_shell_impl.mojo `_mamba3_run`): a carried-state
# forward of L = 4 tokens from a zero state, then one decode step from the
# state it left, and the discard-state (fresh) prefill. Each runs twice, into
# 0xFF-prefilled and into zero-filled y and reports (see
# mamba3_output_coverage_cases.mojo); y, the four reports, the carried
# integers and the written-back state must agree bit for bit. This is the
# native evidence that `generation_array_unfilled` is sound for these
# outputs (`-D MOJOLEARN_MAMBA3_FORWARD_ZERO_FILL` is the zero-filled arm).
#
# The two cases share no state; they run one after another because each is
# a few milliseconds of one-thread host arithmetic on a 2-row fixture (the
# whole suite is far below one second): splitting them over the host pool
# would cost more in pool startup than it saves. Run in a fresh process.
comptime MAMBA3_COVERAGE_CPU_CASES = 2


def _abi(x: List[Float32], weights: List[List[Float32]], state: List[List[Float32]],
         outputs: List[List[Float32]]) -> List[Int]:
    """The 25-slot per-call ABI: x, nine weights, ten state pieces, y, four reports."""
    var a: List[Int] = [Int(x.unsafe_ptr())]
    a.extend(m3_cov_addresses(weights))
    a.extend(m3_cov_addresses(state))
    a.extend(m3_cov_addresses(outputs))
    return a^


def _carried_pass(word: UInt32) raises -> List[List[Float32]]:
    """Forward L tokens from a zero state, then one step from the state it
    left: [y, reports of the forward, y, reports of the step, the final ten
    state pieces, the carried integers as floats]."""
    var shape = m3_cov_shape()
    var b = shape[0]
    var l = shape[1]
    var dm = shape[2]
    var weights = m3_cov_weights()
    var x = m3_cov_x()
    var x1 = m3_cov_token(x, b, l, dm, 0)
    var state = m3_cov_zero_state(b, dm)
    var forward = m3_cov_outputs(b, l, dm, word)
    var q = _mamba3_run(_abi(x, weights, state, forward), b, l, dm, 0, 0, False)
    var step = m3_cov_outputs(b, 1, dm, word)
    var q_step = _mamba3_run(_abi(x1, weights, state, step), b, 1, dm, q, 0, False)
    var out = List[List[Float32]]()
    for i in range(len(forward)):  # small-loop(y and four reports)
        out.append(forward[i].copy())
    for i in range(len(step)):  # small-loop(y and four reports)
        out.append(step[i].copy())
    for i in range(len(state)):  # small-loop(ten state pieces)
        out.append(state[i].copy())
    var words: List[Float32] = [Float32(q), Float32(q_step)]
    out.append(words^)
    _ = x^  # the calls above read x, x1 and the weights by address
    _ = x1^
    _ = weights^
    return out^


def _fresh_pass(word: UInt32) raises -> List[List[Float32]]:
    """The discard-state prefill (`fresh`): the state slots are neither read
    nor written; [y, four reports]."""
    var shape = m3_cov_shape()
    var b = shape[0]
    var l = shape[1]
    var dm = shape[2]
    var weights = m3_cov_weights()
    var x = m3_cov_x()
    # Unread state slots still need non-null addresses for the ABI.
    var state = m3_cov_zero_state(b, dm)
    var outputs = m3_cov_outputs(b, l, dm, word)
    _ = _mamba3_run(_abi(x, weights, state, outputs), b, l, dm, 0, 0, True)
    _ = x^  # the call above read x, the weights and the state by address
    _ = weights^
    _ = state^
    return outputs^


def run_case(unit: Int) raises:
    if unit == 0:
        m3_cov_same_bits(_carried_pass(M3_COV_POISON), _carried_pass(M3_COV_ZERO),
                         "Mamba-3 host carried-state forward + step")
    else:
        m3_cov_same_bits(_fresh_pass(M3_COV_POISON), _fresh_pass(M3_COV_ZERO),
                         "Mamba-3 host fresh prefill")


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(MAMBA3_COVERAGE_CPU_CASES)
    for unit in range(MAMBA3_COVERAGE_CPU_CASES):  # driver-loop: two millisecond cases, serial by design (module docstring)
        try:
            run_case(unit)
        except problem:
            cases.fail(unit, problem)


# ---- The device column (was mamba3_output_coverage_gpu_suite.mojo) ----
# Mamba-3 outputs are written whole on the GPU routes (lane
# fix-mamba3-unfilled-rest, 2026-11-13).
#
# The device bodies behind every Mamba-3 output the public doors now allocate
# unfilled (`generation_array_unfilled`; `-D MOJOLEARN_MAMBA3_FORWARD_ZERO_FILL`
# is the zero-filled arm):
#
#   0  `mamba_block_call` carried-state forward (L = 4 from a zero state) and
#      one decode step from the state it left: state operations 4 and 5 of
#      mamba/native_state_dispatch.mojo and the owned operation 16 run it
#      (`Mamba3Block.forward(state=)`, `.step`, the session's host arm);
#   1  `mamba_block_call` with `fresh` (the discard-state prefill,
#      `mamba3_forward_fresh`'s body);
#   2  the resident session program (`MambaBlockProgram`, what
#      `mamba3_session_step` / `mamba3_session_decode` run): one step, then a
#      decode of L tokens, then the state exported.
#
# Each runs twice, into 0xFF-prefilled and into zero-filled y and reports
# (mamba3_output_coverage_cases.mojo); every output, the carried integers and
# the state must agree bit for bit.
#
# The cases run one after another: they share the process's one device
# context and the per-block program cache (`mamba_block_call` claims a cached
# program per weight set), and each is a few launches on a 2-row fixture.
# Run in a fresh process on a GPU host.
comptime MAMBA3_COVERAGE_GPU_CASES = 3


def _report_addresses(outputs: List[List[Float32]]) -> List[Int]:
    """The four report addresses (outputs[1:5])."""
    var addresses = m3_cov_addresses(outputs)
    var out = List[Int](capacity=4)
    for i in range(1, 5):  # small-loop(the four Mamba-3 reports)
        out.append(addresses[i])
    return out^


def _append_all(mut dst: List[List[Float32]], src: List[List[Float32]]):
    for i in range(len(src)):  # small-loop(at most ten pieces)
        dst.append(src[i].copy())


def _words_as_floats(words: List[Int]) -> List[Float32]:
    var out = List[Float32](capacity=len(words))
    for i in range(len(words)):  # small-loop(the two carried integers)
        out.append(Float32(words[i]))
    return out^


def _carried_pass_gpu(word: UInt32) raises -> List[List[Float32]]:
    var shape = m3_cov_shape()
    var b = shape[0]
    var l = shape[1]
    var dm = shape[2]
    var weights = m3_cov_weights()
    var waddr = m3_cov_addresses(weights)
    var x = m3_cov_x()
    var x1 = m3_cov_token(x, b, l, dm, 0)
    var state = m3_cov_zero_state(b, dm)
    var saddr = m3_cov_addresses(state)
    var forward = m3_cov_outputs(b, l, dm, word)
    var w1 = mamba_block_call(OPS_M3_STEP, waddr, saddr, [0, 0], Int(x.unsafe_ptr()), Int(forward[0].unsafe_ptr()),
                              _report_addresses(forward), b, l, dm, List[Float32](), False)
    require(len(w1) == 2, "Mamba-3 carried forward returned " + String(len(w1)) + " carried integers")
    var step = m3_cov_outputs(b, 1, dm, word)
    var w2 = mamba_block_call(OPS_M3_STEP, waddr, saddr, [w1[0], w1[1]], Int(x1.unsafe_ptr()),
                              Int(step[0].unsafe_ptr()), _report_addresses(step), b, 1, dm, List[Float32](), False)
    var out = List[List[Float32]]()
    _append_all(out, forward)
    _append_all(out, step)
    _append_all(out, state)
    out.append(_words_as_floats(w1))
    out.append(_words_as_floats(w2))
    _ = x^  # the calls above read x, x1, the weights and the state by address
    _ = x1^
    _ = weights^
    return out^


def _fresh_pass_gpu(word: UInt32) raises -> List[List[Float32]]:
    var shape = m3_cov_shape()
    var b = shape[0]
    var l = shape[1]
    var dm = shape[2]
    var weights = m3_cov_weights()
    var x = m3_cov_x()
    var state = m3_cov_zero_state(b, dm)
    var outputs = m3_cov_outputs(b, l, dm, word)
    _ = mamba_block_call(OPS_M3_STEP, m3_cov_addresses(weights), m3_cov_addresses(state), [0, 0],
                         Int(x.unsafe_ptr()), Int(outputs[0].unsafe_ptr()), _report_addresses(outputs), b, l, dm,
                         List[Float32](), True)
    _ = x^  # the call above read x, the weights and the state slots by address
    _ = weights^
    _ = state^
    return outputs^


def _session_pass(word: UInt32) raises -> List[List[Float32]]:
    var shape = m3_cov_shape()
    var b = shape[0]
    var l = shape[1]
    var dm = shape[2]
    var weights = m3_cov_weights()
    var x = m3_cov_x()
    var x1 = m3_cov_token(x, b, l, dm, 0)
    var start = m3_cov_zero_state(b, dm)
    var prog = MambaBlockProgram(OPS_M3_STEP, b, dm, List[Float32](), m3_cov_addresses(weights))
    var out = List[List[Float32]]()
    try:
        prog.load_state(m3_cov_addresses(start), [0, 0])
        var step = m3_cov_outputs(b, 1, dm, word)
        var w1 = prog.step(Int(x1.unsafe_ptr()), Int(step[0].unsafe_ptr()), _report_addresses(step))
        var decoded = m3_cov_outputs(b, l, dm, word)
        var w2 = prog.decode(Int(x.unsafe_ptr()), Int(decoded[0].unsafe_ptr()), l, _report_addresses(decoded))
        # The exported state is not a door output; zero-filled in both passes.
        var exported = m3_cov_zero_state(b, dm)
        var w3 = prog.export_state(m3_cov_addresses(exported))
        _append_all(out, step)
        _append_all(out, decoded)
        _append_all(out, exported)
        out.append(_words_as_floats(w1))
        out.append(_words_as_floats(w2))
        out.append(_words_as_floats(w3))
    finally:
        prog.close()
    _ = x^  # the program read x, x1, the start state and the weights by address
    _ = x1^
    _ = start^
    _ = weights^
    return out^


def case_gpu(unit: Int) raises:
    if unit == 0:
        m3_cov_same_bits(_carried_pass_gpu(M3_COV_POISON), _carried_pass_gpu(M3_COV_ZERO),
                         "Mamba-3 device carried-state forward + step")
    elif unit == 1:
        m3_cov_same_bits(_fresh_pass_gpu(M3_COV_POISON), _fresh_pass_gpu(M3_COV_ZERO), "Mamba-3 device fresh prefill")
    else:
        m3_cov_same_bits(_session_pass(M3_COV_POISON), _session_pass(M3_COV_ZERO),
                         "Mamba-3 resident session step + decode")


def _run_gpu(mut report: Report) raises:
    var cases = report.cases(MAMBA3_COVERAGE_GPU_CASES)
    for unit in range(MAMBA3_COVERAGE_GPU_CASES):  # driver-loop: cases share one device and the program cache (module docstring)
        try:
            case_gpu(unit)
        except problem:
            cases.fail(unit, problem)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("mamba3_output_coverage", cpu=True, gpu=True, serial="", run=run_cases)
