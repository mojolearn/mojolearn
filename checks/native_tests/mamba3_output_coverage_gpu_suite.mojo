# SPDX-License-Identifier: Apache-2.0
"""Mamba-3 outputs are written whole on the GPU routes (lane
fix-mamba3-unfilled-rest, 2026-11-13).

The device bodies behind every Mamba-3 output the public doors now allocate
unfilled (`generation_array_unfilled`; `-D MOJOLEARN_MAMBA3_FORWARD_ZERO_FILL`
is the zero-filled arm):

  0  `mamba_block_call` carried-state forward (L = 4 from a zero state) and
     one decode step from the state it left: state operations 4 and 5 of
     mamba/native_state_dispatch.mojo and the owned operation 16 run it
     (`Mamba3Block.forward(state=)`, `.step`, the session's host arm);
  1  `mamba_block_call` with `fresh` (the discard-state prefill,
     `mamba3_forward_fresh`'s body);
  2  the resident session program (`MambaBlockProgram`, what
     `mamba3_session_step` / `mamba3_session_decode` run): one step, then a
     decode of L tokens, then the state exported.

Each runs twice, into 0xFF-prefilled and into zero-filled y and reports
(mamba3_output_coverage_cases.mojo); every output, the carried integers and
the state must agree bit for bit.

The cases run one after another: they share the process's one device
context and the per-block program cache (`mamba_block_call` claims a cached
program per weight set), and each is a few launches on a 2-row fixture.
Run in a fresh process on a GPU host.
"""
from autograd.block_step_ops import OPS_M3_STEP
from mamba.impl.mamba_block_program import mamba_block_call, MambaBlockProgram
from checks.native_tests.mamba3_output_coverage_cases import (
    require, m3_cov_weights, m3_cov_shape, m3_cov_x, m3_cov_token, m3_cov_zero_state, m3_cov_outputs,
    m3_cov_addresses, m3_cov_same_bits, M3_COV_POISON, M3_COV_ZERO,
)

comptime MAMBA3_COVERAGE_GPU_CASES = 3


def _report_addresses(outputs: List[List[Float32]]) -> List[Int]:
    """The four report addresses (outputs[1:5])."""
    var all = m3_cov_addresses(outputs)
    var out = List[Int](capacity=4)
    for i in range(1, 5):  # small-loop(the four Mamba-3 reports)
        out.append(all[i])
    return out^


def _append_all(mut dst: List[List[Float32]], src: List[List[Float32]]):
    for i in range(len(src)):  # small-loop(at most ten pieces)
        dst.append(src[i].copy())


def _words_as_floats(words: List[Int]) -> List[Float32]:
    var out = List[Float32](capacity=len(words))
    for i in range(len(words)):  # small-loop(the two carried integers)
        out.append(Float32(words[i]))
    return out^


def _carried_pass(word: UInt32) raises -> List[List[Float32]]:
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


def _fresh_pass(word: UInt32) raises -> List[List[Float32]]:
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


def case(unit: Int) raises:
    if unit == 0:
        m3_cov_same_bits(_carried_pass(M3_COV_POISON), _carried_pass(M3_COV_ZERO),
                         "Mamba-3 device carried-state forward + step")
    elif unit == 1:
        m3_cov_same_bits(_fresh_pass(M3_COV_POISON), _fresh_pass(M3_COV_ZERO), "Mamba-3 device fresh prefill")
    else:
        m3_cov_same_bits(_session_pass(M3_COV_POISON), _session_pass(M3_COV_ZERO),
                         "Mamba-3 resident session step + decode")


def run_checks() raises:
    var failed = 0
    for unit in range(MAMBA3_COVERAGE_GPU_CASES):  # driver-loop: cases share one device and the program cache (module docstring)
        try:
            case(unit)
        except problem:
            failed += 1
            print("FAIL Mamba-3 output coverage (GPU) case", unit, String(problem))
    require(failed == 0, "Mamba-3 output coverage (GPU) cases failed: " + String(failed))
    print("PASS Mamba-3 output coverage (GPU)", MAMBA3_COVERAGE_GPU_CASES)
