# SPDX-License-Identifier: Apache-2.0
"""Mamba-3 outputs are written whole on the CPU column (lane
fix-mamba3-unfilled-rest, 2026-11-13).

The host Mamba-3 body every CPU carried-state call reaches
(`_mojolearn_mamba_host`'s state operations 4 and 5 run
mamba/host/mamba_host_shell_impl.mojo `_mamba3_run`): a carried-state
forward of L = 4 tokens from a zero state, then one decode step from the
state it left, and the discard-state (fresh) prefill. Each runs twice, into
0xFF-prefilled and into zero-filled y and reports (see
mamba3_output_coverage_cases.mojo); y, the four reports, the carried
integers and the written-back state must agree bit for bit. This is the
native evidence that `generation_array_unfilled` is sound for these
outputs (`-D MOJOLEARN_MAMBA3_FORWARD_ZERO_FILL` is the zero-filled arm).

The two cases share no state; they run one after another because each is
a few milliseconds of one-thread host arithmetic on a 2-row fixture (the
whole suite is far below one second): splitting them over the host pool
would cost more in pool startup than it saves. Run in a fresh process.
"""
from mamba.host.mamba_host_shell_impl import _mamba3_run
from checks.native_tests.mamba3_output_coverage_cases import (
    require, m3_cov_weights, m3_cov_shape, m3_cov_x, m3_cov_token, m3_cov_zero_state, m3_cov_outputs,
    m3_cov_addresses, m3_cov_same_bits, M3_COV_POISON, M3_COV_ZERO,
)

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


def case(unit: Int) raises:
    if unit == 0:
        m3_cov_same_bits(_carried_pass(M3_COV_POISON), _carried_pass(M3_COV_ZERO),
                         "Mamba-3 host carried-state forward + step")
    else:
        m3_cov_same_bits(_fresh_pass(M3_COV_POISON), _fresh_pass(M3_COV_ZERO),
                         "Mamba-3 host fresh prefill")


def run_checks() raises:
    var failed = 0
    for unit in range(MAMBA3_COVERAGE_CPU_CASES):  # driver-loop: two millisecond cases, serial by design (module docstring)
        try:
            case(unit)
        except problem:
            failed += 1
            print("FAIL Mamba-3 output coverage (CPU) case", unit, String(problem))
    require(failed == 0, "Mamba-3 output coverage (CPU) cases failed: " + String(failed))
    print("PASS Mamba-3 output coverage (CPU)", MAMBA3_COVERAGE_CPU_CASES)
