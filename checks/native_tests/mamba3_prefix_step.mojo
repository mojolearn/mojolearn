# SPDX-License-Identifier: Apache-2.0
"""A Mamba-3 forward equals its prefix forward followed by one decode step,
bit for bit, on the CPU column (lane tmig-sequence-1, 2026-11-14; ported
from the numerical relation of
`test_mamba_native_public.py::test_mamba3_forward_step_ragged_are_native_and_match_the_retained_driver`:
`forward(x[:, :L-1], state)` then `step(x[:, L-1], state)` from a fresh state
returns the bytes of the stateless `forward(x)`).

The host Mamba-3 body every CPU call reaches (`_mojolearn_mamba_host`'s
state operations 4 and 5 run mamba/host/mamba_host_shell_impl.mojo
`_mamba3_run`) on the seeded corpus case of the output-coverage suites
(mamba3_output_coverage_cases.mojo: B 2, L 4, d_model 32): the discard-state
prefill of L tokens against the carried-state forward of L - 1 tokens from a
zero state and one step from the state it left. The Python test's native
audits, refusal texts and ragged admission stay in Python (BOUNDARY).

One fixture of a few milliseconds: it runs on the calling thread (a single
unit; the host pool has nothing to split). Run in a fresh process.
"""
from std.memory import bitcast
from mamba.host.mamba_host_shell_impl import _mamba3_run
from checks.native_tests.mamba3_output_coverage import (
    m3_cov_weights,
    m3_cov_shape,
    m3_cov_x,
    m3_cov_token,
    m3_cov_zero_state,
    m3_cov_outputs,
    m3_cov_addresses,
    M3_COV_ZERO,
)
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The host column (was mamba3_prefix_step_cpu_suite.mojo) ----
def _abi(x: List[Float32], weights: List[List[Float32]], state: List[List[Float32]],
         outputs: List[List[Float32]]) -> List[Int]:
    """The 25-slot per-call ABI: x, nine weights, ten state pieces, y, four reports."""
    var a: List[Int] = [Int(x.unsafe_ptr())]
    a.extend(m3_cov_addresses(weights))
    a.extend(m3_cov_addresses(state))
    a.extend(m3_cov_addresses(outputs))
    return a^


def _prefix(x: List[Float32], b: Int, l: Int, dm: Int, keep: Int) -> List[Float32]:
    """Tokens [0, keep) of every row: (B, keep, d_model)."""
    var out = List[Float32](capacity=b * keep * dm)
    for row in range(b):  # small-loop(two fixture rows)
        for t in range(keep):  # small-loop(fixture tokens)
            for j in range(dm):  # small-loop(d_model 32 fixture values)
                out.append(x[(row * l + t) * dm + j])
    return out^


def _run_cpu(mut report: Report) raises:
    var shape = m3_cov_shape()
    var b = shape[0]
    var l = shape[1]
    var dm = shape[2]
    require(l >= 2, "the corpus case needs at least two tokens")
    var weights = m3_cov_weights()
    var x = m3_cov_x()
    # the stateless forward of all L tokens (state slots neither read nor written)
    var unread = m3_cov_zero_state(b, dm)
    var whole = m3_cov_outputs(b, l, dm, M3_COV_ZERO)
    _ = _mamba3_run(_abi(x, weights, unread, whole), b, l, dm, 0, 0, True)
    # the carried-state forward of L - 1 tokens from a zero state, then one step
    var state = m3_cov_zero_state(b, dm)
    var head = _prefix(x, b, l, dm, l - 1)
    var first = m3_cov_outputs(b, l - 1, dm, M3_COV_ZERO)
    var q = _mamba3_run(_abi(head, weights, state, first), b, l - 1, dm, 0, 0, False)
    var last = m3_cov_token(x, b, l, dm, l - 1)
    var step = m3_cov_outputs(b, 1, dm, M3_COV_ZERO)
    _ = _mamba3_run(_abi(last, weights, state, step), b, 1, dm, q, 0, False)
    for row in range(b):  # small-loop(two fixture rows)
        for t in range(l):  # small-loop(fixture tokens)
            for j in range(dm):  # small-loop(d_model 32 fixture values)
                var want = whole[0][(row * l + t) * dm + j]
                var got = first[0][(row * (l - 1) + t) * dm + j] if t < l - 1 else step[0][row * dm + j]
                if bitcast[DType.uint32](got) != bitcast[DType.uint32](want):
                    raise Error("Mamba-3 prefix + step: row " + String(row) + " token " + String(t) + " word "
                                + String(j) + " is " + String(got) + ", the whole forward's " + String(want))
    _ = x^  # the calls above read x, the prefix, the token and the weights by address
    _ = head^
    _ = last^
    _ = weights^
    _ = unread^
    print("PASS Mamba-3 prefix + step (CPU)", 1)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("mamba3_prefix_step", cpu=True, gpu=False, serial="", run=run_cases)
