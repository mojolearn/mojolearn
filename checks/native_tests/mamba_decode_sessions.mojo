# SPDX-License-Identifier: Apache-2.0
"""The Mamba-1/2/3 host-arm decode session equals the per-call step byte for
byte (lane tmig-core-3, 2026-11-15; ported from
`test_mamba23_decode_sessions.py::test_session_step_is_byte_for_byte_the_per_call_step`
and the Mamba-1 case of
`test_decode_sessions_cpu_route.py::test_session_step_is_byte_for_byte_the_per_call_step`).

The session is the typed entry mamba/mamba_decode_session_run.mojo
(`MambaDecodeSessionRun`, opened on copies of the weights and a zero state,
stepping through the CPU training image's state entry, op 2 * kind - 1, as
`core/mamba_state_entry_table.mojo` `mamba_host_state_ref` resolves it). The
per-call step is the body that entry runs for one token
(mamba/host/mamba_host_shell_impl.mojo `_mamba1_run` / `_mamba2_run` /
`_mamba3_run` at L = 1) on a second, separately carried zero state. Over
every token of the fixture: each token's y, then (after the last token) every
carried state piece, the buffered count (Mamba-2/3) and Mamba-3's pending
flag must be equal word for word; y must be finite.

Fixtures (the seeded corpus cases the other Mamba checks use; the Python
test's ramp weights served the same relation): Mamba-1 corpus case 1
(mamba/checks/mamba_fixture.mojo), Mamba-2 corpus case 1
(mamba/checks/mamba2_fixture.mojo), Mamba-3 case 1
(checks/native_tests/mamba3_output_coverage.mojo). Every token of each case.

The ownership clauses of the Python session classes (stale caller buffers
until sync_state, refusal by name, load_state, close) and the Transformer
session (its host arm is Python, python/mojolearn/_transformer_impl.py
`decode_session`) stay in the Python files.

Host column: the three kinds are independent and run across host cores.
"""
from std.memory import bitcast
from std.math import isfinite
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef
from core.mamba_state_entry_table import mamba_host_state_ref
from mamba.host.neural_entry_host import publish_neural_mamba_host
from mamba.host.mamba_host_shell_impl import _mamba1_run, _mamba2_run, _mamba3_run
from mamba.mamba_decode_session_run import MambaDecodeSessionRun
from autograd.block_step_ops import OPS_M1_STEP, OPS_M2_STEP, block_state_sizes, m2_report_floats, m3_report_floats
from mamba.checks.mamba_fixture import corpus_case, corpus_case_weights, corpus_case_x
from mamba.checks.mamba2_fixture import m2_corpus_case, m2_case_weights, m2_case_x
from checks.native_tests.mamba3_output_coverage import (
    m3_cov_weights, m3_cov_shape, m3_cov_x, m3_cov_token, m3_cov_zero_state,
)
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime MAMBA_DECODE_SESSION_KINDS = 3


struct _Case(Movable):
    var kind: Int
    var b: Int
    var l: Int
    var dm: Int
    var dt_lo: Float64
    var dt_hi: Float64
    var weights: List[List[Float32]]
    var x: List[Float32]
    var state_sizes: List[Int]
    var report_sizes: List[Int]

    def __init__(out self, kind: Int) raises:
        var bb: Int
        var ll: Int
        var dm: Int
        var dt_lo = 0.0
        var dt_hi = 0.0
        var weights = List[List[Float32]]()
        var x: List[Float32]
        var state_sizes: List[Int]
        var report_sizes = List[Int]()
        if kind == 1:
            var c = corpus_case(1)
            bb = c.b
            ll = c.l
            dm = c.d_model
            var w = corpus_case_weights(1)
            weights.append(w.norm_w.copy())
            weights.append(w.w_in.copy())
            weights.append(w.conv_w.copy())
            weights.append(w.conv_b.copy())
            weights.append(w.w_x.copy())
            weights.append(w.w_dt.copy())
            weights.append(w.b_dt.copy())
            weights.append(w.a_log.copy())
            weights.append(w.d_skip.copy())
            weights.append(w.w_out.copy())
            x = corpus_case_x(1)
            var ints: List[Int] = [bb, dm]
            state_sizes = block_state_sizes(OPS_M1_STEP, ints)
        elif kind == 2:
            var c = m2_corpus_case(1)
            bb = c.b
            ll = c.l
            dm = c.d_model
            dt_lo = Float64(c.dt_lo)
            dt_hi = Float64(c.dt_hi)
            var w = m2_case_weights(1)
            weights.append(w.norm_w.copy())
            weights.append(w.w_in.copy())
            weights.append(w.conv_w.copy())
            weights.append(w.conv_b.copy())
            weights.append(w.dt_bias.copy())
            weights.append(w.a_log.copy())
            weights.append(w.d_skip.copy())
            weights.append(w.gnorm_w.copy())
            weights.append(w.w_out.copy())
            x = m2_case_x(1)
            var ints: List[Int] = [bb, dm]
            state_sizes = block_state_sizes(OPS_M2_STEP, ints)
            report_sizes.append(m2_report_floats(bb, dm))
        else:
            var shape = m3_cov_shape()
            bb = shape[0]
            ll = shape[1]
            dm = shape[2]
            weights = m3_cov_weights()
            x = m3_cov_x()
            state_sizes = List[Int]()
            var zero = m3_cov_zero_state(bb, dm)
            for i in range(len(zero)):  # small-loop(the ten Mamba-3 state pieces)
                state_sizes.append(len(zero[i]))
            report_sizes = m3_report_floats(bb, dm)
        self.kind = kind
        self.b = bb
        self.l = ll
        self.dm = dm
        self.dt_lo = dt_lo
        self.dt_hi = dt_hi
        self.weights = weights^
        self.x = x^
        self.state_sizes = state_sizes^
        self.report_sizes = report_sizes^

    def zeros(self, sizes: List[Int]) -> List[List[Float32]]:
        var out = List[List[Float32]]()
        for i in range(len(sizes)):  # small-loop(at most ten pieces)
            out.append(List[Float32](length=sizes[i], fill=Float32(0.0)))
        return out^

    def per_call_step(self, token: List[Float32], mut state: List[List[Float32]], mut y: List[Float32],
                      mut reports: List[List[Float32]], q: Int, pending: Int) raises -> Int:
        """One token through the host state body: x, weights, the state
        (Mamba-1: its two pieces), y, the reports."""
        var a: List[Int] = [Int(token.unsafe_ptr())]
        for i in range(len(self.weights)):  # small-loop(nine or ten weights)
            a.append(Int(self.weights[i].unsafe_ptr()))
        for i in range(len(state)):  # small-loop(at most ten state pieces)
            a.append(Int(state[i].unsafe_ptr()))
        a.append(Int(y.unsafe_ptr()))
        for i in range(len(reports)):  # small-loop(at most four reports)
            a.append(Int(reports[i].unsafe_ptr()))
        var updated = 0
        if self.kind == 1:
            _mamba1_run(a, self.b, 1, self.dm, 0)
        elif self.kind == 2:
            updated = _mamba2_run(a, self.b, 1, self.dm, q, Float32(self.dt_lo), Float32(self.dt_hi))
        else:
            updated = _mamba3_run(a, self.b, 1, self.dm, q, pending, False)
        # the body read and wrote these by address
        _ = token
        _ = state
        _ = y
        _ = reports
        return updated


def _token(x: List[Float32], b: Int, l: Int, dm: Int, t: Int) -> List[Float32]:
    return m3_cov_token(x, b, l, dm, t)


def _same_words(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # small-loop(fixture piece words compared)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error(what + ": word " + String(i) + " is " + String(a[i]) + " vs " + String(b[i]))


def run_kind(entry: EntryRef, kind: Int) raises:
    var c = _Case(kind)
    var name = "Mamba" + String(kind) + " host session"
    # the session: copies of the weights and a zero state
    var names = List[String]()
    var dims = List[List[Int]]()
    for i in range(len(c.state_sizes)):  # small-loop(at most ten state pieces)
        names.append("piece" + String(i))
        dims.append(List[Int](length=1, fill=c.state_sizes[i]))
    var report_dims = List[List[Int]]()
    for i in range(len(c.report_sizes)):  # small-loop(at most four reports)
        report_dims.append(List[Int](length=1, fill=c.report_sizes[i]))
    var weights = List[List[Float32]]()
    for i in range(len(c.weights)):  # small-loop(nine or ten weights)
        weights.append(c.weights[i].copy())
    var session = MambaDecodeSessionRun(kind, entry, c.b, c.dm, weights^, c.zeros(c.state_sizes), names^,
                                        dims^, report_dims^, dt_low=c.dt_lo, dt_high=c.dt_hi)
    var session_reports = c.zeros(c.report_sizes)
    var report_addresses = List[Int]()
    for i in range(len(session_reports)):  # small-loop(at most four reports)
        report_addresses.append(Int(session_reports[i].unsafe_ptr()))
    # the per-call steps on a second zero state
    var plain = c.zeros(c.state_sizes)
    var plain_reports = c.zeros(c.report_sizes)
    var q = 0
    var pending = 0
    for t in range(c.l):  # driver-loop: over the fixture's decode tokens (each reads the state the last left)
        var token = _token(c.x, c.b, c.l, c.dm, t)
        var got = List[Float32](length=c.b * c.dm, fill=bitcast[DType.float32](UInt32(0xFFFFFFFF)))
        var want = List[Float32](length=c.b * c.dm, fill=Float32(0.0))
        session.step(Int(token.unsafe_ptr()), Int(got.unsafe_ptr()), report_addresses)
        q = c.per_call_step(token, plain, want, plain_reports, q, pending)
        if kind == 3:
            pending = 0
        for i in range(len(want)):  # small-loop(B * d_model words)
            require(isfinite(want[i]), name + ": per-call y token " + String(t) + " word " + String(i) + " not finite")
        _same_words(got, want, name + " y token " + String(t))
        _ = token^
        _ = got^
        _ = want^
    for i in range(len(plain)):  # small-loop(at most ten state pieces)
        _same_words(session.plan.parts[i], plain[i], name + " state piece " + String(i))
    if kind != 1:
        require(session.plan.q == q, name + ": buffered tokens " + String(session.plan.q) + " vs " + String(q))
    if kind == 3:
        require(Int(session.plan.pending) == pending, name + ": pending flag differs")
    session.close()
    _ = session^
    _ = session_reports^
    _ = report_addresses^
    _ = plain^
    _ = plain_reports^
    _ = c^


# ---- The host column ----
def _run_cpu(mut report: Report) raises:
    publish_neural_mamba_host()
    var entry = mamba_host_state_ref(Int(GLOBAL_NUMERIC_MODE))
    var cases = report.cases(MAMBA_DECODE_SESSION_KINDS)
    def task(unit: Int) {imm cases, imm entry}:
        try:
            run_kind(entry, unit + 1)
        except error:
            cases.fail(unit, error)
    report.parallel(task, MAMBA_DECODE_SESSION_KINDS)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("mamba_decode_sessions", cpu=True, gpu=False, serial="", run=run_cases)
