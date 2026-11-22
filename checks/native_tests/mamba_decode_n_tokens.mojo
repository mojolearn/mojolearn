# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""One Mamba-1/2/3 session `decode` over N tokens equals N session `step`
calls byte for byte (decode plan D2; lane tests-prune-remaining-models,
2026-11-15; ported from
`test_decode_n_tokens.py::test_mamba_decode_equals_the_steps`).

Per kind, two sessions open on the same weights and a zero state. The first
steps the fixture's L tokens one at a time; the second decodes them in two
calls (the first ceil(L / 2) tokens, then the rest: the second call
continues the first's state). y (B, L, d_model) of the two must be equal
word for word, then the reports (the last token's), every carried state
piece and the carried integers (Mamba-2 buf_len, Mamba-3 buf_len and
pending).

  host column: the typed host-arm session mamba/mamba_decode_session_run.mojo
    `MambaDecodeSessionRun` (`decode` against `step`), opened as
    checks/native_tests/mamba_decode_sessions.mojo opens it;
  device column: the resident program mamba/impl/mamba_block_program.mojo
    `MambaBlockProgram` (`decode`: x uploaded once, the step op per token on
    the device, y down in one copy), what the public
    `Mamba{1,2,3}DecodeSession.decode` doors run.

Fixtures: the seeded corpus cases of mamba_decode_sessions.mojo `_Case`
(Mamba-1 case 1, Mamba-2 case 1, Mamba-3 case 1; L = 4). The refusals
(rows, closed) and the Transformer and Samba sessions stay in the Python
file (their decode arms are Python-assembled).

Host column: the three kinds are independent and run across host cores.
Device column: the kinds run one after another on the process's one device
context.
"""
from std.memory import bitcast
from autograd.block_step_ops import OPS_M1_STEP, OPS_M2_STEP, OPS_M3_STEP
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef
from core.mamba_state_entry_table import mamba_host_state_ref
from mamba.host.neural_entry_host import publish_neural_mamba_host
from mamba.mamba_decode_session_run import MambaDecodeSessionRun
from mamba.impl.mamba_block_program import MambaBlockProgram
from checks.native_tests.mamba_decode_sessions import _Case, _same_words
from checks.native_tests.mamba3_output_coverage import m3_cov_addresses, m3_cov_token
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, CPU_COLUMN_BUILD, check_cases_serial


comptime MAMBA_DECODE_N_KINDS = 3


def _filled(n: Int) -> List[Float32]:
    """`n` floats of 0xFFFFFFFF words (an unwritten word shows as a difference)."""
    return List[Float32](length=n, fill=bitcast[DType.float32](UInt32(0xFFFFFFFF)))


def _chunk(x: List[Float32], b: Int, l: Int, dm: Int, t0: Int, t1: Int) -> List[Float32]:
    """Tokens [t0, t1) of every row: (B, t1 - t0, d_model)."""
    var out = List[Float32](capacity=b * (t1 - t0) * dm)
    for r in range(b):  # small-loop(two fixture rows)
        for i in range(t0 * dm, t1 * dm):  # small-loop(fixture tokens * d_model values)
            out.append(x[r * l * dm + i])
    return out^


def _put(mut y: List[Float32], part: List[Float32], b: Int, l: Int, dm: Int, t0: Int, t1: Int):
    """`part` (B, t1 - t0, d_model) into tokens [t0, t1) of y (B, L, d_model)."""
    var n = t1 - t0
    for r in range(b):  # small-loop(two fixture rows)
        for i in range(n * dm):  # small-loop(fixture tokens * d_model values)
            y[r * l * dm + t0 * dm + i] = part[r * n * dm + i]


def _zeros(sizes: List[Int]) -> List[List[Float32]]:
    var out = List[List[Float32]]()
    for i in range(len(sizes)):  # small-loop(at most ten pieces)
        out.append(List[Float32](length=sizes[i], fill=Float32(0.0)))
    return out^


def _poisoned(sizes: List[Int]) -> List[List[Float32]]:
    var out = List[List[Float32]]()
    for i in range(len(sizes)):  # small-loop(at most four reports)
        out.append(_filled(sizes[i]))
    return out^


def _same_pieces(a: List[List[Float32]], b: List[List[Float32]], what: String) raises:
    require(len(a) == len(b), what + ": piece counts differ")
    for k in range(len(a)):  # small-loop(at most ten pieces)
        _same_words(a[k], b[k], what + " piece " + String(k))


def _same_ints(a: List[Int], b: List[Int], what: String) raises:
    require(len(a) == len(b), what + ": carried integer counts differ")
    for k in range(len(a)):  # small-loop(at most two carried integers)
        require(a[k] == b[k], what + ": carried integer " + String(k) + " is " + String(a[k])
                + " vs " + String(b[k]))


# ---- The host column ----
def _open(c: _Case, entry: EntryRef) raises -> MambaDecodeSessionRun:
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
    return MambaDecodeSessionRun(c.kind, entry, c.b, c.dm, weights^, _zeros(c.state_sizes), names^,
                                 dims^, report_dims^, dt_low=c.dt_lo, dt_high=c.dt_hi)


def run_kind_cpu(entry: EntryRef, kind: Int) raises:
    var c = _Case(kind)
    var name = "Mamba" + String(kind) + " host session decode"
    var b = c.b
    var l = c.l
    var dm = c.dm
    var split = (l + 1) // 2
    # N steps
    var stepped = _open(c, entry)
    var step_reports = _poisoned(c.report_sizes)
    var step_addresses = m3_cov_addresses(step_reports)
    var want = _filled(b * l * dm)
    for t in range(l):  # driver-loop: over the fixture's decode tokens (each reads the state the last left)
        var token = m3_cov_token(c.x, b, l, dm, t)
        var y = _filled(b * dm)
        stepped.step(Int(token.unsafe_ptr()), Int(y.unsafe_ptr()), step_addresses)
        _put(want, y, b, l, dm, t, t + 1)
        _ = token^
        _ = y^
    # two decode calls
    var decoded = _open(c, entry)
    var decode_reports = _poisoned(c.report_sizes)
    var decode_addresses = m3_cov_addresses(decode_reports)
    var got = _filled(b * l * dm)
    var bounds: List[Int] = [0, split, l]
    for k in range(2):  # driver-loop: over the two decode calls (the second continues the first's state)
        var x = _chunk(c.x, b, l, dm, bounds[k], bounds[k + 1])
        var y = _filled(b * (bounds[k + 1] - bounds[k]) * dm)
        decoded.decode(Int(x.unsafe_ptr()), bounds[k + 1] - bounds[k], Int(y.unsafe_ptr()), decode_addresses)
        _put(got, y, b, l, dm, bounds[k], bounds[k + 1])
        _ = x^
        _ = y^
    _same_words(got, want, name + " y")
    _same_pieces(decode_reports, step_reports, name + " reports")
    _same_pieces(decoded.plan.parts, stepped.plan.parts, name + " state")
    if kind != 1:
        require(decoded.plan.q == stepped.plan.q, name + ": buffered tokens " + String(decoded.plan.q)
                + " vs " + String(stepped.plan.q))
    if kind == 3:
        require(decoded.plan.pending == stepped.plan.pending, name + ": pending flag differs")
    stepped.close()
    decoded.close()
    _ = step_addresses^
    _ = step_reports^
    _ = decode_addresses^
    _ = decode_reports^
    _ = c^


def _run_cpu() raises:
    publish_neural_mamba_host()
    var entry = mamba_host_state_ref(Int(GLOBAL_NUMERIC_MODE))
    def task(unit: Int) raises {imm entry}:
        run_kind_cpu(entry, unit + 1)
    check_cases("decode_n_tokens", MAMBA_DECODE_N_KINDS, task)


# ---- The device column ----
def _code(kind: Int) -> Int:
    if kind == 1:
        return OPS_M1_STEP
    if kind == 2:
        return OPS_M2_STEP
    return OPS_M3_STEP


def run_kind_gpu(kind: Int) raises:
    var c = _Case(kind)
    var name = "Mamba" + String(kind) + " resident session decode"
    var b = c.b
    var l = c.l
    var dm = c.dm
    var split = (l + 1) // 2
    var code = _code(kind)
    var fs = List[Float32]()
    if kind == 2:
        fs.append(Float32(c.dt_lo))
        fs.append(Float32(c.dt_hi))
    var waddr = m3_cov_addresses(c.weights)
    var step_reports = _poisoned(c.report_sizes)
    var step_addresses = m3_cov_addresses(step_reports)
    var decode_reports = _poisoned(c.report_sizes)
    var decode_addresses = m3_cov_addresses(decode_reports)
    var want = _filled(b * l * dm)
    var got = _filled(b * l * dm)
    var step_state = _zeros(c.state_sizes)
    var decode_state = _zeros(c.state_sizes)
    var stepped = MambaBlockProgram(code, b, dm, fs, waddr)
    var decoded = MambaBlockProgram(code, b, dm, fs, waddr)
    try:
        stepped.zero_state()
        decoded.zero_state()
        var step_words = List[Int]()
        for t in range(l):  # driver-loop: over the fixture's decode tokens (each reads the state the last left)
            var token = m3_cov_token(c.x, b, l, dm, t)
            var y = _filled(b * dm)
            step_words = stepped.step(Int(token.unsafe_ptr()), Int(y.unsafe_ptr()), step_addresses)
            _put(want, y, b, l, dm, t, t + 1)
            _ = token^
            _ = y^
        var decode_words = List[Int]()
        var bounds: List[Int] = [0, split, l]
        for k in range(2):  # driver-loop: over the two decode calls (the second continues the first's state)
            var x = _chunk(c.x, b, l, dm, bounds[k], bounds[k + 1])
            var y = _filled(b * (bounds[k + 1] - bounds[k]) * dm)
            decode_words = decoded.decode(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), bounds[k + 1] - bounds[k],
                                          decode_addresses)
            _put(got, y, b, l, dm, bounds[k], bounds[k + 1])
            _ = x^
            _ = y^
        _same_ints(decode_words, step_words, name + " returned words")
        var exported_step = stepped.export_state(m3_cov_addresses(step_state))
        var exported_decode = decoded.export_state(m3_cov_addresses(decode_state))
        _same_ints(exported_decode, exported_step, name + " exported words")
        _same_words(got, want, name + " y")
        _same_pieces(decode_reports, step_reports, name + " reports")
        _same_pieces(decode_state, step_state, name + " state")
    finally:
        stepped.close()
        decoded.close()
    # the programs read the weights, x and wrote y, the reports and the state by address
    _ = waddr^
    _ = step_addresses^
    _ = step_reports^
    _ = decode_addresses^
    _ = decode_reports^
    _ = step_state^
    _ = decode_state^
    _ = c^


def _run_gpu() raises:
    # check_cases_serial runs the cases in order: kinds share the one device context (module docstring)
    def ordered_case(unit: Int) raises:
        run_kind_gpu(unit + 1)
    check_cases_serial("decode_n_tokens", MAMBA_DECODE_N_KINDS, ordered_case)


def test_decode_n_tokens() raises:
    comptime if CPU_COLUMN_BUILD:
        _run_cpu()
    else:
        _run_gpu()


def main() raises:
    # The tests by name, not discover_tests: discovery instantiates every function of the module, and the
    # device column's functions do not compile in the host column build.
    var suite = TestSuite()
    suite.test[test_decode_n_tokens]()
    suite^.run()
