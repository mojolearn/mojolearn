# SPDX-License-Identifier: Apache-2.0
"""The Mamba-1/2 per-call forward without its per-call plumbing equals the
old form bit for bit (lane fix-mamba12-forward-exec, 2026-11-13).

New form (the default; `-D MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD` is the
old one, autograd/block_step_ops.mojo):

  - the per-call block program keeps its L > 1 stages in the step entry and
    refills them with zeros per run (`keep_stages`, `rezero`), and a held
    Mamba-2 run's h_last report is read from the held stages;
  - a stateless forward (every state address 0, what the public doors send
    for `forward(x)` with no state) zeroes the state on the device and reads
    nothing back;
  - the door's y and h_last arrays are unfilled.

Old form, in the same binary: a `MambaBlockProgram` with `keep_stages`
cleared (stages built per run and parked), a host zero state loaded
(`load_state`) and read back (`finish_with_state`) on every run, and
zero-filled outputs.

Cases (each a sequence of runs: L, L again (a held set reused), a shorter
L2, then L (a new set after the shape change)):

  0  Mamba-1 stateless forward through `mamba_block_call_abi` (all-zero
     state addresses) vs the old form from a zero state per run;
  1  Mamba-1 carried-state forward (`mamba_block_call`, the state carried
     across the runs) vs the old form carrying the same state;
  2  Mamba-2 stateless forward (y and the h_last report);
  3  Mamba-2 carried-state forward (y, h_last, the state and buf_len).

The new form's outputs are prefilled with 0xFF words (an unwritten byte
shows as a difference), the old form's with zeros. Every y, report, state
piece and carried integer must agree bit for bit. Seeded corpus fixtures
(Mamba-1 case 1: B 2, L 4, d_model 8; Mamba-2 case 1: B 2, L 4,
d_model 32). The cases run one after another: they share the process's one
device context and the per-block program cache. Run in a fresh process on
a GPU host.
"""
from std.memory import bitcast
from autograd.block_step_ops import OPS_M1_STEP, OPS_M2_STEP, block_state_sizes, m2_report_floats
from mamba.impl.mamba_block_program import (
    MambaBlockProgram, mamba_block_call, mamba_block_call_abi, mamba_block_cache_clear,
)
from mamba.checks.mamba_fixture import corpus_case, corpus_case_weights, corpus_case_x
from mamba.checks.mamba2_fixture import m2_corpus_case, m2_case_weights, m2_case_x

comptime MAMBA12_FORWARD_EXEC_CASES = 4
comptime M12_M1_CASE = 1
comptime M12_M2_CASE = 1
comptime M12_POISON: UInt32 = 0xFFFFFFFF
comptime M12_ZERO: UInt32 = 0


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _filled(n: Int, word: UInt32) -> List[Float32]:
    return List[Float32](length=n, fill=bitcast[DType.float32](word))


def _addresses(pieces: List[List[Float32]]) -> List[Int]:
    var out = List[Int](capacity=len(pieces))
    for i in range(len(pieces)):  # small-loop(at most ten buffer addresses)
        out.append(Int(pieces[i].unsafe_ptr()))
    return out^


def _same(a: List[List[Float32]], b: List[List[Float32]], what: String) raises:
    require(len(a) == len(b), what + ": piece counts differ (" + String(len(a)) + " vs " + String(len(b)) + ")")
    for k in range(len(a)):  # small-loop(the runs' output pieces)
        require(len(a[k]) == len(b[k]), what + ": piece " + String(k) + " lengths differ")
        for i in range(len(a[k])):  # small-loop(fixture piece: at most 163840 floats compared)
            if bitcast[DType.uint32](a[k][i]) != bitcast[DType.uint32](b[k][i]):
                raise Error(what + ": piece " + String(k) + " element " + String(i)
                            + " differs between the new form and the old form")


def _prefix(x: List[Float32], b: Int, l: Int, dm: Int, l2: Int) -> List[Float32]:
    """The first l2 tokens of every row: (B, l2, d_model)."""
    var out = List[Float32](capacity=b * l2 * dm)
    for row in range(b):  # small-loop(two fixture rows)
        for i in range(l2 * dm):  # small-loop(l2 * d_model fixture values)
            out.append(x[row * l * dm + i])
    return out^


def _zero_state(code: Int, b: Int, dm: Int) raises -> List[List[Float32]]:
    var ints: List[Int] = [b, dm]
    var sizes = block_state_sizes(code, ints)
    var out = List[List[Float32]]()
    for i in range(len(sizes)):  # small-loop(two or four state pieces)
        out.append(_filled(sizes[i], M12_ZERO))
    return out^


struct _Block(Movable):
    var code: Int
    var b: Int
    var l: Int
    var dm: Int
    var fs: List[Float32]
    var weights: List[List[Float32]]
    var x: List[Float32]
    var x2: List[Float32]
    var l2: Int

    def __init__(out self, code: Int) raises:
        var fs = List[Float32]()
        var weights = List[List[Float32]]()
        var b: Int
        var l: Int
        var dm: Int
        var x: List[Float32]
        if code == OPS_M1_STEP:
            var c = corpus_case(M12_M1_CASE)
            b = c.b
            l = c.l
            dm = c.d_model
            var w = corpus_case_weights(M12_M1_CASE)
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
            x = corpus_case_x(M12_M1_CASE)
        else:
            var c = m2_corpus_case(M12_M2_CASE)
            b = c.b
            l = c.l
            dm = c.d_model
            fs.append(c.dt_lo)
            fs.append(c.dt_hi)
            var w = m2_case_weights(M12_M2_CASE)
            weights.append(w.norm_w.copy())
            weights.append(w.w_in.copy())
            weights.append(w.conv_w.copy())
            weights.append(w.conv_b.copy())
            weights.append(w.dt_bias.copy())
            weights.append(w.a_log.copy())
            weights.append(w.d_skip.copy())
            weights.append(w.gnorm_w.copy())
            weights.append(w.w_out.copy())
            x = m2_case_x(M12_M2_CASE)
        require(l >= 3, "fixture L must be at least 3 (an L > 1 run and a shorter L2 > 1)")
        var l2 = l - 1
        var x2 = _prefix(x, b, l, dm, l2)
        self.code = code
        self.b = b
        self.l = l
        self.dm = dm
        self.fs = fs^
        self.weights = weights^
        self.x = x^
        self.x2 = x2^
        self.l2 = l2

    def runs(self) -> List[Int]:
        """The run sequence's token counts: L, L, L2, L."""
        var out: List[Int] = [self.l, self.l, self.l2, self.l]
        return out^

    def x_of(self, l: Int) -> Int:
        return Int(self.x.unsafe_ptr()) if l == self.l else Int(self.x2.unsafe_ptr())

    def reports(self) raises -> Int:
        return m2_report_floats(self.b, self.dm) if self.code == OPS_M2_STEP else 0


def _old_form(blk: _Block, carried: Bool) raises -> List[List[Float32]]:
    """The old form: per-run stages (keep_stages cleared), the host state
    loaded and read back every run, zero-filled outputs."""
    var prog = MambaBlockProgram(blk.code, blk.b, blk.dm, blk.fs, _addresses(blk.weights))
    prog.s.steps.keep_stages = False
    var out = List[List[Float32]]()
    var state = _zero_state(blk.code, blk.b, blk.dm)
    var words = List[Int]() if blk.code == OPS_M1_STEP else List[Int](length=1, fill=0)
    var ls = blk.runs()
    try:
        for r in range(len(ls)):  # driver-loop: over the case's four runs (each reads the state the last left)
            if not carried:
                state = _zero_state(blk.code, blk.b, blk.dm)
                words = List[Int]() if blk.code == OPS_M1_STEP else List[Int](length=1, fill=0)
            prog.load_state(_addresses(state), words)
            var y = _filled(blk.b * ls[r] * blk.dm, M12_ZERO)
            var rep = _filled(max(blk.reports(), 1), M12_ZERO)
            var reports = List[Int]()
            if blk.code == OPS_M2_STEP:
                reports.append(Int(rep.unsafe_ptr()))
            prog.run(blk.x_of(ls[r]), Int(y.unsafe_ptr()), ls[r], reports)
            prog.finish_with_state(_addresses(state))
            words = prog.words()
            out.append(y^)
            if blk.code == OPS_M2_STEP:
                out.append(rep^)
            if carried:
                for k in range(len(state)):  # small-loop(two or four state pieces)
                    out.append(state[k].copy())
                for k in range(len(words)):  # small-loop(at most one carried integer)
                    out.append(List[Float32](length=1, fill=Float32(words[k])))
    finally:
        prog.close()
    _ = state^  # the program read and wrote the state by address
    return out^


def _new_stateless(blk: _Block) raises -> List[List[Float32]]:
    """The stateless forward through the per-call ABI: every state address
    0, outputs prefilled with 0xFF."""
    var kind = 1 if blk.code == OPS_M1_STEP else 2
    var ns = len(_zero_state(blk.code, blk.b, blk.dm))
    var dt_lo = blk.fs[0] if kind == 2 else Float32(0)
    var dt_hi = blk.fs[1] if kind == 2 else Float32(0)
    var waddr = _addresses(blk.weights)
    var out = List[List[Float32]]()
    var ls = blk.runs()
    for r in range(len(ls)):  # driver-loop: over the case's four runs
        var y = _filled(blk.b * ls[r] * blk.dm, M12_POISON)
        var rep = _filled(max(blk.reports(), 1), M12_POISON)
        var a = List[Int]()
        a.append(blk.x_of(ls[r]))
        for i in range(len(waddr)):  # small-loop(nine or ten weight addresses)
            a.append(waddr[i])
        for _ in range(ns):  # small-loop(two or four state addresses, all 0: the stateless call)
            a.append(0)
        a.append(Int(y.unsafe_ptr()))
        if kind == 2:
            a.append(Int(rep.unsafe_ptr()))
        _ = mamba_block_call_abi(kind, a, blk.b, ls[r], blk.dm, 0, 0, dt_lo, dt_hi)
        out.append(y^)
        if kind == 2:
            out.append(rep^)
    return out^


def _new_carried(blk: _Block) raises -> List[List[Float32]]:
    """The carried-state forward on the cached per-call program (held
    stages), outputs prefilled with 0xFF."""
    var waddr = _addresses(blk.weights)
    var state = _zero_state(blk.code, blk.b, blk.dm)
    var words = List[Int]() if blk.code == OPS_M1_STEP else List[Int](length=1, fill=0)
    var out = List[List[Float32]]()
    var ls = blk.runs()
    for r in range(len(ls)):  # driver-loop: over the case's four runs (each reads the state the last left)
        var y = _filled(blk.b * ls[r] * blk.dm, M12_POISON)
        var rep = _filled(max(blk.reports(), 1), M12_POISON)
        var reports = List[Int]()
        if blk.code == OPS_M2_STEP:
            reports.append(Int(rep.unsafe_ptr()))
        words = mamba_block_call(blk.code, waddr, _addresses(state), words, blk.x_of(ls[r]), Int(y.unsafe_ptr()),
                                 reports, blk.b, ls[r], blk.dm, blk.fs, False)
        out.append(y^)
        if blk.code == OPS_M2_STEP:
            out.append(rep^)
        for k in range(len(state)):  # small-loop(two or four state pieces)
            out.append(state[k].copy())
        for k in range(len(words)):  # small-loop(at most one carried integer)
            out.append(List[Float32](length=1, fill=Float32(words[k])))
    _ = state^  # the calls read and wrote the state by address
    return out^


def case(unit: Int) raises:
    var code = OPS_M1_STEP if unit < 2 else OPS_M2_STEP
    var name = String("Mamba-1") if code == OPS_M1_STEP else String("Mamba-2")
    var blk = _Block(code)
    if unit % 2 == 0:
        _same(_new_stateless(blk), _old_form(blk, False), name + " stateless forward")
    else:
        _same(_new_carried(blk), _old_form(blk, True), name + " carried-state forward")
    _ = blk^  # the programs read the weights and x by address


def run_checks() raises:
    var failed = 0
    for unit in range(MAMBA12_FORWARD_EXEC_CASES):  # driver-loop: cases share one device and the program cache (module docstring)
        try:
            case(unit)
        except problem:
            failed += 1
            print("FAIL Mamba-1/2 forward plumbing (GPU) case", unit, String(problem))
    mamba_block_cache_clear()
    require(failed == 0, "Mamba-1/2 forward plumbing (GPU) cases failed: " + String(failed))
    print("PASS Mamba-1/2 forward plumbing (GPU)", MAMBA12_FORWARD_EXEC_CASES)
