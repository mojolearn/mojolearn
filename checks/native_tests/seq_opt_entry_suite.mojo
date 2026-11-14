# SPDX-License-Identifier: Apache-2.0
"""The sequence optimizer table's host entry, reached natively (lane
calltable-byte-lm, 2026-11-11).

Ordered preconditions first (they share the one process table): a caller
refuses before publication, then the publication `_mojolearn_x_sequence_host`
makes at module init (`publish_seq_opt_host_entries`), a tier key and a
schema mismatch refuse, and an unknown operation refuses through the entry's
sink. Then one case per published operation: each runs the operation
through `seq_opt_call` (the table pointer) and the same typed host function
directly on a twin fixture, and requires the same count, the same state
words and the same float32 bits (or the same refusal text). No Python
participates.

The cases are independent units and run across the native host worker pool.
Run in a fresh process (the first check needs an unpublished table).
"""
from core.entry_table import entry_ref_same
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.byte_lm_entry_table import ByteLmEntryKey, byte_lm_table_entry, BYTE_LM_TABLE_SEQ_OPT
from sequence.exec import HostExec
from sequence.ops import OPT_RMSPROP
from sequence.opt_entry_abi import (
    seq_opt_ref, seq_opt_call, SEQ_OPT_ABI, SO_OP_COUNT, SO_INIT, SO_STEP, SO_LAMB_STEP, SO_ADAFACTOR_STEP, SO_STATE,
    SO_EXPORT, SO_IMPORT, SO_CLOSE,
)
from sequence.opt_entry_host import publish_seq_opt_host_entries
from sequence.opt_public import (
    seq_adafactor_step_on, seq_lamb_step_on, seq_optimizer_close_on, seq_optimizer_export_on,
    seq_optimizer_import_on, seq_optimizer_init, seq_optimizer_state_on, seq_optimizer_step_on,
)

comptime SEQ_OPT_ENTRY_CASES = 8
comptime N = 4


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> ByteLmEntryKey:
    return ByteLmEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


struct Twin(Movable):
    """One optimizer's buffers: 9 state words, a parameter, a gradient, three
    state slots, three scalars and export outputs."""

    var words: List[Int64]
    var param: List[Float32]
    var grad: List[Float32]
    var s1: List[Float32]
    var s2: List[Float32]
    var s3: List[Float32]
    var sc: List[Float32]
    var x1: List[Float32]
    var x2: List[Float32]
    var x3: List[Float32]
    var scx: List[Float32]

    def __init__(out self):
        self.words = List[Int64](length=9, fill=0)
        self.param = [0.5, -1.0, 2.0, 0.25]
        self.grad = [0.1, -0.2, 0.3, -0.4]
        self.s1 = List[Float32](length=N, fill=0.0)
        self.s2 = List[Float32](length=N, fill=0.0)
        self.s3 = List[Float32](length=N, fill=0.0)
        self.sc = List[Float32](length=3, fill=1.0)
        self.x1 = List[Float32](length=N, fill=0.0)
        self.x2 = List[Float32](length=N, fill=0.0)
        self.x3 = List[Float32](length=N, fill=0.0)
        self.scx = List[Float32](length=3, fill=0.0)

    def w(mut self) -> Int:
        return Int(self.words.unsafe_ptr())

    def init_ip(self) -> List[Int]:
        return [0, 1, N]

    def step_addrs(mut self) -> List[Int]:
        return [Int(self.param.unsafe_ptr()), Int(self.grad.unsafe_ptr()), Int(self.s1.unsafe_ptr()),
                Int(self.s2.unsafe_ptr()), Int(self.s3.unsafe_ptr()), Int(self.sc.unsafe_ptr())]

    def copy_addrs(mut self) -> List[Int]:
        return [Int(self.s1.unsafe_ptr()), Int(self.s2.unsafe_ptr()), Int(self.s3.unsafe_ptr()),
                Int(self.x1.unsafe_ptr()), Int(self.x2.unsafe_ptr()), Int(self.x3.unsafe_ptr()),
                Int(self.sc.unsafe_ptr()), Int(self.scx.unsafe_ptr())]


def _step_ip() -> List[Int]:
    # [J, dev, gk_0, size_0, kind, flags]: one host RMSprop tensor, centered off
    return [1, 0, 0, N, OPT_RMSPROP, 0]


def _step_fp() -> List[Float64]:
    # [lr, f1 (alpha), f2, eps, weight_decay, f7 (momentum), initial]: the door's [lr] + _fp
    return [0.01, 0.99, 0.0, 1e-8, 0.0, 0.0, 0.0]


def _with(var head: List[Int], tail: List[Int]) -> List[Int]:
    head.extend(tail.copy())
    return head^


def _same_f32(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + " length")
    for i in range(len(a)):  # small-loop(a four-cell fixture)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), what + " bits")


def _same(a: Twin, b: Twin, what: String) raises:
    for i in range(9):  # small-loop(the nine state words)
        require(a.words[i] == b.words[i], what + " state word " + String(i))
    _same_f32(a.param, b.param, what + " param")
    _same_f32(a.s1, b.s1, what + " s1")
    _same_f32(a.s2, b.s2, what + " s2")
    _same_f32(a.s3, b.s3, what + " s3")
    _same_f32(a.sc, b.sc, what + " scalars")
    _same_f32(a.x1, b.x1, what + " x1")
    _same_f32(a.scx, b.scx, what + " scalars out")


def _ready(mut t: Twin) raises:
    # init and one step through the typed functions: the twin's common start
    var ex = HostExec()
    _ = seq_optimizer_init(t.w(), t.init_ip())
    _ = seq_optimizer_step_on(ex, t.w(), t.step_addrs(), _step_ip(), _step_fp())


def precondition_unpublished() raises:
    var t = Twin()
    try:
        _ = seq_opt_call(seq_opt_ref(_key()), SO_INIT, List[Int](), _with([t.w()], t.init_ip()), List[Float64]())
    except problem:
        _refuses(problem, "provider is not published")
        _ = t^
        return
    raise Error("an unpublished sequence optimizer table did not refuse")


def ref_resolution() raises:
    """Lane entry-table-zero-alloc: the table resolves once into a ref that a
    connection owner reuses (the same pointer and layout word every time),
    and a layout mismatch refuses at resolution, never at a call."""
    var first = seq_opt_ref(_key())
    var second = seq_opt_ref(_key())
    require(entry_ref_same(first, second), "sequence optimizer: a second resolution returned another entry")
    try:
        _ = byte_lm_table_entry[BYTE_LM_TABLE_SEQ_OPT](_key(), 0, String(SEQ_OPT_ABI), SO_OP_COUNT + 1)
        raise Error("sequence optimizer: a layout mismatch did not refuse at resolution")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
        require("ABI schema differs" in String(problem), "sequence optimizer: unexpected refusal: " + String(problem))


def preconditions_published() raises:
    ref_resolution()
    var t = Twin()
    try:
        _ = seq_opt_call(seq_opt_ref(ByteLmEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0)), SO_INIT, List[Int](),
                         _with([t.w()], t.init_ip()), List[Float64]())
        raise Error("a mismatched tier key did not refuse")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
    try:
        _ = byte_lm_table_entry[BYTE_LM_TABLE_SEQ_OPT](_key(), 0, String("mojolearn.sequence.other.v1"), SO_OP_COUNT)
        raise Error("a mismatched schema did not refuse")
    except problem:
        _refuses(problem, "ABI schema differs")
    try:
        _ = seq_opt_call(seq_opt_ref(_key()), SO_OP_COUNT, List[Int](), [t.w()], List[Float64]())
        raise Error("an unknown operation did not refuse")
    except problem:
        _refuses(problem, "unknown operation")
    _ = t^


def run_case(unit: Int) raises:
    var k = seq_opt_ref(_key())
    var ex = HostExec()
    var a = Twin()
    var b = Twin()
    if unit == SO_INIT:
        var n = seq_opt_call(k, SO_INIT, List[Int](), _with([a.w()], a.init_ip()), List[Float64]())
        require(n == seq_optimizer_init(b.w(), b.init_ip()) and n == N, "init count")
        _same(a, b, "init")
    elif unit == SO_STEP:
        _ = seq_optimizer_init(a.w(), a.init_ip())
        _ = seq_optimizer_init(b.w(), b.init_ip())
        var n = seq_opt_call(k, SO_STEP, a.step_addrs(), _with([a.w(), 0], _step_ip()), _step_fp())
        require(n == seq_optimizer_step_on(ex, b.w(), b.step_addrs(), _step_ip(), _step_fp()), "step count")
        _same(a, b, "step")
    elif unit == SO_LAMB_STEP or unit == SO_ADAFACTOR_STEP:
        # A step with no tensors refuses in SeqStep: the entry must route to
        # the same typed function and return its exact refusal text.
        var bad: List[Int] = [0, 0]
        var expected = String("")
        try:
            if unit == SO_LAMB_STEP:
                _ = seq_lamb_step_on(ex, b.w(), List[Int](), bad, List[Float64]())
            else:
                _ = seq_adafactor_step_on(ex, b.w(), List[Int](), bad, List[Float64]())
        except problem:
            expected = String(problem)
        require(expected != "", "the typed step did not refuse")
        try:
            _ = seq_opt_call(k, unit, List[Int](), _with([a.w(), 0], bad), List[Float64]())
            raise Error("the entry step did not refuse")
        except problem:
            require(expected in String(problem), "entry refusal differs: " + String(problem))
    elif unit == SO_STATE:
        _ready(a)
        _ready(b)
        var n = seq_opt_call(k, SO_STATE, List[Int](), [a.w(), 0], List[Float64]())
        require(n == seq_optimizer_state_on(b.w()), "state count")
        _same(a, b, "state")
    elif unit == SO_EXPORT:
        _ready(a)
        _ready(b)
        var ip: List[Int] = [3, N, 3, 3, 1, 0, 0]
        var n = seq_opt_call(k, SO_EXPORT, a.copy_addrs(), _with([a.w(), 0], ip), List[Float64]())
        require(n == seq_optimizer_export_on(b.w(), b.copy_addrs(), ip) and n == 3, "export count")
        _same(a, b, "export")
    elif unit == SO_IMPORT:
        _ready(a)
        _ready(b)
        a.x1 = [1.0, 2.0, 3.0, 4.0]
        b.x1 = [1.0, 2.0, 3.0, 4.0]
        var ip: List[Int] = [3, N, 3, 1, 5, 0, 0]
        var addrs_a = a.copy_addrs()
        var addrs_b = b.copy_addrs()
        # m = 1 source: [slots.., x_0, scalars, scalars_x]
        var ia: List[Int] = [addrs_a[0], addrs_a[1], addrs_a[2], addrs_a[3], addrs_a[6], addrs_a[7]]
        var ib: List[Int] = [addrs_b[0], addrs_b[1], addrs_b[2], addrs_b[3], addrs_b[6], addrs_b[7]]
        var n = seq_opt_call(k, SO_IMPORT, ia, _with([a.w(), 0], ip), List[Float64]())
        require(n == seq_optimizer_import_on(b.w(), ib, ip) and n == 1, "import count")
        require(a.words[0] == 5, "import step word")
        _same(a, b, "import")
    elif unit == SO_CLOSE:
        _ready(a)
        _ready(b)
        var n = seq_opt_call(k, SO_CLOSE, List[Int](), [a.w(), 0, 1], List[Float64]())
        require(n == seq_optimizer_close_on(b.w()), "close count")
        _same(a, b, "close")
    else:
        raise Error("unknown sequence optimizer entry case")
    _ = a^
    _ = b^


def run_checks() raises:
    precondition_unpublished()
    # The publication _mojolearn_x_sequence_host makes at module init.
    publish_seq_opt_host_entries()
    preconditions_published()
    var failures = List[Int](length=SEQ_OPT_ENTRY_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            run_case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL sequence optimizer entry case", unit, String(problem))
    host_parallelize(task, SEQ_OPT_ENTRY_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "sequence optimizer entry cases failed: " + String(failed))
    print("PASS sequence optimizer entry", SEQ_OPT_ENTRY_CASES)
