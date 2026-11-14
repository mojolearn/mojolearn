# SPDX-License-Identifier: Apache-2.0
"""The CausalLM session table, reached natively on a GPU column (lane
causal-lm-sessions-native, 2026-11-11).

The publications `_mojolearn_transformer` and `_mojolearn_mamba` make at
module init (`publish_causal_transformer_device`,
`publish_causal_mamba_device`), repeated as a second init repeats them; then
one case per session-table operation (CREATE, OPEN, OPEN15, EXPORT, LOAD,
CLOSE of the layer decode session; CREATE, OPEN, OPEN15, RUN, FORWARD, CLOSE
of the model session; PROGRAM_READ and PROGRAM_RELEASE), the Mamba entry's
program-only refusal, and the handle owner's close. Each case proves the
published pointer is the image's session dispatch on a registry handle: a
returned handle, the dispatch's own refusal text for a closed or malformed
session, and the registry's refusal of a released handle. The numerical
session bodies are the hand-tuned kernels the public generation tests cover;
these cases add no numerics. No Python participates.

Device cases share one device context and run one after another.
Run in a fresh process.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.causal_lm_entry_table import CausalEntryKey, causal_entry_vendor
from core.causal_lm_entry_table_sessions import (
    causal_sessions_ref,
    causal_sessions_call,
    causal_session_handle_of,
    CAUSAL_SESSIONS_TRANSFORMER,
    CAUSAL_SESSIONS_MAMBA,
    CAUSAL_SESSION_DECODE_CREATE,
    CAUSAL_SESSION_DECODE_OPEN,
    CAUSAL_SESSION_DECODE_OPEN15,
    CAUSAL_SESSION_DECODE_EXPORT,
    CAUSAL_SESSION_DECODE_LOAD,
    CAUSAL_SESSION_DECODE_CLOSE,
    CAUSAL_SESSION_LM_CREATE,
    CAUSAL_SESSION_LM_OPEN,
    CAUSAL_SESSION_LM_OPEN15,
    CAUSAL_SESSION_LM_RUN,
    CAUSAL_SESSION_LM_FORWARD,
    CAUSAL_SESSION_LM_CLOSE,
    CAUSAL_SESSION_PROGRAM_READ,
    CAUSAL_SESSION_PROGRAM_RELEASE,
)
from core.causal_session_handles import CausalSessionHandles
from core.entry_table import entry_ref_same
from transformer.causal_entry_device import publish_causal_transformer_device
from mamba.causal_entry_device import publish_causal_mamba_device
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The device column (was causal_lm_session_gpu_suite.mojo) ----
comptime CAUSAL_SESSION_CASES = 16


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def _key() raises -> CausalEntryKey:
    var vendor = causal_entry_vendor(String(COMPILED_VENDOR))
    require(vendor > 0, "the session suite runs on a GPU column")
    return CausalEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor)


def _call(family: Int, op: Int, a: List[Int], words: List[Int], f: List[Float64]) raises -> Int:
    return causal_sessions_call(causal_sessions_ref(family, _key()), op, a, words, f)


def _t(op: Int, a: List[Int], words: List[Int], f: List[Float64]) raises -> Int:
    return _call(CAUSAL_SESSIONS_TRANSFORMER, op, a, words, f)


def _expect(op: Int, a: List[Int], words: List[Int], f: List[Float64], text: String) raises:
    try:
        _ = _t(op, a, words, f)
    except problem:
        _refuses(problem, text)
        return
    raise Error("session operation " + String(op) + " did not refuse")


def _decode() raises -> Int:
    var handle = _t(CAUSAL_SESSION_DECODE_CREATE, List[Int](), List[Int](), List[Float64]())
    require(handle > 0 and causal_session_handle_of(handle).generation >= 1, "decode create returned no handle")
    return handle


def _lm() raises -> Int:
    var handle = _t(CAUSAL_SESSION_LM_CREATE, List[Int](), List[Int](), List[Float64]())
    require(handle > 0, "lm create returned no handle")
    return handle


def _released(op: Int, handle: Int) raises:
    _expect(op, List[Int](), [handle], List[Float64](), "state is not fitted or was released")


def run_case(unit: Int) raises:
    var none = List[Int]()
    var nof = List[Float64]()
    if unit == 0:
        # DECODE_CREATE: a closed session behind a fresh handle.
        var first = _decode()
        var second = _decode()
        require(first != second, "two creates returned one handle")
        _ = _t(CAUSAL_SESSION_DECODE_CLOSE, none, [first], nof)
        _ = _t(CAUSAL_SESSION_DECODE_CLOSE, none, [second], nof)
    elif unit == 1:
        # DECODE_OPEN: the open dispatch's span refusal; the session stays closed.
        var h = _decode()
        _expect(CAUSAL_SESSION_DECODE_OPEN, none, [h], nof, "transformer decode open: invalid address or scalar span")
        _expect(CAUSAL_SESSION_DECODE_EXPORT, [1, 1], [h], nof, "transformer decode session: not open")
        _ = _t(CAUSAL_SESSION_DECODE_CLOSE, none, [h], nof)
    elif unit == 2:
        # DECODE_OPEN15: the packed open's span refusal.
        var h = _decode()
        _expect(CAUSAL_SESSION_DECODE_OPEN15, [1, 2, 3], [h, 1], nof, "transformer decode open: invalid address or scalar span")
        _ = _t(CAUSAL_SESSION_DECODE_CLOSE, none, [h], nof)
    elif unit == 3:
        # DECODE_EXPORT on a closed session refuses as the Python entry did.
        var h = _decode()
        _expect(CAUSAL_SESSION_DECODE_EXPORT, [1, 1], [h], nof, "transformer decode session: not open")
        _ = _t(CAUSAL_SESSION_DECODE_CLOSE, none, [h], nof)
    elif unit == 4:
        # DECODE_LOAD on a closed session refuses; not left busy (close succeeds).
        var h = _decode()
        _expect(CAUSAL_SESSION_DECODE_LOAD, [1, 1], [h, 0], nof, "transformer decode session: not open")
        _ = _t(CAUSAL_SESSION_DECODE_CLOSE, none, [h], nof)
    elif unit == 5:
        # DECODE_CLOSE releases the handle: every later use refuses.
        var h = _decode()
        require(_t(CAUSAL_SESSION_DECODE_CLOSE, none, [h], nof) == 0, "decode close result")
        _released(CAUSAL_SESSION_DECODE_CLOSE, h)
        _expect(CAUSAL_SESSION_DECODE_EXPORT, [1, 1], [h], nof, "state is not fitted or was released")
    elif unit == 6:
        # LM_CREATE: a closed model session behind a fresh handle.
        var h = _lm()
        _ = _t(CAUSAL_SESSION_LM_CLOSE, none, [h], nof)
    elif unit == 7:
        # LM_OPEN: B, V, D and eps are required.
        var h = _lm()
        _expect(CAUSAL_SESSION_LM_OPEN, [1, 1, 0], [h, 1, 1], [1e-5], "causal lm session open: expected B, V, D and eps")
        _ = _t(CAUSAL_SESSION_LM_CLOSE, none, [h], nof)
    elif unit == 8:
        # LM_OPEN15: five addresses; three refuse with the open dispatch's text.
        var h = _lm()
        _expect(CAUSAL_SESSION_LM_OPEN15, [1, 1, 1], [h, 1, 1, 1], [1e-5], "causal lm session open: invalid descriptor")
        _ = _t(CAUSAL_SESSION_LM_CLOSE, none, [h], nof)
    elif unit == 9:
        # LM_RUN on a closed model session: the grouped reservation refuses
        # and releases nothing it did not reserve.
        var h = _lm()
        var layer = _decode()
        _expect(CAUSAL_SESSION_LM_RUN, [1, 1, 0], [h, 1, layer, 1, 1, 0], nof,
            "causal lm session: busy, not open or lost after a failed call")
        _expect(CAUSAL_SESSION_LM_RUN, [1, 1, 0], [h, 2, layer], nof, "causal lm resident: invalid layer handle span")
        _ = _t(CAUSAL_SESSION_DECODE_CLOSE, none, [layer], nof)
        _ = _t(CAUSAL_SESSION_LM_CLOSE, none, [h], nof)
    elif unit == 10:
        # LM_FORWARD on a closed model session refuses the same way.
        var h = _lm()
        var layer = _decode()
        _expect(CAUSAL_SESSION_LM_FORWARD, [1, 1], [h, 1, layer, 1], nof,
            "causal lm session: busy, not open or lost after a failed call")
        _ = _t(CAUSAL_SESSION_DECODE_CLOSE, none, [layer], nof)
        _ = _t(CAUSAL_SESSION_LM_CLOSE, none, [h], nof)
    elif unit == 11:
        # LM_CLOSE releases the handle.
        var h = _lm()
        require(_t(CAUSAL_SESSION_LM_CLOSE, none, [h], nof) == 0, "lm close result")
        _released(CAUSAL_SESSION_LM_CLOSE, h)
    elif unit == 12:
        # PROGRAM_READ: the read descriptor is admitted before any program opens.
        _expect(CAUSAL_SESSION_PROGRAM_READ, none, [0], nof, "causal lm program: invalid read descriptor")
        var out = List[Int](length=8, fill=0)
        _expect(CAUSAL_SESSION_PROGRAM_READ, none, [0, Int(out.unsafe_ptr()), 2], nof, "causal lm program: invalid read descriptor")
        _ = out
    elif unit == 13:
        # PROGRAM_RELEASE of a handle no program holds.
        _released(CAUSAL_SESSION_PROGRAM_RELEASE, (1 << 32) | 4095)
    elif unit == 14:
        # The Mamba image's entry serves only the decode program.
        try:
            _ = _call(CAUSAL_SESSIONS_MAMBA, CAUSAL_SESSION_DECODE_CREATE, none, none, nof)
            raise Error("the Mamba session entry accepted a Transformer session operation")
        except problem:
            _refuses(problem, "a Mamba image serves only the decode program")
        try:
            _ = _call(CAUSAL_SESSIONS_MAMBA, CAUSAL_SESSION_PROGRAM_RELEASE, none, [(1 << 32) | 4095], nof)
            raise Error("the Mamba program registry released an unknown handle")
        except problem:
            _refuses(problem, "state is not fitted or was released")
    else:
        # The handle owner closes the model session, then every layer session.
        var lm = _lm()
        var layer = _decode()
        var owner = CausalSessionHandles(_key(), CAUSAL_SESSIONS_TRANSFORMER, lm, CAUSAL_SESSION_LM_CLOSE)
        # Lane entry-table-zero-alloc: the owner holds the entry resolved at
        # construction, the same ref a fresh resolution returns.
        require(entry_ref_same(owner.entry, causal_sessions_ref(CAUSAL_SESSIONS_TRANSFORMER, _key())),
                "the session owner does not hold the resolved session entry")
        owner.layers.append(layer)
        owner.close()
        require(owner.primary == 0 and len(owner.layers) == 0, "owner kept closed handles")
        _released(CAUSAL_SESSION_LM_CLOSE, lm)
        _released(CAUSAL_SESSION_DECODE_CLOSE, layer)


def _run_gpu(mut report: Report) raises:
    publish_causal_transformer_device()
    publish_causal_mamba_device()
    publish_causal_transformer_device()
    publish_causal_mamba_device()
    var cases = report.cases(CAUSAL_SESSION_CASES)
    for unit in range(CAUSAL_SESSION_CASES):  # driver-loop: device cases share one device context
        try:
            run_case(unit)
        except problem:
            cases.fail(unit, problem)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("causal_lm_session", cpu=False, gpu=True, serial="", run=run_cases)
