# SPDX-License-Identifier: Apache-2.0
"""The Transformer block session as a registry handle, reached natively on a
GPU column (lane transformer-session-native, 2026-11-12).

The publication `_mojolearn_transformer` makes at module init
(`publish_neural_transformer_device`, repeated as a second init repeats it);
then one case per session rule of the block entry
(transformer/native_block_dispatch.mojo): CREATE returns distinct handles;
RESERVE/RELEASE keep the busy protocol and a busy session refuses CLOSE and a
second RESERVE; an unreserved session refuses execution; CLOSE releases the
handle and every later use refuses with the registry's text; create with
arguments and a null handle word refuse; the native owner
(core/transformer_block_session.mojo) serves its own entry, closes once and
closes on destruction. No numerical body runs: creation, reservation and the
refusals perform no device work. No Python participates.

Run in a fresh process.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.entry_table import EntryRef, entry_ref_call
from core.neural_entry_table import (
    NeuralEntryKey,
    neural_entry_vendor,
    neural_ref,
    NEURAL_TABLE_BLOCK_FORWARD,
    NEURAL_TABLE_BLOCK_TRAIN,
    NEURAL_BLOCK_RESERVE,
    NEURAL_BLOCK_RELEASE,
    NEURAL_BLOCK_SESSION_FORWARD,
    NEURAL_BLOCK_SESSION_CLOSE,
    NEURAL_BLOCK_SESSION_CREATE,
)
from core.transformer_block_session import TransformerBlockSession, transformer_block_session_of
from transformer.neural_entry_device import publish_neural_transformer_device
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The device column (was transformer_block_session_gpu_suite.mojo) ----
comptime BLOCK_SESSION_CASES = 8


def _entry() raises -> EntryRef:
    var vendor = neural_entry_vendor(String(COMPILED_VENDOR))
    require(vendor > 0, "the block session suite runs on a GPU column")
    return neural_ref[NEURAL_TABLE_BLOCK_FORWARD](NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor))


def _call(op: Int, a: List[Int], words: List[Int]) raises -> Int:
    return entry_ref_call(_entry(), op, a, words, List[Float64]())


def _expect(op: Int, a: List[Int], words: List[Int], text: String) raises:
    try:
        _ = _call(op, a, words)
    except problem:
        require(text in String(problem), "unexpected refusal: " + String(problem))
        return
    raise Error("block session operation " + String(op) + " did not refuse")


def _create() raises -> Int:
    var handle = _call(NEURAL_BLOCK_SESSION_CREATE, List[Int](), List[Int]())
    require(handle > 0 and transformer_block_session_of(handle).generation >= 1, "create returned no handle")
    return handle


def case(unit: Int) raises:
    var none = List[Int]()
    if unit == 0:
        # CREATE: distinct handles; CLOSE releases both.
        var first = _create()
        var second = _create()
        require(first != second, "two creates returned one handle")
        _ = _call(NEURAL_BLOCK_SESSION_CLOSE, none, [first])
        _ = _call(NEURAL_BLOCK_SESSION_CLOSE, none, [second])
    elif unit == 1:
        # A reserved (busy) session refuses CLOSE and keeps its handle.
        var h = _create()
        _ = _call(NEURAL_BLOCK_RESERVE, none, [h])
        _expect(NEURAL_BLOCK_SESSION_CLOSE, none, [h], "transformer: session is busy or closed")
        _ = _call(NEURAL_BLOCK_RELEASE, none, [h])
        _ = _call(NEURAL_BLOCK_SESSION_CLOSE, none, [h])
    elif unit == 2:
        # A second RESERVE of a busy session refuses.
        var h = _create()
        _ = _call(NEURAL_BLOCK_RESERVE, none, [h])
        _expect(NEURAL_BLOCK_RESERVE, none, [h], "transformer: session is busy or closed")
        _ = _call(NEURAL_BLOCK_RELEASE, none, [h])
        _ = _call(NEURAL_BLOCK_SESSION_CLOSE, none, [h])
    elif unit == 3:
        # An unreserved session refuses execution before any numerical body.
        var h = _create()
        _expect(NEURAL_BLOCK_SESSION_FORWARD, none, [h], "execution requires a reserved open session")
        _ = _call(NEURAL_BLOCK_SESSION_CLOSE, none, [h])
    elif unit == 4:
        # CLOSE releases the handle: every later use refuses.
        var h = _create()
        _ = _call(NEURAL_BLOCK_SESSION_CLOSE, none, [h])
        _expect(NEURAL_BLOCK_RESERVE, none, [h], "state is not fitted or was released")
        _expect(NEURAL_BLOCK_SESSION_CLOSE, none, [h], "state is not fitted or was released")
    elif unit == 5:
        # Malformed requests refuse: create with words, a null handle word.
        _expect(NEURAL_BLOCK_SESSION_CREATE, none, [1], "create takes no arguments")
        _expect(NEURAL_BLOCK_RESERVE, none, [0], "null session")
    elif unit == 6:
        # The native owner serves its own entry and the training family's
        # (the GPU image publishes one pointer for both); close is once.
        var owner = TransformerBlockSession(_entry())
        var vendor = neural_entry_vendor(String(COMPILED_VENDOR))
        require(owner.serves(_entry()), "the owner does not serve its own entry")
        require(owner.serves(neural_ref[NEURAL_TABLE_BLOCK_TRAIN](NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor))),
                "the owner does not serve the training family of its image")
        var h = owner.word()
        owner.close()
        require(owner.handle == 0 and not owner.serves(_entry()), "a closed owner still serves")
        owner.close()
        _expect(NEURAL_BLOCK_RESERVE, none, [h], "state is not fitted or was released")
    else:
        # Destroying an open owner closes its session.
        var h = 0
        var owner = TransformerBlockSession(_entry())
        h = owner.word()
        _ = owner^
        _expect(NEURAL_BLOCK_RESERVE, none, [h], "state is not fitted or was released")


def _run_gpu(mut report: Report) raises:
    publish_neural_transformer_device()
    publish_neural_transformer_device()
    var cases = report.cases(BLOCK_SESSION_CASES)
    for unit in range(BLOCK_SESSION_CASES):  # driver-loop: cases share the image's registry, one after another
        try:
            case(unit)
        except problem:
            cases.fail(unit, problem)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("transformer_block_session", cpu=False, gpu=True, serial="", run=run_cases)
