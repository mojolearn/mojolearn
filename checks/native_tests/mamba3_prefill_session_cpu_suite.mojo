# SPDX-License-Identifier: Apache-2.0
"""The Mamba-3 prefill session as a registry handle, reached natively on the
CPU column (lane mamba3-prefill-handle, 2026-11-12).

The publication `_mojolearn_mamba_host` makes at module init
(`publish_neural_mamba_host`, repeated as a second init repeats it); then one
case per session rule of the host Mamba entry
(mamba/host/native_block_dispatch.mojo): CREATE returns distinct handles;
RESERVE/RELEASE keep the busy protocol and a busy session refuses CLOSE, INFO
and a second RESERVE; an unreserved session refuses execution; CLOSE releases
the handle and every later use refuses with the registry's text; malformed
requests refuse; INFO of a fresh session writes six zero counters; the
capability bits name no non-owned session prefill and no optional reports;
the native owner (core/mamba3_prefill_session.mojo) serves its own entry,
closes once and closes on destruction. No numerical body runs. No Python
participates.

Run in a fresh process.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef, entry_ref_call
from core.neural_entry_table import (
    NeuralEntryKey, neural_ref, NEURAL_TABLE_MAMBA_TRAIN, NEURAL_MAMBA_RESERVE, NEURAL_MAMBA_RELEASE,
    NEURAL_MAMBA3_OWNED_FORWARD, NEURAL_MAMBA3_SESSION_CLOSE, NEURAL_MAMBA3_SESSION_CREATE,
    NEURAL_MAMBA3_SESSION_INFO, NEURAL_MAMBA3_SESSION_CAPS, MAMBA3_CAP_SESSION_PREFILL,
    MAMBA3_CAP_REPORTS_OPTIONAL,
)
from core.mamba3_prefill_session import Mamba3PrefillHandle, mamba3_prefill_session_of
from mamba.host.neural_entry_host import publish_neural_mamba_host

comptime MAMBA3_SESSION_CASES = 9


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _entry() raises -> EntryRef:
    return neural_ref[NEURAL_TABLE_MAMBA_TRAIN](NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))


def _call(op: Int, a: List[Int], words: List[Int]) raises -> Int:
    return entry_ref_call(_entry(), op, a, words, List[Float64]())


def _expect(op: Int, a: List[Int], words: List[Int], text: String) raises:
    try:
        _ = _call(op, a, words)
    except problem:
        require(text in String(problem), "unexpected refusal: " + String(problem))
        return
    raise Error("Mamba-3 session operation " + String(op) + " did not refuse")


def _create() raises -> Int:
    var handle = _call(NEURAL_MAMBA3_SESSION_CREATE, List[Int](), List[Int]())
    require(handle > 0 and mamba3_prefill_session_of(handle).generation >= 1, "create returned no handle")
    return handle


def case(unit: Int) raises:
    var none = List[Int]()
    if unit == 0:
        # CREATE: distinct handles; CLOSE releases both.
        var first = _create()
        var second = _create()
        require(first != second, "two creates returned one handle")
        _ = _call(NEURAL_MAMBA3_SESSION_CLOSE, none, [first])
        _ = _call(NEURAL_MAMBA3_SESSION_CLOSE, none, [second])
    elif unit == 1:
        # A reserved (busy) session refuses CLOSE and INFO and keeps its handle.
        var h = _create()
        var counters = List[Int](length=6, fill=0)
        _ = _call(NEURAL_MAMBA_RESERVE, none, [h])
        _expect(NEURAL_MAMBA3_SESSION_CLOSE, none, [h], "session busy")
        _expect(NEURAL_MAMBA3_SESSION_INFO, [Int(counters.unsafe_ptr())], [h], "session busy")
        _ = _call(NEURAL_MAMBA_RELEASE, none, [h])
        _ = _call(NEURAL_MAMBA3_SESSION_CLOSE, none, [h])
        _ = counters^
    elif unit == 2:
        # A second RESERVE of a busy session refuses.
        var h = _create()
        _ = _call(NEURAL_MAMBA_RESERVE, none, [h])
        _expect(NEURAL_MAMBA_RESERVE, none, [h], "session busy or unusable")
        _ = _call(NEURAL_MAMBA_RELEASE, none, [h])
        _ = _call(NEURAL_MAMBA3_SESSION_CLOSE, none, [h])
    elif unit == 3:
        # An unreserved session refuses execution before any numerical body.
        var h = _create()
        var fake: List[Int] = [1, 1, 1, 1, 1, 1]
        _expect(NEURAL_MAMBA3_OWNED_FORWARD, fake, [h, 1, 1, 1, 1], "execution requires a reserved usable owner")
        _ = _call(NEURAL_MAMBA3_SESSION_CLOSE, none, [h])
    elif unit == 4:
        # CLOSE releases the handle: every later use refuses.
        var h = _create()
        _ = _call(NEURAL_MAMBA3_SESSION_CLOSE, none, [h])
        _expect(NEURAL_MAMBA_RESERVE, none, [h], "state is not fitted or was released")
        _expect(NEURAL_MAMBA3_SESSION_CLOSE, none, [h], "state is not fitted or was released")
    elif unit == 5:
        # Malformed requests refuse: create with words, a null handle word.
        _expect(NEURAL_MAMBA3_SESSION_CREATE, none, [1], "create takes no arguments")
        _expect(NEURAL_MAMBA_RESERVE, none, [0], "null session")
    elif unit == 6:
        # INFO of a fresh session: six zero counters, written natively.
        var h = _create()
        var counters = List[Int](length=6, fill=-1)
        _ = _call(NEURAL_MAMBA3_SESSION_INFO, [Int(counters.unsafe_ptr())], [h])
        for i in range(6):  # small-loop(six session counters)
            require(counters[i] == 0, "a fresh session reported a nonzero counter")
        _ = _call(NEURAL_MAMBA3_SESSION_CLOSE, none, [h])
    elif unit == 7:
        # The host image has no non-owned session prefill and no optional reports.
        var caps = _call(NEURAL_MAMBA3_SESSION_CAPS, none, none)
        require((caps & (MAMBA3_CAP_SESSION_PREFILL | MAMBA3_CAP_REPORTS_OPTIONAL)) == 0,
                "the host image advertised a GPU-only session capability")
    else:
        # The native owner serves its own entry, closes once, and closes on
        # destruction.
        var owner = Mamba3PrefillHandle(_entry())
        require(owner.serves(_entry()), "the owner does not serve its own entry")
        var h = owner.word()
        owner.close()
        require(owner.handle == 0 and not owner.serves(_entry()), "a closed owner still serves")
        owner.close()
        _expect(NEURAL_MAMBA_RESERVE, none, [h], "state is not fitted or was released")
        var dropped = Mamba3PrefillHandle(_entry())
        var g = dropped.word()
        _ = dropped^
        _expect(NEURAL_MAMBA_RESERVE, none, [g], "state is not fitted or was released")


def run_checks() raises:
    publish_neural_mamba_host()
    publish_neural_mamba_host()
    var failed = 0
    for unit in range(MAMBA3_SESSION_CASES):  # driver-loop: cases share the image's registry, one after another
        try:
            case(unit)
        except problem:
            failed += 1
            print("FAIL Mamba-3 prefill session case", unit, String(problem))
    require(failed == 0, "Mamba-3 prefill session cases failed: " + String(failed))
    print("PASS Mamba-3 prefill session", MAMBA3_SESSION_CASES)
