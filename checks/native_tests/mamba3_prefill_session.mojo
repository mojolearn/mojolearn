# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The CPU training image's Mamba entry has no Mamba-3 prefill session,
checked natively on the CPU column (lane mamba3-prefill-handle, 2026-11-12;
narrowed 2026-11-25 by lane owned-weights-deletion, when the host sessions,
which only ever held the NN40 owned snapshot, were deleted).

The publication `_mojolearn_mamba_host` makes at module init
(`publish_neural_mamba_host`, repeated as a second init repeats it); then one
case per rule of the host Mamba entry (mamba/host/native_block_dispatch.mojo):
the capability bits are zero (no session prefill, no optional reports) and
refuse arguments; every former session operation (reserve, release, the owned
ops 8/9/15/16/17, close, create, info) refuses as unavailable before reading
any word; the native owner (core/mamba3_prefill_session.mojo) cannot open a
session on this image. No numerical body runs. No Python participates.

Run in a fresh process.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef, entry_ref_call
from core.neural_entry_table import (
    NeuralEntryKey,
    neural_ref,
    NEURAL_TABLE_MAMBA_TRAIN,
    NEURAL_MAMBA_RESERVE,
    NEURAL_MAMBA_RELEASE,
    NEURAL_MAMBA3_SESSION_CLOSE,
    NEURAL_MAMBA3_SESSION_CREATE,
    NEURAL_MAMBA3_SESSION_INFO,
    NEURAL_MAMBA3_SESSION_CAPS,
)
from core.mamba3_prefill_session import Mamba3PrefillHandle
from mamba.host.neural_entry_host import publish_neural_mamba_host
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require, check_cases_serial


# ---- The host column (was mamba3_prefill_session_cpu_suite.mojo) ----
comptime MAMBA3_SESSION_CASES = 4


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
    raise Error("Mamba-3 host operation " + String(op) + " did not refuse")


def run_case(unit: Int) raises:
    var none = List[Int]()
    if unit == 0:
        # The host image advertises no capability bit.
        require(_call(NEURAL_MAMBA3_SESSION_CAPS, none, none) == 0,
                "the host image advertised a prefill-session capability")
    elif unit == 1:
        # The capability read takes no arguments.
        _expect(NEURAL_MAMBA3_SESSION_CAPS, none, [1], "capabilities take no arguments")
    elif unit == 2:
        # Every former session operation refuses as unavailable, before any
        # word is read (a plausible handle word and addresses are given).
        var ops: List[Int] = [NEURAL_MAMBA_RESERVE, NEURAL_MAMBA_RELEASE, 8, 9, 15, 16, 17,
                              NEURAL_MAMBA3_SESSION_CLOSE, NEURAL_MAMBA3_SESSION_CREATE, NEURAL_MAMBA3_SESSION_INFO]
        var fake: List[Int] = [1, 1, 1, 1, 1, 1]
        var words: List[Int] = [(1 << 32) | 1, 1, 1, 1, 1]
        for i in range(len(ops)):  # small-loop(ten retired session operation codes)
            if ops[i] == NEURAL_MAMBA3_SESSION_CREATE:
                _expect(ops[i], none, none, "unavailable operation")
            else:
                _expect(ops[i], fake, words, "unavailable operation")
    else:
        # The native owner cannot open a session on the host image.
        try:
            var owner = Mamba3PrefillHandle(_entry())
            _ = owner^
        except problem:
            require("unavailable operation" in String(problem), "unexpected refusal: " + String(problem))
            return
        raise Error("a Mamba-3 prefill session opened on the host image")


def _run_cpu() raises:
    publish_neural_mamba_host()
    publish_neural_mamba_host()
    # check_cases_serial runs the cases in order: cases share the image's registry, one after another
    def ordered_case(unit: Int) raises:
        run_case(unit)
    check_cases_serial("prefill_session", MAMBA3_SESSION_CASES, ordered_case)


def test_prefill_session() raises:
    _run_cpu()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
