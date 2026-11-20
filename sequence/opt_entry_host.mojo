# SPDX-License-Identifier: Apache-2.0
"""The sequence optimizer entry of the CPU binding `_mojolearn_x_sequence_host`
and its table publication (lane calltable-byte-lm, 2026-11-11). ABI in
`sequence/opt_entry_abi.mojo`. The same compiled functions and argument
words as providers/python/sequence_opt_public_host_pyapi.mojo (a `HostExec`,
nothing resident; `handles` is unused on the host, as there)."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.shared_slot import SharedSlot
from core.byte_lm_entry_table import ByteLmEntryKey, byte_lm_table_publish, BYTE_LM_TABLE_SEQ_OPT
from sequence.exec import HostExec
from sequence.opt_entry_abi import (
    SEQ_OPT_ABI, SO_OP_COUNT, SO_INIT, SO_STEP, SO_LAMB_STEP, SO_ADAFACTOR_STEP, SO_STATE, SO_EXPORT,
    SO_IMPORT, SO_CLOSE, seq_opt_rest,
)
from sequence.opt_public import (
    seq_adafactor_step_on,
    seq_lamb_step_on,
    seq_optimizer_close_on,
    seq_optimizer_export_on,
    seq_optimizer_import_on,
    seq_optimizer_init,
    seq_optimizer_state_on,
    seq_optimizer_step_on,
)


def seq_opt_host_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == SO_INIT:
        return seq_optimizer_init(p[0], seq_opt_rest(p, 1))
    if len(p) < 2:
        raise Error("sequence optimizer entry: missing state and handle words")
    if op == SO_STEP:
        var ex = HostExec()
        return seq_optimizer_step_on(ex, p[0], a, seq_opt_rest(p, 2), f)
    if op == SO_LAMB_STEP:
        var ex = HostExec()
        return seq_lamb_step_on(ex, p[0], a, seq_opt_rest(p, 2), f)
    if op == SO_ADAFACTOR_STEP:
        var ex = HostExec()
        return seq_adafactor_step_on(ex, p[0], a, seq_opt_rest(p, 2), f)
    if op == SO_STATE:
        return seq_optimizer_state_on(p[0])
    if op == SO_EXPORT:
        return seq_optimizer_export_on(p[0], a, seq_opt_rest(p, 2))
    if op == SO_IMPORT:
        return seq_optimizer_import_on(p[0], a, seq_opt_rest(p, 2))
    if op == SO_CLOSE:
        return seq_optimizer_close_on(p[0])
    raise Error("sequence optimizer entry: unknown operation " + String(op))


def seq_opt_host_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """Entry 0 of the host sequence optimizer table (`SEQ_OPT_ABI`)."""
    try:
        if op < 0 or op >= SO_OP_COUNT:
            raise Error("sequence optimizer entry: unknown operation " + String(op))
        if result == 0 or np < 1 or (nf > 0 and ff == 0):
            raise Error("sequence optimizer entry: invalid request spans")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = List[Float64](capacity=nf)
        var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # small-loop(the optimizer's floating-point configuration words)
            f.append(fp[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = seq_opt_host_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct SeqOptHostEntriesOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the host optimizer table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime SEQ_OPT_HOST_ENTRIES_OWNER = SharedSlot[SeqOptHostEntriesOwner, "MojolearnSeqOptHostEntriesOwnerV1"]


def publish_seq_opt_host_entries() raises:
    """Publish at module init of `_mojolearn_x_sequence_host` (vendor word 0)."""
    var owner = SEQ_OPT_HOST_ENTRIES_OWNER.get_or_create_ptr()
    var key = ByteLmEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)
    byte_lm_table_publish[BYTE_LM_TABLE_SEQ_OPT](key, Int(owner), [String(SEQ_OPT_ABI)], [SO_OP_COUNT],
        [seq_opt_host_v1])
