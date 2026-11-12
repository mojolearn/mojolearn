# SPDX-License-Identifier: Apache-2.0
"""The sequence optimizer entry of the device binding `_mojolearn_x_sequence`
and its table publication (lane calltable-byte-lm, 2026-11-11). ABI in
`sequence/opt_entry_abi.mojo`. The same compiled functions and argument
words as providers/python/sequence_opt_public_pyapi.mojo."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.shared_slot import SharedSlot
from core.byte_lm_entry_table import ByteLmEntryKey, byte_lm_table_publish, byte_lm_vendor_word, BYTE_LM_TABLE_SEQ_OPT
from sequence.opt_entry_abi import (
    SEQ_OPT_ABI, SO_OP_COUNT, SO_INIT, SO_STEP, SO_LAMB_STEP, SO_ADAFACTOR_STEP, SO_STATE, SO_EXPORT,
    SO_IMPORT, SO_CLOSE, seq_opt_rest,
)
from sequence.opt_public import seq_optimizer_init
from sequence.opt_public_device import (
    seq_adafactor_step_device,
    seq_lamb_step_device,
    seq_optimizer_close_device,
    seq_optimizer_export_device,
    seq_optimizer_import_device,
    seq_optimizer_state_device,
    seq_optimizer_step_device,
)


def seq_opt_device_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == SO_INIT:
        return seq_optimizer_init(p[0], seq_opt_rest(p, 1))
    if len(p) < 2:
        raise Error("sequence optimizer entry: missing state and handle words")
    if op == SO_STEP:
        return seq_optimizer_step_device(p[0], p[1], a, seq_opt_rest(p, 2), f)
    if op == SO_LAMB_STEP:
        return seq_lamb_step_device(p[0], p[1], a, seq_opt_rest(p, 2), f)
    if op == SO_ADAFACTOR_STEP:
        return seq_adafactor_step_device(p[0], p[1], a, seq_opt_rest(p, 2), f)
    if op == SO_STATE:
        return seq_optimizer_state_device(p[0], p[1], a)
    if op == SO_EXPORT:
        return seq_optimizer_export_device(p[0], p[1], a, seq_opt_rest(p, 2))
    if op == SO_IMPORT:
        return seq_optimizer_import_device(p[0], p[1], a, seq_opt_rest(p, 2))
    if op == SO_CLOSE:
        if len(p) != 3:
            raise Error("sequence optimizer entry: close takes words, handles and nh")
        return seq_optimizer_close_device(p[0], p[1], p[2])
    raise Error("sequence optimizer entry: unknown operation " + String(op))


def seq_opt_device_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                      result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """Entry 0 of the sequence optimizer table (`SEQ_OPT_ABI`)."""
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
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = seq_opt_device_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct SeqOptEntriesOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the device optimizer table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime SEQ_OPT_ENTRIES_OWNER = SharedSlot[SeqOptEntriesOwner, "MojolearnSeqOptEntriesOwnerV1"]


def publish_seq_opt_entries() raises:
    """Publish at module init of `_mojolearn_x_sequence` under its compiled
    tier and vendor. Vendor word 0 (cpu) belongs to the host binding; a build
    with no GPU vendor publishes nothing."""
    var word = byte_lm_vendor_word(COMPILED_VENDOR)
    if word <= 0:
        return
    var owner = SEQ_OPT_ENTRIES_OWNER.get_or_create_ptr()
    var key = ByteLmEntryKey(Int(GLOBAL_NUMERIC_MODE), word)
    byte_lm_table_publish[BYTE_LM_TABLE_SEQ_OPT](key, Int(owner), [String(SEQ_OPT_ABI)], [SO_OP_COUNT],
        [seq_opt_device_v1])
