# SPDX-License-Identifier: Apache-2.0
"""The sequence optimizer entry ABI shared by its publishers and its native
caller (lane calltable-byte-lm, 2026-11-11).

Entry 0 of the sequence optimizer table (`core/byte_lm_entry_table.mojo`,
family `BYTE_LM_TABLE_SEQ_OPT`) runs the operations below; publishers are
`sequence/opt_entry_device.mojo` (`_mojolearn_x_sequence`) and
`sequence/opt_entry_host.mojo` (`_mojolearn_x_sequence_host`). Each
operation is the former Python entry of the same name with the same
arguments as plain words (sequence/opt_public.mojo documents them):

    SO_INIT            seq_optimizer_init     p = [words, ip...]
    SO_STEP            seq_optimizer_step     a = addrs, p = [words, handles, ip...], f = fp
    SO_LAMB_STEP       seq_lamb_step          (as SO_STEP)
    SO_ADAFACTOR_STEP  seq_adafactor_step     (as SO_STEP)
    SO_STATE           seq_optimizer_state    a = addrs, p = [words, handles]
    SO_EXPORT          seq_optimizer_export   a = addrs, p = [words, handles, ip...]
    SO_IMPORT          seq_optimizer_import   (as SO_EXPORT)
    SO_CLOSE           seq_optimizer_close    p = [words, handles, nh]

The result is the entry's count. The caller owns every buffer until the
synchronous call returns; nothing is retained and no Python object crosses.
"""
from core.native_descriptor_call import native_descriptor_call
from core.byte_lm_entry_table import ByteLmEntryKey, byte_lm_table_entry, BYTE_LM_TABLE_SEQ_OPT

comptime SEQ_OPT_ABI = "mojolearn.sequence.optimizer.v1"

comptime SO_INIT = 0
comptime SO_STEP = 1
comptime SO_LAMB_STEP = 2
comptime SO_ADAFACTOR_STEP = 3
comptime SO_STATE = 4
comptime SO_EXPORT = 5
comptime SO_IMPORT = 6
comptime SO_CLOSE = 7
comptime SO_OP_COUNT = 8


def seq_opt_rest(p: List[Int], first: Int) raises -> List[Int]:
    """Publisher side: the words p[first:], the former `ip` list."""
    if len(p) < first:
        raise Error("sequence optimizer entry: missing request words")
    var rest = List[Int](capacity=len(p) - first)
    for i in range(first, len(p)):  # driver-loop: over the admitted request words
        rest.append(p[i])
    return rest^


def seq_opt_call(key: ByteLmEntryKey, op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    """Caller side: one optimizer operation in the x_sequence image `key`
    names (a lock-free acquire load and a schema compare, then the pointer)."""
    var entry = byte_lm_table_entry[BYTE_LM_TABLE_SEQ_OPT](key, 0, String(SEQ_OPT_ABI), SO_OP_COUNT)
    return native_descriptor_call(entry, op, a, p, f)
