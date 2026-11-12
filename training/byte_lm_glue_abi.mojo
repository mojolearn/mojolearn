# SPDX-License-Identifier: Apache-2.0
"""The byte LM glue ABI shared by its publishers and its native caller
(lane calltable-byte-lm, 2026-11-11).

Entry 1 of the byte LM trainer table (`core/byte_lm_entry_table.mojo`,
family `BYTE_LM_TABLE_TRAINER`) runs the operations below; entry 0 is the
trainer dispatch (`BYTE_LM_TRAINER_ABI`, 13 operations). Publishers:
`training/byte_lm_glue_entry.mojo` (device binding `_mojolearn_byte_lm`) and
`training/byte_lm_host_glue_entry.mojo` (CPU binding `_mojolearn_byte_lm_host`).

A call is the descriptor shape of `core/native_descriptor_abi.mojo`. A text
answer is written as UTF-8 into a caller buffer: a = [buffer], p[0] = its
capacity; the result is the byte count. `BG_ABSENT` (-1) answers an
operation the image does not carry (the host image has no attention or step
glue arm), where the Python binding had no entry of that name. The caller
owns every buffer until the synchronous call returns; nothing is retained
and no Python object crosses.

These replace the Python entries `byte_lm_numeric_mode`, `byte_lm_vendor`,
`byte_lm_profile`, `byte_lm_arithmetic_suffix`, `byte_lm_config_profile`,
`byte_lm_attention_arm`, `byte_lm_attention_memory_profile`,
`byte_lm_step_glue_arm` for the native trainer door, and the per-call
Python session object of the stateless step (`byte_lm_session_create` and
`byte_lm_native_session_handle_v1`) with a native call-scoped session.
"""
from core.native_descriptor_abi import NativeDescriptorCall
from core.entry_table import EntryRef, entry_ref_call, entry_ref_word
from core.native_string import unsafe_string_from_utf8_cstr
from core.byte_lm_entry_table import ByteLmEntryKey, byte_lm_table_entry, BYTE_LM_TABLE_TRAINER

comptime BYTE_LM_TRAINER_ABI = "mojolearn.training.byte_lm.v1"
comptime BYTE_LM_TRAINER_OPS = 13
comptime BYTE_LM_GLUE_ABI = "mojolearn.training.byte_lm.glue.v1"

comptime BG_NUMERIC_MODE = 0
comptime BG_VENDOR = 1
comptime BG_PROFILE = 2
comptime BG_SUFFIX = 3
#: p = [capacity, 10 shape words]
comptime BG_CONFIG_PROFILE = 4
#: p = [capacity, field]: 0 arm, 1 default, 2 trial_build (int result), 3 resolved_hd64
comptime BG_ATTENTION_ARM = 5
#: p = [capacity, field, 10 shape words]: 0 profile (text), 1 retained exp bytes per layer (int result)
comptime BG_MEMORY_PROFILE = 6
#: p = [capacity, field]: 0 arm (text), 1 trial_build (int result)
comptime BG_STEP_GLUE_ARM = 7
#: result: the address of a new native session owned by the caller until BG_SESSION_FREE
comptime BG_SESSION_NEW = 8
#: a = [session]
comptime BG_SESSION_FREE = 9
comptime BG_OP_COUNT = 10

comptime BG_ABSENT = -1
comptime BG_TEXT_CAPACITY = 512


def byte_lm_glue_text(text: String, a: List[Int], p: List[Int]) raises -> Int:
    """Publisher side: write `text` into the caller's buffer (a[0], p[0])."""
    if len(a) != 1 or a[0] == 0 or len(p) < 1:
        raise Error("byte LM glue: missing text buffer")
    var bytes = text.as_bytes()
    if len(bytes) > p[0]:
        raise Error("byte LM glue: text buffer too short")
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=a[0])
    for i in range(len(bytes)):  # driver-loop: copy one short metadata answer
        dst[i] = bytes[i]
    _ = text
    return len(bytes)


def byte_lm_glue_shape(p: List[Int], first: Int) raises -> List[Int]:
    """Publisher side: the 10 shape words at p[first:first + 10]."""
    if len(p) != first + 10:
        raise Error("byte LM: expected 10 shape words")
    var words = List[Int]()
    for i in range(first, first + 10):  # small-loop(the ten shape words)
        words.append(p[i])
    return words^


def byte_lm_trainer_entry(key: ByteLmEntryKey) raises -> NativeDescriptorCall:
    """Caller side: entry 0, the trainer dispatch of the image `key` names."""
    return byte_lm_table_entry[BYTE_LM_TABLE_TRAINER](key, 0, String(BYTE_LM_TRAINER_ABI), BYTE_LM_TRAINER_OPS)


def byte_lm_trainer_ref(key: ByteLmEntryKey) raises -> EntryRef:
    """Caller side: entry 0 resolved once (lane entry-table-zero-alloc); the
    table refusals raise here."""
    return EntryRef(byte_lm_trainer_entry(key), entry_ref_word(key.mode, key.vendor, BYTE_LM_TRAINER_OPS))


def byte_lm_glue_ref(key: ByteLmEntryKey) raises -> EntryRef:
    """Caller side: entry 1, the glue dispatch of the image `key` names,
    resolved once; the table refusals raise here."""
    return EntryRef(byte_lm_table_entry[BYTE_LM_TABLE_TRAINER](key, 1, String(BYTE_LM_GLUE_ABI), BG_OP_COUNT),
                    entry_ref_word(key.mode, key.vendor, BG_OP_COUNT))


def byte_lm_trainer_call(trainer: EntryRef, op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    """Caller side: one trainer operation (the former provider execute)
    through the resolved entry: one indirect call."""
    return entry_ref_call(trainer, op, a, p, f)


def byte_lm_glue_call(glue: EntryRef, op: Int, a: List[Int], p: List[Int]) raises -> Int:
    """Caller side: one glue operation through the resolved glue entry."""
    var f: Array[Float64, 0] = []
    var result = entry_ref_call(glue, op, Int(a.unsafe_ptr()), len(a), Int(p.unsafe_ptr()), len(p), 0, 0)
    _ = a
    _ = p
    _ = f
    return result


def byte_lm_glue_read_text(glue: EntryRef, op: Int, var p: List[Int]) raises -> Optional[String]:
    """Caller side: a text answer, or None where the image has no such entry.
    `p` holds the operation's words after the capacity word."""
    var buffer = List[UInt8](length=BG_TEXT_CAPACITY + 1, fill=0)
    var words: List[Int] = [BG_TEXT_CAPACITY]
    for i in range(len(p)):  # small-loop(at most twelve request words)
        words.append(p[i])
    var a: List[Int] = [Int(buffer.unsafe_ptr())]
    var size = byte_lm_glue_call(glue, op, a, words)
    if size == BG_ABSENT:
        return None
    if size < 0 or size > BG_TEXT_CAPACITY:
        raise Error("byte LM glue: invalid text answer")
    buffer[size] = 0
    var text = unsafe_string_from_utf8_cstr(buffer.unsafe_ptr())
    _ = buffer^  # keep the answer buffer alive past the copy
    return text^


def byte_lm_glue_read_int(glue: EntryRef, op: Int, var p: List[Int]) raises -> Int:
    """Caller side: an integer answer (BG_ABSENT where the image has none);
    the text buffer is passed so a publisher reads one request shape."""
    var buffer = List[UInt8](length=BG_TEXT_CAPACITY + 1, fill=0)
    var words: List[Int] = [BG_TEXT_CAPACITY]
    for i in range(len(p)):  # small-loop(at most twelve request words)
        words.append(p[i])
    var a: List[Int] = [Int(buffer.unsafe_ptr())]
    var value = byte_lm_glue_call(glue, op, a, words)
    _ = buffer^
    return value


def byte_lm_session_new(glue: EntryRef) raises -> Int:
    """Caller side: a call-scoped native session of the image `key` names."""
    var session = byte_lm_glue_call(glue, BG_SESSION_NEW, List[Int](), List[Int]())
    if session == 0:
        raise Error("byte LM glue: session creation returned a null handle")
    return session


def byte_lm_session_free(glue: EntryRef, session: Int) raises:
    var a: List[Int] = [session]
    _ = byte_lm_glue_call(glue, BG_SESSION_FREE, a, List[Int]())
