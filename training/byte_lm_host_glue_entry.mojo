# SPDX-License-Identifier: Apache-2.0
"""The byte LM glue entry and the trainer table publication of the CPU
binding `_mojolearn_byte_lm_host` (lane calltable-byte-lm, 2026-11-11). ABI
in `training/byte_lm_glue_abi.mojo`; table in `core/byte_lm_entry_table.mojo`.

Each answer is the one the host compatibility export returned
(providers/python/byte_lm_host_compat_pyapi.mojo, named on its operation).
The host image has no attention or step glue arm entries: those operations
answer `BG_ABSENT`, as the missing Python entry did. HOST ONLY: nothing here
imports device code.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.shared_slot import SharedSlot
from core.byte_lm_entry_table import ByteLmEntryKey, byte_lm_table_publish, BYTE_LM_TABLE_TRAINER
from training.byte_lm_glue_abi import (
    BYTE_LM_TRAINER_ABI, BYTE_LM_TRAINER_OPS, BYTE_LM_GLUE_ABI, BG_OP_COUNT, BG_ABSENT,
    BG_NUMERIC_MODE, BG_VENDOR, BG_PROFILE, BG_SUFFIX, BG_CONFIG_PROFILE, BG_ATTENTION_ARM,
    BG_MEMORY_PROFILE, BG_STEP_GLUE_ARM, BG_SESSION_NEW, BG_SESSION_FREE,
    byte_lm_glue_text, byte_lm_glue_shape,
)
from training.byte_lm_host_session import ByteHostSession
from training.byte_lm_host_trainer_dispatch import host_trainer_native_v1
from training.byte_lm_config import ByteConfig, byte_lm_arithmetic_suffix


def _config(words: List[Int]) raises -> ByteConfig:
    # `_config` of the host compatibility export.
    if words[9] != 0 and words[9] != 1:
        raise Error("byte LM: chunked head selector must be 0 or 1")
    var cfg = ByteConfig(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
                         words[8], words[9] == 1)
    cfg.validate()
    return cfg^


def byte_lm_host_glue_operation(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    if op == BG_NUMERIC_MODE:  # host_compat_mode
        return 1
    if op == BG_VENDOR:  # host_compat_vendor
        return byte_lm_glue_text(String("cpu"), a, p)
    if op == BG_PROFILE:  # host_compat_profile: the public config default (chunked=False)
        return byte_lm_glue_text(ByteConfig(chunked_lm_head_v2=False).profile(), a, p)
    if op == BG_SUFFIX:  # host_compat_suffix
        return byte_lm_glue_text(byte_lm_arithmetic_suffix(), a, p)
    if op == BG_CONFIG_PROFILE:  # host_compat_shape_profile
        return byte_lm_glue_text(_config(byte_lm_glue_shape(p, 1)).profile(), a, p)
    if op == BG_ATTENTION_ARM or op == BG_MEMORY_PROFILE or op == BG_STEP_GLUE_ARM:
        return BG_ABSENT
    if op == BG_SESSION_NEW:  # host_compat_create, call-scoped and native
        var owner = alloc[ByteHostSession](1)
        owner.unsafe_write(ByteHostSession())
        return Int(owner)
    if op == BG_SESSION_FREE:
        if len(a) != 1 or a[0] == 0:
            raise Error("byte LM glue: free takes one native session")
        var owner = MutPointer[ByteHostSession, MutUntrackedOrigin](unsafe_from_address=a[0])
        owner.unsafe_deinit_pointee()
        owner.free()
        return 0
    raise Error("byte LM glue: unknown operation " + String(op))


def byte_lm_host_glue_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                         result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """Entry 1 of the host trainer table (`BYTE_LM_GLUE_ABI`)."""
    try:
        if op < 0 or op >= BG_OP_COUNT:
            raise Error("byte LM glue: unknown operation " + String(op))
        if result == 0 or nf != 0:
            raise Error("byte LM glue: invalid request spans")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = byte_lm_host_glue_operation(op, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ByteLmHostEntriesOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the host trainer table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime BYTE_LM_HOST_ENTRIES_OWNER = SharedSlot[ByteLmHostEntriesOwner, "MojolearnByteLmHostEntriesOwnerV1"]


def publish_byte_lm_host_entries() raises:
    """Publish the host trainer table (vendor word 0) at module init of
    `_mojolearn_byte_lm_host`, keyed by the image's compiled tier."""
    var owner = BYTE_LM_HOST_ENTRIES_OWNER.get_or_create_ptr()
    var key = ByteLmEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)
    byte_lm_table_publish[BYTE_LM_TABLE_TRAINER](key, Int(owner),
        [String(BYTE_LM_TRAINER_ABI), String(BYTE_LM_GLUE_ABI)], [BYTE_LM_TRAINER_OPS, BG_OP_COUNT],
        [host_trainer_native_v1, byte_lm_host_glue_v1])
