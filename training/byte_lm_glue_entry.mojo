# SPDX-License-Identifier: Apache-2.0
"""The byte LM glue entry and the trainer table publication of the device
binding `_mojolearn_byte_lm` (lane calltable-byte-lm, 2026-11-11). ABI in
`training/byte_lm_glue_abi.mojo`; table in `core/byte_lm_entry_table.mojo`.

Each answer is the one the binding's former Python entry returned (named on
its operation below), from the same compiled functions. Reads constants, the
shape and (trial builds only) the environment; no GPU operation. Session
creation is lazy, as `byte_lm_session_create` was: no context is created.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.shared_slot import SharedSlot
from core.byte_lm_entry_table import ByteLmEntryKey, byte_lm_table_publish, byte_lm_vendor_word, BYTE_LM_TABLE_TRAINER
from training.byte_lm_glue_abi import (
    BYTE_LM_TRAINER_ABI, BYTE_LM_TRAINER_OPS, BYTE_LM_GLUE_ABI, BG_OP_COUNT,
    BG_NUMERIC_MODE, BG_VENDOR, BG_PROFILE, BG_SUFFIX, BG_CONFIG_PROFILE, BG_ATTENTION_ARM,
    BG_MEMORY_PROFILE, BG_STEP_GLUE_ARM, BG_SESSION_NEW, BG_SESSION_FREE,
    byte_lm_glue_text, byte_lm_glue_shape,
)
from training.byte_lm_binding_impl import ByteLMSession
from training.byte_lm_native_dispatch import byte_lm_native_v1
from training.byte_lm_config import ByteConfig, byte_lm_arithmetic_suffix
from transformer.impl.llama.fused_attention import (
    ATTN_ARM_DEFAULT,
    ATTN_ARM_TRIAL,
    fused_attention_arm_backward_resolved,
    fused_attention_arm_forward_resolved,
    fused_attention_arm_from_env,
    fused_attention_arm_name,
    attention_v1_backward_memory_profile,
    attention_v1_retained_exp_bytes,
)
from core.step_glue import step_glue_arm_from_env, step_glue_arm_name, step_glue_trial_build


def _config(words: List[Int]) raises -> ByteConfig:
    # `_byte_config` of the binding for its 10-word form.
    if words[9] != 0 and words[9] != 1:
        raise Error("byte LM: chunked head selector must be 0 or 1")
    var cfg = ByteConfig(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
                         words[8], words[9] == 1)
    cfg.validate()
    return cfg^


def byte_lm_glue_operation(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    if op == BG_NUMERIC_MODE:  # byte_lm_numeric_mode
        return Int(GLOBAL_NUMERIC_MODE)
    if op == BG_VENDOR:  # byte_lm_vendor
        return byte_lm_glue_text(String(COMPILED_VENDOR), a, p)
    if op == BG_PROFILE:  # byte_lm_profile
        return byte_lm_glue_text(ByteConfig().profile(), a, p)
    if op == BG_SUFFIX:  # byte_lm_arithmetic_suffix
        return byte_lm_glue_text(byte_lm_arithmetic_suffix(), a, p)
    if op == BG_CONFIG_PROFILE:  # byte_lm_config_profile
        return byte_lm_glue_text(_config(byte_lm_glue_shape(p, 1)).profile(), a, p)
    if op == BG_ATTENTION_ARM:  # byte_lm_attention_arm: [arm, default, trial_build, resolved_hd64]
        if len(p) != 2:
            raise Error("byte LM glue: attention arm takes one field word")
        var arm = fused_attention_arm_from_env()
        if p[1] == 0:
            return byte_lm_glue_text(fused_attention_arm_name(arm), a, p)
        if p[1] == 1:
            return byte_lm_glue_text(fused_attention_arm_name(ATTN_ARM_DEFAULT), a, p)
        if p[1] == 2:
            return 1 if ATTN_ARM_TRIAL else 0
        if p[1] == 3:
            var resolved = fused_attention_arm_forward_resolved(arm) | fused_attention_arm_backward_resolved(arm)
            return byte_lm_glue_text(fused_attention_arm_name(resolved), a, p)
        raise Error("byte LM glue: unknown attention arm field")
    if op == BG_MEMORY_PROFILE:  # byte_lm_attention_memory_profile
        var cfg = _config(byte_lm_glue_shape(p, 2))
        if p[1] == 0:
            return byte_lm_glue_text(attention_v1_backward_memory_profile(), a, p)
        if p[1] == 1:
            return attention_v1_retained_exp_bytes(cfg.batch, cfg.length, cfg.n_heads, cfg.length)
        raise Error("byte LM glue: unknown memory profile field")
    if op == BG_STEP_GLUE_ARM:  # byte_lm_step_glue_arm: [arm, trial_build]
        if len(p) != 2:
            raise Error("byte LM glue: step glue arm takes one field word")
        var arm = step_glue_arm_from_env()
        if p[1] == 0:
            return byte_lm_glue_text(step_glue_arm_name(arm), a, p)
        if p[1] == 1:
            return 1 if step_glue_trial_build() else 0
        raise Error("byte LM glue: unknown step glue arm field")
    if op == BG_SESSION_NEW:  # byte_lm_session_create, call-scoped and native
        var owner = alloc[ByteLMSession](1)
        owner.unsafe_write(ByteLMSession())
        return Int(owner)
    if op == BG_SESSION_FREE:
        if len(a) != 1 or a[0] == 0:
            raise Error("byte LM glue: free takes one native session")
        var owner = MutPointer[ByteLMSession, MutUntrackedOrigin](unsafe_from_address=a[0])
        owner.unsafe_deinit_pointee()
        owner.free()
        return 0
    raise Error("byte LM glue: unknown operation " + String(op))


def byte_lm_glue_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """Entry 1 of the trainer table (`BYTE_LM_GLUE_ABI`): addresses aa[0:na],
    words pp[0:np], no scalars. Writes the answer to result; a refusal goes
    to sink."""
    try:
        if op < 0 or op >= BG_OP_COUNT:
            raise Error("byte LM glue: unknown operation " + String(op))
        if result == 0 or nf != 0:
            raise Error("byte LM glue: invalid request spans")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = byte_lm_glue_operation(op, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ByteLmEntriesOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the device trainer table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime BYTE_LM_ENTRIES_OWNER = SharedSlot[ByteLmEntriesOwner, "MojolearnByteLmEntriesOwnerV1"]


def publish_byte_lm_entries() raises:
    """Publish the trainer table at module init of `_mojolearn_byte_lm`. A
    build with no GPU vendor publishes nothing: its callers refuse with the
    table's "provider is not published". A repeated init publishes the same
    pointers; another image with the same key refuses (core/entry_table.mojo)."""
    var word = byte_lm_vendor_word(COMPILED_VENDOR)
    # Vendor word 0 (cpu) belongs to `_mojolearn_byte_lm_host`
    # (training/byte_lm_host_glue_entry.mojo); this dispatch refuses a
    # non-GPU vendor on every training operation anyway.
    if word <= 0:
        return
    var owner = BYTE_LM_ENTRIES_OWNER.get_or_create_ptr()
    var key = ByteLmEntryKey(Int(GLOBAL_NUMERIC_MODE), word)
    byte_lm_table_publish[BYTE_LM_TABLE_TRAINER](key, Int(owner),
        [String(BYTE_LM_TRAINER_ABI), String(BYTE_LM_GLUE_ABI)], [BYTE_LM_TRAINER_OPS, BG_OP_COUNT],
        [byte_lm_native_v1, byte_lm_glue_v1])
