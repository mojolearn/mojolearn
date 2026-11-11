# SPDX-License-Identifier: Apache-2.0
"""The byte LM trainer table of the CPU image, reached natively (lane
calltable-byte-lm, 2026-11-11).

Ordered preconditions first (they share the one process table): the trainer
caller refuses before publication, then the publication
`_mojolearn_byte_lm_host` makes at module init (`publish_byte_lm_host_entries`),
a tier key and a schema mismatch refuse, and an unknown glue operation
refuses through the entry's sink. Then one case per glue operation, each
through `training/byte_lm_glue_abi.mojo` and the published pointer, against
the answer the host compatibility export returned (the same compiled
functions), plus the trainer dispatch (entry 0) on a native call-scoped
session. No Python participates.

The cases are independent units and run across the native host worker pool.
Run in a fresh process (the first check needs an unpublished table).
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.byte_lm_entry_table import ByteLmEntryKey, byte_lm_table_entry, byte_lm_table_published, BYTE_LM_TABLE_TRAINER
from training.byte_lm_config import ByteConfig, byte_lm_arithmetic_suffix
from training.byte_lm_glue_abi import (
    BYTE_LM_GLUE_ABI, BG_OP_COUNT, BG_ABSENT, BG_NUMERIC_MODE, BG_VENDOR, BG_PROFILE, BG_SUFFIX,
    BG_CONFIG_PROFILE, BG_ATTENTION_ARM, BG_MEMORY_PROFILE, BG_STEP_GLUE_ARM, BG_SESSION_NEW, BG_SESSION_FREE,
    byte_lm_glue_call, byte_lm_glue_read_text, byte_lm_glue_read_int, byte_lm_session_new, byte_lm_session_free,
    byte_lm_trainer_call,
)
from training.byte_lm_host_glue_entry import publish_byte_lm_host_entries

comptime BYTE_LM_ENTRY_CASES = 11


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> ByteLmEntryKey:
    return ByteLmEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def _shape() -> List[Int]:
    # (batch, length, d_model, n_heads, n_kv, head_dim, intermediate, n_layers, vocab, chunked)
    return [2, 16, 32, 4, 4, 8, 64, 2, 256, 0]


def _text(key: ByteLmEntryKey, op: Int, var words: List[Int]) raises -> String:
    var text = byte_lm_glue_read_text(key, op, words^)
    require(Bool(text), "a text answer was absent")
    return text.value().copy()


def precondition_unpublished() raises:
    try:
        _ = byte_lm_glue_read_int(_key(), BG_NUMERIC_MODE, List[Int]())
    except problem:
        _refuses(problem, "provider is not published")
        require(not byte_lm_table_published[BYTE_LM_TABLE_TRAINER](_key(), 1, String(BYTE_LM_GLUE_ABI), BG_OP_COUNT),
                "an unpublished table probed as published")
        return
    raise Error("an unpublished byte LM table did not refuse")


def preconditions_published() raises:
    require(byte_lm_table_published[BYTE_LM_TABLE_TRAINER](_key(), 1, String(BYTE_LM_GLUE_ABI), BG_OP_COUNT),
            "the published table did not probe as published")
    try:
        _ = byte_lm_glue_read_int(ByteLmEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0), BG_NUMERIC_MODE, List[Int]())
        raise Error("a mismatched tier key did not refuse")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
    try:
        _ = byte_lm_table_entry[BYTE_LM_TABLE_TRAINER](_key(), 1, String("mojolearn.training.other.v1"), BG_OP_COUNT)
        raise Error("a mismatched schema did not refuse")
    except problem:
        _refuses(problem, "ABI schema differs")
    try:
        _ = byte_lm_glue_call(_key(), BG_OP_COUNT, List[Int](), List[Int]())
        raise Error("an unknown operation did not refuse")
    except problem:
        _refuses(problem, "unknown operation")


def case(unit: Int) raises:
    var k = _key()
    if unit == BG_NUMERIC_MODE:  # host_compat_mode
        require(byte_lm_glue_read_int(k, BG_NUMERIC_MODE, List[Int]()) == 1, "numeric mode")
    elif unit == BG_VENDOR:  # host_compat_vendor
        require(_text(k, BG_VENDOR, List[Int]()) == "cpu", "vendor")
    elif unit == BG_PROFILE:  # host_compat_profile
        require(_text(k, BG_PROFILE, List[Int]()) == ByteConfig(chunked_lm_head_v2=False).profile(), "profile")
    elif unit == BG_SUFFIX:  # host_compat_suffix
        require(_text(k, BG_SUFFIX, List[Int]()) == byte_lm_arithmetic_suffix(), "arithmetic suffix")
    elif unit == BG_CONFIG_PROFILE:  # host_compat_shape_profile
        var s = _shape()
        var cfg = ByteConfig(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], False)
        cfg.validate()
        require(_text(k, BG_CONFIG_PROFILE, _shape()) == cfg.profile(), "config profile")
        var bad = _shape()
        bad[9] = 2
        try:
            _ = byte_lm_glue_read_text(k, BG_CONFIG_PROFILE, bad^)
            raise Error("an invalid chunked selector did not refuse")
        except problem:
            _refuses(problem, "chunked head selector must be 0 or 1")
    elif unit == BG_ATTENTION_ARM or unit == BG_STEP_GLUE_ARM:  # absent on the host image
        require(not byte_lm_glue_read_text(k, unit, [0]), "an absent arm answered")
        require(byte_lm_glue_read_int(k, unit, [1]) == BG_ABSENT, "an absent arm field answered")
    elif unit == BG_MEMORY_PROFILE:  # absent on the host image
        var words: List[Int] = [0]
        words.extend(_shape())
        require(not byte_lm_glue_read_text(k, BG_MEMORY_PROFILE, words^), "an absent memory profile answered")
    elif unit == BG_SESSION_NEW or unit == BG_SESSION_FREE:
        # A call-scoped native session: created closed, its status read
        # through the trainer dispatch (entry 0, operation 5), then freed.
        var session = byte_lm_session_new(k)
        try:
            var info = List[Int](length=4, fill=7)
            var a: List[Int] = [session, Int(info.unsafe_ptr())]
            require(byte_lm_trainer_call(k, 5, a, List[Int](), List[Float64]()) == 4, "session status count")
            require(info[3] == 0 and info[0] == -1, "a new session is not open")
            _ = info^
        finally:
            byte_lm_session_free(k, session)
        try:
            byte_lm_session_free(k, 0)
            raise Error("a null session free did not refuse")
        except problem:
            _refuses(problem, "free takes one native session")
    elif unit == 10:
        # Trainer dispatch operation 0: the image's compiled tier answer.
        require(byte_lm_trainer_call(k, 0, List[Int](), List[Int](), List[Float64]()) == 1, "trainer metadata")
    else:
        raise Error("unknown byte LM entry case")


def run_checks() raises:
    precondition_unpublished()
    # The publication _mojolearn_byte_lm_host makes at module init.
    publish_byte_lm_host_entries()
    preconditions_published()
    var failures = List[Int](length=BYTE_LM_ENTRY_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL byte LM entry case", unit, String(problem))
    host_parallelize(task, BYTE_LM_ENTRY_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "byte LM entry cases failed: " + String(failed))
    print("PASS byte LM entry", BYTE_LM_ENTRY_CASES)
