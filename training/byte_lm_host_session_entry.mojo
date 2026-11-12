# SPDX-License-Identifier: Apache-2.0
"""The resident session entry of the CPU binding `_mojolearn_byte_lm_host`
(lane byte-lm-sessions-native, 2026-11-11). ABI and caller helpers in
`core/byte_lm_entry_table_sessions.mojo`; published by
`publish_byte_lm_host_entries` (training/byte_lm_host_glue_entry.mojo).

The image's resident `ByteHostSession`s live in a generation-checked state
registry; a run holds the session's Arc lease and calls the image's own
trainer dispatch (`host_trainer_native_v1`) with the session address first.
HOST ONLY: nothing here imports device code.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.state_registry import StateRegistry
from core.byte_lm_entry_table_sessions import (
    BS_CREATE, BS_RUN, BS_CLOSE, BS_OP_COUNT, byte_lm_sessions_handle, byte_lm_sessions_write_handle,
    byte_lm_sessions_check_create,
    byte_lm_sessions_run, byte_lm_sessions_floats,
)
from training.byte_lm_host_session import ByteHostSession
from training.byte_lm_host_trainer_dispatch import host_trainer_native_v1


struct ByteLmHostSessionCell(Movable):
    """One resident host session at a stable heap address."""

    var address: Int

    def __init__(out self):
        var owner = alloc[ByteHostSession](1)
        owner.unsafe_write(ByteHostSession())
        self.address = Int(owner)

    def __deinit__(deinit self):
        if self.address != 0:
            var owner = MutPointer[ByteHostSession, MutUntrackedOrigin](unsafe_from_address=self.address)
            owner.unsafe_deinit_pointee()
            owner.free()


comptime BYTE_LM_HOST_SESSIONS = StateRegistry[ByteLmHostSessionCell,
    ("MojolearnByteLmSessionsIdenticalCPUV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnByteLmSessionsFastCPUV1")]


def byte_lm_host_sessions_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == BS_CREATE:
        byte_lm_sessions_check_create(a)
        var handle = BYTE_LM_HOST_SESSIONS.insert(ByteLmHostSessionCell())
        return byte_lm_sessions_write_handle(a, handle)
    var handle = byte_lm_sessions_handle(p)
    if op == BS_CLOSE:
        BYTE_LM_HOST_SESSIONS.release(handle)
        return 0
    if op == BS_RUN:
        var lease = BYTE_LM_HOST_SESSIONS.acquire(handle)
        var result = byte_lm_sessions_run(host_trainer_native_v1, lease[].address, a, p, f)
        _ = lease  # the session outlives the synchronous trainer call
        return result
    raise Error("byte LM sessions: unknown operation " + String(op))


def byte_lm_host_sessions_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                             result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The host image's session entry (`BYTE_LM_SESSIONS_ABI`)."""
    try:
        if op < 0 or op >= BS_OP_COUNT:
            raise Error("byte LM sessions: unknown operation " + String(op))
        if result == 0:
            raise Error("byte LM sessions: null result span")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = byte_lm_sessions_floats(ff, nf)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = byte_lm_host_sessions_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
