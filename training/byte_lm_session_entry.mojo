# SPDX-License-Identifier: Apache-2.0
"""The resident session entry of the device binding `_mojolearn_byte_lm`
(lane byte-lm-sessions-native, 2026-11-11). ABI and caller helpers in
`core/byte_lm_entry_table_sessions.mojo`; published by
`publish_byte_lm_entries` (training/byte_lm_glue_entry.mojo).

The image's resident `ByteLMSession`s live in a generation-checked state
registry. A run holds the session's Arc lease and calls the image's own
trainer dispatch (`byte_lm_native_v1`, the former provider route) with the
session address first, so every trainer operation keeps its descriptors,
words and bits. Releasing the last lease runs `ByteLMSession.__deinit__`
(the device teardown the Python session object's destruction ran).
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.accelerator_target import TARGET_HAS_AMD_GPU, TARGET_HAS_APPLE_GPU, TARGET_HAS_NVIDIA_GPU
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call, native_descriptor_words, native_descriptor_report
from core.state_registry import StateRegistry
from core.byte_lm_entry_table_sessions import (
    BS_CREATE, BS_RUN, BS_CLOSE, BS_OP_COUNT, byte_lm_sessions_handle, byte_lm_sessions_write_handle,
    byte_lm_sessions_check_create,
    byte_lm_sessions_request, byte_lm_sessions_floats,
)
from training.byte_lm_binding_impl import ByteLMSession
from training.byte_lm_native_dispatch import byte_lm_native_v1


struct ByteLmDeviceSessionCell(Movable):
    """One resident session at a stable heap address (the trainer dispatch
    reads the session through its address)."""

    var address: Int

    def __init__(out self):
        # Lazy: creation performs no GPU operation (`byte_lm_session_create`).
        var owner = alloc[ByteLMSession](1)
        owner.unsafe_write(ByteLMSession())
        self.address = Int(owner)

    def __deinit__(deinit self):
        if self.address != 0:
            var owner = MutPointer[ByteLMSession, MutUntrackedOrigin](unsafe_from_address=self.address)
            owner.unsafe_deinit_pointee()
            owner.free()


# One registry per image (tier and vendor), so a handle never reaches a
# session another image owns.
comptime _DEVICE_SESSIONS_NAME: StaticString = (
    ("MojolearnByteLmSessionsIdenticalHIPV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnByteLmSessionsFastHIPV1")
    if TARGET_HAS_AMD_GPU else
    ("MojolearnByteLmSessionsIdenticalCUDAV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnByteLmSessionsFastCUDAV1")
    if TARGET_HAS_NVIDIA_GPU else
    ("MojolearnByteLmSessionsIdenticalMetalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnByteLmSessionsFastMetalV1")
    if TARGET_HAS_APPLE_GPU else
    "MojolearnByteLmSessionsNoGpuV1"
)
comptime BYTE_LM_DEVICE_SESSIONS = StateRegistry[ByteLmDeviceSessionCell, _DEVICE_SESSIONS_NAME]


def byte_lm_device_sessions_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == BS_CREATE:
        byte_lm_sessions_check_create(a)
        var handle = BYTE_LM_DEVICE_SESSIONS.insert(ByteLmDeviceSessionCell())
        return byte_lm_sessions_write_handle(a, handle)
    var handle = byte_lm_sessions_handle(p)
    if op == BS_CLOSE:
        BYTE_LM_DEVICE_SESSIONS.release(handle)
        return 0
    if op == BS_RUN:
        var lease = BYTE_LM_DEVICE_SESSIONS.acquire(handle)
        var request = byte_lm_sessions_request(lease[].address, a, p)
        var result = native_descriptor_call(byte_lm_native_v1, request.operation, request.addresses, request.ints, f)
        _ = lease  # the session outlives the synchronous trainer call
        return result
    raise Error("byte LM sessions: unknown operation " + String(op))


def byte_lm_sessions_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                        result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The device image's session entry (`BYTE_LM_SESSIONS_ABI`)."""
    try:
        if op < 0 or op >= BS_OP_COUNT:
            raise Error("byte LM sessions: unknown operation " + String(op))
        if result == 0:
            raise Error("byte LM sessions: null result span")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = byte_lm_sessions_floats(ff, nf)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = byte_lm_device_sessions_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
