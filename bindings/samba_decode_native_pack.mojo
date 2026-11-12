# SPDX-License-Identifier: Apache-2.0
"""The native door of `SambaDecodeSession` (lane fix-d1-decode-sessions,
2026-11-08).

`SambaStack.decode_session(state)` opens one Samba stack program on the
training image (SAMBA_PROGRAM_OPEN of core/samba_entry_table_programs.mojo,
training/samba_program.mojo, lane x6-samba-on-executor; a native registry
handle since lane samba-sessions-native, 2026-11-12): the registry, every layer's carried state and the
hidden state between layers stay on the device, and a step is one id upload,
one step op per layer and one logits download. It is the same program, the
same ops and the same order as the per-call `SambaStack.step` on a device
build, so the bytes are the per-call step's.

THE HOST ARM. On a host build (no published program entry) or the CPU inference
model, the session steps the caller's own state through the per-call
`samba_forward(step=True)`: there is nothing resident to hold. `sync_state`
and `load_state` are then no-ops on a state that is always current.
"""
# The PACK half of the former `bindings/samba_decode_native.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/samba_decode_native_unpack.mojo`.
from std.python import Python, PythonObject
from std.memory import unsafe_memcpy
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from bindings.generation_buffers import generation_initialize_binding, generation_array
from bindings.causal_lm_public import _mutable, _writeback
from bindings.samba_state_native_door import samba_fields, samba_entries, samba_entry_key
from bindings.samba_inference_native import (
    _samba_inference_ids, samba_forward, samba_program_arm, samba_program_words, samba_program_open_owner,
    samba_program_call, samba_state_words, samba_set_cursors,
)
from core.samba_entry_table_programs import (
    SAMBA_PROGRAM_DECODE_STEP, SAMBA_PROGRAM_DECODE_TOKENS, SAMBA_PROGRAM_DECODE_EXPORT, SAMBA_PROGRAM_DECODE_LOAD,
)
from bindings.samba_admission import samba_admit_value


from bindings.samba_decode_native_unpack import (
    _decode_release,
    _decode_require_open,
)


def samba_decode_sync(session: PythonObject, context: PythonObject) raises -> PythonObject:
    var values = samba_fields(session)
    var what = String("SambaDecodeSession.sync_state")
    _decode_require_open(values, what)
    var state = values["_state"]
    var handle = values["_handle"]
    if handle is None:
        return state
    samba_admit_value(state, context["admission"])
    var model = values["_model"]
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var words = samba_state_words(model, state, what, keep, copies)
    var cursors = List[Int](length=len(words[1]), fill=0)
    var addresses: List[Int] = [Int(cursors.unsafe_ptr())]
    for i in range(len(words[0])):  # driver-loop: over the state's piece addresses (per layer)
        addresses.append(words[0][i])
    var capacity: List[Int] = [len(cursors)]
    _ = samba_program_call(handle, SAMBA_PROGRAM_DECODE_EXPORT, addresses, capacity)
    _writeback(copies)
    _ = keep^
    samba_set_cursors(model, state, cursors)
    _ = cursors
    return state


def samba_decode_close(session: PythonObject, context: PythonObject) raises -> PythonObject:
    var values = samba_fields(session)
    if not Bool(py=py_get(values, "_open", PythonObject(False))):
        return PythonObject(None)
    try:
        _ = samba_decode_sync(session, context)
    finally:
        _decode_release(values)
    return PythonObject(None)


def samba_decode_drop(session: PythonObject) raises -> PythonObject:
    # Finalizer: release an unclosed session without copying its state back.
    _decode_release(samba_fields(session))
    return PythonObject(None)


def samba_decode_is_open(session: PythonObject) raises -> PythonObject:
    return PythonObject(Bool(py=py_get(samba_fields(session), "_open", PythonObject(False))))


def samba_decode_repr(session: PythonObject) raises -> PythonObject:
    var values = samba_fields(session)
    var flag = String("True") if Bool(py=py_get(values, "_open", PythonObject(False))) else String("False")
    var rows = py_get(values, "_rows", PythonObject(0))
    return PythonObject("SambaDecodeSession(open=" + flag + ", rows=" + String(rows) + ")")
