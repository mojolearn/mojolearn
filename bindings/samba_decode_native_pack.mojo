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
model, the session steps a session-owned copy of the caller's state
(bindings/samba_decode_native_unpack.mojo, `_private`): `sync_state` copies it
out to the caller's state, as the device arm's export does.
"""
# The PACK half of the former `bindings/samba_decode_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/samba_decode_native_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.samba_keys_unpack import samba_keys
from bindings.py_balanced import py_item
from std.memory import unsafe_memcpy
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from bindings.generation_buffers_door import generation_initialize_binding, generation_array
from bindings.causal_lm_public_door import _mutable, _writeback
from bindings.samba_state_native_door import samba_fields
from bindings.samba_inference_native_door import (
    _samba_inference_ids, samba_forward, samba_program_words, samba_program_open_owner,
    samba_program_call, samba_state_words, samba_set_cursors,
)
from core.samba_entry_table_programs import (
    SAMBA_PROGRAM_DECODE_STEP, SAMBA_PROGRAM_DECODE_TOKENS, SAMBA_PROGRAM_DECODE_EXPORT, SAMBA_PROGRAM_DECODE_LOAD,
)
from bindings.samba_admission_door import samba_admit_value


from bindings.samba_decode_native_unpack import (
    _decode_release,
    _decode_require_open,
    _samba_state_copy,
)


def samba_decode_sync(session: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var values = samba_fields(session)
    var what = String("SambaDecodeSession.sync_state")
    _decode_require_open(values, what)
    var state = py_item(values, nk[].k__state)
    var handle = py_item(values, nk[].k__handle)
    if handle is None:
        # The host arm: the session-owned state out to the caller's.
        samba_admit_value(state, py_item(context, nk[].k_admission))
        _samba_state_copy(py_item(values, nk[].k__model), py_item(values, nk[].k__private), state, what)
        return state
    samba_admit_value(state, py_item(context, nk[].k_admission))
    var model = py_item(values, nk[].k__model)
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
    var nk = samba_keys()
    var values = samba_fields(session)
    if not Bool(py=py_get(values, nk[].k__open, PythonObject(False))):
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
    var nk = samba_keys()
    return PythonObject(Bool(py=py_get(samba_fields(session), nk[].k__open, PythonObject(False))))


def samba_decode_repr(session: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var values = samba_fields(session)
    var flag = String("True") if Bool(py=py_get(values, nk[].k__open, PythonObject(False))) else String("False")
    var rows = py_get(values, nk[].k__rows, PythonObject(0))
    return PythonObject("SambaDecodeSession(open=" + flag + ", rows=" + String(rows) + ")")
