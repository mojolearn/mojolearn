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


def _decode_mark(state: PythonObject, owner: PythonObject) raises:
    """`owner` (True, or None to release) on the stack state and every
    layer state: the per-call `forward`/`step` refuse a state a session holds
    (`_m3_state_call`, `_causal_layer_run_bound`, `samba_forward`)."""
    var slots = samba_fields(state)
    py_setitem(slots, "_resident_session", owner)
    var layers = slots["layers"]
    for i in range(len(layers)):  # driver-loop: over the stack's layer state owners
        py_setitem(samba_fields(layers[i]), "_resident_session", owner)


def samba_decode_init(session: PythonObject, model: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = native_loaded_module("builtins")
    var what = String("SambaDecodeSession")
    if py_type(model) is not context["model_class"]:
        raise Error("TypeError: " + what + " takes a SambaStack")
    if state is None:
        raise Error(what + ": state is required (allocate_state)")
    if not Bool(py=py_call(bi.isinstance, state, context["state_class"])):
        raise Error("TypeError: " + what + ": state must be a SambaState")
    var values = samba_fields(session)
    if Bool(py=py_get(values, "_open", PythonObject(False))):
        raise Error(what + ": already open; close it first")
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var slots = samba_fields(state)
    if py_get(slots, "_resident_session") is not None:
        raise Error(what + ": the state is owned by an open resident decode session; call sync_state() or close() on it first")
    var layers = slots["layers"]
    if len(layers) != len(config["layers"]):
        raise Error(what + ": the state holds " + String(len(layers)) + " layers, the stack " + String(len(config["layers"])))
    for i in range(len(layers)):  # driver-loop: over the stack's layer state owners
        if py_get(samba_fields(layers[i]), "_resident_session") is not None:
            raise Error(what + ": layer " + String(i) + " state is owned by an open resident decode session")
    var b = Int(py=slots["batch_size"])
    var smax = Int(py=slots["max_tokens"])
    if b < 1 or smax < 1:
        raise Error(what + ": the state holds no rows")
    # Mode/vendor admission of the connected binding; the session then
    # reaches its program through the table key alone (lane
    # samba-sessions-native: no `_entries` namespace kept on the session).
    _ = samba_entries(context, fields["numeric_mode"])
    var key = samba_entry_key(context, fields["numeric_mode"])
    py_setitem(values, "_model", model)
    py_setitem(values, "_state", state)
    py_setitem(values, "_rows", PythonObject(b))
    py_setitem(values, "_vocab", config["vocab"])
    py_setitem(values, "_handle", PythonObject(None))
    if samba_program_arm(fields, key):
        # The device arm: the session owns one stack program (lane x6),
        # loaded with the registry and the state once here, behind its
        # native handle owner (core/samba_program_handle.mojo).
        var keep = List[PythonObject]()
        var copies = List[PythonObject]()
        var words = samba_state_words(model, state, what, keep, copies)
        var opened = samba_program_words(fields, config, b, smax)
        var addresses: List[Int] = [Int(py=fields["flat"]._addr)]
        for i in range(len(words[0])):  # driver-loop: over the state's piece addresses (per layer)
            addresses.append(words[0][i])
        var open_words = opened[0].copy()
        for i in range(len(words[1])):  # driver-loop: over the state's cursor words (two per layer)
            open_words.append(words[1][i])
        var handle = samba_program_open_owner(key, addresses, open_words, opened[1], b, smax)
        _ = keep^
        py_setitem(values, "_handle", handle)
        _decode_mark(state, PythonObject(True))
    py_setitem(values, "_open", PythonObject(True))
    return PythonObject(None)


def samba_decode_open(model: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var session = py_call(bi.object.__new__, context["decode_session_class"])
    _ = samba_decode_init(session, model, state, context)
    return session


def _decode_require_open(values: PythonObject, what: String) raises:
    if not Bool(py=py_get(values, "_open", PythonObject(False))):
        raise Error(what + ": the session is closed")


def samba_decode_step(session: PythonObject, inputs: PythonObject, context: PythonObject) raises -> PythonObject:
    var values = samba_fields(session)
    var what = String("SambaDecodeSession.step")
    _decode_require_open(values, what)
    var handle = values["_handle"]
    if handle is None:
        # The host arm: the per-call step on the caller's own state.
        return samba_forward(values["_model"], inputs, values["_state"], PythonObject(None), True, context)
    var b = Int(py=values["_rows"])
    var v = Int(py=values["_vocab"])
    # The per-call step's own id admission: (B,) or (B, 1) native-endian
    # integers, range-checked against the vocabulary in one native pass.
    var admitted = _samba_inference_ids(inputs, v, True, PythonObject(None), context)
    var ids = admitted[0]
    if Int(py=ids.shape[0]) != b:
        raise Error(what + ": the session holds " + String(b) + " rows, inputs has B = " + String(ids.shape[0]))
    var output = generation_array([b, v])
    var addresses: List[Int] = [Int(py=ids._addr), Int(py=output._addr)]
    var words: List[Int] = [b]
    _ = samba_program_call(handle, SAMBA_PROGRAM_DECODE_STEP, addresses, words)
    _ = ids  # keep the Python object alive through the native call above
    _ = output
    return output


def samba_decode_tokens(session: PythonObject, inputs: PythonObject, context: PythonObject) raises -> PythonObject:
    """`decode(ids)`: (B, N) ids -> (B, N, vocab) logits in one native call
    (decode plan D2). The device arm runs `samba_decode_tokens` (one id
    upload, N step bodies on the device, one logits download); the host arm
    runs the per-call step N times on the caller's own state. Either way the
    logits and the state equal N `step` calls byte for byte."""
    var values = samba_fields(session)
    var what = String("SambaDecodeSession.decode")
    _decode_require_open(values, what)
    var handle = values["_handle"]
    var v = Int(py=values["_vocab"])
    var admitted = _samba_inference_ids(inputs, v, False, PythonObject(None), context)
    var ids = admitted[0]
    var b = Int(py=ids.shape[0])
    var n = Int(py=ids.shape[1])
    var rows = Int(py=values["_rows"])
    if b != rows:
        raise Error(what + ": the session holds " + String(rows) + " rows, inputs has B = " + String(b))
    var output = generation_array([b, n, v])
    if handle is None:
        # The host arm: the per-call step on the caller's own state, token by
        # token, with column t of the ids as the step's (B, 1) input.
        var src = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=ids._addr))
        var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr))
        for t in range(n):  # driver-loop: over decode tokens, one host-arm step each
            var column = generation_array([b, 1], "i")
            var col = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=column._addr))
            for r in range(b):  # driver-loop: over the batch rows, a host-arm copy of token t's id
                col.unsafe_store(r, src.unsafe_load(r * n + t))
            var logits = samba_forward(values["_model"], column, values["_state"], PythonObject(None), True, context)
            var row = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=logits._addr))
            for r in range(b):  # driver-loop: over the batch rows, a host-arm copy of token t's logits
                unsafe_memcpy(dest=dst + (r * n + t) * v, src=row + r * v, count=v)
            _ = logits^
            _ = column^
        _ = ids
        return output
    var addresses: List[Int] = [Int(py=ids._addr), Int(py=output._addr)]
    var words: List[Int] = [b, n]
    _ = samba_program_call(handle, SAMBA_PROGRAM_DECODE_TOKENS, addresses, words)
    _ = ids  # keep the Python object alive through the native call above
    _ = output
    return output


def samba_decode_load(session: PythonObject, context: PythonObject) raises -> PythonObject:
    var values = samba_fields(session)
    var what = String("SambaDecodeSession.load_state")
    _decode_require_open(values, what)
    var state = values["_state"]
    var handle = values["_handle"]
    if handle is None:
        return state
    samba_admit_value(state, context["admission"])
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var words = samba_state_words(values["_model"], state, what, keep, copies)
    _ = samba_program_call(handle, SAMBA_PROGRAM_DECODE_LOAD, words[0], words[1])
    _ = keep^
    return state


def _decode_release(values: PythonObject) raises:
    if not Bool(py=py_get(values, "_open", PythonObject(False))):
        return
    py_setitem(values, "_open", PythonObject(False))
    var handle = py_get(values, "_handle")
    py_setitem(values, "_handle", PythonObject(None))
    var state = py_get(values, "_state")
    if handle is not None and state is not None:
        _decode_mark(state, PythonObject(None))
    # The device session's program is released by its native handle owner
    # (core/samba_program_handle.mojo) with the owner's last reference, here.
    _ = handle^


def samba_decode_state(session: PythonObject) raises -> PythonObject:
    return py_get(samba_fields(session), "_state")


def samba_decode_device(session: PythonObject) raises -> PythonObject:
    return py_get(samba_fields(session), "_handle")
