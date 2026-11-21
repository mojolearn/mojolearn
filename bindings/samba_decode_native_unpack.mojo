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
model, the session steps a session-owned copy of the caller's state (`_private`,
copied in at open) through the per-call `samba_forward(step=True)`, so it keeps
the device arm's contract: `sync_state` copies the session's state out to the
caller's, `load_state` copies the caller's state in, and the caller's state is
not advanced between them. (Before lane cpu-native-crashes, 2026-11-15, the host
arm stepped the caller's state itself and `load_state` could not discard the
steps after a `sync_state`: smoke 91d5 samba-decode-session differed in 7585 of
8192 bytes.)
"""
from std.python import Python, PythonObject
from std.memory import unsafe_memcpy
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from bindings.generation_buffers_door import generation_initialize_binding, generation_array
from bindings.causal_lm_public_door import _mutable, _writeback
from bindings.samba_state_native_door import samba_fields
from bindings.samba_route_unpack import samba_route
from bindings.samba_inference_native_door import (
    _samba_inference_ids, samba_forward, samba_program_words, samba_program_open_owner,
    samba_program_call, samba_state_words, samba_set_cursors, samba_allocate,
)
from training.samba_inference_run import _m3_shapes
from core.samba_entry_table_programs import (
    SAMBA_PROGRAM_DECODE_STEP, SAMBA_PROGRAM_DECODE_TOKENS, SAMBA_PROGRAM_DECODE_EXPORT, SAMBA_PROGRAM_DECODE_LOAD,
)
from bindings.samba_admission_door import samba_admit_value
from bindings.samba_keys_unpack import samba_keys
from bindings.py_balanced import py_item


def _decode_mark(state: PythonObject, owner: PythonObject) raises:
    """`owner` (True, or None to release) on the stack state and every
    layer state: the per-call `forward`/`step` refuse a state a session holds
    (`_m3_state_call`, `_causal_layer_run_bound`, `samba_forward`)."""
    var nk = samba_keys()
    var slots = samba_fields(state)
    py_setitem(slots, nk[].k__resident_session, owner)
    var layers = py_item(slots, nk[].k_layers)
    for i in range(len(layers)):  # driver-loop: over the stack's layer state owners
        py_setitem(samba_fields(layers[i]), nk[].k__resident_session, owner)


def _samba_state_counts(model: PythonObject, state: PythonObject) raises -> List[Int]:
    """The float32 count of every piece `samba_state_words` lists, in its
    order: k_cache and v_cache per attention layer, the ten Mamba-3 pieces
    (`_m3_shapes`) per Mamba-3 layer."""
    var nk = samba_keys()
    var fields = samba_fields(model)
    var config = samba_fields(py_item(fields, nk[].k_config))
    var slots = samba_fields(state)
    var b = Int(py=py_item(slots, nk[].k_batch_size))
    var smax = Int(py=py_item(slots, nk[].k_max_tokens))
    var layers = py_item(slots, nk[].k_layers)
    var blocks = py_item(fields, nk[].k__blocks)
    var counts = List[Int]()
    for i in range(len(py_item(config, nk[].k_layers))):  # driver-loop: over the stack's layer state owners
        if String(py_item(config, nk[].k_layers)[i]) == "attention":
            var lslots = samba_fields(layers[i])
            var n = b * Int(py=py_item(lslots, nk[].k_n_kv_heads)) * smax * Int(py=py_item(lslots, nk[].k_head_dim))
            counts.append(n)
            counts.append(n)
        else:
            var shapes = _m3_shapes(b, Int(py=py_item(samba_fields(blocks[i]), nk[].k_nheads)))
            for k in range(len(shapes)):  # driver-loop: over the fixed Mamba3 continuation buffer ABI
                var n = 1
                for j in range(len(shapes[k])):  # small-loop(piece rank)
                    n *= shapes[k][j]
                counts.append(n)
    return counts^


def _samba_state_copy(model: PythonObject, source: PythonObject, target: PythonObject, what: String) raises:
    """Every piece and cursor of `source` into `target` (two states of the same
    stack, rows and capacity): the host arm's open, sync_state and load_state."""
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var src = samba_state_words(model, source, what, keep, copies)
    var target_keep = List[PythonObject]()
    var target_copies = List[PythonObject]()
    var dst = samba_state_words(model, target, what, target_keep, target_copies)
    var counts = _samba_state_counts(model, source)
    for i in range(len(counts)):  # driver-loop: over the state's pieces (per layer)
        unsafe_memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst[0][i]),
                      src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=src[0][i]), count=counts[i])
    _writeback(target_copies)
    samba_set_cursors(model, target, src[1])
    _ = keep^
    _ = target_keep^
    _ = copies^


def samba_decode_init(session: PythonObject, model: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    _ = generation_initialize_binding(py_item(py_item(context, nk[].k_neural), nk[].k_generation_classes))
    var bi = native_loaded_module("builtins")
    var what = String("SambaDecodeSession")
    if py_type(model) is not py_item(context, nk[].k_model_class):
        raise Error("TypeError: " + what + " takes a SambaStack")
    if state is None:
        raise Error(what + ": state is required (allocate_state)")
    if not Bool(py=py_call(bi.isinstance, state, py_item(context, nk[].k_state_class))):
        raise Error("TypeError: " + what + ": state must be a SambaState")
    var values = samba_fields(session)
    if Bool(py=py_get(values, nk[].k__open, PythonObject(False))):
        raise Error(what + ": already open; close it first")
    var fields = samba_fields(model)
    var config = samba_fields(py_item(fields, nk[].k_config))
    var slots = samba_fields(state)
    if py_get(slots, nk[].k__resident_session) is not None:
        raise Error(what + ": the state is owned by an open resident decode session; call sync_state() or close() on it first")
    var layers = py_item(slots, nk[].k_layers)
    if len(layers) != len(py_item(config, nk[].k_layers)):
        raise Error(what + ": the state holds " + String(len(layers)) + " layers, the stack " + String(len(py_item(config, nk[].k_layers))))
    for i in range(len(layers)):  # driver-loop: over the stack's layer state owners
        if py_get(samba_fields(layers[i]), nk[].k__resident_session) is not None:
            raise Error(what + ": layer " + String(i) + " state is owned by an open resident decode session")
    var b = Int(py=py_item(slots, nk[].k_batch_size))
    var smax = Int(py=py_item(slots, nk[].k_max_tokens))
    if b < 1 or smax < 1:
        raise Error(what + ": the state holds no rows")
    # The stack's route (mode/vendor admission of the connected binding,
    # resolved once with the stack); the session then reaches its program
    # through the table key alone (lane samba-sessions-native: no `_entries`
    # namespace kept on the session).
    var route = samba_route(model, fields, context)
    var key = route.key.copy()
    py_setitem(values, nk[].k__model, model)
    py_setitem(values, nk[].k__state, state)
    py_setitem(values, nk[].k__rows, PythonObject(b))
    py_setitem(values, nk[].k__vocab, py_item(config, nk[].k_vocab))
    py_setitem(values, nk[].k__handle, PythonObject(None))
    if route.program():
        # The device arm: the session owns one stack program (lane x6),
        # loaded with the registry and the state once here, behind its
        # native handle owner (core/samba_program_handle.mojo).
        var keep = List[PythonObject]()
        var copies = List[PythonObject]()
        var words = samba_state_words(model, state, what, keep, copies)
        var opened = samba_program_words(fields, config, b, smax)
        var addresses: List[Int] = [Int(py=py_item(fields, nk[].k_flat)._addr)]
        for i in range(len(words[0])):  # driver-loop: over the state's piece addresses (per layer)
            addresses.append(words[0][i])
        var open_words = opened[0].copy()
        for i in range(len(words[1])):  # driver-loop: over the state's cursor words (two per layer)
            open_words.append(words[1][i])
        var handle = samba_program_open_owner(key, addresses, open_words, opened[1], b, smax)
        _ = keep^
        py_setitem(values, nk[].k__handle, handle)
        _decode_mark(state, PythonObject(True))
    else:
        # The host arm: a session-owned state, the caller's copied in.
        var private = samba_allocate(model, PythonObject(b), PythonObject(smax), context)
        _samba_state_copy(model, state, private, what)
        py_setitem(values, nk[].k__private, private)
    py_setitem(values, nk[].k__open, PythonObject(True))
    return PythonObject(None)


def samba_decode_open(model: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var bi = native_loaded_module("builtins")
    var session = py_call(bi.object.__new__, py_item(context, nk[].k_decode_session_class))
    _ = samba_decode_init(session, model, state, context)
    return session


def _decode_require_open(values: PythonObject, what: String) raises:
    var nk = samba_keys()
    if not Bool(py=py_get(values, nk[].k__open, PythonObject(False))):
        raise Error(what + ": the session is closed")


def samba_decode_step(session: PythonObject, inputs: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var values = samba_fields(session)
    var what = String("SambaDecodeSession.step")
    _decode_require_open(values, what)
    var handle = py_item(values, nk[].k__handle)
    if handle is None:
        # The host arm: the per-call step on the session-owned state.
        return samba_forward(py_item(values, nk[].k__model), inputs, py_item(values, nk[].k__private), PythonObject(None), True, context)
    var b = Int(py=py_item(values, nk[].k__rows))
    var v = Int(py=py_item(values, nk[].k__vocab))
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
    runs the per-call step N times on the session-owned state (`_private`).
    Either way the logits and the state equal N `step` calls byte for byte."""
    var nk = samba_keys()
    var values = samba_fields(session)
    var what = String("SambaDecodeSession.decode")
    _decode_require_open(values, what)
    var handle = py_item(values, nk[].k__handle)
    var v = Int(py=py_item(values, nk[].k__vocab))
    var admitted = _samba_inference_ids(inputs, v, False, PythonObject(None), context)
    var ids = admitted[0]
    var b = Int(py=ids.shape[0])
    var n = Int(py=ids.shape[1])
    var rows = Int(py=py_item(values, nk[].k__rows))
    if b != rows:
        raise Error(what + ": the session holds " + String(rows) + " rows, inputs has B = " + String(b))
    var output = generation_array([b, n, v])
    if handle is None:
        # The host arm: the per-call step on the session-owned state, token by
        # token, with column t of the ids as the step's (B, 1) input.
        var src = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=ids._addr))
        var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr))
        # One (B, 1) id column for every token (was one allocation per token):
        # the step's id admission copies it into its own array before the run.
        var column = generation_array([b, 1], "i")
        var col = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=column._addr))
        for t in range(n):  # driver-loop: over decode tokens, one host-arm step each
            for r in range(b):  # driver-loop: over the batch rows, a host-arm copy of token t's id
                col.unsafe_store(r, src.unsafe_load(r * n + t))
            var logits = samba_forward(py_item(values, nk[].k__model), column, py_item(values, nk[].k__private), PythonObject(None), True, context)
            var row = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=logits._addr))
            for r in range(b):  # driver-loop: over the batch rows, a host-arm copy of token t's logits
                unsafe_memcpy(dest=dst + (r * n + t) * v, src=row + r * v, count=v)
            _ = logits^
        _ = column^  # keep the id column alive past the last step's read
        _ = ids
        return output
    var addresses: List[Int] = [Int(py=ids._addr), Int(py=output._addr)]
    var words: List[Int] = [b, n]
    _ = samba_program_call(handle, SAMBA_PROGRAM_DECODE_TOKENS, addresses, words)
    _ = ids  # keep the Python object alive through the native call above
    _ = output
    return output


def samba_decode_load(session: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var values = samba_fields(session)
    var what = String("SambaDecodeSession.load_state")
    _decode_require_open(values, what)
    var state = py_item(values, nk[].k__state)
    var handle = py_item(values, nk[].k__handle)
    if handle is None:
        # The host arm: the caller's state into the session-owned one.
        samba_admit_value(state, py_item(context, nk[].k_admission))
        _samba_state_copy(py_item(values, nk[].k__model), state, py_item(values, nk[].k__private), what)
        return state
    samba_admit_value(state, py_item(context, nk[].k_admission))
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var words = samba_state_words(py_item(values, nk[].k__model), state, what, keep, copies)
    _ = samba_program_call(handle, SAMBA_PROGRAM_DECODE_LOAD, words[0], words[1])
    _ = keep^
    return state


def _decode_release(values: PythonObject) raises:
    var nk = samba_keys()
    if not Bool(py=py_get(values, nk[].k__open, PythonObject(False))):
        return
    py_setitem(values, nk[].k__open, PythonObject(False))
    py_setitem(values, nk[].k__private, PythonObject(None))
    var handle = py_get(values, nk[].k__handle)
    py_setitem(values, nk[].k__handle, PythonObject(None))
    var state = py_get(values, nk[].k__state)
    if handle is not None and state is not None:
        _decode_mark(state, PythonObject(None))
    # The device session's program is released by its native handle owner
    # (core/samba_program_handle.mojo) with the owner's last reference, here.
    _ = handle^


def samba_decode_state(session: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    return py_get(samba_fields(session), nk[].k__state)


def samba_decode_device(session: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    return py_get(samba_fields(session), nk[].k__handle)
