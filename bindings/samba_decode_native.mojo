# SPDX-License-Identifier: Apache-2.0
"""The native door of `SambaDecodeSession` (lane fix-d1-decode-sessions,
2026-11-08).

`SambaStack.decode_session(state)` opens one resident decode on the training
binding (`samba_decode_open`, training/samba_decode_resident.mojo): the
registry, every layer's carried state and the hidden state between layers
stay on the device, and a step is one id upload, the layer launches and one
logits download. The per-call `SambaStack.step` drove four public block
doors per token with host arrays between layers (0.8.66 samba-decode 1.57x
vs 0.8.55; owed row N094).

THE HOST ARM. Where the loaded training binding has no resident decode
(`samba_decode_open` absent: a build with MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF
or the chunked head, the fast tier, the CPU inference route, or the native
descriptor dispatch route), the session steps the caller's own state through
the per-call `samba_forward(step=True)`: there is nothing resident to hold,
and the bytes are the per-call step's by construction. `sync_state` and
`load_state` are then no-ops on a state that is always current.
"""
from std.python import Python, PythonObject
from std.memory import memcpy
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from bindings.generation_buffers import generation_initialize_binding, generation_array, py_ints
from bindings.causal_lm_public import _native_function, _mutable, _writeback
from bindings.samba_state_native import samba_fields, samba_entries
from bindings.samba_inference_native import _m3_shapes, _m3_names, _samba_inference_ids, samba_forward
from bindings.p3_samba_caps import samba_caps, SAMBA_CAP_RESIDENT_ABI, SAMBA_CAP_RESIDENT_ON, SAMBA_CAP_CHUNKED
from bindings.samba_admission import samba_admit_value


def _decode_device_arm(fields: PythonObject, config: PythonObject, entries: PythonObject) raises -> Bool:
    """Whether this stack's binding can hold the decode resident. The
    resident decode is the resident forward's sibling and takes exactly its
    gates (`_samba_resident_forward`), plus its own entry."""
    var bi = native_loaded_module("builtins")
    if Bool(py=py_get(fields, "_native_cpu_inference", PythonObject(False))):
        return False
    if py_get(fields, "_native_primitives") is not None:
        return False
    if String(fields["numeric_mode"]) != "identical":
        return False
    var caps = samba_caps(entries)
    if not (Bool(py=caps[SAMBA_CAP_RESIDENT_ABI]) and Bool(py=caps[SAMBA_CAP_RESIDENT_ON])):
        return False
    if Bool(py=caps[SAMBA_CAP_CHUNKED]):
        return False
    if not Bool(py=py_call(bi.hasattr, entries, "samba_decode_open")):
        return False
    var blocks = fields["_blocks"]
    for i in range(len(config["layers"])):  # driver-loop: over the stack's layer kinds
        if String(config["layers"][i]) == "attention":
            # The resident layers are full causal attention at the default
            # options, as every SambaStack builds them.
            if Int(py=py_get(samba_fields(blocks[i]), "window", PythonObject(0))) != 0:
                return False
    return True


def _decode_state_words(
    model: PythonObject, state: PythonObject, what: String,
    mut keep: List[PythonObject], mut copies: List[PythonObject],
) raises -> Tuple[List[Int], List[Int]]:
    """The state's piece addresses and cursor words, in the layout
    `samba_decode_load` / `samba_decode_export` take: per layer in stack
    order, the ten Mamba-3 pieces (`_m3_names()` order) and
    (buffered_tokens, pending), or k_cache, v_cache and (cached_tokens, 0)."""
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var slots = samba_fields(state)
    var b = Int(py=slots["batch_size"])
    var smax = Int(py=slots["max_tokens"])
    var layers = slots["layers"]
    if len(layers) != len(config["layers"]):
        raise Error(what + ": the state holds " + String(len(layers)) + " layers, the stack "
                    + String(len(config["layers"])))
    var blocks = fields["_blocks"]
    var names = _m3_names()
    var pieces = List[Int]()
    var cursors = List[Int]()
    for i in range(len(config["layers"])):  # driver-loop: over the stack's layer state owners
        var lslots = samba_fields(layers[i])
        if String(config["layers"][i]) == "attention":
            if Int(py=py_get(lslots, "batch_size", PythonObject(b))) != b or Int(py=lslots["max_tokens"]) != smax:
                raise Error(what + ": layer " + String(i) + " state differs from the stack state's batch or capacity")
            if Int(py=py_get(lslots, "window", PythonObject(0))) != 0:
                raise Error(what + ": layer " + String(i) + " state has a sliding window")
            var n = b * Int(py=lslots["n_kv_heads"]) * smax * Int(py=lslots["head_dim"])
            pieces.append(_mutable(lslots["k_cache"], [n], keep, copies, "k_cache"))
            pieces.append(_mutable(lslots["v_cache"], [n], keep, copies, "v_cache"))
            cursors.append(Int(py=lslots["cached_tokens"]))
            cursors.append(0)
        else:
            var shapes = _m3_shapes(b, Int(py=samba_fields(blocks[i])["nheads"]))
            for k in range(len(names)):  # driver-loop: over the fixed Mamba3 continuation buffer ABI
                pieces.append(_mutable(lslots[names[k]], shapes[k], keep, copies, names[k]))
            cursors.append(Int(py=lslots["buffered_tokens"]))
            cursors.append(1 if Bool(py=lslots["pending"]) else 0)
    return (pieces^, cursors^)


def _decode_mark(state: PythonObject, owner: PythonObject) raises:
    """`owner` (True, or None to release) on the stack state and every
    layer state: the per-call `forward`/`step` refuse a state a session holds
    (`_m3_state_call`, `_causal_layer_run_bound`, `samba_forward`)."""
    var slots = samba_fields(state)
    py_setitem(slots, "_resident_session", owner)
    var layers = slots["layers"]
    for i in range(len(layers)):  # driver-loop: over the stack's layer state owners
        py_setitem(samba_fields(layers[i]), "_resident_session", owner)


def _decode_set_cursors(model: PythonObject, state: PythonObject, cursors: PythonObject) raises:
    var config = samba_fields(samba_fields(model)["config"])
    var layers = samba_fields(state)["layers"]
    for i in range(len(layers)):  # driver-loop: over the stack's layer state owners
        var lslots = samba_fields(layers[i])
        if String(config["layers"][i]) == "attention":
            py_setitem(lslots, "cached_tokens", PythonObject(Int(py=cursors[2 * i])))
        else:
            py_setitem(lslots, "buffered_tokens", PythonObject(Int(py=cursors[2 * i])))
            py_setitem(lslots, "pending", PythonObject(Int(py=cursors[2 * i + 1]) != 0))


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
    var entries = samba_entries(context, fields["numeric_mode"])
    py_setitem(values, "_model", model)
    py_setitem(values, "_state", state)
    py_setitem(values, "_rows", PythonObject(b))
    py_setitem(values, "_vocab", config["vocab"])
    py_setitem(values, "_entries", entries)
    py_setitem(values, "_handle", PythonObject(None))
    if _decode_device_arm(fields, config, entries):
        # The attention layers carry the arithmetic-profile word the per-call
        # step writes (`_causal_layer_run_bound`): same check, same write.
        var blocks = fields["_blocks"]
        for i in range(len(layers)):  # driver-loop: over the stack's layer state owners
            if String(config["layers"][i]) == "attention":
                var lslots = samba_fields(layers[i])
                var pid = py_get(samba_fields(blocks[i]), "_arithmetic_profile")
                var old = py_get(lslots, "arithmetic_profile")
                if old is not None and pid is not None and not Bool(py=old == pid):
                    raise Error(what + ": layer " + String(i) + " state arithmetic profile differs")
                if pid is not None:
                    py_setitem(lslots, "arithmetic_profile", pid)
        var keep = List[PythonObject]()
        var copies = List[PythonObject]()
        var words = _decode_state_words(model, state, what, keep, copies)
        var kinds = bi.list()
        for i in range(len(config["layers"])):  # driver-loop: over model layer metadata at session creation
            _ = py_call(kinds.append, 0 if String(config["layers"][i]) == "mamba3" else 1)
        var params = Python.list(config["vocab"], config["d_model"])
        for name in [String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate")]:  # driver-loop: over resident-session shape metadata
            _ = py_call(params.append, PythonObject(0) if config[name] is None else config[name])
        _ = py_call(params.append, 1 if Bool(py=config["tie_embeddings"]) else 0)
        _ = py_call(params.append, config["norm_eps"])
        _ = py_call(params.append, fields["n_total"])
        _ = py_call(params.append, b)
        _ = py_call(params.append, smax)
        var handle = py_call(_native_function(entries, "samba_decode_open"), kinds, params, fields["flat"]._addr,
                             py_ints(words[0]), py_ints(words[1]))
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
        return samba_forward(values["_model"], inputs, values["_state"], PythonObject(None), True, True, context)
    var b = Int(py=values["_rows"])
    var v = Int(py=values["_vocab"])
    # The per-call step's own id admission: (B,) or (B, 1) native-endian
    # integers, range-checked against the vocabulary in one native pass.
    var admitted = _samba_inference_ids(inputs, v, True, PythonObject(None), context)
    var ids = admitted[0]
    if Int(py=ids.shape[0]) != b:
        raise Error(what + ": the session holds " + String(b) + " rows, inputs has B = " + String(ids.shape[0]))
    var output = generation_array([b, v])
    _ = py_call(_native_function(values["_entries"], "samba_decode_step"), handle,
                Python.list(ids._addr, output._addr), py_ints([b]))
    _ = ids  # keep the Python object alive through the native call above
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
    var v = Int(py=values["_vocab"]) if handle is not None else Int(py=values["_model"].vocab_size)
    var admitted = _samba_inference_ids(inputs, v, False, PythonObject(None), context)
    var ids = admitted[0]
    var b = Int(py=ids.shape[0])
    var n = Int(py=ids.shape[1])
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
            var logits = samba_forward(values["_model"], column, values["_state"], PythonObject(None), True, True, context)
            var row = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=logits._addr))
            for r in range(b):  # driver-loop: over the batch rows, a host-arm copy of token t's logits
                memcpy(dest=dst + (r * n + t) * v, src=row + r * v, count=v)
            _ = logits^
            _ = column^
        _ = ids
        return output
    var rows = Int(py=values["_rows"])
    if b != rows:
        raise Error(what + ": the session holds " + String(rows) + " rows, inputs has B = " + String(b))
    _ = py_call(_native_function(values["_entries"], "samba_decode_tokens"), handle,
                Python.list(ids._addr, output._addr), py_ints([b, n]))
    _ = ids  # keep the Python object alive through the native call above
    return output


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
    var words = _decode_state_words(model, state, what, keep, copies)
    var cursors = py_call(_native_function(values["_entries"], "samba_decode_export"), handle, py_ints(words[0]))
    _writeback(copies)
    _ = keep^
    _decode_set_cursors(model, state, cursors)
    return state


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
    var words = _decode_state_words(values["_model"], state, what, keep, copies)
    _ = py_call(_native_function(values["_entries"], "samba_decode_load"), handle, py_ints(words[0]), py_ints(words[1]))
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
    # The device session's buffers go with its last reference, here.
    _ = handle^


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


def samba_decode_state(session: PythonObject) raises -> PythonObject:
    return py_get(samba_fields(session), "_state")


def samba_decode_device(session: PythonObject) raises -> PythonObject:
    return py_get(samba_fields(session), "_handle")


def samba_decode_repr(session: PythonObject) raises -> PythonObject:
    var values = samba_fields(session)
    var flag = String("True") if Bool(py=py_get(values, "_open", PythonObject(False))) else String("False")
    var rows = py_get(values, "_rows", PythonObject(0))
    return PythonObject("SambaDecodeSession(open=" + flag + ", rows=" + String(rows) + ")")
