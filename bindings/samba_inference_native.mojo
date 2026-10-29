# SPDX-License-Identifier: Apache-2.0
"""Native Samba inference over the existing resident model and block owners.

The resident model/session remains the persistent numerical owner. Public
classes and buffer exporters remain CPython-owned interface records; this
module does not claim that all public state has native struct ownership.
"""
from providers.python.samba_resident_pyapi import samba_resident_native_forward, samba_resident_provider
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from providers.python.samba_execution_pyapi import samba_provider_forward
from providers.python.neural_descriptor_pyapi import neural_descriptor_matches, neural_descriptor_execute
from core.samba_ids_impl import samba_inference_ids_into, samba_copy_row_prefixes
from std.memory import memcpy
from bindings.generation_buffers import generation_initialize_binding, generation_array, generation_address, py_ints
from bindings.samba_state_native import samba_fields, samba_entries
from bindings.samba_training_native import samba_ids
from bindings.neural_block_public import neural_training_entries, neural_block_allocate_binding, neural_block_forward_binding, _neural_lengths, neural_mamba_session_call
from bindings.neural_prefill_native import _prefill_mamba3
from bindings.causal_lm_public import _native_function, _mutable, _writeback
from bindings.causal_lm_construct import _model_bindings
from bindings.py_balanced import py_call, py_get, py_setattr, py_setitem, py_type


def _m3_shapes(b: Int, h: Int) -> List[List[Int]]:
    var result = List[List[Int]]()
    result.append([b, h, 32])
    result.append([b, h, 64, 128])
    result.append([b, 64, h, 128])
    result.append([b, 64, h, 128])
    result.append([b, 64, h, 64])
    result.append([b, 64, h])
    result.append([b, 64, h])
    result.append([b, 64, h])
    result.append([b, h, 128])
    result.append([b, h, 64])
    return result^


def _m3_names() -> List[String]:
    return [String("theta"), String("h"), String("buffer_qrot"), String("buffer_krot"),
            String("buffer_v"), String("buffer_dt"), String("buffer_sig"), String("buffer_adt"),
            String("pending_k"), String("pending_v")]


def samba_state_construct(state: PythonObject, b: PythonObject, maximum: PythonObject, layers: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var fields = samba_fields(state)
    py_setitem(fields, "batch_size", py_call(bi.int, b))
    py_setitem(fields, "max_tokens", py_call(bi.int, maximum))
    py_setitem(fields, "layers", py_call(bi.list, layers))
    return PythonObject(None)


def samba_allocate(model: PythonObject, batch: PythonObject, maximum: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = native_loaded_module("builtins")
    # Integers only (anything with __index__): a str is not parsed and a
    # bool is not counted as 1.
    if py_type(batch) is bi.bool or py_type(maximum) is bi.bool:
        raise Error("TypeError: SambaStack.allocate_state: batch_size and max_tokens must be integers")
    var b = 0
    var cap = 0
    try:
        var operator = native_loaded_module("operator")
        b = Int(py=py_call(operator.index, batch))
        cap = Int(py=py_call(operator.index, maximum))
    except error:
        raise Error("TypeError: SambaStack.allocate_state: batch_size and max_tokens must be integers")
    if b < 1 or cap < 1:
        raise Error("SambaStack.allocate_state: batch_size and max_tokens must be positive")
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var blocks = fields["_blocks"]
    var states = bi.list()
    for i in range(len(config["layers"])):  # driver-loop: over persistent layer state owners
        var block = blocks[i]
        if String(config["layers"][i]) == "attention":
            _ = py_call(states.append, neural_block_allocate_binding(block, PythonObject(b), PythonObject(cap), context["neural"]))
        else:
            var state = py_call(bi.object.__new__, context["mamba3_state_class"])
            var slots = samba_fields(state)
            var shapes = _m3_shapes(b, Int(py=samba_fields(block)["nheads"]))
            var names = _m3_names()
            for j in range(len(names)):  # driver-loop: over the fixed Mamba3 continuation buffer ABI
                py_setitem(slots, names[j], generation_array(shapes[j]))
            py_setitem(slots, "buffered_tokens", 0)
            py_setitem(slots, "pending", False)
            _ = py_call(states.append, state)
    var result = py_call(bi.object.__new__, context["state_class"])
    _ = samba_state_construct(result, PythonObject(b), PythonObject(cap), states)
    return result



def _samba_block_entries(block: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var nc = context["neural"]
    if Bool(py=py_call(bi.isinstance, block, nc["block_classes"]["cpu"]["mamba3"])):
        return _model_bindings(PythonObject("cpu"), "mamba3", nc, PythonObject("identical"))["mamba3"]
    return neural_training_entries(samba_fields(block), nc)

def _m3_state_call(block: PythonObject, x: PythonObject, state: PythonObject, step: Bool, entries: PythonObject,
                   captured_fields: PythonObject, captured_slots: PythonObject) raises -> PythonObject:
    """One carried-state call: `(y, h_last, k_last, v_last, theta_last)`. The
    state's pieces, `buffered_tokens` and `pending` are updated; the caller
    assigns the four reports."""
    # Standalone Mamba operations supply one retained metadata snapshot. Other
    # native owners preserve their established admission and call contract.
    var fields = samba_fields(block) if captured_fields is None else captured_fields
    var slots = samba_fields(state) if captured_slots is None else captured_slots
    if py_get(slots, "_resident_session") is not None:
        raise Error("Mamba3: state belongs to an open resident decode session; synchronize or close it first")
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var d = Int(py=fields["d_model"])
    var h = Int(py=fields["nheads"])
    var shapes = _m3_shapes(b, h)
    var names = _m3_names()
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var addresses: List[Int] = [generation_address(x, keep)]
    var bi = native_loaded_module("builtins")
    # an installed owned snapshot is read on the device (`mamba3_state_owned`)
    var owned = py_get(fields, "_owned_generation")
    var canonical = entries.canonical if Bool(py=py_call(bi.hasattr, entries, "canonical")) else entries
    var on_session = (owned is not None and py_get(fields, "_prefill_session") is not None
                      and py_get(fields, "_prefill_binding") is canonical
                      and Bool(py=py_call(bi.hasattr, entries, "mamba3_state_owned")))
    var weights = fields["_w"]
    for i in range(len(weights)):  # driver-loop: over existing Mamba3 parameter ABI slots
        if on_session:
            addresses.append(0)
        else:
            addresses.append(generation_address(weights[i], keep))
    for i in range(len(names)):  # driver-loop: over mutable continuation buffers
        addresses.append(_mutable(slots[names[i]], shapes[i], keep, copies, names[i]))
    var output = generation_array([b, l, d])
    var reports = Python.list(generation_array([b, h, 64, 128]), generation_array([b, h, 128]),
                              generation_array([b, h, 64]), generation_array([b, h, 32]))
    addresses.append(generation_address(output, keep))
    _ = output  # keep the Python object alive through the native call above
    for i in range(4):  # driver-loop: over the four reporting buffers
        addresses.append(generation_address(reports[i], keep))
        _ = output  # keep the Python object alive through the native call above
    var q = Int(py=slots["buffered_tokens"])
    var pending = 1 if Bool(py=slots["pending"]) else 0
    var updated = PythonObject(None)
    var native = PythonObject(None)
    if not on_session:
        var admitted = py_get(fields, "_native_mamba_state")
        if neural_descriptor_matches(admitted, canonical):
            native = admitted
    if on_session:
        var owned_native = py_get(fields, "_native_mamba_block_provider_cache")
        if neural_descriptor_matches(owned_native, canonical):
            updated = PythonObject(neural_mamba_session_call(owned_native, fields, entries, 16,
                addresses, [b, 1 if step else l, d, q, pending, Int(py=owned), 1 if step else 0]))
        else:
            updated = py_call(_native_function(entries, "mamba3_state_owned"), fields["_prefill_session"], py_ints(addresses),
                              Python.list(b, 1 if step else l, d, q, pending, owned, 1 if step else 0))
        _ = output  # keep the Python object alive through the native call above
    elif native is not None:
        var ints: List[Int] = [b, d, q, pending] if step else [b, l, d, q, pending]
        updated = PythonObject(neural_descriptor_execute(native, 5 if step else 4, addresses, ints, []))
        _ = output
    elif step:
        updated = py_call(_native_function(entries, "mamba3_decode_step"), py_ints(addresses), py_ints([b, d, q, pending]))
        _ = output  # keep the Python object alive through the native call above
    else:
        updated = py_call(_native_function(entries, "mamba3_forward"), py_ints(addresses), py_ints([b, l, d, q, pending]))
        _ = output  # keep the Python object alive through the native call above
    _writeback(copies)
    py_setitem(slots, "buffered_tokens", updated)
    py_setitem(slots, "pending", False)
    _ = keep^
    return Python.tuple(output, reports[0], reports[1], reports[2], reports[3])


def _m3_state_forward(block: PythonObject, x: PythonObject, state: PythonObject, step: Bool, context: PythonObject) raises -> PythonObject:
    # Captured fields/slots are passed explicitly: a PythonObject(None)
    # default is evaluated at compile time and needs a live interpreter.
    var fields = samba_fields(block)
    var result = _m3_state_call(block, x, state, step, _samba_block_entries(block, context), fields, samba_fields(state))
    var report_names = [String("h_last_"), String("k_last_"), String("v_last_"), String("theta_last_")]
    for i in range(4):  # driver-loop: over reporting result metadata
        py_setitem(fields, report_names[i], result[i + 1])
    return result[0]


def _samba_resident_forward(fields: PythonObject, config: PythonObject, entries: PythonObject, ids: PythonObject) raises -> PythonObject:
    if py_get(fields, "_native_primitives") is not None:
        return samba_resident_native_forward(fields, config, entries, ids)
    var bi = native_loaded_module("builtins")
    if Float64(py=config["dropout"]) != 0.0:
        return PythonObject(None)
    var enabled = True
    for name in [String("samba_resident_enabled"), String("samba_resident_open"), String("samba_resident_forward"), String("samba_resident_step")]:  # driver-loop: over existing resident-model capability entries
        enabled = enabled and Bool(py=py_call(bi.hasattr, entries, name))
    if not enabled or not Bool(py=_native_function(entries, "samba_resident_enabled")()):
        return PythonObject(None)
    if Bool(py=py_call(bi.hasattr, entries, "training_chunked_lm_head_enabled")) and Bool(py=_native_function(entries, "training_chunked_lm_head_enabled")()):
        return PythonObject(None)
    var session = py_get(fields, "_resident_handle")
    if session is None:
        var kinds = bi.list()
        for i in range(len(config["layers"])):  # driver-loop: over model layer metadata at session creation
            _ = py_call(kinds.append, 0 if String(config["layers"][i]) == "mamba3" else 1)
        var params = Python.list(config["vocab"], config["d_model"])
        for name in [String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate")]:  # driver-loop: over resident-session shape metadata
            _ = py_call(params.append, PythonObject(0) if config[name] is None else config[name])
        _ = py_call(params.append, 1 if Bool(py=config["tie_embeddings"]) else 0)
        _ = py_call(params.append, config["norm_eps"])
        _ = py_call(params.append, fields["n_total"])
        session = py_call(_native_function(entries, "samba_resident_open"), kinds, params, fields["flat"]._addr)
        py_setitem(fields, "_resident_handle", session)
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var output = generation_array([b, l, Int(py=config["vocab"])])
    _ = py_call(_native_function(entries, "samba_resident_forward"), session,
                Python.list(fields["flat"]._addr, ids._addr, output._addr), py_ints([b, l]))
    _ = output  # keep the Python object alive through the native call above
    return output


def _samba_inference_ids(value: PythonObject, vocab: Int, step: Bool, lengths: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var own = py_type(value) is context["neural"]["array_class"]
    var view = value._mv if own else py_call(bi.memoryview, value)
    var shape = value.shape if own else view.shape
    if (len(shape) != 2 and not (step and len(shape) == 1)):
        raise Error("Samba: inputs must be (B,L) integer ids, or (B,) for step")
    var b = Int(py=shape[0])
    var l = 1 if len(shape) == 1 else Int(py=shape[1])
    if b < 1 or l < 1 or (step and l != 1):
        raise Error("Samba: input batch/length must be positive; step takes one token per row")
    var sizes = List[Int]()
    if lengths is not None:
        sizes = _neural_lengths(lengths, b, l, context["neural"])
    var fmt = String(py_call(view.format.lstrip, "=@<"))
    if fmt != "b" and fmt != "B" and fmt != "h" and fmt != "H" and fmt != "i" and fmt != "I" and fmt != "l" and fmt != "L" and fmt != "q" and fmt != "Q":
        raise Error("Samba: inputs must be native-endian integer ids")
    var packed = py_call(native_loaded_module("array").array, fmt)
    _ = py_call(packed.frombytes, py_call(view.tobytes, "C"))
    var column_major = own and String(value.order) == "F"
    var ids = generation_array([b, l], "i")
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=ids._addr))
    _ = ids  # keep the Python object alive through the native call above
    var signed = fmt == "b" or fmt == "h" or fmt == "i" or fmt == "l" or fmt == "q"
    samba_inference_ids_into(Int(py=packed.buffer_info()[0]), Int(py=packed.itemsize), signed, b, l,
                             column_major, vocab, lengths is not None, sizes, dst)
    _ = packed  # keep the Python object alive through the native call above
    _ = ids  # keep the Python object alive through the native call above
    _ = packed^
    return ids


def samba_forward(model: PythonObject, inputs: PythonObject, state: PythonObject, lengths: PythonObject,
                  step: Bool, resident: Bool, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = native_loaded_module("builtins")
    if step and state is None:
        raise Error("SambaStack.step: state is required")
    if lengths is not None and state is not None:
        raise Error("SambaStack.forward: lengths applies to stateless calls only")
    if state is not None and not Bool(py=py_call(bi.isinstance, state, context["state_class"])):
        raise Error("TypeError: SambaStack: state must be a SambaState")
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var entries = PythonObject(None)
    if Bool(py=py_get(fields, "_native_cpu_inference", PythonObject(False))):
        entries = _model_bindings(PythonObject("cpu"), "mamba3", context["neural"], PythonObject("identical"))["primitives"]
    else:
        entries = samba_entries(context, fields["numeric_mode"])
    var d = Int(py=config["d_model"])
    var v = Int(py=config["vocab"])
    var ids = _samba_inference_ids(inputs, v, step, lengths, context)
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var output = PythonObject(None)
    if state is None and resident:
        output = _samba_resident_forward(fields, config, entries, ids)
    if output is None:
        output = samba_provider_forward(fields, config, entries, ids, state, step, context["neural"])
    if output is None:
        var native = py_get(fields, "_native_primitives")
        if native is not None and not neural_descriptor_matches(native, py_call(bi.getattr, entries, "canonical", entries)):
            native = PythonObject(None)
        var blocks = fields["_blocks"]
        var states = PythonObject(None)
        if state is not None:
            var slots = samba_fields(state)
            if Int(py=slots["batch_size"]) != b or len(slots["layers"]) != len(config["layers"]):
                raise Error("Samba: decode state batch or layer count differs")
            states = slots["layers"]
        var arrays = fields["arrays"]
        var hidden = generation_array([b, l, d])
        if native is not None:
            _ = neural_descriptor_execute(native, 0, [Int(py=hidden._addr), Int(py=arrays["embed.weight"]._addr), Int(py=ids._addr)], [b * l, v, d], [])
        else:
            _ = py_call(_native_function(entries, "embedding_forward"), Python.list(hidden._addr, arrays["embed.weight"]._addr, ids._addr), py_ints([b * l, v, d]))
        _ = hidden  # keep the Python object alive through the native call above
        _ = ids  # keep the Python object alive through the native call above
        for i in range(len(config["layers"])):  # driver-loop: over native model layer execution owners
            var block = blocks[i]
            var result = PythonObject(None)
            if String(config["layers"][i]) == "attention":
                result = neural_block_forward_binding(block, hidden, PythonObject(None) if states is None else states[i], PythonObject(step), PythonObject(None), context["neural"])
            elif states is None:
                result = _prefill_mamba3(block, hidden, _samba_block_entries(block, context))
            else:
                result = _m3_state_forward(block, hidden, states[i], step, context)
            hidden = result
        output = generation_array([b, l, v])
        var head = arrays["embed.weight"] if Bool(py=config["tie_embeddings"]) else arrays["lm_head.weight"]
        var afn = state is None
        for name in [String("samba_afn_norm_head_forward"), String("samba_afn_tail_train"), String("samba_afn_embedding_backward_tied")]:  # driver-loop: over the optional fused-tail ABI
            afn = afn and Bool(py=py_call(bi.hasattr, entries, name))
        if afn:
            var params = py_ints([b * l, v, d])
            _ = py_call(params.append, config["norm_eps"])
            _ = py_call(_native_function(entries, "samba_afn_norm_head_forward"), Python.list(output._addr, hidden._addr, arrays["norm_f.weight"]._addr, head._addr), params)
            _ = head  # keep the Python object alive through the native call above
            _ = hidden  # keep the Python object alive through the native call above
            _ = output  # keep the Python object alive through the native call above
        else:
            var normalized = generation_array([b, l, d])
            var params = py_ints([b * l, d])
            _ = py_call(params.append, config["norm_eps"])
            if native is not None:
                _ = neural_descriptor_execute(native, 2, [Int(py=normalized._addr), Int(py=hidden._addr), Int(py=arrays["norm_f.weight"]._addr)], [b * l, d], [Float64(py=config["norm_eps"])])
            else:
                _ = py_call(_native_function(entries, "rms_norm_forward"), Python.list(normalized._addr, hidden._addr, arrays["norm_f.weight"]._addr), params)
            _ = normalized  # keep the Python object alive through the native call above
            _ = hidden  # keep the Python object alive through the native call above
            if native is not None:
                _ = neural_descriptor_execute(native, 4, [Int(py=output._addr), Int(py=normalized._addr), Int(py=head._addr)], [b * l, v, d], [])
            else:
                _ = py_call(_native_function(entries, "linear_forward"), Python.list(output._addr, normalized._addr, head._addr), py_ints([b * l, v, d]))
            _ = normalized  # keep the Python object alive through the native call above
            _ = head  # keep the Python object alive through the native call above
            _ = output  # keep the Python object alive through the native call above
            _ = normalized^
        _ = head^
        _ = hidden^
    if lengths is not None:
        var sizes = _neural_lengths(lengths, b, l, context["neural"])
        var padded = generation_array([b, l, v])
        samba_copy_row_prefixes(Int(py=padded._addr), Int(py=output._addr), sizes, b, l, v)
        _ = padded  # keep the Python object alive through the native call above
        _ = output  # keep the Python object alive through the native call above
        _ = output^  # the copy above read `output` by address
        output = padded
    if step:
        py_setattr(output, "shape", py_call(bi.tuple, py_ints([b, v])))
        py_setattr(output, "strides", py_call(bi.tuple, py_ints([4 * v, 4])))
        py_setattr(output, "ndim", 2)
    _ = ids^
    return output


def samba_loss(model: PythonObject, inputs: PythonObject, targets: PythonObject, context: PythonObject) raises -> PythonObject:
    var logits = samba_forward(model, inputs, PythonObject(None), PythonObject(None), False, False, context)
    var fields = samba_fields(model)
    var v = Int(py=samba_fields(fields["config"])["vocab"])
    var y = samba_ids(targets, v, True, context)
    var rows = Int(py=logits.shape[0]) * Int(py=logits.shape[1])
    if Int(py=y.size) != rows:
        raise Error("SambaStack.loss: targets must contain one class index per logit row")
    var entries = samba_entries(context, fields["numeric_mode"])
    var loss = generation_array([1])
    var row_loss = generation_array([rows])
    var unused_gradient = generation_array([1])
    var params = py_ints([rows, v, -100, 2, 0, 0])
    _ = py_call(params.append, 0.0)
    var primitive = samba_resident_provider(fields, entries)
    if primitive is not None:
        _ = neural_descriptor_execute(primitive, 8, [Int(py=loss._addr), Int(py=row_loss._addr), Int(py=unused_gradient._addr),
            Int(py=logits._addr), Int(py=y._addr)], [rows, v, -100, 2, 0, 0], [0.0])
    else:
        _ = py_call(_native_function(entries, "ce_loss"), loss._addr, row_loss._addr, unused_gradient._addr, logits._addr, y._addr, params)
    _ = unused_gradient  # keep the Python object alive through the native call above
    _ = row_loss  # keep the Python object alive through the native call above
    _ = loss  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    _ = logits  # keep the Python object alive through the native call above
    var result = PythonObject(Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=loss._addr))[0]))
    _ = loss  # keep the Python object alive through the native call above
    _ = loss^
    _ = row_loss^
    _ = unused_gradient^
    _ = logits^
    _ = y^
    return result
