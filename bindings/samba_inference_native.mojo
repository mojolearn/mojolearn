# SPDX-License-Identifier: Apache-2.0
"""Native Samba inference over the existing resident model and block owners.

The resident model/session remains the persistent numerical owner. Public
classes and buffer exporters remain CPython-owned interface records; this
module does not claim that all public state has native struct ownership.
"""
from std.python import Python, PythonObject
from std.memory import memcpy
from bindings.generation_buffers import generation_initialize_binding, generation_array, generation_address, py_ints
from bindings.samba_state_native import samba_fields, samba_entries
from bindings.samba_training_native import samba_ids
from bindings.neural_block_public import neural_training_entries, neural_block_allocate_binding, neural_block_forward_binding, _neural_lengths
from bindings.neural_prefill_native import _prefill_mamba3
from bindings.causal_lm_public import _native_function, _mutable, _writeback
from bindings.causal_lm_construct import _model_bindings


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
    var bi = Python.import_module("builtins")
    var fields = samba_fields(state)
    fields["batch_size"] = bi.int(b)
    fields["max_tokens"] = bi.int(maximum)
    fields["layers"] = bi.list(layers)
    return PythonObject(None)


def samba_allocate(model: PythonObject, batch: PythonObject, maximum: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = Python.import_module("builtins")
    var b = Int(py=bi.int(batch))
    var cap = Int(py=bi.int(maximum))
    if b < 1 or cap < 1:
        raise Error("SambaStack.allocate_state: batch_size and max_tokens must be positive")
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var blocks = fields["_blocks"]
    var states = bi.list()
    for i in range(len(config["layers"])):  # driver-loop: over persistent layer state owners
        var block = blocks[i]
        if String(config["layers"][i]) == "attention":
            states.append(neural_block_allocate_binding(block, PythonObject(b), PythonObject(cap), context["neural"]))
        else:
            var state = bi.object.__new__(context["mamba3_state_class"])
            var slots = samba_fields(state)
            var shapes = _m3_shapes(b, Int(py=samba_fields(block)["nheads"]))
            var names = _m3_names()
            for j in range(len(names)):  # driver-loop: over the fixed Mamba3 continuation buffer ABI
                slots[names[j]] = generation_array(shapes[j])
            slots["buffered_tokens"] = 0
            slots["pending"] = False
            states.append(state)
    var result = bi.object.__new__(context["state_class"])
    _ = samba_state_construct(result, PythonObject(b), PythonObject(cap), states)
    return result



def _samba_block_entries(block: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var nc = context["neural"]
    if Bool(py=bi.isinstance(block, nc["block_classes"]["cpu"]["mamba3"])):
        return _model_bindings(PythonObject("cpu"), "mamba3", nc, PythonObject("identical"))["mamba3"]
    return neural_training_entries(samba_fields(block), nc)

def _m3_state_forward(block: PythonObject, x: PythonObject, state: PythonObject, step: Bool, context: PythonObject) raises -> PythonObject:
    var fields = samba_fields(block)
    var slots = samba_fields(state)
    if slots.get("_resident_session") is not None:
        raise Error("Mamba3: state belongs to an open resident decode session; synchronize or close it first")
    var entries = _samba_block_entries(block, context)
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var d = Int(py=fields["d_model"])
    var h = Int(py=fields["nheads"])
    var shapes = _m3_shapes(b, h)
    var names = _m3_names()
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var addresses = py_ints([generation_address(x, keep)])
    var weights = fields["_w"]
    for i in range(len(weights)):  # driver-loop: over existing Mamba3 parameter ABI slots
        addresses.append(generation_address(weights[i], keep))
    for i in range(len(names)):  # driver-loop: over mutable continuation buffers
        addresses.append(_mutable(slots[names[i]], shapes[i], keep, copies))
    var output = generation_array([b, l, d])
    var reports = Python.list(generation_array([b, h, 64, 128]), generation_array([b, h, 128]),
                              generation_array([b, h, 64]), generation_array([b, h, 32]))
    addresses.append(generation_address(output, keep))
    for i in range(4):  # driver-loop: over the four reporting buffers
        addresses.append(generation_address(reports[i], keep))
    var q = Int(py=slots["buffered_tokens"])
    var pending = 1 if Bool(py=slots["pending"]) else 0
    var updated = PythonObject(None)
    if step:
        updated = _native_function(entries, "mamba3_decode_step")(addresses, py_ints([b, d, q, pending]))
    else:
        updated = _native_function(entries, "mamba3_forward")(addresses, py_ints([b, l, d, q, pending]))
    _writeback(copies)
    slots["buffered_tokens"] = updated
    slots["pending"] = False
    var report_names = [String("h_last_"), String("k_last_"), String("v_last_"), String("theta_last_")]
    for i in range(4):  # driver-loop: over reporting result metadata
        fields[report_names[i]] = reports[i]
    _ = keep^
    return output


def _samba_resident_forward(fields: PythonObject, config: PythonObject, entries: PythonObject, ids: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if Float64(py=config["dropout"]) != 0.0:
        return PythonObject(None)
    var enabled = True
    for name in [String("samba_resident_enabled"), String("samba_resident_open"), String("samba_resident_forward"), String("samba_resident_step")]:  # driver-loop: over existing resident-model capability entries
        enabled = enabled and Bool(py=bi.hasattr(entries, name))
    if not enabled or not Bool(py=_native_function(entries, "samba_resident_enabled")()):
        return PythonObject(None)
    if Bool(py=bi.hasattr(entries, "training_chunked_lm_head_enabled")) and Bool(py=_native_function(entries, "training_chunked_lm_head_enabled")()):
        return PythonObject(None)
    var session = fields.get("_resident_handle")
    if session is None:
        var kinds = bi.list()
        for i in range(len(config["layers"])):  # driver-loop: over model layer metadata at session creation
            kinds.append(0 if String(config["layers"][i]) == "mamba3" else 1)
        var params = Python.list(config["vocab"], config["d_model"])
        for name in [String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate")]:  # driver-loop: over resident-session shape metadata
            params.append(PythonObject(0) if config[name] is None else config[name])
        params.append(1 if Bool(py=config["tie_embeddings"]) else 0)
        params.append(config["norm_eps"])
        params.append(fields["n_total"])
        session = _native_function(entries, "samba_resident_open")(kinds, params, fields["flat"]._addr)
        fields["_resident_handle"] = session
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var output = generation_array([b, l, Int(py=config["vocab"])])
    _native_function(entries, "samba_resident_forward")(session,
        Python.list(fields["flat"]._addr, ids._addr, output._addr), py_ints([b, l]))
    return output


def _samba_inference_ids(value: PythonObject, vocab: Int, step: Bool, lengths: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var own = bi.type(value) is context["neural"]["array_class"]
    var view = value._mv if own else bi.memoryview(value)
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
    var fmt = String(view.format.lstrip("=@<"))
    if fmt != "b" and fmt != "B" and fmt != "h" and fmt != "H" and fmt != "i" and fmt != "I" and fmt != "l" and fmt != "L" and fmt != "q" and fmt != "Q":
        raise Error("Samba: inputs must be native-endian integer ids")
    var packed = Python.import_module("array").array(fmt)
    packed.frombytes(view.tobytes("C"))
    var column_major = own and String(value.order) == "F"
    var ids = generation_array([b, l], "i")
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=ids._addr))
    for i in range(b * l):
        if lengths is not None and i % l >= sizes[i // l]:
            continue
        var source = i // l + (i % l) * b if column_major else i
        var id = packed[source]
        if Bool(py=id < 0) or Bool(py=id >= vocab):
            raise Error("Samba: input id outside vocabulary")
        dst[i] = Int32(Int(py=id))
    return ids


def samba_forward(model: PythonObject, inputs: PythonObject, state: PythonObject, lengths: PythonObject,
                  step: Bool, resident: Bool, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = Python.import_module("builtins")
    if step and state is None:
        raise Error("SambaStack.step: state is required")
    if lengths is not None and state is not None:
        raise Error("SambaStack.forward: lengths applies to stateless calls only")
    if state is not None and not Bool(py=bi.isinstance(state, context["state_class"])):
        raise Error("TypeError: SambaStack: state must be a SambaState")
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var entries = PythonObject(None)
    if Bool(py=fields.get("_native_cpu_inference", PythonObject(False))):
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
        var blocks = fields["_blocks"]
        var states = PythonObject(None)
        if state is not None:
            var slots = samba_fields(state)
            if Int(py=slots["batch_size"]) != b or len(slots["layers"]) != len(config["layers"]):
                raise Error("Samba: decode state batch or layer count differs")
            states = slots["layers"]
        var arrays = fields["arrays"]
        var hidden = generation_array([b, l, d])
        _native_function(entries, "embedding_forward")(Python.list(hidden._addr, arrays["embed.weight"]._addr, ids._addr), py_ints([b * l, v, d]))
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
            afn = afn and Bool(py=bi.hasattr(entries, name))
        if afn:
            var params = py_ints([b * l, v, d])
            params.append(config["norm_eps"])
            _native_function(entries, "samba_afn_norm_head_forward")(Python.list(output._addr, hidden._addr, arrays["norm_f.weight"]._addr, head._addr), params)
        else:
            var normalized = generation_array([b, l, d])
            var params = py_ints([b * l, d])
            params.append(config["norm_eps"])
            _native_function(entries, "rms_norm_forward")(Python.list(normalized._addr, hidden._addr, arrays["norm_f.weight"]._addr), params)
            _native_function(entries, "linear_forward")(Python.list(output._addr, normalized._addr, head._addr), py_ints([b * l, v, d]))
            _ = normalized^
        _ = head^
        _ = hidden^
    if lengths is not None:
        var sizes = _neural_lengths(lengths, b, l, context["neural"])
        var padded = generation_array([b, l, v])
        for row in range(b):
            memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=padded._addr)) + row * l * v,
                   src=Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=output._addr)) + row * l * v, count=sizes[row] * v)
        output = padded
    if step:
        output.shape = bi.tuple(py_ints([b, v]))
        output.strides = bi.tuple(py_ints([4 * v, 4]))
        output.ndim = 2
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
    params.append(0.0)
    _native_function(entries, "ce_loss")(loss._addr, row_loss._addr, unused_gradient._addr, logits._addr, y._addr, params)
    var result = PythonObject(Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=loss._addr))[0]))
    _ = loss^
    _ = row_loss^
    _ = unused_gradient^
    _ = logits^
    _ = y^
    return result
