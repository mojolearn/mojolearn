# SPDX-License-Identifier: Apache-2.0
"""Native Samba inference: `SambaStack.forward` / `step` and the loss.

A device build runs every call on the Samba stack program
(training/samba_program.mojo, lane x6-samba-on-executor): one executor
session, one step op per layer, the hidden state on the device. The host
route (CPU builds, the CPU inference model) runs the installed CPU kernels
layer by layer. Public classes and buffer exporters remain CPython-owned
interface records.
"""
from providers.python.samba_resident_pyapi import samba_resident_provider
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from providers.python.neural_descriptor_pyapi import neural_descriptor_matches, neural_descriptor_execute
from core.samba_ids_impl import samba_inference_ids_into, samba_copy_row_prefixes
from std.memory import unsafe_memcpy
from bindings.generation_buffers import generation_initialize_binding, generation_array, generation_address, generation_weight_address, py_ints
from bindings.samba_state_native import samba_fields, samba_entries
from bindings.samba_training_native import samba_ids
from bindings.neural_block_public import neural_training_entries, neural_block_allocate_binding, neural_block_forward_binding, _neural_lengths, neural_mamba_session_call
from bindings.neural_prefill_native import _prefill_mamba3
from bindings.causal_lm_public import _native_function, _mutable, _writeback
from bindings.causal_lm_construct import _model_bindings
from bindings.py_balanced import py_call, py_get, py_setattr, py_setitem, py_type
from bindings.byte_perf_helpers import byte_view_bytes


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
            addresses.append(generation_weight_address(weights[i], keep))
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


# TOMBSTONE: SAMBA_THREE_TIER_FORWARD _samba_resident_forward + the provider tier (samba_provider_forward) + the device per-layer walk of samba_forward (the three-tier driver; one program now: SambaStackProgram, training/samba_program.mojo) deleted 2026-11-09 by lane/x6-samba-on-executor; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/SAMBA_THREE_TIER_FORWARD.patch


def samba_program_arm(fields: PythonObject, entries: PythonObject) raises -> Bool:
    """Whether this stack runs on the Samba stack program: a device training
    binding (it registers `samba_program_open`) and not the CPU inference
    route. Otherwise the host per-layer route below runs (CPU builds)."""
    if Bool(py=py_get(fields, "_native_cpu_inference", PythonObject(False))):
        return False
    return Bool(py=py_call(native_loaded_module("builtins").hasattr, entries, "samba_program_open"))


def samba_program_params(fields: PythonObject, config: PythonObject, b: Int, smax: Int) raises -> PythonObject:
    """`samba_program_open`'s layer codes and integer/float words."""
    var bi = native_loaded_module("builtins")
    var kinds = bi.list()
    for i in range(len(config["layers"])):  # driver-loop: over model layer metadata at program creation
        _ = py_call(kinds.append, 0 if String(config["layers"][i]) == "mamba3" else 1)
    var params = Python.list(config["vocab"], config["d_model"])
    for name in [String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate")]:  # driver-loop: over program shape metadata
        _ = py_call(params.append, PythonObject(0) if config[name] is None else config[name])
    _ = py_call(params.append, 1 if Bool(py=config["tie_embeddings"]) else 0)
    _ = py_call(params.append, config["norm_eps"])
    _ = py_call(params.append, fields["n_total"])
    _ = py_call(params.append, b)
    _ = py_call(params.append, smax)
    return Python.tuple(kinds, params)


def samba_state_words(
    model: PythonObject, state: PythonObject, what: String,
    mut keep: List[PythonObject], mut copies: List[PythonObject],
) raises -> Tuple[List[Int], List[Int]]:
    """The state's piece addresses and cursor words, in the layout the
    program's state entries take: per layer in stack order, the ten Mamba-3
    pieces (`_m3_names()` order) and (buffered_tokens, pending), or k_cache,
    v_cache and (cached_tokens, 0)."""
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
            # The arithmetic-profile word the attention state carries
            # (`_causal_layer_run_bound`'s check and write).
            var pid = py_get(samba_fields(blocks[i]), "_arithmetic_profile")
            var old = py_get(lslots, "arithmetic_profile")
            if old is not None and pid is not None and not Bool(py=old == pid):
                raise Error(what + ": layer " + String(i) + " state arithmetic profile differs")
            if pid is not None:
                py_setitem(lslots, "arithmetic_profile", pid)
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


def samba_set_cursors(model: PythonObject, state: PythonObject, cursors: PythonObject) raises:
    """The program's post-call cursor words onto the state's layer records."""
    var config = samba_fields(samba_fields(model)["config"])
    var layers = samba_fields(state)["layers"]
    for i in range(len(layers)):  # driver-loop: over the stack's layer state owners
        var lslots = samba_fields(layers[i])
        if String(config["layers"][i]) == "attention":
            py_setitem(lslots, "cached_tokens", PythonObject(Int(py=cursors[2 * i])))
        else:
            py_setitem(lslots, "buffered_tokens", PythonObject(Int(py=cursors[2 * i])))
            py_setitem(lslots, "pending", PythonObject(Int(py=cursors[2 * i + 1]) != 0))


def _samba_program_reports(fields: PythonObject, config: PythonObject, b: Int, addresses: PythonObject) raises -> PythonObject:
    """Fresh report arrays for every Mamba-3 layer (four each, the shapes
    `_m3_state_call` gives them), their addresses appended to `addresses`
    in layer order; returns them per layer (None for an attention layer)."""
    var bi = native_loaded_module("builtins")
    var blocks = fields["_blocks"]
    var out = bi.list()
    for i in range(len(config["layers"])):  # driver-loop: over the stack's layers (report owners)
        if String(config["layers"][i]) != "mamba3":
            _ = py_call(out.append, PythonObject(None))
            continue
        var h = Int(py=samba_fields(blocks[i])["nheads"])
        var four = Python.list(generation_array([b, h, 64, 128]), generation_array([b, h, 128]),
                               generation_array([b, h, 64]), generation_array([b, h, 32]))
        for k in range(4):  # small-loop(the four Mamba-3 reports)
            _ = py_call(addresses.append, four[k]._addr)
        _ = py_call(out.append, four)
    return out


def _samba_publish_reports(fields: PythonObject, config: PythonObject, reports: PythonObject) raises:
    """Assign each Mamba-3 block's four reports after the run wrote them."""
    var blocks = fields["_blocks"]
    var names = [String("h_last_"), String("k_last_"), String("v_last_"), String("theta_last_")]
    for i in range(len(config["layers"])):  # driver-loop: over the stack's layers (report owners)
        if reports[i] is None:
            continue
        var block_fields = samba_fields(blocks[i])
        for k in range(4):  # small-loop(the four Mamba-3 reports)
            py_setitem(block_fields, names[k], reports[i][k])


def _samba_program_call(model: PythonObject, fields: PythonObject, config: PythonObject, entries: PythonObject,
                        ids: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    """One per-call run on the stack's program: (B, L) ids -> (B, L, vocab)
    logits, from a zero state (stateless) or from the caller's state, which
    holds the post-call state afterwards. The stateless and the stateful
    programs are kept apart (their attention capacity differs: L versus the
    state's max_tokens); each is reopened when its (B, capacity) changes."""
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var smax = l
    var slot = String("_samba_program")
    if state is not None:
        smax = Int(py=samba_fields(state)["max_tokens"])
        slot = String("_samba_state_program")
    var program = py_get(fields, slot)
    if program is not None:
        var shape = py_call(_native_function(entries, "samba_program_shape"), program)
        if Int(py=shape[0]) != b or Int(py=shape[1]) != smax or not Bool(py=shape[2]):
            py_setitem(fields, slot, PythonObject(None))
            program = PythonObject(None)
    if program is None:
        var opened = samba_program_params(fields, config, b, smax)
        program = py_call(_native_function(entries, "samba_program_open"), opened[0], opened[1], fields["flat"]._addr)
        py_setitem(fields, slot, program)
    var output = generation_array([b, l, Int(py=config["vocab"])])
    var addresses = Python.list(fields["flat"]._addr, ids._addr, output._addr)
    # The Mamba-3 blocks' public reports (h_last_, k_last_, v_last_,
    # theta_last_), as the host route's per-block calls assign them: the
    # program's step ops keep the last run's reports, read back with the
    # run's one wait (training/samba_program.mojo enqueue_reports).
    var reports = _samba_program_reports(fields, config, b, addresses)
    if state is None:
        _ = py_call(_native_function(entries, "samba_program_forward"), program, addresses, py_ints([b, l]),
                    Python.list(), Python.list())
        _ = ids  # keep the Python object alive through the native call above
        _samba_publish_reports(fields, config, reports)
        return output
    # The state was admitted at the public door (samba_admit).
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var words = samba_state_words(model, state, "SambaStack", keep, copies)
    var cursors = py_call(_native_function(entries, "samba_program_forward"), program, addresses, py_ints([b, l]),
                          py_ints(words[0]), py_ints(words[1]))
    _samba_publish_reports(fields, config, reports)
    _writeback(copies)
    _ = keep^
    _ = ids  # keep the Python object alive through the native call above
    samba_set_cursors(model, state, cursors)
    return output


def _samba_inference_ids(value: PythonObject, vocab: Int, step: Bool, lengths: PythonObject, context: PythonObject) raises -> Tuple[PythonObject, List[Int]]:
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
    # One native C-order copy of the ids under a buffer pin (formerly
    # view.tobytes then array.frombytes: two Python copies per token step).
    var item = Int(py=view.itemsize)
    var packed = byte_view_bytes(view, b * l * item)
    var column_major = own and String(value.order) == "F"
    var ids = generation_array([b, l], "i")
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=ids._addr))
    _ = ids  # keep the Python object alive through the native call above
    var signed = fmt == "b" or fmt == "h" or fmt == "i" or fmt == "l" or fmt == "q"
    samba_inference_ids_into(Int(packed.unsafe_ptr()), item, signed, b, l,
                             column_major, vocab, lengths is not None, sizes, dst)
    _ = packed  # keep the Python object alive through the native call above
    _ = ids  # keep the Python object alive through the native call above
    _ = packed^
    # The admitted ragged lengths go back to samba_forward, which formerly
    # admitted them a second time for the output prefix copy.
    return (ids, sizes^)


def samba_forward(model: PythonObject, inputs: PythonObject, state: PythonObject, lengths: PythonObject,
                  step: Bool, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = native_loaded_module("builtins")
    if step and state is None:
        raise Error("SambaStack.step: state is required")
    if lengths is not None and state is not None:
        raise Error("SambaStack.forward: lengths applies to stateless calls only")
    if state is not None and not Bool(py=py_call(bi.isinstance, state, context["state_class"])):
        raise Error("TypeError: SambaStack: state must be a SambaState")
    if state is not None and py_get(samba_fields(state), "_resident_session") is not None:
        # A SambaDecodeSession holds this state on the device
        # (bindings/samba_decode_native.mojo); its arrays are stale.
        raise Error("SambaStack: the state is owned by an open decode session (its pieces live in the session and the "
                    + "caller's arrays are stale); call sync_state() or close() on the session first")
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var entries = PythonObject(None)
    if Bool(py=py_get(fields, "_native_cpu_inference", PythonObject(False))):
        entries = _model_bindings(PythonObject("cpu"), "mamba3", context["neural"], PythonObject("identical"))["primitives"]
    else:
        entries = samba_entries(context, fields["numeric_mode"])
    var d = Int(py=config["d_model"])
    var v = Int(py=config["vocab"])
    var admitted = _samba_inference_ids(inputs, v, step, lengths, context)
    var ids = admitted[0]
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var output = PythonObject(None)
    if samba_program_arm(fields, entries):
        # ONE path on a device build: the stack program (lane x6).
        output = _samba_program_call(model, fields, config, entries, ids, state, context)
    else:
        # The host route (CPU builds and the CPU inference model): the
        # installed CPU kernels layer by layer. Device builds never come
        # here (samba_program_arm).
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
        var sizes = admitted[1].copy()
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
    # Row N169: the logits come from `SambaStack.forward`'s one path (the
    # stack program on a device build, lane x6), so no layer loop moves the
    # hidden state through host arrays.
    # OWED: the cross entropy inside the program (logits never leave the
    # device) needs a loss op after the head in training/samba_program.mojo.
    var logits = samba_forward(model, inputs, PythonObject(None), PythonObject(None), False, context)
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
