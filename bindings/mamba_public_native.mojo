# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Native Mamba public operations: Mamba3Block and the Mamba decode sessions.

`Mamba3Block.forward` (plain, carried state and ragged `lengths=`), `step`,
`allocate_state`, the owned weight snapshot (`install_owned_weights`,
`release_owned_weights`, `export_owned_weights`, `session_info`), pickling,
`Mamba3State.set_input_states`, and every method of the Mamba-1/2/3 resident
decode sessions are each ONE call into this module (`mamba_native_call`). The
compiled mamba entries reached here are the ones the retired Python drivers in
`python/mojolearn/_mamba_impl.py` called, with the same addresses in the same
order and the same parameters, so no output bit moves. No Python callable of the
package is invoked: only builtins, buffer exporters and compiled entries.

Public forward/step retain reports in the instance dictionary natively and
return only the output. Internal tuple-returning operations remain available
for other compiled owners. Dictionary insertion retains each report before
the temporary tuple is released.
"""
from std.python import Python, PythonObject
from std.os import getenv
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from bindings.generation_buffers import (
    GENERATION_BOUNDARY, generation_initialize_binding, generation_array, generation_address, py_ints,
)
from bindings.causal_lm_public import _native_function, _mutable, _writeback
from bindings.causal_lm_construct import _model_bindings, _model_block
from bindings.neural_block_public import neural_training_entries, _neural_input, _neural_lengths, _neural_int_words
from bindings.neural_block_admission import neural_block_fields, neural_block_snapshot, neural_block_f32, neural_array_snapshot
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.native_loaded_module import native_loaded_module
from providers.python.autograd_program_arrays_pyapi import program_numeric_scalar
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter
from bindings.neural_prefill_native import _prefill_mamba3_reports
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_handle, neural_descriptor_reserved_execute,
    neural_descriptor_execute, neural_descriptor_matches, neural_descriptor_reuse,
)
from bindings.samba_inference_native import _m3_shapes, _m3_names, _m3_state_call
from core.neural_block_public_impl import neural_copy_row_prefixes
from core.native_descriptor_abi import MAMBA_BLOCK_ABI_NAME, MAMBA_STATE_ABI_NAME
from mamba.impl.mamba_session_impl import mamba_copy_f32, mamba_host_copy


# ---------------------------------------------------------------- Mamba3Block


def _mamba_record_fields(value: PythonObject, expected: PythonObject, names: List[String]) raises -> PythonObject:
    # Validate the exact native record and every attribute route BEFORE the
    # dictionary is read. object.__getattribute__("__dict__") alone would still
    # execute a replaced data descriptor; _fields uses GenericGetDict after
    # checking that descriptor. Raw namespaces below never bind properties.
    var bi = native_loaded_module("builtins")
    if py_type(value) is not expected or py_type(expected) is not bi.type:
        raise Error("TypeError: Mamba state/session requires its exact connected native class")
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, expected)
    for i in range(len(ancestry)):  # driver-loop: record layouts before any attribute read/write
        var namespace = _class_fields(ancestry[i])
        if py_contains(namespace, "__getattr__"):
            raise Error("TypeError: Mamba native records cannot invoke Python attribute hooks")
        for hook in [String("__getattribute__"), String("__setattr__")]:
            if py_contains(namespace, hook) and namespace[hook] is not bi.object.__dict__[hook]:
                raise Error("TypeError: Mamba native records cannot invoke Python attribute hooks")
        for name in names:  # driver-loop: fixed state/session fields cannot be replaced by descriptors
            if py_contains(namespace, name):
                raise Error("TypeError: Mamba native record has a replaced field descriptor: " + name)
    return _fields(value)


def _mamba_scalar(value: PythonObject, name: String) raises:
    var bi = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is bi.str or cls is bi.bytes or program_numeric_scalar(value):
        return
    raise Error("TypeError: Mamba " + name + " requires a native scalar without conversion hooks")


def _mamba_buffer(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(value) is context["array_class"]:
        return neural_array_snapshot(value, context)
    admit_native_buffer_exporter(value)
    # memoryview captures shape/stride metadata once and pins the exact native
    # exporter. Later copies inspect this retained view, never a caller hook.
    return py_call(native_loaded_module("builtins").memoryview, value)


def _mamba_state_names(kind: String) -> List[String]:
    if kind == "mamba1":
        return [String("conv_window"), String("h"), String("profile_id"), String("absolute_position"),
                String("affine_boundary"), String("affine_a"), String("affine_b"), String("_nn34_usable"), String("_resident_session")]
    if kind == "mamba2":
        return [String("conv_window"), String("h"), String("buffer_xbc"), String("buffer_dtraw"),
                String("buffered_tokens"), String("_resident_session")]
    return [String("theta"), String("h"), String("buffer_qrot"), String("buffer_krot"), String("buffer_v"),
            String("buffer_dt"), String("buffer_sig"), String("buffer_adt"), String("pending_k"), String("pending_v"),
            String("buffered_tokens"), String("pending"), String("_resident_session")]


def _mamba_state_fields(state: PythonObject, kind: String, context: PythonObject,
                        initialized: Bool = True) raises -> PythonObject:
    var expected = context["mamba3_state_class"] if kind == "mamba3" else context["generation_classes"][2][kind]
    var names = _mamba_state_names(kind)
    var fields = _mamba_record_fields(state, expected, names)
    if py_contains(_class_fields(expected), "__del__"):
        raise Error("TypeError: Mamba state snapshots cannot invoke Python finalizers")
    if initialized:
        fields = py_call(native_loaded_module("builtins").dict, fields)
        for name in names:  # driver-loop: capture fixed mutable state metadata before execution
            if not py_contains(fields, name):
                continue
            var item = fields[name]
            if name == "profile_id" or name == "absolute_position" or name == "_nn34_usable" or name == "buffered_tokens" or name == "pending" or name == "_resident_session":
                _mamba_scalar(item, name)
            elif item is not None:
                py_setitem(fields, name, _mamba_buffer(item, context))
    return fields


def mamba_state_init(state: PythonObject, kind: String, values: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if kind != "mamba1" and kind != "mamba2" and kind != "mamba3":
        raise Error("TypeError: unknown native Mamba state kind")
    var names = _mamba_state_names(kind)
    var count = 7 if kind == "mamba1" else (5 if kind == "mamba2" else 12)
    if py_type(values) is not bi.tuple or len(values) != count:
        raise Error("TypeError: Mamba state construction requires its native argument tuple")
    var fields = _mamba_state_fields(state, kind, context, False)
    var converted = Python.list()
    for i in range(count):  # driver-loop: constructor reference storage and native scalar conversion
        var value = values[i]
        if (kind == "mamba1" and (i == 2 or i == 3)) or (kind == "mamba2" and i == 4) or (kind == "mamba3" and i == 10):
            _mamba_scalar(value, names[i])
            value = py_call(bi.int, value)
        elif kind == "mamba3" and i == 11:
            _mamba_scalar(value, names[i])
            value = PythonObject(Bool(py=value))
        _ = py_call(converted.append, value)
    for i in range(count):  # driver-loop: publish only after all scalar arguments were admitted
        py_setitem(fields, names[i], converted[i])
    if kind == "mamba1":
        py_setitem(fields, "_nn34_usable", True)
    return PythonObject(None)


def _mamba_session_fields(session: PythonObject, context: PythonObject,
                          initialized: Bool = True) raises -> PythonObject:
    var classes = context["mamba_session_classes"]
    var cls = py_type(session)
    var expected = PythonObject(None)
    for name in [String("mamba1"), String("mamba2"), String("mamba3")]:
        if cls is classes[name]:
            expected = classes[name]
    var fields = _mamba_record_fields(session, expected, [String("_handle"), String("_block"), String("_state")])
    if initialized:
        var hobj = fields["_handle"]
        var handle = hobj.downcast_value_ptr[MambaDecodeHandle]()
        if cls is not classes[_kind_name(handle[].kind)]:
            raise Error("TypeError: Mamba session kind differs from its native handle")
        _ = neural_block_fields(fields["_block"], context)
        _ = _mamba_state_fields(fields["_state"], _kind_name(handle[].kind), context, False)
    return fields


def _m3_entries(block: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # The binding the retired `Mamba3Block._extension()` returned: the shipped
    # host binding for the CPU inference class, the block's tier otherwise.
    var bi = Python.import_module("builtins")
    if Bool(py=py_call(bi.isinstance, block, context["block_classes"]["cpu"]["mamba3"])):
        var entries = _model_bindings(PythonObject("cpu"), "mamba3", context, PythonObject("identical"))["mamba3"]
        var block_provider = PythonObject(None)
        var state_provider = PythonObject(None)
        if getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
            block_provider = neural_descriptor_reuse(py_get(fields, "_native_mamba_block_provider_cache"),
                _native_function(entries, "mamba_block_api_v1"), String(MAMBA_BLOCK_ABI_NAME), 44)
            state_provider = neural_descriptor_reuse(py_get(fields, "_native_mamba_state"),
                _native_function(entries, "mamba_state_api_v1"), String(MAMBA_STATE_ABI_NAME), 6)
        py_setitem(fields, "_native_mamba_block_provider_cache", block_provider)
        py_setitem(fields, "_native_mamba_state", state_provider)
        return entries
    return neural_training_entries(fields, context)


def _m3_canonical(entries: PythonObject) raises -> PythonObject:
    if Bool(py=py_call(Python.import_module("builtins").hasattr, entries, "canonical")):
        return entries.canonical
    return entries


def _m3_allocate(block: PythonObject, fields: PythonObject, batch: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var b = Int(py=py_call(bi.int, batch))
    if b < 1:
        raise Error("Mamba3Block: batch_size must be positive, got " + String(b))
    var state = py_call(bi.object.__new__, context["mamba3_state_class"])
    var slots = _fields(state)
    var shapes = _m3_shapes(b, Int(py=fields["nheads"]))
    var names = _m3_names()
    for j in range(len(names)):  # driver-loop: over the fixed Mamba3 continuation buffer ABI
        py_setitem(slots, names[j], generation_array(shapes[j]))
    py_setitem(slots, "buffered_tokens", 0)
    py_setitem(slots, "pending", False)
    return state


def _m3_fresh(block: PythonObject, fields: PythonObject, x: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    # The retired `_call`: the discard-only prefill where the binding has it,
    # else the carried-state entry from a fresh zero state.
    if Bool(py=py_call(Python.import_module("builtins").hasattr, entries, "mamba3_forward_fresh")):
        return _prefill_mamba3_reports(block, x, entries, fields)
    var state = _m3_allocate(block, fields, x.shape[0], context)
    return _m3_state_call(block, x, state, False, entries, fields, _fields(state))


def _mamba3_forward_captured(
    block: PythonObject, input: PythonObject, state: PythonObject, lengths: PythonObject, step: Bool,
    context: PythonObject, fields: PythonObject,
) raises -> PythonObject:
    """`(y, h_last, k_last, v_last, theta_last)` using one retained snapshot."""
    var d = Int(py=fields["d_model"])
    if step and state is None:
        raise Error("Mamba3Block.step: state is required (a decode step continues a sequence; allocate_state(B) makes the fresh one)")
    var x = _neural_input(input, d, step, context)
    var entries = _m3_entries(block, fields, context)
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    if lengths is not None:
        if state is not None:
            raise Error("Mamba3Block.forward: lengths= (a ragged, right-padded batch) cannot carry a state; run the ragged prefill with state=None, or continue each sequence as its own batch")
        var sizes = _neural_lengths(lengths, b, l, context)
        var padded = generation_array([b, l, d])
        neural_copy_row_prefixes(Int(py=padded._addr), Int(py=x._addr), sizes, b, l, d)
        _ = x^  # the copy above read `x` by address
        var ran = _m3_fresh(block, fields, padded, entries, context)
        var result = generation_array([b, l, d])
        neural_copy_row_prefixes(Int(py=result._addr), Int(py=ran[0]._addr), sizes, b, l, d)
        return Python.tuple(result, ran[1], ran[2], ran[3], ran[4])
    if state is None:
        return _m3_fresh(block, fields, x, entries, context)
    var captured = _mamba_state_fields(state, "mamba3", context)
    var result = _m3_state_call(block, x, state, step, entries, fields, captured)
    var slots = _fields(state)
    py_setitem(slots, "buffered_tokens", captured["buffered_tokens"])
    py_setitem(slots, "pending", captured["pending"])
    return result


def _mamba3_publish_runtime(fields: PythonObject, block: PythonObject) raises:
    var live = _fields(block)
    for name in [String("_native_mamba_block_provider_cache"), String("_native_mamba_state"),
                 String("_prefill_session"), String("_prefill_binding"), String("_owned_generation"),
                 String("_owned_native_provider"), String("_owned_restore_pending")]:
        if py_contains(fields, name):
            py_setitem(live, name, fields[name])


def mamba3_forward(
    block: PythonObject, input: PythonObject, state: PythonObject, lengths: PythonObject, step: Bool,
    context: PythonObject,
) raises -> PythonObject:
    var fields = neural_block_snapshot(block, context)
    try:
        return _mamba3_forward_captured(block, input, state, lengths, step, context, fields)
    finally:
        # Session creation/restore and native invalidation survive a failure,
        # while numerical configuration and captured weights stay local.
        _mamba3_publish_runtime(fields, block)


def mamba3_allocate(block: PythonObject, batch: PythonObject, context: PythonObject) raises -> PythonObject:
    return _m3_allocate(block, _fields(block), batch, context)


def _mamba3_weight_names() -> List[String]:
    # Fixed numerical ABI, not a mutable class attribute lookup.
    return [String("block_norm.weight"), String("in_proj.weight"), String("dt_bias"),
            String("B_norm.weight"), String("C_norm.weight"), String("B_bias"),
            String("C_bias"), String("D"), String("out_proj.weight")]


def _m3_checked_weights(block: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # The current weights through the constructor's own native checks, as the
    # retired `type(self)(...)` re-validation did, without a Python constructor.
    var bi = Python.import_module("builtins")
    var plan = py_call(bi.object.__new__, context["config_context"]["plan_class"])
    py_setattr(plan, "kind", PythonObject("mamba3"))
    py_setattr(plan, "d_model", fields["d_model"])
    py_setattr(plan, "block_kwargs", bi.dict())
    var named = bi.dict()
    var names = _mamba3_weight_names()
    var weights = fields["_w"]
    for i in range(len(names)):  # driver-loop: over the fixed block weight registry
        py_setitem(named, names[i], weights[i])
    var route = fields["_native_route"]
    var checked = _model_block(named, plan, PythonObject("fp32_v1"), route,
                               _model_bindings(route, "mamba3", context, py_get(fields, "numeric_mode")), context)
    return checked.__dict__["_w"]


def mamba3_install_owned(block: PythonObject, context: PythonObject, fields: PythonObject) raises -> PythonObject:
    # Publish session, binding and generation together after successful upload.
    var bi = Python.import_module("builtins")
    var entries = _m3_entries(block, fields, context)
    if not Bool(py=py_call(bi.hasattr, entries, "mamba3_owned_weights_enabled")) or not Bool(py=_native_function(entries, "mamba3_owned_weights_enabled")()):
        raise Error("RuntimeError: Mamba3 owned weights require the NN40 IDENTICAL arm")
    var weights = _m3_checked_weights(block, fields, context)
    var canonical = _m3_canonical(entries)
    var session = py_get(fields, "_prefill_session")
    if session is None or py_get(fields, "_prefill_binding") is not canonical:
        session = _native_function(entries, "mamba3_prefill_session_create")()
    var keep = List[PythonObject]()
    var addresses = py_ints([])
    for i in range(len(weights)):  # driver-loop: over the fixed owned-weight registry
        _ = py_call(addresses.append, generation_address(weights[i], keep))
    var generation = PythonObject(None)
    var native = py_get(fields, "_native_mamba_block_provider_cache")
    if neural_descriptor_matches(native, canonical):
        var handle = neural_descriptor_handle(native,
            _native_function(entries, "mamba_prefill_native_handle_v1"), session)
        generation = PythonObject(neural_descriptor_reserved_execute(native, session, handle, 0, 1, 15,
            _neural_int_words(addresses), [Int(py=fields["d_model"])], []))
    else:
        native = PythonObject(None)
        generation = py_call(_native_function(entries, "mamba3_prefill_session_install_weights"), session, addresses, fields["d_model"])
    py_setitem(fields, "_prefill_session", session)
    py_setitem(fields, "_prefill_binding", canonical)
    py_setitem(fields, "_owned_generation", generation)
    py_setitem(fields, "_owned_native_provider", native)
    py_setitem(fields, "_owned_restore_pending", False)
    _ = keep^
    return generation


def mamba3_install_owned(block: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = neural_block_snapshot(block, context)
    try:
        return mamba3_install_owned(block, context, fields)
    finally:
        _mamba3_publish_runtime(fields, block)


def mamba3_release_owned(block: PythonObject) raises -> PythonObject:
    var fields = _fields(block)
    var session = py_get(fields, "_prefill_session")
    var binding = py_get(fields, "_prefill_binding")
    if session is not None and binding is not None:
        var native = py_get(fields, "_owned_native_provider")
        if neural_descriptor_matches(native, binding):
            var handle = neural_descriptor_handle(native,
                _native_function(binding, "mamba_prefill_native_handle_v1"), session)
            _ = neural_descriptor_execute(native, 18, [], [handle], [], False)
        else:
            _ = py_call(_native_function(binding, "mamba3_prefill_session_close"), session)
    py_setitem(fields, "_prefill_session", PythonObject(None))
    py_setitem(fields, "_prefill_binding", PythonObject(None))
    py_setitem(fields, "_owned_generation", PythonObject(None))
    py_setitem(fields, "_owned_native_provider", PythonObject(None))
    py_setitem(fields, "_owned_restore_pending", PythonObject(False))
    return PythonObject(None)


def mamba3_session_info(block: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = _fields(block)
    var session = py_get(fields, "_prefill_session")
    var binding = py_get(fields, "_prefill_binding")
    if session is None or binding is None or not Bool(py=py_call(bi.hasattr, binding, "mamba3_prefill_session_info")):
        return PythonObject(None)
    var values = py_call(_native_function(binding, "mamba3_prefill_session_info"), session)
    var keys = [String("weight_uploads"), String("weight_recopies"), String("weight_reuses"),
                String("backward_reuses"), String("backward_recomputes"), String("stages_held")]
    var result = bi.dict()
    for i in range(len(keys)):  # small-loop(six named session counters)
        py_setitem(result, keys[i], Int(py=values[i]))
    return result


def mamba3_export_owned(block: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = _fields(block)
    var generation = py_get(fields, "_owned_generation")
    if generation is None:
        raise Error("RuntimeError: Mamba3 has no installed owned snapshot")
    var dm = Int(py=fields["d_model"])
    var di = Int(py=fields["d_inner"])
    var nh = Int(py=fields["nheads"])
    var dip = Int(py=fields["d_in_proj"])
    var shapes = List[List[Int]]()
    shapes.append([dm])
    shapes.append([dip, dm])
    shapes.append([nh])
    shapes.append([128])
    shapes.append([128])
    shapes.append([nh, 128])
    shapes.append([nh, 128])
    shapes.append([nh])
    shapes.append([dm, di])
    var arrays = bi.list()
    var keep = List[PythonObject]()
    var addresses = py_ints([])
    for i in range(len(shapes)):  # driver-loop: over the fixed parameter buffer list
        var array = generation_array(shapes[i])
        _ = py_call(arrays.append, array)
        _ = py_call(addresses.append, generation_address(array, keep))
    var binding = fields["_prefill_binding"]
    var session = fields["_prefill_session"]
    var native = py_get(fields, "_owned_native_provider")
    if neural_descriptor_matches(native, binding):
        var handle = neural_descriptor_handle(native,
            _native_function(binding, "mamba_prefill_native_handle_v1"), session)
        _ = neural_descriptor_reserved_execute(native, session, handle, 0, 1, 17,
            _neural_int_words(addresses), [Int(py=generation)], [])
    else:
        _ = py_call(_native_function(binding, "mamba3_prefill_session_export_weights"), session, addresses, generation)
    var result = bi.dict()
    var names = _mamba3_weight_names()
    for i in range(len(names)):  # driver-loop: over named parameter slots
        py_setitem(result, names[i], arrays[i])
    _ = keep^
    return result


def mamba3_getstate(block: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = _fields(block)
    var result = py_call(bi.dict, fields)
    if py_get(fields, "_owned_generation") is not None:
        var snapshot = mamba3_export_owned(block)
        var names = _mamba3_weight_names()
        var weights = bi.list()
        for i in range(len(names)):  # driver-loop: over the fixed parameter name list
            _ = py_call(weights.append, snapshot[names[i]])
        py_setitem(result, "_w", weights)
        py_setitem(result, "_owned_restore_pending", True)
    py_setitem(result, "_owned_generation", PythonObject(None))
    _ = py_call(result.pop, "_prefill_session", PythonObject(None))
    _ = py_call(result.pop, "_prefill_binding", PythonObject(None))
    var transient_names = [String("_owned_native_provider"), String("_native_mamba_block_provider_cache"),
                           String("_native_mamba_state"), String("_native_forward_owner")]
    for name in transient_names:  # driver-loop: over ephemeral native owners excluded from serialized model state
        _ = py_call(result.pop, name, PythonObject(None))
    return result


def mamba3_setstate(block: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(state) is not bi.dict:
        raise Error("TypeError: Mamba checkpoint state requires an ordinary dictionary")
    var keys = py_call(bi.list, state.keys())
    for i in range(len(keys)):  # driver-loop: never hash a checkpoint key before exact string admission
        if py_type(keys[i]) is not bi.str:
            raise Error("TypeError: Mamba checkpoint state requires exact string keys")
    var cls = py_type(block)
    if cls is not context["block_classes"]["cpu"]["mamba3"] and cls is not context["block_classes"]["gpu"]["mamba3"]:
        raise Error("TypeError: Mamba checkpoint owner requires its exact connected block class")
    # Validate a temporary owner before publishing the supplied checkpoint.
    # This also validates an uninitialized object's layout, as pickle's native
    # allocation has not yet stored _native_kind/dimension metadata.
    var candidate = py_call(bi.object.__new__, cls)
    var candidate_fields = _fields(candidate)
    _ = py_call(candidate_fields.update, state)
    _ = neural_block_fields(candidate, context)
    var fields = _fields(block)
    _ = py_call(fields.update, candidate_fields)
    py_setitem(fields, "_prefill_session", PythonObject(None))
    py_setitem(fields, "_prefill_binding", PythonObject(None))
    py_setitem(fields, "_owned_generation", PythonObject(None))
    py_setitem(fields, "_owned_native_provider", PythonObject(None))
    py_setitem(fields, "_native_mamba_block_provider_cache", PythonObject(None))
    if String(fields["_native_route"]) == "gpu" and String(fields["numeric_mode"]) == "identical" and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
        var entries = _m3_entries(block, fields, context)
        py_setitem(fields, "_prefill_session", _native_function(entries, "mamba3_prefill_session_create")())
        py_setitem(fields, "_prefill_binding", _m3_canonical(entries))
    return PythonObject(None)


def _shape_of(value: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_type(value) is g[].array_cls:
        return value.shape
    return py_call(g[].bi.memoryview, value).shape


def mamba3_set_input_states(
    state: PythonObject, theta: PythonObject, h: PythonObject, k: PythonObject, v: PythonObject,
    context: PythonObject,
) raises -> PythonObject:
    var slots = _mamba_state_fields(state, "mamba3", context)
    var names = [String("theta"), String("h"), String("k"), String("v")]
    var targets = [String("theta"), String("h"), String("pending_k"), String("pending_v")]
    var sources = Python.list(theta, h, k, v)
    for i in range(4):  # driver-loop: over the four named continuation pieces
        var source = neural_block_f32(sources[i], names[i], context)
        var target = slots[targets[i]]
        var want = _shape_of(target)
        if not Bool(py=source.shape == want):
            raise Error("Mamba3State.set_input_states: " + names[i] + " has shape " + String(source.shape) + ", want " + String(want))
        var dims = List[Int]()
        for axis in range(len(want)):  # small-loop(state piece shape dimension words)
            dims.append(Int(py=want[axis]))
        var keep = List[PythonObject]()
        var copies = List[PythonObject]()
        var address = _mutable(target, dims, keep, copies, names[i])
        mamba_copy_f32(address, Int(py=source._addr), Int(py=source.size))
        _ = source^  # the copy above read `source` by address
        _writeback(copies)
        _ = keep^
    py_setitem(_fields(state), "pending", True)
    return PythonObject(None)


# ---------------------------------------------------------------- decode sessions


struct MambaDecodeHandle(Movable, Writable):
    """The native owner of one open Mamba decode session.

    Device arm: the compiled `<kind>_session_*` handle keeps the weights and the
    state resident. Host arm: typed copies of the weights and the state, stepped
    through `<kind>_decode_step` in the per-call address order."""
    var kind: Int
    var binding: PythonObject
    var device: PythonObject
    var provider: PythonObject
    var device_address: Int
    var open: Bool
    var b: Int
    var d_model: Int
    var q: Int
    var pending: Bool
    var position: Int
    var profile: Int
    var dt_limit: PythonObject
    var names: List[String]
    var dims: List[List[Int]]
    var report_dims: List[List[Int]]
    var weights: List[List[Float32]]
    var parts: List[List[Float32]]

    def __init__(out self, kind: Int, binding: PythonObject):
        self.kind = kind
        self.binding = binding
        self.device = PythonObject(None)
        self.provider = PythonObject(None)
        self.device_address = 0
        self.open = False
        self.b = 0
        self.d_model = 0
        self.q = 0
        self.pending = False
        self.position = 0
        self.profile = 1
        self.dt_limit = PythonObject(None)
        self.names = List[String]()
        self.dims = List[List[Int]]()
        self.report_dims = List[List[Int]]()
        self.weights = List[List[Float32]]()
        self.parts = List[List[Float32]]()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("MambaDecodeHandle(kind=", self.kind, ", open=", self.open, ", rows=", self.b, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _kind_name(kind: Int) -> String:
    if kind == 1:
        return "mamba1"
    if kind == 2:
        return "mamba2"
    return "mamba3"


def _session_name(kind: Int) -> String:
    return "Mamba" + String(kind) + "DecodeSession"


def _count(dims: List[Int]) -> Int:
    var n = 1
    for i in range(len(dims)):  # small-loop(state piece shape dimension words)
        n *= dims[i]
    return n


def _m1_check(slots: PythonObject, profile: Int, what: String) raises:
    # The retired `_m1_affine_buffers` metadata checks; the arithmetic is Mojo's.
    if Int(py=py_get(slots, "profile_id", PythonObject(1))) != profile:
        raise Error(what + ": checkpoint numerical profile differs from the binding")
    if profile != 2:
        return
    if not Bool(py=py_get(slots, "_nn34_usable", PythonObject(True))):
        raise Error(what + ": NN34 state was lost after a failed call; restore a checkpoint")
    var position = Int(py=slots["absolute_position"])
    if position < 0 or position > (1 << 62) - 32:
        raise Error(what + ": invalid absolute checkpoint position")


def _session_layout(mut handle: MambaDecodeHandle, fields: PythonObject) raises:
    var b = handle.b
    if handle.kind == 1:
        var di = Int(py=fields["d_inner"])
        handle.names = [String("conv_window"), String("h")]
        handle.dims.append([b, di, 4])
        handle.dims.append([b, di, 16])
        if handle.profile == 2:
            handle.names.append("affine_boundary")
            handle.names.append("affine_a")
            handle.names.append("affine_b")
            handle.dims.append([b, di, 16])
            handle.dims.append([b, di, 16, 6])
            handle.dims.append([b, di, 16, 6])
    elif handle.kind == 2:
        var cd = Int(py=fields["conv_dim"])
        var nh = Int(py=fields["nheads"])
        handle.names = [String("conv_window"), String("h"), String("buffer_xbc"), String("buffer_dtraw")]
        handle.dims.append([b, cd, 4])
        handle.dims.append([b, nh, 64, 128])
        handle.dims.append([b, 256, cd])
        handle.dims.append([b, 256, nh])
        handle.report_dims.append([b, nh, 64, 128])
    else:
        var nh = Int(py=fields["nheads"])
        handle.names = _m3_names()
        handle.dims = _m3_shapes(b, nh)
        handle.report_dims.append([b, nh, 64, 128])
        handle.report_dims.append([b, nh, 128])
        handle.report_dims.append([b, nh, 64])
        handle.report_dims.append([b, nh, 32])


def mamba_session_init(session: PythonObject, block: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = neural_block_snapshot(block, context)
    var kind_name = String(py_get(fields, "_native_kind", PythonObject("")))
    var kind = 1 if kind_name == "mamba1" else (2 if kind_name == "mamba2" else (3 if kind_name == "mamba3" else 0))
    if kind == 0:
        raise Error("TypeError: a Mamba decode session takes a Mamba1Block, Mamba2Block or Mamba3Block")
    var what = _session_name(kind)
    _ = _mamba_record_fields(session, context["mamba_session_classes"][kind_name],
        [String("_handle"), String("_block"), String("_state")])
    var binding = _model_bindings(fields["_native_route"], kind_name, context, py_get(fields, "numeric_mode"))[kind_name]
    var device_arm = Bool(py=py_call(bi.hasattr, binding, kind_name + "_session_create"))
    if not device_arm and not Bool(py=py_call(bi.hasattr, binding, kind_name + "_decode_step")):
        raise Error("NotImplementedError: " + what + ": the loaded binding exports neither a resident decode session nor the per-call " + kind_name + "_decode_step entry")
    if state is None:
        raise Error(what + ": state is required (allocate_state)")
    var slots = _mamba_state_fields(state, kind_name, context)
    if py_get(slots, "_resident_session") is not None:
        raise Error(what + ": the state is owned by an open resident decode session (its pieces live in the session, "
                    + "on the device or on the host, and the caller's buffers are stale); call sync_state() or close() on the session first")
    var hshape = _shape_of(slots["h"])
    if len(hshape) != (3 if kind == 1 else 4) or Int(py=hshape[0]) < 1:
        raise Error(what + ": state.h has an invalid shape " + String(hshape))
    var handle = MambaDecodeHandle(kind, binding)
    handle.b = Int(py=hshape[0])
    handle.d_model = Int(py=fields["d_model"])
    if kind == 1:
        if Bool(py=py_call(bi.hasattr, binding, "mamba1_profile")):
            handle.profile = Int(py=_native_function(binding, "mamba1_profile")())
        _m1_check(slots, handle.profile, what)
        handle.position = Int(py=py_get(slots, "absolute_position", PythonObject(0)))
    elif kind == 2:
        handle.q = Int(py=slots["buffered_tokens"])
        handle.dt_limit = fields["dt_limit"]
    else:
        handle.q = Int(py=slots["buffered_tokens"])
        handle.pending = Bool(py=slots["pending"])
    _session_layout(handle, fields)
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var pieces = List[Int]()
    for i in range(len(handle.names)):  # driver-loop: over the state's named pieces
        pieces.append(_mutable(slots[handle.names[i]], handle.dims[i], keep, copies, handle.names[i]))
    var weights = fields["_w"]
    if device_arm:
        # The weights, then the state pieces, as the retired session opened them.
        var addresses = py_ints([])
        for i in range(len(weights)):  # driver-loop: over block weight slots
            _ = py_call(addresses.append, generation_address(weights[i], keep))
        for i in range(len(pieces)):  # driver-loop: over the state's named pieces
            _ = py_call(addresses.append, pieces[i])
        var params = py_ints([handle.b, handle.d_model])
        if kind == 1:
            if handle.profile == 2:
                _ = py_call(params.append, handle.position)
        elif kind == 2:
            _ = py_call(params.append, handle.q)
            _ = py_call(params.extend, handle.dt_limit)
        else:
            _ = py_call(params.append, handle.q)
            _ = py_call(params.append, 1 if handle.pending else 0)
        handle.device = _native_function(binding, kind_name + "_session_create")()
        if getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1" and String(fields["numeric_mode"]) == "identical":
            handle.provider = neural_descriptor_reuse(py_get(fields, "_native_mamba_block_provider_cache"),
                _native_function(binding, "mamba_block_api_v1"), String(MAMBA_BLOCK_ABI_NAME), 44)
            handle.device_address = neural_descriptor_handle(handle.provider,
                _native_function(binding, kind_name + "_decode_native_handle_v1"), handle.device)
            var p: List[Int] = [handle.b, handle.d_model]
            var f = List[Float64]()
            if kind == 1 and handle.profile != 1:
                raise Error("Mamba1 resident decode requires the binding's supported profile")
            if kind == 2:
                p.append(handle.q)
                f.append(Float64(py=handle.dt_limit[0]))
                f.append(Float64(py=handle.dt_limit[1]))
            elif kind == 3:
                p.append(handle.q)
                p.append(Int(handle.pending))
            var base = 20 + 8 * (kind - 1)
            _ = neural_descriptor_reserved_execute(handle.provider, handle.device, handle.device_address,
                base, base + 2, base + 3, _neural_int_words(addresses), p, f)
        else:
            _ = py_call(_native_function(binding, kind_name + "_session_open"), handle.device, addresses, params)
    else:
        if getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
            handle.provider = neural_descriptor_reuse(py_get(fields, "_native_mamba_state"),
                _native_function(binding, "mamba_state_api_v1"), String(MAMBA_STATE_ABI_NAME))
        # The host arm owns typed copies, the device arm's upload on the host.
        for i in range(len(weights)):  # driver-loop: over block weight slots
            handle.weights.append(mamba_host_copy(generation_address(weights[i], keep), Int(py=weights[i].size)))
        for i in range(len(pieces)):  # driver-loop: over the state's named pieces
            handle.parts.append(mamba_host_copy(pieces[i], _count(handle.dims[i])))
    handle.open = True
    _ = keep^
    var session_fields = _fields(session)
    py_setitem(session_fields, "_handle", PythonObject(alloc=handle^))
    py_setitem(session_fields, "_block", block)
    py_setitem(session_fields, "_state", state)
    py_setitem(_fields(state), "_resident_session", PythonObject(True))
    return PythonObject(None)


def mamba_session_open(block: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var kind = String(py_get(_fields(block), "_native_kind", PythonObject("")))
    var classes = context["mamba_session_classes"]
    if not py_contains(classes, kind):
        raise Error("TypeError: a Mamba decode session takes a Mamba1Block, Mamba2Block or Mamba3Block")
    var session = py_call(bi.object.__new__, classes[kind])
    _ = mamba_session_init(session, block, state, context)
    return session


def _session_require_open(handle: PythonObject, what: String) raises:
    if not handle.downcast_value_ptr[MambaDecodeHandle]()[].open:
        raise Error(what + ": the session is closed")


def mamba_session_step(session: PythonObject, input: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var hobj = session._handle
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    var kind = h[].kind
    var name = _kind_name(kind)
    var what = _session_name(kind) + ".step"
    _session_require_open(hobj, what)
    var x = _neural_input(input, h[].d_model, True, context)
    var b = h[].b
    if Int(py=x.shape[0]) != b:
        raise Error(what + ": the session holds " + String(b) + " rows, x has B = " + String(x.shape[0]))
    var keep = List[PythonObject]()
    var y = generation_array([b, 1, h[].d_model])
    var reports = bi.list()
    for i in range(len(h[].report_dims)):  # driver-loop: over the fixed report buffers
        _ = py_call(reports.append, generation_array(h[].report_dims[i]))
    if h[].device is not None:
        var addresses = py_ints([generation_address(x, keep), generation_address(y, keep)])
        for i in range(len(reports)):  # driver-loop: over the fixed report buffers
            _ = py_call(addresses.append, generation_address(reports[i], keep))
        if h[].provider is not None:
            var base = 20 + 8 * (kind - 1)
            var updated = neural_descriptor_reserved_execute(h[].provider, h[].device, h[].device_address,
                base + 1, base + 2, base + 4, _neural_int_words(addresses), [], [])
            if kind != 1:
                h[].q = updated
        else:
            var updated = py_call(_native_function(h[].binding, name + "_session_step"), h[].device, addresses)
            if kind != 1:
                h[].q = Int(py=updated)
    else:
        # The per-call `<kind>_decode_step` address order with the session's
        # own copies standing where the caller's buffers would: x, weights,
        # state pieces, y, then (Mamba-1) the affine pieces or (Mamba-2/3) the reports.
        var addresses = py_ints([generation_address(x, keep)])
        for i in range(len(h[].weights)):  # driver-loop: over block weight slots
            _ = py_call(addresses.append, Int(h[].weights[i].unsafe_ptr()))
        var lead = 2 if kind == 1 else len(h[].parts)
        for i in range(lead):  # driver-loop: over the state's named pieces
            _ = py_call(addresses.append, Int(h[].parts[i].unsafe_ptr()))
        _ = py_call(addresses.append, generation_address(y, keep))
        for i in range(lead, len(h[].parts)):  # driver-loop: over the Mamba-1 affine pieces
            _ = py_call(addresses.append, Int(h[].parts[i].unsafe_ptr()))
        for i in range(len(reports)):  # driver-loop: over the fixed report buffers
            _ = py_call(addresses.append, generation_address(reports[i], keep))
        var params = py_ints([b, h[].d_model])
        if kind == 1:
            if h[].profile == 2:
                _ = py_call(params.append, h[].position)
        elif kind == 2:
            _ = py_call(params.append, h[].q)
            _ = py_call(params.extend, h[].dt_limit)
        else:
            _ = py_call(params.append, h[].q)
            _ = py_call(params.append, 1 if h[].pending else 0)
        if h[].provider is not None:
            var p: List[Int] = [b, h[].d_model]
            var f = List[Float64]()
            if kind != 1:
                p.append(h[].q)
            if kind == 2:
                f.append(Float64(py=h[].dt_limit[0])); f.append(Float64(py=h[].dt_limit[1]))
            elif kind == 3:
                p.append(Int(h[].pending))
            var updated = neural_descriptor_execute(h[].provider, 2 * kind - 1, _neural_int_words(addresses), p, f)
            if kind != 1:
                h[].q = updated
        else:
            var updated = py_call(_native_function(h[].binding, name + "_decode_step"), addresses, params)
            if kind != 1:
                h[].q = Int(py=updated)
        if kind == 3:
            h[].pending = False
    if kind == 1 and h[].profile == 2:
        h[].position += 1
    _ = keep^
    if kind == 1:
        return y
    if kind == 2:
        return Python.tuple(y, reports[0])
    return Python.tuple(y, reports[0], reports[1], reports[2], reports[3])


def _session_pieces(
    hobj: PythonObject, slots: PythonObject, what: String, mut keep: List[PythonObject], mut copies: List[PythonObject],
) raises -> List[Int]:
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    if h[].kind == 1:
        _m1_check(slots, h[].profile, what)
    var pieces = List[Int]()
    for i in range(len(h[].names)):  # driver-loop: over the state's named pieces
        pieces.append(_mutable(slots[h[].names[i]], h[].dims[i], keep, copies, h[].names[i]))
    return pieces^


def mamba_session_sync(session: PythonObject, context: PythonObject) raises -> PythonObject:
    var hobj = session._handle
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    var kind = h[].kind
    var what = _session_name(kind) + ".sync_state"
    _session_require_open(hobj, what)
    var state = session._state
    var slots = _mamba_state_fields(state, _kind_name(kind), context)
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var pieces = _session_pieces(hobj, slots, what, keep, copies)
    if h[].device is not None:
        var addresses = py_ints(pieces)
        if h[].provider is not None:
            var base = 20 + 8 * (kind - 1)
            var result = neural_descriptor_reserved_execute(h[].provider, h[].device, h[].device_address,
                base + 1, base + 2, base + 5, pieces, [], [])
            if kind == 2:
                h[].q = result
            elif kind == 3:
                h[].q = result // 2
                h[].pending = (result & 1) != 0
        else:
            var exported = py_call(_native_function(h[].binding, _kind_name(kind) + "_session_export_state"), h[].device, addresses)
            if kind == 1:
                if h[].profile == 2:
                    h[].position = Int(py=exported)
            elif kind == 2:
                h[].q = Int(py=exported)
            else:
                h[].q = Int(py=exported[0])
                h[].pending = Bool(py=exported[1])
    else:
        for i in range(len(pieces)):  # driver-loop: over the state's named pieces
            mamba_copy_f32(pieces[i], Int(h[].parts[i].unsafe_ptr()), _count(h[].dims[i]))
    _writeback(copies)
    _ = keep^
    if kind == 1:
        if h[].profile == 2:
            py_setitem(_fields(state), "absolute_position", h[].position)
    else:
        py_setitem(_fields(state), "buffered_tokens", h[].q)
        if kind == 3:
            py_setitem(_fields(state), "pending", PythonObject(h[].pending))
    return state


def mamba_session_load(session: PythonObject, context: PythonObject) raises -> PythonObject:
    var hobj = session._handle
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    var kind = h[].kind
    var what = _session_name(kind) + ".load_state"
    _session_require_open(hobj, what)
    var state = session._state
    var slots = _mamba_state_fields(state, _kind_name(kind), context)
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var pieces = _session_pieces(hobj, slots, what, keep, copies)
    var q = 0
    var pending = False
    var position = h[].position
    if kind == 1:
        if h[].profile == 2:
            position = Int(py=slots["absolute_position"])
    else:
        q = Int(py=slots["buffered_tokens"])
        if kind == 3:
            pending = Bool(py=slots["pending"])
    if h[].device is not None:
        var addresses = py_ints(pieces)
        var name = _kind_name(kind) + "_session_load_state"
        if h[].provider is not None:
            var p = List[Int]()
            if kind != 1:
                p.append(q)
            if kind == 3:
                p.append(Int(pending))
            var base = 20 + 8 * (kind - 1)
            _ = neural_descriptor_reserved_execute(h[].provider, h[].device, h[].device_address,
                base + 1, base + 2, base + 6, pieces, p, [])
        else:
            if kind == 1:
                if h[].profile == 2:
                    _ = py_call(addresses.append, position)
                _ = py_call(_native_function(h[].binding, name), h[].device, addresses)
            elif kind == 2:
                _ = py_call(_native_function(h[].binding, name), h[].device, addresses, py_ints([q]))
            else:
                _ = py_call(_native_function(h[].binding, name), h[].device, addresses, py_ints([q, 1 if pending else 0]))
    else:
        for i in range(len(pieces)):  # driver-loop: over the state's named pieces
            mamba_copy_f32(Int(h[].parts[i].unsafe_ptr()), pieces[i], _count(h[].dims[i]))
    _ = keep^
    h[].q = q
    h[].pending = pending
    h[].position = position
    return state


def _session_release(hobj: PythonObject, state: PythonObject) raises:
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    if not h[].open:
        return
    h[].open = False
    if state is not None:
        py_setitem(_fields(state), "_resident_session", PythonObject(None))
    if h[].device is not None:
        if h[].provider is not None:
            _ = neural_descriptor_execute(h[].provider, 27 + 8 * (h[].kind - 1), [], [h[].device_address], [], False)
        else:
            _ = py_call(_native_function(h[].binding, _kind_name(h[].kind) + "_session_close"), h[].device)
    h[].weights.clear()
    h[].parts.clear()


def mamba_session_close(session: PythonObject, context: PythonObject) raises -> PythonObject:
    var hobj = session._handle
    if not hobj.downcast_value_ptr[MambaDecodeHandle]()[].open:
        return PythonObject(None)
    try:
        _ = mamba_session_sync(session, context)
    finally:
        _session_release(hobj, session._state)
    return PythonObject(None)


def mamba_session_drop(session: PythonObject) raises -> PythonObject:
    # Finalizer: release an unclosed session without copying its state back.
    var values = _fields(session)
    var hobj = py_get(values, "_handle")
    if hobj is None:
        return PythonObject(None)
    _session_release(hobj, py_get(values, "_state"))
    return PythonObject(None)


def mamba_session_is_open(session: PythonObject) raises -> PythonObject:
    return PythonObject(session._handle.downcast_value_ptr[MambaDecodeHandle]()[].open)


def mamba_session_device(session: PythonObject) raises -> PythonObject:
    return session._handle.downcast_value_ptr[MambaDecodeHandle]()[].device


def mamba_session_repr(session: PythonObject) raises -> PythonObject:
    var hobj = session._handle
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    var flag = String("True") if h[].open else String("False")
    return PythonObject(_session_name(h[].kind) + "(open=" + flag + ", rows=" + String(h[].b) + ")")


# ---------------------------------------------------------------- the C boundary


def mamba_native_call(operation: String, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if py_type(values) is not bi.tuple:
        raise Error("TypeError: native Mamba operations require their ordinary argument tuple")
    _ = generation_initialize_binding(context["generation_classes"])
    if operation == "state_init":
        if len(values) != 3 or py_type(values[1]) is not bi.str:
            raise Error("TypeError: native Mamba state requires its kind and arguments")
        return mamba_state_init(values[0], String(values[1]), values[2], context)
    if operation == "session_drop":
        # Finalization has the old best-effort behavior, wholly within native
        # execution. A replaced owner layout is refused without invoking it.
        try:
            _ = _mamba_session_fields(values[0], context, False)
            return mamba_session_drop(values[0])
        except:
            return PythonObject(None)
    if operation == "session_init":
        _ = _mamba_session_fields(values[0], context, False)
        var fields = neural_block_fields(values[1], context)
    elif operation == "set_input_states":
        _ = _mamba_state_fields(values[0], "mamba3", context, False)
    elif operation.startswith("session_") and operation != "session_open" and operation != "session_info":
        var fields = _mamba_session_fields(values[0], context)
        if operation == "session_state":
            return fields["_state"]
        if operation == "session_enter":
            return values[0]
        if operation == "session_exit":
            _ = mamba_session_close(values[0], context)
            return PythonObject(False)
        if operation == "session_step_public":
            var result = mamba_session_step(values[0], values[1], context)
            var hobj = fields["_handle"]
            var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
            if h[].kind == 1:
                return result
            var block_fields = _fields(fields["_block"])
            py_setitem(block_fields, "h_last_", result[1])
            if h[].kind == 3:
                py_setitem(block_fields, "k_last_", result[2])
                py_setitem(block_fields, "v_last_", result[3])
                py_setitem(block_fields, "theta_last_", result[4])
            return result[0]
    elif operation != "setstate":
        var fields = neural_block_fields(values[0], context)
        var kind = String(fields["_native_kind"])
        if operation == "allocate":
            _mamba_scalar(values[1], "batch_size")
    if operation == "forward_public" or operation == "step_public":
        var step = operation == "step_public"
        var lengths = PythonObject(None)
        if not step:
            lengths = values[3]
        var result = mamba3_forward(values[0], values[1], values[2], lengths, step, context)
        # Retain report objects through dictionary insertion before releasing
        # the tuple. No Python assignment or user __setattr__ runs on return.
        var fields = _fields(values[0])
        py_setitem(fields, "h_last_", result[1])
        py_setitem(fields, "k_last_", result[2])
        py_setitem(fields, "v_last_", result[3])
        py_setitem(fields, "theta_last_", result[4])
        return result[0]
    if operation == "forward":
        return mamba3_forward(values[0], values[1], values[2], values[3], False, context)
    if operation == "step":
        return mamba3_forward(values[0], values[1], values[2], PythonObject(None), True, context)
    if operation == "allocate":
        return mamba3_allocate(values[0], values[1], context)
    if operation == "install_owned":
        return mamba3_install_owned(values[0], context)
    if operation == "release_owned":
        return mamba3_release_owned(values[0])
    if operation == "session_info":
        return mamba3_session_info(values[0])
    if operation == "export_owned":
        return mamba3_export_owned(values[0])
    if operation == "getstate":
        return mamba3_getstate(values[0])
    if operation == "setstate":
        return mamba3_setstate(values[0], values[1], context)
    if operation == "set_input_states":
        return mamba3_set_input_states(values[0], values[1], values[2], values[3], values[4], context)
    if operation == "session_open":
        return mamba_session_open(values[0], values[1], context)
    if operation == "session_init":
        return mamba_session_init(values[0], values[1], values[2], context)
    if operation == "session_step":
        return mamba_session_step(values[0], values[1], context)
    if operation == "session_sync":
        return mamba_session_sync(values[0], context)
    if operation == "session_load":
        return mamba_session_load(values[0], context)
    if operation == "session_close":
        return mamba_session_close(values[0], context)
    if operation == "session_drop":
        return mamba_session_drop(values[0])
    if operation == "session_is_open":
        return mamba_session_is_open(values[0])
    if operation == "session_device":
        return mamba_session_device(values[0])
    if operation == "session_repr":
        return mamba_session_repr(values[0])
    raise Error("mamba native: unknown operation " + operation)


def mamba_native_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var bi = native_loaded_module("builtins")
        if py_type(args) is not bi.tuple or len(args) != 3 or py_type(args[0]) is not bi.str:
            raise Error("TypeError: native Mamba call requires an operation name, argument tuple and context")
        var result = mamba_native_call(String(args[0]), args[1], args[2])
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
