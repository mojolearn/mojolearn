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
from bindings.causal_lm_public import _native_function, _native_entry_fields, _native_has_entry, _mutable, _writeback
from bindings.causal_lm_construct import _model_bindings, _model_block
from bindings.neural_block_public import neural_training_entries, _neural_input, _neural_lengths, _neural_int_words, neural_block_snapshot_prepared
from bindings.neural_block_admission import neural_block_fields, neural_block_snapshot, neural_block_f32, neural_array_snapshot, neural_native_scalar
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.native_loaded_module import native_loaded_module, NativeModules
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
from mamba.impl.mamba_decode_plan import MambaDecodePlan


# ---------------------------------------------------------------- Mamba3Block


def _mamba_record_fields(mut mods: NativeModules, value: PythonObject, expected: PythonObject, names: List[String]) raises -> PythonObject:
    # Validate the exact native record and every attribute route BEFORE the
    # dictionary is read. object.__getattribute__("__dict__") alone would still
    # execute a replaced data descriptor; _fields uses GenericGetDict after
    # checking that descriptor. Raw namespaces below never bind properties.
    var bi = mods.module("builtins")
    if py_type(value) is not expected or py_type(expected) is not bi.type:
        raise Error("TypeError: Mamba state/session requires its exact connected native class")
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, expected)
    for i in range(len(ancestry)):  # driver-loop: record layouts before any attribute read/write
        var namespace = _class_fields(mods, ancestry[i])
        if py_contains(namespace, "__getattr__"):
            raise Error("TypeError: Mamba native records cannot invoke Python attribute hooks")
        for hook in [String("__getattribute__"), String("__setattr__")]:  # small-loop(two native record access hooks)
            if py_contains(namespace, hook) and namespace[hook] is not bi.object.__dict__[hook]:
                raise Error("TypeError: Mamba native records cannot invoke Python attribute hooks")
        for name in names:  # driver-loop: fixed state/session fields cannot be replaced by descriptors
            if py_contains(namespace, name):
                raise Error("TypeError: Mamba native record has a replaced field descriptor: " + name)
    return _fields(mods, value)


def _mamba_scalar(mut mods: NativeModules, value: PythonObject, name: String, context: PythonObject) raises:
    var bi = mods.module("builtins")
    var cls = py_type(value)
    if value is None or cls is bi.str or cls is bi.bytes or neural_native_scalar(mods, value, context):
        return
    raise Error("TypeError: Mamba " + name + " requires a native scalar without conversion hooks")


def _mamba_buffer(mut mods: NativeModules, value: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(value) is context["array_class"]:
        return neural_array_snapshot(value, context)
    admit_native_buffer_exporter(value)
    # memoryview captures shape/stride metadata once and pins the exact native
    # exporter. Later copies inspect this retained view, never a caller hook.
    return py_call(mods.module("builtins").memoryview, value)


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


def _mamba_state_fields(mut mods: NativeModules, state: PythonObject, kind: String, context: PythonObject,
                        initialized: Bool = True) raises -> PythonObject:
    var expected = context["mamba3_state_class"] if kind == "mamba3" else context["generation_classes"][2][kind]
    var names = _mamba_state_names(kind)
    var fields = _mamba_record_fields(mods, state, expected, names)
    if py_contains(_class_fields(expected), "__del__"):
        raise Error("TypeError: Mamba state snapshots cannot invoke Python finalizers")
    if initialized:
        fields = py_call(mods.module("builtins").dict, fields)
        for name in names:  # driver-loop: capture fixed mutable state metadata before execution
            if not py_contains(fields, name):
                continue
            var item = fields[name]
            if name == "profile_id" or name == "absolute_position" or name == "_nn34_usable" or name == "buffered_tokens" or name == "pending" or name == "_resident_session":
                _mamba_scalar(mods, item, name, context)
            elif item is not None:
                py_setitem(fields, name, _mamba_buffer(mods, item, context))
    return fields


def mamba_state_init(mut mods: NativeModules, state: PythonObject, kind: String, values: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    if kind != "mamba1" and kind != "mamba2" and kind != "mamba3":
        raise Error("TypeError: unknown native Mamba state kind")
    var names = _mamba_state_names(kind)
    var count = 7 if kind == "mamba1" else (5 if kind == "mamba2" else 12)
    if py_type(values) is not bi.tuple or len(values) != count:
        raise Error("TypeError: Mamba state construction requires its native argument tuple")
    var fields = _mamba_state_fields(mods, state, kind, context, False)
    var converted = Python.list()
    for i in range(count):  # driver-loop: constructor reference storage and native scalar conversion
        var value = values[i]
        if (kind == "mamba1" and (i == 2 or i == 3)) or (kind == "mamba2" and i == 4) or (kind == "mamba3" and i == 10):
            _mamba_scalar(mods, value, names[i], context)
            value = py_call(bi.int, value)
        elif kind == "mamba3" and i == 11:
            _mamba_scalar(mods, value, names[i], context)
            value = PythonObject(Bool(py=value))
        _ = py_call(converted.append, value)
    for i in range(count):  # driver-loop: publish only after all scalar arguments were admitted
        py_setitem(fields, names[i], converted[i])
    if kind == "mamba1":
        py_setitem(fields, "_nn34_usable", True)
    return PythonObject(None)


def _mamba_session_fields(mut mods: NativeModules, session: PythonObject, context: PythonObject,
                          initialized: Bool = True) raises -> PythonObject:
    var classes = context["mamba_session_classes"]
    var cls = py_type(session)
    var expected = PythonObject(None)
    for name in [String("mamba1"), String("mamba2"), String("mamba3")]:  # small-loop(three connected Mamba session classes)
        if cls is classes[name]:
            expected = classes[name]
    var fields = _mamba_record_fields(mods, session, expected, [String("_handle"), String("_block"), String("_state")])
    if initialized:
        var hobj = fields["_handle"]
        var handle = hobj.downcast_value_ptr[MambaDecodeHandle]()
        if cls is not classes[_kind_name(handle[].plan.kind)]:
            raise Error("TypeError: Mamba session kind differs from its native handle")
        _ = neural_block_fields(mods, fields["_block"], context)
        _ = _mamba_state_fields(mods, fields["_state"], _kind_name(handle[].plan.kind), context, False)
    return fields


def _m3_entries(mut mods: NativeModules, block: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # The binding the retired `Mamba3Block._extension()` returned: the shipped
    # host binding for the CPU inference class, the block's tier otherwise.
    var bi = mods.module("builtins")
    # isinstance may consult a replaced block.__class__ descriptor even when
    # field storage was admitted. Test the actual native type ancestry.
    ref cpy = Python().cpython()
    var cls = py_type(block)
    if cpy.lib.get_function[Int32]("PyType_IsSubtype")(
        cls._obj_ptr, context["block_classes"]["cpu"]["mamba3"]._obj_ptr) != 0:
        var entries = _model_bindings(mods, PythonObject("cpu"), "mamba3", context, PythonObject("identical"))["mamba3"]
        var block_provider = PythonObject(None)
        var state_provider = PythonObject(None)
        if getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
            block_provider = neural_descriptor_reuse(py_get(fields, "_native_mamba_block_provider_cache"),
                _native_function(mods, entries, "mamba_block_api_v1"), String(MAMBA_BLOCK_ABI_NAME), 44)
            mods.released()
            state_provider = neural_descriptor_reuse(py_get(fields, "_native_mamba_state"),
                _native_function(mods, entries, "mamba_state_api_v1"), String(MAMBA_STATE_ABI_NAME), 6)
            mods.released()
        py_setitem(fields, "_native_mamba_block_provider_cache", block_provider)
        py_setitem(fields, "_native_mamba_state", state_provider)
        return entries
    return neural_training_entries(mods, fields, context)


def _m3_canonical(mut mods: NativeModules, entries: PythonObject) raises -> PythonObject:
    var fields = _native_entry_fields(mods, entries)
    if py_contains(fields, "canonical"):
        return fields["canonical"]
    return entries


def _mamba_saved_entries(mut mods: NativeModules, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    # _prefill_binding is a mutable model field. Resolve it by identity in the
    # import-connected table BEFORE touching its attributes: a saved binding
    # may be the CPU _HostBinding proxy, whose missing-attribute hook is Python.
    # The exact SimpleNamespace records contain already resolved C functions.
    var bi = mods.module("builtins")
    var table = context["training_bindings"]
    if py_type(table) is not bi.dict:
        raise Error("TypeError: Mamba bindings require their native connection table")
    var keys = py_call(bi.list, table.keys())
    for i in range(len(keys)):  # driver-loop: connection keys before native keyed lookup
        if py_type(keys[i]) is not bi.str:
            raise Error("TypeError: Mamba binding connections require string route keys")
    for i in range(len(keys)):  # driver-loop: locate the retained compiled image by identity
        var families = table[keys[i]]
        if py_type(families) is not bi.dict:
            raise Error("TypeError: Mamba bindings require native family dictionaries")
        var family_names = py_call(bi.list, families.keys())
        for j in range(len(family_names)):  # small-loop(connection family names before lookup)
            if py_type(family_names[j]) is not bi.str:
                raise Error("TypeError: Mamba binding families require string keys")
        var entries = py_get(families, "mamba")
        if entries is None or py_type(entries) is bi.str:
            continue
        if py_type(entries) is not mods.module("types").SimpleNamespace:
            raise Error("TypeError: Mamba requires its connected native entry namespace")
        ref cpy = Python().cpython()
        var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetDict")(entries._obj_ptr, PyObjectPtr())
        if not pointer:
            raise cpy.unsafe_get_error()
        var fields = PythonObject(from_owned=pointer)
        var names = py_call(bi.list, fields.keys())
        for j in range(len(names)):  # driver-loop: exact names before saved-image identity lookup
            if py_type(names[j]) is not bi.str:
                raise Error("TypeError: Mamba native entry namespaces require string keys")
        if binding is py_get(fields, "canonical"):
            return fields
    raise Error("TypeError: Mamba saved binding is not an import-connected native image")


def _mamba_saved_entry(mut mods: NativeModules, entries: PythonObject, name: String, required: Bool = True) raises -> PythonObject:
    var entry = py_get(entries, name)
    if entry is None and not required:
        return entry
    var bi = mods.module("builtins")
    var marker = py_get(entries, "mamba_vendor")
    if py_type(entry) is not py_type(bi.len) or py_type(marker) is not py_type(bi.len):
        raise Error("TypeError: Mamba saved binding requires its compiled " + name + " entry")
    if entry.__self__ is not marker.__self__:
        raise Error("TypeError: Mamba saved entry belongs to a different native image")
    return entry


def _m3_allocate(mut mods: NativeModules, block: PythonObject, fields: PythonObject, batch: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var b = Int(py=py_call(bi.int, batch))
    if b < 1:
        raise Error("Mamba3Block: batch_size must be positive, got " + String(b))
    var state = py_call(bi.object.__new__, context["mamba3_state_class"])
    var slots = _fields(state)
    var shapes = _m3_shapes(b, Int(py=fields["nheads"]))
    mods.released()
    var names = _m3_names()
    mods.released()
    for j in range(len(names)):  # driver-loop: over the fixed Mamba3 continuation buffer ABI
        py_setitem(slots, names[j], generation_array(shapes[j]))
    py_setitem(slots, "buffered_tokens", 0)
    py_setitem(slots, "pending", False)
    return state


def _m3_fresh(mut mods: NativeModules, block: PythonObject, fields: PythonObject, x: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    # The retired `_call`: the discard-only prefill where the binding has it,
    # else the carried-state entry from a fresh zero state.
    if _native_has_entry(mods, entries, "mamba3_forward_fresh"):
        var windowed_result8 = _prefill_mamba3_reports(block, x, entries, fields)
        mods.released()
        return windowed_result8^
    var state = _m3_allocate(mods, block, fields, x.shape[0], context)
    var windowed_result7 = _m3_state_call(block, x, state, False, entries, fields, _fields(state))
    mods.released()
    return windowed_result7^


def _mamba3_forward_captured(mut mods: NativeModules,
    block: PythonObject, input: PythonObject, state: PythonObject, lengths: PythonObject, step: Bool,
    context: PythonObject, fields: PythonObject,
) raises -> PythonObject:
    """`(y, h_last, k_last, v_last, theta_last)` using one retained snapshot."""
    var d = Int(py=fields["d_model"])
    if step and state is None:
        raise Error("Mamba3Block.step: state is required (a decode step continues a sequence; allocate_state(B) makes the fresh one)")
    var x = _neural_input(mods, input, d, step, context)
    var entries = _m3_entries(mods, block, fields, context)
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    if lengths is not None:
        if state is not None:
            raise Error("Mamba3Block.forward: lengths= (a ragged, right-padded batch) cannot carry a state; run the ragged prefill with state=None, or continue each sequence as its own batch")
        var sizes = _neural_lengths(mods, lengths, b, l, context)
        var padded = generation_array([b, l, d])
        neural_copy_row_prefixes(Int(py=padded._addr), Int(py=x._addr), sizes, b, l, d)
        _ = x^  # the copy above read `x` by address
        var ran = _m3_fresh(mods, block, fields, padded, entries, context)
        var result = generation_array([b, l, d])
        neural_copy_row_prefixes(Int(py=result._addr), Int(py=ran[0]._addr), sizes, b, l, d)
        return Python.tuple(result, ran[1], ran[2], ran[3], ran[4])
    if state is None:
        return _m3_fresh(mods, block, fields, x, entries, context)
    var captured = _mamba_state_fields(mods, state, "mamba3", context)
    var result = _m3_state_call(block, x, state, step, entries, fields, captured)
    mods.released()
    var slots = _fields(state)
    py_setitem(slots, "buffered_tokens", captured["buffered_tokens"])
    py_setitem(slots, "pending", captured["pending"])
    return result


def _mamba3_publish_runtime(fields: PythonObject, block: PythonObject) raises:
    var live = _fields(block)
    var names = [String("_native_mamba_block_provider_cache"), String("_native_mamba_state"),
                 String("_prefill_session"), String("_prefill_binding"), String("_owned_generation"),
                 String("_owned_native_provider"), String("_owned_restore_pending")]
    for name in names:  # driver-loop: over native session/cache state published after one operation
        if py_contains(fields, name):
            py_setitem(live, name, fields[name])


def mamba3_forward(mut mods: NativeModules,
    block: PythonObject, input: PythonObject, state: PythonObject, lengths: PythonObject, step: Bool,
    context: PythonObject,
) raises -> PythonObject:
    var fields = neural_block_snapshot_prepared(mods, block, context)
    try:
        return _mamba3_forward_captured(mods, block, input, state, lengths, step, context, fields)
    finally:
        mods.released()
        # Session creation/restore and native invalidation survive a failure,
        # while numerical configuration and captured weights stay local.
        _mamba3_publish_runtime(fields, block)


def mamba3_allocate(mut mods: NativeModules, block: PythonObject, batch: PythonObject, context: PythonObject) raises -> PythonObject:
    return _m3_allocate(mods, block, _fields(block), batch, context)


def _mamba3_weight_names() -> List[String]:
    # Fixed numerical ABI, not a mutable class attribute lookup.
    return [String("block_norm.weight"), String("in_proj.weight"), String("dt_bias"),
            String("B_norm.weight"), String("C_norm.weight"), String("B_bias"),
            String("C_bias"), String("D"), String("out_proj.weight")]


def _m3_checked_weights(mut mods: NativeModules, block: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # The current weights through the constructor's own native checks, as the
    # retired `type(self)(...)` re-validation did, without a Python constructor.
    var bi = mods.module("builtins")
    var plan = py_call(bi.object.__new__, context["config_context"]["plan_class"])
    var plan_fields = _mamba_record_fields(mods, plan, context["config_context"]["plan_class"],
        [String("kind"), String("d_model"), String("block_kwargs")])
    py_setitem(plan_fields, "kind", PythonObject("mamba3"))
    py_setitem(plan_fields, "d_model", fields["d_model"])
    py_setitem(plan_fields, "block_kwargs", bi.dict())
    var named = bi.dict()
    var names = _mamba3_weight_names()
    var weights = fields["_w"]
    for i in range(len(names)):  # driver-loop: over the fixed block weight registry
        py_setitem(named, names[i], weights[i])
    var route = fields["_native_route"]
    var checked = _model_block(mods, named, plan, PythonObject("fp32_v1"), route,
                               _model_bindings(mods, route, "mamba3", context, py_get(fields, "numeric_mode")), context)
    return neural_block_fields(mods, checked, context)["_w"]


def mamba3_install_owned(mut mods: NativeModules, block: PythonObject, context: PythonObject, fields: PythonObject) raises -> PythonObject:
    # Publish session, binding and generation together after successful upload.
    var bi = mods.module("builtins")
    var entries = _m3_entries(mods, block, fields, context)
    if not _native_has_entry(mods, entries, "mamba3_owned_weights_enabled") or not Bool(py=_native_function(mods, entries, "mamba3_owned_weights_enabled")()):
        mods.released()
        raise Error("RuntimeError: Mamba3 owned weights require the NN40 IDENTICAL arm")
    mods.released()
    var weights = _m3_checked_weights(mods, block, fields, context)
    var canonical = _m3_canonical(mods, entries)
    var session = py_get(fields, "_prefill_session")
    if session is None or py_get(fields, "_prefill_binding") is not canonical:
        session = _native_function(mods, entries, "mamba3_prefill_session_create")()
        mods.released()
    var keep = List[PythonObject]()
    var addresses = py_ints([])
    for i in range(len(weights)):  # driver-loop: over the fixed owned-weight registry
        _ = py_call(addresses.append, generation_address(weights[i], keep))
    var generation = PythonObject(None)
    var native = py_get(fields, "_native_mamba_block_provider_cache")
    if neural_descriptor_matches(native, canonical):
        var handle = neural_descriptor_handle(native,
            _native_function(mods, entries, "mamba_prefill_native_handle_v1"), session)
        mods.released()
        generation = PythonObject(neural_descriptor_reserved_execute(native, session, handle, 0, 1, 15,
            _neural_int_words(addresses), [Int(py=fields["d_model"])], []))
        mods.released()
    else:
        native = PythonObject(None)
        generation = py_call(_native_function(mods, entries, "mamba3_prefill_session_install_weights"), session, addresses, fields["d_model"])
        mods.released()
    py_setitem(fields, "_prefill_session", session)
    py_setitem(fields, "_prefill_binding", canonical)
    py_setitem(fields, "_owned_generation", generation)
    py_setitem(fields, "_owned_native_provider", native)
    py_setitem(fields, "_owned_restore_pending", False)
    _ = keep^
    return generation


def mamba3_install_owned(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = neural_block_snapshot(mods, block, context)
    try:
        return mamba3_install_owned(mods, block, context, fields)
    finally:
        mods.released()
        _mamba3_publish_runtime(fields, block)


def mamba3_release_owned(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(block)
    var session = py_get(fields, "_prefill_session")
    var binding = py_get(fields, "_prefill_binding")
    if session is not None and binding is not None:
        var entries = _mamba_saved_entries(mods, binding, context)
        var native = py_get(fields, "_owned_native_provider")
        if neural_descriptor_matches(native, binding):
            var handle = neural_descriptor_handle(native,
                _mamba_saved_entry(mods, entries, "mamba_prefill_native_handle_v1"), session)
            mods.released()
            _ = neural_descriptor_execute(native, 18, [], [handle], [], False)
            mods.released()
        else:
            _ = py_call(_mamba_saved_entry(mods, entries, "mamba3_prefill_session_close"), session)
            mods.released()
    py_setitem(fields, "_prefill_session", PythonObject(None))
    py_setitem(fields, "_prefill_binding", PythonObject(None))
    py_setitem(fields, "_owned_generation", PythonObject(None))
    py_setitem(fields, "_owned_native_provider", PythonObject(None))
    py_setitem(fields, "_owned_restore_pending", PythonObject(False))
    return PythonObject(None)


def mamba3_session_info(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var fields = _fields(block)
    var session = py_get(fields, "_prefill_session")
    var binding = py_get(fields, "_prefill_binding")
    if session is None or binding is None:
        return PythonObject(None)
    var entries = _mamba_saved_entries(mods, binding, context)
    var entry = _mamba_saved_entry(mods, entries, "mamba3_prefill_session_info", False)
    if entry is None:
        return PythonObject(None)
    var values = py_call(entry, session)
    mods.released()
    var keys = [String("weight_uploads"), String("weight_recopies"), String("weight_reuses"),
                String("backward_reuses"), String("backward_recomputes"), String("stages_held")]
    var result = bi.dict()
    for i in range(len(keys)):  # small-loop(six named session counters)
        py_setitem(result, keys[i], Int(py=values[i]))
    return result


def mamba3_export_owned(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
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
    var entries = _mamba_saved_entries(mods, binding, context)
    var session = fields["_prefill_session"]
    var native = py_get(fields, "_owned_native_provider")
    if neural_descriptor_matches(native, binding):
        var handle = neural_descriptor_handle(native,
            _mamba_saved_entry(mods, entries, "mamba_prefill_native_handle_v1"), session)
        mods.released()
        _ = neural_descriptor_reserved_execute(native, session, handle, 0, 1, 17,
            _neural_int_words(addresses), [Int(py=generation)], [])
        mods.released()
    else:
        _ = py_call(_mamba_saved_entry(mods, entries, "mamba3_prefill_session_export_weights"), session, addresses, generation)
        mods.released()
    var result = bi.dict()
    var names = _mamba3_weight_names()
    for i in range(len(names)):  # driver-loop: over named parameter slots
        py_setitem(result, names[i], arrays[i])
    _ = keep^
    return result


def mamba3_getstate(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var fields = _fields(block)
    var result = py_call(bi.dict, fields)
    if py_get(fields, "_owned_generation") is not None:
        var snapshot = mamba3_export_owned(mods, block, context)
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


def mamba3_setstate(mut mods: NativeModules, block: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
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
    _ = neural_block_fields(mods, candidate, context)
    var fields = _fields(block)
    _ = py_call(fields.update, candidate_fields)
    py_setitem(fields, "_prefill_session", PythonObject(None))
    py_setitem(fields, "_prefill_binding", PythonObject(None))
    py_setitem(fields, "_owned_generation", PythonObject(None))
    py_setitem(fields, "_owned_native_provider", PythonObject(None))
    py_setitem(fields, "_native_mamba_block_provider_cache", PythonObject(None))
    if String(fields["_native_route"]) == "gpu" and String(fields["numeric_mode"]) == "identical" and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
        var entries = _m3_entries(mods, block, fields, context)
        py_setitem(fields, "_prefill_session", _native_function(mods, entries, "mamba3_prefill_session_create")())
        mods.released()
        py_setitem(fields, "_prefill_binding", _m3_canonical(mods, entries))
    return PythonObject(None)


def _shape_of(value: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_type(value) is g[].array_cls:
        return value.shape
    return py_call(g[].bi.memoryview, value).shape


def mamba3_set_input_states(mut mods: NativeModules,
    state: PythonObject, theta: PythonObject, h: PythonObject, k: PythonObject, v: PythonObject,
    context: PythonObject,
) raises -> PythonObject:
    var slots = _mamba_state_fields(mods, state, "mamba3", context)
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
    """Provider ownership around a dependency-free native decode plan.

    Python module/session handles stay at this admission/publication boundary.
    Typed state, dimensions, weights and launch preparation belong to the plan.
    """
    var binding: PythonObject
    var device: PythonObject
    var provider: PythonObject
    var device_address: Int
    var dt_limit: PythonObject
    var plan: MambaDecodePlan

    def __init__(out self, kind: Int, binding: PythonObject):
        self.binding = binding
        self.device = PythonObject(None)
        self.provider = PythonObject(None)
        self.device_address = 0
        self.dt_limit = PythonObject(None)
        self.plan = MambaDecodePlan(kind)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("MambaDecodeHandle(kind=", self.plan.kind, ", open=", self.plan.open, ", rows=", self.plan.b, ")")

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


def _session_layout(mut handle: MambaDecodePlan, fields: PythonObject) raises:
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


def mamba_session_init(mut mods: NativeModules, session: PythonObject, block: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var fields = neural_block_snapshot(mods, block, context)
    var kind_name = String(py_get(fields, "_native_kind", PythonObject("")))
    var kind = 1 if kind_name == "mamba1" else (2 if kind_name == "mamba2" else (3 if kind_name == "mamba3" else 0))
    if kind == 0:
        raise Error("TypeError: a Mamba decode session takes a Mamba1Block, Mamba2Block or Mamba3Block")
    var what = _session_name(kind)
    _ = _mamba_record_fields(mods, session, context["mamba_session_classes"][kind_name],
        [String("_handle"), String("_block"), String("_state")])
    var binding = _model_bindings(mods, fields["_native_route"], kind_name, context, py_get(fields, "numeric_mode"))[kind_name]
    var device_arm = _native_has_entry(mods, binding, kind_name + "_session_create")
    if not device_arm and not _native_has_entry(mods, binding, kind_name + "_decode_step"):
        raise Error("NotImplementedError: " + what + ": the loaded binding exports neither a resident decode session nor the per-call " + kind_name + "_decode_step entry")
    if state is None:
        raise Error(what + ": state is required (allocate_state)")
    var slots = _mamba_state_fields(mods, state, kind_name, context)
    if py_get(slots, "_resident_session") is not None:
        raise Error(what + ": the state is owned by an open resident decode session (its pieces live in the session, "
                    + "on the device or on the host, and the caller's buffers are stale); call sync_state() or close() on the session first")
    var hshape = _shape_of(slots["h"])
    if len(hshape) != (3 if kind == 1 else 4) or Int(py=hshape[0]) < 1:
        raise Error(what + ": state.h has an invalid shape " + String(hshape))
    var handle = MambaDecodeHandle(kind, binding)
    handle.plan.b = Int(py=hshape[0])
    handle.plan.d_model = Int(py=fields["d_model"])
    if kind == 1:
        if _native_has_entry(mods, binding, "mamba1_profile"):
            handle.plan.profile = Int(py=_native_function(mods, binding, "mamba1_profile")())
            mods.released()
        _m1_check(slots, handle.plan.profile, what)
        handle.plan.position = Int(py=py_get(slots, "absolute_position", PythonObject(0)))
    elif kind == 2:
        handle.plan.q = Int(py=slots["buffered_tokens"])
        handle.dt_limit = fields["dt_limit"]
    else:
        handle.plan.q = Int(py=slots["buffered_tokens"])
        handle.plan.pending = Bool(py=slots["pending"])
    _session_layout(handle.plan, fields)
    mods.released()
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var pieces = List[Int]()
    for i in range(len(handle.plan.names)):  # driver-loop: over the state's named pieces
        pieces.append(_mutable(slots[handle.plan.names[i]], handle.plan.dims[i], keep, copies, handle.plan.names[i]))
    var weights = fields["_w"]
    if device_arm:
        # The weights, then the state pieces, as the retired session opened them.
        var addresses = py_ints([])
        for i in range(len(weights)):  # driver-loop: over block weight slots
            _ = py_call(addresses.append, generation_address(weights[i], keep))
        for i in range(len(pieces)):  # driver-loop: over the state's named pieces
            _ = py_call(addresses.append, pieces[i])
        var params = py_ints([handle.plan.b, handle.plan.d_model])
        if kind == 1:
            if handle.plan.profile == 2:
                _ = py_call(params.append, handle.plan.position)
        elif kind == 2:
            _ = py_call(params.append, handle.plan.q)
            _ = py_call(params.extend, handle.dt_limit)
        else:
            _ = py_call(params.append, handle.plan.q)
            _ = py_call(params.append, 1 if handle.plan.pending else 0)
        handle.device = _native_function(mods, binding, kind_name + "_session_create")()
        mods.released()
        if getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1" and String(fields["numeric_mode"]) == "identical":
            handle.provider = neural_descriptor_reuse(py_get(fields, "_native_mamba_block_provider_cache"),
                _native_function(mods, binding, "mamba_block_api_v1"), String(MAMBA_BLOCK_ABI_NAME), 44)
            mods.released()
            handle.device_address = neural_descriptor_handle(handle.provider,
                _native_function(mods, binding, kind_name + "_decode_native_handle_v1"), handle.device)
            mods.released()
            var p: List[Int] = [handle.plan.b, handle.plan.d_model]
            var f = List[Float64]()
            if kind == 1 and handle.plan.profile != 1:
                raise Error("Mamba1 resident decode requires the binding's supported profile")
            if kind == 2:
                p.append(handle.plan.q)
                f.append(Float64(py=handle.dt_limit[0]))
                f.append(Float64(py=handle.dt_limit[1]))
            elif kind == 3:
                p.append(handle.plan.q)
                p.append(Int(handle.plan.pending))
            var base = 20 + 8 * (kind - 1)
            _ = neural_descriptor_reserved_execute(handle.provider, handle.device, handle.device_address,
                base, base + 2, base + 3, _neural_int_words(addresses), p, f)
            mods.released()
        else:
            _ = py_call(_native_function(mods, binding, kind_name + "_session_open"), handle.device, addresses, params)
            mods.released()
    else:
        if getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
            handle.provider = neural_descriptor_reuse(py_get(fields, "_native_mamba_state"),
                _native_function(mods, binding, "mamba_state_api_v1"), String(MAMBA_STATE_ABI_NAME))
            mods.released()
        # The host arm owns typed copies, the device arm's upload on the host.
        for i in range(len(weights)):  # driver-loop: over block weight slots
            handle.plan.weights.append(mamba_host_copy(generation_address(weights[i], keep), Int(py=weights[i].size)))
        for i in range(len(pieces)):  # driver-loop: over the state's named pieces
            handle.plan.parts.append(mamba_host_copy(pieces[i], _count(handle.plan.dims[i])))
    handle.plan.open = True
    _ = keep^
    var session_fields = _fields(session)
    py_setitem(session_fields, "_handle", PythonObject(alloc=handle^))
    py_setitem(session_fields, "_block", block)
    py_setitem(session_fields, "_state", state)
    py_setitem(_fields(state), "_resident_session", PythonObject(True))
    return PythonObject(None)


def mamba_session_open(mut mods: NativeModules, block: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var kind = String(py_get(_fields(block), "_native_kind", PythonObject("")))
    var classes = context["mamba_session_classes"]
    if not py_contains(classes, kind):
        raise Error("TypeError: a Mamba decode session takes a Mamba1Block, Mamba2Block or Mamba3Block")
    var session = py_call(bi.object.__new__, classes[kind])
    _ = mamba_session_init(mods, session, block, state, context)
    return session


def _session_require_open(handle: PythonObject, what: String) raises:
    if not handle.downcast_value_ptr[MambaDecodeHandle]()[].plan.open:
        raise Error(what + ": the session is closed")


def mamba_session_step(mut mods: NativeModules, session: PythonObject, input: PythonObject, context: PythonObject) raises -> PythonObject:
    var hobj = session._handle
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    var kind = h[].plan.kind
    var name = _kind_name(kind)
    var what = _session_name(kind) + ".step"
    _session_require_open(hobj, what)
    var x = _neural_input(mods, input, h[].plan.d_model, True, context)
    var b = h[].plan.b
    if Int(py=x.shape[0]) != b:
        raise Error(what + ": the session holds " + String(b) + " rows, x has B = " + String(x.shape[0]))
    var keep = List[PythonObject]()
    var y = generation_array([b, 1, h[].plan.d_model])
    # Reports escape in the returned tuple/public h_last_ fields, so each
    # call still owns fresh arrays. Their temporary collection is native.
    var reports = List[PythonObject](capacity=len(h[].plan.report_dims))
    for i in range(len(h[].plan.report_dims)):  # driver-loop: over the fixed report buffers
        reports.append(generation_array(h[].plan.report_dims[i]))
    var input_address = generation_address(x, keep)
    var output_address = generation_address(y, keep)
    var report_addresses = List[Int](capacity=len(reports))
    for i in range(len(reports)):  # driver-loop: capture newly allocated report buffers at the provider boundary
        report_addresses.append(generation_address(reports[i], keep))
    var addresses = h[].plan.step_addresses(h[].device is not None, input_address, output_address, report_addresses)
    if h[].device is not None:
        if h[].provider is not None:
            var base = 20 + 8 * (kind - 1)
            var updated = neural_descriptor_reserved_execute(h[].provider, h[].device, h[].device_address,
                base + 1, base + 2, base + 4, addresses, [], [])
            mods.released()
            if kind != 1:
                h[].plan.q = updated
        else:
            var updated = py_call(_native_function(mods, h[].binding, name + "_session_step"), h[].device, py_ints(addresses))
            mods.released()
            if kind != 1:
                h[].plan.q = Int(py=updated)
    else:
        # Typed launch preparation has no Python/provider dependency.
        if h[].provider is not None:
            var p = h[].plan.step_parameters()
            var f = List[Float64]()
            if kind == 2:
                f = h[].plan.step_floats(Float64(py=h[].dt_limit[0]), Float64(py=h[].dt_limit[1]))
            var updated = neural_descriptor_execute(h[].provider, 2 * kind - 1, addresses, p, f)
            mods.released()
            if kind != 1:
                h[].plan.q = updated
        else:
            var params = py_ints(h[].plan.step_parameters(False))
            if kind == 2:
                _ = py_call(params.extend, h[].dt_limit)
            var updated = py_call(_native_function(mods, h[].binding, name + "_decode_step"), py_ints(addresses), params)
            mods.released()
            if kind != 1:
                h[].plan.q = Int(py=updated)
        if kind == 3:
            h[].plan.pending = False
    if kind == 1 and h[].plan.profile == 2:
        h[].plan.position += 1
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
    if h[].plan.kind == 1:
        _m1_check(slots, h[].plan.profile, what)
    var pieces = List[Int]()
    for i in range(len(h[].plan.names)):  # driver-loop: over the state's named pieces
        pieces.append(_mutable(slots[h[].plan.names[i]], h[].plan.dims[i], keep, copies, h[].plan.names[i]))
    return pieces^


def mamba_session_sync(mut mods: NativeModules, session: PythonObject, context: PythonObject) raises -> PythonObject:
    var hobj = session._handle
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    var kind = h[].plan.kind
    var what = _session_name(kind) + ".sync_state"
    _session_require_open(hobj, what)
    var state = session._state
    var slots = _mamba_state_fields(mods, state, _kind_name(kind), context)
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var pieces = _session_pieces(hobj, slots, what, keep, copies)
    if h[].device is not None:
        var addresses = py_ints(pieces)
        if h[].provider is not None:
            var base = 20 + 8 * (kind - 1)
            var result = neural_descriptor_reserved_execute(h[].provider, h[].device, h[].device_address,
                base + 1, base + 2, base + 5, pieces, [], [])
            mods.released()
            if kind == 2:
                h[].plan.q = result
            elif kind == 3:
                h[].plan.q = result // 2
                h[].plan.pending = (result & 1) != 0
        else:
            var exported = py_call(_native_function(mods, h[].binding, _kind_name(kind) + "_session_export_state"), h[].device, addresses)
            mods.released()
            if kind == 1:
                if h[].plan.profile == 2:
                    h[].plan.position = Int(py=exported)
            elif kind == 2:
                h[].plan.q = Int(py=exported)
            else:
                h[].plan.q = Int(py=exported[0])
                h[].plan.pending = Bool(py=exported[1])
    else:
        for i in range(len(pieces)):  # driver-loop: over the state's named pieces
            mamba_copy_f32(pieces[i], Int(h[].plan.parts[i].unsafe_ptr()), _count(h[].plan.dims[i]))
    _writeback(copies)
    _ = keep^
    if kind == 1:
        if h[].plan.profile == 2:
            py_setitem(_fields(state), "absolute_position", h[].plan.position)
    else:
        py_setitem(_fields(state), "buffered_tokens", h[].plan.q)
        if kind == 3:
            py_setitem(_fields(state), "pending", PythonObject(h[].plan.pending))
    return state


def mamba_session_load(mut mods: NativeModules, session: PythonObject, context: PythonObject) raises -> PythonObject:
    var hobj = session._handle
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    var kind = h[].plan.kind
    var what = _session_name(kind) + ".load_state"
    _session_require_open(hobj, what)
    var state = session._state
    var slots = _mamba_state_fields(mods, state, _kind_name(kind), context)
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var pieces = _session_pieces(hobj, slots, what, keep, copies)
    var q = 0
    var pending = False
    var position = h[].plan.position
    if kind == 1:
        if h[].plan.profile == 2:
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
            mods.released()
        else:
            if kind == 1:
                if h[].plan.profile == 2:
                    _ = py_call(addresses.append, position)
                _ = py_call(_native_function(mods, h[].binding, name), h[].device, addresses)
                mods.released()
            elif kind == 2:
                _ = py_call(_native_function(mods, h[].binding, name), h[].device, addresses, py_ints([q]))
                mods.released()
            else:
                _ = py_call(_native_function(mods, h[].binding, name), h[].device, addresses, py_ints([q, 1 if pending else 0]))
                mods.released()
    else:
        for i in range(len(pieces)):  # driver-loop: over the state's named pieces
            mamba_copy_f32(Int(h[].plan.parts[i].unsafe_ptr()), pieces[i], _count(h[].plan.dims[i]))
    _ = keep^
    h[].plan.q = q
    h[].plan.pending = pending
    h[].plan.position = position
    return state


def _session_release(mut mods: NativeModules, hobj: PythonObject, state: PythonObject) raises:
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    if not h[].plan.open:
        return
    h[].plan.open = False
    try:
        if state is not None:
            py_setitem(_fields(state), "_resident_session", PythonObject(None))
    finally:
        mods.released()
        # The finalizer may encounter a record whose __dict__ descriptor was
        # replaced after session creation. Refusing that record must still
        # release the already owned native buffers and device session.
        try:
            if h[].device is not None:
                if h[].provider is not None:
                    _ = neural_descriptor_execute(h[].provider, 27 + 8 * (h[].plan.kind - 1), [], [h[].device_address], [], False)
                    mods.released()
                else:
                    _ = py_call(_native_function(mods, h[].binding, _kind_name(h[].plan.kind) + "_session_close"), h[].device)
                    mods.released()
        finally:
            mods.released()
            h[].plan.weights.clear()
            h[].plan.parts.clear()


def mamba_session_close(mut mods: NativeModules, session: PythonObject, context: PythonObject) raises -> PythonObject:
    var hobj = session._handle
    if not hobj.downcast_value_ptr[MambaDecodeHandle]()[].plan.open:
        return PythonObject(None)
    try:
        _ = mamba_session_sync(mods, session, context)
    finally:
        mods.released()
        _session_release(mods, hobj, session._state)
    return PythonObject(None)


def mamba_session_drop(mut mods: NativeModules, session: PythonObject) raises -> PythonObject:
    # Finalizer: release an unclosed session without copying its state back.
    var values = _fields(session)
    var hobj = py_get(values, "_handle")
    if hobj is None:
        return PythonObject(None)
    _session_release(mods, hobj, py_get(values, "_state"))
    return PythonObject(None)


def mamba_session_is_open(session: PythonObject) raises -> PythonObject:
    return PythonObject(session._handle.downcast_value_ptr[MambaDecodeHandle]()[].plan.open)


def mamba_session_device(session: PythonObject) raises -> PythonObject:
    return session._handle.downcast_value_ptr[MambaDecodeHandle]()[].device


def mamba_session_repr(session: PythonObject) raises -> PythonObject:
    var hobj = session._handle
    var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
    var flag = String("True") if h[].plan.open else String("False")
    return PythonObject(_session_name(h[].plan.kind) + "(open=" + flag + ", rows=" + String(h[].plan.b) + ")")


# ---------------------------------------------------------------- the C boundary


def mamba_native_call(mut mods: NativeModules, operation: String, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    if py_type(values) is not bi.tuple:
        raise Error("TypeError: native Mamba operations require their ordinary argument tuple")
    _ = generation_initialize_binding(context["generation_classes"])
    if operation == "state_init":
        if len(values) != 3 or py_type(values[1]) is not bi.str:
            raise Error("TypeError: native Mamba state requires its kind and arguments")
        return mamba_state_init(mods, values[0], String(values[1]), values[2], context)
    if operation == "session_drop":
        # Finalization has the old best-effort behavior, wholly within native
        # execution. A replaced owner layout is refused without invoking it.
        try:
            _ = _mamba_session_fields(mods, values[0], context, False)
            return mamba_session_drop(mods, values[0])
        except:
            mods.released()
            return PythonObject(None)
    if operation == "session_init":
        _ = _mamba_session_fields(mods, values[0], context, False)
        var fields = neural_block_fields(mods, values[1], context)
    elif operation == "set_input_states":
        _ = _mamba_state_fields(mods, values[0], "mamba3", context, False)
    elif operation.startswith("session_") and operation != "session_open" and operation != "session_info":
        var fields = _mamba_session_fields(mods, values[0], context)
        if operation == "session_state":
            return fields["_state"]
        if operation == "session_enter":
            return values[0]
        if operation == "session_exit":
            _ = mamba_session_close(mods, values[0], context)
            return PythonObject(False)
        if operation == "session_step_public":
            var result = mamba_session_step(mods, values[0], values[1], context)
            var hobj = fields["_handle"]
            var h = hobj.downcast_value_ptr[MambaDecodeHandle]()
            if h[].plan.kind == 1:
                return result
            var block_fields = _fields(fields["_block"])
            py_setitem(block_fields, "h_last_", result[1])
            if h[].plan.kind == 3:
                py_setitem(block_fields, "k_last_", result[2])
                py_setitem(block_fields, "v_last_", result[3])
                py_setitem(block_fields, "theta_last_", result[4])
            return result[0]
    elif operation != "setstate":
        var fields = neural_block_fields(mods, values[0], context)
        var kind = String(fields["_native_kind"])
        if operation == "allocate":
            _mamba_scalar(mods, values[1], "batch_size", context)
    if operation == "forward_public" or operation == "step_public":
        var step = operation == "step_public"
        var lengths = PythonObject(None)
        if not step:
            lengths = values[3]
        var result = mamba3_forward(mods, values[0], values[1], values[2], lengths, step, context)
        # Retain report objects through dictionary insertion before releasing
        # the tuple. No Python assignment or user __setattr__ runs on return.
        var fields = _fields(values[0])
        py_setitem(fields, "h_last_", result[1])
        py_setitem(fields, "k_last_", result[2])
        py_setitem(fields, "v_last_", result[3])
        py_setitem(fields, "theta_last_", result[4])
        return result[0]
    if operation == "forward":
        return mamba3_forward(mods, values[0], values[1], values[2], values[3], False, context)
    if operation == "step":
        return mamba3_forward(mods, values[0], values[1], values[2], PythonObject(None), True, context)
    if operation == "allocate":
        return mamba3_allocate(mods, values[0], values[1], context)
    if operation == "install_owned":
        return mamba3_install_owned(mods, values[0], context)
    if operation == "release_owned":
        return mamba3_release_owned(mods, values[0], context)
    if operation == "session_info":
        return mamba3_session_info(mods, values[0], context)
    if operation == "export_owned":
        return mamba3_export_owned(mods, values[0], context)
    if operation == "getstate":
        return mamba3_getstate(mods, values[0], context)
    if operation == "setstate":
        return mamba3_setstate(mods, values[0], values[1], context)
    if operation == "set_input_states":
        return mamba3_set_input_states(mods, values[0], values[1], values[2], values[3], values[4], context)
    if operation == "session_open":
        return mamba_session_open(mods, values[0], values[1], context)
    if operation == "session_init":
        return mamba_session_init(mods, values[0], values[1], values[2], context)
    if operation == "session_step":
        return mamba_session_step(mods, values[0], values[1], context)
    if operation == "session_sync":
        return mamba_session_sync(mods, values[0], context)
    if operation == "session_load":
        return mamba_session_load(mods, values[0], context)
    if operation == "session_close":
        return mamba_session_close(mods, values[0], context)
    if operation == "session_drop":
        return mamba_session_drop(mods, values[0])
    if operation == "session_is_open":
        return mamba_session_is_open(values[0])
    if operation == "session_device":
        return mamba_session_device(values[0])
    if operation == "session_repr":
        return mamba_session_repr(values[0])
    raise Error("mamba native: unknown operation " + operation)


def mamba_native_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var mods = NativeModules()
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var bi = mods.module("builtins")
        if py_type(args) is not bi.tuple or len(args) != 3 or py_type(args[0]) is not bi.str:
            raise Error("TypeError: native Mamba call requires an operation name, argument tuple and context")
        var result = mamba_native_call(mods, String(args[0]), args[1], args[2])
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
