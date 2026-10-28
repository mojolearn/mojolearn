# SPDX-License-Identifier: Apache-2.0
"""Admission and retained metadata for standalone neural public operations."""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_setattr
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.native_loaded_module import native_loaded_module
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter, AdmittedBorrowedBuffer
from bindings.generation_buffers import generation_array, py_ints
from bindings.classical_attributes import classical_connect_numpy
from bindings.sequential_admission import _one_of


def neural_native_scalar(value: PythonObject, context: PythonObject, integer: Bool = False) raises -> Bool:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.int or (not integer and (cls is b.float or cls is b.bool)):
        return True
    var admission = py_get(context, "admission_context")
    if admission is not None:
        classical_connect_numpy(admission)
        return _one_of(cls, admission["integer_types" if integer else "scalar_types"])
    return False


def neural_string_keys(value: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.dict:
        raise Error("TypeError: neural metadata requires an exact dictionary")
    var keys = py_call(b.list, value.keys())
    for i in range(len(keys)):  # driver-loop: admit every key before dictionary lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: neural metadata requires exact string keys")


def neural_plain_value(value: PythonObject, depth: Int = 0, context: PythonObject = PythonObject(None)) raises:
    if depth > 32:
        raise Error("TypeError: neural metadata nesting exceeds the supported depth")
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.int or cls is b.float or cls is b.bool or cls is b.str:
        return
    if context is not None and neural_native_scalar(value, context):
        return
    if cls is b.list or cls is b.tuple:
        for i in range(len(value)):  # driver-loop: passive configuration and descriptor words
            neural_plain_value(value[i], depth + 1, context)
        return
    if cls is b.dict:
        neural_string_keys(value)
        var keys = py_call(b.list, value.keys())
        for i in range(len(keys)):  # driver-loop: passive option dictionaries only
            neural_plain_value(value[keys[i]], depth + 1, context)
        return
    raise Error("TypeError: neural metadata requires native scalar/container values; Python conversion hooks are not supported")


def neural_record_fields(value: PythonObject, names: PythonObject) raises -> PythonObject:
    # _fields rejects a replaced __dict__ descriptor before native dictionary
    # access, including on an otherwise ordinary subclass. No getattr is used.
    var fields = _fields(value)
    var b = native_loaded_module("builtins")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, py_type(value))
    for i in range(len(ancestry)):  # driver-loop: refuse field descriptors before later shared attribute access
        var namespace = _class_fields(ancestry[i])
        for j in range(len(names)):  # small-loop(known operation field names)
            if py_contains(namespace, names[j]):
                var member = namespace[names[j]]
                var member_mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(member))
                for k in range(len(member_mro)):  # small-loop(descriptor protocol ancestry)
                    if py_contains(_class_fields(member_mro[k]), "__get__"):
                        raise Error("TypeError: neural fields require native instance storage, not Python descriptors")
    return fields


def neural_array_slots(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = context["array_class"]
    if py_type(cls) is not b.type:
        raise Error("TypeError: neural arrays require their native storage class")
    var namespace = _class_fields(cls)
    var names = Python.tuple("_store", "_base", "_pin", "_mv", "_addr", "_readonly",
        "shape", "dtype", "ndim", "size", "nbytes", "itemsize", "strides", "order")
    var slots = Python.dict()
    var types = native_loaded_module("types")
    for i in range(len(names)):  # small-loop(Array native slot schema)
        var slot = py_get(namespace, names[i])
        if (py_type(slot) is not types.MemberDescriptorType or slot.__objclass__ is not cls
                or not Bool(py=slot.__name__ == names[i])):
            raise Error("TypeError: neural Array storage descriptors changed")
        py_setitem(slots, names[i], slot)
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(mro)):  # small-loop(Array access ancestry)
        var row = _class_fields(mro[i])
        if py_contains(row, "__getattr__"):
            raise Error("TypeError: neural Array metadata hooks are not supported")
        for name in ["__getattribute__", "__setattr__"]:  # driver-loop: native admission metadata and retained owner registry
            var own = py_get(row, PythonObject(name))
            if own is not None and own is not b.object.__dict__[name]:
                raise Error("TypeError: neural Array metadata hooks are not supported")
    return slots


def neural_array_snapshot(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Capture every Array metadata word once through native member descriptors.
    # The returned passive view retains both the original owner and a real
    # buffer export; execution never consults mutable caller shape/stride slots.
    var b = native_loaded_module("builtins")
    var cls = context["array_class"]
    if py_type(value) is not cls:
        raise Error("TypeError: neural Array input requires its exact native class")
    var slots = neural_array_slots(context)
    var values = Python.dict()
    var names = py_call(b.list, slots.keys())
    for i in range(len(names)):  # small-loop(capture native Array metadata once)
        py_setitem(values, names[i], py_call(slots[names[i]].__get__, value, cls))
    for name in ["_addr", "ndim", "size", "nbytes", "itemsize"]:  # small-loop(native size and address words)
        if py_type(values[name]) is not b.int:
            raise Error("TypeError: neural Array dimensions and addresses require native integers")
    if (py_type(values["shape"]) is not b.tuple or py_type(values["strides"]) is not b.tuple
            or py_type(values["dtype"]) is not b.str or py_type(values["order"]) is not b.str
            or py_type(values["_readonly"]) is not b.bool):
        raise Error("TypeError: neural Array metadata requires native shape, dtype and layout")
    var shape = values["shape"]
    var strides = values["strides"]
    var count = 1
    var itemsize = Int(py=values["itemsize"])
    var dtype = String(values["dtype"])
    var code = "f" if dtype == "<f4" else ("i" if dtype == "<i4" else ("b" if dtype == "<i1" else ("H" if dtype == "<u2" else "")))
    var expected_itemsize = 1 if code == "b" else (2 if code == "H" else 4)
    if code == "" or itemsize != expected_itemsize or len(shape) != len(strides) or Int(py=values["ndim"]) != len(shape):
        raise Error("TypeError: neural Array storage dtype or rank is inconsistent")
    for i in range(len(shape)):  # small-loop(admitted tensor dimensions)
        if py_type(shape[i]) is not b.int or py_type(strides[i]) is not b.int:
            raise Error("TypeError: neural Array shape/strides require native integers")
        var n = Int(py=shape[i])
        if n < 0 or (n != 0 and count > 2147483647 // n):
            raise Error("neural Array shape exceeds native indexing")
        count *= n
    if Int(py=values["size"]) != count or Int(py=values["nbytes"]) != count * itemsize:
        raise Error("neural Array size differs from shape")
    var order = String(values["order"])
    if order != "C" and order != "F":
        raise Error("neural Array requires contiguous C or F layout")
    var step = itemsize
    for i in range(len(shape)):  # small-loop(verify the declared contiguous layout)
        var axis = len(shape) - i - 1 if order == "C" else i
        if count > 0 and Int(py=shape[axis]) > 1 and Int(py=strides[axis]) != step:
            raise Error("neural Array strides differ from its declared layout")
        step *= Int(py=shape[axis])
    var store = values["_store"]
    if store is None:
        store = values["_mv"]
        if py_type(store) is not b.memoryview or store.obj is None:
            raise Error("TypeError: neural Array needs retained native storage")
    admit_native_buffer_exporter(store)
    var pin = AdmittedBorrowedBuffer(store)
    var layout = pin.layout.value()
    var offset = Int(py=values["_addr"]) - layout.address
    if count == 0:
        offset = 0
    var compact = layout.contiguous()
    var f_step = layout.itemsize
    var f_compact = True
    for i in range(len(layout.shape)):  # small-loop(F-contiguous retained native exporters)
        f_compact = f_compact and layout.suboffsets[i] < 0 and (layout.shape[i] <= 1 or layout.strides[i] == f_step)
        f_step *= layout.shape[i]
    compact = compact or f_compact
    if not compact or offset < 0 or offset + count * itemsize > layout.count * layout.itemsize:
        raise Error("neural Array lies outside its retained native storage")
    var retained = py_call(b.memoryview, store)
    var result = py_call(b.object.__new__, cls)
    for i in range(len(names)):  # small-loop(publish passive captured metadata via native slot setters)
        py_call(slots[names[i]].__set__, result, values[names[i]])
    py_call(slots["_store"].__set__, result, retained)
    py_call(slots["_base"].__set__, result, value)
    _ = pin^
    return result


def neural_admit_weights(values: PythonObject, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(values) is not b.list and py_type(values) is not b.tuple:
        raise Error("TypeError: neural weights require exact native slot lists")
    for i in range(len(values)):  # driver-loop: admit each tensor owner before numerical preparation
        if values[i] is None:
            continue
        if py_type(values[i]) is context["array_class"]:
            _ = neural_array_snapshot(values[i], context)
        else:
            admit_native_buffer_exporter(values[i])


def neural_block_fields(block: PythonObject, context: PythonObject, initialized: Bool = True) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var names = Python.tuple("_native_kind", "_native_route", "numeric_mode", "numeric_profile", "weight_format",
        "d_model", "d_inner", "nheads", "conv_dim", "d_in_proj", "dt_rank", "_owned_generation", "_owned_restore_pending", "_discard_reports", "dt_limit", "n_heads", "n_kv_heads", "head_dim",
        "intermediate", "window", "max_positions", "_opts_tail", "_extended", "_w", "_wopt", "_int15",
        "_arithmetic_profile", "_arithmetic_profile_mode", "_arithmetic_profile_changed", "_W_NAMES",
        "norm", "mlp", "qkv_bias", "o_bias", "norm_bias", "mlp_bias", "qk_norm", "rope_scaling",
        "norm_eps", "rope_theta", "rope_dim", "attn_softcap")
    var fields = neural_record_fields(block, names)
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, py_type(block))
    var admitted = False
    for i in range(len(ancestry)):  # driver-loop: passive subclasses of connected public block classes
        for route in ["cpu", "gpu"]:  # driver-loop: native admission metadata and retained owner registry
            for kind in ["transformer", "mamba1", "mamba2", "mamba3"]:  # driver-loop: native admission metadata and retained owner registry
                admitted = admitted or ancestry[i] is context["block_classes"][route][kind]
    if not admitted:
        raise Error("TypeError: neural operation requires a connected native block")
    if not initialized:
        return fields
    for i in range(len(names)):  # small-loop(admit mutable scalar words before conversion or comparisons)
        var name = String(names[i])
        if name != "_w" and name != "_wopt" and name != "_int15" and py_contains(fields, names[i]):
            neural_plain_value(fields[names[i]], 0, context)
    for name in ["_native_kind", "_native_route"]:  # driver-loop: native admission metadata and retained owner registry
        if py_type(fields[name]) is not b.str:
            raise Error("TypeError: neural route metadata requires exact strings")
    if py_type(fields["d_model"]) is not b.int or Int(py=fields["d_model"]) < 1:
        raise Error("TypeError: neural d_model must be a positive native integer")
    return fields


def neural_block_weight_names(block: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = _fields(block)
    var names = py_get(fields, "_W_NAMES")
    if names is None:
        var ancestry = py_call(b.type.__dict__["__mro__"].__get__, py_type(block))
        for i in range(len(ancestry)):  # driver-loop: native admission metadata and retained owner registry
            names = py_get(_class_fields(ancestry[i]), "_W_NAMES")
            if names is not None:
                break
    if py_type(names) is not b.tuple and py_type(names) is not b.list:
        raise Error("TypeError: neural weight registry requires native names")
    for i in range(len(names)):  # driver-loop: native admission metadata and retained owner registry
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: neural weight names require exact strings")
    return py_call(b.tuple, names)


def neural_weights_snapshot(values: PythonObject, context: PythonObject, f32: Bool = True) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(values) is not b.list and py_type(values) is not b.tuple:
        raise Error("TypeError: neural weights require exact native slot lists")
    var result = Python.list()
    for i in range(len(values)):  # driver-loop: capture each weight's storage metadata exactly once
        var value = values[i]
        if value is None:
            py_call(result.append, value)
        elif py_type(value) is context["array_class"]:
            py_call(result.append, neural_block_f32(value, "weight", context) if f32 else neural_array_snapshot(value, context))
        else:
            # Foreign model buffers historically copy to native storage on
            # use. Capture that copy once and retain it for this operation.
            py_call(result.append, neural_block_f32(value, "weight", context))
    return result


def neural_metadata_snapshot(value: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(value) is b.list or py_type(value) is b.tuple:
        var values = Python.list()
        for i in range(len(value)):  # small-loop(copy already admitted option words)
            py_call(values.append, neural_metadata_snapshot(value[i]))
        return py_call(b.tuple, values) if py_type(value) is b.tuple else values
    if py_type(value) is b.dict:
        var values = Python.dict()
        var keys = py_call(b.list, value.keys())
        for i in range(len(keys)):  # driver-loop: native admission metadata and retained owner registry
            py_setitem(values, keys[i], neural_metadata_snapshot(value[keys[i]]))
        return values
    return value


def neural_block_snapshot(block: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = neural_block_fields(block, context)
    var b = native_loaded_module("builtins")
    var snapshot = py_call(b.dict, fields)
    py_setitem(snapshot, "_neural_live_fields", fields)
    py_setitem(snapshot, "_neural_weight_names", neural_block_weight_names(block))
    for name in ["dt_limit", "_opts_tail", "rope_scaling"]:  # driver-loop: native admission metadata and retained owner registry
        if py_contains(fields, name):
            py_setitem(snapshot, name, neural_metadata_snapshot(fields[name]))
    for name in ["_w", "_wopt"]:  # driver-loop: native admission metadata and retained owner registry
        if py_contains(fields, name):
            py_setitem(snapshot, name, neural_weights_snapshot(fields[name], context))
    var packed = py_get(fields, "_int15")
    if packed is not None:
        if py_type(packed) is not b.list and py_type(packed) is not b.tuple:
            raise Error("TypeError: neural packed weights require native plane lists")
        var planes = Python.list()
        for i in range(len(packed)):  # driver-loop: native admission metadata and retained owner registry
            py_call(planes.append, PythonObject(None) if packed[i] is None else neural_weights_snapshot(packed[i], context, False))
        py_setitem(snapshot, "_int15", planes)
    return snapshot


def neural_model_weights(values: PythonObject, context: PythonObject) raises -> PythonObject:
    neural_string_keys(values)
    var b = native_loaded_module("builtins")
    var keys = py_call(b.list, values.keys())
    var result = Python.dict()
    for i in range(len(keys)):  # driver-loop: named constructor weights, before any unpacking/conversion
        var value = values[keys[i]]
        if py_type(value) is context["bf16_class"] or py_type(value) is context["int8_class"]:
            var fields = neural_record_fields(value, Python.tuple("shape", "bits", "codes", "exponents"))
            neural_plain_value(fields["shape"], 0, context)
            var captured = py_call(b.object.__new__, py_type(value))
            var saved = _fields(captured)
            py_setitem(saved, "shape", neural_metadata_snapshot(fields["shape"]))
            for name in ["bits", "codes", "exponents"]:  # small-loop(packed weight storage fields)
                if py_contains(fields, name):
                    var storage = fields[name]
                    if py_type(storage) is context["array_class"]:
                        storage = neural_array_snapshot(storage, context)
                    else:
                        admit_native_buffer_exporter(storage)
                        # A retained native memoryview captures foreign metadata
                        # once for the existing packed-weight unpack operation.
                        storage = py_call(b.memoryview, storage)
                    py_setitem(saved, name, storage)
            py_setitem(result, keys[i], captured)
        else:
            py_setitem(result, keys[i], neural_block_f32(value, String(keys[i]), context))
    return result


def neural_block_f32(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    if py_type(value) is context["array_class"]:
        var captured = neural_array_snapshot(value, context)
        if String(captured.dtype) != "<f4":
            raise Error("TypeError: neural " + name + " requires float32 input")
        if String(captured.order) == "C":
            return captured
        # F-order follows the existing byte-preserving C preparation, with
        # metadata captured once above and the original owner retained.
        from transformer.causal_lm_construct_impl import causal_lm_gather_f32
        from core.hostptr import f32_ptr
        var dims = List[Int]()
        var steps = List[Int]()
        for i in range(len(captured.shape)):  # driver-loop: native admission metadata and retained owner registry
            dims.append(Int(py=captured.shape[i]))
            steps.append(Int(py=captured.strides[i]))
        var result = generation_array(dims)
        causal_lm_gather_f32(f32_ptr(Int(py=captured._addr)), f32_ptr(Int(py=result._addr)), Int(py=captured.size), dims, steps)
        _ = captured^
        return result
    admit_native_buffer_exporter(value)
    var pin = AdmittedBorrowedBuffer(value)
    if pin.format != "f" and pin.format != "<f" and pin.format != "=f" and pin.format != "@f":
        raise Error("TypeError: neural " + name + " requires float32 input")
    var layout = pin.layout.value()
    if layout.itemsize != 4:
        raise Error("TypeError: neural float32 input requires four-byte elements")
    _ = neural_array_slots(context)
    var result = generation_array(layout.shape)
    layout.copy_to(Int(py=result._addr))
    _ = pin^
    return result
