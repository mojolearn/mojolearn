# SPDX-License-Identifier: Apache-2.0
# Packed storage front-end over the common linalg provider. Public Python bodies
# make one binding call; this owner admits buffers once, retains pins, and uses
# the existing linalg numerical entries without redispatching Python wrappers.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.native_loaded_module import native_loaded_module
from bindings.native_instance_item import native_instance_item
from bindings.classical_pandas import classical_admit_value
from bindings.native_buffer_owner import native_storage_allocate
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_wrap_array
from bindings.sequential_inputs import sequential_sequence_f32
from bindings.linalg_ops_buffers import LinBuf, LinAdmission, lin_admitted_c, lin_new, lin_tuple, lin_is_f32, lin_dtype_name
from bindings.linalg_ops_lowbit import lin_connected_binding, lin_lowbit, lin_call, lin_ints
from bindings.optimizer_public_door import base_format, shape_text



def low_fields(value: PythonObject) raises -> PythonObject:
    var fields = py_fields(value)
    if py_type(fields) is not py_type(Python.dict()):
        raise Error("TypeError: packed storage requires an exact native instance dictionary")
    var keys = py_call(native_loaded_module("builtins").list, fields.keys())
    for i in range(len(keys)):  # small-loop(stored packed weight field names)
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: packed storage field names must be exact strings")
    return fields


def low_empty(name: String, shape: List[Int]) raises:
    var what = "feature(s)" if len(shape) == 2 and shape[0] != 0 else "sample(s)"
    var text = shape_text(shape)
    raise Error("mojolearn: " + name + " is empty, shape " + text + ": Found array with 0 " + what
        + " (shape=" + text + ") while a minimum of 1 is required.")

def low_kind(value: PythonObject, context: PythonObject) raises -> Int:
    var cls = py_type(value)
    if cls is context["family_low_bf16"]:
        return 1
    if cls is context["family_low_int8"]:
        return 2
    return 0


def low_format(value: PythonObject) raises -> String:
    if py_type(value) is not py_type(PythonObject("")):
        raise Error("TypeError: mojolearn.lowbit: format must be an exact string")
    var fmt = String(py=value)
    if fmt != "float32" and fmt != "bfloat16" and fmt != "int8":
        raise Error("mojolearn.lowbit: unknown format '" + fmt + "'; one of ('float32', 'bfloat16', 'int8')")
    return fmt


def low_name(value: PythonObject) raises -> String:
    if py_type(value) is not py_type(PythonObject("")):
        raise Error("TypeError: mojolearn.lowbit: tensor names must be exact strings")
    return String(py=value)


def low_binding(mut binding: PythonObject, context: PythonObject) raises -> PythonObject:
    if binding is None:
        binding = lin_lowbit(lin_connected_binding(context), context)
    return binding


def low_wrap(kind: Int, q: LinBuf, e: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = native_storage_allocate(context["family_low_bf16"] if kind == 1 else context["family_low_int8"])
    var fields = py_fields(result)
    py_setitem(fields, "bits" if kind == 1 else "codes", q.keep)
    if kind == 2:
        py_setitem(fields, "exponents", e)
    py_setitem(fields, "shape", lin_tuple(q.shape))
    return result


def low_admit(value: PythonObject, typestr: String, rank: Int, name: String,
              context: PythonObject, force_copy: Bool = False) raises -> LinBuf:
    var input = LinAdmission(value, context)
    var code = base_format(input.view.fmt)
    var valid = (typestr == "<u2" and code == "H" and input.view.itemsize == 2
        or typestr == "<i1" and code == "b" and input.view.itemsize == 1
        or typestr == "<i4" and (code == "i" or code == "l") and input.view.itemsize == 4)
    if not valid or rank >= 0 and len(input.view.shape) != rank:
        if typestr == "<u2":
            raise Error("TypeError: mojolearn.lowbit.BF16Weight: bits must be a 2-D uint16 array")
        if typestr == "<i1":
            raise Error("TypeError: mojolearn.lowbit.Int8Weight: codes must be a 2-D int8 array")
        raise Error("TypeError: mojolearn.lowbit.Int8Weight: exponents must be int32 of shape (rows,)")
    if input.view.count() == 0:
        low_empty(name, input.view.shape)
    return lin_admitted_c(input^, typestr, name, context, force_copy)


def low_exponents(value: PythonObject, rows: Int, context: PythonObject,
                  force_copy: Bool = False) raises -> LinBuf:
    var e = low_admit(value, "<i4", 1, "exponents", context, force_copy)
    if e.dim(0) != rows:
        raise Error("TypeError: mojolearn.lowbit.Int8Weight: exponents must be int32 of shape (rows,)")
    return e^


def low_materialize(value: PythonObject, mut binding: PythonObject, context: PythonObject) raises -> PythonObject:
    var kind = low_kind(value, context)
    if kind == 0:
        return value
    var fields = low_fields(value)
    var q = low_admit(native_instance_item(fields, PythonObject("bits" if kind == 1 else "codes")), "<u2" if kind == 1 else "<i1", 2, "weight", context)
    var result = lin_new(q.shape, "<f4", context)
    if kind == 1:
        _ = lin_call(low_binding(binding, context), context, "from_bf16", PythonObject(result.addr), PythonObject(q.addr), lin_ints(q.count()))
    else:
        var e = low_exponents(native_instance_item(fields, PythonObject("exponents")), q.dim(0), context)
        _ = lin_call(low_binding(binding, context), context, "dequantize_int8", PythonObject(result.addr), PythonObject(q.addr), PythonObject(e.addr), lin_ints(q.dim(0), q.dim(1)))
        _ = e^
    _ = q^
    return result.keep


def low_pack_admitted(var input: LinAdmission, fmt: String, name: String,
                      mut binding: PythonObject, context: PythonObject) raises -> PythonObject:
    if not lin_is_f32(input.view):
        raise Error("TypeError: mojolearn.lowbit: " + name + " has dtype " + lin_dtype_name(input.keep, input.view) + "; pack takes float32")
    if len(input.view.shape) != 2:
        raise Error("mojolearn.lowbit: " + name + " must be 2-D to pack, got shape " + shape_text(input.view.shape))
    if input.view.count() == 0:
        low_empty(name, input.view.shape)
    var a = lin_admitted_c(input^, "<f4", name, context, fmt == "float32")
    if fmt == "float32":
        return a.keep
    var q = lin_new(a.shape, "<u2" if fmt == "bfloat16" else "<i1", context)
    var result = PythonObject(None)
    if fmt == "bfloat16":
        _ = lin_call(low_binding(binding, context), context, "to_bf16", PythonObject(q.addr), PythonObject(a.addr), lin_ints(a.count()))
        result = low_wrap(1, q, PythonObject(None), context)
    else:
        var e = lin_new([a.dim(0)], "<i4", context)
        _ = lin_call(low_binding(binding, context), context, "quantize_int8", PythonObject(q.addr), PythonObject(e.addr), PythonObject(a.addr), lin_ints(a.dim(0), a.dim(1)))
        result = low_wrap(2, q, e.keep, context)
        _ = e^
    _ = a^
    _ = q^
    return result


def low_values(weights: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(weights) is b.dict:
        return py_call(b.list, weights.values())
    if py_type(weights) is b.list or py_type(weights) is b.tuple:
        return weights
    raise Error("TypeError: mojolearn.lowbit: use a native dict, list or tuple; Python iteration hooks are not supported")


def low_formats(weights: PythonObject, context: PythonObject) raises -> String:
    var values = low_values(weights)
    var kind = 0
    for i in range(len(values)):  # driver-loop: over named packed-storage owners
        var current = low_kind(values[i], context)
        if current != 0:
            if kind != 0 and current != kind:
                raise Error("mojolearn.lowbit: weights mix formats ['bfloat16', 'int8']; store a model one way")
            kind = current
    return "float32" if kind == 0 else ("bfloat16" if kind == 1 else "int8")


def lowbit_weight_operation(model: PythonObject, operation: String, args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var binding = PythonObject(None)
    if operation == "family_low_is_packed":
        return PythonObject(low_kind(args[0], context) != 0)
    if operation == "family_low_format_of":
        return PythonObject(low_formats(args[0], context))
    if operation == "family_low_repr":
        var kind = low_kind(model, context)
        var fields = low_fields(model)
        var shape = native_instance_item(fields, PythonObject("shape"))
        classical_admit_value(shape, context)
        var text = String("BF16Weight" if kind == 1 else "Int8Weight")
        return PythonObject(text + "(shape=" + String(py=py_call(native_loaded_module("builtins").repr, shape)) + ")")
    if operation == "family_low_bf16_init" or operation == "family_low_int8_init":
        var kind = 1 if operation == "family_low_bf16_init" else 2
        if low_kind(model, context) != kind:
            raise Error("TypeError: lowbit storage requires its native public class without Python subclass hooks")
        var q = low_admit(args[0], "<u2" if kind == 1 else "<i1", 2, "bits" if kind == 1 else "codes", context)
        var fields = low_fields(model)
        if kind == 2:
            var e = low_exponents(args[1], q.dim(0), context)
            py_setitem(fields, "codes", q.keep)
            py_setitem(fields, "exponents", e.keep)
            _ = e^
        else:
            py_setitem(fields, "bits", q.keep)
        py_setitem(fields, "shape", lin_tuple(q.shape))
        _ = q^
        return PythonObject(None)
    if operation == "family_low_materialize_one":
        return low_materialize(args[0], binding, context)
    if operation == "family_low_widen_bf16":
        var input = LinAdmission(args[0], context)
        var a = lin_admitted_c(input^, "<u2", "bits", context)
        if a.count() == 0:
            raise Error("mojolearn.linalg.from_bf16: bits has no elements")
        var result = lin_new(a.shape, "<f4", context)
        _ = lin_call(low_binding(binding, context), context, "from_bf16", PythonObject(result.addr), PythonObject(a.addr), lin_ints(a.count()))
        _ = a^
        return result.keep
    if operation == "family_low_pack_one":
        var fmt = low_format(args[1])
        var name = low_name(args[2])
        var input = LinAdmission(args[0], context)
        return low_pack_admitted(input^, fmt, name, binding, context)
    var b = native_loaded_module("builtins")
    if operation == "family_low_unpack":
        if py_type(args[0]) is not b.dict:
            return Python.tuple(args[0], "float32")
        var fmt = low_formats(args[0], context)
        if fmt == "float32":
            return Python.tuple(args[0], fmt)
        var result = Python.dict()
        var pairs = py_call(b.list, args[0].items())
        for i in range(len(pairs)):  # driver-loop: over named weights through one admitted provider
            var name = low_name(pairs[i][0])
            py_setitem(result, pairs[i][0], low_materialize(pairs[i][1], binding, context))
        return Python.tuple(result, fmt)
    if operation == "family_low_pack":
        var fmt = low_format(args[1])
        if py_type(args[0]) is not b.dict:
            raise Error("TypeError: mojolearn.lowbit.pack: weights must be a dict keyed by name; Python mapping hooks are not supported")
        var result = Python.dict()
        var pairs = py_call(b.list, args[0].items())
        for i in range(len(pairs)):  # driver-loop: over named weights through one admitted provider
            var key = pairs[i][0]
            var name = low_name(key)
            var value = pairs[i][1]
            var kind = low_kind(value, context)
            if kind != 0 and (kind == 1 and fmt == "bfloat16" or kind == 2 and fmt == "int8"):
                # Same-format repacking copies packed storage directly. It must
                # not widen/requantize, which changes both work and ownership.
                var fields = low_fields(value)
                var q = low_admit(native_instance_item(fields, PythonObject("bits" if kind == 1 else "codes")), "<u2" if kind == 1 else "<i1", 2, name, context, True)
                if kind == 1:
                    py_setitem(result, key, low_wrap(1, q, PythonObject(None), context))
                else:
                    var e = low_exponents(native_instance_item(fields, PythonObject("exponents")), q.dim(0), context, True)
                    py_setitem(result, key, low_wrap(2, q, e.keep, context))
                    _ = e^
                _ = q^
                continue
            if kind != 0:
                value = low_materialize(value, binding, context)
            if py_type(value) is b.list or py_type(value) is b.tuple:
                # Keep the existing direct float32 sequence construction and
                # its large-integer/double-rounding behavior.
                classical_admit_value(value, context)
                var sequence = sequential_sequence_f32(value, context)
                value = classical_wrap_array(sequence[1], sequence[0], "<f4", "C", context)
            var input = LinAdmission(value, context)
            if fmt != "float32" and len(input.view.shape) == 2:
                py_setitem(result, key, low_pack_admitted(input^, fmt, name, binding, context))
            else:
                py_setitem(result, key, value)
                _ = input^
        return result
    raise Error("RuntimeError: unknown lowbit storage operation " + operation)
