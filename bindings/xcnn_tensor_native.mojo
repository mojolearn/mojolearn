# SPDX-License-Identifier: Apache-2.0
# Whole CNN DeviceTensor public operations. Slot identities are import-connected;
# shapes and buffer owners are admitted before conversion or native execution.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.neural_block_admission import neural_plain_value
from bindings.sequential_admission import sequential_admit_value, sequential_integer
from bindings.sequence_array_native import sequence_array_cast
from bindings.xcnn_common import (
    xcnn_builtins, xcnn_tensor, xcnn_tensor_slot, xcnn_tensor_set, xcnn_tensor_admit,
    xcnn_shape, xcnn_size, xcnn_tuple, xcnn_new, xcnn_out_new, xcnn_wrap,
    xcnn_host_of, xcnn_f32, xcnn_entries, xcnn_module_entries, xcnn_device_io_off,
    xcnn_is_t, xcnn_reshape_t,
)
from bindings.py_typed_call import py_call_typed


def _shape(value: PythonObject, context: PythonObject, infer: Bool = False) raises -> List[Int]:
    var b = xcnn_builtins()
    if py_type(value) is not b.list and py_type(value) is not b.tuple:
        raise Error("TypeError: CNN tensor shape requires a native list or tuple")
    var out = List[Int]()
    for i in range(len(value)):  # driver-loop: passive shape words before any conversion
        if not sequential_integer(value[i], context):
            raise Error("TypeError: CNN tensor dimensions require native integers")
        var size = Int(py=value[i])
        if size < (-1 if infer else 0):
            raise Error("ValueError: CNN tensor dimensions must be nonnegative")
        out.append(size)
    return out^


def xcnn_tensor_operation(owner: PythonObject, operation: String, args: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var b = xcnn_builtins()
    if operation == "tensor_to":
        if xcnn_is_t(args[0], context):
            xcnn_tensor_admit(args[0], context)
            return args[0]
        var array = xcnn_f32(args[0], "x", context)
        var settings = Python.dict(numeric_mode=args[1])
        var entries = xcnn_entries(settings, context)
        if xcnn_device_io_off(entries, context):
            return array
        return xcnn_wrap(entries, array, context)
    if operation == "tensor_new" or operation == "tensor_new_out" or operation == "tensor_wrap":
        if owner is not context["xcnn_tensor_class"]:
            raise Error("TypeError: CNN tensor allocation requires the connected class")
        var entries = xcnn_module_entries(args[0], context)
        if operation == "tensor_wrap":
            sequential_admit_value(args[1], context)
            return xcnn_wrap(entries, args[1], context)
        var shape = _shape(args[1], context)
        return xcnn_new(entries, shape, False, context) if operation == "tensor_new" else xcnn_out_new(entries, shape, context)
    if py_type(owner) is not context["xcnn_tensor_class"]:
        raise Error("TypeError: CNN tensor operation requires its exact native owner")
    if operation == "tensor_init":
        var entries = xcnn_module_entries(args[0], context)
        var shape = _shape(args[1], context)
        if args[2] is not None and (py_type(args[2]) is not b.int or Int(py=args[2]) < 1):
            raise Error("TypeError: CNN tensor handle requires a positive native integer")
        if args[3] is not None:
            sequential_admit_value(args[3], context)
        if args[4] is not None and not xcnn_is_t(args[4], context):
            raise Error("TypeError: CNN tensor base requires an exact native tensor")
        xcnn_tensor_set(owner, "b", entries["module"], context)
        xcnn_tensor_set(owner, "shape", xcnn_tuple(shape), context)
        xcnn_tensor_set(owner, "h", args[2], context)
        xcnn_tensor_set(owner, "_host", args[3], context)
        xcnn_tensor_set(owner, "_base", args[4], context)
        return PythonObject(None)
    if operation == "tensor_finalize":
        # Partially initialized objects and interpreter shutdown are legal.
        # Suppression lives in native code, with handles cleared before free.
        try:
            var handle = xcnn_tensor_slot(owner, "h", context)
            var base = xcnn_tensor_slot(owner, "_base", context)
            if base is None and handle is not None:
                if py_type(handle) is not b.int or Int(py=handle) < 1:
                    return PythonObject(None)
                var entries = xcnn_module_entries(xcnn_tensor_slot(owner, "b", context), context)
                xcnn_tensor_set(owner, "h", PythonObject(None), context)
                _ = py_call_typed(entries["symbols"]["x_cnn_res_free"], handle)
        except:
            pass
        return PythonObject(None)
    xcnn_tensor_admit(owner, context)
    var shape = xcnn_shape(owner)
    if operation == "tensor_size":
        return PythonObject(xcnn_size(shape))
    if operation == "tensor_ndim":
        return PythonObject(len(shape))
    if operation == "tensor_numpy" or operation == "tensor_array":
        var result = xcnn_host_of(owner, context)
        if operation == "tensor_numpy":
            return result
        return sequence_array_cast(result, args[0], context["xcnn_array_context"])
    if operation == "tensor_reshape":
        var requested = args[0]
        if py_type(requested) is not b.tuple:
            raise Error("TypeError: CNN reshape requires a native argument tuple")
        if len(requested) == 1 and (py_type(requested[0]) is b.list or py_type(requested[0]) is b.tuple):
            requested = requested[0]
        var dims = _shape(requested, context, True)
        var missing = -1
        var known = 1
        for i in range(len(dims)):  # driver-loop: shape inference from admitted integer metadata
            if dims[i] == -1:
                if missing >= 0:
                    raise Error("ValueError: CNN reshape accepts only one inferred dimension")
                missing = i
            else:
                known *= dims[i]
        if missing >= 0:
            dims[missing] = xcnn_size(shape) // known if known else 0
        return xcnn_reshape_t(owner, dims, context)
    raise Error("ValueError: unknown CNN tensor operation")
