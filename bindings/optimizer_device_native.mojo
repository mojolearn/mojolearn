# SPDX-License-Identifier: Apache-2.0
# Native ownership and consumed outputs of training device arrays.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.optimizer_admission import opt_value, opt_scalar
from bindings.optimizer_public_door import new_array, TensorView, tensor_view, base_format, contiguous_addr
from bindings.optimizer_public_words import optimizer_python_error


def device_entries(binding: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var bindings = context["bindings"]
    for mode in ["identical", "fast"]:  # driver-loop: over the two compiled numeric-mode owners
        var entries = bindings[PythonObject(mode)]
        if py_type(entries) is b.dict and binding is entries["module"]:
            if py_get(entries, "train_dev_alloc") is None or py_get(entries, "train_dev_get") is None or py_get(entries, "train_dev_put") is None or py_get(entries, "train_dev_free") is None:
                raise Error("RuntimeError: this build has no training device arrays (identical GPU builds only)")
            return entries
    raise Error("TypeError: training device arrays require their connected native binding owner")


def device_shape(shape: PythonObject, context: PythonObject) raises -> List[Int]:
    opt_scalar(shape, context)
    var b = Python.import_module("builtins")
    if py_type(shape) is not b.tuple and py_type(shape) is not b.list:
        raise Error("TypeError: training device shape requires a native tuple or list")
    var dims = List[Int]()
    for i in range(len(shape)):  # driver-loop: over native device-array shape words
        var n = Int(py=py_call(b.int, shape[i]))
        if n < 0:
            raise Error("ValueError: training device shape dimensions must be nonnegative")
        dims.append(n)
    return dims^


def shape_count(shape: List[Int]) raises -> Int:
    var size = 1
    for i in range(len(shape)):  # driver-loop: over the native device-array rank
        if shape[i] != 0 and size > 9223372036854775807 // shape[i]:
            raise Error("OverflowError: training device shape exceeds native indexing")
        size *= shape[i]
    return size


def device_set(owner: PythonObject, binding: PythonObject, handle: PythonObject, dims: List[Int], base: PythonObject) raises:
    var fields = py_fields(owner)
    var shape = Python.list()
    for i in range(len(dims)):  # driver-loop: over native device-array shape words
        _ = py_call(shape.append, dims[i])
    py_setitem(fields, "h", handle)
    py_setitem(fields, "shape", py_call(Python.import_module("builtins").tuple, shape))
    py_setitem(fields, "size", PythonObject(shape_count(dims)))
    py_setitem(fields, "_base", base)
    py_setitem(fields, "_b", binding)


def device_new(binding: PythonObject, handle: PythonObject, dims: List[Int], base: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = py_call(Python.import_module("builtins").object.__new__, context["device_class"])
    device_set(result, binding, handle, dims, base)
    return result


def device_finalize(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    # A native method descriptor calls this directly when a temporary output
    # is decref'd during rollback. Retained native types/field access/entries
    # also work after interpreter shutdown has removed builtins/sys.modules.
    var types = context["device_finalizer_types"]
    if py_type(owner) is not context["device_class"]:
        raise Error("TypeError: training device cleanup requires its exact native owner")
    var fields = py_call(context["device_fields_getter"], owner, context["device_class"])
    if py_type(fields) is not types[0]:
        raise Error("TypeError: training device cleanup requires native metadata")
    var keys = py_call(types[4], py_call(fields.keys))
    for i in range(len(keys)):  # driver-loop: over retained device ownership field names
        if py_type(keys[i]) is not types[3]:
            raise Error("TypeError: training device cleanup requires native string keys")
    var binding = py_get(fields, "_b")
    if binding is None or py_get(fields, "_base") is not None:
        return PythonObject(None)
    var handle = py_get(fields, "h")
    if py_type(handle) is not types[1]:
        raise Error("TypeError: training device cleanup requires its native integer handle")
    var entries = PythonObject(None)
    var bindings = context["bindings"]
    for mode in ["identical", "fast"]:  # driver-loop: over the two retained native numeric-mode owners
        var candidate = bindings[PythonObject(mode)]
        if py_type(candidate) is types[0] and binding is candidate["module"]:
            entries = candidate
            break
    if entries is None:
        raise Error("TypeError: training device cleanup requires its connected native binding")
    var release = py_get(entries, "train_dev_free")
    if py_type(release) is not types[2]:
        raise Error("TypeError: training device cleanup requires its compiled release entry")
    try:
        _ = py_call(release, handle)
    except:
        # Preserve the prior best-effort contract and an explicit retry's
        # ownership information when the native release itself fails.
        return PythonObject(None)
    py_setitem(fields, "_b", PythonObject(None))
    return PythonObject(None)


def device_finalize_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        _ = device_finalize(args[1], args[0])
    except error:
        return optimizer_python_error(error)
    return PythonObject(None).steal_data()


def device_operation(operation: String, owner: PythonObject, values: PythonObject, context: PythonObject,
                     selected: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if operation == "device_construct" or operation == "device_new":
        var binding = values[0]
        var entries = device_entries(binding, context)
        var dims = device_shape(values[2] if operation == "device_construct" else values[1], context)
        var size = shape_count(dims)
        if operation == "device_construct":
            if py_type(owner) is not context["device_class"]:
                raise Error("TypeError: training device array requires the exact native owner")
            opt_scalar(values[1], context)
            var base = values[3]
            if base is not None and py_type(base) is not context["device_class"]:
                raise Error("TypeError: training device array base must retain a native device-array owner")
            device_set(owner, binding, py_call(b.int, values[1]), dims, base)
            return PythonObject(None)
        var handle = py_call(entries["train_dev_alloc"], size)
        return device_new(binding, handle, dims, PythonObject(None), context)
    if operation == "device_upload":
        var source = values[0]
        if py_type(source) is context["device_class"]:
            opt_value(source, context)
            return source
        var entries = device_entries(selected["module"], context)
        var view = TensorView()
        if not tensor_view(source, view) or base_format(view.fmt) != "f":
            raise Error("TypeError: mojolearn.train_to_device: a must be float32")
        if view.count() < 1:
            raise Error("ValueError: mojolearn.train_to_device: a must be nonempty")
        var keep = List[List[Float32]]()
        var addr = contiguous_addr(source, view, keep)
        var handle = py_call(entries["train_dev_alloc"], view.count())
        try:
            _ = py_call(entries["train_dev_put"], handle, addr, view.count())
        except error:
            _ = py_call(entries["train_dev_free"], handle)
            raise error^
        _ = keep^
        return device_new(entries["module"], handle, view.shape, PythonObject(None), context)
    if py_type(owner) is not context["device_class"]:
        raise Error("TypeError: training device operation requires its exact native owner")
    var fields = py_fields(owner)
    var keys = py_call(b.list, fields)
    for i in range(len(keys)):  # driver-loop: over stored native device owner field names
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: training device owner metadata requires native string keys")
    opt_value(owner, context)
    var entries = device_entries(fields["_b"], context)
    if operation == "device_reshape":
        var dims = device_shape(values[0], context)
        if shape_count(dims) != Int(py=fields["size"]):
            raise Error("ValueError: training device reshape does not hold the array's floats")
        var base = fields["_base"]
        return device_new(fields["_b"], fields["h"], dims, owner if base is None else base, context)
    if operation == "device_host":
        var dims = device_shape(fields["shape"], context)
        var result = new_array(dims, "<f4")
        _ = py_call(entries["train_dev_get"], fields["h"], result._addr, fields["size"])
        return result
    raise Error("TypeError: unknown training device operation")
