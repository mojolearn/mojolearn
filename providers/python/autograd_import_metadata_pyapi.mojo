# SPDX-License-Identifier: Apache-2.0
# Shared native graph/import metadata admission and error translation.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains, py_fields
from bindings.native_loaded_module import native_loaded_module
from bindings.optimizer_public_words import stash_python_error
from providers.python.autograd_python_errors_pyapi import graph_int
from providers.python.neural_descriptor_pyapi import NativeDescriptorResource


def _fields(value: PythonObject, layout: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is not layout["cls"] or py_type(cls) is not b.type:
        raise Error("TypeError: artifact import requires exact native storage classes")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    if len(mro) != len(layout["mro"]):
        raise Error("TypeError: artifact import class ancestry changed")
    for i in range(len(mro)):  # driver-loop: over connected class ancestry
        if mro[i] is not layout["mro"][i]:
            raise Error("TypeError: artifact import class ancestry changed")
    var hooks = layout["hooks"]
    for i in range(len(hooks)):  # driver-loop: over fixed native storage identities
        var row = py_call(b.type.__dict__["__dict__"].__get__, hooks[i][0])
        if py_get(row, hooks[i][1]) is not hooks[i][2]:
            raise Error("TypeError: artifact import refuses changed Python storage hooks")
    var fields = py_call(layout["fields_get"], value, cls)
    var keys = py_call(b.list, fields)
    for i in range(len(keys)):  # driver-loop: over safe keys before field lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: artifact state requires exact string keys")
    return fields


def _program_specs(value: PythonObject) raises -> PythonObject:
    # Copy passive public metadata only after exact container/scalar admission.
    # No length, index, string or integer hook on an arbitrary object may run.
    var b = native_loaded_module("builtins")
    if py_type(value) is not b.list and py_type(value) is not b.tuple:
        raise Error("TypeError: program descriptors require native list or tuple metadata")
    var result = Python.list()
    for i in range(len(value)):  # driver-loop: one snapshot of declared graph inputs/outputs
        var item = value[i]
        if (py_type(item) is not b.list and py_type(item) is not b.tuple) or len(item) != 3:
            raise Error("TypeError: program descriptors require name, shape and dtype")
        if py_type(item[0]) is not b.str or py_type(item[2]) is not b.str:
            raise Error("TypeError: program descriptor names and dtypes require exact strings")
        var shape = item[1]
        if py_type(shape) is not b.list and py_type(shape) is not b.tuple:
            raise Error("TypeError: program shapes require native list or tuple metadata")
        if len(shape) > 8:
            raise Error("ValueError: program descriptor rank must be at most eight")
        var dimensions = Python.list()
        for axis in range(len(shape)):  # small-loop(shape: admitted graph rank)
            if py_type(shape[axis]) is not b.int:
                raise Error("TypeError: program dimensions require exact native integers")
            if Int(py=shape[axis]) < 0:
                raise Error("ValueError: program dimensions must be nonnegative")
            py_call(dimensions.append, shape[axis])
        py_call(result.append, Python.tuple(item[0], py_call(b.tuple, dimensions), item[2]))
    return py_call(b.tuple, result)


def imported_program_snapshot(program: PythonObject) raises -> PythonObject:
    # Shared by array and Torch execution, before allocations or conversions.
    # Retain owners and capture metadata once for the whole public operation.
    var b = native_loaded_module("builtins")
    var module = native_loaded_module("mojolearn.importers._program")
    var context = py_fields(module)["_IMPORT_CONTEXT"]
    if py_type(context) is not b.dict:
        raise Error("TypeError: imported program requires its native connection")
    var fields = py_call(b.dict, _fields(program, context["layouts"][context["imported"]]))
    var session = fields["_s"]
    var state = py_call(b.dict, _fields(session, context["layouts"][context["session"]]))
    if state["_id"] is None:
        raise Error("ValueError: this session is closed")
    if py_type(fields["_id"]) is not b.int or py_type(state["_id"]) is not b.int:
        raise Error("TypeError: program and Session handles require exact native integers")
    var owner = state["_lifetime"].downcast_value_ptr[NativeDescriptorResource]()
    if owner[].handle == 0 or owner[].handle != Int(py=state["_id"]):
        raise Error("TypeError: Session handle differs from its native registry owner")
    var inputs = _program_specs(fields["inputs"])
    var outputs = _program_specs(fields["outputs"])
    return Python.tuple(fields, state, inputs, outputs)


def _one_of(cls: PythonObject, classes: PythonObject) raises -> Bool:
    for i in range(len(classes)):  # driver-loop: over fixed native metadata types
        if cls is classes[i]:
            return True
    return False


def _scalar(value: PythonObject, classes: PythonObject) raises:
    if not _one_of(py_type(value), classes):
        raise Error("TypeError: shape metadata requires native scalar values; Python conversion hooks are unsupported")


def _shape_words(shapes: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var names = Python.list()
    var flat = Python.list()
    if shapes is None:
        return Python.tuple(names, flat)
    var cls = py_type(shapes)
    if not _one_of(cls, context["shape_mappings"]) and cls is not b.list and cls is not b.tuple:
        raise Error("TypeError: shapes requires a native mapping or key/value sequence; Python mapping callbacks are unsupported")
    # Do not traverse dimension objects yet: a standard torch.Size is a native
    # tuple subtype. It is read through tuple's C iterator below, after checking
    # that its effective iterator has not been replaced by a Python hook.
    var provided = shapes
    if cls is b.list or cls is b.tuple:
        provided = Python.dict()
        for i in range(len(shapes)):  # driver-loop: over native key/value shape pairs
            var pair = shapes[i]
            if (py_type(pair) is not b.list and py_type(pair) is not b.tuple) or len(pair) != 2:
                raise Error("TypeError: shape entries require native key/value pairs")
            var key = pair[0]
            var key_type = py_type(key)
            _scalar(key, context["shape_names"])
            py_setitem(provided, key, pair[1])
    var keys = py_call(b.list, provided)
    for i in range(len(keys)):  # driver-loop: over named input shape metadata
        var key = keys[i]
        _scalar(key, context["shape_names"])
        var name = py_call(b.str, key)
        var dims = provided[key]
        var dims_type = py_type(dims)
        var tuple_storage = False
        if py_type(dims_type) is b.type:
            var mro = py_call(b.type.__dict__["__mro__"].__get__, dims_type)
            tuple_storage = py_contains(mro, b.tuple)
            if tuple_storage:
                for j in range(len(mro)):  # driver-loop: over effective tuple iterator identity
                    var fields = py_call(b.type.__dict__["__dict__"].__get__, mro[j])
                    if py_contains(fields, "__iter__"):
                        if fields["__iter__"] is not b.tuple.__dict__["__iter__"]:
                            raise Error("TypeError: dimensions refuse Python iteration hooks")
                        break
        if tuple_storage:
            dims = py_call(b.tuple, py_call(b.tuple.__dict__["__iter__"], dims))
        else:
            if not _one_of(dims_type, context["shape_containers"]):
                raise Error("TypeError: dimensions require native sequence storage")
            dims = py_call(b.tuple, dims)
        py_call(names.append, name)
        py_call(flat.append, len(dims))
        for j in range(len(dims)):  # driver-loop: over native scalar axis conversion
            _scalar(dims[j], context["shape_scalars"])
            py_call(flat.append, graph_int(dims[j]))
    return Python.tuple(names, flat)


def _raise_object(category: PythonObject, exception: PythonObject) raises:
    # Public CPython exception API; the original class and metadata survive
    # the C entry without calling a Python exception constructor.
    ref cpy = Python().cpython()
    cpy.lib.get_function[NoneType]("PyErr_SetObject")(category._obj_ptr, exception._obj_ptr)
    var failure = stash_python_error()
    raise failure^


def _unsupported(source: PythonObject, problems: PythonObject, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var output = Python.list()
    var all_nodes = True
    for i in range(len(problems)):  # driver-loop: over native lowerer's refusal records
        var item = problems[i]
        py_call(output.append, Python.tuple(item[0], item[1], item[2]))
        all_nodes = all_nodes and item[1] is not None
    var noun = String("node") if all_nodes else String("item")
    if len(problems) != 1:
        noun += "s"
    var basename = py_call(source.rsplit, "/", 1)[-1]
    if len(basename) == 0:
        basename = source
    var message = String(len(problems)) + " " + noun + " of " + String(basename) + " cannot be imported:"
    for i in range(len(output)):  # driver-loop: over complete ordered refusal text
        var item = output[i]
        message += "\n  " + String(item[0])
        if item[1] is not None:
            message += " (" + String(item[1]) + ")"
        message += ": " + String(item[2])
    var category = context["unsupported"]
    var exception = py_call(b.ValueError.__new__, category, PythonObject(message))
    var fields = py_call(b.BaseException.__dict__["__dict__"].__get__, exception, category)
    py_setitem(fields, "source", source)
    py_setitem(fields, "problems", py_call(b.tuple, output))
    _raise_object(category, exception)


def _status(result: PythonObject, source: PythonObject, context: PythonObject) raises:
    var code = Int(py=result[0])
    if code == 0:
        return
    if code == 5:
        _unsupported(source, result[1], context)
    if code == 4:
        raise Error("ValueError: the model graph: " + String(result[1]))
    if code == 6:
        var status = Int(py=result[1])
        var text = String(result[2])
        if status == 1:
            raise Error("ValueError: " + String(source) + " is not an ONNX model: " + text)
        if status == 2:
            var b = native_loaded_module("builtins")
            var exception = py_call(b.FileNotFoundError, 2, "No such file or directory", source) if Int(py=result[3]) != 0 else py_call(b.OSError, String(source) + ": " + text)
            _raise_object(py_type(exception), exception)
        if status == 3 or status == 4:
            raise Error("ValueError: " + String(source) + ": " + text)
    raise Error("RuntimeError: the model import has another result layout; rebuild the x_sequence bindings")


