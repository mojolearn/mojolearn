# SPDX-License-Identifier: Apache-2.0
# Passive graph metadata uses the same compiled import boundary. The record
# protocol descriptors are native, including recursive repr/equality/hash;
# builtin container protocols therefore never reenter generated Python methods.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_owner import native_buffer_view
from bindings.native_json import native_path
from providers.python.autograd_import_metadata_pyapi import _fields, _raise_object
from providers.python.autograd_python_errors_pyapi import graph_python_error
from bindings.optimizer_public_words import optimizer_python_error, PYTHON_ERROR_PENDING


def _admit(value: PythonObject, context: PythonObject, mut seen: List[Int]) raises:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.str or cls is b.bytes or cls is b.bytearray or cls is b.int or cls is b.float or cls is b.bool or cls is b.memoryview:
        return
    if py_type(cls) is not b.type:
        raise Error("TypeError: graph metadata refuses Python metaclass protocols")
    var address = Int(value._obj_ptr)
    for known in seen:  # driver-loop: over metadata owners already checked in this operation
        if known == address:
            return
    seen.append(address)
    if py_contains(context["records"], cls):
        # Validate its native hooks now. Its own compiled protocol captures
        # its fields once when the enclosing builtin reaches that record.
        _ = _fields(value, context["layouts"][cls])
        return
    if cls is b.tuple or cls is b.list:
        for i in range(len(value)):  # driver-loop: over native sequence metadata
            _admit(value[i], context, seen)
        return
    if cls is b.dict:
        var keys = py_call(b.list, value)
        for i in range(len(keys)):  # driver-loop: over admitted native metadata mappings
            # Admit the key before the dictionary performs hash/equality.
            _admit(keys[i], context, seen)
            _admit(value[keys[i]], context, seen)
        return
    raise Error("TypeError: graph metadata protocols require native scalars, buffers, containers or graph records; Python hooks are unsupported")


def _record_values(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_type(value)
    var fields = _fields(value, context["layouts"][cls])
    var names = context["records"][cls]
    var values = Python.list()
    for i in range(len(names)):  # driver-loop: over one captured record's fields
        _ = py_call(values.append, fields[names[i]])
    return py_call(native_loaded_module("builtins").tuple, values)


def graph_record_bound(args: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if len(args) < 2 or py_type(args[0]) is not b.tuple or len(args[0]) != 2:
        raise Error("TypeError: graph protocol requires its bound native descriptor")
    if py_type(args[0][0]) is not b.str or py_type(args[0][1]) is not b.dict:
        raise Error("TypeError: graph protocol requires native operation and context metadata")
    var operation = String(py=args[0][0])
    var arity = 4 if operation == "__setattr__" else (3 if operation == "__eq__" or operation == "__delattr__" else 2)
    if len(args) != arity:
        raise Error("TypeError: invalid graph protocol argument count")
    var context = args[0][1]
    var value = args[1]
    var cls = py_type(value)
    if py_type(cls) is not b.type or not py_contains(context["records"], cls):
        raise Error("TypeError: graph metadata requires an exact connected record class")
    if operation == "__setattr__" or operation == "__delattr__":
        if py_type(args[2]) is not b.str:
            raise Error("TypeError: attribute name must be a native string")
        var verb = String("assign to" if operation == "__setattr__" else "delete")
        var exception = py_call(b.BaseException.__new__, context["frozen_error"], "cannot " + verb + " field " + String(py_call(b.repr, args[2])))
        _raise_object(context["frozen_error"], exception)
    if operation == "__eq__" and py_type(args[2]) is not cls:
        return b.NotImplemented
    var seen = List[Int]()
    var values = _record_values(value, context)
    _admit(values, context, seen)
    if operation == "__eq__":
        var other = _record_values(args[2], context)
        _admit(other, context, seen)
        # Both tuples contain only admitted native protocols. Nested records
        # dispatch to this compiled descriptor, never a Python dataclass hook.
        return values == other
    if operation == "__hash__":
        return py_call(b.hash, values)
    if operation == "__repr__":
        ref cpy = Python().cpython()
        var entered = cpy.lib.get_function[Int32]("Py_ReprEnter")(value._obj_ptr)
        if entered < 0:
            raise cpy.unsafe_get_error()
        if entered > 0:
            return PythonObject("...")
        try:
            var names = context["records"][cls]
            var text = String(context["record_names"][cls]) + "("
            for i in range(len(names)):  # driver-loop: over formatted native record fields
                if i > 0:
                    text += ", "
                text += String(names[i]) + "=" + String(py_call(b.repr, values[i]))
            text += ")"
            cpy.lib.get_function[NoneType]("Py_ReprLeave")(value._obj_ptr)
            return PythonObject(text)
        except:
            cpy.lib.get_function[NoneType]("Py_ReprLeave")(value._obj_ptr)
            raise
    raise Error("ValueError: unknown graph record protocol")


def graph_record_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    # Separate entry prevents a public tuple source from masquerading as a
    # bound graph record operation at the artifact-import door.
    try:
        return graph_record_bound(PythonObject(from_borrowed=packed)).steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        return graph_python_error(error)


def _message(source: PythonObject, problems: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var source_text = source
    if py_type(source) is not b.str:
        var cls = py_type(source)
        if source is None or cls is b.int or cls is b.float or cls is b.bool or cls is b.bytes:
            source_text = py_call(b.str, source)
        else:
            source_text = native_path(source)
    var seen = List[Int]()
    _admit(problems, context, seen)
    if py_type(problems) is not b.tuple and py_type(problems) is not b.list:
        raise Error("TypeError: problems require a native sequence")
    var nodes = True
    for i in range(len(problems)):  # driver-loop: over native refusal entry validation
        var row = problems[i]
        if (py_type(row) is not b.tuple and py_type(row) is not b.list) or len(row) != 3:
            raise Error("ValueError: refusal entries require name, operation and reason")
        nodes = nodes and row[1] is not None
    var noun = String("node" if nodes else "item")
    if len(problems) != 1:
        noun += "s"
    var basename = py_call(source_text.rsplit, "/", 1)[-1]
    if len(basename) == 0:
        basename = source_text
    var text = String(len(problems)) + " " + noun + " of " + String(basename) + " cannot be imported:"
    for i in range(len(problems)):  # driver-loop: over complete ordered native refusal messages
        var row = problems[i]
        text += "\n  " + String(py_call(b.str, row[0]))
        if row[1] is not None:
            text += " (" + String(py_call(b.str, row[1])) + ")"
        text += ": " + String(py_call(b.str, row[2]))
    return PythonObject(text)


def graph_metadata_operation(operation: String, value: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if operation == "record_descriptor":
        if py_type(value) is not context["method_type"] or value.__func__ is not context["record_native_entry"]:
            raise Error("TypeError: graph protocol descriptors require the connected compiled entry")
        ref cpy = Python().cpython()
        var descriptor = cpy.lib.get_function[PyObjectPtr]("PyInstanceMethod_New")(value._obj_ptr)
        if not descriptor:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=descriptor)
    if operation == "record_construct" or operation == "reader_construct":
        var cls = py_type(value)
        if py_type(cls) is not b.type or (cls is not context["reader"] and not py_contains(context["records"], cls)):
            raise Error("TypeError: graph construction requires a connected native record class")
        var fields = _fields(value, context["layouts"][cls])
        var names = Python.tuple("b", "h", "views") if operation == "reader_construct" else context["records"][cls]
        if len(options) != len(names):
            raise Error("TypeError: graph record arguments differ from its connected layout")
        # Construction stores references exactly as a dataclass does. It does
        # not evaluate user values; protocol operations admit them separately.
        for i in range(len(names)):  # driver-loop: over public record constructor fields
            var item = options[i]
            if operation == "reader_construct" and i == 2:
                var seen = List[Int]()
                _admit(item, context, seen)
                item = py_call(b.tuple, item)
            if cls is context["node"] and i == 5 and item is context["attrs_default"]:
                item = Python.dict()
            py_setitem(fields, names[i], item)
        return PythonObject(None)
    if operation == "unsupported_message":
        return _message(value, options, context)
    if operation == "unsupported_construct":
        if py_type(value) is not context["unsupported"]:
            raise Error("TypeError: UnsupportedModel requires its connected native class")
        var message = _message(options[0], options[1], context)
        var fields = py_call(b.BaseException.__dict__["__dict__"].__get__, value, context["unsupported"])
        py_setitem(fields, "source", options[0])
        py_setitem(fields, "problems", py_call(b.tuple, options[1]))
        _ = py_call(b.BaseException.__dict__["args"].__set__, value, Python.tuple(message))
        return PythonObject(None)
    if operation == "supported_ops":
        return py_call(b.list, context["supported_ops"])
    if operation == "dtype_name":
        var seen = List[Int]()
        _admit(value, context, seen)
        var found = py_get(context["dtype_names"], value)
        if found is None:
            found = py_get(context["refused_names"], value)
        return PythonObject("data type " + String(py_call(b.str, value))) if found is None else found
    if operation == "session_shape":
        var seen = List[Int]()
        _admit(value, context, seen)
        var shape = py_call(b.tuple, value)
        if len(shape) >= 2:
            return shape
        return Python.tuple(1, 1) if len(shape) == 0 else Python.tuple(1, shape[0])
    if operation == "buffer_address":
        if value is None:
            return PythonObject(0)
        if context["numpy"] is None or py_type(value) is not context["numpy"].ndarray:
            raise Error("TypeError: buffer_address requires exact native NumPy storage")
        var metadata = py_call(context["numpy_array_interface"], value)
        return metadata["data"][0]
    if operation == "bytes_view":
        if context["numpy"] is None:
            raise Error("ImportError: mojolearn.importers requires NumPy")
        var cls = py_type(value)
        if value is not None and cls is not b.bytes and cls is not b.bytearray and cls is not b.memoryview and cls is not context["array_type"] and cls is not context["numpy"].ndarray:
            raise Error("TypeError: bytes_view requires native buffer storage; Python buffer hooks are unsupported")
        var raw = native_buffer_view(py_call(b.bytes)) if value is None else native_buffer_view(value)
        raw = py_call(raw.cast, "B")
        if Int(py=raw.nbytes) == 0:
            return PythonObject(None)
        return py_call(context["numpy_frombuffer"], raw, context["numpy_uint8"])
    raise Error("ValueError: unknown graph metadata operation")
