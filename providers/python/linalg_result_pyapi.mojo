# SPDX-License-Identifier: Apache-2.0
# Native QRResult/SVDResult tuple construction and named field access.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_attributes import _type_mro
from bindings.mode_metadata_native import _class_fields
from bindings.native_loaded_module import NativeModules
from bindings.native_class_admission import class_record, class_admit
from bindings.cpython_table import cpython_table

# Per class state: the MRO row of the connected result owner (an int, so the
# record never retains the class). A type watcher drops it when the class or
# an ancestor changes, and the full ancestry and namespace check runs again
# (lane ne1-e-table, row N241). Python 3.10 / 3.11 keep the full check.
comptime _RESULT_SLOT = "linalg_result_owner_v1"


def _result_owner(cls: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: linalg results require native class metadata")
    var schemas = context["family_lin_result_schemas"]
    var mods = NativeModules()
    var known = class_record(mods, _RESULT_SLOT, cls)
    var ancestry = _type_mro(cls)
    if known is not None:
        var row = Int(py=known)
        if row >= 0 and row < len(ancestry):
            var recorded = py_get(schemas, ancestry[row])
            if recorded is not None:
                return recorded
    var schema = PythonObject(None)
    var start = -1
    for i in range(len(ancestry)):  # driver-loop: over result class ancestry to find the connected tuple owner
        if py_type(ancestry[i]) is not b.type:
            raise Error("TypeError: linalg result ancestry cannot invoke metaclass hooks")
        if schema is None:
            schema = py_get(schemas, ancestry[i])
            if schema is not None:
                start = i
    if schema is None:
        raise Error("TypeError: linalg results require QRResult or SVDResult")
    if len(ancestry) - start != len(schema[0]):
        raise Error("TypeError: linalg result ancestry changed after connection")
    for i in range(len(schema[0])):  # driver-loop: over the connected result owner ancestry
        if ancestry[start + i] is not schema[0][i]:
            raise Error("TypeError: linalg result ancestry changed after connection")
    var fields = _class_fields(ancestry[start])
    var names = py_call(b.list, fields)
    var expected = schema[1]
    if len(names) != len(expected):
        raise Error("TypeError: linalg result class metadata changed after connection")
    for i in range(len(names)):  # driver-loop: immutable connected class namespace identities
        if not py_contains(expected, names[i]) or fields[names[i]] is not expected[names[i]]:
            raise Error("TypeError: linalg result class metadata changed after connection")
    for i in range(start):  # driver-loop: passive subclasses may not replace tuple access/lifetime protocols
        var fields = _class_fields(ancestry[i])
        for hook in ["__getattribute__", "__getattr__", "__getitem__", "__iter__", "__len__", "__del__"]:  # small-loop(native result protocol names)
            if py_contains(fields, hook):
                raise Error("TypeError: linalg results cannot invoke Python tuple protocols")
    class_admit(mods, _RESULT_SLOT, cls, PythonObject(start))
    return schema


def linalg_result_new(cls: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var schema = _result_owner(cls, context)
    if py_type(values) is not b.tuple or len(values) != Int(py=schema[2]):
        raise Error("TypeError: linalg result constructor has an invalid item count")
    # Invoke the native tuple allocator directly. It does not invoke the
    # subclass's Python __new__, __init__ or iterable protocol.
    return py_call(b.tuple.__new__, cls, values)


def linalg_result_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(args) is not b.tuple or len(args) != 2:
        raise Error("TypeError: linalg result metadata requires its native argument tuple")
    if operation == "family_lin_result_new":
        return linalg_result_new(args[0], args[1], context)
    if operation != "family_lin_result_item":
        raise Error("RuntimeError: unknown linalg result operation")
    var value = args[0]
    _ = _result_owner(py_type(value), context)
    if py_type(args[1]) is not b.int:
        raise Error("TypeError: linalg result field index must be a native integer")
    var index = Int(py=args[1])
    ref cpy = Python().cpython()
    var table = cpython_table()
    var count = table[].PyTuple_Size(value._obj_ptr)
    if index < 0 or index >= count:
        raise Error("IndexError: tuple index out of range")
    var item = table[].PyTuple_GetItem(value._obj_ptr, index)
    if not item:
        raise cpy.unsafe_get_error()
    return PythonObject(from_borrowed=item)
