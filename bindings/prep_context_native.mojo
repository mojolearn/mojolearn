# SPDX-License-Identifier: Apache-2.0
# Stateless preprocessing admits and snapshots its native provider context
# before lazy NumPy setup, dispatch, metadata access, or numerical execution.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains


def _key(key: PythonObject, depth: Int = 0) raises:
    var b = native_loaded_module("builtins")
    var cls = py_type(key)
    if cls is b.str or cls is b.int or cls is b.type:
        return
    if cls is b.tuple and depth < 4:
        for i in range(len(key)):  # small-loop(native provider key tuple)
            _key(key[i], depth + 1)
        return
    raise Error("TypeError: preprocessing provider metadata requires native keys")


def _dictionary(mapping: PythonObject, strings: Bool) raises:
    var b = native_loaded_module("builtins")
    if py_type(mapping) is not b.dict:
        raise Error("TypeError: preprocessing provider metadata requires an exact dictionary")
    var keys = py_call(b.list, mapping)
    for i in range(len(keys)):  # driver-loop: keys before native dict lookup can compare an unadmitted object
        if strings:
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: preprocessing context keys must be exact strings")
        else:
            _key(keys[i])


def _copy_records(source: PythonObject, records: PythonObject, strings: Bool) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    _dictionary(source, strings)
    if py_type(records) is not b.tuple:
        raise Error("TypeError: preprocessing provider identity records must remain immutable")
    var result = Python.dict()
    for i in range(len(records)):  # driver-loop: fixed imported native provider references, copied once per operation
        var pair = records[i]
        if py_type(pair) is not b.tuple or len(pair) != 2:
            raise Error("TypeError: preprocessing provider identity record is invalid")
        var key = pair[0]
        if strings and py_type(key) is not b.str:
            raise Error("TypeError: preprocessing identity names must be exact strings")
        _key(key)
        if not py_contains(source, key) or source[key] is not pair[1]:
            raise Error("TypeError: preprocessing provider identity changed after import")
        py_setitem(result, key, pair[1])
    return result


def _fixed_maps(destination: PythonObject, records: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(records) is not b.tuple:
        raise Error("TypeError: preprocessing fixed provider maps must remain immutable")
    for i in range(len(records)):  # driver-loop: fixed class/descriptor/dtype-code maps
        var record = records[i]
        if py_type(record) is not b.tuple or len(record) != 3 or py_type(record[0]) is not b.str:
            raise Error("TypeError: preprocessing provider map identity is invalid")
        py_setitem(destination, record[0], _copy_records(record[1], record[2], False))


def prep_operation_context(source: PythonObject) raises -> PythonObject:
    # native_loaded_module verifies the bridge module and the exact immutable
    # CONNECTION export captured at import. Replacing a context's own alleged
    # identity table cannot authorize callbacks or new descriptor references.
    var b = native_loaded_module("builtins")
    var connection = native_loaded_module("mojolearn._native_prep_stateless_bridge").CONTEXT_CONNECTION
    if py_type(connection) is not b.tuple or len(connection) != 6 or source is not connection[0]:
        raise Error("TypeError: preprocessing requires its originally connected native provider context")
    var context = _copy_records(source, connection[1], True)
    _fixed_maps(context, connection[2])
    var output = _copy_records(connection[3], connection[4], True)
    _fixed_maps(output, connection[5])
    # The process mode and loaded-module namespace are mutable process data.
    # Admit their native keys, retain original values, and copy each namespace
    # before a readback or input preparation can release the GIL.
    var process = context["backend_fields"]
    var modules = context["modules"]
    _dictionary(process, True)
    _dictionary(modules, True)
    var process_copy = py_call(b.dict, process)
    var modules_copy = py_call(b.dict, modules)
    py_setitem(context, "backend_fields", process_copy)
    py_setitem(context, "modules", modules_copy)
    py_setitem(output, "backend_fields", process_copy)
    py_setitem(output, "modules", modules_copy)
    py_setitem(context, "output_context", output)
    return context
