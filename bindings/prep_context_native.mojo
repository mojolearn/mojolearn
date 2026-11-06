# SPDX-License-Identifier: Apache-2.0
# Stateless preprocessing admits and snapshots its native provider context
# before lazy NumPy setup, dispatch, metadata access, or numerical execution.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.classical_attributes import classical_connect_numpy


def _exact_dict(mapping: PythonObject) raises:
    if py_type(mapping) is not native_loaded_module("builtins").dict:
        raise Error("TypeError: preprocessing provider metadata requires an exact dictionary")


def _copy_records(records: PythonObject) raises -> PythonObject:
    # The operation's copy is built from the immutable import-time records
    # alone (one C-level dict construction), so a later change to the import
    # context can never reach it. The former per-key walk of the source
    # context (key admission plus an identity test per key, every operation)
    # only re-proved that, and is gone.
    var b = native_loaded_module("builtins")
    if py_type(records) is not b.tuple:
        raise Error("TypeError: preprocessing provider identity records must remain immutable")
    return py_call(b.dict, records)


def _fixed_maps(destination: PythonObject, records: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(records) is not b.tuple:
        raise Error("TypeError: preprocessing fixed provider maps must remain immutable")
    for i in range(len(records)):  # driver-loop: fixed class/descriptor/dtype-code maps
        var record = records[i]
        if py_type(record) is not b.tuple or len(record) != 3 or py_type(record[0]) is not b.str:
            raise Error("TypeError: preprocessing provider map identity is invalid")
        py_setitem(destination, record[0], _copy_records(record[2]))


# The context keys classical_connect_numpy publishes. The first operation
# after NumPy is loaded connects them on its copy; later operations reuse
# them instead of rebuilding about forty NumPy dtypes per call.
comptime _NUMPY_CACHE = "prep_operation_numpy"


def _numpy_names() raises -> PythonObject:
    return Python.tuple("ndarray", "scalar_type", "integer", "scalar_types", "integer_types",
                        "native_value_types", "buffer_owner_types", "numeric_dtypes", "class_dtypes",
                        "archive_npy_dtypes", "float_types", "int_types", "bool_types", "float32",
                        "float64", "int32", "int64", "uint64", "bool")


def _connect_numpy_once(source: PythonObject, context: PythonObject) raises:
    if py_get(context, "ndarray") is not None:
        return
    var cached = py_get(source, _NUMPY_CACHE)
    if cached is not None:
        _ = py_call(context.update, cached)
        return
    classical_connect_numpy(context)
    if py_get(context, "ndarray") is None:
        return  # NumPy is not loaded; the next operation looks again
    var names = _numpy_names()
    var kept = Python.dict()
    for i in range(len(names)):  # small-loop(the fixed NumPy admission key names)
        py_setitem(kept, names[i], context[names[i]])
    # A new key of the import context, outside the pinned records: the
    # records (and so every operation copy) are unchanged by it.
    py_setitem(source, _NUMPY_CACHE, kept)


def prep_operation_context(source: PythonObject) raises -> PythonObject:
    # native_loaded_module verifies the bridge module and the exact immutable
    # CONNECTION export captured at import. Replacing a context's own alleged
    # identity table cannot authorize callbacks or new descriptor references.
    var b = native_loaded_module("builtins")
    var connection = native_loaded_module("mojolearn._native_prep_stateless_bridge").CONTEXT_CONNECTION
    if py_type(connection) is not b.tuple or len(connection) != 6 or source is not connection[0]:
        raise Error("TypeError: preprocessing requires its originally connected native provider context")
    var context = _copy_records(connection[1])
    _fixed_maps(context, connection[2])
    var output = _copy_records(connection[4])
    _fixed_maps(output, connection[5])
    # The process mode and loaded-module namespace are mutable process data:
    # copy each namespace before a readback or input preparation can release
    # the GIL.
    var process = context["backend_fields"]
    var modules = context["modules"]
    _exact_dict(process)
    _exact_dict(modules)
    # Both namespaces (the backend module's and sys.modules, ~1000 keys with
    # NumPy loaded) are copied in C; their keys are no longer walked one by
    # one per operation. Every other native family reads them directly.
    var process_copy = py_call(b.dict, process)
    var modules_copy = py_call(b.dict, modules)
    py_setitem(context, "backend_fields", process_copy)
    py_setitem(context, "modules", modules_copy)
    py_setitem(output, "backend_fields", process_copy)
    py_setitem(output, "modules", modules_copy)
    py_setitem(context, "output_context", output)
    _connect_numpy_once(source, context)
    return context
