# SPDX-License-Identifier: Apache-2.0
# AutoARIMA owns ordinary native ARIMA instances as well as primitive buffers.
# Walk their exact storage before releasing old fitted groups; no destructor,
# weakref callback or user descriptor may execute inside state replacement.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.native_owner_lifetime import native_release_admit, native_release_weakrefs
from bindings.classical_attributes import classical_connect_numpy
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from providers.python.autoarima_lifecycle_pyapi import _fields


def autoarima_model_class(context: PythonObject) raises -> PythonObject:
    var cls = context["arima_class"]
    var b = native_loaded_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: AutoARIMA requires the connected ordinary ARIMA class")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var found_storage = False
    for i in range(len(mro)):  # driver-loop: over native model class lifecycle metadata
        var namespace = py_call(b.type.__dict__["__dict__"].__get__, mro[i])
        var keys = py_call(b.list, namespace)
        for j in range(len(keys)):  # driver-loop: over class keys before native lookups
            if py_type(keys[j]) is not b.str:
                raise Error("TypeError: ARIMA class metadata requires exact string keys")
        var storage = py_get(namespace, "__dict__")
        if not found_storage and storage is not None:
            if py_type(storage) is not py_type(b.memoryview.obj):
                raise Error("TypeError: ARIMA instance storage cannot invoke Python descriptors")
            found_storage = True
        if py_get(namespace, "__del__") is not None:
            raise Error("TypeError: AutoARIMA cannot own ARIMA models with Python finalizers")
    return cls


def autoarima_commit(fields: PythonObject, updates: PythonObject, context: PythonObject) raises:
    var ownership = context["tsa_owner_context"]
    classical_connect_numpy(ownership)
    var b = native_loaded_module("builtins")
    var names = py_call(b.list, updates)
    var todo = Python.list()
    var seen = Python.dict()
    for i in range(len(names)):  # driver-loop: over replaced model-state owner references
        var old = py_get(fields, names[i])
        if old is not updates[names[i]]:
            _ = py_call(todo.append, old)
    var cursor = 0
    while cursor < len(todo):  # driver-loop: over the exact retained owner graph, including fitted native ARIMA models
        var item = todo[cursor]
        cursor += 1
        var key = PythonObject(Int(item._obj_ptr))
        if py_contains(seen, key):
            continue
        py_setitem(seen, key, item)
        var cls = py_type(item)
        native_release_weakrefs(item, ownership)
        if cls is b.list or cls is b.tuple:
            for i in range(len(item)):  # driver-loop: over retained group ownership containers
                _ = py_call(todo.append, item[i])
        elif cls is b.dict:
            var pairs = py_call(b.list, item.items())
            for i in range(len(pairs)):  # driver-loop: over native dictionary owner references without key protocols
                _ = py_call(todo.append, pairs[i][0])
                _ = py_call(todo.append, pairs[i][1])
        elif cls is context["arima_class"]:
            _ = autoarima_model_class(context)
            _ = py_call(todo.append, _fields(item, cls))
        else:
            native_release_admit(item, ownership)
    for i in range(len(names)):  # driver-loop: over completely admitted finished state
        py_setitem(fields, names[i], updates[names[i]])
    _ = seen
    _ = todo
