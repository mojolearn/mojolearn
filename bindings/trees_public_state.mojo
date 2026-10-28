# SPDX-License-Identifier: Apache-2.0
# Native forest restoration and fitted-state metadata. State restoration keeps
# the original dictionary identity and checks displaced owners before mutation.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_owner_lifetime import native_release_admit, native_release_weakrefs
from bindings.classical_attributes import classical_check_layout
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.trees_lifetime import TreesResident


def _release_tree_values(values: PythonObject, retained: PythonObject,
                         context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var todo = py_call(b.list, values)
    var seen = Python.dict()
    # A zero-handle native owner has no release work. Its exact class excludes
    # Python subclasses before the supported checked Mojo downcast below.
    var prototype = PythonObject(alloc=TreesResident(PythonObject(None), 0))
    var cursor = 0
    while cursor < len(todo):  # driver-loop: cycle-safe ownership traversal before state replacement
        var item = todo[cursor]
        cursor += 1
        var identity = PythonObject(Int(item._obj_ptr))
        if py_contains(seen, identity):
            continue
        py_setitem(seen, identity, item)
        var kept = False
        for i in range(len(retained)):  # driver-loop: identical retained fields cannot destroy these owners
            kept = kept or item is retained[i]
        var pinned = context["trees_state_retained"]
        for i in range(len(pinned)):  # driver-loop: connected native binding tables have independent owners
            kept = kept or item is pinned[i]
        if kept:
            continue
        native_release_weakrefs(item, context)
        var cls = py_type(item)
        if cls is b.object:
            # Exact object() sentinels have neither Python finalizers nor fields.
            continue
        if cls is b.list or cls is b.tuple:
            for i in range(len(item)):  # driver-loop: native container references, no element protocols
                _ = py_call(todo.append, item[i])
        elif cls is b.dict:
            var pairs = py_call(b.list, item.items())
            for i in range(len(pairs)):  # driver-loop: follow retained dictionary owners without hashing caller keys
                _ = py_call(todo.append, pairs[i][0])
                _ = py_call(todo.append, pairs[i][1])
        elif cls is context["classifier_labels_class"]:
            # Validates the connected namespace/MRO; copies list storage via
            # native list.copy, without ClassLabels iteration or properties.
            _ = py_call(todo.append, classifier_label_list(item, context))
        elif cls is py_type(prototype):
            var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
            for i in range(len(ancestry)):  # driver-loop: reject injected Python finalizers before native owner release
                if py_contains(_class_fields(ancestry[i]), "__del__"):
                    raise Error("TypeError: forest restoration cannot release Python finalizers")
            # Documented std.python API validates the stored Mojo value type:
            # https://mojolang.org/docs/std/python/python_object/PythonObject/#downcast_value_ptr
            var owner = item.downcast_value_ptr[TreesResident]()
            if owner[].release is not None or owner[].handle > 0:
                var releases = context["trees_state_releases"]
                var connected = False
                for i in range(len(releases)):  # driver-loop: only the connected compiled forest release ABI
                    connected = connected or owner[].release is releases[i]
                if not connected or py_type(owner[].release) is not py_type(b.len):
                    raise Error("TypeError: forest restoration requires its connected native release entry")
        elif py_type(cls) is b.type and py_contains(context["trees_constructors"], cls):
            classical_check_layout(item, context)
            _ = py_call(todo.append, _fields(item))
        elif py_type(cls) is b.type and py_contains(context["trees_state_layouts"], cls):
            classical_check_layout(item, context["trees_state_layouts"][cls])
            _ = py_call(todo.append, _fields(item))
        else:
            native_release_admit(item, context)


def forest_restore_state(model: PythonObject, state: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(state) is not b.dict:
        raise Error("TypeError: forest state must be an ordinary dictionary")
    var keys = py_call(b.list, state)
    for i in range(len(keys)):  # driver-loop: validate native state keys before update/hash operations
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: forest state keys must be exact strings")
    var fields = _fields(model)
    var previous = py_call(b.dict, fields)
    var replacement = py_call(b.dict, previous)
    _ = py_call(replacement.update, state)
    # Preserve __setstate__'s update semantics, including unrelated old fields,
    # while resetting the process-local resident holder even if state supplied it.
    py_setitem(replacement, "_resident_holder", Python.dict())
    _release_tree_values(py_call(b.list, previous.values()),
                         py_call(b.list, replacement.values()), context)
    _ = py_call(fields.update, replacement)
    _ = previous
    return PythonObject(None)


def tree_is_fitted(model: PythonObject, forest: Bool) raises -> PythonObject:
    var fields = _fields(model)
    if forest:
        return PythonObject(py_contains(fields, "_offsets"))
    var learner = py_get(fields, "_learner_")
    if learner is None:
        return PythonObject(False)
    return PythonObject(py_get(_fields(learner), "model_") is not None)
