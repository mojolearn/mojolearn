# SPDX-License-Identifier: Apache-2.0
# Native forest restoration and fitted-state metadata. State restoration keeps
# the original dictionary identity and checks displaced owners before mutation.
# The PACK half of the former `bindings/trees_public_state.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/trees_public_state_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.mode_metadata_native_door import _fields, _class_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_owner_lifetime_door import native_release_admit, native_release_weakrefs
from bindings.classical_attributes_door import classical_check_layout
from bindings.classifier_public_lifecycle_door import classifier_label_list
from bindings.trees_lifetime_door import TreesResident
from bindings.py_model_records import bump_model_generation


from bindings.trees_public_state_unpack import (
    _release_tree_values,
)


def forest_restore_state(model: PythonObject, state: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    # Generation word (bindings/py_model_records.mojo, lane ne1-e-records):
    # every stamped record of this object misses from here on.
    _ = bump_model_generation(model)
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
