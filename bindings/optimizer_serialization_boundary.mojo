# SPDX-License-Identifier: Apache-2.0
"""Optimizer copy/pickle UNPACK/PACK; process-local handles never serialize."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_fields, py_get, py_setitem, py_type
from bindings.optimizer_admission import opt_owner, opt_value
from bindings.optimizer_owner_boundary import OptimizerOwner, OptimizerOwnerLease, optimizer_owner_new, optimizer_owner_retained, optimizer_owner_close, optimizer_owner_admit_replacement
from bindings.optimizer_public_door import OPT_DOOR, owner_kind
from bindings.optimizer_public_words import stash_python_error
from bindings.optimizer_scalar_boundary import optimizer_truth
from training.optimizer_owner_control import optimizer_restored_pristine


def optimizer_serialized_field(name: String) -> Bool:
    # Old state dictionaries may contain these retired control mirrors. They
    # are accepted only as external serialization input, never retained state.
    return name != "_native_optimizer_owner" and name != "_native_optimizer_pristine" and name != "_res" and name != "_res_binding" and name != "_res_host" and name != "_host_fresh" and name != "_pristine" and name != "_p_dev" and name != "_g_dev" and name != "_where" and name != "_KIND"


def optimizer_restore_state(owner: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    opt_owner(owner, context)
    opt_value(state, context)
    if py_type(state) is not context["container_types"][2]:
        raise Error("TypeError: optimizer serialized state requires a native dictionary")
    var pristine_value = py_get(state, "_native_optimizer_pristine", py_get(state, "_pristine", False))
    var pristine = optimizer_restored_pristine(optimizer_truth(pristine_value, context),
        py_get(state, "_m") is not None, py_get(state, "_v") is not None)
    var fields = py_fields(owner)
    optimizer_owner_admit_replacement(owner, fields, context)
    var table = cpython_table()
    var door = OPT_DOOR.get_or_create_ptr()
    var saved_pointer = table[].PyDict_Copy(fields._obj_ptr)
    if not saved_pointer:
        var failure = stash_python_error()
        raise failure^
    var saved = PythonObject(from_owned=saved_pointer)
    var replacement = optimizer_owner_new(owner, owner_kind(owner), pristine)
    var publication = OptimizerOwnerLease(owner, replacement)
    var old = py_get(fields, "_native_optimizer_owner")
    if old is not None:
        _ = optimizer_owner_retained(old, owner)
        old.downcast_value_ptr[OptimizerOwner]()[].control.begin()
    try:
        door[].dict_clear(fields._obj_ptr)
        var position = List[Int](length=1, fill=0)
        var pair = List[PyObjectPtr](length=2, fill=PyObjectPtr())
        while table[].PyDict_Next(state._obj_ptr, Int(position.unsafe_ptr()),
                Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):  # driver-loop: admitted external serialized fields
            var key = PythonObject(from_borrowed=pair[0])
            if optimizer_serialized_field(String(key)):
                py_setitem(fields, key, PythonObject(from_borrowed=pair[1]))
        py_setitem(fields, "_native_optimizer_owner", replacement)
        opt_owner(owner, context)
        if old is not None:
            optimizer_owner_close(old, owner)
    except error:
        door[].dict_clear(fields._obj_ptr)
        if door[].dict_update(fields._obj_ptr, saved._obj_ptr) != 0:
            if old is not None:
                old.downcast_value_ptr[OptimizerOwner]()[].control.finish()
            var failure = stash_python_error()
            raise failure^
        if old is not None:
            old.downcast_value_ptr[OptimizerOwner]()[].control.finish()
        raise error^
    if old is not None:
        old.downcast_value_ptr[OptimizerOwner]()[].control.finish()
    _ = old
    _ = publication
    return PythonObject(None)
