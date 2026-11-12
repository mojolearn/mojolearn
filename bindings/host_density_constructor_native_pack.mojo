# SPDX-License-Identifier: Apache-2.0
# Public density constructors store options verbatim. Validation belongs to
# the existing fit/load/inference operations, not to construction.
# The PACK half of the former `bindings/host_density_constructor_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/host_density_constructor_native_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_attributes_door import classical_check_layout, classical_connect_numpy, _descriptor_slot
from bindings.classical_pandas_door import _pd_type_fields
from bindings.host_logistic_native_door import _logistic_label_kind
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.py_cpython_table import cpython_table


from bindings.host_density_constructor_native_unpack import (
    _density_descriptor,
    _density_option,
)


def _density_member(name: PythonObject, plan: PythonObject) raises -> PythonObject:
    var mro = plan["mro"]
    for i in range(len(mro)):  # driver-loop: native class assignment precedence
        var fields = _pd_type_fields(mro[i])
        if py_contains(fields, name):
            return Python.tuple(fields[name], True)
    return Python.tuple(PythonObject(None), False)


def _density_get_slot(model: PythonObject, name: PythonObject, context: PythonObject) raises -> PythonObject:
    # p3-neighbors-cluster: the process-wide resolved CPython tables, not a
    # by-name symbol lookup per slot read (PyErr_Clear only on a miss).
    ref cpy = Python().cpython()
    var attribute_error = context["host_density_attribute_error"]
    var result = cpython_table()[].PyObject_GenericGetAttr(model._obj_ptr, name._obj_ptr)
    if result:
        return Python.tuple(PythonObject(from_owned=result), True)
    if cpython_table()[].PyErr_ExceptionMatches(attribute_error._obj_ptr) != 0:
        cpython_table()[].PyErr_Clear()
        return Python.tuple(PythonObject(None), False)
    raise cpy.unsafe_get_error()


def host_density_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    classical_check_layout(model, context)
    var cls = py_type(model)
    var plan = context["classical_public_layouts"][cls]
    var fields = py_call(plan["fields_get"], model, cls)
    var family = String(args[0])
    var mro = plan["mro"]
    var known = False
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    var found_setter = Optional[PythonObject]()
    for i in range(len(mro)):  # driver-loop: connected constructor ancestry and native assignment slot
        var kind = py_get(context["host_inference_kinds"], mro[i])
        known = known or (kind is not None and String(kind) == family)
        if not found_setter:
            var candidate = py_get(_pd_type_fields(mro[i]), "__setattr__")
            if candidate is not None:
                found_setter = candidate
    var setter = PythonObject(None)
    if found_setter:
        setter = found_setter.value()
    if not known or (family != "kde" and family != "iforest"):
        raise Error("TypeError: density constructor requires its connected native host class")
    if setter is not context["host_density_object_setattr"]:
        raise Error("TypeError: density constructors do not invoke Python attribute assignment hooks")
    var names = Python.tuple("bandwidth", "kernel", "metric", "algorithm", "atol", "rtol", "breadth_first", "leaf_size", "metric_params", "numeric_mode") if family == "kde" else Python.tuple("n_estimators", "max_samples", "max_depth", "max_features", "bootstrap", "contamination", "random_state", "warm_start", "n_jobs", "verbose", "output_type", "numeric_mode")
    if len(args) != len(names) + 1:
        raise Error("TypeError: density constructor argument metadata differs")
    var write = Python.list()
    for i in range(len(names)):  # driver-loop: fixed constructor option schema
        var name = names[i]
        var replacement = args[i + 1]
        var pair = _density_member(name, plan)
        var member = pair[0]
        var present = Bool(py=pair[1])
        var data = _density_descriptor(name, member, present, plan, context)
        var previous = _density_get_slot(model, name, context) if data else Python.tuple(py_get(fields, name), py_contains(fields, name))
        # The old mode wrapper uses hasattr only when None was supplied.
        # A native unset slot is absent; an inherited ordinary value exists.
        if String(name) == "numeric_mode" and replacement is None and (Bool(py=previous[1]) or (present and not data)):
            _ = py_call(write.append, False)
            continue
        _density_option(replacement, context)
        if Bool(py=previous[1]) and previous[0] is not replacement:
            native_release_admit(previous[0], context)
        _ = py_call(write.append, True)
    ref cpy = Python().cpython()
    var set_attr = cpy.lib.get_function[Int32]("PyObject_GenericSetAttr")  # resolved once, not per option
    for i in range(len(names)):  # driver-loop: publish admitted options through native dict/member slots
        if Bool(py=write[i]):
            if set_attr(model._obj_ptr, names[i]._obj_ptr, args[i + 1]._obj_ptr) != 0:
                raise cpy.unsafe_get_error()
    return PythonObject(None)
