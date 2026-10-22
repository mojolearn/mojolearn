# SPDX-License-Identifier: Apache-2.0
# Public density constructors store options verbatim. Validation belongs to
# the existing fit/load/inference operations, not to construction.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_attributes import classical_check_layout, classical_connect_numpy, _descriptor_slot
from bindings.classical_pandas import _pd_type_fields
from bindings.host_logistic_native import _logistic_label_kind
from bindings.native_owner_lifetime import native_release_admit


def _density_option(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: density constructor option nesting exceeds the native bound")
    var cls = py_type(value)
    if cls is py_type(Python.list()) or cls is py_type(Python.tuple()):
        for i in range(len(value)):  # driver-loop: native constructor container admission
            _density_option(value[i], context, depth + 1)
        return
    if cls is py_type(Python.dict()):
        var pairs = py_call(py_type(Python.list()), value.items())
        for i in range(len(pairs)):  # driver-loop: native dictionary entries without user key comparisons
            _density_option(pairs[i][0], context, depth + 1)
            _density_option(pairs[i][1], context, depth + 1)
        return
    # The class-label scalar admission includes native complex/string NumPy
    # scalars and then the ordinary exact native buffer/container admission.
    # No conversion is requested: preserve each supplied option by identity.
    _ = _logistic_label_kind(value, context)


def _density_member(name: PythonObject, plan: PythonObject) raises -> PythonObject:
    var mro = plan["mro"]
    for i in range(len(mro)):  # driver-loop: native class assignment precedence
        var fields = _pd_type_fields(mro[i])
        if py_contains(fields, name):
            return Python.tuple(fields[name], True)
    return Python.tuple(PythonObject(None), False)


def _density_descriptor(name: PythonObject, member: PythonObject, present: Bool,
                        plan: PythonObject, context: PythonObject) raises -> Bool:
    if not present:
        return False
    var cls = py_type(member)
    var data = _descriptor_slot(cls, "__set__") or _descriptor_slot(cls, "__delete__")
    var getter = _descriptor_slot(cls, "__get__")
    if data or (String(name) == "numeric_mode" and getter):
        var binding = py_get(plan["bindings"], name)
        var known = binding is not None and binding[0] is member
        # Ordinary Python __slots__ uses CPython's fixed member descriptor;
        # arbitrary extension get/set descriptors have no audited callback
        # closure merely because their descriptor object is native.
        var native = cls is context["host_density_member_type"]
        # Plain functions bind through a native C descriptor; a mode method
        # counts as an existing attribute but is never called here.
        if not data and cls is context["host_density_function_type"]:
            native = True
        if not known or not native:
            raise Error("TypeError: density constructors do not invoke Python property or descriptor hooks")
    return data


def _density_get_slot(model: PythonObject, name: PythonObject, context: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    var attribute_error = context["host_density_attribute_error"]
    var matches = cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")
    var clear = cpy.lib.get_function[NoneType]("PyErr_Clear")
    var get = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetAttr")
    var result = get(model._obj_ptr, name._obj_ptr)
    if result:
        return Python.tuple(PythonObject(from_owned=result), True)
    if matches(attribute_error._obj_ptr) != 0:
        clear()
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
    var setter = PythonObject(None)
    for i in range(len(mro)):  # driver-loop: connected constructor ancestry and native assignment slot
        var kind = py_get(context["host_inference_kinds"], mro[i])
        known = known or (kind is not None and String(kind) == family)
        if setter is None:
            setter = py_get(_pd_type_fields(mro[i]), "__setattr__")
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
    for i in range(len(names)):  # driver-loop: publish admitted options through native dict/member slots
        if Bool(py=write[i]):
            if cpy.lib.get_function[Int32]("PyObject_GenericSetAttr")(model._obj_ptr, names[i]._obj_ptr, args[i + 1]._obj_ptr) != 0:
                raise cpy.unsafe_get_error()
    return PythonObject(None)
