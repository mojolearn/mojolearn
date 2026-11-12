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
from bindings.cpython_table import cpython_table


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
