# SPDX-License-Identifier: Apache-2.0
# PCA fitting keeps admitted model references separately from its immutable
# operation setting references. Each consumed scalar is decoded once before
# the solver, at its original refusal point. Native scalar conversion preserves NumPy's public
# parsing/version/warning contract through the shared scalar provider.
# The PACK half of the former `providers/python/pca_fit_model_pyapi.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/pca_fit_model_pyapi_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.sequential_admission_door import sequential_admit_value
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from providers.python.pca_fit_scalar_pyapi_unpack import PCAFitScalar, pca_fit_scalar_admit
from bindings.pca_descriptor_boundary_door import pca_connection, pca_mode_name


def pca_fit_fields(state: PythonObject, operation: String, context: PythonObject) raises -> PythonObject:
    var fields = py_call(py_type(Python.dict()).__dict__["copy"], state)
    var owners = Python.list()
    var names = context["pca_parameters"] + context["pca_state"]
    # Row N262: one module view for the field admissions of this call, not
    # one NativeModules per value; a scalar admission (NumPy's scalar
    # protocol) is followed by released(), so the next value re-checks.
    var mods = NativeModules()
    for index in range(len(names)):  # driver-loop: admitted references retained before settings are replaced
        if py_contains(fields, names[index]):
            if index < len(context["pca_parameters"]):
                pca_fit_scalar_admit(fields[names[index]], context)
                mods.released()
            else:
                sequential_admit_value(mods, fields[names[index]], context)
            _ = py_call(owners.append, fields[names[index]])
    var b = native_loaded_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("TypeError: numeric_mode must be identical, fast or None")
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "_pca_fit_owners", owners)
    return fields
