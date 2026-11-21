# SPDX-License-Identifier: Apache-2.0
"""Linear archive class/format-metadata UNPACK and fitted-model PACK.

The typed schema, shape/refusal decisions and loaded scalar values live in
core.native_archive_linear. This module admits native class/NumPy metadata and
publishes the writable coefficient owner plus its native state pin.
"""
# The PACK half of the former `bindings/classical_archive_linear.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/classical_archive_linear_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.classical_archive_output_door import archive_numeric_pack
from bindings.classical_arrays_door import classical_numpy
from bindings.classical_archive_door import _archive_class_name_unpack
from providers.python.linear_state_boundary_door import linear_state_publish
from core.native_archive_read import NativeArchiveDecoded
from core.native_archive_linear import NativeLinearArchivePlan, native_archive_needs_float_text
from bindings.classical_keys_unpack import classical_keys


def archive_linear_pack(cls: PythonObject, plan: NativeLinearArchivePlan,
                         members: List[NativeArchiveDecoded], ridge: Bool, context: PythonObject) raises -> PythonObject:
    var coefficients = archive_numeric_pack(members[plan.coefficients], context)
    var model = native_storage_allocate(cls)
    ref cpy = Python().cpython()
    var raw = cpython_table()[].PyObject_GenericGetDict(model._obj_ptr, PyObjectPtr())
    if not raw:
        raise cpy.unsafe_get_error()
    var fields = PythonObject(from_owned=raw)
    var nk = classical_keys()
    py_setitem(fields, nk[].k_fit_intercept, plan.fit_intercept)
    py_setitem(fields, nk[].k_numeric_mode, PythonObject(plan.mode))
    # `device` is a constructor parameter, not model state: the file does not
    # carry it. A loaded public model takes the default; the explicit Host*
    # classes (mojolearn.host_model) run on the CPU by class and say so.
    py_setitem(fields, "device", "cpu" if Bool(py=py_get(context, nk[].k_classical_explicit_cpu, False)) else "gpu")
    py_setitem(fields, nk[].k_coef_, coefficients)
    py_setitem(fields, nk[].k_intercept_, plan.intercept)
    py_setitem(fields, nk[].k_n_features_in_, plan.features)
    if ridge:
        py_setitem(fields, nk[].k_alpha, plan.alpha)
        py_setitem(fields, nk[].k_solver, "auto")
        py_setitem(fields, nk[].k_normalize, False)
        py_setitem(fields, nk[].k_solver_, "eig")
    linear_state_publish(model, coefficients, PythonObject(None), plan.features, plan.intercept, 0.0, False)
    return model
