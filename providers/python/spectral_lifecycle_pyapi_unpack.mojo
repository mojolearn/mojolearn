# SPDX-License-Identifier: Apache-2.0
# Whole estimator metadata/lifecycle boundary for the spectral public methods.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native_door import _fields, _class_fields
from bindings.classical_attributes_door import _type_mro
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_arrays_door import classical_kind
from bindings.classical_validate_unpack import classical_view, classical_estimator_name, RESHAPE_HINT
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.family_raw_attr_pack import family_raw_get, family_raw_get_or
from providers.python.classical_family_admission_pyapi_unpack import classical_family_admit
from providers.python.array_metadata_pyapi_door import array_shape_once


def spectral_input(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    # Preserve the former _sk_validate dense-only class contract without
    # reading a custom metaclass's __module__ attribute.
    var module = py_call(b.type.__dict__["__module__"].__get__, py_type(source))
    if py_type(module) is b.str and String(module).startswith("scipy.sparse"):
        raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array. (" + name + " takes dense X only.)")
    var view = classical_view(source, context, "X")
    var kind = classical_kind(view, context)
    if kind == "c":
        raise Error("ValueError: Complex data not supported: " + name + " got a complex X.")
    var shape = array_shape_once(view, context)
    if len(shape) < 2:
        raise Error("ValueError: Expected 2D array, got " + ("scalar" if len(shape) == 0 else "1D") + " array instead. " + RESHAPE_HINT)
    if len(shape) == 2 and shape[0] == 0:
        raise Error("ValueError: Found array with 0 sample(s) while a minimum of 1 is required by " + name + ".")
    if len(shape) == 2 and shape[1] == 0:
        raise Error("ValueError: Found array with 0 feature(s) while a minimum of 1 is required by " + name + ".")
    # Keep the original dtype: precomputed affinity's COO route consumes
    # float64 before its established conversion; validation must not narrow it.
    return view
