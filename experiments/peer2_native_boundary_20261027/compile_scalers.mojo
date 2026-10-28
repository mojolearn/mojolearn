# SPDX-License-Identifier: Apache-2.0
# Focused public scaler entries; compile status is independent of PLS.
from std.python._cpython import PyObjectPtr
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, raise_python_exception
from bindings.scaler_native import scaler_fit_c
from bindings.scaler_lifecycle_native import scaler_lifecycle_c
from bindings.minmax_native import minmax_lifecycle_c
from providers.python.preprocessing_input_pyapi import register_scaler_snapshot


@export
def peer2_scaler_fit(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return scaler_fit_c(py_self, args_ptr)


@export
def peer2_scaler_lifecycle(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return scaler_lifecycle_c(py_self, args_ptr)


@export
def peer2_minmax_lifecycle(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    return minmax_lifecycle_c(py_self, args_ptr)


@export
def PyInit_peer2_scalers() abi("C") -> PythonObject:
    # The production builder invokes this same registration helper. Keeping
    # the affected entries together covers type publication during compilation.
    try:
        var module = PythonModuleBuilder("peer2_scalers")
        register_scaler_snapshot(module)
        module.def_py_c_function(scaler_fit_c, "standard_scaler_fit_native")
        module.def_py_c_function(scaler_lifecycle_c, "standard_scaler_native_call")
        module.def_py_c_function(minmax_lifecycle_c, "minmax_scaler_native_call")
        return module.finalize()
    except error:
        _ = raise_python_exception(error)
        return PythonObject(None)
