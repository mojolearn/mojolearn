# SPDX-License-Identifier: Apache-2.0
# The compiled owner of GaussianProcessRegressor/Classifier public methods and the parallel_gaussian_process functions (python/mojolearn/_gp_impl.py, _gpc_impl.py, parallel_gaussian_process.py): one native call per public method,
# reached from bindings/family_native.mojo for every `family_gp_*` operation.
# The regressor's operations are bindings/gp_regressor_native.mojo's, the
# classifier's bindings/gp_classifier_native.mojo's.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from bindings.native_loaded_module import native_loaded_module
from providers.python.estimator_binding_pyapi import estimator_arguments, estimator_binding_mode
from bindings.gp_regressor_native import gp_regressor_operation
from bindings.gp_classifier_native import gp_classifier_operation
from bindings.gp_kernel_public_native import gp_kernel_operation


def gp_operation(model: PythonObject, operation: String, var args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    if operation.startswith("family_gp_kernel_"):
        return gp_kernel_operation(model, operation, args, context)
    # Thin public wrappers reserve None for native binding admission. Internal
    # host/archive/parallel owners retain their explicitly admitted modules.
    if operation != "family_gp_gpc_lml":
        args = estimator_arguments(model, "_mojolearn_gp", args, context)
    if operation == "family_gp_gpc_fit" or operation == "family_gp_gpc_predict" or operation == "family_gp_gpc_score":
        if args[1] is None:
            var base = estimator_binding_mode(PythonObject("identical"), "_mojolearn", context)
            var prepared = Python.list(args[0], base)
            for i in range(2, len(args)):  # driver-loop: fixed public argument envelope
                _ = py_call(prepared.append, args[i])
            args = py_call(native_loaded_module("builtins").tuple, prepared)
    if operation.startswith("family_gp_gpr_"):
        return gp_regressor_operation(model, operation, args, context)
    if operation.startswith("family_gp_gpc_"):
        return gp_classifier_operation(model, operation, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
