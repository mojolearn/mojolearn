# SPDX-License-Identifier: Apache-2.0
# The compiled owner of GaussianProcessRegressor/Classifier public methods and the parallel_gaussian_process functions (python/mojolearn/_gp_impl.py, _gpc_impl.py, parallel_gaussian_process.py): one native call per public method,
# reached from bindings/family_native.mojo for every `family_gp_*` operation.
# The regressor's operations are bindings/gp_regressor_native.mojo's, the
# classifier's bindings/gp_classifier_native.mojo's.
from std.python import Python, PythonObject
from bindings.gp_regressor_native import gp_regressor_operation
from bindings.gp_classifier_native import gp_classifier_operation


def gp_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    if operation.startswith("family_gp_gpr_"):
        return gp_regressor_operation(model, operation, args, context)
    if operation.startswith("family_gp_gpc_"):
        return gp_classifier_operation(model, operation, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
