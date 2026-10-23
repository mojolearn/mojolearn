# SPDX-License-Identifier: Apache-2.0
# The compiled owner of GaussianProcessRegressor/Classifier public methods and the parallel_gaussian_process functions (python/mojolearn/_gp_impl.py, _gpc_impl.py, parallel_gaussian_process.py): one native call per public method,
# reached from bindings/family_native.mojo for every `family_gp_*` operation.
from std.python import Python, PythonObject


def gp_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown family operation " + operation)
