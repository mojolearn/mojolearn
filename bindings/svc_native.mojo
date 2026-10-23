# SPDX-License-Identifier: Apache-2.0
# The compiled owner of SVC, SVR, LinearSVC/LinearSVR and OneClassSVM public
# methods (python/mojolearn/_svm_impl.py, svm.py, _expansion_neighbors.py
# OneClassSVM): one native call per public method, reached from
# bindings/family_native.mojo for every `family_svc_*` operation. The linear
# machines are bindings/svc_linear_native.mojo.
from std.python import Python, PythonObject
from bindings.svc_linear_native import svc_linear_operation


def svc_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    if operation.startswith("family_svc_linear_"):
        return svc_linear_operation(model, operation, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
