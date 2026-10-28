# SPDX-License-Identifier: Apache-2.0
# The compiled owner of SVC, SVR, LinearSVC/LinearSVR and OneClassSVM public
# methods (python/mojolearn/_svm_impl.py, svm.py, _expansion_neighbors.py
# OneClassSVM): one native call per public method, reached from
# bindings/family_native.mojo for every `family_svc_*` operation. The linear
# machines are bindings/svc_linear_native.mojo, the kernel machines
# bindings/svc_kernel_native.mojo, OneClassSVM bindings/svc_ocsvm_native.mojo.
from std.python import Python, PythonObject
from bindings.svc_linear_native import svc_linear_operation
from bindings.svc_kernel_native import svc_kernel_operation
from bindings.svc_ocsvm_native import svc_ocsvm_operation
from providers.python.svc_public_state_pyapi import svc_public_state_operation


def svc_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    if operation.startswith("family_svc_public_"):
        return svc_public_state_operation(model, operation, args, context)
    if operation.startswith("family_svc_linear_"):
        return svc_linear_operation(model, operation, args, context)
    if operation.startswith("family_svc_svc_") or operation.startswith("family_svc_svr_"):
        return svc_kernel_operation(model, operation, args, context)
    if operation.startswith("family_svc_ocsvm_"):
        return svc_ocsvm_operation(model, operation, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
