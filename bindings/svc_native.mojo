# SPDX-License-Identifier: Apache-2.0
# The compiled owner of SVC, SVR, NuSVC/NuSVR, LinearSVC/LinearSVR and OneClassSVM public methods (python/mojolearn/_svm_impl.py, svm.py, _expansion_neighbors.py OneClassSVM): one native call per public method,
# reached from bindings/family_native.mojo for every `family_svc_*` operation.
from std.python import Python, PythonObject


def svc_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown family operation " + operation)
