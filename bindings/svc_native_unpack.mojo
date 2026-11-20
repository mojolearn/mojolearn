# SPDX-License-Identifier: Apache-2.0
# The compiled owner of SVC, SVR, LinearSVC/LinearSVR and OneClassSVM public
# methods (python/mojolearn/_svm_impl.py, svm.py, _expansion_neighbors.py
# OneClassSVM): one native call per public method, reached from
# bindings/family_native.mojo for every `family_svc_*` operation. The linear
# machines are bindings/svc_linear_native_door.mojo, the kernel machines
# bindings/svc_kernel_native.mojo, OneClassSVM bindings/svc_ocsvm_native_door.mojo.
from std.python import Python, PythonObject
from bindings.svc_linear_native_door import svc_linear_operation
from bindings.svc_kernel_native_door import svc_kernel_operation
from bindings.svc_ocsvm_native_door import svc_ocsvm_operation
from providers.python.svc_public_state_pyapi_pack import svc_public_state_operation
from providers.python.family_public_lifecycle_pyapi_door import family_lifecycle_begin, family_lifecycle_finish
from bindings.operation_names_svc import (
    SVC_OPN_FAMILY_SVC_PUBLIC_SVR_SETSTATE, SVC_OPN_FAMILY_SVC_LINEAR_SVR_SCORE, SVC_OPN_FAMILY_SVC_SVR_VALIDATE,
    SVC_OPN_FAMILY_SVC_OCSVM_SCORE_SAMPLES,
)
from providers.python.neighbors_public_lifecycle_pyapi_door import neighbors_public_end, neighbors_public_refusal
from bindings.gp_svm_gp_state_door import gp_x_mark, gp_x_release
from bindings.host_sparse_native_door import host_sparse_format
from bindings.py_optional import py_present


def _svc_operation_route(model: PythonObject, operation: Int, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """`operation` is the leaf code of bindings/operation_names_svc.mojo,
    numbered by sub-owner: each former prefix test is one range test."""
    if operation <= SVC_OPN_FAMILY_SVC_PUBLIC_SVR_SETSTATE:
        return svc_public_state_operation(model, operation, args, context)
    if operation <= SVC_OPN_FAMILY_SVC_LINEAR_SVR_SCORE:
        return svc_linear_operation(model, operation, args, context)
    if operation <= SVC_OPN_FAMILY_SVC_SVR_VALIDATE:
        return svc_kernel_operation(model, operation, args, context)
    if operation <= SVC_OPN_FAMILY_SVC_OCSVM_SCORE_SAMPLES:
        return svc_ocsvm_operation(model, operation, args, context)
    raise Error("RuntimeError: unknown family operation " + String(operation))
