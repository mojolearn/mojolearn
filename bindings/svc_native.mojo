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
from providers.python.family_public_lifecycle_pyapi import family_lifecycle_spec, family_lifecycle_begin, family_lifecycle_finish
from providers.python.neighbors_public_lifecycle_pyapi import neighbors_public_end, neighbors_public_refusal


def _svc_operation_route(model: PythonObject, operation: String, args: PythonObject,
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


def svc_operation(model: PythonObject, operation: String, args: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """The public operation: the estimators that formerly took the `_mode`
    class-creation wrappers run their whole lifecycle around the owner
    (providers/python/family_public_lifecycle_pyapi.mojo); every other
    operation goes straight to the owner."""
    var lifecycle = family_lifecycle_spec(operation)
    if lifecycle.method.byte_length() == 0:
        return _svc_operation_route(model, operation, args, context)
    var call = family_lifecycle_begin(model, operation, lifecycle, args, 1, context)
    var result = PythonObject(None)
    try:
        result = _svc_operation_route(model, operation, call.args, context)
    except error:
        var failure = neighbors_public_refusal(error, call, model)
        raise failure^
    finally:
        neighbors_public_end(model, call)
    return family_lifecycle_finish(model, lifecycle, result, call, context)
