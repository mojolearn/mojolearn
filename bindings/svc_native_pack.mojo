# SPDX-License-Identifier: Apache-2.0
# The compiled owner of SVC, SVR, LinearSVC/LinearSVR and OneClassSVM public
# methods (python/mojolearn/_svm_impl.py, svm.py, _expansion_neighbors.py
# OneClassSVM): one native call per public method, reached from
# bindings/family_native.mojo for every `family_svc_*` operation. The linear
# machines are bindings/svc_linear_native_door.mojo, the kernel machines
# bindings/svc_kernel_native.mojo, OneClassSVM bindings/svc_ocsvm_native_door.mojo.
# The PACK half of the former `bindings/svc_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/svc_native_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.svc_linear_native_door import svc_linear_operation
from bindings.svc_kernel_native_door import svc_kernel_operation
from bindings.svc_ocsvm_native_door import svc_ocsvm_operation
from providers.python.svc_public_state_pyapi_pack import svc_public_state_operation
from providers.python.family_public_lifecycle_pyapi_door import family_lifecycle_begin, family_lifecycle_finish
from providers.python.family_public_lifecycle_run import FamilyLifecycle
from bindings.operation_names_svc import (
    SVC_OPN_FAMILY_SVC_LINEAR_SVC_FIT, SVC_OPN_FAMILY_SVC_LINEAR_SVC_PREDICT, SVC_OPN_FAMILY_SVC_LINEAR_SVC_SCORE,
    SVC_OPN_FAMILY_SVC_LINEAR_SVR_FIT, SVC_OPN_FAMILY_SVC_LINEAR_SVR_SCORE, SVC_OPN_FAMILY_SVC_OCSVM_DECISION,
    SVC_OPN_FAMILY_SVC_OCSVM_FIT, SVC_OPN_FAMILY_SVC_OCSVM_FIT_PREDICT, SVC_OPN_FAMILY_SVC_OCSVM_PREDICT,
    SVC_OPN_FAMILY_SVC_OCSVM_SCORE_SAMPLES,
)
from providers.python.neighbors_public_lifecycle_pyapi_door import neighbors_public_end, neighbors_public_refusal
from bindings.gp_svm_gp_state_door import gp_x_mark, gp_x_release
from bindings.host_sparse_native_door import host_sparse_format
from bindings.py_optional import py_present


from bindings.svc_native_unpack import (
    _svc_operation_route,
)


def svc_unknown_operation(operation: String) raises:
    """A `family_svc_*` name the operation table does not hold (the shells
    pass table names only): the text the route raised for it."""
    if operation.startswith("family_svc_public_"):
        raise Error("RuntimeError: unknown SVM state operation")
    if operation.startswith("family_svc_linear_"):
        raise Error("RuntimeError: unknown linear SVM operation " + operation)
    if operation.startswith("family_svc_svc_") or operation.startswith("family_svc_svr_"):
        raise Error("RuntimeError: unknown kernel SVM operation " + operation)
    if operation.startswith("family_svc_ocsvm_"):
        raise Error("RuntimeError: unknown OneClassSVM operation " + operation)
    raise Error("RuntimeError: unknown family operation " + operation)


def _svc_lifecycle(operation: Int) -> FamilyLifecycle:
    """The lifecycle of OneClassSVM and LinearSVC/LinearSVR (lane bw2-svm:
    moved from the shared name chain of
    providers/python/family_public_lifecycle_run.mojo, same flags; the kernel
    machines and the pickle state run outside it)."""
    # OneClassSVM (bindings/svc_ocsvm_native_door.mojo). sample_weight is not a target.
    if operation == SVC_OPN_FAMILY_SVC_OCSVM_FIT:
        return FamilyLifecycle("fit", scan=True, keeps_x=False)
    if operation == SVC_OPN_FAMILY_SVC_OCSVM_FIT_PREDICT:
        return FamilyLifecycle("fit_predict", scan=True, keeps_x=False)
    if operation == SVC_OPN_FAMILY_SVC_OCSVM_SCORE_SAMPLES:
        return FamilyLifecycle("score_samples", scan=True, widen=True)
    if operation == SVC_OPN_FAMILY_SVC_OCSVM_DECISION:
        return FamilyLifecycle("decision_function", scan=True, widen=True)
    if operation == SVC_OPN_FAMILY_SVC_OCSVM_PREDICT:
        return FamilyLifecycle("predict", scan=True)
    # LinearSVC / LinearSVR (bindings/svc_linear_native_door.mojo).
    if operation == SVC_OPN_FAMILY_SVC_LINEAR_SVC_FIT:
        return FamilyLifecycle("fit", scan=True, replace=False, target=2, classifier=True, required=True)
    if operation == SVC_OPN_FAMILY_SVC_LINEAR_SVC_PREDICT:
        return FamilyLifecycle("predict", scan=True, replace=False)
    if operation == SVC_OPN_FAMILY_SVC_LINEAR_SVC_SCORE:
        return FamilyLifecycle("score", scan=True, replace=False)
    if operation == SVC_OPN_FAMILY_SVC_LINEAR_SVR_FIT:
        return FamilyLifecycle("fit", scan=True, replace=False, target=2, required=True)
    if operation == SVC_OPN_FAMILY_SVC_LINEAR_SVR_SCORE:
        return FamilyLifecycle("score", scan=True, replace=False)
    return FamilyLifecycle("")


def svc_operation(model: PythonObject, operation: Int, args: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """The public operation: the estimators that formerly took the `_mode`
    class-creation wrappers run their whole lifecycle around the owner
    (providers/python/family_public_lifecycle_pyapi.mojo); every other
    operation goes straight to the owner. `operation` is the leaf code of
    bindings/operation_names_svc.mojo."""
    var lifecycle = _svc_lifecycle(operation)
    if lifecycle.method.byte_length() == 0:
        return _svc_operation_route(model, operation, args, context)
    # The lifecycle's operation parameter is not read (its text is the spec's method).
    var call = family_lifecycle_begin(model, "", lifecycle, args, 1, context)
    # Perf (lane perf3-cluster-gp-gate; perf2-gp-svm row 20): OneClassSVM's
    # door replaced a dense X by its admitted C-order float32 snapshot
    # (family_lifecycle_begin: rank, emptiness and finite scan). Register it
    # for this operation (the GP door's registry, bindings/gp_svm_gp_state.mojo)
    # so bindings/svc_ocsvm_native_door.mojo does not snapshot it again. A sparse X
    # is not replaced and not registered; the registry also takes only an
    # exact ndarray or Array.
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var marked = Optional[PythonObject]()
    if (operation >= SVC_OPN_FAMILY_SVC_OCSVM_DECISION and operation <= SVC_OPN_FAMILY_SVC_OCSVM_SCORE_SAMPLES
            and lifecycle.replace):
        if host_sparse_format(args[1]).byte_length() == 0:
            marked = py_present(call.args[1])
            gp_x_mark(marked.value())
    var result = PythonObject(None)
    try:
        result = _svc_operation_route(model, operation, call.args, context)
    except error:
        var failure = neighbors_public_refusal(error, call, model)
        raise failure^
    finally:
        if Bool(marked):
            gp_x_release(marked.value())
        neighbors_public_end(model, call)
    return family_lifecycle_finish(model, lifecycle, result, call, context)
