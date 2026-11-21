# SPDX-License-Identifier: Apache-2.0
# OneClassSVM `fit`, `score_samples`, `decision_function`, `predict` and
# `fit_predict` (python/mojolearn/_expansion_neighbors.py), one native call
# each, in three parts:
#   UNPACK  bindings/svc_ocsvm_boundary_unpack.mojo: the binding of the
#           call, every parameter check of the Python methods, the admitted
#           buffers and their owners, the typed requests;
#   RUN     svm/svc_ocsvm_native_run.mojo over core/ocsvm_plan.mojo, the GIL
#           released: the x_neighbors primitives the Python `_XNeighbors`
#           methods ran (`xn_pos_compact`, `xn_take_rows`, `xn_take_cols`,
#           `xn_variance`, `xn_kernel`, `xn_ocsvm`,
#           `x_neighbors_ocsvm_alpha_init`, the resident solve
#           `x_neighbors_ocsvm_resident[_idn]`, `xn_kernel_matmul`,
#           `xn_matmul`, `xn_unary`, `xn_p2m_sign_label`) with the same
#           buffers, the same integer/float words and the same launch order;
#   PACK    bindings/svc_ocsvm_boundary_pack.mojo: the public arrays and the
#           fitted fields.
# This file only sequences them. Private scratch is Mojo-owned; public
# arrays remain at the boundary. Numerical implementations are unchanged.
# The binding is the estimator's (its binding route); every entry is asked
# of that module: no call here asks which device the process runs on.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.ocsvm_provider_unpack import OcConnection
from core.ocsvm_plan import oc_take_rows_into
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.labels_descriptor_boundary_door import labels_scope
from svm.svc_ocsvm_native_run import OcFitRun, oc_fit_run, oc_query_run, oc_decision_run, oc_labels_run
from bindings.svc_ocsvm_boundary_unpack import oc_binding_unpack, oc_fit_unpack, oc_query_unpack
from bindings.svc_ocsvm_boundary_pack import oc_fit_outputs, oc_fit_pack, oc_public_vector
from bindings.operation_names_svc import (
    SVC_OPN_FAMILY_SVC_OCSVM_FIT, SVC_OPN_FAMILY_SVC_OCSVM_SCORE_SAMPLES, SVC_OPN_FAMILY_SVC_OCSVM_DECISION,
    SVC_OPN_FAMILY_SVC_OCSVM_PREDICT, SVC_OPN_FAMILY_SVC_OCSVM_FIT_PREDICT,
)


def _ocsvm_fit(model: PythonObject, binding: PythonObject, source: PythonObject, sample_weight: PythonObject,
               context: PythonObject, mut connection: OcConnection) raises:
    var unpacked = oc_fit_unpack(model, binding, source, sample_weight, context, connection)
    var run = OcFitRun()
    with GILReleased(Python()):
        run = oc_fit_run(unpacked.driver, unpacked.resident, unpacked.request)
    # The weight-row gather is a label step on the estimator's device (labels_scope).
    var outputs = oc_fit_outputs(unpacked, run, labels_scope(model, context))
    if not unpacked.request.precomputed:
        # The support vectors: the support rows of X, gathered into the public array.
        with GILReleased(Python()):
            oc_take_rows_into(unpacked.driver, unpacked.request.x, unpacked.request.rows, unpacked.request.features,
                              outputs.support_address, outputs.supports, outputs.vectors_address)
    oc_fit_pack(unpacked, run, outputs, context)
    # The owners of X, the weights and the public arrays outlive RUN and PACK.
    _ = outputs
    _ = run
    _ = unpacked


def _ocsvm_scores(model: PythonObject, binding: PythonObject, source: PythonObject,
                  context: PythonObject, mut connection: OcConnection) raises -> PythonObject:
    var unpacked = oc_query_unpack(model, binding, source, False, False, context, connection)
    var result = oc_public_vector(unpacked.request.rows, "<f4", context)
    with GILReleased(Python()):
        oc_query_run(unpacked.driver, unpacked.request, result.address)
    _ = unpacked
    return result.value


def _ocsvm_decision(model: PythonObject, binding: PythonObject, source: PythonObject,
                    context: PythonObject, mut connection: OcConnection) raises -> PythonObject:
    var unpacked = oc_query_unpack(model, binding, source, False, True, context, connection)
    var result = oc_public_vector(unpacked.request.rows, "<f4", context)
    with GILReleased(Python()):
        oc_decision_run(unpacked.driver, unpacked.request, result.address)
    _ = unpacked
    return result.value


def _ocsvm_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                   context: PythonObject, mut connection: OcConnection) raises -> PythonObject:
    var unpacked = oc_query_unpack(model, binding, source, True, True, context, connection)
    var result = oc_public_vector(unpacked.request.rows, "<i8", context)
    with GILReleased(Python()):
        oc_labels_run(unpacked.driver, unpacked.request, result.address)
    _ = unpacked
    return result.value


def svc_ocsvm_operation(model: PythonObject, operation: Int, args: PythonObject,
    context: PythonObject) raises -> PythonObject:
    var binding = oc_binding_unpack(model, args[0], context)
    classical_connect_numpy(context)
    var connection = OcConnection(binding)
    try:
        if operation == SVC_OPN_FAMILY_SVC_OCSVM_FIT:
            _ocsvm_fit(model, binding, args[1], args[2], context, connection)
            return model
        if operation == SVC_OPN_FAMILY_SVC_OCSVM_SCORE_SAMPLES:
            return _ocsvm_scores(model, binding, args[1], context, connection)
        if operation == SVC_OPN_FAMILY_SVC_OCSVM_DECISION:
            return _ocsvm_decision(model, binding, args[1], context, connection)
        if operation == SVC_OPN_FAMILY_SVC_OCSVM_PREDICT:
            return _ocsvm_predict(model, binding, args[1], context, connection)
        if operation == SVC_OPN_FAMILY_SVC_OCSVM_FIT_PREDICT:
            _ocsvm_fit(model, binding, args[1], PythonObject(None), context, connection)
            return _ocsvm_predict(model, binding, args[1], context, connection)
        raise Error("RuntimeError: unknown OneClassSVM operation " + String(operation))
    finally:
        _ = connection
        _ = args
        _ = binding
