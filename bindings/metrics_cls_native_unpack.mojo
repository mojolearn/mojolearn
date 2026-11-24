# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the metrics of python/mojolearn/_metrics_impl.py: one native call per public method,
# reached from bindings/family_native.mojo for every `family_mcls_*` operation (lane mcb2-mcls).
#
# Public shells pass a None binding slot. This owner selects the import-connected
# native module using the explicit mode (or the live default), then owns input
# admission, argument preparation, algorithm dispatch and finished results.
# The groups:
#   bindings/mcls_cluster.mojo   label-pair cluster metrics, entropy, kl_divergence, silhouette,
#                                trustworthiness, mean_squared/absolute/root_mean_squared_error, r2_score
#   bindings/mcls_classify_door.mojo  accuracy_score, confusion_matrix, precision/recall/f1_score, log_loss
#   bindings/mcls_ranking_door.mojo   roc_auc_score, precision_recall_curve
from bindings.operation_names_mcls import (
    MCLS_OPN_FAMILY_MCLS_ACCURACY,
    MCLS_OPN_FAMILY_MCLS_ACCURACY_FRACTION,
    MCLS_OPN_FAMILY_MCLS_CONFUSION,
    MCLS_OPN_FAMILY_MCLS_LOG_LOSS,
    MCLS_OPN_FAMILY_MCLS_PRF,
    MCLS_OPN_FAMILY_MCLS_PR_CURVE,
    MCLS_OPN_FAMILY_MCLS_R2,
    MCLS_OPN_FAMILY_MCLS_REGRESSION_ERROR,
    MCLS_OPN_FAMILY_MCLS_ROC_AUC,
)
from std.python import Python, PythonObject
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_pandas_door import classical_admit_value
from bindings.metrics_native_support_door import metric_arguments_to
from bindings.metrics_device_door import metric_device_admit, metric_device_identical
from core.function_device import FunctionDevice, function_device_default
from providers.python.function_device_boundary_door import function_device_scope
from providers.python.metric_binding_pyapi_pack import metric_binding_arguments
from bindings.mcls_cluster_door import mcls_cluster_operation
from bindings.mcls_classify_door import mcls_classify_operation
from bindings.mcls_ranking_door import mcls_ranking_operation


def metrics_cls_unknown_operation(operation: String) raises:
    """A `family_mcls_*` name the operation table does not hold (the shells
    pass table names only): the text the cluster owner's chain raised for it."""
    raise Error("RuntimeError: unknown family operation " + operation)


def metrics_cls_operation(model: PythonObject, operation: Int, var args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    """`operation` is the leaf code of bindings/operation_names_mcls.mojo:
    classify, ranking and cluster codes are three ranges. The native callers'
    form (a model's score: bindings/pls_native_door.mojo,
    bindings/xseq_score_native_door.mojo, bindings/trees_public_meta_door.mojo):
    no `device` item, the route of the caller's context."""
    var end = len(args)
    return _metrics_cls_run(operation, args^, end, function_device_default(), context)


def metrics_cls_public(operation: Int, var args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The family door's form (bindings/family_native_door.mojo): the public
    shell's tuple ends with its `device` (lane vdf-metrics), read FIRST, before
    any admission or work (bindings/metrics_device_door.mojo)."""
    var end = len(args) - 1
    if end < 0:
        raise Error("TypeError: a public metric call carries its device as the last argument")
    var device = metric_device_admit(args[end], "this mojolearn.metrics function", context)
    if device.host_beside:
        metric_device_identical(_mcls_mode(operation, args, end), context)
    return _metrics_cls_run(operation, args^, end, device, function_device_scope(device, context))


def _mcls_mode(operation: Int, args: PythonObject, end: Int) raises -> PythonObject:
    """The `numeric_mode` item of a public metric (explicit device="cpu" beside
    a GPU set only): the last item of the operations that carry one, None
    (the default mode) for the cluster label metrics, which carry none."""
    if (operation == MCLS_OPN_FAMILY_MCLS_ACCURACY or operation == MCLS_OPN_FAMILY_MCLS_R2
            or operation == MCLS_OPN_FAMILY_MCLS_REGRESSION_ERROR or operation == MCLS_OPN_FAMILY_MCLS_ROC_AUC
            or operation == MCLS_OPN_FAMILY_MCLS_PR_CURVE or operation == MCLS_OPN_FAMILY_MCLS_CONFUSION
            or operation == MCLS_OPN_FAMILY_MCLS_PRF or operation == MCLS_OPN_FAMILY_MCLS_LOG_LOSS
            or operation == MCLS_OPN_FAMILY_MCLS_ACCURACY_FRACTION):
        var item = args[end - 1]  # a local, never a returned subscript (AGENTS.md part 6)
        return item
    return PythonObject(None)


def _metrics_cls_run(operation: Int, var args: PythonObject, end: Int, device: FunctionDevice,
                     context: PythonObject) raises -> PythonObject:
    """`args[:end]` are the metric's own arguments; `context` is already the
    device's (`function_device_scope`)."""
    args = metric_binding_arguments(operation, args, end, device, context)
    classical_connect_numpy(context)
    # The first argument is a connected binding for these operations; all
    # remaining arguments are user data/configuration and must be admitted
    # before any truth, equality, length or scalar conversion can run.
    var first = 0 if operation == MCLS_OPN_FAMILY_MCLS_LOG_LOSS or operation == MCLS_OPN_FAMILY_MCLS_ACCURACY_FRACTION else 1
    for index in range(first, end):  # driver-loop: metric input owners and native option metadata
        classical_admit_value(args[index], context)
    args = metric_arguments_to(args, first, end, context)
    if operation <= MCLS_OPN_FAMILY_MCLS_LOG_LOSS:
        return mcls_classify_operation(operation, args, context)
    if operation <= MCLS_OPN_FAMILY_MCLS_PR_CURVE:
        return mcls_ranking_operation(operation, args, context)
    return mcls_cluster_operation(operation, args, context)
