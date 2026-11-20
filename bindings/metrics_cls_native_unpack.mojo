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
    MCLS_OPN_FAMILY_MCLS_ACCURACY_FRACTION,
    MCLS_OPN_FAMILY_MCLS_LOG_LOSS,
    MCLS_OPN_FAMILY_MCLS_PR_CURVE,
)
from std.python import Python, PythonObject
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_pandas_door import classical_admit_value
from bindings.metrics_native_support_door import metric_arguments
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
    classify, ranking and cluster codes are three ranges."""
    args = metric_binding_arguments(operation, args, context)
    classical_connect_numpy(context)
    # The first argument is a connected binding for these operations; all
    # remaining arguments are user data/configuration and must be admitted
    # before any truth, equality, length or scalar conversion can run.
    var first = 0 if operation == MCLS_OPN_FAMILY_MCLS_LOG_LOSS or operation == MCLS_OPN_FAMILY_MCLS_ACCURACY_FRACTION else 1
    for index in range(first, len(args)):  # driver-loop: metric input owners and native option metadata
        classical_admit_value(args[index], context)
    args = metric_arguments(args, first, context)
    if operation <= MCLS_OPN_FAMILY_MCLS_LOG_LOSS:
        return mcls_classify_operation(operation, args, context)
    if operation <= MCLS_OPN_FAMILY_MCLS_PR_CURVE:
        return mcls_ranking_operation(operation, args, context)
    return mcls_cluster_operation(operation, args, context)
