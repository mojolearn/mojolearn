# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the metrics of python/mojolearn/_metrics_impl.py: one native call per public method,
# reached from bindings/family_native.mojo for every `family_mcls_*` operation (lane mcb2-mcls).
#
# Each public function there is `_NATIVE(None, "family_mcls_<op>", (_get_binding(numeric_mode), ...), _NATIVE_CONTEXT)`:
# the shell resolves the metrics binding for the call's numeric mode (binding acquisition) and this owner
# does the rest: the argument and label checks in the order and wording the Python made them, the label
# encoding (bindings/mcls_common.mojo), the route choice between the cuML-backed entry of the metrics
# binding and the x_metrics program of the options, the entry or program itself and the result.
# The groups:
#   bindings/mcls_cluster.mojo   label-pair cluster metrics, entropy, kl_divergence, silhouette,
#                                trustworthiness, mean_squared/absolute/root_mean_squared_error, r2_score
#   bindings/mcls_classify.mojo  accuracy_score, confusion_matrix, precision/recall/f1_score, log_loss
#   bindings/mcls_ranking.mojo   roc_auc_score, precision_recall_curve
from std.python import Python, PythonObject
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.metrics_native_support import metric_arguments
from bindings.mcls_cluster import mcls_cluster_operation
from bindings.mcls_classify import mcls_classify_operation
from bindings.mcls_ranking import mcls_ranking_operation


def metrics_cls_operation(model: PythonObject, operation: String, var args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    # The first argument is a connected binding for these operations; all
    # remaining arguments are user data/configuration and must be admitted
    # before any truth, equality, length or scalar conversion can run.
    var first = 0 if operation == "family_mcls_log_loss" or operation == "family_mcls_accuracy_fraction" else 1
    for index in range(first, len(args)):  # driver-loop: metric input owners and native option metadata
        classical_admit_value(args[index], context)
    args = metric_arguments(args, first, context)
    if (operation == "family_mcls_accuracy" or operation == "family_mcls_accuracy_fraction" or operation == "family_mcls_confusion"
            or operation == "family_mcls_prf" or operation == "family_mcls_log_loss"):
        return mcls_classify_operation(operation, args, context)
    if operation == "family_mcls_roc_auc" or operation == "family_mcls_pr_curve":
        return mcls_ranking_operation(operation, args, context)
    return mcls_cluster_operation(operation, args, context)
