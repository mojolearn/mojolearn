# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the classification, ranking and clustering metrics of python/mojolearn/_expansion_metrics.py: one native call per public method,
# reached from bindings/family_native.mojo for every `family_mexp_*` operation.
# The function groups live in bindings/metrics_exp_cls.mojo (classification),
# bindings/metrics_exp_curve.mojo (curves, average precision, top-k, auc),
# bindings/metrics_exp_rank.mojo (probabilities, hinge, ranking) and
# bindings/metrics_exp_cluster.mojo (clustering); the shared program, arena
# and label pieces in bindings/metrics_exp_common.mojo.
from std.python import Python, PythonObject
from bindings.classical_attributes import classical_connect_numpy
from bindings.metrics_exp_cls import metrics_exp_cls_operation
from bindings.metrics_exp_curve import metrics_exp_curve_operation
from bindings.metrics_exp_rank import metrics_exp_rank_operation
from bindings.metrics_exp_cluster import metrics_exp_cluster_operation


def metrics_exp_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    if operation.startswith("family_mexp_cls_"):
        return metrics_exp_cls_operation(operation, args, context)
    if operation.startswith("family_mexp_curve_"):
        return metrics_exp_curve_operation(operation, args, context)
    if operation.startswith("family_mexp_rank_"):
        return metrics_exp_rank_operation(operation, args, context)
    if operation.startswith("family_mexp_cluster_"):
        return metrics_exp_cluster_operation(operation, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
