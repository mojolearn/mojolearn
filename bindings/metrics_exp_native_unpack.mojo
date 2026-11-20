# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the classification, ranking and clustering metrics of python/mojolearn/_expansion_metrics.py: one native call per public method,
# reached from bindings/family_native.mojo for every `family_mexp_*` operation.
# The function groups live in bindings/metrics_exp_cls_door.mojo (classification),
# bindings/metrics_exp_curve_door.mojo (curves, average precision, top-k, auc),
# bindings/metrics_exp_rank_door.mojo (probabilities, hinge, ranking) and
# bindings/metrics_exp_cluster.mojo (clustering); the shared program, arena
# and label pieces in bindings/metrics_exp_common_unpack.mojo.
from bindings.operation_names_mexp import (
    MEXP_OPN_FAMILY_MEXP_CLS_REPORT,
    MEXP_OPN_FAMILY_MEXP_CURVE_TOP_K,
    MEXP_OPN_FAMILY_MEXP_RANK_LOSS,
)
from std.python import Python, PythonObject
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_pandas_door import classical_admit_value
from bindings.metrics_native_support_door import metric_arguments
from bindings.metrics_exp_cls_door import metrics_exp_cls_operation
from bindings.metrics_exp_curve_door import metrics_exp_curve_operation
from bindings.metrics_exp_rank_door import metrics_exp_rank_operation
from bindings.metrics_exp_cluster_door import metrics_exp_cluster_operation


def metrics_exp_unknown_operation(operation: String) raises:
    """A `family_mexp_*` name the operation table does not hold (the shells
    pass table names only): the text the route raised for it, chosen by the
    same prefixes."""
    if operation.startswith("family_mexp_cls_"):
        raise Error("RuntimeError: unknown classification metric " + operation)
    if operation.startswith("family_mexp_curve_"):
        raise Error("RuntimeError: unknown curve metric " + operation)
    if operation.startswith("family_mexp_rank_"):
        raise Error("RuntimeError: unknown ranking metric " + operation)
    if operation.startswith("family_mexp_cluster_"):
        raise Error("RuntimeError: unknown clustering metric " + operation)
    raise Error("RuntimeError: unknown family operation " + operation)


def metrics_exp_operation(model: PythonObject, operation: Int, var args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    """`operation` is the leaf code of bindings/operation_names_mexp.mojo:
    cls, curve, rank and cluster codes are four ranges."""
    classical_connect_numpy(context)
    classical_admit_value(args, context)
    args = metric_arguments(args, 0, context)
    if operation <= MEXP_OPN_FAMILY_MEXP_CLS_REPORT:
        return metrics_exp_cls_operation(operation, args, context)
    if operation <= MEXP_OPN_FAMILY_MEXP_CURVE_TOP_K:
        return metrics_exp_curve_operation(operation, args, context)
    if operation <= MEXP_OPN_FAMILY_MEXP_RANK_LOSS:
        return metrics_exp_rank_operation(operation, args, context)
    return metrics_exp_cluster_operation(operation, args, context)
