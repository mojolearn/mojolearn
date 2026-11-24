# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the classification, ranking and clustering metrics of python/mojolearn/_expansion_metrics.py: one native call per public method,
# reached from bindings/family_native.mojo for every `family_mexp_*` operation.
# The function groups live in bindings/metrics_exp_cls_door.mojo (classification),
# bindings/metrics_exp_curve_door.mojo (curves, average precision, top-k, auc),
# bindings/metrics_exp_rank_door.mojo (probabilities, hinge, ranking) and
# bindings/metrics_exp_cluster.mojo (clustering); the shared program, arena
# and label pieces in bindings/metrics_exp_common_unpack.mojo.
from bindings.operation_names_mexp import (
    MEXP_OPN_FAMILY_MEXP_CLS_LIKELIHOOD_RATIOS,
    MEXP_OPN_FAMILY_MEXP_CURVE_AUC,
    MEXP_OPN_FAMILY_MEXP_CURVE_AVERAGE_PRECISION,
    MEXP_OPN_FAMILY_MEXP_CLS_REPORT,
    MEXP_OPN_FAMILY_MEXP_CURVE_TOP_K,
    MEXP_OPN_FAMILY_MEXP_RANK_LOSS,
)
from std.python import Python, PythonObject
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_pandas_door import classical_admit_value
from bindings.metrics_native_support_door import metric_arguments_to
from bindings.metrics_device_door import metric_device_admit, metric_device_identical, metric_device_inputs
from core.device_route import DEVICE_INSTALL
from providers.python.function_device_boundary_door import function_device_scope
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
    cls, curve, rank and cluster codes are four ranges. The native callers'
    form: no `device` item, the route of the caller's context."""
    var end = len(args)
    return _metrics_exp_run(operation, args^, end, context)


def metrics_exp_public(operation: Int, var args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The family door's form (bindings/family_native_door.mojo): the public
    shell's tuple ends with its `device` (lane vdf-metrics), read FIRST, before
    any admission or work (bindings/metrics_device_door.mojo). The programs
    and label steps run in `function_device_scope`'s context."""
    var end = len(args) - 1
    if end < 0:
        raise Error("TypeError: a public metric call carries its device as the last argument")
    var device = metric_device_admit(args[end], "this mojolearn.metrics function", context)
    if device.request != DEVICE_INSTALL:
        metric_device_inputs(device, args, end, context)
    if device.host_beside:
        metric_device_identical(_mexp_mode(operation, args, end), context)
    return _metrics_exp_run(operation, args^, end, function_device_scope(device, context))


def _mexp_mode(operation: Int, args: PythonObject, end: Int) raises -> PythonObject:
    """The `numeric_mode` item (explicit device="cpu" beside a GPU set only):
    the last item, except the likelihood ratios (item 5, before the selector
    of the single-ratio forms), average_precision_score (item 4, before
    `average`) and auc, which carries none (the default mode)."""
    if operation == MEXP_OPN_FAMILY_MEXP_CURVE_AUC:
        return PythonObject(None)
    if operation == MEXP_OPN_FAMILY_MEXP_CLS_LIKELIHOOD_RATIOS:
        var first_mode = args[5]  # a local, never a returned subscript (AGENTS.md part 6)
        return first_mode
    if operation == MEXP_OPN_FAMILY_MEXP_CURVE_AVERAGE_PRECISION:
        var second_mode = args[4]  # a local, never a returned subscript (AGENTS.md part 6)
        return second_mode
    var last_mode = args[end - 1]  # a local, never a returned subscript (AGENTS.md part 6)
    return last_mode


def _metrics_exp_run(operation: Int, var args: PythonObject, end: Int, context: PythonObject) raises -> PythonObject:
    """`args[:end]` are the metric's own arguments; `context` is already the
    device's."""
    classical_connect_numpy(context)
    classical_admit_value(args, context)
    args = metric_arguments_to(args, 0, end, context)
    if operation <= MEXP_OPN_FAMILY_MEXP_CLS_REPORT:
        return metrics_exp_cls_operation(operation, args, context)
    if operation <= MEXP_OPN_FAMILY_MEXP_CURVE_TOP_K:
        return metrics_exp_curve_operation(operation, args, context)
    if operation <= MEXP_OPN_FAMILY_MEXP_RANK_LOSS:
        return metrics_exp_rank_operation(operation, args, context)
    return metrics_exp_cluster_operation(operation, args, context)
