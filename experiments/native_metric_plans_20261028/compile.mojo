# SPDX-License-Identifier: Apache-2.0
# Compile actual affected metric public-operation bodies, never execute.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.metrics_exp_native import metrics_exp_operation
from bindings.metrics_cls_native import metrics_cls_operation
from bindings.arena_prog_native import _metrics_run_fields


def expansion(operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return metrics_exp_operation(PythonObject(None), String(operation), args, context)


def classification(operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return metrics_cls_operation(PythonObject(None), String(operation), args, context)


@export
def PyInit__native_metric_plans() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_native_metric_plans")
        module.def_function[expansion]("expansion")
        module.def_function[classification]("classification")
        module.def_function[_metrics_run_fields]("public_program")
        return module.finalize()
    except error:
        abort(String(error))
