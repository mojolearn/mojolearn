# Compile-only: complete classification, regression and expanded owners.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.metrics_cls_native import metrics_cls_operation
from bindings.metrics_reg_native import metrics_reg_operation
from bindings.metrics_exp_native import metrics_exp_operation

def classifications(model: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return metrics_cls_operation(model, String(operation), args, context)

def regressions(operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return metrics_reg_operation(String(operation), args, context)

def expansions(model: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return metrics_exp_operation(model, String(operation), args, context)

@export
def PyInit__mojolearn_metrics_boundaries_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_metrics_boundaries_compile")
        m.def_function[classifications]("classifications")
        m.def_function[regressions]("regressions")
        m.def_function[expansions]("expansions")
        return m.finalize()
    except error:
        abort(String(error))
