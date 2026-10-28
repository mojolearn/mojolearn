# Compile-only: metric owners plus the retained public arena Prog wrapper.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.metrics_cls_native import metrics_cls_operation
from bindings.arena_prog_native import _metrics_run
from bindings.metrics_reg_native import metrics_reg_operation
from bindings.metrics_exp_native import metrics_exp_operation

def classifications(model: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return metrics_cls_operation(model, String(operation), args, context)

def regressions(operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return metrics_reg_operation(String(operation), args, context)

def expansions(model: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return metrics_exp_operation(model, String(operation), args, context)

def program(prog: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    return _metrics_run(prog, mode, context)

@export
def PyInit__mojolearn_metrics_program_fields_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_metrics_program_fields_compile")
        m.def_function[program]("program")
        m.def_function[classifications]("classifications")
        m.def_function[regressions]("regressions")
        m.def_function[expansions]("expansions")
        return m.finalize()
    except error:
        abort(String(error))
