# Compile-only fixture: full report operations and transitive native owners.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.metrics_reg_native import metrics_reg_operation
from bindings.metrics_exp_native import metrics_exp_operation

def regression_report_compile(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return metrics_reg_operation("family_metric_regression_report", args, context)

def ranking_report_compile(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return metrics_exp_operation(PythonObject(None), "family_mexp_curve_ranking_report", args, context)

@export
def PyInit__mojolearn_metrics_reports_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_metrics_reports_compile")
        m.def_function[regression_report_compile]("regression_report")
        m.def_function[ranking_report_compile]("ranking_report")
        return m.finalize()
    except error:
        abort(String(error))
