# Compile-only fixture: full report operations and transitive native owners.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.metrics_reg_door import metrics_reg_operation
from bindings.metrics_exp_native_unpack import metrics_exp_operation
from bindings.operation_names_metric import METRIC_OPN_FAMILY_METRIC_REGRESSION_REPORT
from bindings.operation_names_mexp import MEXP_OPN_FAMILY_MEXP_CURVE_RANKING_REPORT

def regression_report_compile(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return metrics_reg_operation(METRIC_OPN_FAMILY_METRIC_REGRESSION_REPORT, args, context)

def ranking_report_compile(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return metrics_exp_operation(PythonObject(None), MEXP_OPN_FAMILY_MEXP_CURVE_RANKING_REPORT, args, context)

@export
def PyInit__mojolearn_metrics_reports_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_metrics_reports_compile")
        m.def_function[regression_report_compile]("regression_report")
        m.def_function[ranking_report_compile]("ranking_report")
        return m.finalize()
    except error:
        abort(String(error))
