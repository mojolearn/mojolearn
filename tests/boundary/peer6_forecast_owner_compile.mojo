# Compile-only actual constructor providers and their shared release admission.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from providers.python.forecast_initialize_pyapi import forecast_initialize
from providers.python.autoarima_lifecycle_pyapi import autoarima_initialize
from bindings.xseq_forecast_native import xseq_forecast

def initialize(model: PythonObject, kind: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return forecast_initialize(model, String(py=kind), args, context)

def forecast(model: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return xseq_forecast(model, String(py=operation), args, context)

@export
def PyInit__peer6_forecast_owner_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_peer6_forecast_owner_compile")
        module.def_function[initialize]("forecast_initialize")
        module.def_function[autoarima_initialize]("autoarima_initialize")
        module.def_function[forecast]("xseq_forecast")
        return module.finalize()
    except error:
        abort(String(error))
