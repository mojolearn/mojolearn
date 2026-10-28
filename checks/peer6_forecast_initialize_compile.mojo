# Compile-only constructor provider; no runtime or identity execution.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from providers.python.forecast_initialize_pyapi import forecast_initialize

def initialize(model: PythonObject, kind: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return forecast_initialize(model, String(py=kind), args, context)

@export
def PyInit__peer6_forecast_initialize_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_peer6_forecast_initialize_compile")
        module.def_function[initialize]("forecast_initialize")
        return module.finalize()
    except error:
        abort(String(error))
