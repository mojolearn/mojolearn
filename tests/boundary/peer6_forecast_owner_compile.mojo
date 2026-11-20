# Compile-only actual constructor providers and their shared release admission.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from providers.python.forecast_initialize_pyapi_pack import forecast_initialize
from providers.python.autoarima_lifecycle_pyapi_door import autoarima_initialize
from bindings.xseq_forecast_native_door import xseq_forecast
from bindings.xseq_keys_unpack import xseq_keys
from bindings.py_keys import py_op_code

def initialize(model: PythonObject, kind: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return forecast_initialize(model, String(py=kind), args, context)

def forecast(model: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # The owner takes the xseq operation code and the caller's str (bw2-sequence).
    var nk = xseq_keys()
    return xseq_forecast(model, py_op_code(nk[].ops, operation), operation, args, context)

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
