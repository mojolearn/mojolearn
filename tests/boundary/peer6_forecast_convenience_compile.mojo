# Compile-only public xseq dispatch; no runtime or identity execution.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.xseq_native import xseq_c

@export
def PyInit__peer6_forecast_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_peer6_forecast_compile")
        module.def_py_c_function(xseq_c, "xseq_native_call")
        return module.finalize()
    except error:
        abort(String(error))
