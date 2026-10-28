# Compile-only fixture for the complete public result initialization owner.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from providers.python.resample_public_pyapi import resample_result_initialize

@export
def PyInit__mojolearn_resample_results_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_resample_results_compile")
        m.def_function[resample_result_initialize]("initialize")
        return m.finalize()
    except error:
        abort(String(error))
