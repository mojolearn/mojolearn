# Compile-only: native pandas storage reader. No public/runtime execution.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from providers.python.resample_pandas_pyapi import resample_pandas_array

@export
def PyInit__peer6_resample_pandas_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_peer6_resample_pandas_compile")
        module.def_function[resample_pandas_array]("resample_pandas_array")
        return module.finalize()
    except error:
        abort(String(error))
