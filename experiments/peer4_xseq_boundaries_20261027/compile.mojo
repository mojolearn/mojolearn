# Compile-only recurrent/MLP owner, public lifecycle and native admission.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.xseq_native import xseq_c

@export
def PyInit__mojolearn_xseq_boundaries_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_xseq_boundaries_compile")
        module.def_py_c_function(xseq_c, "xseq_native_call")
        return module.finalize()
    except error:
        abort(String(error))
