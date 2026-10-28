# Compile-only fixture: actual CNN public dispatcher, no module import/execution.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.xcnn_native import xcnn_c

@export
def PyInit__mojolearn_xcnn_public_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_xcnn_public_compile")
        m.def_py_c_function(xcnn_c, "xcnn_native_call")
        return m.finalize()
    except error:
        abort(String(error))
