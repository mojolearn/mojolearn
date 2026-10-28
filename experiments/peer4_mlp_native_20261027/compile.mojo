# Compile-only MLPInference and SmallMLPTrainer complete native owners.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.small_mlp_native import small_mlp_c, small_mlp_finalize_c

@export
def PyInit__mojolearn_small_mlp_public_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_small_mlp_public_compile")
        m.def_py_c_function(small_mlp_c, "small_mlp_call")
        m.def_py_c_function(small_mlp_finalize_c, "small_mlp_finalize")
        return m.finalize()
    except error:
        abort(String(error))
