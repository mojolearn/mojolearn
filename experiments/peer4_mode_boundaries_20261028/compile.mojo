# Compile-only native public slot and full-reset reconstruction owners.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.public_attributes_native import public_attribute_c
from bindings.full_reset_params_native import full_reset_params_c

@export
def PyInit__mojolearn_mode_boundaries_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_mode_boundaries_compile")
        module.def_py_c_function(public_attribute_c, "public_attribute_native_call")
        module.def_py_c_function(full_reset_params_c, "full_reset_params_native_call")
        return module.finalize()
    except error:
        abort(String(error))
