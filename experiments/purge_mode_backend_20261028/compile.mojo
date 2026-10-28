# Compile-only roots of the purge-mode-backend lane: the generic public
# attribute slot, the native set_params owner (connected schemas plus the
# generic rebuild), constructor replay, backend and numeric-profile owners.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.public_attributes_native import public_attribute_c, public_attribute_generic_c
from bindings.full_reset_params_native import full_reset_params_c
from bindings.full_reset_constructor_native import full_reset_constructor_c
from bindings.backend_public_native import backend_public_c
from bindings.numeric_profile_native import numeric_profile_c

@export
def PyInit__mojolearn_purge_mode_backend_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_purge_mode_backend_compile")
        module.def_py_c_function(public_attribute_c, "public_attribute_native_call")
        module.def_py_c_function(public_attribute_generic_c, "public_attribute_generic_native_call")
        module.def_py_c_function(full_reset_params_c, "full_reset_params_native_call")
        module.def_py_c_function(full_reset_constructor_c, "full_reset_constructor_native_call")
        module.def_py_c_function(backend_public_c, "backend_public_native_call")
        module.def_py_c_function(numeric_profile_c, "numeric_profile_native_call")
        return module.finalize()
    except error:
        abort(String(error))
