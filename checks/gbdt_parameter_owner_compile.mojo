# Compile-only native GBDT parameter lifecycle and shared forest signature path.
# This fixture is not imported, loaded or executed.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.trees_gbdt_parameters import gbdt_adapter_construct, gbdt_adapter_set_params
from bindings.trees_constructor_native import forest_construct
from bindings.params_metadata_native import params_metadata_c

@export
def PyInit__mojolearn_gbdt_parameter_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_gbdt_parameter_compile")
        module.def_function[gbdt_adapter_construct]("construct")
        module.def_function[gbdt_adapter_set_params]("set_params")
        module.def_function[forest_construct]("forest_construct")
        module.def_py_c_function(params_metadata_c, "get_params")
        return module.finalize()
    except error:
        abort(String(error))
