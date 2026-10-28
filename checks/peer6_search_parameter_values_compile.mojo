# SPDX-License-Identifier: Apache-2.0
# Compile-only fixture for the actual search parameter-value provider, including
# native float16 Array scalar preparation. No module is loaded or executed.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.msel_search_state import msel_parameter_values


def parameter_values_compile(value: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_parameter_values(value, context)


@export
def PyInit__mojolearn_search_parameter_values_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_search_parameter_values_compile")
        module.def_function[parameter_values_compile]("parameter_values")
        return module.finalize()
    except error:
        abort(String(error))
