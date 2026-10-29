# SPDX-License-Identifier: Apache-2.0
# Actual SmallMLP public entry, focused compilation and installed regression entry.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.small_mlp_native import small_mlp_c


@export
def PyInit__small_mlp_items() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_small_mlp_items")
        module.def_py_c_function(small_mlp_c, "small_mlp")
        return module.finalize()
    except error:
        abort(String(error))
