# SPDX-License-Identifier: Apache-2.0
# Compile-only entry for the actual public TSA dispatcher and reachable owners.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.tsa_public_native import tsa_c

@export
def PyInit__mojolearn_legacy_tsa_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_legacy_tsa_compile")
        module.def_py_c_function(tsa_c, "tsa_native_call")
        return module.finalize()
    except error:
        abort(String(error))
