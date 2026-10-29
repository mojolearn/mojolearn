# SPDX-License-Identifier: Apache-2.0
# Actual affected public entries; compile only, never execute this fixture.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.causal_lm_public import causal_public_c



@export
def PyInit__generation_pick_buffers() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_generation_pick_buffers")
        module.def_py_c_function(causal_public_c[3], "forward")
        module.def_py_c_function(causal_public_c[5], "read")
        module.def_py_c_function(causal_public_c[7], "generate")
        return module.finalize()
    except error:
        abort(String(error))
