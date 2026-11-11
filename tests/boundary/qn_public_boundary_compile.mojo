# SPDX-License-Identifier: Apache-2.0
# Compile the actual QN public native C entry, including lifetime/lifecycle.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.qn_native import qn_c

@export
def PyInit__mojolearn_qn_public_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_qn_public_compile")
        module.def_py_c_function(qn_c, "qn_native_call")
        return module.finalize()
    except problem:
        abort(String(problem))
