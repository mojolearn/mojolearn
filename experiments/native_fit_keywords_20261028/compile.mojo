# SPDX-License-Identifier: Apache-2.0
# Actual preprocessing entry, compile only; never execute this fixture.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.prep_encode_native import prep_encode_c


@export
def PyInit__native_fit_keywords() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_native_fit_keywords")
        module.def_py_c_function(prep_encode_c, "prep")
        return module.finalize()
    except error:
        abort(String(error))
