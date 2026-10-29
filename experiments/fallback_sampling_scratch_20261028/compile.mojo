# SPDX-License-Identifier: Apache-2.0
# Actual public read/generate entries on integrated resident/native source.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.causal_lm_public import causal_public_c


@export
def PyInit__fallback_sampling_scratch() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_fallback_sampling_scratch")
        module.def_py_c_function(causal_public_c[5], "read")
        module.def_py_c_function(causal_public_c[7], "generate")
        return module.finalize()
    except error:
        abort(String(error))
