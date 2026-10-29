# SPDX-License-Identifier: Apache-2.0
# Compile only the repaired public host-SVC attribute root; never execute it.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.host_svc_native import host_svc_attribute_c


@export
def PyInit__host_svc_mode_recheck() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_host_svc_mode_recheck")
        module.def_py_c_function(host_svc_attribute_c, "host_svc_attribute")
        return module.finalize()
    except error:
        abort(String(error))
