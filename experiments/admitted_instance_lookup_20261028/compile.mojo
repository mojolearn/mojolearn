# SPDX-License-Identifier: Apache-2.0
# Actual native attribute entry, compile only; never execute this fixture.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.public_attributes_native import public_attribute_c


@export
def PyInit__admitted_instance_lookup() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_admitted_instance_lookup")
        module.def_py_c_function(public_attribute_c, "attribute")
        return module.finalize()
    except error:
        abort(String(error))
