# SPDX-License-Identifier: Apache-2.0
# Compile affected public label lifecycle and accuracy owner; never execute.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.prep_owner_native import prep_owner_c
from bindings.classical_labels import labels_accuracy


def accuracy(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return PythonObject(labels_accuracy(args[0], args[1], args[2], args[3], context))


@export
def PyInit__label_type_reuse() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_label_type_reuse")
        module.def_py_c_function(prep_owner_c, "prep")
        module.def_function[accuracy]("accuracy")
        return module.finalize()
    except error:
        abort(String(error))
