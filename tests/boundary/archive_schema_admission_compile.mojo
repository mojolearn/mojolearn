# SPDX-License-Identifier: Apache-2.0
# Compile the actual shared estimator archive-save owner and field admission.
from std.os import abort
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder, raise_python_exception
from providers.python.classical_family_admission_pyapi import classical_family_admit
from bindings.archive_schema_native import archive_schema_save


def archive_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        return archive_schema_save(args[0], args[1], args[2]).steal_data()
    except problem:
        return raise_python_exception(problem)


@export
def PyInit__mojolearn_archive_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_archive_compile")
        module.def_py_c_function(archive_c, "archive")
        return module.finalize()
    except problem:
        abort(String(problem))
