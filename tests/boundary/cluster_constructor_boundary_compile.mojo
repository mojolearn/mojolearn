# SPDX-License-Identifier: Apache-2.0
# Compile the public lifecycle owner including the newly native constructors.
from std.os import abort
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder, raise_python_exception
from providers.python.classical_family_admission_pyapi_unpack import classical_family_admit
from providers.python.cluster_public_lifecycle_pyapi_door import cluster_public_operation


def cluster_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=packed)
        _ = classical_family_admit(args[0])
        # args[1]: the route owner, args[2]: its leaf code (lane bw2-cluster).
        return cluster_public_operation(args[0], Int(py=args[1]), Int(py=args[2]), "", args[3], args[4]).steal_data()
    except problem:
        return raise_python_exception(problem)


@export
def PyInit__mojolearn_cluster_constructor_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_cluster_constructor_compile")
        module.def_py_c_function(cluster_c, "cluster")
        return module.finalize()
    except problem:
        abort(String(problem))
