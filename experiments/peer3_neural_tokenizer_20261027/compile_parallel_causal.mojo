# SPDX-License-Identifier: Apache-2.0
# Compile-only parallel parent/worker dispatch and remote-state lifetime type.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder
from bindings.parallel_causal_native import parallel_causal_c, RemoteStateLifetime


@export
def peer3_parallel_causal_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return parallel_causal_c(py_self, args)


@export
def PyInit_peer3_parallel_causal_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("peer3_parallel_causal_compile")
        module.def_py_c_function(parallel_causal_c, "parallel_causal_call")
        _ = module.add_type[RemoteStateLifetime]("_RemoteStateLifetime")
        return module.finalize()
    except:
        return PythonObject(None)
