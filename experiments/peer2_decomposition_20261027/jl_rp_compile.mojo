# SPDX-License-Identifier: Apache-2.0
# Compile-only exported roots for the JL scalar boundary increment.
# The coordinator compiles this with the supported runtime-classical flags;
# do not execute these forwarding symbols or count compilation as runtime
# qualification. No full decomp_c family dispatch is imported here.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.jl_native import jl_c
from bindings.rp_native import rp_fit


@export
def peer2_jl_compile_root(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return jl_c(py_self, args)


@export
def peer2_rp_jl_compile_root(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    # Force rp_fit's changed jl_min_dim(n, eps, context) call to instantiate
    # without pulling in unrelated ICA/FA/NMF/LDA/KernelPCA operation doors.
    try:
        var call = PythonObject(from_borrowed=args)
        var result = rp_fit(call[0], call[1], Bool(py=call[2]), call[3])
        return result.steal_data()
    except error:
        return raise_python_exception(error)
