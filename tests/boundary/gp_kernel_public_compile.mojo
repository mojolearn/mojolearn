# Compile-only coverage of all GP kernel metadata dispatch branches and admission.
# This extension is never loaded or executed by the compilation controller.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.gp_kernel_public_native import gp_kernel_operation, gp_admit_kernel, gp_kernel_nodes

def kernel_operation(kernel: PythonObject, operation: PythonObject, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    return gp_kernel_operation(kernel, String(operation), args, context)

@export
def PyInit__mojolearn_gp_kernel_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_gp_kernel_compile")
        module.def_function[kernel_operation]("gp_kernel_operation")
        module.def_function[gp_admit_kernel]("gp_admit_kernel")
        module.def_function[gp_kernel_nodes]("gp_kernel_nodes")
        return module.finalize()
    except error:
        abort(String(error))
