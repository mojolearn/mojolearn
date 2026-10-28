# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the linalg functions of python/mojolearn/_linalg_impl.py (quantize/dequantize, eigh, svd, ...), resample.py and parallel_graph.py (family_lin_, family_rs_, family_pg_ operations): one native call per public method,
# reached from bindings/family_native.mojo for every `family_lin_*`, `family_rs_*`
# and `family_pg_*` operation. The functions live in bindings/linalg_ops_*.mojo.
from std.python import Python, PythonObject
from bindings.classical_attributes import classical_connect_numpy
from bindings.linalg_ops_lowbit import lin_lowbit_operation
from bindings.linalg_ops_resample import rs_operation
from bindings.linalg_ops_decomp import lin_decomp_operation
from providers.python.linalg_result_pyapi import linalg_result_operation


def linalg_ops_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    if operation.startswith("family_lin_result_"):
        return linalg_result_operation(operation, args, context)
    if (operation == "family_lin_qr" or operation == "family_lin_eigh" or operation == "family_lin_svdvals"
            or operation == "family_lin_svd"):
        return lin_decomp_operation(operation, args, context)
    if operation.startswith("family_lin_"):
        return lin_lowbit_operation(operation, args, context)
    if operation.startswith("family_rs_"):
        return rs_operation(operation, args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
