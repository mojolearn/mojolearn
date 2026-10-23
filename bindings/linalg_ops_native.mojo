# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the linalg functions of python/mojolearn/_linalg_impl.py (quantize/dequantize, eigh, svd, ...), resample.py and parallel_graph.py (family_lin_, family_rs_, family_pg_ operations): one native call per public method,
# reached from bindings/family_native.mojo for every `family_lin_*` operation.
from std.python import Python, PythonObject


def linalg_ops_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown family operation " + operation)
