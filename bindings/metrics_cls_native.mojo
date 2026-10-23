# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the classification metrics of python/mojolearn/_metrics_impl.py: one native call per public method,
# reached from bindings/family_native.mojo for every `family_mcls_*` operation.
from std.python import Python, PythonObject


def metrics_cls_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown family operation " + operation)
