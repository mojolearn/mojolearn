# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the classification, ranking and clustering metrics of python/mojolearn/_expansion_metrics.py: one native call per public method,
# reached from bindings/family_native.mojo for every `family_mexp_*` operation.
from std.python import Python, PythonObject


def metrics_exp_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown family operation " + operation)
