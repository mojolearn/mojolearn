# SPDX-License-Identifier: Apache-2.0
from std.python import Python, PythonObject


def metrics_exp_curve_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown family operation " + operation)
