# SPDX-License-Identifier: Apache-2.0
# Filled by the next commit of lane mcb2-mcls.
from std.python import Python, PythonObject


def mcls_classify_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown family operation " + operation)
