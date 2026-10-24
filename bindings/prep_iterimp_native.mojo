# SPDX-License-Identifier: Apache-2.0
# The native owner of IterativeImputer (dispatched by bindings/prep_owner_native.mojo).
from std.python import Python, PythonObject


def prep_iterimp_call(model: PythonObject, operation: String, args: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    raise Error("TypeError: the native IterativeImputer owner does not serve " + operation)
