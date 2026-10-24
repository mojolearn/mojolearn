# SPDX-License-Identifier: Apache-2.0
# The native owner of RFE and of mutual_info_classif / mutual_info_regression
# (dispatched by bindings/prep_owner_native.mojo).
from std.python import Python, PythonObject


def prep_rfe_call(model: PythonObject, operation: String, args: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    raise Error("TypeError: the native RFE owner does not serve " + operation)


def prep_mutual_info_call(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    raise Error("TypeError: the native mutual-information owner does not serve " + operation)
