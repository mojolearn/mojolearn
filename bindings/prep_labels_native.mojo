# SPDX-License-Identifier: Apache-2.0
# The native owner of LabelEncoder, LabelBinarizer and MultiLabelBinarizer
# (dispatched by bindings/prep_owner_native.mojo).
from std.python import Python, PythonObject


def prep_labels_call(model: PythonObject, kind: String, operation: String, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    raise Error("TypeError: the native label owner does not serve " + operation)
