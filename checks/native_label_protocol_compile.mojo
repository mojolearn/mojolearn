# SPDX-License-Identifier: Apache-2.0
# Compile-only actual shared label operations, all SVM owners, and GP label callers.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.classical_labels import (
    _label_list, _flatten_labels, _sorted_classes, labels_label_kind,
    labels_encode_labels, labels_decode_labels, _classification_labels, labels_class_weight_rows,
)
from bindings.classifier_public_lifecycle import classifier_label_list, _public_labels
from bindings.svc_native import svc_operation
from bindings.gp_labels_native import gp_encode_labels, gp_decode_labels

@export
def native_label_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var a = PythonObject(from_borrowed=packed)
        var operation = String(a[0])
        var value = a[1]
        var context = a[2]
        if operation == "list":
            return _label_list(value, context).steal_data()
        if operation == "flatten":
            return _flatten_labels(value, context).steal_data()
        if operation == "kind":
            return PythonObject(labels_label_kind(value, context)).steal_data()
        if operation == "decode":
            return labels_decode_labels(value, a[3], context).steal_data()
        if operation == "classify":
            var labels = _classification_labels(value, "labels", context)
            return labels.values.steal_data()
        if operation == "public":
            return _public_labels(classifier_label_list(value, context), context).steal_data()
        var encoded = _sorted_classes(value, context) if operation == "sort" else labels_encode_labels(value, context)
        if operation == "weights":
            return labels_class_weight_rows(a[3], encoded, a[4], context).steal_data()
        return Python.tuple(encoded.classes, encoded.codes).steal_data()
    except error:
        return raise_python_exception(error)

@export
def native_svm_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var a = PythonObject(from_borrowed=packed)
        return svc_operation(a[0], String(a[1]), a[2], a[3]).steal_data()
    except error:
        return raise_python_exception(error)

@export
def native_gp_label_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var a = PythonObject(from_borrowed=packed)
        if String(a[0]) == "encode":
            return gp_encode_labels(a[1], a[2], a[3]).steal_data()
        return gp_decode_labels(a[1], a[2], a[3], a[4]).steal_data()
    except error:
        return raise_python_exception(error)
