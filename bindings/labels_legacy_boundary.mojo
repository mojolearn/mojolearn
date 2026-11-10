# SPDX-License-Identifier: Apache-2.0
"""Lazy scalar UNPACK for the legacy public label entries.

These compatibility entries retain the original refusal order and do not
convert fields an empty operation never used. Internal callers use one fully
typed request; the numerical implementation is shared with those callers.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.labels_native_api import LabelsPrepareRequest
from core.labels_prepare import LabelsPreparationKernels, labels_prepare_targets
from core.labels_admission import (
    labels_unique_shape, labels_prepare_count, labels_prepare_label_count,
    labels_require_address,
)


def labels_legacy_address(value: PythonObject, message: String) raises -> Int:
    var address = Int(py=value)
    labels_require_address(address, message)
    return address


def labels_legacy_unique[K: LabelsPreparationKernels](src_addr: PythonObject,
    n: PythonObject, kind: PythonObject, classes_addr: PythonObject,
    codes_addr: PythonObject) raises -> PythonObject:
    var count = Int(py=n)
    var native_kind = Int(py=kind)
    labels_unique_shape(count, native_kind)
    var message = String("unique_inverse: null buffer address")
    var source = labels_legacy_address(src_addr, message)
    var classes = labels_legacy_address(classes_addr, message)
    var codes = labels_legacy_address(codes_addr, message)
    var result = 0
    with GILReleased(Python()):
        result = K.unique(source, count, native_kind, classes, codes)
    return PythonObject(result)


def labels_legacy_prepare[K: LabelsPreparationKernels](y_addr: PythonObject,
    y_code: PythonObject, n: PythonObject, classes_addr: PythonObject,
    codes_addr: PythonObject, w_addr: PythonObject, w_code: PythonObject,
    w_flags: PythonObject) raises -> PythonObject:
    var count = Int(py=n)
    labels_prepare_count(count)
    var labels = Int(py=y_addr)
    var result = 0
    if labels != 0:
        labels_prepare_label_count(count)
        var message = String("prepare_targets: null label output address")
        var classes = labels_legacy_address(classes_addr, message)
        var codes = labels_legacy_address(codes_addr, message)
        var label_code = Int(py=y_code)
        var request = LabelsPrepareRequest(labels, classes, codes, 0,
            label_code, count, 0, 0)
        # Finish label errors before converting weight metadata, matching the
        # original public entry. This calls the same typed preparation body.
        with GILReleased(Python()):
            result = labels_prepare_targets[K](request)
    var weights = Int(py=w_addr)
    if weights != 0 and count > 0:
        var weight_code = Int(py=w_code)
        var flags = Int(py=w_flags)
        var request = LabelsPrepareRequest(0, 0, 0, weights, 0, count,
            weight_code, flags)
        var status = 0
        with GILReleased(Python()):
            status = labels_prepare_targets[K](request)
        if status < 0:
            return PythonObject(status)
    return PythonObject(result)
