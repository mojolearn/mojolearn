# SPDX-License-Identifier: Apache-2.0
"""Lazy scalar UNPACK for the legacy public label entries.

These compatibility entries retain the original refusal order and do not
convert fields an empty operation never used. Internal callers use one fully
typed request; the numerical implementation is shared with those callers.
"""
# The PACK half of the former `bindings/labels_legacy_boundary.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/labels_legacy_boundary_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.labels_native_api import LabelsPrepareRequest
from core.labels_prepare import LabelsPreparationKernels, labels_prepare_targets
from core.labels_admission import (
    labels_unique_shape, labels_prepare_count, labels_prepare_label_count,
    labels_require_address,
)


from bindings.labels_legacy_boundary_unpack import (
    labels_legacy_address,
)


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
