# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Legacy public host label wrappers: scalar UNPACK, native RUN, scalar PACK."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.labels_native_host import LabelsHostKernels
from core.labels_native_api import LabelsPrepareRequest
from core.labels_prepare import (
    PT_ENCODE_CAP, PT_W_NONFINITE, PT_W_NEGATIVE, PT_W_ALL_ZERO,
    PT_ALLOW_NEGATIVE, PT_ALLOW_ALL_ZERO,
)


def unique_inverse_binding(src_addr: PythonObject, n: PythonObject, kind: PythonObject,
    classes_addr: PythonObject, codes_addr: PythonObject) raises -> PythonObject:
    var count = Int(py=n)
    var native_kind = Int(py=kind)
    var source = Int(py=src_addr)
    var classes = Int(py=classes_addr)
    var codes = Int(py=codes_addr)
    var result = 0
    with GILReleased(Python()):
        result = LabelsHostKernels.unique(source, count, native_kind, classes, codes)
    return PythonObject(result)


def prepare_targets_binding(y_addr: PythonObject, y_code: PythonObject, n: PythonObject,
    classes_addr: PythonObject, codes_addr: PythonObject, w_addr: PythonObject,
    w_code: PythonObject, w_flags: PythonObject) raises -> PythonObject:
    var request = LabelsPrepareRequest(Int(py=y_addr), Int(py=classes_addr), Int(py=codes_addr),
        Int(py=w_addr), Int(py=y_code), Int(py=n), Int(py=w_code), Int(py=w_flags))
    var result = 0
    with GILReleased(Python()):
        result = LabelsHostKernels.prepare(request)
    return PythonObject(result)
