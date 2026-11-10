# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Legacy public host label wrappers: scalar UNPACK, native RUN, scalar PACK."""
from std.python import PythonObject
from core.labels_native_host import LabelsHostKernels
from bindings.labels_legacy_boundary import labels_legacy_unique, labels_legacy_prepare
from core.labels_prepare import (
    PT_ENCODE_CAP, PT_W_NONFINITE, PT_W_NEGATIVE, PT_W_ALL_ZERO,
    PT_ALLOW_NEGATIVE, PT_ALLOW_ALL_ZERO,
)


def unique_inverse_binding(src_addr: PythonObject, n: PythonObject, kind: PythonObject,
    classes_addr: PythonObject, codes_addr: PythonObject) raises -> PythonObject:
    return labels_legacy_unique[LabelsHostKernels](src_addr, n, kind, classes_addr, codes_addr)


def prepare_targets_binding(y_addr: PythonObject, y_code: PythonObject, n: PythonObject,
    classes_addr: PythonObject, codes_addr: PythonObject, w_addr: PythonObject,
    w_code: PythonObject, w_flags: PythonObject) raises -> PythonObject:
    return labels_legacy_prepare[LabelsHostKernels](y_addr, y_code, n,
        classes_addr, codes_addr, w_addr, w_code, w_flags)
