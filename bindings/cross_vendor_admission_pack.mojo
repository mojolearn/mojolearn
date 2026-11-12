# SPDX-License-Identifier: Apache-2.0
"""Exact native types at the distributed transport boundary; no protocols."""
# The PACK half of the former `bindings/cross_vendor_admission.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/cross_vendor_admission_unpack.mojo`.
from std.math import isfinite
from std.python import Python, PythonObject
from bindings.native_json import native_path
from bindings.py_balanced import py_call, py_type, py_get
from core.shared_slot import SharedSlot


from bindings.cross_vendor_admission_unpack import (
    CROSS_VENDOR_TYPES,
    cv_sequence,
    cv_text,
)


def cv_path(value: PythonObject) raises -> PythonObject:
    var c = CROSS_VENDOR_TYPES.get_or_create_ptr()
    var kind = py_type(value)
    if kind is not c[].bi.str and kind is not c[].bi.bytes and kind is not c[].path and kind is not c[].pure_path:
        raise Error("cross_vendor: path requires str, bytes or a standard Path; Python path hooks are not accepted")
    if kind is c[].path or kind is c[].pure_path:
        var parts = PythonObject(None)
        try:
            parts = py_call(c[].bi.object.__getattribute__, value, "_raw_paths")
        except:
            parts = py_call(c[].bi.object.__getattribute__, value, "_parts")
        var components = cv_sequence(parts, "path storage")
        for i in range(len(components)):  # driver-loop: over stored native path components
            _ = cv_text(components[i], "path component")
    return native_path(value)
