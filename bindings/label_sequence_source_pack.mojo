# SPDX-License-Identifier: Apache-2.0
"""Borrowed boundary label slots, shared by public and native pin sequences."""
# The PACK half of the former `bindings/label_sequence_source.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/label_sequence_source_unpack.mojo`.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_borrowed_pack import borrowed_item


from bindings.label_sequence_source_unpack import (
    label_source_raw,
)


def label_source_value[native: Bool](source: PythonObject, pins: List[PythonObject],
                                     index: Int) raises -> PythonObject:
    return PythonObject(from_borrowed=label_source_raw[native](source, pins, index))
