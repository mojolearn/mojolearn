# SPDX-License-Identifier: Apache-2.0
# Array label values from one retained logical dtype/shape/stride snapshot.
# Keep the GP-specific helper and its API intact; both use native scalar bits.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from bindings.p3_labels_metrics_native_door import p3_array_numbers
from bindings.native_scalar_decode_unpack import NativeScalarBuffer
from core.native_scalar_decode import NativeRealBits, native_scalar_real
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin, array_shape_once, array_metadata_once


from bindings.label_snapshot_native_pack import (
    _label_axis,
)


def label_array_list(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var shape = array_shape_once(value, context)
    var pin = ArrayPin()
    var snapshot = array_metadata_once(value, shape^, "labels", context, pin, allow_unsupported=True)
    # Higher rank metadata is admission-only for numerical CastIn. This label
    # decoder uses captured address/width/strides and declared dtype directly;
    # it never consumes the placeholder CastIn.kind when castable is false.
    var result = _label_axis(snapshot, 0, 0)
    _ = pin^  # the door's pin holds the label storage through the decode
    return result
