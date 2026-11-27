"""Nested Python lists reach the float32-only estimators (the scalers) as
float32 through their native admission, with the SAME bytes as the float32
Array of the same list; a float64 array is converted to float32 once
(round-to-nearest-even), the same bytes as the float32 Array of its values.

Host-side only: the input checks run before any device work.
"""
import array
import struct

import pytest

from mojolearn import MinMaxScaler, StandardScaler
from mojolearn._array import Array

# A tie between two float32 neighbors (round-to-nearest-even and
# round-half-away disagree), a float32 subnormal, a value that is not a
# float32, and a plain int.
_TIE = struct.unpack("<d", struct.pack("<Q", 0x3FF0000010000000))[0]
_ROWS = [[0.1, 1.0 / 3.0, _TIE], [2.5, 1e-40, 7]]
_TARGET = [0.1, 1.0 / 3.0, _TIE, -2.0, 3]


# TOMBSTONE: the `_buffer.materialize_f32_lists` / `as_f32_c` tests (they tested the deleted Python
# conversion layer) deleted 2026-11-15 by lane py-tail-2; code recoverable at 1a9d8c01bd.


def test_float32_list_rows_are_one_rounding():
    # the float32 Array of a list rounds each double once (round-to-nearest-even)
    assert Array.from_list(_TARGET, dtype="<f4").tobytes() == array.array("f", _TARGET).tobytes()


def test_float64_buffer_is_converted_once():
    # Andrew, 2026-11-27: float64 is converted at the native boundary with
    # round-to-nearest-even, the bytes of the float32 Array of the same values
    # (_TIE is the tie case where a second rounding rule would differ).
    wide = StandardScaler().fit(Array.from_list(_ROWS, dtype="<f8"))
    narrow = StandardScaler().fit(Array.from_list(_ROWS, dtype="<f4"))
    for name in ("mean_", "var_", "scale_"):
        assert getattr(wide, name).tobytes() == getattr(narrow, name).tobytes(), name


@pytest.mark.parametrize("scaler", [StandardScaler, MinMaxScaler])
def test_scaler_input_accepts_lists(scaler):
    # the native admission (bindings/scaler_native_door.mojo) takes the list
    # rows as float32, the same words as the array
    from_list = scaler().fit(_ROWS).transform(_ROWS)
    rows = Array.from_list(_ROWS, dtype="<f4")
    from_array = scaler().fit(rows).transform(rows)
    assert from_list.tobytes() == from_array.tobytes()
