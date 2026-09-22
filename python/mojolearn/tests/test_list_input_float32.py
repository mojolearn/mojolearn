"""Nested Python lists reach the float32-only estimators (the scalers and
the boosted regressor's targets) as float32, with the SAME bytes every
other estimator gets from `as_f32_c` for the same list; buffer inputs keep
their dtype, so a float64 array is still refused by name.

Host-side only: the input checks run before any device work, so nothing
here fits a model.
"""
import array
import struct

import pytest

from mojolearn import GradientBoostingRegressor, MinMaxScaler, StandardScaler
from mojolearn._array import Array
from mojolearn._buffer import as_f32_c, materialize_f32_lists

# A tie between two float32 neighbors (round-to-nearest-even and
# round-half-away disagree), a float32 subnormal, a value that is not a
# float32, and a plain int.
_TIE = struct.unpack("<d", struct.pack("<Q", 0x3FF0000010000000))[0]
_ROWS = [[0.1, 1.0 / 3.0, _TIE], [2.5, 1e-40, 7]]
_TARGET = [0.1, 1.0 / 3.0, _TIE, -2.0, 3]


def test_nested_list_bytes_equal_as_f32_c():
    got, copied = materialize_f32_lists(_ROWS, "X")
    ref = as_f32_c(_ROWS, ndim=2, name="X")[0]
    assert copied and got.dtype == "<f4" and got.shape == (2, 3)
    assert got.tobytes() == ref.tobytes()
    flat = materialize_f32_lists(_TARGET, "y")[0]
    assert flat.dtype == "<f4"
    assert flat.tobytes() == as_f32_c(_TARGET, ndim=1, name="y")[0].tobytes()
    assert flat.tobytes() == array.array("f", _TARGET).tobytes()


def test_tuples_and_int_lists_convert():
    assert materialize_f32_lists(((1, 2), (3, 4)), "X")[0].dtype == "<f4"
    assert materialize_f32_lists([1, 2, 3], "y")[0].dtype == "<f4"


def test_float32_input_is_untouched():
    a = Array.from_list(_ROWS, dtype="<f4")
    got, copied = materialize_f32_lists(a, "X")
    assert got is a and not copied


def test_float64_buffer_keeps_its_dtype():
    store = array.array("d", [0.1, 0.2, 0.3])
    got = materialize_f32_lists(store, "y")[0]
    assert got.dtype == "<f8"
    with pytest.raises(TypeError, match="float32"):
        StandardScaler._input(Array.from_list(_ROWS, dtype="<f8"))
    with pytest.raises(TypeError, match="float32"):
        GradientBoostingRegressor._regression_target(got, "y")


@pytest.mark.parametrize("scaler", [StandardScaler, MinMaxScaler])
def test_scaler_input_accepts_lists(scaler):
    values = scaler._input(_ROWS)
    assert values.dtype == "<f4"
    assert values.tobytes() == as_f32_c(_ROWS, ndim=2, name="X")[0].tobytes()


def test_regressor_target_accepts_lists():
    target = materialize_f32_lists(_TARGET, "y")[0]
    GradientBoostingRegressor._regression_target(target, "y")
