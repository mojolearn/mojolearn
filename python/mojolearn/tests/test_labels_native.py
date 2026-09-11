"""DEVIATION 2500: the native label helpers in the base binding compute the
SAME answer as the Python ORDER RULE they shortcut.

`_labels.encode_labels` is `sorted_classes(flatten_labels(y))` for one
numeric buffer; `argmax_rows` and `decode_labels` are the same scan and
the same gather. The Python routines are the definition; this file holds
the compiled copies to them object for object (classes are compared by
value, type and, for floats, sign of zero) and code for code. NumPy is the
input factory here and nothing else.

Skips when the base binding is not built.
"""
import math

import numpy as np
import pytest

from mojolearn._array import Array
from mojolearn import _labels
from mojolearn._labels import (
    _encode_labels_native, argmax_rows, decode_labels, encode_labels,
    flatten_labels, sorted_classes,
)

try:
    from mojolearn._buffer import _native
    _native("encode_labels_f32")
    _native("argmax_rows_f32")
    _native("gather_i64")
except ImportError as exc:  # pragma: no cover - build state
    pytest.skip(f"base binding without DEVIATION 2500 helpers: {exc}", allow_module_level=True)

_RNG = np.random.default_rng(1)


def _same_classes(got, ref):
    if len(got) != len(ref):
        return False
    for a, b in zip(got, ref):
        if a != b or type(a) is not type(b):
            return False
        if isinstance(a, float) and math.copysign(1.0, a) != math.copysign(1.0, b):
            return False
    return True


_CASES = {
    "f32 two classes": _RNG.integers(0, 2, 1000).astype(np.float32),
    "f32 minus zero first": np.array([-0.0, 0.0, 1.0, -0.0], np.float32),
    "f32 plus zero first": np.array([0.0, -0.0, 1.0], np.float32),
    "f64 halves": _RNG.integers(-50, 50, 5000).astype(np.float64) * 0.5,
    "i32 signed": _RNG.integers(-3, 3, 777).astype(np.int32),
    "i64 beyond float precision": np.array([2**60 + 1, 2**60, 5, -7, 2**60 + 1], np.int64),
    "u8": _RNG.integers(0, 255, 3000).astype(np.uint8),
    "u32": _RNG.integers(0, 2**32 - 1, 3000, dtype=np.uint32),
    "bool": _RNG.integers(0, 2, 100).astype(bool),
    "bool one class": np.ones(10, bool),
    "one class": np.full(10, 3.5, np.float32),
    "column": _RNG.integers(0, 3, 50).astype(np.float32).reshape(-1, 1),
    "row": _RNG.integers(0, 3, 50).astype(np.float32).reshape(1, -1),
    "F-order column": np.asfortranarray(_RNG.integers(0, 3, 50).astype(np.float64).reshape(-1, 1)),
    "int16 widened": _RNG.integers(-9, 9, 100).astype(np.int16),
    "descending": np.arange(100, 0, -1).astype(np.int64),
    "strided view": _RNG.integers(0, 3, 200).astype(np.int32)[::2],
    "cap exactly": np.arange(4096, dtype=np.int32)[::-1],
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_native_encode_equals_order_rule(name):
    y = _CASES[name]
    ref_classes, ref_codes = sorted_classes(flatten_labels(y))
    got = _encode_labels_native(y)
    assert got is not None, "native arm declined a supported buffer"
    classes, codes = got
    assert _same_classes(classes, ref_classes)
    assert codes.dtype == "<i4" and codes.ndim == 1
    assert codes.tolist() == ref_codes


def test_array_input_takes_the_native_arm():
    y = Array.from_list(_RNG.integers(0, 4, 300).astype(np.float32).tolist(), "<f4")
    ref = sorted_classes(flatten_labels(y))
    got = _encode_labels_native(y)
    assert got is not None
    assert _same_classes(got[0], ref[0]) and got[1].tolist() == ref[1]


def test_more_classes_than_the_cap_falls_back():
    y = np.arange(_labels._NATIVE_ENCODE_MAX_CLASSES + 1, dtype=np.int32)
    assert _encode_labels_native(y) is None
    classes, codes = encode_labels(y)
    assert classes == list(range(len(y))) and codes.tolist() == list(range(len(y)))


def test_matrix_labels_fall_back():
    assert _encode_labels_native(_RNG.integers(0, 3, (5, 4)).astype(np.float32)) is None


def test_lists_and_strings_take_the_python_rule():
    classes, codes = encode_labels([3, 1, 2, 1])
    assert classes == [1, 2, 3] and codes.tolist() == [2, 0, 1, 0] and codes.dtype == "<i4"
    classes, codes = encode_labels(["b", "a"])
    assert classes == ["a", "b"] and codes.tolist() == [1, 0]


def test_nan_is_refused_with_the_rule_s_message():
    y = np.array([1.0, float("nan")], np.float32)
    with pytest.raises(ValueError) as ref:
        sorted_classes(flatten_labels(y))
    with pytest.raises(ValueError) as got:
        encode_labels(y)
    assert str(got.value) == str(ref.value)


def test_empty_is_refused():
    with pytest.raises(ValueError, match="y is empty"):
        encode_labels(np.zeros(0, np.float32))


def _scores():
    sc = _RNG.random((1000, 5)).astype(np.float32)
    sc[3, :] = 0.5          # a full tie keeps column 0
    sc[7, :] = np.nan       # NaN never replaces: column 0
    sc[9, 2] = np.inf
    sc[11, 1:] = -np.inf
    ref = [0 if np.isnan(r).any() else int(np.argmax(r)) for r in sc]
    return sc, ref


@pytest.mark.parametrize("dtype", ["<f4", "<f8"])
def test_argmax_rows_native_equals_the_scan(dtype):
    sc, ref = _scores()
    arr = Array.from_buffer(sc if dtype == "<f4" else sc.astype(np.float64))
    got = argmax_rows(arr)
    assert got.dtype == "<i8" and got.tolist() == ref


@pytest.mark.parametrize("classes", [
    [10, 20, 30, 40, 50],
    [0.5, 1.5, 2.5, 3.5, 4.5],
    [True, False, True, False, True],
    ["a", "b", "c", "d", "e"],
    [2**70, 1, 2, 3, 4],
])
def test_decode_labels_matches_the_gather(classes):
    sc, ref = _scores()
    codes = argmax_rows(Array.from_buffer(sc))
    got = decode_labels(classes, codes)
    expect = [classes[i] for i in ref]
    if isinstance(got, Array):
        kind = _labels.label_kind(classes)
        assert got.dtype == ("<i8" if kind == "int" else "<f8")
        assert got.tolist() == expect
    else:
        assert got == expect
