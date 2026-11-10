# SPDX-License-Identifier: Apache-2.0
"""Input dictionary/scalar admission and descriptor connection seams."""
import numpy as np
import pytest

from mojolearn._array import Array
from mojolearn._class_weights import _trees_x_bind
from mojolearn._native_labels_bridge import labels_entry, LABELS_OP_CLASS_WEIGHT_ROWS


def _rows(weight, classes, codes, binding=None):
    entry, context = labels_entry()
    xbind = _trees_x_bind() if binding is None else binding
    table = {"label_weights_native_api_v1": xbind.label_weights_native_api_v1}
    return entry(LABELS_OP_CLASS_WEIGHT_ROWS,
                 (weight, classes, Array.from_list(codes, "<i4"), table), context)


def test_class_weight_numpy_scalar_admission_and_defaults():
    assert _rows({np.int64(7): np.float32(1)}, [1, 7], [0, 1, 1]) is None
    got = _rows({np.int64(7): np.float64(0.5)}, [1, 7], [0, 1, 1])
    assert got.dtype == "<f4" and got.tolist() == [1.0, 0.5, 0.5]


def test_class_weight_large_integer_and_float_key_membership():
    huge = 1 << 16000
    got = _rows({huge: 0.5, np.float64(1.0): 2}, [huge, 1], [1, 0, 1])
    assert got.tolist() == [2.0, 0.5, 2.0]


def test_class_weight_unknown_key_precedes_scalar_value_range():
    with pytest.raises(ValueError, match="labels absent from y"):
        _rows({2: -1}, [1], [0])
    with pytest.raises(TypeError, match="exact native"):
        _rows({2: object()}, [1], [0])
    with pytest.raises(ValueError, match="finite nonnegative real"):
        _rows({1: True}, [1], [0])


def test_class_weight_extended_source_one_is_not_rounded_to_unweighted():
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform has no extended-precision scalar beyond binary64")
    value = np.longdouble(1) + np.finfo(np.longdouble).eps
    assert value != 1 and float(value) == 1.0
    got = _rows({1: value}, [1], [0, 0])
    assert got is not None and got.tolist() == [1.0, 1.0]


def test_class_weight_uses_descriptor_without_legacy_compiled_calls(monkeypatch):
    xbind = _trees_x_bind()
    called = []

    def old_entry(*args):
        called.append(args)
        raise AssertionError("legacy compiled-Python class-weight connection used")

    monkeypatch.setattr(xbind, "x_trees_code_counts", old_entry)
    monkeypatch.setattr(xbind, "x_trees_class_rows", old_entry)
    got = _rows("balanced", [1, 7], [0, 1, 1], xbind)
    assert got.tolist() == [1.5, 0.75, 0.75]
    assert called == []
