# SPDX-License-Identifier: Apache-2.0
"""Python-specific scalar admission and public label packing seams."""
import numpy as np
import pytest

from mojolearn._array import Array
from mojolearn._native_labels_bridge import (
    LABELS_OP_SCORE_FROM_CODES,
    LABELS_OP_UNION_ENCODE,
    labels_entry,
)


def _pair(left, right):
    entry, context = labels_entry()
    return entry(LABELS_OP_UNION_ENCODE, (left, right, "confusion_matrix"), context)


def test_numpy_integral_scalars_pack_as_exact_python_ints():
    # Unlike the pure Mojo key fixtures, this checks NumPy scalar type admission
    # and CPython normalization/ownership, including an out-of-Int64 value.
    left = [np.bool_(True), False, np.int16(7)]
    right = [np.uint64(2**64 - 1), 7, True]
    tc, ti, pc, pi, kind, present = _pair(left, right)
    assert kind == "integer"
    assert list(tc) == [0, 1, 7] and ti.tolist() == [1, 0, 2]
    assert list(pc) == [1, 7, 2**64 - 1] and pi.tolist() == [2, 1, 0]
    assert list(present) == [0, 1, 7, 2**64 - 1]
    assert all(type(value) is int for values in (tc, pc, present) for value in values)


def test_numpy_strings_and_surrogates_pack_as_exact_python_strs():
    tc, ti, pc, pi, kind, present = _pair(
        [np.str_("β"), "a", "\ud800"], ["😀", np.str_("a"), "β"]
    )
    assert kind == "string"
    assert list(tc) == ["a", "β", "\ud800"] and ti.tolist() == [1, 0, 2]
    assert list(pc) == ["a", "β", "😀"] and pi.tolist() == [2, 0, 1]
    assert list(present) == ["a", "β", "\ud800", "😀"]
    assert all(type(value) is str for values in (tc, pc, present) for value in values)


def test_large_python_int_admission_does_not_require_decimal_conversion():
    huge = 1 << 16000
    tc, ti, pc, pi, kind, present = _pair([huge, -huge, 0], [huge + 1, huge, -huge])
    assert kind == "integer"
    assert list(tc) == [-huge, 0, huge] and ti.tolist() == [2, 0, 1]
    assert list(pc) == [-huge, huge, huge + 1] and pi.tolist() == [2, 1, 0]
    assert list(present) == [-huge, 0, huge, huge + 1]


def test_rebound_duplicate_fitted_classes_keep_last_index_mapping():
    entry, context = labels_entry()
    codes = Array.from_list([0, 2, 1], "<i4")
    # The existing dictionary construction assigns index2 to the final7.
    got = entry(LABELS_OP_SCORE_FROM_CODES, ([7, 7, 1], [7, 1, 7], codes, None, None), context)
    assert got == pytest.approx(2 / 3, abs=1e-15)


@pytest.mark.parametrize("value", [-0.0, 0.0, float("nan")])
def test_score_label_float_refusal_is_unchanged(value):
    with pytest.raises((TypeError, ValueError)):
        _pair([value], [0])


def test_label_scalar_hooks_are_refused_before_execution():
    calls = []

    class Scalar:
        def __index__(self):
            calls.append("index")
            return 1

        def __int__(self):
            calls.append("int")
            return 1

        def __str__(self):
            calls.append("str")
            return "1"

    with pytest.raises(TypeError):
        _pair([Scalar()], [1])
    assert calls == []
