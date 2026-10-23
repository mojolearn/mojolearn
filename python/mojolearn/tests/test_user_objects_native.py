# SPDX-License-Identifier: Apache-2.0
"""The native user-object reader (bindings/user_objects_native.mojo) against
the Python each operation replaced."""
import numbers

import pytest

from mojolearn import _user_objects as uo


def _has_entry():
    try:
        from mojolearn.tokenizer import _binding
        return hasattr(_binding()._module, "user_objects_native_call")
    except Exception:  # noqa: BLE001  no host binding in this install
        return False


pytestmark = pytest.mark.skipif(not _has_entry(), reason="the tokenizer host binding has no user-object reader")


def test_take_first_seen_and_lookup():
    import array
    vals = ["b", "a", 3, "b", 3.0, None]
    assert uo.take(vals, array.array("q", [5, 0, -1, 2])) == [None, "b", None, 3]
    with pytest.raises(IndexError):
        uo.take(vals, array.array("q", [6]))
    classes, codes = uo.call("first_seen", vals)
    index = {v: i for i, v in enumerate(dict.fromkeys(vals))}
    assert classes == list(index) and list(codes) == [index[v] for v in vals]
    codes, missing = uo.call("lookup", ["a", "b", "a"], ["a", "z", "b", "z", 7])
    assert list(codes) == [2, -1, 1, -1, -1] and missing == ["z", 7]


def test_kinds_match_isinstance():
    vals = ["s", True, 3, 2.0, 2.5, float("inf"), (1,), [2]]
    got = uo.kinds(vals)
    assert got[uo.KIND_STR] == 1 and got[uo.KIND_BOOL] == 1 and got[uo.KIND_EXACT_INT] == 1
    assert got[uo.KIND_INTEGRAL] == sum(isinstance(v, numbers.Integral) for v in vals)
    assert got[uo.KIND_REAL] == sum(isinstance(v, numbers.Real) for v in vals)
    assert got[uo.KIND_INTVAL] == 3 and got[uo.KIND_SEQ] == 2 and got[uo.KIND_REAL_NONBOOL] == 4


def test_conversions_and_shapes():
    assert list(uo.call("ints", [1, True, 2.9])) == [1, 1, 2]
    assert list(uo.call("floats", (1, 2.5))) == [1.0, 2.5]
    assert uo.call("column", [[1, "a"], [2, "b"]], 1) == ["a", "b"]
    assert uo.call("transpose", [[1, 2, 3], ["a", "b"]]) == [[1, "a"], [2, "b"]]
    assert uo.call("flatten", [[1, (2, 3)], 4]) == [1, 2, 3, 4]
    assert uo.call("nested", [[1, 2], [3, 4]], type(None)) == ((2, 2), [1, 2, 3, 4])
    with pytest.raises(ValueError, match="not rectangular"):
        uo.call("nested", [[1, 2], [3]], type(None))
    assert uo.call("f32_exact", [1, 2.5]) == [1.0, 2.5]
    assert uo.call("f32_exact", [16777217]) is None and uo.call("f32_exact", [True]) is None


def test_sorted_codes_is_the_type_value_order():
    vals = [3, "b", -1, 1, "a", 3]
    order = sorted({v for v in vals if v != -1}, key=lambda v: (str(type(v)), v))
    code = {v: float(i) for i, v in enumerate(order)}
    want = [-1.0 if v == -1 else code[v] for v in vals]
    assert list(uo.call("sorted_codes", vals, -1)) == want
