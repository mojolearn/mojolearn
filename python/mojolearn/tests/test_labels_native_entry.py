"""Lane ne1-h-labels: the one label entry (`labels_native_call`,
bindings/labels_native_entry.mojo) gives the answers of the shell steps it
replaced: `encode_labels` the ORDER RULE classes and codes of
`sorted_classes(flatten_labels(y))`, the metric pair the per-side encodes and
the sorted union, a classifier score from predicted codes the accuracy of the
decoded prediction, bit for bit.

Skips when the runtime classical host binding is not built.
"""
import numpy as np
import pytest

from mojolearn._array import Array
from mojolearn import _labels

try:
    from mojolearn._native_labels_bridge import (
        labels_entry, LABELS_OP_SCORE_FROM_CODES, LABELS_OP_UNION_ENCODE)
    labels_entry()
except (ImportError, AttributeError, TypeError) as exc:  # pragma: no cover - build state
    pytest.skip(f"labels_native_call is not built: {exc}", allow_module_level=True)


def _reference(y):
    return _labels.sorted_classes(_labels.flatten_labels(y))


CASES = [
    [3, 1, 2, 1, 3],
    [2.5, -0.0, 0.0, 2.5],
    ["b", "a", "c", "a"],
    ("x", "y", "x"),
    [[1], [0], [1]],
    [True, False, True],
    np.array([4, 2, 2, 9], dtype=np.int64),
    np.array([1.5, 0.5, 1.5], dtype=np.float32),
    np.array(["q", "p", "q"]),
    np.array([True, False, True]),
    np.array(["u", "t", "u"], dtype=object),
]


@pytest.mark.parametrize("y", CASES, ids=range(len(CASES)))
def test_encode_matches_order_rule(y):
    classes, codes = _labels.encode_labels(y)
    ref_classes, ref_codes = _reference(y)
    assert list(classes) == list(ref_classes)
    assert [type(c) for c in classes] == [type(c) for c in ref_classes]
    assert codes.dtype == "<i4"
    assert codes.tolist() == list(ref_codes)


@pytest.mark.parametrize("y", [[1, "a"], [float("nan"), 1.0], [None, 1], []])
def test_encode_refusals_match(y):
    with pytest.raises(ValueError):
        _labels.encode_labels(y)
    with pytest.raises(ValueError):
        _reference(y)


def test_decode_matches_python():
    for classes in ([3, 7, 11], [0.5, 2.5], ["a", "b", "c"], [False, True]):
        codes = Array.from_list([1, 0, len(classes) - 1, 1], "<i4")
        got = _labels.decode_labels(classes, codes)
        want = [classes[c] for c in (1, 0, len(classes) - 1, 1)]
        values = got.tolist() if isinstance(got, Array) else list(got)
        assert values == want
        assert [type(v) for v in values] == [type(v) for v in want]


def test_union_encode():
    entry, context = labels_entry()
    tc, tcodes, pc, pcodes, kind, present = entry(
        LABELS_OP_UNION_ENCODE, (["b", "a", "b"], ["c", "a", "a"], "confusion_matrix"), context)
    assert kind == "string"
    assert list(tc) == ["a", "b"] and tcodes.tolist() == [1, 0, 1]
    assert list(pc) == ["a", "c"] and pcodes.tolist() == [1, 0, 0]
    assert list(present) == ["a", "b", "c"]
    with pytest.raises(TypeError):
        entry(LABELS_OP_UNION_ENCODE, ([1, 2], ["a", "b"], "confusion_matrix"), context)
    with pytest.raises(ValueError):
        entry(LABELS_OP_UNION_ENCODE, ([1, 2], [1], "confusion_matrix"), context)


@pytest.mark.parametrize("classes,y", [
    (["cat", "dog"], ["dog", "cat", "cat", "bird"]),
    ([2, 5, 9], np.array([5, 5, 2, 7], dtype=np.int64)),
])
@pytest.mark.parametrize("weighted", [False, True])
def test_score_from_codes_same_bits(classes, y, weighted):
    from mojolearn._expansion_metrics import accuracy_fraction
    entry, context = labels_entry()
    codes = Array.from_list([1, 0, 0, 1], "<i4")
    w = Array.from_list([0.5, 1.25, 2.0, 0.75], "<f4") if weighted else None
    got = entry(LABELS_OP_SCORE_FROM_CODES, (y, classes, codes, w, None), context)
    want = accuracy_fraction(y, _labels.decode_labels(classes, codes), w)
    assert np.float64(got).tobytes() == np.float64(want).tobytes()


@pytest.mark.parametrize("bad", [2, 7])
def test_score_from_codes_out_of_range_raises_like_decode(bad):
    entry, context = labels_entry()
    classes = ["cat", "dog"]
    codes = Array.from_list([1, 0, bad, 1], "<i4")
    with pytest.raises(IndexError):
        _labels.decode_labels(classes, codes)
    with pytest.raises(IndexError):
        entry(LABELS_OP_SCORE_FROM_CODES, (["dog", "cat", "cat", "dog"], classes, codes, None, None), context)
