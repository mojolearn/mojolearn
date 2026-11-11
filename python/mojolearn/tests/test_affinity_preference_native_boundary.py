"""Preference UNPACK compatibility and callback traps, not kernel timing.

The native suite owns conversion arithmetic. These retained Python cases cover
the Python storage formats and their scalar-kind/public-refusal seams.
"""
from array import array
from collections import OrderedDict, defaultdict

import numpy as np
import pytest

import mojolearn as ml

__mojolearn_user_boundary__ = True


def _model(preference):
    return ml.AffinityPropagation(
        affinity="precomputed", preference=preference, damping=0.5,
        max_iter=1, convergence_iter=1, random_state=13,
        numeric_mode="identical",
    )


def _fit(preference):
    similarities = np.array([[0, -4, -36], [-4, 0, -16], [-36, -16, 0]], dtype=np.float32)
    return _model(preference).fit(similarities)


def _assert_preferences(preference, values):
    model = _fit(preference)
    # First responsibility diagonal is (preference - best off-diagonal)/2;
    # availability is zero in the initial iteration. Existing epsilon noise
    # is bounded by the same 1e-4 used by the independent native fixture.
    expected = (np.asarray(values, dtype=np.float64) + [4, 4, 16]) * 0.5
    np.testing.assert_allclose(np.asarray(model._message_diag)[3:], expected, rtol=0, atol=1e-4)
    return model


@pytest.mark.parametrize("kind", [
    "list", "tuple", "range", "bytes", "bytearray", "array", "memoryview",
    "ndarray", "dict", "ordered", "default", "list_iterator", "tuple_iterator",
    "range_iterator", "reverse_iterator",
])
def test_preference_native_iterable_scalar_kinds(kind):
    values = [0, 1, 2]
    sources = {
        "list": values, "tuple": tuple(values), "range": range(3),
        "bytes": bytes(values), "bytearray": bytearray(values),
        "array": array("q", values), "memoryview": memoryview(array("h", values)),
        "ndarray": np.asarray(values, dtype=np.uint64),
        "dict": dict.fromkeys(values), "ordered": OrderedDict.fromkeys(values),
        "default": defaultdict(None, dict.fromkeys(values)),
        "list_iterator": iter(values), "tuple_iterator": iter(tuple(values)),
        "range_iterator": iter(range(3)), "reverse_iterator": reversed([2, 1, 0]),
    }
    _assert_preferences(sources[kind], values)


def test_preference_array_storage_does_not_call_array_iterator_or_tolist(monkeypatch):
    values = ml.Array.from_buffer(array("f", [0, 1, 2]))
    called = []

    def trap(*args, **kwargs):
        called.append("Array conversion")
        raise AssertionError("Python Array conversion executed")

    monkeypatch.setattr(ml.Array, "__iter__", trap, raising=False)
    monkeypatch.setattr(ml.Array, "tolist", trap)
    _assert_preferences(values, [0, 1, 2])
    assert called == []


@pytest.mark.parametrize("kind", ["ordinary", "nullable_int", "nullable_float", "nullable_bool", "categorical", "sparse", "frame"])
def test_preference_pandas_passive_storage_does_not_call_iterators(kind, monkeypatch):
    pd = pytest.importorskip("pandas")
    expected = [0, 1, 2]
    if kind == "ordinary":
        source = pd.Series(expected, dtype="uint64")
    elif kind == "nullable_int":
        source = pd.Series(expected, dtype="Int64")
    elif kind == "nullable_float":
        source = pd.Series(expected, dtype="Float32")
    elif kind == "nullable_bool":
        expected = [False, True, False]
        source = pd.Series(expected, dtype="boolean")
    elif kind == "categorical":
        source = pd.Series(pd.Categorical(expected, categories=[2, 0, 1]))
    elif kind == "sparse":
        source = pd.Series(pd.arrays.SparseArray(expected, fill_value=0))
    else:
        source = pd.DataFrame([["ignored", "cells", "here"]], columns=expected)
    called = []

    def trap(*args, **kwargs):
        called.append("pandas iterator")
        raise AssertionError("Python pandas iteration executed")

    monkeypatch.setattr(pd.Series, "__iter__", trap)
    monkeypatch.setattr(pd.DataFrame, "__iter__", trap)
    monkeypatch.setattr(pd.Index, "__iter__", trap)
    _assert_preferences(source, expected)
    assert called == []


@pytest.mark.parametrize("encoding", ["primitive", "dictionary", "run_end", "union"])
def test_preference_arrow_numeric_encoding(encoding, monkeypatch):
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    if encoding == "primitive":
        data = pa.array([0, 1, 2], type=pa.float32())
    elif encoding == "dictionary":
        data = pa.DictionaryArray.from_arrays(pa.array([1, 2, 0]), pa.array([2, 0, 1]))
    elif encoding == "run_end":
        data = pa.RunEndEncodedArray.from_arrays(pa.array([1, 2, 3], type=pa.int16()), pa.array([0, 1, 2]))
    else:
        data = pa.UnionArray.from_dense(
            pa.array([0, 1, 0], type=pa.int8()), pa.array([0, 0, 1], type=pa.int32()),
            [pa.array([0, 2]), pa.array([1.0])],
        )
    source = pd.Series(pd.arrays.ArrowExtensionArray(data))
    called = []

    def trap(*args, **kwargs):
        called.append("ArrowExtensionArray iterator")
        raise AssertionError("Python Arrow iteration executed")

    monkeypatch.setattr(pd.arrays.ArrowExtensionArray, "__iter__", trap)
    _assert_preferences(source, [0, 1, 2])
    assert called == []


@pytest.mark.parametrize("kind", ["none", "nullable", "nan"])
def test_preference_missing_or_nonfinite_refusal_precedes_wrong_length(kind):
    if kind == "nullable":
        pd = pytest.importorskip("pandas")
        preference = pd.Series([pd.NA], dtype="Float64")
    elif kind == "none":
        preference = [None]
    else:
        preference = [np.nan]
    model = _model(preference)
    with pytest.raises((TypeError, ValueError)) as caught:
        model.fit(np.eye(3, dtype=np.float32))
    assert "one value per sample" not in str(caught.value)
    assert "labels_" not in vars(model)


def test_preference_custom_iterator_is_refused_without_hooks():
    called = []

    class Custom:
        def __iter__(self):
            called.append("iter")
            return iter([0, 1, 2])

        def __len__(self):
            called.append("len")
            return 3

    with pytest.raises(TypeError):
        _fit(Custom())
    assert called == []


@pytest.mark.parametrize("value", [np.float32(1), np.int64(1), np.array(1, dtype=np.float32)])
def test_preference_noniterable_numpy_scalar_keeps_refusal(value):
    with pytest.raises(TypeError):
        _fit(value)
