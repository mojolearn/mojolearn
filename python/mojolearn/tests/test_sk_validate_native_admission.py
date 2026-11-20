# SPDX-License-Identifier: Apache-2.0
"""Boundary tests for the native X and y admission of `_sk_validate`.

`_check_X` and `_check_y` are one native call each
(bindings/labels_admission_unpack.mojo `admission_x_op` / `admission_y_op`):
the input is described from its own export (shape and format, never its
values) and refused in scikit-learn's words. These tests pin the description
of every input container and dtype, non-contiguous and read-only buffers,
the refusals and their exception types, the returned objects, and that no
Python buffer export (`_buffer.Buf`) runs on the way.
"""
import warnings

import numpy as np
import pytest

from mojolearn import _buffer, _sk_validate
from mojolearn._array import Array
from mojolearn.linear_model import LogisticRegression, Ridge
from mojolearn._sk_validate import _check_X, _check_y, describe


@pytest.fixture(autouse=True)
def no_python_buffer_export(monkeypatch):
    """The description is native: the Python buffer export is never entered."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Python buffer export entered")
    monkeypatch.setattr(_buffer, "Buf", forbidden)
    monkeypatch.setattr(_buffer, "_device_array_kind", forbidden)


# ---------------------------------------------------------------- describe


@pytest.mark.parametrize("dtype, kind", [
    (np.float32, "float"), (np.float64, "float"), (np.float16, "float"),
    (np.int8, "int"), (np.int32, "int"), (np.int64, "int"), (np.uint8, "int"),
    (np.uint64, "int"), (np.bool_, "int"), (np.complex64, "complex"),
    (np.complex128, "complex"),
])
def test_describe_buffer_dtypes(dtype, kind):
    x = np.zeros((4, 3), dtype=dtype)
    assert describe(x) == ("buffer", (4, 3), kind)


def test_describe_object_and_text_buffers_have_no_kind():
    assert describe(np.array([[1, "a"]], dtype=object)) == ("buffer", (1, 2), None)
    assert describe(np.array(["a", "b"])) == ("buffer", (2,), None)


def test_describe_noncontiguous_readonly_and_scalar_buffers():
    x = np.arange(60, dtype=np.float64).reshape(6, 10)
    assert describe(x[::2, ::3]) == ("buffer", (3, 4), "float")
    assert describe(np.asfortranarray(x)) == ("buffer", (6, 10), "float")
    ro = x.copy()
    ro.flags.writeable = False
    assert describe(ro) == ("buffer", (6, 10), "float")
    assert describe(np.float64(1.0).reshape(())) == ("buffer", (), "float")
    assert describe(x.astype(">f8")) == ("buffer", (6, 10), "float")


def test_describe_other_buffer_exporters():
    import array
    assert describe(bytearray(5)) == ("buffer", (5,), "int")
    assert describe(memoryview(np.zeros((2, 2), np.float32))) == ("buffer", (2, 2), "float")
    assert describe(array.array("d", [1.0, 2.0])) == ("buffer", (2,), "float")


def test_describe_mojolearn_array():
    assert describe(Array.from_list([1.0, 2.0], "<f4")) == ("buffer", (2,), "float")
    assert describe(Array.from_list([1, 2], "<i8")) == ("buffer", (2,), "int")


def test_describe_lists_and_tuples():
    assert describe([[1.0, 2.0], [3.0, 4.0]]) == ("list", (2, 2), "float")
    assert describe(((1, 2), (3, 4), (5, 6))) == ("list", (3, 2), "int")
    assert describe([True, False]) == ("list", (2,), "int")
    assert describe([]) == ("list", (0,), "int")
    assert describe([[], []]) == ("list", (2, 0), "int")
    assert describe([[1j]]) == ("list", (1, 1), "complex")
    assert describe([["a", "b"]]) == ("list", (1, 2), None)
    assert describe([np.float32(1.0)]) == ("list", (1,), "float")
    assert describe([np.int64(1)]) == ("list", (1,), "int")


def test_describe_not_tabular():
    for value in (None, "abc", b"abc", {"a": 1}, 3.0, object()):
        assert describe(value) is None


class _Frame:
    """No buffer, a `shape` tuple of ints: a frame."""
    shape = (3, 2)


class _Device:
    """A device array announces itself; its memory is never read."""
    shape = (4, 2)
    __cuda_array_interface__ = {}


class _DLPackDevice(_Frame):
    def __dlpack_device__(self):
        return (2, 0)


class _BadShape:
    shape = (3, "2")


def test_describe_frames_and_device_arrays():
    assert describe(_Frame()) == ("frame", (3, 2), None)
    assert describe(_Device()) == ("frame", (4, 2), None)
    assert describe(_DLPackDevice()) == ("frame", (3, 2), None)
    assert describe(_BadShape()) is None


def test_describe_sparse():
    sparse = pytest.importorskip("scipy.sparse")
    m = sparse.csr_matrix(np.eye(3, dtype=np.float32))
    assert describe(m) == ("sparse", (3, 3), None)


class _Raises:
    @property
    def tocsr(self):
        raise RuntimeError("probe failed")


def test_describe_keeps_a_probe_exception_type():
    with pytest.raises(RuntimeError, match="probe failed"):
        describe(_Raises())


# ---------------------------------------------------------------- X


def test_check_x_returns_the_description_shape():
    x = np.ones((5, 3), np.float32)
    info = _check_X(Ridge(), x)
    assert info[1] == (5, 3)
    assert _check_X(Ridge(), x[::2]) [1] == (3, 3)
    assert _check_X(Ridge(), [[1.0, 2.0]])[1] == (1, 2)
    assert _check_X(Ridge(), [["a", "b"]]) is None


@pytest.mark.parametrize("x, message", [
    (np.ones(3), r"Expected 2D array, got 1D array instead \(shape=\(3,\)\)"),
    (np.float64(1.0).reshape(()), r"Expected 2D array, got scalar array instead"),
    (np.ones((0, 3)), r"Found array with 0 sample\(s\) \(shape=\(0, 3\)\) while a minimum of 1 is required by Ridge"),
    (np.ones((3, 0)), r"Found array with 0 feature\(s\) \(shape=\(3, 0\)\) while a minimum of 1 is required by Ridge"),
    (np.ones((3, 2), np.complex128), r"Complex data not supported: Ridge got a complex X"),
    (np.array([[1.0, np.nan]]), r"Input X contains NaN or infinity \(Ridge\)"),
    (np.array([[1.0, np.inf]], np.float32)[:, ::-1], r"Input X contains NaN or infinity \(Ridge\)"),
    ([[1.0, float("nan")]], r"Input X contains NaN or infinity \(Ridge\)"),
])
def test_check_x_refusals(x, message):
    with pytest.raises(ValueError, match=message):
        _check_X(Ridge(), x)


def test_check_x_readonly_and_integer_inputs_pass():
    x = np.arange(12, dtype=np.float64).reshape(4, 3)
    x.flags.writeable = False
    assert _check_X(Ridge(), x)[1] == (4, 3)
    assert _check_X(Ridge(), np.ones((2, 2), np.int32))[1] == (2, 2)


def test_check_x_refuses_sparse_for_a_dense_estimator():
    sparse = pytest.importorskip("scipy.sparse")
    with pytest.raises(TypeError, match=r"Sparse data was passed for X, but dense data is required.*\(Ridge takes dense X only.\)"):
        _check_X(Ridge(), sparse.csr_matrix(np.eye(2)))


def test_check_x_tag_words_are_cached_per_class():
    _check_X(Ridge(), np.ones((2, 2)))
    words = _sk_validate._x_words(Ridge)
    assert words[0] is Ridge.__mro__
    assert words[1] & _sk_validate._X_SCAN and words[1] & _sk_validate._X_TABULAR
    assert not words[1] & _sk_validate._X_SPARSE_OK


# ---------------------------------------------------------------- y


def test_check_y_passes_and_returns_y():
    y = np.arange(4, dtype=np.float64)
    assert _check_y(Ridge(), y, 4) is y
    assert _check_y(LogisticRegression(), [0, 1, 0, 1], 4) == [0, 1, 0, 1]


@pytest.mark.parametrize("est, y, rows, message", [
    (Ridge(), np.array([1.0, np.nan]), 2, r"Input y contains NaN or infinity \(Ridge\)"),
    (Ridge(), np.ones(3), 2, r"Found input variables with inconsistent numbers of samples: \[2, 3\]"),
    (LogisticRegression(), np.array([0.5, 1.0]), 2, r"Unknown label type: continuous"),
    (Ridge(), np.ones(2, np.complex128), 2, r"Complex data not supported: Ridge got a complex y"),
])
def test_check_y_refusals(est, y, rows, message):
    with pytest.raises(ValueError, match=message):
        _check_y(est, y, rows)


def test_check_y_column_vector_warns_and_ravels_before_refusing():
    y = np.ones((3, 1))
    with pytest.warns(Warning, match="A column-vector y was passed"):
        out = _check_y(Ridge(), y, 3)
    assert out.shape == (3,)
    bad = np.array([[1.0], [np.nan]])
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Input y contains NaN"):
            _check_y(Ridge(), bad, 2)
    assert any("column-vector" in str(w.message) for w in seen)


def test_check_y_column_list_is_not_raveled():
    y = [[1.0], [2.0]]
    assert _check_y(Ridge(), y, 2) is y


def test_check_y_missing_target():
    with pytest.raises(ValueError, match="requires y to be passed"):
        _check_y(Ridge(), None)


# ---------------------------------------------------------------- public calls


def test_public_fit_and_predict_go_through_the_native_admission():
    x = np.random.default_rng(0).random((20, 3))
    y = x @ np.array([1.0, -2.0, 0.5])
    model = Ridge().fit(x[:, ::1], y)
    assert model.n_features_in_ == 3
    assert model.predict(x[::2]).shape == (10,)
    with pytest.raises(ValueError, match="X has 2 features, but Ridge is expecting 3 features as input."):
        model.predict(x[:, :2])
