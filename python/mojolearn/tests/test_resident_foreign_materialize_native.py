# SPDX-License-Identifier: Apache-2.0
"""_Resident.take and DeviceRows.materialize on foreign fold containers.

Lists, tuples, ranges, memoryview/array.array, pandas and Arrow-backed
pandas values are materialized by the native model-selection input owner,
not by the Python `_buffer._materialize` step: the whole take is one native
call, the original dtype is kept (float64 that float32 would round, int64
beyond 2**24), and caller hooks (__array__, __iter__, __float__, DLPack)
are refused before they can run.
"""
import array
import gc

import numpy as np
import pytest

from mojolearn import _buffer
from mojolearn._arena_io import DeviceRows
from mojolearn._buffer import _native
from mojolearn.model_selection import _Resident
from mojolearn._verify_native_runtime import audit_native_call

INDICES = np.array([2, 0], dtype="i8")
BIG = 2 ** 24 + 1


def _live():
    return int(_native("msel_live")())


@pytest.fixture
def owner():
    before = _live()
    result = _Resident()
    try:
        yield result
    finally:
        result.close()
        gc.collect()
        assert _live() == before


@pytest.fixture(autouse=True)
def no_python_materializer(monkeypatch):
    calls = []
    original = _buffer._materialize

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(_buffer, "_materialize", spy)
    yield
    assert not calls


def _rows(value):
    return np.asarray(value.tolist())


@pytest.mark.parametrize("value,dtype", [
    ([[0.1, 1.0], [2.0, 3.0], [4.0, 5.5]], "<f8"),
    (([BIG, 1], [2, 3], [4, 5]), "<i8"),
    ([[True, False], [False, True], [True, True]], "<i8"),
    (range(3), "<i8"),
])
def test_take_on_sequences_is_native_and_keeps_natural_dtype(owner, value, dtype):
    rows = audit_native_call(owner.take, value, INDICES).assert_native()
    expected = np.asarray(value if not isinstance(value, range) else list(value), dtype=dtype)[[2, 0]]
    assert rows.dtype == dtype and np.array_equal(_rows(rows), expected)
    assert owner._entries[id(value)][0] is value


def test_take_on_array_array_and_memoryview_is_native(owner):
    store = array.array("d", [0.1, 1.0, 2.0, 3.0, 4.0, 5.0])
    view = memoryview(store).cast("B").cast("d", (3, 2))
    rows = audit_native_call(owner.take, view, INDICES).assert_native()
    assert rows.dtype == "<f8" and np.array_equal(_rows(rows), np.asarray(store).reshape(3, 2)[[2, 0]])


@pytest.mark.parametrize("kind", ["pandas_f64", "pandas_i64", "arrow_f64", "arrow_i64", "arrow_bool",
                                  "arrow_string_index", "arrow_chunked"])
def test_take_on_pandas_and_arrow_is_native_and_dtype_preserving(owner, kind):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [0.1, 1.0, 2.0], "b": [3.0, 4.0, 5.5]})
    if kind == "pandas_i64":
        frame = pd.DataFrame({"a": [BIG, 1, 2], "b": [3, 4, 5]})
    if kind.startswith("arrow"):
        pa = pytest.importorskip("pyarrow")
        if kind == "arrow_f64":
            frame = frame.astype("float64[pyarrow]")
        elif kind == "arrow_i64":
            frame = pd.DataFrame({"a": [BIG, 1, 2], "b": [3, 4, 5]}).astype("int64[pyarrow]")
        elif kind == "arrow_bool":
            frame = pd.DataFrame({"a": [True, False, True]}).astype("bool[pyarrow]")
        elif kind == "arrow_chunked":
            column = pa.chunked_array([[0.1, 1.0], [2.0]])
            frame = pd.DataFrame({"a": pd.arrays.ArrowExtensionArray(column)})
        else:
            frame.index = pd.Index(pd.array(["w", "x", "y"], dtype="string[pyarrow]"))
    expected = np.asarray(frame)[[2, 0]]
    rows = audit_native_call(owner.take, frame, INDICES).assert_native()
    if kind == "arrow_bool":
        # A bool block takes the established Array bool width (uint8).
        assert np.dtype(rows.dtype).kind in "bu"
        assert np.array_equal(_rows(rows).astype(bool), expected.astype(bool))
        return
    want = "<i8" if kind.endswith("i64") else "<f8"
    assert rows.dtype == want
    assert np.array_equal(_rows(rows), expected.astype(want))


def test_arrow_missing_float_reads_nan_and_missing_integer_is_refused(owner):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    floats = pd.DataFrame({"a": [1.0, None, 3.0]}).astype("float64[pyarrow]")
    rows = audit_native_call(owner.take, floats, np.array([1, 2], dtype="i8")).assert_native()
    assert rows.dtype == "<f8" and np.isnan(_rows(rows)[0, 0]) and _rows(rows)[1, 0] == 3.0
    ints = pd.DataFrame({"a": [1, None, 3]}).astype("int64[pyarrow]")
    audit = audit_native_call(owner.take, ints, INDICES)
    assert type(audit.exception) is TypeError and not audit.frames
    assert id(ints) not in owner._entries


@pytest.mark.parametrize("hook", ["__array__", "__iter__", "__float__", "__dlpack__", "__len__"])
def test_foreign_hooks_are_refused_before_they_run(owner, hook):
    calls = []

    def poisoned(self, *args, **kwargs):
        calls.append(hook)
        raise AssertionError("caller hook ran")

    value = type("Foreign", (), {hook: poisoned})()
    audit = audit_native_call(owner.take, value, INDICES)
    assert audit.exception is not None and not calls and not audit.frames
    assert id(value) not in owner._entries


def test_device_rows_materialize_on_a_float64_base_is_native(owner):
    # DeviceRows reads its shape and dtype from exact Array/ndarray metadata
    # (54cd01bff, bindings/arena_array_metadata.mojo); a list base has none
    # and is refused at construction. model_selection only builds DeviceRows
    # over admitted arrays. The materialize step keeps the float64 dtype.
    listed = [[0.1, 1.0], [2.0, 3.0], [4.0, 5.5]]
    with pytest.raises(TypeError, match="DeviceRows requires ordinary Array or ndarray metadata"):
        DeviceRows(listed, np.array([2, 0], dtype="i8"), owner.take)
    base = np.asarray(listed, dtype="<f8")
    rows = DeviceRows(base, np.array([2, 0], dtype="i8"), owner.take)
    output = audit_native_call(rows.materialize).assert_native()
    assert output.dtype == "<f8" and np.array_equal(_rows(output), np.asarray(base)[[2, 0]])
    assert audit_native_call(rows.materialize).assert_native() is output


def test_upload_is_reused_for_the_same_foreign_owner(owner):
    base = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    owner.take(base, INDICES)
    live = _live()
    entry = owner._entries[id(base)]
    audit_native_call(owner.take, base, np.array([1], dtype="i8")).assert_native()
    assert owner._entries[id(base)] is entry and _live() == live + 1
