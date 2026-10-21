# SPDX-License-Identifier: Apache-2.0
"""Existing deterministic NPZ format, observed through the private native owner."""
import io
import pathlib
import zipfile

import numpy as np
import pytest

from mojolearn import LinearRegression, Ridge
from mojolearn._native_classical_bridge import connect
from mojolearn._verify_native_runtime import audit_native_call

ENTRY, CONTEXT = connect(Ridge, LinearRegression)


def native_save(model, path):
    return ENTRY(model, "save", (path,), CONTEXT)


def clean(audit, error=None):
    report = audit.report()
    assert not report["python_frames"]
    assert not report["forbidden_native_calls"]
    assert not report["wrapper_work"]
    assert not report["unobserved_existing_threads"]
    if error is None:
        return audit.assert_native()
    assert isinstance(audit.exception, error)


def fitted(cls):
    model = cls(fit_intercept=True, numeric_mode="identical")
    # Arbitrary exact coefficient bits make the format oracle independent of
    # fit arithmetic. Public save only requires these fitted metadata fields.
    model.coef_ = np.array([0x80000000, 0x3F800001, 0xBF000000], dtype="u4").view("f4")
    model.intercept_ = float.fromhex("0x1.0000000000001p-3")
    model.n_features_in_ = 3
    if cls is Ridge:
        model.alpha = float.fromhex("0x1.0000000000001p-2")
    return model


@pytest.mark.parametrize("cls", [LinearRegression, Ridge])
@pytest.mark.parametrize("destination", ["path", "str", "bytesio", "buffered_file"])
def test_native_save_preserves_complete_legacy_file_bytes(tmp_path, cls, destination):
    model = fitted(cls)
    legacy = io.BytesIO()
    model.save(legacy)
    expected = legacy.getvalue()
    path = tmp_path / "native.npz"
    if destination == "bytesio":
        target = io.BytesIO()
    elif destination == "buffered_file":
        target = path.open("w+b")
    elif destination == "str":
        target = str(path)
    else:
        target = path
    try:
        assert clean(audit_native_call(native_save, model, target)) is target
        if isinstance(target, io.BytesIO):
            assert not target.closed
            actual = target.getvalue()
        else:
            if destination == "buffered_file":
                assert not target.closed
                target.flush()
            actual = path.read_bytes()
    finally:
        if destination == "buffered_file":
            target.close()
    assert actual == expected
    with np.load(io.BytesIO(actual), allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["coef"].view("u4"), model.coef_.view("u4"))
        assert arrays["intercept"].view("u8")[0] == np.float64(model.intercept_).view("u8")
        assert arrays["estimator"][0] == cls.__name__
        np.testing.assert_array_equal(arrays["meta"], [3, 1])
    restored = cls.load(io.BytesIO(actual))
    np.testing.assert_array_equal(np.asarray(restored.coef_).view("u4"), model.coef_.view("u4"))
    assert np.float64(restored.intercept_).view("u8") == np.float64(model.intercept_).view("u8")


@pytest.mark.parametrize("cls", [LinearRegression, Ridge])
def test_native_save_bypasses_python_serializer_and_file_instance_shadows(monkeypatch, cls):
    from mojolearn import _serialize
    model = fitted(cls)
    expected = io.BytesIO()
    model.save(expected)
    target = io.BytesIO()

    def forbidden(*args, **kwargs):
        raise AssertionError("Python archive callback reached")

    target.write = forbidden
    target.tell = forbidden
    target.flush = forbidden
    target.close = forbidden
    with monkeypatch.context() as patch:
        patch.setattr(_serialize, "write_npz", forbidden)
        patch.setattr(zipfile, "ZipFile", forbidden)
        patch.setattr(np, "savez", forbidden)
        patch.setattr(np, "asarray", forbidden)
        assert clean(audit_native_call(native_save, model, target)) is target
    assert target.getvalue() == expected.getvalue()


def test_native_save_preserves_existing_seekable_prefix():
    model = fitted(Ridge)
    expected, actual = io.BytesIO(b"prefix"), io.BytesIO(b"prefix")
    expected.seek(6)
    actual.seek(6)
    model.save(expected)
    clean(audit_native_call(native_save, model, actual))
    assert actual.getvalue() == expected.getvalue()


@pytest.mark.parametrize("cls", [LinearRegression, Ridge])
def test_native_save_unfitted_refuses_before_file_creation(tmp_path, cls):
    path = tmp_path / "unfitted.npz"
    audit = audit_native_call(native_save, cls(), path)
    clean(audit, RuntimeError)
    assert "not fitted" in str(audit.exception)
    assert not path.exists()


@pytest.mark.parametrize("kind", ["pathlike", "path_subclass", "file_subclass", "buffered_python_raw"])
def test_native_save_refuses_python_file_protocols_before_callbacks(tmp_path, kind):
    calls = []

    class Pathlike:
        def __fspath__(self):
            calls.append("fspath")
            raise AssertionError("Python path protocol reached")

    class PathSubclass(type(pathlib.Path())):
        def __fspath__(self):
            calls.append("fspath")
            raise AssertionError("Python path subclass reached")

    class FileSubclass(io.BytesIO):
        def write(self, value):
            calls.append("write")
            raise AssertionError("Python file subclass reached")

    class PythonRaw(io.RawIOBase):
        def writable(self):
            return True

        def write(self, value):
            calls.append("write")
            raise AssertionError("Python raw-file callback reached")

    target = {"pathlike": lambda: Pathlike(), "path_subclass": lambda: PathSubclass(tmp_path / "bad.npz"),
              "file_subclass": FileSubclass, "buffered_python_raw": lambda: io.BufferedWriter(PythonRaw())}[kind]()
    clean(audit_native_call(native_save, fitted(Ridge), target), TypeError)
    assert calls == []
    if kind == "buffered_python_raw":
        target.close()
