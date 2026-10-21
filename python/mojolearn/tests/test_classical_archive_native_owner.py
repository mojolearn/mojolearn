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


def native_load(cls, path):
    return ENTRY(cls, "load", (path,), CONTEXT)


def native_predict(model, x):
    return ENTRY(model, "predict", (x,), CONTEXT)


def payloads(cls):
    model = fitted(cls)
    values = {"format": np.array(["mojolearn-linear-1"]), "estimator": np.array([cls.__name__]),
              "numeric_mode": np.array(["identical"]), "coef": model.coef_,
              "intercept": np.array([model.intercept_], dtype="f8"), "meta": np.array([3, 1], dtype="i8")}
    if cls is Ridge:
        values["alpha"] = np.array([model.alpha], dtype="f8")
    return values


def archive(values, compression=zipfile.ZIP_STORED, version=(1, 0), force_zip64=False):
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=compression) as writer:
        for name, value in values.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, value, version=version, allow_pickle=False)
            with writer.open(name + ".npy", "w", force_zip64=force_zip64) as output:
                output.write(member.getvalue())
    return result.getvalue()


_COMPRESSIONS = [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA]
if hasattr(zipfile, "ZIP_ZSTANDARD"):
    _COMPRESSIONS.append(zipfile.ZIP_ZSTANDARD)


@pytest.mark.parametrize("cls", [LinearRegression, Ridge])
@pytest.mark.parametrize("compression", _COMPRESSIONS)
def test_native_load_preserves_standard_archive_codecs_and_consumed_prediction(cls, compression):
    raw = archive(payloads(cls), compression)
    expected = cls.load(io.BytesIO(raw))
    target = io.BytesIO(raw)
    got = clean(audit_native_call(native_load, cls, target))
    assert not target.closed
    assert type(got) is cls
    assert got.get_params() == expected.get_params()
    np.testing.assert_array_equal(np.asarray(got.coef_).view("u4"), np.asarray(expected.coef_).view("u4"))
    assert np.float64(got.intercept_).view("u8") == np.float64(expected.intercept_).view("u8")
    x = np.array([[1, 2, 3], [-1, 0, 1], [0.5, -0.25, 0]], dtype="f4")
    actual = clean(audit_native_call(native_predict, got, x))
    np.testing.assert_array_equal(np.asarray(actual).view("u4"), np.asarray(expected.predict(x)).view("u4"))


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("fortran", [False, True])
def test_native_load_preserves_npy_versions_and_decodes_extra_members(version, fortran):
    values = payloads(Ridge)
    # read_npz validates every member, including unrelated numeric/text arrays;
    # both C/F extra shapes exercise the native metadata decoder independently.
    values["extra_numeric"] = np.arange(6, dtype="f2").reshape(2, 3)
    values["extra_text"] = np.array([["α", "", "hello"], ["z", "β", "bye"]])
    values["extra_bytes"] = np.array([[b"a", b"b"], [b"c", b""]])
    values["extra_boolean"] = np.array([[True, False], [False, True]])
    if fortran:
        values = {name: np.asfortranarray(value) if value.ndim == 2 else value for name, value in values.items()}
    raw = archive(values, version=version, force_zip64=True)
    expected = Ridge.load(io.BytesIO(raw))
    actual = clean(audit_native_call(native_load, Ridge, io.BytesIO(raw)))
    np.testing.assert_array_equal(np.asarray(actual.coef_).view("u4"), np.asarray(expected.coef_).view("u4"))


@pytest.mark.parametrize("kind", ["format", "mode", "estimator", "coef_dtype", "coef_shape", "features",
                                   "meta_dtype", "meta_length", "bias_dtype", "bias_length", "alpha_dtype",
                                   "alpha_length", "missing", "extra_dtype"])
def test_native_load_preserves_schema_refusals_and_validates_extra_members(kind):
    values = payloads(Ridge)
    if kind == "format":
        values["format"] = np.array(["other"])
    elif kind == "mode":
        values["numeric_mode"] = np.array(["unknown"])
    elif kind == "estimator":
        values["estimator"] = np.array(["LinearRegression"])
    elif kind == "coef_dtype":
        values["coef"] = values["coef"].astype("f8")
    elif kind == "coef_shape":
        values["coef"] = values["coef"].reshape(1, 3)
    elif kind == "features":
        values["meta"][0] = 2
    elif kind == "meta_dtype":
        values["meta"] = values["meta"].astype("i4")
    elif kind == "meta_length":
        values["meta"] = np.array([3], dtype="i8")
    elif kind == "bias_dtype":
        values["intercept"] = values["intercept"].astype("f4")
    elif kind == "bias_length":
        values["intercept"] = np.array([1, 2], dtype="f8")
    elif kind == "alpha_dtype":
        values["alpha"] = values["alpha"].astype("f4")
    elif kind == "alpha_length":
        values["alpha"] = np.array([1, 2], dtype="f8")
    elif kind == "missing":
        del values["coef"]
    elif kind == "extra_dtype":
        values["unused"] = np.array([1 + 2j], dtype="c8")
    raw = archive(values)
    with pytest.raises(ValueError):
        Ridge.load(io.BytesIO(raw))
    clean(audit_native_call(native_load, Ridge, io.BytesIO(raw)), ValueError)


@pytest.mark.parametrize("corruption", ["crc", "payload_truncated", "bad_header", "overlap", "missing_footer"])
def test_native_load_checks_zip_framing_before_publishing_a_model(corruption):
    raw = bytearray(archive(payloads(Ridge)))
    with zipfile.ZipFile(io.BytesIO(raw)) as reader:
        entry = reader.getinfo("coef.npy")
        offset = entry.header_offset
        data = offset + 30 + len(entry.filename) + len(entry.extra)
    if corruption == "crc":
        raw[data + entry.file_size - 1] ^= 1
    elif corruption == "payload_truncated":
        del raw[data + entry.file_size - 2:data + entry.file_size]
    elif corruption == "bad_header":
        raw[offset] = 0
    elif corruption == "overlap":
        central = raw.find(b"PK\x01\x02")
        while raw[central + 46:central + 54] != b"coef.npy":
            central = raw.find(b"PK\x01\x02", central + 4)
            assert central >= 0
        raw[central + 20:central + 24] = (len(raw)).to_bytes(4, "little")
    else:
        del raw[-22:]
    with pytest.raises((ValueError, zipfile.BadZipFile, EOFError)):
        Ridge.load(io.BytesIO(raw))
    clean(audit_native_call(native_load, Ridge, io.BytesIO(raw)), ValueError)


def test_native_load_preserves_duplicate_last_entry_and_prepended_bytes():
    raw = io.BytesIO(archive(payloads(Ridge)))
    with zipfile.ZipFile(raw, "a") as writer:
        value = io.BytesIO()
        np.lib.format.write_array(value, np.array([2.5], dtype="f8"), allow_pickle=False)
        with pytest.warns(UserWarning, match="Duplicate name"):
            writer.writestr("intercept.npy", value.getvalue())
    prefixed = b"external-prefix" + raw.getvalue()
    expected = Ridge.load(io.BytesIO(prefixed))
    actual = clean(audit_native_call(native_load, Ridge, io.BytesIO(prefixed)))
    assert actual.intercept_ == expected.intercept_ == 2.5


@pytest.mark.parametrize("cls", [LinearRegression, Ridge])
def test_native_load_bypasses_python_helpers_and_instance_file_shadows(monkeypatch, cls):
    from mojolearn import _serialize
    raw = archive(payloads(cls), zipfile.ZIP_DEFLATED)
    target = io.BytesIO(raw)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python archive loading callback reached")

    target.read = forbidden
    target.seek = forbidden
    target.tell = forbidden
    target.close = forbidden
    with monkeypatch.context() as patch:
        patch.setattr(_serialize, "read_npz", forbidden)
        patch.setattr(_serialize, "decode_npy", forbidden)
        patch.setattr(zipfile, "ZipFile", forbidden)
        patch.setattr(np, "load", forbidden)
        patch.setattr(np, "frombuffer", forbidden)
        got = clean(audit_native_call(native_load, cls, target))
    assert type(got) is cls
    assert got.n_features_in_ == 3
    assert not target.closed
