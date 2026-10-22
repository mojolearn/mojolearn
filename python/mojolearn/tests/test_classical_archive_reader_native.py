# SPDX-License-Identifier: Apache-2.0
"""Public model loads retain archive state and refuse runtime Python hooks."""
import builtins
import dis
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from mojolearn import LinearRegression, Ridge
from mojolearn._classical_host import HostLinearRegression, HostRidge
from mojolearn._verify_native_runtime import audit_native_call
import mojolearn._verify_native_runtime as observer


def _payload(compression=zipfile.ZIP_STORED, **changes):
    values = dict(format="mojolearn-linear-1", estimator="HostLinearRegression",
                  numeric_mode="identical", coef=np.array([.75, -.5, .125], "<f4"),
                  intercept=np.array([.25], "<f8"), meta=np.array([3, 1], "<i8"),
                  extra_bytes=np.array([b"a", b"b"], "S4"))
    values.update(changes)
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=compression) as archive:
        for name, value in values.items():
            member = io.BytesIO()
            np.save(member, value, allow_pickle=False)
            archive.writestr(name + ".npy", member.getvalue())
    return result.getvalue()


def _input(tmp_path, payload, kind):
    path = tmp_path / "model.npz"
    path.write_bytes(payload)
    if kind == "str":
        return str(path)
    if kind == "bytes":
        return os.fsencode(path)
    if kind == "path":
        return path
    if kind == "pure_path":
        return PurePosixPath(path)
    if kind == "bytesio":
        return io.BytesIO(payload)
    if kind == "fileio":
        return io.FileIO(path, "rb")
    return io.open(path, "rb")


def _close(value):
    if type(value) in (io.BytesIO, io.FileIO, io.BufferedReader):
        value.close()


def _state(model):
    assert np.asarray(model.coef_, "<f4").tobytes() == np.array([.75, -.5, .125], "<f4").tobytes()
    assert np.asarray([model.intercept_], "<f8").tobytes() == np.array([.25], "<f8").tobytes()
    assert model.n_features_in_ == 3
    assert model.fit_intercept is True
    assert model.numeric_mode == "identical"


def _refused(audit, category):
    report = audit.report()
    assert not report["python_frames"], report["python_frames"]
    assert not report["forbidden_native_calls"], report["forbidden_native_calls"]
    assert not report["wrapper_work"], report["wrapper_work"]
    assert not report["unobserved_existing_threads"]
    assert isinstance(audit.exception, category), repr(audit.exception)
    assert audit.result is None


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED,
                                         zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
@pytest.mark.parametrize("kind", ["str", "bytes", "path", "pure_path", "bytesio", "fileio", "buffered"])
def test_public_load_preserves_compressed_state_and_input_ownership(tmp_path, compression, kind):
    value = _input(tmp_path, _payload(compression), kind)
    try:
        model = audit_native_call(HostLinearRegression.load, value).assert_native()
        if type(value) in (io.BytesIO, io.FileIO, io.BufferedReader):
            assert not value.closed
            assert value.tell() > 0
    finally:
        _close(value)
    # Output storage survives closing the input and consuming all learned state.
    _state(model)


@pytest.mark.parametrize("name", ["open", "sorted", "list", "int", "tuple", "memoryview",
                                  "bytes", "bytearray", "chr", "str", "object", "type"])
def test_public_load_ignores_replaced_builtin_helpers(tmp_path, monkeypatch, name):
    value = _input(tmp_path, _payload(), "str" if name == "open" else "bytesio")
    loader = HostLinearRegression.load
    original = getattr(builtins, name)
    touched = []
    def foreign(*args, **kwargs):
        touched.append(name)
        return original(*args, **kwargs)
    try:
        with monkeypatch.context() as patch:
            # Isolate only test-controller lookups. The product receives the
            # genuinely replaced builtin throughout the public load call.
            patch.setattr(dis, name, original, raising=False)
            patch.setattr(observer, name, original, raising=False)
            patch.setattr(builtins, name, foreign)
            audit = audit_native_call(loader, value)
        model = audit.assert_native()
        assert touched == []
    finally:
        _close(value)
    _state(model)


@pytest.mark.parametrize("kind", ["str", "bytes", "path", "pure_path", "bytesio", "fileio", "buffered"])
def test_public_load_error_repr_preserves_standard_names_without_python(tmp_path, monkeypatch, kind):
    value = _input(tmp_path, _payload(estimator="Bogus"), kind)
    expected = f"mojolearn: {value!r} was saved by Bogus, not HostLinearRegression"
    loader = HostLinearRegression.load
    original = repr
    touched = []
    def foreign(*args, **kwargs):
        touched.append("repr")
        return original(*args, **kwargs)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(dis, "repr", original, raising=False)
            patch.setattr(observer, "repr", original, raising=False)
            patch.setattr(builtins, "repr", foreign)
            audit = audit_native_call(loader, value)
        _refused(audit, ValueError)
        assert str(audit.exception) == expected
        assert touched == []
        if type(value) in (io.BytesIO, io.FileIO, io.BufferedReader):
            assert not value.closed
    finally:
        _close(value)


@pytest.mark.parametrize("kind", ["missing", "directory"])
@pytest.mark.parametrize("path_kind", ["str", "path", "pure_path", "bytes"])
def test_public_load_retains_file_exception_attributes(tmp_path, kind, path_kind):
    # The pre-native _serialize.read_npz at 1af843f4e^ entered ZipFile directly:
    # str and standard Path inputs retain the original filesystem exception.
    # Bytes paths were not accepted by ZipFile. The accepted native loader
    # added them with UTF-8/surrogateescape normalization, which stays intact.
    path = tmp_path / "missing.npz" if kind == "missing" else tmp_path
    value = {"str": str(path), "path": path, "pure_path": PurePosixPath(path),
             "bytes": os.fsencode(path)}[path_kind]
    if path_kind == "bytes":
        oracle = lambda: io.open(os.fsdecode(value), "rb")
    else:
        oracle = lambda: zipfile.ZipFile(value, "r")
    with pytest.raises(OSError) as expected:
        oracle()
    audit = audit_native_call(HostLinearRegression.load, value)
    _refused(audit, type(expected.value))
    assert type(audit.exception) is type(expected.value)
    assert str(audit.exception) == str(expected.value)
    for name in ("errno", "filename", "filename2"):
        actual_value = getattr(audit.exception, name, None)
        expected_value = getattr(expected.value, name, None)
        assert type(actual_value) is type(expected_value)
        assert actual_value == expected_value


def test_public_load_preserves_closed_input_exception():
    old, actual = io.BytesIO(), io.BytesIO()
    old.close(); actual.close()
    with pytest.raises(ValueError) as expected:
        zipfile.ZipFile(old, "r")
    audit = audit_native_call(HostLinearRegression.load, actual)
    _refused(audit, ValueError)
    assert str(audit.exception) == str(expected.value)


@pytest.mark.parametrize("cls", [Path, PurePosixPath])
def test_public_load_filesystem_errors_use_standard_path_lexical_spelling(tmp_path, cls):
    value = cls(str(tmp_path) + "/.//missing/../child.npz")
    with pytest.raises(FileNotFoundError) as expected:
        zipfile.ZipFile(value, "r")
    audit = audit_native_call(HostLinearRegression.load, value)
    _refused(audit, FileNotFoundError)
    assert audit.exception.errno == expected.value.errno
    assert audit.exception.filename == expected.value.filename
    assert str(audit.exception) == str(expected.value)


def test_public_load_refuses_path_protocol_without_entering_it():
    touched = []
    class Foreign:
        def __fspath__(self):
            touched.append("fspath")
            return "model.npz"
    _refused(audit_native_call(HostLinearRegression.load, Foreign()), TypeError)
    assert touched == []


def test_public_load_error_refuses_custom_file_name_repr(tmp_path):
    value = _input(tmp_path, _payload(estimator="Bogus"), "fileio")
    touched = []
    class Foreign:
        def __repr__(self):
            touched.append("repr")
            return "foreign-name"
    value.name = Foreign()
    try:
        _refused(audit_native_call(HostLinearRegression.load, value), TypeError)
        assert touched == []
        assert not value.closed
    finally:
        value.close()


@pytest.mark.parametrize("kind", ["fileio", "buffered"])
@pytest.mark.parametrize("name_kind", ["int", "float", "deleted", "path", "pure_path"])
def test_public_load_error_preserves_native_file_name_display(tmp_path, kind, name_kind):
    raw = _input(tmp_path, _payload(estimator="Bogus"), "fileio")
    value = raw if kind == "fileio" else io.BufferedReader(raw)
    if name_kind == "deleted":
        del raw.name
    else:
        raw.name = {"int": 3, "float": 1.25, "path": Path("a//./b/../c"),
                    "pure_path": PurePosixPath("//a/./b")}[name_kind]
    try:
        expected = f"mojolearn: {value!r} was saved by Bogus, not HostLinearRegression"
        audit = audit_native_call(HostLinearRegression.load, value)
        _refused(audit, ValueError)
        assert str(audit.exception) == expected
        assert not value.closed
    finally:
        value.close()


def test_public_load_error_preserves_native_file_repr_cycle_exception(tmp_path):
    value = _input(tmp_path, _payload(estimator="Bogus"), "fileio")
    value.name = value
    try:
        with pytest.raises(RuntimeError) as expected:
            repr(value)
        audit = audit_native_call(HostLinearRegression.load, value)
        _refused(audit, RuntimeError)
        assert str(audit.exception) == str(expected.value)
    finally:
        value.name = "restored"
        value.close()


@pytest.mark.parametrize("cls", [LinearRegression, Ridge, HostLinearRegression, HostRidge])
def test_public_load_allocates_the_connected_model_and_restores_ridge_settings(cls):
    extra = {"alpha": np.array([.375], "<f8")} if cls in (Ridge, HostRidge) else {}
    stream = io.BytesIO(_payload(estimator=cls.__name__, **extra))
    model = audit_native_call(cls.load, stream).assert_native()
    assert type(model) is cls
    stream.close()
    _state(model)
    if extra:
        assert model.alpha == .375 and model.solver == "auto"
        assert model.solver_ == "eig" and model.normalize is False


@pytest.mark.parametrize("field,value,message", [
    ("format", "wrong", "model archive format does not match mojolearn-linear-1"),
    ("numeric_mode", "wrong", "invalid saved numeric_mode"),
    ("meta", np.array([3], "<i8"), "model meta must hold two fields"),
    ("coef", np.ones(2, "<f4"), "model coef does not match n_features_in_"),
])
def test_public_load_malformed_state_preserves_refusal_and_borrowed_input(field, value, message):
    stream = io.BytesIO(_payload(**{field: value}))
    audit = audit_native_call(HostLinearRegression.load, stream)
    _refused(audit, ValueError)
    assert str(audit.exception) == message
    assert not stream.closed


@pytest.mark.parametrize("name", [None, "bytes", "bytearray", "memoryview", "int", "tuple"])
def test_public_load_without_numpy_preserves_native_owned_arrays(name):
    script = r'''
import builtins,dis,io,json,sys
from mojolearn._classical_host import HostLinearRegression
from mojolearn._verify_native_runtime import audit_native_call
import mojolearn._verify_native_runtime as observer
assert "numpy" not in sys.modules
stream=io.BytesIO(bytes.fromhex(sys.argv[1]))
loader=HostLinearRegression.load
name=sys.argv[2]
original=getattr(builtins,name) if name else None
touched=[]
def foreign(*args,**kwargs):
    touched.append(name)
    return original(*args,**kwargs)
if name:
    setattr(dis,name,original)
    setattr(observer,name,original)
    setattr(builtins,name,foreign)
try:
    audit=audit_native_call(loader,stream)
finally:
    if name:setattr(builtins,name,original)
model=audit.assert_native()
assert touched==[],touched
assert "numpy" not in sys.modules
assert not stream.closed
stream.close()
assert list(model.coef_._mv)==[.75,-.5,.125]
assert model.intercept_==.25 and model.n_features_in_==3
print(json.dumps({"native":True,"numpy_absent":True}))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(sys.modules["mojolearn"].__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, "-S", "-c", script, _payload().hex(), name or ""],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == dict(native=True, numpy_absent=True)
