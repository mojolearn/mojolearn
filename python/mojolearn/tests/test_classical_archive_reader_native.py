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
import threading
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
def test_public_load_refuses_replaced_builtin_helpers_before_callbacks(tmp_path, monkeypatch, name):
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
            if name == "list":
                # The observer snapshots threading.enumerate before profiling.
                # Its two list copies are test preparation, not product work.
                patch.setattr(threading, "list", original, raising=False)
            patch.setattr(builtins, name, foreign)
            audit = audit_native_call(loader, value)
        # Connected module exports are identity-checked before execution.
        # Restoring the export must recover the public operation and ownership.
        _refused(audit, TypeError)
        assert str(audit.exception) == "native module connection export changed: builtins." + name
        assert touched == []
        if type(value) in (io.BytesIO, io.FileIO, io.BufferedReader):
            assert not value.closed and value.tell() == 0
        model = audit_native_call(loader, value).assert_native()
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
        _refused(audit, TypeError)
        assert str(audit.exception) == "native module connection export changed: builtins.repr"
        assert touched == []
        if type(value) in (io.BytesIO, io.FileIO, io.BufferedReader):
            assert not value.closed and value.tell() == 0
        restored = audit_native_call(loader, value)
        _refused(restored, ValueError)
        assert str(restored.exception) == expected
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


@pytest.mark.parametrize("cls", [Path, PurePosixPath])
def test_public_load_file_error_preserves_double_slash_root(tmp_path, cls):
    value = cls("//" + str(tmp_path).lstrip("/") + "/missing.npz")
    with pytest.raises(FileNotFoundError) as expected:
        io.open(value, "rb")
    audit = audit_native_call(HostLinearRegression.load, value)
    _refused(audit, FileNotFoundError)
    assert type(audit.exception.filename) is str
    assert audit.exception.filename == expected.value.filename
    assert audit.exception.errno == expected.value.errno
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


@pytest.mark.parametrize("kind", ["fileio", "buffered"])
@pytest.mark.parametrize("name_kind", [
    "complex", "complex_special", "bytearray", "memoryview", "list", "tuple",
    "tuple_singleton", "dict", "set", "frozenset", "empty_list", "empty_tuple",
    "empty_dict", "empty_set", "empty_frozenset", "nested_paths",
])
def test_public_load_error_preserves_builtin_container_names(tmp_path, kind, name_kind):
    values = {
        "complex": complex(-0., 1.25), "complex_special": complex(float("inf"), float("nan")),
        "bytearray": bytearray(b"a\x00\xff"), "memoryview": memoryview(b"owned"),
        "list": [1, "x", b"\xff", None], "tuple": (1., 2j, False), "tuple_singleton": ("only",),
        "dict": {"one": [1, 2], 3: (None,)}, "set": {1, 3j, "x"},
        "frozenset": frozenset((2, 4j, "y")), "empty_list": [], "empty_tuple": (),
        "empty_dict": {}, "empty_set": set(), "empty_frozenset": frozenset(),
        "nested_paths": {PurePosixPath("//a/./b"): [Path("a/../b"), {PurePosixPath("c")}]} ,
    }
    raw = _input(tmp_path, _payload(estimator="Bogus"), "fileio")
    value = raw if kind == "fileio" else io.BufferedReader(raw)
    raw.name = values[name_kind]
    try:
        expected = f"mojolearn: {value!r} was saved by Bogus, not HostLinearRegression"
        audit = audit_native_call(HostLinearRegression.load, value)
        _refused(audit, ValueError)
        assert str(audit.exception) == expected
        assert raw.name is values[name_kind] and not value.closed
    finally:
        raw.name = "restored"
        value.close()


@pytest.mark.parametrize("kind", ["fileio", "buffered"])
@pytest.mark.parametrize("cycle", ["list", "dict", "tuple_list"])
def test_public_load_error_preserves_native_container_cycles(tmp_path, kind, cycle):
    raw = _input(tmp_path, _payload(estimator="Bogus"), "fileio")
    value = raw if kind == "fileio" else io.BufferedReader(raw)
    name = {} if cycle == "dict" else []
    if cycle == "dict":
        name["cycle"] = name
    else:
        name.append(name if cycle == "list" else (name,))
    raw.name = name
    try:
        expected = f"mojolearn: {value!r} was saved by Bogus, not HostLinearRegression"
        audit = audit_native_call(HostLinearRegression.load, value)
        _refused(audit, ValueError)
        assert str(audit.exception) == expected
    finally:
        raw.name = "restored"
        value.close()
        name.clear()


@pytest.mark.parametrize("kind", ["fileio", "buffered"])
@pytest.mark.parametrize("container", [list, dict])
def test_public_load_error_preserves_file_container_cycle_exception(tmp_path, kind, container):
    raw = _input(tmp_path, _payload(estimator="Bogus"), "fileio")
    value = raw if kind == "fileio" else io.BufferedReader(raw)
    raw.name = [value] if container is list else {"file": value}
    try:
        with pytest.raises(RuntimeError) as expected:
            repr(value)
        audit = audit_native_call(HostLinearRegression.load, value)
        _refused(audit, RuntimeError)
        assert str(audit.exception) == str(expected.value)
    finally:
        raw.name = "restored"
        value.close()


@pytest.mark.parametrize("kind", ["fileio", "buffered"])
def test_public_load_error_repeated_native_file_names_are_not_cycles(tmp_path, kind):
    raw = _input(tmp_path, _payload(estimator="Bogus"), "fileio")
    value = raw if kind == "fileio" else io.BufferedReader(raw)
    peer = io.FileIO(tmp_path / "peer.bin", "w+")
    peer.name = PurePosixPath("//a/./b")
    raw.name = [peer, peer]
    try:
        expected = f"mojolearn: {value!r} was saved by Bogus, not HostLinearRegression"
        audit = audit_native_call(HostLinearRegression.load, value)
        _refused(audit, ValueError)
        assert str(audit.exception) == expected
    finally:
        raw.name = "restored"
        peer.name = "restored"
        value.close()
        peer.close()


@pytest.mark.parametrize("kind", ["fileio", "buffered"])
@pytest.mark.parametrize("name_kind", ["list", "tuple", "dict_key", "dict_value", "set", "frozenset", "list_subclass"])
def test_public_load_error_refuses_nested_foreign_name_hooks(tmp_path, kind, name_kind):
    touched = []
    class Foreign:
        def __repr__(self):
            touched.append("repr")
            return "foreign"
    class ForeignList(list):
        def __iter__(self):
            touched.append("iter")
            return super().__iter__()
        def __repr__(self):
            touched.append("repr")
            return "foreign-list"
    foreign = Foreign()
    names = {"list": [foreign], "tuple": (foreign,), "dict_key": {foreign: 1},
             "dict_value": {"value": foreign}, "set": {foreign}, "frozenset": frozenset((foreign,)),
             "list_subclass": ForeignList([1])}
    raw = _input(tmp_path, _payload(estimator="Bogus"), "fileio")
    value = raw if kind == "fileio" else io.BufferedReader(raw)
    raw.name = names[name_kind]
    try:
        _refused(audit_native_call(HostLinearRegression.load, value), TypeError)
        assert touched == [] and not value.closed
    finally:
        raw.name = "restored"
        value.close()


@pytest.mark.parametrize("kind", ["fileio", "buffered"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("key_kind", ["foreign", "str_subclass"])
def test_public_load_error_refuses_file_name_dictionary_key_hooks(tmp_path, kind, nested, key_kind):
    raw = _input(tmp_path, _payload(estimator="Bogus"), "fileio")
    value = raw if kind == "fileio" else io.BufferedReader(raw)
    peer = io.FileIO(tmp_path / "peer.bin", "w+") if nested else raw
    touched = []
    class Foreign:
        def __hash__(self):
            return hash("name")
        def __eq__(self, other):
            touched.append("comparison")
            return other == "name"
    class NativeSubclass(str):
        __hash__ = Foreign.__hash__
        __eq__ = Foreign.__eq__
    key = Foreign() if key_kind == "foreign" else NativeSubclass("name")
    del peer.name
    peer.__dict__[key] = "hidden-name"
    if nested:
        raw.name = [peer]
    touched.clear()
    try:
        _refused(audit_native_call(HostLinearRegression.load, value), TypeError)
        assert touched == []
    finally:
        peer.__dict__.clear()
        peer.name = "restored"
        raw.name = "restored"
        value.close()
        if nested:
            peer.close()


def test_public_load_error_bounds_deep_native_name_recursion(tmp_path):
    # CPython 3.14 guards native recursion by C stack, independently of the
    # Python recursion-limit setting. Discover an actual native repr refusal
    # in an isolated process; a fixed 2*sys.getrecursionlimit() was not one.
    # The search bound is a fixture memory budget, never a product depth cap.
    path = tmp_path / "deep.npz"
    path.write_bytes(_payload(estimator="Bogus"))
    code = r'''
import io, json, sys
from mojolearn._classical_host import HostLinearRegression
from mojolearn._verify_native_runtime import audit_native_call
value = io.FileIO(sys.argv[1], "rb")
try:
    for exponent in range(7, 19):
        name = []
        for _ in range(2**exponent):
            name = [name]
        value.name = name
        try:
            repr(value)
        except RecursionError:
            depth = 2**exponent
            break
    else:
        raise AssertionError("native repr guard not reached within fixture memory budget")
    audit = audit_native_call(HostLinearRegression.load, value)
    assert type(audit.exception) is RecursionError, audit.report()
    assert audit.result is None
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report
    # Unwinding must leave the supported recursion guard balanced. A shallow
    # exact display immediately after the refusal must still succeed.
    value.name = [[1], [2]]
    expected = f"mojolearn: {value!r} was saved by Bogus, not HostLinearRegression"
    audit = audit_native_call(HostLinearRegression.load, value)
    assert type(audit.exception) is ValueError, audit.report()
    assert str(audit.exception) == expected
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report
    print(json.dumps({"native_oracle_depth": depth, "python_recursion_limit": sys.getrecursionlimit()}))
finally:
    value.name = "restored"
    value.close()
'''
    result = subprocess.run([sys.executable, "-c", code, str(path)], text=True,
                            capture_output=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["native_oracle_depth"] > 0


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
# -S disables site initialization; it does not make installed NumPy absent.
# Model a truly unavailable optional dependency, not a broken import.
class MissingNumPy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("NumPy deliberately unavailable", name="numpy")
sys.meta_path.insert(0, MissingNumPy())
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
if name:
    # A replaced connected builtin export is refused before any callback
    # (bindings/native_loaded_module.mojo, contract of 31fc85981 and
    # test_native_cv_module_connections.py); the restored export then loads.
    assert type(audit.exception) is TypeError,audit.report()
    assert str(audit.exception)=="native module connection export changed: builtins."+name,str(audit.exception)
    assert not audit.report()["python_frames"],audit.report()
    assert touched==[],touched
    assert not stream.closed
    stream.seek(0)
    audit=audit_native_call(loader,stream)
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


@pytest.mark.parametrize("kind", ["fileio", "buffered"])
@pytest.mark.parametrize("shape", ["alias", "list_cycle", "dict_cycle", "tuple_cycle", "file_cycle"])
def test_public_load_error_preserves_path_graph_aliases_and_cycles(tmp_path, kind, shape):
    raw = _input(tmp_path, _payload(estimator="Bogus"), "fileio")
    value = raw if kind == "fileio" else io.BufferedReader(raw)
    path = PurePosixPath("//a/./b")
    if shape == "alias":
        shared = [path, 2j]
        name = [shared, shared]
    elif shape == "dict_cycle":
        name = {"path": path}
        name["cycle"] = name
    elif shape == "tuple_cycle":
        shared = [path]
        name = (shared,)
        shared.append(name)
    elif shape == "file_cycle":
        name = [path, value]
    else:
        name = [path]
        name.append(name)
    raw.name = name
    try:
        if shape == "file_cycle":
            with pytest.raises(RuntimeError) as expected:
                repr(value)
            error_type, message = RuntimeError, str(expected.value)
        else:
            error_type = ValueError
            message = f"mojolearn: {value!r} was saved by Bogus, not HostLinearRegression"
        audit = audit_native_call(HostLinearRegression.load, value)
        _refused(audit, error_type)
        assert str(audit.exception) == message
        assert raw.name is name
    finally:
        raw.name = "restored"
        value.close()
        if shape == "tuple_cycle":
            shared.clear()
        elif isinstance(name, (dict, list)):
            name.clear()


@pytest.mark.parametrize("kind", ["fileio", "buffered"])
@pytest.mark.parametrize("depth", [64, 256])
def test_public_load_error_formats_deep_supported_path_names(tmp_path, kind, depth):
    raw = _input(tmp_path, _payload(estimator="Bogus"), "fileio")
    value = raw if kind == "fileio" else io.BufferedReader(raw)
    name = PurePosixPath("//a/./b")
    for _ in range(depth):
        name = [name]
    raw.name = name
    try:
        # These depths are valid for both retained CPython versions. The test
        # requires the original exact display, not merely successful traversal.
        message = f"mojolearn: {value!r} was saved by Bogus, not HostLinearRegression"
        audit = audit_native_call(HostLinearRegression.load, value)
        _refused(audit, ValueError)
        assert str(audit.exception) == message
        assert raw.name is name
    finally:
        raw.name = "restored"
        value.close()


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_LZMA])
@pytest.mark.parametrize("case", ["linear", "kde", "late_schema_error"])
def test_public_load_repeated_archive_reads_preserve_none_ownership(tmp_path, case, compression):
    # Mojo 1.0's PythonObject slice conversion decremented borrowed None bounds.
    # The full public load regression runs in a fresh interpreter: CPython 3.11
    # exposes the refcount loss; immortal-None versions still verify consumed
    # state, native calls and a clean process exit. No product refcount tuning.
    if case == "kde":
        payload = _payload(compression, format="mojolearn-kde-1", estimator="KernelDensity",
                           meta=np.array([2, 3, 1], "<i8"), bandwidth=np.array([.75], "<f8"),
                           kernel="gaussian", metric="euclidean",
                           x=np.array([[1, 2], [3, 4], [5, 6]], "<f4"),
                           weights=np.array([1, 2, 1], "<f4"))
    else:
        payload = _payload(compression, **({"format": "unknown-format"} if case == "late_schema_error" else {}))
    path = tmp_path / "repeated-load.npz"
    path.write_bytes(payload)
    code = r'''
import gc, io, json, sys
import numpy as np
from mojolearn._classical_host import HostLinearRegression, HostKernelDensity
from mojolearn._verify_native_runtime import audit_native_call
path, case = sys.argv[1:]
payload = open(path, "rb").read()
loader = HostKernelDensity.load if case == "kde" else HostLinearRegression.load

def consume(model):
    if case == "kde":
        assert model.bandwidth == .75 and model.n_features_in_ == 2
        assert np.asarray(model._x, "<f4").tobytes() == np.array([[1,2],[3,4],[5,6]], "<f4").tobytes()
        assert np.asarray(model._w, "<f4").tobytes() == np.array([1,2,1], "<f4").tobytes()
    else:
        assert model.n_features_in_ == 3 and model.intercept_ == .25
        assert np.asarray(model.coef_, "<f4").tobytes() == np.array([.75,-.5,.125], "<f4").tobytes()

stream = io.BytesIO(payload)
audit = audit_native_call(loader, stream)
report = audit.report()
assert not report["python_frames"], report
assert not report["forbidden_native_calls"], report
assert not report["wrapper_work"], report
assert not report["unobserved_existing_threads"], report
if case == "late_schema_error":
    assert type(audit.exception) is ValueError, repr(audit.exception)
    expected_error = str(audit.exception)
else:
    assert audit.exception is None, repr(audit.exception)
    consume(audit.result)
assert not stream.closed
stream.close()
del stream, audit, report

def operation():
    stream = io.BytesIO(payload)
    try:
        if case == "late_schema_error":
            try:
                loader(stream)
            except ValueError as error:
                assert str(error) == expected_error
            else:
                raise AssertionError("malformed model unexpectedly loaded")
        else:
            model = loader(stream)
            consume(model)
        assert not stream.closed and stream.tell() > 0
    finally:
        stream.close()

for _ in range(1024):
    operation()
# The count is process-global: CPython can retire unrelated None references
# while caches warm. Retained fresh-process diagnostics show that loss decays,
# unlike the original 139-reference loss on every load. Use a fixed warmup and
# repeated large batches with a fixed allowance, never a per-call allowance.
# This detects sustained borrowed-singleton loss, not global count invariance.
samples = []
for batch in range(3):
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        before = sys.getrefcount(None)
        for _ in range(512):
            operation()
        raw_after = sys.getrefcount(None)
    finally:
        if enabled:
            gc.enable()
    gc.collect()
    after = sys.getrefcount(None)
    samples.append(dict(calls=512, before=before, raw_after=raw_after,
                        after=after, delta=after-before,
                        raw_delta=raw_after-before, gc_delta=after-raw_after))
from mojolearn._verify import binding_artifacts
receipt = dict(case=case, calls=1536, warmup=1024, fixed_allowance=16,
               samples=samples, native=True, consumed_state=True,
               bindings=binding_artifacts())
print(json.dumps(receipt), flush=True)
assert all(sample["delta"] >= -16 and sample["raw_delta"] >= -16
           for sample in samples), receipt
'''
    result = subprocess.run([sys.executable, "-c", code, str(path), case],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    receipt = json.loads(result.stdout)
    # Retain measured deltas and the actual subprocess native artifacts in
    # pytest's captured output, including when the lifetime check succeeds.
    print(json.dumps(receipt, sort_keys=True))
    assert receipt["calls"] == 1536 and receipt["warmup"] == 1024
    assert receipt["native"] and receipt["consumed_state"]
    assert receipt["fixed_allowance"] == 16 and len(receipt["samples"]) == 3
    assert all(sample["calls"] == 512 and sample["delta"] >= -16
               and sample["raw_delta"] >= -16 for sample in receipt["samples"])
