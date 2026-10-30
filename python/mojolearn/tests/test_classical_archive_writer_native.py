# SPDX-License-Identifier: Apache-2.0
"""Native archive IO preserves deterministic bytes and native file errors."""
import builtins
import dis
import io
import os
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from mojolearn import Array
import mojolearn as ml
from mojolearn._classical_host import HostLinearRegression
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_density_archive as legacy
from mojolearn.tests.test_host_inference_native import _refused


def _model(layout="c"):
    model = HostLinearRegression(numeric_mode="identical")
    source = np.asarray([.75, -.5, .125], dtype="<f4")
    if layout in ("fortran", "fortran_array"):
        # The incumbent save accepts this assigned coefficient block. This
        # tests serialization of the exact shape, not linear inference.
        source = np.asfortranarray([[.75, -.5], [.125, -0.], [np.inf, np.nan]], dtype="<f4")
    elif layout == "strided":
        source = np.asarray([.75, 0., -.5, 0., .125, 0.], dtype="<f4")[::2]
    elif layout == "empty":
        source = np.empty(0, dtype="<f4")
    elif layout == "uint8_array":
        source = np.asarray([1, 2, 255], dtype="u1")
    if layout in ("fortran_array", "uint8_array"):
        source = Array.from_buffer(source)
    model.coef_ = source
    model.intercept_ = .25
    model.n_features_in_ = 3
    return model


def _values(model):
    return dict(format="mojolearn-linear-1", estimator="HostLinearRegression", numeric_mode="identical",
                coef=vars(model)["coef_"], intercept=np.asarray([.25], dtype="<f8"),
                meta=np.asarray([3, 1], dtype="<i8"))


def _expected(model):
    out = io.BytesIO()
    legacy.write_npz(out, _values(model))
    return out.getvalue()


def _scaler_model(owner):
    model = ml.StandardScaler() if owner == "standard" else ml.MinMaxScaler()
    state = vars(model)
    state.update(numeric_mode_="identical", n_features_in_=2, n_samples_seen_=3)
    if owner == "standard":
        state.update(with_mean_=True, with_std_=True, mean_=Array.from_list([1., 2.], "<f4"),
                     var_=Array.from_list([1., 1.], "<f4"), scale_=Array.from_list([1., 1.], "<f4"))
    else:
        state.update(feature_range_=(0., 1.), clip_=False)
        for name, values in dict(data_min=[1., 2.], data_max=[2., 3.], data_range=[1., 1.],
                                 scale=[1., 1.], min=[-1., -2.]).items():
            state[name + "_"] = Array.from_list(values, "<f4")
    return model


@pytest.mark.parametrize("destination", ["str", "bytes", "path", "pure_path", "bytesio", "fileio", "buffered"])
@pytest.mark.parametrize("layout", ["c", "fortran", "fortran_array", "strided", "empty", "uint8_array"])
def test_public_linear_save_preserves_frozen_archive_bytes(tmp_path, destination, layout):
    model = _model(layout)
    expected = _expected(model)
    filename = tmp_path / "model.npz"
    targets = dict(str=str(filename), bytes=os.fsencode(filename), path=filename,
                   pure_path=PurePosixPath(filename), bytesio=io.BytesIO())
    target = io.FileIO(filename, "wb") if destination == "fileio" else io.open(filename, "wb") if destination == "buffered" else targets[destination]
    try:
        assert audit_native_call(model.save, target).assert_native() is target
        if destination == "bytesio":
            actual = target.getvalue()
        else:
            actual = filename.read_bytes()
        assert actual == expected
        if destination in ("bytesio", "fileio", "buffered"):
            assert not target.closed
    finally:
        if destination in ("bytesio", "fileio", "buffered"):
            target.close()


@pytest.mark.parametrize("name", ["open", "sorted", "repr", "bytearray", "bytes"])
def test_archive_writer_ignores_replaced_builtin_helpers(tmp_path, monkeypatch, name):
    model = _model("empty" if name == "bytes" else "c")
    expected = _expected(model)
    filename = str(tmp_path / "poisoned.npz")
    touched = []
    original = getattr(builtins, name)
    def foreign(*args, **kwargs):
        touched.append(name)
        return original(*args, **kwargs)
    # Bind first: the native attribute read consults the connected builtins.
    save = model.save
    with monkeypatch.context() as patch:
        # Pin only the test auditor's disassembler; the product still sees
        # the actual poisoned builtins module for the complete save call.
        patch.setattr(dis, name, original, raising=False)
        patch.setattr(builtins, name, foreign)
        audit = audit_native_call(save, filename)
    # A replaced connected builtin export is refused before any callback
    # (bindings/native_loaded_module.mojo, contract of 31fc85981 and
    # test_native_cv_module_connections.py); the restored export then saves.
    assert type(audit.exception) is TypeError, audit.report()
    assert str(audit.exception) == "native module connection export changed: builtins." + name
    assert not audit.report()["python_frames"], audit.report()
    assert touched == []
    assert audit_native_call(model.save, filename).assert_native() == filename
    assert Path(filename).read_bytes() == expected
    assert touched == []


@pytest.mark.parametrize("owner", ["linear", "standard", "minmax"])
@pytest.mark.parametrize("kind", ["missing_parent", "directory", "closed", "readonly"])
def test_archive_file_errors_preserve_native_class_and_attributes(tmp_path, kind, owner):
    model = _model() if owner == "linear" else _scaler_model(owner)
    def destination():
        if kind == "missing_parent":
            return str(tmp_path / "missing" / "model.npz")
        if kind == "directory":
            return str(tmp_path)
        if kind == "closed":
            file = io.BytesIO(); file.close(); return file
        path = tmp_path / "readonly.npz"
        path.write_bytes(b"")
        return io.FileIO(path, "rb")
    old_path, actual_path = destination(), destination()
    try:
        with pytest.raises(Exception) as expected:
            # Native file failures are independent of member contents; retain
            # the same fixed legacy payload for all three public C doors.
            legacy.write_npz(old_path, _values(_model()))
        audit = audit_native_call(model.save, actual_path)
        _refused(audit, type(expected.value))
        assert type(audit.exception) is type(expected.value)
        assert str(audit.exception) == str(expected.value)
        for name in ("errno", "filename", "filename2"):
            assert getattr(audit.exception, name, None) == getattr(expected.value, name, None)
    finally:
        if kind == "readonly":
            old_path.close(); actual_path.close()


@pytest.mark.parametrize("cls", [Path, PurePosixPath])
def test_archive_writer_file_error_preserves_double_slash_root(tmp_path, cls):
    model = _model()
    value = cls("//" + str(tmp_path).lstrip("/") + "/missing/model.npz")
    with pytest.raises(FileNotFoundError) as expected:
        io.open(value, "wb")
    audit = audit_native_call(model.save, value)
    _refused(audit, FileNotFoundError)
    assert type(audit.exception.filename) is str
    assert audit.exception.filename == expected.value.filename
    assert audit.exception.errno == expected.value.errno
    assert str(audit.exception) == str(expected.value)


def test_archive_path_protocols_are_refused_before_callbacks(tmp_path):
    touched = []
    class ForeignPath:
        def __fspath__(self):
            touched.append("fspath")
            return str(tmp_path / "foreign.npz")
    _refused(audit_native_call(_model().save, ForeignPath()), TypeError)
    assert touched == []


@pytest.mark.parametrize("field", ["shape", "dtype", "order", "_mv"])
def test_archive_array_metadata_refuses_python_hooks(field):
    model = _model("fortran_array")
    array = vars(model)["coef_"]
    touched = []
    class Foreign:
        def __str__(self): touched.append("str"); return "<f4"
        def __repr__(self): touched.append("repr"); return "foreign"
        def __len__(self): touched.append("len"); return 2
        def __iter__(self): touched.append("iter"); return iter((3, 2))
        def __index__(self): touched.append("index"); return 2
    setattr(array, field, Foreign())
    _refused(audit_native_call(model.save, io.BytesIO()), TypeError)
    assert touched == []


@pytest.mark.parametrize("poison", [None, "bytes", "bytearray", "tuple"])
def test_archive_fortran_array_without_numpy_matches_frozen_codec(poison):
    model = _model("fortran_array")
    expected = _expected(model).hex()
    script = r'''
import builtins,dis,io,json,sys
# Model an environment without the optional dependency before package import.
import importlib.abc
class BlockNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("NumPy is absent in this fixture", name="numpy")
sys.meta_path.insert(0, BlockNumPy())
from array import array
from mojolearn import Array
from mojolearn._classical_host import HostLinearRegression
from mojolearn._verify_native_runtime import audit_native_call
assert "numpy" not in sys.modules
model=HostLinearRegression(numeric_mode="identical")
model.coef_=Array._owned(array("f", [.75,.125,float("inf"),-.5,-0.,float("nan")]),(3,2),"<f4","F")
model.intercept_=.25
model.n_features_in_=3
output=io.BytesIO()
name=sys.argv[1]
original=getattr(builtins,name) if name else None
touched=[]
def foreign(*args,**kwargs):
    touched.append(name)
    return original(*args,**kwargs)
save=model.save
if name:
    setattr(dis,name,original)
    setattr(builtins,name,foreign)
try:
    audit=audit_native_call(save,output)
finally:
    if name: setattr(builtins,name,original)
if name:
    # A replaced connected builtin export is refused before any callback
    # (contract of 31fc85981); the restored export then saves.
    assert type(audit.exception) is TypeError, audit.report()
    assert str(audit.exception)=="native module connection export changed: builtins."+name, str(audit.exception)
    assert not audit.report()["python_frames"], audit.report()
    assert touched==[], touched
    output=io.BytesIO()
    audit=audit_native_call(model.save,output)
audit.assert_native()
assert touched==[], touched
assert "numpy" not in sys.modules
print(json.dumps(output.getvalue().hex()))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(sys.modules["mojolearn"].__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, "-S", "-c", script, poison or ""], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == expected
