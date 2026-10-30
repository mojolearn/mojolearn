# SPDX-License-Identifier: Apache-2.0
"""Public CPU density save owns preparation, deterministic IO and error paths."""
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import numpy as np
import pytest

from mojolearn import Array, host_model
from mojolearn import _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_density_archive as legacy
from mojolearn.tests._legacy_host_inference import iforest as old_iforest
from mojolearn.tests.test_host_archive_native import _state
from mojolearn.tests.test_host_inference_native import _refused


def _prepared(family, layout="array", weighted=False, mode="identical", cls=None):
    cls = cls or (host.HostKernelDensity if family == "kde" else host.HostIsolationForest)
    model = cls(numeric_mode=mode) if family == "kde" else cls(n_estimators=7, max_samples=3, max_depth=2, random_state=11, numeric_mode=mode)
    x = np.asarray([[.25, -.75], [1.5, .125], [-.5, .875]], dtype="<f4")
    if layout == "fortran":
        x = Array.from_buffer(np.asfortranarray(x))
    elif layout == "strided":
        x = np.repeat(x, 2, axis=1)[:, ::2]
    elif layout == "array":
        x = Array.from_buffer(x)
    fields = vars(model)
    fields.update(_x=x, n_features_in_=2, _mojolearn_fitted=True)
    if family == "kde":
        fields.update(n_samples_fit_=3, _w=Array.from_list([.5, 1., 2.], "<f4") if weighted else None)
    else:
        fields.update(_n_estimators=7, _max_samples_mode=1, _max_samples_int=3, _max_samples_frac=1.,
                      _max_depth=2, _max_features_mode=0, _max_features_int=0, _max_features_frac=1.,
                      _bootstrap=weighted, _seed=11, _contamination_auto=not weighted,
                      _contamination=.25 if weighted else 0., offset_=-.4 if weighted else -.5, max_samples_=3)
    return model


def _oracle(model, family, *, canonical=True):
    output = io.BytesIO()
    (legacy.save_kde if family == "kde" else legacy.save_iforest)(model, output, canonical=canonical)
    return output.getvalue()


@pytest.mark.parametrize("family", ["kde", "iforest"])
@pytest.mark.parametrize("layout", ["array", "fortran", "ndarray", "strided"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("destination", ["str", "path", "bytesio"])
def test_public_density_save_preserves_schema_bytes_and_state(tmp_path, family, layout, weighted, destination):
    model = _prepared(family, layout, weighted)
    expected = _oracle(model, family)
    state = _state(vars(model))
    path = tmp_path / "model.npz"
    target = io.BytesIO() if destination == "bytesio" else str(path) if destination == "str" else path
    assert audit_native_call(model.save, target).assert_native() is target
    actual = target.getvalue() if destination == "bytesio" else path.read_bytes()
    assert actual == expected
    assert _state(vars(model)) == state


@pytest.mark.parametrize("family", ["kde", "iforest"])
@pytest.mark.parametrize("mode", [None, " identical ", "FAST", "fast", "", False, 0, np.str_("identical"), np.asarray(0)])
def test_density_save_native_mode_forms_preserve_old_bytes(family, mode):
    model = _prepared(family)
    model.numeric_mode = mode
    expected = _oracle(model, family)
    target = io.BytesIO()
    assert audit_native_call(model.save, target).assert_native() is target
    assert target.getvalue() == expected


@pytest.mark.parametrize("family", ["kde", "iforest"])
@pytest.mark.parametrize("kind", ["unfitted", "mode", "mode_number", "missing_size", "missing_parent"])
def test_density_save_native_errors_preserve_partial_state(tmp_path, family, kind):
    old, model = _prepared(family), _prepared(family)
    for item in (old, model):
        if kind == "unfitted": del vars(item)["_x"]
        elif kind == "mode": item.numeric_mode = "bad"
        elif kind == "mode_number": item.numeric_mode = 3
        elif kind == "missing_size": del vars(item)["n_features_in_"]
    target = str(tmp_path / "missing" / "model.npz") if kind == "missing_parent" else io.BytesIO()
    with pytest.raises(Exception) as expected:
        (legacy.save_kde if family == "kde" else legacy.save_iforest)(old, target)
    audit = audit_native_call(model.save, target)
    _refused(audit, type(expected.value))
    assert type(audit.exception) is type(expected.value)
    assert str(audit.exception) == str(expected.value)
    assert _state(vars(model)) == _state(vars(old))


@pytest.mark.parametrize("family", ["kde", "iforest"])
@pytest.mark.parametrize("historic", [False, True])
@pytest.mark.parametrize("route", ["class", "generic"])
def test_density_saved_models_and_historic_host_tags_load_natively(family, historic, route):
    model = _prepared(family, weighted=True)
    payload = _oracle(model, family, canonical=not historic)
    operation = type(model).load if route == "class" else host_model
    restored = audit_native_call(operation, io.BytesIO(payload)).assert_native()
    query = np.asarray([[.5, -.25], [.125, .75]], dtype="f4")
    expected = model.score_samples(query) if family == "kde" else old_iforest(model, query, 0)
    actual = audit_native_call(restored.score_samples, query).assert_native()
    assert _state(actual) == _state(expected)
    # Saving again keeps the canonical archive schema after a historic load.
    target = io.BytesIO()
    audit_native_call(restored.save, target).assert_native()
    assert target.getvalue() == _oracle(restored, family)


@pytest.mark.parametrize("family", ["kde", "iforest"])
@pytest.mark.parametrize("tag", ["other_family", "unregistered_subclass"])
def test_density_historic_tag_admission_keeps_exact_family_bounds(family, tag):
    payload = _oracle(_prepared(family), family, canonical=False)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    other = "HostIsolationForest" if family == "kde" else "HostKernelDensity"
    fields["estimator"] = other if tag == "other_family" else "UnregisteredDensitySubclass"
    output = io.BytesIO()
    legacy.write_npz(output, fields)
    _refused(audit_native_call(host_model, io.BytesIO(output.getvalue())), ValueError)


def _archive_route(route):
    return host.HostKernelDensity.load if route == "class" else host.host_model if route == "classical" else host_model


@pytest.mark.parametrize("route", ["class", "classical", "generic"])
@pytest.mark.parametrize("kind", ["str", "bytes", "path", "pure_path", "bytesio", "fileio", "buffered"])
def test_density_load_admits_archive_inputs_at_the_native_reader(tmp_path, route, kind):
    original = _prepared("kde", weighted=True)
    payload = _oracle(original, "kde")
    filename = tmp_path / "model.npz"
    filename.write_bytes(payload)
    value = {"str": str(filename), "bytes": os.fsencode(filename), "path": filename,
             "pure_path": PurePosixPath(filename), "bytesio": io.BytesIO(payload)}.get(kind)
    if kind == "fileio": value = io.FileIO(filename, "rb")
    elif kind == "buffered": value = io.open(filename, "rb")
    try:
        restored = audit_native_call(_archive_route(route), value).assert_native()
        if kind in ("bytesio", "fileio", "buffered"):
            assert not value.closed
            assert value.tell() > 0
    finally:
        if kind in ("bytesio", "fileio", "buffered"): value.close()
    query = np.asarray([[.5, -.25], [.125, .75]], dtype="f4")
    expected = original.score_samples(query)
    actual = audit_native_call(restored.score_samples, query).assert_native()
    assert _state(actual) == _state(expected)


@pytest.mark.parametrize("route", ["class", "classical", "generic"])
@pytest.mark.parametrize("kind", ["protocol", "bytesio_subclass", "path_subclass", "buffered_raw"])
def test_density_load_archive_admission_refuses_python_file_protocols(tmp_path, route, kind):
    touched = []
    class Foreign:
        def __fspath__(self): touched.append("fspath"); return str(tmp_path / "model.npz")
        def read(self, *args): touched.append("read"); return b""
        def seek(self, *args): touched.append("seek"); return 0
    class ForeignBytesIO(io.BytesIO):
        def read(self, *args): touched.append("read"); return super().read(*args)
    class ForeignPath(type(tmp_path)):
        def __fspath__(self): touched.append("fspath"); return super().__fspath__()
    class ForeignRaw(io.FileIO):
        def read(self, *args): touched.append("read"); return super().read(*args)
        def seek(self, *args): touched.append("seek"); return super().seek(*args)
    filename = tmp_path / "model.npz"
    filename.write_bytes(_oracle(_prepared("kde"), "kde"))
    value = Foreign() if kind == "protocol" else ForeignBytesIO(filename.read_bytes()) if kind == "bytesio_subclass" else ForeignPath(filename) if kind == "path_subclass" else io.BufferedReader(ForeignRaw(filename, "rb"))
    touched.clear()  # construction is test setup, outside the public operation.
    try:
        _refused(audit_native_call(_archive_route(route), value), TypeError)
        assert touched == []
    finally:
        if kind in ("bytesio_subclass", "buffered_raw"): value.close()


@pytest.mark.parametrize("family", ["kde", "iforest"])
@pytest.mark.parametrize("field", ["numeric_mode", "n_features_in_", "_x"])
def test_density_save_refuses_python_state_hooks_before_invocation(family, field):
    model = _prepared(family)
    touched = []
    class Foreign:
        def __float__(self): touched.append("float"); return 1.
        def __int__(self): touched.append("int"); return 2
        def __index__(self): touched.append("index"); return 2
        def __bool__(self): touched.append("bool"); return True
        def __str__(self): touched.append("str"); return "identical"
        def __repr__(self): touched.append("repr"); return "foreign"
    vars(model)[field] = Foreign()
    _refused(audit_native_call(model.save, io.BytesIO()), TypeError)
    assert touched == []


@pytest.mark.parametrize("field", ["kernel", "metric"])
@pytest.mark.parametrize("form", ["str", "numpy", "bytes", "numpy_bytes", "bytearray", "subclass"])
def test_density_save_retains_native_text_option_conversion(field, form):
    class NativeText(str):
        pass
    text = "gaussian" if field == "kernel" else "euclidean"
    value = dict(str=text, numpy=np.str_(text), bytes=text.encode(), numpy_bytes=np.bytes_(text),
                 bytearray=bytearray(text.encode()), subclass=NativeText(text))[form]
    model = _prepared("kde")
    setattr(model, field, value)
    expected = _oracle(model, "kde")
    output = io.BytesIO()
    audit_native_call(model.save, output).assert_native()
    assert output.getvalue() == expected


@pytest.mark.parametrize("field", ["kernel", "metric"])
@pytest.mark.parametrize("value", [None, 3, True, 2 + 3j, ["gaussian"], ("euclidean",), {"name": "gaussian"}])
def test_density_save_keeps_builtin_assigned_option_archive_bytes(field, value):
    model = _prepared("kde")
    setattr(model, field, value)
    expected = _oracle(model, "kde")
    output = io.BytesIO()
    audit_native_call(model.save, output).assert_native()
    assert output.getvalue() == expected


@pytest.mark.parametrize("family", ["kde", "iforest"])
@pytest.mark.parametrize("mode", [["identical"], {"mode": "identical"}, ("identical",), bytearray(b"identical")])
def test_density_save_invalid_native_container_mode_preserves_error(family, mode):
    model = _prepared(family)
    model.numeric_mode = mode
    with pytest.raises(ValueError) as expected:
        _oracle(model, family)
    audit = audit_native_call(model.save, io.BytesIO())
    _refused(audit, ValueError)
    assert type(audit.exception) is type(expected.value)
    assert str(audit.exception) == str(expected.value)


@pytest.mark.parametrize("form", [1, np.float32(.75), np.float64(.75), np.asarray(.75), "0.75", b"0.75"])
def test_density_save_retains_native_bandwidth_scalar_conversion(form):
    model = _prepared("kde")
    model.bandwidth = form
    expected = _oracle(model, "kde")
    output = io.BytesIO()
    audit_native_call(model.save, output).assert_native()
    assert output.getvalue() == expected


def _resolved_names():
    return ("_max_samples_mode", "_max_samples_int", "_max_samples_frac", "_max_depth",
            "_max_features_mode", "_max_features_int", "_max_features_frac", "_contamination_auto",
            "_contamination", "_n_estimators", "_bootstrap", "_seed")


def _resolved_snapshot(model):
    return _state(vars(model)), [(name, _state(getattr(model, name, None))) for name in _resolved_names()]


@pytest.mark.parametrize("slotted", [False, True])
@pytest.mark.parametrize("case", ["valid", "samples", "depth", "features", "contamination", "seed", "warm_start"])
def test_iforest_save_resolves_missing_seed_with_exact_partial_state(slotted, case):
    class SlotForest(host.HostIsolationForest):
        __slots__ = _resolved_names()
    cls = SlotForest if slotted else host.HostIsolationForest
    old, model = _prepared("iforest", cls=cls), _prepared("iforest", cls=cls)
    for item in (old, model):
        for name in _resolved_names():
            vars(item).pop(name, None)
        if case == "samples": item.max_samples = 0
        elif case == "depth": item.max_depth = 0
        elif case == "features": item.max_features = 0
        elif case == "contamination": item.contamination = .75
        elif case == "seed": item.random_state = -1
        elif case == "warm_start": item.warm_start = True
    reference = io.BytesIO()
    try:
        legacy.save_iforest(old, reference, canonical=True)
    except Exception as expected:
        expected_error = expected
    else:
        expected_error = None
    output = io.BytesIO()
    audit = audit_native_call(model.save, output)
    if expected_error is None:
        audit.assert_native()
        assert output.getvalue() == reference.getvalue()
    else:
        _refused(audit, type(expected_error))
        assert type(audit.exception) is type(expected_error)
        assert str(audit.exception) == str(expected_error)
    assert _resolved_snapshot(model) == _resolved_snapshot(old)


@pytest.mark.parametrize("family", ["kde", "iforest"])
def test_density_save_reads_registered_native_slots(family):
    base = host.HostKernelDensity if family == "kde" else host.HostIsolationForest
    class SlotModel(base):
        __slots__ = ("_x", "n_features_in_")
    model = _prepared(family, cls=SlotModel)
    for name in ("_x", "n_features_in_"):
        setattr(model, name, vars(model).pop(name))
    expected = _oracle(model, family)
    output = io.BytesIO()
    audit_native_call(model.save, output).assert_native()
    assert output.getvalue() == expected


@pytest.mark.parametrize("family", ["kde", "iforest"])
@pytest.mark.parametrize("slot", ["__setattr__", "shape", "__del__"])
def test_density_save_metadata_allocation_refuses_replaced_array_hooks(family, slot):
    # Class-slot changes, including finalization flags, stay in an isolated
    # child. Product allocation must refuse before it publishes any Array.
    script = r'''
import io,json,runpy,sys,types
from pathlib import Path
from mojolearn import Array
from mojolearn._verify_native_runtime import audit_native_call
# Tests are external to an installed wheel. Load this retained test file only
# for fixture preparation, before the audited public save operation.
tests=types.ModuleType('mojolearn.tests');tests.__path__=[str(Path(sys.argv[3]).parent)]
sys.modules['mojolearn.tests']=tests
model=runpy.run_path(sys.argv[3])['_prepared'](sys.argv[1]);name=sys.argv[2];calls=[]
old=vars(Array).get(name);present=name in vars(Array)
def hook(*args):calls.append(name)
replacement=property(lambda self: calls.append(name),lambda self,value:calls.append(name)) if name=='shape' else hook
setattr(Array,name,replacement)
try:audit=audit_native_call(model.save,io.BytesIO())
finally:
    if present:setattr(Array,name,old)
    else:delattr(Array,name)
assert isinstance(audit.exception,TypeError),audit.report()
assert calls==[],calls
report=audit.report()
assert not any(report[k] for k in ('python_frames','forbidden_native_calls','unresolved_native_calls','wrapper_work','unobserved_existing_threads')),report
print(json.dumps({'refused':True,'calls':calls}))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(sys.modules["mojolearn"].__file__).resolve().parents[1]))
    child = subprocess.run([sys.executable, "-c", script, family, slot, str(Path(__file__).resolve())],
                           env=env, capture_output=True, text=True)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == {"refused": True, "calls": []}


@pytest.mark.parametrize("family", ["kde", "iforest"])
def test_density_save_without_numpy_preserves_frozen_archive(family):
    expected = _oracle(_prepared(family), family).hex()
    script = r'''
import io,json,sys,importlib.abc
# -S does not hide NumPy when the installed package's site directory is
# explicitly supplied through PYTHONPATH. Enforce actual optional absence.
class BlockNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy' or fullname.startswith('numpy.'):
            raise ModuleNotFoundError('NumPy intentionally unavailable', name='numpy')
sys.meta_path.insert(0, BlockNumPy())
from array import array
from mojolearn import Array
from mojolearn._classical_host import HostKernelDensity,HostIsolationForest
from mojolearn._verify_native_runtime import audit_native_call
family=sys.argv[1]
model=HostKernelDensity(numeric_mode='identical') if family=='kde' else HostIsolationForest(n_estimators=7,max_samples=3,max_depth=2,random_state=11,numeric_mode='identical')
fields=vars(model)
fields.update(_x=Array._owned(array('f',[.25,-.75,1.5,.125,-.5,.875]),(3,2),'<f4','C'),n_features_in_=2,_mojolearn_fitted=True)
if family=='kde':fields.update(n_samples_fit_=3,_w=None)
else:fields.update(_n_estimators=7,_max_samples_mode=1,_max_samples_int=3,_max_samples_frac=1.,_max_depth=2,_max_features_mode=0,_max_features_int=0,_max_features_frac=1.,_bootstrap=False,_seed=11,_contamination_auto=True,_contamination=0.,offset_=-.5,max_samples_=3)
assert 'numpy' not in sys.modules
output=io.BytesIO()
audit_native_call(model.save,output).assert_native()
assert 'numpy' not in sys.modules
print(json.dumps(output.getvalue().hex()))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(sys.modules["mojolearn"].__file__).resolve().parents[1]))
    child = subprocess.run([sys.executable, "-S", "-c", script, family], env=env, capture_output=True, text=True)
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == expected
