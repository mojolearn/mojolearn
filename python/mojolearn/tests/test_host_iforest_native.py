# SPDX-License-Identifier: Apache-2.0
"""Explicit CPU isolation-forest scoring preserves state and native boundaries."""
import ast
import gc
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from mojolearn import _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._connected_exports import assert_connected_export_refused
from mojolearn.tests import _legacy_host_inference as legacy
from mojolearn.tests.test_host_archive_native import _archive, _state
from mojolearn.tests.test_host_inference_native import _input, _refused, test_application_subclass_keeps_constructor_fit_and_score_guards as _subclass_guards
from mojolearn.tests.test_host_kernel_inference_native import FORMATS

METHODS = ("score_samples", "decision_function", "predict")
CONTROLS = ("_n_estimators", "_max_samples_mode", "_max_samples_int", "_max_samples_frac", "_max_depth", "_max_features_mode", "_max_features_int", "_max_features_frac", "_seed", "_contamination", "_bootstrap", "_contamination_auto")


def _model(tmp_path, sample_mode=0, feature_mode=0, bootstrap=False, auto=True):
    return host.HostIsolationForest.load(_archive(tmp_path, "IsolationForest", override={
        "ints": np.asarray([7, sample_mode, 2, 2, feature_mode, 1, int(bootstrap), 11, int(auto), 2, 3], dtype="<i8"),
        "reals": np.asarray([.75, .5, 0 if auto else .25, -.5], dtype="<f8")}))


def _oracle(model, x, method, public=True):
    copied = object.__new__(type(model))
    vars(copied).update(vars(model))
    out = legacy.iforest(copied, x, METHODS.index(method), public=public)
    return out, _state(vars(copied))


def _check(model, x, method):
    expected, state = _oracle(model, x, method)
    actual = audit_native_call(getattr(model, method), x).assert_native()
    assert type(actual) is np.ndarray
    assert actual.dtype == np.dtype("<i4" if method == "predict" else "<f8")
    assert _state(actual) == _state(expected)
    assert _state(vars(model)) == state
    return actual


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("sample_mode", [0, 1, 2])
@pytest.mark.parametrize("feature_mode", [0, 1])
@pytest.mark.parametrize("bootstrap,auto", [(False, True), (True, False)])
def test_iforest_resolved_controls_preserve_scores_and_state(tmp_path, method, sample_mode, feature_mode, bootstrap, auto):
    _check(_model(tmp_path, sample_mode, feature_mode, bootstrap, auto), _input(2, "float32"), method)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["float32", "list", "fortran", "strided", "big_endian", "half", "int16", "bool", "uint64"])
def test_iforest_dense_input_casts_preserve_complete_results(tmp_path, method, kind):
    _check(_model(tmp_path), _input(2, kind), method)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("format", FORMATS)
def test_iforest_existing_sparse_refusal_remains_native(tmp_path, method, format):
    sp = pytest.importorskip("scipy.sparse")
    model = _model(tmp_path)
    x = getattr(sp, format + "_matrix")(_input(2, "float32"))
    audit = audit_native_call(getattr(model, method), x)
    _refused(audit, TypeError)
    assert "dense data is required" in str(audit.exception)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("layout", ["c", "fortran", "readonly"])
def test_iforest_saved_buffer_physical_layout_is_preserved(tmp_path, method, layout):
    model = _model(tmp_path)
    model._x = np.array(model._x, order="F" if layout == "fortran" else "C", copy=True)
    if layout == "readonly":
        model._x.flags.writeable = False
    _check(model, _input(2, "float32"), method)


@pytest.mark.parametrize("field", CONTROLS + ("n_features_in_", "_mojolearn_fitted", "_mojolearn_fitting"))
def test_iforest_native_scalar_controls_preserve_results(tmp_path, field):
    model = _model(tmp_path)
    setattr(model, field, np.asarray(vars(model).get(field, False)))
    _check(model, _input(2, "float32"), "score_samples")


@pytest.mark.parametrize("field", CONTROLS + ("n_features_in_", "_mojolearn_fitted", "_mojolearn_fitting"))
def test_iforest_scalar_controls_refuse_python_hooks_before_conversion(tmp_path, field):
    model = _model(tmp_path)
    touched = []
    class Scalar:
        def __int__(self):
            touched.append("int")
            return 1
        def __float__(self):
            touched.append("float")
            return .5
        def __bool__(self):
            touched.append("bool")
            return True
    setattr(model, field, np.asarray(Scalar(), dtype=object))
    _refused(audit_native_call(model.score_samples, _input(2, "float32")), TypeError)
    assert touched == []


@pytest.mark.parametrize("field", ["n_estimators", "max_samples", "max_depth", "max_features", "bootstrap", "contamination", "random_state", "warm_start", "verbose"])
def test_iforest_missing_seed_resolution_refuses_python_scalar_hooks(tmp_path, field):
    model = _model(tmp_path)
    del model._seed
    touched = []
    class Scalar:
        def __int__(self):
            touched.append("int")
            return 1
        def __float__(self):
            touched.append("float")
            return .5
        def __bool__(self):
            touched.append("bool")
            return True
    setattr(model, field, np.asarray(Scalar(), dtype=object))
    _refused(audit_native_call(model.score_samples, _input(2, "float32")), TypeError)
    assert touched == []


@pytest.mark.parametrize("sample_mode", [0, 1, 2])
@pytest.mark.parametrize("feature_mode", [0, 1])
@pytest.mark.parametrize("auto", [False, True])
def test_iforest_manually_assembled_state_resolves_controls_natively(tmp_path, sample_mode, feature_mode, auto):
    model = _model(tmp_path, sample_mode, feature_mode, auto=auto)
    del model._seed
    _check(model, _input(2, "float32"), "decision_function")


@pytest.mark.parametrize("field,value", [("max_samples", np.int64(2)), ("max_samples", "AUTO"), ("max_samples", np.float32(.75)), ("max_features", np.int32(1)), ("max_features", np.float64(.5)), ("max_depth", None), ("random_state", None), ("contamination", "AUTO"), ("contamination", np.float32(.25)), ("n_estimators", np.asarray(4)), ("bootstrap", np.asarray(True))])
def test_iforest_constructor_scalar_forms_resolve_like_old_path(tmp_path, field, value):
    model = _model(tmp_path)
    del model._seed
    setattr(model, field, value)
    _check(model, _input(2, "float32"), "score_samples")


@pytest.mark.parametrize("field,value", [("n_estimators", 0), ("max_samples", 0), ("max_samples", 2.), ("max_samples", True), ("max_samples", "bogus"), ("max_depth", 0), ("max_features", 0), ("max_features", 2.), ("max_features", True), ("contamination", 0.), ("contamination", "bogus"), ("random_state", -1)])
def test_iforest_invalid_unresolved_controls_keep_refusal(tmp_path, field, value):
    model = _model(tmp_path)
    del model._seed
    setattr(model, field, value)
    with pytest.raises(ValueError):
        _oracle(model, _input(2, "float32"), "score_samples")
    _refused(audit_native_call(model.score_samples, _input(2, "float32")))


@pytest.mark.parametrize("field,value", [("warm_start", True), ("n_jobs", 1), ("verbose", True), ("output_type", "numpy")])
def test_iforest_existing_unsupported_control_refusals_are_native(tmp_path, field, value):
    model = _model(tmp_path)
    del model._seed
    setattr(model, field, value)
    with pytest.raises(NotImplementedError):
        _oracle(model, _input(2, "float32"), "score_samples")
    _refused(audit_native_call(model.score_samples, _input(2, "float32")), NotImplementedError)


@pytest.mark.parametrize("field", ["_x", "query"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("method", METHODS)
def test_iforest_nonfinite_inputs_preserve_public_exception_message(tmp_path, field, value, method):
    model = _model(tmp_path)
    x = _input(2, "float32")
    if field == "query":
        x[0, 0] = value
    else:
        model._x = np.asarray(model._x).copy()
        model._x[0, 0] = value
    with pytest.raises(Exception) as old:
        _oracle(model, x, method)
    assert type(old.value) is Exception
    before = _state(vars(model))
    audit = audit_native_call(getattr(model, method), x)
    _refused(audit)
    assert str(audit.exception) == f"Input X contains NaN or infinity (HostIsolationForest: {old.value})"
    assert _state(vars(model)) == before


@pytest.mark.parametrize("method", METHODS)
def test_iforest_native_kernel_error_preserves_plain_exception(tmp_path, method):
    model = _model(tmp_path)
    model._seed = -1
    with pytest.raises(Exception) as old:
        _oracle(model, _input(2, "float32"), method)
    assert type(old.value) is Exception
    audit = audit_native_call(getattr(model, method), _input(2, "float32"))
    _refused(audit, Exception)
    assert type(audit.exception) is Exception and str(audit.exception) == str(old.value)


def test_iforest_array_output_and_state_survive_input_lifetime(tmp_path):
    model = _model(tmp_path)
    x = _input(2, "float32")
    outputs = []
    for method in METHODS:
        expected, _ = _oracle(model, x, method, public=False)
        out = getattr(model, method)(x)
        assert type(out).__name__ == "Array" and _state(out) == _state(expected)
        outputs.append((out, _state(expected)))
    del model, x
    gc.collect()
    assert all(_state(value) == expected for value, expected in outputs)


def test_iforest_query_protocol_and_saved_object_buffer_refuse_hooks(tmp_path):
    model = _model(tmp_path)
    touched = []
    class Input:
        def __array__(self, *args, **kwargs):
            touched.append("array")
            return np.ones((2, 2), dtype="<f4")
        def __float__(self):
            touched.append("float")
            return 1.
    _refused(audit_native_call(model.score_samples, Input()), TypeError)
    # np.full converts its fill value through __array__ before the audit.
    # Scalar object assignment installs the hostile payload without hooks.
    storage = np.empty((3, 2), dtype=object)
    for index in np.ndindex(storage.shape):
        storage[index] = Input()
    assert touched == []
    model._x = storage
    _refused(audit_native_call(model.score_samples, [[1., 2.]]), TypeError)
    assert touched == []


def test_iforest_fixed_numerical_entry_refuses_python_callback(tmp_path, monkeypatch):
    model = _model(tmp_path)
    touched = []
    def callback(*args):
        touched.append("run")
        return 0
    monkeypatch.setitem(host._CLASSICAL["host_inference_context"]["host_inference_entries"], "iforest_run", callback)
    _refused(audit_native_call(model.predict, [[1., 2.]]), TypeError)
    assert touched == []


def test_iforest_subclass_native_attributes_finalizer_and_outputs(tmp_path):
    cls = type("OrdinaryHostIForest", (host.HostIsolationForest,), {})
    model = object.__new__(cls)
    vars(model).update(vars(_model(tmp_path)))
    for method in METHODS:
        audit_native_call(getattr, model, method).assert_native()
        _check(model, _input(2, "float32"), method)
    # the resident forest's release is a native lifetime object since lane
    # migrate-trees-2 (bindings/trees_public_iforest.mojo `_if_lifetime`):
    # no Python finalizer is defined anywhere in the class's MRO
    assert not any("__del__" in vars(base) for base in cls.__mro__)


def test_iforest_inherited_application_guards_remain_intact():
    _subclass_guards("IsolationForest")


@pytest.mark.parametrize("method", ["fit", "fit_predict"])
@pytest.mark.parametrize("auto", [False, True])
def test_iforest_inherited_training_preserves_nested_prediction(tmp_path, method, auto):
    x = _input(2, "float32")
    model = host.HostIsolationForest(n_estimators=7, max_samples=3, max_depth=2, max_features=.5, contamination="auto" if auto else .25, random_state=11, numeric_mode="identical")
    result = getattr(model, method)(x)
    if method == "fit":
        assert result is model
    else:
        assert _state(result) == _state(legacy.iforest(model, x, 2))
    assert vars(model)["_mojolearn_fitted"] is True and "_mojolearn_fitting" not in vars(model)
    for operation in METHODS:
        _check(model, x, operation)


def test_iforest_dataframe_feature_warning_and_torch_inputs(tmp_path):
    pd = pytest.importorskip("pandas")
    model = _model(tmp_path)
    _check(model, pd.DataFrame(_input(2, "float32"), columns=["a", "b"]), "score_samples")
    model.feature_names_in_ = np.asarray(["a", "b"], dtype=object)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit_native_call(model.score_samples, _input(2, "float32")).assert_native()
    assert len(caught) == 1
    del model.feature_names_in_
    torch = pytest.importorskip("torch")
    _check(model, torch.tensor(_input(2, "float32")), "predict")


@pytest.mark.parametrize("method", METHODS)
def test_iforest_unfitted_mode_shape_and_complex_refusals_stay_native(tmp_path, method):
    unfitted = object.__new__(host.HostIsolationForest)
    _refused(audit_native_call(getattr(unfitted, method), [[1., 2.]]))
    model = _model(tmp_path)
    for x in (np.ones((3, 3), dtype="<f4"), np.ones(2, dtype="<f4"), np.empty((0, 2), dtype="<f4"), np.ones((3, 2), dtype="complex64")):
        _refused(audit_native_call(getattr(model, method), x))
    model.numeric_mode = "fast"
    _refused(audit_native_call(getattr(model, method), [[1., 2.]]))


def test_iforest_public_methods_are_single_native_calls():
    tree = ast.parse(Path(host.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HostIsolationForest")
    for name in METHODS:
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        body = method.body[1:]
        assert len(body) == 1 and isinstance(body[0], ast.Return)
        assert isinstance(body[0].value, ast.Call) and body[0].value.func.id == "_native_classical"


_NUMPY_IMPORT_CHILD = r'''
import importlib.abc, json, sys
class BlockNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            # Model a missing optional dependency; a broken import must still propagate.
            raise ModuleNotFoundError("NumPy deliberately unavailable in this test", name="numpy")
blocker = BlockNumPy()
sys.meta_path.insert(0, blocker)
from mojolearn import _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
assert "numpy" not in sys.modules
payload = json.load(sys.stdin)
if payload["order"] == "late":
    sys.meta_path.remove(blocker)
    import numpy
rows = []
for record in payload["records"]:
    model = host.HostIsolationForest.load(record["path"])
    for method in ("score_samples", "decision_function", "predict"):
        result = audit_native_call(getattr(model, method), payload["source"]).assert_native()
        assert type(result).__name__ == ("ndarray" if payload["order"] == "late" else "Array")
        rows.append({"case":record["case"], "method":method, "bytes":result.tobytes().hex(), "shape":list(result.shape), "offset":model.offset_, "samples":model.max_samples_})
if payload["order"] == "absent":
    assert "numpy" not in sys.modules
print(json.dumps(rows))
'''


@pytest.mark.parametrize("order", ["absent", "late"])
def test_iforest_numpy_absent_and_late_import_preserve_consumed_results(tmp_path, order):
    records, expected = [], []
    source = _input(2, "list")
    for mode in (0, 1, 2):
        folder = tmp_path / str(mode)
        folder.mkdir()
        model = _model(folder, sample_mode=mode, feature_mode=mode % 2, auto=mode == 0)
        records.append(dict(case=mode, path=str(folder / "IsolationForest.npz")))
        for index, method in enumerate(METHODS):
            result = legacy.iforest(model, source, index)
            expected.append(dict(case=mode, method=method, bytes=result.tobytes().hex(), shape=list(result.shape), offset=model.offset_, samples=model.max_samples_))
    done = subprocess.run([sys.executable, "-c", _NUMPY_IMPORT_CHILD],
                          input=json.dumps(dict(order=order, records=records, source=source)), text=True,
                          capture_output=True, env=dict(os.environ), timeout=90)
    assert done.returncode == 0, done.stderr[-6000:]
    assert json.loads(done.stdout) == expected


def test_iforest_resolution_never_calls_replaced_builtin_float(tmp_path, monkeypatch):
    import builtins
    model = _model(tmp_path, sample_mode=2, auto=False)
    del model._seed
    # Use already admitted native NumPy scalars throughout the stored float
    # options so this reaches resolution even while builtins.float is patched.
    for name, value in tuple(vars(model).items()):
        if type(value) is float:
            setattr(model, name, np.float64(value))
    x = _input(2, "float32")
    expected, state = _oracle(model, x, "score_samples")
    touched = []
    def callback(value):
        touched.append("float")
        return .5
    score = model.score_samples  # bind first: the native attribute read consults the connected builtins
    before = _state(vars(model))
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "float", callback)
        audit = audit_native_call(score, x)
    # A replaced connected builtins export refuses (see _connected_exports)
    # before resolution: the model keeps its pre-call state (`state` is the
    # oracle's state after a completed resolution, which stores `_seed`).
    assert_connected_export_refused(audit, "builtins", "float")
    assert _state(vars(model)) == before and touched == []
    audit = audit_native_call(model.score_samples, x)
    assert _state(audit.assert_native()) == _state(expected)
    assert _state(vars(model)) == state and touched == []
