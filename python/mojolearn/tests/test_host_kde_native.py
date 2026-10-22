# SPDX-License-Identifier: Apache-2.0
"""CPU KDE scores preserve the saved numerical contract without Python work."""
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
from mojolearn.tests import _legacy_host_inference as legacy
from mojolearn.tests.test_host_archive_native import _archive, _state
from mojolearn.tests.test_host_inference_native import _input, _refused, test_application_subclass_keeps_constructor_fit_and_score_guards as _subclass_guards
from mojolearn.tests.test_host_kernel_inference_native import FORMATS

KERNELS = ("gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine")
METHODS = ("score_samples", "score")


def _model(tmp_path, kernel="gaussian", weighted=True):
    model = host.HostKernelDensity.load(_archive(tmp_path, "KernelDensity", override={"kernel": kernel}))
    if not weighted:
        model._w = None
    return model


def _check(model, x, method):
    expected = legacy.kde(model, x, method == "score")
    actual = audit_native_call(getattr(model, method), x).assert_native()
    if method == "score":
        assert type(actual) is float
    else:
        assert type(actual) is np.ndarray and actual.dtype == np.dtype("<f4")
    assert _state(actual) == _state(expected)
    return actual


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["float32", "list", "fortran", "strided", "big_endian", "half", "int16", "bool", "uint64"])
def test_kde_dense_scores_preserve_complete_output_bits(tmp_path, kernel, weighted, method, kind):
    model = _model(tmp_path, kernel, weighted)
    before = _state(vars(model))
    _check(model, _input(2, kind), method)
    assert _state(vars(model)) == before


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("method", METHODS)
def test_kde_sparse_inputs_keep_original_dense_expansion(tmp_path, format, method):
    sp = pytest.importorskip("scipy.sparse")
    _check(_model(tmp_path), getattr(sp, format + "_matrix")(_input(2, "float32")), method)


@pytest.mark.parametrize("metric", ["euclidean", "l2", "sqeuclidean", "l1", "cityblock", "manhattan", "chebyshev", "cosine", "minkowski"])
@pytest.mark.parametrize("method", METHODS)
def test_kde_metric_aliases_preserve_results(tmp_path, metric, method):
    model = _model(tmp_path)
    model.metric = metric
    _check(model, np.asarray([[1., 2.], [2., 3.], [6., 7.]], dtype="<f4"), method)


@pytest.mark.parametrize("field", ["_x", "_w"])
@pytest.mark.parametrize("layout", ["c", "fortran", "readonly"])
@pytest.mark.parametrize("method", METHODS)
def test_kde_saved_native_buffer_layouts_preserve_physical_storage(tmp_path, field, layout, method):
    model = _model(tmp_path)
    value = np.array(vars(model)[field], order="F" if layout == "fortran" else "C", copy=True)
    if layout == "readonly":
        value.flags.writeable = False
    setattr(model, field, value)
    _check(model, _input(2, "float32"), method)


@pytest.mark.parametrize("kind", ["float", "string", "bytes", "numpy32", "zero_dim", "infinity"])
@pytest.mark.parametrize("method", METHODS)
def test_kde_bandwidth_native_scalar_forms_preserve_results(tmp_path, kind, method):
    model = _model(tmp_path)
    model.bandwidth = {"float": .75, "string": ".75", "bytes": b".75", "numpy32": np.float32(.75), "zero_dim": np.asarray(.75), "infinity": np.inf}[kind]
    _check(model, _input(2, "float32"), method)


@pytest.mark.parametrize("field", ["n_features_in_", "_mojolearn_fitted", "_mojolearn_fitting"])
def test_kde_native_scalar_metadata_preserves_results(tmp_path, field):
    model = _model(tmp_path)
    setattr(model, field, np.asarray(vars(model).get(field, False)))
    _check(model, _input(2, "float32"), "score_samples")


@pytest.mark.parametrize("field", ["bandwidth", "n_features_in_", "_mojolearn_fitted", "_mojolearn_fitting"])
def test_kde_scalar_state_refuses_python_conversion_hooks(tmp_path, field):
    model = _model(tmp_path)
    touched = []
    class Scalar:
        def __float__(self):
            touched.append("float")
            return .75
        def __int__(self):
            touched.append("int")
            return 2
        def __bool__(self):
            touched.append("bool")
            return True
    setattr(model, field, np.asarray(Scalar(), dtype=object))
    _refused(audit_native_call(model.score_samples, [[1., 2.]]), TypeError)
    assert touched == []


@pytest.mark.parametrize("field", ["_x", "_w"])
def test_kde_saved_object_buffers_refuse_element_hooks(tmp_path, field):
    model = _model(tmp_path)
    touched = []
    class Scalar:
        def __float__(self):
            touched.append("float")
            return 1.
    setattr(model, field, np.full(vars(model)[field].shape, Scalar(), dtype=object))
    _refused(audit_native_call(model.score_samples, [[1., 2.]]), TypeError)
    assert touched == []


@pytest.mark.parametrize("field", ["kernel", "metric"])
def test_kde_string_options_refuse_python_subclass_hooks(tmp_path, field):
    model = _model(tmp_path)
    touched = []
    class Name(str):
        def __str__(self):
            touched.append("str")
            return "gaussian"
    setattr(model, field, Name(vars(model)[field]))
    _refused(audit_native_call(model.score_samples, [[1., 2.]]), TypeError)
    assert touched == []


@pytest.mark.parametrize("method", METHODS)
def test_kde_protocol_inputs_refused_before_hooks(tmp_path, method):
    model = _model(tmp_path)
    touched = []
    class Input:
        def __array__(self, *args, **kwargs):
            touched.append("array")
            return np.ones((2, 2), dtype="<f4")
    _refused(audit_native_call(getattr(model, method), Input()), TypeError)
    assert touched == []


def test_kde_score_does_not_inspect_ignored_y(tmp_path):
    model = _model(tmp_path)
    touched = []
    class Ignored:
        def __array__(self, *args, **kwargs):
            touched.append("array")
        def __bool__(self):
            touched.append("bool")
            return True
    x = _input(2, "float32")
    result = audit_native_call(model.score, x, Ignored()).assert_native()
    assert result == legacy.kde(model, x, True) and touched == []


@pytest.mark.parametrize("method", METHODS)
def test_kde_fixed_numerical_entry_refuses_python_replacement(tmp_path, method, monkeypatch):
    model = _model(tmp_path)
    touched = []
    def callback(*args):
        touched.append("entry")
        return 0
    monkeypatch.setitem(host._CLASSICAL["host_inference_context"]["host_inference_entries"], "kde_score_samples", callback)
    _refused(audit_native_call(getattr(model, method), [[1., 2.]]), TypeError)
    assert touched == []


@pytest.mark.parametrize("field", ["_x", "_w", "query"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_kde_nonfinite_data_preserves_existing_native_refusals(tmp_path, field, value):
    model = _model(tmp_path)
    x = _input(2, "float32")
    if field == "query":
        x[0, 0] = value
    else:
        data = np.asarray(vars(model)[field]).copy()
        data.flat[0] = value
        setattr(model, field, data)
    with pytest.raises(Exception) as old:
        legacy.kde(model, x)
    assert type(old.value) is Exception
    audit = audit_native_call(model.score_samples, x)
    # The established public wrapper converts nonfinite diagnostics to
    # ValueError while the positive-weight check remains builtin Exception.
    normalized = field != "_w" or value == np.inf
    _refused(audit, ValueError if normalized else Exception)
    expected = f"Input X contains NaN or infinity (HostKernelDensity: {old.value})" if normalized else str(old.value)
    assert type(audit.exception) is (ValueError if normalized else Exception)
    assert str(audit.exception) == expected


@pytest.mark.parametrize("field,value", [("bandwidth", 0.), ("bandwidth", -1.), ("bandwidth", np.nan), ("kernel", "bogus"), ("metric", "bogus")])
def test_kde_invalid_kernel_configuration_preserves_refusal(tmp_path, field, value):
    model = _model(tmp_path)
    setattr(model, field, value)
    with pytest.raises(Exception) as old:
        legacy.kde(model, [[1., 2.]])
    assert type(old.value) is Exception
    audit = audit_native_call(model.score_samples, [[1., 2.]])
    _refused(audit, Exception)
    assert type(audit.exception) is Exception and str(audit.exception) == str(old.value)


@pytest.mark.parametrize("shape", [(0, 2), (2, 3), (3,)])
def test_kde_invalid_query_shape_is_refused_natively(tmp_path, shape):
    _refused(audit_native_call(_model(tmp_path).score_samples, np.ones(shape, dtype="<f4")))


def test_kde_unfitted_and_numeric_mode_guards_remain_native(tmp_path):
    model = object.__new__(host.HostKernelDensity)
    _refused(audit_native_call(model.score_samples, [[1., 2.]]))
    model = _model(tmp_path)
    model.numeric_mode = "fast"
    _refused(audit_native_call(model.score_samples, [[1., 2.]]))


def test_kde_ordinary_subclass_keeps_native_attributes_and_scores(tmp_path):
    cls = type("OrdinaryHostKDE", (host.HostKernelDensity,), {})
    model = object.__new__(cls)
    vars(model).update(vars(_model(tmp_path)))
    for method in METHODS:
        audit_native_call(getattr, model, method).assert_native()
        _check(model, _input(2, "float32"), method)
    audit_native_call(cls.__del__, model).assert_native()


@pytest.mark.parametrize("field", ["bandwidth", "kernel", "metric"])
@pytest.mark.parametrize("storage", ["slot", "inherited", "instance_override"])
@pytest.mark.parametrize("method", METHODS)
def test_kde_native_option_attributes_preserve_bits_and_state(tmp_path, field, storage, method):
    value = {"bandwidth": .75, "kernel": "exponential", "metric": "manhattan"}[field]
    members = {"__slots__": (field,)} if storage == "slot" else {field: value}
    parent = type("NativeKDEOptions", (host.HostKernelDensity,), members)
    cls = type("InheritedKDEOptions", (parent,), {})
    model = object.__new__(cls)
    vars(model).update(vars(_model(tmp_path)))
    if storage == "slot":
        vars(model).pop(field)
        setattr(model, field, value)
    elif storage == "inherited":
        vars(model).pop(field)
    before = _state(vars(model))
    original = getattr(model, field)
    _check(model, _input(2, "float32"), method)
    assert _state(vars(model)) == before and getattr(model, field) is original


@pytest.mark.parametrize("field", ["bandwidth", "kernel", "metric"])
@pytest.mark.parametrize("storage", ["property", "descriptor", "slot_scalar"])
def test_kde_resolved_options_refuse_python_hooks(tmp_path, field, storage):
    touched = []
    def hook(*args):
        touched.append("get")
        return .75 if field == "bandwidth" else "gaussian"
    class Descriptor:
        __get__ = hook
    class Scalar:
        def __float__(self):
            touched.append("float")
            return .75
        def __str__(self):
            touched.append("str")
            return "gaussian"
    members = {field: property(hook) if storage == "property" else Descriptor()}
    if storage == "slot_scalar":
        members = {"__slots__": (field,)}
    cls = type("RefusedKDEOption", (host.HostKernelDensity,), members)
    model = object.__new__(cls)
    vars(model).update(vars(_model(tmp_path)))
    vars(model).pop(field)
    if storage == "slot_scalar":
        setattr(model, field, Scalar())
    before = _state(vars(model))
    _refused(audit_native_call(model.score_samples, _input(2, "float32")), TypeError)
    assert touched == [] and _state(vars(model)) == before


def test_kde_native_finalizer_masks_device_base_python_finalizer(tmp_path, monkeypatch):
    model = _model(tmp_path)
    touched = []
    def finalizer(self):
        touched.append("del")
    monkeypatch.setattr(host.KernelDensity, "__del__", finalizer)
    _check(model, _input(2, "float32"), "score_samples")
    audit_native_call(host.HostKernelDensity.__del__, model).assert_native()
    assert touched == []


@pytest.mark.parametrize("name", ["KernelDensity", "ElasticNet", "KMeans", "KernelRidge"])
def test_resolved_model_python_finalizer_is_refused_before_attribute_binding(tmp_path, name, monkeypatch):
    cls = getattr(host, "Host" + name)
    model = cls.load(_archive(tmp_path, name))
    touched = []
    def finalizer(self):
        touched.append("del")
    with monkeypatch.context() as patch:
        patch.setattr(cls, "__del__", finalizer, raising=False)
        _refused(audit_native_call(getattr, model, "score_samples" if name == "KernelDensity" else "predict"), TypeError)
        assert touched == []


def test_kde_internal_output_owner_survives_model_and_input(tmp_path):
    model = _model(tmp_path)
    x = _input(2, "float32")
    expected = _state(legacy.kde(model, x))
    out = model.score_samples(x)
    assert type(out).__name__ == "Array" and _state(out) == expected
    audit_native_call(model.score_samples, x + 1).assert_native()
    del model, x
    gc.collect()
    assert _state(out) == expected


def test_kde_dataframe_feature_name_warning_and_torch_inputs(tmp_path):
    pd = pytest.importorskip("pandas")
    model = _model(tmp_path)
    x = pd.DataFrame(_input(2, "float32"), columns=["a", "b"])
    _check(model, x, "score_samples")
    model.feature_names_in_ = np.asarray(["a", "b"], dtype=object)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit_native_call(model.score_samples, _input(2, "float32")).assert_native()
    assert len(caught) == 1
    del model.feature_names_in_
    torch = pytest.importorskip("torch")
    _check(model, torch.tensor(_input(2, "float32")), "score_samples")


def test_kde_inherited_application_guards_remain_intact():
    _subclass_guards("KernelDensity")


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("weighted", [False, True])
def test_kde_inherited_fit_preserves_saved_state_and_scoring(tmp_path, kernel, weighted):
    x = np.asarray([[1., 2.], [3., 4.], [5., 6.]], dtype="<f4")
    model = host.HostKernelDensity(bandwidth=.75, kernel=kernel, numeric_mode="identical")
    assert model.fit(x, sample_weight=np.asarray([1., 2., 1.], dtype="<f4") if weighted else None) is model
    assert vars(model)["_mojolearn_fitted"] is True
    assert "_mojolearn_fitting" not in vars(model)
    for method in METHODS:
        _check(model, _input(2, "float32"), method)


def test_kde_public_methods_are_single_native_calls():
    tree = ast.parse(Path(host.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HostKernelDensity")
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
            raise ImportError("NumPy deliberately unavailable in this test")
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
    model = host.HostKernelDensity.load(record["path"])
    result = audit_native_call(model.score_samples, payload["source"]).assert_native()
    assert type(result).__name__ == ("ndarray" if payload["order"] == "late" else "Array")
    total = audit_native_call(model.score, payload["source"]).assert_native()
    assert type(total) is float
    rows.append({"kernel":record["kernel"], "bytes":result.tobytes().hex(), "shape":list(result.shape), "total":total})
if payload["order"] == "absent":
    assert "numpy" not in sys.modules
print(json.dumps(rows))
'''


@pytest.mark.parametrize("order", ["absent", "late"])
def test_kde_numpy_absent_and_late_import_preserve_consumed_results(tmp_path, order):
    records, expected = [], []
    source = _input(2, "list")
    for kernel in KERNELS:
        folder = tmp_path / kernel
        folder.mkdir()
        model = _model(folder, kernel)
        records.append(dict(kernel=kernel, path=str(folder / "KernelDensity.npz")))
        result = legacy.kde(model, source)
        expected.append(dict(kernel=kernel, bytes=result.tobytes().hex(), shape=list(result.shape), total=legacy.kde(model, source, True)))
    done = subprocess.run([sys.executable, "-c", _NUMPY_IMPORT_CHILD],
                          input=json.dumps(dict(order=order, records=records, source=source)), text=True,
                          capture_output=True, env=dict(os.environ), timeout=90)
    assert done.returncode == 0, done.stderr[-6000:]
    assert json.loads(done.stdout) == expected


def test_kde_bandwidth_conversion_never_calls_replaced_builtin_float(tmp_path, monkeypatch):
    import builtins
    model = _model(tmp_path)
    model.bandwidth = ".75"
    x = _input(2, "float32")
    expected = _state(legacy.kde(model, x))
    touched = []
    def callback(value):
        touched.append("float")
        return .75
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "float", callback)
        audit = audit_native_call(model.score_samples, x)
    assert _state(audit.assert_native()) == expected
    assert touched == []
