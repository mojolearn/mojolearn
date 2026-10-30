# SPDX-License-Identifier: Apache-2.0
"""SVR retains saved-kernel behavior across its complete native boundary."""
import ast
import gc
import hashlib
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

KERNELS = ("linear", "rbf", "poly", "sigmoid", "precomputed")


def _model(tmp_path, kernel="rbf", zero=False):
    changes = dict(kernel=kernel, kernel_params=np.asarray([3, .25], dtype="<f8"))
    if zero:
        changes.update(meta=np.asarray([2, 0, 5, -1, 1000], dtype="<i8"),
                       dual_coef=np.zeros((1, 0), dtype="<f4"),
                       support_vectors=np.zeros((0, 2), dtype="<f4"),
                       support=np.zeros(0, dtype="<i4"))
    return host.HostSVR.load(_archive(tmp_path, "SVR", override=changes))


def _expected(model, x):
    # The public NumPy prediction widens the completed f32 kernel output.
    return np.asarray(legacy.svr(model, x), dtype=np.float64)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("kind", ["float32", "list", "fortran", "strided", "big_endian", "half", "int16", "bool", "uint64"])
def test_svr_dense_predictions_preserve_complete_output_bits(tmp_path, kernel, kind):
    model = _model(tmp_path, kernel)
    x = _input(2, kind)
    before = _state(vars(model))
    expected = _expected(model, x)
    actual = audit_native_call(model.predict, x).assert_native()
    assert type(actual) is np.ndarray and actual.dtype == np.dtype("<f8")
    assert _state(actual) == _state(expected)
    assert _state(vars(model)) == before


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("format", FORMATS)
def test_svr_sparse_prediction_keeps_original_dense_expansion(tmp_path, kernel, format):
    sp = pytest.importorskip("scipy.sparse")
    model = _model(tmp_path, kernel)
    x = getattr(sp, format + "_matrix")(_input(2, "float32"))
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(_expected(model, x))


@pytest.mark.parametrize("kernel", KERNELS)
def test_svr_zero_support_preserves_bias_only_predictions(tmp_path, kernel):
    model = _model(tmp_path, kernel, True)
    x = _input(2, "float32")
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(_expected(model, x))


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("kernel", KERNELS)
def test_svr_native_coefficient_property_preserves_contract(tmp_path, kernel, zero):
    model = _model(tmp_path, kernel, zero)
    if kernel != "linear":
        _refused(audit_native_call(getattr, model, "coef_"), AttributeError)
        return
    expected = legacy.svr_coefficients(model)
    result = audit_native_call(getattr, model, "coef_").assert_native()
    assert type(result) is np.ndarray and result.dtype == np.dtype("<f4")
    assert _state(result) == _state(expected)
    assert type(model.coef_).__name__ == "Array"
    assert _state(model.coef_) == _state(expected)


@pytest.mark.parametrize("field", ["dual_coef_", "support_vectors_"])
@pytest.mark.parametrize("layout", ["c", "fortran", "readonly"])
def test_svr_fitted_native_buffer_layout_is_preserved(tmp_path, field, layout):
    model = _model(tmp_path, "linear")
    value = np.array(vars(model)[field], order="F" if layout == "fortran" else "C", copy=True)
    if layout == "readonly":
        value.flags.writeable = False
    setattr(model, field, value)
    x = _input(2, "float32")
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(_expected(model, x))
    assert _state(audit_native_call(getattr, model, "coef_").assert_native()) == _state(legacy.svr_coefficients(model))


@pytest.mark.parametrize("kind", ["list", "tuple", "int32", "int64", "float64", "strided", "readonly"])
def test_svr_precomputed_support_column_selection_retains_order(tmp_path, kind):
    model = _model(tmp_path, "precomputed")
    if kind == "list":
        model.support_ = [1, 0]
    elif kind == "tuple":
        model.support_ = (1, 0)
    elif kind == "strided":
        model.support_ = np.asarray([1, 99, 0, 99], dtype=np.int32)[::2]
    else:
        model.support_ = np.asarray([1, 0], dtype=np.int32 if kind == "readonly" else kind)
        if kind == "readonly":
            model.support_.flags.writeable = False
    x = _input(2, "float32")
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(_expected(model, x))


@pytest.mark.parametrize("kind", ["owned", "positive_slice", "negative_slice", "readonly_buffer"])
def test_svr_precomputed_support_array_storage_is_contiguous(tmp_path, kind):
    from mojolearn import Array

    if kind == "positive_slice":
        support = Array.from_buffer(np.asarray([1, 99, 0, 99], dtype=np.int32))[::2]
    elif kind == "negative_slice":
        support = Array.from_buffer(np.asarray([0, 1], dtype=np.int32))[::-1]
    else:
        source = np.asarray([1, 0], dtype=np.int32)
        if kind == "readonly_buffer":
            source.flags.writeable = False
        support = Array.from_buffer(source)
        if kind == "owned":
            support = support.copy()
    # Ordinary Array slicing copies its selection; the saved support ABI
    # therefore consumes contiguous int32 words even for reversed slices.
    assert support.strides == (4,)
    model = _model(tmp_path, "precomputed")
    model.support_ = support
    x = _input(2, "float32")
    expected = _expected(model, x)
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(expected)


@pytest.mark.parametrize("kernel", KERNELS)
def test_svr_internal_prediction_and_coefficient_owners_survive_calls(tmp_path, kernel):
    model = _model(tmp_path, kernel)
    x = _input(2, "float32")
    expected = _state(legacy.svr(model, x))
    result = model.predict(x)
    assert type(result).__name__ == "Array"
    assert _state(result) == expected
    audit_native_call(model.predict, x + 1).assert_native()
    del model, x
    gc.collect()
    assert _state(result) == expected


@pytest.mark.parametrize("field", ["n_features_in_", "n_support_", "_gamma", "_hp_cache_size", "_hp_degree", "_hp_coef0", "_mojolearn_fitted", "_mojolearn_fitting"])
def test_svr_ordinary_numeric_scalar_state_keeps_results(tmp_path, field):
    model = _model(tmp_path, "poly")
    setattr(model, field, np.asarray(vars(model).get(field, False)))
    x = _input(2, "float32")
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(_expected(model, x))


@pytest.mark.parametrize("field", ["n_features_in_", "n_support_", "_gamma", "_hp_cache_size", "_hp_degree", "_hp_coef0", "_mojolearn_fitted", "_mojolearn_fitting"])
def test_svr_scalar_state_refuses_python_hooks_before_conversion(tmp_path, field):
    model = _model(tmp_path, "poly")
    touched = []
    class Scalar:
        def __bool__(self):
            touched.append("bool")
            return True
        def __int__(self):
            touched.append("int")
            return 2
        def __float__(self):
            touched.append("float")
            return .5
    setattr(model, field, np.asarray(Scalar(), dtype=object))
    _refused(audit_native_call(model.predict, _input(2, "float32")), TypeError)
    assert touched == []


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_svr_nonfinite_queries_keep_original_kernel_results(tmp_path, kernel, value):
    model = _model(tmp_path, kernel)
    x = _input(2, "float32")
    x[0, 0] = value
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(_expected(model, x))


def test_svr_complex_queries_keep_original_refusal(tmp_path):
    model = _model(tmp_path)
    x = _input(2, "float32").astype(np.complex64)
    _refused(audit_native_call(model.predict, x))


def test_svr_input_protocols_are_refused_before_python(tmp_path):
    model = _model(tmp_path)
    touched = []
    class Input:
        def __array__(self, *args, **kwargs):
            touched.append("array")
            return _input(2, "float32")
    _refused(audit_native_call(model.predict, Input()), TypeError)
    assert touched == []


@pytest.mark.parametrize("entry,kernel,attribute", [("svr_predict", "rbf", False), ("svc_pair_epilogue", "precomputed", False), ("svc_dual_gemv", "linear", True)])
def test_svr_fixed_native_entries_refuse_python_replacements(tmp_path, entry, kernel, attribute, monkeypatch):
    model = _model(tmp_path, kernel)
    touched = []
    def callback(*args):
        touched.append(entry)
        return 0
    monkeypatch.setitem(host._CLASSICAL["host_inference_context"]["host_inference_entries"], entry, callback)
    audit = audit_native_call(getattr, model, "coef_") if attribute else audit_native_call(model.predict, _input(2, "float32"))
    _refused(audit, TypeError)
    assert touched == []


def test_svr_property_replacement_is_refused_before_python(tmp_path, monkeypatch):
    model = _model(tmp_path, "linear")
    touched = []
    def callback(self):
        touched.append("property")
        return None
    monkeypatch.setattr(host.SVR, "coef_", property(callback))
    _refused(audit_native_call(getattr, model, "coef_"), TypeError)
    assert touched == []


def test_svr_ordinary_subclass_keeps_native_methods_and_property(tmp_path):
    cls = type("OrdinarySVRSubclass", (host.HostSVR,), {})
    original = _model(tmp_path, "linear")
    model = object.__new__(cls)
    vars(model).update(vars(original))
    entry = audit_native_call(getattr, model, "predict").assert_native()
    assert _state(audit_native_call(entry, _input(2, "float32")).assert_native()) == _state(_expected(model, _input(2, "float32")))
    assert _state(audit_native_call(getattr, model, "coef_").assert_native()) == _state(legacy.svr_coefficients(model))


def test_svr_inherited_methods_keep_existing_application_guards():
    _subclass_guards("SVR")


def test_svr_dataframe_and_feature_name_warning_are_preserved(tmp_path):
    pd = pytest.importorskip("pandas")
    model = _model(tmp_path)
    x = pd.DataFrame(_input(2, "float32"), columns=["a", "b"])
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(_expected(model, x))
    model.feature_names_in_ = np.asarray(["a", "b"], dtype=object)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit_native_call(model.predict, _input(2, "float32")).assert_native()
    assert len(caught) == 1


def test_svr_public_method_is_one_native_call():
    tree = ast.parse(Path(host.__file__).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HostSVR")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "predict")
    body = method.body[1:]
    assert len(body) == 1 and isinstance(body[0], ast.Return)
    assert isinstance(body[0].value, ast.Call) and body[0].value.func.id == "_native_classical"


# Existing CPU fit/score remain migration debt. These saved 32011bb32
# results guard the nested inference calls affected by native installation.
_FITTED_RESULTS = {'linear': {'n_support': 3,
            'prediction': '886309dd9e4623d92aaf016c52213bc7e8bc611e576aa554ec1b8a8fcc86837b',
            'score': 0.9742311383576671},
 'rbf': {'n_support': 5,
         'prediction': '3fc84e526b8e1e19b0aab3696015dce0f8e3ba7b1dafa68f0ce055eb2cb9f9b0',
         'score': 0.864387674716556},
 'poly': {'n_support': 5,
          'prediction': '29680366643fe5f227bcf0fb2f49c38e555ea8138aea5eff40ba03f3c2ea16d9',
          'score': 0.7585669376390397},
 'sigmoid': {'n_support': 6,
             'prediction': '156754aae9ed0643cbe2424f915eb85195924f538db46a4f5e8b286189f17a9d',
             'score': 0.8081568046535612},
 'precomputed': {'n_support': 3,
                 'prediction': '886309dd9e4623d92aaf016c52213bc7e8bc611e576aa554ec1b8a8fcc86837b',
                 'score': 0.9742311383576671}}


@pytest.mark.parametrize("kernel", KERNELS)
def test_svr_inherited_fit_and_nested_score_preserve_results(kernel):
    x = np.asarray([[-2., -1.], [-1., -2.], [0., 0.], [.5, .25],
                    [2., 1.], [1., 2.], [3., 3.], [4., 3.]], dtype="<f4")
    y = x[:, 0] * .25 - x[:, 1] * .5
    data = x @ x.T if kernel == "precomputed" else x
    model = host.HostSVR(kernel=kernel, gamma=.1, coef0=.25 if kernel in ("poly", "sigmoid") else 0,
                         max_iter=4, numeric_mode="identical")
    assert model.fit(data, y) is model
    assert model.n_support_ == _FITTED_RESULTS[kernel]["n_support"]
    assert vars(model)["_mojolearn_fitted"] is True
    assert "_mojolearn_fitting" not in vars(model)
    result = audit_native_call(model.predict, data).assert_native()
    assert hashlib.sha256(result.tobytes()).hexdigest() == _FITTED_RESULTS[kernel]["prediction"]
    assert model.score(data, y) == _FITTED_RESULTS[kernel]["score"]
    if kernel == "linear":
        assert _state(audit_native_call(getattr, model, "coef_").assert_native()) == _state(legacy.svr_coefficients(model))


@pytest.mark.parametrize("kind", ["float16", "float32", "float64", "int16", "list", "tuple"])
def test_svr_intercept_uses_original_native_scalar_conversion(tmp_path, kind):
    model = _model(tmp_path, "linear")
    model.intercept_ = [2.75] if kind == "list" else ((2.75,) if kind == "tuple" else np.asarray([2.75], dtype=kind))
    x = _input(2, "float32")
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(_expected(model, x))


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
    model = host.HostSVR.load(record["path"])
    result = audit_native_call(model.predict, payload["source"]).assert_native()
    assert type(result).__name__ == ("ndarray" if payload["order"] == "late" else "Array")
    rows.append({"case":record["case"], "method":"predict", "bytes":result.tobytes().hex(), "shape":list(result.shape)})
    if record["kernel"] == "linear":
        result = audit_native_call(getattr, model, "coef_").assert_native()
        assert type(result).__name__ == ("ndarray" if payload["order"] == "late" else "Array")
        rows.append({"case":record["case"], "method":"coef_", "bytes":result.tobytes().hex(), "shape":list(result.shape)})
if payload["order"] == "absent":
    assert "numpy" not in sys.modules
print(json.dumps(rows))
'''


@pytest.mark.parametrize("order", ["absent", "late"])
def test_svr_preserves_numpy_free_and_late_numpy_contracts(tmp_path, order):
    records, expected = [], []
    source = _input(2, "list")
    for kernel, zero in [(k, False) for k in KERNELS] + [("linear", True)]:
        case = kernel + ("-zero" if zero else "")
        folder = tmp_path / case
        folder.mkdir()
        model = _model(folder, kernel, zero)
        records.append(dict(case=case, kernel=kernel, path=str(folder / "SVR.npz")))
        result = _expected(model, source) if order == "late" else legacy.svr(model, source)
        expected.append(dict(case=case, method="predict", bytes=result.tobytes().hex(), shape=list(result.shape)))
        if kernel == "linear":
            result = legacy.svr_coefficients(model)
            expected.append(dict(case=case, method="coef_", bytes=result.tobytes().hex(), shape=list(result.shape)))
    done = subprocess.run([sys.executable, "-c", _NUMPY_IMPORT_CHILD],
                          input=json.dumps(dict(order=order, records=records, source=source)), text=True,
                          capture_output=True, env=dict(os.environ), timeout=90)
    assert done.returncode == 0, done.stderr[-6000:]
    assert json.loads(done.stdout) == expected



def test_svr_unfitted_property_mode_and_shape_guards_remain_native(tmp_path):
    model = object.__new__(host.HostSVR)
    _refused(audit_native_call(getattr, model, "coef_"), AttributeError)
    _refused(audit_native_call(model.predict, [[1., 2.]]))
    model = _model(tmp_path, "linear")
    for x in (np.ones((3, 4), dtype=np.float32), np.ones(3, dtype=np.float32), np.ones((0, 2), dtype=np.float32)):
        _refused(audit_native_call(model.predict, x))
    model.numeric_mode = "fast"
    _refused(audit_native_call(model.predict, [[1., 2.]]))
    _refused(audit_native_call(getattr, model, "coef_"))


@pytest.mark.parametrize("field", ["n_features_in_", "n_support_"])
def test_svr_computed_property_refuses_scalar_hooks(tmp_path, field):
    model = _model(tmp_path, "linear")
    touched = []
    class Scalar:
        def __int__(self):
            touched.append("int")
            return 2
    setattr(model, field, np.asarray(Scalar(), dtype=object))
    _refused(audit_native_call(getattr, model, "coef_"), TypeError)
    assert touched == []


@pytest.mark.parametrize("field", ["dual_coef_", "support_vectors_"])
def test_svr_computed_property_refuses_object_array_hooks(tmp_path, field):
    model = _model(tmp_path, "linear")
    touched = []
    class Scalar:
        def __float__(self):
            touched.append("float")
            return 1.
    value = np.empty(np.asarray(vars(model)[field]).shape, dtype=object)
    value.fill(Scalar())
    setattr(model, field, value)
    _refused(audit_native_call(getattr, model, "coef_"), TypeError)
    assert touched == []


def test_svr_application_property_override_is_refused_before_python(tmp_path):
    touched = []
    class Subclass(host.HostSVR):
        @property
        def coef_(self):
            touched.append("property")
            return None
    model = object.__new__(Subclass)
    vars(model).update(vars(_model(tmp_path, "linear")))
    _refused(audit_native_call(getattr, model, "coef_"), TypeError)
    assert touched == []
