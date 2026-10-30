# SPDX-License-Identifier: Apache-2.0
"""CPU logistic inference preserves scores, links, labels and native boundaries."""
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
from mojolearn._labels import ClassLabels, label_kind
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_host_inference as legacy
from mojolearn.tests.test_host_archive_native import _archive, _state
from mojolearn.tests.test_host_inference_native import _input, _refused, test_application_subclass_keeps_constructor_fit_and_score_guards as _subclass_guards

METHODS = ("decision_function", "predict", "predict_proba", "predict_log_proba")


def _model(tmp_path, count=3, intercept=True):
    targets = 1 if count == 2 else count
    weights = np.linspace(-.75, .875, (2 + int(intercept)) * targets, dtype=np.float32)
    return host.HostLogisticRegression.load(_archive(tmp_path, "LogisticRegression", override={
        "meta": np.asarray([2, intercept], dtype="<i8"),
        "classes": np.arange(count, dtype=np.int64) * 3 + 2,
        "w": weights}))


def _expected(model, x, method):
    return legacy.logistic(model, x, method)


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["float32", "list", "fortran", "strided", "big_endian", "half", "int16", "bool", "uint64"])
def test_logistic_dense_public_results_keep_exact_bits(tmp_path, count, method, kind):
    model = _model(tmp_path, count)
    x = _input(2, kind)
    expected = _expected(model, x, method)
    before = _state(vars(model))
    result = audit_native_call(getattr(model, method), x).assert_native()
    assert type(result) is np.ndarray
    assert _state(result) == _state(expected)
    assert _state(vars(model)) == before


@pytest.mark.parametrize("count", [2, 5])
@pytest.mark.parametrize("method", METHODS)
def test_logistic_without_intercept_retains_weight_layout(tmp_path, count, method):
    model = _model(tmp_path, count, False)
    x = _input(2, "float32")
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))


LABELS = [
    ["cold", "warm"], [False, True], [-3, 9], [-.25, 1.5], [2**70, 2**70 + 1],
    [2**70, 1.5], [np.int32(-2), np.int32(5)], [np.float32(.25), np.float32(.5)],
    [np.str_("red"), np.str_("blue")], [np.bool_(False), np.bool_(True)],
]


@pytest.mark.parametrize("labels", LABELS)
@pytest.mark.parametrize("container", ["list", "ClassLabels", "tuple", "ndarray"])
def test_logistic_label_values_types_and_container_semantics(tmp_path, labels, container):
    model = _model(tmp_path, 2)
    value = list(labels)
    if container == "ClassLabels":
        value = ClassLabels(value)
    elif container == "tuple":
        value = tuple(value)
    elif container == "ndarray":
        # Object storage preserves large integer, bool and numpy scalar label
        # identities; each element must still pass native scalar admission.
        value = np.asarray(value, dtype=object)
    model.classes_ = value
    x = np.asarray([[-4, 2], [4, -2], [0, 0]], dtype=np.float32)
    if container == "ndarray" and label_kind(value) in ("int", "float"):
        # Existing numeric ndarray state fails decode's truth test; retain
        # that explicit negative contract instead of silently extending it.
        with pytest.raises(ValueError, match="truth value"):
            _expected(model, x, "predict")
        _refused(audit_native_call(model.predict, x))
        return
    expected = _expected(model, x, "predict")
    result = audit_native_call(model.predict, x).assert_native()
    assert type(result) is (np.ndarray if type(expected).__name__ == "Array" else list)
    assert _state(result) == _state(expected)
    if type(expected) is list:
        assert [type(v) for v in result] == [type(v) for v in expected]


@pytest.mark.parametrize("dtype", ["int32", "uint64", "float32", "float64", "longlong", "ulonglong", "longdouble", "U5", "S5", "bool"])
def test_logistic_native_label_array_dtypes_and_strides(tmp_path, dtype):
    model = _model(tmp_path, 2)
    values = np.asarray([0, 1, 1, 0], dtype=dtype)
    model.classes_ = values[::2]
    x = _input(2, "float32")
    if np.dtype(dtype).kind in "iuf":
        with pytest.raises(ValueError, match="truth value"):
            _expected(model, x, "predict")
        _refused(audit_native_call(model.predict, x))
        return
    expected = _expected(model, x, "predict")
    actual = audit_native_call(model.predict, x).assert_native()
    assert _state(actual) == _state(expected)
    if type(expected) is list:
        assert [type(v) for v in actual] == [type(v) for v in expected]


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("method", ["decision_function", "predict_proba", "predict_log_proba"])
def test_logistic_numeric_ndarray_class_state_remains_valid_outside_decode(tmp_path, count, method):
    model = _model(tmp_path, count)
    model.classes_ = np.asarray(model.classes_, dtype=np.int64)
    x = _input(2, "float32")
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("count", [2, 3])
def test_logistic_ties_extreme_scores_and_probability_zeros(tmp_path, method, count):
    model = _model(tmp_path, count, False)
    x = np.asarray([[0., 0.], [1e30, -1e30], [-1e30, 1e30]], dtype=np.float32)
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))


@pytest.mark.parametrize("method", METHODS)
def test_logistic_internal_outputs_keep_owner_lifetimes(tmp_path, method):
    model = _model(tmp_path)
    x = _input(2, "float32")
    expected = _state(_expected(model, x, method))
    result = getattr(model, method)(x)
    assert type(result).__name__ == "Array"
    assert _state(result) == expected
    audit_native_call(getattr(model, method), x + 2).assert_native()
    del model, x
    gc.collect()
    assert _state(result) == expected


@pytest.mark.parametrize("method", METHODS)
def test_logistic_classlabels_indexing_helpers_never_run(tmp_path, method, monkeypatch):
    model = _model(tmp_path)
    model.classes_ = ClassLabels(model.classes_)
    x = _input(2, "float32")
    expected = _expected(model, x, method)
    touched = []
    def callback(*args, **kwargs):
        touched.append("ClassLabels")
        raise AssertionError("Python class-label helper ran")
    with monkeypatch.context() as patch:
        for name in ("__getitem__", "__iter__", "__len__"):
            patch.setattr(ClassLabels, name, callback, raising=False)
        actual = audit_native_call(getattr(model, method), x).assert_native()
    assert touched == []
    assert _state(actual) == _state(expected)


@pytest.mark.parametrize("container", ["list", "ClassLabels", "ndarray"])
def test_logistic_object_labels_refused_before_any_conversion_hook(tmp_path, container):
    model = _model(tmp_path, 2)
    touched = []
    class Label:
        def __float__(self):
            touched.append("float")
            return 1.
        def __int__(self):
            touched.append("int")
            return 1
        def __str__(self):
            touched.append("str")
            return "label"
    value = [Label(), Label()]
    model.classes_ = value if container == "list" else (ClassLabels(value) if container == "ClassLabels" else np.asarray(value, dtype=object))
    _refused(audit_native_call(model.predict, _input(2, "float32")), TypeError)
    assert touched == []


def test_logistic_classlabels_finalizer_is_refused(tmp_path, monkeypatch):
    model = _model(tmp_path)
    model.classes_ = ClassLabels(model.classes_)
    touched = []
    def finalizer(self):
        touched.append(True)
    with monkeypatch.context() as patch:
        patch.setattr(ClassLabels, "__del__", finalizer, raising=False)
        audit = audit_native_call(model.predict, _input(2, "float32"))
    _refused(audit, TypeError)
    assert touched == []


@pytest.mark.parametrize("method", METHODS)
def test_logistic_native_method_lookup_and_fitted_attributes(tmp_path, method):
    cls = type("OrdinaryLogisticSubclass", (host.HostLogisticRegression,), {})
    original = _model(tmp_path)
    model = object.__new__(cls)
    vars(model).update(vars(original))
    entry = audit_native_call(getattr, model, method).assert_native()
    coefficients = audit_native_call(getattr, model, "coef_").assert_native()
    assert type(coefficients) is np.ndarray
    assert np.shares_memory(coefficients, np.asarray(vars(model)["coef_"]))
    x = _input(2, "float32")
    assert _state(audit_native_call(entry, x).assert_native()) == _state(_expected(model, x, method))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_logistic_nonfinite_inputs_are_refused_without_python(tmp_path, method, value):
    model = _model(tmp_path)
    x = _input(2, "float32")
    x[0, 0] = value
    _refused(audit_native_call(getattr(model, method), x))


@pytest.mark.parametrize("method", METHODS)
def test_logistic_dataframe_preparation_matches_frozen_results(tmp_path, method):
    pd = pytest.importorskip("pandas")
    model = _model(tmp_path)
    x = pd.DataFrame(_input(2, "float32"), columns=["a", "b"])
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))


@pytest.mark.parametrize("method", METHODS)
def test_logistic_unfitted_mode_and_shape_guards_are_native(tmp_path, method):
    model = object.__new__(host.HostLogisticRegression)
    _refused(audit_native_call(getattr(model, method), [[1., 2.]]))
    model = _model(tmp_path)
    for x in (np.ones((3, 4), dtype=np.float32), np.ones(3, dtype=np.float32), np.ones((0, 2), dtype=np.float32)):
        _refused(audit_native_call(getattr(model, method), x))
    model.numeric_mode = "fast"
    _refused(audit_native_call(getattr(model, method), [[1., 2.]]))


@pytest.mark.parametrize("field", ["fit_intercept", "n_features_in_", "_mojolearn_fitted", "_mojolearn_fitting"])
def test_logistic_scalar_state_refuses_python_element_hooks(tmp_path, field):
    model = _model(tmp_path)
    touched = []
    class Scalar:
        def __bool__(self):
            touched.append("bool")
            return True
        def __int__(self):
            touched.append("int")
            return 2
    setattr(model, field, np.asarray(Scalar(), dtype=object))
    _refused(audit_native_call(model.predict, _input(2, "float32")), TypeError)
    assert touched == []


@pytest.mark.parametrize("field", ["fit_intercept", "n_features_in_", "_mojolearn_fitted", "_mojolearn_fitting"])
def test_logistic_ordinary_numeric_scalar_state_keeps_results(tmp_path, field):
    model = _model(tmp_path)
    setattr(model, field, np.asarray(vars(model).get(field, 0)))
    x = _input(2, "float32")
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(_expected(model, x, "predict"))


@pytest.mark.parametrize("method", METHODS)
def test_logistic_input_protocols_are_refused_before_python(tmp_path, method):
    model = _model(tmp_path)
    touched = []
    class Input:
        def __array__(self, *args, **kwargs):
            touched.append("array")
            return _input(2, "float32")
    _refused(audit_native_call(getattr(model, method), Input()), TypeError)
    assert touched == []


@pytest.mark.parametrize("entry,count,method", [
    ("qn_predict_binary", 2, "predict"), ("qn_decision_function", 3, "decision_function"),
    ("qn_sigmoid", 2, "predict_proba"), ("qn_softmax", 3, "predict_proba"),
    ("py2mojo_linear_flags", 2, "predict_log_proba"), ("py2mojo_rows", 2, "predict_log_proba")])
def test_logistic_native_entries_reject_python_replacements(tmp_path, entry, count, method, monkeypatch):
    model = _model(tmp_path, count)
    touched = []
    def callback(*args):
        touched.append(entry)
        return 0
    monkeypatch.setitem(host._CLASSICAL["host_inference_context"]["host_inference_entries"], entry, callback)
    _refused(audit_native_call(getattr(model, method), _input(2, "float32")), TypeError)
    assert touched == []


def test_logistic_installation_keeps_inherited_constructor_fit_score_guards():
    _subclass_guards("LogisticRegression")


@pytest.mark.parametrize("method", METHODS)
def test_logistic_feature_name_warnings_preserve_original_nested_methods(tmp_path, method):
    model = _model(tmp_path)
    model.feature_names_in_ = np.asarray(["a", "b"], dtype=object)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit_native_call(getattr(model, method), _input(2, "float32")).assert_native()
    assert len(caught) == {"decision_function": 1, "predict": 2, "predict_proba": 2, "predict_log_proba": 3}[method]
    assert "was fitted with feature names" in str(caught[0].message)


def test_logistic_public_methods_are_one_native_call():
    tree = ast.parse(Path(host.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HostLogisticRegression")
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in METHODS:
            body = node.body[1:]
            assert len(body) == 1 and isinstance(body[0], ast.Return)
            assert isinstance(body[0].value, ast.Call) and body[0].value.func.id == "_native_classical"


@pytest.mark.parametrize("selected", [0, 1])
def test_logistic_label_overflow_depends_on_selected_values(tmp_path, selected):
    from mojolearn._array import Array
    model = _model(tmp_path, 2)
    model.classes_ = [7, 2**70]
    model._w = Array.from_list([0., 0., -1. if selected == 0 else 1.], "<f4")
    x = _input(2, "float32")
    expected = _expected(model, x, "predict")
    result = audit_native_call(model.predict, x).assert_native()
    assert type(result) is (np.ndarray if selected == 0 else list)
    assert _state(result) == _state(expected)


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
    model = host.HostLogisticRegression.load(record["path"])
    for method in payload["methods"]:
        result = audit_native_call(getattr(model, method), payload["source"]).assert_native()
        assert type(result).__name__ == ("ndarray" if payload["order"] == "late" else "Array")
        rows.append({"count":record["count"], "method":method, "bytes":result.tobytes().hex(), "shape":list(result.shape)})
if payload["order"] == "absent":
    assert "numpy" not in sys.modules
print(json.dumps(rows))
'''


@pytest.mark.parametrize("order", ["absent", "late"])
def test_logistic_preserves_numpy_free_and_late_numpy_contracts(tmp_path, order):
    records, expected = [], []
    source = _input(2, "list")
    for count in (2, 3):
        folder = tmp_path / str(count)
        folder.mkdir()
        model = _model(folder, count)
        records.append(dict(count=count, path=str(folder / "LogisticRegression.npz")))
        for method in METHODS:
            result = _expected(model, source, method)
            expected.append(dict(count=count, method=method, bytes=result.tobytes().hex(), shape=list(result.shape)))
    done = subprocess.run([sys.executable, "-c", _NUMPY_IMPORT_CHILD],
                          input=json.dumps(dict(order=order, records=records, methods=METHODS, source=source)),
                          text=True, capture_output=True, env=dict(os.environ), timeout=90)
    assert done.returncode == 0, done.stderr[-6000:]
    assert json.loads(done.stdout) == expected


# Frozen 32011bb32 explicit CPU fit receipt, with four consumed outputs.
# Constructor, fit and score retain their existing implementation; changing
# their nested public inference calls must not change these saved results.
_FITTED_OUTPUT_HASHES = {2: {'decision_function': '1ad2d18e9861959ee0f6736fda9e3fa194128288322c4d20c14c5cb39d378c1b',
     'predict': 'a49d0784ca312e2168a570689ebc45669b6dc9967dda56264245e13beedec673',
     'predict_proba': 'ef218666e8fefce3a2a18e9af1f716a4b3c25391f4f9298c8cab7fec7249a4c3',
     'predict_log_proba': '7fb56bfd34f64dcb59631f23fe7b09ca8829b0673c781bf4102268bdf2a5aa07'},
 3: {'decision_function': '1d8bc5231d366c3d3725cd094e54e7308687f457f845135accf52a54a912dda4',
     'predict': '23c2c9c0a4327dd4daa18b1b53f645ccecd3e6ee4b62d65d6d2ce906c9cd0445',
     'predict_proba': '20dd4774ca1c43f58b4279070dbd9e4f2568655258fdecb316d81adebb35810c',
     'predict_log_proba': '349ecb3fbc4e26c84cad3c528df65250997c41e8dffb52ce88e624af58df0d4f'}}


@pytest.mark.parametrize("count", [2, 3])
def test_logistic_inherited_fit_state_and_nested_score_keep_results(count):
    x = np.asarray([[-2., -1.], [-1., -2.], [0., 0.], [.5, .25],
                    [2., 1.], [1., 2.], [3., 3.], [4., 3.]], dtype="<f4")
    y = np.arange(len(x), dtype=np.int64) % count
    model = host.HostLogisticRegression(max_iter=4, numeric_mode="identical")
    assert model.fit(x, y) is model
    assert type(model.classes_) is ClassLabels
    assert vars(model)["_mojolearn_fitted"] is True
    assert "_mojolearn_fitting" not in vars(model)
    assert model.n_features_in_ == 2
    for method in METHODS:
        result = audit_native_call(getattr(model, method), x).assert_native()
        assert hashlib.sha256(result.tobytes()).hexdigest() == _FITTED_OUTPUT_HASHES[count][method]
    assert model.score(x, y) == {2: .5, 3: .625}[count]


@pytest.mark.parametrize("scalar", [complex, np.complex64, np.complex128, np.clongdouble])
@pytest.mark.parametrize("container", ["list", "tuple", "ClassLabels", "ndarray"])
def test_logistic_complex_class_labels_preserve_native_objects(tmp_path, scalar, container):
    model = _model(tmp_path, 2)
    labels = [scalar(1 + 2j), scalar(-3 + 4j)]
    if container == "tuple":
        labels = tuple(labels)
    elif container == "ClassLabels":
        labels = ClassLabels(labels)
    elif container == "ndarray":
        # Primitive complex dtype indexing stays native, including the
        # platform's extended complex scalar. No numeric label conversion.
        labels = np.asarray([labels[0], labels[1], labels[1], labels[0]])[::2]
    model.classes_ = labels
    x = _input(2, "float32")
    expected = _expected(model, x, "predict")
    result = audit_native_call(model.predict, x).assert_native()
    assert type(result) is list
    assert result == expected
    assert [type(v) for v in result] == [type(v) for v in expected]


@pytest.mark.parametrize("scalar", [complex, np.complex64, np.complex128, np.clongdouble])
def test_logistic_complex_subclasses_are_refused_without_hooks(tmp_path, scalar):
    touched = []
    class Label(scalar):
        def __complex__(self):
            touched.append("complex")
            return 1 + 2j
        def __float__(self):
            touched.append("float")
            return 1.
        def __str__(self):
            touched.append("str")
            return "complex label"
    model = _model(tmp_path, 2)
    model.classes_ = [Label(1 + 2j), Label(-3 + 4j)]
    _refused(audit_native_call(model.predict, _input(2, "float32")), TypeError)
    assert touched == []


@pytest.mark.parametrize("method", METHODS)
def test_logistic_complex_label_support_keeps_complex_input_refusal(tmp_path, method):
    model = _model(tmp_path, 2)
    model.classes_ = [1 + 2j, -3 + 4j]
    x = _input(2, "float32").astype(np.complex64)
    _refused(audit_native_call(getattr(model, method), x))
