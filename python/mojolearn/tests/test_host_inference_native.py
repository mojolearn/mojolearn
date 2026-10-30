# SPDX-License-Identifier: Apache-2.0
"""Explicit CPU prediction owners retain original bits and native boundaries."""
import ast
import builtins
import warnings
from pathlib import Path

import numpy as np
import pytest

from mojolearn import _classical_host as host
from mojolearn.cluster import _METRIC_NAMES
from mojolearn._buffer import frombytes
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._connected_exports import assert_connected_export_refused
from mojolearn.tests import _legacy_host_inference as legacy
from mojolearn.tests.test_host_archive_native import _archive, _state
from mojolearn.tests.test_host_tree_native import _layout

NAMES = ("ElasticNet", "Lasso", "KMeans")


def _model(tmp_path, name):
    return getattr(host, "Host" + name).load(_archive(tmp_path, name))


def _input(features, kind):
    values = np.arange(6 * features).reshape(6, features) / 4 - 1
    if kind == "half":
        return values.astype("<f2")
    if kind == "int16":
        return values.astype("<i2")
    if kind == "bool":
        return values > 1
    if kind == "uint64":
        return (np.arange(6 * features, dtype="<u8") + np.uint64(2**53 + 2**29)).reshape(6, features)
    return _layout(values, kind)


def _refused(audit, exception=ValueError):
    assert isinstance(audit.exception, exception), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("kind", ["float32", "list", "fortran", "strided", "big_endian", "half", "int16", "bool", "uint64"])
def test_public_host_prediction_retains_bits_and_native_preparation(tmp_path, name, kind):
    model = _model(tmp_path, name)
    x = _input(model.n_features_in_, kind)
    reference = legacy.kmeans(model, x) if name == "KMeans" else legacy.public_cd_predict(model, x)
    result = audit_native_call(model.predict, x).assert_native()
    assert _state(result) == _state(reference)
    if name == "KMeans":
        assert _state(audit_native_call(model.transform, x).assert_native()) == _state(legacy.kmeans(model, x, True))


@pytest.mark.parametrize("metric", list(_METRIC_NAMES) + [0, 1])
def test_kmeans_aliases_and_integer_metric_configuration(tmp_path, metric):
    model = _model(tmp_path, "KMeans")
    model.metric = metric
    x = np.asarray([[2, 3], [1, 2], [3, 4]], dtype="<f4")
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(legacy.kmeans(model, x))
    assert _state(audit_native_call(model.transform, x).assert_native()) == _state(legacy.kmeans(model, x, True))


@pytest.mark.parametrize("name", ["ElasticNet", "Lasso"])
def test_coordinate_descent_preserves_signed_zero_and_subnormal_words(tmp_path, name):
    model = _model(tmp_path, name)
    words = np.asarray([0x00000000, 0x80000000, 0x00000001, 0x3f000000, 0xbf000000, 0x3f800000], dtype="<u4")
    x = frombytes(words.tobytes(), "<f4", (2, 3))
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(legacy.public_cd_predict(model, x))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("method", ["predict", "transform"])
def test_kmeans_refuses_nonfinite_queries_natively(tmp_path, value, method):
    model = _model(tmp_path, "KMeans")
    _refused(audit_native_call(getattr(model, method), [[value, 1.]]))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shape", [(0, 2), (2, 5), (4,)])
def test_invalid_input_shapes_are_refused_without_python(tmp_path, name, shape):
    model = _model(tmp_path, name)
    _refused(audit_native_call(model.predict, np.zeros(shape, dtype="<f4")))


@pytest.mark.parametrize("name", NAMES)
def test_explicit_cpu_models_keep_identical_mode_restriction(tmp_path, name):
    model = _model(tmp_path, name)
    model.numeric_mode = "fast"
    _refused(audit_native_call(model.predict, _input(model.n_features_in_, "float32")))


@pytest.mark.parametrize("name", NAMES)
def test_unfitted_models_fail_without_python(tmp_path, name):
    model = object.__new__(getattr(host, "Host" + name))
    _refused(audit_native_call(model.predict, [[1., 2.]]))


@pytest.mark.parametrize("name", NAMES)
def test_host_prediction_never_calls_python_binding_or_input_protocols(tmp_path, name, monkeypatch):
    model = _model(tmp_path, name)
    x = _input(model.n_features_in_, "float32")
    touched = []
    def poisoned(*args, **kwargs):
        touched.append("binding")
        raise AssertionError("Python binding selection was invoked")
    monkeypatch.setattr(host._HostBound, "_bind", poisoned)
    audit_native_call(model.predict, x).assert_native()
    class Input:
        def __array__(self, *args, **kwargs):
            touched.append("array")
            return x
    _refused(audit_native_call(model.predict, Input()), TypeError)
    assert touched == []


def test_public_host_prediction_wrappers_are_single_native_calls():
    tree = ast.parse(Path(host.__file__).read_text())
    expected = {("_HostCD", "predict"), ("HostKMeans", "predict"), ("HostKMeans", "transform")}
    found = set()
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or (cls.name, method.name) not in expected:
                continue
            body = method.body[1:] if ast.get_docstring(method) is not None else method.body
            assert len(body) == 1 and isinstance(body[0], ast.Return)
            assert sum(isinstance(node, ast.Call) for node in ast.walk(body[0])) == 1
            found.add((cls.name, method.name))
    assert found == expected


@pytest.mark.parametrize("name", ["ElasticNet", "Lasso"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_coordinate_descent_public_validation_rejects_nonfinite_before_cast(tmp_path, name, value):
    model = _model(tmp_path, name)
    _refused(audit_native_call(model.predict, [[value, 1., 2.]]))


@pytest.mark.parametrize("name", ["ElasticNet", "Lasso"])
def test_finite_float64_overflow_keeps_previous_prediction_semantics(tmp_path, name):
    model = _model(tmp_path, name)
    x = np.asarray([[1e300, 1., 2.], [-1e300, 1., 2.]], dtype="<f8")
    expected = legacy.public_cd_predict(model, x)
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(expected)


@pytest.mark.parametrize("name", NAMES)
def test_internal_array_and_external_numpy_result_contract(tmp_path, name):
    model = _model(tmp_path, name)
    x = _input(model.n_features_in_, "float32")
    internal = audit_native_call(model.predict, x).assert_native()
    external = audit_native_call(model.predict, x).assert_native()
    # Enter the public boundary explicitly; pytest module names must not
    # select the package-internal Array output policy.
    # Keep every value, subsequent-call and owner-lifetime check below.
    assert type(internal) is np.ndarray
    assert type(external) is np.ndarray
    assert _state(internal) == _state(external)
    assert external.dtype == (np.dtype("int32") if name == "KMeans" else np.dtype("float64"))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("index_kind", ["object", "string", "category"])
def test_dataframe_preparation_preserves_public_output_and_name_semantics(tmp_path, name, index_kind):
    pd = pytest.importorskip("pandas")
    model = _model(tmp_path, name)
    labels = [str(i) for i in range(model.n_features_in_)]
    columns = pd.CategoricalIndex(labels) if index_kind == "category" else pd.Index(labels, dtype=index_kind)
    x = pd.DataFrame(_input(model.n_features_in_, "float32"), columns=columns)
    expected = legacy.kmeans(model, x) if name == "KMeans" else legacy.public_cd_predict(model, x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = audit_native_call(model.predict, x).assert_native()
    assert _state(got) == _state(expected)
    # The legacy wrapper materializes pandas before its feature-name check.
    assert caught == []


@pytest.mark.parametrize("name", NAMES)
def test_stored_feature_names_and_warning_error_filter_are_native(tmp_path, name):
    pd = pytest.importorskip("pandas")
    model = _model(tmp_path, name)
    labels = [str(i) for i in range(model.n_features_in_)]
    x = pd.DataFrame(_input(model.n_features_in_, "float32"), columns=labels)
    model.feature_names_in_ = np.asarray(labels, dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _refused(audit_native_call(model.predict, x), UserWarning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit_native_call(model.predict, x.to_numpy()).assert_native()
    assert len(caught) == 1
    assert "was fitted with feature names" in str(caught[0].message)


@pytest.mark.parametrize("name", NAMES)
def test_native_method_lookup_and_fitted_attribute_publication(tmp_path, name):
    model = _model(tmp_path, name)
    method = audit_native_call(getattr, model, "predict").assert_native()
    field = "cluster_centers_" if name == "KMeans" else "coef_"
    value = audit_native_call(getattr, model, field).assert_native()
    assert type(value) is np.ndarray
    assert type(getattr(model, field)).__name__ == "Array"
    assert np.shares_memory(value, np.asarray(vars(model)[field]))
    audit_native_call(method, _input(model.n_features_in_, "float32")).assert_native()


@pytest.mark.parametrize("name", NAMES)
def test_ordinary_host_subclass_keeps_native_prediction_and_attributes(tmp_path, name):
    cls = type("NativeHostSubclass", (getattr(host, "Host" + name),), {})
    model = cls.load(_archive(tmp_path, name))
    assert type(model) is cls
    method = audit_native_call(getattr, model, "predict").assert_native()
    x = _input(model.n_features_in_, "float32")
    expected = legacy.kmeans(model, x) if name == "KMeans" else legacy.public_cd_predict(model, x)
    assert _state(audit_native_call(method, x).assert_native()) == _state(expected)


@pytest.mark.parametrize("name", ["ElasticNet", "Lasso"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_dataframe_validation_rejects_nonfinite_before_prediction(tmp_path, name, value):
    pd = pytest.importorskip("pandas")
    model = _model(tmp_path, name)
    x = pd.DataFrame([[value, 1., 2.]], dtype="float64")
    _refused(audit_native_call(model.predict, x))


@pytest.mark.parametrize("name", ["ElasticNet", "Lasso"])
def test_dataframe_finite_float64_overflow_preserves_old_cast_semantics(tmp_path, name):
    pd = pytest.importorskip("pandas")
    model = _model(tmp_path, name)
    x = pd.DataFrame([[1e300, 1., 2.], [-1e300, 1., 2.]], dtype="float64")
    expected = legacy.public_cd_predict(model, x)
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(expected)


@pytest.mark.parametrize("name", NAMES)
def test_application_subclass_keeps_constructor_fit_and_score_guards(name):
    # Application overrides remain ordinary application code. Migrating the
    # package's prediction methods must not disable the inherited setup of
    # constructor options, fit validation/tracking or score fitted checks.
    calls = []
    class Custom(getattr(host, "Host" + name)):
        def __init__(self):
            pass
        def fit(self, X, y=None):
            calls.append(vars(self).get("_mojolearn_fitting"))
            return self
        def score(self, X, y=None):
            return 123
    model = Custom(numeric_mode="identical") if name == "KMeans" else Custom()
    if name == "KMeans":
        assert vars(model)["numeric_mode"] == "identical"
    with pytest.raises(ValueError, match="not fitted"):
        model.score([[1., 2.]])
    with pytest.raises(ValueError):
        model.fit(np.asarray([1., 2.]), np.asarray([0., 1.]))
    assert calls == []
    assert model.fit(np.asarray([[1., 2.], [3., 4.]]), np.asarray([0., 1.])) is model
    assert calls == [1]
    assert vars(model)["_mojolearn_fitted"] is True
    assert "_mojolearn_fitting" not in vars(model)
    assert model.n_features_in_ == 2
    assert model.score([[1., 2.]]) == 123


@pytest.mark.parametrize("method", ["fit", "fit_predict", "fit_transform"])
def test_kmeans_inherited_fit_operations_keep_state_and_prediction_bits(method):
    # Frozen CPU baseline 32011bb32, before this prediction-owner migration.
    # These inherited training operations remain existing runtime debt; this
    # regression checks that native method installation preserves their guards
    # and the predictions/distances they consume.
    x = np.asarray([[-2., -1.], [-1., -2.], [0., 0.], [.5, .25],
                    [2., 1.], [1., 2.], [3., 3.], [4., 3.]], dtype="<f4")
    model = host.HostKMeans(n_clusters=2, n_init=1, max_iter=4, init="random",
                            random_state=5, numeric_mode="identical")
    result = getattr(model, method)(x)
    if method == "fit":
        assert result is model
        result = model.predict(x)
    result = np.asarray(result)
    if method == "fit_transform":
        assert result.dtype == np.dtype("<f4")
        assert result.shape == (8, 2)
        expected = ("0080f6410080fe3f0080f2410080ee3f0000354100005d3f0000004100400940"
                    "0000e83f00d01b410000144000d01d410000503f00e8d5410000344000f40b42")
    else:
        assert result.dtype == np.dtype("<i4")
        assert result.shape == (8,)
        expected = "0100000001000000010000000100000000000000000000000000000000000000"
    assert result.tobytes().hex() == expected
    assert vars(model)["_mojolearn_fitted"] is True
    assert "_mojolearn_fitting" not in vars(model)
    assert model.n_features_in_ == 2


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("hook", ["function", "init", "str", "del", "metaclass"])
def test_host_warning_category_hooks_are_refused_before_native_publication(tmp_path, name, hook, monkeypatch):
    model = _model(tmp_path, name)
    model.feature_names_in_ = np.asarray([str(i) for i in range(model.n_features_in_)], dtype=object)
    touched = []
    original = builtins.UserWarning
    if hook == "function":
        def category(*args, **kwargs):
            touched.append("function")
            return original(*args, **kwargs)
    elif hook == "metaclass":
        class Meta(type):
            def __getattribute__(self, name):
                touched.append("metaclass")
                return super().__getattribute__(name)
        class category(original, metaclass=Meta):
            pass
    else:
        def callback(self, *args, **kwargs):
            touched.append(hook)
            return "warning" if hook == "str" else None
        category = type("HookedWarning", (original,), {"__" + hook + "__": callback})
    # Bind the operation first: the native attribute read itself consults the
    # connected builtins, which refuse once an export is replaced.
    predict, x = model.predict, _input(model.n_features_in_, "float32")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "UserWarning", category)
            audit = audit_native_call(predict, x)
    _refused(audit, TypeError)
    assert_connected_export_refused(audit, "builtins", "UserWarning")
    assert touched == []
    assert caught == []


@pytest.mark.parametrize("name", NAMES)
def test_native_warning_subclass_identity_survives_error_filter(tmp_path, name, monkeypatch):
    model = _model(tmp_path, name)
    model.feature_names_in_ = np.asarray([str(i) for i in range(model.n_features_in_)], dtype=object)
    class NativeWarning(UserWarning):
        pass
    predict, x = model.predict, _input(model.n_features_in_, "float32")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)
        with monkeypatch.context() as patch:
            # Replacing the connected builtins.UserWarning refuses (see
            # _connected_exports); the native category is the setup-time one.
            patch.setattr(builtins, "UserWarning", NativeWarning)
            audit = audit_native_call(predict, x)
        assert_connected_export_refused(audit, "builtins", "UserWarning")
        audit = audit_native_call(predict, x)
    _refused(audit, UserWarning)
    assert type(audit.exception) is UserWarning
    assert "was fitted with feature names" in str(audit.exception)
