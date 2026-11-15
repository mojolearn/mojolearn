# SPDX-License-Identifier: Apache-2.0
"""Saved SVC public inference preserves its complete native operation boundary."""
import gc
import warnings

import numpy as np
import pytest

from mojolearn import Array
from mojolearn import _classical_host as host
from mojolearn._labels import ClassLabels
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_host_inference as legacy
from mojolearn.tests.test_host_archive_native import _archive, _state, _svc_ovo
from mojolearn.tests.test_host_inference_native import _input, _refused, test_application_subclass_keeps_constructor_fit_and_score_guards as _subclass_guards
from mojolearn.tests.test_host_kernel_inference_native import FORMATS

KERNELS = ("linear", "rbf", "poly", "sigmoid", "precomputed")
METHODS = ("decision_function", "predict", "predict_proba", "predict_log_proba")


def _model(tmp_path, kernel="rbf", multi=False, shape="ovr", ties=False, probability=True, zero=False):
    changes = _svc_ovo() if multi else {}
    changes.update(kernel=kernel, coef0=np.asarray([.25], dtype="<f8"))
    if multi:
        changes.update(shape=shape, break_ties=np.asarray([ties], dtype="<i8"))
        if kernel == "precomputed":
            changes["pair_support"] = np.asarray([0, 1, 0, 1, 0, 1], dtype="<i4")
        if zero:
            changes.update(pair_meta=np.asarray([0, 1, 0, 5, 0, 2, 0, 6, 1, 2, 0, 7], dtype="<i8"),
                           pair_support=np.zeros(0, dtype="<i4"), pair_dual=np.zeros(0, dtype="<f4"), pair_sv=np.zeros(0, dtype="<f4"))
    elif zero:
        changes.update(meta=np.asarray([2, 0, 5, -1, 1000, 3], dtype="<i8"), dual_coef=np.zeros((1, 0), dtype="<f4"),
                       support_vectors=np.zeros((0, 2), dtype="<f4"), support=np.zeros(0, dtype="<i4"))
    if probability:
        if not multi:
            changes["prob_ab"] = np.asarray([.5, -.5], dtype="<f8")
    else:
        changes.pop("prob_ab", None)
    return host.HostSVC.load(_archive(tmp_path, "SVC", override=changes))


def _expected(model, x, method):
    value = legacy.svc(model, x, method)
    return value if type(value) is list else np.asarray(value)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["float32", "list", "fortran", "strided", "half"])
def test_svc_complete_predictions_preserve_dtype_bits_and_state(tmp_path, kernel, multi, method, kind):
    model = _model(tmp_path, kernel, multi)
    x = _input(2, kind)
    expected = _expected(model, x, method)
    before = _state(vars(model))
    result = audit_native_call(getattr(model, method), x).assert_native()
    assert type(result) is (list if type(expected) is list else np.ndarray)
    assert _state(result) == _state(expected)
    assert _state(vars(model)) == before


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("kernel", ["rbf", "precomputed"])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_svc_sparse_queries_preserve_expansion_and_outputs(tmp_path, format, kernel, multi, method):
    sp = pytest.importorskip("scipy.sparse")
    model = _model(tmp_path, kernel, multi)
    x = getattr(sp, format + "_matrix")(_input(2, "float32"))
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("shape,ties", [("ovo", False), ("ovr", False), ("ovr", True)])
@pytest.mark.parametrize("method", METHODS)
def test_svc_multiclass_decision_layout_and_tie_policy(tmp_path, kernel, shape, ties, method):
    model = _model(tmp_path, kernel, True, shape, ties)
    x = np.asarray([[0., 0.], [1., 1.], [-1., -1.], [1., -1.]], dtype=np.float32)
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_svc_zero_support_keeps_bias_calibration_and_votes(tmp_path, kernel, multi, method):
    model = _model(tmp_path, kernel, multi, zero=True)
    x = _input(2, "float32")
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("zero", [False, True])
def test_svc_linear_coefficients_keep_pair_sign_and_owner(tmp_path, multi, zero):
    model = _model(tmp_path, "linear", multi, zero=zero)
    expected = np.asarray(legacy.svc_coefficients(model))
    result = audit_native_call(getattr, model, "coef_").assert_native()
    assert type(result) is np.ndarray and result.dtype == np.float32
    assert _state(result) == _state(expected)
    del model
    gc.collect()
    assert _state(result) == _state(expected)


@pytest.mark.parametrize("kernel", KERNELS[1:])
def test_svc_nonlinear_coefficients_keep_attribute_refusal(tmp_path, kernel):
    _refused(audit_native_call(getattr, _model(tmp_path, kernel), "coef_"), AttributeError)


@pytest.mark.parametrize("method", METHODS[2:])
@pytest.mark.parametrize("probability", [False, True])
def test_svc_probability_method_availability_is_native(tmp_path, method, probability):
    model = _model(tmp_path, probability=probability)
    audit = audit_native_call(getattr, model, method)
    if probability:
        entry = audit.assert_native()
        assert callable(entry)
        x = _input(2, "float32")
        assert _state(audit_native_call(entry, x).assert_native()) == _state(_expected(model, x, method))
    else:
        _refused(audit, AttributeError)
        assert not hasattr(model, method)


@pytest.mark.parametrize("method", METHODS[2:])
def test_svc_missing_calibration_keeps_fitted_probability_refusal(tmp_path, method):
    model = _model(tmp_path)
    del model._prob_ab
    _refused(audit_native_call(getattr(model, method), _input(2, "float32")), AttributeError)


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_svc_internal_completed_outputs_keep_owner_lifetimes(tmp_path, multi, method):
    model = _model(tmp_path, multi=multi)
    x = _input(2, "float32")
    expected = _state(_expected(model, x, method))
    result = audit_native_call(getattr(model, method), x).assert_native()
    # Enter the public boundary explicitly; pytest module names must not
    # select the package-internal Array output policy.
    # Keep every value, subsequent-call and owner-lifetime check below.
    assert type(result) is (list if method == "predict" else np.ndarray)
    assert _state(result) == expected
    audit_native_call(getattr(model, method), x + 3).assert_native()
    del model, x
    gc.collect()
    assert _state(result) == expected


@pytest.mark.parametrize("dtype", ["int32", "int64", "float64"])
@pytest.mark.parametrize("array", [False, True])
def test_svc_binary_precomputed_support_converts_ordinary_array_words(tmp_path, dtype, array):
    model = _model(tmp_path, "precomputed")
    support = np.asarray([1, 0], dtype=dtype)
    model.support_ = Array.from_buffer(support) if array else support
    x = _input(2, "float32")
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(_expected(model, x, "predict"))


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("kind", ["fortran", "readonly"])
def test_svc_saved_support_matrix_preserves_physical_storage(tmp_path, multi, kind):
    model = _model(tmp_path, "linear", multi)
    if multi:
        for pair in model._pairs:
            value = np.asfortranarray(pair["sv"]) if kind == "fortran" else np.asarray(pair["sv"]).copy()
            if kind == "readonly":
                value.flags.writeable = False
            pair["sv"] = value
    else:
        value = np.asfortranarray(model.support_vectors_) if kind == "fortran" else np.asarray(model.support_vectors_).copy()
        if kind == "readonly":
            value.flags.writeable = False
        model.support_vectors_ = value
    x = _input(2, "float32")
    assert _state(audit_native_call(model.decision_function, x).assert_native()) == _state(_expected(model, x, "decision_function"))
    assert _state(audit_native_call(getattr, model, "coef_").assert_native()) == _state(legacy.svc_coefficients(model))


@pytest.mark.parametrize("labels", [[-2, 4], [-2., 4.5], [False, True], [None, b"warm"], [1j, 2j], [2**70, 2**71]])
@pytest.mark.parametrize("container", ["list", "tuple", "ClassLabels"])
def test_svc_native_label_kinds_preserve_result_type_and_values(tmp_path, labels, container):
    model = _model(tmp_path)
    model.classes_ = list(labels) if container == "list" else (tuple(labels) if container == "tuple" else ClassLabels(labels))
    x = _input(2, "float32")
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(_expected(model, x, "predict"))


@pytest.mark.parametrize("method", METHODS)
def test_svc_dataframe_and_warning_count_are_preserved(tmp_path, method):
    pd = pytest.importorskip("pandas")
    model = _model(tmp_path)
    x = pd.DataFrame(_input(2, "float32"), columns=["a", "b"])
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))
    model.feature_names_in_ = np.asarray(["a", "b"], dtype=object)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit_native_call(getattr(model, method), _input(2, "float32")).assert_native()
    assert len(caught) == 1


@pytest.mark.parametrize("method", METHODS)
def test_svc_nonfinite_query_keeps_existing_kernel_result(tmp_path, method):
    model = _model(tmp_path)
    x = np.asarray([[np.nan, 0], [np.inf, 0], [-np.inf, 0]], dtype=np.float32)
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))


@pytest.mark.parametrize("field", ["n_features_in_", "_gamma", "_hp_cache_size", "_hp_degree", "_hp_coef0", "_label0", "_label1", "n_support_", "probability"])
def test_svc_scalar_hooks_are_refused_before_conversion(tmp_path, field):
    model = _model(tmp_path)
    touched = []
    class Scalar:
        def __int__(self):
            touched.append("int")
            return 1
        def __float__(self):
            touched.append("float")
            return 1.
        def __bool__(self):
            touched.append("bool")
            return True
    setattr(model, field, Scalar())
    _refused(audit_native_call(model.predict, _input(2, "float32")), TypeError)
    assert touched == []


@pytest.mark.parametrize("slot", ["i", "j", "b", "calibration"])
def test_svc_pair_and_calibration_scalar_hooks_are_refused(tmp_path, slot):
    model = _model(tmp_path, multi=True)
    touched = []
    class Scalar:
        def __int__(self):
            touched.append("int")
            return 1
        def __float__(self):
            touched.append("float")
            return 1.
    if slot == "calibration":
        model._prob_ab[0] = (Scalar(), 0.)
    else:
        model._pairs[0][slot] = Scalar()
    _refused(audit_native_call(model.predict_proba, _input(2, "float32")), TypeError)
    assert touched == []


@pytest.mark.parametrize("method", METHODS)
def test_svc_input_protocols_are_refused_without_callbacks(tmp_path, method):
    model = _model(tmp_path)
    touched = []
    class Input:
        def __array__(self, *args, **kwargs):
            touched.append("array")
            return np.zeros((2, 2), dtype=np.float32)
    _refused(audit_native_call(getattr(model, method), Input()), TypeError)
    assert touched == []


@pytest.mark.parametrize("entry", ["svc_predict", "svc_pair_epilogue", "svc_dual_gemv"])
def test_svc_fixed_entries_refuse_python_replacement(tmp_path, entry, monkeypatch):
    model = _model(tmp_path, "linear")
    touched = []
    def callback(*args):
        touched.append(entry)
        raise AssertionError("Python entry called")
    monkeypatch.setitem(host._CLASSICAL["host_inference_context"]["host_inference_entries"], entry, callback)
    audit = audit_native_call(getattr, model, "coef_") if entry == "svc_dual_gemv" else audit_native_call(model.predict, _input(2, "float32"))
    _refused(audit, TypeError)
    assert touched == []


def test_svc_coefficient_property_replacement_is_refused(tmp_path, monkeypatch):
    model = _model(tmp_path, "linear")
    touched = []
    def callback(self):
        touched.append("property")
        return 1
    monkeypatch.setattr(host.SVC, "coef_", property(callback))
    _refused(audit_native_call(getattr, model, "coef_"), TypeError)
    assert touched == []


def test_svc_ordinary_subclass_keeps_methods_property_and_availability(tmp_path):
    class Child(host.HostSVC):
        pass
    base = _model(tmp_path, "linear")
    model = object.__new__(Child)
    vars(model).update(vars(base))
    x = _input(2, "float32")
    for method in METHODS:
        assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))
    assert _state(audit_native_call(getattr, model, "coef_").assert_native()) == _state(legacy.svc_coefficients(model))
    model.probability = False
    _refused(audit_native_call(getattr, model, "predict_proba"), AttributeError)


def test_svc_inherited_methods_keep_existing_application_guards():
    _subclass_guards("SVC")


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("kind", ["tuple", "ndarray", "fortran", "strided", "big_endian", "Array", "array_rows"])
def test_svc_native_calibration_storage_preserves_scalar_conversion(tmp_path, multi, kind):
    model = _model(tmp_path, multi=multi)
    value = np.asarray(model._prob_ab, dtype=np.float64)
    if kind == "tuple":
        value = tuple(tuple(row) for row in value)
    elif kind == "fortran":
        value = np.asfortranarray(value)
    elif kind == "strided":
        expanded = np.repeat(value, 2, axis=1)
        value = expanded[:, ::2]
    elif kind == "big_endian":
        value = value.astype(">f8")
    elif kind == "Array":
        value = Array.from_buffer(value)
    elif kind == "array_rows":
        value = [row.copy() for row in value]
    model._prob_ab = value
    x = _input(2, "float32")
    for method in METHODS[2:]:
        assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))


@pytest.mark.parametrize("method", METHODS[2:])
def test_svc_probability_availability_refuses_boolean_hook(tmp_path, method):
    model = _model(tmp_path)
    touched = []
    class Flag:
        def __bool__(self):
            touched.append("bool")
            return True
    model.probability = Flag()
    _refused(audit_native_call(getattr, model, method), TypeError)
    assert touched == []


@pytest.mark.parametrize("multi", [False, True])
def test_svc_classlabels_python_indexing_helpers_never_run(tmp_path, multi, monkeypatch):
    model = _model(tmp_path, multi=multi)
    model.classes_ = ClassLabels(model.classes_)
    x = _input(2, "float32")
    expected = _expected(model, x, "predict")
    touched = []
    def callback(*args, **kwargs):
        touched.append("ClassLabels")
        raise AssertionError("class label helper ran")
    with monkeypatch.context() as patch:
        for name in ("__getitem__", "__iter__", "__len__"):
            patch.setattr(ClassLabels, name, callback, raising=False)
        actual = audit_native_call(model.predict, x).assert_native()
    assert _state(actual) == _state(expected)
    assert touched == []


@pytest.mark.parametrize("field", ["dual_coef_", "support_vectors_", "pair_dual", "pair_sv"])
def test_svc_computed_property_refuses_object_storage_hooks(tmp_path, field):
    model = _model(tmp_path, "linear", field.startswith("pair"))
    touched = []
    class Scalar:
        def __float__(self):
            touched.append("float")
            return 1.
    if field.startswith("pair"):
        key = field.removeprefix("pair_")
        before = model._pairs[0][key]
    else:
        before = vars(model)[field]
    value = np.empty(before.shape, dtype=object)
    value.fill(Scalar())
    if field.startswith("pair"):
        model._pairs[0][key] = value
    else:
        setattr(model, field, value)
    _refused(audit_native_call(getattr, model, "coef_"), TypeError)
    assert touched == []


@pytest.mark.parametrize("multi", [False, True])
def test_svc_inherited_fit_is_native_and_precomputed_matches_linear(multi):
    """For every kernel the inherited fit publishes a fitted model whose every inference method is
    one native call, and a precomputed linear Gram gives the linear kernel's
    outputs bit for bit in the same build."""
    x = np.asarray([[-3, -2], [-2, -1], [-1, -2], [-2, -3], [0, 1], [1, 0],
                    [1, 2], [2, 1], [3, 4], [4, 3], [3, 3], [4, 4]], dtype=np.float32)
    y = np.asarray([0] * 4 + [1] * 4 + [2] * 4 if multi else [0] * 6 + [1] * 6, dtype=np.int64)
    outputs = {}
    for kernel in KERNELS:
        data = x @ x.T if kernel == "precomputed" else x
        model = host.HostSVC(C=1.5, kernel=kernel, gamma=.5, degree=3, coef0=.25 if kernel in ("poly", "sigmoid") else 0.,
                             probability=True, random_state=17, max_iter=100, numeric_mode="identical")
        assert model.fit(data, y) is model
        assert vars(model)["_mojolearn_fitted"] is True
        assert "_mojolearn_fitting" not in vars(model)
        outputs[kernel] = {method: np.asarray(audit_native_call(getattr(model, method), data).assert_native())
                           for method in METHODS}
    for method in METHODS:
        np.testing.assert_array_equal(outputs["precomputed"][method], outputs["linear"][method])


@pytest.mark.parametrize("method", METHODS)
def test_svc_numeric_ndarray_classes_keep_existing_decode_refusal(tmp_path, method):
    model = _model(tmp_path)
    model.classes_ = np.asarray([0, 1], dtype=np.int64)
    x = _input(2, "float32")
    if method == "predict":
        _refused(audit_native_call(model.predict, x), ValueError)
    else:
        assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))


def test_svc_binary_support_list_keeps_existing_tolist_refusal(tmp_path):
    model = _model(tmp_path, "precomputed")
    model.support_ = [0, 1]
    _refused(audit_native_call(model.predict, _input(2, "float32")), AttributeError)


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
    model = host.HostSVC.load(record["path"])
    for method in ("predict", "decision_function", "predict_proba", "predict_log_proba", "coef_"):
        if method == "coef_":
            result = audit_native_call(getattr, model, method).assert_native()
        else:
            result = audit_native_call(getattr(model, method), payload["source"]).assert_native()
        if type(result) is list:
            value = result
        else:
            assert type(result).__name__ == ("ndarray" if payload["order"] == "late" else "Array")
            value = dict(bytes=result.tobytes().hex(), shape=list(result.shape), dtype=str(result.dtype))
        rows.append(dict(case=record["case"], method=method, value=value))
if payload["order"] == "absent":
    assert "numpy" not in sys.modules
print(json.dumps(rows))
'''


@pytest.mark.parametrize("order", ["absent", "late"])
def test_svc_preserves_numpy_free_and_late_numpy_contracts(tmp_path, order):
    import json
    import os
    import subprocess
    import sys
    records, expected = [], []
    source = _input(2, "list")
    for multi in (False, True):
        case = "multi" if multi else "binary"
        folder = tmp_path / case
        folder.mkdir()
        model = _model(folder, "linear", multi)
        records.append(dict(case=case, path=str(folder / "SVC.npz")))
        for method in ("predict", "decision_function", "predict_proba", "predict_log_proba", "coef_"):
            result = legacy.svc_coefficients(model) if method == "coef_" else legacy.svc(model, source, method)
            if type(result) is list:
                value = result
            else:
                value = dict(bytes=result.tobytes().hex(), shape=list(result.shape),
                             dtype=str(np.asarray(result).dtype) if order == "late" else result.dtype)
            expected.append(dict(case=case, method=method, value=value))
    done = subprocess.run([sys.executable, "-c", _NUMPY_IMPORT_CHILD],
                          input=json.dumps(dict(order=order, records=records, source=source)), text=True,
                          capture_output=True, env=dict(os.environ), timeout=90)
    assert done.returncode == 0, done.stderr[-6000:]
    assert json.loads(done.stdout) == expected


@pytest.mark.parametrize("kind", ["big_endian", "int16", "bool", "uint64"])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_svc_additional_native_query_casts_preserve_results(tmp_path, kind, multi, method):
    model = _model(tmp_path, multi=multi)
    x = _input(2, kind)
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))


@pytest.mark.parametrize("kernel,degree,gamma,coef0,cache", [
    ("poly", 0, 0., -1.25, 1.), ("poly", 2, .125, .75, 64.),
    ("poly", 32, .125, -.25, 64.), ("sigmoid", 3, 0., -1.25, 1.),
    ("sigmoid", 3, 1.5, .75, 64.), ("rbf", 3, 0., 0., 1.),
])
@pytest.mark.parametrize("method", METHODS)
def test_svc_kernel_controls_retain_existing_numerical_parameters(tmp_path, kernel, degree, gamma, coef0, cache, method):
    model = _model(tmp_path, kernel, True)
    model._hp_degree, model._gamma, model._hp_coef0, model._hp_cache_size = degree, gamma, coef0, cache
    x = _input(2, "float32")
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, method))


@pytest.mark.parametrize("method", METHODS[2:])
def test_svc_cached_probability_method_keeps_fitted_calibration_contract(tmp_path, method):
    model = _model(tmp_path)
    x = _input(2, "float32")
    expected = _expected(model, x, method)
    bound = getattr(model, method)
    model.probability = False
    _refused(audit_native_call(getattr, model, method), AttributeError)
    assert _state(audit_native_call(bound, x).assert_native()) == _state(expected)
    assert _state(audit_native_call(getattr(type(model), method), model, x).assert_native()) == _state(expected)
