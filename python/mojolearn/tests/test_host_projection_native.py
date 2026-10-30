# SPDX-License-Identifier: Apache-2.0
"""Explicit CPU projection retains the complete native operation boundary."""
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
from mojolearn.tests.test_host_kernel_inference_native import FORMATS, _duplicates

CASES = ("pca", "whiten", "tsvd")
METHODS = ("transform", "inverse_transform")


def _model(tmp_path, case):
    name = "TruncatedSVD" if case == "tsvd" else "PCA"
    model = getattr(host, "Host" + name).load(_archive(tmp_path, name))
    if name == "PCA":
        model.whiten = case == "whiten"
    return model


def _expected(model, x, case, method):
    return legacy.projection(model, x, "TruncatedSVD" if case == "tsvd" else "PCA", method == "inverse_transform")


def _query(method, kind="float32"):
    return _input(2 if method == "inverse_transform" else 3, kind)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["float32", "list", "fortran", "strided", "big_endian", "half", "int16", "bool", "uint64"])
def test_projection_dense_public_outputs_preserve_exact_bits(tmp_path, case, method, kind):
    model = _model(tmp_path, case)
    x = _query(method, kind)
    before = _state(vars(model))
    expected = _expected(model, x, case, method)
    actual = audit_native_call(getattr(model, method), x).assert_native()
    assert type(actual) is np.ndarray
    assert actual.dtype == np.dtype("<f4")
    assert _state(actual) == _state(expected)
    assert _state(vars(model)) == before


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("format", FORMATS)
def test_projection_sparse_formats_keep_consumed_outputs(tmp_path, case, method, format):
    sp = pytest.importorskip("scipy.sparse")
    model = _model(tmp_path, case)
    x = getattr(sp, format + "_matrix")(_query(method))
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, case, method))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("format", FORMATS)
def test_projection_sparse_array_containers_keep_results(tmp_path, method, format):
    sp = pytest.importorskip("scipy.sparse")
    model = _model(tmp_path, "whiten")
    x = getattr(sp, format + "_array")(_query(method, "int16"))
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, "whiten", method))


@pytest.mark.parametrize("format", ["csr", "csc", "coo", "bsr", "lil"])
@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_projection_sparse_duplicate_order_is_unchanged(tmp_path, format, dtype):
    model = _model(tmp_path, "tsvd")
    x = _duplicates(format, dtype)
    assert _state(audit_native_call(model.transform, x).assert_native()) == _state(_expected(model, x, "tsvd", "transform"))


@pytest.mark.parametrize("format", FORMATS + ("bsr_endian", "dia_endian"))
def test_projection_preserves_existing_sparse_storage_refusals(tmp_path, format):
    sp = pytest.importorskip("scipy.sparse")
    model = _model(tmp_path, "tsvd")
    if format.endswith("_endian"):
        x = getattr(sp, format.split("_")[0] + "_matrix")(_query("transform"))
        x.data = x.data.astype(">f8")
        error = ValueError
    else:
        x = getattr(sp, format + "_matrix")(np.asarray([[2**63 + 1, 0, 1]], dtype=np.uint64))
        error = OverflowError
    with pytest.raises(error):
        _expected(model, x, "tsvd", "transform")
    _refused(audit_native_call(model.transform, x), error)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", METHODS)
def test_projection_internal_outputs_and_owners_survive_later_calls(tmp_path, case, method):
    model = _model(tmp_path, case)
    x = _query(method)
    result = getattr(model, method)(x)
    assert type(result).__name__ == "Array"
    expected = _state(_expected(model, x, case, method))
    assert _state(result) == expected
    other = audit_native_call(getattr(model, method), x + 2).assert_native()
    del x, model
    gc.collect()
    assert _state(result) == expected
    assert np.asarray(other).shape == np.asarray(result).shape


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_projection_nonfinite_dense_values_are_refused_natively(tmp_path, case, method, value):
    model = _model(tmp_path, case)
    x = _query(method)
    x[0, 0] = value
    _refused(audit_native_call(getattr(model, method), x))


@pytest.mark.parametrize("case", ["pca", "tsvd"])
@pytest.mark.parametrize("method", METHODS)
def test_unwhitened_projection_keeps_finite_float64_overflow_cast(tmp_path, case, method):
    model = _model(tmp_path, case)
    x = _query(method).astype(np.float64)
    x[0, 0] = 1e300
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, case, method))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", METHODS)
def test_sparse_nonfinite_values_keep_existing_kernel_contract(tmp_path, case, method):
    sp = pytest.importorskip("scipy.sparse")
    model = _model(tmp_path, case)
    x = _query(method)
    x[0, 0] = np.nan
    x = sp.csr_matrix(x)
    if case == "whiten":
        with pytest.raises(Exception, match="finite"):
            _expected(model, x, case, method)
        _refused(audit_native_call(getattr(model, method), x))
    else:
        assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, case, method))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", METHODS)
def test_dataframe_projection_preserves_preparation_and_results(tmp_path, case, method):
    pd = pytest.importorskip("pandas")
    model = _model(tmp_path, case)
    x = pd.DataFrame(_query(method).astype(np.float64))
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x.to_numpy(), case, method))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", METHODS)
def test_projection_width_and_unfitted_guards_are_native(tmp_path, case, method):
    model = _model(tmp_path, case)
    _refused(audit_native_call(getattr(model, method), np.zeros((3, 5), dtype=np.float32)))
    unfitted = object.__new__(type(model))
    _refused(audit_native_call(getattr(unfitted, method), _query(method)))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", METHODS)
def test_explicit_cpu_projection_refuses_saved_fast_mode(tmp_path, case, method):
    model = _model(tmp_path, case)
    model.numeric_mode = "fast"
    _refused(audit_native_call(getattr(model, method), _query(method)))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", METHODS)
def test_projection_method_lookup_attributes_and_subclasses_are_native(tmp_path, case, method):
    base = _model(tmp_path, case)
    cls = type("OrdinaryProjectionSubclass", (type(base),), {})
    model = cls.load(_archive(tmp_path, "TruncatedSVD" if case == "tsvd" else "PCA"))
    if case != "tsvd":
        model.whiten = case == "whiten"
    entry = audit_native_call(getattr, model, method).assert_native()
    value = audit_native_call(getattr, model, "components_").assert_native()
    assert type(value) is np.ndarray
    assert np.shares_memory(value, np.asarray(vars(model)["components_"]))
    x = _query(method)
    assert _state(audit_native_call(entry, x).assert_native()) == _state(_expected(model, x, case, method))


@pytest.mark.parametrize("name", ["PCA", "TruncatedSVD"])
def test_projection_installation_preserves_inherited_fit_and_score_guards(name):
    _subclass_guards(name)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_svd_inherited_training_keeps_nested_projection_state_and_bits(method):
    # Frozen 32011bb32 CPU fit/fit_transform receipt. Training remains existing
    # runtime debt; changing its nested transform must preserve these results.
    x = np.asarray([[-2., -1., 2.], [-1., -2., 1.], [0., 0., 0.], [.5, .25, 1.],
                    [2., 1., -1.], [1., 2., 3.], [3., 3., 2.], [4., 3., 1.]], dtype="<f4")
    model = host.HostTruncatedSVD(n_components=2, algorithm="covariance_eigh", numeric_mode="identical")
    result = getattr(model, method)(x)
    if method == "fit":
        assert result is model
        result = model.transform(x)
    result = np.asarray(result)
    assert result.dtype == np.dtype("<f4")
    assert result.shape == (8, 2)
    assert result.tobytes().hex() == (
        "a0d4c6bf702524405433e0bf6ecea83f0000000000000000c0ef473fe7ea463f"
        "7056e83f6e45cfbf4a1d3340aea31d4041a49340dc03583fbf18a2406257dabe")
    assert vars(model)["_mojolearn_fitted"] is True
    assert "_mojolearn_fitting" not in vars(model)
    assert model.n_features_in_ == 3


@pytest.mark.parametrize("field", ["components_", "mean_", "singular_values_", "explained_variance_"])
@pytest.mark.parametrize("method", METHODS)
def test_whitening_accepts_ordinary_readonly_numpy_state(tmp_path, field, method):
    model = _model(tmp_path, "whiten")
    value = np.array(getattr(model, field), copy=True)
    value.flags.writeable = False
    setattr(model, field, value)
    x = _query(method)
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, "whiten", method))


@pytest.mark.parametrize("fault", ["shape", "dtype", "fortran", "negative_singular", "nonfinite_mean", "samples"])
@pytest.mark.parametrize("method", METHODS)
def test_whitening_invalid_state_is_refused_before_prediction(tmp_path, fault, method):
    model = _model(tmp_path, "whiten")
    if fault == "shape":
        model.explained_variance_ = np.ones(1, dtype=np.float32)
    elif fault == "dtype":
        model.singular_values_ = np.ones(2, dtype=np.float64)
    elif fault == "fortran":
        model.components_ = np.asfortranarray(np.asarray(model.components_))
    elif fault == "negative_singular":
        model.singular_values_ = np.asarray([-1, 2], dtype=np.float32)
    elif fault == "nonfinite_mean":
        model.mean_ = np.asarray([np.nan, 1, 2], dtype=np.float32)
    else:
        model.n_samples_ = 1
    _refused(audit_native_call(getattr(model, method), _query(method)))


@pytest.mark.parametrize("variance", [-1., np.nan, np.inf])
@pytest.mark.parametrize("method", METHODS)
def test_whitening_preserves_metadata_only_explained_variance_contract(tmp_path, variance, method):
    # The old _validate_whiten_state checks this field's dtype/shape/order.
    # Its values are not consumed: whitening uses singular_values_, and the
    # CPU ABI receives no explained_variance_ pointer. Preserve that behavior
    # separately from the required singular-value/mean refusal tests above.
    model = _model(tmp_path, "whiten")
    model.explained_variance_ = np.asarray([variance, 1.], dtype=np.float32)
    x = _query(method)
    expected = _expected(model, x, "whiten", method)
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(expected)


@pytest.mark.parametrize("case", ["pca", "tsvd"])
@pytest.mark.parametrize("method", METHODS)
def test_unwhitened_components_keep_existing_fortran_physical_layout(tmp_path, case, method):
    model = _model(tmp_path, case)
    model.components_ = np.asfortranarray(np.asarray(model.components_))
    x = _query(method)
    assert _state(audit_native_call(getattr(model, method), x).assert_native()) == _state(_expected(model, x, case, method))


@pytest.mark.parametrize("field", ["n_features_in_", "n_components_", "n_samples_", "whiten", "_mojolearn_fitted", "_mojolearn_fitting"])
def test_projection_scalar_state_cannot_invoke_element_hooks(tmp_path, field):
    model = _model(tmp_path, "whiten")
    touched = []
    class Hook:
        def __int__(self):
            touched.append("int")
            return 2
        def __index__(self):
            touched.append("index")
            return 2
        def __bool__(self):
            touched.append("bool")
            return True
    setattr(model, field, np.asarray(Hook(), dtype=object))
    _refused(audit_native_call(model.transform, _query("transform")), TypeError)
    assert touched == []


@pytest.mark.parametrize("field", ["n_features_in_", "n_components_", "n_samples_", "whiten"])
def test_projection_ordinary_scalar_arrays_keep_legacy_behavior(tmp_path, field):
    model = _model(tmp_path, "whiten")
    setattr(model, field, np.asarray(getattr(model, field)))
    x = _query("transform")
    assert _state(audit_native_call(model.transform, x).assert_native()) == _state(_expected(model, x, "whiten", "transform"))


@pytest.mark.parametrize("case", CASES)
def test_projection_refuses_python_input_protocols_before_calling_them(tmp_path, case):
    model = _model(tmp_path, case)
    touched = []
    class Protocol:
        def __array__(self, *args, **kwargs):
            touched.append("array")
            return _query("transform")
        def toarray(self):
            touched.append("toarray")
            return _query("transform")
        @property
        def nnz(self):
            touched.append("nnz")
            return 1
    _refused(audit_native_call(model.transform, Protocol()), TypeError)
    assert touched == []


@pytest.mark.parametrize("entry,case,method", [
    ("tsvd_transform", "tsvd", "transform"), ("pca_transform", "pca", "transform"),
    ("inverse_transform", "tsvd", "inverse_transform"),
    ("pca_whiten_transform", "whiten", "transform"),
    ("pca_whiten_inverse_transform", "whiten", "inverse_transform")])
def test_projection_refuses_replaced_native_entries_without_callback(tmp_path, entry, case, method, monkeypatch):
    model = _model(tmp_path, case)
    touched = []
    def callback(*args):
        touched.append(True)
    monkeypatch.setitem(host._CLASSICAL["host_inference_context"]["host_inference_entries"], entry, callback)
    _refused(audit_native_call(getattr(model, method), _query(method)), TypeError)
    assert touched == []


@pytest.mark.parametrize("case", CASES)
def test_only_forward_projection_preserves_feature_name_warning(tmp_path, case):
    model = _model(tmp_path, case)
    model.feature_names_in_ = np.asarray(["a", "b", "c"], dtype=object)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit_native_call(model.transform, _query("transform")).assert_native()
        audit_native_call(model.inverse_transform, _query("inverse_transform")).assert_native()
    assert len(caught) == 1
    assert "was fitted with feature names" in str(caught[0].message)


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
    model = getattr(host, record["class"]).load(record["path"])
    if record["case"] != "tsvd":
        model.whiten = record["case"] == "whiten"
    for method, source in record["queries"].items():
        result = audit_native_call(getattr(model, method), source).assert_native()
        assert type(result).__name__ == ("ndarray" if payload["order"] == "late" else "Array")
        rows.append({"case":record["case"], "method":method, "bytes":result.tobytes().hex(), "shape":list(result.shape)})
if payload["order"] == "absent":
    assert "numpy" not in sys.modules
print(json.dumps(rows))
'''


@pytest.mark.parametrize("order", ["absent", "late"])
def test_projection_preserves_numpy_free_and_late_numpy_contracts(tmp_path, order):
    records, expected = [], []
    for case in CASES:
        name = "TruncatedSVD" if case == "tsvd" else "PCA"
        model = _model(tmp_path, case)
        queries = {method: _query(method, "list") for method in METHODS}
        records.append(dict(case=case, **{"class": "Host" + name}, path=str(_archive(tmp_path, name)), queries=queries))
        for method, source in queries.items():
            result = _expected(model, source, case, method)
            expected.append(dict(case=case, method=method, bytes=result.tobytes().hex(), shape=list(result.shape)))
    done = subprocess.run([sys.executable, "-c", _NUMPY_IMPORT_CHILD],
                          input=json.dumps(dict(order=order, records=records)), text=True,
                          capture_output=True, env=dict(os.environ), timeout=90)
    assert done.returncode == 0, done.stderr[-6000:]
    assert json.loads(done.stdout) == expected


def test_projection_public_methods_are_one_native_binding_call():
    tree = ast.parse(Path(host.__file__).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_HostProjection")
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in METHODS:
            body = node.body[1:]
            assert len(body) == 1 and isinstance(body[0], ast.Return)
            call = body[0].value
            assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            assert call.func.id == "_native_classical"
