# SPDX-License-Identifier: Apache-2.0
"""Whole CPU kernel-model operations preserve dense/sparse public behavior."""
import ast
from pathlib import Path

import numpy as np
import pytest

from mojolearn import _classical_host as host
from mojolearn._buffer import frombytes
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_host_inference as legacy
from mojolearn.tests.test_host_archive_native import _archive, _state
from mojolearn.tests.test_host_inference_native import _input, _refused, test_application_subclass_keeps_constructor_fit_and_score_guards as _subclass_guards

NAMES = ("KernelRidge", "Nystroem", "RBFSampler")
FORMATS = ("csr", "csc", "coo", "bsr", "dia", "dok", "lil")


def _model(tmp_path, name):
    return getattr(host, "Host" + name).load(_archive(tmp_path, name))


def _method(model, name):
    return model.predict if name == "KernelRidge" else model.transform


def _sparse(format, container="matrix", dtype="float64"):
    sp = pytest.importorskip("scipy.sparse")
    values = np.asarray([[0, 2, -1], [3, 0, 1], [0, 0, 0], [1, 2, 0]], dtype="<i8")
    if np.dtype(dtype).kind in "ub":
        values = np.abs(values)
    return getattr(sp, format + "_" + container)(values.astype(dtype))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("kind", ["float32", "list", "fortran", "strided", "big_endian", "half", "int16", "bool", "uint64"])
def test_kernel_dense_input_preparation_and_public_bits(tmp_path, name, kind):
    model = _model(tmp_path, name)
    x = _input(3, kind)
    expected = legacy.kernel_model(model, x, name)
    assert _state(audit_native_call(_method(model, name), x).assert_native()) == _state(expected)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("container", ["matrix", "array"])
def test_all_supported_sparse_formats_keep_consumed_prediction_bits(tmp_path, name, format, container):
    model = _model(tmp_path, name)
    x = _sparse(format, container)
    expected = legacy.kernel_model(model, x, name)
    assert _state(audit_native_call(_method(model, name), x).assert_native()) == _state(expected)


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("dtype", ["bool", "int8", "uint16", "int32", "uint64", "longlong", "ulonglong", "float32", "float64"])
def test_sparse_source_dtype_cast_is_preserved(tmp_path, format, dtype):
    model = _model(tmp_path, "Nystroem")
    x = _sparse(format, dtype=dtype)
    if dtype == "uint64":
        dense = np.asarray([[2**53 + 2**29 + 1, 0, 2**63 - 1], [1, 2, 0]], dtype=dtype)
        x = type(x)(dense)
    expected = legacy.kernel_model(model, x, "Nystroem")
    assert _state(audit_native_call(model.transform, x).assert_native()) == _state(expected)


def _duplicates(format, dtype):
    sp = pytest.importorskip("scipy.sparse")
    if np.dtype(dtype).kind == "f":
        data = np.asarray([2**24, 1, -2**24, -0.0, np.nextafter(np.float32(0), np.float32(1))], dtype=dtype)
    elif dtype == "int8":
        data = np.asarray([127, 1, 1, 0, 1], dtype=dtype)
    else:
        data = np.asarray([1, 1, 0, 0, 1], dtype=dtype)
    if format == "coo":
        return sp.coo_matrix((data, (np.asarray([0, 0, 0, 1, 1]), np.asarray([0, 0, 0, 1, 2]))), shape=(2, 3))
    if format == "csr":
        return sp.csr_matrix((data, np.asarray([0, 0, 0, 1, 2]), np.asarray([0, 3, 5])), shape=(2, 3))
    if format == "csc":
        return sp.csc_matrix((data, np.asarray([0, 0, 0, 1, 1]), np.asarray([0, 3, 4, 5])), shape=(2, 3))
    if format == "bsr":
        return sp.bsr_matrix((data.reshape(-1, 1, 1), np.asarray([0, 0, 0, 1, 2]), np.asarray([0, 3, 5])), shape=(2, 3))
    x = sp.lil_matrix((2, 3), dtype=dtype)
    x.rows[0], x.data[0] = [0, 0, 0], data[:3].tolist()
    x.rows[1], x.data[1] = [1, 2], data[3:].tolist()
    return x


@pytest.mark.parametrize("format", ["csr", "csc", "coo", "bsr", "lil"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int8", "bool"])
def test_sparse_duplicate_order_and_original_dtype_accumulation(tmp_path, format, dtype):
    model = _model(tmp_path, "KernelRidge")
    # Linear kernel consumes the expanded entries without a nonlinear kernel
    # hiding differences from premature f32 narrowing or integer overflow.
    model._kernel_params = (0, 3, .1, 1., .25)
    x = _duplicates(format, dtype)
    expected = legacy.kernel_model(model, x, "KernelRidge")
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(expected)


@pytest.mark.parametrize("format", ["csr", "csc", "coo", "bsr", "dia"])
def test_sparse_strides_endian_and_wide_indices(tmp_path, format):
    model = _model(tmp_path, "Nystroem")
    x = _sparse(format, dtype="float64")
    original = x.data
    # BSR/DIA's existing COO conversion rejects nonnative byte order. Their
    # accepted strided native storage remains covered; CSR/CSC/COO additionally
    # preserve their accepted big-endian data buffers.
    store = np.zeros(original.shape[:-1] + (original.shape[-1] * 2,),
                     dtype="<f8" if format in ("bsr", "dia") else ">f8")
    store[..., ::2] = original
    x.data = store[..., ::2]
    if format == "coo":
        x.coords = tuple(a.astype(np.int64) for a in x.coords)
    elif format == "dia":
        x.offsets = x.offsets.astype(np.int64)
    else:
        x.indices = x.indices.astype(np.int64)
        x.indptr = x.indptr.astype(np.int64)
    expected = legacy.kernel_model(model, x, "Nystroem")
    assert _state(audit_native_call(model.transform, x).assert_native()) == _state(expected)


@pytest.mark.parametrize("format", FORMATS + ("dense",))
def test_uint64_outside_legacy_signed_storage_range_keeps_native_refusal(tmp_path, format):
    model = _model(tmp_path, "Nystroem")
    x = np.asarray([[2**63 + 1, 0, 1], [1, 2, 0]], dtype=np.uint64)
    if format != "dense":
        x = type(_sparse(format))(x)
    with pytest.raises(OverflowError, match="int too big"):
        legacy.kernel_model(model, x, "Nystroem")
    _refused(audit_native_call(model.transform, x), OverflowError)


@pytest.mark.parametrize("format", ["bsr", "dia"])
def test_block_and_diagonal_nonnative_byte_order_keeps_native_refusal(tmp_path, format):
    model = _model(tmp_path, "Nystroem")
    x = _sparse(format)
    x.data = x.data.astype(">f8")
    with pytest.raises(ValueError, match="does not support dtype"):
        legacy.kernel_model(model, x, "Nystroem")
    _refused(audit_native_call(model.transform, x))


@pytest.mark.parametrize("name", NAMES)
def test_block_sparse_rectangular_blocks_keep_layout(tmp_path, name):
    sp = pytest.importorskip("scipy.sparse")
    model = _model(tmp_path, name)
    x = sp.bsr_matrix(np.arange(12, dtype=np.float32).reshape(4, 3), blocksize=(2, 3))
    expected = legacy.kernel_model(model, x, name)
    assert _state(audit_native_call(_method(model, name), x).assert_native()) == _state(expected)


@pytest.mark.parametrize("format", FORMATS)
def test_sparse_python_conversion_methods_are_never_entered(tmp_path, format, monkeypatch):
    model = _model(tmp_path, "Nystroem")
    x = _sparse(format)
    expected = legacy.kernel_model(model, x, "Nystroem")
    touched = []
    def poisoned(*args, **kwargs):
        touched.append("hook")
        raise AssertionError("Python sparse conversion or property was called")
    for name in ("toarray", "tocsr", "tocoo", "__getattribute__"):
        monkeypatch.setattr(type(x), name, poisoned)
    assert _state(audit_native_call(model.transform, x).assert_native()) == _state(expected)
    assert touched == []


@pytest.mark.parametrize("format", FORMATS)
def test_sparse_subclasses_are_refused_without_input_hooks(tmp_path, format):
    model = _model(tmp_path, "Nystroem")
    original = _sparse(format)
    touched = []
    class Custom(type(original)):
        def toarray(self, *args, **kwargs):
            touched.append("toarray")
            return original.toarray()
    x = Custom(original)
    _refused(audit_native_call(model.transform, x), TypeError)
    assert touched == []


@pytest.mark.parametrize("format", ["csr", "dok", "lil"])
def test_sparse_custom_metaclasses_are_refused_without_metadata_hooks(tmp_path, format):
    model = _model(tmp_path, "Nystroem")
    original = _sparse(format)
    touched = []
    class Meta(type(type(original))):
        def __getattribute__(cls, name):
            touched.append(name)
            return super().__getattribute__(name)
    class Custom(type(original), metaclass=Meta):
        pass
    x = Custom(original)
    touched.clear()  # class construction is test preparation
    _refused(audit_native_call(model.transform, x), TypeError)
    assert touched == []


@pytest.mark.parametrize("format", ["csr", "lil"])
def test_exact_sparse_classes_with_python_finalizers_are_refused(tmp_path, format, monkeypatch):
    model = _model(tmp_path, "Nystroem")
    x = _sparse(format)
    touched = []
    def finalizer(self):
        touched.append("del")
    with monkeypatch.context() as patch:
        patch.setattr(type(x), "__del__", finalizer, raising=False)
        audit = audit_native_call(model.transform, x)
    _refused(audit, TypeError)
    assert touched == []


@pytest.mark.parametrize("format", ["dok", "lil"])
def test_sparse_object_values_are_admitted_before_conversion_hooks(tmp_path, format):
    model = _model(tmp_path, "Nystroem")
    x = _sparse(format)
    touched = []
    class Value:
        def __float__(self):
            touched.append("float")
            return 1.0
    if format == "dok":
        x._dict[(0, 1)] = Value()
    else:
        x.data[0][0] = Value()
    _refused(audit_native_call(model.transform, x), TypeError)
    assert touched == []


@pytest.mark.parametrize("damage", ["pointer", "index", "coordinate", "block_count", "shape", "dtype_hook"])
def test_sparse_invalid_storage_is_refused_before_native_reads(tmp_path, damage):
    model = _model(tmp_path, "Nystroem")
    touched = []
    x = _sparse("coo" if damage == "coordinate" else "csr")
    exception = ValueError
    if damage == "pointer":
        x.indptr[-1] = x.data.size + 1
    elif damage == "index":
        x.indices[0] = 100
    elif damage == "coordinate":
        x.coords[0][0] = 100
    elif damage == "block_count":
        x = _sparse("bsr")
        x.data = x.data[:-1]
    elif damage == "shape":
        vars(x)["_shape"] = (-1, 3)
    else:
        x = _sparse("dok")
        class Dtype:
            @property
            def itemsize(self):
                touched.append("itemsize")
                return 8
        vars(x)["dtype"] = Dtype()
        exception = TypeError
    _refused(audit_native_call(model.transform, x), exception)
    assert touched == []


@pytest.mark.parametrize("name", NAMES)
def test_kernel_dataframe_preparation_retains_public_results(tmp_path, name):
    pd = pytest.importorskip("pandas")
    model = _model(tmp_path, name)
    x = pd.DataFrame(_input(3, "float32"), columns=["a", "b", "c"])
    expected = legacy.kernel_model(model, x, name)
    assert _state(audit_native_call(_method(model, name), x).assert_native()) == _state(expected)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rbf_dense_finite_validation_stays_before_native_kernel(tmp_path, value):
    model = _model(tmp_path, "RBFSampler")
    _refused(audit_native_call(model.transform, np.asarray([[value, 1., 2.]], dtype="<f8")))


@pytest.mark.parametrize("name", NAMES)
def test_kernel_unfitted_and_wrong_mode_fail_without_python(tmp_path, name):
    model = object.__new__(getattr(host, "Host" + name))
    _refused(audit_native_call(_method(model, name), [[1., 2., 3.]]))
    model = _model(tmp_path, name)
    model.numeric_mode = "fast"
    _refused(audit_native_call(_method(model, name), [[1., 2., 3.]]))


@pytest.mark.parametrize("name", NAMES)
def test_kernel_none_mode_keeps_explicit_cpu_behavior_with_fast_default(tmp_path, name, monkeypatch):
    from mojolearn import _backend
    model = _model(tmp_path, name)
    model.numeric_mode = None
    monkeypatch.setattr(_backend, "_DEFAULT_MODE", "fast")
    x = [[1., 2., 3.]]
    expected = legacy.kernel_model(model, x, name)
    assert _state(audit_native_call(_method(model, name), x).assert_native()) == _state(expected)


@pytest.mark.parametrize("name", NAMES)
def test_kernel_application_subclasses_keep_inherited_fit_and_score_guards(name):
    _subclass_guards(name)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (2, 4), (3,)])
def test_kernel_invalid_shapes_are_refused_without_python(tmp_path, name, shape):
    model = _model(tmp_path, name)
    _refused(audit_native_call(_method(model, name), np.zeros(shape, dtype=np.float32)))


@pytest.mark.parametrize("name", ["KernelRidge", "Nystroem"])
@pytest.mark.parametrize("kernel", [0, 1, 2, 3, 5])
def test_all_saved_cpu_kernel_codes_retain_prediction_bits(tmp_path, name, kernel):
    model = _model(tmp_path, name)
    model._kernel_params = (kernel,) + model._kernel_params[1:]
    x = _input(3, "float32")
    expected = legacy.kernel_model(model, x, name)
    assert _state(audit_native_call(_method(model, name), x).assert_native()) == _state(expected)


def test_kernel_ridge_single_target_keeps_public_dtype_and_rank(tmp_path):
    model = _model(tmp_path, "KernelRidge")
    model.n_targets_, model._squeeze = 1, True
    model.dual_coef_ = frombytes(np.asarray([1., 2.], dtype="<f4").tobytes(), "<f4", (2,))
    x = _input(3, "float32")
    expected = legacy.kernel_model(model, x, "KernelRidge")
    assert _state(audit_native_call(model.predict, x).assert_native()) == _state(expected)


@pytest.mark.parametrize("name", NAMES)
def test_kernel_subclass_and_fitted_attribute_publication(tmp_path, name):
    cls = type("NativeKernelSubclass", (getattr(host, "Host" + name),), {})
    model = cls.load(_archive(tmp_path, name))
    field = {"KernelRidge": "dual_coef_", "Nystroem": "components_", "RBFSampler": "random_weights_"}[name]
    assert type(audit_native_call(getattr, model, field).assert_native()) is np.ndarray
    expected = legacy.kernel_model(model, _input(3, "float32"), name)
    assert _state(audit_native_call(_method(model, name), _input(3, "float32")).assert_native()) == _state(expected)


def test_kernel_public_methods_are_single_native_calls():
    tree = ast.parse(Path(host.__file__).read_text())
    wanted = {"HostKernelRidge": "predict", "HostNystroem": "transform", "HostRBFSampler": "transform"}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in wanted:
            continue
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == wanted[cls.name])
        assert len(method.body) == 2
        assert isinstance(method.body[1], ast.Return)
        call = method.body[1].value
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        assert call.func.id == "_native_classical"


SCALAR_SLOTS = [
    ("KernelRidge", "n_features_in_", None), ("KernelRidge", "n_targets_", None),
    *(('KernelRidge', '_kernel_params', i) for i in range(5)),
    ("KernelRidge", "info_", None), ("KernelRidge", "_squeeze", None),
    ("Nystroem", "_kernel_params", 0), ("Nystroem", "_kernel_params", 2),
    ("Nystroem", "sweeps_", None), ("RBFSampler", "_params", 0),
    ("RBFSampler", "_params", 1), ("RBFSampler", "sigma_", None), ("RBFSampler", "scale_", None),
]


def _set_scalar_slot(model, field, item, value):
    if item is None:
        setattr(model, field, value)
    else:
        values = list(vars(model)[field])
        values[item] = value
        setattr(model, field, tuple(values))


@pytest.mark.parametrize("name,field,item", SCALAR_SLOTS)
def test_kernel_object_array_scalar_state_is_refused_before_element_hooks(tmp_path, name, field, item):
    model = _model(tmp_path, name)
    touched = []
    class Value:
        def __int__(self):
            touched.append("int")
            return 1
        def __index__(self):
            touched.append("index")
            return 1
        def __float__(self):
            touched.append("float")
            return 1.0
        def __bool__(self):
            touched.append("bool")
            return True
    value = np.empty((), dtype=object)
    value[()] = Value()
    _set_scalar_slot(model, field, item, value)
    _refused(audit_native_call(_method(model, name), [[1., 2., 3.]]), TypeError)
    assert touched == []


@pytest.mark.parametrize("name", NAMES + ("ElasticNet", "Lasso", "KMeans"))
@pytest.mark.parametrize("field", ["_mojolearn_fitted", "_mojolearn_fitting"])
def test_host_fitted_state_object_arrays_never_call_element_truth_hooks(tmp_path, name, field):
    model = _model(tmp_path, name)
    features = model.n_features_in_
    touched = []
    class Value:
        def __bool__(self):
            touched.append("bool")
            return True
    value = np.empty((), dtype=object)
    value[()] = Value()
    setattr(model, field, value)
    method = model.predict if name in ("KernelRidge", "ElasticNet", "Lasso", "KMeans") else model.transform
    _refused(audit_native_call(method, _input(features, "float32")), TypeError)
    assert touched == []


@pytest.mark.parametrize("name,field,item", SCALAR_SLOTS)
def test_ordinary_zero_dimensional_scalar_state_keeps_legacy_results(tmp_path, name, field, item):
    model = _model(tmp_path, name)
    original = vars(model)[field] if item is None else vars(model)[field][item]
    _set_scalar_slot(model, field, item, np.asarray(original))
    x = _input(3, "float32")
    expected = legacy.kernel_model(model, x, name)
    assert _state(audit_native_call(_method(model, name), x).assert_native()) == _state(expected)


@pytest.mark.parametrize("name", NAMES + ("ElasticNet", "Lasso", "KMeans"))
@pytest.mark.parametrize("dtype,shape", [("bool", ()), ("int32", (1,)), ("complex64", (1, 1))])
def test_numeric_fitted_flag_arrays_keep_legacy_truth_semantics(tmp_path, name, dtype, shape):
    model = _model(tmp_path, name)
    model._mojolearn_fitted = np.ones(shape, dtype=dtype)
    model._mojolearn_fitting = np.zeros(shape, dtype=dtype)
    x = _input(model.n_features_in_, "float32")
    if name in NAMES:
        expected = legacy.kernel_model(model, x, name)
    elif name == "KMeans":
        expected = legacy.kmeans(model, x)
    else:
        expected = legacy.public_cd_predict(model, x)
    method = model.predict if name in ("KernelRidge", "ElasticNet", "Lasso", "KMeans") else model.transform
    assert _state(audit_native_call(method, x).assert_native()) == _state(expected)


@pytest.mark.parametrize("storage", ["shape", "array", "dict", "row_list", "coordinate", "field_key"])
def test_sparse_mutated_metadata_cannot_enter_python_hooks(tmp_path, storage):
    model = _model(tmp_path, "Nystroem")
    x = _sparse("dok" if storage in ("dict", "coordinate") else ("lil" if storage == "row_list" else "csr"))
    touched = []
    class Number(int):
        def __int__(self):
            touched.append("int")
            return 1
        def __index__(self):
            touched.append("index")
            return 1
    exception = TypeError
    if storage == "shape":
        vars(x)["_shape"] = (Number(4), 3)
        exception = ValueError
    elif storage == "array":
        class Storage(np.ndarray):
            def __getattribute__(self, name):
                touched.append(name)
                return super().__getattribute__(name)
        vars(x)["data"] = np.asarray([1.]).view(Storage)
    elif storage == "dict":
        class Dictionary(dict):
            def items(self):
                touched.append("items")
                return super().items()
        vars(x)["_dict"] = Dictionary(vars(x)["_dict"])
    elif storage == "row_list":
        class Row(list):
            def __len__(self):
                touched.append("len")
                return super().__len__()
        vars(x)["data"][0] = Row([1., 2.])
    elif storage == "coordinate":
        vars(x)["_dict"] = {(Number(0), 0): 1.}
    else:
        class Key(str):
            __hash__ = str.__hash__
            def __eq__(self, other):
                touched.append("eq")
                return super().__eq__(other)
        object.__setattr__(x, "__dict__", {Key("_shape"): (4, 3)})
    # ndarray subclass construction may query __array_finalize__; fixture
    # construction is outside the audited product operation.
    touched.clear()
    _refused(audit_native_call(model.transform, x), exception)
    assert touched == []


@pytest.mark.parametrize("value", [2**63, 2**63 + 1, 2**64 - 1])
@pytest.mark.parametrize("overwritten", [False, True])
def test_lil_uint64_high_word_keeps_final_storage_refusal(tmp_path, value, overwritten):
    sp = pytest.importorskip("scipy.sparse")
    model = _model(tmp_path, "Nystroem")
    x = sp.lil_matrix((2, 3), dtype=np.uint64)
    # LIL stores native Python ints; a duplicate column keeps the last value.
    # An earlier high word must not cause a premature signed-index failure.
    x.rows[0] = [0, 0] if overwritten else [0]
    x.data[0] = [value, 1] if overwritten else [value]
    x.rows[1], x.data[1] = [1], [2]
    if overwritten:
        expected = legacy.kernel_model(model, x, "Nystroem")
        assert _state(audit_native_call(model.transform, x).assert_native()) == _state(expected)
    else:
        with pytest.raises(OverflowError):
            legacy.kernel_model(model, x, "Nystroem")
        _refused(audit_native_call(model.transform, x), OverflowError)
