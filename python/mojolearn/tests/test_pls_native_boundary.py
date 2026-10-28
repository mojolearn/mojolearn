"""Focused public PLS boundary controls; execute after the native owner is rebuilt."""
import numpy as np
import pytest
import mojolearn as ml
from mojolearn.exceptions import NotFittedError

__mojolearn_user_boundary__ = True


@pytest.mark.parametrize("name", ["PLSRegression", "PLSCanonical", "CCA"])
@pytest.mark.parametrize("target_kind", ["vector", "column", "strided", "list"])
def test_pls_target_shape_dtype_and_original_storage(name, target_kind):
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    y = np.array([1, 3, 2, 5, 4, 8, 6, 9], dtype=np.float32)
    if target_kind == "column":
        target = y[:, None]
    elif target_kind == "strided":
        backing = np.zeros(16, dtype=np.float64)
        backing[::2] = y
        target = backing[::2]
    elif target_kind == "list":
        target = y.tolist()
    else:
        target = y
    original_x, original_y = x.copy(), np.asarray(target).copy()
    model = getattr(ml, name)(n_components=1, numeric_mode="identical")
    model.fit(x, target)
    prediction = np.asarray(model.predict(x))
    assert prediction.shape == ((8, 1) if target_kind == "column" else (8,))
    assert prediction.dtype == np.float64
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(np.asarray(target), original_y)


@pytest.mark.parametrize("hook", ["array", "shape", "list"])
def test_pls_target_rejects_python_protocol_before_invocation(hook):
    calls = []
    class ArrayHook:
        def __array__(self, *args, **kwargs):
            calls.append("array")
            return np.ones(8, dtype=np.float32)
    class ShapeHook(np.ndarray):
        @property
        def shape(self):
            calls.append("shape")
            return (8,)
    class ListHook(list):
        def __getitem__(self, key):
            calls.append("item")
            return super().__getitem__(key)
        def __len__(self):
            calls.append("len")
            return super().__len__()
    target = (ArrayHook() if hook == "array" else
              np.ones(8, dtype=np.float32).view(ShapeHook) if hook == "shape" else ListHook(range(8)))
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    with pytest.raises(TypeError):
        ml.PLSRegression(n_components=1, numeric_mode="identical").fit(x, target)
    assert calls == []


def test_pls_native_target_keeps_finite_length_and_fitted_checks():
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    model = ml.PLSRegression(n_components=1, numeric_mode="identical")
    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict(x)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        model.fit(x, np.ones(7, dtype=np.float32))
    y = np.ones(8, dtype=np.float32)
    y[-1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(x, y)


@pytest.mark.parametrize("name", ["PLSRegression", "PLSCanonical", "CCA"])
def test_pls_public_lifecycle_and_output_dtypes(name):
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    y = np.array([1, 3, 2, 5, 4, 8, 6, 9], dtype=np.float32)
    model = getattr(ml, name)(n_components=1, numeric_mode="identical")
    projected = model.fit_transform(x, y)
    assert type(projected) is tuple
    assert all(type(value) is np.ndarray and value.dtype == np.float32 for value in projected)
    assert vars(model)["_mojolearn_fitted"] is True
    assert "_mojolearn_fitting" not in vars(model)
    assert vars(model)["n_features_in_"] == 3
    prediction = model.predict(x)
    assert type(prediction) is np.ndarray and prediction.dtype == np.float64
    with pytest.raises(ValueError, match="expecting 3 features"):
        model.predict(x[:, :2])
    with pytest.raises(ValueError, match="Expected 2D array"):
        model.transform(x[:, 0])
    with pytest.raises(ValueError, match="0 sample"):
        model.predict(x[:0])


def test_pls_native_lifecycle_refuses_mutable_scalar_hook():
    calls = []
    class Number:
        def __int__(self):
            calls.append("int")
            return 1
    model = ml.PLSRegression(n_components=1, numeric_mode="identical")
    vars(model)["n_components"] = Number()
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    with pytest.raises(TypeError):
        model.fit(x, np.arange(8, dtype=np.float32))
    assert calls == []


@pytest.mark.parametrize("name", ["PLSRegression", "PLSCanonical", "CCA"])
def test_pls_public_sparse_admission_matches_family_contract(name):
    sparse = pytest.importorskip("scipy.sparse")
    x = sparse.csr_matrix(np.arange(24, dtype=np.float32).reshape(8, 3))
    y = np.array([1, 3, 2, 5, 4, 8, 6, 9], dtype=np.float32)
    model = getattr(ml, name)(n_components=1, numeric_mode="identical")
    if name == "PLSRegression":
        assert model.fit(x, y) is model
        assert model.predict(x).shape == (8,)
    else:
        with pytest.raises(TypeError, match="Sparse data was passed for X"):
            model.fit(x, y)


@pytest.mark.parametrize("argument", ["X", "y"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_pls_public_complex_error_before_numerical_execution(argument, dtype):
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    y = np.array([1, 3, 2, 5, 4, 8, 6, 9], dtype=np.float32)
    if argument == "X":
        x = x.astype(dtype)
    else:
        y = y.astype(dtype)
    with pytest.raises(ValueError, match=f"Complex data not supported: PLSRegression got a complex {argument}."):
        ml.PLSRegression(n_components=1, numeric_mode="identical").fit(x, y)
