"""Focused public PLS boundary controls; execute after the native owner is rebuilt."""
import numpy as np
import pytest
import mojolearn as ml


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
    assert prediction.dtype == np.float32
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
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(x)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        model.fit(x, np.ones(7, dtype=np.float32))
    y = np.ones(8, dtype=np.float32)
    y[-1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(x, y)
