# SPDX-License-Identifier: Apache-2.0
"""Ordinary pandas compatibility without invoking its Python conversion API."""
import os

import numpy as np
import pytest

try:
    import pandas as pd
except ImportError:
    if os.environ.get("MOJOLEARN_REQUIRE_PANDAS") == "1":
        raise
    pytest.skip("pandas is unavailable", allow_module_level=True)

from mojolearn import LinearRegression, Ridge
from mojolearn._native_classical_bridge import connect
from mojolearn._verify_native_runtime import audit_native_call

ENTRY, CONTEXT = connect(Ridge, LinearRegression)


def ridge_fit(model, x, y, weights=None):
    return ENTRY(model, "ridge_fit", (x, y, weights), CONTEXT)


def linear_fit(model, x, y, weights=None):
    return ENTRY(model, "ols_fit", (x, y, weights), CONTEXT)


def predict(model, x):
    return ENTRY(model, "predict", (x,), CONTEXT)


def score(model, x, y):
    return ENTRY(model, "score", (x, y), CONTEXT)


def clean(audit, error=None):
    report = audit.report()
    assert not report["python_frames"]
    assert not report["forbidden_native_calls"]
    assert not report["wrapper_work"]
    assert not report["unobserved_existing_threads"]
    if error is None:
        return audit.assert_native()
    assert isinstance(audit.exception, error)


def bits(value):
    return np.asarray(value, dtype="f4").view("u4")


def fixture(case):
    rng = np.random.default_rng(909)
    base = rng.normal(size=(19, 3)).astype("f4")
    y = (base @ np.array([0.25, -0.5, 0.75], dtype="f4") + 0.5).astype("f4")
    weights = None
    if case == "float32":
        x = pd.DataFrame(base, columns=["a", "b", "c"])
    elif case == "mixed":
        x = pd.DataFrame({"a": np.arange(19, dtype="i8"), "b": base[:, 1].astype("f8"), "c": base[:, 2]})
    elif case == "fragmented_reordered":
        x = pd.DataFrame({"a": base[:, 0]})
        x["c"] = base[:, 2]
        x["b"] = base[:, 1]
        x = x[["c", "a", "b"]].iloc[::-2]
        y = y[::-2]
    elif case == "nullable_integer":
        x = pd.DataFrame({name: pd.array(np.arange(19) ** (i + 1), dtype="Int64") for i, name in enumerate("abc")})
    elif case == "nullable_float":
        x = pd.DataFrame({name: pd.array(base[:, i], dtype="Float32" if i != 1 else "Float64") for i, name in enumerate("abc")})
    elif case == "boolean_mixture":
        x = pd.DataFrame({"a": base[:, 0] > 0, "b": base[:, 1].astype("f8"), "c": base[:, 2]})
    elif case == "nullable_boolean":
        x = pd.DataFrame({name: pd.array(base[:, i] > 0, dtype="boolean") for i, name in enumerate("abc")})
    elif case == "object_native_numbers":
        x = pd.DataFrame(base.astype(object), columns=list("abc"))
        x.iloc[0, 0] = np.float32(0.5)
        x.iloc[1, 1] = np.int64(2)
    elif case == "series_target_weight":
        x = pd.DataFrame(base)
        y = pd.Series(pd.array(y, dtype="Float32"))
        weights = pd.Series(pd.array(np.linspace(0.5, 1.5, len(base)), dtype="Float64"))
    else:
        raise AssertionError(case)
    return x, y, weights


@pytest.mark.parametrize("cls,operation", [(Ridge, ridge_fit), (LinearRegression, linear_fit)])
@pytest.mark.parametrize("case", ["float32", "mixed", "fragmented_reordered", "nullable_integer", "nullable_float", "boolean_mixture", "nullable_boolean", "object_native_numbers", "series_target_weight"])
def test_native_pandas_whole_fit_predict_score_preserves_old_public_driver(cls, operation, case):
    x, y, weights = fixture(case)
    # Old public methods and this independently materialized typed matrix are
    # both retained. Test preparation is intentionally outside the audit.
    canonical = np.asarray(x).astype("f4")
    target = np.asarray(y).astype("f4")
    w = None if weights is None else np.asarray(weights).astype("f4")
    old = cls().fit(x, y, sample_weight=weights)
    native = cls()
    clean(audit_native_call(operation, native, x, y, weights))
    np.testing.assert_array_equal(bits(native.coef_), bits(old.coef_))
    assert np.float64(native.intercept_).view("u8") == np.float64(old.intercept_).view("u8")
    out = clean(audit_native_call(predict, native, x))
    np.testing.assert_array_equal(bits(out), bits(old.predict(x)))
    np.testing.assert_allclose(out, canonical.astype("f8") @ np.asarray(native.coef_, dtype="f8") + native.intercept_, rtol=3e-5, atol=3e-5)
    assert np.float64(clean(audit_native_call(score, native, x, y))).view("u8") == np.float64(old.score(x, y)).view("u8")
    # The same buffers sent through the existing array API must yield the
    # same fitted bytes, including weighted target/Series conversion.
    independent = cls().fit(canonical, target, sample_weight=w)
    np.testing.assert_array_equal(bits(native.coef_), bits(independent.coef_))


def test_native_pandas_does_not_enter_standard_python_conversion_or_metadata(monkeypatch):
    x, y, weights = fixture("nullable_float")
    old = Ridge().fit(x, y)
    expected = np.asarray(old.predict(x), dtype="f4")
    from pandas.core.internals.blocks import ExtensionBlock
    from pandas.core.internals.managers import BlockManager
    from pandas.core.arrays.floating import FloatingArray

    calls = []

    def poison(*args, **kwargs):
        calls.append("python")
        raise AssertionError("pandas Python callback reached")

    native = Ridge()
    with monkeypatch.context() as patch:
        for name in ("__array__", "to_numpy"):
            patch.setattr(pd.DataFrame, name, poison)
            patch.setattr(FloatingArray, name, poison)
        for name in ("shape", "values", "dtypes"):
            patch.setattr(pd.DataFrame, name, property(poison))
        patch.setattr(BlockManager, "blocks", property(poison))
        patch.setattr(ExtensionBlock, "values", property(poison))
        clean(audit_native_call(ridge_fit, native, x, y))
        out = clean(audit_native_call(predict, native, x))
    assert calls == []
    np.testing.assert_array_equal(bits(out), bits(expected))


@pytest.mark.parametrize("case", ["integer", "mixed_float", "mixed_unsigned", "object"])
def test_native_pandas_preserves_promotion_before_float32_at_double_rounding_boundaries(case):
    # An integer one unit above a float32 midpoint rounds upward directly,
    # but its intervening float64 representation lands on the even midpoint.
    # This distinguishes pandas common-dtype promotion from a per-block cast.
    value = 2**62 + 2**38 + 1
    first = np.array([value, -value, value - 2, -value + 2], dtype="i8")
    if case == "integer":
        x = pd.DataFrame({"a": first})
    elif case == "mixed_float":
        x = pd.DataFrame({"a": first, "b": np.zeros(4, "f4")})
    elif case == "mixed_unsigned":
        x = pd.DataFrame({"a": first, "b": np.zeros(4, "u8")})
    else:
        x = pd.DataFrame({"a": first.astype(object)})
    canonical = np.asarray(x).astype("f4")
    model = Ridge(fit_intercept=False)
    model.coef_ = np.zeros(canonical.shape[1], dtype="f4")
    model.coef_[0] = 1
    model.intercept_ = 0.0
    model.n_features_in_ = canonical.shape[1]
    expected = np.asarray(model.predict(x), dtype="f4")
    actual = clean(audit_native_call(predict, model, x))
    np.testing.assert_array_equal(bits(actual), bits(expected))
    np.testing.assert_array_equal(bits(actual), canonical[:, 0].view("u4"))
    if case == "integer":
        assert canonical[0, 0] > np.float32(np.float64(value))
    else:
        assert canonical[0, 0] == np.float32(np.float64(value))


@pytest.mark.parametrize("kind", ["subclass", "object_hook", "array_subclass", "nullable_mask", "manager", "field_key"])
def test_native_pandas_refuses_custom_hooks_before_reading_them(kind):
    calls = []

    class Poison:
        def __float__(self):
            calls.append("float")
            raise AssertionError("Python numeric hook reached")

        def __getattribute__(self, name):
            calls.append("attribute")
            raise AssertionError("Python attribute hook reached")

    class Frame(pd.DataFrame):
        def __array__(self, *args, **kwargs):
            calls.append("array")
            raise AssertionError("Python frame hook reached")

    class Array(np.ndarray):
        @property
        def dtype(self):
            calls.append("dtype")
            raise AssertionError("Python array hook reached")

    class Key(str):
        def __eq__(self, other):
            calls.append("eq")
            raise AssertionError("Python metadata key reached")

        __hash__ = str.__hash__

    x, y, _ = fixture("nullable_float" if kind == "nullable_mask" else "float32")
    if kind == "subclass":
        x = Frame(x)
    elif kind == "object_hook":
        x = x.astype(object)
        x._mgr.blocks[0].values[0, 0] = Poison()
    elif kind == "array_subclass":
        x._mgr.blocks[0].values = x._mgr.blocks[0].values.view(Array)
    elif kind == "nullable_mask":
        x._mgr.blocks[0].values._mask = x._mgr.blocks[0].values._mask.view(Array)
    elif kind == "manager":
        object.__getattribute__(x, "__dict__")["_mgr"] = Poison()
    else:
        object.__getattribute__(x, "__dict__")[Key("extra")] = 1
    clean(audit_native_call(ridge_fit, Ridge(), x, y), TypeError)
    assert calls == []


@pytest.mark.parametrize("case", ["missing", "strings", "complex", "empty_rows", "empty_columns"])
def test_native_pandas_preserves_ordinary_refusals(case):
    x, y, _ = fixture("float32")
    if case == "missing":
        x = x.astype("Float32")
        x.iloc[0, 0] = pd.NA
    elif case == "strings":
        x = x.astype(str)
    elif case == "complex":
        x = x.astype("complex128")
    elif case == "empty_rows":
        x, y = x.iloc[:0], y[:0]
    else:
        x = x.iloc[:, :0]
    with pytest.raises((TypeError, ValueError)) as old:
        Ridge().fit(x, y)
    clean(audit_native_call(ridge_fit, Ridge(), x, y), type(old.value))
