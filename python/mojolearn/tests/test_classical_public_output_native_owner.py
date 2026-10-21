# SPDX-License-Identifier: Apache-2.0
"""Private public-output operation preserves the existing float64 interface."""
import numpy as np
import pytest
from mojolearn import LinearRegression, Ridge
from mojolearn._native_classical_bridge import connect
from mojolearn._verify_native_runtime import audit_native_call

ENTRY, CONTEXT = connect(Ridge, LinearRegression)

# The incumbent wrapper distinguishes callers by their module name. This
# independent oracle always represents an external user, even when pytest
# collects this file as mojolearn.tests rather than an isolated test module.
_USER_ORACLE = {"__name__": "classical_public_output_oracle"}
exec("def predict(model, x):\n    return model.predict(x)\n", _USER_ORACLE)
legacy_public_predict = _USER_ORACLE["predict"]


def native_predict_public(model, x):
    return ENTRY(model, "predict_public", (x,), CONTEXT)


def native_predict_internal(model, x):
    return ENTRY(model, "predict", (x,), CONTEXT)


def native_score(model, x, y):
    return ENTRY(model, "score", (x, y), CONTEXT)


def clean(audit):
    report = audit.report()
    assert not report["python_frames"]
    assert not report["forbidden_native_calls"]
    assert not report["wrapper_work"]
    assert not report["unobserved_existing_threads"]
    return audit.assert_native()


@pytest.mark.parametrize("cls", [LinearRegression, Ridge])
@pytest.mark.parametrize("layout", ["float32", "float64", "strided"])
def test_native_public_prediction_widens_after_kernel_and_preserves_score_bits(cls, layout, monkeypatch):
    rng = np.random.default_rng(872)
    x = rng.normal(size=(17, 3)).astype("f4")
    y = (x @ np.array([0.75, -1.5, 0.125], dtype="f4") + np.float32(0.25)).astype("f4")
    model = cls().fit(x, y)
    query = x.astype("f8") if layout == "float64" else x[::-1] if layout == "strided" else x
    expected = legacy_public_predict(model, query)
    expected_score = model.score(x, y)
    assert expected.dtype == np.dtype("f8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Python output conversion reached")

    from mojolearn import _mode, _buffer
    with monkeypatch.context() as patch:
        patch.setattr(_mode, "_public_result", forbidden)
        patch.setattr(_mode, "_ndarray", forbidden)
        patch.setattr(_buffer, "as_f32_c", forbidden)
        patch.setattr(np, "asarray", forbidden)
        actual = clean(audit_native_call(native_predict_public, model, query))
        internal = clean(audit_native_call(native_predict_internal, model, query))
        score = clean(audit_native_call(native_score, model, x, y))
    assert actual.dtype == np.dtype("f8")
    assert internal.dtype == np.dtype("f4")
    np.testing.assert_array_equal(actual.view("u8"), expected.view("u8"))
    np.testing.assert_array_equal(actual.view("u8"), internal.astype("f8").view("u8"))
    assert np.float64(score).view("u8") == np.float64(expected_score).view("u8")
