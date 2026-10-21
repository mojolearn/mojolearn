# SPDX-License-Identifier: Apache-2.0
"""The explicit Host* interface keeps CPU inference and coefficient hashing."""
import hashlib
import io

import numpy as np
import pytest

from mojolearn import LinearRegression, Ridge
from mojolearn._classical_host import HostLinearRegression, HostRidge
from mojolearn._native_classical_bridge import connect_host
from mojolearn._verify_native_runtime import audit_native_call

ENTRY, CONTEXT = connect_host(HostRidge, HostLinearRegression)


def native_load(cls, path):
    return ENTRY(cls, "load", (path,), CONTEXT)


def native_save(model, path):
    return ENTRY(model, "save", (path,), CONTEXT)


def native_predict(model, x):
    return ENTRY(model, "predict_public", (x,), CONTEXT)


def native_score(model, x, y):
    return ENTRY(model, "score_public", (x, y), CONTEXT)


def native_hash(model):
    return ENTRY(model, "model_sha256", (), CONTEXT)


def native_vendor(model):
    return ENTRY(model, "vendor_used", (), CONTEXT)


def native_mode(model):
    return ENTRY(model, "numeric_mode_used", (), CONTEXT)


def clean(audit, error=None):
    report = audit.report()
    assert not report["python_frames"]
    assert not report["forbidden_native_calls"]
    assert not report["wrapper_work"]
    assert not report["unobserved_existing_threads"]
    if error is None:
        return audit.assert_native()
    assert isinstance(audit.exception, error)


def saved(cls, mode="identical"):
    model = cls(numeric_mode=mode)
    model.coef_ = np.array([0.75, -0.5, 0.125], dtype="f4")
    model.intercept_ = 0.25
    model.n_features_in_ = 3
    buffer = io.BytesIO()
    model.save(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("cls,host_cls", [(Ridge, HostRidge), (LinearRegression, HostLinearRegression)])
@pytest.mark.parametrize("source", ["ordinary", "host"])
def test_native_host_load_cpu_predict_score_hash_and_save_match_existing_contract(cls, host_cls, source, monkeypatch):
    archive = saved(cls if source == "ordinary" else host_cls)
    old = host_cls.load(io.BytesIO(archive))
    x = np.random.default_rng(208).normal(size=(13, 3)).astype("f4")
    y = np.linspace(-1, 1, len(x), dtype="f4")
    expected_prediction = np.asarray(old.predict(x), dtype="f4").astype("f8")
    expected_score = np.float64(old.score(x, y))
    expected_hash = old.model_sha256()
    expected_saved = io.BytesIO()
    old.save(expected_saved)
    assert expected_hash == hashlib.sha256(np.asarray(old.coef_, dtype="f4").tobytes()).hexdigest()
    from mojolearn import _backend, _serialize
    from mojolearn._classical_host import _HostBound

    def poison(*args, **kwargs):
        raise AssertionError("Python host model preparation or binding selection reached")

    with monkeypatch.context() as patch:
        patch.setattr(_HostBound, "_bind", poison)
        patch.setattr(_backend, "binding", poison)
        patch.setattr(_serialize, "read_npz", poison)
        patch.setattr(_serialize, "write_npz", poison)
        native = clean(audit_native_call(native_load, host_cls, io.BytesIO(archive)))
        result = clean(audit_native_call(native_predict, native, x))
        score = clean(audit_native_call(native_score, native, x, y))
        digest = clean(audit_native_call(native_hash, native))
        assert clean(audit_native_call(native_vendor, native)) == "cpu"
        assert clean(audit_native_call(native_mode, native)) == "identical"
        output = io.BytesIO()
        clean(audit_native_call(native_save, native, output))
    assert type(native) is host_cls
    assert result.dtype == np.dtype("f8")
    np.testing.assert_array_equal(result.view("u8"), expected_prediction.view("u8"))
    np.testing.assert_allclose(result, x.astype("f8") @ np.asarray(native.coef_, dtype="f8") + 0.25, rtol=3e-6, atol=3e-7)
    assert np.float64(score).view("u8") == expected_score.view("u8")
    assert digest == expected_hash
    assert output.getvalue() == expected_saved.getvalue()


@pytest.mark.parametrize("host_cls", [HostRidge, HostLinearRegression])
def test_native_host_hash_preserves_none_empty_and_noncontiguous_storage(host_cls):
    model = host_cls()
    clean(audit_native_call(native_hash, model), AttributeError)
    model.coef_ = None
    assert clean(audit_native_call(native_hash, model)) == hashlib.sha256(b"").hexdigest()
    for value in [np.empty(0, dtype="f4"), np.arange(12, dtype="f4")[::-2]]:
        model.coef_ = value
        assert clean(audit_native_call(native_hash, model)) == hashlib.sha256(value.tobytes()).hexdigest()


@pytest.mark.parametrize("cls,host_cls", [(Ridge, HostRidge), (LinearRegression, HostLinearRegression)])
def test_native_host_preserves_saved_fast_mode_refusal_at_inference(cls, host_cls):
    archive = saved(cls, "fast")
    native = clean(audit_native_call(native_load, host_cls, io.BytesIO(archive)))
    old = host_cls.load(io.BytesIO(archive))
    x = np.ones((2, 3), dtype="f4")
    with pytest.raises(ValueError):
        old.predict(x)
    clean(audit_native_call(native_predict, native, x), ValueError)


@pytest.mark.parametrize("host_cls", [HostRidge, HostLinearRegression])
def test_native_host_not_fitted_message_uses_the_supported_subclass_name(host_cls):
    model = host_cls()
    x = np.ones((2, 3), dtype="f4")
    with pytest.raises(ValueError) as old:
        model.predict(x)
    audit = audit_native_call(native_predict, model, x)
    clean(audit, type(old.value))
    assert str(audit.exception) == str(old.value)
