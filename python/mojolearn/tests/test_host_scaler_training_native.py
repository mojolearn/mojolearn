# SPDX-License-Identifier: Apache-2.0
"""Explicit CPU scaler training, routing, and optional-ABI failures."""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend
from mojolearn._classical_host import HostStandardScaler, HostMinMaxScaler
from mojolearn.preprocessing import _STANDARD_CONTEXT, _MINMAX_CONTEXT
from mojolearn._verify_native_runtime import audit_native_call


def _data():
    x = np.random.default_rng(243).normal(size=(41, 4)).astype(np.float32)
    x[::5, 0] = np.nan
    x[:, 2] = np.nan
    x[:, 3] = 2.5
    return x


def _bits(value):
    return np.asarray(value).tobytes()


@pytest.mark.parametrize("host,ordinary", [(HostStandardScaler, ml.StandardScaler), (HostMinMaxScaler, ml.MinMaxScaler)])
@pytest.mark.parametrize("operation", ["fit", "partial_fit", "fit_transform"])
def test_explicit_host_training_is_native_and_matches_existing_kernels(host, ordinary, operation):
    x = _data()
    m, reference = host(), ordinary(numeric_mode="identical")
    kwargs = {"sample_weight": np.linspace(.5, 1.5, len(x), dtype=np.float32)} if host is HostStandardScaler else {}
    if operation == "partial_fit":
        for part in (slice(0, 17), slice(17, None)):
            options = {"sample_weight": kwargs["sample_weight"][part]} if kwargs else {}
            assert audit_native_call(m.partial_fit, x[part], **options).assert_native() is m
            reference.partial_fit(x[part], **options)
    else:
        result = audit_native_call(getattr(m, operation), x, **kwargs).assert_native()
        expected = getattr(reference, operation)(x, **kwargs)
        if operation == "fit_transform":
            assert _bits(result) == _bits(expected)
        else:
            assert result is m
    names = ("mean_", "var_", "scale_") if host is HostStandardScaler else ("data_min_", "data_max_", "data_range_", "scale_", "min_")
    for name in names + ("n_samples_seen_",):
        assert _bits(getattr(m, name)) == _bits(getattr(reference, name)), name
    assert _bits(audit_native_call(m.transform, x).assert_native()) == _bits(reference.transform(x))


@pytest.mark.parametrize("host,context,name", [
    (HostStandardScaler, _STANDARD_CONTEXT, "standard_fit"),
    (HostMinMaxScaler, _MINMAX_CONTEXT, "minmax_fit"),
])
def test_host_training_entry_is_fixed_cpu_abi(host, context, name):
    expected = getattr(_backend.load_host_module("_mojolearn_preprocessing_host"), name)
    entries = context["scaler_host_contexts"][host]["scaler_bindings"]["identical"]
    assert entries[name] is expected
    if _backend.vendor() != "cpu":
        assert context["scaler_bindings"]["identical"][name] is not expected


@pytest.mark.parametrize("host,context,name", [
    (HostStandardScaler, _STANDARD_CONTEXT, "standard_fit"),
    (HostMinMaxScaler, _MINMAX_CONTEXT, "minmax_fit"),
])
@pytest.mark.parametrize("operation", ["fit", "partial_fit", "fit_transform"])
def test_missing_training_abi_refuses_natively_and_keeps_inference(host, context, name, operation, monkeypatch):
    x = _data()
    retained = host().fit(x)
    expected = _bits(retained.transform(x))
    entries = context["scaler_host_contexts"][host]["scaler_bindings"]["identical"]
    monkeypatch.delitem(entries, name)
    monkeypatch.setitem(entries, "scaler_training_error", "retained missing-training-ABI probe")
    audit = audit_native_call(getattr(host(), operation), x)
    assert isinstance(audit.exception, ImportError)
    assert "_mojolearn_preprocessing_host" in str(audit.exception)
    assert "missing-training-ABI probe" in str(audit.exception)
    assert not audit.frames
    assert not audit.wrapper_work
    assert _bits(audit_native_call(retained.transform, x).assert_native()) == expected
