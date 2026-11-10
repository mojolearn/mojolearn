# SPDX-License-Identifier: Apache-2.0
"""Generic MinMax archive routing, file ownership and public buffer seams.

Numerical schema/refusal matrices live in checks/native_tests/archive_minmax.mojo.
NumPy prepares independent archive bytes and inspects published buffer owners.
"""
import io

import numpy as np
import pytest

import mojolearn
from mojolearn import Array, _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call


def _saved_bytes(meta=None):
    stream = io.BytesIO()
    words = np.array([0x80000000, 0x7FC12345], dtype="<u4")
    values = words.view("<f4")
    np.savez(
        stream, format="mojolearn-scaler-1", estimator="MinMaxScaler",
        numeric_mode="identical", meta=np.array([2, 7, 1] if meta is None else meta, dtype="<i8"),
        feature_range=np.array([-1, 3], dtype="<f8"),
        data_min=values, data_max=values, data_range=values, scale=values,
        min=values,
    )
    return stream.getvalue()


@pytest.mark.parametrize("route", ["host_model", "classical_host_model", "requested_class"])
@pytest.mark.parametrize("source_kind", ["path", "caller_stream"])
def test_generic_minmax_preserves_writable_owners_after_input_release(tmp_path, route, source_kind):
    if source_kind == "path":
        source = tmp_path / "minmax.npz"
        source.write_bytes(_saved_bytes())
    else:
        source = io.BytesIO(_saved_bytes())
    if route == "host_model":
        audit = audit_native_call(mojolearn.host_model, source)
    elif route == "classical_host_model":
        audit = audit_native_call(host.host_model, source)
    else:
        # The explicit generic class door remains a supported connected ABI;
        # HostMinMaxScaler.load itself now uses the dedicated typed scaler door.
        audit = audit_native_call(host._native_classical, host.HostMinMaxScaler,
                                  "host_archive_load", (source,), host._CLASSICAL)
    audit.assert_native()
    model = audit.result
    assert type(model) is host.HostMinMaxScaler
    assert model.feature_range == model.feature_range_ == (-1.0, 3.0)
    assert model.clip is model.clip_ is True
    assert model.n_features_in_ == 2 and model.n_samples_seen_ == 7
    if route != "requested_class":
        assert model.estimator == "MinMaxScaler"
    if source_kind == "path":
        source.unlink()
    else:
        assert not source.closed
        source.close()  # no outstanding borrowed export may escape PACK
    names = ("data_min_", "data_max_", "data_range_", "scale_", "min_")
    for name in names:
        value = getattr(model, name)
        assert type(value) is Array
        assert value.shape == (2,) and value.order == "C"
        view = np.asarray(value)
        assert view.flags.writeable
        np.testing.assert_array_equal(view.view("<u4"), [0x80000000, 0x7FC12345])
    model.scale_[0] = 2.0
    assert float(model.data_min_[0]) == 0.0
    assert np.signbit(float(model.data_min_[0]))
    saved = io.BytesIO()
    audit_native_call(model.save, saved).assert_native()
    assert not saved.closed
    saved.seek(0)
    with np.load(saved, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["scale"].view("<u4"), [0x40000000, 0x7FC12345])


def test_generic_minmax_refusal_releases_caller_file_export():
    source = io.BytesIO(_saved_bytes(meta=[2]))
    audit = audit_native_call(mojolearn.host_model, source)
    assert isinstance(audit.exception, ValueError), audit.report()
    assert str(audit.exception) == "model meta must hold 3 fields"
    report = audit.report()
    assert not any(report[key] for key in (
        "python_frames", "forbidden_native_calls", "unresolved_native_calls",
        "wrapper_work", "unobserved_existing_threads",
    )), report
    assert not source.closed
    source.close()  # exceptional schema exit must release the admitted export
