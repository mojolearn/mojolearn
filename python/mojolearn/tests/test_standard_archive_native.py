# SPDX-License-Identifier: Apache-2.0
"""StandardScaler archive owner and public scalar-type seams.

Native schema/refusal/count-word matrices live in archive_standard_suite.mojo.
Python prepares independent NPZ fixtures and inspects final API owners only.
"""
import io
import zipfile

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import Array, _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call


def _archive(count_kind):
    words = np.array([0x80000000, 0x7FC12345], dtype="<u4")
    fields = dict(
        format="mojolearn-scaler-1", estimator="StandardScaler", numeric_mode="identical",
        meta=np.array([2, 17 if count_kind == "integer" else -1, 1, 1], dtype="<i8"),
        mean=words.view("<f4"), var=words.view("<f4"), scale=words.view("<f4"),
    )
    if count_kind == "float_vector":
        fields["n_samples_seen"] = np.array([-0.0, 1.25], dtype="<f4")
    elif count_kind == "integer_vector":
        fields["n_samples_seen"] = np.array([-1, 9007199254740993], dtype="<i8")
    elif count_kind == "float_scalar":
        fields["n_samples_seen"] = np.array([1.25], dtype="<f4")
    elif count_kind == "integer_scalar":
        fields["n_samples_seen"] = np.array([9007199254740993], dtype="<i8")
    elif count_kind == "invalid_text":
        fields["n_samples_seen"] = np.array("wrong")
    # Encode the supported rank-one F-order flag explicitly: every loaded
    # statistic and count vector must still be a fresh writable C-order owner.
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as saved:
        for name, value in fields.items():
            value = np.asarray(value)
            raw = io.BytesIO()
            np.lib.format.write_array_header_1_0(raw, {
                "descr": np.lib.format.dtype_to_descr(value.dtype),
                "fortran_order": value.ndim == 1, "shape": value.shape,
            })
            raw.write(value.tobytes())
            saved.writestr(name + ".npy", raw.getvalue())
    stream.seek(0)
    return stream


@pytest.mark.parametrize("route", ["public", "explicit_host", "generic_host"])
@pytest.mark.parametrize("count_kind", [
    "integer", "float_vector", "integer_vector", "float_scalar", "integer_scalar",
])
def test_standard_archive_owns_writable_statistics_and_preserves_count_types(route, count_kind):
    source = _archive(count_kind)
    loader = {"public": ml.StandardScaler.load, "explicit_host": host.HostStandardScaler.load,
              "generic_host": ml.host_model}[route]
    model = audit_native_call(loader, source).assert_native()
    assert not source.closed
    source.close()
    assert model.copy and model.with_mean and model.with_std
    assert model.with_mean_ and model.with_std_ and model.numeric_mode is None
    for name in ("mean_", "var_", "scale_"):
        value = getattr(model, name)
        assert type(value) is Array and value.order == "C" and value.shape == (2,)
        assert np.asarray(value).flags.writeable
        np.testing.assert_array_equal(np.asarray(value).view("<u4"), [0x80000000, 0x7FC12345])
    count = model.n_samples_seen_
    if count_kind == "integer":
        assert type(count) is int and count == 17
    elif count_kind.endswith("scalar"):
        assert type(count) is float
        assert count == (1.25 if count_kind == "float_scalar" else 9007199254740992.0)
    else:
        assert type(count) is Array and count.order == "C" and count.shape == (2,)
        assert np.asarray(count).flags.writeable
        expected = [-0.0, 1.25] if count_kind == "float_vector" else [-1, 9007199254740993]
        np.testing.assert_array_equal(np.asarray(count), expected)
        count[0] = 2
    model.mean_[0] = 3.0
    assert np.asarray(model.var_).view("<u4")[0] == 0x80000000
    model.mean_ = model.scale_  # public rebinding to another admitted owner
    model.scale_[0] = 4.0
    saved = io.BytesIO()
    audit_native_call(model.save, saved).assert_native()
    assert not saved.closed
    saved.seek(0)
    with np.load(saved, allow_pickle=False) as fields:
        np.testing.assert_array_equal(fields["mean"].view("<u4"), [0x40800000, 0x7FC12345])
        np.testing.assert_array_equal(fields["scale"].view("<u4"), [0x40800000, 0x7FC12345])
        np.testing.assert_array_equal(fields["var"].view("<u4"), [0x80000000, 0x7FC12345])
        if count_kind == "integer":
            np.testing.assert_array_equal(fields["meta"], [2, 17, 1, 1])
            assert "n_samples_seen" not in fields
        elif count_kind.endswith("scalar"):
            assert fields["n_samples_seen"].dtype == np.dtype("<f4")
            np.testing.assert_array_equal(fields["n_samples_seen"], np.array([count], dtype="<f4"))
        else:
            np.testing.assert_array_equal(fields["n_samples_seen"], np.asarray(count))
    with zipfile.ZipFile(io.BytesIO(saved.getvalue())) as archive:
        expected = ["estimator.npy", "format.npy", "mean.npy", "meta.npy"]
        if count_kind != "integer":
            expected.append("n_samples_seen.npy")
        assert archive.namelist() == expected + ["numeric_mode.npy", "scale.npy", "var.npy"]


@pytest.mark.parametrize("loader", [ml.StandardScaler.load, ml.host_model])
def test_standard_archive_refusal_preserves_exception_and_releases_export(loader):
    source = _archive("invalid_text")
    audit = audit_native_call(loader, source)
    assert isinstance(audit.exception, AttributeError), audit.report()
    assert str(audit.exception) == "'str' object has no attribute 'dtype'"
    report = audit.report()
    assert not any(report[key] for key in (
        "python_frames", "forbidden_native_calls", "unresolved_native_calls",
        "wrapper_work", "unobserved_existing_threads",
    )), report
    assert not source.closed
    source.close()
