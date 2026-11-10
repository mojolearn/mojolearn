# SPDX-License-Identifier: Apache-2.0
"""Cholesky archive file, buffer-rebinding and protocol-admission seams.

Native schema/refusal/raw-word coverage lives in archive_cholesky_suite.mojo.
NumPy prepares independent saved bytes and inspects public writable owners.
"""
import io
import zipfile

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import Array, _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call


def _archive(order="C", mode="identical"):
    words = np.array([[0x80000000, 0x7FC12345], [0x3F800000, 0x40000000]], dtype="<u4")
    factor = np.array(words.view("<f4"), order=order)
    scalars = np.array([0x7FF8000000000042, 0x8000000000000000], dtype="<u8").view("<f8")
    result = io.BytesIO()
    np.savez(result, format="mojolearn-cholesky-1", estimator="Cholesky", numeric_mode=mode,
             meta=np.array([2, -7, 32], dtype="<i8"), scalars=scalars,
             jitter_param=np.array([-1.0], dtype="<f8"), L=factor)
    result.seek(0)
    return result, words


@pytest.mark.parametrize("route", ["public", "explicit_host", "generic_host"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("replacement_layout", ["fortran", "negative_stride"])
def test_cholesky_archive_owns_writable_c_factor_and_saves_rebound_buffers(route, order, replacement_layout):
    source, expected = _archive(order)
    loader = {"public": ml.Cholesky.load, "explicit_host": host.HostCholesky.load,
              "generic_host": ml.host_model}[route]
    model = audit_native_call(loader, source).assert_native()
    assert not source.closed
    source.close()
    assert type(model.L_) is Array and model.L_.order == "C" and model.L_.shape == (2, 2)
    assert np.asarray(model.L_).flags.writeable
    np.testing.assert_array_equal(np.asarray(model.L_).view("<u4"), expected)
    assert model.n_ == 2 and model.info_ == -7 and model.nb_ == 32 and model.jitter is None
    np.testing.assert_array_equal(np.array([vars(model)["_logdet"], model.jitter_]).view("<u8"),
                                  [0x7FF8000000000042, 0x8000000000000000])
    np.asarray(model.L_)[0, 0] = 4.0
    assert np.asarray(model.L_).view("<u4")[0, 1] == 0x7FC12345
    if replacement_layout == "fortran":
        replacement = np.array([[4.0, 1.0], [2.0, 8.0]], dtype="<f4", order="F")
    else:
        replacement = np.arange(16, dtype="<f4").reshape(4, 4)[::-2, ::2]
    replacement.flags.writeable = False
    expected_bytes = replacement.tobytes(order="C")
    model.L_ = replacement
    # Existing native scalar admissions permit float-to-int truncation and
    # primitive NumPy scalars. The saved mode is normalized natively.
    model.n_ = 2.9
    model.info_ = -7.9
    model.nb_ = np.int32(16)
    model.numeric_mode = "\u2003IDENTICAL\t"
    saved = io.BytesIO()
    audit_native_call(model.save, saved).assert_native()
    assert not saved.closed and not replacement.flags.writeable
    assert replacement.tobytes(order="C") == expected_bytes
    saved.seek(0)
    with np.load(saved, allow_pickle=False) as archive:
        assert archive["L"].tobytes(order="C") == expected_bytes
        assert archive["L"].shape == (2, 2) and archive["L"].dtype == np.dtype("<f4")
        np.testing.assert_array_equal(archive["meta"], [2, -7, 16])
        np.testing.assert_array_equal(archive["scalars"].view("<u8"), [0x7FF8000000000042, 0x8000000000000000])
        assert archive["numeric_mode"].item() == "identical"
        assert archive["estimator"].item() == "Cholesky"
    with zipfile.ZipFile(io.BytesIO(saved.getvalue())) as archive:
        assert archive.namelist() == ["L.npy", "estimator.npy", "format.npy", "jitter_param.npy",
                                     "meta.npy", "numeric_mode.npy", "scalars.npy"]


def test_cholesky_save_refuses_scalar_conversion_hooks_before_file_output():
    source, _ = _archive()
    model = audit_native_call(ml.Cholesky.load, source).assert_native()
    source.close()
    calls = []

    class Hook:
        def __float__(self):
            calls.append("float")
            return 1.0

    model._logdet = Hook()
    destination = io.BytesIO(b"sentinel")
    audit = audit_native_call(model.save, destination)
    assert isinstance(audit.exception, TypeError), audit.report()
    assert "without Python conversion hooks" in str(audit.exception)
    assert calls == [] and destination.getvalue() == b"sentinel" and destination.tell() == 0
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "unresolved_native_calls",
                                          "wrapper_work", "unobserved_existing_threads")), report


def test_cholesky_bad_saved_mode_preserves_quoted_error_and_releases_input():
    mode = "'bad\n"
    source, _ = _archive(mode=mode)
    audit = audit_native_call(ml.Cholesky.load, source)
    assert isinstance(audit.exception, ValueError), audit.report()
    assert str(audit.exception) == "mojolearn: invalid saved numeric_mode " + repr(mode)
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "unresolved_native_calls",
                                          "wrapper_work", "unobserved_existing_threads")), report
    assert not source.closed
    source.close()
