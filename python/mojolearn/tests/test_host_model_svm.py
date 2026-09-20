# SPDX-License-Identifier: Apache-2.0
"""Saved-model CPU inference for SVC's linear and polynomial kernels and for
SVR, rbf and linear (lane/inference-svm, 2026-09-15).

The refusal and format tests need no binding. The round-trip tests fit
through the bindings this process routes (a GPU set, or the reference host
set under `reference_training()` on a CPU-only install), save, load through
`mojolearn.host_model` and require the host answer to be the same bytes as
the fitted model's; they skip when the svm host binding is not built.
Cross-vendor identity is tools/classical_host_gate.py's and
tools/identity_break.py's, not this file's.
"""
import os
import tempfile

import numpy as np
import pytest

import mojolearn
from mojolearn import _backend, _classical_host, _serialize
from mojolearn._cpu_reference import reference_training


def _rows(n=256, d=6, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    yr = (X @ np.arange(1, d + 1, dtype=np.float32) + 0.5).astype(np.float32)
    yc = (yr > np.median(yr)).astype(np.int32)
    return X, yc, yr


def test_svr_format_is_a_host_model_format():
    assert "mojolearn-svr-1" in _classical_host.CLASSICAL_FORMATS
    assert _classical_host._FORMATS["mojolearn-svr-1"] == {"SVR": _classical_host.HostSVR}


def test_unfitted_svr_refuses_save():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError, match="not fitted"):
            mojolearn.SVR().save(os.path.join(tmp, "m.npz"))


def _host_built():
    return os.path.exists(_backend.host_module_path("_mojolearn_svm_host"))


CASES = [
    ("SVC", lambda: mojolearn.SVC(C=1.0, kernel="linear", max_iter=100), "c",
     lambda e, X: (e.decision_function(X), e.predict(X))),
    ("SVC", lambda: mojolearn.SVC(C=1.0, kernel="poly", degree=3, gamma=0.1, coef0=1.0, max_iter=100), "c",
     lambda e, X: (e.decision_function(X), e.predict(X))),
    ("SVR", lambda: mojolearn.SVR(C=1.0, kernel="rbf", epsilon=0.1, max_iter=100), "r",
     lambda e, X: (e.predict(X),)),
    ("SVR", lambda: mojolearn.SVR(C=1.0, kernel="linear", epsilon=0.1, max_iter=100), "r",
     lambda e, X: (e.predict(X),)),
]


@pytest.mark.parametrize("name,make,target,probe", CASES)
def test_host_model_reproduces_the_fitted_model(name, make, target, probe):
    if not _host_built():
        pytest.skip("mojolearn/host/_mojolearn_svm_host.so is not built")
    X, yc, yr = _rows()
    with reference_training():
        est = make().fit(X[:192], yc[:192] if target == "c" else yr[:192])
    Xh = X[192:]
    want = [np.asarray(a).tobytes() for a in probe(est, Xh)]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.npz")
        est.save(path)
        host = mojolearn.host_model(path)
        assert type(host).__name__ == "Host" + name and host.estimator == name
        assert host.vendor_used() == "cpu"
        assert [np.asarray(a).tobytes() for a in probe(host, Xh)] == want
        # A host model saves under its own class name; every other member
        # must be the same bytes.
        again = os.path.join(tmp, "again.npz")
        host.save(again)
        first = _serialize.read_npz(path, _classical_host.CLASSICAL_FORMATS)
        second = _serialize.read_npz(again, _classical_host.CLASSICAL_FORMATS)
        assert sorted(first) == sorted(second)
        assert _serialize.scalar_str(second, "estimator") == "Host" + name
        for member in sorted(first):
            if member == "estimator":
                continue
            a, b = first[member], second[member]
            if isinstance(a, (str, bytes, list)):
                assert a == b, member
            else:
                assert (a.dtype, tuple(a.shape), a.tobytes()) == (b.dtype, tuple(b.shape), b.tobytes()), member


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
