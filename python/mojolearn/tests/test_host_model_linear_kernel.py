# SPDX-License-Identifier: Apache-2.0
"""Saved-model CPU inference for the scalers, lasso and elasticnet and the
kernel methods (lane/inference-linear-svm, 2026-09-15).

The refusal tests need no binding. The round-trip tests fit through the
bindings this process routes (a GPU set, or the reference host set under
`reference_training()` on a CPU-only install), save, load through
`mojolearn.host_model` and require the host answer to be the same bytes as
the fitted model's; they skip when the estimators host binding is not built.
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


def _rows(n=300, d=6, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X @ np.arange(1, d + 1, dtype=np.float32) + 0.5).astype(np.float32)
    return X, y


def test_formats_are_host_model_formats():
    for fmt in ("mojolearn-scaler-1", "mojolearn-cd-1", "mojolearn-kernel-ridge-1",
                "mojolearn-nystroem-1", "mojolearn-rbf-sampler-1"):
        assert fmt in _classical_host.CLASSICAL_FORMATS


def _host_built():
    return os.path.exists(_backend.host_module_path("_mojolearn_estimators_host"))


def _fit(make, X, y):
    with reference_training():
        return make().fit(X, y)


CASES = [
    ("StandardScaler", lambda: mojolearn.StandardScaler(), lambda e, X: (e.transform(X), e.inverse_transform(e.transform(X)))),
    ("StandardScaler", lambda: mojolearn.StandardScaler(with_mean=False), lambda e, X: (e.transform(X),)),
    ("StandardScaler", lambda: mojolearn.StandardScaler(with_std=False), lambda e, X: (e.transform(X),)),
    ("MinMaxScaler", lambda: mojolearn.MinMaxScaler(feature_range=(-1.0, 1.0), clip=True),
     lambda e, X: (e.transform(X * 2), e.inverse_transform(e.transform(X)))),
    ("Lasso", lambda: mojolearn.Lasso(alpha=0.01, max_iter=50), lambda e, X: (e.predict(X),)),
    ("ElasticNet", lambda: mojolearn.ElasticNet(alpha=0.01, l1_ratio=0.0, fit_intercept=False, max_iter=50),
     lambda e, X: (e.predict(X),)),
    ("KernelRidge", lambda: mojolearn.KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5), lambda e, X: (e.predict(X[:32]),)),
    ("Nystroem", lambda: mojolearn.Nystroem(kernel="rbf", gamma=0.5, n_components=16, random_state=7),
     lambda e, X: (e.transform(X[:32]),)),
    ("RBFSampler", lambda: mojolearn.RBFSampler(gamma=0.5, n_components=24, random_state=1),
     lambda e, X: (e.transform(X),)),
]


@pytest.mark.parametrize("name,make,probe", CASES)
def test_host_model_reproduces_the_fitted_model(name, make, probe):
    if not _host_built():
        pytest.skip("mojolearn/host/_mojolearn_estimators_host.so is not built")
    X, y = _rows()
    X = X[:128] if name in ("KernelRidge", "Nystroem") else X
    y = y[:X.shape[0]]
    est = _fit(make, X, y)
    want = [np.asarray(a).tobytes() for a in probe(est, X)]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.npz")
        est.save(path)
        host = mojolearn.host_model(path)
        assert type(host).__name__ == "Host" + name and host.estimator == name
        assert host.vendor_used() == "cpu"
        got = [np.asarray(a).tobytes() for a in probe(host, X)]
        assert got == want
        # A host model saves under its own class name, as every host
        # subclass does; every other member must be the same bytes.
        again = os.path.join(tmp, "again.npz")
        host.save(again)
        first = _serialize.read_npz(path, _classical_host.CLASSICAL_FORMATS)
        second = _serialize.read_npz(again, _classical_host.CLASSICAL_FORMATS)
        assert sorted(first) == sorted(second)
        assert _serialize.scalar_str(first, "estimator") == name
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
