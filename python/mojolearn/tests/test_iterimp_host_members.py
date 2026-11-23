# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IterativeImputer with an explicit Host* estimator (lane mf-iterimp3b): the
host classes carry their own member marks, so their steps are fitted through
the native member on the CPU image their own methods bind. On the CPU route
the answer is the public class's with device="cpu", bit for bit. A host class
whose member cannot fit (an inference-only host image) is refused in the
words it had before its mark."""
import numpy as np
import mojolearn as ml
from mojolearn import _classical_host as host


def _data(seed=0, n=400, d=5):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 2))
    X = (Z @ rng.standard_normal((2, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    X[rng.random((n, d)) < 0.15] = np.nan
    return X


def _labels(seed=0, n=400, d=4):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    X = np.stack([labels + (j % 2) for j in range(d)], axis=1).astype(np.float32)
    rows = rng.random(n) < 0.6
    cols = rng.integers(0, d, n)
    X[rows, cols[rows]] = np.nan
    return X


def _same(host_estimator, public_estimator, X, Xh, **kw):
    a = ml.IterativeImputer(estimator=host_estimator, **kw)
    b = ml.IterativeImputer(estimator=public_estimator, **kw)
    xa, xb = np.asarray(a.fit_transform(X)), np.asarray(b.fit_transform(X))
    assert np.array_equal(xa, xb), (type(host_estimator).__name__, np.argwhere(xa != xb)[:4])
    assert a.n_iter_ == b.n_iter_
    assert len(a.imputation_sequence_) == len(b.imputation_sequence_)
    for (ja, na, ea), (jb, nb, eb) in zip(a.imputation_sequence_, b.imputation_sequence_):
        assert ja == jb and list(na) == list(nb)
        assert type(ea) is type(host_estimator)
        if hasattr(eb, "coef_"):
            assert np.array_equal(np.asarray(ea.coef_), np.asarray(eb.coef_))
    assert np.array_equal(np.asarray(a.transform(Xh)), np.asarray(b.transform(Xh)))


def test_host_linear_members():
    X, Xh = _data(2), _data(3)
    for kw in (dict(), dict(imputation_order="random", random_state=3, add_indicator=True)):
        _same(host.HostRidge(alpha=1.0), ml.Ridge(alpha=1.0, device="cpu"), X, Xh, **kw)
        _same(host.HostLinearRegression(), ml.LinearRegression(device="cpu"), X, Xh, **kw)


def test_host_logistic_member():
    # a classifier host class, fitted on the float target as scikit-learn fits it
    X, Xh = _labels(0), _labels(1)
    _same(host.HostLogisticRegression(), ml.LogisticRegression(device="cpu"), X, Xh)


def test_host_inference_only_refused():
    # HostLasso's own fit refuses (its host image predicts a saved model only), so its member does not
    # serve the fit: the refusal it had before its mark
    try:
        ml.IterativeImputer(estimator=host.HostLasso()).fit(_data(0))
        raise AssertionError("no TypeError")
    except TypeError as error:
        assert "IterativeImputer requires a mojolearn estimator; got HostLasso" in str(error), error


if __name__ == "__main__":
    test_host_linear_members()
    test_host_logistic_member()
    test_host_inference_only_refused()
    print("PASS test_iterimp_host_members")
