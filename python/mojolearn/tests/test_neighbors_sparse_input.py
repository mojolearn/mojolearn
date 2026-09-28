# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Sparse X on the neighbors family's existing estimators (2026-09-27): a
scipy.sparse input is densified exactly (_buffer.as_f32_dense_c), so every
answer must be its dense twin's bytes. Skips, saying so, without scipy or a
binding."""

import numpy as np
import pytest

import mojolearn as ml

sp = pytest.importorskip("scipy.sparse")


def _data(n=120, d=6, seed=8):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    x[rng.random(size=(n, d)) < 0.5] = 0.0
    return x, (x[:, 0] > 0).astype(np.int64), (x[:, 1] + 0.5 * x[:, 2]).astype(np.float32)


def _same(a, b):
    if isinstance(a, tuple):
        return all(_same(u, v) for u, v in zip(a, b))
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


CASES = [
    ("SVC", lambda: ml.SVC(C=0.5), "yc", lambda m, q: (m.decision_function(q), m.predict(q))),
    ("SVR", lambda: ml.SVR(C=0.5), "yr", lambda m, q: m.predict(q)),
    ("KernelRidge", lambda: ml.KernelRidge(alpha=1.0, kernel="rbf", gamma=0.2), "yr", lambda m, q: m.predict(q)),
    ("Nystroem", lambda: ml.Nystroem(n_components=16, random_state=1), None, lambda m, q: m.transform(q)),
    ("RBFSampler", lambda: ml.RBFSampler(n_components=16, random_state=1), None, lambda m, q: m.transform(q)),
    ("KNeighborsClassifier", lambda: ml.KNeighborsClassifier(n_neighbors=5), "yc", lambda m, q: m.predict(q)),
    ("KNeighborsRegressor", lambda: ml.KNeighborsRegressor(n_neighbors=5), "yr", lambda m, q: m.predict(q)),
    ("NearestNeighbors", lambda: ml.NearestNeighbors(n_neighbors=4), None, lambda m, q: m.kneighbors(q)),
    ("KernelDensity", lambda: ml.KernelDensity(bandwidth=0.8), None, lambda m, q: m.score_samples(q)),
]


@pytest.mark.parametrize("name, make, target, probe", CASES, ids=[c[0] for c in CASES])
def test_sparse_is_its_dense_twin(name, make, target, probe):
    x, yc, yr = _data()
    y = {"yc": yc, "yr": yr, None: None}[target]
    args = () if y is None else (y,)
    try:
        dense = make().fit(x, *args)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no binding for {name} on this install: {exc}")
    sparse = make().fit(sp.csr_matrix(x), *args)
    q = x[:40]
    assert _same(probe(dense, q), probe(sparse, sp.csr_matrix(q)))
    assert _same(probe(dense, q), probe(sparse, sp.csc_matrix(q)))
