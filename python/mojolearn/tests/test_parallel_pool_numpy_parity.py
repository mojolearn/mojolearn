# SPDX-License-Identifier: Apache-2.0
"""Pool workers load NumPy when their parent had it loaded.

`import mojolearn` no longer loads NumPy, so a fresh pool worker had none,
and the pooled Gram PCA (`parallel_classical.fit_gram_estimator`, lane
par-gram-pca) failed in the worker with "PCA requires optional NumPy
support" although the parent had NumPy (identity record, AMD MI325X,
5b1606ab1). The native PCA owner reads NumPy's connected types during fit
(bindings/pca_native.mojo), so the worker must run under the parent's
optional dependency set.
"""
import sys

import numpy as np
import pytest

from mojolearn import _backend
from mojolearn import _parallel_pool as pool


def test_worker_environment_follows_the_parents_numpy(monkeypatch):
    launched = []

    class Fake:
        def __init__(self, args, env, **kwargs):
            launched.append(env)

        def poll(self):
            return 0

    monkeypatch.setattr(pool.subprocess, "Popen", Fake)
    monkeypatch.setattr(_backend, "vendor", lambda: "cpu")
    monkeypatch.setenv("MOJOLEARN_POOL_IMPORT_NUMPY", "1")
    assert "numpy" in sys.modules
    pool.DevicePool((0,))._start()
    assert launched[-1]["MOJOLEARN_POOL_IMPORT_NUMPY"] == "1"
    monkeypatch.delitem(sys.modules, "numpy")
    pool.DevicePool((0,))._start()
    assert "MOJOLEARN_POOL_IMPORT_NUMPY" not in launched[-1]


def test_pooled_gram_pca_fits_in_a_fresh_worker():
    if _backend.vendor() in (None, "cpu"):
        pytest.skip("the cooperative Gram worker needs a GPU device")
    import mojolearn as ml
    from mojolearn.parallel_classical import fit_gram_estimator
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4096, 12)).astype(np.float32)
    par = fit_gram_estimator(ml.PCA(n_components=4), X, devices=(0,))
    plain = ml.PCA(n_components=4).fit(X)
    assert np.asarray(par.components_).tobytes() == np.asarray(plain.components_).tobytes()
    assert np.asarray(par.transform(X)).tobytes() == np.asarray(plain.transform(X)).tobytes()
