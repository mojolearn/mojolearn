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
    # The pooled model crossed a pickle: its arrays carry new (equal) dtype
    # instances, which the PCA state admission once refused by identity.
    for name in ("components_", "mean_", "explained_variance_", "explained_variance_ratio_",
                 "singular_values_"):
        got, want = getattr(par, name), getattr(plain, name)
        assert type(got) is type(want), name
        assert got.dtype == want.dtype and got.shape == want.shape, name
        assert got.flags.c_contiguous == want.flags.c_contiguous, name
        assert np.asarray(got).tobytes() == np.asarray(want).tobytes(), name
    for name in ("n_components_", "n_features_in_", "noise_variance_"):
        assert getattr(par, name) == getattr(plain, name), name
    # transform and inverse_transform run in the parent on the pooled state.
    assert np.asarray(par.transform(X)).tobytes() == np.asarray(plain.transform(X)).tobytes()
    Z = np.asarray(plain.transform(X))
    assert np.asarray(par.inverse_transform(Z)).tobytes() == np.asarray(plain.inverse_transform(Z)).tobytes()


# THE par-kernel-ridge DTYPE FLIP (2026-10-24). Loading NumPy in the worker
# made it hand back PUBLIC results (a regressor's `predict` widened to
# float64), because the worker runs as `__main__` and was taken for a user.
# The worker is internal now (`_parallel_worker.__mojolearn_internal__`), so
# what it sends back must not depend on MOJOLEARN_POOL_IMPORT_NUMPY.

_WORKER_PROBE = r"""
import os, runpy, sys
if os.environ.get("MOJOLEARN_POOL_IMPORT_NUMPY") != "1":
    sys.modules["numpy"] = None   # the NumPy-less worker: no import possible
from mojolearn import _mode
from mojolearn._array import Array
g = runpy.run_module("mojolearn._parallel_worker", run_name="__main__")

class Regressor:
    _estimator_type = "regressor"
    def _raw(self, X):
        return Array((3,), "<f4")   # the native float32 result
    predict = _mode._public_output(_raw, "predict")

g["est"] = Regressor()
exec("def _ask():\n    return est.predict(None)\n", g)
out = g["_ask"]()
sys.stderr.write("PROBE %s %s %s\n" % (type(out).__module__, type(out).__name__, out.dtype))
"""


def _probe_worker(numpy_flag):
    import os
    import subprocess
    env = dict(os.environ)
    if numpy_flag:
        env["MOJOLEARN_POOL_IMPORT_NUMPY"] = "1"
    else:
        env.pop("MOJOLEARN_POOL_IMPORT_NUMPY", None)
    done = subprocess.run([sys.executable, "-c", _WORKER_PROBE], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    lines = [line for line in done.stderr.decode().splitlines() if line.startswith("PROBE ")]
    assert len(lines) == 1, done.stderr[-2000:]
    return lines[0]


def test_worker_public_method_result_ignores_the_numpy_flag():
    with_numpy, without = _probe_worker(True), _probe_worker(False)
    assert with_numpy == without
    assert with_numpy.endswith(" Array <f4"), with_numpy


def test_pooled_kernel_ridge_predict_bytes_ignore_the_workers_numpy(monkeypatch):
    if _backend.vendor() in (None, "cpu"):
        pytest.skip("the cooperative kernel-methods worker needs a GPU device")
    import mojolearn as ml
    from mojolearn import parallel_classical
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, 6)).astype(np.float32)
    y = rng.standard_normal(256).astype(np.float32)
    est = ml.KernelRidge(alpha=1.0, kernel="rbf").fit(X, y)
    # this module is mojolearn's own (`mojolearn.tests.*`): raw results
    plain = est.predict(X)
    real_pool = pool.DevicePool
    results = {}
    for flag in (True, False):
        def started(devices, cooperative=False, _flag=flag):
            p = real_pool(devices, cooperative=cooperative)
            if _flag:
                p._start()
            else:
                saved = sys.modules.pop("numpy")
                try:
                    p._start()   # the worker environment drops the NumPy flag
                finally:
                    sys.modules["numpy"] = saved
            return p
        monkeypatch.setattr(parallel_classical, "DevicePool", started)
        results[flag] = parallel_classical.apply_kernel_method(est, X, devices=(0,))
    for got in results.values():
        assert type(got) is type(plain)
        assert got.dtype == plain.dtype
        assert np.asarray(got).tobytes() == np.asarray(plain).tobytes()
