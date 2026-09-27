# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every x_neighbors entry point called TWICE in one process, on the GPU
binding and on the CPU host binding, with the second call's bits equal to
the first's.

The regression it guards: each GPU driver built its own DeviceContext, whose
DeviceBuffers outlived it, and the SECOND GPU call in a process hung
(LocalOutlierFactor on an RTX 4090, 2026-09-27). The lane checks call each
entry point once per process, so they could not see it. x_neighbors/gen.py
now hands every driver one process-lifetime context (`xn_ctx`).

Each column runs in its own subprocess (the CPU one with MOJOLEARN_VENDOR=cpu
and the host bindings the lane check builds under python/mojolearn/host),
with a timeout, so a hang fails the test instead of stalling the run.
Runs on the lane's pod:
`python -m pytest python/mojolearn/tests/test_x_neighbors_repeat.py`."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

PKG = Path(__file__).resolve().parents[1]

_SCRIPT = r'''
import hashlib
import numpy as np
import mojolearn as ml

rng = np.random.default_rng(0)
X = rng.standard_normal((96, 5)).astype(np.float32)
Xh = rng.standard_normal((24, 5)).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
A = ((np.floor(X[:48, 3] * 2)[:, None] == np.floor(X[:48, 3] * 2)[None, :])
     & ~np.eye(48, dtype=bool)).astype(np.float32)
H = X.copy(); H[::7, 2] = np.nan
lab = np.where(np.arange(96) % 3 == 0, y, -1)


def calls():
    out = []
    lof = ml.LocalOutlierFactor(n_neighbors=5, novelty=True).fit(X)
    out += [lof.score_samples(Xh), lof.predict(Xh)]
    out += [ml.LocalOutlierFactor(n_neighbors=5).fit_predict(X)]
    nc = ml.NearestCentroid(shrink_threshold=0.2).fit(X, y)
    out += [nc.predict(Xh), nc.predict_proba(Xh), nc.predict_log_proba(Xh)]
    oc = ml.OneClassSVM(nu=0.3).fit(X)
    out += [oc.decision_function(Xh), oc.predict(Xh)]
    out += [ml.KernelPCA(n_components=3, kernel="rbf").fit(X).transform(Xh)]
    out += [ml.PolynomialCountSketch(n_components=16, random_state=1).fit(X).transform(Xh)]
    out += [ml.AdditiveChi2Sampler().fit(np.abs(X)).transform(np.abs(Xh))]
    out += [ml.SkewedChi2Sampler(n_components=16, random_state=1).fit(np.abs(X)).transform(np.abs(Xh))]
    out += [ml.LabelPropagation().fit(X, lab).predict_proba(Xh)]
    out += [ml.LabelSpreading().fit(X, lab).predict_proba(Xh)]
    out += [ml.KNNImputer(n_neighbors=3).fit(H).transform(H)]
    out += [ml.PageRank().fit(A).pagerank_]
    out += [ml.connected_components(A, directed=False)[1]]
    out += [ml.Louvain().fit(A).labels_]
    sg = ml.SVGP(n_inducing=8).fit(X, X[:, 0])
    out += list(sg.predict_f(Xh))
    h = hashlib.sha256()
    for a in out:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


first = calls()
second = calls()
print("FIRST", first)
print("SECOND", second)
assert first == second, "a second call in the same process moved the bits"
'''


def _run(env_extra):
    env = dict(os.environ, PYTHONPATH=str(PKG.parent), MOJOLEARN_NUMERIC_MODE="identical")
    env.update(env_extra)
    proc = subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True,
                          text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return [l for l in proc.stdout.splitlines() if l.startswith("FIRST")][0].split()[1]


def test_every_entry_point_twice_on_gpu_and_cpu():
    gpu = _run({})
    cpu = _run({"MOJOLEARN_VENDOR": "cpu", "MOJOLEARN_HOST_DIR": str(PKG / "host")})
    assert gpu == cpu, (gpu, cpu)
