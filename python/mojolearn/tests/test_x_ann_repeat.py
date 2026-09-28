# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every ann-family entry point called TWICE in one process, on the GPU
bindings and on the CPU host bindings, with the second call's bits equal to
the first's and the GPU column equal to the CPU column.

Entry points: IVFPQIndex / IVFSQIndex / IVFRaBitQIndex fit and search (with
and without the sample filter), refine, CagraIndex fit and search, TSNE fit,
and IVF-Flat's IVFIndex fit, search and extend (bindings/_mojolearn_x_ann*
and bindings/_mojolearn_ivf*).

The regression it guards (CURRENT DIRECTIVES, 2026-09-27): x_cluster and
x_neighbors built a DeviceContext per call whose buffers outlived it, and the
SECOND GPU call in a process hung. The lane checks call each entry point once
per process, so they could not see it. x_ann now runs every device entry on
one process-lifetime context (x_ann/device_ctx.mojo, `x_ann_ctx`).

Each column runs in its own subprocess with a timeout, so a hang fails the
test instead of stalling the run. Runs on the lane's pod:
`python -m pytest python/mojolearn/tests/test_x_ann_repeat.py`."""
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
X = rng.standard_normal((1024, 16)).astype(np.float32)
Q = rng.standard_normal((32, 16)).astype(np.float32)
keep = (np.arange(1024) % 3) != 0
cand = ((np.arange(32 * 16, dtype=np.int64).reshape(32, 16) * 7919 + 13) % 1024).astype(np.int32)
cand[:, 5] = -1


def calls():
    out = []
    pq = ml.IVFPQIndex(n_lists=8, n_probes=3, pq_dim=4, pq_bits=4, n_neighbors=5,
                       pq_kmeans_n_iters=5, random_state=3).fit(X)
    out += [pq.codes_, *pq.search(Q), *pq.search(Q, filter=keep)]
    sq = ml.IVFSQIndex(n_lists=8, n_probes=3, n_neighbors=5, random_state=3).fit(X)
    out += [sq.codes_, *sq.search(Q), *sq.search(Q, filter=keep)]
    rq = ml.IVFRaBitQIndex(n_lists=8, n_probes=3, n_neighbors=5, random_state=3).fit(X)
    out += [rq.codes_, *rq.search(Q), *rq.search(Q, filter=keep)]
    out += list(ml.refine(X, Q, cand, 5))
    cg = ml.CagraIndex(graph_degree=8, intermediate_graph_degree=16, n_neighbors=5, itopk_size=16,
                       n_seeds=8).fit(X[:512])
    out += [cg.graph_, *cg.search(Q)]
    ts = ml.TSNE(perplexity=5.0, max_iter=60, random_state=5).fit(X[:120])
    out += [ts.embedding_, np.float32(ts.kl_divergence_)]
    iv = ml.IVFIndex(n_lists=8, n_probes=3, n_neighbors=5, random_state=3).fit(X[:768])
    out += [*iv.search(Q), iv.n_candidates_]
    iv.extend(X[768:])
    out += [iv.list_indices_, *iv.search(Q)]
    h = hashlib.sha256()
    for a in out:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


first = calls()
second = calls()
print("FIRST", first, ml.vendor())
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


def test_every_ann_entry_point_twice_on_gpu_and_cpu():
    gpu = _run({})
    cpu = _run({"MOJOLEARN_VENDOR": "cpu", "MOJOLEARN_HOST_DIR": str(PKG / "host")})
    assert gpu == cpu, (gpu, cpu)
