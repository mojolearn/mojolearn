# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every x_cluster entry point (x_cluster/entries.mojo, `_E_*` in
python/mojolearn/_expansion_cluster.py, `_E_AGGLO` in _hierarchy_impl.py,
`_E_SPECTRAL_ASSIGN` in _spectral_impl.py) called at least TWICE in one process,
on the GPU binding and on the CPU host binding, with the second call's output
equal to the first's byte for byte.

Regression for the hang the cpu lane found on an RTX 4090 (main acbb70e0a):
`x_cluster/device_ops.mojo` built a DeviceContext per call and held the
call's buffers in fields declared after it, so the context died before its
buffers and the SECOND `x_cluster_call` in a process never returned. The
context is now one per process (`x_cluster_ctx`, the x_cnn `_Global`
pattern). Each backend runs in a child process with a timeout, so a hang is a
failure, not a stuck test run."""
import os
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent('''
    import numpy as np
    import mojolearn as ml

    rng = np.random.default_rng(7)
    X = np.ascontiguousarray(np.concatenate([rng.normal(c, 0.4, size=(60, 3)) for c in (0.0, 4.0, 8.0)]),
                             dtype=np.float32)
    Q = np.ascontiguousarray(X[::7] + np.float32(0.1))

    def run():
        out = []
        m = ml.MiniBatchKMeans(n_clusters=3, batch_size=64, max_iter=3, random_state=1).fit(X)   # _E_MINIBATCH
        out += [m.cluster_centers_, m.labels_, m.predict(Q), m.transform(Q)]                      # _E_NEAREST, _E_DISTANCES
        p = ml.MiniBatchKMeans(n_clusters=3, batch_size=64, random_state=1)
        p.partial_fit(X[:90]); p.partial_fit(X[90:])                                              # _E_MINIBATCH_PARTIAL
        out += [p.cluster_centers_]
        b = ml.BisectingKMeans(n_clusters=3, random_state=1).fit(X)                                # _E_BISECT
        out += [b.cluster_centers_, b.labels_, b.predict(Q)]                                      # _E_BISECT_PREDICT
        s = ml.MeanShift(bandwidth=1.5).fit(X)                                                     # _E_MEANSHIFT
        out += [s.cluster_centers_, s.labels_]
        o = ml.OPTICS(min_samples=5).fit(X)                                                        # _E_OPTICS
        out += [o.ordering_, o.reachability_, o.labels_]
        a = ml.AffinityPropagation(random_state=0).fit(X[:90])                                    # _E_AFFINITY
        out += [a.cluster_centers_indices_, a.labels_]
        g = ml.BayesianGaussianMixture(n_components=3, max_iter=20, random_state=1).fit(X)        # _E_BGMM
        out += [g.means_, g.score_samples(Q)]                                                     # _E_BGMM_SCORE
        h = ml.AgglomerativeClustering(n_clusters=3, linkage="ward", compute_distances=True).fit(X)  # _E_AGGLO
        out += [h.children_, h.distances_, h.labels_]
        for how in ("discretize", "cluster_qr"):                                                   # ENTRY_SPECTRAL_ASSIGN
            c = ml.SpectralClustering(n_clusters=3, assign_labels=how, random_state=1).fit(X)
            out += [c.labels_]
        return [np.asarray(v).tobytes() for v in out]

    first = run()
    second = run()
    assert len(first) == len(second)
    for i, (u, v) in enumerate(zip(first, second)):
        assert u == v, f"output {i} of the second call differs from the first"
    print("TWICE OK", ml.vendor())
''')


def _run(env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=900)


def _binding_present(name):
    try:
        from mojolearn import _backend
        _backend.binding(name, "identical")
        return True
    except Exception:
        return False


@pytest.mark.parametrize("backend", ["gpu", "cpu"])
def test_every_x_cluster_entry_twice_in_one_process(backend):
    if backend == "gpu":
        if not _binding_present("_mojolearn_x_cluster"):
            pytest.skip("no GPU x_cluster binding in this install")
        r = _run({})
    else:
        r = _run({"MOJOLEARN_VENDOR": "cpu"})
    if r.returncode != 0 and "No module named" in r.stderr and backend == "cpu":
        pytest.skip("no CPU host binding in this install")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    assert "TWICE OK" in r.stdout
