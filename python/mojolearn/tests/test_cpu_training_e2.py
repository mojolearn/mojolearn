# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E batch 2, CPU training for the kmeans, metrics, spectral,
standard-scaler, minmax-scaler and logistic lanes, run only where the host
bindings are built and the package took the CPU-only path (skipped, and SAID
to be skipped, otherwise).

Each public fit runs twice on a small draw and returns the same bytes, with
its outputs in range; the metrics host entries give the same answers with one
and seven row tasks (each count in its own process); each binding reads back
sabotage False. They are plumbing checks. The bit claim against the GPU
columns is the CPU identity gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_e2
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import os
import sys
from pathlib import Path

import mojolearn
from mojolearn import _backend


def _run_with_threads(threads, code):
    """Run `code` in a fresh interpreter with MOJOLEARN_CPU_THREADS set to
    `threads` (None: unset) and return its stdout lines. The host thread
    policy is read once per process (core/host_predict_threads.mojo), so a
    count set inside this process after the first host call changes
    nothing; only a new process honours it. The child imports the same
    mojolearn package this process holds."""
    import subprocess
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if threads is None:
        env.pop("MOJOLEARN_CPU_THREADS", None)
    else:
        env["MOJOLEARN_CPU_THREADS"] = threads
    pkg_parent = str(Path(mojolearn.__file__).resolve().parent.parent)
    proc = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, sys.argv.pop(1)); exec(sys.argv.pop(1))",
         pkg_parent, code],
        env=env, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (threads, proc.stdout[-2000:], proc.stderr[-4000:])
    return proc.stdout.strip().splitlines()


#: The metrics host thread-count check, run once per thread count in its own
#: process: the same draw as the test, the address entries, one line out.
_METRICS_THREADS_CODE = """
import numpy as np
from mojolearn import _backend
module = _backend.load_host_module("_mojolearn_metrics_host")
rng = np.random.default_rng(1)
x = rng.standard_normal((300, 3)).astype(np.float32)
yt = (rng.integers(0, 3, 300)).astype(np.int32)
yp = (yt + (rng.random(300) < 0.2)).astype(np.int32) % 3
acc = module.accuracy_score(yt.ctypes.data, yp.ctypes.data, [len(yt)])
cm = np.empty(9, dtype=np.int64)
module.confusion_matrix(yt.ctypes.data, yp.ctypes.data, cm.ctypes.data, [len(yt), 3, 0])
print(repr(float(acc)), cm.tobytes().hex())
"""


def test_spectral_runs_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_metrics_host" not in _backend.host_families_built():
        print("SKIP: the metrics host binding is not built")
        return
    import numpy as np
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.standard_normal((60, 3)) + 6.0, rng.standard_normal((60, 3)) - 6.0]).astype(np.float32)
    got = [np.asarray(mojolearn.SpectralClustering(n_clusters=2, n_neighbors=8, random_state=5).fit(x).labels_)
           for _ in range(2)]
    assert got[0].tobytes() == got[1].tobytes(), "two host fits returned different labels"
    assert set(got[0].tolist()) <= {0, 1} and got[0].shape == (120,)
    assert (got[0][:60] == got[0][0]).all() and (got[0][60:] == got[0][60]).all() and got[0][0] != got[0][60], \
        "two separated blobs were not split into the two clusters"


def test_scalers_run_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_preprocessing_host" not in _backend.host_families_built():
        print("SKIP: the preprocessing host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_preprocessing_host")
    assert not bool(module.preprocessing_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(3)
    x = rng.standard_normal((700, 5)).astype(np.float32)
    x[:, 4] = np.float32(2.5)
    s = mojolearn.StandardScaler().fit(x)
    t = np.asarray(s.transform(x[:64]))
    back = np.asarray(s.inverse_transform(t))
    assert np.asarray(s.var_)[4] == 0.0 and np.asarray(s.scale_)[4] == 1.0 and np.asarray(s.mean_)[4] == np.float32(2.5)
    assert np.abs(np.asarray(s.mean_)[:4] - x[:, :4].mean(0)).max() < 1e-4
    assert np.abs(back - x[:64]).max() < 1e-4
    m = mojolearn.MinMaxScaler().fit(x)
    u = np.asarray(m.transform(x[:64]))
    assert np.asarray(m.data_min_).tolist() == x.min(0).tolist() and np.asarray(m.data_max_).tolist() == x.max(0).tolist()
    assert u[:, :4].min() >= 0.0 and u[:, :4].max() <= 1.0
    again = np.asarray(mojolearn.MinMaxScaler().fit(x).transform(x[:64]))
    assert again.tobytes() == u.tobytes(), "two host fits returned different bytes"


def test_logistic_runs_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_estimators_host" not in _backend.host_families_built():
        print("SKIP: the estimators host binding is not built")
        return
    import numpy as np
    rng = np.random.default_rng(4)
    x = rng.standard_normal((400, 3)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.int32)
    fits = [mojolearn.LogisticRegression(max_iter=30).fit(x, y) for _ in range(2)]
    a, b = (np.asarray(f.coef_) for f in fits)
    assert a.tobytes() == b.tobytes(), "two host fits returned different coefficients"
    assert fits[0].retcode_ in (0, 3) and int(np.asarray(fits[0].n_iter_)[0]) >= 1
    pred = np.asarray(fits[0].predict(x))
    assert (pred == y).mean() > 0.9, "the host fit does not separate a separable draw"
    # An l1 penalty takes the host OWL-QN arm (lane/cpu-training-batch3).
    l1 = [np.asarray(mojolearn.LogisticRegression(penalty="l1", C=1.0, max_iter=30).fit(x, y).coef_)
          for _ in range(2)]
    assert l1[0].tobytes() == l1[1].tobytes(), "two host OWL-QN fits returned different coefficients"


def test_metrics_run_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_metrics_host" not in _backend.host_families_built():
        print("SKIP: the metrics host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_metrics_host")
    assert not bool(module.metrics_host_sabotage()), "a sabotage build loaded outside the gate"
    mt = mojolearn.metrics
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 3)).astype(np.float32)
    yt = (rng.integers(0, 3, 300)).astype(np.int32)
    yp = (yt + (rng.random(300) < 0.2)).astype(np.int32) % 3
    # One and seven row tasks must give the same accuracy and confusion
    # matrix bytes; each count runs in its own process (read once).
    serial = _run_with_threads("1", _METRICS_THREADS_CODE)
    parallel = _run_with_threads("7", _METRICS_THREADS_CODE)
    assert serial and serial == parallel, (serial, parallel)
    got = [(mt.accuracy_score(yt, yp), mt.adjusted_rand_score(yt, yp), mt.v_measure_score(yt, yp),
            mt.r2_score(x[:, 0], x[:, 0] * np.float32(0.9)), mt.silhouette_score(x, yp))
           for _ in range(2)]
    assert got[0] == got[1], "two host runs returned different values"
    acc, ari, vm, r2, sil = got[0]
    assert acc == float((yt == yp).sum()) / 300.0 or abs(acc - (yt == yp).mean()) < 1e-6
    assert -0.5 <= ari <= 1.0 and 0.0 <= vm <= 1.0 and 0.0 <= r2 <= 1.0 and -1.0 <= sil <= 1.0


def test_kmeans_fit_runs_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_core_host" not in _backend.host_families_built():
        print("SKIP: the core host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_core_host")
    assert not bool(module.core_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 5)).astype(np.float32)
    outs = []
    for _ in range(2):
        m = mojolearn.KMeans(n_clusters=4, random_state=7).fit(x)
        outs.append((np.asarray(m.cluster_centers_).tobytes(), np.asarray(m.labels_).tobytes(),
                     m.inertia_, m.n_iter_, m.sum_scale_, m.weight_scale_))
    assert outs[0] == outs[1], "two host fits returned different bytes"
    labels = np.asarray(mojolearn.KMeans(n_clusters=4, random_state=7).fit(x).labels_)
    assert labels.shape == (600,) and labels.min() >= 0 and labels.max() < 4
    assert outs[0][3] >= 1 and outs[0][4] > 0 and outs[0][5] > 0


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
