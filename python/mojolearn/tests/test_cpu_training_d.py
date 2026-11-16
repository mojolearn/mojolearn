# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the workstream D estimators: cholesky, rbf-sampler,
kernel-ridge, nystroem, gmm, gmm-random-init, hdbscan and hdbscan-leaf.

The runtime checks (skipped, and SAID to be skipped, when a binding is absent
or a GPU set loaded) fit each estimator twice through the public API on a
CPU-only install and require the same bytes, and run HDBSCAN once with the
default host workers and once with one task, each in its own process. They
are plumbing checks; the bit claim against the GPU columns is the CPU
identity gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_d
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import os
import sys
from pathlib import Path

import mojolearn
from mojolearn import _backend


def _built(basename):
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    if basename not in _backend.host_families_built():
        print(f"SKIP: {basename} is not built")
        return False
    return True


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


#: The HDBSCAN default-workers versus one-task check, run once per thread
#: setting in its own process.
_HDBSCAN_THREADS_CODE = """
import numpy as np
import mojolearn
rng = np.random.default_rng(0)
x = rng.standard_normal((96, 4)).astype(np.float32)
m = mojolearn.HDBSCAN(min_cluster_size=5).fit(x)
print(np.asarray(m.labels_).tobytes().hex())
print(np.asarray(m.core_distances_).tobytes().hex())
print(m.n_clusters_, m.n_outliers_, m.n_boruvka_rounds_)
"""


def _twice(fit):
    a, b = fit(), fit()
    assert a == b, "two host fits returned different bytes"


def test_estimators_fit_on_the_host_when_built():
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.standard_normal((96, 4)).astype(np.float32)
    y = rng.standard_normal(96).astype(np.float32)
    if _built("_mojolearn_gp_host"):
        a = (x.T @ x + 4.0 * np.eye(4)).astype(np.float32)
        _twice(lambda: np.asarray(mojolearn.Cholesky().fit(a).solve(np.ones((4, 1), np.float32))).tobytes())
    if _built("_mojolearn_kernel_methods_host"):
        _twice(lambda: np.asarray(mojolearn.KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5).fit(x, y).predict(x[:8])).tobytes())
        _twice(lambda: np.asarray(mojolearn.Nystroem(gamma=0.5, n_components=8, random_state=1).fit(x).transform(x[:8])).tobytes())
        _twice(lambda: np.asarray(mojolearn.RBFSampler(gamma=0.5, n_components=8, random_state=1).fit(x).transform(x[:8])).tobytes())
    if _built("_mojolearn_mixture_host"):
        _twice(lambda: np.asarray(mojolearn.GaussianMixture(n_components=2, max_iter=5, random_state=3).fit(x).score_samples(x[:8])).tobytes())
    if _built("_mojolearn_hdbscan_host"):
        _twice(lambda: np.asarray(mojolearn.HDBSCAN(min_cluster_size=5).fit(x).labels_).tobytes())
        # 96**2 is below the Boruvka parallel threshold.  Its default-worker
        # answer must cover every row just like the explicit one-task path.
        # Each setting runs in its own process (the thread policy is read
        # once per process): labels, core distances and the counts.
        default = _run_with_threads(None, _HDBSCAN_THREADS_CODE)
        serial = _run_with_threads("1", _HDBSCAN_THREADS_CODE)
        assert len(default) == 3 and default == serial, (default, serial)


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)

