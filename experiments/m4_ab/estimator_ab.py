"""Local Apple FAST estimator A/B on the M4 (testing code, never product execution).

Each call times one arm on one estimator and shape in a fresh process: one
excluded warmup fit and one scored fit (one run per arm). The fit clock
includes the host-to-device copy, as the board's does. Data is seeded
standard normal with a planted cluster structure, written once per shape.

  python estimator_ab.py prepare DATA ROWS FEATURES
  MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=<arm pkg> python estimator_ab.py run DATA ROWS FEATURES ESTIMATOR ARM OUT

The driver script alternates arm order per shape (A B, then B A) so laptop
heating does not favor one arm.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def data_path(root, rows, features):
    return Path(root) / ('x-%d-%d.npy' % (rows, features))


def prepare(root, rows, features):
    path = data_path(root, rows, features)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    centers = rng.standard_normal((16, features)).astype(np.float32) * 4
    x = rng.standard_normal((rows, features), dtype=np.float32)
    x += centers[rng.integers(0, 16, rows)]
    np.save(path, x)


def make(name):
    if name == 'kmeans':
        from mojolearn.cluster import KMeans
        return KMeans(n_clusters=16, n_init=1, max_iter=20, tol=1e-12, random_state=0)
    if name == 'pca':
        from mojolearn.decomposition import PCA
        return PCA(n_components=16)
    if name == 'dtree_reg':
        from mojolearn import DecisionTreeRegressor
        return DecisionTreeRegressor(random_state=0)
    if name == 'ivf_sq':
        from mojolearn import IVFSQIndex
        return IVFSQIndex(n_lists=1024, n_probes=32, n_neighbors=8, random_state=0)
    raise ValueError(name)


def target(x):
    # Nonlinear synthetic target, not a board dataset (testing code only).
    rng = np.random.default_rng(11)
    return (3 * np.sin(x[:, 0]) + x[:, 1] * x[:, 2] + np.abs(x[:, 3]) + 0.1 * rng.standard_normal(len(x))).astype(np.float32)


def quality(name, model, x):
    if name == 'kmeans':
        return dict(inertia=float(model.inertia_))
    if name == 'dtree_reg':
        from sklearn.metrics import r2_score
        held = x[-50000:]
        return dict(heldout_r2=float(r2_score(target(held), np.asarray(model.predict(held)))))
    if name == 'ivf_sq':
        q = x[:2000]
        _, got = model.search(q)
        xx = (x * x).sum(1)[None, :]
        exact = np.concatenate([np.argpartition(xx - 2 * c @ x.T, 8, axis=1)[:, :8]
                                for c in np.array_split(q, 16)])
        hits = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(np.asarray(got), exact))
        return dict(recall_at_8=hits / (8 * len(q)))
    return dict(explained_variance_ratio_sum=float(np.asarray(model.explained_variance_ratio_).sum()))


def quiet(limit=float(os.environ.get('M4AB_MAX_LOAD', '2.0')), wait_s=600):
    # A/A on a loaded laptop (load 6.3: other sessions' mojo builds, a simulator, Metal compiles)
    # differed by 39% on identical code, so each arm waits for a quiet machine and records the load.
    start = time.time()
    while os.getloadavg()[0] > limit and time.time() - start < wait_s:
        time.sleep(15)
    return os.getloadavg()[0]


def run(root, rows, features, name, arm, out):
    x = np.load(data_path(root, rows, features))
    load = quiet()
    times = []
    for _ in range(2):
        model = make(name)
        fit_x, fit_args = x, ()
        if name == 'dtree_reg':
            fit_x, fit_args = x[:-50000], (target(x[:-50000]),)
        start = time.perf_counter_ns()
        model.fit(fit_x, *fit_args)
        times.append((time.perf_counter_ns() - start) / 1e6)
    import mojolearn
    record = dict(estimator=name, arm=arm, rows=rows, features=features, warmup_ms=times[0],
                  scored_ms=times[1], load_1min=load, quiet=load <= float(os.environ.get('M4AB_MAX_LOAD', '2.0')), module=str(Path(mojolearn.__file__).parent), **quality(name, model, x))
    with open(out, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7])
