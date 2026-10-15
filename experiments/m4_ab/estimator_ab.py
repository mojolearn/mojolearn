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
        return KMeans(n_clusters=16, n_init=1, max_iter=20, tol=0.0, random_state=0)
    if name == 'pca':
        from mojolearn.decomposition import PCA
        return PCA(n_components=16)
    raise ValueError(name)


def quality(name, model, x):
    if name == 'kmeans':
        return dict(inertia=float(model.inertia_))
    return dict(explained_variance_ratio_sum=float(np.sum(model.explained_variance_ratio_)))


def run(root, rows, features, name, arm, out):
    x = np.load(data_path(root, rows, features))
    times = []
    for _ in range(2):
        model = make(name)
        start = time.perf_counter_ns()
        model.fit(x)
        times.append((time.perf_counter_ns() - start) / 1e6)
    import mojolearn
    record = dict(estimator=name, arm=arm, rows=rows, features=features, warmup_ms=times[0],
                  scored_ms=times[1], module=str(Path(mojolearn.__file__).parent), **quality(name, model, x))
    with open(out, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7])
