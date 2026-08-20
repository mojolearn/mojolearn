"""Import the installed wheel and run a real fit on both estimators.

Not a hello-world. It plants four separated clusters and requires k-means to
recover them, and requires k-NN's nearest neighbor of a point to be itself.
An import-only smoke test would pass on a wheel whose GPU path is broken.

`--no-gpu` runs everything EXCEPT the fits: import, version, the estimator
classes, the named absences, and argument validation. It exists for one
specific environment and should not be used anywhere else.

    GitHub's hosted macOS runner is an "Apple M1 (Virtual)" that reports no
    GPU to system_profiler and CANNOT create a Metal function using
    `block_sum` -- measured 2026-08-20, tools/hosted_block_probe.mojo fails
    there while the trivial elementwise kernel passes. So CI can verify that
    the wheel BUILDS and INSTALLS on 3.10 through 3.14, and cannot verify
    that it COMPUTES. Pretending otherwise, by letting the GPU failure pass
    under continue-on-error, produced a green run in which every interpreter
    had failed. A check that cannot run must say so, not report success.

GPU execution is verified on real Apple silicon by
packaging/macos/verify_wheel.sh without this flag.
"""
import sys

import numpy as np

import mojolearn

NO_GPU = "--no-gpu" in sys.argv

if NO_GPU:
    # Everything that does not touch the device. Import already exercised the
    # extension load, which is what the dylib staging and the wheel tag are
    # actually about.
    assert mojolearn.__version__
    assert hasattr(mojolearn, "KMeans")
    assert hasattr(mojolearn, "NearestNeighbors")
    for absent in ("DBSCAN", "PCA", "LinearRegression"):
        try:
            getattr(mojolearn, absent)
        except AttributeError as exc:
            assert "no caller-facing surface" in str(exc), absent
        else:  # pragma: no cover
            raise AssertionError(f"{absent} should be a named absence")
    # Validation happens before any device work, so it is reachable here.
    try:
        mojolearn.KMeans(n_clusters=99).fit(np.zeros((4, 2), dtype=np.float32))
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("n_clusters > n_samples should raise")
    print(
        f"v{mojolearn.__version__} py{sys.version_info.major}."
        f"{sys.version_info.minor} IMPORT+API ok (--no-gpu, device untested)"
    )
    raise SystemExit(0)

rng = np.random.default_rng(0)
centers = np.array([[0, 0], [40, 40], [0, 40], [40, 0]], dtype=np.float32)
X = np.repeat(centers, 100, axis=0) + rng.normal(0, 1, (400, 2)).astype(np.float32)

km = mojolearn.KMeans(
    n_clusters=4, init="array", init_centroids=centers + 4, max_iter=50
).fit(X)
for c in range(4):
    block = km.labels_[c * 100 : (c + 1) * 100]
    assert len(set(block.tolist())) == 1, f"cluster {c} split across labels"
assert len({tuple(r) for r in km.cluster_centers_.round(0)}) == 4, "centroids merged"

nn = mojolearn.NearestNeighbors(n_neighbors=3).fit(X)
d, i = nn.kneighbors(X[:50])
assert (i[:, 0] == np.arange(50)).all(), "a point is not its own nearest neighbour"
assert (d[:, 0] < 1e-3).all(), "self-distance is not ~0"

print(
    f"v{mojolearn.__version__} py{sys.version_info.major}.{sys.version_info.minor}"
    f" kmeans n_iter={km.n_iter_} knn tile={nn.used_query_tile_}"
)
