"""Import the installed wheel and run a real fit on both estimators.

Not a hello-world. It plants four separated clusters and requires k-means to
recover them, and requires k-NN's nearest neighbour of a point to be itself.
An import-only smoke test would pass on a wheel whose GPU path is broken.
"""
import sys

import numpy as np

import mojolearn

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
