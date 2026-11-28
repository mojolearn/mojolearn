# Clustering

Clustering groups rows that are close to each other without using labels. Pick the method by the shape of
the groups you expect.

| Use | Estimators |
|---|---|
| Round groups of similar size, with a known count | `KMeans`, `MiniBatchKMeans`, `BisectingKMeans` |
| Groups of any shape, with noise points left out | `DBSCAN`, `HDBSCAN`, `OPTICS` |
| Soft assignments and a density model | `GaussianMixture`, `BayesianGaussianMixture` |
| A tree of merges | `AgglomerativeClustering` |
| Groups found from a similarity graph | `SpectralClustering`, `AffinityPropagation`, `MeanShift` |
| Communities in a graph you already have | `Leiden`, `Louvain` |

## K-means

```python
import numpy as np
import mojolearn

X = np.random.default_rng(0).random((10_000, 8), dtype=np.float32)

km = mojolearn.KMeans(n_clusters=4, n_init=3).fit(X)
labels = km.predict(X)
print(km.cluster_centers_.shape, km.inertia_)
```

`KMeans` follows cuVS's defaults, so `n_init` is 1 unless you set it. Set the same `n_init` when you
compare with scikit-learn.

`km.score(X)` is the opposite of the inertia of `X` on the fitted centers, as in scikit-learn. On the
training rows it equals `-km.inertia_`, and it is the score a search with `scoring=None` ranks by.

## Density-based clustering

`DBSCAN` and `HDBSCAN` find groups of any shape and label points in no group `-1`.

```python
db = mojolearn.DBSCAN(eps=0.5, min_samples=5).fit(X)
hdb = mojolearn.HDBSCAN(min_cluster_size=10).fit(X)
print(set(db.labels_), hdb.n_clusters_)
```

## Gaussian mixtures

A mixture gives each row a probability for each component and can draw new samples.

```python
gmm = mojolearn.GaussianMixture(n_components=3).fit(X)
proba = gmm.predict_proba(X[:5])
print(gmm.bic(X))
```

## Communities in a graph

`Leiden` and `Louvain` take a symmetric weighted adjacency matrix (dense, scipy.sparse, or an
`(indptr, indices, n)` tuple) and return a community for every node. `Leiden` is the method single-cell
pipelines use (Scanpy's `tl.leiden`): it refines each community before merging, so every community it
returns is connected.

```python
import numpy as np
import mojolearn

M = np.random.default_rng(0).random((2_000, 2_000)) < 0.005
A = (M | M.T).astype(np.float32)

leiden = mojolearn.Leiden(resolution=1.0).fit(A)
print(leiden.n_communities_, leiden.quality_)

cpm = mojolearn.Leiden(resolution=0.05, objective_function="CPM").fit_predict(A)
```

`resolution` sets the community size: larger values give more, smaller communities. With
`objective_function="CPM"` it is the edge density a community must exceed. `n_iterations=-1` (the
default) repeats the algorithm until the partition stops improving, and `random_state` fixes the order
in which nodes move. The moves run in parallel in a fixed order, so the partition is reproducible but
can differ from leidenalg's.

## Saving a model

`KMeans`, `DBSCAN`, `HDBSCAN`, `AgglomerativeClustering`, `SpectralClustering` and `GaussianMixture`
save to a file and load back:

```python
km.save("kmeans.npz")
loaded = mojolearn.KMeans.load("kmeans.npz")
```

## Devices and numeric modes

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. `identical` is the
default. On Apple GPUs you can choose `fast` instead, which is tuned for speed and does not promise
identical bits: call `mojolearn.set_numeric_mode()` with `"fast"`. See [Numeric modes](numeric_modes.md).

## See also

- [Choosing an estimator](choosing_an_estimator.md)
- [Decomposition and embeddings](decomposition.md), to reduce dimensions before clustering
- API reference: [Clustering and mixture models](../api/clustering.rst)
