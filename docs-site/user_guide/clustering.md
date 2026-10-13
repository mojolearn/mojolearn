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
