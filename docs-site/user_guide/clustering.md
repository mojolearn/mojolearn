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

## Single-cell workflows

The steps of the Scanpy and Seurat tutorials after normalization and PCA are graph steps, and each has a
Mojolearn counterpart. They all start from the k nearest neighbors of every cell, computed on the PCA
coordinates (`pcs` below, one row per cell).

```python
import numpy as np
import mojolearn

pcs = np.random.default_rng(0).normal(size=(5_000, 30)).astype(np.float32)
knn = mojolearn.NearestNeighbors(n_neighbors=20).fit(pcs).kneighbors(pcs, return_distance=False)
```

**Seurat** (`FindNeighbors`, then `FindClusters`): `shared_neighbors_graph` joins two cells when their
neighbor lists overlap, weighted by the Jaccard index of the two lists, and drops weights below
`prune_snn` (Seurat's `prune.SNN`, 1/15 by default). Each cell counts among its own neighbors, as
`kneighbors` on the training data returns it, so the result matches Seurat's `snn` graph. Cluster it with
`Louvain`, Seurat's default algorithm, or with `Leiden`:

```python
snn = mojolearn.shared_neighbors_graph(knn, prune_snn=1 / 15)   # CSR: snn.indptr, snn.indices, snn.data
clusters = mojolearn.Louvain(resolution=0.8).fit_predict(snn)
```

**Scanpy** (`pp.neighbors`, `tl.leiden`, `tl.diffmap`, `tl.paga`, `tl.umap`): cluster the neighbor graph
with `Leiden`, then summarize it. `DiffusionMap` returns Scanpy's `X_diffmap` (`embedding_`) and
`diffmap_evals` (`eigenvalues_`) from a symmetric weighted graph; the first component is the stationary
one, as in Scanpy. `PAGA` compares the edges between every two clusters with the number expected if
edges were random (Scanpy's `paga` model `'v1.2'`) and keeps the spanning tree of the strongest links:

```python
graph = snn   # or any symmetric connectivity matrix, such as UMAP's fuzzy graph
groups = mojolearn.Leiden(resolution=1.0).fit_predict(graph)

dm = mojolearn.manifold.DiffusionMap(n_comps=15).fit(graph)
print(dm.eigenvalues_[:5])

paga = mojolearn.PAGA().fit(graph, groups)
print(paga.connectivities_.shape, paga.connectivities_tree_.shape)
```

`paga.connectivities_` and `paga.connectivities_tree_` are sparse like Scanpy's, in the same CSR form as
`snn`. Scanpy counts PAGA's edges on the directed neighbor lists; pass those (each row's neighbors as an
`(indptr, indices, n)` tuple) to count them the same way.

The clusters are usually drawn on a UMAP map. Scanpy's `tl.umap` uses `min_dist=0.5`, while the `UMAP`
default is 0.1, so pass it to get the same spacing:

```python
embedding = mojolearn.UMAP(n_neighbors=15, min_dist=0.5).fit_transform(pcs)
```

Every step runs on the GPU and gives the same result on every device in `identical` mode. Partitions,
diffusion components and PAGA trees can differ from Scanpy's and Seurat's in tie-breaks and in the last
digits: the clustering moves nodes in a fixed parallel order, and the eigenvectors come from a seeded
solver in single precision.

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
