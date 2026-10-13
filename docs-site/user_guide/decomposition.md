# Decomposition and embeddings

Decomposition methods rewrite the data in fewer columns. Use them to compress features before another
model, to remove noise, or to draw the data in two dimensions.

| Use | Estimators |
|---|---|
| Linear projections | `PCA`, `IncrementalPCA`, `TruncatedSVD` |
| Parts-based and sparse factors | `NMF`, `SparsePCA`, `DictionaryLearning`, `LatentDirichletAllocation` |
| Independent sources and latent factors | `FastICA`, `FactorAnalysis` |
| Random projections | `GaussianRandomProjection`, `SparseRandomProjection` |
| Two related sets of columns | `CCA`, `PLSRegression`, `PLSCanonical` |
| Embeddings for visualization | `UMAP`, `TSNE`, `Isomap`, `SpectralEmbedding`, `MDS` |

## PCA

```python
import numpy as np
import mojolearn

X = np.random.default_rng(0).random((10_000, 8), dtype=np.float32)

pca = mojolearn.PCA(n_components=3).fit(X)
Z = pca.transform(X)
X_back = pca.inverse_transform(Z)
print(pca.explained_variance_ratio_)
```

For data that does not fit in memory at once, `IncrementalPCA` fits batch by batch with `partial_fit`.
`TruncatedSVD` skips the centering step.

## Non-negative factors

```python
nmf = mojolearn.NMF(n_components=3, random_state=0)
W = nmf.fit_transform(X)
H = nmf.components_
```

## A 2-D map with UMAP

```python
umap = mojolearn.UMAP(n_neighbors=15, n_components=2).fit(X)
print(umap.embedding_.shape)        # (10000, 2)
Y = umap.transform(X[:10])          # place new rows on the same map
```

## Saving a model

`PCA`, `TruncatedSVD` and `UMAP` save to a file and load back:

```python
pca.save("pca.npz")
loaded = mojolearn.PCA.load("pca.npz")
```

## Devices and numeric modes

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. `identical` is the
default. On Apple GPUs you can choose `fast` instead, which is tuned for speed and does not promise
identical bits: call `mojolearn.set_numeric_mode()` with `"fast"`. See [Numeric modes](numeric_modes.md).

## See also

- [Choosing an estimator](choosing_an_estimator.md)
- [Clustering](clustering.md), [Preprocessing](preprocessing.md)
- API reference: [Decomposition and manifold learning](../api/decomposition.rst)
