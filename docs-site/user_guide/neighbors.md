# Neighbors and vector search

Neighbor methods answer "which stored rows are closest to this one?". Exact search compares a query with
every stored row. Approximate indexes answer large searches much faster by looking at a part of the data.

| Use | Estimators |
|---|---|
| Exact nearest neighbors | `NearestNeighbors`, `RadiusNeighbors` |
| Classification and regression by neighbors | `KNeighborsClassifier`, `KNeighborsRegressor`, `NearestCentroid` |
| Approximate vector search | `IVFIndex`, `IVFPQIndex`, `IVFSQIndex`, `IVFRaBitQIndex`, `CagraIndex` |
| Outlier scores from neighbors | `LocalOutlierFactor` |
| Labels spread over a neighbor graph | `LabelPropagation`, `LabelSpreading` |

## Exact neighbors

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
X = rng.random((10_000, 8), dtype=np.float32)

nn = mojolearn.NearestNeighbors(n_neighbors=5).fit(X)
distances, indices = nn.kneighbors(X[:10])
```

Neighbors come back nearest first; ties in distance go to the lower index.

## Classification by neighbors

```python
y = (X[:, 0] > 0.5).astype(np.int32)
clf = mojolearn.KNeighborsClassifier(n_neighbors=5).fit(X, y)
labels = clf.predict(X[:10])
```

## Approximate search

`IVFIndex` splits the stored rows into `n_lists` groups and searches the `n_probes` groups closest to
each query. `CagraIndex` searches a graph of neighbors. Raising `n_probes` or `itopk_size` finds more of
the true neighbors at the cost of time.

```python
index = mojolearn.IVFIndex(n_lists=16, n_probes=4, n_neighbors=5).fit(X)
distances, indices = index.search(X[:10])

graph = mojolearn.CagraIndex(n_neighbors=5).fit(X)
distances, indices = graph.search(X[:10])
```

`IVFIndex.extend` adds rows to a fitted index without refitting it.

## Saving an index

`NearestNeighbors`, `RadiusNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor` and every index
save to a file and load back:

```python
index.save("ivf.npz")
loaded = mojolearn.IVFIndex.load("ivf.npz")
```

## Devices and numeric modes

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. `identical` is the
default. On Apple GPUs you can choose `fast` instead, which is tuned for speed and does not promise
identical bits: call `mojolearn.set_numeric_mode()` with `"fast"`. See [Numeric modes](numeric_modes.md).

## See also

- [Choosing an estimator](choosing_an_estimator.md)
- API reference: [Neighbors and vector search](../api/neighbors.rst)
