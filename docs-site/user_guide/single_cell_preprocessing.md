# Single-cell preprocessing

The usual preprocessing of a single-cell count matrix (cells as rows, genes as columns) runs in Mojolearn in the
order of the Scanpy tutorial. Every step takes the count matrix as a dense array or as a sparse matrix in CSR form,
and the steps that keep zeros at zero return a sparse result.

| Scanpy | Mojolearn |
|---|---|
| `pp.normalize_total` | `TotalCountNormalizer` |
| `pp.log1p` | `Log1pTransformer` |
| `pp.highly_variable_genes` | `HighlyVariableGenes` |
| `pp.regress_out` | `RegressOut` |
| `pp.scale` | `ClippedStandardScaler` |

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
counts = rng.poisson(rng.gamma(0.5, 2.0, 2_000), (5_000, 2_000)).astype(np.float32)
batch = rng.integers(0, 2, 5_000)

# variable genes on raw counts (seurat_v3), per batch
hvg = mojolearn.HighlyVariableGenes(flavor="seurat_v3", n_top_genes=500).fit(counts, batch)

X = mojolearn.TotalCountNormalizer(target_sum=1e4).fit_transform(counts)
X = mojolearn.Log1pTransformer().fit_transform(X)
X = hvg.transform(X)

total = counts.sum(axis=1, keepdims=True)
X = mojolearn.RegressOut().fit_transform(X, total)
X = mojolearn.ClippedStandardScaler(max_value=10).fit_transform(X)
```

`HighlyVariableGenes` offers Scanpy's three recipes. `flavor="seurat"` and `flavor="cell_ranger"` expect
log-transformed data and rank genes by their dispersion within bins of similar mean; select by `n_top_genes` or
by the `min_mean`, `max_mean`, `min_disp` and `max_disp` cutoffs. `flavor="seurat_v3"` expects raw counts, fits the
variance against the mean with a local quadratic fit (`span`) and requires `n_top_genes`. Pass each cell's batch
as `y` to score the batches separately and combine them, as Scanpy's `batch_key` does. The fitted attributes
(`highly_variable_`, `means_`, `dispersions_norm_`, `variances_norm_`, `highly_variable_nbatches_`, ...) hold
what Scanpy writes to `adata.var`.

`TotalCountNormalizer(exclude_highly_expressed=True, max_fraction=0.05)` leaves genes that dominate some cell
out of the totals, as Scanpy does; `columns=` gives the genes to count yourself.

`RegressOut` takes the covariates of each cell as `y`, an array of shape (cells, covariates), for example the total
counts and the fraction of mitochondrial reads. Pass categorical covariates as indicator columns.

`ClippedStandardScaler` scales with the sample standard deviation, as Scanpy's `pp.scale` does; `mean_` and
`std_` are Scanpy's `var['mean']` and `var['std']`. With `zero_center=False` a sparse matrix stays sparse.

For the next steps (the embedding, the neighbor graph, clustering and marker genes) see
[clustering](clustering.md). For UMAP in a single-cell workflow pass `min_dist=0.5`, the value Scanpy uses.
