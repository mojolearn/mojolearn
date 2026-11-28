# Single-cell analysis

Mojolearn covers the steps of a single-cell workflow after the count matrix is normalized and log-transformed:
the embedding, batch integration, clustering and marker genes. The functions follow Scanpy, harmonypy and Seurat,
so a tutorial written for those tools maps onto them step by step.

| Step | Mojolearn |
|---|---|
| Embedding | `PCA` |
| Batch integration | `Harmony` |
| Marker genes per cluster | `rank_genes_groups` |

## Batch integration with Harmony

`Harmony` removes batch effects from a cell embedding, usually the leading principal components. It clusters the
cells softly while favoring clusters that mix every batch, then corrects each cluster's batch offsets by a ridge
regression, and repeats until the objective stops falling. Build the neighbor graph, the clusters and the UMAP
from the corrected embedding.

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
X = np.log1p(rng.poisson(1.0, (3000, 2000))).astype(np.float32)
batch = rng.integers(0, 3, 3000)

pcs = mojolearn.PCA(n_components=50).fit_transform(X)
harmony = mojolearn.Harmony(random_state=0)
Z = harmony.fit_transform(pcs, batch)
print(Z.shape, harmony.n_iter_, harmony.converged_)
```

The parameters are harmonypy's: `theta` (the diversity penalty), `sigma` (the softness of the clusters), `lamb`
(the ridge penalty), `n_clusters` (by default `min(round(n_cells / 30), 100)`), `tau`, `block_size`,
`max_iter_harmony` and `max_iter_kmeans`. To correct several covariates at once, such as the sample and the
technology, pass one label array per covariate, with one `theta` and `lamb` each if you want them to differ:

```python
tech = rng.integers(0, 2, 3000)
Z = mojolearn.Harmony(theta=[2.0, 1.0]).fit_transform(pcs, [batch, tech])
```

The soft cluster assignments are in `R_` and the objective of each round in `objective_`.

## Marker genes

`rank_genes_groups` tests every gene for each group against the rest of the cells (or against one reference
group), as Scanpy's `tl.rank_genes_groups` and Seurat's `FindMarkers`. `method="wilcoxon"` is the Wilcoxon
rank-sum test and `method="t-test"` Welch's t-test. The input is cells x genes, dense or a sparse matrix in CSR
form (`indptr`, `indices`, `data`, `shape`), with log1p-transformed values.

```python
labels = mojolearn.KMeans(n_clusters=6, n_init=3).fit_predict(Z)
res = mojolearn.rank_genes_groups(X, labels, method="wilcoxon")

names = np.asarray(res["names"])
pvals_adj = np.asarray(res["pvals_adj"])
for i, group in enumerate(res["groups"]):
    print(group, names[i, :5], pvals_adj[i, :5])
```

The result is laid out as Scanpy's: one row per group, the genes sorted by score from the highest (a tie goes
to the lower gene index). `names` holds the gene (column) indices in that order and `scores`, `pvals`,
`pvals_adj` (Benjamini-Hochberg per group by default, `corr_method="bonferroni"` for Bonferroni) and
`logfoldchanges` the values beside them. With
`reference=<label>` each group is compared with that group's cells, and the reference itself has no row.
`tie_correct=True` applies the tie correction to the Wilcoxon variance. Every group needs at least two cells.

For a UMAP of single-cell data, pass `min_dist=0.5` to match Scanpy's `tl.umap`.
