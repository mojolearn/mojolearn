# Using AnnData

[AnnData](https://anndata.readthedocs.io) is the standard container of single-cell data in Python: a cells x
genes matrix `X` (often sparse), per-cell annotations in `obs`, per-gene annotations in `var`, cell
coordinates such as PCA and UMAP in `obsm`, and cell-to-cell graphs in `obsp`, saved together in one `.h5ad`
file. `mojolearn.adapters.anndata` runs Mojolearn estimators on an AnnData object with the function names,
parameter names and result slots of [Scanpy](https://scanpy.readthedocs.io), so a Scanpy workflow keeps its
shape and its downstream code finds every result where it expects it.

In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.

## Install

```bash
pip install "mojolearn[anndata]"
```

The extra installs anndata (with NumPy and pandas). Scanpy is not needed. `import mojolearn` never imports
anndata; `mojolearn.adapters.anndata` raises an `ImportError` naming the extra when it is missing.

## Read, compute, write

Reading and writing `.h5ad` files is anndata's own job:

```python
import anndata
from mojolearn.adapters.anndata import pp, tl

adata = anndata.read_h5ad("pbmc3k.h5ad")
pp.pca(adata, n_comps=50)
tl.umap(adata, n_neighbors=15)
tl.tsne(adata)
adata.write_h5ad("pbmc3k_mojolearn.h5ad")
```

## The functions

| Function | Mojolearn estimator | Reads | Writes |
| --- | --- | --- | --- |
| `pp.pca(adata, n_comps=50)` | `PCA` | `X` or `layer` | `obsm['X_pca']`, `varm['PCs']`, `uns['pca']` |
| `pp.scale(adata)` | `StandardScaler` | `X` or `layer` (dense) | `X` or `layer`, `var['mean']`, `var['std']` |
| `tl.umap(adata)` | `UMAP` | `obsm['X_pca']`, or `use_rep` | `obsm['X_umap']`, `uns['umap']` |
| `tl.tsne(adata)` | `TSNE` | `obsm['X_pca']`, or `use_rep` | `obsm['X_tsne']`, `uns['tsne']` |
| `tl.louvain(adata)` | `Louvain` | `obsp['connectivities']`, or `adjacency` | `obs['louvain']`, `uns['louvain']` |

Each function reads its input from the AnnData object, runs the estimator once, and writes the results; the
estimator's own page describes the algorithm and every parameter. The Scanpy parameters these functions take
keep their Scanpy names (`n_comps`, `use_rep`, `n_pcs`, `layer`, `key_added`, `neighbors_key`, `adjacency`,
`use_weights`, `copy`) and their Scanpy defaults. A Scanpy parameter a function does not take raises a
`TypeError` that names it.

- `key_added` writes the results under that key instead of the default one: `pp.pca(adata, key_added="raw_pca")`
  fills `obsm['raw_pca']`, `varm['raw_pca']` and `uns['raw_pca']`.
- `layer` reads `adata.layers[layer]` instead of `adata.X` (and `pp.scale` writes the scaled matrix back to the
  same layer).
- `tl.umap` takes `n_neighbors`, `metric`, `use_rep` and `n_pcs` from `uns['neighbors']['params']` when you
  leave them out, as Scanpy does. Mojolearn's UMAP builds its own exact neighbor graph from that
  representation. Its `min_dist` defaults to 0.5, Scanpy's choice for `tl.umap`; `mojolearn.UMAP` itself
  defaults to umap-learn's 0.1.
- `tl.louvain` clusters the graph in `obsp['connectivities']`, the graph Scanpy's `pp.neighbors` writes (or the
  `connectivities_key` that `uns[neighbors_key]` names, or the symmetric sparse matrix you pass as `adjacency`). With the default `use_weights=False` every edge counts as weight 1, as in Scanpy.
  The clusters are a categorical column with categories `'0'`, `'1'`, ..., numbered in order of each cluster's
  first cell.
- `pp.pca` always centers the data. To use only the highly variable genes, pass that subset:
  `pp.pca(adata[:, adata.var["highly_variable"]].copy())`.
- `pp.scale` divides by the population standard deviation, as scikit-learn's `StandardScaler`; Scanpy divides
  by the sample standard deviation (number of cells minus one).

## Data types

Mojolearn computes in float32. A float64 matrix is converted once when it is read; results are float32. `X`
may be a dense array or a SciPy sparse matrix; `pp.pca` accepts both, and `pp.scale` needs a dense matrix.

## Copies, views and backed files

The functions modify `adata` in place and return None. With `copy=True` they leave `adata` unchanged and
return a modified copy.

A view such as `adata[adata.obs["group"] == "B"]` is accepted with `copy=True`. To modify it in place, make it
an object of its own first: `subset = adata[mask].copy()`.

An object opened in backed mode (`anndata.read_h5ad(path, backed="r")`) is refused; load it with
`adata.to_memory()` first.
