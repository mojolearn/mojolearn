# Lane c-cluster (classical family)

Read `docs/plans/cpu-gpu-cleanup/COMMON_BRIEF.md` and `docs/plans/HOST_ROUTE_REMOVAL.md` first, then `CLAUDE.md`.

Worklist: `docs/plans/cpu-gpu-cleanup/rows/c-cluster.tsv` (74 rows). Columns: rule, class, owner, state, path, occ, text.
`text` is the offending line; `occ` is its occurrence index among identical lines in that file.

Files you own (prefixes): `cluster/`, `hdbscan/`, `dbscan/`, `hierarchy/`, `mixture/`, `spectral/`, `umap/`, `embedding/`, `x_cluster/`, `bindings/_mojolearn_hdbscan.mojo`, `bindings/_mojolearn_mixture.mojo`, `bindings/_mojolearn_x_cluster.mojo`, `bindings/_mojolearn_embedding.mojo`, `python/mojolearn/_expansion_cluster.py`, `bindings/_mojolearn_umap`, `bindings/_mojolearn_spectral`, `bindings/_mojolearn_cluster`

For every row: make the GPU path do that work on the device, with a fixed fold order and no serial chain, and delete the host route.
Or, where the row is CPU-only-install code that a GPU install imports by mistake, move the import behind the CPU-only path.
Clear as many rows as you can, working from the biggest problems down.
