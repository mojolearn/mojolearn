# HDBSCAN

GPU HDBSCAN derived from cuML, cuVS, and RAFT.

The lane covers the graph, minimum-spanning-tree, hierarchy, condensed-tree, and labeling stages,
and, since 2026-09-15, cuML's prediction data and `approximate_predict` for held-out points
(`impl/prediction_data.mojo`, `impl/detail/predict.mojo`, the host restatement in
`host/hdbscan_host_oracle.mojo`), and cuML's soft clustering, `membership_vector` and
`all_points_membership_vectors` (`impl/detail/soft_clustering.mojo`), in float32 on every column
(DEVIATION 1616).
Supported and refused behavior is defined by `NOT_IMPLEMENTED.tsv`.
The checks, rather than old investigation prose, define current status.

## Verify

```bash
pixi run check-hdbscan
pixi run hdbscan-main
```

## Current focus

- expand fixtures that distinguish valid trees from accidentally convenient ones;
- preserve deterministic tie handling;
- collect complete Apple, AMD, and NVIDIA identity cards before broadening claims.
