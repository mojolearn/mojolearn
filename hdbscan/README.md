# HDBSCAN

GPU HDBSCAN derived from cuML, cuVS, and RAFT.

The lane covers the graph, minimum-spanning-tree, hierarchy, condensed-tree, and labeling stages.
Supported and refused behavior is defined by `DERIVATION_MAP.tsv` and `NOT_IMPLEMENTED.tsv`.
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
