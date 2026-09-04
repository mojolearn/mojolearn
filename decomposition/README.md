# Decomposition

GPU PCA and dense SVD primitives derived from RAFT.

The implementation uses explicit eigensolver and sign conventions so equivalent mathematical
factorizations do not masquerade as bitwise-identical outputs. `DERIVATION_MAP.tsv` records
provenance and `NOT_IMPLEMENTED.tsv` defines the supported boundary.

## Verify

```bash
mojo run -I . decomposition/pca_main.mojo
mojo run -I . decomposition/pca_wide_main.mojo
mojo run -I . decomposition/svd_full_main.mojo
```

Wide matrices, repeated singular values, and sign canonicalization are the highest-value fixtures.
