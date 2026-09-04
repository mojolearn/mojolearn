# Cholesky

Dense GPU Cholesky primitives used by statistical estimators. Scope and provenance are recorded in
`DERIVATION_MAP.tsv` and `NOT_IMPLEMENTED.tsv`.

```bash
pixi run check-cholesky
pixi run cholesky-main
```

Diagonal checks, update order, and failure behavior for non-positive-definite inputs need explicit tests.
