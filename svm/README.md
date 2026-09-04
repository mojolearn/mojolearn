# SVM

Dense FP32 binary C-SVC and epsilon-SVR derived from cuML's SMO solver and cuVS kernel matrices.

`DERIVATION_MAP.tsv` maps the port to upstream code. `NOT_IMPLEMENTED.tsv` defines unsupported
multiclass, sparse, kernel, and parameter combinations. Unsupported behavior must fail clearly
rather than silently selecting a different algorithm.

## Verify

```bash
pixi run check-svm
pixi run check-svm-oracle
```

Identity depends on working-set selection, tie handling, kernel evaluation, and reduction order.
Performance changes must preserve those choices in IDENTICAL mode.
