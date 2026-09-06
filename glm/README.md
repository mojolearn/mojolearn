# Generalized linear models

GPU linear-model primitives and estimators, currently centered on ordinary least squares. The TSV
ledgers define upstream derivation and unsupported behavior.

```bash
mojo run -I . glm/ols_main.mojo
pixi run check-linalg-identity
```

Solver choice and reduction schedule must be explicit whenever bitwise identity is claimed.
