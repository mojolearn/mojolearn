# Kernel methods

GPU kernel primitives and estimators derived from cuML and cuVS. Provenance and unsupported
combinations are defined in `DERIVATION_MAP.tsv` and `NOT_IMPLEMENTED.tsv`.

```bash
pixi run check-kernel-methods
pixi run kernel-methods-main
```

Kernel dispatch, parameter validation, and fixed reduction order are part of the observable contract.
