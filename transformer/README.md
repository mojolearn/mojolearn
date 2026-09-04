# Transformer

GPU transformer forward and backward primitives with an explicit cross-vendor identity contract.

`IDENTICAL_TRANSFORMER_CONTRACT.md` is normative. `DERIVATION_MAP.tsv` and `NOT_IMPLEMENTED.tsv`
define provenance and scope; corpus fixtures live under `corpus/`.

```bash
tools/with_identical_mode.sh pixi run check-transformer
tools/with_identical_mode.sh pixi run check-transformer-backward
```

Optimized FAST kernels must remain isolated from the arithmetic schedule promised by IDENTICAL mode.
