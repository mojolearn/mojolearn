# GEMM

Cross-vendor FP32 matrix multiplication with separate FAST and IDENTICAL execution modes.

`IDENTICAL_FP32_CONTRACT.md` is the normative arithmetic and scheduling specification.
`DERIVATION_MAP.tsv` maps the implementation to upstream sources, and `NOT_IMPLEMENTED.tsv`
records intentionally unsupported behavior.

## Verify

```bash
pixi run check-gemm-identity
pixi run check-gemm-oracle
pixi run speed-gemm
```

IDENTICAL prioritizes reproducible output bits across supported GPUs. FAST may use a different
schedule but must be measurably faster on the benchmarked hardware; a speed claim without a
recorded fixture and repeated timings is not accepted.
