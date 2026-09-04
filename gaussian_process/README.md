# Gaussian process

Cross-vendor exact dense Gaussian-process regression in FP32.

The implementation, upstream mapping, and refused surface live in `estimator.mojo`,
`DERIVATION_MAP.tsv`, and `NOT_IMPLEMENTED.tsv`. The identity check covers factorization, solves,
prediction, and uncertainty output rather than accepting a final-value-only comparison.

## Verify

```bash
pixi run check-gaussian-process
pixi run gaussian-process-main
```

The principal constraints are stable Cholesky behavior, explicit ridge handling, and fixed
reduction order. Broader kernels or approximate GP methods should be separate lanes.
