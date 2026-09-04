# Kernel density estimation

GPU KernelDensity derived from cuML.

`DERIVATION_MAP.tsv` identifies the ported upstream paths and `NOT_IMPLEMENTED.tsv` lists kernels,
metrics, and options that are deliberately refused. `estimator.mojo` is the public Mojo surface.

## Verify

```bash
pixi run check-kde
```

Changes should test normalization, metric dispatch, bandwidth edge cases, and reduction order.
FAST-mode approximations must remain explicit and must not leak into IDENTICAL mode.
