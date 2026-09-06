# Holt-Winters

GPU Holt-Winters exponential smoothing derived from cuML.

The supported seasonal and initialization behavior is captured in `DERIVATION_MAP.tsv` and
`NOT_IMPLEMENTED.tsv`. This lane is maintained for compatibility even though upstream has marked
the algorithm for deprecation; expansion should be justified by a concrete user need.

## Verify

```bash
pixi run check-holtwinters
```

Identity work should focus on initialization, seasonal indexing, and fixed-order error reduction.
