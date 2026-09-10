# Resampling

GPU resampling primitives with explicit deterministic and identity behavior. See
`NOT_IMPLEMENTED.tsv` for refusals.

```bash
pixi run check-resample
pixi run resample-card
```

Random streams, index generation, and tie behavior must be tested independently of final aggregates.
