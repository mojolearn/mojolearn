# Resampling

GPU resampling primitives with explicit deterministic and identity behavior. See
`DERIVATION_MAP.tsv` for provenance and `NOT_IMPLEMENTED.tsv` for refusals.

```bash
pixi run check-resample
pixi run resample-card
```

Random streams, index generation, and tie behavior must be tested independently of final aggregates.
