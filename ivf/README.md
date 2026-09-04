# IVF

GPU inverted-file indexing and search derived from cuVS. `DERIVATION_MAP.tsv` and
`NOT_IMPLEMENTED.tsv` define provenance and scope.

```bash
pixi run check-ivf
pixi run ivf-card
```

Training assignment, probe ordering, candidate ties, and top-k selection are identity-sensitive.
