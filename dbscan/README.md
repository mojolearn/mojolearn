# DBSCAN

GPU density clustering with FAST and IDENTICAL execution modes. `DERIVATION_MAP.tsv` records
provenance and `NOT_IMPLEMENTED.tsv` prevents silent compatibility claims.

```bash
pixi run check-dbscan
pixi run check-dbscan-identity
```

Neighborhood construction and label canonicalization are tested separately. FAST must outperform
IDENTICAL on its declared benchmark fixture.
