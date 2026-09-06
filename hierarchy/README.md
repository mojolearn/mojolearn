# Hierarchical clustering

GPU linkage and hierarchy construction. `DERIVATION_MAP.tsv` records upstream provenance;
`NOT_IMPLEMENTED.tsv` records unsupported linkage and metric combinations.

```bash
pixi run check-linkage
pixi run linkage-main
```

Tie handling and merge ordering are contract behavior, not implementation details.
