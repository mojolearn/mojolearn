# Neighbors

GPU exact, radius, and ball-cover neighbor search. Provenance and the deliberately unsupported
surface are in the two TSV ledgers.

```bash
pixi run check-knn-identity
pixi run check-ball-cover-knn
pixi run check-radius
pixi run check-metric
```

Distance evaluation, candidate ordering, and equal-distance ties are part of the output contract.
