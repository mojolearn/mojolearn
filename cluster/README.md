# Clustering

GPU clustering estimators, currently including k-means. The TSV ledgers define provenance and
unsupported behavior.

```bash
pixi run check-kmeans
pixi run check-kmeans-identity
```

Initialization, empty-cluster repair, assignment ties, and centroid reduction order require direct tests.
