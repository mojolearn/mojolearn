# Source frozen from the qualified two-H100 pod

`multi-gpu-qualified-source.tgz` was copied directly from the completed
`rbtojh7e0esekh` working tree after the shared optimizer, byte-LM pool,
QR/full-PCA, wider Gram and pointwise batches. Its SHA256 identifies the exact
source used for the subsequent RTX 5090 builds, independently of git labels.

The initial `commit.txt` inside names `eaec62839`; later source overlays are
recorded in the preceding family receipts through `a8b158e91`. That marker
alone is not the final source identity. This archive contains those overlays.
It excludes installed pixi environments, compiled extensions, Python caches,
corpora and result archives. The enwik8 corpus is fetched separately from R2
and qualified by its recorded SHA256.

All H100 jobs were done and all evidence downloaded before termination.
`h100-termination.log` records DELETE 204 and GET 404. No local tests/builds.
