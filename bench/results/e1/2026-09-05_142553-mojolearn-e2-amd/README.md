# AMD integrated UMAP qualification, 2026-09-05

Frozen source e2cfe1ac. This source includes public CSR UMAP fitting and
unseen-sample transform. All 22 follow-up jobs passed, including fresh
metrics bindings, fit/transform API checks and both held-out quality cases
in FAST, DETERMINISTIC and IDENTICAL modes. The IDENTICAL retained training
and query inputs and embeddings match the Apple public-CSR capture exactly.
These are source-binding results, not installed Linux wheel qualification.

Native transform, sparse graph, independent dense-versus-sparse fit and
FAST/DETERMINISTIC sparse optimizer checks passed. Public k-NN legacy and
experimental dispatch checks passed with exactly equal retained outputs;
this run does not measure whole-request speed and does not enable the flag.

The corrected Mamba source API passed 102 checks. Native backward and
transformer forward/backward jobs passed; the transformer API passed its
44 checks while retaining the explicit independent corpus-oracle debt.
The original frozen certificates retain their separate scope.

One DigitalOcean AMD rental, CPU affinity limited to four cores. Droplet
598039434 was deleted after collection; GET returned HTTP 404 at 14:34:03 UTC.
Read source and binary manifests alongside each card.
