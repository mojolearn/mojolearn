# Public source CSR integration, 2026-09-05

Public native fit delegates to the CSR estimator. The dense graph helper and
explicit dense validation composition remain independent.

Apple M4 clean metrics builds passed in FAST, DETERMINISTIC and IDENTICAL.
Each passed all 12 fit/transform Python test groups and both held-out transform
quality cases. Training and query embeddings in every mode match the earlier
dense public-fit quality records byte for byte within that mode. IDENTICAL
API tests retain the original and broader pinned fit-layout arrays.

These are source-built API checks, not a new published wheel. The previous
AMD campaign and current frozen NVIDIA campaign qualify the sparse native
candidate, not this later public integration. Fresh remote public integration
and installed-artifact qualification remain separate work.

Each quality JSON retains source hashes, binary SHA-256, mode and exact input
and output cells. Original pre-integration evidence is retained alongside this
record and is not relabeled as newer source.
