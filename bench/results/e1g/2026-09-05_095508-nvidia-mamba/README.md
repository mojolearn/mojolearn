# NVIDIA UMAP and public kNN qualification, 2026-09-05

Frozen source 72da212bbf639a2e6885ba02f3876d76edd67f60, RTX 4090.
All native Mamba cases and all 16 follow-up jobs passed. The pod was destroyed
and absence verified with HTTP 404. Artifacts were fetched before deletion.

Both UMAP held-out transform cases passed quality and their retained inputs,
training embeddings and query embeddings match Apple and AMD byte for byte.
Native transform cells also match; sparse estimator gates passed in IDENTICAL,
FAST (including the 1024-row optimizer threshold) and DETERMINISTIC.
Public UMAP fit/transform tests passed in IDENTICAL. This frozen source still
uses dense public fitting; later source commit 274ba7c0 integrates public CSR
and is qualified separately. Nothing here is a complete installed Linux wheel.

All 133,348 public kNN selected pairs match between legacy and experimental
IDENTICAL dispatch and match Apple. Thirty fixtures cover q1/257/1000, K8/10/16,
fallback K4/15, duplicate distances and changes of query tiling. The activation
headers confirm the flag was absent in one binary and present in the other.
No timing claim is made by this public dispatch correctness driver.

The source Mamba API passed again after the native certificate. Native backward
and Python inference/state checks remain distinct; no Python backward API is
claimed. Read commit/source/binary manifests with the retained raw records.
