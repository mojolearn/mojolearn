# Current parallel GMM and UMAP references

These original columns were captured from the installed 0.8.25 candidate at source `9ab5d5269f60667efbe3da56307edc4f84261601`, using the admitted `sm_89` native payload from `3dded3bd47b9db2928ed3b7919f502b20c576e9c`. The AMD GMM column retains its own earlier recorded source and native provenance. Files are copied byte-for-byte; `raw-inventory.json` records their original locations and SHA-256 values. No columns were combined, relabelled, or rewritten.

Both parallel lanes passed all nine fixtures on one and two physical NVIDIA devices. Each column records all eight applicable parts, and each two-device column has its actual hardware witness. This is physical NVIDIA evidence, not a cross-vendor multi-GPU claim.

Reference admission uses the ordinary strict `build_table` and `merge_reference_lanes` functions. GMM is corroborated by the original current-revision AMD column and the fresh one-device NVIDIA columns. Parallel UMAP has a fresh two-repeat one-device NVIDIA column for every fixture. Every repeated part is stable and matches the separately captured one-device and physical two-device results. Two-device columns are retained as physical evidence and are not presented to the builder as ordinary one-device reference columns.

The stale GMM table predated `classic-kmeanspp-init-1`: its four numeric parts change on all nine fixtures, and four explicit property declarations are added per fixture. The parallel UMAP table predated `save-option-extras-contract-2`, which includes saved-model option metadata. Current references replace those stale contracts. Exact values and old-to-new device-class coverage are in `admission-proof.json`.

Current GMM table evidence is AMD and NVIDIA; current parallel UMAP table evidence is repeated NVIDIA only. Stale Apple/AMD class claims are not carried forward without qualifying current-revision records. Historical raw files and records remain preserved. Ordinary UMAP and all unrelated table cells are unchanged. No model algorithm, native code, fixture, or hash computation changes in this admission.
