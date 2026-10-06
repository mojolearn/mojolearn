# I15 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual public exact neighbor cases, fused cross-block stable top-k, tie geometry invariance and certified bound-compaction select-min/max arms with runtime reach witnesses.

Remaining original-card scope: The existing Apple-only coarse MMA → canonical exact rescore → per-query certificate → compacted exact fallback driver now has actual public AUTO-vs-TILED witnesses. Separated neighbors must certify at least one query, excluded duplicate ties must fall back every query, and cancellation fixtures compare every exact word. An opt-in metadata counter uses the existing fallback-count readback, adding no device work. NVIDIA/AMD coarse MMA is explicitly unsupported; their exact public outputs remain checked, without a coarse-path claim. Whole-query timing and fallback-rate qualification are owed.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.
