# Native classification metrics qualification

Verdict: PASS on the local Apple GPU in FAST, DETERMINISTIC, and IDENTICAL. This is correctness evidence, not a performance measurement or cross-vendor qualification.

`native-{mode}.build.log` and `native-{mode}.run.log` correspond to retained executables in `build/classification_metrics/`; `native-binaries.sha256` records their hashes and `native-toolchain.txt` records the compiler. Each binary prints its numeric mode before the final PASS marker. All compilation and GPU execution was serialized with `tools/with_build_lock.sh`.

The independent native check reconstructs integer confusion counts and Float64 precision/recall/F1 oracles from original pairs. It exercises every average and normalization, selected labels retaining outside false positives/negatives, excluded confusion labels, absent classes, zero division 0/1 and undefined flags, weighted zero-support fallback, direct F1, ragged launches and repeat bits. Synthetic GPU count fixtures cover counts above Float32 exactness and Int32 maximum, including widened F1 denominators and exact Int64 confusion output. Input and allocation bounds have negative tests.

The initial `mojo run` FAST check passed in `native-fast.log`. Two earlier compilation failures are preserved separately: `native-fast.initial-fixture-type-failure.log` (fixture scalar comparisons) and `native-fast.initial-generic-store-failure.log` (explicit generic output casts). These failures were corrected before all retained binaries were built. All native runs also emit `Context leak detected, CoreAnalytics returned false`; the diagnostic is retained, and each process exits successfully with its PASS marker.

Implementation limits: unweighted encoded single-label inputs, positive row count through Int32 maximum, dense confusion capped at 4096 classes, and Float32 normalized output. Integer numerators and denominators convert separately to Float32. PRF uses O(k) GPU counts and one GPU thread folds selected classes in ascending order; no CPU metric reduction or speed claim is made.
