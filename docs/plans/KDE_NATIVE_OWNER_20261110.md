# KDE native requests and ownership

The KernelDensity public fit/score boundary now unpacks into native requests and calls the estimator image's fixed native descriptor. This is the KDE estimator provider, not the separate `XnDriver` image. The numerical implementations and switch defaults are unchanged.

Dependency order: registry/workspace repair `8a624987ef`; native protocol/providers `f052c4918c`; exported factories and native state `78be03c73b`; then this public switch. Only these commits belong in the integration handoff, not earlier adoption merges.

## Native execution and retained behavior

- `core/kde_native_api.mojo` defines the seven-operation v1 wire protocol, typed preparation/direct-query/resident-query requests and exact integer handles. Names use length-delimited UTF-8 spans, including embedded NULs. Float64 score totals use owned native result storage.
- `core/kde_native_host.mojo` calls the existing host score executor. `core/kde_native_device.mojo` preserves the existing direct and resident kernels. The original Float32 bandwidth rounding, weight copy, finite/status checks, score folds and transfer routes remain.
- `core/kde_native_state.mojo` holds native fit identity, dimensions, names, width, owner/provider identity, take-once handle state and admission metadata words.
- `bindings/kde_native_boundary.mojo` retains only image/source lifetime pins around the native state. An active query pins its resident owner through GIL-free execution. Replacement or explicit release drops the model's owner; idle handles release immediately and active handles release after their final query pin. Teardown remains quiet, including interpreter finalization. Native owner allocation failure retains cleanup ownership.
- `bindings/neighbors_kde_native.mojo` retains public admission/publication order, sparse preparation, borrowed-input finite-copy, immutable-fit switch behavior, read-only snapshots, weight snapshots, lazy resident upload and public `_x`/`_w`/setting rebinds. The dense preparation envelope is a native boundary struct. Native output layouts replace shape tuples constructed solely to allocate outputs.
- Both old binding surfaces now unpack into the same native provider implementations. Compatibility exports remain, while their duplicate execution bodies and weight-copy plumbing are removed.

Registry metadata locking and workspace leases are separately documented in `KDE_RESIDENT_OWNERSHIP_20261110.md`. Ordinary sequential pooled reuse retains its allocation/reuse counters. Concurrent active calls can receive independent pooled scratch instead of the previous contention temporary-buffer path; this intentional overlap allocation change is **unmeasured**. No lock spans whole numerical operations. The separate reproduced Modular shared-context submission crash is unresolved; this migration adds no workaround or concurrency-success claim.

## Admission and Python boundary

`bindings/kde_admission_boundary.mojo` replaces per-call Python identity-key lists with the existing weak native stamped-record registry, native content records and native array-metadata words. It tracks nested metric-parameter mutation, model class, address, rank/dimensions, dtype identity and C-contiguity. Missing inherited constructor fields disable caching as before. The cache does not pin a resident owner, so cached admission cannot delay explicit device release. Records are stamped before execution; public state changed while the GIL is released cannot be blessed without admission.

The density lifecycle's KDE branch accepts the opaque native owner and keeps pickle stripping/restoration and explicit release. DBSCAN execution is untouched. The single runtime-family host image registers the resident and admission types with the existing lookup/add/attach pattern. KernelDensity's obsolete import-time tuple-finalizer registration is removed; the native resident owner performs cleanup.

Invalid ndarray-valued diagnostics use `<native scalar array>` rather than invoking NumPy's configured Python formatting hooks. Primitive scalar diagnostic text remains. Exact native scalar admission precedes conversions/comparisons. Public namespace keys are re-admitted after preparation/teardown GIL windows before publication or additional field reads.

Python still appears in UNPACK/PACK files: public attribute values, array/storage pins, import-connected schema/provider tables, original/shared public argument tuples, pickle dictionaries and result boxing. The shared neighbor lifecycle and density constructor envelope still stage boundary metadata. This is not a claim that every family or boundary file is Python-free. Native KDE requests, resident execution/state and the numerical RUN closure contain no Python objects/imports.

## Source evidence and checks owed

Source-only audit: `~/mojolearn-evidence/python-mojo-migration-20261110/kde-native-owner/source-audit.json` records **142 native transitive sources, zero Python imports**, and **24 unchanged KDE/host executor Mojo sources** relative to the independently reviewed registry repair `8a624987ef`. The earlier registry repair is intentionally outside that unchanged-source comparison. Full originals, patches and normal hook logs remain beside the report.

`python/mojolearn/tests/test_kde_native_boundary.py` adds **11 definitions / 22 expanded cases** covering legacy-entry bypass, descriptor refusal, rebound scalar/dictionary callback traps, formatter refusal, copied fit storage, mutable public owners/settings, pickle/release/reuse and model lifetime. AST parsing and `git diff --check` succeeded. **No lane compilation, runtime tests, timing or cross-vendor identity were run.** The native-test lane owns pure contracts and the original native numerical/ownership workloads.

Compile scope: `bindings/_mojolearn_runtime_family_host.mojo`, `bindings/_mojolearn_estimators_host.mojo`, `bindings/_mojolearn_estimators.mojo` and their standard target/switch variants. Existing native KDE workload handoff: original six-kernel × four-metric × weighted/unweighted cases (37×5 training/11×5 query); sequential workspace fixture 131×7 training, 9 then 3 queries, weighted Gaussian/Euclidean, width 0.25, allocations 4/reuse 1. Parent orchestrates the combined compile and native suites.

Removed: resident/key Python tuples; KDE prepared-key Python lists and equality fallback; dense-result tuple; per-query option lists and compiled-callable dispatch; duplicate legacy execution/copy bodies; obsolete KDE tuple finalizer. No numerical kernel, tuned route, switch or supported public method removed. No new A/B define. No intended algorithm/bit change; runtime behavior, performance and release identity remain separate checks owed to the orchestrator.
