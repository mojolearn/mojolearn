# PCA independent source review

Reviewed frozen source `187a672d708da28ca4f90d6cba7f4bb74b79de26`, containing
`e8958ae5e2` and `b2b211b86c`, against their baseline `290a6ae27e`.
This is a read-only product review. No compilation, runtime tests, timing,
cross-device identity or product changes were performed by this lane.

## Findings requiring coordinated repair

All three findings are inherited in unchanged numerical plumbing. They are
reachable from the new native randomized PCA descriptor and matter when calls
run concurrently with the GIL released. They are source defects, not observed
runtime failures. The integration owner received each finding; product edits
remain coordinated with the PCA/TSVD owner.

1. **PCA-R1: context first publication is unsynchronized.**
   `x_decomp/device.mojo:315` (`xd_ctx`) tests and assigns a shared
   `Optional[DeviceContext]` without a lock or publication flag. Concurrent
   first callers can create, overwrite and destroy contexts while another
   caller reads the same Optional. Reachability is
   `PCADeviceKernels.fit` → `pca_randomized_owned` → `DevRDev.alloc` →
   `pool_alloc` → `xd_ctx`. The root assigned the narrow accessor repair to
   the PCA/TSVD lane, using the existing synchronized `process_ctx` owner,
   an explicit root-zero plan and a fresh slot name.
2. **PCA-R2: resident pool metadata is unguarded.**
   `x_decomp/resident.mojo:124` (`pool_alloc`), `:150` (`pool_free`) and
   `:187` (`_ptr`) mutate/read the process-wide `X_DECOMP_POOL` Lists and
   counters without synchronization. Concurrent fits can race free-list
   pops, ID allocation and buffer-list reallocation while another operation
   reads a buffer. `DevRDev` forwards allocation, upload, numerical operations
   and download to this same pool. A context-publication repair alone does
   not fix these races.
3. **PCA-R3: pinned staging ownership is unguarded.**
   `x_decomp/device.mojo:1698` (`_xd_stage_ptr`) lazily creates one shared
   `HostBuffer` without synchronized publication. Independently, `:1735`
   (`_down`) shares that buffer across the complete DMA → synchronize → host
   copy interval without a lease. A second large Apple download can overwrite
   the stage before the first caller's host copy finishes. Randomized PCA
   reaches this through `DevRDev.download` → `dev_download_run` → `_down`.
   Both first creation and continued ownership require repair; retain the
   existing measured staged-transfer path.

## Reviewed contracts with no further concrete defect identified

- The five descriptor operations, request spans, address/parameter ordering
  and result words agree between `PCANativeDriver` and native dispatch.
- The sole `PCAQueryCache` registration is in
  `_mojolearn_runtime_decomp_host`; its guard attaches the existing type.
  Native dimensions, whitening, connection and generation replace the old
  Python cache tuple/overlay. Identity keys retain their Python owners.
- Generation and identity cache paths preserve the prior eligibility rules:
  mutable decoded scalar arrays are not cached, rebinding changes the key or
  generation, selected factory identity is checked, and fitted arrays are
  captured again per query. Whitening rescans mutable fitted values.
- Fit/query field admission, provider selection, width/layout checks,
  whitening checks and publication order match in the reviewed paths.
  Invalid changed factories take full field admission before provider refusal.
- Drivers are copied and provider images pinned before releasing the GIL.
  Inputs, fitted state, outputs and prefix-copy sources remain owned through
  native calls. Outer estimator and fold contexts use the synchronized
  `process_ctx`; the randomized transitive accessor is the exception above.
- Fold descriptors remain in the base image that owns the store and use
  `msel_fold_entry_context`. Exact/dense, randomized, TSQR, fused projection,
  whitening and resident-fold calls retain their existing numerical owners,
  flags, transfer routes and synchronization points.

Static byte comparisons found all 14 sampled numerical/context source files
unchanged. This does not prove concurrency safety, numerical correctness or
performance; the open findings demonstrate why unchanged kernels alone are
insufficient. Python remains in public admission, storage/image pins and output
publication. Legacy exported numerical entries remain available, while the PCA
public owner no longer dispatches through their Python callables.

Evidence: `~/mojolearn-evidence/python-mojo-migration-20261110/pca-independent-review/`.
`review-results.json` records exact revisions, file hashes and the three open
findings. `source/` preserves the frozen reviewed sources; `boundary.diff`,
`fit.diff`, `provider-registration.diff` and the before-files preserve the
comparison evidence. Compilation and runtime coverage remain the orchestrator's
work. No A/B define or bits change was introduced. Nothing was removed by this
review because product files were read-only.
