# Remaining work and execution plan

Updated September 6, 2026. This is the execution order for
[FEATURE_COMPLETION_PLAN.md](FEATURE_COMPLETION_PLAN.md); it preserves the
original feature, paper and validation requests.

## Execution rules

Root alone runs tests, builds, models, measurements and rentals. Subagents
only read, author and review source. Use two CPU cores/threads by default,
never more than three; execute one model/build/measurement job at a time.
Prefer remote NVIDIA/AMD. Necessary MacBook checks require explicit memory
and time bounds as well as thread caps. Every rental has a deletion watchdog,
and every failed/stopped attempt retains its original status and artifacts.

## 1. Publish the requested PyPI 0.6.0 alpha API release

- [x] Expose public `linalg`, `umap`, and `training` modules and existing
  optimizer/loss functions; retain public Mamba/Transformer and fixed-trainer
  exports with precise native availability limits.
- [x] Locate base macOS and AMD wheels and their evidence. Build explicit
  alpha API overlays with unchanged native/runtime bytes and new Python
  source hashes. Initial `0.6.0a1` candidates passed file checks and bounded
  macOS import checks; they are retained but are not the requested final
  version number and have not been published.
- [x] Reassemble as **0.6.0**, explicitly opting into the `alpha-api` release
  profile and keeping the Alpha classifier. Validate both exact final wheels
  and the manifest; do not inherit numerical qualification from base wheels.
- [x] Stage exact assets and manifest, run the existing Trusted Publisher
  workflow's explicit alpha path, and verify PyPI filenames and SHA256 values.
  Keep NVIDIA Linux source-build-only for this artifact. Publishing an API
  does not assert that every feature, shape or cross-vendor certificate passes.
- [x] Verify actual PyPI 0.6.0 filenames/SHA256 after workflow `34066704839`
  succeeded; update current installation/support wording. The first upload
  failed core-metadata validation before publication; repaired r2 bytes are
  the published artifacts, with original failure retained.

## 2. Finish bounded neural training and real checkpoint resume

Follow [the LM claim and acceptance plan](docs/LM_TRAINING_CLAIM_PLAN.md):
NVIDIA, then AMD, then Apple; actual continuation and per-step state, followed
by measured step cost. A three-vendor or priority claim requires its own proof.

- [x] NVIDIA small MLP: all 18 jobs and admission passed at `8f6ed41`, including
  independent reference/edge checks, 16 steps and complete same-device
  checkpoint continuation. [Evidence](bench/results/resume/2026-09-06-root-training-nvidia/README.md).
- [ ] Run the same frozen MLP numerical sources on AMD; compare complete
  per-step parameters, gradients, moments, flags, counters and loss bytes.
- [x] NVIDIA two-block, 34,944-parameter byte LM: all 12 jobs and fetched
  admission passed at `d921eade`, including the independent first-step FP64
  gradient/AdamW oracle and controls, followed by 128 real-text training steps.
  Held-out loss fell from 5.5412986 to 2.8436419 (ratio 0.51317).
  [Evidence](bench/results/resume/2026-09-06-root-byte-lm-nvidia/README.md).
  This closes single-vendor learning only; cross-vendor/resume remain open.
- [x] AMD run 3 passed all 12 jobs and fetched admission from identical
  numerical source at `d921eade`, with an explicit 1 GiB runtime pool. All
  128 raw steps, heldout batches and final checkpoints match NVIDIA. This is
  the continuous two-vendor result; full protocol admission awaits resume.
- [ ] Add the separately guarded Metal column after NVIDIA/AMD resume passes.
- [ ] Execute actual NVIDIA↔AMD checkpoint transfer/resume in fresh processes,
  comparing with continuous runs and requiring effective missing-moments
  controls. The native driver/comparator and
  [serial runner](training/TRAINING_RESUME_SERIAL.md) are authored, not yet run.
  Matching checkpoint files alone does not close resume behavior.

## 3. Close Mamba and symmetric-tree feature gaps

- [ ] After the active LM campaign, refresh the historical 209-case matrix
  sequentially using [REFUSAL_SEQUENTIAL_EXECUTION.md](docs/REFUSAL_SEQUENTIAL_EXECUTION.md)
  and [REFUSAL_PORT_PLAN.md](docs/REFUSAL_PORT_PLAN.md). The 27 old refusal rows
  are not 27 current missing features: several are implemented and others
  are invalid combinations or intentional algorithm refusals. Keep kd-tree
  unsupported; retain the two explicit CPU tree rows as host evidence.
- [ ] Validate additive PCA whitening binding exposure and bounded k-NN
  k=257–1024 selector extension after source review. Existing distance-weighted
  voting, Manhattan and cosine need current evidence, not duplicate ports.
  Do not add these later edits to the already frozen 0.6.0 artifacts silently.

- [ ] Validate installed Mamba1/2/3 forward and zero-state IDENTICAL Python
  backward on matching NVIDIA/AMD artifacts, including the shared B2/L8/D32
  shape. Existing native certificates and bounded NVIDIA Python results are
  separate evidence. [Public scope](mamba/PUBLIC_ALPHA_SURFACE.md).
- [ ] Add carried-state/state-cotangent backward and other missing requested
  modes through real native interfaces; these cannot be completed by exports.
- [ ] Reconcile the reported Apple Mamba3 long intermediate failure with its
  exact source and policy. Later NVIDIA/AMD long-profile compositional
  certificates do not silently recertify Apple or a direct FP64 intermediate
  claim. Preserve the paper's narrower claim until the relevant proof exists.
- [ ] Implement learning-to-rank objectives, grouping/pair contracts and their
  derivatives; then validate gradient, leaf and model behavior. Ordered RMSE
  is already implemented but is **not ranking**.
- [ ] Complete categorical/CTR and other CatBoost feature gaps from the
  [tree inventory](docs/TREE_ALPHA_FEATURE_STATUS.md). Do not call the entire
  symmetric-tree subsystem CatBoost-complete or universally bitwise validated.

## 4. UMAP and missing hardware evidence

- [x] Bounded NVIDIA digits neighborhood preservation and one matched cuML
  timing workload: all 15 jobs passed, including k=10 trustworthiness and
  retention against umap-learn. [Evidence](bench/results/resume/2026-09-06-root-umap-nvidia/README.md).
- [ ] Broaden declared real-data fixtures and perform matching NVIDIA/AMD
  IDENTICAL comparisons; one digits experiment is not general coverage.
- [x] Locate existing fitted ARIMA NVIDIA/AMD functional passes and distinguish
  them from filter-only or full-fitter identity claims.
- [ ] Close full ARIMA fitter identity and NVIDIA Holt-Winters, spectral and
  Gaussian-process identity cells using the
  [evidence inventory](docs/MISSING_VENDOR_EVIDENCE_QUEUE.md). Existing smoke
  and historical FAST timings do not fill full identity columns.

## 5. Paper claims and performance work

- [x] Place Table 6's k-NN/GEMV implementation-gap explanation beside the
  numbers and rebuild both paper variants.
- [ ] Optimize the two kernels without changing the declared numerical
  contract; follow [optimization queue](docs/KNN_GEMV_OPTIMIZATION_QUEUE.md).
- [ ] Root measures FAST and IDENTICAL against exactly one mathematically
  matched external NVIDIA implementation per workload. External tolerance
  correctness/performance and own cross-device bitwise equality are separate.
- [ ] Finish fourth-machine coverage and per-block source/provenance audit.
  Expand each claim only to the source, hardware, shapes and state actually
  checked; retain unmeasured cells and failing fixtures explicitly.

No item is complete merely because its source, test, driver or workflow exists.
Publication, functional execution, reference correctness, bitwise agreement,
resume and speed each require their own retained evidence.
