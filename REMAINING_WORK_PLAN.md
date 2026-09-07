# Remaining work and execution plan

**Controlling campaign plan:** [Identity and NVIDIA comparison](docs/IDENTITY_AND_NVIDIA_COMPARISON_EXECUTION_PLAN.md). New performance runs use ONLY our IDENTICAL mode against incumbent FAST and DETERMINISTIC. No our-FAST column or worker. Small M4/NVIDIA/AMD identity coverage is a separate exhaustive evidence audit and gap campaign. This supersedes older timing instructions below.

## September 7 update: Apple continuous training completed

Root built the Metal tiny byte-LM and ran all128 real-text training steps.
Complete raw states, held-out bytes and final checkpoint match NVIDIA and
DigitalOcean AMD bit for bit. Loss fell from5.5413 to2.8436 on all three.
Evidence: `bench/results/resume/2026-09-07-root-byte-lm-three-vendor/README.md`.
The user explicitly removed the fixed free-memory minimum; the separately
recorded user-tiny policy retained two-thread limits, a2GiB RSS cap, pressure,
swap/compression, CPU and deadline stops, watchdog and verified cleanup.
The earlier blocked/readiness sections below are historical, superseded for
continuous Metal training. Metal resume, separate MLP Metal, and installed
byte-LM wheel coverage remain open. No further paper work was performed.


Updated September 7, 2026. This is the execution order for
[FEATURE_COMPLETION_PLAN.md](FEATURE_COMPLETION_PLAN.md); it preserves the
original feature, paper and validation requests.

## Execution rules

Root alone runs tests, builds, models, measurements and rentals. Subagents
only read, author and review source. Use two CPU cores/threads by default,
never more than three; execute one model/build/measurement job at a time.
Prefer remote NVIDIA/AMD. Necessary MacBook checks require explicit memory
and time bounds as well as thread caps. Every rental has a deletion watchdog,
and every failed/stopped attempt retains its original status and artifacts.

## Current execution order — explicit DigitalOcean and Apple follow-up

**Latest priority:** focus on mojolearn; no further paper work. Root prepared
and attempted the Apple tiny-LM first-step entrypoint. Its 258-file source
check passed, but memory admission stopped execution before compiler/runtime
probes or GPU launch (167 MiB free/speculative at the attempt, 4 GiB required).
[Retained attempt](bench/results/resume/2026-09-07-root-apple-followup/first-step-attempt1/README.md).
Retry with a new output after unused applications are closed; do not lower
the guard. First step, full128 continuous comparison, then Metal resume remain
distinct gates. Remote native jobs must not overlap the Apple job.

1. **Completed:** DigitalOcean AMD MI325X run6 passed all 12 jobs, independent
   gradient/AdamW verification, 128 real-text training steps and matching
   head64 state. Loss fell from 5.5413 to 2.8436. Root validated fetched bytes
   and verified deletion. [Evidence](bench/results/resume/2026-09-07-root-byte-lm-do-amd/README.md).
2. Add Apple build/readback and one-step checks only after actual memory
   admission. The latest real supervisor check refused before starting even
   its harmless Python child; no Metal kernel has run.
   [Retained readiness evidence](bench/results/resume/2026-09-07-root-apple-followup/guard-readiness.json).
3. **Completed:** NVIDIA RunPod RTX4090 run2 passed all twelve jobs at
   `eac39c36`. Root compared its complete 128-step raw trajectory with DO AMD
   run6: all state, held-out tokens/losses and final checkpoint bytes agree;
   both independent FP64 gradient/AdamW oracles passed. The common inventory
   contains 258 files, including 45 transitive Mamba files. Both rentals were
   deleted and verified absent. [Expanded-source result](bench/results/resume/2026-09-07-root-byte-lm-expanded-comparison/README.md).
   Actual cross-vendor resume on this expanded-source round and Metal remain
   unrun; the older bidirectional-resume proof remains separate.
4. Finish actual Metal trajectory/checkpoint evidence, then the separate MLP
   Apple leg when local memory admission permits. While Apple is blocked,
   continue remote NVIDIA wheel delivery and the feature/evidence queue;
   do not weaken the MacBook guard to unblock it.

## 1. Published PyPI 0.6.0 alpha API release

### Authorized next release: 0.6.1 combined Linux wheel

[Execution plan](docs/RELEASE_0_6_1_EXECUTION_PLAN.md). Publication is authorized;
0.6.0 remains the currently published release until final 0.6.1 verification.

- [x] Author explicit three-set packer profile: CUDA sm_89, CUDA sm_90,
  HIP gfx942, 135 standard extensions and per-architecture build provenance.
  Root passed three bounded packer fixture tests; no native build implied.
- [x] Author per-architecture installed qualification and release admission;
  root's bounded file fixtures pass. Hardware qualification remains below.
- [ ] Freeze 0.6.1 source; build all three sets serially with two cores.
- [ ] Resolve any remaining Mamba FAST/DETERMINISTIC accuracy failure without
  changing tolerances to obtain a pass; retain historical failed attempts.
- [ ] Assemble/audit one final wheel and measure its actual compressed size.
- [ ] Run all 24 installed jobs on actual sm_89, sm_90 and gfx942 against
  the same final wheel hash. Two rentals suffice only if NVIDIA access covers
  both architectures; compiling an architecture is not runtime qualification.
- [ ] Publish 0.6.1 and verify PyPI file hashes, then update installation text.
  The separate byte-LM native extension is explicitly absent from this initial
  packer profile; its installed availability remains a coordinated follow-up.

DigitalOcean follow-up: run1 refused before dependency/model launch because
the render node lacked direct vendor identity; deletion was verified by HTTP
404. Root passed 25 guard tests plus six subtests for PCI-ancestor discovery.
Run2 refused locally before any API call because its token argument was missing.
Run3 used the corrected guarded transport at `e0bee4ac`, with all 213 numerical
source files matching the common Metal snapshot. It also refused before setup:
retained topology identifies seven AMD XCP platform placeholders beside the
physical AMD GPU. Deletion is verified HTTP 404. Add narrowly recognized XCP
handling and test the recorded topology before another attempt. Keep its outcome separate
from the already qualified RunPod NVIDIA/AMD training result.

Run4 passed guarded AMD device admission, pinned ROCm/PyTorch installation and
the actual MI325X GPU preflight. It stopped before model work because the Pixi
installer request returned an HTTP redirect page. Artifacts were fetched and
deletion verified HTTP 404. Run5 retains the installer, follows HTTPS redirects,
checks shell syntax and executes it under the AMD memory/time guard. NVIDIA's
prepared orchestration revision `34fb352f` additionally guards locked dependency
installation and stops on failure; root passed 50 bounded rehearsal checks and
a mocked failure-path check. No new training result is inferred from bootstrap.

Run5 subsequently reached compilation and exposed missing transitive Mamba
sources in the trimmed upload. Run6 at `eac39c36` includes all 45 Mamba Mojo
files and passed all 12 jobs, fetched admission and head64 verification.
Its expanded 258-file capture inventory matches prepared Metal `45cc2d9d`.
Root also verified the historical Mamba bytes against both old continuous
archives, all six full inventories and the retained resume snapshot; original
213-file capture records remain unchanged. The matching NVIDIA refresh and
continuous128 raw comparison now passed at `eac39c36`; held-out loss is
5.5412986 → 2.8436419 on both devices. See the
[NVIDIA campaign](bench/results/resume/2026-09-07-root-byte-lm-nvidia-common/README.md)
and [expanded-source comparison](bench/results/resume/2026-09-07-root-byte-lm-expanded-comparison/README.md).
New expanded-source resume and Metal are not admitted. PyPI 0.6.1 still awaits
actual native builds and installed qualification.

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
  [Actual index verification](bench/results/releases/2026-09-06-alpha-api/pypi-0.6.0-verification.json).
- [ ] Close NVIDIA binary installation in a new artifact release using the
  [NVIDIA wheel closure plan](docs/NVIDIA_WHEEL_CLOSURE_PLAN.md): include CUDA
  and HIP payloads, explicitly inventory the byte-LM extension, then run the
  installed checks on the exact final wheel sequentially. Do not replace
  published 0.6.0 bytes or infer qualification from source-only passes.

## 2. Finish bounded neural training and real checkpoint resume

[Current neural-training status](docs/NEURAL_TRAINING_STATUS.md) answers the
AMD, separate neural network, MacBook and NVIDIA-wheel questions. Root's
September 7 Mac telemetry check was readable, but the 4 GiB launch reserve
was absent: no local native compile or model launch is admitted yet.

Follow [the LM claim and acceptance plan](docs/LM_TRAINING_CLAIM_PLAN.md):
NVIDIA, then AMD, then Apple; actual continuation and per-step state, followed
by measured step cost. A three-vendor or priority claim requires its own proof.

- [x] NVIDIA small MLP: all 18 jobs and admission passed at `8f6ed41`, including
  independent reference/edge checks, 16 steps and complete same-device
  checkpoint continuation. [Evidence](bench/results/resume/2026-09-06-root-training-nvidia/README.md).
- [x] Run the same frozen MLP numerical sources on AMD; compare complete
  per-step parameters, gradients, moments, flags, counters and loss bytes.
  All 18 AMD jobs and admission passed; all 16 raw steps match NVIDIA at
  `8f6ed41`. [Retained comparison and teardown](bench/results/resume/2026-09-07-root-mlp-amd/README.md).
  MLP Metal and foreign-vendor MLP resume remain separate.
- [x] NVIDIA two-block, 34,944-parameter byte LM: all 12 jobs and fetched
  admission passed at `d921eade`, including the independent first-step FP64
  gradient/AdamW oracle and controls, followed by 128 real-text training steps.
  Held-out loss fell from 5.5412986 to 2.8436419 (ratio 0.51317).
  [Evidence](bench/results/resume/2026-09-06-root-byte-lm-nvidia/README.md).
  This closes the NVIDIA learning/oracle result; AMD continuous agreement is
  recorded below, along with separately admitted bidirectional resume.
- [x] AMD run 3 passed all 12 jobs and fetched admission from identical
  numerical source at `d921eade`, with an explicit 1 GiB runtime pool. All
  128 raw steps, heldout batches and final checkpoints match NVIDIA. This is
  the continuous two-vendor result; both resume directions are separately admitted below.
  [Raw comparison evidence](bench/results/resume/2026-09-06-root-byte-lm-cross-vendor/README.md).
- [x] Refresh NVIDIA/DO AMD on common `eac39c36` with the expanded 258-file
  inventory; admit both twelve-job campaigns and full continuous128 equality.
- [ ] Execute new expanded-source cross-vendor checkpoint continuation and
  bounded Metal build/readback, one-step, resume and 128-step comparisons.
  Preserve earlier qualified results and the supplemental transitive audit.
  Darwin build/vendor and immutable-byte capture source is now authored;
  compilation, Metal execution and three-vendor admission remain open.
  Root passed 88 host-only tests plus seven subtests after fixing Darwin
  temporary-directory fixtures. The 06:28 UTC memory recheck still refuses
  a native launch; user has been asked to free memory, not to reauthorize work.
  Common snapshot `d17c1aa1` (following initial `59e35532`) and both source
  archives are retained in the
  [Metal preparation record](bench/results/resume/2026-09-07-root-metal-preparation/README.md).
  Optional Metal comparison passed 38 focused tests plus seven subtests;
  both retained NVIDIA/AMD raw resume proofs still admit under the updated
  comparator. No Metal model has executed.
  The newer common-source LM result uses DigitalOcean MI325X and RunPod
  RTX4090. The earlier LM bidirectional-resume and classifier results used
  RunPod; their source and provider scopes remain separate.
- [x] NVIDIA → AMD actual checkpoint continuation passed the final all-raw
  comparator, including both independent oracles and effective missing-moments
  control. [Qualified evidence](bench/results/resume/2026-09-06-root-byte-lm-cross-vendor/README.md).
- [x] AMD → NVIDIA separately captured head/resume/control passed the same
  all-raw comparator. Bidirectional checkpoint continuation is admitted for
  this bounded real-text fixture. Metal remains separate.

## 3. Close Mamba and symmetric-tree feature gaps

The historical 209-case certificate remains **180 + 2 + 27**. Root compared
all refusal IDs, settings and exact errors across retained round-11 NVIDIA/AMD
and round-13 NVIDIA/AMD/Apple records: they match. The paper audit is now bound
to round 13 and corrected to **10 intentional-at-audit / 10 source-identified /
7 remaining-work** cells. This is source classification, not new certification.
[Raw provenance audit](bench/results/audits/2026-09-06-refusal-round-binding.json).
Both paper variants rebuilt successfully after the correction.

- [ ] After the active LM campaign, refresh the historical 209-case matrix
  sequentially using [REFUSAL_SEQUENTIAL_EXECUTION.md](docs/REFUSAL_SEQUENTIAL_EXECUTION.md)
  and [REFUSAL_PORT_PLAN.md](docs/REFUSAL_PORT_PLAN.md). The 27 old refusal rows
  are not 27 current missing features: several are implemented and others
  are invalid combinations or intentional algorithm refusals. Keep kd-tree
  unsupported; retain the two explicit CPU tree rows as host evidence.
- [ ] Build and validate the newly authored public PCA `full` binding export;
  source and host dispatch gates passed; no native build/numerical test has run. Preserve the distinction
  between an existing native kernel and a callable public route.
- [ ] Validate Manhattan DBSCAN through its supported `brute` route. The
  historical default `rbc` combination still refuses; the brute implementation
  does not retroactively turn that old refusal row into a pass.
- [ ] Validate additive PCA whitening binding exposure and bounded k-NN
  k=257–1024 selector extension after source review. Existing distance-weighted
  voting, Manhattan and cosine need current evidence, not duplicate ports.
  Audited source `2e53699e` lacks the PCA whitening binding export; this is
  not an inspection of the differently sourced published native payload.
  Later export additions remain unqualified and are not retroactive release coverage.
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
