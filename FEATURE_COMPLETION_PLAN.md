# Feature completion and release follow-up

Updated 2026-09-06. This is the requested actionable supplement to
[ROADMAP.md](ROADMAP.md), not a certificate or a release announcement.
Unchecked boxes are planned work. Evidence applies only to its recorded
source, artifact, device, numeric mode, shape and feature configuration.

The original scope remains: continue Mamba2/3 implementation, complete the
missing symmetric-tree CatBoost features and UMAP features, prove the claimed
IDENTICAL behavior, and compare FAST and IDENTICAL against exactly one
appropriate external implementation per workload on NVIDIA through RunPod.
The six follow-ups below add to that scope; they do not replace it.

## Execution and resource rules

- **Latest execution direction:** skip Apple testing. Focus new validation
  and measurements on remote NVIDIA and AMD machines. Any necessary local
  host-only checks must use at most **two CPU cores and two threads** (the
  requested ceiling is two or three cores); do not run local Mojo builds or
  GPU workloads. The macOS release item below uses retained evidence and
  does not authorize fresh Apple tests.
- [ ] Keep **all tests, builds, model execution, measurements and benchmarks
  in the root/main thread**. Explicitly tell every subagent: **NEVER run
  tests, measurements, benchmarks, builds, GPU/model code or provisioning**.
  Parallel agents may inspect sources, implement changes and author checks.
- [ ] Run GPU jobs serially with bounded fixtures, memory limits, CPU/compiler
  thread caps, deadlines and independent rental cleanup watchdogs. Preserve
  time to fetch evidence and confirm deletion. Keep GPU/model/build jobs
  serial across rentals. A bidirectional checkpoint exchange may retain a
  second idle remote host, with its own watchdog; it does not authorize
  simultaneous execution. A stalled process must not start an overlapping
  replacement. Record stopped/failed jobs as such.
- [ ] For each NVIDIA performance workload, retain FAST, IDENTICAL and one
  external NVIDIA arm in the same rental session, with matched input bytes,
  dtype, settings, transfer/timing boundaries and independently checked
  outputs. Check GPU execution explicitly, including prediction. No CPU
  performance baseline, cross-session ratio or universal identity claim.
- [ ] Store commands, exit statuses, raw outputs, input/output hashes, source
  inventory, binding/wheel hashes, dependency/compiler versions, GPU model,
  driver, launch parameters and numeric mode with every acceptance result.
  Keep per-block arithmetic/dispatch provenance where the claim needs it.

See [NVIDIA feature inventory](bench/NVIDIA_FEATURE_COVERAGE_2026-09-06.md)
and [current campaign record](bench/results/resume/2026-09-06-root-feature-nvidia/README.md).

## Near-term training milestone (active)

### Alpha API exposure (user priority, September 6)

Public API availability does not require completion of every numerical
certificate. Expose implemented operations now, mark experimental limits,
and continue NVIDIA/AMD validation independently. Never invent an API for an
unimplemented algorithm or describe a missing native binary as available.

- [ ] Add normal public `linalg`, `umap`, and training primitive modules.
- [ ] Inventory Mamba 1/2/3 forward/backward and tree objective coverage in
  installed alpha documentation; distinguish ordered boosting from ranking.
- [ ] Assemble an explicitly prerelease alpha artifact with exact base-native
  and updated-Python provenance. Preserve existing stable release checks;
  an alpha overlay is not a fresh installed numerical certificate.
- [ ] Validate package contents and selected remote NVIDIA/AMD callable
  surfaces, publish through the existing Trusted Publisher identity, and
  verify actual index filenames/hashes. No new Apple execution.
- [ ] Implement missing ranking objectives and carried-state backward
  separately; alpha exposure alone does not complete those algorithms.

Root alone runs tests, builds and measurements, using at most three CPU
cores (two remains the default). Subagents only author/review source.

### Paper-gap execution queue (September 6 follow-up)

- [ ] **Public neural training:** expose and validate a public FP32 complete
  training step; existing fixed internal Transformer evidence is not a public
  multi-block learning claim. Public optimizer/loss modules and the fixed MLP
  and byte-LM trainers are authored; finish native integration and exact-wheel
  callable checks. Mamba 1/2/3 zero-state IDENTICAL Python backward is already
  exposed; finish shared-shape and installed-artifact validation rather than
  describing the methods as absent.
- [ ] **Mamba-3 long intermediate:** locate the reported Apple failure and
  preserve its exact source/profile/intermediate. Reconcile it with the later
  NVIDIA/AMD long-profile record at `395d9421` linked in
  [backward certification](mamba/BACKWARD_CERTIFICATION.md). Its compositional
  arithmetic policy does not claim that every intermediate passes a direct
  whole-FP64 threshold, and it does not recertify Apple. Diagnose any remaining
  failure under the actual declared policy; keep it outside paper claims
  until supported by the appropriate evidence.
- [x] **UMAP bounded real-data neighborhood preservation:** locate the retained
  NVIDIA digits result. It measures k=10 trustworthiness **and retention**
  against umap-learn on pinned 1,024 fit / 256 held-out rows, with controls;
  all 15 campaign jobs passed. See [retained result](bench/results/resume/2026-09-06-root-umap-nvidia/README.md).
  This specific experiment is measured, not synthetic-only or still unmeasured.
- [ ] **UMAP broader coverage:** test additional declared real-data fixtures
  and compare matching IDENTICAL outputs across NVIDIA/AMD. Do not infer
  general dataset coverage from the digits result or bitwise equality to cuML.
- [ ] **Actual cross-vendor checkpoint resume:** compile the authored driver,
  produce same-source continuous and head checkpoints on NVIDIA/AMD, transfer
  the actual files in both directions, continue training in fresh processes,
  and compare complete state bytes with effective missing-moments controls.
  Identical checkpoint bytes alone remain the narrower supported claim until
  execution and admission finish. Follow
  [root commands](training/PUBLIC_TRAINING_RESUME_COMMANDS.md).

The immediate training target is a small neural network followed by a small
language model that learns next-token prediction from real text. This extends
item 4 and uses item 3's resume proof; it does not replace the six follow-ups.

- [ ] Validate the public fixed FP32 8→16→3 MLP: forward, independent gradient
  reference, AdamW, complete state snapshots, and checkpoint continuation.
  NVIDIA run 3 built the binding and passed all 20 surface/reference checks
  and four numerical-edge checks. Two overflow assertions expected the wrong
  Python exception type; both native calls correctly refused overflow. The
  exact-message assertion repair passed in frozen run 4 (`8f6ed41`): all
  18 jobs and admission passed, including 16-step learning and same-device
  full-state checkpoint continuation. AMD comparison remains open.
- [x] Independently check the existing one-block Transformer gradients before
  extending its arithmetic claim. On NVIDIA at `201e5ebc`, all eleven tensors
  and loss passed the preset FP64 gate; both deliberate-error controls were
  effective. See [scoped evidence](bench/results/resume/2026-09-06-root-training-nvidia/README.md).
  This does not qualify the new two-block model or cross-vendor training.
- [ ] Implement a two-block byte-level decoder with vocabulary 256, width 32,
  context 32 and batch 2, using caller-supplied, pinned real-text token bytes.
  Expose the complete training step and explicit data continuation cursor.
  Source is authored in `training/byte_lm.mojo` and the public
  `SmallByteLanguageModelTrainer` API: 34,944 FP32 parameters, 20 tensors.
  The remote profile 5 runner requires an explicit GPU architecture, checks
  an independent one-step gradient/AdamW oracle before a fixed 128-step run,
  and retains complete state plus root guard receipts. Compilation and
  numerical qualification of this new model remain open.
- [ ] Demonstrate learning with a predeclared held-out loss criterion and
  retained training trace. Fix train/validation split, initialization, token
  schedule and optimizer settings before execution; no selected lucky run.
- [ ] Compare full parameters, gradients, AdamW moments, counters and loss bits
  at every step on remote NVIDIA and AMD from the same frozen source. Transfer
  a checkpoint in both directions and compare against continuous runs.
- [ ] Record numerical reference checks separately from bitwise agreement:
  identical outputs alone cannot show that both implementations are correct.
- [ ] Recertify Metal separately when fresh Apple execution is authorized.
  Historical three-vendor toy-fixture evidence is not certification of this
  new model, source or real-text experiment.

This is a **small language model**, not a large language model. Existing
13,376-parameter single-block training evidence makes it a plausible near-term
engineering milestone, but neither learning quality nor the new cross-vendor
claim has passed yet. Several working days is an estimate conditional on the
gradient checks and remote integration, not a delivery guarantee.

## 1. Fourth-machine hedges and per-block provenance caveat

- [ ] Inventory every existing hedge that limits identity to the tested
  machines, and the associated per-block provenance caveat. Keep these two
  qualifications together in the resulting evidence and public wording.
- [ ] Select additional hardware that exercises a meaningful architecture or
  launch/partition difference; freeze the exact candidate and workloads
  before renting. Budget **two sequential rentals and about one day** as an
  estimate, not an executed booking, guaranteed duration or result.
- [ ] Root reruns the named IDENTICAL fixtures on the additional machines;
  compare complete arrays/cards and relevant per-block partials against the
  retained reference. Record dispatch, reduction topology and compilation
  provenance so a matching final digest cannot hide an unexamined path.
- [ ] Close each hedge only for the combinations actually exercised. A
  fourth machine expands empirical coverage; it does not establish identity
  on arbitrary hardware or remove a missing per-block provenance caveat.

Acceptance: retained machine manifests, per-block evidence where applicable,
raw-byte comparisons and explicit remaining exclusions. See
[identity path audit](IDENTITY_PATHS.md) and
[conformance evidence](docs/CONFORMANCE.md).

## 2. Publish 0.6.0 for macOS and AMD Linux

Concrete packaging and admission edits are tracked in the
[AMD-only release implementation plan](packaging/linux/AMD_ONLY_RELEASE_PLAN.md).

Existing evidence was inspected: the
[installed gap-closure record](bench/results/resume/2026-09-06-installed-gap-closure/README.md)
qualifies **24 AMD installed jobs at frozen source
`eb835021dcd79a59a7e8f78c754a75db3c1fea83`**, normalized wheel SHA256
`7c5f9af825cbcbd74a293adc75ad15670a30a179d3a9a8a8cfd993cd9476f7be`.
It does not qualify newer source, a replacement wheel or a combined wheel.
The [Apple follow-up](bench/results/resume/2026-09-06-feature-finish/README.md)
and roadmap record fifteen interpreter/mode jobs for a later Apple candidate;
that is candidate evidence, not proof of publication.

- [ ] Freeze the intended release source and inspect actual package-index
  files and hashes before deciding whether 0.6.0 remains available to
  publish. Never replace an already uploaded file under the same version.
- [ ] Review retained qualification for the exact macOS artifact without
  rerunning Apple tests, and refresh AMD Linux qualification remotely
  for the final source/artifact; do not reuse `eb835021` results to certify
  the later serialization, sequence or current feature changes.
- [ ] Resolve the **AMD-only Linux artifact/admission dependency** explicitly:
  the current [Linux release admission](packaging/linux/RELEASE_QUALIFICATION.md)
  requires both vendors to qualify the same combined HIP/CUDA wheel. An AMD
  release with NVIDIA source-build-only needs an explicit AMD-only package
  contract, payload/installer behavior and admission checks; retain the
  existing combined-wheel refusal until that supported path is implemented
  and validated. A passing AMD candidate is insufficient for combined release.
- [ ] Keep NVIDIA **source-build-only** until a fresh full installed candidate
  clears the original five failures: two Mamba FAST/DETERMINISTIC accuracy
  failures and three Transformer stalls. Source overlays removed the stalls
  and improved accuracy, but are not final-wheel qualification; retain the
  original failures and demand all required jobs on the final bytes.
- [ ] Publish the qualifying macOS and AMD artifacts using the
  [release runbook](docs/PYPI_RELEASE.md), then clean-install the actual index
  artifacts and retain the workflow, index hashes and post-publication jobs.
- [ ] Remove stale “unpublished”, “release pending” and equivalent current
  wording **only after actual publication and retained evidence**. Preserve
  historical failure/candidate records and unresolved NVIDIA restrictions.

Acceptance: final-source provenance, exact uploaded artifact hashes, complete
claimed-platform installed matrices, supported-architecture metadata and
successful installation from the actual public index. No publication or
rental is performed by creating this plan.

## 3. Cross-vendor checkpoint resume

The fixed-profile native driver and evidence comparator are authored; see
[remote resume commands](training/PUBLIC_TRAINING_RESUME_COMMANDS.md).
Compilation, numerical validation and cross-vendor execution remain open.

- [ ] Audit the current checkpoint implementation and retained runs; the
  [checkpoint format document](training/CHECKPOINT_FORMAT.md) contains an
  older “written, not run” status that must not be promoted or dismissed
  without matching current execution evidence.
- [ ] Freeze one IDENTICAL training fixture, inputs, optimizer configuration
  and step count. Run a same-device continuous reference, then write a real
  checkpoint at a fixed intermediate step, terminate the process and resume
  it in a new process on another GPU vendor.
- [ ] Compare the resumed final state with the same-device continuous digest
  and retained raw tensors: parameters, optimizer moments, counters/flags,
  RNG and all other carried state required by the contract. Test both
  directions for each vendor pair being claimed.
- [ ] Verify equal-state checkpoint bytes and deliberate corruption/missing
  state refusals; record any intentionally excluded metadata separately.

Acceptance: checkpoint hashes, source/destination manifests, continuous and
resumed step traces, final complete-state digest and byte comparison. An
in-process resume or equal output alone does not close this item.

## 4. Public FP32 Transformer training step and Mamba backward

The source audit and proposed state/API contract are in the
[public training implementation plan](training/PUBLIC_TRAINING_IMPLEMENTATION_PLAN.md).

- [ ] Provide one public FP32 Transformer training step composing **embedding,
  forward, backward, loss, AdamW and checkpoint save/load**. Specify every
  tensor shape, parameter owner, gradient layout and supported numeric mode.
- [ ] Root validates all gradients and optimizer updates against independent
  reference calculations, then validates repeated steps and the real-file
  resume path from item 3. Retain loss and complete training-state evidence.
- [ ] Keep Mamba2/3 Python backward shape-compatible with the corresponding
  forward inputs/parameters, including explicit refusals for unsupported
  cache-continuation VJPs and modes. Add missing supported paths in separately
  reviewable slices with matching shape and ownership checks.
- [ ] Add Mamba Python backward validation at the public training fixture's
  **B=2, L=8, model width=32**, with explicit family-specific parameter
  layouts. The retained small backward fixture does not close this shared
  shape requirement by itself.
- [x] Attach the root's newly reported NVIDIA result: **IDENTICAL zero-state
  Mamba2/3 backward matches all 20 exposed gradient tensors for the tested
  fixtures**. See the [retained test output](bench/results/resume/2026-09-06-root-feature-nvidia/run/remote/feature-supplement/mamba23-backward.log),
  [job statuses](bench/results/resume/2026-09-06-root-feature-nvidia/run/remote/feature-supplement/results.tsv)
  and [campaign provenance](bench/results/resume/2026-09-06-root-feature-nvidia/README.md).
  This result does not certify FAST backward,
  continuation-cache gradients, all shapes or cross-vendor backward.
- [ ] After correctness admission, root measures FAST and IDENTICAL public
  training workloads against one matching NVIDIA reference per workload.
  Preserve unsupported arms as unsupported rather than substituting a
  different mathematical workload.

Acceptance: public API checks plus independently validated complete gradient
and state tensors at the declared shapes, checkpoint continuation evidence,
and separately scoped timing records. See
[Mamba backward certification](mamba/BACKWARD_CERTIFICATION.md),
[loss contract](training/IDENTICAL_LOSS_CONTRACT.md) and
[optimizer contract](training/IDENTICAL_OPTIMIZER_CONTRACT.md).

## 5. Fill the missing hardware columns

**OPEN — reaffirmed by the user on September 6.** No new hardware-column
run is admitted by this queue update. First locate and audit retained results
to avoid rerunning completed fixtures or promoting smoke tests into identity
certificates. The user's “ARIMA fitter Apple only” note describes a gap to
investigate: `SUPPORT_MATRIX.md` already points to fitted ARIMA smoke passes
in frozen AMD/NVIDIA installed candidates. Those passes must be reconciled
with the broader batched-fitter claim, rather than erased or assumed to
certify it. Start with
[installed gap-closure evidence](bench/results/resume/2026-09-06-installed-gap-closure/README.md).

- [ ] Inventory exact ARIMA fitter, Holt-Winters, spectral and Gaussian-process
  result paths, commits, devices, modes and shapes; distinguish filter-only,
  fitted smoke, independent correctness and cross-device identity evidence.
- [ ] Root alone schedules and executes any missing runs, serially, preferring
  remote NVIDIA/AMD. Necessary MacBook runs are now permitted by the user,
  with two CPU cores/threads by default (never above three), explicit memory
  and time limits, and no concurrent tests/builds/measurements. CPU limits
  alone do not bound GPU or unified-memory allocation; size those explicitly.
  Subagents must never execute tests, builds, models or measurements.

- [ ] **Batched ARIMA: NVIDIA and AMD.** Exercise the actual batched estimator
  fit/filter/predict shapes and declared options; retain separate evidence
  from existing single-fixture filtering and fitted smoke jobs.
- [ ] **Holt-Winters: NVIDIA.** Validate supported trend/seasonal configurations,
  fitted state and forecast outputs in the claimed numeric modes.
- [ ] **Spectral: NVIDIA.** Enumerate the public spectral workloads being
  claimed, validate their intermediate/final outputs, and distinguish a
  component eigensolver check from complete estimator coverage.
- [ ] **Gaussian process: NVIDIA.** Validate the declared fit/predict and
  uncertainty surface with bounded matrix sizes and retained numerical checks.
- [ ] Update each [support-matrix](SUPPORT_MATRIX.md) column only after its
  current-source hardware evidence passes. For performance, select one
  mathematically matching NVIDIA comparator; if unavailable, retain a
  correctness-only result and explicitly leave performance unmeasured.

Acceptance: per-workload device/source/mode/shape manifests, independent
correctness results, raw-byte identity comparisons for declared IDENTICAL
fixtures, and retained failures. Old Apple–AMD cards do not fill NVIDIA cells.

## 6. Real-dataset UMAP neighborhood preservation

The [digits quality runner](tools/umap_real_dataset_quality.py) passed remotely
on NVIDIA at frozen source `6146b121608d4cf73706d6540bb884e134df409c`.
See the [retained completion evidence](bench/results/resume/2026-09-06-root-umap-nvidia/README.md).
Closure below is scoped to the pinned digits experiment and separate bounded
cuML comparison; it does not assert arbitrary dataset coverage.

- [x] Select and pin a real dataset with provenance, checksum, bounded sample
  size and fixed preprocessing/train/held-out split. Declare the neighborhood
  metric and acceptance thresholds before viewing the results.
- [x] Measure neighborhood preservation (for example trustworthiness and
  neighbor overlap) for public UMAP fit and held-out transform in FAST and
  IDENTICAL. Keep seeds, input bytes, neighborhood size, metric and effective
  optimization parameters in the evidence; check IDENTICAL repetition by bits.
- [x] Compare quality against pinned **umap-learn on CPU**. This is the
  explicitly requested **external quality-only baseline exception**, not a
  CPU performance baseline or an exception to NVIDIA-only performance
  comparisons. Do not derive speed ratios from this arm.
- [x] Keep the separate NVIDIA performance workload at exactly one external
  NVIDIA comparator, such as cuML, with matched settings. Different algorithms
  and embeddings require independent quality criteria; do not claim bitwise
  equality to umap-learn or cuML without actual bitwise evidence.

Acceptance: dataset/preprocessing hashes, retained embeddings, neighborhood
metric definitions and results, held-out quality, mode provenance and explicit
pass/fail thresholds. Synthetic fixtures and dispatch checks alone do not
close real-data quality. See the [UMAP scope in the support matrix](SUPPORT_MATRIX.md)
and [current NVIDIA inventory](bench/NVIDIA_FEATURE_COVERAGE_2026-09-06.md).

## Added queue: Table 6 context and k-NN/GEMV optimization

- [x] Locate the current paper's Table 6 (reported page 7) and place the
  explanation in the same column, directly beside the approximately 25×
  k-NN and 21× GEMV cost figures. Preserve the measured values and provenance.
- [x] Ground the adjacent sentence in the kernel investigation: the reported
  slowdowns expose current implementation/operand-reuse optimization gaps in
  the named kernels; they do not establish an unavoidable cost of the
  identity contract. Distinguish established causes from hypotheses for
  each kernel. Root reviews the eventual rendered placement.
  Implemented in the sibling `mlsys/paper/paper.tex` Table 6 caption;
  root rebuilt named and anonymous PDFs serially with two-thread caps and
  confirmed page 7 placement. k-NN's operand-reuse cause is stated; GEMV's
  corresponding causal attribution remains unresolved. Measured ratios are
  unchanged. This closes the editorial placement, not the optimization work.
- [ ] Queue operand-reuse and dispatch improvements for the pinned distance
  and GEMV kernels, preserving exact operation/reduction order wherever the
  contract requires it. Root validates complete bytes before admitting speed.
- [ ] Prefer NVIDIA through RunPod for new measurements. Compare FAST and
  IDENTICAL with exactly one matching NVIDIA external implementation per
  workload: cuML for k-NN when its settings and distance/neighbor semantics
  match; a suitable NVIDIA BLAS implementation for GEMV if cuML exposes no
  equivalent public workload. Match inputs, dtype, output semantics, transfer
  boundaries, warmup and timed rounds; report independent quality checks.
- [ ] Separately compare MojoLearn IDENTICAL raw outputs across the remote
  NVIDIA/AMD devices being claimed. cuML is a NVIDIA quality/performance
  reference, not the cross-vendor bitwise oracle. Fresh Metal remains deferred.
- [ ] Revisit Table 6 only with retained measurements from the optimized
  candidate. Keep historical ratios and source scopes available; do not
  rewrite them as new results or make a universal overhead claim.

All tests, rendering/build checks, models and measurements remain root/main
only. Subagents may inspect and edit sources but must never execute them.
This queue addition does not interrupt the active MLP, language-model,
checkpoint, release or other feature work above.

## Dependency order and closure

1. Inventory current evidence and freeze source/artifact scopes; establish the
   resource guardrails and exact admissions before any new rental.
2. Complete the public training/state interfaces and remaining correctness
   fixes (item 4); use these to drive real-file checkpoint resume (item 3).
3. Prepare fourth-machine fixtures and provenance checks (item 1), missing
   hardware workloads (item 5), and real-data UMAP checks (item 6) in parallel
   source-only work; root executes them serially on the appropriate hardware.
4. Resolve AMD-only release packaging/admission and qualify final artifacts
   (item 2). This release may be deliberately narrower than all six future
   items, but it must not advertise incomplete features as certified.
5. Publish only admitted artifacts; update current release wording and scoped
   feature/support claims with links to the corresponding retained evidence.

Continue the original symmetric-tree backlog alongside these steps: ranking
and group targets, multi-target/uncertainty losses, general Ordered boosting,
categorical combinations, leaf weights and heterogeneous histogram layout.
Continue Mamba2/3 and UMAP missing surfaces as separate slices. New metadata
APIs, a narrow backward fixture or one successful NVIDIA comparison do not
establish full feature parity or bitwise identity for every feature.
