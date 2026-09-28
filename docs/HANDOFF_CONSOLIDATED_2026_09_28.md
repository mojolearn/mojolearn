# Consolidation handoff — 2026-09-28

Written before conversation compaction, approximately 19:56 UTC. This is a
snapshot, not a live job status. Read the queue before drawing timing conclusions.

Latest operational update, approximately 21:28 UTC:

- Numerical candidate and packaging inputs remain frozen at `b8dfa7710`;
  subsequent handoff edits do not change that candidate. All five CUDA prebuilds
  passed. Jobs `nvc1-0016` (445 lanes) and `nvc1-0019` (physical par-gmm) remain
  queued; no numerical results from either yet. Runner owns monitoring.
- AMD packaging preparation completed: 75 witnessed copies plus 35 new lower-tier
  builds, preserving copied binding/math bytes. All 110 native files and build
  provenance are archived locally under the durable evidence root at
  `packaging-preparation/amd-b8dfa7710-build-inputs.tar.gz`; SHA-256
  `f5091a33a69413137ea8fd5f4f279bf39e1971296af13003f8613388cf31d227`.
  These are build inputs, not a staged or qualified wheel. AMD's observed
  shutdown deadline is 23:01:42 UTC; the archive is already local.
- M4's scheduled retirement at 21:16:30 UTC interrupted packaging preparation.
  Last preserved receipt: 34/45 builds; new binaries were not recovered. M3 Ultra
  also retired. M2 Pro remains running but busy with another owner's checks;
  no idle Apple host or reusable copies of the M4 outputs have been confirmed.
- User now authorizes additional NVIDIA/AMD capacity if needed. Another AMD box
  is unnecessary. A second shared NVIDIA pod is already being provisioned by
  another workflow; our idempotent request deferred to its fleet lock. Four-GPU
  stock attempts failed and two-GPU attempts are pending. No new pod is ready.
  Do not start a competing provisioner or claim a quote reserves capacity.
- Migration archive is prepared locally in `/tmp/mojolearn-cuda-migration/`.
  Runner must validate actual architecture/toolchain/source closures before reuse.
  If job 0016 starts first, leave it running and migrate only queued 0019. Cancel
  only our old queued entries after replacement readiness, never duplicate runs.
- Live durable state: `cuda-runner-state.json`, `packaging-preparation/state.json`,
  and `cuda-comparison-plan.json` in the evidence root listed below. The final
  comparison command is prepared but has not run. NumPy policy remains pending.

Post-compaction update, 20:07 UTC: main is `c46f77614`. The parallel candidate
table gap described below is fixed and pushed (48 focused tests passed); use
`docs/PAR_GMM_CANDIDATE_WORKFLOW.md` for the prepared qualification recipe.
No physical par-gmm run has happened yet. NVIDIA job `nvc1-0016` remains queued;
native prebuild is progressing without a current failure. Older Apple jobs
0006–0009 precede it, including an overlapping sweep on a modified source tree;
their results cannot automatically substitute for the pinned candidate.
An explicit optional runtime NumPy extra versus fully NumPy-free runtime policy
question is pending with the user. Do not infer an answer from elapsed time.

## Release answer

**No commit is yet qualified for the requested next PyPI release.** Main is
`b447ec05701fe35fb96cef10b332c76e9c7f40c6` before this documentation-only handoff.
It is the integrated candidate, not an approved release. Apple/AMD qualification
is complete for the scoped 445-lane comparison; CUDA, `par-gmm`, the strict
runtime NumPy policy, and candidate-wheel/release gates remain outstanding.
No version bump, PyPI publication, or arXiv email was performed.

## User intent and operating constraints

- Merge the Apple work first, then perform one coordinated check; do not rerun
  whole sweeps after every fix. Target only affected lanes after failures.
- The specifically requested `apple-merged-owed` (`79559ad0f`) and final
  `neighbors-apple2` (`75ddcae2e`) are already merged and pushed to main, alongside
  the consolidated Apple branches. Do not repeat those merges.
- Main pushes and subagent work are authorized. Never force-push. Other sessions
  can advance main, so fetch/reconcile before the next push.
- Work in `/Users/andrewhendel/mojolearn-wt/apple-consolidated`, branch
  `fix/apple-consolidated-verifier`. Original workspace
  `/Users/andrewhendel/CascadeProjects/mojolearn` contains unrelated user work;
  do not reset or edit it as the integration checkout.
- Use existing hardware. Do not allocate a machine or cancel another owner's
  jobs. A previous unanswered request for a replacement NVIDIA allocation is
  obsolete: another workflow supplied existing capacity.
- Keep failed raw records and strict admission checks. No hash normalization,
  dropping mismatches, fabricated evidence, or silent relaxation of NumPy policy.

## Completed checks and source repairs

See [validation status](CONSOLIDATED_VALIDATION_2026_09_28.md) and the committed
[final evidence scope](../bench/results/consolidated_check/2026-09-28_final-base/README.md).

- 445 single-device base configurations agree across Metal M4 and HIP AMD:
  1,671 matching numeric parts, zero missing/incomplete/different parts. Each
  selected GPU record also passed its local CPU comparison.
- Inventory is 504 lanes. The 59 physical parallel-device configurations are
  explicitly outside this default scope, not silently counted as passes.
- Original base sweep at `308878e80679` ran once. Later checks were targeted.
  All-nine-fixture evidence was collected where required for reference repairs;
  new batch probes used base plus the other eight disjoint fixtures.
- 230 missing-reference lanes and two ordinary stale GMM lanes are repaired:
  232 scoped admissions. Only `par-gmm` remains stale.
- Optional `mojolearn[verify]`, actionable missing-NumPy guidance, lazy exports,
  isolated verifier workers, 120-second arm bounds, incremental progress and
  unhealthy-GPU stopping are integrated. All formerly undeclared batch properties
  now have executable probes or explicit structural inapplicability reasons.
- Fixed platform-dependent GLM fixture exponential arithmetic, unreachable
  TreeSHAP zero-cover divisions, optional preprocessing host-export detection,
  BCa/NMF/LayerNorm probe contracts, and metrics undefined weighted-score NaNs.
- Metal eigensolver defaults to the established implementation because the new
  path crashes the Metal compiler. Experimental opt-in remains unqualified;
  CUDA/HIP and new SVD behavior are retained. Do not remove required fences.
- Apple/AMD passed the 12-case radix regression; CUDA coverage is in the queue.
- Comparator enforces expected lane and batch revisions, complete records,
  per-lane source consistency, and original numeric hashes. Wheel audit checks
  generated lane fragments and CTR payload against source.

## Active NVIDIA work — coordinate with runner, do not duplicate

Subagent `/root/runner` owns preparation, queueing, and monitoring. Ask it for live
status. `/root/applicability` completed reference/admission and par-gmm analysis;
`/root/integration_audit` completed source/NumPy audits and is idle.

The old 2×4090 host retired at 19:19 UTC before numerical checks. Its job IDs
are obsolete. The existing replacement `nvc1` has two A40 GPUs, architecture
`sm_86`; this task uses one GPU for the 445-lane continuation.

- Exactly one task-owned GPU job is queued: **`nvc1-0016`**, lane
  `apple-consolidated`, cap 120 minutes. Do not confuse this with a historical
  identically numbered job on the retired host.
- CPU native prebuild is healthy, two build workers within one invocation.
  Last report: byte-LM/forest and estimators completed; GBDT CUDA/host compiling.
  One old SIGTERM143 entry is from the deliberate jobs1→jobs2 restart, not a
  current build failure. Do not launch a second checker against its build tree.
- The queued wrapper waits up to 60 minutes for explicit
  `prebuild-complete.json`, verifies PID/start time, and fails closed via
  `prebuild-failed.txt`. It does not touch checker bindings before readiness.
- Remote repository: `/root/mojolearn-apple-consolidated`, clean pinned
  `5ee51236a` snapshot. Five private source worktrees and native cache:
  `/root/apple-consolidated-cuda-5ee51236a`.
- Prebuild log:
  `/root/mojolearn-apple-consolidated/.git/apple-cuda-prebuild.log`.
- Use `bash tools/nvidia_central.sh queue` and
  `bash tools/nvidia_central.sh status nvc1-0016` for read-only status.
  Existing-host shell access is `bash tools/nvidia_central.sh sh
  apple-consolidated 'COMMAND'`. Runner owns mutations and job commands.

Pinned recipe: [CUDA continuation](../tools/consolidated_check/CUDA_RESUME_20260928.md),
`tools/consolidated_check/cuda_resume.py`, and
`tools/consolidated_check/cuda_resume_20260928.json`.
It covers all 445 lanes exactly once across seven groups/five commits:

| Commit | Scope |
| --- | --- |
| `308878e80679` | 337 unaffected lanes, base; radix regression |
| `9d64cb98b284` | 97 repaired/prep/property lanes, base |
| `9d64cb98b284` | 4 GLM lanes, all nine fixtures |
| `9d64cb98b284` | 2 SHAP lanes, all nine fixtures |
| `ad1680bbeaaf` | BCa, unpaired resampling, NMF, all nine fixtures |
| `64f7d941396d` | LayerNorm, all nine fixtures |
| `2c81a1293390` | Regression metrics, all nine fixtures |

Evidence is under the workspace's `evidence/COMMIT-GROUP/clean_0of1` directories.
After completion, retrieve all saved records and compare against Apple/AMD with
the explicit per-lane overlays and latest expected plan. Local CUDA/CPU agreement
alone is insufficient. Preserve failures and rerun only affected lanes after fixes.

## Evidence and resume locations

- Committed final report and expected plan:
  `bench/results/consolidated_check/2026-09-28_final-base/{crossvendor,plan}.json`.
- Committed admissions/raw JSON:
  `bench/results/identity_break/2026-09-28_admitted-*`.
- Persistent full archive:
  `/Users/andrewhendel/mojolearn-evidence/consolidated-2026-09-28/`.
  Contains `saved-json-records.tar.gz`, SHA manifest and reports; 2,470 original
  JSONs, including failures (~57.2 MB unpacked). Prefer this over transient /tmp.
- Local live comparison inputs:
  `/tmp/mojolearn-current-308878e80679`,
  `/tmp/mojolearn-current-repairs-9d64cb98b284`,
  `/tmp/mojolearn-current-batch-remaining8-9d64cb98b284`.
- Local final report: `/tmp/mojolearn-final-445-crossvendor.json`.
  Expected plan: `/tmp/mojolearn-final-metrics-plan/plan.json`.
- Remote Apple/AMD evidence roots under `~/mojolearn-evidence/`:
  `consolidated-main-308878e80679`, `consolidated-repairs-9d64cb98b284`,
  `consolidated-probe-fixes-ad1680bbeaaf`,
  `consolidated-batch-remaining8-9d64cb98b284`,
  `consolidated-layernorm-64f7d941396d`, `consolidated-metrics-2c81a1293390`.

## par-gmm: reviewed next step, no new job submitted

An eligible current AMD ONE-device all-nine-fixture record exists at
`/tmp/consolidated-existing-records/do-amd/ev-apple-merged-9f20e20ac/s/clean_0of1/par-gmm.gpu.json`.
SHA-256: `5abd39df5941bd63cc0990b4357c787caa8102064e3f279414a35ee3c17160e7`.
Strict admission passes: HIP class, complete, one repeat, `par_devices=0`,
`classic-kmeanspp-init-1`, current fixture/held-out hashes, eight usable parts.
Its CPU counterpart refused and is not a witness.

A bounded future job on the existing two A40s can collect:

1. CUDA ONE-device par-gmm, all nine fixtures, full properties, enforced CUDA.
2. A scoped candidate baseline from matching AMD and CUDA one-device witnesses.
3. TWO-device par-gmm on devices `0,1`, all nine fixtures, compared against that
   candidate, retaining `_verify_par.PoolWitness` placement for every fixture.
4. Promotion only after successful physical comparison.

Require two distinct GPU UUIDs and the cooperative worker's requested group;
GMM uses one worker containing both GPUs, not two workers. Check binding
`gmm_parallel_available()`. This qualifies NVIDIA sharding, not AMD two-device.
`build_table()` intentionally admits ONE-device records; `admit(par_axis=True)`
is coverage-only. **Tool gap:** `verify --par` currently ignores
`--reference-table` and reads the shipped table. Add explicit candidate-path
support or use an isolated candidate table before qualification. Do not compare
against stale shipped hashes and call the expected failure a new regression.

## NumPy and release gates

See [NumPy audit](NUMPY_RELEASE_BLOCKERS_2026-09-28.md). Core metadata has no
mandatory dependencies and `[verify]` supplies NumPy, but the strict runtime
audit still reports 23 import sites in 19 files, including sequence/CNN/ANN APIs.
Base import without NumPy does not prove those APIs work without it. The user has
not chosen between a bit-preserving runtime rewrite and a documented optional
runtime-feature dependency policy. Do not silently allowlist estimator imports.

Candidate wheel build/install and normal release gates are still owed. Relevant
audits: `python packaging/portable_math/wheel.py --python-tree STAGED_ROOT` and
`python packaging/portable_math/wheel.py --audit-only CANDIDATE.whl`.
Local targeted pytest interpreter:
`/tmp/mojolearn-pypi-check.fLA6dk/venv/bin/python` (system Python lacks pytest).
Completed targeted tests already passed; do not rerun broad suites without a
new change or unresolved concern.

New `py-*` branches are separate in-progress performance/correctness work, not
NumPy removal; they remain outside main. Do not automatically merge unfinished
branches while waiting for CUDA. Preserve current numerical fixes in later merges.

## Resume order

1. Contact runner and inspect existing prebuild/job0016; preserve its work.
2. Finish CUDA comparison, repairing only actual failing lanes.
3. Address candidate-table handling and bounded physical par-gmm qualification
   on existing hardware when the required two GPUs can be scheduled.
4. Resolve runtime NumPy policy/implementation and candidate wheel gates.
5. Commit evidence/status updates, fetch/reconcile concurrent main, push normally.
   Do not declare release-ready or publish until remaining gates are satisfied.
