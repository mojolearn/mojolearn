# Plan: make the code fast, then verify everything once per release (Andrew, 2026-11-05)

READ THIS FIRST after a compact or in a new session. It replaces the "reuse old verification runs" goal as the main
plan. State as of 2026-11-06 ~03:00Z, main `f069cb12ba`.

## The plan in one paragraph

Stabilising Mojolearn goes through performance. First remove the per-call Python and repeated work from the hot paths (most of it came from
taking Python out of the runtime and doing it badly: Mojo code that still asks Python for each data item). Fix them in
Mojo; never bring Python back. Shrink verification fixtures so no lane is slow. Then each release does ONE verification
round: build the wheels once, run every lane and every check once on every device from those wheels, compare, publish.
Nothing is verified after a release. Narrow re-verification ("only what changed") is no longer a goal.

## Decisions taken on 2026-11-05 (Andrew)

1. Verification is pre-release only. No step after the first publish runs a lane on any device (also in `CLAUDE.md`).
2. One verification round per release, on the built wheels. No separate gate that repeats work.
3. Run everything every release. Reuse of old runs is an opt-in bonus (`--reuse-unchanged`), not the default.
   Reason: the lane selector cannot soundly narrow changes to shared files, and it under-selects in places.
4. Fixtures are small. Target: no lane above about 5 s on any device.
5. One NVIDIA package: the PTX sm_80 set ships inside `mojolearn-nvidia`; `mojolearn-nvidia-ptx` is gone (five
   publishes). The PTX identity column STAYS; native and PTX are recorded in parallel on one pod.
6. The loader keeps native sm_89 first on Ada cards, then PTX.
7. CPU work runs free on GitHub runners (the repo is public): the CPU column and every check that needs no GPU.
8. No users yet: no upgrade-path engineering; redoing work is fine.
9. Performance fixes are implemented in Mojo. Python stays out of the runtime; the rule is kept.
12. No same-bits (ID) checks during development; the one global verify at the very end is the only one. Bits may
    change between versions; identity is only across GPU vendors within one version.
10. Fix everything first, then build once and verify once. No compile or rental until the fixes are in and Andrew
    says go.
11. Rented hosts run at full cores; the Mac is for Apple verification only; CPU work that can be handed off goes to
    RunPod CPU pods (`tools/runpod_cpu_leg.sh`).

## What is on main already (tested on RunPod CPU pods, never run for real)

- One NVIDIA wheel (`lane/nvidia-one-wheel-simple`).
- Single-step release behind `tools/release.py --identity-on-wheels` (default off): record from installed wheels, check
  units with reuse keys, data-only table repack (`tools/wheel_table_repack.py`), publish admission accepting the repack
  receipt, CPU column and checks in `.github/workflows/cpu-identity-gate.yml` (`mode=wheel`), no device run after
  publish, `identity-at-freeze` report.
- Tooling: lane-selection cache, bincache one-key-one-content rule, dependency worktree cleanup, shared GPU type list
  for prebuild rentals, `tools/identity_lane_timings.py`, pip in the test environment.
- Test debt: the tools suite on a CPU pod went from 134 failing tests to 27.
- Run everything by default under `--identity-on-wheels` (`--reuse-unchanged` is the opt-in), and 41 lanes with smaller
  fixtures (their digests change, so the next record run re-records every lane). Green on a CPU pod (leg16: no new
  failures against main, 35 fixture floors hold).

## Branches written tonight, NOT merged (each needs the step named)

| Branch | What | Needs |
|---|---|---|
| `lane/trees-boost-train-regression` @ 7580d25b71 | boosting classifier labels read natively (replaces the per-label walk added by 2c61cfe498) | compile classical runtime binding; time 5 tree lanes on NV+AMD |
| `lane/parallel-pool-spawn-cost` @ 9eaff35ad3 | parallel spawn/close of pool workers, worker exits at once, pool timing record | build 2 bindings; `tools/pool_spawn_cost_probe.py` A/B on NV+AMD |
| `lane/per-element-python-loops` @ eff066dd6a | 13 per-item Python loops found on array inputs, 8 fixed (labels for LDA/QDA/NB/logistic/forest/GBDT/sequence/GP/KNN/kernel fits, host predict decode, stratified folds); table `~/mojolearn-evidence/per-element-python-loops.md` | compile many bindings; ID checks NV+AMD; timings at 20k rows. Owed: float-y walks, SVC fit labels (`svc_common.mojo:201`), permutation_test_score (`msel_search.mojo:789`), other `msel_flatten` callers, experimental classification metrics |
| `lane/regressions-since-0855` | the per-lane timing reader `tools/identity_timing_compare.py` | merged into the integration branch |
| `lane/lane-select-precision-2` @ 8cabefab4c | selector under-selection fixes + import-time rule | PARKED (2 precision tests fail); only matters with `--reuse-unchanged` |
| `lane/build-sizing-all-cores` @ 4331938e2f | build jobs from cores and memory, no 16 cap | HELD: merged builds peak 46 GB per job; needs a measured figure |
| `lane/lane-select-glue-attribution` | REJECTED (unsound) | nothing |
| `lane/nvidia-one-wheel` (path rename), `lane/nvidia-one-wheel-columns` | not wanted / parked | nothing |

Briefs for all of these: `~/mojolearn-evidence/briefs-20261105/`. CPU pod test results and scripts:
`~/mojolearn-evidence/integ-tooling-20261105b/` (`run-leg*.sh`, `leg*/remote/leg_out/`; main's result is `leg13`).

## The fix lanes (started 2026-11-06 ~04:30Z)

The stored identity timings per lane and device (reader: `tools/identity_timing_compare.py`) pointed at a fixed
extra cost per public predict/transform call and per-label Python checks on classifier fits; those are what the lanes
removed.

Base for all product fixes: `integ/perf-20261105` (main + boost fix + per-element loop fixes + pool fix + the timing
reader). Six Opus fix lanes branch from it; brief `~/mojolearn-evidence/briefs-20261105/perf-fix-lanes.md`:
`lane/perf-public-call-cost`, `lane/perf-labels-buffer-read`, `lane/perf-neighbors-resident`,
`lane/perf-meta-estimators`, `lane/perf-svm-and-msel`, `lane/perf-trees-misc`. None compiles or runs anything. When
they report: merge into `integ/perf-20261105`, then the ONE compile and the ONE timing pass (step 3-4 below), on
Andrew's go.

### Python-removal sweep (Andrew, 2026-11-05 night: "be thorough ... eliminating unnecessary python calls")

Removing Python from a hot path is assumed good: no measuring, no checks of any kind until the one global verify at
the very end. Five more Opus lanes read every binding file they own and remove per-item Python work, calls back into
Python, repeated attribute traffic and redundant calls (double validation, several native calls where one suffices,
state rebuilt per call). Brief: `~/mojolearn-evidence/briefs-20261105/perf-python-sweep-lanes.md`. Lanes (same base):
`lane/perf-sweep-prep`, `lane/perf-sweep-classical`, `lane/perf-sweep-metrics-tsa`, `lane/perf-sweep-neural`,
`lane/perf-sweep-parallel-native`. Each leaves a site table at `~/mojolearn-evidence/perf-sweep-<lane>.md`. Second
wave, after the six fix lanes merge (they own those files now): `trees_*`, `msel_*`, `neighbors_*`, `svc_*`, `host_*`.

### State 2026-11-06 (after the fix lanes): ALL MERGED, NOTHING COMPILED

`integ/perf-20261105` @ `d097c8f90f` (on origin) = main + 24 fix lanes: 386 files, +13.5k/-3.4k lines. No lane
compiled, ran, timed or checked anything. Push this branch with `MOJOLEARN_WIDE_REBUILD_OK=1` (the rebuild fence
flags the shared modules it changes; never `--no-verify`). Per-lane site tables: `~/mojolearn-evidence/perf-sweep-*.md`.

- What remained was ~1.5-2.5 ms per public call on every column (repeated module/attribute/admission checks)
  and per-label Python checks on tree and discriminant-analysis fits; the lanes removed both.
- Ideas file (Fable): `~/mojolearn-evidence/ideas-cpu-per-call-cost.md`. HELD until after the first compile: one
  `AdmittedCall` record per public call, and a native kernel call table between binding modules (also the owed
  manifold driver inside `_mojolearn_x_decomp`).
- Compile watch list: `bindings/cpython_table.mojo` (new pointer table, 53+ bindings, prototypes never compiled),
  `bindings/native_class_admission.mojo` (type/dict watchers, Python >= 3.12), process-wide provider snapshot cache
  (`providers/python/x_neighbors_admission_pyapi.mojo`; revert if sharing across extensions misbehaves),
  `neighbor_query_native.mojo:123` and `parallel_shard_native.mojo:972` (ternaries), cluster host cells now parallel
  (`cluster/host/host_cells.mojo:64`), identity records kept in native side tables (`bindings/trees_side_records.mojo`).
- After the compile: scikit-learn estimator checks (dict unchanged by predict, pickle) on trees, meta, boosting,
  isolation forest, neighbours, IVF; one profile of a one-row predict on a CPU build.

## Known product defects found tonight, not yet assigned

- Isolation forest on CPU refits the whole forest on every score call (`bindings/trees_public_iforest.mojo:198-212`).
- GBDT border types run one GPU thread per column (`gbdt/grid_creator/border_types.mojo:94`): GPU 2-4x slower than CPU.
- `trees-calibrated` training 2.1 -> 11.4 s on AMD only.
- Apple-only slow lanes (robust covariance, factor analysis, dictionary learning, two SVC lanes, auto-ARIMA,
  metrics splitters/search) and CPU-only mean shift: causes not found.
- `neural-trainer-unfused` opens two worker pools per cell.
- Selector: at least 160 package files' import-time edits can under-select (fixed only on the parked branch).
- 27 tools tests still fail on a CPU pod: selector tests, tests needing unshipped data or built binaries, a Gaussian
  process expectation in `test_native_prep_plans`, one `test_six_lane_grid` assertion.

## Order of work

1. Finish the per-item Python audit; hand every item to an Opus fix lane
   (Fable judges, Opus implements; lanes write code and push, they never compile or run).
2. Test tooling and harness changes on RunPod CPU pods; merge when green against main's result.
3. When the fix lanes are in: ONE integration branch, ONE compile (all switches, NVIDIA and AMD targets on cheap CPU
   build boxes, macOS on the Mac), fix what breaks.
4. ONE timing pass on NVIDIA and AMD boxes for the performance fixes (before/after against main, all cores busy).
5. Merge, refreeze, and do the first real run of `--identity-on-wheels`: that run is also release 0.8.66. It records
   every lane (fixtures changed), rents one NVIDIA pod and one AMD droplet, uses the Mac for the Apple column and
   GitHub for CPU. Watch list: the single-step lane's replies, summarised in
   `~/mojolearn-evidence/HANDOFF_2026-11-05_release-0866.md`.
6. Publish five packages. Nothing runs after.

## Release 0.8.66 right now

Stopped, unpublished, nothing rented. The old-path run froze `41c7312110` and stopped at the box step (no AMD droplet in
stock); its `--status` shows identity owed on every column because two glue files changed after the last record. It will
ship through step 5 above, not through the old path. PyPI latest is 0.8.65.

## What needs Andrew's word

- Go for the compile and the box timing pass (step 3-4), and go for the release run (step 5).
- Whether the remaining slow Apple lanes may keep sizes above the 5 s target where the checked branch needs them.

## BEFORE THE RELEASE: every improvement lands first (Andrew, 2026-11-06)

Andrew: "i want to do all of our improvements ALL OF OUR IMPROVEMENTS before the release." Nothing on this list
is deferred to a later version. "After the compile" below means after the first compile and BEFORE the freeze,
the verification round and the publish. The release (0.8.66) does not start until every item is done or Andrew
has struck it.

Order: (1) third-pass fix lanes merge -> (2) ONE compile of `integ/perf-20261105`, fix what breaks -> (3) the
post-compile items below (code, then recompile only what they touch) -> (4) the profile pass and the fixes it
points to -> (5) tool tests on CPU runners, measurement smoke and multi-GPU gate dry run on the rented boxes ->
(6) fast-forward main, freeze, the ONE verification round on the built wheels, TestPyPI rehearsal, publish.

State 2026-11-06 evening: `integ/perf-20261105` = main + 24 night lanes + 10 `perf2-*` + 5 `perf3-*` + 6
`pass2-*` + multi-GPU logical-shard gate + harness per-row lane + third-pass lanes as they finish (14 `p3-*`).
Three Fable ideas passes done: `~/mojolearn-evidence/ideas-every-slow-lane-pass1.md`, `-pass2.md`,
`ideas-pass3-from-scratch.md`. UNCOMPILED. Nothing rented.

### A. Structural cleanups held until the first compile (do right after it)
- Fold the three CPython function tables into one: `bindings/cpython_table.mojo`,
  `bindings/perf2_shared_cpython_table.mojo`, `bindings/p3_lm_tokenizer_cpython.mojo` (and any further copy a
  `p3-*` lane adds). One table, one global name, all users moved; delete the copies.
- Fold the per-model record helpers into one (`trees_side_records`, `cluster_side_records`, `gp_svm_records`,
  `msel_plan_records`, prepared-registry slots), fixing the `msel_plan_records` one-entry-per-model leak.
- `AdmittedCall`: one admitted record per class and mode and per fitted model, threaded through every public
  call (design: `~/mojolearn-evidence/ideas-cpu-per-call-cost.md`); the per-family records become its users.
- Native kernel call table between bindings and the manifold driver inside `_mojolearn_x_decomp`
  (`bindings/mani_kit.mojo`: 78 per-op Python-level calls per manifold fit).
- 26 calls left in an unproven form (`cpy.PyModule_GetDict`, `cpy.PyObject_GetIter` / `PyIter_Next`): settle the
  form at the compile, align all.
- Lock on the sequence resident store (`sequence/perf3_seq_resident.mojo`).

### B. Rows the fix lanes left OWED (each lane's table has the `file:line` and the design)
`~/mojolearn-evidence/p3-*.md`, `perf2-*.md`, `perf3-*.md`, `pass2-*.md`: every row marked OWED or OWED-SHARED.
Known so far: native byte LM session record and multi-step entry; one request per contiguous worker span in
parallel causal; engine output written into the caller's address; device greedy pick for ragged lengths; kept
sessions owning a stream's KV; legacy GradientBoosting predict reading X twice; forest predict-codes operation
for meta-estimator members; RFE / IterativeImputer one crossing per feature; per-fold DeviceRows built natively;
BayesianGaussianMixture blocks without the per-query copy; decode-session per-step re-admission;
`estimator_arguments` per cluster query; pandas class cache in `classical_pandas.mojo`; IsolationForest refit on
every `fast` score (`_mojolearn_svm.mojo:643`); `MultiOutputClassifier.score` with string labels; `x-decomp-lda`
and `x-decomp-ipca` (Apple / CPU). After the 14 `p3-*` lanes finish, collect their OWED rows here and run one
more round of Opus lanes on them.

### C. Product gaps found on the way
- `LogisticRegression` (and every GLM) with several devices runs on one device in `identical` mode (the tiled
  gradient never reaches `glm/impl/qn/multi_gpu.mojo`): shard the tiled gradient.
- Unit tests of the multi-device split logic (no binding exposes it yet).
- Python 3.10 / 3.11 (wheels declare `>=3.10`): the "check once" fixes need 3.12 watchers; decide a watcher-free
  fast path (per-class version stamp) so those versions are not left on the slow path.

### D. Only a compiled build can answer (ONE profile pass of the integ wheel, with the previous release as the reference)
The 16 profiles in `ideas-every-slow-lane-pass2.md` ("Post-compile profile list") plus the forest predict
profile in `~/mojolearn-evidence/forest-regressor-other-stage.md`. Fix what they show, before the freeze.

### E. Release-process changes (tools; in flight 2026-11-06)
- Multi-GPU logical-shard gate (12 fixed lanes, 1 / 2 / 3 logical devices on one GPU) in every NVIDIA and AMD
  gate job; one dry run on a box before the freeze.
- Measurement smoke as a gate phase (every measurement recipe once at smoke size, our GPU arm, pass = runs and
  returns the board's fields).
- TestPyPI rehearsal before the first real publish; after publish only the sha256 comparison remains.
- Harness: per-row Python removed, harness work in its own `-harness` stage (`timing_env.harness_timing=2`):
  timings are not comparable with records before this change; the profile pass in D is the reference.
- Tool tests on CPU runners for everything changed in tools/ since 2026-11-05; the 27 failing tools tests.
- Release-tool log wording: stop printing "owed" in the run-everything path.

### Decisions taken 2026-11-06 (Fable judging; Andrew can reverse)
- Fitted arrays that have a resident device copy become read-only at the buffer level (in-place writes raise
  NumPy's read-only error; replacing the attribute re-uploads). No per-call content hashes.
- Byte LM `train_step` default returns loss and scalars only; gradients on explicit request (Andrew).
- Lazy `models` import reverted. PCA whitening scans its fitted buffers natively on each whitened call.

### Decision 2026-11-06 (evening): the owed rows move to AFTER 0.8.66 (Andrew)

Supersedes item B (and A, C where they need new native entry points) of "BEFORE THE RELEASE" above: the rows the fix
lanes left owed (new native entry points: device-resident fitted state, native training sessions, folds staying on
the device, fused whole-step entries, the single admission record, the manifold driver) are a planned program after
the release. Plan: `docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md` (branch `lane/plan-native-entry-points` until
merged). 0.8.66 ships what is on `integ/perf-20261105` once it compiles and passes: tests on the compiled build,
then freeze, the ONE verification round on the built wheels (every lane, every column; multi-GPU logical-shard
check; measurement smoke), TestPyPI rehearsal, publish. Still before the release: items D (profile pass, run on the
gate box while it is rented; results rank the after-release plan, they do not block the release unless a lane is
slower for a code reason that is quick to fix) and E (release-process changes, done).

Compile state: every GPU binding compiles on sm_89, sm_80 and gfx942 (GitHub cross-compile, 128 + 65 jobs); the
Linux release build of e289f32355 compiled all eleven merged shards (three device, eight CPU) and assembled; macOS
development build in progress on the Mac. Compiler stop without a diagnostic in the model selection group: fixed by
keeping the estimator clone recursion on PythonObject results (cause not isolated; a standalone reproducer compiles).
