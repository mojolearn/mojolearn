# Plan: make the code fast, then verify everything once per release (Andrew, 2026-11-05)

READ THIS FIRST after a compact or in a new session. It replaces the "reuse old verification runs" goal as the main
plan. State as of 2026-11-06 ~03:00Z, main `f069cb12ba`.

## The plan in one paragraph

Stabilising Mojolearn goes through performance. First find and fix the time regressions since 0.8.55 (most come from
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
| `lane/trees-boost-train-regression` @ 7580d25b71 | boosting classifier labels read natively (20x fit regression from 2c61cfe498) | compile classical runtime binding; time 5 tree lanes on NV+AMD |
| `lane/parallel-pool-spawn-cost` @ 9eaff35ad3 | parallel spawn/close of pool workers, worker exits at once, pool timing record | build 2 bindings; `tools/pool_spawn_cost_probe.py` A/B on NV+AMD |
| `lane/per-element-python-loops` @ eff066dd6a | 13 per-item Python loops found on array inputs, 8 fixed (labels for LDA/QDA/NB/logistic/forest/GBDT/sequence/GP/KNN/kernel fits, host predict decode, stratified folds); table `~/mojolearn-evidence/per-element-python-loops.md` | compile many bindings; ID checks NV+AMD; timings at 20k rows. Owed: float-y walks, SVC fit labels (`svc_common.mojo:201`), permutation_test_score (`msel_search.mojo:789`), other `msel_flatten` callers, experimental classification metrics |
| `lane/regressions-since-0855` | timing table 0.8.55 -> now per lane and device, attribution, fix list (`~/mojolearn-evidence/regressions-since-0855.md`) | in flight; its fix list feeds new fix lanes |
| `lane/lane-select-precision-2` @ 8cabefab4c | selector under-selection fixes + import-time rule | PARKED (2 precision tests fail); only matters with `--reuse-unchanged` |
| `lane/build-sizing-all-cores` @ 4331938e2f | build jobs from cores and memory, no 16 cap | HELD: merged builds peak 46 GB per job; needs a measured figure |
| `lane/lane-select-glue-attribution` | REJECTED (unsound) | nothing |
| `lane/nvidia-one-wheel` (path rename), `lane/nvidia-one-wheel-columns` | not wanted / parked | nothing |

Briefs for all of these: `~/mojolearn-evidence/briefs-20261105/`. CPU pod test results and scripts:
`~/mojolearn-evidence/integ-tooling-20261105b/` (`run-leg*.sh`, `leg*/remote/leg_out/`; main's result is `leg13`).

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

1. Finish the regression table against 0.8.55 and the per-item Python audit; hand every item to an Opus fix lane
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
