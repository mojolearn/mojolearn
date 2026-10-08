# Running the IDENTICAL switch grid over lq

`tools/six_lane_grid_lq.py` turns a grid plan into `lq add` lines, one file per box. After the runs it turns the
lq results and job logs into the three inputs that `tools/six_lane_grid_decide.py` reads. It never builds, races
or connects to a box. Lanes queue nothing. The orchestrator feeds the lines and copies the results back.

## Commands

```bash
# 0. Freeze one commit for the round and push it. Both arms build from this branch.
#    A moving main mixes heads; collect then reports INCOMPLETE identity.
git push origin <sha>:refs/heads/grid/freeze-<date>
python3 tools/six_lane_grid.py --crosses auto --cap 0 --out ~/mojolearn-evidence/grid-lq/plan

# 1. Render one lines file per box (laptop, metadata only)
for v in nvidia amd; do
  python3 tools/six_lane_grid_lq.py render --plan-dir ~/mojolearn-evidence/grid-lq/plan --vendor $v \
    --branch grid/freeze-<date> --b-repeats 3 --out ~/mojolearn-evidence/grid-lq/$v.lines
done

# 2. Feed each box (orchestrator; one loop per box, resumable).
#    Waits while the box has >= 4 unfinished jobs; checks every 300 s.
bash tools/six_lane_grid_lq_feed.sh nv  ~/mojolearn-evidence/grid-lq/nvidia.lines 4 300 > ~/mojolearn-evidence/grid-lq/feed-nv.log 2>&1 &
bash tools/six_lane_grid_lq_feed.sh amd ~/mojolearn-evidence/grid-lq/amd.lines    4 300 > ~/mojolearn-evidence/grid-lq/feed-amd.log 2>&1 &

# 3. Progress (grep-sized): the run id is in <lines>.json totals.run_id
lq results nv  '<run_id>' | tail -5
lq results amd 'NO-RESULT|CMD .* rc=[1-9]' | tail -5

# 4. Collect. The orchestrator copies two files per box:
#      /root/lq/results.txt
#      a dump made ON the box: grep -H -E '^(ALGOS|GRIDBB)' /root/lq/out/*/*.log > /root/lq/grid-logs.txt
python3 tools/six_lane_grid_lq.py collect --plan-dir ~/mojolearn-evidence/grid-lq/plan \
  --results ~/mojolearn-evidence/grid-lq/nv-results.txt ~/mojolearn-evidence/grid-lq/amd-results.txt \
  --logs ~/mojolearn-evidence/grid-lq/nv-grid-logs.txt ~/mojolearn-evidence/grid-lq/amd-grid-logs.txt \
  --out ~/mojolearn-evidence/grid-lq/collected

# 5. Decide (unchanged tool; pass the plan's matrix, not the committed default)
python3 tools/six_lane_grid_decide.py --matrix ~/mojolearn-evidence/grid-lq/plan/grid-matrix.json.gz \
  --verdicts ~/mojolearn-evidence/grid-lq/collected/grid-verdicts.json \
  --identity ~/mojolearn-evidence/grid-lq/collected/summary.json \
  --quality  ~/mojolearn-evidence/grid-lq/collected/quality.json \
  --out ~/mojolearn-evidence/grid-lq/decide
```

`render` options: `--only-lanes`, `--phase factorial|pairwise`, `--budget-hours H`, `--b-group-size N`
(default 8 workloads per incumbent line), and `--lanes-json`/`--board-json` (registries for tests).

`--budget-hours H` keeps A lines in order while their races plus the new workloads' B repeats fit in H hours.
Each race is costed at half the plan's pair time: nvidia 38 s, amd 24 s per pair.

`collect` options: `--min-samples` (default 2 incumbent repeats for a floor) and `--run-id`.

## Two routes, one line per build

Every line carries `MOJOLEARN_GRID_TAG=<run>.<...>`. The run id is a hash of the plan files, so collect ignores
lines from another campaign.

A lines carry `MOJOLEARN_BUILD_DEFINES=<D1=1,D2=3>` with the pack's define set. A pack is the planner's set of
configurations that share one define set (`grid-plan.json` `builds.packs`), so one build covers them all. B lines
(the incumbent) carry no defines. Each workload's B is repeated `--b-repeats` times, spread evenly through the
file; the repeats are the noise floor.

**RACE: `expanded:` workloads** (tools/bench_board_algos.py lanes)

```
lq add nv RACE <branch> <lane[,lane]> <ds[,ds]> MOJOLEARN_GRID_TAG=<run>.<pack> MOJOLEARN_BUILD_DEFINES=... BUILDS=build_x,...
lq add nv RACE <branch> <lane@ds,...> PAIRS MOJOLEARN_GRID_TAG=<run>.B<grp>r<k> BUILDS=build_x
```

- box_job.sh runs overlay_race_job2.sh and prints one `ALGOS ... median_ms= quality={json} digest=<16 hex>` line
  per race.
- A binding the line did not list is built with the same exported defines, and the race is retried.

**CMD: every other family**, through tools/bench_board.py

```
lq add nv CMD <branch> <run>.<pack>.bb MOJOLEARN_GRID_TAG=<run>.<pack>.bb MOJOLEARN_BUILD_DEFINES=... \
   $PWD/.pixi/envs/default/bin/python tools/six_lane_grid_bb.py --tag <run>.<pack>.bb --vendor nvidia \
   --race classical:kmeans:taxi --race trees:rf:istella ... BUILDS=build,build_<x>,...
```

- The B lines use the tag `<run>.C<grp>r<k>`.
- The mapping:
  - `classical:L@dataset=D` -> `classical/L/D`
  - `more:L@dataset=D` -> `classical2/L/D`
  - `neural:L` -> `neural/L/<its data>`
  - tree `L:D` -> `trees/L/<board dataset>`. Task lanes map their driver dataset back through
    `TREE_TASK_DATASETS`: `gbdt-multiclass:taximc` -> taxi, `gbdt-rank-*:istellarank` -> istella,
    `gbdt-categorical:taxicat` -> taxi.

  Each (family, lane, dataset) must be a race that `bench_board.plan_races` plans in IDENTICAL on the vendor.
- `BUILDS=`: the base `build` (core `_mojolearn`), plus the pack's bindings, plus the members' workload bindings
  from the plan. The CMD path has no retry-on-missing-binding loop, so this list must be complete.
- box_job.sh (CMD) builds those bindings in a detached worktree of the branch, with every `MOJOLEARN_*` token
  exported. That export is the box delegate's change of 2026-10-07; the tokens also sit before the interpreter
  as an environment prefix. It then runs the command in the worktree.
- tools/six_lane_grid_bb.py:
  1. Puts the tree's `python/` on the pixi interpreter with a `.pth` file. bench_board drops `PYTHONPATH`, and
     the `.pth` is how the built bindings get installed here. It also installs scikit-learn if it is missing.
  2. Runs `bench_board.py --modes identical --rows full --rounds 1 --no-infer --skip-install --python-env
     <that python> --cache /root/board-0833/cache --no-smoke-gate`. Our arm only, since the opponent store at
     `<job dir>/opponent-store.jsonl` is empty. Output goes to `--out /root/lq/out/<id>/bb` (the job dir outlives
     the worktree). It runs once per (family, dataset set), so only the requested cells race.
  3. Prints one `GRIDBB tag= vendor= head= family= lane= dataset= status= median_ms= hash=<16 hex> quality={json}`
     line per requested race, from `bb/board.json`, and then `GRIDBB-DONE`. results.txt gets only the CMD log's
     last line, so collect reads the GRIDBB lines from the log dump.

**Output digests.** Every bench_board cell carries `hash`:
- trees: the FSPEED prediction hash.
- classical, classical2, algos: the last round digest.
- neural: the last round digest. The training lanes (lm-train-step, samba-train-step) had none, so
  `bench_board_neural.outputs_digest` now hashes the race's saved outputs (sha256, 16 hex). It is recorded as
  `outputs_digest`, printed as `NEURAL-DIGEST`, and used by `bench_board.classical_cells` when the rounds carry
  no digest.

**Box prerequisites for CMD lines.**
- The board caches under `/root/board-0833/cache`: ctd-data, more-data and algos-data blocks. bench_board preps a
  missing block itself, untimed, once.
- The taxi/Istella npz under bench_board's `--data-root` default (`$GBM_BENCH_DATA` or `~/datasets/gbm-bench`).
- pip in the pixi env. The script bootstraps it with ensurepip, as overlay_race_job2.sh does.

## Prebuilt bindings (build once on a CPU box, race on the GPU boxes)

Each lq line makes the GPU box build `build` + its `BUILDS=` scripts (about 4 min per line; the builds, not the
races, dominate GPU time). `tools/six_lane_grid_prebuild.py` builds every needed (binding, define set) once on a
CPU-only Linux x86_64 box and ships the `.so` files to the boxes; `render --prebuilt /root/grid-prebuilt` adds
`PREBUILT=/root/grid-prebuilt` to every line, and box_job.sh (tools/ops/box_job_prebuilt.patch) installs from the
store instead of building, falling back to the build when the store cannot serve the line.

```
# 1. plan (Mac or build box; metadata only): the deduplicated artifacts for the rendered lines
python3 tools/six_lane_grid_prebuild.py plan --plan-dir ~/mojolearn-evidence/grid-lq/plan --vendor nvidia --out <store>
python3 tools/six_lane_grid_prebuild.py plan --lines ~/mojolearn-evidence/grid-lq/nv.txt --vendor nvidia --out <store>  # the exact queued lines
# 2. build (CPU-only Linux box, in the frozen tree, `pixi install` done): parallel cross compiles, resumable
python3 tools/six_lane_grid_prebuild.py build --vendor nvidia --out <store> --jobs 16           # sm_89
python3 tools/six_lane_grid_prebuild.py build --vendor amd    --out <store> --jobs 16           # gfx942
# 3. pack + ship: one tar per vendor; the credential stays on the Mac (presigned PUT), or push from the Mac with aws
python3 tools/six_lane_grid_prebuild.py pack --out <store>                                       # <store>/grid-prebuilt-<vendor>-<sha12>.tar.gz + .json
python3 tools/six_lane_grid_prebuild.py presign-put --key grid-prebuilt/<source_sha>/nvidia-<tarsha12>.tar.gz    # Mac
python3 tools/six_lane_grid_prebuild.py push --sidecar <store>/grid-prebuilt-nvidia-<sha12>.tar.gz.json --put-url '<URL>'  # box
# 4. the GPU box fetches, verifies and unpacks to /root/grid-prebuilt/<vendor>/ (script printed on the Mac, run via lq CMD or by the orchestrator)
python3 tools/six_lane_grid_prebuild.py box-script nvidia <store>/grid-prebuilt-nvidia-<sha12>.tar.gz.json > nv-fetch.sh
# 5. render with the token
python3 tools/six_lane_grid_lq.py render ... --prebuilt /root/grid-prebuilt --out ~/mojolearn-evidence/grid-lq/nv.txt
```

- Dedup: a define changes a binding's bits only when its NAME is referenced by a file in that binding's Mojo import
  closure (the hook's import graph, tools/hooks/no_host_routes.py Tree, the same one the box uses to pick rebuilds;
  `core/six_lane_experiment_guards.mojo` names every define and is excluded). Each (binding, defines that reach it) is
  compiled once and serves every full define set that narrows to it. At f5ea97003: 1388 lines x bindings = 3276 box
  builds per vendor -> 2186 distinct (binding, full set) -> 1488 artifacts per vendor (24 bindings, 179 defines).
- Fidelity: `build` runs the real `bindings/build_<x>.sh` with the box's environment (`MOJOLEARN_NUMERIC_MODE=identical`,
  `MOJOLEARN_TARGET_COLUMN=<vendor>`, `MOJOLEARN_GPU_ARCHS=sm_89|gfx942`, `MOJOLEARN_BUILD_DEFINES=<effective set>`,
  `MOJOLEARN_COMPILE_JOBS=1`) under a `pixi` shim that records the `mojo build` argv instead of compiling, then runs that
  argv with the real compiler and `-o` at the artifact. Host bindings are not prebuilt (races never load them; ID lines
  never carry `PREBUILT=`).
- Store layout: `<store>/<vendor>/<binding>/<key>/_mojolearn_<x>.so` + `receipt.json` (defines, serves, argv, script
  sha, closure sha, compiler version, source sha, host), `<vendor>/manifest.json`, `<vendor>/lookup.tsv`
  (`binding  sha256(sorted full define set)  artifact  artifact_sha256  dest`). `plan.json` carries the counts and
  `estimate_minutes_at_jobs` (60 s per compile).
- Install (`tools/six_lane_grid_install_prebuilt.sh <tree> <vendor> <defines-csv|-> <bindings-csv> [store]`): exit 0
  when every listed binding was copied to its `dest` and sha-verified; exit 2 when the store is missing, was built from
  another commit than the tree's HEAD, a host binding is listed, the (binding, define set) is absent or a file is
  corrupt: box_job.sh then builds everything as before. rc.txt shows `<build> rc=prebuilt` or `prebuilt rc=miss ...`.
- A Mac can run `plan` and a `build` smoke (the dry run emulates Linux, so the recorded argv is the box's), but its
  artifacts are Mach-O: `box_usable=false`, excluded from lookup.tsv and pack. Real artifacts come from Linux x86_64.
- Tests: `cd tools && python3 test_six_lane_grid_prebuild.py` (synthetic tree, fake compiler, fake .so files).

## Collect outputs (`--out`)

| file | schema | read by |
| --- | --- | --- |
| `grid-verdicts.json` | `mojolearn.six-lane-timing-verdicts/1` | decide `--verdicts` |
| `summary.json` | `mojolearn.six-lane-comparison/1` | decide `--identity` |
| `quality.json` | rows | decide `--quality` |
| `floors.json` | `mojolearn.six-lane-aa-floors/1` | the floors from the B repeats |
| `collect-report.json` | | coverage and job health |

**`grid-verdicts.json`.**
- scored = A_ms / median(B_ms), with B from the same head when it exists.
- floor = log(p90/p10) of the B repeats with 4 or more samples, else log(max/min) (`six_lane_timing._spread`).
- The verdict comes from `six_lane_timing.judge`: FASTER or SLOWER only beyond the floor on both NVIDIA and AMD,
  in the same direction. Only the scored phase is judged.

**`summary.json`.** `arms.A` and `arms.B` compare the NVIDIA digest with the AMD digest. Each arm is one of:
- MATCH: the digests are equal.
- MISMATCH: same head, different digests. Also used when the incumbent's digest changes from run to run on one
  vendor.
- INCOMPLETE: a digest is missing or the race failed. Also used when the digests differ and the heads differ.

**`quality.json`.** One row per configuration, workload and vendor. The A quality JSON is compared with B's,
metric by metric:
- WORSE or BETTER: a metric moves beyond 1e-6 relative.
  - Lower is better: rmse, logloss, inertia, error, residual, diff, ...
  - Higher is better: accuracy, auc, r2, trustworthiness, recall, silhouette, ...
- SAME: every judged metric is within that tolerance.
- FAIL: the A race failed.
- PENDING: there is no incumbent.

Keys with no known direction are listed in `unjudged`.

**`collect-report.json`.** Coverage per vendor (MEASURED/FAILED/MISSING/REFUSED), plus these job-health lists,
which should all be empty:
- `no_result`: RACE jobs that reported no result.
- `jobs_without_algos`: a RACE job whose results line lost its ALGOS text to the cut, with no logs to fill it.
- `truncated_without_log`.
- `cmd_jobs_without_gridbb`: a CMD job with no GRIDBB lines in the dump.

## Totals at this branch (`--crosses auto --cap 0`, after the planner dataset fix; nvidia and amd identical)

| | per box |
| --- | --- |
| plan cells | 3,354 |
| cells raced (A) | 2,361 (70%) |
| RACE A cells (expanded) | 535 |
| CMD A cells: neural / trees / classical / classical2 | 932 / 670 / 174 / 50 |
| lines (= builds) | 1,285 |
| A lines: RACE / CMD | 145 / 1,038 (factorial 252, pairwise 931) |
| B lines: RACE / CMD | 36 / 66 (3 repeats; 186 + 201 races) |
| configurations / workloads | 1,707 / 129 |
| projected race hours | nvidia 14.5, amd 9.2 |

The projected hours count only the races, at the plan's median pair time. Build time is not counted: each of
the 1,285 lines is a fresh worktree, pixi install, libMojolearnMath and its bindings. Tree and neural races also
run longer than the median.

**Refused cells: 993 per vendor (28 workloads), each named with its reason in `<lines>.json`.**

| refused | cells | reason |
| --- | --- | --- |
| rf, et: `istellamc`, `istellareg`, `taximc`, `taxireg`, `year` | 480 | tree driver datasets with no board race (the board races these lanes on taxi/istella only) |
| gbdt-symmetric(-1000), gbdt-depthwise, gbdt-lossguide: `istellareg`, `taxireg`, `year` | 336 | same |
| gbdt-ordered: `istellareg`, `taxireg` | 94 | no board dataset in `TREE_TASK_DATASETS` |
| `iforest:anomaly` | 47 | the board races iforest on taxi/istella |
| `gbdt-categorical:criteo` | 32 | criteo is not a board dataset |
| `expanded:sgd-reg@regression_report` | 4 | lane variant; no race form |

**Fixed in the planner** (`tools/six_lane_grid.py map_workloads`). `expanded:` and `more:` workloads now keep only
the datasets their board lane races:
- the time-series lanes race `taxi-hourly` and `synthetic`;
- `lu-solve`, `more:arima` and `more:ets` race `synthetic`.

**Accepted with a note.** 22 `@input=` variant workloads (classification-full-v1, tsvd-full-v1) race the board's
own full-row block, not the registered variant input.
