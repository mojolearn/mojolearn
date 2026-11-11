# Pre-0.8.67 measurement: the measurement build against 0.8.55 and 0.8.66

One NVIDIA L40S and one AMD MI300X. One job per box measures the measurement build (wheels built from one commit to time them; never a release
candidate, AGENTS.md), not a published release,
against the mlsys paper's 0.8.55 numbers and the 0.8.66 boards on main, plus every row tonight's lanes left
owed. Our arms only. Opponents are never rerun; the compare reuses the paper's stored opponent cells.

| piece | file | what it does |
|---|---|---|
| job | `tools/ops/paper_compare_job.sh` | on the box: plan, install, smoke, timed boards, drivers, compare (header has every knob) |
| stage | `tools/ops/measure_candidate.py stage` | on the Mac: validate the packed wheels, copy them through the vendor wrapper with sha256 checks, print the submit line |
| install | `tools/bench_board.py --install-only --ours-install-only` | one venv per arm: our wheels, scikit-learn, pandas, no opponent stack |
| candidate bytes | `tools/candidate_wheel_install.py` (through `bench_board --candidate-wheel`) | RECORD, metadata, tags and payload source-commit checks; sha256 of each file in `job/wheel-manifest-candidate.json` and in every board's `box.mojolearn.wheel` |
| paper compare | `tools/paper_compare.py compare --prior-compare` | three columns: 0.8.55, 0.8.66, candidate. Flags: REGRESSION vs each, RECOVERED, STILL-REGRESSED. Label: `candidate <sha12> (<version>)` |
| owed rows | `bench/results/paper_compare/owed-pre0867.json`, `tools/measure_owed.py` | 29 rows: define A/Bs, deletions made without measurement, CV / search / RFE / stacking, pandas input, neighbour queries, trees, and one neighbouring non-board shape per family |
| drivers | `tools/measure_owed_driver.py` | the rows the board cannot express: `while_loop` `check_every`, pandas-input fits, an MLP fit on a non-board shape |

## 0. Inputs

1. **The measurement build.** `tools/release.py candidate` (the subcommand keeps its name) builds the wheels from one
   exact commit and never touches main,
   tags, PyPI or the release ledger (`tools/release_candidate.py`). It runs five steps of the release and no
   other: `reuse-plan`, `linux-builds` (`.github/workflows/release-linux-build.yml` on GitHub's free runners,
   one run per set: cuda sm_89, cuda sm_80, hip gfx942), `linux-wait`, `linux-assemble`, `linux-pack`. The
   Release runs with `--only` those five, so `freeze-version`, `freeze-changelog`, `freeze-docs-facts`,
   `freeze-commit` (the version bump commit and its push), `identity-at-freeze`, `rehearsal`, `readiness`,
   `cross-compile`, every gate, column, publish, `finish-line` and `record` never run. The commit is pinned in the
   build's own `state.json`. The version is local, `0.8.67.dev0+<commit12>[.<label>]`, and exists only in the
   packed wheels: PyPI refuses local versions, so they cannot be uploaded. Each measurement build gets its own work dir,
   `~/mojolearn-evidence/release-candidates/<version>/`, with its own source worktree and local ledger. The
   wheels land in `<work dir>/<commit12>/linux/final/` beside `candidate.json`, where
   `measure_candidate.py stage` looks for a local `--version`. At the end the tool checks that origin/main and
   the tag lists are unchanged.

   The run needs the commit pushed, and a tooling checkout whose branch is pushed at its HEAD, because the
   workflow runs from that branch. Pass it with `--ref`, or run from a checkout of that branch. The tool refuses
   a work dir inside the main checkout.

   A set first (it fills the binding cache the B sets reuse):
   ```
   python3 tools/release.py candidate --commit <SHA40> --ref <pushed tooling branch>
   ```
   Then the four B sets, all at once. Each is the same commit with `--defines`, which the workflow passes to
   every compile as `MOJOLEARN_BUILD_DEFINES`. The label defaults to `d` plus 8 hex digits of the defines; the
   owed arm names below are clearer:
   ```
   python3 tools/release.py candidate --commit <SHA40> --ref <branch> --defines "-D MOJOLEARN_EM_LOOP_CHECK_EVERY=1" --label em1
   python3 tools/release.py candidate --commit <SHA40> --ref <branch> --defines "-D MOJOLEARN_EM_LOOP_CHECK_EVERY=8" --label em8
   python3 tools/release.py candidate --commit <SHA40> --ref <branch> --defines "-D MOJOLEARN_SGD_LOOP_CHECK_EVERY=1" --label sgd1
   python3 tools/release.py candidate --commit <SHA40> --ref <branch> --defines "-D MOJOLEARN_XD_FAST_SVD_ONE_BLOCK" --label svd1
   ```
   Add `-D MOJOLEARN_GBDT_DEVICE_ROUNDS_OFF` as its own B set once a `.mojo` file of the measured commit reads
   it; none does at measure-prep 3d67f6c437. The tool refuses any define that no `.mojo` or `bindings/` file of
   the commit names, because such a B set would be byte-identical to the A set.
   Lane t3-gbdt-greedy-on-device-loop adds `-D MOJOLEARN_GBDT_GREEDY_DEVICE_ROUNDS_OFF` (read by
   `gbdt/methods/doc_parallel_boosting.mojo`): its own B set, label `greedyoff`, for the greedy-searcher
   (default) SymmetricTree GBDT rows; the pointwise rows stay on `MOJOLEARN_GBDT_DEVICE_ROUNDS_OFF`, which turns
   both arms off.

   A B set recompiles only the bindings that can read its define. The reuse plan builds every row whose closure
   names the define (`release_reuse.apply_defines`). The binding cache keys and builds every other binding without
   the variable (`tools/bincache.py scope_build_defines`), so those are cache hits on the A set's bytes.
   `core/em_loop_chunk.mojo` and `x_linear/device.mojo` are the readers today. The defines are recorded in the
   run name, `GHA/defines.txt`, each wheel's `LINUX_PAYLOAD.json` (`candidate.defines`), the version label and
   the box's `wheel-manifest-<arm>.json`.

   Expected wall time: the three sets build at once, about 1.5 h for the A set. The 0.8.66 release legs took
   5217 to 5513 s, and 0.8.65's took 2712 to 3892 s. linux-pack then runs about 10 to 20 min on the Mac
   (pack, the auditwheel docker run, the strip). Each B set is mostly cache-hit assembly plus the reader groups'
   compiles, about 30 to 60 min plus the pack. Three B sets at once fan out about 3 x 3 x (shards + 1) jobs, so
   GitHub's concurrent-job cap may queue some of them. Use `--dry-run` to print the five steps' plan without
   running anything.
2. **Job tree.** The tree the job runs `tools/` from must contain this lane's commits (the integration branch
   after merge). It is a different commit from the measurement build's source commit.
3. **Datasets** on each box: `sh tools/dataset_store.sh stage "<ssh flags+target>" gbm-bench/taxi/taxi_speed.npz gbm-bench/istella/istella_speed.npz gbm-bench/istella/istella_rank.npz`
   (the rank rows need `istella_rank`; without it both installs refuse).
4. **B wheel sets** for the define A/Bs. Stage each with the A set: `--b-arm em-check-1=<B work dir>/<commit12>/linux/final@<SHA40>`
   (likewise `em-check-8`, `sgd-check-1`). The stage step reads each B set's own version from its `candidate.json`.
   `svd-one-block` is `-D MOJOLEARN_XD_FAST_SVD_ONE_BLOCK` (label `svd1`): B routes PCA / TSVD / randomized-svd through `svd_op_tall`. Rows without a staged set report
   `B-NOT-RUN`.
5. **The ours-only venv** gets our wheels, `scikit-learn` (the trees opponent pin, 1.7.2), and `pandas` and
   `huggingface_hub` at their `pixi.lock` versions (3.0.5 and 1.33.0). The board's drivers import nothing else
   for our arm: the other third-party imports belong to opponent arms or to the parquet download step.
   `huggingface_hub` fetches the `causal-lm-generate` checkpoint when the box lacks it. The **prior arm** (published
   0.8.66) runs on this tree's drivers. Two driver changes since the 0.8.66 run need APIs that 0.8.66 lacks, and
   both now fall back: mamba2/3-decode and samba-decode use 0.8.66's public per-call `step` (it has no public
   `decode_session`; `decode_n` refuses there), and the `while_loop` driver passes `check_every` only when the case
   names one.

## 1. Submit commands (orchestrator; boxes already up)

NVIDIA (lane tree `/root/mojolearn-measure`; the stage step syncs it from `--tree`):
```
python3 tools/ops/measure_candidate.py stage nvidia --version '0.8.67.dev0+<sha12>' --source-commit <SHA40> --tree <job worktree> \
    [--b-arm em-check-1=<dir>@<sha40> ...]
# prints, among others:
SUBMIT bash tools/nvidia_central.sh submit measure --cap 240 --note pc-<sha12>-nvidia --results box-out/pc-<sha12>-nvidia \
    --env PC_WHEEL_DIR=/root/pc-in/<sha12>/nvidia --env PC_SOURCE_COMMIT=<SHA40> [--env PC_B_ARMS=...] tools/ops/paper_compare_job.sh
```
The NVIDIA queue caps a job at 240 minutes, so split it into three submits on the same `--results` dir. Add
these to the printed line:
```
--env PC_PARTS=plan,install,smoke,board,compare                          # paper + owed measurement-build full rows
--env PC_PARTS=install,owed,compare --env PC_OWED_ARMS=candidate,em-check-1,em-check-8,sgd-check-1
--env PC_PARTS=install,owed,drivers,compare --env PC_OWED_ARMS=prior
```
AMD (the droplet's `/root/mojolearn` must be at the pushed job-tree head; the stage step refuses otherwise):
```
bash tools/do_amd_steward.sh update <job tree sha>
python3 tools/ops/measure_candidate.py stage amd --version '0.8.67.dev0+<sha12>' --source-commit <SHA40> --tree <job worktree>
nohup bash tools/do_amd_steward.sh run pc-<sha12>-amd box-out/pc-<sha12>-amd \
    'PC_WHEEL_DIR=/root/pc-in/<sha12>/amd PC_SOURCE_COMMIT=<SHA40> bash tools/ops/paper_compare_job.sh' \
    > ~/mojolearn-evidence/measure-candidate/amd-run.log 2>&1 &
```
`run` holds an ssh session for the whole job, hence `nohup`. The results persist to R2 either way. AMD has no
240-minute cap, so it runs as one job. Use `--dry-run` on the stage step to print every copy command without
touching a box.

## 2. Phases, wall time, disk, busy cores

Estimates are scaled from the 0.8.66 AMD paper run (20 cores, 235 GB RAM): smoke took 527 s with 2 shards, and
the timed board took 3405 s for 62 races, about 55 s per race.

| phase | runs on | parallel | wall (est.) | disk (est.) |
|---|---|---|---|---|
| plan | CPU | one process, seconds | < 1 min | - |
| install | CPU (+1 GPU set load per arm) | every arm at once + the dry run | 3-6 min | ~1 GB per arm venv (candidate, prior, each B) under `/root/pc-cache` |
| smoke | one GPU, not timed | min(nproc/4, MemAvailable/12 GB, 8, races) shards on ONE venv | 10-15 min | data blocks per shard cache, ~1-2 GB each |
| board (paper + owed measurement-build full rows, ~62 + ~90 races) | one GPU, timed | serial by nature, nothing else on the box | 2.5 h | results ~1.5 GB |
| owed: B arms and neighbour rows (~30 races) | one GPU, timed | serial | 30-40 min | ~0.3 GB |
| owed: prior 0.8.66 (~80 races) | one GPU, timed | serial | 1.2-1.5 h | ~1 GB |
| drivers | one GPU, timed | serial | 5-10 min | - |
| compare + report | CPU | seconds | < 1 min | - |

Total: about 5 h per box. Budget 20 GB of free disk. The 0.8.66 smoke made a 13 GB venv per shard because it
installed the opponent stack. An ours-only run installs our wheels, scikit-learn and pandas once per arm.

Busy cores: `job/cores.log` samples every 60 s and `job/job.log` gets a `BUSY` line every 5 minutes. Install and
smoke should use most cores. A timed board shows only the cores its driver uses (2-3 of 20 on the 0.8.66 AMD
board). That is by design: a timed race must not share the GPU or the CPU with other work, so state busy cores of
total in the report and do not add work during the board phases.

## 3. Reading the result (one-line summaries only)

```
python3 tools/box_persist.py fetch --tag <tag> --extract --dest ~/mojolearn-evidence/measure-candidate/<tag>
head -n 12 <dest>/SUMMARY-<vendor>.txt     # PAPER-COMPARE vendor=.. 0.8.55->0.8.66->candidate_<sha12>_(0.8.67) ..
head -n 12 <dest>/OWED-SUMMARY-<vendor>.txt # OWED-REPORT vendor=.. ; slower than the prior ; B arm faster ; drivers
tail -n 8 <dest>/job/phases.tsv; tail -n 3 <dest>/job/cores.log; cat <dest>/job/status.json
```
How to read the flags:

- Paper rows: `RECOVERED` is a 0.8.66 regression fixed. `STILL-REGRESSED` and `REGRESSION-VS-PRIOR` are open work.
- Owed rows: `REGRESSION` means the measurement build is slower than 0.8.66. `<b>:B-FASTER` means the define's B arm wins
  that cell. The define decision takes the NVIDIA and AMD average (CLAUDE.md measurement item 2).

Every file carries the label `candidate <sha12> (<version>)` and the wheel sha256s. It can never be read as a
release.

Store the result under `bench/results/paper_compare/candidate-<sha12>/`. Never overwrite the 0.8.66
`compare-<vendor>.json`: it is the middle column of the next compare.
