# Pre-0.8.67 measurement: the candidate against 0.8.55 and 0.8.66

One NVIDIA L40S and one AMD MI300X. One job per box measures the candidate build, not a published release,
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

1. **Candidate wheels.** Run the release pipeline on the frozen 0.8.67 source up to `linux-pack`. The packed
   wheels land in `~/mojolearn-evidence/release/0.8.67/<commit12>/linux/final/` (`Release.split_final`). The
   GitHub workflow `release-linux-build.yml` uploads binding sets, not wheels. The wheels only exist after
   `linux-pack`, so the job installs those packed files and nothing else. Nothing is published.
   ```
   python3 tools/release.py 0.8.67 --only freeze-version,freeze-changelog,freeze-docs-facts,freeze-commit,rehearsal,reuse-plan,readiness,cross-compile,linux-builds,linux-wait,linux-assemble,linux-pack
   ```
   A fix after the measurement needs `--refreeze`, which creates a new candidate commit and new wheels.
2. **Job tree.** The tree the job runs `tools/` from must contain this lane's commits (the integration branch
   after merge). It is a different commit from the candidate's source commit.
3. **Datasets** on each box: `sh tools/dataset_store.sh stage "<ssh flags+target>" gbm-bench/taxi/taxi_speed.npz gbm-bench/istella/istella_speed.npz`.
4. **B wheel sets** (optional, define A/Bs only). Each is a full core plus vendor set, packed the same way from a
   commit that changes the define's default:
   `em-check-1` (`MOJOLEARN_EM_LOOP_CHECK_EVERY=1`), `em-check-8` (`=8`), `sgd-check-1`
   (`MOJOLEARN_SGD_LOOP_CHECK_EVERY=1`). `svd-round-launches` has no switch in code yet (see the owed file).
   Without the sets, those rows report `B-NOT-RUN`.

## 1. Submit commands (orchestrator; boxes already up)

NVIDIA (lane tree `/root/mojolearn-measure`; the stage step syncs it from `--tree`):
```
python3 tools/ops/measure_candidate.py stage nvidia --version 0.8.67 --source-commit <SHA40> --tree <job worktree> \
    [--b-arm em-check-1=<dir>@<sha40> ...]
# prints, among others:
SUBMIT bash tools/nvidia_central.sh submit measure --cap 240 --note pc-<sha12>-nvidia --results box-out/pc-<sha12>-nvidia \
    --env PC_WHEEL_DIR=/root/pc-in/<sha12>/nvidia --env PC_SOURCE_COMMIT=<SHA40> [--env PC_B_ARMS=...] tools/ops/paper_compare_job.sh
```
The NVIDIA queue caps a job at 240 minutes, so split it into three submits on the same `--results` dir. Add
these to the printed line:
```
--env PC_PARTS=plan,install,smoke,board,compare                          # paper + owed candidate full rows
--env PC_PARTS=install,owed,compare --env PC_OWED_ARMS=candidate,em-check-1,em-check-8,sgd-check-1
--env PC_PARTS=install,owed,drivers,compare --env PC_OWED_ARMS=prior
```
AMD (the droplet's `/root/mojolearn` must be at the pushed job-tree head; the stage step refuses otherwise):
```
bash tools/do_amd_steward.sh update <job tree sha>
python3 tools/ops/measure_candidate.py stage amd --version 0.8.67 --source-commit <SHA40> --tree <job worktree>
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
| board (paper + owed candidate full rows, ~62 + ~90 races) | one GPU, timed | serial by nature, nothing else on the box | 2.5 h | results ~1.5 GB |
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
- Owed rows: `REGRESSION` means the candidate is slower than 0.8.66. `<b>:B-FASTER` means the define's B arm wins
  that cell. The define decision takes the NVIDIA and AMD average (CLAUDE.md measurement item 2).

Every file carries the label `candidate <sha12> (<version>)` and the wheel sha256s. It can never be read as a
release.

Store the result under `bench/results/paper_compare/candidate-<sha12>/`. Never overwrite the 0.8.66
`compare-<vendor>.json`: it is the middle column of the next compare.
