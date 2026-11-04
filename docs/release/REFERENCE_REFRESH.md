# Reference refresh after a merge

`tools/reference_refresh.sh <sha>` keeps the verifier's reference table
(`python/mojolearn/verify_reference/table.json`) current with main. For one
merged commit it does four things:

1. records only the identity lanes whose source closure changed, on every column;
2. admits all the fetched records into the table in one call;
3. commits the admission on branch `ref/<sha12>`;
4. pushes that branch.

The orchestrator merges `ref/<sha12>` into main. The next release freeze then
finds every row already admitted, so identity work does not run at release time.

## When to run it

Run it after every merge to main that touches `bindings/`, `core/`, a lane's
Python route under `python/mojolearn/`, `tools/identity_break.py` or anything
else a lane's source closure reaches.

You don't have to decide whether a merge counts. Run `--plan`; it answers per
column:

```
tools/reference_refresh.sh --plan <sha>
PLAN nvidia REUSED 419 / RECORD 9 of 428
PLAN amd    REUSED 419 / RECORD 9 of 428
PLAN cpu    REUSED 416 / RECORD 12 of 428
PLAN apple  NOTHING-CHANGED (REUSED 428 of 428)
```

A merge of docs, evidence or selection tooling plans `NOTHING-CHANGED`
everywhere, and the refresh then submits nothing.

Run one refresh at a time, oldest merge first, and merge each `ref/<sha12>`
before you start the next. Each admission branch starts from `origin/main`, so
it is built on every admission already merged.

## Usage

| command | what it does |
|---|---|
| `tools/reference_refresh.sh --plan <sha> [columns]` | prints REUSED / RECORD counts per column; submits and saves nothing |
| `tools/reference_refresh.sh <sha> [columns]` | runs the full refresh: submit, wait, fetch, admit, commit, push |
| `tools/reference_refresh.sh --no-wait <sha> [columns]` | submits the owed jobs and exits; rerun the same command to fetch and admit |
| `tools/reference_refresh.sh --status <sha>` | prints the saved state of each column and of the admission |
| `tools/reference_refresh.sh --resubmit COL <sha> [columns]` | forgets a dead box's job, then continues |

`columns` is a comma list of `nvidia`, `amd`, `cpu` and `apple`. The default is
all four.

The commit must be on `origin/main`. The refresh records merged commits only.

## What each column runs

Each column runs its existing tool. `--plan` prints the same lane selection
that tool will make.

| column | where | job | lane selection |
|---|---|---|---|
| nvidia | the shared NVIDIA pod, lane tree `/root/mojolearn-$REFRESH_NV_LANE` (default `identity`) | `nvidia_central.sh sync`, then `submit tools/identity_box_job.sh`: `box_merged_verify.sh` builds the merged bindings, runs `python -m mojolearn verify` and records the column | `ONLY_CHANGED=table`, backend cuda |
| amd | the DO AMD droplet | `do_amd_steward.sh update <sha>`, then `run ... tools/identity_box_job.sh amd` (same chain as nvidia) | `ONLY_CHANGED=table`, backend hip |
| cpu | the NVIDIA pod, in the tree the nvidia job built | `submit tools/identity_box_job_cpu_ptx.sh` once the nvidia job has finished | auto reuse, class cpu |
| apple | this Mac, under `tools/mac_slot.py metal` | `record_identity_column.sh apple-<chip>` in `REFRESH_APPLE_TREE` | auto reuse, class apple |

Some rules about the columns:

- **Results persist from the box.** Every box job goes through `box_persist_run.sh`,
  so its results reach R2 from the box itself.
- **cpu can need an nvidia job.** If cpu owes lanes and nvidia owes none, the
  nvidia job still runs, because it builds the tree the cpu record runs in.
- **apple needs a built tree.** `REFRESH_APPLE_TREE` must be a checkout at
  `<sha>` with its Apple bindings already built. The refresh never compiles.
  If apple owes lanes and no tree is given, the column is `blocked`. Build the
  tree and rerun, or drop `apple` from the columns.
- **The PTX column is not part of a refresh.** It needs the release leg's PTX
  set, so `identity_box_job_cpu_ptx.sh` skips it.

The refresh rents nothing and releases nothing. It uses the NVIDIA pod and the
AMD droplet that the orchestrator has already brought up, and it stops with a
message when one is missing. One flag is the exception: `--extend-amd MIN`.
While the AMD job runs, it moves the droplet's lease to now + MIN at each poll.

## Rerunning after a crash

You can rerun the same command after a crash. Every decision is saved to
`~/mojolearn-evidence/reference-refresh/<sha12>/state.json` before the next
step starts.

A rerun keeps the plan it started with and never submits a column a second
time. For each submitted job it reads the job's R2 manifest
(`tools/box_persist.py fetch --list`) and acts on what it finds:

| manifest | what the rerun does |
|---|---|
| none yet (job queued) or `running` | waits |
| `final` with exit 0 | fetches the results and collects the records made at `<sha>` |
| `final` with a non-zero exit | marks the column `failed` |
| `flushed` | marks the column `dead`: the box died mid-job. Use `--resubmit COL` to start it again |

The AMD launcher holds its ssh session for the whole job, so it runs detached
on the Mac. The job ignores HUP on the droplet, and the rerun learns its tag
from the launcher's log.

When admission runs:

- It runs only after every requested column is `done` or `nothing`.
- An admitted commit is pushed. It is never made twice.
- A failed push is retried on the next run.

Before a sync, the refresh checks both sides:

- **The source tree** (`REFRESH_SRC_TREE`, a detached checkout at `<sha>`) must
  be clean, with no untracked files, so nothing but `<sha>` reaches a box.
- **The box tree** has its untracked record outputs removed first
  (`bench/results/identity_break`, `bench/results/reference_admissions`). At
  0.8.65, a resubmit refused "untracked source
  bench/results/identity_break/retained/...".

Records from any other commit found in a fetched results directory are never
admitted.

## What it costs

| plan result | box work |
|---|---|
| no column owes a lane | none |
| a GPU column owes lanes | one job on that box: a merged-bindings build, one verify, and fits of the changed lanes only (`REPEATS=1`) |
| cpu owes lanes | adds one job on the NVIDIA pod that compiles nothing |
| apple owes lanes | one local Metal job |

The boxes are billed by the hour while they are up. The 2026-11-04 rates were:

- RTX 4090 pod: $0.89/hr;
- MI300X droplet: $2.59/h.

A refresh keeps them busy only for its own jobs.

The admission runs on the Mac in `REFRESH_ADMIT_TREE`, which defaults to
`~/mojolearn-wt/reference-admit`:

- **What it does.** `admit_identity_columns.sh --build-host` checks the records,
  regenerates the table, and runs the host self-test and coverage.
- **What it builds.** `--build-host` builds only the host bindings that are
  missing (`estimators`, `x_decomp`) through the compile slot.
- **Forcing a rebuild.** Delete those bindings in the admission tree after a
  change to them.

## How the release freeze consumes it

Once `ref/<sha12>` is merged, the table already holds rows at `<sha>` for every lane whose closure moved. At
the freeze, record each column with `record_identity_column.sh` in its default mode (auto reuse; no
`ONLY_CHANGED`). That mode measures each lane from the newest good evidence of the column's own device
class, so the rows the refresh admitted count. It prints `NOTHING-CHANGED` for every column whose lanes did
not move between the last refreshed merge and the freeze. The release then admits nothing new.

Two limits remain:

- **`ONLY_CHANGED=table` can reselect a refreshed lane.** That mode measures a lane from the oldest
  record among its agreeing columns. If one class kept an older row (for example, apple planned
  `NOTHING-CHANGED` for a CUDA-only change), the lane is selected again.
- **The version bump reaches some lanes.** A lane whose inputs include `python/mojolearn/_version.py`
  (the `hf-*` lanes at 0.8.65) is reached by the freeze's own version bump.

Run the refresh on any commit that reaches a lane before freezing. The release ledger's own checks are
separate from this command.

## Environment

| variable | default | meaning |
|---|---|---|
| `REFRESH_HOME` | `~/mojolearn-evidence/reference-refresh` | state, logs, fetched jobs, the admission inputs |
| `REFRESH_PYTHON` | this tree's pixi Python, else the main checkout's | selection, `box_persist.py`, admission |
| `REFRESH_SRC_TREE` | `~/mojolearn-wt/refresh-src` | the clean detached tree at `<sha>` that is planned and synced |
| `REFRESH_ADMIT_TREE` | `~/mojolearn-wt/reference-admit` | the admission checkout, switched to `ref/<sha12>` from `origin/main` |
| `REFRESH_NV_LANE` | `identity` | the NVIDIA pod lane tree |
| `REFRESH_APPLE_TREE` | unset | a built Mac checkout at `<sha>` for the apple column |
| `REFRESH_APPLE_LABEL` | `apple-<chip of this Mac>` | the Apple column label |
| `REFRESH_POLL_SEC` / `REFRESH_WAIT_MIN` | 300 / 600 | poll interval and how long one invocation waits before it exits (rerun to continue) |
| `REFRESH_SCOPE` | `routine` | the lane scope, the same as the box jobs' default |

Tests: `tools/test_reference_refresh.py`. They run with no box and no network.
