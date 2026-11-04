# Reference refresh after a merge

`tools/reference_refresh.sh [<sha>]` keeps the verifier's reference table
(`python/mojolearn/verify_reference/table.json`) current with main. With no
`<sha>` it targets the tip of `origin/main`. For that commit it does four things:

1. records, on every column, only the identity lanes whose source closure
   differs from their admitted record;
2. admits all the fetched records into the table in one call;
3. commits the admission on branch `ref/<sha12>`;
4. pushes that branch.

The orchestrator merges `ref/<sha12>` into main. The next release freeze then
finds every row already admitted, so identity work does not run at release time.

## When to run it

Run it after merges to main that touch `bindings/`, `core/`, a lane's
Python route under `python/mojolearn/`, `tools/identity_break.py` or anything
else a lane's source closure reaches.

You don't have to decide whether a merge counts. Run `--plan`; it answers per
column with the lanes owed, the lanes reused and the gaps:

```
tools/reference_refresh.sh --plan
SCOPE columns nvidia,amd,cpu,ptx,apple x lanes all (the supported surface)
PLAN nvidia REUSED 419 / RECORD 9 of 572
PLAN amd    REUSED 419 / RECORD 9 of 572
PLAN cpu    REUSED 524 / RECORD 12 of 536
PLAN ptx    GAP 85 owed lane(s) not recorded / REUSED 484 of 569: no sm_80 PTX set: ...
PLAN apple  NOTHING-CHANGED (REUSED 572 of 572)
GAPS ptx=85
```

A merge of docs, evidence or selection tooling plans `NOTHING-CHANGED`
everywhere, and the refresh then submits nothing.

## Several merges: one refresh, at the newest commit

A refresh always targets the newest selected commit. N merges to one family cost
one build and one record per column, not N:

- **No sha means the tip of `origin/main`.** Run the refresh once after a burst
  of merges, not once per merge.
- **Pending older refreshes are superseded.** A refresh of an older commit that
  has not been admitted yet is marked `superseded_by` the target. Its state and
  any records it fetched stay under its home dir, but it is never admitted, and
  the refresh does not wait for its jobs. A job it already started on a box is
  listed (`SUPERSEDED <old> by <new>; not waited for: ...`) and runs to its end.
- **An older request joins the newer pending refresh.** Asking for a commit that
  a pending refresh already contains prints `COALESCED <old> into the pending
  refresh of the newer <new>` and continues that one.
- **Each owed lane is recorded once.** The target records only the lanes whose
  closure differs from their admitted record. The changes of all the superseded
  commits are measured together.
- **Admissions merge before the next refresh.** A pushed `ref/<sha12>` that is
  not merged into the target stops the refresh. Otherwise its lanes would be
  recorded a second time. Merge it, then rerun.
- **A finished newer refresh leaves nothing to do.** Asking for a commit older
  than an admitted refresh is refused.
- A pending refresh of a commit that is neither an ancestor nor a descendant of
  the target stays pending, and the refresh says so. Refreshing the tip of
  `origin/main` covers both.

`--plan` prints the same decisions (`WOULD SUPERSEDE ...`) and saves nothing.

## The scope is the supported surface

The scope is never a recorder default. It is what the release supports:

- **Columns:** every column of the table at `<sha>`, today `nvidia`, `amd`,
  `cpu`, `ptx` and `apple`.
- **Lanes:** every lane the harness ships (`identity_columns.py --scope all`):
  the routine lanes and the neural-training lanes.

The box jobs get the same scope explicitly (`SCOPE=all`, the cpu job's
`ID_SCOPE=all`). Before this, the routine default left 85 neural-training lanes
per column and the whole PTX column unrecorded after the multi-GPU
encapsulation merge.

The scope can only be narrowed, and only with a reason:

- `--scope routine` or `--scope neural-training` narrows the lanes;
- naming fewer columns narrows the columns.

Both need `--reason TEXT`. The reason is stored in the state and written to
the admission README. The plan then prints `NARROWED n owed lane(s) out of
scope` per column. A column the table does not hold, or any other scope, is
refused. A rerun keeps the scope its refresh was planned with.

## Gaps

A gap is a column that owes lanes but has no box or runner to record them:

| gap | why |
|---|---|
| nvidia, cpu, ptx | no shared NVIDIA pod is up (`MOJOLEARN_DEVPOD_STATE/nvc*/state.env`) |
| amd | no AMD droplet is up (`MOJOLEARN_STEWARD_DO_STATE/state.env`) |
| apple | `REFRESH_APPLE_TREE` is not set |
| ptx | no sm_80 PTX set (`REFRESH_PTX_SET_TGZ`, `_SHA256`, `_COMMIT`) |
| any other table column | the refresh has no runner for it (owed count `?`) |

What happens with a gap:

- **Reported.** `--plan` and the run print each gap with its owed count and
  reason, then a `GAPS ...` summary line.
- **Not blocking.** The other columns are still recorded and admitted.
- **Recorded.** The admission README gets a "Reference refresh scope" section
  with the per-column counts (owed, reused, recorded, narrowed out), the gaps
  and the superseded commits. The commit message lists the gaps too.
- **Exit code 4.** A run that admits with gaps exits 4.
- **Cleared on a rerun.** If the box or runner comes up before admission, a
  rerun clears the gap and records the lanes.
- **Filled by the next refresh.** After admission, the lanes still differ from
  the admitted table, so the next refresh at a newer commit records them.

## Usage

| command | what it does |
|---|---|
| `tools/reference_refresh.sh --plan [<sha>] [columns]` | prints lanes owed / reused and the gaps per column; submits and saves nothing |
| `tools/reference_refresh.sh [<sha>] [columns]` | runs the full refresh: submit, wait, fetch, admit, commit, push |
| `tools/reference_refresh.sh --no-wait [<sha>] [columns]` | submits the owed jobs and exits; rerun the same command to fetch and admit |
| `tools/reference_refresh.sh --status <sha>` | prints the saved scope, the state of each column and of the admission |
| `tools/reference_refresh.sh --resubmit COL [<sha>] [columns]` | forgets a dead box's job, then continues |
| `... --scope routine\|neural-training --reason TEXT` | narrows the lanes (see above) |

`columns` is a comma list of `nvidia`, `amd`, `cpu`, `ptx` and `apple`. It
follows `<sha>` (write `origin/main` to name the tip). The default is every
column of the table; fewer columns need `--reason`. `ptx` needs `cpu`, because
it records in the cpu job.

The commit must be on `origin/main`. The refresh records merged commits only.

## What each column runs

Each column runs its existing tool. `--plan` prints the same lane selection
that tool will make.

| column | where | job | lane selection |
|---|---|---|---|
| nvidia | the shared NVIDIA pod, lane tree `/root/mojolearn-$REFRESH_NV_LANE` (default `identity`) | `nvidia_central.sh sync`, then `submit tools/identity_box_job.sh`: `box_merged_verify.sh` builds the merged bindings, runs `python -m mojolearn verify` and records the column | `ONLY_CHANGED=table`, backend cuda |
| amd | the DO AMD droplet | `do_amd_steward.sh update <sha>`, then `run ... tools/identity_box_job.sh amd` (same chain as nvidia) | `ONLY_CHANGED=table`, backend hip |
| cpu | the NVIDIA pod, in the tree the nvidia job built | `submit tools/identity_box_job_cpu_ptx.sh` once the nvidia job has finished | auto reuse, class cpu |
| ptx | the same cpu job, with the release leg's sm_80 PTX set | `PTX_SET_TGZ`, `PTX_SET_SHA256` and `PTX_SET_COMMIT` passed to it (`submit --env`) | auto reuse, class ptx, backend cuda |
| apple | this Mac, under `tools/mac_slot.py metal` | `record_identity_column.sh apple-<chip>` in `REFRESH_APPLE_TREE` | auto reuse, class apple |

Some rules about the columns:

- **Results persist from the box.** Every box job goes through `box_persist_run.sh`,
  so its results reach R2 from the box itself.
- **cpu can need an nvidia job.** If cpu owes lanes and nvidia owes none, the
  nvidia job still runs, because it builds the tree the cpu record runs in.
- **cpu can need an nvidia job, and ptx a cpu job.** If ptx owes lanes and cpu
  owes none, the cpu job still runs, since it records the PTX column after the
  cpu column. One fetch serves both. The records under `cpu-ptx/ptx/` go to the
  PTX column and the rest go to cpu.
- **apple needs a built tree.** `REFRESH_APPLE_TREE` must be a checkout at
  `<sha>` with its Apple bindings already built. The refresh never compiles.
  If apple owes lanes and no tree is given, the column is a gap.
- **ptx needs the release leg's set.** `REFRESH_PTX_SET_TGZ` is the path of a
  `cuda-sm_80` set tarball already on the pod.
  `identity_box_job_cpu_ptx.sh` checks it against `REFRESH_PTX_SET_SHA256`
  and checks that `REFRESH_PTX_SET_COMMIT` is code-equal to `<sha>`. Without a
  set, an owed PTX column is a gap.

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

- It runs only after every requested column is `done`, `nothing` or a gap. A
  `failed` or `dead` column stops it.
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
| `REFRESH_PTX_SET_TGZ` / `REFRESH_PTX_SET_SHA256` / `REFRESH_PTX_SET_COMMIT` | unset | the sm_80 PTX set on the pod for the ptx column |
| `MOJOLEARN_DEVPOD_STATE` / `MOJOLEARN_STEWARD_DO_STATE` | the box tools' defaults | where the refresh looks for a live NVIDIA pod and AMD droplet (gaps) |

`REFRESH_SCOPE` is refused: the scope is the supported surface, narrowed only
by `--scope` with `--reason`.

Tests: `tools/test_reference_refresh.py`. They run with no box and no network.
