# lane/saved-model-reference-gaps

The audit in `docs/lanes/LANE_STATUS_lane-expose-inference-surface.md` and the
registry note at `python/mojolearn/host_surface.py` (`SAVED_MODEL_INFERENCE_OWED`)
say DBSCAN, agglomerative and spectral PREDICT are shipped with their GPU
reference recordings still owed, and that spectral's precomputed-affinity
variant is pending. This file establishes what is actually missing, from the
registry read BY IMPORT and from the records on main, before anything is
recorded.

## How the gap was measured

* The lane registry was read by import, never by grep:
  `python3 -c "import identity_break; sorted(identity_break.LANES)"` reads
  **212** lanes at `06295a5da`. Twelve of them match dbscan, agglomerative or
  spectral.
* Every `bench/results/identity_break/**/*.json` on main was walked and each
  `cells[lane/fixture]` part was classified by whether the recorded value is a
  real digest.

### The first scan was wrong, and the way it was wrong is the point

The first pass accepted a part as recorded when its value was a list of
16-character strings. `n/a:transductive` is **exactly sixteen characters**, so
that scan reported NVIDIA, AMD and Apple columns for every one of these lanes.
It was a check that could not fail. The kept copy is
`~/mojolearn-evidence/saved-model-reference-gaps/gap-scan-LOOSE-WRONG.txt`, and
the corrected scan, which requires `^[0-9a-f]{16}$`, is
`gap-scan-strict.txt` beside it. The corrected scan's first act was to print
the matches rather than a count, which is how the placeholder was seen.

The same false positive is in the record itself: at commit `1eea14f80`
(`bench/results/identity_break/2026-09-14_166-lanes/`) every one of these lanes
carries `"infer": ["n/a:transductive", "n/a:transductive"]` and
`"model": ["n/a:no-save", "n/a:no-save"]` on all three GPU columns, because
`predict` did not exist until 2026-09-15.

## The true gap

No GPU or CPU column holds a real `infer`, `model` or `batch` digest for these
lanes anywhere except the two 2026-09-15 lane records. What exists:

| lane | part | Apple/Metal | NVIDIA | AMD | CPU |
|---|---|---|---|---|---|
| `dbscan` | infer, model, batch | 4 fixtures | **none** | none | 4 fixtures |
| `dbscan-brute-l1` | infer, model, batch | 4 fixtures | **none** | none | 4 fixtures |
| `dbscan-weighted` | infer, model, batch | 4 fixtures | **none** | none | 4 fixtures |
| `agglomerative` | infer, model, batch | 4 fixtures | **none** | none | 4 fixtures |
| `spectral` | infer, model, batch | 2 fixtures | **none** | none | 9 fixtures |
| `spectral-precomputed` | infer, model, batch | 2 fixtures | **none** | none | 9 fixtures |

* Apple/Metal, transductive lanes: `bench/results/identity_break/2026-09-15_transductive-predict/apple-m4.json`
  at `ef1647619`, fixtures base, ties, dupes, denormal. Its `vendor` field reads
  `arm64` rather than `apple-m4`, which is why a vendor-keyed search misses it;
  that record's README names it the Metal column.
* Apple/Metal, spectral lanes: `bench/results/identity_break/2026-09-15_spectral-predict/metal/apple-m4.json`
  at `b886dbc97`, fixtures base and ties. It sits one directory deeper than the
  other columns, so a non-recursive glob misses it.
* CPU: `2026-09-15_transductive-predict/cpu-x86.json` (`bd9ef2eee`) and
  `2026-09-15_spectral-predict/cpu-x86.json` (`0a6957015`).

**So the owed column is NVIDIA, and only NVIDIA.** AMD is owed too but is left
alone entirely by Andrew's standing instruction.

## Two things the audit got wrong

1. **Spectral's precomputed-affinity variant is NOT pending.** It is
   implemented and measured. `identity_break.LANES` registers
   `spectral-precomputed`; `python/mojolearn/_spectral_impl.py` accepts
   `affinity="precomputed"` in `fit`, in `predict` (an `(n_new, n_train)`
   matrix) and in `save`/`load`, which write and read the `affinity` scalar;
   `python/mojolearn/tests/test_spectral_predict.py` covers it. Both its CPU
   column (9 fixtures) and its Metal column (2 fixtures) carry real digests.
   Nothing is owed there but the NVIDIA column.
2. **The `owed.json` files in those two records are stale.** Both were written
   before their own Metal columns were taken and still list `apple-m4` as
   missing.

## What IS missing besides the column

`host_surface.inference_lanes()` and `tools/classical_host_gate.py`'s `LANES`
agree exactly, at 79 lanes, and none of the four predict lanes is in either.
So there is no `bench/results/classical_host/` recording for them, and there
cannot be one until the gate declares them. That is a code gap, not just a
recording gap, and it is this lane's first change.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

## The defect that explains why nothing had been recorded

`tools/classical_host_gate.py record` was DEAD on main, and had been since
`--lane-rule-only` was added by lane/ties-sabotage on 2026-09-15:

```
if args.lane_rule_only and not args.every_fixture:
AttributeError: 'Namespace' object has no attribute 'lane_rule_only'
```

That flag is declared on the `check` subparser only, so a `record` Namespace
never carries it, and `main()` read it unguarded ahead of every other refusal.
`record` therefore raised before `do_record` ran a line, on any box, for any
lane. It is why `SAVED_MODEL_INFERENCE_OWED` could say "waiting on ONE thing, a
GPU recording" for four lanes and nobody could produce one: the tool that makes
a recording could not start.

It was found the only way it could be found, by a rented GPU box refusing.
The fix is `getattr(args, 'lane_rule_only', None)`. Both sides were watched:
the unfixed tool raises here too, and the fixed one reaches the estimator's
`fit` (`~/mojolearn-evidence/saved-model-reference-gaps/record-crash-proof.txt`),
while `check --expect-mismatch --lane-rule-only dbscan` still refuses the flag
without `--every-fixture`.

## The NVIDIA column

`bench/results/identity_break/2026-09-16_predict-nvidia/`. One RunPod RTX 2000
Ada Generation, sm_89, $0.24/h, seven minutes billed, pod deleted and verified
gone (HTTP 404). **No spec was pinned**: the runner's default RTX 4090 answered
"There are no instances currently available" and nothing was billed for it, so
the leg was re-driven over a list of NVIDIA specs in price order until one
created. That list, not a pin, is what kept this lane off the starvation that
cost a Hot Aisle leg thirty minutes this morning.

    cells=54 stable=54 moved=0 refused=0
    infer: stable=54   model: stable=54   batch: stable=54

Six lanes, nine fixtures, two repeats. Against the Apple/Metal and x86 CPU
columns of the two 2026-09-15 lane records:

| lanes | verdict |
|---|---|
| dbscan, dbscan-brute-l1, dbscan-weighted, agglomerative | IDENTICAL x3 on all 32 infer/model parts and all 16 batch parts those columns carry; the other five fixtures are ONE-COLUMN, because those columns ran four |
| spectral-precomputed | IDENTICAL x3 on base and ties, IDENTICAL x2 on the other seven |
| spectral | ONE-COLUMN on all nine, and NOT a divergence |

`spectral`'s two older columns were excluded by the diff tool's own
`LANE_REVISIONS`, which prints "hashed ... spectral ... at an older lane
revision; its cells there are not compared and read as absent". The lane was
shrunk from 2000 rows to 512 at `e2bb9e541`, and
`git merge-base --is-ancestor e2bb9e541 0a6957015` answers NO, so neither the
CPU column (`0a6957015`) nor the Metal column (`b886dbc97`) was taken at the
published size. **This NVIDIA column is the first spectral column that is**, and
spectral's Apple and CPU columns are owed again, which nothing had recorded.

The batch part's negative control fired on every lane:
`MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1` gives `BATCH_MOVED=6 of 6`, each naming
the element that moved (`dbscan/base ... whole 0x00000001 vs alone 0x00000000`).


## The recording, which is the deliverable

`bench/results/classical_host/2026-09-16-nvidia-predict/`, 36 fixture
directories (four lanes x nine fixtures), recorded on a RunPod NVIDIA
A100-SXM4-80GB (sm_80). Its own README has the detail. The verdicts:

| arm | verdict |
|---|---|
| `check`, x86-64 host bindings on the recording box | `gate verdict IDENTICAL (36 fixtures, exit 0)` |
| `check`, arm64 host bindings on the M4 | `gate verdict IDENTICAL (36 fixtures, exit 0)` |
| sabotage, `--every-fixture`, x86-64 | `EXPECTED MISMATCH SEEN`, `unmoved` EMPTY |
| sabotage, `--every-lane`, x86-64 | `EXPECTED MISMATCH SEEN`, `unmoved` EMPTY |
| sabotage, `--every-fixture`, arm64 | `EXPECTED MISMATCH SEEN`, `unmoved` EMPTY |

`unmoved` EMPTY is the sentence that matters. The rule the lane was given was
that each recorded cell's check must be seen to FAIL before its PASS is
believed, and `--every-fixture` is the only rule that asks that of every cell
rather than of one per lane. All 36 moved, named individually in the reports.

The second NVIDIA column (A100 sm_80) came off the same box. Over all six
lanes and nine fixtures, against the RTX 2000 Ada sm_89 column and the Metal
column of the same tree:

    summary: IDENTICAL=54
    summary (infer/model): IDENTICAL=108
    summary (batch): IDENTICAL=54

## What it cost, and the two things that nearly wasted it

Four rental attempts, three of which billed nothing.

1. **Nothing was pinned, and that is why there is a box at all.** The runner's
   default RTX 4090 answered "There are no instances currently available", and
   so did the RTX 2000 Ada, the A6000, the 4090 again and the A100 PCIe on the
   second round. The A100 SXM answered. A leg that had pinned one spec would
   have starved, which is what happened to a Hot Aisle leg this morning.
2. **The first box lost its recording phase to the `record` crash above**, and
   its host builds to a second defect: this lane's body exports
   `MOJOLEARN_GPU_ARCHS` for the GPU builds, and
   `bindings/build_host_family.sh` refuses a Linux CPU build that carries one.
   All four host builds exited 2 in zero seconds and took the check and both
   sabotage arms with them. Both are fixed in
   `tools/saved_model_predict_nvidia_leg.sh` (`env -u MOJOLEARN_GPU_ARCHS`),
   along with the commit witness the runner does not write for this payload
   and the two diff phases that cannot run on a box `git archive` gave no
   `bench/results/` to.
3. A third box was killed by SIGTERM two minutes into its payload and billed
   three minutes for nothing. Its pod was terminated cleanly and verified
   gone. The leg is launched through `os.setsid` now so a signal aimed at the
   harness's background tasks cannot reach it.

Billed: about seven minutes of RTX 2000 Ada at $0.24/h, three minutes of the
same for nothing, and about ten minutes of A100 SXM at $1.59/h. Every pod was
deleted and verified gone (HTTP 404), and the account listed no live pod at
the end.

## Verification scope

`python3 tools/lane_select.py --changed-since origin/main` reads:

    # python/mojolearn/host_surface.py: declares the CPU surface itself: every lane
    # tools/classical_host_gate.py: NOT ATTRIBUTABLE: no lane's derived source set names it, so every lane
    # FALLING BACK TO EVERY LANE.
    # 212 of 212 lanes selected

That is a full sweep and it was NOT run on the Mac, which is the rule. What
ran instead is this lane's own six lanes, on two NVIDIA columns and a Metal
column, plus its four gate lanes on two CPU host architectures, plus
`test_host_surface` (155 passed) and `docs_facts --check`. No Mojo source was
changed by this lane, so no other lane's cells can have moved.

Two checks in this lane were watched failing before they were trusted:
`test_inference_lanes_are_classical_gate_lanes` fails when one lane is dropped
from the gate table, and `test_recordings_and_columns_exist` fails on a
one-character change to the recording's path.

## Still owed

* The AMD recording and AMD identity cells, at the next release record. AMD
  was left alone entirely, by instruction.
* `spectral`'s x86 CPU identity column at the 512-row size. Its Apple column
  was retaken here; the CPU one was not, and the 2026-09-15 one is superseded.
* `kmeans` stays the one entry in `SAVED_MODEL_INFERENCE_OWED`, and it waits
  on a serialization format, not on a box.
* `tools/lane_select.py` cannot attribute `tools/classical_host_gate.py` to
  any lane, so a change there falls back to all 212. That is the selector
  lane's map, not this one's, and it is written down rather than fixed here.
