# lane/umap-batch-fix, 2026-09-16

lane/umap-batch-determinism measured four batch couplings in UMAP's
saved-model transform and deliberately scoped the repair rather than applying
it. This lane applies all four, in both spellings, and closes the gate that
lane wrote.

## The decision, and who made it, so it can be reversed

Andrew asked for the work to continue and did not choose between the four
repairs. The choice made here was to land ALL FOUR NOW, and the reasoning is
written down so he can disagree with it.

Three of the four move every recorded UMAP transform cell on every column,
which normally means an expensive re-record and would normally make this a
release decision rather than a lane merge. But `release/0.8.7` ALREADY
requires all four record columns to be retaken: `MERGE_SAME` pins the commit
and `lane_revisions`, thirteen lanes go stale, and the wheel's host-family
count changed. Landing these now therefore costs NOTHING EXTRA, and landing
them later costs a whole re-record on Apple, NVIDIA and AMD. This is the cheap
moment and it does not come again.

If that reasoning is wrong, the lane to revert is this one, and the four
changes are four small hunks in `umap/transform.mojo` and its host
restatement.

## The four arms, before and after

CPU host route, `umap/checks/batch_determinism_check.mojo`, one core, 8.6 s.
Both runs are in this worktree, not quoted from the branch:
`~/mojolearn-evidence/umap-batch-fix/before-cpu-host-nsr.log` and
`after-cpu-host-nsr.log`.

| arm | before | after |
|---|---|---|
| SOLO, nsr=5 | True, all 8 rows, largest `-1.9688573` | **False** |
| ORDER, nsr=5 | True, all 8 rows, largest `0.45718908` | **False** |
| COMPANY, nsr=5 | True, `0.069869995` | **False** |
| SOLO, nsr=0 | True, 7 of 8 rows | **False** |
| ORDER, nsr=0 | False | False |
| COMPANY, nsr=0 | True | **False** |
| FLOOR, thousandfold companion mean | True, `0.023` | **False** |
| CLIFF, 10,000 -> 10,001, nsr=5 | `1.3616371` | **`0.0`** |
| CLIFF, 10,000 -> 10,001, nsr=0 | `0.022047043` | **`0.0`** |
| CONTROL, 9,999 -> 10,000 | `0.0` | `0.0` |
| REPEAT | bitwise equal | bitwise equal |
| ULP ladder | fires at 16,384, moves the row by `0.005` | fires at 8,192, moves it by `0.0086` |

The ladder still fires, which is what makes every False above worth reading.
It fires EARLIER and by a slightly larger amount, and that is the fix's doing
rather than noise: see the RNG key below.

Device route on Metal, alone in the slot,
`umap/checks/batch_determinism_device_check.mojo`, 73 s, log
`after-device-metal.log`. `SUMMARY device.nsr5 solo False order False company
False` and the same at nsr=0, and the ULP ladder fires at the same 8,192 ULPs
as the host. Every one of the 32 batch-of-eight cells the device printed is
bit for bit the cell the host printed, at both negative sample rates, so the
repository's cross-route IDENTICAL claim still holds exactly here.

## The four changes

All four are in `umap/transform.mojo` and restated character for character in
`umap/host/umap_oracle.mojo`, whose only textual difference stays the sabotage
arm's `draw_epoch`.

1. **The sigma floor's mean** is the row's own k neighbor distances, not the
   request's. Measured effectively inert on ordinary data by the branch this
   builds on, and repaired anyway, because an inert coupling is still a
   coupling.
2. **The edge schedule's maximum** is the row's own largest membership, not
   the request's.
3. **The negative-sample counter** is keyed on a hash of the row's own k
   neighbor INDICES, not on `row * k + j`, which was a position in the
   request.
4. **The refinement epoch count** with `n_epochs` unset is 100 at every
   request size, not 100 below ten thousand queries and 30 above.

### The RNG key is the indices and NOT the memberships, deliberately

The branch's note suggested "a hash over that row's k (index, weight) pairs".
That was tried first and measured, and it is worse. Hashing the membership
bits makes the negative-sample draw a discontinuous function of the row's own
input: a 4-ULP change to one feature moved the output by `0.146`, where the
shipped-before spelling needed 16,384 ULPs to move it by `0.005`. That trades
a batch coupling for an input discontinuity thirty times larger from a
perturbation four thousand times smaller, which is not a repair.

The neighbor indices are integers and do not wobble. With the index-only key
the ULP ladder fires at 8,192 and moves the row by `0.0086`, the same order as
before. The cost of the choice is that two queries with the identical ordered
k-neighbor list draw the same negative samples; their memberships, their
initialization and their positions still differ, and two queries that are
byte-identical SHOULD get identical answers.

## The epoch cost, measured, not quoted

The branch predicted 3.3x from the epoch counts alone. That is arithmetic, not
a measurement, so `umap/checks/batch_epoch_cost_check.mojo` measures it. CPU
host route, which is the right place: the device route puts only the k-NN on a
`DeviceContext` and runs `refine_transform` on the host exactly as this does,
so refinement time is host time on both routes. 1,000 training rows in 8
features, k=10, `negative_sample_rate=5`, `n_epochs` unset, one core, two
repetitions of each cell on each build.

| size | before | after | ratio |
|---|---|---|---|
| 5,000 queries (100 epochs both sides) | 2215, 2342 ms | 3243, 3282 ms | 1.46x, 1.40x |
| 10,001 queries (30 epochs before, 100 after) | 2252, 2219 ms | 6979, 6731 ms | 3.10x, 3.03x |

**3.07x above the old boundary**, slightly UNDER the 3.3x predicted, because
the k-NN and the setup are a fixed cost the epochs do not touch.

**1.43x BELOW the old boundary, which nobody predicted.** That is change 2:
with a per-row maximum every row's strongest edge has `scaled == 1.0` and
fires every epoch, where before only the batch argmax row's did. It is the
intended per-row behavior rather than a side effect, but it is not free and a
transform of any size now costs about half again what it did.

The brief said to stop and say so if the measured cost were far worse than
3.3x. It is not, so this landed.

## What was made to fail

Nothing here is a null that was never seen to fire. Every arm was run against
the unfixed build, in a throwaway worktree at the parent commit, and watched
to fail.

| arm | on the unfixed build |
|---|---|
| SOLO / ORDER / COMPANY | `UMAP transform is batch dependent at nsr5: a query alone does not match the same query in a batch`, exit 1 |
| FLOOR | `ARM FLOOR_ABSURD ... True` then `the sigma floor still reads the whole request's mean`, exit 1 |
| FLOOR_BIND | the pair's two rows are IDENTICAL (`0.84648645` both, the shared batch mean 6000.2) and row 0 ALONE is `0.7788105`; raises `the sigma floor binds on neither row; it was deleted, not made per row` |
| CLIFF | `ARM CLIFF nsr0 ... largest move 0.022047043`, and the EPOCHS arm reads `0.0` because at 10,001 queries the old spelling had already forced 30 epochs, so asking for 30 by name changed nothing. The defect states itself twice. |
| TILE, on Metal | row 0 moves by `0.038`, raises `the k-NN query tile reaches the row` |
| the harness batch part | `BATCH_MOVED:transform:row 0 of 64 alone:output 0 element 0: whole 0xbf99e1c6 vs alone 0xbfb692af`, exit 1, which is the pair the old declaration cited from 2026-09-14 |
| `test_host_surface` | `these lanes moved past the shipped reference and they are still public: ['umap']` |

The sigma floor arm had to be rebuilt, not just re-expected. Its old control
worked by moving the BATCH mean a thousandfold, and after the repair that
control cannot fire by construction, which would have left the arm unable to
fail. The replacement is `_floor_still_binds`: two hand-built neighbor rows
differing only in a last distance the 64-iteration sigma search underflows to
exactly zero, so their unfloored sigma is bit identical and any difference
between them is the floor and nothing else. They differ (`0.7788105` against
`0.88249964`), and row 0 alone is bitwise equal to row 0 in the pair. That is
the floor still binding, per row, which a repair that simply deleted the floor
would fail.

## The runtime silence, and the residual

The brief required that whatever batch dependence remains be gone or refused
BY NAME. It is gone, at every size measured: 1, 8, 300, 5,000, 9,999, 10,000,
10,001 queries on the host route, and 1, 8 and 300 on Metal. The prose that
said otherwise is corrected in every place it appeared rather than left to
contradict the code:

* `umap/transform.mojo`'s module docstring,
* `UMAP.transform`, `UMAP.save` and `UMAP.load` in
  `python/mojolearn/_umap_impl.py`,
* `docs/lanes/CPU_INFERENCE_BOUNDARY_2026-09-15.md`,
* `README.md`, `SUPPORT_MATRIX.md`, `python/mojolearn/ALPHA_API.md`,
  `umap/README.md` (which keeps its 2026-09-05 paragraph as the record of what
  was true then),
* `host_surface.inference_display` and the `HostUMAP` docstring, which are the
  user-facing sentences, the first of them generated into README.md and
  SUPPORT_MATRIX.md through `tools/docs_facts.py --write`,
* the declaration at `tools/identity_break.py`.

**THE ONE RESIDUAL, named rather than hidden.** On the device route
`plan_query_tile` and `knn_device_count` still read `n_queries`, so the k-NN
kernel's tiling and its device split are functions of the request size. The
TILE arm measures that the tiling does not reach a row's answer at 1 against
300 queries on Metal, which crosses the clamp and the multi-tile loop. It does
NOT measure the multi-device split, which needs more than one GPU, and it does
not isolate the tile from other size effects on the fixed build, because after
the repair there are none left to separate. A GPU column's harness batch part
is what would catch it.

## The harness, so this cannot regress unnoticed

`tools/identity_break.py` declared `umap` and `par-graph-umap`
`n/a:batch-dependent-by-contract`, which EXEMPTED them from the batch part.
The exemption is now a PART. `_batch_umap` asks `transform` for each of 64
held-out rows alone against the whole request; `_batch_par_graph_umap` does the
same through `transform_umap`. Run on this build with the metrics and core
host bindings, `umap batch` reads a real hash `39ed4f97e478fd2c`,
`batch_verdict STABLE`, on all nine fixtures; on the unfixed build the same
part reads BATCH_MOVED on all nine.

`par-graph-umap` REFUSES on a CPU-only install, because no host binding
restates the cooperative multi-GPU driver, so its four parts are OWED to a GPU
column.

Shared registry files were touched additively and it is worth saying which,
because `lane/inference-coverage-complete` is working in `host_surface.py` at
the same time: two `LANE_REVISIONS` keys, one `PUBLIC_PENDING_LANES` key, two
batch declarations replacing one exemption, one corrected set difference in
`test_host_surface.py`, and one `inference_display` string. None of it is
coverage work.

## The re-record this owes, measured rather than assumed

Running the umap lane on all nine fixtures against both builds:

| part | before vs after |
|---|---|
| `train` | SAME on all nine, the fit is untouched |
| `model` | SAME on all nine, the saved bytes are the fit |
| `infer` | MOVED on all nine |
| `batch` | BATCH_MOVED -> STABLE on all nine |

So every recorded UMAP TRANSFORM cell moves and no fit cell does. That covers
18 fixture keys in `python/mojolearn/verify_reference/table.json`.

`umap` is a PUBLIC reference lane, so without a declaration a CPU-only
`verify --all` would have read DIVERGENT for it on a machine that is fine.
Both lanes get a `LANE_REVISIONS` entry, which makes those cells read OWED to
the next record instead, and `umap` joins `PUBLIC_PENDING_LANES` as
`stale reference` until that record. `LANE_REVISIONS` held only INPUT changes
until now and its comment said so; the comment now says an arithmetic change
stales a reference the same way and that these two are the first.

**A check that could not have fired.** `test_host_surface` asserts that no
lane with a stale reference stays public. It fired on `umap` as it should. It
could NOT have fired correctly on `par-graph-umap`: the assertion subtracted
`PUBLIC_PENDING_LANES` and the host-only lanes but not the prefix-excluded
ones, so it demanded an entry for a `par-` lane that the same test's own loop
would then have rejected for not being a covered lane. `par-graph-umap` is the
first `par-` lane to get a `LANE_REVISIONS` entry, which is why nobody had met
it. The assertion now subtracts the excluded prefixes.

## Verification scope

This lane's own checks only, never a sweep.

    MODULAR_HOME=<pixi default env>/share/max \
    MAC_SLOTS=4 bash ~/mojolearn-evidence/tools/mac_slot.sh run nice -n 19 \
      mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
      umap/checks/batch_determinism_check.mojo
    ... umap/checks/batch_epoch_cliff_check.mojo
    ... umap/checks/batch_epoch_cost_check.mojo
    bash ~/mojolearn-evidence/tools/mac_slot.sh metal ... \
      umap/checks/batch_determinism_device_check.mojo
    cd python && MOJOLEARN_NUMERIC_MODE=fast python -m pytest -q \
      mojolearn/tests/test_host_surface.py mojolearn/tests/test_verify_all.py \
      mojolearn/tests/test_verify_reference_admit.py
    PYTHONPATH=$PWD/python MOJOLEARN_NUMERIC_MODE=identical python \
      tools/identity_break.py --lanes umap,par-graph-umap --vendor cpu-apple-m4

210 of 211 python tests pass;
`test_cli_compare_exit_codes_are_what_a_stranger_scripts_against` fails in any
fresh worktree for want of a built identical binding and fails identically on
the unfixed tree, so it is not this lane's.

`docs/VERIFICATION_MATRIX.md` is generated and is stale on main for reasons
outside this lane (13 lanes, several sabotage and column counts), so only the
two rows this lane changes were taken from a regeneration rather than
committing the whole sweep. Someone should run
`python3 tools/verification_matrix.py --write` on its own.

One thing noticed and NOT fixed, because it is another lane's:
`python/mojolearn/tests/test_host_surface.py` has no `__main__` block, so the
documented invocation `cd python && python3 -m mojolearn.tests.test_host_surface`
runs nothing and exits 0. It must be run under pytest, or it is a check that
cannot fail.

## Evidence

`~/mojolearn-evidence/umap-batch-fix/`: `before-cpu-host-nsr.log`,
`after-cpu-host-nsr.log`, `after-epoch-cliff.log`, `after-device-metal.log`,
`stock-gate-determinism.log`, `stock-gate-floor.log`,
`stock-gate-floorbind.log`, `stock-gate-cliff.log`, `stock-device-tile.log`,
`stock-cost.log`, `ib-umap-after.json`, `ib-umap-stock.json`,
`ib-allfix-*.json`.
