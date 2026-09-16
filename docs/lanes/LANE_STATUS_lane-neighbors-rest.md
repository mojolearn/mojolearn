# Lane status: lane/neighbors-rest (2026-09-15)

Andrew, Sep 15 2026: "do the rest of neighbors as a lane". Work through
`neighbors/NOT_IMPLEMENTED.tsv` (31 rows on origin/main at a76c02d27):
implement what a scikit-learn or cuML user would call, bitwise identical on
the GPU with the CPU verifier and public CPU inference, and give every other
row an exact reason.

**STOPPED ON PURPOSE, NOT FINISHED.** The M4's Metal command queue was
leaking (AGXCommandQueue 6754 against a limit of 512, every fit roughly 20x
slow) and a restart was due within the hour, so this lane stopped queuing
for the GPU, pushed everything it could prove, and wrote the resume steps
below. No RunPod pod was ever rented, so nothing is owed on a box and
nothing needs reaping.

## Where it is

Branch `lane/neighbors-rest`, pushed. Nothing merged to main: the CPU column
is owed, and unrun code stays on its branch.

    973f2e01e  triage of the 31 rows
    08ab0b3d0  item 1: the seven brute force metrics
    783e21441  classical_host_gate: record no longer dies on --lane-rule-only
    (this commit)  item 2: kneighbors(X=None), evidence and this status

## Triage (all 31 rows)

Counts: (a) user-facing 2, (b) internal reference plumbing 20,
(c) intentionally excluded 9.

| # | Row (reference symbol) | Class | Decision |
|---|---|---|---|
| 1 | raft select_warpsort `warp_sort_filtered` | b | selector internals; the shipped selectors already return the exact top k |
| 2 | raft select_warpsort `warp_sort_distributed` | b | same |
| 3 | raft select_warpsort `warp_sort_distributed_ext` | b | same |
| 4 | raft select_warpsort `calc_launch_parameter` | b | launch geometry is pinned; no answer depends on it |
| 5 | raft select_warpsort `select_k` host entry | b | chooses between queues; one queue here |
| 6 | raft select_warpsort `csr_layout` | b | the selector's CSR row layout, not sparse input to an estimator |
| 7 | raft select_warpsort `kMaxGridDimY` loop | b | host chunking for a CUDA grid cap |
| 8 | raft select_radix multi-block `radix_topk` | b | one very long row; brute force is one block per query |
| 9 | cuVS `DistanceEpilogue` hook | b | C++ template hook |
| 10 | cuVS brute_force index/search API | b | the C++ index object; the capability (save, load, search on a CPU) ships. Row rewritten to name what is actually absent: their precomputed index norms |
| 11 | cuVS ivf_flat, ivf_pq, cagra | c | split: ivf_flat SHIPS as `IVFIndex`; ivf_pq and CAGRA stay out of scope, approximate by construction |
| 12 | cuML `precomp_lbls=true` | b | MNMG reduction arm |
| 13 | cuML `class_vote_kernel` label_cache | b | where a label array is read from; scheduling |
| 14 | cuML `get_next_usable_stream` | b | CUDA streams |
| 15 | raft `getUniquelabels` on device | b | host sort stands in; same set |
| 16 | cuML `approx_knn_build_index` / `approx_knn_search` | c | an approximate search behind the exact class's name; the capability is `IVFIndex`, asked for by name |
| 17 | cuVS Policy4x4 tile | b | speed only |
| 18 | cuVS `cosine_cutlass_op` | b | CUTLASS hook |
| 19 | cuVS `expensive_inner_loop` | b | compiler hint |
| 20 | cuVS distance table (10 metrics) | **a** | **ITEM 1, DONE (evidence partly owed):** canberra, correlation, jensenshannon, inner_product, braycurtis, hamming, russellrao implemented; haversine refused by name (needs an arcsine no pinned primitive provides); hellinger, kldivergence, jaccard, dice split into their own row, reachable by no public metric name in either reference |
| 21 | cuVS fused arm for L2Unexpanded | b | a missing arm, same distance |
| 22 | cuVS fused arm powf post-processing | c | dead code plus a reference bug on multi-partition input |
| 23 | scikit-learn kd_tree / ball_tree | c | engineering refusal; `algorithm='rbc'` serves the request |
| 24 | cuVS `rbc_all_knn_query` | **a** | **ITEM 2, WRITTEN (evidence owed):** `kneighbors(X=None)` is scikit-learn's all-kNN query on both algorithms; only their convenience overload stays absent |
| 25 | cuVS ball cover `z` (Ptolemaic) bound | b | speed only; the triangle bound is exact |
| 26 | cuVS ball cover post-filter registers | b | only the approximate mode needs it |
| 27 | cuVS ball cover `weight` / `perform_post_filtering` | c | approximate mode |
| 28 | cuVS ball cover asserts | b | guards staging this implementation does not do |
| 29 | rbc metric='cosine', 'sqeuclidean', Lp p < 1 | c | not metrics; the cover prunes on the triangle inequality |
| 30 | `core/row_norms.mojo` stdlib sqrt | b | row was stale: the defect is FIXED; only the duplicated cosine norm kernel remains, and collapsing it moves no bit |
| 31 | scikit-learn `weights=<callable>` | c | reason sharpened: the weights already form on the host, so the real reason is that a caller's arbitrary Python arithmetic cannot carry this tree's identity claim |

## Item 1: the seven brute force metrics (code done, CPU column owed)

`metric='canberra'`, `'correlation'`, `'jensenshannon'`, `'inner_product'`,
`'braycurtis'`, `'hamming'`, `'russellrao'` on `NearestNeighbors`,
`KNeighborsClassifier` and `KNeighborsRegressor`, on the GPU and from a
saved model on a CPU with no GPU. Each is a core and an epilogue in
`metric_distance_kernel`, one thread per cell, features ascending, every
primitive from `checks/numerics.mojo`; the tiled arm's `use_norms = false`
branch takes all seven in both modes, so there is no new arm. The host
oracle calls the same cores over its `List` boundary.
DEVIATIONS 2898 to 2901 (`distance_ops.mojo`, THE SEVEN METRICS): correlation's
in-cell row statistics with the constant-row and flushed-variance refusals,
jensenshannon's negative-entry refusal and rectifier, russellrao's boolean
reading and braycurtis's zero rules, and inner product's select-max as an
ascending select over the stored negation with `weights='distance'` refused.

PROVEN (on the M4, one core; `bench/results/identity_break/2026-09-15_neighbors-rest/README.md`
has the numbers): the Apple column, 63 cells STABLE across train, infer,
model and batch, taken twice and agreeing byte for byte; `check-metric-identical`
with 15 DistanceType values x 1961 cells, 29,415 bit-equal, 0 differ, and every
metric priced against a float64 reference; 14 Python tests; the saved-model
recording, 63 of 63 RECORDED with `reload_equal`; and the existing neighbor and
KDE lanes unchanged on the `base` fixture (0 DIVERGENT, 0 MOVED).

## Item 2: `kneighbors(X=None)` (code written, NOTHING measured)

The all-kNN query: search at `k + 1`, drop each row's own index, drop the
FIRST column instead where duplicates crowded it out (scikit-learn
`_base.py:868-889`), on both 'brute' and 'rbc'. Integer bookkeeping over
slots the search already returned, so no float moves.
`RadiusNeighbors.radius_neighbors(X=None)` still keeps the self edge
(cuVS's policy, DBSCAN's requirement) and both docstrings say so.
`python/mojolearn/tests/test_knn_self_query.py` is written and HAS NOT RUN:
it was queued for the Metal slot when the lane stopped.

## RESUME, for a session with none of this context

Worktree: `git worktree add -b lane/neighbors-rest <dir> origin/lane/neighbors-rest`
(the old one lived in a session scratchpad under /private/tmp and is gone
after the restart). Build once, one core:

    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 bash bindings/build.sh
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=cpu bash bindings/build_core_host.sh

1. ITEM 2 IS UNMEASURED. Run its tests first; if any fail, fix before anything else:

       cd python && python3 -m mojolearn.tests.test_knn_self_query
       cd python && python3 -m mojolearn.tests.test_knn_extended_metrics

2. ITEM 2 HAS NO IDENTITY CELL. Either add a lane or decide, in writing, that
   the equality test above is the claim (the search is the same call, so a
   lane would hash the same arithmetic twice). If a lane is added, it needs a
   batch declaration and a `KNN_METRIC_INPUT`-style entry is NOT needed.

3. THE CPU COLUMN AND THE SABOTAGE COLUMN (the only thing standing between
   item 1 and main). One RunPod CPU pod, from this branch:

       bash tools/runpod_cpu_leg.sh --lane neighbors-rest \
         --cmd-file <the script below> --worktree <this worktree> \
         --include bench/results/identity_break/2026-09-15_neighbors-rest \
         --include bench/results/classical_host/2026-09-15-apple-m4-neighbors-rest \
         --include bench/results/identity_break/2026-09-14_166-lanes \
         --build core --sabotage-build core --vcpu 8 --lease 90 \
         --envs default,test --out <out dir>        # add --rent to create it

   The pod script (it was `scratchpad/nr/pod_leg_cmd.sh`, now gone; rewrite it
   from these steps) must run, with `NEW` the seven lane names:
   `identity_break --lanes $NEW --repeats 2` for the CPU column; the same
   under `MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage
   MOJOLEARN_HOST_ALLOW_SABOTAGE=1` for the sabotage column;
   `identity_break --diff bench/results/identity_break/2026-09-15_neighbors-rest/apple-m4.json <cpu json>
   --lanes $NEW --owed-json owed.json`; `tools/cpu_identity_gate_check.py owed`
   with the production and sabotage JSONs; `classical_host_gate.py check
   bench/results/classical_host/2026-09-15-apple-m4-neighbors-rest` and the
   same with `--expect-mismatch --every-fixture` under the sabotage host set;
   and `python3 -m mojolearn.tests.test_knn_extended_metrics`,
   `test_knn_self_query` and `pytest python/mojolearn/tests/test_host_surface.py`.
   WHAT IT MUST READ: Apple vs CPU IDENTICAL on all 63 cells with NVIDIA and
   AMD OWED; every production cell part MOVED by the sabotage build; the gate
   green in production and mismatching on every fixture under sabotage.

4. THEN MERGE. `git merge origin/main`, `python3 tools/docs_facts.py --check`,
   `python3 packaging/wheel_ci.py pins .`,
   `python3 packaging/wheel_ci.py inventory python/mojolearn`, push HEAD:main,
   confirm it landed, and remove the worktree.

5. OWED BEYOND THIS LANE: the NVIDIA and AMD recordings and columns for the
   seven lanes, at the next release record (no GPU box is rented between
   releases). Evidence goes under
   `bench/results/identity_break/2026-09-15_neighbors-rest/` (columns, with the
   README there) and
   `bench/results/classical_host/2026-09-15-apple-m4-neighbors-rest/` (the
   saved-model recording, one directory per lane and fixture).

## Not done, and not started

`kneighbors_graph` / `radius_neighbors_graph` (scikit-learn's CSR wrappers
over the two queries) are not in the TSV and were not attempted; they are
pure host bookkeeping over calls that already exist, and scipy is not a
dependency, so the return type would have to be decided first.

## DROPPED 2026-09-16. Still owed, and deliberately not taken.

This lane was picked up on 2026-09-16 to close its owed evidence and was
**dropped without recording anything.** Nothing here changed; the branch is
exactly as the 2026-09-15 session left it.

**Why it was dropped.** Metal is the scarcest resource we have, one machine and
one GPU shared by every agent, and a lane's Metal time is for proving that
lane's OWN NEW CELLS rather than working through a backlog of owed columns lane
by lane. Item 2's tests were the only part of this lane's owed work that needed
the GPU, and they were not worth another lock acquisition while a fixture-shrink
lane needed it.

**The x86 CPU column cannot be taken the way step 3 below describes.** That step
rents a RunPod CPU pod, and renting was not permitted in that session. A local
substitute was prepared and NOT run: the Apple M4's own host route (`identical/`
moved aside, `MOJOLEARN_HOST_DIR` pointed at the core host binding), which is a
genuinely independent arithmetic path from Metal and would make the diff a real
two-column claim. **It would NOT discharge the x86 column**, which stays owed to
a CPU pod or the release record.

### Still owed, unchanged

1. The CPU column and the sabotage column (step 3 below).
2. The host gate check, green in production and mismatching on every fixture
   under the sabotage host set.
3. **Item 2 (`kneighbors(X=None)`) is STILL UNMEASURED.**
   `python/mojolearn/tests/test_knn_self_query.py` has never run. Note its
   suite SKIPS itself when no k-NN binding loads (`_binding_works()` swallows
   the exception), so a green-looking run that reports skips is NOT a pass:
   whoever runs it must confirm the tests actually executed.
4. The NVIDIA and AMD columns, to the next release record.

### Item 2's open question, answered in writing

Step 2 below asks for a decision: add an identity lane, or record in writing
that the equality test is the claim. **Recorded here: no new lane is needed.**
Read the code (`kneighbors`, `query_is_train = X is None`): when `X is None` the
query matrix IS the fitted index, handed to the SAME search call, and the only
additions are a search at `k + 1` and integer selection among slots the search
already returned. No new float arithmetic exists for a lane to hash, so a lane
would hash the existing search twice and could not fail independently. The
bit-for-bit equality test against the manual construction is the claim. This is
a decision on the record, not a measurement: the test still has to RUN.

### A correction to this file

The reason given above for stopping on 2026-09-15 ("the M4's Metal command
queue was leaking, AGXCommandQueue 6754 against a limit of 512, every fit
roughly 20x slow") is **wrong and should not be repeated.** Those queues belong
to an Apple SYSTEM SERVICE, not to our processes; measured directly, one of our
lanes held a single queue while DockHelper held thousands, and with our own
lane holding the GPU on 2026-09-16 the count read 1. The committed Apple column
is unaffected: its trustworthiness rests on the three independent checks named
in its README (two full runs agreeing byte for byte, 29,415 of 29,415 cells
against the host oracle, every recorded model reloading equal), not on any
claim about the machine's health.
