# WHICH LANES ARE LAUNCH-BOUND (lane/metal-launch-overhead, 2026-09-16)

`docs/lanes/LANE_STATUS_lane-metal-launch-overhead.md` measures what a Metal
launch and a Metal wait cost, and spends that cost model on the byte LM step.
This file spends it on **the rest of the Apple column**, and it answers one
question per lane: is the lane's Metal time a function of a count we could
lower, or of a count its own claim pins?

Everything below is traced from the SOURCE, the method of `a7b3b0393`. Where a
number is a prediction rather than a measurement it says so, and it names the
run that would falsify it.

## 0. The test, and why the grid does not count

From the cost model: an `enqueue_function` costs about **0.30 ms**, a
`synchronize` with pending work about **4.15 ms**, and a `synchronize` on an
empty queue **12.7 us**. The probe's `launch_only` arm enqueues a ONE-thread
kernel and `launch_4096` enqueues a 4096-thread kernel, both draining once at
the end, and the two differ by well under a single wait.

So a lane's Metal bill is

    launches * 0.30 ms  +  waits * 4.15 ms

and **a quantity that only enters `grid_dim` is free**. That is the whole test:

> **A lane is LAUNCH-BOUND when its launch and wait counts are set by an
> ITERATION or STRUCTURE count (boosting rounds, tree depth, Boruvka rounds,
> optimizer iterations, decoder blocks, heads) and the DATA SIZE enters only as
> `grid_dim`, a buffer length, or a scalar the kernel strides over.**

For a launch-bound lane, shrinking the data removes arithmetic that was never
being paid for. This is why the 2026-09-16 fixture shrink bought the neural
lanes nothing on Metal, and section 2 argues it will not have bought the GBDT,
hdbscan or holtwinters lanes anything either.

### 0.1 The control: the per-operation cost is FLAT in the batch size

Multiplying a count by a constant is only legitimate if the constant is a
constant. `PROBE_N` was swept while holding everything else fixed, one binary
built once and run five times under the Metal lock, three repeats each.
Minimum microseconds per operation:

| `PROBE_N` | `launch_only` | `launch_sync` | `sync_only` | `d2h_only` | `launch_4096` |
|---:|---:|---:|---:|---:|---:|
| 32 | 478.6 | 3844.6 | 29.5 | 4473.8 | 530.4 |
| 128 | 279.9 | 4249.7 | 38.8 | 4382.9 | 625.3 |
| 256 | 290.8 | 4592.2 | 39.8 | 4861.5 | 574.0 |
| 480 | 163.7 | 2814.8 | 35.9 | 3058.2 | 150.7 |
| 1024 | 209.6 | 3024.6 | 32.0 | 3332.4 | 211.7 |

**There is no trend in `PROBE_N` and no cliff.** Enqueuing 1024 kernels before
draining costs the same per kernel as enqueuing 32, so there is no queue-depth
backpressure to worry about and `launches * constant` is a valid model. The
spread across the rows tracks the machine's load average, which fell from about
16 to about 10 during the sweep, and not the batch size: the two cheapest rows
are the two largest `PROBE_N`.

Two readings to avoid. This is **not** a test of the 512 Metal command-QUEUE
limit: all 1024 launches go to one `DeviceContext` and therefore one queue, so
that limit is not reached and nothing here speaks to it. And `launch_4096`
tracks `launch_only` across the sweep (150.7 against 163.7 at the cleanest row),
so the 4096x larger grid is worth at most a few hundred microseconds against a
wait's 4.15 ms. Grid is free. That is the claim section 0 rests on and it is
measured, not assumed.

Raw logs: `~/mojolearn-evidence/metal-launch-overhead-2026-09-16/probe/`, five
runs of mine plus three taken independently in another session, all consistent.

### 0.2 The model predicts the byte LM step, and the alternative misses by 20x

`a7b3b0393` traced `(64 + 23*G)*L + (32 + 5*G)` launches per byte LM training
step, which at `G = 1` is 211 at two blocks and 124 at one, and measured the
step on Metal. Those two numbers have never been multiplied together. Doing it
discriminates the two candidate cost models sharply:

| shape | launches | measured resident step | at 4.15 ms per ROUND TRIP | at 0.30 ms per ENQUEUE |
|---|---:|---:|---:|---:|
| 2 blocks, d32, ff64 | 211 | **1.380 s** | 0.88 s | 0.063 s |
| 1 block, d32, ff64 | 124 | **0.849 s** | 0.51 s | 0.037 s |
| 1 block, d16, ff32 | 124 | **0.644 s** | 0.51 s | 0.037 s |

The round-trip model lands within 1.3x to 1.6x and under-predicts, which is the
right direction: real kernels are not one-thread stores, and the composed
operations wait more than once per launch. The enqueue-only model is **20x to
22x low** and is refuted. So a launch on the shipped path is being paid for as a
host round trip, not as a queued enqueue, which is exactly what makes the wait
count and not the launch count the thing to attack.

## 1. GBDT, the largest non-neural block of the Apple column

Ten GBDT lanes total roughly **7,500 s**, about a quarter of the column
(`docs/lanes/LANE_STATUS_lane-identity-fixtures-light.md` sections 1a, 1b).

Traced `gbdt/train.mojo:1967` to `gbdt/methods/doc_parallel_boosting.mojo:921`
to `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:4677`, which
is the default arm (`grow_policy=SymmetricTree`, `use_pointwise_searcher=False`).

**The launch count, per fit:**

    n_trees * [ max_depth * (3B + ~10) + ~10 ]  +  ~5 per iteration

`B` is the number of feature-grouping POLICIES that are live (binary,
half-byte, one-byte), so `B <= 3` **and it is not the feature count**
(`gbdt/gpu_data/feature_blocks.mojo:41`: a policy with no features produces no
block and therefore no launch).

This is not my arithmetic. The repository states it itself, in
`gbdt/methods/greedy_subsets_searcher/structure_searcher_template.mojo:99-144`,
which computes `var launches = 3 * b + 12` per level and says in as many words

> the launch count does not depend on the leaf sizes at all

`grow_tree_schedule(max_depth, n_rows, feature_blocks)` at `:134` even TAKES
`n_rows` and never reads it for the launch count.

**Rows do not appear.** The loop nest is boosting iterations
(`doc_parallel_boosting.mojo:1598`), then depth
(`greedy_search_helper.mojo:5069`), then feature BLOCKS (`:2746`). There is no
host loop over rows or row blocks that launches per block. Rows enter as
`grid_dim` arithmetic, as buffer sizes, as an `Int32(n_rows)` argument the
histogram kernels stride over internally (their grid is
`(groups*replicas, n_live, stat_count)`, with no row term at all), and as two
step thresholds.

**The two thresholds, and why the shrink did not cross either.**

| constant | value | file | effect |
|---|---:|---|---|
| `GATHER_INPLACE_SIZE` | 1024 | `gbdt/methods/greedy_subsets_searcher/kernel/split_points.mojo:592`, tested `:829` | `max_leaf_rows <= 1024` takes the 1-launch inplace arm |
| `FAST_SORT_SIZE` | 500000 | `gbdt/gpu_util/kernel/reorder_one_bit.mojo:78`, tested `gbdt/gpu_util/kernel/reorder_single_pass.mojo:538` | above it, 1 launch; below it, `launch_stable_partition` is 3 |

The driver passes `max_live_rows = n_rows` (`greedy_search_helper.mojo:5067`,
deliberately the row BOUND and not a measured leaf size, so that no host read
happens per level). The four shrunk lanes went **20,000 rows to 1,500**. Both
sizes are above 1024 and both are below 500,000, so **both sides take the same
arm and the launch count is literally identical**. Shrinking further, below
1024, would REMOVE one launch per level; going the other way cannot help.

**What this predicts.** `gbdt-parametric-losses`, `gbdt-nan-modes`,
`gbdt-lossguide-newtoncosine` and `gbdt-pair-logit` were shrunk on evidence of
**1.4x to 2.3x measured on the CPU HOST route**
(`LANE_STATUS_lane-identity-fixtures-light.md`, the table at its line 410). On
the CPU route the row-proportional cost is real and large: `gbdt/train.mojo`
carries `for r in range(n_rows)` host loops at `:1157`, `:1261`, `:1421` and
`:1609`, `doc_parallel_boosting.mojo:1223` another, and quantization copies
`n_rows` floats per feature through pinned staging with a `ctx.synchronize()`
every eight features (`train.mojo:349`, `:358-360`, `:389`).

So the honest prediction is split, and it is falsifiable:

* the **device** bill (launches and waits) is unchanged by the row cut, exactly,
* the **host** bill falls with rows on the Metal route too, because those host
  loops run on the CPU whichever device the kernels go to,
* therefore the Metal gain is bounded by the host share and **must not be
  assumed to be the 1.4x to 2.3x the CPU route showed**.

**Nobody has measured a GBDT lane on Metal before and after the shrink.** That
is one lane at one fixture, two runs, and it is the cheapest outstanding check
in this area:

    mac_slot.sh metal python tools/identity_break.py --lanes gbdt-nan-modes \
      --fixtures base --repeats 2

against the same at the pre-shrink row count. The three lanes where the Metal
transfer WAS checked (`byte-lm` steps 231.8 s to 242.0 s, `samba` worse,
`mamba2-dtlimit` 92.8 s to 90.5 s) all came back at zero or worse, which is
what this section predicts for a row cut as well.

**Where GBDT's launches actually are, if anyone wants them.**

| term | multiplier | file |
|---|---|---|
| per level, default symmetric arm | `3B + 12` | `structure_searcher_template.mojo:99-144` |
| quantization, once per fit | 1 kernel + 2 copies per FEATURE, one wait per 8 | `gbdt/train.mojo:349`, `:358-360`, `:389` |
| Depthwise / Lossguide arms | **per LEAF** | `greedy_search_helper_depthwise.mojo:1317`, `:1661` |
| leaf estimation (anything but RMSE + Newton + 1 iteration + 1 permutation) | per tree PER PERMUTATION | `doc_parallel_boosting.mojo:2039`, `:2254`, gate at `:1465-1479` |
| the leaf partitioner's radix sort | 4 launches per bit of `ceil(log2(n_leaves))` | `gbdt/gpu_util/kernel/radix_sort.mojo:257` |

Device buffers are pooled per FIT, not per tree: about 80 for a whole fit, and
`run_tree_layout_traced` (`greedy_search_helper.mojo:4677-5910`) contains no
`enqueue_create_buffer` at all. Per-tree allocation reappears only on the
non-default arms (`doc_parallel_boosting.mojo:2060`, `:2177`, `:2289`).

## 2. hdbscan and hdbscan-leaf, 910 s, and the floor IS the launch term

`hierarchy/impl/sparse/solver/mst_solver.mojo:242`

    for _i in range(mst_iterations):

wraps about fifteen launches (`:311` through `:442`) plus an inner colour
propagation loop at `:276`

    while not done:

carrying two more (`:279`, `:289`). So

    launches ~= mst_iterations * (15 + 2 * colour_passes)

and rows enter only as `grid_dim`; `colour_passes` grows at most
logarithmically in n.

These two lanes were cut 6000 rows to **4000** and to **2000**, and both are
FLOORED at a **Boruvka round count of 5**, measured, because the round count is
one of the integers the lanes HASH
(`docs/lanes/FIXTURE_SHRINK_SCOPE.md`, section A).

That is the sharpest case in this file. **The one term that would remove
launches is the one the claim pins.** Rows are free on the device; rounds are
the whole device bill; and rounds are the claim. So hdbscan is launch-bound,
the row cut removes no launches, and the lane cannot be cut further on Metal
without deleting what it asserts. Its measured 1.4x and 2.3x are, again, CPU
host route numbers.

## 3. arima and holtwinters: observations are grid, iterations are the bill

`arima/impl/batched_arima.mojo:618`

    for i in range(N):

issues `perturb_kernel` (`:619`), `grad_kernel` (`:626`) and `reset_param_kernel`
(`:631`): **three launches per PARAMETER**, a finite-difference gradient. That
sits under `arima/impl/batched_fit.mojo:414`

    while k <= param.max_iterations:

with a line search at `:451`, `for _t in range(param.max_linesearch):`. So

    launches ~= max_iterations * (1 + max_linesearch) * 3N

The observation count enters only inside `batched_kalman_loop_kernel`'s grid
(`arima/impl/batched_kalman.mojo:881`).

`arima-seasonal-c` (231 s) and `arima-011` (165 s) are therefore launch-bound
in the optimizer, not in the series. `holtwinters` was cut **512 observations
to 128** by the fixture lane; by the same reading that removes grid and not
launches, and no Metal gain is predicted there either.

## 4. The repo-wide wait density, which is where the money is

The cost model says a wait is worth about a dozen launches, so the interesting
static quantity is not launch sites but **how many launches are immediately
followed by a wait**. Counted over every non-check `.mojo` file, an
`enqueue_function` counts as paired when a `.synchronize()` follows it within
fourteen lines with no intervening launch:

| area | launch sites | wait-paired | ratio |
|---|---:|---:|---:|
| `transformer/` | 79 | 34 | 0.43 |
| `spectral/` | 28 | 21 | 0.75 |
| `hdbscan/` | 17 | 13 | 0.76 |
| `training/` | 11 | 8 | 0.73 |
| `decomposition/` | 21 | 12 | 0.57 |
| `cluster/` | 43 | 18 | 0.42 |
| `mamba/` | 134 | 19 | 0.14 |
| `gbdt/` | 248 | 17 | 0.07 |
| `extratrees/` | 35 | 0 | 0.00 |
| whole repo | 1099 | 266 | 0.24 |

The densest single files are
`hdbscan/impl/detail/soft_clustering.mojo` (8 of 9),
`spectral/impl/sparse/solver/detail/lanczos.mojo` (14 of 17),
`transformer/impl/llama/modeling_llama.mojo` (17 of 26) and
`mamba/impl/modules/mamba2.mojo` (8 of 12).

Read this as a MAP, not a bill. It is a static site count, so a site inside a
loop is counted once and a site never reached is still counted. What it is good
for is pointing item 1 of the LANE_STATUS's ranked levers ("the waits inside the
composed operations") at the files where the pattern is densest, and recording
that the GBDT family is NOT one of them: at 0.07 its device cost is launches
and arithmetic, not host round trips.

The counting script is
`~/mojolearn-evidence/metal-launch-overhead-2026-09-16/pairs.py`.

## 5. Sorting the candidates by SHAPE versus SESSION

`a7b3b0393` got this right and it decides everything before a stopwatch is
touched. **A lane whose claim is a SHAPE cannot be shrunk. A lane whose claim
is a SESSION or an INVARIANT can be, as long as the cut does not touch the
count the invariant names.**

| lane group | Apple s | what the claim IS | the launch term | can it be cut? |
|---|---:|---|---|---|
| `byte-lm` | 1995 | a **SHAPE**, the published profile, pinned comptime at `training/byte_lm.mojo:92`, oracle defined against two blocks | blocks | **NO.** It is the shipped shape's only device column |
| `byte-lm-resident` | 1735 | a **SESSION**, the resident export equals the stateless bytes at whatever shape | blocks | **CUT**, 2.25x, `a7b3b0393` |
| `samba`, `samba-untied-dropout-accum` | 2516 | a **STACK**, mamba3 plus attention; the heterogeneous composition is the point | layers | **NO.** Cutting a layer deletes the claim |
| `mamba1/2/3`, `mamba2-dtlimit`, `transformer`, `transformer-window` | 3475 | one block each at a d_model floor | blocks = 1 | **NO axis exists** |
| `gbdt-*`, ten lanes | ~7500 | **INVARIANTS**: loss parametrization, NaN modes, grow policy, pair logit, CTR | `n_trees x max_depth` | rows are free; trees and depth are the bill **and they are also the claim** |
| `hdbscan`, `hdbscan-leaf` | 910 | an **INVARIANT** pinned to the Boruvka round count | rounds | **NO.** The launch term is the hashed integer |
| `arima-seasonal-c`, `arima-011` | 396 | an **INVARIANT**, the seasonal order and (0,1,1) | `iterations x linesearch x params` | **CANDIDATE.** `max_iterations` is a fit knob, not the order the lane asserts |

The one row of that table that is a live candidate and has not been taken is
ARIMA, and it is 396 s. Everything larger is pinned by its own claim. That is
the same conclusion `lane/neural-shape-shrink` reached for the neural family,
now extended across the column: **the Apple column is not expensive because its
fixtures are big, and it cannot be made cheap by making them smaller.** It is
expensive per host round trip, and the only lever that is not also a claim is
the one the LANE_STATUS ranks first, the waits inside the composed operations.

## 6. What this file does NOT claim

* No lane here was measured on Metal before and after by this lane. Sections 1,
  2 and 3 are SOURCE traces plus the measured cost model, and each names the run
  that would falsify it. The prediction they share (a data-size cut buys nothing
  on Metal) has been CONFIRMED three times by others, on `byte-lm`, `samba` and
  `mamba2-dtlimit`, and never yet tested on a GBDT, hdbscan or holtwinters lane.
* Section 4's table is static site counts, not dynamic counts. It maps where to
  look; it does not price anything.
* No shrink is proposed here, so no sabotage gate is owed here. Any cut that
  comes out of section 5 takes the two-sided gate: the arm confirmed DIVERGENT
  at the CURRENT size first, then production STABLE and the arm still DIVERGENT
  at the new size.
