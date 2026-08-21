# ensemble/ — Random Forest and ExtraTrees, the plan

Written 2026-08-21. This promotes ROADMAP.md Phases 2 and 3 out of that file.
Landscape verified this day against upstream source; every claim below carries
its citation. Nothing in this directory is built yet.

## What exists upstream, verified 2026-08-21

**RF training on GPU:**

| library | RF on GPU | ET on GPU | status |
|---|---|---|---|
| cuML | **YES** — native, `cpp/src/randomforest/` + `cpp/src/decisiontree/batched-levelalgo/` | **NO** — open FEA [#8133](https://github.com/rapidsai/cuml/issues/8133) (May 2026), unblocked but unlanded | alive, main @ `c068a20` (2026-08-20), v26.08.00 |
| LightGBM | **YES** — `boosting=rf` (`src/boosting/rf.hpp`) composes with `device_type=cuda` | **YES** — `extra_trees` implemented INSIDE the CUDA split kernel (`cuda_best_split_finder.cu`, `USE_RAND` template) | alive |
| XGBoost | YES as configuration (`num_parallel_tree` loop in `gbm/gbtree.cc`, no dedicated learner; XGBRF* wrappers deprecated 3.4.0) | NO | alive |
| ThunderGBM | yes on paper | no | dead (last release Sep 2022) |
| TF-DF/YDF, H2O, sklearn, CatBoost | no (sklearn forests are Cython, outside Array-API dispatch; CatBoost RF request [#545](https://github.com/catboost/catboost/issues/545) open since 2018) | no | — |

So: **cuML is the only native GPU RF alive**, and **LightGBM is the only
maintained library with ET on a GPU** — and LightGBM's ET is histogram-space
(one random BIN per feature per node), not the Geurts random real threshold.
Nobody ships the histogram-free formulation on a GPU. ROADMAP.md's Phase 2/3
claims all survived verification; what is NEW since its 2026-08-20 rewrite is
cuML FEA #8133: **cuML has now DESIGNED ExtraTrees** (separate estimators, a
random-splitter flag in `DecisionTreeParams`, a "threshold-driven
split-evaluation kernel", sklearn defaults `bootstrap=False`,
`max_features='sqrt'` clf / `1.0` reg), blocked-then-unblocked by the
weighted-RF PR [#8132](https://github.com/rapidsai/cuml/pull/8132) (merged
Jun 1 2026). Road A of Phase 3 is no longer a composition we assemble — it is
mirroring the incumbent's own written design, and possibly their code if the
PR lands before we get there. **Check #8133 for a linked PR before starting
Step 2.**

## Step 0 — the gate, DEFERRED by Andrew on 2026-08-21

The gather probe (ROADMAP.md, "The open gate"): random-index gather of
4M x 32 floats, GPU vs CPU, achieved GB/s. It prices RF, ET and all future
histogram work at once, and it has still never run.

**It no longer gates the port.** Andrew's instruction on 2026-08-21 was to
get the basics mirroring cuML first and to take no timing measurement at
all this round. The probe moves to the first timing round; nothing below
waits on it, and no number from this directory may be quoted until it has
run. See the session log at the end of this file.

Their gather shape, now read rather than assumed, is
`kernels/builder_kernels_impl.cuh:336-341`: per sampled instance, TWO
random-index gathers — one into the feature column via
`dataset.value(row, col)` and one into `dataset.labels[row]` — where `row`
comes from `row_ids`, which at the root is `raft::random::uniformInt`
output (`randomforest.cuh:140-142`), i.e. unsorted uniform draws WITH
replacement. Deeper nodes see an ascending, gapped subset instead, because
their partition is a stable scan. Both shapes belong in the probe.

## Step 1 — Random Forest, from cuML

**Pin**: rapidsai/cuml v26.08.00, commit
`265b9da6a0e75dbef071a3168398b993a5ff6f0e`, cloned read-only on 2026-08-21
to `~/CascadeProjects/upstream/cuml-v26.08.00` — a SEPARATE checkout from
`~/CascadeProjects/upstream/cuml`, which stays at `00094f7` because
`PORTING_RULES.md:0a` pins that one for `dbscan/`, `decomposition/` and
`glm/`. Cite everything `file.cuh:line` against the v26.08.00 path, per
STANDING_ORDERS day-one rule 1.

**Mirror** (their `cpp/src/` prefix dropped, the way `gbdt/` drops
`catboost/cuda/`):

    ensemble/randomforest.mojo            <- randomforest/randomforest.cu + .cuh
    ensemble/decisiontree/decisiontree.mojo   <- decisiontree/decisiontree.cu + .cuh
    ensemble/decisiontree/batched_levelalgo/
        builder.mojo        <- builder.cuh      (NodeQueue, work batching, train loop)
        bins.mojo           <- bins.cuh         (histogram bin types + AtomicAdd)
        objectives.mojo     <- objectives.cuh   (Gini/Entropy; MSE/Poisson/Gamma/IG)
        quantiles.mojo      <- quantiles.cuh/.h (global per-feature quantiles, once per forest)
        split.mojo          <- split.cuh        (Split, evalBestSplit, tie-break total order)
        dataset.mojo        <- dataset.h
        random_utils.mojo   <- random_utils.cuh (FNV-1a32 seed chain)
        kernels/builder_kernels.mojo       <- kernels/builder_kernels.cuh
        kernels/builder_kernels_impl.mojo  <- kernels/builder_kernels_impl.cuh
    (the eight per-objective .cu instantiation TUs collapse into comptime
     specialization — record as a deviation, same shape as gbdt DEVIATION 63)

Their public header `include/cuml/ensemble/randomforest.hpp` shapes the
estimator surface (`RF_params`, `fit`, `predict`, `score`).

All five files the recon had only inferred — `dataset.h`, `quantiles.h`,
`flatnode.h` and the `kernels/*.cu` TUs — have now been read against the
pin; what they contain is in the session log at the end of this file. There
are EIGHT instantiation TUs -- {classification, weighted-classification,
regression, weighted-regression} x {float, double} -- and NINE `.cu` files
once `node-split.cu` is counted. An earlier revision of this line said TEN,
which its own enumeration contradicts.

**Vendor-call mapping** (rule: port the CALL; MAX has no device sort and no
device scan, checked 2026-08-20 — hand-write portably only where nothing
exists, zero warp intrinsics, smem from queried budget):

| their call | site | our arm |
|---|---|---|
| `cub::DeviceSegmentedRadixSort::SortKeys` | quantiles | hand-write portable segmented radix (RAFT `select_radix` precedent: zero warp intrinsics); check whether `gbdt/gpu_util`'s ReorderBins radix machinery lifts into `core/` first |
| `cub::BlockScan::InclusiveSum` (pdf→cdf) | findBestSplits | `gpu.primitives.block` scan (exists; probe before denying anything) |
| `thrust::inclusive_scan_by_key` + tabulate writer | node partition | `gbdt/gpu_util/kernel/scan.mojo` family is the precedent; segmented scan-and-scatter is already ported there — lift shared pieces into `core/`, do not import `gbdt/` from `ensemble/` |
| `cub::BlockReduce` (left counts) | countLocalLeft | block reduce primitive |
| `cuda::std::lower_bound` (bin search) | histogram kernel | plain device binary search (the GBDT evaluator already measured search-shape costs on Apple — see kernel_matrix.quantize_search_for before choosing) |
| `cuda::std::minstd_rand` + `shuffle_iterator` | per-node feature sampling | port the generator exactly — seeds are FNV-1a32(seed, treeid, nodeid), determinism depends on it |
| `raft::random::PCGenerator`, `uniformInt`, `uniform` | bootstrap + quantile subsample | port the generators bit-for-bit (permutation-oracle precedent: `tools/permutation_oracle/` compiled their MT19937-64 — do the same for PCG/minstd if any doubt) |
| double `atomicAdd` on `RegressionBin.label_sum` | histogram | NO float64 on device — fixed-point accumulate (`core/` fixed_point, the cluster/ transfer precedent) |

**Known non-ports, recorded up front:**

- `n_streams` OpenMP fan-out over CUDA streams: no streams on Metal
  (PORTING_RULES.md:195). Faithful port gets zero cross-tree parallelism and
  it changes no OUTPUT bit, for two reasons verified in their source
  2026-08-21: their own non-OpenMP build defines `omp_get_max_threads()` to
  1 (`randomforest.cuh:38-43`) and `set_rf_params` takes
  `min(cfg_n_streams, omp_get_max_threads())` (`randomforest.cu:584`), so a
  cuML built without OpenMP is single-stream regardless of what the user
  passed; and both RNG draws are pure hashes of `(seed, tree_id)`
  (`randomforest.cuh:120-122`) and `(seed, treeid, nodeid)`
  (`builder_kernels.cuh:88`) rather than a stream drawn in order, so tree
  7's rows and node 12's columns are the same values whether 1 stream or 8
  produced them. DEVIATION 117, worth a paper sentence.

  **The earlier justification here was FALSE and is deleted rather than
  annotated:** it claimed "cuML's own docs say `n_streams=1` for
  reproducibility (`randomforestclassifier.pyx:182`)". That file does not
  exist at this pin (both estimators are plain `.py`), its `n_streams`
  docstring is `randomforestclassifier.py:94-95` and says only "Number of
  parallel streams used for forest building", and a grep for
  `reproduc|deterministic` across their ensemble, randomforest and
  decisiontree trees returns nothing of the kind for RF. The claim had also
  propagated into `random_utils.mojo`, where it has been struck too.
- Multi-GPU quantile path (raft comms): out of scope, single-device library.
- Inference: cuML's own path is treelite -> nvForest (in-repo FIL is
  REMOVED on main). We will not port treelite. The C-API `predict()`
  traversal is ported — and it is NOT in `randomforest.cu`, as this line
  used to say. `randomforest.cu` holds only the C-API wrappers; they
  forward to `RandomForest::predict` (`randomforest.cuh:382-436`), which
  calls the per-tree walk `DT::DecisionTree::predict_one`
  (`decisiontree.cuh:370-389`). Both halves are ported, each in the file
  that mirrors it. The gbdt evaluator is oblivious-shaped and does not fit
  deep unbalanced trees. OPEN DECISION, unchanged: whether a device
  evaluator is worth it at v1 or host predict suffices behind the
  estimator — noting that cuML's host walk is a legitimate port target (it
  is what their C-API and their own tests exercise) but is NOT what a cuML
  user timing inference runs, so any inference comparison must say which of
  the two it ran.

**Ship classification first.** `ClassificationBin{count}` is integer
`atomicAdd` — order-independent by construction, bit-identical across
Metal/CUDA/HIP with no fixed-point machinery, and `split.cuh` breaks ties on
a total order (metric, colid, quesval). The cleanest IDENTICAL column in the
library, against NVIDIA's flagship being nondeterministic at its own defaults.
VERIFIED at the pin, 2026-08-21, and the caution is discharged:
`bins.cuh:14` makes `BinCountT` an `unsigned long long int` and `:31` adds
to it with an integer `atomicAdd`. PR #8132's `double` is a separate
`weight` field on `WeightedClassificationBin` (`:55-57`) that the unweighted
path never instantiates — and `bootstrap=True`, their default, forces the
unweighted path regardless, because `tree_sample_weight()` returns
`nullptr` whenever bootstrapping is on (`randomforest.cuh:167`). The
integer-atomics story holds.

What does NOT hold is the 64-bit width: this device has no 64-bit integer
atomic at all (measured — see the session log). A 32-bit counter is exact
here rather than approximate, because a bin count is bounded by
`n_sampled_rows` and `IdxT` is `int` in every cuML RF instantiation.

**Oracles and gates** (rule 4: analytic answers + the competitor's own
output, never real datasets):

1. cuML cannot run on this box. Run THEIR wheel on the NVIDIA column via
   `tools/remote_gpu.sh`, `n_streams=1`, fixed seed, and dump quantiles,
   per-node histograms, chosen splits and leaf values on 4096-row hashed
   fixtures. That is the per-cell oracle, same discipline as CatBoost's.
2. Analytic fixtures: hand-computable Gini/MSE splits, adversarial
   cardinality at every packing/quantile boundary, hashed scattered bins per
   STANDING_ORDERS rule 8, sabotage per mechanism.
3. sklearn is a QUALITY oracle only (cuML splits are quantile-quantized;
   exact match with sklearn is impossible by design) — holdout-accuracy
   bands, the cluster/ pattern.

**Benchmark arms** (rule 5, and gpu-baseline-if-it-exists): sklearn
RandomForest/ExtraTrees have NO GPU path on the Mac (verified: forests are
outside Array-API dispatch), so the honest comparison is our GPU vs sklearn
CPU (`n_jobs=-1`) on the same machine, defaults theirs, interleaved arms, one
process. Secondary column: cuML on the NVIDIA box, their GPU vs our GPU on
that box's own terms.

**Size**: ~130 KB of C++ core, ~3,000 lines of Mojo. MEDIUM-HARD (ROADMAP's
estimate stands).

## The lane split (added 2026-08-21, same day)

RF and ExtraTrees run as TWO PARALLEL LANES, and the split is decided by
file convergence, not by algorithm names (rule 12):

- **This lane (RF)** owns `ensemble/` and any `core/` additions (lifted
  scan/partition/RNG primitives). Deviation numbers **100–129** reserved
  (ledger stood at 90).
- **The ET lane** is `extratrees/PLAN.md`: the histogram-free Geurts
  formulation, a port-from-paper in its own top-level directory, sharing NO
  compute machinery with this lane — no quantiles, no bins, no histograms,
  and `bootstrap=False` means not even the row sampler. Substrate both need
  (tree node, predict, partition) is duplicated there on purpose;
  reconciliation is a merge-time decision. Deviations **130–159**.
- **What CANNOT be a parallel lane** is the cuML-design ET below (Step 2):
  it is a flag on this lane's builder and split kernel — every file it
  touches is a file this lane is writing. It stays a serial follow-on here.
- The Step 0 gather probe is shared and runs ONCE, before either lane
  starts kernels. Own checks only per lane; full suite once at merge.

## Step 2 — ExtraTrees (sklearn-API-parity mode), mirroring cuML's own design

Serial follow-on AFTER Step 1 — see the lane split above for why. This is a
different product from `extratrees/` (random BIN-space thresholds on the
histogram builder vs random real thresholds with no histogram); both can
ship, the way CatBoost-mode and lossguide both shipped in mojotrees.

Order of preference, strict:

1. **If cuML's #8133 has landed code by then: port it.** Same directory,
   their files.
2. **If not: implement their WRITTEN design** — random-splitter flag in the
   ported `DecisionTreeParams`, threshold-driven split evaluation, sklearn
   defaults (`bootstrap=False`, `max_features='sqrt'` clf / `1.0` reg) —
   with LightGBM's CUDA `USE_RAND` kernel
   (`cuda_best_split_finder.cu`: `rand_threshold = NextInt(0, num_bin-2)`,
   scan gated to that bin) as the second citable precedent. Two documented
   upstreams, zero invention; the deviation block says which sentence came
   from which.
3. One open design fork to resolve from the #8133 thread at port time:
   random BIN (LightGBM's histogram-space rule) vs random REAL threshold in
   the node's feature range (Geurts 2006 / sklearn). They are different
   algorithms with different variance behavior; sklearn parity as a quality
   oracle requires the second.

**Road B — the histogram-free Geurts formulation — is no longer a candidate
here: it is the parallel lane**, promoted to `extratrees/PLAN.md` (nobody
has it on a GPU: verified again this round; the only ET-on-GPU
implementations in existence are LightGBM's histogram-space kernel and
cuML's unlanded plan).

## Considered and declined

- **RF mode inside `gbdt/`** (LightGBM `rf.hpp` is ~a screen of code at the
  boosting layer): declined. `gbdt/` mirrors CatBoost and CatBoost has no RF
  (issue #545 open since 2018) — there is no CatBoost source to be faithful
  to, and grafting LightGBM's boosting-layer semantics onto CatBoost's
  learner is exactly the invention rule 1 forbids. RF lives in `ensemble/`
  mirroring cuML, the way `cluster/` mirrors cuVS.
- **Porting LightGBM's CUDA tree learner** to get their ET: declined. ET is
  ~a template flag on top of a full second GBDT learner; the surface is the
  learner, not the flag. Their kernel remains the cited precedent for Step 2,
  not a port target. Known CUDA-learner gaps recorded for the citation file:
  sparse unsupported (#6725), max_depth ignored (#6969).
- **XGBoost as RF source**: declined — RF there is an emergent configuration
  with no dedicated learner to mirror, wrappers deprecated.

## Sequencing against the rest of the library

ROADMAP order stands: Phase 0 bindings and Phase 1 HDBSCAN outrank this
directory. This plan exists so that when RF's turn comes, the first hour is
the gather probe and the second is `git clone` at the pin — not a research
round.

---

# Session log: 2026-08-21, the RF lane opens

## What changed in the brief, and by whom

**Step 0's gather probe is DEFERRED, by Andrew, mid-session.** The
instruction was explicit: *"DO NOT DO NOT check for time in subagents or in
your flow. this should be getting the basics. I will optimize time later.
but you should be mirroring what cuml does."* So no timing number was taken
this round, by this lane or by any subagent it launched, and
`bench/results/GATHER_PROBE_2026-08-XX.md` **does not exist and was not
written**. The ET lane was told this file would gate its kernel work; it
does not, this round. Nothing in this session may be quoted as a
performance result, because nothing in this session measured one.

The gate itself is not cancelled, only deferred — it still prices RF, ET
and all future histogram work at once, and it still has to run before any
number from this directory is quoted. It moves to the first timing round.

## The pin, and why it is not where the brief said to put it

`~/CascadeProjects/upstream/cuml` **already existed** at `00094f7`
(branch-25.08), and `PORTING_RULES.md:0a` pins exactly that checkout as the
upstream for `dbscan/`, `decomposition/` and `glm/`. Checking out a
different tag in it would have silently moved three other sections' upstream
out from under them — file convergence, the one thing rule 12 says predicts
integration pain.

So this lane cloned a SECOND, separate read-only checkout:

    ~/CascadeProjects/upstream/cuml-v26.08.00
    tag v26.08.00, commit 265b9da6a0e75dbef071a3168398b993a5ff6f0e

Every citation in `ensemble/` is against that path and that commit.
`PORTING_RULES.md:0a`'s table needs a row for it at merge time; this lane
did not edit that file because other sessions are in it.

## Five files verified against the pin, the ones the recon flagged as inferred

`dataset.h`, `quantiles.h`, `flatnode.h` and the `kernels/*.cu` TUs were
recorded in the plan above as inferred from names and sizes. All read now:

- **`dataset.h` is 48 lines**: a struct of pointers and strides plus one
  accessor, `value(row, col) = data[row*row_stride + col*col_stride]`. That
  accessor is the memory shape of the whole learner.
- **`quantiles.h` is 20 lines**: `{DataT* quantiles_array (col-major);
  IdxT* n_bins_array}`. The per-feature bin count is a real array, not a
  constant — a feature with fewer distinct values than `max_n_bins` gets
  fewer bins, via a `thrust::unique` inside their batched kernel.
- **`flatnode.h` is 66 lines**: `SparseTreeNode` with private fields behind
  accessors, `left_child_id == -1` marking a leaf and
  `RightChildId() == LeftChildId() + 1`.
- **The ten `kernels/*.cu` TUs** are instantiation-only: classification /
  weighted-classification / regression / weighted-regression x float/double,
  plus `node-split.cu`. They collapse into comptime specialization.

## THE CAUTION IN STEP 1 IS RESOLVED, and the answer is the good one

The plan above warned: *"PR #8132's weighted bin variants widened counts to
double — verify at the pin which bin type the UNWEIGHTED path
instantiates."* Verified at `bins.cuh`:

    using BinCountT = unsigned long long int;            // :14
    struct ClassificationBin { BinCountT count; ... }     // :17-18
    atomicAdd(&address->count, val.count);                // :31

The unweighted classification bin is **a 64-bit unsigned INTEGER with an
integer atomicAdd**. #8132's widening to `double` is a separate `weight`
field on `WeightedClassificationBin` (`:55-57`), which the unweighted path
never instantiates — and `bootstrap=True`, their default, forces the
unweighted path anyway, because `RowSampler::tree_sample_weight()` returns
`nullptr` whenever bootstrapping is on (`randomforest.cuh:167`).

So the integer-atomics story survives contact with the source. **Ship
classification first** stands, and it stands for the stated reason rather
than a hoped-for one.

## MEASURED: this device has no 64-bit integer atomic, and it says so loudly

`ensemble/mojo_only/atomic_width_probe.mojo`. `Atomic.fetch_add` on a
`UInt64` is a hard **compile** error on the Apple target:

    error: Atomic operation is not supported for this type on Apple GPU
    error: failed to legalize operation 'pop.atomic.rmw' ...

This is the best kind of denial — it cannot be mistaken for a working
kernel that drops writes. Two call sites depend on it: `ClassificationBin::
AtomicAdd` (`bins.cuh:31`) and `countLocalLeftKernel`'s
`atomicAdd(reinterpret_cast<unsigned long long*>(&splits[nid].local_nLeft),
...)` (`kernels/builder_kernels_impl.cuh:79-81`).

The resolution is EXACT rather than approximate and the argument comes from
their own types: both quantities are bounded by `n_sampled_rows`, `IdxT` is
instantiated as `int` throughout cuML's RF, so neither can exceed 2^31-1 and
neither can overflow a 32-bit counter. Their 64-bit width buys headroom
their own index type never lets them reach. Priced under DEVIATION 101 (bins
lane) and under the kernels lane's numbers for `local_nLeft`.

## MEASURED, and it is a finding rather than a port detail

Two things came out of `ensemble/mojo_only/split_check.mojo`, both by
running rather than by reasoning:

1. **Their split reduction operator is NOT associative.** `Split::update`
   merges an equivalent threshold range only when two candidates agree on
   `global_nLeft` (`split.cuh:174-177`) and otherwise falls through to a
   `quesval` maximum. Three candidates with equal gain and equal column —
   X(nLeft 5,[1,1],q10), X2(nLeft 5,[3,3],q30), Y(nLeft 7,[2,2],q20) — give
   **bin 2, quesval 20** under one grouping and **bin 3, quesval 30** under
   the other. Arm B of the check runs both groupings and passes only when
   they DISAGREE, so if a future edit accidentally makes the operator
   associative, the check fails and says why.

   Why it matters: `evalBestSplit` merges blocks into the node's global slot
   under `while (atomicCAS(mutex, 0, 1))` (`:251`), which orders blocks by
   arrival. Their comment at `:123-125` says the midpoint rule exists "so
   deterministic tie-breaking does not pick an edge", so determinism is
   plainly their intent.

   **OPEN ITEM (Andrew's, or the NVIDIA column's).** Whether cuML's GPU RF
   is in fact reproducible run-to-run in this tie class is a question about
   THEIR binary, and this repository settles those exactly one way: run
   their binary on a constructed fixture in this tie class on the NVIDIA
   column via `tools/remote_gpu.sh` with `n_streams=1` and a fixed seed, and
   read their per-cell output. Not by reasoning — reasoning is what produced
   the paragraph above, and see finding 2 for what reasoning is worth here.
   Until that runs, this port claims bit-identity **to itself across
   backends**, and makes no claim of bit-identity **to cuML** inside this
   tie class. Recorded so the paper cannot quietly claim the stronger one.

2. **`select_split_range_midpoint` republishes the threshold even on a unit
   range.** The check's arm D(iii) was written predicting *no movement*, on
   the reasoning that their rule maps `[b,b]` to bin `b + (b-b+1)/2 = b` and
   is therefore the identity. It measured **35 of 37 nodes moved**, and the
   measurement was right: `:126-134` picks the bin and then assigns
   `quesval = quantiles[bin]`. The rule is the identity on the BIN INDEX and
   is not the identity on the THRESHOLD.

   The consequence is a semantic worth knowing before porting the kernels: a
   candidate's `quesval` exists only as a tie-break key; the threshold
   finally published for a node is always read back out of the quantiles
   array. The arm now holds that to an exact per-node count rather than to
   "most of them".

## Lane state

DONE and checked on device: `dataset.mojo`, `random_utils.mojo`,
`split.mojo` (+ `split_check.mojo`, four arms, three sabotages, all green),
`atomic_width_probe.mojo`.

IN FLIGHT, one dedicated session each, non-converging files: `bins.mojo` +
`objectives.mojo`; `quantiles.mojo`; `core/block_reduce.mojo` +
`core/block_scan.mojo` + `core/scan_by_key.mojo`; `randomforest.mojo` +
`decisiontree/decisiontree.mojo` + `flatnode.mojo`.

NOT STARTED, deliberately serial because every file it touches is a file
above: `builder.mojo` and `kernels/builder_kernels{,_impl}.mojo`.

Deviation numbers issued from the reserved 100-129 range: 100/102 dataset,
101/105-107 bins+objectives and split, 108-111 quantiles, 112-115 core
primitives, 116-119 estimator surface, 103 + 120-129 held for builder and
kernels.

## A shared-checkout incident, recorded because rule 12 predicted it

Commit `974e4fb` ("The package promised CPU support it does not have"),
written by a concurrent session, swept this lane's in-progress `ensemble/`
files — `PLAN.md`, `dataset.mojo`, `random_utils.mojo`,
`atomic_width_probe.mojo` and four `__init__.mojo` — into a commit about
`gbdt/`, `extratrees/`, `pixi.toml` and `python/`. Nothing was lost and
nothing was corrupted, but a commit that spans four sections cannot be
reverted for one of them, and its message describes none of what it carried.
This is the `git add -A` failure the standing orders forbid by name, from
the other side. No action needed beyond knowing it happened; this lane
commits by explicit path.

---

# The deviation ledger for this directory, RECONCILED

Five lanes ran in parallel and the numbering collided, exactly as rule 3
warns and as the five-lane round before this one did. **The collision was
this lane's fault, not the lanes':** 105-107 were handed to the
bins/objectives lane in its brief and then used by `split.mojo`, written
concurrently by the orchestrator. The bins lane found the clash itself, kept
101 as assigned, and moved to the next free numbers rather than overwriting.

Numbers as actually used, which is what PORTING.md takes:

| # | file | what |
|---|---|---|
| 100 | `dataset.mojo` | `sample_weight` float64 -> Float32 |
| 101 | `bins.mojo` | counter width 64 -> 32 (exact); the four float64 accumulators -> Int32 fixed point; reduction buffers not ported |
| 102 | `dataset.mojo` | Int64/Int32 field widths, spelling only |
| 103 | `kernels/builder_kernels_impl.mojo` | **NOT RETIRED — this row said it was.** 103a: their dynamic shared-memory byte count becomes a comptime `SMEM_BIN_SLOTS` blob, because `stack_allocation` is static. 103b: their runtime `use_global_memory_histogram` argument becomes a comptime parameter, so the two arms are two kernels. 103c: the leaf kernel's blob is capped the same way. All three are live and cited from `builder.mojo` and `builder_kernels_check.mojo`. (103 is ALSO in use in `gbdt/methods/pointwise_scores_calcer.mojo` — a cross-lane collision this ledger does not own.) |
| 104 | `split.mojo` | queried `WARP_SIZE`, not their hardcoded 32 |
| 105 | `split.mojo` | their reduction operator is not associative (MEASURED) |
| 106 | `split.mojo` | their `atomicCAS`/`threadfence`/`atomicExch` mutex as acquire-load + weak relaxed CAS + release store |
| 107 | `split.mojo` | `printSplits` declined, priced |
| 108 | `quantiles.mojo` | the distributed arm not ported; `comm_size` still a parameter that RAISES |
| 109 | `quantiles.mojo` | their two-element `thrust::inclusive_scan` runs on the host |
| 110 | `quantiles.mojo` | `double bin_width` + `round` as a host Float64 index table |
| 111 | `quantiles.mojo` | `cub::DeviceSegmentedRadixSort::SortKeys` hand-written |
| 112 | `objectives.mojo` **and** `core/block_reduce.mojo` | **COLLIDED.** objectives: the eight per-objective `.cu` TUs collapse into comptime specialization (there are NINE `.cu` files -- eight instantiations plus `node-split.cu`; "ten" was wrong here and at the head of this file). core: queried warp width. |
| 113 | `objectives.mojo` **and** `core/block_reduce.mojo` | **COLLIDED.** objectives: `raft::log` -> `std.math.log`. core: 32-bit flush atomic. |
| 114 | `objectives.mojo` **and** `core/block_scan.mojo` | **COLLIDED.** objectives: float64 per site. core: CUB's unpadded `HAS_IDENTITY == false` arm. |
| 115 | `core/scan_by_key.mojo` | three-phase decoupling instead of CUB's lookback; the functor travels in a device buffer |
| 116 | `flatnode.mojo` | private fields, the phantom `LabelT`, their asymmetric widths |
| 117 | `randomforest.mojo` | `n_streams` |
| 118 | `decisiontree.mojo` | phantom `L`; tree dumps not ported; `train_time` inert |
| 119 | `randomforest.mojo` | `fit` not ported; treelite not ported; the device-pointer boundary; float64 per site; summation order in `score` |
| 120 | `kernels/builder_kernels.mojo` | `alignPointer` declined, padding still transcribed |
| 121 | `kernels/builder_kernels.mojo` | `sample_features` — OPENED, then CLOSED |
| 122 | `kernels/builder_kernels.mojo` | multi-GPU pack/unpack declined |
| **123** | `quantiles.mojo` | **NEW, assigned here: the subnormal flush. See below.** |
| 124 | `core/block_reduce.mojo`, `core/block_scan.mojo` | queried warp width (was 112) |
| 125 | `core/block_reduce.mojo` | 32-bit flush atomic (was 113) |
| 126 | `core/block_scan.mojo` | CUB's unpadded `HAS_IDENTITY == false` arm (was 114) |
| 127 | `kernels/builder_kernels_impl.mojo` | no 64-bit device atomic, so `local_nLeft` accumulates through a 32-bit shadow and a third kernel widens it back before the scan reads it |
| 128 | `kernels/builder_kernels_impl.mojo` | 128a: a Mojo struct cannot be a kernel argument, so each launcher's by-value arguments travel in a one-element device buffer. 128b: `NodeSplitPartitionOps` duplicates four `DatasetView` fields and `value()` because `ScanByKeyOps` needs `TrivialRegisterPassable` |
| 129 | `objectives.mojo`, `kernels/builder_kernels_impl.mojo` | 129a: `ObjectiveLike` written down, because Mojo traits are nominal — **CLOSED**, the adapters and launcher overloads are gone. 129b: `ScanBin` trait adapter. 129c: `lower_bound` duplicated per address space |
| 300 | `builder.mojo` | `n_streams` absent from the builder; priced at 117 |
| 301 | `builder.mojo` | the four kernel launches — **CLOSED**, they are wired, and `Builder` is generic over `O: ObjectiveLike` (the "classification-only" caveat this row used to carry is gone with 129a) |
| 302 | `builder.mojo` | the distributed all-reduce path not ported; workspace total unchanged on one device |
| 303 | `builder.mojo` | `TimerCPU`/`train_time` not ported |
| 304 | `builder.mojo` | `enqueue_memset` takes a buffer, not a range, so the histogram workspace is zeroed whole |
| 305 | `randomforest.mojo` | `RowSampler`'s host staging and the `n_selected` field their `resize` carries implicitly |
| 306 | `randomforest.mojo` | the weighted-bootstrap CDF is a sequential HOST scan; theirs is `thrust::inclusive_scan` on device, which is a different summation order |
| 307 | `builder.mojo` | `CRITERION_END` resolves in `Builder`'s constructor. Theirs resolves it in `DecisionTree::fit` (`decisiontree.cuh:251-256`), which also dispatches the objective FAMILY on the same integer-label test; ours has the family fixed by `O`, so the constructor is the first point that sees both `params` and `O.LabelT` |
| 308 | `randomforest.mojo` | `fit_forest` / `RandomForest.fit` take `BinScales`, not an objective. Theirs has no caller-supplied objective at all — `builder.cuh:592-596` builds it from `params` — so after that rewiring the scales are the only thing a caller still supplies |
| 309 | `randomforest.mojo` | the Python layer's four derived-parameter transforms (`randomforest_common.pyx:520-536`, `validation.py:73-79`). Mojo has no `int \| float` parameter, so each fraction arm is its own entry point; the `n_bins` clamp lives in `fit_forest` because that is where `n_rows` is known and this port has no separate Python layer |
| 311 | `randomforest.mojo` | OOB. The mask buffer is allocated by `fit_forest`, not by a caller, and `oob_score_` / `oob_decision_function_` / `oob_prediction_` land on `RandomForestMetaData` rather than on an estimator object -- they are Python ATTRIBUTES there, not C++ members. The per-tree predictions come from the host tree walk rather than `nvforest.predict_per_tree` (which is DEVIATION 119b's declined path), and the whole scoring pass is on the host where theirs is cupy on device |
| 310 | `objectives.mojo` | `10 * numeric_limits<DataT>::epsilon()` as an IEEE-754 literal, because `nextafter` crashes the Metal backend with no line number. Value is bit-identical; checked against `nextafter` on the host in `regression_check.mojo` |

**RESOLVED, 2026-08-21, in the same session:** `core/`'s 112/113/114 were
renumbered to **124/125/126** in `core/block_reduce.mojo` and
`core/block_scan.mojo`, because `objectives.mojo`'s were written first and
because `core/` is one file tree away from the rest of this ledger. The row
above is kept as history; the files now read 124/125/126 and nothing holds
112/113/114 twice.

The lesson is not "assign numbers up front" — they WERE assigned up front. It
is that the orchestrator must not spend numbers from a range it has already
handed out.

## DEVIATION 123 — this GPU flushes float32 subnormals in arithmetic

Found by the quantiles lane, measured rather than inferred, and it is the
only deviation this round that changes an OUTPUT rather than a spelling.

**Measured with a two-kernel probe:** a float32 subnormal survives host ->
device -> kernel -> host BIT FOR BIT, through `enqueue_copy`, a Float32
load/store and a bitcast. But in ARITHMETIC and COMPARISON it is flushed:
the same subnormal compares equal to `+0.0`, to `-0.0`, and to a DIFFERENT
subnormal (`0x006CE3EE == 0x00000001` returned true), and
`subnormal + (-0.0)` returned `+0.0`. The smallest normal, `0x00800000`, is
correct throughout, so the boundary is exactly the subnormal range.

**Where it bites: one line.** `thrust::unique` at `quantiles.cuh:104`, which
collapses duplicate quantiles. Any feature carrying subnormals gets a
SMALLER `n_bins_array[col]` here than on CUDA — measured 5 vs 6, 4 vs 6,
5 vs 7, 5 vs 7 across four shapes. It does NOT touch the sort (integer keys;
the same column matches cell for cell) nor the `quantiles_array` values
(gathered, never arithmetic).

**It cannot be engineered away.** Comparing bit patterns instead would stop
`-0.0`/`+0.0` collapsing, which cuML DOES collapse — so that would be a
different function from theirs, not a fix. The check therefore runs TWO host
references, one IEEE (what cuML computes) and one with the measured flush,
gates the port on what it controls, and prints the divergence per column as
a named line rather than folding it into a pass/fail.

**This belongs in `mojo_only/kernel_matrix.mojo` as a CAPABILITY row** — not
a NUMERIC or SCHEDULING row, because those are knobs this project chooses
and nobody chose this. It is stated in `quantiles.mojo` for now. **OPEN
merge-time item.**

## Other open merge-time items, from the lanes' reports

- **`DatasetView` cannot be passed to a kernel as written.** A Mojo struct
  is refused as a kernel argument unless it conforms to `DevicePassable`,
  which the core lane could not satisfy from outside the stdlib. Their
  workaround (`core/scan_by_key.mojo`) puts the struct in a one-element
  device buffer and loads it in the kernel's first line. `dataset.mojo` is
  `(Copyable, Movable)` and will hit this the first time the histogram
  kernel tries to take it. Decide once, apply everywhere.
- **`ensemble/mojo_only/` can never be `mojo precompile`d.** Every check
  file has a `main()`, which the packager rejects. Three lanes hit this and
  all three reported it as somebody else's failure. Either checks move to a
  sibling outside the package or that command leaves the briefs.
- **`gbdt/gpu_util/kernel/segmented_sort.mojo` is duplicated** into
  `ensemble/mojo_only/segmented_sort.mojo`, deliberately, because
  `ensemble/` must not import `gbdt/`. Collapse at merge.
- **`Split` should reconcile with the ET lane's**, which is writing its own
  under `extratrees/`.
- No `pixi.toml` task exists for any check in this directory; all of them
  run by path. Adding them is a single-file edit nobody could make safely
  while five sessions were open.

## Corrections to this file, made in the same round that falsified them

- The `n_streams` justification cited "cuML's own docs say `n_streams=1` for
  reproducibility (`randomforestclassifier.pyx:182`)". **That file does not
  exist at this pin and cuML says no such thing.** Deleted above, and in
  `random_utils.mojo` where it had propagated. The deviation survives on
  their `#else` branch and on the RNG being a pure hash.
- "Port the C-API `predict()` traversal in `randomforest.cu`" named the
  wrong file; the per-tree walk is `decisiontree.cuh:370-389`. Corrected
  above.
- "the eight per-objective `.cu` instantiation TUs" — there are TEN.
- The bins caution and the Step 0 gate: see the session log above.

---

# State at the end of 2026-08-21: the directory trains

`ensemble/mojo_only/train_check.mojo` grows real trees on the device, end to
end — quantiles, feature sampling, histogram, cdf, gain, split reduction,
partition, leaf. All **ten** checks in this directory pass together:
`split`, `shuffle`, `atomic_width_probe`, `objectives`, `quantiles`,
`core_primitives`, `predict`, `builder`, `builder_kernels`, `train`.

**The identity claim has its first end-to-end evidence.** Two fits of a
3-class, 4-feature hashed dataset produce 53 nodes and 159 leaf values that
are BIT-IDENTICAL. That is the property the classification path was chosen
for — an integer counter under an integer atomic, so the histogram cannot
depend on block arrival order, plus `Split::update`'s total-order tie-break.
It is evidence about THIS port on THIS backend; bit-identity across
Metal/CUDA/HIP, and against cuML itself, still needs the NVIDIA column.

## What is NOT done

0. **BOOTSTRAP NOW WORKS** (2026-08-21, later). `RowSampler`'s default arm
   is wired to a bit-exact port of RAFT's Philox `uniformInt`, held to
   RAFT's own compiled output per cell at 10 cuML call sites. `fit_forest`
   trains cuML's DEFAULT configuration. Twelve checks green.

   **AND IT SURFACED A VERIFIED FINDING ABOUT cuML ITSELF.** Read
   first-hand in RAFT v26.08.00 (`ebf9268`), not taken on report:
   `call_rng_kernel` launches `4 * getMultiProcessorCount()` blocks of 256
   threads (`rng_impl.cuh:70-71`); `rngKernel` gives thread `tid`
   subsequence `tid` and strides by `gridDim.x * blockDim.x`
   (`rng_device.cuh:680-694`); the SM count is
   `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)`
   (`cudart_utils.hpp:301-308`).

   **WITH THE BOUND, which the first write-up of this omitted.** When
   `len <= stride` the loop body runs once per thread and element `idx`
   comes from subsequence `idx`, draw 0 -- independent of `stride`, hence
   identical on every GPU. The dependence appears only for `len > stride`.
   So: cuML's bootstrap sample is identical across NVIDIA parts when
   `n_sampled_rows <= 4 * SM_count * 256`, and differs above it.
   `4*SM*256` is 59,392 on an L4, 81,920 on a V100, 110,592 on an A100,
   131,072 on a 4090, 135,168 on an H100. A 50k-row fit is reproducible
   everywhere; covtype (581,012) and epsilon (400k) are not.

   DEVIATION 184 freezes 4x108x256 and prices the loss: below the threshold
   we match cuML on ANY part, above it we match a 108-SM part.

   Second finding, same read: cuML **truncates the user's 64-bit seed to 32
   bits** (`randomforest.cuh:121` calls `fnv1a32` directly rather than
   `fnv1a32_combine`), so `seed = 0` and `seed = 2^32` give the identical
   forest. Their per-NODE chain does fold both halves; the asymmetry is
   theirs and is transcribed.

1. **Regression does not train.** `Builder` is classification-only. Theirs
   is `Builder<ObjectiveT>`; ours cannot be until `objectives.mojo` declares
   a trait the launchers can dispatch on — Mojo traits are nominal, so the
   launchers are overloaded on the concrete objective type instead. One
   change deletes two adapters, six launcher overloads and this restriction
   together. It is the single highest-value cleanup left.
2. **Weighted bootstrap and zero-weight row removal** are not ported; both
   need `sample_weight`, which this port does not accept, and the first also
   needs a float64 prefix scan. Both raise by name.
3. **NOTHING HAS EVER BEEN COMPARED AGAINST cuML.** Every check here is
   either an ANALYTIC fixture or a diff against a LIBRARY PRIMITIVE's
   compiled output (CCCL's `shuffle_iterator`, RAFT's Philox). That
   establishes a faithful, self-consistent port. **It is not parity.**
   Parity needs cuML's own per-node histograms, chosen splits and leaf
   values, dumped on the NVIDIA column via `tools/remote_gpu.sh`, and that
   has never been run. `bench/results/` still contains no RF file.
4. **No accuracy comparison** against sklearn as a quality band.
5. **No timing measurement of any kind exists**, by instruction. The Step 0
   gather probe still has not run. Nothing here may be quoted as a
   performance result, in either direction.

## Merge-time items still open

- `DatasetView` is not `DevicePassable`; every kernel takes it through a
  one-element device-buffer blob. Making it `TrivialRegisterPassable` would
  delete that indirection and the four duplicated fields in the scan functor.
- `lower_bound` needs an `address_space` parameter in `builder_kernels.mojo`;
  the kernels file carries a duplicate for want of it.
- `Bin` could conform to `core/block_scan.BlockScanElement` directly,
  deleting the `ScanBin` adapter.
- The subnormal-flush row (DEVIATION 123) belongs in
  `mojo_only/kernel_matrix.mojo` as a CAPABILITY row.
- `ensemble/mojo_only/segmented_sort.mojo` duplicates
  `gbdt/gpu_util/kernel/segmented_sort.mojo`.
- `ensemble/mojo_only/` can never be `mojo precompile`d while its checks
  carry `main()`.
- No `pixi.toml` task exists for any check here; all ten run by path.


---

# THE SECOND NUMBERING COLLISION, and it was the same mistake

`builder.mojo` was written with DEVIATIONS 130-134. **The ExtraTrees lane
holds 130-175**, reserved in this very file's lane-split section, and 130-134
are five LIVE deviations of theirs — sklearn's draw order, the feature
sampler, constant-feature rediscovery, the tie-break, and the swap
partition. A verbatim five-way collision.

Renumbered to **176-180**. `ensemble/` had used its whole reserved 100-129
range and this lane took "the next number after 129" without checking who
owned it.

**That is the same mistake as the 112-114 collision earlier the same day**,
and the earlier note — "the orchestrator must not spend numbers from a range
it has already handed out" — was not enough, because this time the range had
been handed out to a lane in a DIFFERENT DIRECTORY and the lane was not
running. The rule that would actually have prevented both:

> Before spending a deviation number, grep the WHOLE REPOSITORY for it, not
> just the files you are writing. A reserved range is only as real as the
> check that reads it.

`ensemble/` owns **100-129** and **300-310**. The 176-199 block it was
given was never used: `builder.mojo` went to 300-304 and the row sampler to
305-306, while 176-196 filled up in OTHER lanes. This ledger now lists what
the code says rather than what the range said, which is the only version
worth having.

Two collisions this ledger does not own and cannot fix from here: **103** is
live in both `kernels/builder_kernels_impl.mojo` and
`gbdt/methods/pointwise_scores_calcer.mojo`, and **119** means one thing in
`randomforest.mojo` and another in `PORTING.md` (`## 119. RUNG 2 IS NOT A
SECOND SEARCHER`). A bare "DEVIATION 119" reference lands on the wrong entry
about half the time. Resolving either needs an orchestrator, not a lane.


---

# What is left, 2026-08-21 end of session

Fourteen checks pass: `split`, `shuffle`, `atomic_width_probe`,
`objectives`, `quantiles`, `core_primitives`, `predict`, `builder`,
`builder_kernels`, `train`, `forest`, `philox`, `regression`, `criteria`.

**Classification and regression forests both train, bagged, and predict.**
All six criteria cuML accepts are exercised; MAE is refused as cuML refuses
it. `RandomForest.fit` works as a method. Two fits of the same data are
bit-identical.

## Functional gaps, in the order they block a user

1. **`sample_weight` is DONE.** This entry used to say it "is not accepted
   anywhere" and that all three dependent arms "raise by name". Both
   sentences were false by the time anyone read them. `fit_forest` takes
   `sample_weight_host`, all four `RowSampler` arms run, and
   `sample_weight_check` covers BOTH directions of their
   `tree_sample_weight` rule (`randomforest.cuh:166-167`): with bootstrap
   off the weights change the forest, with bootstrap on the weighted bin
   and the plain bin give the same forest, so nothing is counted twice.
   The CDF is a host scan rather than their device one (DEVIATION 306).

   **OOB IS DONE TOO.** `store_bootstrap_mask` (`randomforest.cuh:170-183`)
   is two kernels in their order (fill, then scatter a constant, which is
   idempotent so the duplicate ids a bootstrap produces are not a race), and
   it is called from `sample()` AFTER the four-way dispatch as theirs is at
   `:163`, so every arm records a mask. `fit_forest(oob_score=True)`
   allocates the `n_trees x n_rows` buffer and runs `compute_oob_score`,
   the port of `randomforest_common.pyx:695-753`. `oob_check` covers it.
2. **Inference is the host walk only.** cuML's production path is
   treelite -> nvForest and is not ported and not planned. Any inference
   comparison must say which of the two it ran. OPEN: whether a device
   evaluator is worth v1.
3. **float64 features are declined** (no float64 on device), so their
   `-double.cu` instantiations have no counterpart. Multi-GPU likewise.
4. `max_leaves` and `max_batch_size` are honoured and unit-checked in
   `builder_check`, but no end-to-end fit varies them. Cheap to add.

## The gap that is not functional, and matters more than all of the above

**NOTHING HAS EVER BEEN COMPARED AGAINST cuML.** Every check here is an
ANALYTIC fixture or a diff against a LIBRARY PRIMITIVE's compiled output
(CCCL's `shuffle_iterator`, RAFT's Philox). That is a faithful,
self-consistent, well-checked port. **It is not parity, and no number in
this repository says otherwise.**

What would change that, and it is one specific piece of work: run cuML on
the NVIDIA column via `tools/remote_gpu.sh` at `n_streams=1` with a fixed
seed, dump quantiles, per-node histograms, chosen splits and leaf values on
4096-row hashed fixtures, and diff per cell. `bench/results/` still contains
no RF file. It also settles the DEVIATION 105 tie-class question, which
bounds what bit-identity can even mean.

Below that: sklearn as a quality band, then timing -- the Step 0 gather
probe first, since it prices everything downstream. **No timing measurement
of any kind exists**, by instruction, so nothing here may be quoted as a
performance result in either direction.

## Merge-time cleanups, none blocking

- `DatasetView` is not `DevicePassable`; every kernel takes it through a
  one-element device-buffer blob.
- `lower_bound` needs an `address_space` parameter; the kernels file
  carries a duplicate for want of it.
- `Bin` could conform to `BlockScanElement` directly, deleting `ScanBin`.
- The subnormal-flush row (DEVIATION 123) belongs in
  `mojo_only/kernel_matrix.mojo` as a CAPABILITY row.
- `ensemble/mojo_only/segmented_sort.mojo` duplicates the `gbdt/` one.
- `ensemble/mojo_only/` cannot be `mojo precompile`d while its checks carry
  `main()`; all fourteen run by path, and no `pixi.toml` task exists for
  any of them.
