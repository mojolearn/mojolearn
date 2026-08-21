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

## Step 0 — the gate, unchanged and still unrun

The gather probe (ROADMAP.md, "The open gate"): random-index gather of
4M x 32 floats, GPU vs CPU, achieved GB/s. Half a day. It prices RF, ET and
all future histogram work at once. **Nothing below starts before it runs.**
No `bench/results/` file covers it as of this writing (checked).

## Step 1 — Random Forest, from cuML

**Pin**: rapidsai/cuml v26.08.00 (main was `c068a20`, 2026-08-20, structurally
identical for RF). Cite everything `file.cuh:line` against the pin, per
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

Port-time caveat from the recon: `dataset.h`, `quantiles.h`, `flatnode.h` and
the `kernels/*.cu` TU roles were inferred from names/sizes — re-read those
five small files against the pin before citing them.

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
  it does not matter — the intra-tree grid (up to 4096 nodes x columns x
  row-blocks) is already full, and cuML's own docs say `n_streams=1` for
  reproducibility (`randomforestclassifier.pyx:182`). Our only mode IS their
  reproducible mode. Deviation, priced at ~nothing, worth a paper sentence.
- Multi-GPU quantile path (raft comms): out of scope, single-device library.
- Inference: cuML's own path is treelite -> nvForest (in-repo FIL is
  REMOVED on main). We will not port treelite. Port the C-API `predict()`
  traversal in `randomforest.cu` (host/device flat-tree walk over
  `SparseTreeNode`) — the gbdt evaluator is oblivious-shaped and does not
  fit deep unbalanced trees. OPEN DECISION: whether a device evaluator is
  worth it at v1 or host predict suffices behind the estimator.

**Ship classification first.** `ClassificationBin{count}` is integer
`atomicAdd` — order-independent by construction, bit-identical across
Metal/CUDA/HIP with no fixed-point machinery, and `split.cuh` breaks ties on
a total order (metric, colid, quesval). The cleanest IDENTICAL column in the
library, against NVIDIA's flagship being nondeterministic at its own defaults.
CAUTION from the recon: PR #8132's weighted bin variants widened counts to
double (their own ~20% cost) — verify at the pin which bin type the UNWEIGHTED
path instantiates before claiming the integer-atomics story.

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
