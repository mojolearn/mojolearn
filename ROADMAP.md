# What else the GPU is worth pointing at

**Nothing here is in scope until it is promoted out of this file.** The tree
under `gbdt/` is still a transliteration of CatBoost's symmetric learner and
"COPY, DO NOT IMPROVE" still governs every file in it. This document exists so
a decision made later is made against a written list rather than a remembered
one.

Rewritten 2026-08-20. The previous version ranked Random Forest first and
called it "nearly free from here." **That was wrong on three counts and the
sentences are deleted rather than annotated, per the standing rule.** What
replaced them is below, with the measurement or the source line that killed
each one.

## The thesis this list follows from

The advantage is **access**, not tier. On Apple Silicon, XGBoost, LightGBM and
CatBoost all fall back to CPU, because their GPU backends are CUDA and OpenCL
and neither exists here. There is a processor in the machine that no
competitor can reach at all.

That is categorical rather than incremental, and it does not depend on the GPU
being fast in absolute terms. It also does not depend on trees. **The wedge is
the device, so anything that runs on the device inherits it.**

The comparison that measures it is therefore always **our GPU against their
CPU on the same machine**, never against their GPU on hardware they own.

### The thesis has a leak, found 2026-08-20

**scikit-learn can already reach the Apple GPU for some estimators**, through
Array API dispatch onto a PyTorch MPS tensor. This is not speculative; the
tags are in the installed source. In scikit-learn 1.9, `tags.array_api_support`
is set by:

| reaches Apple GPU today | does not |
|---|---|
| `PCA` (solver not `arpack`/`randomized`) | `KMeans` |
| `Ridge` | `DBSCAN` |
| `GaussianMixture` (`init_params="random"`) | `NearestNeighbors` |
| `LogisticRegression`/`GLM` (lbfgs), `LDA`, `naive_bayes`, `preprocessing` | `RandomForest`, `HDBSCAN` |

Two consequences, and both are binding:

1. **An arm whose scikit-learn counterpart is Array-API-supported must be
   benchmarked against the MPS path, not only against CPU.** `bench_sklearn.py`
   currently runs `PCA(svd_solver="covariance_eigh")` and
   `Ridge(alpha=0, solver="cholesky")` on CPU. Both are supported. The OLS
   ratio and the PCA ratio are therefore measured against a baseline with an
   unused GPU path. This is the same defect as the `n_jobs` leak, caught before
   publication instead of after.
2. **An algorithm on the supported side is a bad target**, because the win
   would have to be over scikit-learn-on-Metal rather than
   scikit-learn-on-CPU. That is a different and much harder fight, and it is
   exactly the fight the thesis says we should not have to have.

Mitigating, and stated fairly: MPS support in PyTorch is incomplete, float32
only, and `array_api_dispatch` is experimental and off by default, so few real
users do this today. A reviewer would. A published claim would not survive it.

## The precedent, on the other side

**RAPIDS cuML** GPU-accelerates most of the scikit-learn surface on NVIDIA. It
does not run on Apple Silicon. The general shape of what this could become is
that library's Mac-shaped counterpart, and gradient boosting is one tile of it
rather than the whole thing.

## The filter every candidate passes through

Two measurements govern everything below, and they have overturned three
confident predictions in two days.

**Arithmetic intensity decides winnability, but it does NOT set the ceiling
where this file first claimed.** The M4 has ~4 TFLOP/s FP32 on the GPU against
~1-1.5 on the CPU with AMX, but **both draw on the same 120 GB/s of unified
memory**. The GPU lever is therefore ~3x on FLOPs and **1.0x on bandwidth**,
so compute-bound algorithms are the winnable ones and that much stands.

**What is deleted: "memory-bound ones cap near parity. This is why k-NN wins
at 1.99x and k-means loses at 0.45x."** k-means was the entire evidence for
that sentence and k-means now WINS AT 2.89x, measured
2026-08-20 (`WINDOW_2026-08-20_pca-arm-split.md`). PCA went the same way,
0.60x to 3.64x. Four k-means lanes and the split-K Gram lane did it, which is
exactly the "matter of tuning" this paragraph said it was not.

The corrected reading: arithmetic intensity says which algorithms have room,
not how much of it a given implementation has already taken. A 0.45x is
evidence about our kernel, not about the hardware ceiling. Every "capped"
claim in this file should be read as unproven until something has been tuned
to a standstill, and nothing here has.

**Occupancy is the denominator, traffic is only the numerator.** The fused
k-NN kernel was predicted 6-10x faster from a traffic argument and measured
**0.85x**, because at `gridDim.x == 1` the block count is
`ceil(n_queries / Mblk)` and ~125 blocks each stream the whole index. Any
estimate that counts bytes without counting blocks is worthless.

**The open gate.** Every histogram-shaped candidate below (RF, ExtraTrees, and
any future tree work) is bandwidth-bound on a scattered gather. Whether the
GPU beats the CPU there depends on **achieved gather bandwidth**, not peak: a
GPU hides random-access latency across thousands of threads where 10 CPU cores
are limited by outstanding misses. **This has not been measured.** One
half-day experiment -- random-index gather of 4M x 32 floats, GPU vs CPU,
achieved GB/s -- prices RF, ExtraTrees and all future histogram work at once.
Nothing in that family should be started before it runs.

---

# The plan, in order

## Phase 0. Ship the arm that already wins. `NearestNeighbors`.

**This is the largest gap in the repository and it is not an algorithm.**
`neighbors/__init__.mojo` is empty. Every entry point under `neighbors/` is a
`*_main.mojo` benchmark or a `mojo_only/*_check.mojo` verifier. There are no
bindings and no estimator type anywhere in the repo. The brute-force
k-NN -- **cannot be called by a user.**

STATUS 2026-08-20: `neighbors/estimator.mojo` now exists and `knn_search` is
callable, so this phase is partly done; building it found two bugs, one of
them arms that silently disagreed about output ORDER. What remains is
bindings. And k-NN is no longer "the only measured win": the board is now
k-means 2.89x, k-NN 1.53x, PCA 3.64x, OLS 4.33x matched, DBSCAN a tie.

Five algorithms are implemented and ONE has a callable surface. That ratio,
not the algorithm count, is what gates a release.

What the surface needs:

- `kneighbors(X, n_neighbors, return_distance)` -- the kernel exists
- `kneighbors_graph` -- sparse wrapper over the above
- `radius_neighbors` -- a threshold rather than a top-k, and **`ball_cover`
  already does radius search** for DBSCAN
- `KNeighborsClassifier` / `KNeighborsRegressor` -- majority vote and mean over
  returned neighbors, both trivial
- metrics beyond L2 (cosine, L1) -- cuVS has them, they are a port not a design

**Difficulty: EASY.** Days, not weeks. Highest value per line on this entire
document, because the hard part is done and measured.

It also **de-risks Phase 1**, which calls the same primitive.

Blocking caveat: `select_warpsort` is ported and cannot be wired
(instantiating it at a launch site crashes `mojo build`), and RAFT's dispatch
prefers that family for every k a user actually asks for. The 1.53x is on
their *second* choice. Ship on the radix path; keep the warpsort blocker
filed.

Second caveat, from `UNWIRED.md:371`: RAFT places k-NN output with `atomicAdd`
and has no index tie-break, so **which of several equidistant neighbors is
returned is not reproducible**. An `IDENTICAL` column cannot cover k-NN
indices until that is fixed, and fixing it is an improvement on the upstream.
Ship the estimator with the limitation documented, not hidden.

## Phase 1. HDBSCAN.

**The highest-value algorithm on this list, and nothing about it is
compromised.** `HDBSCAN` has no Array API path in scikit-learn, cuML has a
complete GPU implementation to mirror faithfully, and it is built on the
primitive Phase 0 just shipped.

**Would anyone use it?** More than anything else here. It is the clustering
step in BERTopic, the standard in single-cell genomics (`scanpy`), and the
default partner to UMAP. scikit-learn only absorbed it in 1.3 and it is slow
there.

Difficulty, per file, from the pinned checkouts:

| file | lines | difficulty |
|---|---|---|
| `cuml/.../hdbscan/detail/reachability.cuh` | 162 | **easy** -- calls `brute_force_knn` at `:110`, then core distances |
| `cuvs/.../cluster/detail/mst.cuh` | 345 | **hard** -- Boruvka, iterative, atomics. The one real kernel |
| `cuvs/.../cluster/detail/agglomerative.cuh` | 328 | **easy** -- `:83` says "Agglomerative labeling on host"; `build_dendrogram_host` copies edges back and runs a plain `UnionFind`. **Sequential Mojo, no kernel** |
| `condense.cuh` | 161 | moderate -- tree traversal |
| `stabilities.cuh` | 216 | moderate |
| `extract.cuh` | 330 | moderate |
| `select.cuh` | 462 | moderate |

~2,000 lines for the core path, comparable to `cluster/`. `soft_clustering.cuh`
(630) and `predict.cuh` (277) are optional and should be deferred.

**Overall: MEDIUM, with exactly one hard kernel.** The Boruvka MST is the risk
and it should be the first thing built, not the last.

Falls out nearly free alongside it: **single-linkage / agglomerative
clustering**, since cuVS's `single_linkage.cuh` is the same substrate. Lower
standalone usage, near-zero marginal cost.

## Phase 2. Random Forest, from cuML. GATED.

**Do not start before the gather measurement.** RF's inner loop is a scattered
gather feeding a histogram accumulate -- the same shape `e4eb7cc` measured as
38 of 41.7 ms per tree in the GBDT path. If the GPU does not clear roughly 2x
the CPU's achieved gather bandwidth, this lands where k-means and PCA
landed, after several thousand lines. NOTE 2026-08-20: both of those have
since been tuned into wins (2.89x and 3.64x), so the warning is now about
COST, several thousand lines before a number, rather than about a ceiling.

Three corrections to what this document used to claim:

1. **It is not "nearly free" and it reuses nothing from `gbdt/`.**
   `catboost_options.mojo:441` refuses `Depthwise` and `Lossguide`; the ported
   learner is oblivious-only. cuML reads raw floats with a per-row
   `lower_bound` into shared quantiles rather than a compressed bin index,
   keeps its histogram in shared memory one column per `blockIdx.y`, and
   batches a work queue of up to 4,096 nodes rather than a tree level. Every
   layer differs. It is a new sibling directory (`ensemble/`) mirroring
   `cuml/cpp/src/decisiontree/batched-levelalgo/`, the way `cluster/` mirrors
   cuVS.
2. **"Trees are independent" is not where cuML's parallelism comes from.**
   `randomforest.cuh:164-166` is `#pragma omp parallel for` over `n_streams`
   **CUDA streams**, default 4. `ctx.stream()` raises on Metal
   (`PORTING_RULES.md:195`). A faithful port gets zero cross-tree parallelism,
   and it does not matter: the intra-tree grid is nodes x columns x row-blocks
   and is already full. The real advantage over the oblivious learner is that
   the node batch is **4,096 wide where depth-6 oblivious is 64**.
3. **RF is not cheaper than boosting, it is more parallel.** cuML grows to
   purity by default (`max_depth = -1`). It does more total work. The GPU/CPU
   *ratio* improves; the absolute time does not.

**The determinism prize, and it is large.** `split.cuh:81-89` breaks ties on a
total order -- metric, then `colid`, then `quesval` -- so the split choice is
independent of the reduction tree. And `bins.cuh`:

- **Classification, `CountBin { int x }`, integer `atomicAdd`.**
  Order-independent by construction. **RF classification is bit-identical
  across Metal / CUDA / ROCm for free**, with no fixed-point machinery at all.
  This would be the cleanest `IDENTICAL` column in the library.
- **Regression, `AggregateBin { double label_sum }`.** float64 atomics, which
  Metal does not have at any speed. Needs the fixed-point accumulation already
  built in `cluster/mojo_only/reduce_by_key.mojo`.

Ship classification first for that reason.

Worth citing beside it: cuML's own documentation says **"For nearly
reproducible results, set `n_streams=1`"** (`randomforestclassifier.pyx:182`).
NVIDIA's flagship RF is nondeterministic at its defaults. Having no streams
stops being a limitation and becomes the feature.

**Difficulty: MEDIUM-HARD, ~3,000 lines.**

## Phase 3. ExtraTrees. Two roads, and they are not the same product.

**Road A -- ride on RF. Cheap, small win.**

cuML's RF learner plus LightGBM's documented `extra_trees` rule: at split
evaluation, check one randomly-chosen threshold per feature instead of scanning
all bins (`config.h:389`). Roughly 150 lines once Phase 2 exists, and it is a
composition of two documented upstreams rather than an invention -- already the
house pattern, where CatBoost-mode mirrors CatBoost and lossguide mirrors
LightGBM.

**But it does not dodge the bottleneck.** LightGBM's `extra_trees` still builds
the full histogram and only skips the bin scan, which is the cheap part. Most
of ExtraTrees' CPU speedup is work the GPU was already hiding. Expect
near-parity with RF, not a step change. Worth having for the API surface;
not worth sequencing ahead of anything.

**Road B -- the histogram-free formulation. The only genuinely novel thing on
this list.**

scikit-learn's ExtraTrees, and the original Geurts, Ernst & Wehenkel (2006)
specification, draw a random *real-valued* threshold in the feature's range at
that node. There is no histogram. Per node the state is roughly **four
accumulators per candidate feature instead of 128 bins x 2**, register or
threadgroup resident. The 3.26M-cell footprint that `e4eb7cc` measured as ~90%
of the per-tree fixed cost **does not exist**, and neither does the per-row
`lower_bound` binary search.

- **Nobody has this on a GPU.** Not cuML, not LightGBM's CUDA or OpenCL
  backend, not RAPIDS, not scikit-learn. Verified by grep across cuML's `cpp/`
  and `python/`.
- **It is not an invention.** The algorithm is published and specified; the
  gap is that no one wrote a GPU version. Porting from a paper is a different
  category from improving on a competitor's source, which is what rule 0
  actually forbids. Say so explicitly in the DEVIATION block rather than
  letting it look like drift.
- **The risk is real.** It is still a gather, so it is still bandwidth-bound.
  It wins on *footprint*, not on arithmetic intensity. The gather measurement
  gates it exactly as it gates RF.

If the library is ever going to contribute something upstream rather than
transliterate, this is the candidate, and it is also the paper hook.

**Difficulty: MEDIUM once RF exists, and the correctness oracle is
scikit-learn's ExtraTrees at a fixed seed.**

---

# Deferred, with the reason recorded

| item | verdict | why |
|---|---|---|
| **Gaussian Mixture Models** | **demoted 2026-08-20** | Best arithmetic intensity of anything on this list -- the full-covariance E-step is `d^2` FLOPs per point per component against k-means' `~d`, and `_gaussian_mixture.py` is already GEMM-shaped (`y = (X @ prec_chol) - (mu @ prec_chol)`). But `mixture/_gaussian_mixture.py:1002` sets `array_api_support` under `init_params="random"`, so **scikit-learn's GMM already reaches the Apple GPU.** Racing scikit-learn-on-Metal, not scikit-learn-on-CPU. Also no faithful upstream: cuML issue #2034 has been open since April 2020 with no branches or PRs, and `branch-0.11` never had a `gmm` directory. The GPU implementations that exist ([PyCave](https://github.com/borchero/pycave), [pomegranate](https://github.com/jmschrei/pomegranate)) are PyTorch wrappers over cuBLAS; the CUDA ones are unmaintained academic code. **THE MPS MEASUREMENT HAS NOW RUN, and it moves this back up.** `bench/results/SKLEARN_GPU_BASELINE_2026-08-20.md`: scikit-learn's Array API path on MPS is **1.22x SLOWER** than the same call on torch's CPU, on both arms tested, non-overlapping ranges. `aten::_linalg_eigh` is not implemented on MPS at all, `Ridge(cholesky)` is refused outright, and `linalg_svd` silently falls back to the host. So "scikit-learn's GMM already reaches the Apple GPU" was the right FACT and the wrong INFERENCE: reaching it is not the same as being fast on it, and the demotion rested on the second. **What is still needed is one measurement, not a decision:** run `GaussianMixture(init_params="random")` under `array_api_dispatch` on mps and see whether the full-covariance E-step's Cholesky is implemented there. If it is refused, or refused-then-slow like PCA, GMM is the best arithmetic intensity on this list with no live competitor. The upstream problem is unchanged and is the real cost: cuML issue #2034 open since April 2020, no faithful CUDA source to mirror. |
| **Spectral clustering** | **skip** | cuML's is two thin files delegating to RAFT's Lanczos. Sparse iterative eigensolver: memory-bound and sequential across iterations. Wrong shape, and it is mostly UMAP's initialization anyway. |
| **Gaussian Processes** | **still the largest single win, still the most work** | O(n^3) Cholesky is close to the ideal GPU workload: dense, regular, high arithmetic intensity. Underserved on every platform, so the gap is wider than for anything else here. Cholesky is a call; numerical robustness, kernel choice and marginal-likelihood gradients are the work. |
| **UMAP / t-SNE** | **hard, do not start here** | Reduce high-dimensional data to two dimensions for plotting. Dominant in single-cell genomics and in inspecting embedding spaces, and cuML accelerates both, so Mac users get the CPU versions and wait. But UMAP needs fuzzy simplicial sets, a spectral initialization and an SGD layout phase; t-SNE needs Barnes-Hut or an FFT approximation. Neither is verifiable against a hand calculation the way a histogram is. **HDBSCAN first -- it is UMAP's usual downstream partner and shares the kNN.** |
| **Bootstrap, permutation tests, Monte Carlo** | **trivial, and still unclaimed** | Embarrassingly parallel by construction, compute-bound if the statistic is, and statisticians run them on laptops constantly. Least glamorous entry, possibly the best ratio of user-visible speedup to difficulty. A good filler between phases. |
| **IVF index for k-NN** | **only if brute force stops winning** | Distances to centroids are a matmul, selecting `nprobe` lists is a top-k, the search inside is brute force. Dense and regular. FAISS-GPU implements IVFFlat and IVFPQ. Measure where exact brute force stops winning first. |
| **HNSW** | **never** | Graph traversal: pointer chasing, irregular access, every hop dependent on the last. Close to the worst case for a wide machine. FAISS ships HNSW CPU-only for exactly this reason. |
| **GBDT pointwise loss extras** (Quantile, MAE, Poisson, Tweedie, Huber, Expectile, Lq, MAPE, LogLinQuantile) | **LANDED 2026-08-21** | All nine train, through the same generic kernel their `PointwiseTargetKernel` dispatches to (`pointwise_targets.cu:447-519`), comptime-specialized instead of template-instantiated (DEVIATION 63). The leaf estimator is CatBoost's choice per loss, not ours: `gbdt/options/loss_description.mojo` ports `SetLeavesEstimationDefault` (`catboost_options.cpp:273-360`), so Tweedie gets Newton at TWENTY iterations, LogLinQuantile and sub-2 Lq get Gradient, and MAE / MAPE / Quantile get the EXACT weighted-quantile estimator -- which meant porting their segmented sort, segmented weight scan, `ComputeNeedWeights` and sixteen-iteration binary search (`exact_estimation.cu`, DEVIATIONS 65-67). Gated by `check-pointwise-target` (per-cell against libm through FFI, six sabotages) and `check-exact-estimation` (analytic weighted quantile, four sabotages). STILL TRUE: zero effect on any current benchmark row; this is admission, not speed. |
| **GBDT MultiClass** | **next, and the hard part is not where I first said** | Scoped against their source 2026-08-21. Four things move. (1) **The der kernels**: `MultiLogitValAndFirstDerImpl` and `MultiLogitSecondDerRowImpl`, about 160 lines of `targets/kernel/multilogit.cu`'s 800 -- the rest of that file is MultiRMSE, MultiCrossEntropy, RMSEWithUncertainty and OneVsAll. (2) **The cursor and the leaf become multi-dimensional**, and this is the DOMINANT cost: leaf values are vectors, so `TAdditiveModel`, the text format `check-model-io` gates, the host apply and the device evaluator all change shape together. (3) **The oracle's `rowSize > 1` branch**: `ComputeSecondDerRowLowerTriangleForAllBlocks` on the device, then a BLOCKED lower-triangular Hessian per leaf. **This was first written up here as the hard part and reading their file killed that** -- their solve is `UpdateMoveDirectionBlockedHessian` (`descent_helpers.cpp:91-117`), twenty-five lines of HOST code calling `SolveLinearSystemCholesky` on a `rowSize x rowSize` dense matrix, and `rowSize` is `n_classes - 1`: six by six for covtype. Our walker is already host-side and already has the diagonal arm theirs branches away from, so this is a second arm on an existing function, not a subsystem. (4) **`stat_count` becomes `1 + n_classes`**, and this is nearly free: our histogram family is `stat_count`-generic because theirs is (`gridDim.z * 2 + SkipFirst`), so only the driver's literal `2` and the two-lane fixed-point magnitude reduce widen. |
| **GBDT querywise / ranking** (QueryRMSE, PairLogit, QuerySoftMax, YetiRank -- `querywise_targets_impl.h:92-271`) | **long-term maybe, dated 2026-08-21** | Genuinely niche: applies only to query-grouped data (`group_id`), which means search and recsys teams, and no benchmark in our arena exercises it. CORRECTED 2026-08-22: the machinery cost is SMALLER than first recorded here -- these four declare `EOracleType::Pointwise` (`querywise_targets_impl.h:300`), so they ride the ported doc-parallel learner; the work is per-query target kernels only. Only `PairLogitPairwise` (`Pairwise`) and `QueryCrossEntropy` (`Groupwise`) need a second learner (`pairwise_oblivious_trees/`), and THOSE two are the worst ratio on the list. Revisit the four cheap ones if a real ranking user shows up; the two expensive ones probably never. |

## The uncomfortable note: MEASURED FALSE, then measured the other way

An earlier version of this section said, from the `e4eb7cc` measurement
(41.7 ms per-tree fixed cost, per-row work 2.1x behind CatBoost CPU), that
**no dataset scale closes the gap on its own**. That was true when measured
and is false now, so the sentence is replaced rather than kept:
`PERF_2026-08-20_fixed-cost.md` has the current decomposition -- per-row
work about 2x FASTER than their CPU, fixed cost 9.43 ms against their 2.20
-- and the alternated windows read 1.25-1.32x ahead at 800k, 1.35-1.51x at
2M, 1.8-2.1x at 4M, MSE matching to eight significant figures. Scale is
exactly what closes the gap; the crossover sits near 436k rows and small
datasets below it are conceded to the launch floor.

What survives of the original point: GBDT remains the hardest, most
competitive target on this list, and finishing it is still not the only
thing between here and a product. Phase 0 says the same thing more bluntly.

## On maintenance, which is the real objection to a broad scope

Most of the treadmill is per REPOSITORY, not per algorithm: language churn,
interpreter releases, packaging, CI. That is paid once whether one algorithm
ships or six.

Per-algorithm carry varies enormously, and not in the direction intuition
suggests. Gradient boosting is *expensive* to maintain because it has a large
parameter surface and accuracy gates against three competitors. k-means has
almost none, because there is nothing in it to drift. So the cost of a broad
scope is concentrated in FIRST implementation and validation rather than in
ongoing carry, and several entries above are cheap on both.

## Is any of this "AI"

Not in the 2026 sense of the word, which means generative models. All of it is
**machine learning and statistics**, which is what scikit-learn is and always
was.

Against: if the audience is investors or press, "AI" is the word that opens
doors and none of this is it.

For: tabular ML is where a large share of deployed, revenue-bearing model work
actually sits. Fraud, credit, churn, demand forecasting, ranking, survival. It
is unglamorous, it is enormous, and its practitioners are exactly the people
sitting in front of Macs running scikit-learn on a CPU while the GPU idles.

There is one real bridge: **k-NN is retrieval**, and retrieval is the
infrastructure under RAG. Fast exact nearest neighbours on a laptop GPU is an
AI-adjacent claim that is honestly earned rather than stretched -- and Phase 0
is what makes it callable.

## On the name

`mojotrees` describes about a third of what is now here: k-means, brute-force
k-NN, PCA, OLS and DBSCAN are already in, and HDBSCAN is Phase 1. A
tree-specific name would misdescribe the library and would have to be
abandoned later at a worse moment.

**`mojolearn` is the right name and is already the one this repository uses.**
It maps onto scikit-learn, which is the one reference point every target user
already has. `mojotrees` remains the name of the superseded GBDT repository.

`cuML` is not a counterexample. NVIDIA can afford an opaque name because
RAPIDS is a marketing umbrella with brand gravity behind it and a sales
organization explaining it. This project has neither, so the name has to do
the work that cuML's does not have to do. "Learn" does that; "cu" only works
because everyone already knows what "cu" prefixes.

Two practical follow-ups, neither urgent:

- **Claim `mojolearn` on PyPI before the first release.** `mojotrees` is
  currently live there and should eventually get a tombstone release pointing
  at the new name rather than being deleted.
- **"Mojo" is Modular's trademark.** The required attribution line is already
  in `NOTICE`. Community packages use the prefix widely, so this is worth
  knowing rather than worth changing course over.
