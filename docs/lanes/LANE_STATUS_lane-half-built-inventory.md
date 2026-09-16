# Lane status: lane/half-built-inventory (2026-09-16)

Andrew, Sep 16 2026: "what do we have half built that needs to be fully built,
and can we finish it?" This lane answers the first half completely and the
second half honestly: it triages every `NOT_IMPLEMENTED.tsv` row in the 25
families nobody had triaged, and it finishes the one class of work that could
be finished today without the Metal lock.

Base: `origin/main` at `bfb8f725a`. Branch `lane/half-built-inventory`.
Worktree `scratchpad/wt-inventory`. Main only (0.8.7); `release/0.8.6` and the
frozen `db9047b9f` were not touched.

## THE CONSTRAINT THAT SHAPED THIS LANE, READ IT FIRST

The 0.8.6 Apple record was running on the Metal slot for this lane's whole
life (`run_apple_groups.sh`, pid 29471, `mac_slot.sh metal`). **No Metal job
was run and the Metal lock was never taken.** No box, GPU or CPU, was rented.
All local work was one core at `nice -n 19`.

That is why this lane ships documentation corrections and a triage rather than
a new capability: **every user-facing row worth building needs a new identity
cell, and a new cell needs a Metal column that this lane could not take.**
Saying so is the result, not an excuse. The ranked list below is what the next
session should build, with the Metal column priced into each estimate.

**A SECOND reason, independent of the first, and the one that decided it.**
Late in this lane the machine reached 53 MB free of 16 GB at a load average of
42 on 10 cores, from too many building agents at once, and the coordinator
stopped all compiles. **This lane had started none and so had nothing to
abandon**, but the instruction stands for whoever picks the list up: do not
start item 1 or item 2 until the machine is quiet AND the Metal slot is free.
Both items are compile-heavy (`_mojolearn_estimators.so` is a Metal AOT build
of 141 AIR blobs). The edit sites below are recorded exactly so that the build
is a mechanical step when the machine can take it.

Note also `docs/lanes/HANDOFF_2026-09-15_evening.md`: **Andrew, Sep 15 at 19:50
ET, "no more new lanes."** Four `*-rest` lanes were taken off the board then.
This lane read that before doing anything and deliberately opened no new
implementation lane. If the Sep 16 question re-authorizes that work, the
ranked list is ready; if it does not, the triage still stands on its own.

## Part 1: the inventory

### What was already triaged, and was NOT redone

| lane | family | rows | verdict |
| --- | --- | --- | --- |
| `lane/gbdt-rest` (`2355b6a27`) | gbdt | 28 | (a) 10, (b) 13, (c) 5 |
| `lane/neighbors-rest` (`973f2e01e`) | neighbors | 31 | (a) 2, (b) 20, (c) 9 |
| `lane/cpu-training-par-wave3` (`df9234099`) | the 28 `par-*` CPU lanes | 28 | not a NOT_IMPLEMENTED triage; 2 coverable now, 24 after named work, 2 never |

### The 25 families nobody had triaged

Every family except `gbdt` and `neighbors`: **arima, cholesky, cluster,
dbscan, decomposition, extratrees, gaussian_process, gemm, glm, hdbscan,
hierarchy, holtwinters, isolation_forest, ivf, kde, kernel_methods, metrics,
mixture, resample, solver, spectral, svm, tokenizer, transformer, tsa.**

Counts are real data rows (`grep -cv '^#\|^[[:space:]]*$'`), not `wc -l`.
**425 rows in the tree, 59 previously triaged, 366 triaged here.**

Classes: **(a)** a capability a user reaches for; **(b)** internal reference
plumbing no user calls; **(c)** intentionally excluded, with a reason that is
a decision and not a backlog; **closed** the row says IMPLEMENTED/RESOLVED and
is kept as history, so it is not a gap at all.

| family | rows | (a) | (b) | (c) | closed |
| --- | --: | --: | --: | --: | --: |
| arima | 24 | 6 | 8 | 7 | 3 |
| cholesky | 11 | 0 | 6 | 4 | 1 |
| cluster | 7 | 0 | 0 | 5 | 2 |
| dbscan | 11 | 4 | 4 | 2 | 1 |
| decomposition | 10 | 2 | 0 | 5 | 3 |
| extratrees | 20 | 4 | 0 | 11 | 5 |
| gaussian_process | 12 | 7 | 2 | 3 | 0 |
| gemm | 12 | 0 | 6 | 6 | 0 |
| glm | 12 | 6 | 2 | 2 | 2 |
| hdbscan | 14 | 2 | 8 | 3 | 1 |
| hierarchy | 13 | 2 | 6 | 5 | 0 |
| holtwinters | 20 | 1 | 2 | 15 | 2 |
| isolation_forest | 8 | 1 | 1 | 5 | 1 |
| ivf | 21 | 5 | 3 | 13 | 0 |
| kde | 16 | 4 | 2 | 10 | 0 |
| kernel_methods | 16 | 7 | 4 | 5 | 0 |
| metrics | 16 | 2 | 7 | 6 | 1 |
| mixture | 17 | 9 | 2 | 6 | 0 |
| resample | 18 | 9 | 2 | 7 | 0 |
| solver | 15 | 2 | 6 | 5 | 2 |
| spectral | 20 | 2 | 4 | 12 | 2 |
| svm | 20 | 6 | 2 | 10 | 2 |
| tokenizer | 11 | 3 | 1 | 5 | 2 |
| transformer | 14 | 5 | 0 | 8 | 1 |
| tsa | 8 | 1 | 1 | 5 | 1 |
| **total** | **366** | **90** | **79** | **165** | **32** |

**(a) is the load-bearing number and every one of its 90 rows is named below.**
The (b)/(c) boundary inside the remaining 244 is judgment applied row by row;
the headline is that two thirds of what these files record as "not
implemented" is either a deliberate exclusion or reference plumbing with no
caller.

### The 90 (a) rows, by family

- **arima (6)** confidence intervals / `level`; missing observations (NaN as
  missing); the CSS log-likelihood method and `truncate`; device
  `information_criterion`; caller-supplied `start_params` / `set_fit_params`;
  AutoARIMA's p/q/P/Q/k search.
- **dbscan (4)** `core_sample_indices_` (an attribute cuML computes on every
  default fit); `metric='cosine'`; L1 on the ball cover, so
  `metric='manhattan'` stops refusing `algorithm='rbc'`; `metric='precomputed'`.
- **decomposition (2)** randomized SVD (`PCA(svd_solver='randomized')`,
  `TruncatedSVD(algorithm='randomized')`); tSVD `explained_variance_` computed
  from the transformed data as theirs is.
- **extratrees (4)** the Poisson, Gamma and Inverse Gaussian criteria;
  `sample_weight`.
- **gaussian_process (7)** `predict(return_cov=True)`; `DotProduct`;
  `RationalQuadratic`; `ExpSineSquared`; the `Exponentiation` /
  `PairwiseKernel` / `CompoundKernel` family; multi-output `y`;
  `GaussianProcessClassifier`'s optimizer.
- **glm (6)** `lstsqSvdJacobi` and `lstsqQR` (accuracy on ill-conditioned
  designs, the one property the shipped routes cannot have); `ridgeSVD`;
  `sample_weight` on the QN losses; the `ols_fit_weighted` BINDING (the device
  arm exists and Python does the scaling itself); a Python door for the six
  `glm_linear` / `glm_svm` one-target losses, which run and are exposed nowhere.
- **hdbscan (2)** `probabilities_`; `cluster_selection_epsilon`.
- **hierarchy (2)** the k-NN connectivity arm, **which is cuML's own Python
  default**; the L1 and cosine metrics.
- **holtwinters (1)** a caller-supplied `OptimParams` surface.
- **isolation_forest (1)** DEVIATION 750, which of `curand_u64`'s two draws is
  the high word. Until one number comes off an NVIDIA box every gate in that
  lane is us agreeing with us. **OWED to the next release record.**
- **ivf (5)** caller-supplied ids on `extend`; `CosineExpanded`; the
  inner-product arm; the sample filter; serialize / deserialize of an index.
- **kde (4)** `metric='minkowski'` with p != 2 (see Part 2, ranked #1);
  `sample()`; `bandwidth='scott'|'silverman'`; the eight refused distance
  metrics.
- **kernel_methods (7)** `AdditiveChi2Sampler`; `SkewedChi2Sampler`;
  `PolynomialCountSketch`; per-target `alpha`; `sample_weight`; the cosine /
  chi2 / additive_chi2 pairwise kernels; `gamma='scale'`.
- **metrics (2)** `median_absolute_error`; `silhouette_score` at a metric
  other than euclidean. **Plus three live refusals that are not TSV rows at
  all** and are ranked in Part 2: `sample_weight` on the regression metrics,
  `roc_auc_score` multiclass and `max_fpr`, and
  `precision_recall_curve(drop_intermediate=True)`.
- **mixture (9)** `covariance_type` tied / diag / spherical;
  `init_params='k-means++'` and `'random_from_data'`; `n_init>1`;
  `warm_start`; `means_init` / `weights_init` / `precisions_init`;
  `BayesianGaussianMixture`.
- **resample (9)** `method='BCa'`; `paired=False`; `permutation_type='samples'`
  and `'pairings'`; the exhaustive permutation arm; `stratify`;
  `sample_weight`; `replace=False`; an order-statistic permutation test.
  **Note a gap no row records: there is no `mojolearn.utils.resample`
  function at all**, so rows about its keywords describe options of a door
  that does not exist.
- **solver (2)** `sample_weight`; `solver='qn'` (OWL-QN).
- **spectral (2)** the `SpectralEmbedding` surface (no such class exists);
  `coo_reduce_duplicates`, which is the alternative to the duplicate-key
  refusal.
- **svm (6)** the `tanh` / sigmoid kernel; `kernel='precomputed'`;
  `sample_weight` / `class_weight`; sparse input; **multiclass (one-vs-rest),
  which is the largest single hole in the SVM surface**; `probability` (Platt).
- **tokenizer (3)** pre-tokenizer patterns other than GPT-2's; raising on a
  disallowed special token; the completion-time surfaces.
- **transformer (5)** FlashAttention / online softmax (a v2 with its own
  contract); the non-default RoPE types; chunked / prefix / packed masks and
  attention sinks; dropout; **running `transformer/corpus/`, which exists and
  has never been executed, so agreement with a float64 reference is claimed
  nowhere.**
- **tsa (1)** seasonal `D` selection (their `seas_test`, an STL).

## Part 2: what was finished

**Six documentation rows corrected against the code, each one an absence claim
that had gone false.** This is the `fix-docs-on-discovery` and
`absence-in-one-file-is-not-never-measured` rule applied to the very files
whose job is to be honest about gaps. Every correction was made by reading the
implementation, not the prose.

| file:line | the claim that had gone false | what the code says |
| --- | --- | --- |
| `decomposition/NOT_IMPLEMENTED.tsv:15` | `pcaInverseTransform` "not implemented yet ... trivial once anything needs it" | `PCA.inverse_transform` (`decomposition.py:389`) runs the plain arm through `inverse_transform` and the whitened arm through `pca_whiten_inverse_transform`; `TruncatedSVD.inverse_transform` (:581) calls the same entry with `add_mean = 0`; the host column carries both names (`_classical_host.py:210`, `host_surface.py:1155`) |
| `holtwinters/NOT_IMPLEMENTED.tsv:16` | `get_level` / `get_trend` / `get_season` / `score` / `forecast(index=)` "NOT IMPLEMENTED" | all five are in `_tsa_impl.py` (`_component` :812, :835, :838, :841, `score` :791, `forecast(h, index=)` :584) with cuML's return shapes and DEVIATION 2422. Narrowed to the cudf/cupy plumbing, which is NOT APPLICABLE under the numpy-free contract |
| `glm/NOT_IMPLEMENTED.tsv:5` | "`linear_model.py` still refuses > 2 classes at the Python door" | `lane/logistic-multiclass` opened it: `_QN_LOSS_SOFTMAX` at `n_classes > 2`, the 4-field `qn_decision_function`, `argmax_rows` predict, float64 host softmax `predict_proba`, identity lane `logistic-multiclass`. Narrowed to the two DEVICE kernels, which really are absent, plus the separate live refusal of l1/elasticnet at C > 2 |
| `spectral/NOT_IMPLEMENTED.tsv:25` | `spectral_clustering.pyx` "HAND-OFF, not implemented ... so a binding author does not have to rediscover them" | `SpectralClustering` carries the whole enumerated contract (`_n_components` :453, the affinity and `assign_labels` refusals :341/:349, `eigen_tol`, `n_init`, `n_neighbors`) plus save/load/predict and `prediction_data`. Row 26 (`SpectralEmbedding`) is genuinely still open and now says so |
| `kde/NOT_IMPLEMENTED.tsv:4` | `sample()` "Refused by absence (no entry); the Python surface should raise NotImplementedError by name" | `density.py:714` already raises by name. The row's own instruction was done; the algorithm half still stands |
| `kde/NOT_IMPLEMENTED.tsv:6` + `density.py` x2 | the binding slot cited as `_mojolearn_estimators.mojo:392-396` | the real check is `:647-651`, **and it is duplicated in `_mojolearn_estimators_host.mojo:213-217`**, which neither the row nor `density.py` mentioned. A fix touching only the GPU binding would make the CPU verifier refuse what the GPU accepts |

### Evidence

No numerics moved, so there is no identity column to take. What was run:

    python3 tools/docs_facts.py --check                    # OK: 13 facts, 12 marked spans
    python3 packaging/wheel_ci.py pins .                   # OK: 56 build scripts
    python3 packaging/wheel_ci.py inventory python/mojolearn  # OK: 85 modules, all importable
    python3 -m py_compile python/mojolearn/density.py      # OK
    # every edited TSV row still has exactly 3 tab-separated fields, line counts unchanged

**The probe, and why the obvious one is void.** Grepping for the old sentence
cannot work here: two corrections QUOTE the sentence they supersede, exactly
the trap in `quoting-the-old-text-kills-your-probe`. The probe that does work
reads the STATUS FIELD (field 2), which the new text does not quote, and it
was run on both sides so it could be seen to differ:

    for spec in glm:5 kde:4 decomposition:15 holtwinters:16 spectral:25; do
      f="${spec%%:*}/NOT_IMPLEMENTED.tsv"; n="${spec##*:}"
      printf 'NOW : %s\n' "$(sed -n "${n}p" "$f" | cut -f2)"
      printf 'MAIN: %s\n' "$(git show origin/main:"$f" | sed -n "${n}p" | cut -f2)"
    done

## What remains, ranked by user value against cost

Costs assume one core locally and **include the Metal column**, which is what
makes every one of these bigger than its diff.

1. **KDE `metric='minkowski'`, p != 2.** The arithmetic is already there and
   gated: `kde_score_samples_host_ptr` takes `metric_arg: Float32 = 2.0`
   (`kde/estimator.mojo:181`) and the kernel computes any finite positive
   normal p. What blocks it is a `len(params) != 5` check in TWO binding
   files. **Cost: 2 lines x 2 files, then the real work** -- a new identity
   lane beside `kde-cosine-minkowski`, its `host_surface.py` entry, a rebuild
   of `_mojolearn_estimators.so` (a Metal AOT build, 141 AIR blobs) and
   `_mojolearn_estimators_host.so` plus a sabotage host build, a Metal column,
   a CPU column and a sabotage seen to fail. Half a day, and it cannot start
   while the Apple record holds the slot. **This is the highest value per unit
   of work on the board.**
2. **`sample_weight` on the regression metrics** (`mean_absolute_error`,
   `mean_squared_error`, `root_mean_squared_error`). `_metrics_impl.py:712`
   refuses it; the pattern is already built twice next door
   (`accuracy_score_weighted` and `r2_score_weighted` over
   `metrics/impl/weighted_scores.mojo` with the `PINNED_SUM_W` tree), so this
   is a third instance of a solved shape, not a new one. **Cost: a new kernel
   and binding entry per metric, a host twin, 3 new cells with full proof.
   One to two days.**
3. **`precision_recall_curve(drop_intermediate=True)` and `roc_auc_score`
   multiclass / `max_fpr`.** Host bookkeeping over curves the device already
   returns. Cheapest of the metrics work, but each still adds recorded values
   and therefore cells. **Half a day each.**
4. **SVM multiclass (one-vs-rest).** The largest hole in a shipped estimator's
   surface: `SVC` asserts two classes. Real work -- k binary fits, a label
   map, a decision-function shape change and a saved-model format change.
   **Several days.**
5. **`kneighbors_graph` / `radius_neighbors_graph`.** Absent entirely and not
   in any TSV. Pure host bookkeeping over queries that already exist, so no
   new arithmetic -- but the return type has to be decided first, because
   scipy is not a dependency and there is no sparse type in the tree. **A day,
   most of it the design decision.**
6. **`GaussianProcessRegressor.predict(return_cov=True)`.** One more kernel
   matrix and one GEMM at `OP_TN`. Small and self-contained. **A day.**
7. **`RationalQuadratic`** is named in its own row as the cheapest
   unimplemented GP kernel, because `identical_pow` is already pinned.
   **A day.**
8. **Run `transformer/corpus/`.** The float64 reference exists, is committed,
   and has never been executed by anyone; agreement with it is claimed
   nowhere. This is a RUN plus a `pixi.toml` task line, not new code.
   **Half a day, and it needs no Metal column because it is a tolerance
   comparison, not an identity cell.** Best value of anything that can be done
   while the Metal slot is busy.
9. **DEVIATION 750 (isolation forest `curand_u64` word order).** One number off
   an NVIDIA box. **OWED to the next release record**, cannot be done locally.

Explicitly NOT recommended: the pairwise GBDT learner family, the ivf_pq /
CAGRA / HNSW families, `BayesianGaussianMixture`, and FlashAttention. Each is a
new lane with its own contract, and each is correctly (c) today.

## Resume

    cd /Users/andrewhendel/CascadeProjects/mojolearn
    git worktree add <dir> lane/half-built-inventory      # or work from main, this is merged
    # The triage is this file. The corrections are in the six files below.
    git show lane/half-built-inventory --stat

To pick up item 1 (KDE minkowski), the exact edit sites, re-read on 2026-09-16:

    bindings/_mojolearn_estimators.mojo:647       if len(params) != 5:
    bindings/_mojolearn_estimators_host.mojo:213  if len(params) != 5:   # MUST change with it
    kde/estimator.mojo:181                        metric_arg already present, default 2.0
    python/mojolearn/density.py:620-643           the refusal to delete
    python/mojolearn/host_surface.py:358          kde-cosine-minkowski, where the new lane goes

Do not start it while `mac_slot.sh metal` is held by the release record, and do
not merge it on a CPU column alone: unlike a documentation fix, it moves bits.

## Rules this lane ran under

One core, `nice -n 19`, own worktree, one process at a time. No Metal job and
no Metal lock. Nothing rented. `release/0.8.6` and `db9047b9f` untouched. No
`git stash` in the shared checkout. Branch checked immediately before the
commit.
