# The next two objectives, and they are one prerequisite

Written 2026-08-21 for the session that picks this up. Read `PORTING_RULES.md`
first, then `PORTING.md` 88, then this.

## THE CORRECTION THAT REORDERS EVERYTHING

I told Andrew that feature combinations were the cheaper of the two -- 1,640
lines, no new kernel family, do them before ordered boosting. **That was
wrong, and the source says so in two places.**

**Tree CTRs live in the FEATURE-PARALLEL searcher and nowhere else.**
`TTreeCtrDataSetsHelper` and `TTreeCtrDataSetVisitor` appear in
`catboost/cuda/methods/oblivious_tree_structure_searcher.cpp:106-209`.
Grep for `TreeCtr` in `oblivious_tree_doc_parallel_structure_searcher.{h,cpp}`
and in all of `greedy_subsets_searcher/` -- the learner this repository
ported -- and there are **zero hits**.

Their option resolution says the same thing from the other side
(`cuda/train_lib/train.cpp:73-84`):

    if (MaxTensorComplexity > 1 && featuresManager.GetCatFeatureIds().size()) {
        return;                                  // STAY feature-parallel
    } else {
        if (BoostingType == Plain) { DataPartitionType = DocParallel; }
    }

So a fit with `max_ctr_complexity > 1` and categorical features **stays on
the feature-parallel learner even in Plain mode**. Combinations are not a
feature we can add to what we have; they are a feature OF THE OTHER LEARNER.

**Therefore the two objectives are one prerequisite and two things that ride
on it**, and the prerequisite is the expensive part.

---

## OBJECTIVE 1 (the prerequisite): the feature-parallel learner

`TDynamicBoosting` + `TFeatureParallelPointwiseObliviousTree`, which is what
`pointwise.cpp` -> `train_template_pointwise.h:27-48` selects whenever the
data partition is FeatureParallel -- and `EDataPartitionType` DEFAULTS to
FeatureParallel (`boosting_options.cpp:25`).

### What has to be ported

| piece | their file | lines |
|---|---|---|
| the boosting loop | `methods/dynamic_boosting.h` | 673 |
| the dataset + builder | `gpu_data/feature_parallel_dataset{,_builder}.{h,cpp}` | 583 |
| the weak learner | `methods/feature_parallel_pointwise_oblivious_tree.{h,cpp}` | ~150 |
| the structure searcher | `methods/oblivious_tree_structure_searcher.{h,cpp}` | 713 |
| the subsets | `methods/pointwise_optimization_subsets.{h,cpp}` | 184 |
| the scorer | `methods/pointwise_scores_calcer.h` + `pointwise_score_calcer.cpp` | 125 |
| histogram driver | `methods/histograms_helper.{h,cpp}` | 519 |
| helpers | `methods/helpers.{h,cpp}` | 277 |
| kernel host wrappers | `methods/pointwise_kernels.{h,cpp}` | 699 |
| the apply | `methods/add_oblivious_tree_model_feature_parallel.{h,cpp}` | 133 |
| **the second histogram kernel family** | `methods/kernel/pointwise_hist1.cu`, `pointwise_hist2*.cu`, `compute_point_hist2_loop.cuh`, `split_properties_helpers.cuh` | **3,736** |

**~4,000 lines of host code and ~3,700 of kernels.** For scale, the histogram
family already ported (`greedy_subsets_searcher/kernel/`) is 6,707 lines of
CUDA and was the single biggest piece of this port.

### What TRANSFERS, and it is a lot

CatBoost shares these between its two learners, and so do we:

* every target kernel (`targets/kernel/`), including multilogit
* the leaf estimator: `pointwise_oracle`, `descent_helpers`, the walker, the
  Exact weighted-quantile path, `linear_system`
* bootstrap, borders/quantization, `nan_mode`, simple CTRs and their tables
* the model, the text format, the device evaluator, the row-wise apply
* **two files in `methods/kernel/` are ALREADY PORTED from that very
  directory**: `exact_estimation.cu` and `linear_solver.cu`
* **the permutation machinery landed 2026-08-21 and is the direct
  prerequisite**: `TDynamicBoosting::Fit` runs the SAME per-permutation
  cursor loop `doc_parallel_boosting.h` runs, with folds added on top.
  `compute_bins_for_model` and `partition_from_bins`
  (`methods/leaves_estimation/doc_parallel_leaves_estimator.mojo`) are what
  its fold tasks need too.

### What is genuinely new

1. **The feature-parallel data layout.** Doc-parallel splits ROWS across
   devices and gives every device all features; feature-parallel splits
   FEATURES. On one GPU that distinction is thinner than it sounds, and the
   first question to answer is how much of `TFeatureParallelDataSet` is
   multi-device bookkeeping that collapses to nothing at device count 1.
   **Answer that before porting it.** It may be much cheaper than 583 lines.
2. **The multi-task structure searcher.** Theirs takes N `(learnTarget,
   validateTarget)` pairs through `AddTask` and sums their scores; ours
   takes one dataset and one cursor. This is the piece with no counterpart
   here, and it is the piece that makes ordered boosting possible.
3. **The second histogram family**, which is a different memory layout from
   the one we ported, not a configuration of it.

---

## OBJECTIVE 2: the two features that ride on it

Once objective 1 exists, both of these are small, and that is the whole
argument for doing them in this order.

### 2a. Ordered boosting -- NEARLY FREE once the learner is there

`CreateFolds` (`dynamic_boosting.h:189-223`) is **35 lines**: nested prefix
folds growing geometrically by `fold_len_multiplier` (2.0) from
`min_fold_size` (100), about `log2(n/m)` of them. The Plain arm collapses to
a single fold covering everything (`:205-209`), which is exactly what this
port already does.

The searcher's `AddTask` has to exist either way, because the Plain arm calls
`SetTarget` and the Ordered arm calls `AddTask` per fold (`:305-330`) -- both
on the same object. So **if objective 1 is ported faithfully, ordered
boosting is the fold construction plus the loop that feeds it.**

This is CatBoost's shipped GPU default for every non-multiclass loss
(`catboost_options.cpp:803-807`), so until it lands, every comparison
against CatBoost must pin `boosting_type=Plain` on their side to be a
matched comparison. That is legitimate and it is what the benchmarks do; it
also means we cannot claim parity with CatBoost AS SHIPPED.

### 2b. Feature combinations (tree CTRs)

| piece | lines |
|---|---|
| `methods/tree_ctrs.{h,cpp}`, `tree_ctrs_dataset.{h,cpp}`, `tree_ctr_datasets_visitor.{h,cpp}`, `batch_feature_tensor_builder.{h,cpp}`, `ctr_from_tensor_calcer.{h,cpp}`, `tree_ctr_memory_usage_estimator.h` | 1,640 |

**They are computed DURING tree growth, not as a preprocessing pass.** A
tree CTR's tensor is (the splits already in the current tree) x (a
categorical feature), so the candidate set grows level by level and the CTR
values are built inside the searcher loop --
`oblivious_tree_structure_searcher.cpp:137` builds the helper, `:202-209`
visits the datasets for the level's candidates. That is why they cannot be
bolted onto a preprocessing-time CTR path like the one `train()` has.

`IsTreeCtrsEnabled()` is `catFeatures present && MaxTensorComplexity > 1`
(`binarizations_manager.h:65-68`), and `UseAsBaseTensorForTreeCtr` is
`tensor.GetComplexity() < MaxTensorComplexity` (`:70-72`). Their default
complexity is 4 (`cat_feature_options.cpp:231`); ours is pinned at 1 and
`TCatFeatureParams.check()` refuses more
(`gbdt/options/catboost_options.mojo:985`). `combination_ctrs` already
exists as a field on `TCatFeatureParams` and is unread.

---

## Staging, with a gate at every step

Nothing here lands without a check that can fail. Per `PORTING_RULES`, and
per the `check-permutation-count` precedent: **a gate that would pass with
the feature absent is not a gate.**

1. **Answer the device-count question first** (one day, no code): how much of
   `TFeatureParallelDataSet` survives at device count 1. Write the answer
   into `PORTING.md` whichever way it comes out.
2. **The histogram family, alone, against the one we have.** Both compute the
   same histogram from the same rows; on one device they must agree cell for
   cell. That is a differential with an existing correct implementation on
   the other side, which is the strongest gate available and it exists
   before any boosting loop does.
3. **The searcher at ONE task**, against `run_tree_layout`: same rows, same
   grid, same gradients, same splits. If it disagrees at one task it will
   never be right at N.
4. **The searcher at N tasks**, against a host tally over the fold slices.
5. **The loop**, gated the way `permutation_count` was: an identity that
   holds (Plain with one fold must reproduce today's model) plus a control
   that must differ.
6. **Widen the CatBoost differential.** `tools/catboost_oracle.py` takes the
   border budget from the environment already; depth and feature count are
   the same shape of change, and a categorical fixture is the one that has
   never been compared split-for-split against them. Andrew asked what
   "narrow" meant and this is the answer: the gate is narrow, not the code.
7. **Ordered boosting**, gated on their fold arithmetic: the fold boundaries
   are a closed-form function of `n`, `min_fold_size` and
   `fold_len_multiplier`, so they can be asserted against a host calculation
   before any tree is grown.
8. **Tree CTRs**, gated first on the TENSOR (their hash of a feature
   combination, against `tools/` oracle output) and only then on a fit.

Deviation numbers start at **91**; 62-90 are taken.

## Traps already known

* **`--target-accelerator` at ANY value produces an artifact with zero GPU
  kernels** (PORTING.md 70). Do not add it back.
* **`mojo run` JITs and is unaffected by the AOT blob defect**, so a green
  check says nothing about whether the wheel works.
* Their `learnPermutationCount - 1` modulus means permutation 2 of 4 is never
  searched on. Transcribed, not corrected -- the same expression is in
  `dynamic_boosting.h:286-289`.
* `TDynamicBoosting` uses `TMirrorMapping` where the doc-parallel learner uses
  `TStripeMapping`. On one device both are "the whole buffer", but the types
  carry which one CatBoost meant, and the port should say which it is.
