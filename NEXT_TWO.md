# The road to CatBoost's symmetric trees, corrected twice

Written 2026-08-21, revised the same day after staged item 1 was answered.
Read `PORTING_RULES.md` first, then `PORTING.md` 88 and 91, then this.

## STATUS

Staged item 1 -- "how much of `TFeatureParallelDataSet` survives at device
count 1" -- is **ANSWERED and written up in `PORTING.md` 91**. The answer
changed the plan twice over, so this note is the revised plan and the two
corrections are recorded rather than quietly edited away.

## CORRECTION 1 (from the first draft of this note)

I said feature combinations were the cheaper objective and should come before
ordered boosting. **Wrong.** `TTreeCtrDataSetsHelper` and
`TTreeCtrDataSetVisitor` appear only in
`methods/oblivious_tree_structure_searcher.cpp:106-209`; grep `TreeCtr` in the
doc-parallel searcher or anywhere in `greedy_subsets_searcher/` and there are
zero hits. Combinations are a feature OF the other learner, so they share
ordered boosting's prerequisite.

## CORRECTION 2 (from answering item 1)

The first draft said "CatBoost has two GPU learners" and priced one
indivisible ~7,700-line prerequisite. **There are three searchers, the two
oblivious ones share their entire stack, and at device count 1 the two data
layouts build a bit-identical compressed index** -- proof in `PORTING.md` 91 A.
So the work is a ladder with a gate on every rung, not a cliff.

And the finding that matters most, `PORTING.md` 91 F: **the searcher this
repository ported is the one CatBoost runs for MULTICLASS symmetric trees**
(`multiclass.cpp:5-14`), not the one it runs for single-target RMSE symmetric
trees. The registration that would have sent single-target symmetric trees our
way is commented out in `pointwise_non_symmetric.cpp:36`. Our splits match
CatBoost 48/48, but that oracle passes no `task_type` and so runs their CPU
learner. **We have never compared against either GPU single-target oblivious
searcher, because neither is ported.** Rung 1 is what closes that.

---

## THE LADDER

Every rung ships and is gated on its own. Deviation numbers start at **92**;
62-91 are taken.

### RUNG 1 -- the pointwise stack and `TDocParallelObliviousTree`

This is CatBoost's `boosting_type=Plain` GPU oblivious learner: the arm every
matched benchmark in this repository already pins CatBoost to.

**LANDED 2026-08-21** (`check-pointwise-offsets`, `check-pointwise-loop`):

    split_properties_helpers.cuh   -> gbdt/methods/kernel/, pointwise part
    compute_point_hist2_loop.cuh   -> all three entry points, ONE shared
                                      loop behind a `PointHist2` trait

**NEXT, in this order:**

1. `pointwise_hist2_one_byte_templ.cuh` + the 5/6/7/8-bit accumulators.
   These are `PointHist2` implementors and nothing else -- the loop is
   done, so each one is `AddPoint`, `Reduce` and a writeback.
2. `pointwise_hist2_binary.cu`, `pointwise_hist2_half_byte.cu`.
3. `pointwise_hist2.cu`, the dispatcher, plus `pointwise_scores.cu` and
   `score_calcers.cuh`.
4. `pointwise_kernels.{h,cpp}` host wrappers, then `histograms_helper`,
   `pointwise_optimization_subsets`, `pointwise_scores_calcer`.
5. `oblivious_tree_doc_parallel_structure_searcher` and the 85-line weak
   learner, at which point the rung-1 gate (agreement with the
   greedy-subsets histograms cell for cell) becomes runnable.

| piece | their file | lines |
|---|---|---|
| the subsets | `methods/pointwise_optimization_subsets.{h,cpp}` | 184 |
| the scorer | `methods/pointwise_scores_calcer.h` + `pointwise_score_calcer.cpp` | 125 |
| the histogram driver | `methods/histograms_helper.{h,cpp}` | 519 |
| the kernel wrappers | `methods/pointwise_kernels.{h,cpp}` | 699 (minus the dead `TComputeHist1Kernel`) |
| helpers | `methods/helpers.{h,cpp}` | 277 |
| the searcher | `methods/oblivious_tree_doc_parallel_structure_searcher.{h,cpp}` | 295 |
| the weak-learner glue | `methods/doc_parallel_pointwise_oblivious_tree.h` | 85 |
| **the hist2 kernel family** | `kernel/pointwise_hist2*.{cu,cuh}`, `compute_point_hist2_loop.cuh`, `split_properties_helpers.cuh`, `pointwise_scores.cu`, `score_calcers.cuh` | **3,563** |

`kernel/pointwise_hist1.cu` (935) is **not** on this list: it is dead upstream
(`PORTING.md` 91 D, and `gbdt/UNPORTED.tsv`).

**What already exists and is reused unchanged**: the boosting loop
(`TBoosting` is the same class for both weak learners), `TDocParallelDataSet`,
every target kernel, the leaf estimator and its oracle/walker/Exact path,
bootstrap, borders and quantization, `nan_mode`, simple CTRs, the model, the
text format, the device evaluator, the apply, and the permutation machinery
that landed 2026-08-21.

**Gate**: the new histograms and the greedy-subsets histograms this repository
already has must agree cell for cell, on the same rows and the same compressed
index. Two independent implementations, one of them already gated against
CatBoost -- the strongest differential available, and it exists before any
boosting loop is touched. Sabotage: perturb one bin of one feature and watch
the disagreement appear at exactly that cell.

### RUNG 2 -- `TFeatureParallelObliviousTreeSearcher` at ONE fold (~400 lines)

With `PORTING.md` 91 A proving the compressed index identical at one device,
this searcher differs from rung 1 by the fold bits and the tree-CTR block and
nothing else. At `FoldBits == 0` it is the same program.

**Gate**: an identity. One fold must reproduce rung 1's model to the bit.
Plus a control that must differ, the way `check-permutation-count` does it.

### RUNG 3 -- folds, `TDynamicBoosting`, ordered boosting (~700 lines)

`CreateFolds` (`dynamic_boosting.h:189-223`) is 35 lines: nested prefix folds
growing geometrically by `fold_len_multiplier` (2.0) from `min_fold_size`
(100), about `log2(n/m)` of them. The Plain arm collapses to a single fold
(`:205-209`), which is exactly rung 2.

The whole of ordered boosting at the subsets level is three lines
(`PORTING.md` 91 B): the fold id occupies the LOW bits of the bin, depth bits
sit above it, and every downstream call reads `CurrentDepth + FoldBits`.

**Gate**: the fold boundaries are closed-form in `n`, `min_fold_size` and
`fold_len_multiplier`, so they are asserted against a host calculation before
any tree is grown.

This is CatBoost's shipped GPU default for every non-multiclass loss
(`catboost_options.cpp:803-807`). Until it lands, every comparison against
CatBoost must pin `boosting_type=Plain` on their side to be matched -- which
is what the benchmarks do, and which means we cannot claim parity with
CatBoost AS SHIPPED.

### RUNG 4 -- tree CTRs / feature combinations (1,640 lines)

`methods/tree_ctrs.{h,cpp}`, `tree_ctrs_dataset.{h,cpp}`,
`tree_ctr_datasets_visitor.{h,cpp}`, `batch_feature_tensor_builder.{h,cpp}`,
`ctr_from_tensor_calcer.{h,cpp}`, `tree_ctr_memory_usage_estimator.h`.

Computed DURING tree growth, not as a preprocessing pass: a tree CTR's tensor
is (the splits already in the current tree) x (a categorical feature), so the
candidate set grows level by level inside the searcher loop
(`oblivious_tree_structure_searcher.cpp:137`, `:202-209`).

`IsTreeCtrsEnabled()` is `catFeatures present && MaxTensorComplexity > 1`
(`binarizations_manager.h:65-68`). Their default complexity is 4
(`cat_feature_options.cpp:231`); ours is pinned at 1 and
`TCatFeatureParams.check()` refuses more (`gbdt/options/catboost_options.mojo:985`).
`combination_ctrs` already exists as a field and is unread.

**Gate**: the TENSOR first -- their hash of a feature combination against a
`tools/` oracle -- and only then a fit.

### RUNG 5 -- widen the CatBoost differential

Independent of the rest and worth doing alongside. `tools/catboost_oracle.py`
takes the border budget from the environment already; depth and feature count
are the same shape of change. Two things it has never done: **run with
`task_type="GPU"`** (see `PORTING.md` 91 F -- today it compares against their
CPU learner) and **compare a categorical fixture split-for-split**. When
Andrew asked what "narrow" meant, this is the answer: the gate is narrow, not
the code.

---

## Traps already known

* **`--target-accelerator` at ANY value produces an artifact with zero GPU
  kernels** (`PORTING.md` 70). Do not add it back.
* **`mojo run` JITs and is unaffected by the AOT blob defect**, so a green
  check says nothing about whether the wheel works.
* Their `learnPermutationCount - 1` modulus means permutation 2 of 4 is never
  searched on. Transcribed, not corrected -- the same expression is in
  `dynamic_boosting.h:286-289`.
* `TMirrorMapping` vs `TStripeMapping` is a no-op at one device
  (`PORTING.md` 91 A) but records which one CatBoost meant. Keep the
  distinction in the types even though it costs nothing today: erasing it is
  correct now and unportable to a second device later.
* `gbdt/train.mojo` is shared with another session. Rungs 1-4 are new files;
  do not edit it without checking who else is in it.
