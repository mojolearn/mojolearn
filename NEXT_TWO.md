# Where the CatBoost symmetric-tree port stands, and what is left

Rewritten 2026-08-21 after rung 1 shipped. Read `PORTING_RULES.md` first,
then `PORTING.md` 91, then this.

## DONE: rung 1. `fit` runs CatBoost's single-target symmetric learner

    fit(..., use_pointwise_searcher=True)   TDocParallelObliviousTreeSearcher

which is what CatBoost runs for single-target symmetric trees at
`boosting_type=Plain` (`PORTING.md` 91 F). The default stays False --
`TGreedySubsetsSearcher`, what this repository has always run and what
CatBoost runs for MULTICLASS symmetric trees.

**The two arms are bit-identical over twenty boosting iterations**
(`pixi run check-fit-pointwise`). They share the compressed index, the weak
target, the bootstrap, leaf estimation, the apply and the loss; only the
structure searcher differs.

~9,000 lines, fourteen checks:

    check-cindex-packing              our index feeds their decode, all 3 policies
    check-pointwise-offsets           the fold stripe, the partial-pass sibling
    check-pointwise-loop              every point once, 160 cases x 5 entry points
    check-pointwise-hist2             5/6/7-bit accumulators, per cell
    check-pointwise-hist2-8bit        the fixed-point one, per cell
    check-pointwise-hist2-half-byte   binary + half-byte, both readers
    check-pointwise-driver            the one-byte driver and its dispatch
    check-pointwise-small-bin-driver  the other two drivers
    check-pointwise-dispatch          the whole family together, 3,686 cells
    check-pointwise-scores            five calcers, three split kernels
    check-pointwise-subsets           CreateSubsets + Split, CONTENT and ORDER
    check-histograms-helper           the full-pass state machine, host only
    check-pointwise-vs-greedy         RUNG 1's GATE: two searchers, one tree
    check-fit-pointwise               `fit` both ways, 20 iterations identical

## WHAT IS LEFT

### Rung 2 -- `TFeatureParallelObliviousTreeSearcher` at ONE fold (713 lines)

`methods/oblivious_tree_structure_searcher.{h,cpp}`. At `FoldBits == 0` and
one device it is the same program as rung 1 -- `PORTING.md` 91 A proves the
two data layouts build a bit-identical compressed index at device count 1,
and 91 B shows the two searchers share their whole stack and differ in three
lines of `CreateSubsets`.

**Gate**: an identity. One fold must reproduce rung 1's model to the bit,
plus a control that must differ, the way `check-pointwise-vs-greedy` does it.

Everything it needs is now ported. This is the cheapest rung by far.

### Rung 3 -- ordered boosting (1,353 lines)

`methods/dynamic_boosting.h` + `feature_parallel_pointwise_oblivious_tree` +
the feature-parallel dataset. `CreateFolds` (`dynamic_boosting.h:189-223`) is
35 lines: nested prefix folds growing geometrically by `fold_len_multiplier`
(2.0) from `min_fold_size` (100).

The whole of it at the subsets level is three lines (`PORTING.md` 91 B): the
fold id occupies the LOW bits of a document's bin, and every downstream index
reads `CurrentDepth + FoldBits` rather than `CurrentDepth`. **The fold-stripe
arithmetic is already ported and gated** (`PointwisePartOffsetsHelper`), and
`TOptimizationSubsets` already carries `fold_count`/`fold_bits`.

**Gate**: the fold boundaries are closed-form in `n`, `min_fold_size` and
`fold_len_multiplier`, so they assert against a host calculation before any
tree grows.

THIS IS CATBOOST'S SHIPPED GPU DEFAULT for every non-multiclass loss
(`catboost_options.cpp:803-807`). Until it lands, **every comparison against
CatBoost must pin `boosting_type=Plain` on their side**, and we cannot claim
parity with CatBoost as shipped.

### Rung 4 -- tree CTRs / feature combinations (1,640 lines)

Computed DURING tree growth, not as a preprocessing pass: a tree CTR's tensor
is (the splits already in the current tree) x (a categorical feature). Needs
rung 2 first -- they live only in the feature-parallel searcher.

**Gate**: the TENSOR first, their hash against a `tools/` oracle, before any
fit.

### Rung 5 -- widen the CatBoost differential (independent, do any time)

**CORRECTED 2026-08-21. This page said twice that running the oracle with
`task_type="GPU"` was the highest-value item here. It is not an item at
all: CatBoost's GPU arm CANNOT RUN ON THIS MACHINE.**

    catboost.CatBoostRegressor(task_type='GPU').fit(...)
    -> catboost/libs/train_lib/trainer_env.cpp:9:
       Environment for task type [GPU] not found

CatBoost's GPU build is CUDA. That is not a gap in this repository's
discipline -- it is the project's thesis
([[mojotrees-benchmark-arena]]): their GPU arms cannot run on Apple silicon,
which is the win condition rather than a caveat. Comparing against their GPU
output needs an NVIDIA box, which is what `tools/nvidia_bench.sh` and
`tools/remote_gpu.sh` are for.

**And the CPU oracle is the right reference anyway.** CatBoost's CPU learner
is the canonical implementation of the algorithm; their two arms are meant to
agree on which splits an oblivious tree takes. `PORTING.md` 108's 144 of 144
is a comparison with CatBoost, not a weaker substitute for one.

What is genuinely left here, all doable locally:

* **a CATEGORICAL fixture, split-for-split.** Never done, and the one place
  the differential has no coverage at all.
* depth and feature-count sweeps -- the same shape of change as the border
  budget the oracle already takes from the environment.
* a fixture whose widest feature lands in each of the four one-byte bit
  widths, since `PORTING.md` 108 showed a whole kernel can be reached by
  exactly one of the three existing fixtures.

## OPEN DECISIONS, both needing a MEASUREMENT and no benchmark authorised

1. **`PORTING.md` 98a -- which partition reducer.** `update_partition_props`
   (the one their dispatch names) is ported, unused, and now a drop-in after
   three layout changes. Calling it replaces six launches and two scratch
   buffers with one, taking `Split` from 17 to 12. What is still bought by
   declining is depth-0 occupancy, where their form puts the whole dataset
   through one threadgroup. The swap is one call in `update_subsets_stats`.

2. **The `use_pointwise_searcher` default.** The arms are bit-identical, so
   this is purely a performance question. The pointwise arm pays a host round
   trip per tree in `split_stat_planes`, which disappears entirely if the
   boosting loop carries the weak target as two buffers throughout --
   `stats` is read by the greedy searcher, the estimator and the bootstrap,
   so that is a real change and not a rename.

## TRAPS, all of them found the hard way

* **The two scorers disagree about SIGN** (`PORTING.md` 94a).
  `pointwise_scores.mojo` keeps CatBoost's -- `FLT_MAX`, `gain < bestGain`,
  lower wins. `greedy_subsets/compute_scores.mojo` folds in a negation and
  flips every comparison. A searcher that mixes them picks the WORST split
  at every level and still returns a well-formed tree.
* **`TakeBest` folds in OPPOSITE directions** at its two call sites, and both
  are theirs. The calcer keeps the incumbent on a tie; the searcher takes the
  new one.
* **`IntLog2` is `ceil`.** Floor silently unhistograms every one-byte feature
  whose fold count is not a power of two.
* **`result_size` IS the scorer's grid.** Each block writes its own record;
  reading block 0 is an argmin over 128 bin features.
* **`BIN_SPLIT_TAKE_BIN` is 0 and `TAKE_GREATER` is 1**, the opposite of what
  the names suggest.
* **`TCFeature::Offset` is an ELEMENT offset**; this tree's layout stores a
  COLUMN index and strides by `n_rows`.
* **Two views of one buffer cannot both reach a kernel.** `unsafe_bitcast`
  does not launder the origin and the check fires at `enqueue_function`
  itself (`PORTING.md` 97.2).
* **`--target-accelerator` at ANY value produces an artifact with zero GPU
  kernels** (`PORTING.md` 70). `mojo run` JITs and is unaffected, so a green
  check says nothing about the wheel.
* Their `learnPermutationCount - 1` modulus means permutation 2 of 4 is never
  searched on. Transcribed, not corrected.

## TWO THINGS WORTH RE-EXAMINING

* **`PORTING.md` 92's hypothesis about item 11.** A divergent threadgroup
  barrier is benign until the barrier is LOAD-BEARING; item 11's original
  symptom ("every feature's histogram read 0.0") has now been produced in
  this repository by a racing tally twice and by a reused async staging
  buffer once, and by a divergent barrier zero times. Untested.
* **The greedy family's duplicated loop.** `PORTING.md` 13's blocker was
  false -- a shared pointer crosses a function boundary if the callee
  parameterizes the origin (`check-shared-pointer`). The pointwise family is
  written CatBoost's way because of it. Unifying `hist_binary.mojo` and
  `hist_half_byte.mojo` is now possible and is NOT done; until it is, treat
  those two files as one.

## RULES THIS ROUND EARNED

* **Gate a kernel against a REAL accumulator, not a convenient one.** A
  private-slot tally measures coverage and is the histogram equivalent of
  uniform test data; the contention a real accumulator creates is where the
  bugs are (`PORTING.md` 92).
* **A gate whose sabotage does not move it is not coverage** -- say so in the
  check rather than let a green tick imply otherwise.
* **Beware "reached but inert"**: a path that runs but whose effect is a
  no-op at the chosen parameters. It bit four times this round -- a kernel
  that returned immediately, a scale of 1.0 making a division a no-op, a
  fixture where every power-of-two fold count made two offsets coincide, and
  a check that spelled a layout with the library's own constants.
* **Wiring finds what no layer below it can.** Fifteen gates were green while
  three integration bugs sat in plain sight, because none of them crossed a
  layer boundary.
