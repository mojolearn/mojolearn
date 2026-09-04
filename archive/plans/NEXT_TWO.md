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

## CLOSED 2026-08-31: the covtype leg of the searcher-parity claim

`checks/searcher_parity_covtype_check.mojo`, at the FULL 581,012 rows and
53 mixed-policy features, both reps:

    mse greedy 0.9486324077643835  pointwise 0.9486324077643835  identical True

That greedy value is the one the defect was reported against, and the
pointwise arm -- which produced 1.1843180519507341 -- now matches it bit
for bit. The rung-1 "two searchers, one tree, same bits" claim holds on
mixed-policy shapes, which is the case it was blocked on.

THE BUG, and it was not where the localization pointed. The suspects
recorded below were the per-policy `bin_sums` segment offsets and the
cross-policy best-split id mapping; both were innocent. In
`gbdt/methods/pointwise_scores_calcer.mojo` the per-feature weights
buffer was ALLOCATED at the BIN-FEATURE count and INDEXED by GLOBAL
feature id. On a single-policy shape those two numberings coincide and
nothing is wrong -- which is exactly why each policy alone passed and the
mix failed, and why every oracle fixture was silent.

The localization below is kept because it was correct and it is what made
the bug findable: each policy ALONE bit-identical, the two together not,
is the observation that says "numbering", not "arithmetic".

WHAT THE 4k FIXTURE COULD NOT SETTLE. The fix was first confirmed on a
4,000-row slice. That slice carries the same fold file and therefore the
same policy split and the same global numbering, so it was a strong
argument -- but the divergence was MEASURED at 581k and an argument is not
a measurement. The full arm above is the closing evidence.

CORRECTED WHILE CLOSING: the note below reads "pure one-byte POINTWISE
BEAT GREEDY 1.5x at 4k x 10". True where it was taken, and it does not
survive scale. At 581k the same arm is 2.8x SLOWER (greedy 0.247 s,
pointwise 0.701 s, ratio 0.352 and 0.361 across the two reps). The 4k
number is left standing as the small-shape datum it is; it is not
evidence about the shipped size.

LOCALIZED (2026-08-22, performance lane): each policy ALONE is
bit-identical -- covtype's 10 one-byte features pass (`covob_*`
fixtures), its 43 binary features pass (`covbin_*`), the 53 together
fail. Reproduce the split with the `covob`/`covbin` fixtures in
~/.cache/mojolearn, built by slicing the covtype fixture by fold count.

## BEFORE ANYTHING ELSE: DEVIATION 134 IS OPEN

One run in roughly a hundred of a 4096 x 8 depth-1 fit produced a WRONG BUT
COHERENT model on BOTH searchers, **and the two searchers disagreed with each
other** -- which is the one thing `check-fit-pointwise` exists to forbid. Not
reproduced in ~100 further runs of that cell.

THE SENTENCE THAT STOOD HERE -- "half a mechanism is known (the greedy
float-atomic flush) and it does not cover the pointwise arm" -- WAS STALE
THE EVENING IT WAS WRITTEN and is deleted per [[fix-docs-on-discovery]].
`PORTING.md` 134a-134d is the current record: 134a's 600 clean warm reps
ruled the float atomic OUT for both arms (an atomic fires every fit; there
was exactly one loss value per arm), and 134c found, fixed and reproduced
ON DEMAND the pointwise half -- `_estimate_and_apply`'s `h_po`/`h_ps`
staging pair freed at its last use, [[mojo-buffer-freed-at-last-use]],
closed at the ownership level by DEVIATION 1890, forced corruption
matching the recorded 8/12 first-div-t8 signature to five significant
figures.

**NO SPEED NUMBER AND NO PARITY CLAIM SHOULD BE QUOTED UNTIL THIS CLOSES.**
A learner that produces a different model one run in a hundred does not
have a loss to compare.

RUN RECORD, 2026-09-01 evening (PORTING.md 134f carries the full numbers):
the un-sabotaged FAST soak was CLEAN at 1000 reps (0 CatBoost
disagreements, 0 arm disagreements, one loss value per arm), and the
REQUIRED-RED positive control (`-D MOJOLEARN_134_CONTROL=1`, the
pre-a4aee262 ctor) was QUIET at a combined N=1200 -- so on a quiet box
the lifetime mechanism ALONE does not reproduce the sighting, the 134b
LOAD WINDOW IS A NECESSARY INGREDIENT, and 134 STAYS OPEN. The loaded
control RAN 2026-09-01 midday under Andrew's delegation (bounded, 8+1
processes, watchdog): ~3,000 clean under-load process-reps at load 4-8
including the sabotaged arm — still no 134 signature — then a
box-saturation event at load 18 turned every process's fits to silent
garbage simultaneously (a DIFFERENT defect, recorded as DEVIATION 2002)
and the watchdog tore the window down. A reduced sustained-load rerun
(6 processes, cap 10) and the IDENTICAL-mode soak (rebuild required)
remain the owed halves of the closing pair; full record PORTING.md 134f
run record #2. The bit-inert holds themselves are verified:
check-ordered-boosting and check-fit-pointwise both PASS at HEAD.

MECHANISM HUNT, 2026-09-01 (write-only lane; the code-read claims below
were written before the run record above):

* **The greedy half's ranked candidate was live at the sighting and has
  been closed since the next morning.** The sighting is 3e6ead33
  (2026-08-21 16:28); 134d's audit (written 16:56 the same day) ranked
  `TTreeWorkspace.__init__`'s dead staging buffers as the greedy
  candidate; a4aee262 (2026-08-22 10:41) then put holds-past-drain on
  exactly those fourteen buffers -- plus `upload_blocks`, the estimator
  and the oracle -- and d638ecbb (10:57) swept 39 host-staging and 19
  device holds across the tree. 134d's "UNTESTED" was never reconciled
  against that: the HAZARD it names is gone; only the ATTRIBUTION (force
  the ctor corruption, reproduce greedy 2/12 first-div-t2 by 134c's
  positive-control technique) is still owed. So BOTH halves of 134 are
  the same mechanism class -- staging freed at last use under 134b's
  20-30-process load window -- and no arithmetic account is needed for
  either.
* **The soak cell's fit path audits clean at HEAD.** A whole-of-`gbdt/`
  last-use sweep (every locally created host/device buffer whose final
  textual reference is an enqueue argument, no hold, no field, no
  transfer) finds NOTHING on the path the soak exercises:
  `fit_with_test`, `run_tree_layout_traced`, `TTreeWorkspace.__init__`,
  `upload_blocks`, the pointwise searcher and the estimation workspace
  are all held or pool-owned. The flags in `run_one_level`/`run_tree` are
  check-driver-only paths covered by DEVIATION 1905's per-site audit.
* **134d's "the rest have not been recounted" is now recounted, and six
  unguarded sites of the class remained -- all OFF the soak cell's path,
  all given the same bit-inert hold-past-drain fix in this commit:**
  `gbdt/models/cuda/evaluator.mojo` `pack_model_for_evaluator` (seven
  staging buffers, none held -- `h1` died with six copies not yet
  enqueued behind it; the device-evaluator apply path, never on the
  soak's mse, which reads the fit's own loss track);
  `gbdt/gpu_util/kernel/bootstrap.mojo` `create_bootstrap_seeds` (the
  65536-seed upload -- ON the fit path whenever bootstrap is on; garbage
  seeds are a wrong-but-coherent model exactly in 134's signature);
  `gbdt/ctrs/ctr_bins_builder.mojo` `__init__` and
  `add_cat_feature_bins`; `gbdt/ctrs/ctr_calcers.mojo`
  `set_binarized_sample` and `TWeightedBinFreqCalcerGpu.trivial` (the
  CTR fit path, categorical fixtures only). ALL SIX FIXES ARE
  UNVERIFIED, RUN OWED (commands below).
* Two benign findings, recorded not fixed: `fit_with_test`'s `h_mags`
  (`doc_parallel_boosting.mojo:1228`) is allocated and never used (dead
  since the `hm` readbacks took over) -- delete when the file is next
  open for real work; `ctr_calcers.mojo`'s `dst`/`h_col` loop pair look
  like the class but are loop-scoped, so their lifetimes extend past the
  per-iteration drain.

`pixi run soak-determinism` is the driver, and the experiment it was built
for is UNCHANGED AND STILL OWED: soak at the shipping `NUMERIC_FAST`, then
rebuild with `GLOBAL_NUMERIC_MODE = NUMERIC_IDENTICAL` and soak again --
if the greedy arm goes silent and the pointwise arm still drifts, there is
a SECOND defect and it is not the histogram flush. A clean pair of soaks
at HEAD is now the EXPECTED outcome (both known mechanisms are fixed);
what closes 134 is that pair PLUS the greedy positive control, ideally
under 134b's synthetic load.

THE POSITIVE CONTROL IS NOW WRITTEN (PORTING.md 134f, 2026-09-01):
`-D MOJOLEARN_134_CONTROL=1` compiles out `TTreeWorkspace.__init__`'s
fourteen holds, recreating the pre-a4aee262 ctor; the soak driver's
verdict inverts under it (RED required = mechanism confirmed; QUIET
raises and is a finding -- the 134b load window was the missing
ingredient -- NOT a pass). The loaded-run recipe is in 134f and the soak
docstring. RUN OWED, exact commands, cheapest first:

    # the control (build included -- the comptime-if transfer corner is
    # untested), quiet box first:
    pixi run mojo run -D MOJOLEARN_134_CONTROL=1 -I . checks/determinism_soak.mojo
    # if quiet: the loaded control -- PORTING.md 134f's 24-process recipe,
    # in a scheduled window (it deliberately violates the quiet-box rule)
    # the closing un-sabotaged pair, same box:
    pixi run soak-determinism
    tools/with_identical_mode.sh pixi run soak-determinism
    # and the CTR/bootstrap/evaluator holds' gates:
    pixi run check-ordered-boosting && pixi run check-fit-pointwise

## WHAT IS LEFT

### Rung 2 -- DONE 2026-08-21. `TFeatureParallelObliviousTreeSearcher` at ONE fold

`gbdt/methods/oblivious_tree_structure_searcher.mojo` +
`gbdt/methods/oblivious_tree_bin_builder.mojo`,
`pixi run check-feature-parallel-identity`. `PORTING.md` 120-124. PORTED, NO
CALLER -- `fit()` cannot select it yet; see `UNWIRED.md`.

The gate is the identity this page asked for, at TWO row counts: 3 splits
identical to the bit, 16,434 per-document leaf ids exact, plus a control that
must and does differ.

**THIS PAGE SAID "everything it needs is now ported, this is the cheapest rung
by far", off `PORTING.md` 91 B. THAT WAS WRONG AND THE SENTENCE IS DELETED.**
91 A held. 91 B did not: `TSubsetsHelper<TMirrorMapping>` has no
`CreateSubsets` at all, the two `Split`s are different code calling different
kernels, and the feature-parallel arm needs `docBins` -- a per-document
bit-packed array with no counterpart on the doc-parallel path, built by three
kernels. 1,067 lines of `gbdt/`, not wiring. The IDENTITY claim the gate rests
on survived; only the price was fiction.

Two rows of its sabotage table are worth carrying forward: a defect in the
bit-packing was invisible at 4,000 rows and red at 16,434, because a
compression block is 8,192 documents and `blockIdx.x` is otherwise always 0.

### Rung 3 -- ordered boosting. THE DATA STRUCTURES RUN; the boosting loop is untouched

**STATUS 2026-08-21.** The fold axis is carried and gated end to end through
the layer both searchers share -- `pixi run check-ordered-boosting`, seven
gates at `FoldCount 12 / FoldBits 4 / stripe 16`, 1,463 positions over 600
rows.
`PORTING.md` 125-129. It is the first time a fold axis has run a kernel in this
tree.

Three things stood between a fold-based TREE and this page. Two are now gone:

* ~~`PolicyScoreHelper` hard-codes `foldCount = 1`~~ **CLOSED FOR REAL
  2026-09-03** (DEVIATION 126). It was never a hard-coding: the calcer has
  always taken `fold_count` and forwarded it, and the doc-parallel searcher
  simply omitted the argument at construction, so the helpers were built at
  1 while the layout was built at 12 and the consistency check refused. One
  argument. `check-ordered-boosting` O7 now asserts that A FOLD-BASED TREE
  GROWS at FoldCount 12 instead of asserting the refusal, and all seven
  gates pass. The differential still stands at 288 of 288.
* The fold arm is wired into the DOC-PARALLEL searcher, **which upstream
  cannot have folds at all** (DEVIATION 127). It belongs in
  `oblivious_tree_structure_searcher.mojo`, which now exists. Three lines, and
  it is a decision rather than a mechanical edit -- OPEN.
* `TDynamicBoosting::Fit` is **not ported at all**, and it is the bulk of the
  1,353 lines: per-permutation datasets, the per-fold approx CURSORS,
  `ComputeWeakTarget`'s fold arm (a gradient per fold at THAT fold's cursor),
  per-fold leaf estimation, and model averaging
  (`dynamic_boosting.h:230-470`). **Without cursors there is no ordered
  boosting, only an ordered partition.** OPEN, and this is the real remaining
  work.

So `boosting_type=Plain` STAYS PINNED on every CatBoost comparison, and this
repository still cannot claim parity with CatBoost AS SHIPPED.

Not a blocker, contrary to fear: the dynamic scorer supports only 3 of the 7
score functions, and the GPU default is Cosine, which is one of the three
(DEVIATION 128).

### Rung 3 -- the original entry (1,353 lines)

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

### Rung 4 -- tree CTRs / feature combinations

Computed DURING tree growth, not as a preprocessing pass: a tree CTR's tensor
is (the splits already in the current tree) x (a categorical feature). Needs
rung 2 first -- they live only in the feature-parallel searcher.

**FRONT HALF DONE 2026-08-21** (`2302da0`), and the gate was the tensor first
exactly as this page asked: `gbdt/methods/batch_feature_tensor_builder.mojo` +
`pixi run check-feature-tensor`, against CatBoost's own source compiled by
clang (`tools/feature_tensor_oracle/`). `PORTING.md` 116-118. PORTED, NO
CALLER -- see `UNWIRED.md` for the five things that must exist before it has
one, of which the last is rung 2. Do not take the `max_ctr_complexity > 1`
guard off until they all do.

WHAT REMAINS: `binarizations_manager`'s tensor -> feature-id map
(`InverseCtrs`), `tree_ctrs_dataset.{h,cpp}` + `tree_ctrs.{h,cpp}`,
`ctr_from_tensor_calcer.h`, and `TFeatureTensorTracker`.

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

**OUR GPU ARM AGAINST THEIR CPU ARM IS THE COMPARISON.** That is the plan's
thesis -- GPU ACCESS, NOT TIER -- and it is what `PORTING.md` 108's 144 of
144 already is: our GPU searcher reproducing, split for split, the trees
their CPU learner chose on the same data and the same grid. Not a weaker
substitute for anything. The SPEED half of that same comparison is a
benchmark and has not been run.

What is genuinely left here, all doable locally:

* ~~a CATEGORICAL fixture, split-for-split~~ **DONE 2026-08-21** (`233349b`),
  `bench/oracle_cat.txt`, and it immediately found a shipped bug: the
  pointwise scorer's one-hot flag array was a hardcoded constant, 0 of 18
  one-hot splits matching. Now **192 of 192 across all four fixtures on both
  searchers, 21 of 21 one-hot**. ONE-HOT ONLY, and that is their CPU
  learner's limit rather than a shape chosen to pass (`PORTING.md` 113):
  `IsSupportedCtrType(CPU, FeatureFreq)` is FALSE and `max_ctr_complexity`
  above 1 is refused. **So the CTR half of our categorical path has no oracle
  coverage at all**, and DEVIATION 109 does not repair it -- this is the one
  place their CPU arm cannot express the feature rather than merely running
  it slower.
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
