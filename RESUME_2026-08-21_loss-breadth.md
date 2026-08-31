# The loss-breadth round, 2026-08-21

**This is a separate file on purpose.** `RESUME.md` was already modified in
the working tree by another session, so this round's handoff does not go in
it -- a shared file with two sessions writing into it is the merge point
PORTING_RULES 12 says to avoid. `bindings/_mojolearn_estimators.mojo`,
`bindings/build_estimators.sh`, `decomposition/estimator.mojo` and the
`bench/results/` additions are that session's and were not touched.

Fold this into `RESUME.md` when the tree is quiet.

## What landed

**Nine more pointwise losses.** Quantile, MAE, LogLinQuantile, MAPE, Poisson,
Lq, Expectile, Tweedie, Huber, joining RMSE / Logloss / CrossEntropy. They go
through one generic `pointwise_target_kernel[objective, estimation]`, which is
their `PointwiseTargetImpl<TTarget>` with the host switch
(`pointwise_targets.cu:447-519`) ported as `launch_pointwise_target_kernel`
and the objective moved from a template instantiation to a comptime parameter
(DEVIATION 63).

**The options layer that decides their leaves.** `gbdt/options/
loss_description.mojo` ports `GetEstimationMethodDefaults` and
`SetLeavesEstimationDefault` (`catboost_options.cpp:30-360`). This is not
optional decoration: four of the nine have a second derivative that is
identically zero, and a Newton step on them divides by `lambda` alone -- it
does not crash and it does not look wrong, it fits a different model. The
resolved table now matches theirs exactly (verified by printing it):

    RMSE            Newton  1     Logloss/CrossEntropy  Newton 10
    Quantile        Exact   1     MAE                   Exact  1
    MAPE            Exact   1     LogLinQuantile        Gradient 1
    Poisson         Newton 10     Lq (q<2)              Gradient 1
    Expectile       Newton  5     Tweedie               Newton 20
    Huber           Newton  1

**The Exact estimator**, because that table demands it for three losses:
`gbdt/methods/kernel/exact_estimation.mojo` (their four kernels),
`gbdt/methods/leaves_estimation/leaves_estimation_helper.mojo`
(`ComputeWeightedQuantile` + `ComputeExactApprox`), and
`gbdt/gpu_util/kernel/segmented_sort.mojo` (DEVIATION 65). The oracle
gained `estimate_exact` and a Gradient arm in `write_second_derivatives`.

**Bernoulli and Poisson bootstrap**, their `UniformBootstrapImpl` and
`PoissonBootstrapImpl` beside the Bayesian arm already here, plus `NextNormal`
and `NextPoisson`. NOT filtered (DEVIATION 69).

**A caller-facing surface**: `gbdt/estimator.mojo` and
`python/mojolearn/ensemble.py`. The model crosses as its own text, so the
boundary inherits `check-model-io`'s gate instead of needing a new one.

## What to pick up

1. **THE PYTHON BINDING DOES NOT LOAD** and it is a toolchain problem, not a
   port problem -- PORTING.md 70, five hypotheses killed. `mojo build --emit
   shared-lib` embeds 0 compiled Metal kernels against an executable's 81,
   and the JIT fails on the shared-memory ones. Also found on the way:
   **`bindings/build.sh` passes `--target-accelerator metal:1`, which is not
   a value the compiler recognises** -- `--print-supported-accelerators`
   lists `apple-m1-metal4` .. `apple-m5-metal4`. Fixing that alone does not
   fix the loader, but it should be fixed.
2. **MultiClass**, scoped against their source in ROADMAP.md. The Hessian
   solve is NOT the hard part -- it is twenty-five lines of host Cholesky on
   a 6x6 (`descent_helpers.cpp:91-117`). The dominant cost is
   multi-dimensional leaf values through the model, its text format and both
   evaluators.
3. **Two open items in the ledger**, both unmeasured: DEVIATION 62 (Tweedie's
   parameter, never compared against their CPU on a real fixture) and
   DEVIATION 69's cost (how much of the histogram we waste by not filtering).
   DEVIATION 64 was open and is now measured: 8192/8192 bit-identical.

## Mirror repairs made the same day

Two files of this round were at the wrong address and both are fixed, because
PORTING_RULES 4 -- their paths are our paths -- is what lets a reviewer open
one of their files beside one of ours and diff branch for branch:

* `GetEstimationMethodDefaults` and `SetLeavesEstimationDefault` were written
  into `gbdt/options/loss_description.mojo`, beside the parameters they read.
  They are `catboost_options.cpp:30-360`, so they live in
  `gbdt/options/catboost_options.mojo` now. That also fixed an import
  pointing the wrong way: their graph runs `catboost_options.h` ->
  `loss_description.h` and never back.
* `segmented_radix_sort.mojo` is `segmented_sort.mojo`, because their file is
  `cuda_util/kernel/segmented_sort.cu`.

`gbdt/NOT_IMPLEMENTED.tsv` is new and lists eleven symbols this section deliberately
does not port, with the reason for each -- the register `cluster/` already
had and `gbdt/` did not.

## MultiClass: the estimation half is done

DONE, each with its own gate:

* `gbdt/targets/kernel/multilogit.mojo` -- their two MultiLogit kernels
  (`multilogit.cu:10-169`). `check-multilogit`.
* `gbdt/lapack/linear_system.mojo` -- `SolveLinearSystemCholesky`, which is
  their `dposv_` transcribed, INCLUDING its behaviour when the matrix is not
  positive definite: the right-hand side is left as the raw gradient and
  their own `CB_ENSURE(info >= 0)` passes. DEVIATION 74.
* The walker's BLOCKED arm, `descent_helpers.cpp:91-117`. Gated in
  `newton_walker_check` by a block-quadratic whose exact minimizer is known,
  at rowSize 2/3/6, with two sabotages.
* The oracle's `rowSize > 1` arm: the multi-column reduce, the gradient's
  reconstructed last component, the blocked lower-triangular Hessian, the
  GAUGE FIXING in `MakeEstimationResult`, a multi-dimensional `MoveTo` that
  projects before it subtracts, and a `Regularize` that zeroes every approx
  dimension. `check-multiclass-oracle`.
* `add_model_value_kernel` gained an approx-dimension axis. DEVIATION 76.

TWO STRUCTURAL FACTS THE PORT NEARLY GOT WRONG, both found by reading their
header rather than by reasoning:

1. **`SingleBinDim() = cursorDim + 1` for MultiClass**
   (`pointwise_oracle.h:57-64`). The LEAF has one more dimension than the
   cursor. The Hessian is therefore `numClasses x numClasses`, which is why
   `MultiLogitSecondDerRowImpl`'s `else` branch at `:157` exists -- it is
   the PINNED class's row, not a defensive arm. A port that stopped the row
   loop at `cursorDim` would build a matrix one row short and Cholesky would
   solve a different system.
2. **`HessianBlockSize()` returns 1 unless the method is Newton AND the
   Hessian type is not Diagonal** (`:30-39`). A Gradient-method MultiClass
   fit is DIAGONAL. Reading it as "MultiClass is blocked" sends a Gradient
   fit down the wrong arm.

* The SCORE kernel's MultiClass arm (`compute_scores.cu:105-110`): the extra
  leaf contribution standing for the pinned class. `AddLeaf` was factored out
  of the split loop first, because their `calcer.AddLeaf` is called from
  three places once MultiClass exists and writing it three times is how the
  three drift apart. `check-multiclass-score`, and `check_scores` re-run to
  confirm the extraction left the single-stat path alone.

**MULTICLASS TRAINS END TO END as of 2026-08-21.** `check-multiclass-train`
covers numClasses 2, 3 and 7: the loss falls, argmax accuracy reaches
0.94-0.99 against chance while a RANDOM-label control stays far below it,
probabilities sum to one, the model's `dim` is `numClasses - 1`, and the
save/load round trip is bit-identical over up to 24,576 approxes.

THE GATE THAT MATTERS THERE is the cross-check between the two apply paths:
training updates the cursor through `add_model_value_kernel`, which walks
PARTITIONS and never evaluates a split, while prediction goes through
`compute_bins_and_add_kernel`, which walks ROWS and re-derives each leaf.
Different kernels, different buffers, different orders -- and on the learn
set the loss they produce agrees to 1.7e-8 to 5.3e-7 relative. That is what
would catch a plane-major buffer read as bin-major, which is a live risk
because `leaf_values` is bin-major and the cursor is plane-major.

TWO DEFECTS IT FOUND on its first run, both recorded (PORTING.md 80, 82):
the walker was not applying `MakeEstimationResult` on return, and the text
format wrote `n_leaves` values per tree instead of `n_leaves * dim`, which
would have lost every dimension but the first on save.

NOT DONE, and this is the other half:

* **The DEVICE EVALUATOR** (`gbdt/models/cuda/evaluator.mojo`) is still
  one-dimensional. `predict_multi_floats` goes through the tree-wise apply,
  which is multi-dimensional; the separate device evaluator that
  `check-ctr-apply` and `check-catboost-apply` exercise is not, and will
  refuse or mispredict a `dim > 1` model. Nothing routes a MultiClass model
  to it today, which is exactly the condition PORTING_RULES 8 warns about.
* **`MultiClassOneVsAll`**, whose `GetHessianType()` is Diagonal rather than
  Symmetric (`multiclass_targets.h:118-123`) and which therefore takes the
  DIAGONAL walker arm. Its kernels are in `multilogit.cu` beside the ported
  ones and are listed in `gbdt/NOT_IMPLEMENTED.tsv`.
* **The Python surface for MultiClass** -- `gbdt/estimator.mojo` and
  `python/mojolearn/ensemble.py` are one-dimensional and would need
  `predict_proba` routed through `multiclass_probabilities`. Blocked behind
  the loader anyway (PORTING.md 70).

## New checks

    pixi run check-pointwise-target    10 objectives x 4 modes per cell
                                       against libm through FFI, 6 sabotages
    pixi run check-exact-estimation    the analytic weighted quantile over
                                       10 ragged leaves, 4 sabotages
    pixi run check-multilogit          MultiClass's two kernels, four gates,
                                       3 sabotages + 1 recorded impossible

`original/bootstrap_check.mojo` gained `check_bernoulli_and_poisson`, which
pins Bernoulli at rate 1.0 and 0.0 as exact identities and Poisson(1) against
its own distribution (measured: 36,797 zeros of 100,000 against p(0)=0.3679;
8,046 draws >= 3 against p=0.0803).

**Nothing is committed.** Every change above is in the working tree.
