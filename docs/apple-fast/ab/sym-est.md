# lane/apple-fast-sym-est: leaf estimation for symmetric trees under FAST on Apple

Binding `gbdt`. All code is in `gbdt/methods/leaves_estimation/apple_fast_est.mojo`; the hooks are in
`gbdt/methods/doc_parallel_boosting.mojo` (`_estimate_and_apply` head, the loop head's derivative dispatch, the
symmetric estimation call site). Every switch is compiled only under FAST + Apple and defaults OFF; IDENTICAL and
every other vendor compile main's code unchanged. Profile: `docs/apple-fast/notes/sym-est.md`.

Which board cells can move: the board pins `leaf_estimation_method=Newton`, `leaf_estimation_iterations=1`.
taxi is RMSE at one permutation, so its leaves come from the searcher's own Newton step (DEVIATION 64) and it never
reaches the estimator: taxi lines are null controls. istella is Logloss and runs one estimation task per tree, so
istella is the deciding dataset. `gbdt-ordered` estimates through `ordered_boosting.mojo`'s batched path and is a null
control too.

Any one of the defines below routes the task (single-dim pointwise loss, Newton or Gradient, no query/pair/YetiRank
grouping) through `apple_fast_estimate_and_apply`; Exact, multiclass, MultiRMSE and ranking keep main's path.

`MOJOLEARN_EST_STATS_FUSED`: the per-leaf fold, the Newton step, the Hessian regularizer and `RegularizeImpl` run on
the device (`est_walk_kernel`, one block, one thread per leaf) instead of reading the leaf sums back and doing the
arithmetic on the host. Per istella tree: 2 drains -> 1, 2 readbacks -> 1 (the leaf values, riding the tail drain),
no `d_est` upload, no per-tree host buffer allocations (the task's buffers are a fit-owned pool), no identity or
bins fill (the shift is never used at one iteration). Expected: a few launches and one ~0.2 ms+ wait fewer per tree.
Risk: the device walker computes in f32 what the host computed in f64 (gradient/Hessian quotient), last-bit leaf
differences; quality should be inside FAST spread.

`MOJOLEARN_EST_REUSE_PART`: on top of the above, the evaluation reads the live target, weight and cursor buffers in
ROW order through the searcher's partition (`compute_partition_stats_gather` over `row_index`) instead of gathering
them into bin order first. Removes the 2 (3 weighted) full-row gather passes. Risk: the gathered reduction reads rows
in a scattered order, which may cost more than the gathers it saves on large leaves; that is the question the A/B
answers.

`MOJOLEARN_EST_SHRINK_FUSED`: on top of STATS_FUSED, the cursor add (`AppendModels`) is fused with the NEXT tree's
derivative pass: one full-row kernel (`est_apply_derivs_*_kernel`) adds the leaf value to the row's cursor and writes
the two search planes, the value partials and (when needed) the magnitude partials the loop head would compute; the
loop head skips its `launch_approximate`. Only with one permutation (with more, the next learn permutation is a fresh
draw). Removes one full-row pass and one launch per tree. Risk: none on quality (the same kernels, the same cursor);
the fused kernel adds a row -> leaf scatter launch when REUSE_PART is off.

`MOJOLEARN_EST_ITERS_DEVICE`: `leaf_estimation_iterations > 1` (the Logloss default when the user sets nothing; not
on the board): the whole AnyImprovement line search runs on the device, one `est_walk_kernel` per evaluation deciding
accept / halve / stop, so a tree with k iterations has one drain instead of k+1. No request line: the board pins one
iteration, so a board A/B would be a null. Measure with a user-default Logloss fit if wanted.

`MOJOLEARN_SYM_EST_ALL`: all four.

Not done: `MOJOLEARN_EST_EXACT_SORTLESS` (Exact estimation without the segmented sort). Exact losses (MAE, Quantile,
MAPE) are not on the board, and a selection that returns the same weighted quantile as the sort plus binary search
needs its own design; left for a later pass.

Request lines: `docs/apple-fast/ab/sym-est.txt` (istella singles first, then ALL; symmetric-1000 istella for ALL and
SHRINK_FUSED; symmetric-1000 taxi and ordered taxi as null controls for ALL).
