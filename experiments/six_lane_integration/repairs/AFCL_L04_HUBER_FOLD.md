# AFCL-L04 Huber final-fold repair

`compensated-final-v1` repairs an uncorrected float32 reduction in the
default-off Apple FAST Huber candidate. It is source implemented only:
**NOT COMPILED, NOT EXECUTED, QUALITY NOT VERIFIED, NOT MEASURED**. The
retained combined candidate's Huber/Istella quality failure remains a
failure. This source change does not establish that the repaired candidate
passes or that L04 alone caused the earlier result.

The original `MOJOLEARN_AFCL_L04=1` changed row partials from 512 to 256
rows. Each partial still accumulates from zero, and the final fold receives
approximately twice as many float32 words. That final fold previously used
one uncorrected ascending sum for every gradient cell and every loss or
weight sum. Its rounding can perturb the objective, gradient, Armijo
decision, curvature history and subsequent L-BFGS path. Shorter row partials
do not by themselves control error in the longer final reduction.

The repaired L04 kernel retains the 256-row schedule and carries a second
float32 rounding residual using the existing `x_linear.ff.ff_add_f`
TwoSum-based arithmetic. It rounds the accumulated high and low words back
to float32 before the existing objective/optimizer step. Outlier counts
remain integer sums of bitcast count words. The launch grid, scratch layout
and witness slots are unchanged. No new hardware feature or toolchain mode
is required by the source. The compensation applies to all shapes; there
is no dataset or dimension dispatch.

The compile-time branch selects this fold only under Apple FAST L04.
The incumbent still calls `hg_fold_kernel` and uses its existing default
512-row schedule. IDENTICAL, other vendors, the shared grid fold, row
partial expressions, solver settings, sample-weight rules, stopping budget
and acceptance gates are untouched. This repair does not address rounding
inside the partials, the residual map, or the remaining float32 optimizer.

The failed retained experiment used one excluded warmup and one scored
sample per arm, with the complete proposed bundle enabled in A. On full
Istella, A/B R2 was -0.007089342908900065/-0.005952189726329049 and RMSE was
0.8373928001448464/0.836919896299495. Scored whole-operation times were
2.4091830409597605/2.404323499999009 seconds. These are **old-source
observations**, not measurements of this repair. The committed
[source audit](../measurements/20261006/apple-huber-istella-source-audit.json)
retains the failure and artifact hashes. No full fitted-state export was
available, so output hashes do not establish model-state identity.

The companion [repair record](AFCL_L04_HUBER_FOLD.json) retains exact saved
input recipes, hashes, settings, prior evidence references and the future
comparison scopes. It is a review record, not an executable queue. The
existing implementation ID remains `AF.C.AFCL-L04`; the revision ID is
`AF.C.AFCL-L04:compensated-final-v1`, distinguished by its new source and
binary hashes. No existing result or catalog freeze is relabeled.

The measurement owner should freeze the reviewed source, compile the
changed Apple FAST candidate binding configurations and actual dependents,
and create fresh receipts. Old candidate binaries cannot execute the new
fold. Preserve the incumbent B configuration and its genuine provenance.
Do not mutate an active frozen package or queue.

After that freeze, use the existing full regression inputs for both
`algos/huber@dataset=taxi` (5,250,086 fit rows, 11 features) and
`algos/huber@dataset=istella` (2,043,304 fit rows, 220 features), each with
500,000 held-out query rows. Compare isolated repaired L04 against the
incumbent, then the repaired complete proposed configuration against the
same incumbent. Each pair uses one excluded warmup and one scored sample
per arm, serially under the owner's device lock, with the original
whole-operation boundary and separate fit/inference reporting. Preserve
the saved standardized arrays, splits, seeds, constructor settings and
existing quality/opponent rules. Do not rerun the historically capped
default input preparation.

The existing race does not exercise sample weights or `fit_intercept=False`,
nor retain a complete model state, per-trial objective/gradient/line-search
history, or all outlier diagnostics. Those remain explicit coverage gaps;
the repair does not invent new races or treat the two recipes as proof of
these properties. Cross-vendor bitwise equality is not Apple FAST's
acceptance requirement. Retain failures, timings and qualification status
separately, and publish only through the existing board tools.

No compiler, binding import, estimator, data-array load, numerical fixture,
quality evaluation, identity check or timing run was executed for this
patch. Validation is restricted to source review, JSON consistency and
`git diff --check`.
