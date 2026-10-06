# Preprocessing, metrics, resampling and statistical forecasting implementation

All 14 controls are OFF. Source written only: **uncompiled, unverified and unmeasured**, with zero samples. No test, lint, syntax check, build, benchmark or device job was run. No quality or performance claim follows from these edits.

The exact A/B controls are in [preprocessing.json](preprocessing.json). Source edits use supported patterns already present in the repository, but toolchain acceptance remains unknown. No Python runtime numerical work was added.

## AFCL-P01 — Scaler reduction independent accumulators

Two independent per-lane FP32 sums in the two-pass col_stats path; two independent Welford chains followed by existing Chan merge in the current SI_ONEPASS/fused-transform cs_tile path.

Callers: SimpleImputer and PowerTransformer column-statistic consumers; any OP_COL_STATS caller including scaler programs.

Limits: Only column-statistic kernels change; pt_fold_fast_kernel and class_stats_fast_kernel are unchanged. Current SI_ONEPASS is covered; an additional rollback-path comparison may use MOJOLEARN_SI_ONEPASS_OFF in both arms. No claim of coverage for separate core scaler kernels.

Quality still required: Large offsets, NaNs, zero variance, transformed outputs and downstream model quality.

Source: `x_prep/fastred.mojo`, `x_prep/fastpt.mojo`.

## AFCL-P02 — MaxAbs row tiling

MaxAbs direct-fit row chunks 1024/128 become 512/64; feature-width policy is untouched.

Callers: MaxAbsScaler direct FAST fit.

Limits: Requires existing PREP3_MAXABS entrance, enabled by default. Doubles partial buffer size; no change to input pool ownership.

Quality still required: Exact finite maxima, NaN/zero handling and transformed output.

Source: `x_prep/fastmaxabs.mojo`.

## AFCL-P03 — Categorical Naive Bayes count grouping

Categorical count launch uses 64 threads instead of 128; final integer-count conversion launch is unchanged.

Callers: CategoricalNB unweighted categorical atomic count path.

Limits: The existing atomic path is opt-in and must be enabled in both arms. Weighted data stays on its existing fallback; this is not a new atomic algorithm.

Quality still required: Class/feature counts, smoothing, unseen categories and heldout logloss.

Source: `x_prep/fastnb.mojo`, `x_prep/device.mojo`.

## AFCL-P04 — Sparse Naive Bayes row grouping

CSR histogram groups contain 32 rows instead of 64; the 256-thread block and all nonzero processing remain.

Callers: MultinomialNB and ComplementNB CSR count fitting.

Limits: No BernoulliNB coverage claim; dense fitting and CSR prediction traversal are unchanged. Existing NB_TEXT_CSR export must remain enabled.

Quality still required: Counts, class priors, negative-input refusal, posterior quality and empty rows.

Source: `x_prep/fastnb_csr.mojo`.

## AFCL-P05 — Target-encoding work grouping

TargetEncoder global/category folds use independent TE_TGR=128 rather than shared TGR=256, including scratch, strides, tree fold and launch size.

Callers: TargetEncoder global and gathered-category fold stages.

Limits: Existing TE_GLOBAL and TE_ENC paths stay enabled in both arms. Non-gathered category fallback is unchanged; all cross-fitting and smoothing are retained.

Quality still required: Fold isolation, category smoothing, unseen-category behavior and downstream task quality.

Source: `x_prep/fastprep2.mojo`.

## AFCL-P06 — Iterative-imputer convergence fold

Existing imputer row-max convergence reduction uses 512 rather than 256 lanes, with matching shared array, stride and launch.

Callers: IterativeImputer II_CONV when precomputed row sums exist.

Limits: Runtime II_CONV prerequisite is identical in both arms. Does not enable row sums or change tolerance, iteration count, flag observation or the Gram path.

Quality still required: Imputed values, observed entries unchanged, stopping semantics and downstream quality.

Source: `x_prep/fastprep2.mojo`.

## AFCL-P07 — Feature-selection row tile

Feature statistic row tiles use RPT=32 rather than 64, RT=256 rather than 512, retaining the 32-column by 8-row-lane block.

Callers: f_regression, r_regression, f_classif and applicable selectors.

Limits: Both scratch sizing and kernels consume existing shared constants. Existing SELECT_FREG/FCLS defaults must remain enabled; large-class fallback remains as before.

Quality still required: Scores, p-values, selected features, class imbalance and force_finite behavior.

Source: `x_prep/select_fast.mojo`.

## AFCL-P08 — Regression metric independent accumulators

Two independent software-binary64 accumulation chains in uniform and weighted multioutput regression score averaging.

Callers: Regression metric KIND_AVG epilogues for multiple outputs.

Limits: Per-observation metric folds do not change. Scalar-output callers are not a speed hypothesis; all zero-weight/NaN rules and binary64 arithmetic remain. The host-run FAST unit on an Apple build shares the guarded epilogue; no host fallback is added.

Quality still required: Independent FP64 metric agreement, sample weights, multioutput and finite policy.

Source: `x_metrics/reg_epi.mojo`.

## AFCL-P09 — Ranking metric prefix work grouping

Weighted/unweighted curve-prefix chunk size 1024 becomes 512 in the planner and emitted stage parameters.

Callers: ROC/PR/AUC curve callers and weighted-percentile consumers of CURVE_CHUNK.

Limits: Sort order, equal-score groups and curve-fold chunks remain unchanged. Extra prefix scratch can trigger the existing bounded-plan fallback; all affected callers need future full-workload coverage. Host planned FAST programs on Apple use matching chunk parameters.

Quality still required: Exact threshold membership, tie groups, weighted areas and degenerate labels.

Source: `x_metrics/plan.mojo`.

## AFCL-P10 — Resampling gather tile geometry

Grouped resample gather tile 32 features by 8 rows becomes 16 features by 16 rows, retaining 256 threads.

Callers: Public float32 resample GPU grouped-gather path.

Limits: Three existing opt-in prerequisites in both arms. Nonreplacement additionally needs MOJOLEARN_RESAMPLE_FAST_DEVICE_PERMUTE in both. Does not change bootstrap/permutation_test independent statistical kernels or promise all dtypes enter the gather path.

Quality still required: Exact paired-array row draws, old-output lifetime and statistical interval/p-value quality.

Source: `resample/gather_fast.mojo`, `resample/estimator.mojo`.

## AFCL-P11 — Stationarity series reduction schedule

Two independent per-thread chains for series sums/squared sums followed by the existing STATS_TPB block reduction.

Callers: KPSS and AutoARIMA differencing selection using series_sum_kernel.

Limits: Narrowed from proposed launch-width change to local accumulation scheduling to avoid changing shared core STATS_TPB. Decisions near significance thresholds require future quality qualification; no scan/order/search settings change.

Quality still required: KPSS statistics, differencing decisions, finite-input refusals and forecast quality.

Source: `tsa/impl/timeSeries/stationarity.mojo`.

## AFCL-P12 — Batched ARIMA likelihood work grouping

Kalman series block default 32 becomes 64; propagated through public likelihood/forecast helpers, packed likelihood, optimizer and retained-evaluation workspace entrances.

Callers: ARIMA and AutoARIMA scalar-lane Kalman evaluation/forecast routes.

Limits: Every lane retains its complete recurrence. Explicit user/caller kalman_tpb overrides are respected. Separate time-scan kernels, if selected by an existing opt-in, are not changed. No reduced precision or fewer optimizer evaluations.

Quality still required: Likelihood/gradient quality, selected orders, optimizer status and heldout forecast error.

Source: `arima/impl/batched_kalman.mojo`, `arima/impl/batched_arima.mojo`, `arima/impl/fast_eval_ws.mojo`, `arima/estimator.mojo`.

Before any future qualification, resolve every affected full-workload recipe and run the declared independent quality gates. Include P01+P05/P07 preprocessing/model interactions, P08+P09 metric consumption, and P11+P12 complete AutoARIMA. No existing board values were changed.

## AFCL-P13 — Gaussian Naive Bayes class-statistic chains

Two independent per-lane chains for unweighted class-filtered sums and centered squared deviations; existing count and reduction tree remain.

Callers: GaussianNB and unweighted OP_CLASS_STATS consumers.

Limits: Existing MOJOLEARN_XPREP_FAST_FOLDS must remain enabled; weighted class-statistic and feature-selector specialized routes are unchanged. This does not claim the separate naive_bayes/ native fit kernel was modified.

Quality still required: Class counts/means/variances, variance smoothing, rare classes and heldout logloss.

Source: `x_prep/fastred.mojo`.

## AFCL-P14 — Holt-Winters series launch geometry

Default optimizer group width HW_OPTIM_TPB changes 128 to 64; existing callers continue using the shared default.

Callers: HoltWinters fitting and applicable optimizer/evaluation helpers that use HW_OPTIM_TPB.

Limits: Separate HW_PREDICT_TPB output geometry is unchanged. Explicit tpb_optim overrides still win; unrelated ETS implementations are not changed. Every seasonal recurrence, parameter search and stopping setting remains.

Quality still required: Heldout forecast error, smoothing parameters, initial states, seasonal tails and refusal handling.

Source: `holtwinters/impl/internal/hw_utils.mojo`.
