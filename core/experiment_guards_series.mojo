# SPDX-License-Identifier: Apache-2.0
"""Opt-in strategy admission for the time-series bindings (tsa, ARIMA, forecast, resample).

Each binding of this family imports SERIES_GUARDS_OK and asserts it in its build function, so every compile of one of
those bindings (per-binding or merged) evaluates these guards. A guard whose define is read in code by bindings
of more than one family belongs in core/six_lane_experiment_guards.mojo, which every binding reaches through
checks/numerics.mojo.
"""
from std.sys.compile import is_defined, get_defined_int
# The entries below moved verbatim, in their original order, from core/six_lane_experiment_guards.mojo on
# 2026-11-04 (lane split-hot-groups; the placement of each entry is docs/build/guard_split.json). A header comment
# that headed lines of several families is carried with each of them.


def _check_configuration() -> Bool:
    # Deleted 2026-10-07 (lane serial-cleanup): forbidden serial shape; incumbent route is parallel. Recoverable at origin/integration/switches-20261007 608a7cf4a.
    comptime assert not is_defined["MOJOLEARN_C60_DIFF_REUSE"](), "removed: C60_DIFF_REUSE was dead code (select_d never reaches d == 2)"
    # Deleted 2026-10-08 (lane grid-losers-1): IDENTICAL grid ge123e6f9 losers (docs/apple-fast/EXPERIMENTS.md). Recoverable at main ad7ed2370.
    # C25_PROJECTION_REUSE: deleted 2026-10-08 above, restored 2026-10-10 (lane c25-restore-promote) from the tombstone patch after grid pass 2.
    comptime assert not is_defined["MOJOLEARN_C58_SHARED_PREP"](), "removed: MOJOLEARN_C58_SHARED_PREP retired 2026-10-08: noise, NV 0.996x / AMD 0.980x on ets (grid ge123e6f9); see EXPERIMENTS.md"
    # Lane owed-deletions-D1 (2026-10-09): DROPPED rows in docs/apple-fast/EXPERIMENTS.md whose code was still on main (docs/TOMBSTONES.md). Deleted code recoverable at main b639a2bd2.
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_CSS_SEARCH"](), "removed 2026-10-09 (lane/owed-deletions-D1): AutoARIMA CSS-scored order search was DROPPED-quality: rab26 synthetic -45.2% / taxi-hourly -30.5% ms but forecast_rmse 2.465 -> 3.485 / 75.71 -> 76.06; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_STEPWISE"](), "removed 2026-10-09 (lane/owed-deletions-D1): AutoARIMA Hyndman-Khandakar stepwise device search was DROPPED-slower: rab26 synthetic +51.6% / taxi-hourly +102.5% ms, taxi forecast_rmse 75.71 -> 90.54; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_GROUPS_CONCURRENT"](), "removed 2026-10-09 (lane/owed-deletions-D1): AutoARIMA Kalman groups in one concurrent round loop was DROPPED-slower: autoarima +14-16% (M3 verdicts batch 6, 1 run per arm); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_D_CONCURRENT"](), "removed 2026-10-09 (lane/owed-deletions-D1): AutoARIMA d groups in one native search call was DROPPED-slower: autoarima +5-7% (M3 verdicts batch 6, 1 run per arm); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_RESAMPLE_FAST_RANK_SORT"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple bootstrap distribution sorted by one rank launch was DROPPED-slower: bootstrap taxi 14.38 -> 13.64 ms, istella 11.59 -> 14.46 ms (M3 afc_ab_def ab1 d51f4b4bf, mixed: slower on istella); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_RESAMPLE_FAST_ONE_FOLD"](), "removed 2026-10-09 (lane/owed-deletions-D1): FAST Apple bootstrap mean / diff_means replicates folded once per block was DROPPED-slower: bootstrap taxi 16.31 -> 13.89 ms, istella 12.84 -> 14.23 ms (M3 afc_ab_def ab1 d51f4b4bf, mixed: slower on istella); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_TSA2_KPSS"](), "removed 2026-10-03 by 7ff2caf99 (refusal added 2026-10-09, lane/owed-deletions-D1): FAST Apple one-launch kpss_fused was DROPPED-noise: kpss taxi-hourly -73%, synthetic -62%, the same gain as the kept TSA_FAST_KPSS_PACK; code at 7ff2caf99^; see docs/TOMBSTONES.md"
    # wave-2 w2-decide-neural (2026-10-11): A/B on main c8ef6c66d, one run per arm, NV L40S nv2 / AMD MI325X, digests NV==AMD
    # on every cell (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch).
    # Promoted: the old positive define is refused; -D <NAME>_OFF restores the old path.
    comptime assert not is_defined["MOJOLEARN_IDN_ETS_TEAM"](), "promoted 2026-10-11 (lane/w2-decide-neural): the ETS(A, A|Ad, N) team is the IDENTICAL default (damped-ets 0.014-0.017x, bits change NV==AMD, forecast_rmse equal, nv2 v0099 / amd a1681); drop -D MOJOLEARN_IDN_ETS_TEAM, -D MOJOLEARN_IDN_ETS_TEAM_OFF restores op_ets"
    # Lane k1-delete-unverdicted-fast (2026-11-09): Apple FAST experiments with no verdict, never to be measured (Apple FAST
    # investment stopped); deleted (docs/TOMBSTONES.md "Unverdicted FAST experiments deleted (lane K1)").
    # Restore: git apply experiments/removed/MOJOLEARN_ARIMA_FAST_K3_DF_PROBE.patch (both defines)
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_K3_DF_PROBE"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_ARIMA_FAST_K3_DF_PROBE, the private Apple FAST double-float K3 likelihood probe (`_arima_k3_df_probe`, arima/impl/fast_scalar_df.mojo), was never measured on a fit (private-kernel quality only) and never will be; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_PRODUCT_DF_TAIL"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_ARIMA_FAST_PRODUCT_DF_TAIL (F16), the Apple FAST compensated Kalman likelihood and finite-difference tail (arima/impl/fast_eval_df.mojo), had no verdict row (F16 M3 one-sample B/A 1.13 / 1.04 / 1.01, left off) and will not be measured; code at f59d913b0b; see docs/TOMBSTONES.md"

    comptime assert not is_defined["MOJOLEARN_IDN_ARIMA_ASYNC_OFF"](), "removed 2026-11-09 (lane/n10-dictl-arima-loops): the lock-step ARIMA L-BFGS this kept under IDENTICAL is deleted; the per-series solver (fast_lbfgs_async.mojo, same bits per series) is the device's only arm; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_ASYNC_OFF"](), "removed 2026-11-09 (lane/n10-dictl-arima-loops): the lock-step ARIMA L-BFGS this restored is deleted (M3 A/B autoarima taxi-hourly 39,639 -> 24,563 ms, synthetic 28,511 -> 14,918 ms for the per-series solver); code at f59d913b0b; see docs/TOMBSTONES.md"
    # Lane expdel-02-decomposition-series (2026-11-15): owed deletions of docs/plans/EXPERIMENT_DELETIONS_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_ARIMA_FAST_LLONLY_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_LLONLY_OFF"](), "removed 2026-11-15 (lane/expdel-02-decomposition-series): ARIMA_FAST_LLONLY_OFF (FAST+Apple rollback to the storing Kalman loop kernel inside the fit evaluations) is gone; the LL_ONLY loop is the FAST+Apple path (autoarima taxi-hourly 50,459 -> 47,023 ms, rmse identical, KEPT 6c54b7e87); restore: experiments/removed/MOJOLEARN_ARIMA_FAST_LLONLY_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ARIMA_FAST_EVAL_WS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_EVAL_WS_OFF"](), "removed 2026-11-15 (lane/expdel-02-decomposition-series): ARIMA_FAST_EVAL_WS_OFF (FAST+Apple rollback to about 45 device buffers allocated and the series copied N + 1 times per candidate point) is gone; the held stacked workspace is the FAST+Apple path (autoarima taxi-hourly 50,299 -> 42,921 ms alone, 50,500 -> 39,504 ms with LLONLY, KEPT 6c54b7e87); restore: experiments/removed/MOJOLEARN_ARIMA_FAST_EVAL_WS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RESAMPLE_FAST_IDX_DIRECT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RESAMPLE_FAST_IDX_DIRECT_OFF"](), "removed 2026-11-15 (lane/expdel-02-decomposition-series): RESAMPLE_FAST_IDX_DIRECT_OFF (FAST+Apple rollback to the host buffer -> List -> store-loop index route) is gone with RESAMPLE_IDX_DIRECT and the binding's List route; every tier copies the drawn indices straight into the caller's buffer (resample taxi 71.7 -> 62.8 ms, same integers); restore: experiments/removed/MOJOLEARN_RESAMPLE_FAST_IDX_DIRECT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RESAMPLE_FAST_PERM_SELECT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RESAMPLE_FAST_PERM_SELECT_OFF"](), "removed 2026-11-15 (lane/expdel-02-decomposition-series): RESAMPLE_FAST_PERM_SELECT_OFF (FAST+Apple rollback to perm_select_stat_kernel for the mean / diff_means permutation null) is gone; the 4-bit radix select is the FAST+Apple path (permutation-test taxi 55.41 -> 35.49 ms, istella 55.28 -> 35.28 ms, pvalue identical); restore: experiments/removed/MOJOLEARN_RESAMPLE_FAST_PERM_SELECT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CV_FAST_SLICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CV_FAST_SLICE_OFF"](), "removed 2026-11-15 (lane/expdel-02-decomposition-series): CV_FAST_SLICE_OFF (FAST+Apple rollback to four _take_rows byte gathers per KFold fold) is gone; the row-range views are the FAST+Apple path (cross-val-score taxi 157.93 -> 126.18 ms, istella 3114.45 -> 2920.08 ms, mean_r2 identical); restore: experiments/removed/MOJOLEARN_CV_FAST_SLICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CV_FAST_TRUST_FOLDS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CV_FAST_TRUST_FOLDS_OFF"](), "removed 2026-11-15 (lane/expdel-02-decomposition-series): CV_FAST_TRUST_FOLDS_OFF (FAST+Apple rollback to the _indices / _overlap checks on the native default folds) is gone; trusting the native partition is the FAST+Apple path (cross-val-score taxi 161.86 -> 142.47 ms, mean_r2 identical); restore: experiments/removed/MOJOLEARN_CV_FAST_TRUST_FOLDS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TSA_FAST_SELD_FUSED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TSA_FAST_SELD_FUSED_OFF"](), "removed 2026-11-15 (lane/expdel-02-decomposition-series): TSA_FAST_SELD_FUSED_OFF (FAST+Apple rollback to the per-round select_d launches) is gone; select_d as one launch per batch is the FAST+Apple path (select-d taxi-hourly 8.5 -> 2.0 ms, synthetic 8.0 -> 2.2 ms, KEPT 6c0ba4379); restore: experiments/removed/MOJOLEARN_TSA_FAST_SELD_FUSED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C11_DRAW_GATHER.patch
    comptime assert not is_defined["MOJOLEARN_CLASSICAL_C11_DRAW_GATHER"](), "removed 2026-11-15 (lane/expdel2-07-base): CLASSICAL_C11_DRAW_GATHER (Resample (bootstrap draw gather, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at bfc7696dd0; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_C58_SERIES4.patch
    comptime assert not is_defined["MOJOLEARN_C58_SERIES4"](), "removed 2026-11-15 (lane/expdel2-07-base): C58_SERIES4 (Time series (Holt-Winters four series per thread, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 236e97faa2; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_C59_TRIAL_STATE.patch
    comptime assert not is_defined["MOJOLEARN_C59_TRIAL_STATE"](), "removed 2026-11-15 (lane/expdel2-07-base): C59_TRIAL_STATE (Time series (ARIMA trial state reuse, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 7935a1db35; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_C60_LAG4.patch
    comptime assert not is_defined["MOJOLEARN_C60_LAG4"](), "removed 2026-11-15 (lane/expdel2-07-base): C60_LAG4 (Time series (shared detrended observations across lag tasks, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at ad78b8054f; see docs/TOMBSTONES.md"
    # Lane expdel2-28-neural (2026-11-15): unmeasured arms and superseded old paths deleted, plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history) (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_ETS_TEAM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ETS_TEAM_OFF"](), "removed 2026-11-15 (lane/expdel2-28-neural): ETS_TEAM_OFF (old path of the FAST Apple default) is gone; the default is kept; code at 8d47995bad; restore: experiments/removed/MOJOLEARN_ETS_TEAM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ETS_TEAM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ETS_TEAM_OFF"](), "removed 2026-11-15 (lane/expdel2-28-neural): IDN_ETS_TEAM_OFF (old path beside the measured default) is gone; the default is kept; code at 0fe5074767; restore: experiments/removed/MOJOLEARN_IDN_ETS_TEAM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NM_SPEC_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NM_SPEC_OFF"](), "removed 2026-11-15 (lane/expdel2-28-neural): IDN_NM_SPEC_OFF (old path beside the measured default) is gone; the default is kept; code at 33ca37483e; restore: experiments/removed/MOJOLEARN_IDN_NM_SPEC_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_CROSTON_REG_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_CROSTON_REG_OFF"](), "removed 2026-11-15 (lane/expdel2-28-neural): SEQ_CROSTON_REG_OFF (old path of the FAST Apple default) is gone; the default is kept; code at a2b075952e; restore: experiments/removed/MOJOLEARN_SEQ_CROSTON_REG_OFF.patch"
    # Lane expdel2-36-neural-series-linear (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_SELECT_D_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SELECT_D_OFF"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): SELECT_D_OFF (old path beside a recorded default, Series (AutoARIMA select_d FAST+Apple device rounds)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_SELECT_D_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SELECT_D_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SELECT_D_OFF"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): IDN_SELECT_D_OFF (old path, no A/B recorded, Series (AutoARIMA select_d IDENTICAL device rounds)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_SELECT_D_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): IDN_KPSS_ONE_WAIT_OFF (old path, no A/B recorded, Series (KPSS one-wait binding call)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_KPSS_ONE_WAIT_OFF.patch"
    # MOJOLEARN_AFCL_P11: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KPSS_SCAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KPSS_SCAN_OFF"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): IDN_KPSS_SCAN_OFF (old path, no A/B recorded, Series (KPSS blocked scan)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_KPSS_SCAN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TSA_FAST_KPSS_PACK_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TSA_FAST_KPSS_PACK_OFF"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): TSA_FAST_KPSS_PACK_OFF (old path beside a recorded default, Series (KPSS packed one-launch FAST+Apple)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_TSA_FAST_KPSS_PACK_OFF.patch"
    # ---- end lane expdel2-36-neural-series-linear
    # Lane expdel2-37-series (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # MOJOLEARN_AFCL_P10: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_P12: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_P14: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ARIMA_ASYNC_ALL.fb12-loser-arms.patch
    comptime assert not is_defined["MOJOLEARN_ARIMA_ASYNC_ALL"](), "removed 2026-11-17 (lane/fb12-loser-arms): ARIMA_ASYNC_ALL (decided B arm, per-series L-BFGS on every route; L40S ba08dac006 B 1.04x, no gain) is gone; the default path is the only one; code recoverable at 2203db7094; restore: experiments/removed/MOJOLEARN_ARIMA_ASYNC_ALL.fb12-loser-arms.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ARIMA_FAST_BATCH_GRAD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_BATCH_GRAD_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): ARIMA_FAST_BATCH_GRAD_OFF (old path, Time series (ARIMA stacked gradient, Apple FAST); no A/B recorded) is gone; the default is kept; code recoverable at b22c6673d4; restore: experiments/removed/MOJOLEARN_ARIMA_FAST_BATCH_GRAD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ARIMA_FAST_CONST_BOTH.patch
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_CONST_BOTH"](), "removed 2026-11-15 (lane/expdel2-37-series): ARIMA_FAST_CONST_BOTH (opt-in arm, Time series (AutoARIMA with/without constant grid); no A/B recorded) is gone; the default path is the only one; code recoverable at f1218d6715; restore: experiments/removed/MOJOLEARN_ARIMA_FAST_CONST_BOTH.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ARIMA_FAST_KPSS_D.patch
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_KPSS_D"](), "removed 2026-11-15 (lane/expdel2-37-series): ARIMA_FAST_KPSS_D (opt-in arm, Time series (AutoARIMA statsforecast ndiffs d); no A/B recorded) is gone; the default path is the only one; code recoverable at 2044a9635d; restore: experiments/removed/MOJOLEARN_ARIMA_FAST_KPSS_D.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ARIMA_FUSED_EVAL_TAIL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ARIMA_FUSED_EVAL_TAIL_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): ARIMA_FUSED_EVAL_TAIL_OFF (old path, Time series (ARIMA fused evaluation tail); no A/B recorded) is gone; the default is kept; code recoverable at 96d454df19; restore: experiments/removed/MOJOLEARN_ARIMA_FUSED_EVAL_TAIL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ARIMA_ID_BATCH_GRAD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ARIMA_ID_BATCH_GRAD_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): ARIMA_ID_BATCH_GRAD_OFF (old path, Time series (ARIMA stacked gradient, Apple IDENTICAL); no A/B recorded) is gone; the default is kept; code recoverable at b22c6673d4; restore: experiments/removed/MOJOLEARN_ARIMA_ID_BATCH_GRAD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ARIMA_ORDER_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ARIMA_ORDER_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): ARIMA_ORDER_BATCH_OFF (old path, Time series (AutoARIMA grouped order fits); no A/B recorded) is gone; the default is kept; code recoverable at e4949779dc; restore: experiments/removed/MOJOLEARN_ARIMA_ORDER_BATCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_HW_EST_BLOCK64.patch
    comptime assert not is_defined["MOJOLEARN_HW_EST_BLOCK64"](), "removed 2026-11-15 (lane/expdel2-37-series): HW_EST_BLOCK64 (opt-in arm, Time series (Holt-Winters estimate block, Apple FAST); no A/B recorded) is gone; the default path is the only one; code recoverable at 82df561cdb; restore: experiments/removed/MOJOLEARN_HW_EST_BLOCK64.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ALL_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_ALL_OFF (old path, Every family (umbrella: every IDENTICAL switch off); no A/B recorded) is gone; the default is kept; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_ALL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ARIMA_EVAL_WS_EXOG_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ARIMA_EVAL_WS_EXOG_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_ARIMA_EVAL_WS_EXOG_OFF (old path, Time series (ARIMA held workspace for exog fits); no A/B recorded) is gone; the default is kept; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_ARIMA_EVAL_WS_EXOG_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ARIMA_EVAL_WS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ARIMA_EVAL_WS_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_ARIMA_EVAL_WS_OFF (old path, Time series (ARIMA held evaluation workspace, IDENTICAL); no A/B recorded) is gone; the default is kept; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_ARIMA_EVAL_WS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ARIMA_FUSED_TAIL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ARIMA_FUSED_TAIL_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_ARIMA_FUSED_TAIL_OFF (old path, Time series (ARIMA fused evaluation tail, IDENTICAL); no A/B recorded) is gone; the default is kept; code recoverable at 96d454df19; restore: experiments/removed/MOJOLEARN_IDN_ARIMA_FUSED_TAIL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ARIMA_LLONLY_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ARIMA_LLONLY_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_ARIMA_LLONLY_OFF (old path, Time series (ARIMA log-likelihood-only filter, IDENTICAL); no A/B recorded) is gone; the default is kept; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_ARIMA_LLONLY_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ARIMA_ORDER_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ARIMA_ORDER_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_ARIMA_ORDER_BATCH_OFF (old path, Time series (AutoARIMA grouped order fits, IDENTICAL); no A/B recorded) is gone; the default is kept; code recoverable at e4949779dc; restore: experiments/removed/MOJOLEARN_IDN_ARIMA_ORDER_BATCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_BOOT_SELECT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_BOOT_SELECT_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_BOOT_SELECT_OFF (old path, Time series (resample bootstrap order statistic); no A/B recorded) is gone; the default is kept; code recoverable at e780161ad2; restore: experiments/removed/MOJOLEARN_IDN_BOOT_SELECT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KALMAN_RD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KALMAN_RD_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_KALMAN_RD_OFF (old path, Time series (Kalman per-rd kernel, IDENTICAL); no A/B recorded) is gone; the default is kept; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_KALMAN_RD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_X0_PAR_LS.patch
    comptime assert not is_defined["MOJOLEARN_IDN_X0_PAR_LS"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_X0_PAR_LS (opt-in arm, Time series (ARIMA start-point parallel least squares); no A/B recorded) is gone; the default path is the only one; code recoverable at 713890f21c; restore: experiments/removed/MOJOLEARN_IDN_X0_PAR_LS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_X0_PAR_LS_MIN256.patch
    comptime assert not is_defined["MOJOLEARN_IDN_X0_PAR_LS_MIN256"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_X0_PAR_LS_MIN256 (opt-in arm, Time series (ARIMA parallel LS threshold); no A/B recorded) is gone; the default path is the only one; code recoverable at 713890f21c; restore: experiments/removed/MOJOLEARN_IDN_X0_PAR_LS_MIN256.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KALMAN_FAST_RD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KALMAN_FAST_RD_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): KALMAN_FAST_RD_OFF (old path, Time series (Kalman per-rd kernel); no A/B recorded) is gone; the default is kept; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_KALMAN_FAST_RD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KALMAN_TIME_SCAN.patch
    comptime assert not is_defined["MOJOLEARN_KALMAN_TIME_SCAN"](), "removed 2026-11-15 (lane/expdel2-37-series): KALMAN_TIME_SCAN (opt-in arm, Time series (Kalman steady-state chunked scan, Apple FAST); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_KALMAN_TIME_SCAN.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KALMAN_TIME_SCAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KALMAN_TIME_SCAN_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): KALMAN_TIME_SCAN_OFF (old path, Time series (Kalman chunked scan off switch); no A/B recorded) is gone; the default is kept; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_KALMAN_TIME_SCAN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_PERM_COUNT_RANK.patch
    comptime assert not is_defined["MOJOLEARN_PERM_COUNT_RANK"](), "removed 2026-11-15 (lane/expdel2-37-series): PERM_COUNT_RANK (opt-in arm, Time series (resample permutation counting rank); no A/B recorded) is gone; the default path is the only one; code recoverable at 6b87cc7de9; restore: experiments/removed/MOJOLEARN_PERM_COUNT_RANK.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RESAMPLE_FAST_DEVICE_PERMUTE.patch
    comptime assert not is_defined["MOJOLEARN_RESAMPLE_FAST_DEVICE_PERMUTE"](), "removed 2026-11-15 (lane/expdel2-37-series): RESAMPLE_FAST_DEVICE_PERMUTE (opt-in arm, Time series (resample device permutation, Apple FAST); no A/B recorded) is gone; the default path is the only one; code recoverable at 6b87cc7de9; restore: experiments/removed/MOJOLEARN_RESAMPLE_FAST_DEVICE_PERMUTE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RESAMPLE_FAST_GATHER_NARROW.patch
    comptime assert not is_defined["MOJOLEARN_RESAMPLE_FAST_GATHER_NARROW"](), "removed 2026-11-15 (lane/expdel2-37-series): RESAMPLE_FAST_GATHER_NARROW (opt-in arm, Time series (resample narrow-row device gather, Apple FAST); no A/B recorded) is gone; the default path is the only one; code recoverable at 6b87cc7de9; restore: experiments/removed/MOJOLEARN_RESAMPLE_FAST_GATHER_NARROW.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RESAMPLE_FAST_GATHER_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RESAMPLE_FAST_GATHER_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): RESAMPLE_FAST_GATHER_OFF (old path, Time series (resample device gather, Apple FAST); no A/B recorded) is gone; the default is kept; code recoverable at 6b87cc7de9; restore: experiments/removed/MOJOLEARN_RESAMPLE_FAST_GATHER_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RESAMPLE_FAST_TILED_GATHER_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RESAMPLE_FAST_TILED_GATHER_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): RESAMPLE_FAST_TILED_GATHER_OFF (old path, Time series (resample tiled gather, Apple FAST); no A/B recorded) is gone; the default is kept; code recoverable at 6b87cc7de9; restore: experiments/removed/MOJOLEARN_RESAMPLE_FAST_TILED_GATHER_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RESAMPLE_FAST_WAIT_PAIR_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RESAMPLE_FAST_WAIT_PAIR_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): RESAMPLE_FAST_WAIT_PAIR_OFF (old path, Time series (resample paired transfers, Apple FAST); no A/B recorded) is gone; the default is kept; code recoverable at 6b87cc7de9; restore: experiments/removed/MOJOLEARN_RESAMPLE_FAST_WAIT_PAIR_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RESAMPLE_GATHER_NARROW_BYTES.patch
    comptime assert not is_defined["MOJOLEARN_RESAMPLE_GATHER_NARROW_BYTES"](), "removed 2026-11-15 (lane/expdel2-37-series): RESAMPLE_GATHER_NARROW_BYTES (opt-in arm, Time series (resample narrow gather width knob); no A/B recorded) is gone; the default path is the only one; code recoverable at 6b87cc7de9; restore: experiments/removed/MOJOLEARN_RESAMPLE_GATHER_NARROW_BYTES.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_X0_FAST_LS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X0_FAST_LS_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): X0_FAST_LS_OFF (old path, Time series (ARIMA start-point LS, Apple FAST); no A/B recorded) is gone; the default is kept; code recoverable at 8c2c6d7d01; restore: experiments/removed/MOJOLEARN_X0_FAST_LS_OFF.patch"
    # ---- end lane expdel2-37-series
    # Lane expdel2-47-promote (2026-11-15): old paths deleted and ARIMA_FAST_SEARCH_REUSE promoted (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # Promoted: the old positive define is refused; -D MOJOLEARN_ARIMA_FAST_SEARCH_REUSE_OFF restores the refit path.
    comptime assert not is_defined["MOJOLEARN_ARIMA_FAST_SEARCH_REUSE"](), "promoted 2026-11-15 (lane/expdel2-47-promote): AutoARIMA search-fit reuse is the FAST + Apple default (M3 rab16-arimareuse: synthetic 18229.7 -> 13688.1 ms, taxi-hourly 16386.5 -> 9872.2 ms, forecast_rmse identical; IDENTICAL unchanged); drop -D MOJOLEARN_ARIMA_FAST_SEARCH_REUSE, -D MOJOLEARN_ARIMA_FAST_SEARCH_REUSE_OFF restores the refit"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ARIMA_IC_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ARIMA_IC_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-47-promote): IDN_ARIMA_IC_DEVICE_OFF (old path, Time series (AutoARIMA IC argmin on device, IDENTICAL); no A/B recorded) is gone; the default is kept; restore: experiments/removed/MOJOLEARN_IDN_ARIMA_IC_DEVICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ARIMA_ORDER_CAP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ARIMA_ORDER_CAP_OFF"](), "removed 2026-11-15 (lane/expdel2-47-promote): IDN_ARIMA_ORDER_CAP_OFF (old path, Time series (AutoARIMA order-group workspace cap, IDENTICAL); no A/B recorded) is gone; the default is kept; restore: experiments/removed/MOJOLEARN_IDN_ARIMA_ORDER_CAP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ARIMA_ORDER_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ARIMA_ORDER_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-47-promote): IDN_ARIMA_ORDER_DEVICE_OFF (old path, Time series (AutoARIMA order search on device, IDENTICAL); no A/B recorded) is gone; the default is kept; restore: experiments/removed/MOJOLEARN_IDN_ARIMA_ORDER_DEVICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ARIMA_ORDER_SEASONAL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ARIMA_ORDER_SEASONAL_OFF"](), "removed 2026-11-15 (lane/expdel2-47-promote): IDN_ARIMA_ORDER_SEASONAL_OFF (old path, Time series (AutoARIMA seasonal grouped search, IDENTICAL); no A/B recorded) is gone; the default is kept; restore: experiments/removed/MOJOLEARN_IDN_ARIMA_ORDER_SEASONAL_OFF.patch"
    # ---- end lane expdel2-47-promote
    # ---- lane fx-verdicts5 (2026-11-20): verdicts of A/B round 3 (NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad)
    # TOMBSTONE: MOJOLEARN_RESAMPLE_FAST_PERM_SELECT_EARLY (LOSS, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm: permutation-test istella 1.075, taxi 1.101, digests equal) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
    # Restore: git apply experiments/removed/MOJOLEARN_RESAMPLE_FAST_PERM_SELECT_EARLY.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_RESAMPLE_FAST_PERM_SELECT_EARLY"](), "removed 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm; LOSS: permutation-test istella 1.075, taxi 1.101, digests equal): the early stop of perm_select_stat_kernel's key passes; see docs/TOMBSTONES.md"
    # ---- end lane fx-verdicts5
    return True


comptime SERIES_GUARDS_OK = _check_configuration()
