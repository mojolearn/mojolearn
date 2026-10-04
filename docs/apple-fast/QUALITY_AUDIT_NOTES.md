# FAST quality audit notes

Rows of `~/mojolearn-evidence/board-quality-audit-2026-10-04.md` where FAST is below the best opponent and the cause is not a FAST quality loss. Each lane appends its own section.

## Regressors (lane apple-fast-q-reg, 2026-10-04)

Numbers come from the M3 board JSON (`board-m3ultra-0834.json`), which has the IDENTICAL, FAST and opponent cells side by side.

| lane / dataset | FAST | IDENTICAL | opponent | verdict |
|---|---|---|---|---|
| linearsvr / istella | r2 -0.10675 | r2 -0.10676 | sklearn r2 -0.0257 (fill) | artifact |
| pa-reg / istella | r2 -0.2823 | r2 +0.2943 | sklearn r2 -0.1282 | artifact (unstable estimator) |
| adaboost-reg / taxi | r2 -0.606 (0.216 before) | r2 +0.679 | sklearn r2 0.564 | unstable estimator; fix only via the DT bins change; A/B owed |

**linearsvr / istella: metric artifact.** FAST and IDENTICAL agree to 6e-6 in r2, so FAST takes no shortcut. Ours (and cuML) minimize the primal with L-BFGS and do not penalize the intercept. scikit-learn runs liblinear's dual coordinate descent with `max_iter=1000`, which does not converge on raw Istella, and with `intercept_scaling=1` it penalizes the intercept. That is a different model (see the mismatches in tools/bench_board_more.py `linearsvr`). With `epsilon=0` the loss is absolute error: the fit is a median regression, and r2 does not measure what it minimizes. The opponent cell is a fill: this board never measured it. Whitelist the row, or compare the epsilon-insensitive objective instead of r2.

**pa-reg / istella: unstable estimator, no FAST step.** FAST and IDENTICAL run the same minibatch PA-I code. x_linear/device.mojo `_sgd_mb_grid` has no mode branch. The only difference is rounding: x_linear/ops.mojo `fa`/`fm`/`fd`/`fmad` pin products and divides under IDENTICAL and use plain `*` and `/` under FAST (checks/numerics.mojo `identical_mul`, `identical_div`). That alone moves the held-out r2 from +0.29 to -0.28. With C=1 on raw Istella, each PA-I step jumps to fit its batch, and the reported model is the last iterate after 20 epochs (`tol=None`). Its quality is a draw from a wide spread. scikit-learn's draw (-0.13) falls inside that spread too. On taxi the order is FAST 0.854, IDENTICAL 0.901, sklearn 0.795. No code change: averaging or a step cap would change the estimator's semantics. Whitelist the row as seed-level noise. A stable comparison needs several seeds per arm.

**adaboost-reg / taxi: unstable estimator; no FAST-only step found.** These paths are mode-free: the AdaBoost.R2 host steps (xtrees/ops.mojo `r2_step`, `weighted_median`, `weighted_sample`) and the member trees, whose histograms are integer or fixed point. Every FAST builder switch in builder_kernels_impl.mojo is documented as giving the same forest. FAST does fit members through the data session (`TE_ADA_SESSION`). bindings/_mojolearn_rf.mojo `rf_regressor_fit_session_rows` gathers the same row bytes on the device and calls the same `fit_forest`. Its A/B (te-adareg-taxi) recorded identical quality. Two FAST builds gave r2 0.216 and -0.606 on the same data, so the estimator itself is unstable here. On heavy-tailed taxi targets, AdaBoost.R2's linear loss, normalized by the largest error, concentrates the weighted bootstrap on a few outlier rows, and rounding-level changes then change which rows are drawn. The DT bins fix (FAST n_bins 256) changes these members too. RUN OWED: the FAST A/B with `-D MOJOLEARN_TE_ADA_SESSION_OFF` against the default, plus `-D MOJOLEARN_DT_BINS_QOLD`, to confirm or rule out a session effect before whitelisting.
