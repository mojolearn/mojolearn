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
Rows of ~/mojolearn-evidence/board-quality-audit-2026-10-04.md that are metric artifacts or stale cells, not real quality losses. Each section says why, so `tools/af_board_quality_audit.py` can whitelist the row.

## Dense linear algebra and tree-shap (lane apple-fast-q-linalg, 2026-10-04)

Fixed in code (FAST default, QOLD define restores; rows in EXPERIMENTS.md, READY-AB): lu-solve and lu-factor synthetic (`LU_QFIX`), svd istella and taxi reconstruction (`SVD_QFIX`), tsvd istella (`TSVD_QFIX`). See x_decomp/qfix.mojo.

Not fixed in code:

- **svd istella `max_rel_singular_value_error` (41,531 vs numpy 37): artifact.** The metric divides each singular value's error by max(s_k, 1e-12 s_0) (tools/bench_board_algos.py, the `svd` quality block), and the float64 reference itself comes from `eigvalsh(X^T X)`, accurate only to about 1e-8 s_0 for small values. istella has 21 constant features and near-dependent columns, so it has singular values below float32's resolution of about 6e-8 s_0. Any float32 SVD returns those as noise of size eps s_0, which the 1e-12 floor turns into errors of 1e4 to 1e6: torch-gpu (float32) scores 4.9e6, our earlier FAST 2.9e4. numpy's 37 comes from LAPACK's structure on exactly dependent columns, not from more precision. For a float32 input, singular values under eps s_0 are zero to the user. The whitelist should apply the metric only to s_k >= 2^-23 s_0. The taxi row's singular-value metric is already SAME.
- **qr istella (relative_gram_difference 9.03e-04 vs numpy 2.47e-08): stale cell.** The 9.03e-04 is the 0.8.34 FAST cell (`ours-fast`, board-m3ultra-0834.json), from FAST's route before the blocked TSQR (python/mojolearn/_linalg_impl.py `_qr_tsqr`, lane neural-pass140). Main's `qr(mode='reduced')` now runs the TSQR on both tiers with the same kernels, and 0.8.34's TSQR cell for istella (`ours`) is 1.54e-07: within the audit's abs_tol 1e-6 of numpy. taxi moved the same way (2.0e-03 -> 5.58e-07, SAME). One FAST re-measure of qr istella is owed. No code change.
- **tree-shap istella and taxi (max_additivity_error 1.2e-06 and 3.9e-05 vs lightgbm 4e-15 and 6e-13): artifact.** Our SHAP values, base value and margin are float32 (python/mojolearn/_expansion_trees.py:2976-2982: float32 throughout, because Metal has no float64; phi is "<f4" at :3052). LightGBM computes in float64. The taxi margin is fare_amount (tools/classical_two_datasets.py:632), up to a few hundred dollars, so one float32 ulp of the margin is 1.5e-05 to 6e-05: 3.9e-05 is about one ulp. istella margins are O(1-4), ulp 2.4e-07 to 4.8e-07: 1.2e-06 is a few ulps over a sum of 220 phi values. The float32 opponents score worse: shap-cpu and xgboost-cpu 1.2e-04 on taxi, and 1.86e-06 on istella. The whitelist should compare against float32 opponents, or allow a few float32 ulps of the margin.
- **mb-dict-learning taxi (relative_reconstruction_error 0.4967 vs sklearn 0.4770, -3.95%): likely a randomized-method artifact.** The algorithm follows sklearn step by step (python/mojolearn/_expansion_decomp.py `MiniBatchDictionaryLearning.fit`: `_minibatch_step`, `_update_inner_stats` theta/beta, `_update_dict`, `_check_convergence` with tol 1e-3 and max_no_improvement 10). It differs only in its random streams: the shuffle is a Philox permutation where sklearn uses its RandomState, the unused-atom resample likewise, and the initial dictionary is the exact SVD where sklearn uses a randomized SVD. With max_iter=10 and batch_size=256, the noisy early stop on the batch-cost average ends each run at a stream-dependent step. The deterministic counterpart shows the core steps are sound: dict-learning on the same taxi data scores ours 0.4592 vs sklearn 0.4606 (ours better). istella is -0.67% (not material). OWED before whitelisting: a seed spread, sklearn and ours over random_state 0..4 on the taxi mid subset. If sklearn's spread does not reach 0.497, reopen as a real loss.
Rows of `~/mojolearn-evidence/board-quality-audit-2026-10-04.md` where FAST is
worse than the best opponent but the gap is a property of the metric or the
draw, not a FAST loss. Each section says why, so `tools/af_board_quality_audit.py`
can whitelist the row. Lanes append their own section.

## lane apple-fast-q-misc (2026-10-04)

Board cells: `~/mojolearn-evidence/q-misc-board-cells.txt` (dumped from
`board-quality/board-m3ultra-0834.json`). Simulations below are NumPy/SciPy
only, on `~/mojolearn-evidence/kernel-pca-gpu-investigation-20260930/taxi-board-10000.npy`
(10,000 standardized taxi rows); no mojolearn code was run.

| lane | dataset | metric FAST vs opponent | verdict | reason |
|---|---|---|---|---|
| gaussian-rp | istella | distortion 0.6807 vs 0.1780 | ARTIFACT (draw) | Same n_components=10 on every arm (tools/bench_board_algos.py:396; eps is unused when n_components is set), so sklearn's number does not come from more components. A Gaussian projection is rotation invariant: for any pair, projected/original squared distance is chi2_10/10 whatever the data, and our scale is 1/sqrt(n_components) (python/mojolearn/_expansion_decomp.py:1500). On Istella the 2,000 pair differences point along a few huge-variance features, so the board metric is close to ONE draw of abs(chi2_10/10 - 1): its quantiles (10/25/50/75/90%) are 0.057/0.144/0.299/0.491/0.685, mean 0.351. 0.68 (ours) and 0.18 (sklearn, numpy RandomState(7)) are both ordinary draws. The FAST draw is main's (unchanged bits). |
| gaussian-rp | taxi | 0.3458 vs 0.3398 | ARTIFACT (draw) | As above; both at the expected 0.351. |
| sparse-rp | taxi | 0.264 vs FAST before 0.1472 (sklearn 0.381) | ARTIFACT (draw) | Same n_components=10 and density='auto' (tools/bench_board_algos.py:402). XD_FAST_SRP_STRAT (EXPERIMENTS.md, w2-kfeat-srp-q, 40 seeds per arm) lowered the 40-seed mean distortion 0.346 -> 0.233 on taxi and 0.946 -> 0.571 on istella; seed 7 alone moved 0.147 -> 0.264 on taxi and is still better than sklearn's 0.381. The "after worse than before" row is one seed of a draw that improved on average. |
| rbf-sampler | taxi | kernel_rel_error 0.1085 vs 0.0838 | ARTIFACT (draw) | Same gamma=1/d, n_components=256, seed 7 (tools/bench_board_more.py:475); our cos is the pinned `identical_cos` (kernel_methods/checks/random_features.mojo), the weights N(0, 2 gamma), the offsets U(0, 2 pi), the scale sqrt(2/D). Over 40 draws of the same random Fourier features on the 1,000 check rows the error is mean 0.1028, sd 0.0119, min 0.0812, 10%/90% 0.0889/0.1202: ours (0.1085) is a median draw, sklearn's (0.0838) near the minimum. |
| rbf-sampler | istella | 0.1420 vs 0.1374 | ARTIFACT (draw) | 3% apart, well inside one sd of the draw spread (12% on taxi). FAST draw = main's (KM_FAST_RBF_RESIDENT is byte-identical, tools/kfeat_quality.py). |
| nystroem | taxi | 0.04503 vs 0.04437 | ARTIFACT (draw) | 1.5% apart; landmark draws alone give sd 0.0033 on mean 0.028 (12%) on the 10k sample. |
| resample | taxi | max_mean_shift_over_std 0.00292 vs 0.00226 | ARTIFACT (Monte Carlo) | The metric is the largest column of abs(bootstrap mean - population mean)/sd over a 1,000,000-row bootstrap: each column is about N(0, 1e-3), so the max over 11 columns is about 2e-3 and over 220 columns about 3e-3. Both arms sit there; the drawn rows are each library's own stream (tools/bench_board_algos.py:673). xmean is float64 (tools/bench_board_algos.py:3835). |
| resample | istella | 0.00320 vs 0.00255 | ARTIFACT (Monte Carlo) | As above. |
| spectral | taxi | silhouette 0.0399 vs 0.0899 | ARTIFACT (k-means start on a degenerate embedding) | IDENTICAL is 0.0399 too and FAST/IDENTICAL ARI 0.997, so no FAST step causes it. Istella agrees with sklearn (0.1477 vs 0.1477, ARI 0.9998). On the taxi sample the 10-NN graph has 2 connected components and the 8 smallest normalized-Laplacian eigenvalues are 0, 0, 1.7e-4 ... 3.1e-3 (a near-degenerate subspace); k-means with n_init=1 (SPECTRAL_N_INIT=1, tools/bench_board_more.py:165) on the EXACT eigenvectors gives silhouette -0.032 .. 0.093 across 8 seeds (0.069, -0.032, 0.093, 0.075, -0.024, 0.069, -0.026, 0.060). 0.040 and 0.090 are both inside that spread. |
| minibatch-kmeans | taxi | silhouette 0.1380 vs 0.1655 | ARTIFACT (start/batch draw) | n_init=1, stochastic batches from each library's own stream; FAST = IDENTICAL (ARI 1.0). On istella ours is better (0.1167 vs 0.1118). Silhouette is not the objective k-means minimizes. |
| kmeans | istella | inertia 6.051e17 vs 5.959e17 (+1.5%) | ARTIFACT (local optimum, n_init=1) | FAST = IDENTICAL bits (inertia_over_ours 1.0). Our init is RAFT's greedy k-means++ (cluster/checks/plus_plus.mojo), as sklearn's; one start each, different streams. torch-gpu lands between (5.991e17). On taxi ours beats sklearn (3.093e8 vs 3.166e8). |
| bayesian-gmm | taxi | mean_log_likelihood 4.896 vs 6.178 | ARTIFACT (likely; local optimum) - multi-seed check owed | FAST = IDENTICAL (4.89559 vs 4.89560), so no FAST shortcut. The updates follow sklearn's `_bayesian_mixture.py` (python/mojolearn/_expansion_cluster.py:595); the board scores every arm with the same plain GMM log-likelihood from weights/means/covariances (tools/bench_board_algos.py:4442). One kmeans start each (n_init=1, each library's own KMeans stream) on taxi's partly discrete columns with reg_covar 1e-6: a component that locks onto a near-constant sub-population changes the held-out log-likelihood by whole nats. Not proven: a 5-seed run of both arms would settle it. |
| als | taxi-zones | recall_at_10 0.05400 vs 0.05679 | ARTIFACT (likely; init draw) - multi-seed check owed | FAST = IDENTICAL (0.0539953). Same factors/regularization/alpha/iterations, exact solver both sides, init uniform*0.01 both sides, confidence alpha*r as implicit (python/mojolearn/_expansion_decomp.py:4818). On the text dataset the two agree (0.5487 vs 0.5482). The remaining difference is each library's own initial factors. |

Fixed in code on this lane (not artifacts): knn istella (KNN_FAST_REFINE),
ivf-sq taxi and ivf-filter istella (IVF_COARSE_FAISS_INIT), autoarima
taxi-hourly (board search maxiter). See EXPERIMENTS.md, QUALITY-FIX rows.
Rows of `tools/af_board_quality_audit.py` (board-quality-audit-2026-10-04) that are metric artifacts or stale measurements, with the reason, so the audit can whitelist them. Each lane appends its own section.

## Classifiers (lane/apple-fast-q-clf, 2026-10-04)

### pa-clf taxi: stale row (no code change)

- Audit: FAST accuracy 0.58383 vs sklearn 0.74474 (and IDENTICAL 0.76036), from the 0.8.34 board JSON.
- The 0.8.34 wheel (release 42b06235e) fit PassiveAggressiveClassifier per sample (no `batch_size`). Main fits it in batches of 256, the batch moving by the mean of its rows' PA-I steps (lane/neural-pass132, merged after 0.8.34; python/mojolearn/_expansion_linear.py `PassiveAggressiveClassifier`, x_linear/sgd.mojo `mb_row_dot` / `mb_step`). The row measures code that is no longer on main.
- A float32 numpy model of main's step on the board's taxi block (1M fit rows, standardized, 20 epochs, C=1) gives 0.7717 / 0.7702 / 0.7665 / 0.7667 / 0.7703 over five shuffle seeds, all above sklearn's 0.74474 (`~/mojolearn-evidence/apple-fast-q-clf/sim_mb_perceptron.py`).
- Needed: one FAST re-measure of pa-clf taxi on main. Whitelist the 0.8.34 row.

### decision-tree-clf taxi: metric artifact (no code change)

- Audit: accuracy SAME (0.7563 vs 0.75656), log loss 1.20224 vs sklearn 1.14955. FAST and IDENTICAL are the same words.
- Cause: a depth-16 unpruned tree's leaves are mostly pure, so `predict_proba` is exactly 0 or 1 in every library (leaf frequencies, not a float32 effect). The board clips at 1e-15, so each test row that falls in a pure leaf of the wrong class costs 34.5 nats. The 0.053-nat gap is about 0.15% of the 100,000 test rows landing in wrong pure leaves.
- Which rows those are depends on the split candidates: ours splits on 128 quantile bins per feature (the documented mismatch in tools/bench_board_algos.py `decision-tree-clf`), sklearn on exact thresholds. The sign flips by dataset: on Istella ours is better (0.7583 vs sklearn 0.7917).
- Not a quality loss: accuracy is equal, and the log loss of 0/1 leaves measures how many errors fall in pure leaves. Changing n_bins to match this board would be tuning to the dataset.

### Fixed in code (see EXPERIMENTS.md, "Quality fixes, classifiers")

- perceptron taxi: the minibatch last iterate is a lottery. The fix averages the epoch-end iterates (`SGD_PERC_AVG`, QOLD `MOJOLEARN_SGD_PERC_QOLD`). The audit row itself is also stale: 0.8.34 fit per sample.
- gaussian-nb, bernoulli-nb, multinomial-nb, complement-nb, qda, nearest-centroid on istella: float32 probabilities saturated at 1.0. Fixed by float64 `predict_proba` (QOLD `MOJOLEARN_PROBA64_QOLD`). The two text rows (multinomial-nb 0.559529 vs 0.557319, complement-nb 0.559491 vs 0.557285) have the same cause and are covered by the same fix.
