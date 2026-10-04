# FAST quality audit notes

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
