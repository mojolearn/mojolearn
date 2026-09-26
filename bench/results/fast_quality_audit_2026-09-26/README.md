# FAST quality audit (Sep 26 2026, Andrew: accuracy over speed)
Method: paired by seed/subset, >= 5 seeds, >= 2 datasets, current FAST vs the pre-approximation reference build.

| Item | Result | Verdict |
|---|---|---|
| Lossguide 16-leaf batch (e608ee4b8) | Istella reg RMSE +2% every subset; taxi AUC/logloss slightly worse | REVERTED (opt-in -D MOJOLEARN_GBDT_LG_BATCH=1) main 9ff8d7b16 |
| UMAP init tol 1e-3 (e2db5dd45/f7e0987b5) | trust covtype -0.0083 (all 8 seeds worse), taxi -0.0043 | REVERTED (opt-in -D MOJOLEARN_UMAP_INIT_TOL=1) main 9ff8d7b16 |
| ExtraTrees 16-bit border codes, wide reg (c1f1ac438) | istellareg RMSE +0.022% (sd 0.001, mixed signs), year +0.015% (mixed) over 5 x 300k subsets, 50 trees | KEEP |
| SVC/SVR working set 2048 + inline update_f (291a1ab6a, ad03c090d) | SVC taxi 50k acc identical on 5/5 subsets (8.5 -> 6.0 s); SVR r2 diffs ~1e-4 mixed | KEEP |
| ARIMA time-parallel Kalman + stall rule (58e139d45, f5b7ba2c4) | llf lower in every group at 20k (mean -0.035/-0.038/-0.122, worst -0.371); identical below 4096 obs | REVERTED (opt-in -D MOJOLEARN_KALMAN_TIME_SCAN=1) |
| ARIMA TSQR start params (211cb23b8) | identical fits at 2000 obs (scan inactive) | KEEP |
| k-NN expanded distances + fused top-k / MMA (3c33ed941, 6dbb238db) | KNeighborsClassifier acc identical 10/10 (taxi, covtype x 5 subsets 200k); recall@10 vs float64: taxi +0.00004, covtype -0.00008 (worst -0.0002, near-ties) | KEEP |
| GMM parallel sums / fused cov / chol / E-step (faeb9756f..856ffdf14) | taxi test ll +0.0017 mean (min -0.00006); covtype at reg_covar 1e-3: -0.0009 mean (range -0.005..+0.001 on ~-100), mixed; covtype at default reg: FAST refuses at init like sklearn (ill-defined covariance) while the reference returns a garbage fit (ll down to -2e9) | KEEP |
| Lasso/ElasticNet Gram-form CD (40652f7e4) | taxireg R2 identical 10/10; year R2 +3e-6 better on 10/10 (300k subsets) | KEEP |
| FAST Cholesky (vendor GEMM, blocked/inverse TRSM, recursive panel; 5c7f43c47, 8c2ca5cc7, 4322907ba) | at 3000 rows (paths exercised): GPR R2 and GPC acc identical to 7 decimals 5/5 | KEEP |
| LogisticRegression split-K X^T dZ + gemv (7f5d39eda, 913a3b07f) | taxi identical; covtype 7-class at max_iter 200 (unconverged) logloss +0.0031 (4/5 worse); at max_iter 1000: acc +0.00017 (range +-0.0007), logloss +0.0002 (range +-0.0033), mixed | KEEP |
| Spectral/Lanczos FAST (ncv 48, device restarts, simdgroup SpMV; 91ce51ade, 3a67941ed) | SpectralClustering 8 blobs 20k: ARI vs truth equal or better 5/5 (mean +0.053; reference 0.89/0.84 on two seeds, FAST 1.0); SpectralEmbedding trust taxi/covtype within +-4e-5 mixed | KEEP |

Not audited (no accuracy surface or already covered): IVF 256-row trainset (earlier 16-run paired audit: recall +0.004), forest inference engine under FAST (labels exact in its own check), KDE lgamma recurrence (exact at integer/half-integer arguments), SVM fused kernels on NVIDIA/AMD (a compile routing fix).
Summary: 3 reverted (Lossguide batch, UMAP init tol, ARIMA time scan), 9 kept with paired evidence.
