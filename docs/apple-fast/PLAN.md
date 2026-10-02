# Apple FAST (classical + trees): one ranked plan for lane/apple-fast (2026-10-02)

Branch `lane/apple-fast` (worktree ~/mojolearn-wt/apple-fast) = origin/main + lane/apple-fast-classical
(ee03623fa) + lane/apple-fast-trees2 (50dfdcca0), merged without conflicts; FAST Metal builds of every
changed binding compile (x_linear, x_neighbors, core build.sh, gbdt, svm). Details:
PLAN-classical.md and PLAN-trees.md beside this file. Every change is FAST + Apple comptime; IDENTICAL
must be re-checked on the M3 for both halves (owed).

## 0. Read what is queued (m3, unread)
Classical (branch lane/apple-fast-classical): 37 afc-base2, 39 afc-m1build, 40 afc-m1bayes, 41 afc-m1enet,
42 afc-m1knn, 44 afc-dbg. Trees (lane/apple-fast-trees2): 36, 38, 43, 45-54 (55 VOID).
Decide per change: keep default-on, or set the switch default off (classical: env `=0`; trees: `_OFF` defines).

## 1. Ranked next experiments (expected gain x certainty)
1. [classical] Verify the four unmeasured classical changes (BayesianRidge/ARD sse, LassoCV/ENetCV grid,
   k-NN d>32 MMA, NearestCentroid chunks): A/B + paired quality taxi+istella; IDENTICAL hash at head.
2. [trees] Flip Lossguide batch-width / SM_X / RF / ET arms that win (A/Bs 46-54 already queued).
3. [classical] FAST slower than IDENTICAL: theta, rbf-sampler, pca Istella; take IDENTICAL's arm or fix.
4. [classical] Grid Gram for Lars/LassoLars/RidgeClassifier/RidgeCV/LDA/QDA (one-block Gram chains in
   x_linear/lars.mojo, ridge.mojo, x_prep); generalize enetcv_fast's ef_gram_kernel into a shared FAST grid Gram.
5. [trees] Depthwise split chain fusion (greedy_search_helper_depthwise.mojo:1151, ~1800-2100), taxi 14.9 s vs XGBoost 9.9.
6. [trees] YetiRank est.approx relaunches (yeti_rank.mojo:772, pointwise_oracle.mojo ~600).
7. [classical] PCA Istella one-block Jacobi (decomposition/impl/linalg/detail/pca.mojo eig_and_truncate) -> jacobi2 / top-k.
8. [classical] Isotonic parallel PAVA (x_linear/isotonic.mojo); knn-imputer/LLE through fast_mma_knn.
9. [trees] RF/ET Python all_finite -> device scan (randomforest.py:481, extratrees.py:278); iforest sampled upload.
10. [classical] Label encoders/binarizers/onehot/minmax host overhead (_expansion_prep.py); meanshift block-per-seed opt-in.
11. [trees] seg_scan one-thread blocks (segmented_sort.mojo:372, randomforest.mojo:1842); categorical CTR stages.

## 2. Needs a fresh M3 board at lane/apple-fast's head (classical algos + trees family), FAST and IDENTICAL.
