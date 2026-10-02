# Orchestrator status (2026-10-02)
Launched (wave 1, 20 agents): core tsa ann cluster(+cluster2) gram(+kernel) bayes decomp(linalg+sparse) neighbors(isotonic-knn+neighbors2) linear prep(+prep2) resample trees-depthwise(+trees-scan) trees-ensembles(+trees-io,rfet-scan) depthwise yetirank(+trees-yeti) trees-symmetric select kapprox meta dart
Pending launch: ets (launching now); resample DONE head 50b96e795
Results: (filled as lanes report)
- resample DONE 50b96e795: merged main, env->defines, 7 taxi A/B lines
- linear DONE 44ec8018d: merged main, env->defines, istella A/B lines
- tsa DONE 21e28b348: merged main, LLONLY + EVAL_WS defines, 3 autoarima taxi-hourly A/B lines
- bayes DONE e9312ac06 (+ard line drop pending): BAYES_GRID_GUARD on main's grid kernels
- trees-scan DONE 43430ca0f: merged main, dropped its scan variant, 2 A/B lines; trees-depthwise DONE 92e87fe17: merged main, kept CTR_FAST_SCAN, 3 taxi lines; heads mutually clean
- core DONE 67258a0df: merged main, OLS_FAST_DEVICE_CENTER finished, core-ols-dcenter-taxi
- trees-ensembles DONE d7887faa1 (env->bit-set defines, BAG_SESSION dropped as main covers it, 8 lines); trees-io DONE 9d916e91f (5 lines); rfet-scan DONE b0d5b1fab (docs merge only)
- trees-symmetric DONE ae0774468: SYM_DEVICE_LEVEL two-stage level score, tsym-level-istella
- yetirank DONE c9eb1ce14 (merged main + trees-yeti; SCORE_GRID, SYM_HIST_UNROLL8; 2 istellarank lines); trees-yeti DONE b157e5ee0 (merge only)
- prep DONE 387211293 (PREP_FAST_MINMAX define, 5 lines); prep2 DONE cc3b27d5f (PREP2_FAST_EIGH_BLOCK define, 7 lines); mutually clean
- gram DONE 033f6c096 (3 defines, 6 lines); kernel DONE 4e08139d4 (merged gram; 4 defines, 4 istella lines)
- kapprox DONE 015510968: KAPPROX_DEVICE (chi2 samplers device fit/transform), 2 lines; gaussian-rp left to decomp-sparse; CHECK overlap with neighbors2/isotonic-knn (x_neighbors gen bindings)
- select DONE baf8e1681: SELECT_FREG/FCLS/D defines, 4 lines
- cluster DONE 2d5b3dffa (2 defines, 10 lines); cluster2 DONE 670baf47a (merged cluster; 5 defines, 18 lines; optics serial launch rewritten as per-step grid); trim to one dataset requested

## Paused 2026-10-02 (weekly API limit, resets 22:00Z); user asked: commit and push everything, discard nothing
Pushed heads: resample 50b96e795, linear 44ec8018d, tsa 21e28b348, bayes 0f7218b29, trees-scan 43430ca0f, trees-depthwise 92e87fe17, core 67258a0df, trees-ensembles d7887faa1, trees-io 9d916e91f, rfet-scan b0d5b1fab, trees-symmetric ae0774468, yetirank c9eb1ce14, trees-yeti b157e5ee0, prep 387211293, prep2 cc3b27d5f, gram 033f6c096, kernel 4e08139d4, kapprox 015510968, cluster 2d5b3dffa, cluster2 670baf47a, select cb5c31a60 (prep2 merged in; merge.log committed as-is), meta 497bcad7d (CALIB_GNB_FOLDS; multioutput-reg not started), decomp-linalg 8fdbdd0e8 and decomp-sparse e021e3498 (main merged; WIP .txt/.md and overlap check not done), ann 59afb759a (main merged; was mid-push), depthwise 341ab92ea (dw_tree_sync.mojo WIP, unreferenced), dart c7fe11e34 (WIP), ets 7433af169 (WIP, unreferenced).
NOT pushed: neighbors2 local merge 6aff39c27 in ~/mojolearn-wt/neighbors2 (pre-push hook refused 13 host-route findings in x_neighbors/iter_device.mojo:1869-2005 `_buf`/`_down` calls and svgp_fast.mojo:110 one-block launch; origin stays at 67157e40a, unmerged with main). Patch of that head vs origin/main: neighbors2-unpushed-merge.patch beside this file. isotonic-knn 4f4cd51e2 = origin (main merged, pushed by the lane).
Open follow-ups: cluster/cluster2 request files not yet trimmed to one dataset per change; kapprox x neighbors2/isotonic-knn overlap on x_neighbors gen bindings unchecked; select x prep2 settled by the select lane's merge.
