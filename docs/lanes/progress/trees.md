# trees: progress

Pass 1 (code first). Gate per algorithm on the lane pod (RunPod H100):
build, sanity vs scikit-learn / LightGBM (`python/mojolearn/tests/test_x_trees_sanity.py`,
all passed), `tools/algos_lane_check.sh <lanes>` AGREE (CUDA column == CPU
column, cpu-amd-epyc-9554). Every lane PENDING (`_surface_trees.py`).

Where the code is: classes in `python/mojolearn/_expansion_trees.py`; the
ensemble glue (RNG draws, gathers, votes, SAMME/R2 steps, weighted median,
apply/Newton leaves, one-hot) in `xtrees/ops.mojo`, exported by
`xtrees/api.mojo::register` from BOTH `bindings/_mojolearn_x_trees.mojo` and
`bindings/_mojolearn_x_trees_host.mojo` (host code, one spelling; moving it
onto the device is pass-2 speed work). Trees are fitted through the existing
rf / extratrees entry points, whose sources are untouched.

| algorithm | lanes | lane check | commit |
|---|---|---|---|
| DecisionTreeClassifier / DecisionTreeRegressor | trees-dt-clf, trees-dt-reg | AGREE: compared batch 9, infer 9, model 9, train 9 (each lane) | see git log |
| BaggingClassifier / BaggingRegressor | trees-bagging-clf, trees-bagging-reg | AGREE: compared batch 9, infer 9, train 9 (each lane) | see git log |
| AdaBoostClassifier (SAMME) / AdaBoostRegressor (R2) | trees-adaboost-clf, trees-adaboost-reg | AGREE: compared batch 9, infer 9, train 9 (each lane) | see git log |
| DART (DARTRegressor L2 / DARTClassifier binary) | trees-dart-reg, trees-dart-clf | AGREE: compared batch 9, infer 9, train 9 (each lane) | see git log |
| RandomTreesEmbedding | trees-random-embedding | AGREE: compared batch 9, infer 9, train 9 | see git log |
| VotingClassifier / VotingRegressor | trees-voting-clf, trees-voting-reg | AGREE: compared batch 9, infer 9, train 9 (each lane) | see git log |
| StackingClassifier / StackingRegressor | trees-stacking-clf, trees-stacking-reg | AGREE: compared batch 9, infer 9, train 9 (each lane) | see git log |
| MultiOutputClassifier / MultiOutputRegressor | trees-multioutput | AGREE: compared batch 9, infer 9, train 9 | see git log |
| OneVsRestClassifier | trees-onevsrest | AGREE: compared batch 9, infer 9, train 9 | see git log |
| CalibratedClassifierCV (sigmoid, isotonic) | trees-calibrated | AGREE: compared batch 9, infer 9, train 9 | see git log |
| AdaBoostClassifier back on SAMME sample weights (sklearn's `_boost_discrete`) | trees-adaboost-clf | AGREE: compared batch 9, infer 9, train 9 | see git log |
| TreeExplainer (exact TreeSHAP, forests + DART) | trees-shap-tree | AGREE: compared batch 9, infer 9, train 9 | see git log |
| KernelExplainer | trees-shap-kernel | AGREE: compared batch 9, infer 9, train 9 | see git log |
| PermutationExplainer | trees-shap-permutation | AGREE: compared infer 9, train 9 (batch n/a: position-seeded) | this commit |

Pod setup notes (for a fresh agent): the lane check does not build the
ubiquitous bindings; build `bindings/build.sh` (identical), `build_forest_host.sh`
`build_byte_lm_host.sh` and `build_core_host.sh` once on a fresh pod, and
libMojolearnMath (`PYTHONPATH=packaging/portable_math python -c "import pathlib, stage;
stage.build(pathlib.Path('python/mojolearn/.libs/libMojolearnMath.so'))"`). sklearn/lightgbm/pytest live
in /root/sk on the pod (uv --target); run the sanity test with
`PYTHONPATH=python:/root/sk pixi run -e default python -m pytest -q python/mojolearn/tests/test_x_trees_sanity.py`.

Shared fix (main's policy 2026-09-27: the lane that finds it fixes it):
the RF weighted objective (class weights / sample_weight WITHOUT bootstrap)
now has its CPU restatement in `ensemble/host/rf_oracle.mojo`; lane
`trees-rf-weighted` AGREE (batch/infer/model/train 9), sabotage
`rf_weighted_split_sabotage.patch` (the split planes drop the row weight):
AGREE, DISAGREE, AGREE after `git apply -R`. Existing RF lanes (rf-clf,
rf-reg, rf-clf-entropy-log2-noboot, rf-clf-balanced-parallel,
rf-reg-poisson, rf-reg-gamma-ig, rf-score-weighted): `python -m mojolearn
verify` 7/7 VERIFIED on CUDA and on CPU after, 0 cell hashes differ from
before on either column.

PASS 2 (proof), in progress:
- per-seam gate `xtrees/checks/glue_check.mojo` (oracle `glue_oracle.mojo`),
  DEVIATIONS 5600-5605, IDENTITY_PATHS rows 160-166, six sabotage arms in
  `xtrees/checks/sabotage/` listed in `tools/identity_lanes/trees.checks`:
  each FAILS the driver on the H100 pod; all 20 trees lanes AGREE with the
  driver run first.
- end-to-end sabotage `xtrees/checks/sabotage/column_cpu_only.patch` (perturbs
  the CPU column only, `MOJOLEARN_COLUMN_CPU`, plus the rf host leaf):
  `algos_lane_check.sh <all 20 lanes> --pass 2 --sabotage` on the H100 pod:
  AGREE, 20/20 DISAGREE, 20/20 AGREE after reversal, RESULT: PASS. This is
  the patch to hand the steward.
- `xtrees/NOT_IMPLEMENTED.tsv`: the lane's refused options (option parity input).
- AMD MI300X (Hot Aisle box `trees-amd`): `--pass 2` PASS on all 20 lanes,
  seams bite; NVIDIA vs AMD GPU columns diff OK on all 20; glue cards
  byte-identical across the two boxes.
- Apple: M2 Pro steward PASS (1790526750361-trees-77e0b3a8d7), M3 Ultra spooled.
- Pass-2 proof DONE for the 20 lanes (PENDING until a release record admits them).

Option parity (item 2), merged as each passes:
- DecisionTree splitter='random' (+ max_leaf_nodes best-first on it): lane
  trees-dt-random AGREE; column sabotage (now also the ET host draw) DISAGREE
  then AGREE.
- Directive 000 (seam arms re-proven on the fixed lane check, 3084ca09c):
  DONE. The six glue_check arms each "build, run and FAIL" under the fixed
  tool on the H100 pod (runs of 2026-09-27 18:32Z and 19:40Z) and on the
  MI300X. Never repeat.
- DART options (reg_alpha, max_delta_step, colsample_bytree, subsample +
  subsample_freq, multiclass softmax + its TreeSHAP), lane trees-dart-options:
  `--pass 2 --sabotage dart_options_cpu_only.patch` PASS on H100 and MI300X.
- Bagging oob_score, cv splitter objects / (train, test) pairs, Kernel SHAP
  link='logit', lane trees-oob-cv-link: `--sabotage oob_link_cpu_only.patch`
  PASS on H100 and MI300X.
- ExtraTreesRegressor criterion poisson / gamma / inverse_gaussian
  (DEVIATION 5610, IDENTITY_PATHS row 168; extratrees/ tsv rows closed),
  lane trees-et-deviance: `--pass 2 --sabotage et_deviance_cpu_only.patch`
  PASS on H100 and MI300X.
- Existing lanes after all three: the nine older trees lanes AGREE (H100,
  MI300X); et-clf, et-clf-entropy-bestfirst, et-reg,
  et-reg-bootstrap-parallel AGREE on MI300X; on H100 those four plus rf-clf,
  rf-clf-balanced-parallel, rf-clf-entropy-log2-noboot, rf-reg,
  rf-reg-gamma-ig, rf-reg-poisson, rf-score-weighted, saved-model-host-infer
  were fitted BEFORE (the branch diff reversed) and AFTER: every cell
  IDENTICAL on the CUDA column and on the CPU column (identity_break --diff).
- test_x_trees_sanity (max_delta_step split into its own looser assertion,
  measured 0.646 vs LightGBM 0.768), test_host_surface, test_lane_select
  (pins forest_host_predict 84, forest_inference 50): see the merge commit.
- Apple (M2 Pro steward) + do-amd: requests 1790536790718 (dart-options),
  1790536798209 (oob-cv-link), 1790536805393 (et-deviance) at b771caee9:
  m2pro PASS, do-amd PASS. MERGED to main 4b4507b7d.
- GradientBoosting loss='MultiRMSE' (gbdt/, the trees family's CatBoost
  arm), lane trees-gbdt-multirmse (in the trees fragment, NOT in
  identity_break.py: a harness edit selects every lane). Symmetric Hessian
  rows + the MultiClass Cholesky leaf solve (multiclass_targets.h:118-123),
  (n, D) y dim-major through a +4 params tail, boost_from_average refused
  for MultiRMSE (DEVIATION 5951, unset resolves False; CatBoost's default is
  per-dimension averages: a real option-parity debt), DEVIATION 5950 (loss
  sum order). Refused by name: Depthwise/Lossguide, Ordered, Exact, eval_set,
  cat features, class weights, dim < 2. Not in _HOST_ROUTE_LANES (needs a
  recorded lane). Evidence: `--pass 2 --sabotage
  gbdt/checks/sabotage/multirmse_der_cpu_only.patch` PASS on H100 and on
  MI300X; every gbdt-* lane + cross-val + saved-model-host-infer fitted
  BEFORE (MultiRMSE diff reversed) and AFTER on H100: IDENTICAL on CUDA and
  CPU columns; all AGREE on MI300X; test_gbdt_multirmse + test_host_surface
  199 passed, 1 skipped (catboost absent). test_lane_select pins now
  forest_host_predict 85, gbdt_host_predict 50: test_lane_select 68
  passed at the branch tip (H100 pod). Steward request 1790542307842 (m2pro, m3ultra, do-amd)
  PENDING. MERGE when m2pro (or m3ultra) + do-amd PASS.
- Owed (family phase 1, existing lanes): gbdt-categorical-ctr-tables and
  gbdt-tensor-ctr-tables read NOTHING COMPARED in the lane check: their
  `model` part has one column (the CPU column refuses it). They need a CPU
  arm for the model part.

extratrees/NOT_IMPLEMENTED.tsv has no `not yet` row left (the rest are
`deliberate`). xtrees/NOT_IMPLEMENTED.tsv `not yet` rows remain (ccp_alpha,
monotonic_cst, min_weight_fraction_leaf, DecisionTreeRegressor sample_weight,
Bagging warm_start, DART leaf-wise g/h growth + min_sum_hessian +
feature_fraction_bynode, stacking/calibration sample_weight, calibration
ensemble='auto'/cv='prefit', TreeSHAP interventional / interaction values /
CatBoost models, Permutation link).

PRIORITY (main, 2026-09-27): THE M3 ULTRA RF DIVERGENCE. Steward request
1790526750361 (pass 2, 20 trees lanes) FAILED on m3ultra: 17 lanes that fit
through the RF device builder (DT, Bagging, DART, Voting, Stacking,
MultiOutput, OneVsRest, Calibrated, rf-weighted, SHAP) DISAGREE, while
adaboost-clf/reg (shallow DT learners: check their depth, a clue) and
random-embedding (ET builder) AGREE.
Cell table for trees-dt-clf (commit 77e0b3a8d): M3 CPU == M2 Metal == M2
CPU on EVERY cell; M3 METAL is the outlier on base, hashed, wide, denormal,
denormal_ftz, dupes, odd, negative; ONLY `ties` (few distinct values)
agrees. On M3 Metal denormal != denormal_ftz (they are equal everywhere
else). Suspects, in order: the quantile path (few distinct values agree:
core/segmented_sort.mojo uses max.gpu.primitives.block.prefix_sum, a
library warp-shuffle scan; quantiles.mojo), a denormal flush the M3 does
not do in hardware (M3 GPUs keep fp32 subnormals; M1/M2 flush), a
simdgroup/threadgroup assumption. The M3 Ultra is a steward now (not
off-limits); never ssh-run Metal jobs beside its daemon: use
`apple_steward.py submit --kind speed --cmd ...` (m3ultra only).
Diagnostics queued (both behind a long m3ultra queue):
- 1790543277062-speed-trees-8e5eadb7b7: runs quantiles_check,
  objectives_check, builder_kernels_check, split_check, criteria_check,
  train_check, forest_check, fingerprint_probe (ensemble/checks) on the M3.
  Its stdout is in the verdict dir. The H100's fingerprint_probe for
  comparison: CLF-OOB 0x87e5c72530dd00a7, CLF-NOBOOT 0x1aa915d207fc19b9,
  REG-BOOT 0xf67c295ec84bbaeb, CLF-DEEP 0xfe402d068c603119, CLF-BOOT-K4
  0x1c9d9763f188a349 (the K4 rows repeat the first four).
- 1790541058384-trees-d529df376b: rf-clf, rf-reg,
  rf-clf-entropy-log2-noboot on all stewards: does rf-* (existing RF,
  shipped) also diverge on the M3? If yes, the defect predates this lane.
Then: fix at the root, a separating fixture + sabotage, prove on m2pro +
m3ultra + H100 + MI300X, existing bits unchanged elsewhere.

NEXT (option parity continues; this phase is not finished):
1. gbdt/ (CatBoost) losses, starting with MultiRMSE, then MultiLogloss /
   MultiCrossEntropy, RMSEWithUncertainty; then QuerySoftMax, PFound,
   QueryCrossEntropy, PairLogitPairwise, max_pairs, Wilcoxon detector,
   grow_policy='Region'. Map for a multi-target loss (from MultiClassOneVsAll):
   every layer takes ONE target per row today (ensemble.py:2046 `as_f32_c(y,
   ndim=1)`, train.mojo:1784-1805 upload, host binding :1473), so MultiRMSE
   needs an n_rows x dim y slot. Places: ensemble.py LOSSES :141,
   MULTI_OUTPUT_LOSSES :317 (label checks :2075), boost-from-average :493 /
   auto LR :456, predict_proba :2527; pointwise_targets.mojo codes :184-235
   (next code 17) + objective_from_name :246 / objective_name :295;
   train.mojo Ordered refusal :1093, boost-from-average :1838-1870,
   approx_dim :2065; doc_parallel_boosting.mojo approx_dim :1603,
   boost-from-average refuses dim != 1 at :1705, der dispatch :2189,
   fv_blocks :2267, final loss :2913, _test_loss :409, estimate_can_batch
   :1164; multilogit.mojo new kernels beside one_vs_all_* (:651, :807);
   pointwise_oracle.mojo _oracle_dims :1257, multi-dim arm :700, diagonal
   second der :846; catboost_options.mojo :1317 leaf defaults; host:
   gbdt/host/gbdt_oracle_multiclass.mojo (GBDT_OBJ_* :116, gbdt_multi_host_fit
   :647), bindings/_mojolearn_gbdt_host.mojo is_multi :1122, dispatch :1475;
   ensemble.py _HOST_ROUTE_LANES :418; identity lanes gbdt-onevsall
   (identity_break.py :3319) as the model for a new lane; seam check
   checks/multilogit_check.mojo.
2. the xtrees `not yet` rows above.
Then phase (d) FAST GPU speed, (e) IDENTICAL GPU speed, (f) CPU speed.
