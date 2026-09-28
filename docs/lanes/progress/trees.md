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

SESSION A (verification), 2026-09-27 evening. DONE, merged:
- THE M3 RF DIVERGENCE, FIXED AT THE ROOT (DEVIATION 5611, IDENTITY_PATHS
  row 169). Diagnosis from the FAIL verdict's own cells (steward request
  1790526750361 on m3ultra): depth-2/3 learners (AdaBoost) and a
  one-column-per-node forest (RandomTreesEmbedding) AGREE, every depth-8
  multi-column tree DISAGREES, and `denormal` vs `denormal_ftz` (the same
  X after the IDENTICAL flush) gave DIFFERENT forests on the one M3: a
  run-to-run race, not arithmetic. The only in-kernel cross-threadgroup
  payload in the RF builder was `_publish_to_global`: the node's `Split`
  read and written with PLAIN loads/stores inside a device-mutex critical
  section (a lost candidate on M3). Fix: under IDENTICAL, on every vendor,
  `HIST_SPLIT_CANDIDATES_DEFAULT` is on: each column block stores its
  pinned (DEVIATION 404) winner in its own slot
  (`Split.eval_best_split_pinned_to_candidate`) and
  `merge_split_candidates_kernel` folds a node's slots with `update` (a
  total order) after the kernel boundary. The ET `split_reduce_kernel`
  runs one block per node under IDENTICAL (`ET_SPLIT_REDUCE_ONE_BLOCK`),
  so its mutex merge never runs either.
  Evidence (H100 pod, commit b55282c5): 36 lanes (every rf-*, et-*,
  saved-model-host-infer, all 24 trees-* RF/ET lanes) AGREE CPU == CUDA;
  every CUDA cell of all 36 IDENTICAL to the pre-fix runs (/root/ev/et_after,
  /root/ev/m_trees); sabotage `xtrees/checks/sabotage/seam_5611_lost_candidate.patch`
  (the fold drops the last column block) on trees-dt-clf, trees-dt-reg,
  rf-clf, rf-reg: AGREE, DISAGREE, AGREE after reversal. MI300X (trees-amd):
  the same 36 AGREE CPU == HIP, HIP cells unchanged where a prior exists
  (26), and the HIP cells equal the CUDA cells on all 36.
  POST-MERGE, QUEUED: steward request 1790536790720-trees-b25e0fdfe8
  (m2pro, m3ultra-b, m4-a, do-amd; coalesced every earlier queued trees
  request): the 20 trees lanes + trees-dart-options, trees-et-deviance,
  rf-clf, rf-reg, rf-clf-entropy-log2-noboot, trees-gbdt-multirmse, pass 2,
  sabotage xtrees/checks/sabotage/steward_combo_cpu_only.patch (the four
  CPU-only column patches in one). Read it with `apple_steward.py status`.
  An M3 FAIL comes back as a fix commit before any B item.
  If M3 still disagrees, the next suspect is any other plain cross-block
  read in the RF builder (none found by grep) or the quantile path.
  OWED (FAST, not identity): ET under FAST on Apple still merges bpn > 1
  blocks through the mutex (quality risk on M3 when k > TPB);
  neighbors/impl/detail/fused_l2_knn.mojo has the same mutex-payload
  pattern (neighbors family, reported to main).
- MultiRMSE (trees-gbdt-multirmse) MERGED under the NVIDIA + CPU gate.
- REPEATED CALLS (directive): python/mojolearn/tests/test_trees_repeat.py
  calls every trees entry point (RF/ET clf+reg, IsolationForest, GBDT
  Logloss/RMSE/Depthwise/Lossguide/MultiRMSE, OrderedRMSE, FeatureFreq,
  host_predict on saved RF and GBDT files, DT, Bagging, AdaBoost,
  DART, RandomTreesEmbedding, Voting, OneVsRest, TreeExplainer) twice in
  one process on GPU and on CPU, asserting first == second and GPU == CPU.
  The trees bindings build a DeviceContext per call but return host data
  only (no buffer outlives its context). PASS on the H100 (GPU == CPU,
  first == second) and on the MI300X (run without pytest, the box has
  none). test_host_surface + test_trees_repeat 201 passed and
  test_lane_select 69 passed at the merge tip (H100). The trees lanes
  iforest, iforest-tuned, trees-gbdt-multirmse, gbdt-yeti-rank,
  saved-model-host-infer AGREE (H100); iforest, trees-gbdt-multirmse,
  gbdt-symmetric, gbdt-feature-freq, gbdt-ordered-rmse AGREE (MI300X).
- GradientBoosting.predict on a CPU install routes a model whose text
  carries CTR tables / a tensor CTR registry through HostGBDT (the gbdt
  host binding's walk refuses those records by name). Inert for every
  existing lane (30 gbdt lanes AGREE, CUDA cells unchanged).
- test_lane_select pins after merging main: kmeans_oracle 71,
  gbdt_host_predict 51, forest_host_predict 86 (x-metrics-search from
  the metrics lane reaches both).

SESSION C, 2026-09-28 ~00:20Z (RunPod balance negative: every RunPod pod,
the trees H100 included, is gone and `dev_pod.sh up` is refused; work ran on
the Hot Aisle MI300X `trees-amd` only). STOPPED at this checkpoint by the
coordinator.
- STEWARD 1790536790720 (the 5611 RF fix, 26 lanes): m3ultra-b PASS (THE M3
  DIVERGENCE IS FIXED), do-amd PASS, m4-a working, m2pro FAIL: clean
  trees-gbdt-multirmse DISAGREE (8 of 9 fixtures, `ties` IDENTICAL; parts
  differ: predict, infer, model, batch). The M2 Pro METAL column is the odd
  one: its CPU column, the M3 Ultra Metal and CPU columns and the M4 Pro
  (1790542307842) all read 4226ed22.. on base; M2 Metal reads 5ae37033..
  Built from source at the commit (not a seeded binding). Every other lane
  in the request AGREES on the M2. Unknown yet whether ANY gbdt lane agrees
  on the M2 (no gbdt lane has run there before). DIAGNOSTIC QUEUED:
  speed request 1790553780797-speed-trees-b23b38412e on m2pro: lane check
  (pass 1) of gbdt-symmetric, gbdt-rmse, gbdt-multiclass, gbdt-onevsall,
  trees-gbdt-multirmse, then ~/mojolearn-evidence/trees/diag_m2_multirmse.py
  (Metal vs CPU model text at n_estimators 1, 2, 3, 6, 12 on base and ties,
  MultiRMSE and RMSE; prints the first differing lines). The script reads
  all IDENTICAL on the MI300X (HIP vs CPU). Read it with
  `apple_steward.py status` / `cloudmac.sh ssh m2pro 'cat
  ~/mojolearn-evidence/apple-steward/done/1790553780797-speed-trees-b23b38412e/*'`.
  FIX AT ROOT BEFORE ANY B ITEM once it says where the M2 first moves.
- trees-oob-cv-link: its only M3 verdict (1790536798209, m3ultra FAIL) is
  from b771caee9, BEFORE the 5611 fix; it was not in 1790536790720. Owed:
  put it in the next identity submit (M3 re-run).
- DONE ON THE BRANCH (lane/algos-trees, NOT merged: the NVIDIA gate is
  owed): gbdt-tensor-ctr-tables trains on the CPU column. The host
  FeatureFreq oracle (gbdt/host/gbdt_oracle_feature_freq.mojo) no longer
  refuses a level winner on the tensor column: `persist_synchronized_mixed_path`
  restated (the winning level's table at model column n_features + k, its
  split history remapped to an earlier tensor winner's stable column), the
  canonical `TFeatureTensor.get_hash` restated on the host (the device module
  imports the CTR bin builder), `tensor_ctr_registry` / `feature_freq_tensor`
  records and `type tensor_ctr` columns in the text. On this lane BOTH
  levels win on the tensor column (registry 6 2). Lane body: fits on both
  columns (`_ctr_saved_or_fit` dropped for this lane). GradientBoosting.load
  on a CPU install reads a CTR model's dim through HostGBDT's parser (the
  host binding's gbdt_model_dim refuses CTR records; it was the only
  refusal left, on the model part). Sabotage
  gbdt/checks/sabotage/tensor_ctr_count_cpu_only.patch (row 0 counted in the
  neighbouring key; a doubled count is refused by the reader instead).
  Evidence, MI300X (trees-amd, cpu-intel-r-xeon-r-platinum-8470):
  `--pass 2 --sabotage tensor_ctr_count_cpu_only.patch` on
  gbdt-tensor-ctr-tables: AGREE (batch/infer/model/train 9), DISAGREE (model
  DIVERGENT 9/9), AGREE after reversal: PASS. gbdt-feature-freq,
  gbdt-categorical-ctr, saved-model-host-infer, gbdt-symmetric AGREE.
  OWED ON AN NVIDIA POD before merge (the gate): `sh tools/algos_lane_check.sh
  gbdt-tensor-ctr-tables --pass 2 --sabotage
  gbdt/checks/sabotage/tensor_ctr_count_cpu_only.patch`; `algos_lane_check.sh
  gbdt-feature-freq,gbdt-categorical-ctr,saved-model-host-infer` AGREE; the
  CUDA cells of gbdt-tensor-ctr-tables and gbdt-feature-freq unchanged vs
  main (GPU code untouched: host oracle, lane body, a CPU-only load branch);
  test_host_surface; test_lane_select (its inputs changed: ensemble.py,
  identity_break.py lane body, the gbdt host oracle). Then merge + push, then
  one steward identity submit: gbdt-tensor-ctr-tables with that sabotage,
  plus trees-oob-cv-link (M3 re-run).
- NOT STARTED: gbdt-categorical-ctr-tables (multi-permutation CPU boosting,
  map below); type B; the ExtraTrees FAST Apple mutex-merge quality check
  (bpn > 1 blocks under FAST still fold through the mutex).
- NOTE from the cpu lane (DEVIATION 5900): GBDT's small-fit border search
  runs serially under IEEE on the CPU but on FTZ+DAZ workers on the device.
- MY ERROR this session: `dev_pod.sh extend trees 0` (a probe for the ssh
  target) set a zero-minute lease on the H100; the balance deletion took
  every RunPod pod at the same time, so the pod was gone either way, but
  never pass 0 to extend.

SESSION trees-cpu (the trees CPU-speed lane, branch lane/trees-cpu, pod
`trees-cpu` RunPod H100, cpu-intel-r-xeon-r-platinum-8480), 2026-09-28:
- DONE: gbdt-categorical-ctr-tables TRAINS ON THE CPU COLUMN (item 2 of the
  list below). New gbdt/host/gbdt_oracle_ctr.mojo restates train's CTR
  prelude and column loop: the default GPU simple CTRs (Borders at three
  priors, ParamId 0, Uniform 15; FeatureFreq (0,1), MinEntropy 15), the
  target grid, one CTR order per permutation
  (`ctrs_estimation_permutation(n, p).fill_order()`), the FeatureFreq column
  (`TWeightedBinFreqCalcer`) and the ORDERED Borders columns per permutation
  (`THistoryBasedCtrCalcerGpu`: integer counts before the row in the
  permutation's stable category order, one Float32 divide), permutation 0's
  values deciding a dependent column's grid, one compressed index per
  permutation, `build_ctr_tables` (reused, host code on the device path)
  and the CTR model text. gbdt/host/gbdt_oracle.mojo's boosting loop is now
  `gbdt_host_boost` over one index and one cursor per permutation: the learn
  permutation draw (`TRandom(iteration + seed)`, `Advance(10)`, their
  `% (learnPermutationCount - 1)`), the structure searched on the learn
  permutation's index and cursor, every permutation estimating the same
  structure on its own cursor (the learn one over the searcher's partition,
  the others over `compute_bins_for_model` + the stable `partition_from_bins`),
  the in-loop learn loss from the learn cursor, the final loss from the
  estimation cursor. One permutation is the old loop statement for statement.
  Binding: the flags arm dispatches to the CTR arm when a categorical column
  is above one_hot_max_size (eval_set with CTR columns refused by name);
  `GradientBoosting.fit`/`.load` on a CPU install read a CTR model's dim
  through HostGBDT's parser (the load hunk is lane/algos-trees' own, same
  text). Lane body: fits on both columns (`_ctr_saved_or_fit` no longer used
  by this lane). test_trees_repeat gains a CTR fit.
  Evidence (H100 pod): `algos_lane_check.sh gbdt-categorical-ctr-tables
  --pass 2 --sabotage gbdt/checks/sabotage/ctr_ordered_cpu_only.patch` (the
  host ordered statistic counts the row's own target): AGREE (batch, infer,
  model, train 9 each), DISAGREE (every part), AGREE after reversal: PASS.
  A direct fit (1500 rows, 20 depth-6 trees, 8 CTR columns, 4 permutations)
  wrote a CPU model text byte-identical to the CUDA one on the first run.
- DONE: DEVIATION 5900's GBDT item (the cpu lane's note 4). `train`'s phase
  B border search now asks `calc_quantization(..., flush_subnormals=True)`,
  whose `best_split` flushes by bits exactly as the host oracle's
  `_best_split_phase_b` (values flushed on entry, both halves and their sum
  flushed, no fma), so the device borders no longer depend on whether the
  task runs on an FTZ+DAZ `sync_parallelize` worker or an IEEE
  `host_parallelize` task. The oracle's small serial fits were already
  env-independent (explicit ftz everywhere). This also closes the
  unmeasured arm where the fused `0.5*a + 0.5*b` kept a subnormal half the
  oracle flushed. gbdt/resident_model.mojo's `sync_parallelize` predict
  tasks are NOT pinned (owed, if the cpu lane moves them to host_parallelize).
- EXISTING BITS: the 36 non-par lanes lane_select attributes to the branch,
  fitted on main f237f1996 and on the branch (merged with main 9f2d2b120,
  which brought no gbdt change): every
  CUDA and CPU cell IDENTICAL (71 of 72 column files; the 72nd is the CTR
  lane's CPU column, which refused before), 35 lanes AGREE after
  (gbdt-tensor-ctr-tables was NOTHING COMPARED on both sides: lane/algos-trees'
  CPU arm was not on main yet; after merging it, see below). The 15 par-* lanes need two GPUs and were not run (the
  phase-B pin is inert on every recorded fixture).
- AFTER MERGING MAIN (lane/algos-trees' tensor CTR CPU arm, 276990727):
  gbdt-categorical-ctr-tables, gbdt-tensor-ctr-tables, gbdt-feature-freq,
  gbdt-symmetric, gbdt-categorical-ctr, saved-model-host-infer AGREE.
  test_host_surface then found `host_model` with NO lane (both CTR table
  lanes had reached it through `_ctr_saved_or_fit`); the CTR tables lane
  now checks `ml.host_model(<saved file>)` answers exactly what the fitted
  estimator answers, as a raise and not a part (its cells unchanged on both
  columns, identity_break --diff IDENTICAL). test_lane_select pin
  neural_inference.py 41 -> 40 (gbdt-tensor-ctr-tables no longer reaches
  it). test_host_surface + test_trees_repeat 201 passed; test_lane_select
  82 passed.

NEXT SESSION: FIRST the CTR-table CPU paths (main's request 2026-09-27),
then type B (features).
1. gbdt-tensor-ctr-tables: the CPU column fits
   ExperimentalTwoLevelFeatureFreq through gbdt/host/gbdt_oracle_feature_freq.mojo,
   which REFUSES "a level-one winner on the FeatureFreq tensor column"
   (:639) and a level-two one (:665). Restate on the host: the level
   winner on the tensor column (the split-history table after it,
   `stage_next_feature_freq_after_winner`), the tensor_ctr_registry and
   feature_freq_tensor records with their canonical tensor hash
   (gbdt/models/tensor_ctr_value_table.mojo, model_text.mojo:374-670), and
   the `features n m` header with the tensor column. Then change the lane
   body in tools/identity_break.py to fit on both columns (drop
   `_ctr_saved_or_fit`; a lane-body edit selects only that lane); predict
   already routes through HostGBDT (above). Sabotage: the tensor count or
   the prior in the host restatement.
2. gbdt-categorical-ctr-tables: CPU TRAINING with CTR categoricals
   (cat_features above one_hot_max_size: Borders at three priors +
   FeatureFreq per column). Map: gbdt/train.mojo:1533-1660 (the column
   prep: `compute_simple_ctrs` host, `compute_simple_ctrs_gpu` per
   permutation over `ctrs_estimation_permutation`, `build_ctr_tables`),
   :1745-1778 (one compressed index per permutation, est_perm =
   permutation_count - 1), doc_parallel_boosting.mojo:1821+ (one cursor
   per permutation, `perm_cindexes`). The host oracle gbdt/host/gbdt_oracle.mojo
   runs ONE permutation; the new arm needs the per-permutation ordered CTR
   columns and cursors. Host binding refusal: `_refuse` "no CTR
   categoricals" in bindings/_mojolearn_gbdt_host.mojo.
3. Type B: the remaining xtrees/extratrees/gbdt NOT_IMPLEMENTED rows
   (list below and item 1 of NEXT).

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

SESSION D, 2026-09-28 ~01:05Z (RunPod funded; NVIDIA pod `trees` H100
vgv1hkkbm9comp; AMD `trees-amd` Hot Aisle MI300X kept to ~01:10Z Sep 29).
- GATE DONE, MERGED: gbdt-tensor-ctr-tables CPU training. H100: `--pass 2
  --sabotage gbdt/checks/sabotage/tensor_ctr_count_cpu_only.patch` AGREE
  (batch/infer/model/train 9), DISAGREE (model DIVERGENT 9/9), AGREE after
  reversal: PASS. Every other lane lane_select names (34, par-* excluded)
  AGREE on CUDA == CPU; gbdt-categorical-ctr-tables reads NOTHING COMPARED
  (the CPU column refuses CTR categoricals: item 2 of the CTR list) and reads
  the same on the merge base with the lane patch reversed, so it is not this
  change. test_host_surface 200 passed; test_lane_select OK (0 failures).
  MI300X: the same lane set incl. gbdt-tensor-ctr-tables AGREE (RC 0).
- trees-oob-cv-link M3 RE-RUN: 1790558067096 m3ultra-b PASS, m4pro-a PASS
  (at aefdd7f9f, after the 5611 fix). OWED ITEM 3 CLOSED.
- M2 DIAGNOSTIC (1790553780797) READ: NOT MultiRMSE. On the M2 Pro Metal
  column EVERY gbdt lane checked DISAGREES (gbdt-symmetric, gbdt-rmse,
  gbdt-multiclass, gbdt-onevsall, trees-gbdt-multirmse) on the non-integer
  fixtures; `ties` (6 distinct values, <= 5 borders) is IDENTICAL. The model
  header (borders) is identical; the Metal tree is `depth 1`, `split 0 0 0
  0` (feature 0, bin 0: the first candidate, i.e. every score equal) where
  the CPU grows depth 6. So the fault is in the split search on 128-border
  (one-byte) features on Apple8 (M2) only; M3/M4 agree. Kernel checks
  queued: m2pro 1790558473013, control m4pro-a 1790558481928.
