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
- OWED: AMD column (box trees-amd up on Hot Aisle), M2 Pro steward, card diff across boxes.

Next: AMD box (`tools/dev_pod.sh up trees 240 --vendor amd`), then the steward, then option parity.
