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
| CalibratedClassifierCV (sigmoid, isotonic) | trees-calibrated | AGREE: compared batch 9, infer 9, train 9 | this commit |

Pod setup notes (for a fresh agent): the lane check does not build the
ubiquitous bindings; build `bindings/build.sh` (identical), `build_forest_host.sh`
and `build_byte_lm_host.sh` once on a fresh pod. sklearn/lightgbm/pytest live
in /root/sk on the pod (uv --target); run the sanity test with
`PYTHONPATH=python:/root/sk pixi run -e default python -m pytest -q python/mojolearn/tests/test_x_trees_sanity.py`.

Known gaps (reported to main):
- the RF weighted objective (sample_weight / class_weight without bootstrap)
  has no CPU restatement in `ensemble/host/rf_oracle.mojo`, so
  `DecisionTreeClassifier.fit(sample_weight=...)` is GPU-only (the rf host
  binding refuses it by name) and AdaBoostClassifier fits weighted bootstraps
  instead of weighted trees.

Next: the RF weighted objective on the CPU (ensemble/host/rf_oracle.mojo; main's new policy: the lane fixes it), then AdaBoostClassifier on SAMME sample weights and DecisionTree sample_weight on CPU, then SHAP TreeExplainer, KernelExplainer, PermutationExplainer.
