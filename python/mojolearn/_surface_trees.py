"""THE TREES LANE'S HOST SURFACE FRAGMENT (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `trees` expansion lane and merged by host_surface.py at import
(`EXPANSION_LANES`). Literal data only: no imports, no calls but dict(...).
The names this lane may declare are its own:

  GPU_BINDINGS          ("_mojolearn_x_trees",) once bindings/build_x_trees.sh builds
                        it; every packaging list then carries it
                        (`host_surface.py --expansion-gpu-bindings`)
  FAMILIES              (dict(family="x_trees", binding="_mojolearn_x_trees_host",
                        routes="_mojolearn_x_trees", ...),), the same keys as a
                        family in host_surface.FAMILIES
  TRAINING_LANE_NAMES   {lane: "the name the docs use"} for the family's training_lanes
  PUBLIC_PENDING_LANES  {lane: "no reference"} until a release record admits it
"""
GPU_BINDINGS = ("_mojolearn_x_trees",)
FAMILIES = (
    dict(
        family="x_trees",
        binding="_mojolearn_x_trees_host",
        routes="_mojolearn_x_trees",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(
            "trees-dt-clf",
            "trees-dt-reg",
            "trees-dt-random",
            "trees-rf-weighted",
            "trees-et-deviance",
            "trees-gbdt-multirmse",
            "trees-bagging-clf",
            "trees-bagging-reg",
            "trees-adaboost-clf",
            "trees-adaboost-reg",
            "trees-dart-reg",
            "trees-dart-clf",
            "trees-dart-options",
            "trees-random-embedding",
            "trees-voting-clf",
            "trees-voting-reg",
            "trees-stacking-clf",
            "trees-stacking-reg",
            "trees-multioutput",
            "trees-onevsrest",
            "trees-calibrated",
            "trees-oob-cv-link",
            "trees-shap-tree",
            "trees-shap-kernel",
            "trees-shap-permutation",
        ),
        inference_lanes=(),
        forest_kinds=(),
        classes=(
            "DecisionTreeClassifier",
            "DecisionTreeRegressor",
            "BaggingClassifier",
            "BaggingRegressor",
            "AdaBoostClassifier",
            "AdaBoostRegressor",
            "DARTRegressor",
            "DARTClassifier",
            "RandomTreesEmbedding",
            "VotingClassifier",
            "VotingRegressor",
            "StackingClassifier",
            "StackingRegressor",
            "MultiOutputClassifier",
            "MultiOutputRegressor",
            "OneVsRestClassifier",
            "CalibratedClassifierCV",
            "TreeExplainer",
            "KernelExplainer",
            "PermutationExplainer",
        ),
        display="the trees expansion lane's ensemble glue",
        host_modules=("xtrees/ops.mojo", "xtrees/api.mojo", "xtrees/shap.mojo"),
        exports=(
            "x_trees_host_numeric_mode", "x_trees_host_vendor", "x_trees_host_column", "x_trees_host_sabotage",
            "x_trees_numeric_mode", "x_trees_vendor",
            "x_trees_sample_indices", "x_trees_weighted_sample", "x_trees_gather_f32", "x_trees_gather_i32",
            "x_trees_accumulate", "x_trees_accumulate_onehot", "x_trees_accumulate_cols",
            "x_trees_argmax_rows", "x_trees_argmax_rows_f32", "x_trees_scale", "x_trees_softmax_rows", "x_trees_scale_to_f32", "x_trees_put_f32",
            "x_trees_samme_step", "x_trees_r2_step", "x_trees_weighted_median",
            "x_trees_apply", "x_trees_gradients", "x_trees_leaf_newton", "x_trees_tree_score_add", "x_trees_uniform",
            "x_trees_onehot_leaves", "x_trees_transpose_f32", "x_trees_log64",
            "x_trees_normalize_rows", "x_trees_scatter", "x_trees_platt_fit", "x_trees_platt_apply",
            "x_trees_isotonic_fit", "x_trees_isotonic_predict", "x_trees_node_cover", "x_trees_tree_shap",
            "x_trees_expected_value", "x_trees_mask_expand", "x_trees_block_mean", "x_trees_kernel_solve",
            "x_trees_leaf_newton_rows", "x_trees_logit", "x_trees_accumulate_rows",
            "x_trees_check_weights_f32", "x_trees_mul_f32",
            "x_trees_exact_sum_f32", "x_trees_margin2",
        ),
        gate="tools/algos_lane_check.sh",
        wheel_note="Ships: the trees expansion lane's ensemble glue (pass 1, PENDING).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {
    "trees-dt-clf": "DecisionTreeClassifier",
    "trees-dt-reg": "DecisionTreeRegressor",
    "trees-dt-random": "DecisionTreeClassifier splitter=random",
    "trees-rf-weighted": "RandomForestClassifier weighted objective",
    "trees-et-deviance": "ExtraTreesRegressor poisson, gamma, inverse_gaussian",
    "trees-gbdt-multirmse": "GradientBoosting MultiRMSE (multi-target regression)",
    "trees-bagging-clf": "BaggingClassifier",
    "trees-bagging-reg": "BaggingRegressor",
    "trees-adaboost-clf": "AdaBoostClassifier",
    "trees-adaboost-reg": "AdaBoostRegressor",
    "trees-dart-reg": "DARTRegressor",
    "trees-dart-clf": "DARTClassifier",
    "trees-dart-options": "DARTClassifier multiclass and LightGBM options",
    "trees-random-embedding": "RandomTreesEmbedding",
    "trees-voting-clf": "VotingClassifier",
    "trees-voting-reg": "VotingRegressor",
    "trees-stacking-clf": "StackingClassifier",
    "trees-stacking-reg": "StackingRegressor",
    "trees-multioutput": "MultiOutputRegressor",
    "trees-onevsrest": "OneVsRestClassifier",
    "trees-calibrated": "CalibratedClassifierCV",
    "trees-oob-cv-link": "Bagging oob_score, cv splitter objects, KernelExplainer link=logit",
    "trees-shap-tree": "TreeExplainer",
    "trees-shap-kernel": "KernelExplainer",
    "trees-shap-permutation": "PermutationExplainer",
}
PUBLIC_PENDING_LANES = {
    "trees-dart-options": "no reference",
    "trees-shap-tree": "no reference",
    }
