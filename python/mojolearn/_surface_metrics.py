"""THE METRICS LANE'S HOST SURFACE FRAGMENT (the lane was added 2026-09-27).

Owned by the `metrics` expansion lane and merged by host_surface.py at import
(`EXPANSION_LANES`). Literal data only: no imports, no calls but dict(...).
The metrics family's EXISTING binding (`_mojolearn_metrics`) and its host
family stay declared in host_surface.py; this fragment adds the lane's new
binding, `_mojolearn_x_metrics`, which carries every metric and
model_selection option the expansion added.
"""
GPU_BINDINGS = ("_mojolearn_x_metrics",)
FAMILIES = (
    dict(
        family="x_metrics",
        binding="_mojolearn_x_metrics_host",
        routes="_mojolearn_x_metrics",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(
            "x-metrics-classification",
            "x-metrics-regression",
            "x-metrics-ranking",
            "x-metrics-cluster",
            "x-metrics-splitters",
            "x-metrics-search",
        ),
        inference_lanes=(),
        forest_kinds=(),
        classes=(
            "metrics.balanced_accuracy_score",
            "metrics.class_likelihood_ratios",
            "metrics.classification_report",
            "metrics.cohen_kappa_score",
            "metrics.fbeta_score",
            "metrics.hamming_loss",
            "metrics.jaccard_score",
            "metrics.matthews_corrcoef",
            "metrics.multilabel_confusion_matrix",
            "metrics.precision_recall_fscore_support",
            "metrics.zero_one_loss",
            "metrics.d2_absolute_error_score",
            "metrics.d2_pinball_score",
            "metrics.d2_tweedie_score",
            "metrics.explained_variance_score",
            "metrics.max_error",
            "metrics.mean_absolute_percentage_error",
            "metrics.mean_gamma_deviance",
            "metrics.mean_pinball_loss",
            "metrics.mean_poisson_deviance",
            "metrics.mean_squared_log_error",
            "metrics.mean_tweedie_deviance",
            "metrics.median_absolute_error",
            "metrics.root_mean_squared_log_error",
            "metrics.auc",
            "metrics.average_precision_score",
            "metrics.brier_score_loss",
            "metrics.coverage_error",
            "metrics.d2_brier_score",
            "metrics.d2_log_loss_score",
            "metrics.dcg_score",
            "metrics.det_curve",
            "metrics.hinge_loss",
            "metrics.label_ranking_average_precision_score",
            "metrics.label_ranking_loss",
            "metrics.ndcg_score",
            "metrics.roc_curve",
            "metrics.top_k_accuracy_score",
            "metrics.adjusted_mutual_info_score",
            "metrics.calinski_harabasz_score",
            "metrics.contingency_matrix",
            "metrics.davies_bouldin_score",
            "metrics.normalized_mutual_info_score",
            "metrics.pair_confusion_matrix",
            "model_selection.KFold",
            "model_selection.StratifiedKFold",
            "model_selection.GroupKFold",
            "model_selection.StratifiedGroupKFold",
            "model_selection.TimeSeriesSplit",
            "model_selection.ShuffleSplit",
            "model_selection.StratifiedShuffleSplit",
            "model_selection.GroupShuffleSplit",
            "model_selection.LeaveOneOut",
            "model_selection.LeavePOut",
            "model_selection.LeaveOneGroupOut",
            "model_selection.LeavePGroupsOut",
            "model_selection.RepeatedKFold",
            "model_selection.RepeatedStratifiedKFold",
            "model_selection.PredefinedSplit",
            "model_selection.train_test_split",
            "model_selection.check_cv",
            "model_selection.cross_validate",
            "model_selection.cross_val_predict",
            "model_selection.ParameterGrid",
            "model_selection.ParameterSampler",
            "model_selection.GridSearchCV",
            "model_selection.RandomizedSearchCV",
            "model_selection.validation_curve",
            "model_selection.learning_curve",
            "model_selection.permutation_test_score",
            "model_selection.get_scorer",
            "model_selection.make_scorer",
            "model_selection.get_scorer_names",
        ),
        display="the evaluation metrics and model_selection helpers added by the metrics lane",
        host_modules=(
            "x_metrics/host/program.mojo", "x_metrics/common.mojo", "x_metrics/units.mojo",
            "x_metrics/group.mojo", "x_metrics/regression.mojo", "x_metrics/ranking.mojo", "x_metrics/cluster.mojo", "x_metrics/split.mojo",
            "x_metrics/epilogue.mojo",
        ),
        exports=(
            "x_metrics_host_numeric_mode", "x_metrics_host_vendor", "x_metrics_host_column",
            "x_metrics_host_sabotage", "x_metrics_run", "x_metrics_run_out", "x_metrics_curve_roc", "x_metrics_expected_mi", "x_metrics_row_sum_range", "x_metrics_numeric_mode", "x_metrics_vendor",
            # lane py-misc-metrics (2026-09-28): x_metrics/epilogue.mojo
            "x_metrics_curve_pr", "x_metrics_curve_det", "x_metrics_ndcg_mean", "x_metrics_class_sums",
            "x_metrics_auc_xy", "x_metrics_mi_contingency", "x_metrics_centroids", "x_metrics_ch_extra",
            "x_metrics_db_score",
            # lane metrics-apple3 (2026-09-28): cross_val_predict's row scatter, integer label plumbing
            "x_metrics_scatter_rows", "x_metrics_encode_small_i64", "x_metrics_first_rows", "x_metrics_ovo_pair",
            "x_metrics_expected_mi_tasks", "x_metrics_row_sum_range_tasks",
            # lane pyglue-sweep (2026-10-03): the clustering metrics' contingency epilogue
            "x_metrics_contingency_stats",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note="Ships: the metrics lane's CPU route (the added evaluation metrics and model_selection helpers).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {
    "x-metrics-classification": "balanced_accuracy, matthews_corrcoef, cohen_kappa, jaccard, fbeta, "
                                "precision_recall_fscore_support, hamming/zero-one loss, multilabel_confusion_matrix, "
                                "class_likelihood_ratios and the sample_weight options of the classification metrics",
    "x-metrics-regression": "the regression metrics the lane added (MSLE, MAPE, pinball, median absolute error, "
                            "max error, explained variance, Tweedie deviances, the D^2 scores) and the "
                            "sample_weight / multioutput / force_finite options of the existing ones",
    "x-metrics-ranking": "the ranking and probabilistic scores the lane added (ROC/DET curves, average "
                         "precision, top-k accuracy, Brier, hinge, DCG/NDCG, the label-ranking scores, the "
                         "D^2 log-loss and Brier scores) and the weighted / partial / multiclass options of "
                         "roc_auc_score, precision_recall_curve and log_loss",
    "x-metrics-cluster": "the clustering scores the lane added (Calinski-Harabasz, Davies-Bouldin, normalized "
                         "and adjusted mutual information, the contingency and pair-confusion matrices)",
    "x-metrics-splitters": "the model_selection splitters (KFold ... PredefinedSplit), train_test_split and the "
                           "parameter grid / sampler, seeded by the counter RNG",
    "x-metrics-search": "cross_validate, cross_val_predict, scorer names and GridSearchCV",
}
PUBLIC_PENDING_LANES = {
    }
