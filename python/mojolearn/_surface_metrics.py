"""THE METRICS LANE'S HOST SURFACE FRAGMENT (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md;
the lane was added 2026-09-27, docs/lanes/ALGORITHM_EXPANSION_PLAN.md item 1a).

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
        ),
        display="the evaluation metrics and model_selection helpers added by the metrics lane",
        host_modules=(
            "x_metrics/host/program.mojo", "x_metrics/common.mojo", "x_metrics/units.mojo",
            "x_metrics/group.mojo",
        ),
        exports=(
            "x_metrics_host_numeric_mode", "x_metrics_host_vendor", "x_metrics_host_column",
            "x_metrics_host_sabotage", "x_metrics_run", "x_metrics_numeric_mode", "x_metrics_vendor",
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
}
PUBLIC_PENDING_LANES = {
    "x-metrics-classification": "no reference",
}
