# metrics: progress

Lane `metrics` (docs/lanes/ALGORITHM_EXPANSION_PLAN.md LANE CHARTER): the
evaluation metrics, cross-validation and model_selection. Worktree
`~/mojolearn-wt/metrics`, branch `lane/metrics`, pod `metrics` (RunPod RTX
4090). Numbers: DEVIATION 6100-6199, IDENTITY_PATHS rows 190-199 (claimed in
the registry tables).

## How the lane computes
The EXISTING metrics (`_mojolearn_metrics`, python/mojolearn/_metrics_impl.py)
are untouched on their default paths. Everything the lane added runs on ONE
new binding, `_mojolearn_x_metrics` (+ host `_mojolearn_x_metrics_host`), in
the prep lane's program model: `x_metrics_run` runs a program of units over
one float32 arena, one thread per unit on the GPU (x_metrics/device.mojo, one
process-lifetime DeviceContext), the same units in a loop on the CPU
(x_metrics/host/program.mojo). Units: x_metrics/group.mojo (stable counting
sort by key, PairSum per group, pair keys), regression.mojo (per-element
terms, per-column stable sort, weighted percentile, column max), ranking.mojo
(binary curve = sort + cumulative counts, per-row scores), cluster.mojo
(row-to-centroid distance), split.mojo (counter-RNG permutation). Op table
x_metrics/units.mojo == `_OPS` in python/mojolearn/_expansion_metrics.py.
O(n) work is on the device; the O(classes) epilogue is host binary64 with
+ - * / sqrt only and `_portable_math` (DEVIATION 6106). New metric
functions live in `_expansion_metrics.py` and are re-exported from
`mojolearn.metrics`; splitters / search in `model_selection.py`. Existing
functions route to x_metrics ONLY when a new option is used (sample_weight,
multioutput, force_finite=False, max_fpr, multiclass AUC, drop_intermediate,
normalize=False, scorer names).

Sanity vs scikit-learn 1.9 on the pod: `PYTHONPATH=python:/root/skl
.pixi/envs/default/bin/python python/mojolearn/tests/test_x_metrics_sanity.py`
(sklearn installed with the pixi python: `.pixi/envs/default/bin/python -m
ensurepip; ... -m pip install --target /root/skl scikit-learn==1.9.0`).
295 metric comparisons + the unshuffled splitters' exact indices PASS.

Pod sync: the full `dev_pod.sh sync` uploads ~800 MB and crawled; use
`~/mojolearn-evidence/metrics/psync.sh` (a git patch against the merge base,
applied with --index on the pod; seconds).

## Phase 1+2 (verification + option parity): this session
Added (all with sample_weight where scikit-learn has it):
- classification: balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
  jaccard_score, fbeta_score, precision_recall_fscore_support, hamming_loss,
  zero_one_loss, multilabel_confusion_matrix, class_likelihood_ratios,
  classification_report; sample_weight on precision/recall/f1/confusion_matrix,
  accuracy_score(normalize=False).
- regression: mean_squared_log_error, root_mean_squared_log_error,
  mean_absolute_percentage_error, mean_pinball_loss, median_absolute_error,
  max_error, explained_variance_score, mean_tweedie/poisson/gamma_deviance,
  d2_tweedie/pinball/absolute_error_score; sample_weight + multioutput on
  MSE/MAE/RMSE/r2, r2 force_finite=False.
- ranking / probabilistic: roc_curve, det_curve, auc, average_precision_score,
  top_k_accuracy_score, brier_score_loss, d2_brier_score, d2_log_loss_score,
  hinge_loss, dcg_score, ndcg_score, coverage_error,
  label_ranking_average_precision_score, label_ranking_loss; roc_auc_score
  sample_weight / max_fpr / multiclass ovr+ovo / labels / averages;
  precision_recall_curve sample_weight + drop_intermediate; log_loss sample_weight.
- clustering: normalized/adjusted_mutual_info_score, contingency_matrix,
  pair_confusion_matrix, calinski_harabasz_score, davies_bouldin_score.
- model_selection: KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold,
  TimeSeriesSplit, ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit,
  LeaveOneOut, LeavePOut, LeaveOneGroupOut, LeavePGroupsOut, RepeatedKFold,
  RepeatedStratifiedKFold, PredefinedSplit, train_test_split, check_cv,
  cross_validate, cross_val_predict, ParameterGrid, ParameterSampler,
  GridSearchCV, RandomizedSearchCV, validation_curve, learning_curve,
  permutation_test_score, get_scorer / make_scorer / get_scorer_names;
  cross_val_score scoring by name.
Refused by name / replaced: metrics/NOT_IMPLEMENTED.tsv (multilabel-indicator
targets, sparse contingency, numpy RandomState / scipy distributions, n_jobs>1).

| lanes | verdict |
|---|---|
| x-metrics-classification, -regression, -ranking, -cluster, -splitters, -search | AGREE, train 9 fixtures, cuda RTX 4090 vs CPU (Ryzen 7950X), 2026-09-27 |
| seams 6100-6108 (`x_metrics/seams/metrics_check.mojo`, 8 seams host+device) | PASS on RTX 4090; `--pass 2` arms: see below |

(pass-2 results, steward verdicts and merge commit are appended below as they land)

## Next phases
- **3. FAST speed** (next session): every x_metrics unit is one thread per
  work item and several are single-thread (group_sort, col_sort, bin_curve,
  permute, wpercentile). FAST: parallel radix/segmented sorts, block-parallel
  group sums and prefix scans, one thread block per row for the row metrics;
  the existing `_mojolearn_metrics` kernels are in the family too. Measure at
  1M+ rows on R2 data (NVIDIA pod, AMD via do-amd, Apple via the stewards,
  `--kind speed`), paired quality check vs scikit-learn at 5+ seeds on 2+
  datasets. The splitters' and the search's Python bookkeeping (fold masks,
  `_encode_*`) also want native helpers at 1M rows.
- **4. IDENTICAL speed**: the same units parallel under IDENTICAL with the
  same bits (PairSum's fixed tree is already a function of the count, so a
  block-parallel fold that reproduces its leaves is bit-compatible).
- **5. CPU speed**: threads / SIMD in the host runner, bits identical at
  every thread count.
- Option-parity leftovers (NOT_IMPLEMENTED.tsv): multilabel-indicator targets
  across the classification metrics; silhouette's other distance metrics.
