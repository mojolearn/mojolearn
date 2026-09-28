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
| seams 6100-6108 (`x_metrics/seams/x_metrics_check.mojo`, 8 seams host+device) | PASS on RTX 4090 |
| `--pass 2`, 5 lanes + `e2e_host_fadd.patch` (host-only +1 ulp in `fadd`) | all 8 seam arms FAIL under their patch and PASS after reversal; the 5 lanes AGREE, DISAGREE on 9/9 fixtures under the sabotage, AGREE after (RTX 4090) |
| `--pass 2`, x-metrics-splitters + `e2e_host_permute.patch` (host-only reversed key order) | AGREE, DISAGREE on 9/9 fixtures under it, AGREE after (RTX 4090). The first run was refused because the selector did not see model_selection reach x_metrics; fixed (`_execute` is a module-level function, model_selection names `_SPLIT_BINDING`) |
| FAST tier sanity | the FAST x_metrics build passes the same 295 scikit-learn comparisons (the existing FAST bindings were not built on the pod) |
| the 103 lanes the diff selects (`lane_select.py --lanes-for-paths`) | 83 AGREE CUDA vs CPU (every existing metrics, cross-val, rf/et/gbdt/trees, spectral lane in the set); 18 `par-*` lanes not run: they have no CPU arm by design (cooperative multi-GPU drivers); `gbdt-categorical-ctr-tables` and `gbdt-tensor-ctr-tables` read NOTHING COMPARED on the model stage because the CPU column loads the GPU's saved file (`n/a:gpu-saved-file`) and `--require-columns 2` then cannot be met: a harness property of those two lanes, unrelated to this diff, reported to main |
| test_lane_select (feature tree) | OK: 0 failures |
| registration (EXPANSION_LANES, door, empty fragment) | merged to main a6f8617a8 on its own (manifest byte-identical; test_host_surface, test_lane_select OK), so the feature diff selects 103 lanes instead of all |
| steward identity requests at be39c62b6 | 1790544421098 (5 lanes, e2e_host_fadd), 1790544803666 (splitters, e2e_host_permute): queued on m2pro, m3ultra-b, m4pro, do-amd. Post-merge release gates (CURRENT DIRECTIVES 0000b): the next session reads `tools/apple_steward.py status`; a FAIL is fixed at the root as its own commit |

## Phase 3+4 (FAST and IDENTICAL GPU speed): session C, 2026-09-27
One change serves both tiers: the FAST x_metrics build runs the same units,
so the parallel schedules are the same bits in both modes.
- **Planner** (x_metrics/plan.mojo) rewrites every single-thread stage
  (group_sort, group_sum, col_sort, wpercentile, bin_curve, permute) into
  wide stages (x_metrics/par.mojo): chunked stable counting sort, PairSum
  fold as leaves + pairwise levels (the fixed tree is a function of the
  count, so the leaves reproduce it), merge sort by rank of the strict
  (key, row) order, gathered sequential prefixes. Both runners (device and
  host) run the planned program. `wpercentile_unit` is split: the CDF is
  gather + prefix, then `wpct_select` (op 26) does the search and average.
- The weighted-percentile CDF and the curve walk stay SEQUENTIAL Float32
  prefixes (DEVIATION 6107); they run as HOST stages inside the device
  runner (read slots down, same unit on the host, write slots up): 4-14 ms
  per 1M rows instead of 60-260 ms on one GPU thread.
- Host epilogue: `_fsum` returns `_portable_math.fsum`'s bits via
  math.fsum (3000 randomized trials incl. signed zeros, subnormals,
  overflow agree bit for bit); splitters build fold rows with
  itertools.compress into array('q').
- `MOJOLEARN_XMETRICS_PROFILE=1` prints each planned stage's device time.
- Speed board: `bench/x_metrics_speed.py` (1M rows, taxi + HIGGS from R2,
  `XMSPEED <case> <s> <digest>`). FAST paired quality check:
  `bench/x_metrics_fast_quality.py` (5 seeds x taxi + HIGGS vs scikit-learn
  1.9, per-case relative error JSON).
- Process-lifetime DeviceContext (CURRENT DIRECTIVES): the binding already
  held one (`metrics_ctx`); `python/mojolearn/tests/test_x_metrics_repeat.py`
  calls 21 entry families twice per process on GPU and host, same bytes,
  GPU == host.

| check | verdict |
|---|---|
| speed, RTX 4090, 1M rows, `--reps 2`, board total | IDENTICAL 84.86 s -> 5.75 s (14.8x); FAST 81.56 s -> 5.75 s (14.2x). Median absolute error 5.04 s -> 0.011 s, d2_absolute_error 4.80 -> 0.018, shuffle_split 18.8 -> 0.26, train_test_split 3.76 -> 0.05, roc_curve 5.56 -> 0.26, roc_auc ovr 8.48 -> 1.79; AMI 1.44 -> 1.34 (Python-bound, unchanged); max_error unchanged |
| bits unchanged | every one of the 29 board digests equal before/after, in BOTH modes (before = the base .so files, same pod, same run) |
| FAST quality (paired, 5 seeds x 2 datasets vs scikit-learn) | the FAST build's per-case relative-error JSON is byte-identical before and after (q_old.json == q_new.json): quality unchanged |
| seam gate `check_parallel_schedules` (80acfbe71) | unplanned host vs planned host vs planned device, every arena word agrees (244735 words), RTX 4090 |
| `--pass 2`, 5 lanes + e2e_host_fadd | AGREE, 9/9 DISAGREE under sabotage, AGREE after; all 8 seam arms PASS/FAIL/PASS (arms retargeted at the planned schedules, 73d473008; 6105 arm build fixed 835bca4ea) |
| `--pass 2`, x-metrics-splitters + e2e_host_permute (retargeted at par.mojo) | AGREE, 9/9 DISAGREE, AGREE after; 8 seam arms bite |
| the other lanes the diff selects (77) | AGREE CUDA vs CPU; the same two harness-property lanes `gbdt-categorical-ctr-tables` / `gbdt-tensor-ctr-tables` NOTHING COMPARED as in phase 2 (unrelated) |
| test_lane_select | OK, 0 failures |
| test_host_surface, test_x_metrics_repeat (after merging origin/main) | PASS |
| Apple speed | queued (m4pro-b): before 1790549640312-speed-metrics-51f2ef6215, after 1790549643846-speed-metrics-835bca4ead |
| AMD speed + post-merge identity | submitted at the merge (see the session report / `apple_steward.py status`, requests `*-metrics-*`) |

## Steward results read 2026-09-28 ~00:00Z (session D)
Nothing FAILED; nothing to fix. 1790544421098 (phase 2, e2e_host_fadd): PASS
m3ultra-b, m4pro-a (the m2pro copy coalesced into 1790544421099).
1790544803666 (splitters, e2e_host_permute): PASS m4pro-b. Still QUEUED, not
run: post-merge identity 1790544421099, 1790553421721 (do-amd, m3ultra-b,
m4pro-b/m4-a, m2pro); Apple speed 1790549640312 / 1790549643846 (m4pro-b);
AMD speed 1790548802096 / 1790548807312 (`pixi run -e default`, both modes in
one cmd) and 1790553428450..1790553431411 (direct `.pixi/envs/default/bin/
python`, one mode each; not failed, so not resubmitted; the steward's build
step runs `pixi run -e default` first, which creates that env in the
worktree). The next session reads `apple_steward.py status` again.

## Phase 5 (CPU speed): session D, 2026-09-28, CODE COMMITTED, POD GATE OWED
RunPod balance went negative and every pod was deleted (the `metrics` pod
gwwujixoh5dtve is gone; `dev_pod.sh up` refused "balance too low"; state
cleared with `dev_pod.sh down`). Do not retry renting until Andrew tops up.
Work on lane/metrics (pushed, NOT merged):
- 83c5c7847 `x_metrics/host/program.mojo`: every planned stage's units split
  into contiguous ranges on the MOJOLEARN_CPU_THREADS pool
  (`core/host_predict_threads.mojo`), joined before the next stage. A stage
  is one device launch, so its units are independent by construction; a
  stage fans out only at >= HOST_TASK_WORK (16384) rows of work per task.
  The host merge pass is `sort_merge_span_unit` (par.mojo, MERGE_SPAN=4096
  outputs per unit, started by a co-rank search), replacing the per-pair
  two-pointer unit, so the last passes split too. The seam gate's
  `check_parallel_schedules` now also runs the planned host at 1, 2, 3 and 8
  tasks against the sequential units; new arm
  `x_metrics/seams/sabotage/seam_host_threads.patch` in metrics.checks.
- bench/x_metrics_speed.py prints vendor and thread count.
- Checked locally (one mac_slot core, MOJOLEARN_CPU_THREADS=1): the host
  binding compiles; col_sort through the binding equals numpy's stable
  argsort at n = 200003, 70001 x 3, 4097 x 2, 5 (the span merge incl.
  co-rank starts). Multi-thread runs were NOT done locally (1-core rule).

**OWED ON A POD (in this order), then merge:**
1. `~/mojolearn-evidence/metrics/psync.sh` (new pod: `dev_pod.sh up metrics
   240`), build host + identical + fast (`pbuild.sh host identical fast`).
2. Seam gate: `x_metrics/seams/x_metrics_check.mojo` PASS (host 1/2/3/8
   tasks + device); `--pass 2` on the seam arms incl. seam_host_threads.patch
   (must FAIL under it, PASS after).
3. `algos_lane_check.sh` on the 6 x-metrics lanes (AGREE CUDA vs CPU) with
   `e2e_host_fadd.patch` / `e2e_host_permute.patch` biting, once at
   MOJOLEARN_CPU_THREADS=1 and once at the default.
4. CPU board `MOJOLEARN_VENDOR=cpu python bench/x_metrics_speed.py --reps 2`
   at base (origin/main host .so) vs new, and new at MOJOLEARN_CPU_THREADS=1,
   3 and default: every digest equal across all of them and equal to the
   GPU board's digests; FAST board likewise (FAST quality JSON unchanged).
5. test_host_surface; test_lane_select (metrics.checks changed);
   test_x_metrics_repeat.
6. Merge + push; ONE batched steward identity request (x-metrics lanes,
   e2e_host_fadd) plus CPU-speed timing on do-amd if wanted.

## Next phases
- **Read first**: `apple_steward.py status` for the metrics speed and identity
  requests; a FAIL is fixed at the root as its own commit before phase 5.
- Remaining GPU speed headroom (optional, a later pass): roc_auc ovr (1.8 s)
  and AMI (1.3 s, the expected-MI term is Python binary64) are the largest
  cases left; stratified_kfold (0.5 s) is Python bookkeeping.
- **5. CPU speed**: code committed (session D above); the pod gate list above
  is owed. After it: SIMD in the host units is the next lever (a SIMD width
  is a PIN; only per-element units such as reg_term/sort_key/gather qualify).
- Option-parity leftovers (NOT_IMPLEMENTED.tsv): multilabel-indicator targets
  across the classification metrics; silhouette's other distance metrics.
