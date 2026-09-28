# metrics-apple3: progress

Lane `metrics-apple3` (Apple FAST speed round 3, ~/mojolearn-evidence/apple3_speed_brief.md):
the metrics family (the 24 metrics, CV splitters, model_selection over tree and
classical estimators). Worktree `~/mojolearn-wt/metrics-apple3`, branch
`lane/metrics-apple3`, forked from lane/apple3-merged 6856b5f8f. Round 2:
docs/lanes/progress/metrics-apple2.md. Evidence: ~/mojolearn-evidence/metrics-apple3/.

## Targets (from the measurements we already have)
Round 2's step 3 (m4pro-a, steward 1790619894321, FAST, 1M rows) and its
profile lines, largest seconds first:

| case | FAST s | where the time is (round-2 profile) |
|---|---|---|
| roc_auc_ovr | 0.228 | `x_metrics_row_sum_range` 0.119 (host, one fsum with a heap list per row), device 0.059, arena fill 0.019, 5 curve epilogues 0.020, label encode 0.018 |
| adjusted_mutual_info_score | 0.137 | `x_metrics_expected_mi` 0.114 (host, one thread, 25 cells) |
| stratified_kfold_shuffle | 0.069 | device 0.037, first-seen label encode in Python 0.022 |
| shuffle_split | 0.054 | device 0.061 under the profiler (5 sorts) |
| every classification count metric | 0.022 to 0.029 | `encode_labels_i64` twice = 0.017 (core binding), device 0.006 |

Model selection had no Apple measurement at all (py-misc-msel: "Apple: not
run"). Read from the code, per fit: the fold's rows of X are gathered again
for every candidate of a search, every value of a validation curve and every
permutation; every scorer of a multimetric fold calls predict again;
learning_curve gathers each training subset twice and the test rows once
per size, from Python lists; cross_val_predict makes a Python object per row.

## How it is measured
`bench/x_metrics_apple_ab.sh <base sha>` (round 2's script, extended): base
tree and head tree on the same Mac in one steward job.
- XMAB_BUILDS: the estimators' bindings built in the head tree, copied to the base tree.
- The board (bench/x_metrics_speed.py) base/head, IDENTICAL and FAST, two passes.
- XMAB-EQ: tools/apple_speed_metrics/eq_cases.py (round 2's cases plus wide
  and exact score rows at four sizes and many-class AMI / NMI), base == head per mode.
- XMAB_MSEL=1: tools/apple_speed_metrics/msel_eq.py (78 model selection
  results: base tree, head under MOJOLEARN_MSEL3_BEFORE=1, head as shipped;
  all equal per mode), then bench/x_msel_speed.py (1M rows, taxi + HIGGS,
  phases per case) before / after in the same build.
- XMAB_EXTRAS=1: the epilogue python-vs-native table, the whole-arena arm
  (MOJOLEARN_ARENA_RANGES=0 against 1), py-misc-msel's check.py.

## Changes
All keep every word: IDENTICAL and FAST run the same code, no FAST-only
approximation was added, so the paired quality numbers are unchanged.

1. `row_sum_range` (x_metrics/epilogue.mojo): each row's `fsum` by a running
   binary64 sum with Fast2Sum's error term; while every error is 0 the running
   sum is the exact sum, which is fsum's value; the first nonzero error sends
   the row to `fsum` itself. Rows run as host tasks (`host_parallelize`); a
   largest and a smallest value do not depend on the order of the rows.
2. `expected_mi`: the (a, b) cells run as host tasks in the caller's
   floating-point environment; each task reduces its cells' terms to
   math.fsum's partials (an exact expansion of their sum) and the partials are
   fsum-ed. fsum returns the correctly rounded exact sum, so grouping moves no bit.
3. `scatter_rows` + `x_metrics_scatter_rows` (both x_metrics bindings):
   cross_val_predict's row scatter as a byte copy.
4. model_selection.py, each with its definition kept as the before arm
   (`MOJOLEARN_MSEL3_BEFORE=1`) and the fallback:
   - `_FoldRows`: a search, a validation curve and a permutation test gather
     each fold's rows once (kept while all folds fit in 1 GiB,
     `MOJOLEARN_MSEL_FOLD_CACHE_MB`);
   - the scorers of one multimetric fold share one prediction per response
     method (`_Scorer.__call__(_memo=)`); a callable scorer still gets the estimator;
   - learning_curve: one gather per fold at the largest size, every smaller
     size a prefix view, test rows once per fold; the same draws;
   - cross_val_predict: partition test by `check_indices_i64`, assembly by
     `scatter_rows`, widened as the definition widens.

5. Integer labels (x_metrics/epilogue.mojo `encode_small_i64`,
   `first_rows_i32`; python `_metrics_impl._encode_small_native`,
   `model_selection._first_seen_native`): labels that span fewer than 65536
   values are encoded by host tasks (the span, a seen-byte per value, one
   table load per row) instead of the core encoder's binary search per row
   on one thread; the same ascending classes and ranks. StratifiedKFold reads
   its first-seen codes from them (each class's first row ranks the classes)
   instead of a Python object per row. A wider span, too many classes or an
   older binary takes the core encoder as before.
6. The host arena of a program of 65536 words or more is an anonymous zero
   mapping (`_expansion_metrics._Arena`): the same zeros, but only the pages
   the inputs and outputs live in are ever touched
   (`array.array("f", bytes(4 * size))` wrote every page twice).
7. tools/apple_speed_metrics/epilogue3_words.mojo: changes 1, 2, 3 and 5
   against their sequential definitions at 1, 2, 3 and 8 host tasks.

Changes 4, 5 (the Python side) and 6 share the before arm
`MOJOLEARN_MSEL3_BEFORE=1`.

SHARED CODE: none outside x_metrics, `_metrics_impl.py`, `_expansion_metrics.py`
and model_selection.py so far. The
epilogue now imports core/host_parallel.mojo and core/host_predict_threads.mojo
(read only, not edited).

## Results
| steward | Mac | commit | what | state |
|---|---|---|---|---|
| 1790626827717 | m4-a | 52469ad90 (= the base + the phase board) | the base measured: model selection phases, the board, the epilogue table, py-misc-msel's timing | PASS |
| 1790628637582 | m3ultra-b | 81e13a96c | the A/B of changes 1 to 7 (base 6856b5f8f against head; before against after in the head build) | queued |

### Job 1790626827717 (m4-a, Apple M4): the base, no change of this lane in it
Evidence: ~/mojolearn-evidence/metrics-apple3/1790626827717-52469ad90.txt.
The FAST estimators `_mojolearn_metrics.so` was not built in this job, so the
tree cases and the multimetric case have IDENTICAL numbers only.

Model selection, 1M rows x 16 features (taxi for regression, HIGGS for
classification), seconds, one run after a warm run. Phases are exclusive.

| case | mode | wall | fit | predict | score (less predict) | take_rows | folds | other (Python) |
|---|---|---|---|---|---|---|---|---|
| cross_val_score Ridge, cv 5 | FAST | 0.298 | 0.173 | 0.014 | 0.061 | 0.027 | 0.020 | 0.002 |
| cross_val_score LogisticRegression | FAST | 0.500 | 0.391 | 0.016 | 0.033 | 0.028 | 0.030 | 0.002 |
| cross_val_score GaussianNB | FAST | 0.252 | 0.146 | 0.023 | 0.032 | 0.030 | 0.020 | 0.002 |
| cross_val_score GaussianNB | IDENTICAL | 1.149 | 1.020 | 0.030 | 0.040 | 0.035 | 0.021 | 0.003 |
| cross_val_score DecisionTree(8) | IDENTICAL | 0.395 | 0.263 | 0.016 | 0.052 | 0.030 | 0.030 | 0.004 |
| cross_val_score RandomForest(10 trees, 8) | IDENTICAL | 1.173 | 1.035 | 0.025 | 0.051 | 0.029 | 0.030 | 0.004 |
| cross_val_score GradientBoostingRegressor(20, 4) | IDENTICAL | 1.120 | 1.039 | 0.024 | 0.006 | 0.029 | 0.019 | 0.003 |
| cross_validate GaussianNB, 3 scorers + train scores | IDENTICAL | 1.793 | 1.000 | 0.419 (30 calls) | 0.320 | 0.031 | 0.020 | 0.003 |
| GridSearchCV Ridge, 4 candidates x 5 folds | FAST | 1.181 | 0.728 | 0.059 | 0.245 | 0.117 (80 gathers) | 0.020 | 0.012 |
| validation_curve Ridge, 4 values x 5 folds | FAST | 2.292 | 0.689 | 0.204 | 1.248 | 0.119 | 0.020 | 0.012 |
| learning_curve Ridge, 5 sizes x 5 folds | FAST | 2.032 | 0.514 | 0.165 | 0.991 | 0.157 (150 gathers) | 0.019 | 0.185 |
| learning_curve GaussianNB, shuffled | FAST | 2.290 | 0.415 | 0.391 | 0.499 | 0.356 | 0.020 | 0.610 |
| cross_val_predict Ridge | FAST | 0.478 | 0.172 | 0.016 | | 0.028 | 0.020 | 0.242 |
| cross_val_predict GaussianNB predict_proba, 200k rows | FAST | 0.110 | 0.031 | 0.008 | | 0.006 | 0.004 | 0.062 |
| permutation_test_score GaussianNB, 5 permutations, cv 5 | FAST | 1.577 | 0.851 | 0.135 | 0.189 | 0.166 (120 gathers) | 0.117 (6 draws) | 0.120 |

What it says: a fit is one binding call in every estimator measured
(their families own it). This family's share is the row gathers repeated
per candidate and per permutation, one predict per scorer, the Python lists
of learning_curve (0.19 to 0.61 s) and the Python objects of
cross_val_predict (0.24 s of 0.48 s).

The board at the base (29 cases, 1M rows, `--reps 2`): FAST 1.101 s,
IDENTICAL 1.114 s. Largest: roc_auc_ovr 0.266, adjusted_mutual_info_score
0.138, shuffle_split 0.077, stratified_kfold_shuffle 0.075.

lane py-misc-metrics' epilogues had never run on Apple. Python route against
native route, the same build, 1M rows (FAST; IDENTICAL is within 10%):

| case | python s | native s | bits |
|---|---|---|---|
| precision_recall_curve, weighted | 0.185 | 0.041 | equal |
| det_curve | 0.186 | 0.034 | equal |
| roc_auc ovr weighted, 1M x 5 | 0.330 | 0.303 | equal |
| d2_log_loss_score, weighted | 0.055 | 0.023 | equal |
| ndcg_score, 200k x 5 | 0.024 | 0.009 | equal |
| auc, 1M points | 0.175 | 0.016 | equal |
| normalized_mutual_info, 1000 x 1000 classes | 0.704 | 0.423 | equal |
| calinski_harabasz, 100k x 50, k 300 | 0.026 | 0.023 | equal |
| davies_bouldin, 100k x 50, k 300 | 0.398 | 0.115 | equal |

lane py-misc-msel's splitters had never run on Apple either
(tools/py_misc_msel/check.py time, IDENTICAL, 1M rows, the digest of before
and after the same in every row):

| case | before s | after s |
|---|---|---|
| LeaveOneGroupOut, 1000 groups | 56.576 | 3.258 |
| LeavePGroupsOut(2), 30 groups | 25.117 | 1.753 |
| GroupKFold(5), 1000 groups | 0.370 | 0.024 |
| GroupKFold(5, shuffle) | 0.359 | 0.024 |
| StratifiedGroupKFold(5), 100 groups | 0.420 | 0.095 |
| GroupShuffleSplit(10), 1000 groups | 0.527 | 0.054 |
| StratifiedShuffleSplit(10) | 1.645 | 0.379 |
| PredefinedSplit, 5 folds | 0.304 | 0.054 |
| KFold(5) unshuffled | 0.234 | 0.016 |
| iterable cv, 5 pairs | 0.137 | 0.016 |
| check_cv stratify test (float y) | 0.307 | 0.001 |
| scorer roc_auc column, (n, 2) f4 | 0.120 | 0.025 |
| permutation_test_score 10 perms, KFold(5) | 3.427 | 0.513 |
| permutation_test_score 10 perms, cv 5 (stratified) | 3.537 | 1.371 |
| permutation_test_score 3 perms, 1000 groups, GroupKFold(5) | 50.790 | 1.321 |

## For other lanes (found by the phase board, not changed here)
- linear: `Ridge.score` / `LinearRegression.score` computes R^2 in Python per
  row (`linear_model._r2_host`, `_r2_sums`, `tolist`): 27 to 31 ms per call at
  800k rows on the M4, 1.24 s of the 2.29 s validation curve and 0.99 s of
  the 2.03 s learning curve above. `scoring="r2"` goes through
  mojolearn.metrics and does not pay it.
- prep: the classifiers' `score` (`_expansion_prep.py` `score`, a Python `sum`
  over a generator per row) is 17 to 27 ms per call at 200k to 800k rows:
  0.86 s of the shuffled GaussianNB learning curve. GaussianNB's fit is
  0.20 s IDENTICAL against 0.03 s FAST at 800k x 16 (one `x_prep_run_ranges` call).
- core (`bindings/hotpath_helpers.mojo` `_encode_labels`): one thread, a
  binary search per row, 5 ms per 800k int64 labels; every classifier fit and
  every classification metric calls it (twice per metric). Change 5 answers
  the metrics' calls inside this family; the core encoder is unchanged.

## Unproven
- Changes 1 to 7 until job 1790628637582 reports: the Mojo changes have not been built.
