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
(none yet: the first job is queued)

| steward | Mac | commit | what | state |
|---|---|---|---|---|
| 1790626827717 | m4-a | 52469ad90 | phase board at the base (x_msel_speed.py, cProfile), board, epilogue table | queued |

## Unproven
- Everything above until its job reports: the Mojo changes have not been built.
