# py-misc-metrics: progress (metrics epilogues, DEVIATION 6106)

Branch `lane/py-misc-metrics` (worktree ~/mojolearn-wt/py-misc-metrics),
forked from lane/py-misc 342469dae; the parent lane py-misc merges it.
Brief: ~/mojolearn-evidence/py_work_brief.md; audit
~/mojolearn-evidence/python_work_audit.md, metrics items 12, 15, 16, 17, 19, 20.

## What moved (0a581a9a5)

Each is the Python it stands in for, operation for operation, in
`x_metrics/epilogue.mojo`, exported by BOTH x_metrics bindings (GPU and host):

| audit item | Python before | Mojo entry |
|---|---|---|
| 12 | `precision_recall_curve_options` (weighted or drop_intermediate): lists, keep comprehension, per-point t/(t+f), t/T, reversal, 3 `from_list` | `x_metrics_curve_pr` (`pr_arrays`) |
| 12 | `det_curve`: lists, drop, T - v, bisects, v/N, v/P, reversals | `x_metrics_curve_det` (`det_arrays`) |
| 15 | `ndcg_score`: per-row a/b, `_fsum`, weighted zip multiply | `x_metrics_ndcg_mean` (`ndcg_mean`) |
| 16 | `_class_weights` (d2_log_loss, d2_brier) and `_ovr`'s weighted support | `x_metrics_class_sums` (`class_sums`) |
| 17 | public `auc()` on 1-D float32/float64 buffers | `x_metrics_auc_xy` (`auc_xy`) |
| 19 | `_mi_from_contingency` cell loop (two loop-invariant logs hoisted: the same values) | `x_metrics_mi_contingency` (`mi_contingency`) |
| 20 | calinski_harabasz centroids and between-cluster sum; davies_bouldin centroids, k^2 d distances, scores | `x_metrics_centroids`, `x_metrics_ch_extra`, `x_metrics_db_score` |

Operation rules: + - / correctly rounded binary64; every product
`pinned_mul_f64`; `fsum_strict` = CPython `math.fsum` step for step, raising
(so the Python decides) on a non-finite term, a possible intermediate
overflow or a non-finite sum, which are exactly the cases where `_fsum`
leaves `math.fsum`; `portable_log_c` for `_portable_math.log`; IEEE sqrt for
`_portable_math.sqrt` (both are the hardware square root). Every case the
Python handles by a warning, an exception or a non-finite value raises in
Mojo and falls back to the unchanged Python. `auc()` on lists, integers or
n-D input keeps the Python conversion.

Reference arm: `MOJOLEARN_METRICS_EPILOGUE=python` (or
`MOJOLEARN_HOTPATH=python`) takes the Python route in the same build.

`** 2` (libm pow, outside 6106's allowed operations) became `v * v` in
calinski_harabasz, davies_bouldin and d2_brier (`_sq`). Bits move only where
the host's pow(x, 2) is not correctly rounded. NOT done in
`model_selection.py` (StratifiedGroupKFold std at 903/915, search std at
1688): lane py-misc-msel is editing StratifiedGroupKFold's hunks. That
change is OWED to py-misc-msel or py-bugs.

## DEVIATION and ledger changes (18939b4f5)

- Row 198 (DEVIATION 6106) narrowed to the audit's text: Python binary64
  only on O(classes) / O(outputs) / O(candidates x folds) scalars; `**`, pow
  and libm transcendentals forbidden; every row-, point-, cell- or k^2 d
  scaling epilogue in `x_metrics/epilogue.mojo` under the same operation rules.
  The module docstring of `_expansion_metrics.py` says the same.
- Row 139 (decomp host control flow) narrowed: Python arithmetic only on
  scalars and O(d) / O(components) quantities (stopping tests, FactorAnalysis
  O(d) sums, chi2 bisection, PCA-mle rank); no new data-length Python work.
  The data-length sorts, Isomap reconstruction error and MDS diagonal are
  OWED moves (lane py-decomp-nbrs).

## Proof (one light job, tools/py_misc/metrics_job.sh)

PENDING (see below).
