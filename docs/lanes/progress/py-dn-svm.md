# py-dn-svm (SVM sub-lane of py-decomp-nbrs), 2026-09-28

Branch `lane/py-dn-svm`, cut from `lane/py-decomp-nbrs`. Audit rows: neighbors
2, 3, 4, 15 (labels part), 21 (not moved), core 7 (Platt part); ranks 6 and 14.

## What moved from Python into Mojo

`svm/host/svc_proba.mojo` (host only, compiled into `_mojolearn_svm` and
`_mojolearn_svm_host`, exports `svc_pair_epilogue`, `svc_platt_train`,
`svc_splitmix_perm`, `svc_portable_math`):

| Python before | Mojo now |
|---|---|
| `predict_proba`: per row, per pair `_sigmoid_predict`, clamp, `_multiclass_probability` | `svc_pair_epilogue` mode 3, rows split over host tasks |
| `predict_log_proba`: `math.log` per element of `predict_proba` | mode 4 (`pm_log`), `v <= 0` raises "math domain error" as before |
| multiclass `predict`: Python vote loop | mode 2, int64 codes into the native `decode_labels` |
| `predict(break_ties=True)`, `decision_function` ovr: `_ovr_scores` | mode 1, then the native `argmax_rows` |
| `decision_function` ovo: transpose comprehension | mode 0 |
| binary `predict`: `[1 if v == label1 else 0 ...]` | mode 5 |
| `_pair_decisions`: one `tolist()` per pair | the pair machines write one `(n_pairs, n)` float32 Array |
| fit, probability=True: `_splitmix_perm` (O(n) Python per pair) | `svc_splitmix_perm` |
| fit, probability=True: `_sigmoid_train` / `_platt_fval` (n portable exp and log per Newton step) | `svc_platt_train` |

Same bits by construction: binary64 throughout, the Python loop order, every
product that feeds an add or subtract `pinned_mul_f64`, CPython's `min`/`max`
argument semantics for the clamp, and `pm_exp` / `pm_log`, line for line
twins of `mojolearn_exp` / `mojolearn_log` in
`packaging/portable_math/portable_math.c` (the library `_portable_math`
loads). NOT `checks/numerics.mojo::portable_exp64`: its reduction `k` is one
fused rounding, the C's (built `-ffp-contract=off`) is two.

The Python functions (`_splitmix_perm`, `_platt_fval`, `_sigmoid_train`,
`_sigmoid_predict`, `_multiclass_probability`, `_ovr_scores`) remain as the
reference the Mojo is held to (and that `tests/test_svc_probability.py`
exercises); the estimator no longer calls them.

Not moved: the per fold row gathers (`_rows_copy`, `_precomputed_*`), the
per pair index lists of `_fit_probability`/`_fit_ovo`, `coef_`'s
`_dual_times_sv` (DEVIATION 2372 unchanged), the per pair `svc_predict`
re-upload of q (DEVIATION 873 unchanged).

## DEVIATION ledger

The SVC Platt / coupling binary64 host arithmetic has no ledger row; the
py-bugs lane owns adding it (origin/lane/py-bugs had no change and no
progress file when this lane was cut, 2026-09-28 19:20Z). This lane does
not add a duplicate. For that row: the arithmetic now runs in Mojo
(`svm/host/svc_proba.mojo`), the Python implementation is retired from the
runtime and kept as the reference only.

## Proof and timing

PENDING (see below).
