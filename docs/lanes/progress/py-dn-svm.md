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

## Proof and timing: UNPROVEN (checking stopped by Andrew's order, 2026-09-28)

Checking stopped on Andrew's order before any lane check or timing ran;
py-consolidated merges this branch and runs one global check. What did run,
on the shared pod nvc1 (2x A40, x86-64, CPU only, `sh`, no GPU):

- `_mojolearn_svm_host` and `_mojolearn_svm` BUILD (host rebuilt after the
  last source change; the GPU binding was built one commit earlier and is
  stale by stamp).
- `svc_equal.py` on the HOST binding (MOJOLEARN_VENDOR=cpu), EQUAL RESULT PASS:
  `pm_exp` and `pm_log` equal `mojolearn_exp` / `mojolearn_log` (portable_math.c
  through `_portable_math._native`) on 1,000,025 inputs each, including the
  overflow and underflow cutoffs, subnormals, infinities, NaN and inputs next
  to the exp reduction boundary (0 differ); `svc_splitmix_perm` equals
  `_splitmix_perm` at n = 1, 2, 50, 1000, 4097, 100000; `svc_platt_train`
  equals `_sigmoid_train` bit for bit in 60 random trials (n 1 to 5000,
  one class only included); epilogue modes ovo, ovr, votes, proba, log proba
  equal the Python reference at K = 2, 3, 4, 7, 10 (3000 rows, decisions with
  exact 0.0, -0.0 and integer ties); binary codes equal.
- One bug the check caught and the branch fixes: `sigmoid_train` read its
  target list through a pointer after the List's last use (Mojo destroys it
  there); it is now kept alive to the end.

Not run (owed to the global check): the lanes
x-neighbors-svc-probability, x-neighbors-svc-multiclass, svc, svc-linear,
svc-poly, x-neighbors-svc-sigmoid, x-neighbors-svm-precomputed,
x-neighbors-svm-weights, par-svm (GPU == CPU and SAME against the base
columns), `svc_equal.py` on the GPU binding, and every timing (job scripts
~/mojolearn-evidence/py-dn-svm/job_a.sh, job_b.sh, svc_bench.py,
svc_micro.py; copies on the pod in /root/ev-py-dn-svm/). Job nvc1-0013
(job_a.sh) was queued before the order and was left queued, not cancelled.
