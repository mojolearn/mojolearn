# py-misc-msel: progress (model selection sub-lane of py-misc)

Branch `lane/py-misc-msel` (worktree ~/mojolearn-wt/py-misc-msel), forked from
lane/py-misc 342469dae, merged origin/lane/py-shared 05bf6ad0d (native label
encoder in `_encode_sorted`). Brief: ~/mojolearn-evidence/py_work_brief.md;
audit: ~/mojolearn-evidence/python_work_audit.md, metrics + model selection
items 1, 2, 3, 4, 7, 22.

## Changes (python/mojolearn/model_selection.py; no Mojo, no rebuild)

Every route uses core helpers both binaries already export (`_mojolearn`
and `_mojolearn_core_host`): `encode_labels_*`, `fold_ids` (per-group row
counts), `gather_i32` (per-group table to per-row words), `select_fold_i64`
(each split's ascending test and train rows), `gather_i64` (permuted row
lists), `gather_rows_bytes`, `transpose_f32`, `reduce_stat` (integral test),
and the x_metrics `group_sort` / `rows64` stages. Integer bookkeeping and
byte moves only; the same rows in the same order and the same RNG draws in
the same order. The Python code stays as the definition and the route
under `MOJOLEARN_HOTPATH=python`, below 256 rows, under the fold-order
sabotage control, or when a helper is missing. `MOJOLEARN_MSEL_BEFORE=1`
(read per call) is the before arm in the same build.

| item | path | before | after |
|---|---|---|---|
| 1 | LeaveOneGroupOut, LeavePGroupsOut | per split, a Python filter over all rows + mask + two compress passes | groups encoded once natively; per split one `select_fold_i64` (LeavePGroupsOut: a per-group table through `gather_i32` first) |
| 7 | GroupKFold (both), StratifiedGroupKFold, GroupShuffleSplit | per fold a Python comprehension over rows with set lookups | per-group fold table, one `gather_i32`, one `select_fold_i64` per fold (StratifiedGroupKFold's assignment `_assign` unchanged, shared by both routes) |
| 7 | StratifiedShuffleSplit (and train_test_split(stratify=)) | per split per class a list comprehension, list concat, two more permuted comprehensions | class rows once by `select_fold_i64`; per split the same draws via `permutation_rows`, `gather_i64` straight into train/test, `gather_i64` for the final shuffles |
| 7 | PredefinedSplit, unshuffled KFold | per fold comprehension / mask + compress | native encode of the fold values + `select_fold_i64`; KFold through `_native_default_folds` |
| 22 | `_IterableCV` | `flatten_labels` (tolist) + array('q') per side | int64/int32 buffer widened in C |
| 3 | `_Scorer` binary proba column | `pred.tolist()` + comprehension + `from_list` | `as_f32_c` + `transpose_f32` + one memmove |
| 4 | permutation_test_score | per permutation Python list gather + `from_list`; with groups a Python filter per group per permutation | y converted once; per permutation `gather_rows_bytes` by the permutation (groups: one program of the same per-group draws, `gather_i64` per group over the group-sorted rows, inverse sort by a second `group_sort`); folds computed once when the splitter cannot read y and is deterministic |
| 2 (part) | `check_cv` stratify test | `flatten_labels` + two Python `all` scans per call | numeric 1-D buffer: dtype, or `reduce_stat` integral test |
| 2 | GridSearch / RandomizedSearch / validation_curve per-candidate setup | | NOT TOUCHED: blocked on py-bugs (owns the fold-redraw fix; lane/py-bugs had no commits at 19:15Z). `_cross_validate_on` (cross_validate's loop over validated folds) is now a separate function py-bugs can reuse. |

## Verification: job nvc1-0001 (NVIDIA A40 pod ayjqqutlqcebzt, Xeon Gold 6342; 19:36 to 20:05Z, exit 0)

Tree 308878e8 (this branch). Evidence on the pod: /root/ev-py-misc-msel/20260928T193611Z.

- Before == after, identity lanes x-metrics-splitters + x-metrics-search
  (identity_break, --repeats 1, before arm MOJOLEARN_MSEL_BEFORE=1):
  GPU 18 of 18 cells identical hashes, CPU (host bindings) 18 of 18. PASS.
- Equality script (tools/py_misc_msel/check.py equal: 15 splitters x 4 label
  and group kinds x 3 sizes, iterable cv, check_cv decisions, 4 binary proba
  scorers on f4 / f8 / F-order proba with a NaN, permutation_test_score over
  7 cv/group/y-kind combinations plus GaussianNB): GPU 257 of 257 byte-equal,
  CPU 257 of 257. PASS.
- GPU == CPU lane check (tools/algos_lane_check.py): RESULT FAIL before the
  diff, NOT from this lane: metrics.checks line 7's sabotage patch
  x_metrics/seams/sabotage/seam_6105_contraction.patch does not apply to
  x_metrics/par.mojo on the lane/py-misc base either (checked with
  `git apply --check` on 342469dae). The GPU and CPU identity JSONs of this
  job were not cross-diffed (stop order); py-consolidated's global check owns
  GPU == CPU.

Timing, 1,000,000 rows, one process, before then after (seconds):

| case | GPU arm before | GPU arm after | x | CPU arm before | CPU arm after | x |
|---|---|---|---|---|---|---|
| LeaveOneGroupOut, 1000 groups | 218.690 | 14.273 | 15.3 | 88.547 | 8.682 | 10.2 |
| LeavePGroupsOut(2), 30 groups | 44.033 | 4.951 | 8.9 | 42.218 | 5.098 | 8.3 |
| GroupKFold(5), 1000 groups | 0.780 | 0.066 | 11.9 | 0.784 | 0.077 | 10.2 |
| GroupKFold(5, shuffle) | 0.774 | 0.075 | 10.4 | 0.700 | 0.065 | 10.8 |
| StratifiedGroupKFold(5), 100 groups | 0.826 | 0.199 | 4.2 | 0.750 | 0.208 | 3.6 |
| GroupShuffleSplit(10), 1000 groups | 1.216 | 0.123 | 9.9 | 1.215 | 0.133 | 9.1 |
| StratifiedShuffleSplit(10) | 3.975 | 0.425 | 9.4 | 6.578 | 2.346 | 2.8 |
| PredefinedSplit, 5 folds | 0.558 | 0.123 | 4.5 | 0.670 | 0.162 | 4.1 |
| KFold(5) unshuffled | 0.385 | 0.044 | 8.7 | 0.479 | 0.066 | 7.3 |
| iterable cv, 5 pairs | 0.331 | 0.042 | 7.9 | 0.406 | 0.073 | 5.6 |
| check_cv stratify test (float y) | 0.658 | 0.003 | 207 | 0.807 | 0.005 | 172 |
| scorer roc_auc column, (n, 2) f4 | 0.264 | 0.023 | 11.5 | 0.344 | 0.083 | 4.1 |
| permutation_test_score 10 perms, KFold(5) | 6.018 | 1.022 | 5.9 | 8.126 | 2.175 | 3.7 |
| permutation_test_score 10 perms, cv=5 (stratified) | 7.805 | 2.799 | 2.8 | 10.027 | 4.372 | 2.3 |
| permutation_test_score 3 perms, 1000 groups, GroupKFold(5) | 83.077 | 1.039 | 80.0 | 85.106 | 0.919 | 92.6 |

Every row: the digest of the before and after results is the same ("same yes").
The permutation cases use a trivial hash estimator, so they time the plumbing only.

## DEVIATION changes

None. DEVIATION 3104 (Python fold bookkeeping below 256 rows) still holds:
every new route keeps the 256-row floor.

## Unproven / not done

- GPU == CPU for the two lanes on this tree (the lane check stopped at the
  pre-existing seam_6105 patch failure; left to py-consolidated).
- Apple: not run (m2pro only after 21:16Z; stop order).
- StratifiedGroupKFold still encodes y first-seen and fills the group x class
  table in Python (O(n)); its `** 2` std (audit conformance gap) unchanged.
- cross_val_predict (item 9), split_descriptor (21), learning_curve (py-bugs).
