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

## Verification (one light job, tools/py_misc_msel/job.sh)

PENDING: no shared NVIDIA pod was up at 19:19Z (nvidia_central: only the
orchestrator brings pods up).

## DEVIATION changes

None. DEVIATION 3104 (Python fold bookkeeping below 256 rows) still holds:
every new route keeps the 256-row floor.

## Unproven / not done

- Everything above until the job runs.
- StratifiedGroupKFold still encodes y first-seen and fills the group x class
  table in Python (O(n)); its `** 2` std (audit conformance gap) unchanged.
- cross_val_predict (item 9), split_descriptor (21), learning_curve (py-bugs).
