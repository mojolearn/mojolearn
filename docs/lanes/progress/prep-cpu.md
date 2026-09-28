# prep-cpu: progress

Phase D (CPU speed, FAST and IDENTICAL) for the prep family's `x_prep` host path.
Worktree `~/mojolearn-wt/prep-cpu`, branch `lane/prep-cpu`, pod `prep-cpu` (RunPod RTX 4090, AMD Ryzen 9
7950X, cgroup quota 6.8 CPUs: "default threads" below means 16 tasks on about 7 cores).
The family's own progress (phases A-C, option parity) is docs/lanes/progress/prep.md.

## What changed (all IDENTICAL: the same words at every thread count, CPU == GPU)
The host binding `_mojolearn_x_prep_host` runs the same units as the device (x_prep/common.mojo).
1. **Threads** (x_prep/host/program.mojo): each stage's units split into contiguous ranges over
   `core/host_parallel.mojo` (tasks in the caller's FP environment, DEVIATION 5900; the count is
   `core/host_predict_threads.mojo`'s MOJOLEARN_CPU_THREADS policy), joined before the next stage.
   A unit's words never depend on which thread runs it.
2. **Host spellings of the same words** (x_prep/host/, each file says why its words are the unit's):
   - `sort_cols`: std sort of the 32-bit `key`s (one-to-one off NaN), heap sort for a column of two
     NaN payloads (sort.mojo; IDENTITY_PATHS row 142 widened).
   - `te_enc`: one ascending walk per (fold, feature, target column) folds each row into its
     category, instead of one walk per category: O(n + CMAX) for O(n CMAX) (target.mojo; the
     finishing formula `te_value` is shared with the unit).
   - `pt_fit`: each row's logarithm and the Jacobian sum once per column, one exponential per row per
     lambda, T kept for the variance pass (power.mojo; `power_log` / `power_from_log` shared with
     the unit's `power`).
   - `mi_cd`, `mi_dc`, `mi_cc`: the k-th neighbour pair and the counts from value-sorted columns
     (binary search + tie groups; the Chebyshev search stops a side past the k-th primary) instead
     of an O(n^2) scan; a non-finite column runs the units (mutual_info.mojo; row 147 widened).
   - `matmul` per output row (B read once, in order), `class_stats` per column (all classes in one
     walk), `qda_cov` per covariance row, `qda_dec` per projection (dense.mojo); `kbins_edges`
     kmeans centre update in one walk (x_prep/kbins.mojo `kbins_edges[True]`). Row 140 widened.
3. **Proof**: `check_host_sort`, `check_host_power`, `check_host_te`, `check_host_mi`,
   `check_host_dense` in x_prep/seams/prep_check.mojo compare each host spelling to the device's
   unit called directly, word for word; arms `seam_5402_host_sort`, `host_mi_ties`, `host_te_fold`,
   `host_power_jacobian`, `host_dense_order` (tools/identity_lanes/prep.checks) each FAIL the check
   (pod, 2026-09-28).

FAST on the CPU: the CPU route is IDENTICAL only (`_backend._cpu_only_binding`, and
bindings/build_host_family.sh refuses any other mode), so a FAST call on a CPU install runs this
same path; there is no separate FAST CPU arithmetic to speed up and no quality check owed.

## Before -> after, CPU, 1M rows (seconds, wall per fit+transform/predict, one process)
Data from R2: HIGGS (first 1M x 28, NaN-injected 10% for the imputers), covtype tiled to 1M
(LOWCARD = slope + soil + wilderness codes; COUNTS = |first 10 columns|; CAT = the 44 indicators).
Harness ~/mojolearn-evidence/prep-cpu/prof.py (digests of every output: 0 mismatches between the
baseline and every thread count). base = origin/main f237f1996, serial.

| case | base | 1 thread | 3 threads | default |
|---|---|---|---|---|
(filled from the final run below)

## Gate
(filled below)

## Session 2026-09-28 ~04:55Z (resume): BLOCKED, nothing merged
- Step 0 coverage audit (reading only): every algorithm in the Lane 5 table and its 2026-09-27
  additions has an x-prep-* lane in tools/identity_lanes/prep.py (CPU and GPU arms) and a biting
  sabotage recorded in docs/lanes/progress/prep.md (e2e_host_branch for the summing lanes,
  e2e_store_branch for the store lanes, seams 5400-5409, and this lane's six host arms). No gap found.
- Branch gate state (pod uk8yx8d9ajdnd1, before it died): the 48 prep lanes SAME BITS at threads
  1 / 3 / default vs base; kbins lanes AGREE at 1 / 3 / default; e2e_host_branch AGREE, DISAGREE,
  AGREE. OWED: test_host_surface (the pod's default env has no pytest: run it in the test env),
  the before -> after table, merge.
- Blocker 1: the pod is gone and `dev_pod.sh up prep-cpu` refuses: RunPod "account balance is too
  low to rent a pod". The RunPod pod listing is EMPTY (every lane's pod is gone).
- Blocker 2: this branch imports core/host_parallel.mojo, which is still only on lane/cpu
  (15 commits ahead of main). Merging lane/prep-cpu before lane/cpu merges would break main's
  x_prep host build. Merge order: lane/cpu first, then this branch (re-run the gate on main's copy).

## Next
- Python overhead now dominates many fits at 1M rows (the arena copy in `_Prog.run` / `get`,
  0.1-0.5 s; TargetEncoder's shuffled fold assignment in Python, ~1 s; LabelEncoder /
  LabelBinarizer's per-value Python label checks, ~1.3 s). The fold assignment is integer
  splitmix64 Fisher-Yates: a native unit would be bit-exact by construction, but it needs a new op
  id, so it waits for lane algos-prep2's op additions (87-100 and its WIP) to land first.
- `qt_apply` (binary search per element), `gnb` `class_stats` at K = 2 (28 groups), kmeans
  assignment (O(n nb) per iteration; the rows are sorted, a bisection over sorted centres would
  need the unit's first-minimum tie rule reproduced exactly).
- The scalers (preprocessing/host, StandardScaler / MinMaxScaler) and resample host paths are lane
  cpu's `host_parallelize` conversion (lane/cpu); measure them after it merges.
