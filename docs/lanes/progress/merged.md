# lane/merged: integration branch progress

Andrew, 2026-09-28: stop per-branch gating. Every finished lane branch is
merged into lane/merged, checked once globally, and merged into main only
if clean. Source branches are never modified; lane/merged only merges FROM
them.

Base: origin/main at 9a8f9e390 (branch created off it).

## Merge log

One `git merge --no-ff` per branch. Where a local lane ref was ahead of its
origin ref (local merged origin/main but was not pushed), the local ref was
merged; every such local ref was a strict superset of its origin ref.


### lane/cpu, lane/algos-linear, lane/linear-cpu

- lane/cpu: merged clean. It removed core/host_fp_env.mojo; that code is now
  part of core/host_parallel.mojo (`host_parallelize`).
- lane/algos-linear: merged clean (team fits, x_linear/team.mojo and tops.mojo).
- lane/linear-cpu: 24 conflicted files, resolved in 0ee6ba194. Both sides
  claim the pass-1 bits. The rule for every fit both sides rewrote: ONE entry
  point that carries the team `t` and the host scratch, with
  `comptime if is_gpu()`. The device runs the algos-linear team schedule and
  the host runs linear-cpu's map-then-fold schedule.
  - lbfgs.mojo: the Objective is now (t, x, y, n, d, ip, fp, theta, toff,
    grad, goff, sc), and lbfgs takes both t and sc. Callers: huber_fit
    (sc = fw + lbfgs_work(p)) and logcv_fit (sc = cptr + 1, work moved past
    the n*(K'+1) scratch).
  - huber, logcv (logistic_objective), glm (_objective), cd (_prep):
    `_team` (HEAD) and `_host` (linear-cpu) bodies behind one dispatcher.
  - ridge, quantile: the whole fit is split into `_team` and `_host`.
    ridge's final solve was the same in both, so it is factored into
    `_ridge_solve_best`.
  - glm_fit (gradient/Hessian block), bayes_ridge_fit and ard_fit (means,
    Gram, X'y), lars_fit (the same), enetcv_fit (the per-l1 path and its
    held-out errors), logcv_fit (the held-out hit map; the host version maps
    the hit flags with par_rows into team row 0), sgd_fit (problems on
    threads vs par_rows units): only the row-pass block is split. The lead's
    algebra is shared. bayes `_t_sse` gains sc and folds on the host through
    `_sse`/`_wsse`.
  - ops.mojo: the union of both sides' imports.
  - python/mojolearn/_expansion_linear.py: every fw size is linear-cpu's,
    which is a superset. The device receives the host's scratch and does not
    use it. LogisticRegressionCV keeps HEAD's 5 iw words.
  - bench/x_linear_speed.py: both sides made the same fix; HEAD's spelling
    (ML) is kept.
  - Sabotage patches: e2e_device_fold and the opt_* patches were re-derived
    on the device (team) code. The redundant is_gpu import hunks were
    dropped because the resolved files import it. opt_sgd_weights takes
    linear-cpu's version, since sgd_one is shared. Every patch in
    x_linear/checks/sabotage (the seam ones included) passes
    `git apply --check`, applies, and passes `--check -R` on a scratch copy.
  - docs/lanes/progress/linear.md: both sections kept, plus a combined note.
  - Not done (behaviour change, needs its own identity check): par_rows is
    still serial. It is not yet threaded through core/host_parallel.mojo.
  - Checked: the x_linear host binding builds (`mojo build`, one core, via
    tools/mac_slot.py). The device (is_gpu) branches have NOT been compiled
    on any GPU target. The first NVIDIA, AMD or Apple build of
    _mojolearn_x_linear must show that they compile, and the lane's identity
    gate must show the bits did not move.

### Note on lane/algos-linear's ref (orchestrator, 2026-09-28)

lane/algos-linear was merged from the LOCAL ref eeff40047 before the
instruction to use origin refs arrived. eeff40047 = origin/lane/algos-linear
(6a0d37701) plus one local `Merge origin/main` whose remerge-diff is empty
(a clean auto-merge, no hand resolution), so the content is the same as
origin's merged with main. Every later branch is merged from its origin/ ref.

### lane/algos-neighbors, lane/neighbors-cpu

- origin/lane/algos-neighbors: clean.
- origin/lane/neighbors-cpu: 4 conflicted files.
  - core/knn_host_predict.mojo (4 sites), gaussian_process/host/gpr_oracle.mojo
    (1): neighbors-cpu's `sync_parallelize(...)` + `_ = <buf>^` lifetime pins vs
    lane/cpu's `host_parallelize(...)`. Kept host_parallelize (the one split,
    DEVIATION 5900) AND the lifetime pins. check_host_parallel_sites: PASS.
  - kernel_methods/host/km_host_oracle.mojo: algos-neighbors added the
    chi2 / additive-chi2 / cosine kernel paths before the dot; neighbors-cpu
    replaced `gemm_oracle` with `host_gemm_identical` for the dot. Kept both:
    the new kernel paths as algos-neighbors wrote them (cosine still on
    gemm_oracle, as it was gated), the plain dot on host_gemm_identical.
  - svm/host/smo_oracle.mojo: algos-neighbors added the KERNEL_PRECOMPUTED
    branch with a List workspace; neighbors-cpu moved the workspace to
    stack_allocation. Kept the stack workspace and the precomputed branch
    (its `_ = partials^` dropped: nothing to keep alive on the stack).

### lane/algos-prep2, lane/prep-cpu

- origin/lane/algos-prep2: IDENTITY_PATHS.md range collision. prep's second
  range was 200-209, which main gives to neural (with cluster's rows 200-201
  inside it; that overlap is main's own and was left as is). Renumbered prep
  to **220-229** (orchestrator's suggestion, next free after cpu's 210-219);
  row 200 (normal CDF / inverse, DEVIATION 5410) -> **220**; references fixed
  in checks/ndtri_check.mojo (2), checks/numerics.mojo, resample/checks/intervals.mojo,
  resample/NOT_IMPLEMENTED.tsv. Next free row is now 230.
  Seen, not changed: the neural files (training/maximize.mojo,
  training/checks/maximize_check.mojo, test_optim_maximize_seam.py,
  training/IDENTICAL_OPTIMIZER_CONTRACT.md, training/NOT_IMPLEMENTED.tsv)
  still say "IDENTITY_PATHS row 200" for maximize=, which main's table lists as
  row 202. Owner: neural (pre-existing on main, not from this integration).
- origin/lane/prep-cpu: tools/identity_lanes/prep.checks. Union: the ndtri
  arm (algos-prep2) + the six host arms (prep-cpu). Every listed patch
  passes `git apply --check` on the merged tree.

### lane/algos-decomp, lane/decomp-cpu

- origin/lane/algos-decomp:
  - tools/algos_lane_check.py: the per-arm ARM_TIMEOUT / HUNG handling
    (algos-decomp) wraps the subprocess run; main's "exit 1 with its JSON
    written under a sabotage, the diff decides" rule follows it. Both kept.
  - tools/test_lane_select.py kmeans_oracle pin: 75 (HEAD) vs 76
    (algos-decomp). Took 76 provisionally; every pin is recomputed with
    tools/lane_select.py after the LAST merge (the pins move with each
    family), see "test_lane_select pins" below.
- origin/lane/decomp-cpu:
  - tools/identity_lanes/decomp.checks: union (5320 householder, 5321 ALS-CG
    from algos-decomp; eleven host arms from decomp-cpu). Every listed patch
    passes `git apply --check`.
  - x_decomp/checks/dense_check.mojo: header "DEVIATIONS 5307-5309, 5320" +
    decomp-cpu's host QR / Jacobi SVD / eigh / LU equalities.
  - x_decomp/checks/fold_ew_check.mojo: import union (F32Ptr, FOLD_BLOCK,
    bidx, ew_cell).
  - docs/lanes/progress/decomp.md: both sections; the Isomap-at-10k-on-AMD
    OWED note from the unpushed local 08ba64ec6 re-added by hand.
  - RISK, for the global check: algos-decomp moved geqrf/orgqr/getrf onto
    the device in parallel steps (same cells as the serial routine) and
    decomp-cpu added fast host QR/Jacobi/LU equal to the serial replay; both
    claim the serial bits, so they must agree; dense_check and the decomp
    lanes prove it.
