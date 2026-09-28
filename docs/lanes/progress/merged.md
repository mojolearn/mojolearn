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
