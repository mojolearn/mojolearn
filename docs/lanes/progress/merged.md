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

### lane/algos-cluster, lane/cluster-cpu

- origin/lane/algos-cluster: IDENTITY_PATHS.md. Its new row 202 (AP tie
  noise, `x_cluster/bodies.mojo::ap_noise_cell`, DEVIATION 5122) collided with
  neural's row 202 (maximize=). Renumbered to **230**, range **230-239**
  registered to cluster; next free row 240. No other file cited "row 202" for
  it (cluster.checks names the DEVIATION, not the row).
- origin/lane/cluster-cpu: mixture/host/gmm_host_oracle.mojo. Main split the
  E-step's Mahalanobis over components (host_parallelize); cluster-cpu
  splits over rows (host_cells) with host_gemm_oracle. Took cluster-cpu's
  row split (it states the old component split no longer changes anything).
- FIX AT THE ROOT (integration): check_host_parallel_sites FAILED after this
  point on resample/host/resample_host.mojo lines 454/687/797 — three
  `sync_parallelize` calls that algos-prep2's new resample paths added and
  that prep-cpu's rewrite (which moved the other two to host_parallelize)
  never saw. Now host_parallelize; the check PASSES.
- OWED (not done here, behaviour change needing its own check):
  cluster/host/host_cells.mojo and x_linear/ops.mojo::par_rows still run
  their tasks serially "until core/host_parallel.mojo is on main". It is on
  lane/merged now; threading them is the owning families' next step
  (cluster, linear).

### lane/algos-trees, lane/trees-cpu

- Both merged with no textual conflict (origin refs at merge time:
  algos-trees dc27fe36d, trees-cpu 390f24b03). algos-trees may get one more
  push (the M2 GBDT hist kernel fix); origin/lane/algos-trees is merged again
  right before the build.
- FIX AT THE ROOT (integration): trees-cpu's RF (ensemble/host/rf_oracle.mojo)
  and ExtraTrees (extratrees/impl/randomforest/randomforest.mojo) host fits
  imported `core.host_fp_env` (host_ieee_fp_enter/leave in each tree task),
  a file lane/cpu deleted; and they plus xtrees/ops.mojo called
  sync_parallelize directly (check_host_parallel_sites FAIL, 3 sites).
  Converted to host_parallelize (caller's environment in every task, which is
  what host_ieee_fp_enter installed), the same conversion lane/cpu made for
  linear; their comments now cite core/host_parallel.mojo. Check: PASS.

### lane/algos-ann-b, lane/ann-cpu

- origin/lane/algos-ann-b:
  - tools/dev_pod.sh: both sides fixed the same patch-sync bug (a file the
    previous patch added, tracked by the new base, deleted from the box).
    Kept main's form (reset, then remove old additions except what the new
    base tracks); ann's reorder is the same fix in another spelling.
  - tools/test_lane_select.py: both REMEASURED notes kept; pin provisional.
- origin/lane/ann-cpu: clean.

### lane/algos-sequence, lane/sequence-cpu

- origin/lane/algos-sequence: clean.
- origin/lane/sequence-cpu:
  - DEVIATION number collision: 5540 was algos-sequence's LR schedulers seam
    (sched_check.py, seam_5540_sched_exact_round.patch) and sequence-cpu's
    host GEMM vector cells (sequence/host_gemm.mojo). Renumbered the later
    one (host GEMM) to **5544**: IDENTITY_PATHS row 150 (DEVIATION and the
    biting-arms list), sequence/README.md (table row + header),
    sequence/checks/oracle.mojo, sequence/checks/seams_check.mojo (check
    "5544_host_gemm_vector"; PASS line now 27 seams), the patch renamed to
    sequence/checks/sabotage/seam_5544_host_gemm_unfused.patch (its comment
    too), tools/identity_lanes/sequence.checks, docs/lanes/progress/sequence.md.
    All 44 listed sequence patches apply.
  - tools/dev_pod.sh: the same sync fix again; main's form kept.
  - FOUND AFTER THE MERGE (global duplicate-row scan): algos-sequence's new
    IDENTITY_PATHS row 199 (forecasters, DEVIATIONS 5541-5543) duplicated
    cluster's row 199 on main. Renumbered to **240**, range **240-249**
    registered to sequence, next free row **250**; sequence/README.md fixed.
    After this, neither IDENTITY_PATHS rows nor any README DEVIATION table
    number repeats anywhere in the tree (scan of every `| NNNN |` row).

### lane/neural-cpu, lane/neural-cpu-threads, lane/algos-cnn, lane/metrics

- All four merged with no textual conflict.
- FIX AT THE ROOT (integration): algos-cnn (x_cnn/host/gemm_host.mojo 5
  sites, x_cnn/host/ops_host.mojo 7) and metrics (x_metrics/host/program.mojo
  1) called sync_parallelize directly; lane/cpu's rule routes every non-GBDT
  host split through host_parallelize (caller's FP environment, DEVIATION
  5900). Converted (import + call); check_host_parallel_sites PASS over
  1773 files. RISK: bits move only if a subnormal met a pool worker; the CPU
  column at MOJOLEARN_CPU_THREADS 1/3/default in the global check decides.

### test_lane_select pins (after the last merge)

Recomputed with tools/lane_select.py on the merged tree and on an archive of
origin/main 9a8f9e390, lane sets diffed:
- cluster/host/kmeans_oracle.mojo 75 (main) -> **83**: + ivf-filter,
  x-ann-cagra-filter, x-ann-refine-euclidean, x-ann-tsne-pca (algos-ann-b),
  resample-bca, resample-perm-samples, resample-unpaired (algos-prep2),
  x-decomp-umap-options (algos-decomp). None dropped.
- core/forest_host_predict.mojo 86 -> **88**: + bootstrap,
  metrics-classification (old lanes whose binding closure now includes the
  merged host modules). None dropped.
- gbdt_host_predict 51, forest_inference 50, neural_inference.py 41: unchanged.
- Registry: 504 lanes on lane/merged.

## Build (step 3)

- Every binding the 504 registered lanes run (75: GPU + host families) built
  on the central AMD box (gfx942) at ce54c30f9 except one:
  **_mojolearn_x_cluster** — `_ap_noise_kernel(m: Int)`; the pinned toolchain
  refuses Int as a device argument ("Int and UInt do not conform to
  DevicePassable"). Owner: algos-cluster (WIP 649b55f7b). FIXED on lane/merged
  (003ea19ba: Int64 for the call, same cells; e2e_device_fold_reversed.patch
  context follows). After it: 75/75 built on gfx942. NVIDIA (sm_89, nvc1):
  75/75 built at 003ea19ba. x_linear's merged team/host dispatch compiles on
  both.

## Global check (step 4): how it runs

- Exposed lanes = host_surface.covered_lanes(): **465** of 504 registered.
  The other 39 (par-* multi-GPU drivers, cross-val-folds, resample-bca,
  resample-perm-samples, resample-unpaired, resample-utils) have no CPU route;
  their CPU arms refuse by design ("no CPU implementation of the cooperative
  multi-GPU driver ..."). Not counted as failures. NOTE for prep: the four
  new resample-* lanes are not in host_surface's covered set.
- Driver: tools/merged_check/merged_check.py (untracked, synced to the boxes;
  reuses algos_lane_check's needed_bindings / build+stamp / run_arm /
  compare). clean = GPU arm once, CPU arm per thread setting, each CPU column
  diffed cell for cell with the GPU column.
- NVIDIA nvc1 (2x RTX 4090): clean in 4 resumable shards, CPU column at
  MOJOLEARN_CPU_THREADS=1, 3 and default; e2e sabotage sequence (46 family
  patch lines, tools/merged_check/sab_plan.tsv) in a separate tree
  (/root/mojolearn-merged-sab); tests job (test_host_surface,
  test_lane_select, every test_x_*_repeat at threads 1/3/default).
- AMD central (MI300X): clean in 4 shards (CPU default), tests job.
- Apple + do-amd (MI325X): 24 clean shards via apple_steward speed jobs at
  003ea19ba, each on m2pro, m3ultra-b, do-amd and one M4 (rotating
  m4pro-a/m4pro-b/m4-a); a build job first on each. m3ultra drained.

## Failures found (step 5)

- **x-decomp-* (every x_decomp estimator), GPU and CPU**: `TypeError: a
  bytes-like object is required` in `_expansion_decomp._M.from_input`
  (array.frombytes given the Array's 2-D float32 memoryview). Owner:
  algos-decomp (2f2c2891b, "an input reaches the x_decomp store in one copy").
  Not from the integration (_array/_buffer equal algos-decomp's). FIXED on
  lane/merged b90431acf (`mv.cast("B")`, same bytes, as _array.Array does);
  the decomp lanes are re-run after the shards.
- **kmeans, kmeans-random, kmeans-sqrt: CPU arm REFUSED** on x86 (Xeon 8470
  and EPYC): `transform(X)[i, labels_[i]] and transform(X).min(axis=1)
  differ: 77306 bytes of 80000`. Reproduced on a pure origin/lane/cluster-cpu
  tree (ec159b796): 10933 of 80000. Owner: cluster-cpu (host KMeans rewrite:
  8f9a54482 / a8cf410cf / d9a4453ec / 778f2e677). kmeans_oracle.mojo on
  lane/merged is byte-identical to cluster-cpu's. Fix in progress on
  lane/merged-kmeans-fix (root cause, then same bits as main's host path).

## The MI300X column is OWED (2026-09-28 ~09:10Z)

Hot Aisle terminated the central AMD box (MI300X) at about 09:10Z (account
balance -$54). Its lane/merged clean shards (0015-0018) and tests job (0019)
were lost mid-run. What they had recorded before the box went: shards 0 and 1
were running; shard 1 finished 126 lanes (every exposed lane AGREE except the
kmeans and x-decomp failures above). The AMD column of the global check is
do-amd (MI325X, gfx942) through the steward shards. **The MI300X column is
owed until Hot Aisle is topped up.**

## Seam arms (found by the first Apple/do-amd shards at 003ea19ba)

algos_lane_check runs every family's .checks seam arms at the clean stage and
stops the whole run at the first arm that fails, so the first 22 shard runs
reached no lane. What they found:
- **ann.checks:11 cagra_5820_prune_high_rank.patch did not apply** (16 runs):
  cut by algos-ann-b before ann-cpu moved CAGRA prune into host tasks
  (3cfc52318). INTEGRATION: re-cut on lane/merged (8ab426638), same edit.
- **trees.checks seam_5601_fused / seam_5603_libm_exp did not apply**: my own
  xtrees/ops.mojo import change (sync_parallelize -> host_parallelize) was in
  their context. INTEGRATION: fixed in 8ab426638. Every .checks patch of every
  family now passes `git apply --check`.
- **x_decomp/checks/sabotage/host_ew_onemsq_fused.patch NOT SEEN** by
  fold_ew_check.mojo on m3ultra-b and m4-a (Arm): the host `1 - x*x` spelled
  fused is not separated from the pinned unfused spelling there. Owner:
  decomp-cpu (its host arm). If the Arm host compiler contracts the pinned
  spelling, the decomp lanes' Apple CPU column will show it.
- **x_cluster/checks/sabotage/5103_kth.patch NOT SEEN** by kth_check.mojo on
  m4-a (4 runs). Owner: algos-cluster.
The Apple/do-amd shards were withdrawn (70 queued copies, my own) and
resubmitted at the lane/merged tip with the seam step skipped (lane verdicts
only; seam arms are not part of the global check's lane comparison), one lane
at a time so one failing arm cannot stop a shard.

### kmeans CPU fix (merged from lane/merged-kmeans-fix, d57f6b123)

Root cause: `cluster/host/kmeans_oracle.mojo::host_kmeans_transform` (the
row-task form from lane/cluster-cpu) read `x_norm` / `c_norm` through
untracked pointers after their last tracked use, so Mojo freed both lists
before `host_cells(_row, ...)` ran and every transform cell read freed memory
(labels_/predict were fine: host_assign holds its norms as arguments). Fix:
`_ = x_norm^` and `_ = c_norm^` after the join (no arithmetic change). Evidence
(nvc1 EPYC CPU arm): kmeans, kmeans-random, kmeans-sqrt, kmeans-array,
kmeans-classic-pp, kmeans-weighted pass at MOJOLEARN_CPU_THREADS 1/3/default,
and `--diff` against origin/main's CPU build is IDENTICAL on every cell;
removing the two lines brings back the exact 59264/80000 refusal. Owner of the
bug: cluster-cpu. Open: no kmeans host sabotage switch is wired into
cluster.checks.
