# Apple FAST, next pass (updated 2026-10-02 ~19:30Z by the M3 manager session)

Classical ML and trees only. FAST is the Apple GPU tier: faster, quality held, GPU only.

## How this pass works

- **You write code. You never run anything.** No builds, tests, timing or identity runs. The M3 manager session builds every
  branch, measures on the M3 Ultra, merges winners into main and publishes results.
- **Base: `origin/main`.** Main now carries all Oct 2 FAST work that won (merge 269ffa57a) plus the host-route-removal
  (hr2-*) lanes. Branches cut from `lane/apple-fast` must first `git merge origin/main` (no rebase) and resolve conflicts:
  main's IDENTICAL behavior and its host-route removals win; the FAST change is re-expressed on top of them.
- One branch per family, `lane/apple-fast-<family>` (required prefix; the M3 queue refuses others). Never push to `main` or
  `lane/apple-fast`. Never `--no-verify`. The draft PRs #132-#149 target `lane/apple-fast`; ignore them, the request files
  are what the M3 reads.
- Each change: default OFF, compiled only under FAST on Apple
  (`comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`), IDENTICAL compiles main's code
  unchanged. Prefer a `-D MOJOLEARN_<NAME>` define to an env switch (an env read is a host step; switches are deleted once
  they win). Mojo pointer types: pass `MutPointer[T, MutAnyOrigin]` explicitly (the tier branch failed on an origin-typed
  `unsafe_ptr()` argument); read docs/apple-fast/m3/build-errors.txt for the M3 compiler's exact messages.
- **Request runs** with `docs/apple-fast/ab/<family>.txt`, one line per comparison:

      CMD lane/apple-fast-<family> <tag> AFC_FAMILY=<algos|classical|classical2> bash tools/afc_ab.sh <tag> <lane> <dataset> 1 2 - <ENV=1>
      CMD lane/apple-fast-<family> <tag> AFC_FAMILY=<family> bash tools/afc_ab_def.sh <tag> <binding> <lane> <dataset> 1 2 "" "-D <DEFINE>"
      CMD lane/apple-fast-<family> <tag> AFT_OUT=$HOME/aft-ab/<tag> bash tools/aft_ab.sh <binding> <lane> <dataset> 2 "" "-D <DEFINE>"

  LIGHT A/Bs: old FAST vs new FAST, one alternation, board-size data. No `-ident` lines (skipped). One dataset per change
  first; the second only after the first wins. One tag per lane x dataset (never loop lanes under one tag). Opponents are never
  re-run; their times come from the stored board.
- **Read results** on `lane/apple-fast-results`: `docs/apple-fast/m3/results.txt`, `queue.txt`, `build-errors.txt`
  (refreshes every 10 min). After every push, check build-errors first: a branch that does not compile yields nothing.
- Keep rule: faster on the M3 and held-out quality within FAST run-to-run spread -> FAST default (switch removed) -> main.

## Same bits

The cloud box has no Apple GPU and no Mojo toolchain: no runtime identity checks there. Do the static check and say so in each
commit message: every changed line is inside a FAST + Apple guard (or a FAST-only Python branch) and nothing IDENTICAL compiles
changes. The M3 manager runs the real check before any merge: IDENTICAL Metal model hash on the branch vs main, lane by lane
(`tools/aft_idcheck.sh`, `AFT_ID_HOST=0`).

## FAST is GPU only

No host compute in a FAST fit/transform/predict: no host loops over rows, no NumPy/sklearn math in the Python layer, no
device-to-host round trips inside an iteration, no one-thread/one-block launches over runtime sizes, no env reads on the hot
path. The pre-push hook `tools/hooks/no_host_routes.py` refuses new findings; never edit its baseline. Existing debt rows
(`tools/hooks/host_routes_baseline.tsv`) are assigned to sub-lanes by another session's plan, `docs/plans/cpu-gpu-cleanup/`
on `origin/claude/lucid-ride-74ce6a`: take a debt row only if it is on a FAST classical/trees path and its file is not
assigned there.

## Done and on main (M3 FAST, before -> after)

LassoCV/ElasticNetCV istella 70.8/75.1 s -> 0.56/0.58 s, taxi 3.1 s -> 0.10 s (now 2-12x faster than sklearn, r2 equal or
better); YetiRank istellarank 55.4 s -> 5.6 s; Lossguide exact batches width 64 (taxi 78 -> 14.9 s, istella 102 -> 20.7 s);
k-NN classifier istella 864 -> 290 ms; ARD istella 7.1 -> 0.86 s; ExtraTrees partition rows per thread (-21% taxi, -5%
istella); RF 8-column histogram tiles (-1%); iforest row-major upload + device finite scan; NearestCentroid chunked stats.
Dropped after A/B: GBDT SM_X4/SM_X8, Lossguide width 16, RF small-node / node-split / batch16k, ET TPB 256, ET batch 65536.

## Unfinished work (take these first)

1. **Merge `origin/main` into every family branch and resolve.** Test merges against main conflict in:
   ann (x_ann/knn_device, tsne_device), cluster and cluster2 (x_cluster/device_ops, host/host_ops, ops; cluster2 also bisect,
   optics), core (bindings/_mojolearn_estimators), decomp-linalg (x_decomp/device, python/_linalg_impl), decomp-sparse
   (x_decomp/device, python/_expansion_decomp), gram (x_linear/device, lars, ridge), isotonic-knn (x_neighbors/iter_device,
   python/_expansion_decomp), kernel (kernel_methods/estimator, x_linear/bayes, x_linear/device), linear
   (bindings/_mojolearn_solver, glm/impl/qn/glm_base, python/_solver_impl), neighbors2 (x_neighbors bindings + iter_device,
   python/_surface_neighbors), prep (preprocessing binding/estimator, python/_expansion_prep, preprocessing), prep2
   (python/_expansion_prep), resample (resample/estimator), trees-depthwise (gbdt/train), trees-ensembles (xtrees/api),
   trees-io (isolation_forest/impl/isolation_forest). Clean: pca-eig, seq, tier, trees-scan, trees-yeti, tsa.
   The M3 measures each branch at its head, so a merge commit is needed before a winner can land on main anyway.
2. **`core` WIP (20b2cda13)**: an unfinished FAST change in glm/estimator.mojo (+253), bindings/_mojolearn_estimators.mojo,
   python/mojolearn/linear_model.py. Finish it behind a define, then add its lines to `docs/apple-fast/ab/core.txt`
   (the manager already queued A/Bs for core's four finished switches: dbscan scan/cc-batch, kde slices, knn k64, kmeans x3).
3. **`tsa` WIP (6c72cf77a)**: arima batched_kalman (+234), new arima/impl/fast_eval_ws.mojo (+226). Finish, add
   `docs/apple-fast/ab/tsa.txt` (arima lanes on taxi-hourly).
4. **`trees-symmetric`**: empty. SymmetricTree (CatBoost-style) FAST is open: depth-wise oblivious splits on the device.
5. **BayesianRidge FAST on main's grid path**: the merge rebuilt the Gram sse shortcut in x_linear/device.mojo
   (`bayes_yy_parts_kernel`, `bayes_step_gram_kernel`). The same shortcut went NaN on istella in the old path
   (y'y - 2w'X'y + w'Gw cancels in f32 along near-null directions). `lane/apple-fast-bayes` (79f7ace5f) guarded the old path
   with a delta form and an error bound that falls back to a row pass: port that guard to the grid kernels. M3 check
   `br-main-q` is queued. sklearn itself is r2 -426,163 on istella: the bar is finite and no worse than the row-pass arm.
6. **Overlaps to settle** (from the peer's STATUS.md): trees-scan and trees-depthwise both parallelise
   `launch_scan_vector_u32` (gbdt/gpu_util/kernel/scan.mojo) under different defines, so keep one; gram and kernel both edit
   x_linear/device.mojo; cluster and cluster2 share x_cluster files; prep/prep2 and isotonic-knn/decomp-sparse share Python
   expansion files. Whoever merges second resolves.
7. **Depthwise** (taxi ~14.9 s vs XGBoost 9.9 s): fused chain and one wait per level are on `lane/apple-fast-depthwise`
   (A/Bs queued). Next: one wait per tree (leaf selection, split records and sizes on the device across levels).
8. **YetiRank** (5.6 s): bitonic + merge-path sort on `lane/apple-fast-yetirank` (A/B queued); next tree_search and
   sym.hist. Coordinate with trees-yeti (TASK16K kernel, symmetric histogram block).

## Gaps with no branch yet (M3 0.8.34 board; x = times slower than the best opponent)

- feature selection: select-f-regression, select-r-regression, select-f-classif (taxi ~3.1-3.3x)
- kernel approximation: skewed-chi2 taxi 4.5x, additive-chi2 istella 2.9x, gaussian-rp taxi 3.7x / istella 2.4x
  (decomp-sparse touches gaussian-rp: check it first)
- calibrated taxi 2.4x, multioutput-reg taxi 2.1x
- trees: dart / dart-reg istella ~2x vs LightGBM; gbdt-categorical taxi 1.35x (CTR stages; trees-depthwise touches CTR prep)
- lstsq istella 1.9x (decomp-linalg has a tiled lstsq arm), damped-ets taxi-hourly 3.3x, select-d 2.1x

The full table is `docs/apple-fast/m3-gaps-0834.tsv` (ratio first). Families with queued A/Bs are not to be restarted;
wait for their results.
