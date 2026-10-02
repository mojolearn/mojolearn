# Apple FAST, next pass (written 2026-10-02 by the M3 manager session)

Classical ML and trees only. FAST is the Apple GPU tier: it must be faster, quality must hold, and it runs on the GPU only.

## How this pass works

- **You write code. You never run anything.** No builds, tests, timing or identity runs. The M3 manager session builds every
  branch, runs every measurement on the M3 Ultra, merges winners into main, and publishes results.
- Base every branch on `origin/main` (main now carries all of the Oct 2 FAST work). One branch per family:
  `lane/apple-fast-<family>`. The prefix is required; the M3 queue refuses other names. Never push to `main` or `lane/apple-fast`.
- Each change sits behind a switch that defaults OFF and compiles only under FAST on Apple
  (`comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`), so IDENTICAL compiles exactly main's
  code. A `-D MOJOLEARN_<NAME>` define is preferred over an env switch: an env read is a host step, and every switch is deleted
  once it wins anyway.
- **Request runs** with `docs/apple-fast/ab/<family>.txt` on your branch, one line per comparison:

      CMD lane/apple-fast-<family> <tag> AFC_FAMILY=<algos|classical|classical2> bash tools/afc_ab.sh <tag> <lane> <dataset> 1 2 - <ENV=1>
      CMD lane/apple-fast-<family> <tag> AFC_FAMILY=<family> bash tools/afc_ab_def.sh <tag> <binding> <lane> <dataset> 1 2 "" "-D <DEFINE>"
      CMD lane/apple-fast-<family> <tag> AFT_OUT=$HOME/aft-ab/<tag> bash tools/aft_ab.sh <binding> <lane> <dataset> 2 "" "-D <DEFINE>"

  These are LIGHT A/Bs: old FAST against new FAST, one alternation, board-size data. Do not add `-ident` baseline lines
  (they are skipped). One dataset per change first (istella, or taxi where the change targets taxi's shape); add the second
  only after the first wins. One tag per lane x dataset (do not loop several lanes under one tag).
- **Read results** on branch `lane/apple-fast-results`: `docs/apple-fast/m3/results.txt` (every result line),
  `docs/apple-fast/m3/queue.txt` (queue with positions), `docs/apple-fast/m3/build-errors.txt` (compile errors from the M3,
  check it after every push: a branch that does not compile produces no numbers). Refreshes every 10 minutes.
- Keep rule: a change becomes the FAST default when the M3 A/B is faster and held-out quality stays within FAST run-to-run
  spread. Then the switch is removed and the arm is the code.

## Same bits: what you can and cannot check here

The cloud box has no Apple GPU and no Mojo toolchain, so it cannot run identity checks. What you can do, and must do, is the
static check: every changed line is inside a FAST + Apple comptime guard (or a FAST-only Python branch), and nothing the
IDENTICAL build compiles changes. Say so in the commit message. The M3 manager runs the real check before merging: the
IDENTICAL Metal model hash on your branch against main (`tools/aft_idcheck.sh`, `AFT_ID_HOST=0`), lane by lane.

## FAST runs on the GPU only

No host compute inside a FAST fit/transform/predict: no host loops over rows, no NumPy/sklearn math in the Python layer, no
device-to-host round trips inside an iteration, no one-thread/one-block launches over runtime sizes, no env reads in kernels'
callers. The pre-push hook `tools/hooks/no_host_routes.py` refuses new findings (never `--no-verify`, never edit its baseline).
Existing debt rows are in `tools/hooks/host_routes_baseline.tsv`; another session's plan `docs/plans/cpu-gpu-cleanup/` (branch
`origin/claude/lucid-ride-74ce6a`) assigns that debt to sub-lanes. Take a debt row only if it sits on a FAST classical/trees path
and its file is not assigned there; otherwise leave it to that plan.

## Unfinished work (take these first)

1. **`lane/apple-fast-tier` does not compile in FAST**: `core/gemm.mojo:411` passes `z.unsafe_ptr()` (origin-typed) to
   `_apple_fast_gemm_nt_tiled`, which takes `MutPointer[Float32, MutAnyOrigin]`. Fix the signature or cast; its 10 queued
   A/Bs (pca/ols/kmeans GEMM arms, rbf/nystroem pinned, theta) produce nothing until then.
2. **BayesianRidge FAST on the grid path**: the main merge rebuilt the Gram sse shortcut on main's new grid driver
   (`x_linear/device.mojo` `bayes_yy_parts_kernel` / `bayes_step_gram_kernel`). The same shortcut gave NaN on istella in the
   old path (sse = y'y - 2w'X'y + w'Gw cancels in f32 on near-null directions). `lane/apple-fast-bayes` (79f7ace5f) fixed the
   old path with a delta form and an error bound that falls back to a row pass; port that guard to the grid kernels.
   M3 check `br-main-q` will show whether main's FAST BayesianRidge istella is finite. Note: sklearn's own BayesianRidge on
   istella is r2 -426,163, so the bar there is finite and no worse than the row-pass arm, not a good r2.
3. **Depthwise** (FAST taxi ~14.9 s vs XGBoost 9.9 s): `lane/apple-fast-depthwise` fused the split chain
   (`-D MOJOLEARN_GBDT_DW_FUSED_CHAIN`) and cut host waits to one per level (`-D MOJOLEARN_GBDT_DW_NO_LEVEL_SYNC`); A/Bs
   queued. Remaining: one wait per tree (leaf selection + split records + sizes entirely on the device across levels).
4. **YetiRank**: 5.6 s on istellarank now. `lane/apple-fast-yetirank` adds a bitonic + merge-path sort
   (`-D MOJOLEARN_YETI_FAST_SORT`, A/B queued). Next: tree_search and sym.hist stages.
5. **`lane/apple-fast-tsa`** has code and no request file. Add `docs/apple-fast/ab/tsa.txt`.
6. **`tools/afc_ab.sh` summary fix** (one tag looping over lanes pooled every lane's runs) is on main; branches cut before it
   carry the old copy. Merge `origin/main` into your branch.

## Gaps with no branch yet (M3 0.8.34 board, FAST vs best opponent; ratio = how many times slower we are)

- feature selection: `select-f-regression`, `select-r-regression`, `select-f-classif` taxi (~3.1-3.3x)
- kernel approximation: `skewed-chi2` taxi 4.5x, `additive-chi2` istella 2.9x, `gaussian-rp` taxi 3.7x / istella 2.4x
- `calibrated` taxi 2.4x, `multioutput-reg` taxi 2.1x
- trees: `dart` / `dart-reg` istella ~2x vs LightGBM; `gbdt-categorical` taxi 1.35x (CTR stages)
- `lstsq` istella 1.9x (also touched by decomp-linalg), `damped-ets` taxi-hourly 3.3x, `select-d` 2.1x

Everything else on the gap list (`docs/apple-fast/m3-gaps-0834.tsv`, ratio first) already has a branch with A/Bs queued:
huber/BayesianRidge/ARD (kernel), isotonic/knn-imputer/LLE (isotonic-knn), connected-components/meanshift/minibatch-kmeans
(cluster), label/onehot/ordinal/minmax/multilabel encoders (prep), lars/lasso-lars/ridge-clf/lda/qda (gram), pca/ipca
(pca-eig), knn family (core, neighbors2). Do not start those again; wait for their results on the results branch.
