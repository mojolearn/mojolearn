# lane/apple-fast-prep2: prep lanes not owned by `prep` or `gram` (PLAN.md item 10, the rest of xlane="prep")

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check
(binding x_prep: `bindings/build_x_prep.sh`, new file `x_prep/fastprep2.mojo`, 3 lines in
`x_prep/device.mojo`, Python `_expansion_prep.py` `_prep2_qselect` + SimpleImputer / RobustScaler fit).
Every switch is compiled under FAST + Apple only (`PREP2_FAST`) and defaults OFF; IDENTICAL compiles
the old code. Lanes: target-encoder, simple-imputer, robust-scaler, iterative-imputer (the others of the
family are already element-parallel, see "Not changed").

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `MOJOLEARN_X_PREP_FAST_TE_GLOBAL=1` | env, host at dispatch (`Prep2Switches`) | `x_prep/fastprep2.mojo` te_global_fast_kernel, dispatched in `prep2_fast_stage` | te_global by a 256-thread threadgroup per (fold, target) with a tree (count+sum, then squared deviations) |
| `MOJOLEARN_X_PREP_FAST_TE_ENC=1` | env | te_enc_fast_kernel | te_enc by a threadgroup per (fold, column, category, target) over the category's gathered bucket (needs BK and GB > 0, the default program; else the unit runs) |
| `MOJOLEARN_X_PREP_FAST_II_CONV=1` | env | ii_conv_fast_kernel | the inf-norm max over `ii_rowabs`' row sums by one threadgroup tree (a max is exact: the same word) |
| `MOJOLEARN_X_PREP_FAST_EIGH_BLOCK=1` | env | eigh_block_fast_kernel (m <= 32, else the unit) | the cyclic Jacobi of `eigh` on a 32-thread threadgroup per matrix: same sweeps, same rotation order and tests, each rotation's row/column updates strided over the threads, A and V in shared memory (each element the unit's own expression: the unit's words) |
| `MOJOLEARN_X_PREP_FAST_II_GRAM_TILE=1` | env | ii_gram_tile_kernel + ii_gram_tile_reduce_kernel (d <= 32, else the existing fold) | ii_gram as a grid of 512-row blocks, each tile of 128 rows x d columns loaded into shared memory once, every thread 4 of the d*d cells, partials reduced by a tree per cell (scratch: chunks x d*d words) |
| `MOJOLEARN_X_PREP_FAST_QSELECT=1` | env, read in Python on the FAST tier of a Metal binding (`_prep2_qselect`) | `_expansion_prep.py` SimpleImputer.fit (median), RobustScaler.fit; device `qselect_device` (the `quantile` stage with SELECT = 1 as its 8th parameter) | the median / the three quantiles by a device radix select over `dradix.mojo`'s keys (one strided key load, then per 8-bit pass a histogram per (column, chunk of 16384 rows) carrying both order statistics of every fraction, a pick per task, then the unit's lerp) instead of sort_cols + quantile; no sort, no n*d sorted scratch; the same order statistics, so the same words |

## Causes (file:line on origin/lane/apple-fast f5f61bde)

- target-encoder: `x_prep/target.mojo:21` te_global_unit, ONE thread per (fold, target) = 5 threads for
  cv=4, each walking the 1M rows twice (mean, then variance). `x_prep/target.mojo:249` te_enc_unit, ONE
  thread per (fold, column, category) walking that category's bucket: taxi's largest zone category is
  hundreds of thousands of rows on one thread while the other blocks idle. The buckets themselves
  (te_hist .. te_hscatter, 256 chunks per column) and te_gather are parallel already.
- iterative-imputer (32 columns, 100k rows, 10 rounds = up to 320 BayesianRidge fits in one program):
  `x_prep/eigh.mojo:23` eigh_unit, a 31 x 31 cyclic Jacobi on ONE thread per fit (465 rotations x ~124
  loads+stores per sweep, 6-10 sweeps: ~0.5M dependent memory ops per fit, the largest serial section
  of the program); `x_prep/fastred.mojo:282` ii_gram_fast_kernel, 1024 threadgroups each walking every
  row of two columns (2 n d^2 = 200M loads per fit; the tile reads n d = 3.2M once);
  `x_prep/iterative.mojo:378` ii_conv_unit, one thread over the 100k row sums once per round.
- simple-imputer (median) and robust-scaler: `_expansion_prep.py` SimpleImputer.fit / RobustScaler.fit
  stage sort_cols on every column (`x_prep/dradix.mojo` radix_sort_cols_device: load, 4 x (histogram,
  scan, random scatter), store, over two n*d key blocks plus the n*d sorted block: Istella 220 x 1M =
  1.76 GB of key scratch + 880 MB sorted) to read 1 or 3 order statistics per column (`quantile_unit`).

## Expected effect

te_global: 5 threads x 2M serial loads -> 5 x 256 threads; te_enc: the largest bucket's serial walk
-> a tree. eigh: ~8-30x fewer dependent steps per fit (32 threads, one barrier per rotation).
ii_gram tile: ~60x fewer global loads per fit. qselect: fewer passes and no scatter; the key copy
stays (880 MB on Istella; 1.76 GB + 880 MB before). Quality: te_* and eigh produce the unit's words
(independent elements, same order); ii_gram's sums are pairwise (never less accurate than row order);
qselect selects the same words. Expected digests: unchanged for QSELECT, TE_*, EIGH_BLOCK and II_CONV on
Apple FAST; II_GRAM_TILE may move the last bits.

## Keep rule

A switch becomes the FAST default when its arm is faster on the M3 and held-out quality stays within
FAST's run-to-run spread; then the env read goes and the arm is the code. The `prep2-*-ident-istella`
lines are the IDENTICAL baselines (AFC_ARM=ours) for the "FAST slower than IDENTICAL" test.

## Not changed (listed with shape and cost estimate)

- iterative-imputer `ii_br_unit` (`x_prep/iterative.mojo:131`): BayesianRidge's iterations on ONE
  thread, O(p^2) per iteration (p = 31: ~2k mul/div), up to 300 iterations, typically 5-30: ~0.1-0.5 ms
  per fit, ~30-150 ms per program. Block version: threads over the predictors, 5 tree reductions per
  iteration. Second to eigh.
- iterative-imputer `ii_sub_unit` (one thread, p^2 copies): ~1k moves per fit, negligible.
- iterative-imputer per-feature launches: 7 stages x 32 features x 10 rounds = 2240 launches in one
  program (~20-50 ms of launch latency on Metal); a fused per-feature kernel would need the Gram's grid
  inside it (a cooperative launch), not cheap.
- target-encoder categories='auto': `_fit_categories` -> `unique_cols_unit` (`x_prep/prims.mojo:238`),
  one thread per column walking the 1M sorted rows (5 threads on taxi, 8 on istella); ~30-60 ms. The
  helper is shared with the `prep` family's encoders (OneHot/Ordinal); the chunked run scan of
  `x_prep/labels.mojo` (uniq_count/scan/write, one column) generalised to (column, chunk) units would
  replace it, as a copy under a new name for TargetEncoder.
- target-encoder te_hsum (one thread per (column, category) over 256 chunks) and te_hstart: tiny.
- rfe: `_expansion_prep.py` RFE.fit is a host loop by construction (5 LogisticRegression fits for
  Istella's 220 -> 110 at step 0.1; the fits are x_linear's, not this lane's); each step re-uploads X
  through `_gather` (200k x 220 words) and reads coef_ back; `sqsum_cols_unit` one thread per column
  over the coef rows (1 x d: trivial). Candidate: one resident upload and a device column gather per
  step (x_prep's `x_prep_dev_put` store), ~5 x 176 MB of host->device copies saved.
- power-transformer: FAST runs the staged search (50 dependent rounds of pt_map over n*d + a
  threadgroup fold per column, `x_prep/device.mojo` OP_PT_FOLD); the speculated search (IDENTICAL's,
  `_pt_spec_depth`) is 3x fewer dependent rounds for 2.3x more element work, and its `pt_sfold_unit` is a
  one-thread fold per candidate (`x_prep/transform.mojo:662`): a FAST arm needs a pt_sfold threadgroup
  fold and `spec` enabled on FAST; expected gain small since FAST's folds are already trees and pt_map
  (an exp/log per element) dominates. Not written.
- kbins (quantile, linear), quantile-transformer, spline (quantile knots): sort every column (radix,
  parallel) then read order statistics; a select pays only for <= 4 fractions (kbins needs 17, the
  quantile transformer 1000), so these keep the sort.
- maxabs-scaler, normalizer, binarizer, poly-features, variance-threshold, standard-scaler
  (preprocessing/standard.mojo, chunked grid), spline / kbins transform: every stage is a threadgroup
  fold or one thread per element already; what remains is the host arena copy and the bus (n*d up and
  down per call), outside this lane.
- simple-imputer transform: `fill_unit` one thread per element; nothing serial.
