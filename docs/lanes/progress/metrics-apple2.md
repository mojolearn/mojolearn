# metrics-apple2: progress

Lane `metrics-apple2` (Apple speed round 2, ~/mojolearn-evidence/apple2_speed_brief.md):
the metrics family (the 24 metrics, CV splitters, model_selection), IDENTICAL
and FAST. Worktree `~/mojolearn-wt/metrics-apple2`, branch
`lane/metrics-apple2`, forked from lane/apple-merged 037daa353. Round 1:
docs/lanes/progress/metrics-apple.md.

## How it is measured
`bench/x_metrics_apple_ab.sh <base sha> [reps] [profile] [only] [tests]`, one
steward speed job per step: the base commit in a temporary worktree (sharing
the job worktree's pixi env and prebuilt bindings) and the head, x_metrics
built in both modes in both trees, then the board (bench/x_metrics_speed.py,
1M rows, taxi + HIGGS from R2) base/head IDENTICAL, base/head FAST, then
reversed. Before the timing, `tools/apple_speed_metrics/eq_cases.py` prints a
digest per case (KFold/StratifiedKFold/ShuffleSplit/train_test_split/
RepeatedKFold at 7 sizes, roc_curve/AUC/max_fpr/AP/PR/DET/ovr/ovo curves with
ties and weights) in both trees and both modes (XMAB-EQ SAME required), and
`tools/apple_speed_metrics/*_words.mojo` checks the new units word for word
(planned vs sequential, host 1/2/4/8 tasks and device).

## Changes (all meant to keep IDENTICAL bits; FAST runs the same code)
1. Declared output ranges (8b2075c22). `_Prog.want(off, n)`; a program that
   declares outputs calls the new `x_metrics_run_out`, and the device
   downloads only those ranges (not the inputs, order slots, thresholds an
   AUC never reads). Every read (`floats`, `ints`, `words`, `get`, keep
   bytes) refuses an undeclared word on every backend, so a missing
   declaration fails loudly instead of reading zeros.
2. `fold_rows` (op 36; planned as fr_scatter / fr_cnt / fr_off / fr_fill,
   ops 37 to 40) and `rows64` (op 41) (8b2075c22). KFold(shuffle) and
   StratifiedKFold get every fold's ascending (test, train) rows as Int64
   words in one program; ShuffleSplit (and so train_test_split) gets its
   permutations as Int64 words. No Python int is made per row (round 1's
   `_rows_of` lever).
3. AUC / AP curves skip the thresholds and (AP) the keep flags (8b2075c22).
4. The curve epilogue in host binary64 (5b330a3ff): `x_metrics/epilogue.mojo`
   (`x_metrics_curve_auc`, `_ap`, `_roc`) runs the same binary64 operations
   as `_drop_collinear`, `_trapezoid`, `_binary_auc`, `_binary_ap` and
   roc_curve over the arena words (products pinned, CPython's fsum step for
   step), so no Python float is made per curve point. Python keeps its path
   for empty classes, zero denominators, `MOJOLEARN_HOTPATH=python` and an
   older binary.

5. Native expected MI (25ab062db): x_metrics/epilogue.mojo expected_mi,
   with portable_log_c = packaging/portable_math/portable_math.c
   mojolearn_log operation for operation (products pinned, fm = fma).
6. Native multiclass row-sum check (459a122b7): row_sum_range.
7. Count-bounded downloads (quads [lo, hi, CNT, mult]) and device
   compaction of the kept curve points (bin_curve params 12, 13; ops
   ck_cnt / ck_off / ck_fill 42 to 44) for the AUCs and roc_curve
   (e22d0ba22).
8. StratifiedKFold in one device program: group_sort, the class
   permutations, strat_codes (op 45), fold_rows (cf20f1062).

SHARED CODE: none outside x_metrics (the bindings `_mojolearn_x_metrics` and
`_mojolearn_x_metrics_host` gain exports; `_surface_metrics.py` lists them;
`tools/lane_select.py` gains `tools/apple_speed_metrics/` as measurement tooling).

## Results
### Step 1: m4pro-b, steward 1790603676019 (head 2035801320 = changes 1 to 4 + skip-inputs, base 037daa353)
Evidence: ~/mojolearn-evidence/metrics-apple2/1790603676019-2035801320.txt.
XMAB-EQ identical SAME and fast SAME (473 cases); WORDS PASS (fold_rows +
rows64, planned == sequential, host 1/2/4/8 tasks and device, 0 words
differ). Board totals (29 cases, `--reps 2`, min of the two passes):
IDENTICAL base 2.306 s* / head 1.116 s; FAST base 2.323 s* / head 1.089 s.
*The base tree lacked libMojolearnMath.dylib (the A/B script copied only
.so files; fixed in c695c9ebb), so 3 base cases FAILED and are missing
from the base total (round 1 had them at ~0.33 s on this Mac, so the base
is ~2.64 s). Their head digests equal round 1's (1790587224015, m4pro-b):
matthews 24ce1cfb, AMI 5a9f3918, davies_bouldin a067ed0a. Every other
case digest is equal base vs head in both modes.

| case | IDENTICAL base | head | x | FAST base | head | digest |
|---|---|---|---|---|---|---|
| median_absolute_error | 0.0163 | 0.0093 | 1.8x | 0.0162 | 0.0093 | equal |
| median_absolute_error_w | 0.0284 | 0.0200 | 1.4x | 0.0228 | 0.0144 | equal |
| mean_pinball_loss | 0.0093 | 0.0039 | 2.4x | 0.0090 | 0.0039 | equal |
| explained_variance_score | 0.0339 | 0.0131 | 2.6x | 0.0334 | 0.0124 | equal |
| mean_tweedie_deviance | 0.0104 | 0.0049 | 2.1x | 0.0109 | 0.0049 | equal |
| max_error | 0.0049 | 0.0032 | 1.5x | 0.0049 | 0.0029 | equal |
| d2_absolute_error_score | 0.0260 | 0.0147 | 1.8x | 0.0264 | 0.0147 | equal |
| r2_score_w_mo | 0.0775 | 0.0252 | 3.1x | 0.0770 | 0.0250 | equal |
| mean_squared_log_error | 0.0111 | 0.0053 | 2.1x | 0.0103 | 0.0053 | equal |
| balanced_accuracy_score | 0.0360 | 0.0251 | 1.4x | 0.0356 | 0.0244 | equal |
| matthews_corrcoef | FAILED* | 0.0254 | - | FAILED* | 0.0240 | equal |
| cohen_kappa_score | 0.0282 | 0.0233 | 1.2x | 0.0282 | 0.0242 | equal |
| prfs_w | 0.0428 | 0.0292 | 1.5x | 0.0431 | 0.0288 | equal |
| hamming_loss | 0.0314 | 0.0236 | 1.3x | 0.0304 | 0.0243 | equal |
| roc_curve | 0.1322 | 0.0261 | 5.1x | 0.1336 | 0.0264 | equal |
| roc_curve_w | 0.2290 | 0.0482 | 4.8x | 0.2237 | 0.0372 | equal |
| average_precision_score | 0.1264 | 0.0249 | 5.1x | 0.1266 | 0.0250 | equal |
| roc_auc_max_fpr | 0.1233 | 0.0262 | 4.7x | 0.1239 | 0.0259 | equal |
| roc_auc_ovr | 0.5326 | 0.2102 | 2.5x | 0.5314 | 0.2090 | equal |
| log_loss_w | 0.0402 | 0.0202 | 2.0x | 0.0397 | 0.0206 | equal |
| brier_score_loss | 0.0222 | 0.0123 | 1.8x | 0.0222 | 0.0125 | equal |
| top_k_accuracy_score | 0.0318 | 0.0170 | 1.9x | 0.0315 | 0.0170 | equal |
| adjusted_mutual_info_score | FAILED* | 0.2472 | - | FAILED* | 0.2459 | equal |
| calinski_harabasz_score | 0.0431 | 0.0240 | 1.8x | 0.0468 | 0.0245 | equal |
| davies_bouldin_score | FAILED* | 0.0241 | - | FAILED* | 0.0239 | equal |
| kfold_shuffle | 0.2262 | 0.0272 | 8.3x | 0.2307 | 0.0273 | equal |
| stratified_kfold_shuffle | 0.2739 | 0.1136 | 2.4x | 0.2758 | 0.1032 | equal |
| shuffle_split | 0.1311 | 0.0518 | 2.5x | 0.1309 | 0.0524 | equal |
| train_test_split | 0.0283 | 0.0108 | 2.6x | 0.0275 | 0.0108 | equal |


### Step 3: m4pro-a, steward 1790619894321 (head ee2576390 = changes 1 to 8 + origin/main merged, base 037daa353)
Evidence: ~/mojolearn-evidence/metrics-apple2/1790619894321-ee2576390.txt.
XMAB-EQ identical SAME and fast SAME (607 cases, incl. 100 AMI, rare-class
StratifiedKFold, RepeatedStratifiedKFold, ovr micro/None, row-sum edge
cases); WORDS PASS for fold_rows + rows64 and for the curve compaction
(host 1/2/4/8 tasks and device, 0 words differ). No case FAILED; every
case digest equal base vs head in both modes.

Board totals (29 cases, `--reps 2`, the two passes):
IDENTICAL base 2.595 / 2.602 s, head 0.968 / 0.993 s (2.7x);
FAST base 2.604 / 2.619 s, head 0.953 / 0.957 s (2.7x).

| case | IDENTICAL base | head | x | FAST base | head | digest |
|---|---|---|---|---|---|---|
| median_absolute_error | 0.0164 | 0.0095 | 1.7x | 0.0165 | 0.0094 | equal |
| median_absolute_error_w | 0.0280 | 0.0192 | 1.5x | 0.0223 | 0.0138 | equal |
| mean_pinball_loss | 0.0101 | 0.0038 | 2.7x | 0.0102 | 0.0038 | equal |
| explained_variance_score | 0.0339 | 0.0120 | 2.8x | 0.0337 | 0.0119 | equal |
| mean_tweedie_deviance | 0.0110 | 0.0047 | 2.3x | 0.0111 | 0.0047 | equal |
| max_error | 0.0050 | 0.0026 | 1.9x | 0.0053 | 0.0026 | equal |
| d2_absolute_error_score | 0.0295 | 0.0150 | 2.0x | 0.0294 | 0.0150 | equal |
| r2_score_w_mo | 0.0667 | 0.0253 | 2.6x | 0.0664 | 0.0252 | equal |
| mean_squared_log_error | 0.0116 | 0.0051 | 2.3x | 0.0116 | 0.0051 | equal |
| balanced_accuracy_score | 0.0303 | 0.0233 | 1.3x | 0.0305 | 0.0233 | equal |
| matthews_corrcoef | 0.0305 | 0.0234 | 1.3x | 0.0307 | 0.0231 | equal |
| cohen_kappa_score | 0.0263 | 0.0217 | 1.2x | 0.0265 | 0.0216 | equal |
| prfs_w | 0.0413 | 0.0292 | 1.4x | 0.0412 | 0.0291 | equal |
| hamming_loss | 0.0305 | 0.0233 | 1.3x | 0.0301 | 0.0231 | equal |
| roc_curve | 0.1321 | 0.0251 | 5.3x | 0.1326 | 0.0245 | equal |
| roc_curve_w | 0.2231 | 0.0427 | 5.2x | 0.2187 | 0.0368 | equal |
| average_precision_score | 0.1257 | 0.0249 | 5.0x | 0.1242 | 0.0252 | equal |
| roc_auc_max_fpr | 0.1214 | 0.0262 | 4.6x | 0.1223 | 0.0264 | equal |
| roc_auc_ovr | 0.5305 | 0.2272 | 2.3x | 0.5264 | 0.2281 | equal |
| log_loss_w | 0.0352 | 0.0201 | 1.8x | 0.0354 | 0.0198 | equal |
| brier_score_loss | 0.0201 | 0.0120 | 1.7x | 0.0201 | 0.0121 | equal |
| top_k_accuracy_score | 0.0290 | 0.0163 | 1.8x | 0.0290 | 0.0167 | equal |
| adjusted_mutual_info_score | 0.2562 | 0.1350 | 1.9x | 0.2540 | 0.1365 | equal |
| calinski_harabasz_score | 0.0439 | 0.0241 | 1.8x | 0.0443 | 0.0239 | equal |
| davies_bouldin_score | 0.0441 | 0.0241 | 1.8x | 0.0446 | 0.0239 | equal |
| kfold_shuffle | 0.2225 | 0.0279 | 8.0x | 0.2235 | 0.0277 | equal |
| stratified_kfold_shuffle | 0.2679 | 0.0683 | 3.9x | 0.2690 | 0.0685 | equal |
| shuffle_split | 0.1344 | 0.0541 | 2.5x | 0.1362 | 0.0538 | equal |
| train_test_split | 0.0285 | 0.0111 | 2.6x | 0.0290 | 0.0113 | equal |

Note: roc_auc_ovr is 0.210 s at step 1 (m4pro-b) and 0.227 s at step 3
(m4pro-a), different Macs; the compaction's own effect on ovr was not
separated by an A/B (the freeze came first). Its digest is equal.

## FINAL (2026-09-28, freeze)
- Default ON, proven (IDENTICAL bits equal, FAST bits equal, the same
  Mac for both arms): changes 1 to 8. No opt-in defines; nothing reverted.
- Board, m4pro-a: IDENTICAL 2.60 s -> 0.97 s, FAST 2.61 s -> 0.95 s.
  Round 1 baseline for reference: 5.12 s (835bca4ea, m4pro-b).
- Unproven commits: none in the default path. The later combined run
  (m2pro, NVIDIA, AMD, CPU) still has to cover the new units (fold_rows
  36-40, rows64 41, ck_* 42-44, strat_codes 45) and the host epilogue
  (x_metrics/epilogue.mojo) on the other columns: the host binding builds
  (checked locally on one core) but was not run on CPU/NVIDIA/AMD here.
- Seams: the new units have word tests in tools/apple_speed_metrics/,
  not yet fixtures in x_metrics/seams/x_metrics_check.mojo.
- Levers left: roc_auc_ovr (~0.21 s: device sort + upload of 13M words),
  stratified_kfold_shuffle label encoding (Python), shuffle_split's five
  device sorts.

## Unproven
- None in the default path (see FINAL). Build failures on the way:
  1790604342204 (`out` argument) and 1790619175919 (`fn` local), fixed by
  main's renames (ca5c76ace, db92af88c), merged into the lane.
