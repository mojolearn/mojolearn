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


## Unproven
- 8b2075c22, 5b330a3ff: queued as steward request 1790603676019 on m4pro-b
  (its name keeps the first sha; the request was retargeted in place to 5b330a3ff).
