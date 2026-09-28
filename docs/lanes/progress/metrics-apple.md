# metrics-apple: progress

Lane `metrics-apple` (~/mojolearn-evidence/apple_speed_brief.md): Apple
Metal speed for the metrics family (metrics, CV, model_selection), FAST and
IDENTICAL. Worktree `~/mojolearn-wt/metrics-apple`, branch
`lane/metrics-apple` (off origin/main 5b622763d). Home Mac for speed: m4-a.
Evidence: ~/mojolearn-evidence/metrics-apple/.

## Board
`bench/x_metrics_speed.py` (1M rows, taxi + HIGGS from R2, `--reps 2`, the
minimum per case; `XMSPEED <case> <s> <digest>`). This lane added
`XMSPEED-INPUT` lines (a digest per input array, so a case digest that
differs between two boxes is judged against its inputs) and `--cprofile N`.
`MOJOLEARN_XMETRICS_PROFILE=1` now also prints the setup (plan + alloc +
upload) and download time of each device call.

## Before (read first: the metrics lane's queued Apple speed results)
| Mac | request | mode | board total | note |
|---|---|---|---|---|
| m4pro-b | 1790549640312 (51f2ef621, before the lane's phase C) | IDENTICAL | 138.76 s | |
| m4pro-b | 1790549640312 | FAST | 138.21 s | |
| m4pro-b | 1790549643846 (835bca4ea, phase C) | IDENTICAL | 5.12 s | 27x, every digest equal to before |
| m4pro-b | 1790549643846 | FAST | 5.09 s | every digest equal to before |
| m4-a | 1790580253728 (ea5591d10 = main + board flags) | IDENTICAL | 5.97 s | this lane's baseline |
| m4-a | 1790580253728 | FAST | 5.94 s | |

Where Apple stood (m4-a baseline, per call): roc_auc_ovr 1.61 s (Python row
check, flag lists, collinear drop; 0.45 s device for 5 sorts), AMI 1.05 s
(expected-MI walk over the full support, ~870k portable logs), stratified
kfold 0.53 s, shuffle_split 0.53 s (5 device sorts, 16 merge passes each),
kfold 0.31 s (index arrays via map(int)), the curves 0.17-0.28 s (merge
passes: ~150 ms of device time per 1M-row sort with profile syncs), the
medians 0.09-0.10 s, max_error 0.089 s (one GPU thread walks 1M rows).

## Changes (all IDENTICAL = same bits; FAST runs the same units, so FAST gets every change with its bits unchanged too)
1. Device merge pass = merge path (`sort_merge_path_unit`, 8ad21b657): one
   diagonal binary search per 8 outputs, then a two-pointer merge. The sort
   order is strict (6101), so the output is unique. Host test
   (mtest/merge_path_host.mojo): 0 words differ from the host merge, 18
   shapes. Device merge time per 1M-row sort ~150 ms -> ~40 ms (profile).
2. `col_max` planned (8ad21b657): chunk maxima (earliest largest non-NaN) +
   a left-to-right final with the first-NaN rule; host test vs
   `col_max_unit` (NaN first/inside, +-0 ties, subnormals): 0 differ.
   max_error 0.089 s -> 0.008 s.
3. Expected MI (8ad21b657): each walk from the mode stops at the first
   u == 0 (every later u is 0 times a finite ratio: nothing added to the
   exact z, its term skipped as pr == 0); portable logs memoized. 304 random
   cases vs the pre-lane code: 0 differ. AMI 1.05 s -> 0.27 s.
4. roc_auc / curves in Python (8ad21b657, 61d623f1b): the ovr row-sum check
   by map(math.fsum) + max/min (|fl(s-1)| is monotone either side of 1);
   flags by bytes.translate into '<i4' words; support by bytes.count;
   `_drop_collinear`, `_trapezoid`, `_binary_auc`, `_binary_ap`, roc_curve
   ratios by operator maps (the same binary64 ops). py_eq_orig.py vs the
   pre-lane code (5b622763d): 400 curves, 349 AP cases, 3000 row checks, 0
   differ.
5. Splitters (8ad21b657, e4f1bbd78, 220ec99ff): masks and index arrays built
   in C; `_as_index` takes lists as they are; first-seen codes by
   dict.fromkeys; StratifiedKFold fold allocation by floor division over the
   class runs (no sort) and test masks straight from the fold bytes (a
   `_Mask` the split loop takes as its mask). ms_eq_orig.py vs the pre-lane
   module: 411 splits (KFold, StratifiedKFold 2/3/5/7 folds, ShuffleSplit,
   mixed label types), 0 differ.
6. Unweighted prefixes in parallel (4e71f0263): the unweighted percentile
   CDF is Float32(i + 1) below 2^24 (`wpct_iota_unit`), and the unweighted
   curve walk is integer counting (chunk counts, offsets, fill). Integers
   do not depend on order, so the words are the sequential units'; the
   device no longer copies these slots to the host and back. Host test
   (mtest/plan_host.mojo): planned vs sequential arena, 155197 words, 0
   differ at 1 and 4 tasks; it bites under two sabotages (60036 and 5412
   words). The seam schedule fixture gains the unweighted percentile and
   col_max.
7. Collinear-drop flags on the device for unweighted curves (4f371e7bd):
   bin_curve params 10/11 = KEEP offset + flag; the planner adds
   `curve_keep_unit` after the curve (and after the unplanned fallback);
   roc_curve and every AUC compress the lists by the flags. Counts are
   integer-valued, so Int step compares decide as Python's binary64 ones
   (mtest/keep_host.mojo: 9150 slots, 0 differ). Weighted curves keep the
   Python rule.

Measured (probe, bench/x_metrics_copy_probe.mojo, m4-a): a device->host
copy runs at ~2-3.4 GB/s (12 MB: 3.5 ms, 120 MB: 37 ms) while host->device
runs at ~14 GB/s; a pinned host buffer or kernels on host-buffer memory are
no faster. The arena download is now the largest Metal overhead left
(roc_auc_ovr: 38 ms of its device call).

## Results (m4-a, board `--reps 2`, 1M rows)
Every one of the 29 case digests is equal to the baseline's in BOTH modes
at every round (and the XMSPEED-INPUT digests are equal).

| round | commit | IDENTICAL | FAST |
|---|---|---|---|
| baseline | ea5591d10 (main + board flags) | 5.97 s | 5.94 s |
| 1 | 8ad21b657 (items 1-5 first cut) | 3.46 s | 3.43 s |
| 2 | 99426ce36 (+ lane/merged, AP/roc ratios, fromkeys) | 3.51 s | 3.46 s |
| 3 | 4e71f0263 (+ parallel unweighted prefixes) | 3.35 s | 3.36 s |
| 4 | 4f371e7bd (+ device keep flags) | 2.99 s | 2.99 s |
| 5 | 220ec99ff (+ StratifiedKFold) | **2.82 s** | **2.80 s** |

On m4pro-b the same commit (1790587224015) runs in 2.66 s IDENTICAL and 2.64 s FAST, against 5.12 s / 5.09 s at 835bca4ea (1790549643846) on that Mac. All 29 digests are equal in both modes.

Per case, baseline -> round 5 (seconds; speedup on IDENTICAL):

| case | IDENTICAL before | after | x | FAST before | after |
|---|---|---|---|---|---|
| median_absolute_error | 0.0897 | 0.0238 | 3.8x | 0.0836 | 0.0213 |
| median_absolute_error_w | 0.0923 | 0.0375 | 2.5x | 0.0860 | 0.0282 |
| mean_pinball_loss | 0.0114 | 0.0117 | 1.0x | 0.0116 | 0.0114 |
| explained_variance_score | 0.0374 | 0.0392 | 1.0x | 0.0375 | 0.0383 |
| mean_tweedie_deviance | 0.0123 | 0.0136 | 0.9x | 0.0124 | 0.0131 |
| max_error | 0.0888 | 0.0079 | 11.2x | 0.0844 | 0.0075 |
| d2_absolute_error_score | 0.1042 | 0.0365 | 2.9x | 0.0983 | 0.0365 |
| r2_score_w_mo | 0.0703 | 0.0717 | 1.0x | 0.0682 | 0.0749 |
| mean_squared_log_error | 0.0125 | 0.0133 | 0.9x | 0.0124 | 0.0135 |
| balanced_accuracy_score | 0.0320 | 0.0348 | 0.9x | 0.0319 | 0.0344 |
| matthews_corrcoef | 0.0329 | 0.0325 | 1.0x | 0.0320 | 0.0346 |
| cohen_kappa_score | 0.0290 | 0.0292 | 1.0x | 0.0276 | 0.0301 |
| prfs_w | 0.0432 | 0.0444 | 1.0x | 0.0451 | 0.0499 |
| hamming_loss | 0.0327 | 0.0342 | 1.0x | 0.0324 | 0.0346 |
| roc_curve | 0.2553 | 0.1415 | 1.8x | 0.2585 | 0.1393 |
| roc_curve_w | 0.2815 | 0.2410 | 1.2x | 0.2788 | 0.2334 |
| average_precision_score | 0.1722 | 0.1310 | 1.3x | 0.1708 | 0.1344 |
| roc_auc_max_fpr | 0.2462 | 0.1354 | 1.8x | 0.2445 | 0.1362 |
| roc_auc_ovr | 1.6089 | 0.5570 | 2.9x | 1.6191 | 0.5602 |
| log_loss_w | 0.0379 | 0.0376 | 1.0x | 0.0362 | 0.0372 |
| brier_score_loss | 0.0226 | 0.0225 | 1.0x | 0.0210 | 0.0221 |
| top_k_accuracy_score | 0.0300 | 0.0297 | 1.0x | 0.0290 | 0.0300 |
| adjusted_mutual_info_score | 1.0481 | 0.2730 | 3.8x | 1.0511 | 0.2548 |
| calinski_harabasz_score | 0.0497 | 0.0521 | 1.0x | 0.0524 | 0.0520 |
| davies_bouldin_score | 0.0524 | 0.0487 | 1.1x | 0.0527 | 0.0524 |
| kfold_shuffle | 0.3079 | 0.2405 | 1.3x | 0.3027 | 0.2368 |
| stratified_kfold_shuffle | 0.5304 | 0.2846 | 1.9x | 0.5317 | 0.2811 |
| shuffle_split | 0.5278 | 0.1607 | 3.3x | 0.5259 | 0.1628 |
| train_test_split | 0.1059 | 0.0357 | 3.0x | 0.1071 | 0.0351 |

FAST quality: FAST's digests are identical before and after every change
(the changes keep the words in both tiers), so the paired quality numbers
(bench/x_metrics_fast_quality.py, 5 seeds x taxi + HIGGS vs scikit-learn)
are unchanged from the metrics lane's recorded ones; no FAST-only
approximation was added.

## Next levers (not done)
- `_rows_of` (itertools.compress into array('q')) is ~19 ms per 1M-row index
  array: 0.19 s of kfold_shuffle and of stratified_kfold_shuffle. A native
  mask-to-int64-rows export on the x_metrics binding would remove it, but it
  adds a binding export (and its surface-list entry), so it was left for a
  later pass.
- Downloading only the slots Python reads (inputs are downloaded back
  today): ~2-4% of the board on Apple.
- Weighted curves still drop collinear points in Python (roc_curve_w 0.24 s).
- The roc_auc_ovr digest differs between do-amd and the Macs (below).

## Open items found
- roc_auc_ovr's board digest differs between do-amd (c58b6b0f768b25fe) and
  the Macs (0309f2824a780685); the 28 other cases agree. The board's inputs
  come from numpy lstsq/exp; the new XMSPEED-INPUT lines settle whether the
  inputs or the metric differ (an AMD board run is needed).
- m2pro FAIL 1790544421099 (x-metrics-search DISAGREE at c236fa10f, every
  estimator-dependent part incl. `predict`), while 1790553421721 at the same
  commit PASSed on m2pro: the search lane fits GradientBoostingRegressor, and
  m2pro's gbdt Metal-vs-CPU disagreements show in the trees lane at the same
  time (gbdt-bfa-quantile, -exact-mae, -parametric-losses, -stochastic-arms,
  -multirmse, -categorical-ctr-tables). A trees-family (M2 gbdt) issue,
  reported to the orchestrator.
- Identity request 1790581934145 (8ad21b657, all six x-metrics lanes, the
  combined e2e_host_fadd + e2e_host_permute patch) was submitted before
  Andrew's "no verification in your lane" directive; a withdraw found every
  copy already claimed, and it no longer shows in `status`. No other
  identity request was made. Verification is the orchestrator's one check
  on lane/apple-merged.
- The merge of origin/lane/merged (275dd1bcf) brought the metrics CPU
  lane's host merge spans; `x_metrics/host/program.mojo` got one line so the
  host fans out the col_max and unweighted-curve chunk stages.
