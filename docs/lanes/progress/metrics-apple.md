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

## Changes (all IDENTICAL = same bits; FAST runs the same units)
1. Device merge pass = merge path (`sort_merge_path_unit`): one diagonal
   binary search per 8 outputs, then a two-pointer merge. The sort order is
   strict (6101), so the output is unique: the same words as the rank merge
   and the host's pair merge. Host test
   (~/mojolearn-evidence/metrics-apple/mtest/merge_path_host.mojo): 0 words
   differ from the pair merge over 18 shapes, output sorted.
2. `col_max` planned: chunk maxima (earliest largest non-NaN) + a
   left-to-right final with the first-NaN rule; host test against
   `col_max_unit` (NaN first/inside, +-0 ties, subnormals): 168 columns, 0
   differ.
3. Expected MI: each walk from the mode stops at the first u == 0 (every
   later u is 0 times a finite ratio: adds nothing to the exact z, and its
   term is skipped as pr == 0); portable logs memoized. 304 random cases,
   old == new bit for bit (1M case: 6.0 s -> 0.86 s on the laptop core).
4. roc_auc ovr: the row-sum check by map(math.fsum) over zip of column
   slices + max/min (|fl(s-1)| is monotone either side of 1); flags by
   bytes.translate into '<i4' words; support by bytes.count;
   `_drop_collinear`, `_trapezoid`, `_binary_auc`, `_binary_ap`, roc_curve
   ratios by operator maps (the same binary64 ops). Checked bit for bit vs
   the old code (~/mojolearn-evidence/metrics-apple/py_eq.py: 400 curves, 349
   AP cases, 3000 row-check cases, 0 differ).
5. Splitters: the test mask by map(mask.__setitem__), index arrays by
   array('q', compress(range(n), mask)); `_as_index` takes lists/ranges as
   they are; StratifiedKFold fold assignment by iterators + Counter. Checked
   vs the old module (ms_eq.py: 300 splits, 0 differ).

## Results
(pending: m4-a after = 1790581930499; identity = 1790581934145 with the
combined e2e_host_fadd + e2e_host_permute sabotage)

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
