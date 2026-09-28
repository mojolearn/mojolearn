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
(pending)

## Unproven
- 8b2075c22, 5b330a3ff: queued as steward request 1790603676019 on m4pro-b
  (its name keeps the first sha; the request was retargeted in place to 5b330a3ff).
