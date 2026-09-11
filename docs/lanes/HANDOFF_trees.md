# Trees lane handoff (branch `lane/trees-identical`, 2026-09-09)

Current consolidated plan: [DECISION_TREE_ROADMAP.md](DECISION_TREE_ROADMAP.md).

Current scope and persistent project decisions: [TREE_GROWTH_SCOPE.md](TREE_GROWTH_SCOPE.md).

Symmetric GBDT (CatBoost mirror), RF (cuML mirror), ET. Our IDENTICAL arm
against each opponent's FAST arm on NVIDIA. Wind-down ordered by the
orchestrator before tasks 4 and 5 were measured on the H100; everything
below is either measured with a log path or marked not run.

## 2026-09-11 H100 confirmation leg (branch `lane/nvidia-identical-trees-0911`, source 352d9781)

Box: RunPod NVIDIA H100 80GB HBM3 (81559 MiB), driver 580.126.09, kernel
6.8.0-106, runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04, the
same GPU model, image and driver as the 2026-09-10 night leg (7cebeecf) on
a different physical pod. Pod bai1webdyjdqtx 04:00 to 04:26 UTC (26 minutes
of a 90-minute lease), reaped, HTTP 404 verified. Ours IDENTICAL only
(MOJOLEARN_NUMERIC_MODE=identical), alone in the process; no opponent ran,
no FAST or DETERMINISTIC arm was built. A measurement leg: main at 352d9781
carries, since 7cebeecf, DEVIATION 2512 (histogram zero and gbdt fills as
kernel launches), DEVIATION 2502 (a pure classification node is a leaf,
never retried) and the DEVIATION 2510 stamps (off unless
MOJOLEARN_STAGE_TIMES=1). Evidence:
`bench/results/trees_identical/h100_2026-09-11/{speed,ib,logs}/`;
`logs/batchP.sh` is the whole run in order, `logs/ab.txt` the exit-code
ledger, `logs/bins_sha256.txt` the four .so (the setup script's build and
the brief's rebuild of each binding produced byte-equal .so,
`logs/setup_so_sha256.txt` = `logs/baseline_so_sha256.txt`),
`speed/SUMMARY.md` the table below from `logs/summarize_speed.py`.

### Fingerprints (identity_break, IDENTICAL, 9 fixtures x2 per lane)

| lane | vs Sep 10 night H100 (`ib/sep10b_baseline.json`, 7cebeecf) | vs Apple M4 2502 (`ib/apple_rf2502.json`) |
|---|---|---|
| rf-clf | MOVED 9 of 9, parts predict and proba (expected: DEVIATION 2502) | IDENTICAL 9 of 9 |
| rf-reg | IDENTICAL 9 of 9 | IDENTICAL 9 of 9 |
| et-clf | IDENTICAL 9 of 9 | not in the Apple file |
| et-reg | IDENTICAL 9 of 9 | not in the Apple file |
| gbdt-symmetric | IDENTICAL 9 of 9 | not in the Apple file |
| gbdt-depthwise | IDENTICAL 9 of 9 | not in the Apple file |
| gbdt-lossguide | IDENTICAL 9 of 9 | not in the Apple file |
| gbdt-rmse | IDENTICAL 9 of 9 | not in the Apple file |
| kmeans | IDENTICAL 9 of 9 | not in the Apple file |

Exactly the brief's expectation: the only movement is rf-clf, and the new
rf-clf forest is bit-identical to the Apple M4's on all nine fixtures
(cross-vendor identity of the DEVIATION 2502 forest). DEVIATION 2512
moved nothing (`ib/diff.sep10b_baseline.baseline.txt`,
`ib/diff.apple_rf2502.baseline.txt`). No finding.

### Timing, ours IDENTICAL alone in the process, HIGGS first-N rows, ms median (min..max)

Run in this order, interleaved so drift is visible: rf 1M (pass1),
symmetric 1M (pass1), rf 2M, symmetric 1M (pass2), depthwise 1M, lossguide
1M, et 1M, rf 1M (pass2), then the rf 1M stage replicate. Opponent rows are
quoted from `bench/OPPONENT_REFERENCE.md` by name and were not re-run;
"Sep 9 row" = the 2026-09-09 same-image table, "Aug 28 row" = the
`e1g/2026-08-28_030908` table (different container).

| lane | rows | rounds | median | min..max | hash | Sep 10 night (7cebeecf) | opponent row (ms) | ours / theirs |
|---|---|---|---|---|---|---|---|---|
| rf (pass1) | 1M | 5 | 1088 | 1071..1108 | efd14ab2c09ff57c | 1516 (3ffa2951595422d4) | cuML RF 3314 (Sep 9 row) | 0.33x |
| rf (pass2, last cell) | 1M | 5 | 1081 | 1051..1099 | efd14ab2c09ff57c | 1521 (baseline2) | same | 0.33x |
| rf | 2M | 5 | 1760 | 1716..1792 | 7fd9fda29a4fa81d | 2283 (67d883dc6079b90f) | cuML RF 4543.0 (Aug 28 row) | 0.39x |
| gbdt-symmetric (pass1) | 1M | 7 | 527 | 463..602 | dac2cf366e219cec | 533 | CatBoost GPU symmetric 900 (Sep 9); 846.1 (Aug 28) | 0.59x; 0.62x |
| gbdt-symmetric (pass2) | 1M | 7 | 493 | 472..578 | dac2cf366e219cec | 533 | same | 0.55x; 0.58x |
| gbdt-depthwise | 1M | 7 | 1016 | 986..1083 | 592afa74b0d96982 | 1070 | XGBoost GPU depthwise 617.3; CatBoost GPU depthwise 1232.5 (Aug 28 rows) | 1.65x; 0.82x |
| gbdt-lossguide | 1M | 7 | 1610 | 1602..1649 | 4f02e5cb8088b281 | 1652 | XGBoost GPU lossguide 816.9; LightGBM CUDA 1313.7; CatBoost 1600.6 (Aug 28 rows) | 1.97x; 1.23x; 1.01x |
| et | 1M | 5 | 2504 | 2476..2585 | 2c192f6b12dbb6c5 | 2578 | no valid NVIDIA row | n/a |

FSPEED-ACC: rf 1M logloss 0.538817 / auc 0.809830 on both passes (the
Apple M4 DEVIATION 2502 numbers exactly; the previous forest gave 0.538850 /
0.809906); rf 2M 0.536959 / 0.811575 (previous forest 0.537060 / 0.811483);
symmetric 0.542067 / 0.800716, depthwise 0.525450 / 0.813518, lossguide
0.525348 / 0.813238, et 0.622379 / 0.762400, all equal to the Sep 10 night
cells, as their hashes require.

Reading: the RF classifier at 1M went 1516 to 1088 and 1081 (0.72x of the
Sep 10 night round) and at 2M 2283 to 1760 (0.77x); the drift control
(first and last cell, 25 minutes apart) agrees within 7 ms of median. The
forest changed (DEVIATION 2502), so this is a different fit, not the same
fit made cheaper: the behavior change is Andrew's call (see the Apple
section below). Every hash-unchanged lane is within 3 to 7 percent of the
Sep 10 night pod and lower: symmetric 533 to 527/493, depthwise 1070 to
1016, lossguide 1652 to 1610, et 2578 to 2504. Those gaps are on a
different physical pod and were not isolated against a 7cebeecf build in
the same window, so they are NOT attributed to DEVIATION 2512; the
brief's expectation (neutral on CUDA) is consistent with them and nothing
here contradicts it. The symmetric 478 (Sep 10) vs 533 (Sep 10 night) vs
527/493 (here) spread is the size of one pod's drift and stays UNRESOLVED
as a gbdt-path question.

### RF HIGGS 1M per-stage split (MOJOLEARN_STAGE_TIMES=1, one replicate, drains per stage, not a timing)

`speed/baseline.rf.higgs.r1000000.stage.log`, round 1139 (warm-up 1876),
hash efd14ab2c09ff57c; the Sep 10 night `stamps2` column is round 1552.

| where | stage | Sep 11 ms | Sep 10 night ms |
|---|---|---|---|
| fit_forest | device_wait | 454 | 761 |
| fit_forest | host_enq_partition (stage + enqueue the node split) | 112 | 102 |
| fit_forest | host_queue_push (children into the host tree) | 111 | 104 |
| fit_forest | host_enq_hist (next batch's histogram round) | 104 | 82 |
| fit_forest | host_enq_hist_retry (retry sampling rounds) | stamp did not fire: no retry round | 89 |
| fit_forest | leaf_values | 71 | 71 |
| fit_forest | tree_copy | 35 | 33 |
| fit_forest | host_read_splits | 18 | 18 |
| fit_forest | flush_splits | 5 | 6 |
| fit_forest | host_begin_tree + row_sampling + quantiles + bin_dataset + host_setup + host_teardown | 7 | 6 |
| fit_forest | fit_total | 941 | 1342 |
| binding | bind_host_copy 20 + bind_h2d 2 + ctx/pinned/release/retain < 0.2 | 23 | 22 |
| binding | binding_total | 964 | 1361 |
| Python | round minus binding_total | 175 | 191 |

The 352d9781 stamps also print the pieces nested INSIDE `host_enq_hist`
(host_stage_items 31, host_phase_upload 8, host_launch_setup 7,
host_hist_zero 5, host_hist_launch 5, host_best_launch 5); they are
double-counted against the parent, which is why `other` prints negative
(-37). Under DEVIATION 2502 the retry stamp never fired on HIGGS 1M: every
node that ended a batch without a valid split was pure, so the 89 ms of
retry enqueue is gone along with the retry rounds' device time; device_wait
761 to 454 is the rest of the round-count drop. The remaining host-side
enqueue and queue work (about 330 ms of the 941) is unchanged from the Sep
10 night candidates list, which still applies.

### RUN OWED

- Isolate DEVIATION 2512 on CUDA: a 7cebeecf-source and a 352d9781-source
  build of the gbdt and rf bindings, both IDENTICAL, interleaved on one pod
  in one window at 1M and 2M (the 3 to 7 percent lower hash-unchanged
  cells here are drift-sized and unattributed).
- RF 2M stage split and the gbdt 1M stage splits at 352d9781 (only the RF
  1M stage replicate ran here).
- Depthwise and lossguide 2M cells at 352d9781 (the brief's list stopped at
  1M for the gbdt lanes; the Sep 10 night 2M cells are 1917 and 2451).
- Extra trees still has no valid NVIDIA opponent row.
- The nine-lane Apple identity JSON regeneration (DEVIATION 2340 dtype,
  DEVIATION 2502 rf-clf) is still owed; this leg diffed against the
  two-lane Apple 2502 file only.
- Andrew's decision on DEVIATION 2502 as the default classifier forest;
  the cross-vendor identity of the new forest is now established
  (Apple M4 and H100 18 of 18).

### What landed (commits, `%h parent %p`)

- the evidence directory, this section and the OPPONENT_REFERENCE note:
  see the branch `lane/nvidia-identical-trees-0911`, NOT merged to main.

## 2026-09-10 night, Apple M4: RF host tax and pure nodes (branch `lane/trees-perf-0910b`)

Evidence: `bench/results/rf_fast_mac_2026-09-10/` (README carries every
number). HIGGS 1M, 100 trees, depth 16, the harness's RF config, M4, MAX
26.5, wall time of `fit`, runs not interleaved with each other.

### What the M4 fit was doing

A 17.5 s IDENTICAL fit: device_wait 4.3 s, `other` 12.8 s (the H100's
`other` was 0.49 s). Stamps inside the batch driver found 9.5 s of it in
the per-round histogram `enqueue_memset`: under MAX 26.5 on Metal a memset
between two kernel launches costs the host about 110 us more than the
launches (microbenchmarks in the evidence directory), and this one, up to
21 MB and 14,640 times per fit, averaged 650 us. A copy or a kernel in the
same position pays nothing extra. MAX's own floors on the M4: 19 us per
`enqueue_function`, 155 us per launch plus synchronize, one command buffer
per launch.

### DEVIATION 2500 (Python labels, on main since 59d7fbea)

`sorted_classes(flatten_labels(y))` was 400 ms of pure Python per
1,000,000-row classifier fit on the M4 (about 0.9 s of the H100's 2.3 s
RF round). `encode_labels` runs the same ORDER RULE in the base binding
for one numeric buffer: 8 ms. Forest hash unchanged. The H100 night leg
measured its effect: RF 1M 2234 -> 1516, 2M 3743 -> 2283, ET 1M 3314 -> 2578.

### DEVIATION 2512 (`core/device_zero.mojo`)

The histogram zero rides a kernel launch. Hash unchanged
(3ffa2951595422d4, 3 of 3), M4 FAST fit 12.4 s. `pixi run check-device-zero`.
The same `enqueue_memset`-between-launches pattern exists in gbdt
(`greedy_search_helper*.mojo` histogram memsets, `pointwise_scores_calcer`,
`dynamic_boosting`) and is the first thing to try on the gbdt lanes on the
M4; not yet measured there.

### DEVIATION 2502 (a pure node is a leaf) -- CHANGES THE FOREST

Rounds: 14,640 histogram rounds for 3,640 node batches per fit. Round 0
processed 2.54M nodes over 1.59 billion rows; rounds 1 to 5 each re-ran
about 527k nodes averaging 9 rows, 99.2% of which ended the batch with no
split after every column was tried, all of them pure (leaf vectors
audited). The other 0.8% were pure nodes with a numerically nonzero Gini
gain that split into two pure children on a retry. The split kernel now
writes the node's purity into the slot and the host treats a pure node as
a leaf and never retries it (scikit-learn's rule). Rounds 3,627; M4 fit
9.1 s FAST, 8.9 s IDENTICAL; hash efd14ab2c09ff57c on 8 of 8 FAST and 3 of
3 IDENTICAL fits, FAST equals IDENTICAL; logloss 0.538850 -> 0.538817, AUC
0.809906 -> 0.809830.

The forest differs because the skipped splits shift later nodes' tree
indices, which seed the column sampler. rf-reg fingerprints are unchanged
(regression never marks a node); rf-clf moved on 9 of 9 fixtures
(`bench/results/identity_break/apple-m4.identical.rf-2502-2026-09-10.json`).
ON BY DEFAULT again since 2026-09-11 evening: it was opt-in for one day
(a one-dataset decision), then won on both section-9 datasets at 1M on the
M4 with equal logloss (`bench/results/rf_2502_m4_2026-09-11/`) and Andrew
ratified the flip; opt out with `-D MOJOLEARN_2502_RETRY_PURE=1`.
`rf_perf_candidates_check` ALL ARMS GREEN under the IDENTICAL define.

### DEVIATION 2512 in gbdt (`bench/results/gbdt_fast_mac_2026-09-10/`)

All 32 `enqueue_memset` sites under gbdt/ go through
`core/device_zero.enqueue_fill`. Same bytes, hashes unchanged, IDENTICAL
gbdt fingerprints 36 of 36 equal. M4 FAST HIGGS 1M: symmetric 4,312 ->
2,544..2,591 ms; depthwise 8,499 -> 6,318..6,500; lossguide 17,713 ->
10,865..11,232. IDENTICAL symmetric 2,837..2,871 (hash dac2cf366e219cec).

### ExtraTrees on the M4 (`bench/results/et_fast_mac_2026-09-10/`)

FAST 1M fit 11.0 s: stage + feature sampler 2.0 s, range pass 3.0 s, score
pass 3.8 s, partition 2.4 s, host 0.1 s. Kernel time, no host tax; a
kernel lane (the range and score passes read the node rows through
row ids; the M4 has about 100 GB/s).

### RUN OWED

- H100: RF 1M and 2M at this source. RAN 2026-09-11 (section above):
  RF 1M 1088/1081, 2M 1760, hash efd14ab2c09ff57c at 1M equal to the M4;
  the retry stamp did not fire and device_wait fell 761 to 454; rf-clf
  moved 9 of 9 and nothing else moved.
- Apple M4: gbdt-symmetric, depthwise, lossguide FAST stage splits at 1M
  to price their memsets (DEVIATION 2512 candidates); ET 1M stage split.
- The nine-lane Apple identity JSON regeneration now has two reasons
  (DEVIATION 2340 predict dtype, DEVIATION 2502 rf-clf).

## 2026-09-10 night H100 leg (branch `lane/nvidia-identical-trees-0910b`, source 7cebeecf)

Box: RunPod NVIDIA H100 80GB HBM3 (81559 MiB), driver 580.126.09, kernel
6.8.0-106, runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04,
224-core host, catboost 1.2.10, cuml 26.08.00, numpy 2.4.6 installed but NO
opponent ran; ours IDENTICAL only (MOJOLEARN_NUMERIC_MODE=identical,
-D MOJOLEARN_NUMERIC_IDENTICAL=1). Pod lztaetqqwol9jj 02:35 to 03:24 UTC
(49 minutes of a 120-minute lease), reaped, HTTP 404 verified. Evidence:
`bench/results/trees_identical/h100_2026-09-10b/{ib,speed,logs,patches}/`;
`logs/batchK.sh`, `batchL.sh`, `batchM.sh`, `batchN.sh` ran in that order,
`logs/ab.txt` is the exit-code ledger, `logs/bins_sha256.txt` every binary
set, `speed/SUMMARY.md` the table below from `logs/summarize_speed.py`.
Binary sets: `baseline` = the four .so the setup script built from 7cebeecf
(`baseline2` is the same four, a second RF 1M pass adjacent to the A/B
switch); `stamps`/`stamps2`/`stampsg` carry the DEVIATION 2510 stamps (RF
binding twice, gbdt once) and were used for stage runs only; `exp2511` is
the baseline .so with the DEVIATION 2511 Python patch.

### Fingerprints (identity_break, IDENTICAL, 9 fixtures x2)

- baseline (7cebeecf) vs the retained Sep 10 set
  (`h100_2026-09-10/ib/baseline.json`, deb01bcf): 81/81 IDENTICAL, all nine
  lanes (`ib/diff.sep10_baseline.baseline.txt`). The DEVIATION 2340 dtype
  movement of the Sep 9 comparison is gone against a Sep 10 file, as the
  brief expected. rf-clf/rf-reg vs `flip2011.json`: 18/18 IDENTICAL.
- exp2511 (DEVIATION 2511 patch) vs baseline, rf-clf, rf-reg, et-clf,
  et-reg: 36/36 IDENTICAL (`ib/diff.baseline.exp2511.txt`).
- No lane moved; nothing was excluded from optimization on that ground.

### Timing, ours IDENTICAL alone in the process, HIGGS first-N rows, ms median (min..max)

Opponent rows are quoted from `bench/OPPONENT_REFERENCE.md` by name and were
not re-run. "Sep 9 row" = the 2026-09-09 same-image table; "Aug 28 row" =
the `e1g/2026-08-28_030908` table (different container).

| set | lane | rows | rounds | median | min..max | hash | opponent row (ms) | ours / theirs |
|---|---|---|---|---|---|---|---|---|
| baseline | rf | 1M | 5 | 1516 | 1476..1579 | 3ffa2951595422d4 | cuML RF 3314 (Sep 9 row) | 0.46x |
| baseline2 | rf | 1M | 5 | 1521 | 1485..1576 | 3ffa2951595422d4 | same | 0.46x |
| baseline | rf | 2M | 5 | 2283 | 2231..2293 | 67d883dc6079b90f | cuML RF 4543.0 (Aug 28 row) | 0.50x |
| baseline | gbdt-depthwise | 1M | 7 | 1070 | 987..1121 | 592afa74b0d96982 | XGBoost GPU depthwise 617.3; CatBoost GPU depthwise 1232.5 (Aug 28 rows) | 1.73x; 0.87x |
| baseline | gbdt-depthwise | 2M | 5 | 1917 | 1905..1962 | e9e6dff4ba5496ae | XGBoost 1039.4; CatBoost 1560.6 (Aug 28 rows) | 1.84x; 1.23x |
| baseline | gbdt-lossguide | 1M | 7 | 1652 | 1639..1690 | 4f02e5cb8088b281 | XGBoost GPU lossguide 816.9; LightGBM CUDA 1313.7; CatBoost 1600.6 (Aug 28 rows) | 2.02x; 1.26x; 1.03x |
| baseline | gbdt-lossguide | 2M | 5 | 2451 | 2437..2509 | ff9ddbb329632a2d | XGBoost 1272.0; LightGBM 1669.3; CatBoost 1954.5 (Aug 28 rows) | 1.93x; 1.47x; 1.25x |
| baseline | gbdt-symmetric | 1M | 7 | 533 | 490..570 | dac2cf366e219cec | CatBoost GPU symmetric 900 (Sep 9 row); 846.1 (Aug 28 row) | 0.59x; 0.63x |
| baseline | et | 1M | 5 | 2578 | 2543..2616 | 2c192f6b12dbb6c5 | no valid NVIDIA row (Sep 10 ours: 3314) | n/a |
| exp2511 | rf | 1M | 5 | 1484 | 1480..1534 | 3ffa2951595422d4 | A/B vs baseline/baseline2 | see DEVIATION 2511 |
| exp2511 | rf | 2M | 5 | 2180 | 2175..2212 | 67d883dc6079b90f | A/B vs baseline | see DEVIATION 2511 |

FSPEED-ACC, every set and rung equal to Sep 10 where a Sep 10 cell exists:
rf 1M logloss 0.538850 / auc 0.809906, rf 2M 0.537060 / 0.811483; depthwise
1M 0.525450 / 0.813518, 2M 0.524432 / 0.814172; lossguide 1M 0.525348 /
0.813238, 2M 0.524618 / 0.813858; symmetric 1M 0.542067 / 0.800716; et 1M
0.622379 / 0.762400. The RF hashes are the ones the brief required.

Against the Sep 10 leg (deb01bcf, same GPU model and image): RF 1M 2234 ->
1516, RF 2M 3743 -> 2283, ET 1M 3314 -> 2578; DEVIATION 2500 (native label
encoding, 59d7fbea) is the change between them on the forest path. Symmetric
1M 478 -> 533 moved the other way on a different physical pod; drift or a
gbdt-path regression between deb01bcf and 7cebeecf is UNRESOLVED (RUN OWED
below). Depthwise and lossguide had no post-boundary-tax H100 number; the
Sep 8 ratios (1.6-2.2x and 2.1-2.5x of XGBoost) are now 1.73-1.84x and
1.93-2.02x.

### Per-stage splits (MOJOLEARN_STAGE_TIMES=1, one replicate each, drains per stage, not a timing)

RF HIGGS 1M, set `stamps2` (DEVIATION 2510 stamps, `speed/stamps2.rf.higgs.r1000000.stage.log`),
round 1552, hash 3ffa2951595422d4:

| where | stage | ms |
|---|---|---|
| fit_forest | device_wait | 761 |
| fit_forest | host_queue_push (children into the host tree) | 104 |
| fit_forest | host_enq_partition (stage + enqueue the node split) | 102 |
| fit_forest | host_enq_hist_retry (retry sampling rounds) | 89 |
| fit_forest | host_enq_hist (next batch's histogram round) | 82 |
| fit_forest | leaf_values | 71 |
| fit_forest | tree_copy | 33 |
| fit_forest | host_read_splits | 18 |
| fit_forest | flush_splits | 6 |
| fit_forest | host_begin_tree + row_sampling + quantiles + bin_dataset + host_setup + host_teardown | 6 |
| fit_forest | other (unstamped) | 68 |
| fit_forest | fit_total | 1342 |
| binding | bind_host_copy 20 + bind_h2d 2 + ctx/pinned/release/retain < 0.2 | 22 |
| binding | binding_total | 1361 |
| Python | round minus binding_total | 191 |

The Python 191 ms by perf_counter in `logs/rf_residual_profile.baseline.log`
and `logs/rf_export_profile.baseline.log` (3 reps each): `as_f32_colmajor`
77 (the C-to-F transpose DEVIATION 1840 keeps on purpose), `encode_labels`
16-18 (was 277 before DEVIATION 2500), `del Xf` 5-7, export destination
allocation 40-48 (`empty()`, zero-filled, 4,019,922 nodes = 96 MB), export
copy 19-35, export release 0.3-3. The Sep 10 round-minus-fit residual of
937 ms is now 191 + 22.

The unstamped `other` of the Sep 10 table (492-499 ms) is therefore host
work in the level loop: 395 ms of it is enqueue/staging plus the queue push,
not a hidden kernel. The stamps are `stop_host` (no drain), off unless the
variable is set, and are on this branch (DEVIATION 2510, ca439973).

GBDT HIGGS 1M, set `stampsg` (gbdt entry and `train` host-phase stamps,
DEVIATION 2510; `speed/stampsg.gbdt-{depthwise,lossguide}.higgs.r1000000.stage.log`),
plus the per-tree tables of the baseline stage runs summed over the 100
trees (`speed/baseline.gbdt-*.higgs.r1000000.stage.log`):

| stage | depthwise (round 1160) | lossguide (round 1944) |
|---|---|---|
| Python (round minus gbdt_fit_total) | 105 | 106 |
| gbdt_fit_host_copy_in (112 MB List copy of X at the binding) | 73 | 73 |
| train_pre_quantize (per-column host build, a second copy) | 99 | 99 |
| train_quantize_borders | 59 | 60 |
| train_cindex_build | 17 | 20 |
| train_targets_upload + train_pre_fit | 5 | 5 |
| train_fit_with_test (the boosting loop) | 790 | 1562 |
| of which: 100 tree-structure tables summed | 210 | 975 |
| of which: est.pstats/approx/move/readback | 68 | 68 |
| of which: unwrapped per-iteration work (derivatives, bootstrap, cursor update, loss readback) | ~512 | ~519 |
| gbdt_fit_model_text | 12 | 14 |

The two GBDT lanes share about 880 ms that is not tree search: ~105 Python,
~250 host copies and quantization before the loop, ~512 unwrapped inside
the loop, ~13 model text. For depthwise that is 76 percent of the round.

### What was tried: DEVIATION 2511, raw-malloc export destinations (NOT KEPT under the leg's rule)

`_export_fit_result` (python/mojolearn/_forest_protocol.py) allocated the
five model arrays with `empty()` (zero-filled `array.array`); the native
`forest_export` writes every element of every one, so the fill is a second
pass over 96 MB (1M) / 192 MB (2M) of fresh pages every round (the previous
model's arrays were just freed). The patch allocates them with
`_output_store` (`PyMem_RawMalloc`, the DEVIATION 2473/2500 pattern) and
`Array._owned`; bytes unchanged, fingerprints 36/36 IDENTICAL, hashes equal.
Patch: `bench/results/trees_identical/h100_2026-09-10b/patches/dev2511_forest_export_raw_store.patch`.

| rung | baseline | baseline2 (adjacent pass) | exp2511 | rule |
|---|---|---|---|---|
| RF 1M | 1516 (1476..1579) | 1521 (1485..1576) | 1484 (1480..1534) | median lower by 32-37; range 1480..1534 OVERLAPS the baseline medians: FAIL |
| RF 2M | 2283 (2231..2293) | not run | 2180 (2175..2212) | median lower by 103; range below 2283: PASS |

Kept only if both rungs pass, so the tree was reverted and the patch
retained. Mechanism check (`logs/export_alloc_micro.log`, CPU only):
`empty()` for the five shapes 48.5 ms on fresh pages, 6.5 ms when the
allocator reuses them; `_output_store` 0.02-0.15 ms plus a 5-34 ms first
touch that the native export then pays instead of the fill. Orchestrator's
call: apply it on the 2M evidence, or re-run the 1M rung interleaved with
7+ rounds. `logs/rf_export_profile.exp2511.log` is NOT a measurement of the
patch (the scratch script hardcodes `empty()`); ignore it.

### Candidates not taken, with numbers (all mechanical, none touch arithmetic or fold order)

1. GBDT `gbdt_fit_host_copy_in` 73 ms + `train_pre_quantize` 99 ms: two
   host copies of the 112 MB column-major X before quantization (the List
   at `gbdt/estimator.mojo` and the per-column build in `gbdt/train.mojo`).
   Borrowing the caller's buffer the way WP1 did for the forests removes
   both; 172 ms of a 1070 ms depthwise round, the same 172 of 1652 lossguide.
2. GBDT unwrapped per-iteration work, ~512 ms in both lanes (5 ms per
   iteration): derivative launch, deterministic sums, bootstrap, make_sequence,
   cursor update, loss readback (`gbdt/methods/doc_parallel_boosting.mojo`
   1410-2110). Needs its own stamps before anything is attributed.
3. RF enqueue/staging, 273 ms (`host_enq_partition` 102, `host_enq_hist_retry`
   89, `host_enq_hist` 82) over roughly 1600 batches: per-launch host cost of
   `enqueue_function` plus pinned staging. Candidate: count launches per batch
   with RF_LAUNCH_LOG at 1M and fold the per-batch small copies into the
   DEVIATION 1908 packed upload. Retry rounds alone are 89 ms.
4. RF `host_queue_push` 104 ms for 4.02M nodes is the transcribed
   `NodeQueue::push` with a reserve; near its floor, not a target.
5. RF `tree_copy` 33 ms: `ts.queue.tree.copy()` in `_finish_tree` plus
   `states[k].tree.copy()` in fit_forest, two copies per tree; a move would
   remove one. Small.
6. Python `as_f32_colmajor` 77 ms is DEVIATION 1840's deliberate cost.

### What landed (commits, `%h parent %p`)

- ca439973 parent 7cebeecf: RF stage stamps (DEVIATION 2510), three files.
- 69c64d11 parent ca439973: gbdt entry and `train` host-phase stamps (DEVIATION 2510).
- 849b7ccf parent 69c64d11: the evidence directory.
- the commit after 849b7ccf: this section and the OPPONENT_REFERENCE note.
- NOT committed to the tree: DEVIATION 2511 (patch retained under `patches/`).

### RUN OWED

- DONE 2026-09-11 (orchestrator, Apple M4, merged state 7356fb67):
  identity_break gbdt-symmetric, gbdt-depthwise, gbdt-lossguide, gbdt-rmse,
  rf-reg against the retained `apple-m4.identical.json`: 45 of 45
  IDENTICAL. rf-clf moved on 9 of 9 by DEVIATION 2502 (below), not by the
  stamps.
- Symmetric 1M 533 vs Sep 10's 478 on a different pod: one interleaved RF +
  symmetric session on one H100 to separate drift from a gbdt-path change
  between deb01bcf and 7cebeecf.
- DEVIATION 2511 at 1M, interleaved, 7+ rounds, if the orchestrator wants
  the rule satisfied before applying it.
- Extra trees NVIDIA opponent row (unchanged from Sep 10).
- 5M rungs were not started (brief).

## 2026-09-10 H100 leg (branch `lane/nvidia-identical-trees-0910`, source deb01bcf)

Box: RunPod NVIDIA H100 80GB HBM3, driver 580.126.09, kernel 6.8.0-106,
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04, catboost 1.2.10,
cuml 26.08.00, numpy 2.4.6 (no opponent ran; ours IDENTICAL only). Pod
51pwsv6pe55tzc 20:34 to 21:44 UTC, reaped, HTTP 404 verified. Evidence:
`bench/results/trees_identical/h100_2026-09-10/{ib,speed,gates,logs}/`;
`logs/batchI.sh` and `logs/batchJ.sh` are the scripts that ran, `logs/ab.txt`
the exit-code ledger, `logs/bins_sha256.txt` every binary set, `speed/SUMMARY.md`
the table below regenerated by `logs/summarize_speed.py`.

### Fingerprints (identity_break, IDENTICAL, 9 fixtures x2)

- baseline (deb01bcf) vs the Sep 9 H100 set (`h100_2026-09-09/ib/baseline.json`,
  a9ba6818): 63/81 IDENTICAL (rf-reg, et-reg, all four gbdt lanes, kmeans),
  18/81 DIVERGENT: every rf-clf and et-clf fixture, `predict` only, `proba`
  equal (`ib/diff.sep9_h100_baseline.baseline.txt`). UNDERSTOOD, not a forest
  change: `ib/predict_dtype_witness.txt` (script beside it) refits the 18
  models; classifier `predict` is now an int64 `Array` (DEVIATION 2340,
  NumPy-free Python layer 637de940, on main since 36d48b07, after a9ba6818)
  where Sep 9 returned the int32 label dtype, and the fingerprint hashes the
  dtype. Cast back to int32, today's `predict` reproduces the Sep 9 hash on
  18/18; `proba` equals on 18/18. Large witness: RF HIGGS 1M prediction hash
  3ffa2951595422d4 and logloss/AUC 0.538850/0.809906 equal Sep 9's. The
  Apple JSON `bench/results/identity_break/apple-m4.identical.json` (e616906e)
  also predates 2340 and will show the same 18 moves against any current
  build: RUN OWED (Apple, orchestrator) to regenerate it.
- rf2010, rf2011, rf2012 vs baseline: 18/18 equal each (rf-clf, rf-reg).
- fused, sp, spnf vs baseline: 36/36 equal each (four gbdt lanes).
- flip2011 (the flipped SOURCE, no define) vs baseline and vs rf2011: 18/18
  equal each.

### Boundary-tax gates on NVIDIA (all `-D MOJOLEARN_NUMERIC_IDENTICAL=1`)

| gate | check | result | log |
|---|---|---|---|
| WP4 | ensemble/checks/oob_check.mojo | PASS (ALL OK) | gates/wp4_oob_check.run.log |
| WP1 | extratrees/checks/borrowed_upload_check.mojo | PASS (ALL OK) | gates/wp1_borrowed_upload_check.run.log |
| WP8 | extratrees/checks/stage_upload_bytes_check.mojo | PASS | gates/wp8_stage_upload_bytes_check.run.log |
| WP2 | checks/forest_export_protocol.mojo (-I bindings) | PASS | gates/wp2_forest_export_protocol.run.log |
| WP2 | checks/forest_export_public.py --mode identical --vendor cuda | PASS, 5 cases | gates/wp2_forest_export_public.run.log |
| WP3 | checks/forest_inference_model.mojo, separate arrays | RESIDENT_FOREST_PASS, VENDOR cuda | gates/wp3_forest_inference_separate.run.log |
| WP3 | same, -D MOJOLEARN_FOREST_PACKED_NODES=1 | RESIDENT_FOREST_PASS, VENDOR cuda | gates/wp3_forest_inference_packed.run.log |
| WP5 | checks/gbdt_cindex_staging_check.mojo | PASS 1/19, 257/35, 8193/67 | gates/wp5_gbdt_cindex_staging.run.log |
| WP5 | checks/nan_mode_check.mojo | PASS | gates/wp5_nan_mode_check.run.log |
| reach | ensemble/checks/rf_perf_candidates_check.mojo, shipped source | ALL ARMS GREEN | logs/rfgate.src.log |
| reach | same, flipped source (HIST_ITEMS_PER_THREAD 4) | ALL ARMS GREEN | logs/rfgate.flip2011src.log |
| reach | same, -D MOJOLEARN_2011_HIST_ITEMS1=1 (the opt-out) | ALL ARMS GREEN | logs/rfgate.flip2011optout.log |

### Timing, ours IDENTICAL only (HIGGS first-N rows, test = last 500k, ms median, min..max, one dropped warm-up)

Cells ran sequentially, one set after another, not interleaved; the
`driftcheck` row is the baseline 1M cell re-run at the end of the leg as
the drift control for that ordering (the box drifted 100 ms down over the
hour).

| set | lane | rows | rounds | median | min..max | hash | log |
|---|---|---|---|---|---|---|---|
| baseline | rf | 1M | 5 | 2475 | 2385..2532 | 3ffa2951595422d4 | speed/baseline.rf.higgs.r1000000.ours.log |
| baseline | rf | 2M | 5 | 4037 | 3890..4135 | 67d883dc6079b90f | speed/baseline.rf.higgs.r2000000.ours.log |
| baseline (driftcheck, 21:40 UTC) | rf | 1M | 5 | 2375 | 2340..2406 | 3ffa2951595422d4 | speed/baseline.rf.higgs.r1000000.ours.driftcheck.log |
| rf2010 (-D MOJOLEARN_2010_ROWS_SORTED=1) | rf | 1M | 5 | 2476 | 2453..2499 | 3ffa2951595422d4 | speed/rf2010.rf.higgs.r1000000.ours.log |
| rf2010 | rf | 2M | 5 | 4229 | 4183..4284 | 67d883dc6079b90f | speed/rf2010.rf.higgs.r2000000.ours.log |
| rf2011 (-D MOJOLEARN_2011_HIST_ITEMS4=1) | rf | 1M | 5 | 2274 | 2239..2315 | 3ffa2951595422d4 | speed/rf2011.rf.higgs.r1000000.ours.log |
| rf2011 | rf | 2M | 5 | 3743 | 3719..3761 | 67d883dc6079b90f | speed/rf2011.rf.higgs.r2000000.ours.log |
| rf2012 (-D MOJOLEARN_2012_SMEM_COPIES4=1) | rf | 1M | 5 | 2389 | 2340..2416 | 3ffa2951595422d4 | speed/rf2012.rf.higgs.r1000000.ours.log |
| rf2012 | rf | 2M | 5 | 3957 | 3951..4036 | 67d883dc6079b90f | speed/rf2012.rf.higgs.r2000000.ours.log |
| flip2011 (flipped source, no define) | rf | 1M | 5 | 2234 | 2216..2277 | 3ffa2951595422d4 | speed/flip2011.rf.higgs.r1000000.ours.log |
| baseline | et | 1M | 5 | 3314 | 3272..3430 | 2c192f6b12dbb6c5 | speed/baseline.et.higgs.r1000000.ours.log |
| baseline | et | 2M | 5 | 6099 | 5994..6335 | a9a9ff528583d3bd | speed/baseline.et.higgs.r2000000.ours.log |
| baseline | gbdt-symmetric Logloss | 1M | 7 | 478 | 459..528 | dac2cf366e219cec | speed/baseline.gbdt-symmetric.higgs.r1000000.ours.log |
| baseline | gbdt-symmetric Logloss | 2M | 5 | 753 | 741..926 | 2000f16a43380f98 | speed/baseline.gbdt-symmetric.higgs.r2000000.ours.log |
| baseline | gbdt-symmetric RMSE (higgsreg) | 1M | 7 | 396 | 377..476 | 037aa3188d25c889 | speed/baseline.gbdt-symmetric.higgsreg.r1000000.ours.log |
| fused (-D MOJOLEARN_2030_FUSED_EST_MOVE=1) | gbdt-symmetric Logloss | 1M | 7 | 508 | 442..563 | dac2cf366e219cec | speed/fused.gbdt-symmetric.higgs.r1000000.ours.log |
| fused | gbdt-symmetric Logloss | 2M | 5 | 781 | 728..937 | 2000f16a43380f98 | speed/fused.gbdt-symmetric.higgs.r2000000.ours.log |
| sp (2030 + -D MOJOLEARN_IDENTICAL_SINGLE_PASS_PARTITION=1) | gbdt-symmetric Logloss | 1M | 7 | 541 | 493..676 | dac2cf366e219cec | speed/sp.gbdt-symmetric.higgs.r1000000.ours.log |
| sp | gbdt-symmetric Logloss | 2M | 5 | 824 | 823..1026 | 2000f16a43380f98 | speed/sp.gbdt-symmetric.higgs.r2000000.ours.log |
| sp | gbdt-symmetric RMSE (higgsreg) | 1M | 7 | 448 | 434..525 | 037aa3188d25c889 | speed/sp.gbdt-symmetric.higgsreg.r1000000.ours.log |
| spnf (single-pass define only) | gbdt-symmetric Logloss | 1M | 7 | 548 | 516..595 | dac2cf366e219cec | speed/spnf.gbdt-symmetric.higgs.r1000000.ours.log |
| spnf | gbdt-symmetric Logloss | 2M | 5 | 852 | 838..1040 | 2000f16a43380f98 | speed/spnf.gbdt-symmetric.higgs.r2000000.ours.log |

RF 1M quality (same 500k tail, every RF set): logloss 0.538850, AUC
0.809906; 2M: 0.537060 / 0.811483. ET 1M: 0.622379 / 0.762400. RF
per-phase split (`speed/baseline.rf.higgs.r1000000.stage.log`, one untimed
replicate, drains per stage): fit_total 1.41 s of a 2.34 s round,
device_wait 0.84, other 0.49, leaf_values 0.07, quantiles/bin/row_sampling
under 3 ms; about 0.9 s of each round is outside the Mojo fit.

Opponent rows quoted, NOT re-measured (bench/OPPONENT_REFERENCE.md): cuML
RandomForestClassifier 3314 ms (3257..3950) at 1M, "Trees, HIGGS 1M,
2026-09-09 trees lane" table, same GPU model and image as this leg; cuML
RandomForest 4543.0 at 2M, Aug 28 `e1g/2026-08-28_030908` table (different
container, driver 580.126.09). Ours IDENTICAL RF at the flipped default: 2234
ms at 1M (0.67 of the same-box cuML row) and, from the rf2011 cell, 3743 at
2M (0.82 of the Aug 28 row). Sep 9's ours row on this GPU model was 5762
(5116..6451) with the same hash. ET has no valid NVIDIA opponent row; ours
alone above. Symmetric: CatBoost GPU 900 (864..939) Logloss / 699 (680..759)
RMSE at 1M in the Sep 9 table; ours 478 / 396 today, 775 / 806 on Sep 9
(Sep 9 rounds were interleaved with the opponent in one process, today's
were not, so the two ours rows are not an A/B of any single change).

### What flipped

- DEVIATION 2011, `HIST_ITEMS_PER_THREAD` default 1 -> 4
  (ensemble/decisiontree/batched_levelalgo/builder.mojo). Fingerprints
  18/18 equal at 1 and 4; medians lower at both rungs (1M 2475 -> 2274, 2M
  4037 -> 3743), and the candidate's whole range sits below the baseline's
  minimum at both rungs (2315 < 2385, 3761 < 3890), which exceeds the round
  spread. The flipped source rebuilt on the pod reproduces it (flip2011:
  18/18 equal, 2234 (2216..2277) at 1M against an adjacent baseline
  re-run of 2375 (2340..2406)); reach gate green with and without the new
  opt-out `-D MOJOLEARN_2011_HIST_ITEMS1=1`.

### What did not flip, with the numbers

- rf2010 (ROWS_SORTED): 2476 vs 2475 at 1M, 4229 vs 4037 at 2M. Negative.
- rf2012 (SMEM_COPIES 4): 2389 vs 2475 at 1M (86 ms inside the baseline's
  147 ms spread), 3957 vs 4037 at 2M (79 ms inside 245). Not a win by the
  spread rule; not flipped.
- DEVIATION 2030 fused walker move: 508 vs 478 at 1M, 781 vs 753 at 2M.
  Higher median at both rungs on the H100; the Sep 9 L40S 4%/2% win does
  not reproduce here. Stays OFF.
- single-pass IDENTICAL partition (define `MOJOLEARN_IDENTICAL_SINGLE_PASS_PARTITION`,
  which is main's replacement for the Sep 9 kernel_matrix patch; the patch
  no longer applies): sp 541/824, spnf 548/852 vs 478/753. Higher at both
  rungs; the 1M and 2M hashes equal baseline's on every set (dac2cf366e219cec,
  037aa3188d25c889, 2000f16a43380f98), so the arm IS identity-safe above
  500k rows per leaf on NVIDIA, and it is slower. Stays OFF; the patch file
  under bench/results/trees_identical/patches/ is superseded by the define.
- fold: already in baseline (4ccccde6 is an ancestor of deb01bcf); no
  separate set built.

### RUN OWED

- Apple M4 identity JSON regeneration after DEVIATION 2340 (orchestrator):
  `MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py --lanes rf-clf,rf-reg,et-clf,et-reg,gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse,kmeans --vendor apple-m4 --json bench/results/identity_break/apple-m4.identical.json`
  (expect the 18 `predict` cells to move by dtype only, `proba` equal).
- DONE 2026-09-10 (orchestrator, Apple M4, main at c061fe82): the rf
  binding rebuilt IDENTICAL at the flipped default;
  `rf_perf_candidates_check.mojo` ALL ARMS GREEN under
  `-D MOJOLEARN_NUMERIC_IDENTICAL=1`; `identity_break.py --lanes rf-clf,rf-reg`
  (`bench/results/identity_break/apple-m4.identical.rf-flip2011-2026-09-10.json`)
  diffed against the retained `apple-m4.identical.json`: rf-reg 9/9
  IDENTICAL, rf-clf `proba` 9/9 equal, `predict` differs on 9/9 by dtype
  only (today's int64 cast to the int32 label dtype reproduces the retained
  hash 9/9), the same picture as the H100. The flip is forest-identical on
  Apple. The full nine-lane Apple JSON regeneration above is still owed and
  should wait until DEVIATION 2340's `predict` dtype is settled (sklearn
  returns predictions in the `classes_` dtype; int64 is a behavior change,
  not just a fingerprint change).
- AMD execution and timing of every cell above (no AMD box this leg).
- 5M rungs: not started by directive.
- RF 2M at the flipped SOURCE (only rf2011's define build has the 2M cell):
  `sh tools/trees_identical_ab.sh speed flip2011 rf higgs 2000000 5 ours`.

## What landed (commits, `%h parent %p`)

- `609747b5 parent f3d76e8d` tools/trees_leg.sh (guarded RunPod session:
  rent, arm the on-pod watchdog first, ship source archive, ssh, extend,
  reap+verify) and tools/trees_identical_remote.sh (pod setup: pixi,
  IDENTICAL bindings base/gbdt/rf/trees, CatBoost, cuML, LightGBM built
  with USE_CUDA=ON, HIGGS).
- `4ccccde6 parent 609747b5`
  - core/pinned_reduce.mojo: `two_phase_halving_sum`, the same halving
    tree (same additions, same order) in 3 barriers instead of
    log2(block)+2; `halving_block_sum` calls it.
  - gbdt/targets/kernel/pointwise_targets.mojo: its IDENTICAL
    `pinned_block_sum` arm imports that fold instead of carrying a copy.
  - ensemble/randomforest.mojo, ensemble/decisiontree/batched_levelalgo/
    builder.mojo, .../kernels/builder_kernels_impl.mojo: ROWS_SORTED_SAMPLE,
    HIST_ITEMS_PER_THREAD, HIST_SMEM_COPIES_DEFAULT selected by
    `-D MOJOLEARN_2010_ROWS_SORTED=1`, `-D MOJOLEARN_2011_HIST_ITEMS4=1`,
    `-D MOJOLEARN_2012_SMEM_COPIES4=1`; shipped values unchanged.
  - bindings/build_rf.sh: MOJOLEARN_EXTRA_DEFINES pass-through.
  - tools/speed_gbdt_arm.py, bench/speed/forest_speed_arm.py: dataset
    `higgsreg` (HIGGS label as a float target, the RMSE cell).
  - tools/trees_identical_ab.sh: on-pod A/B helper (named binary sets,
    identity_break fingerprints and diffs, speed cells, RF reach gate).
- `a9ba6818 parent 4ccccde6` L40S artifacts, HIGGS via UCI static zip in
  the setup body, and bench/results/trees_identical/patches/
  kernel_matrix_single_pass_identical_nvidia.patch (checks/kernel_matrix.mojo
  is NOT edited; the kNN lane owns it).
- this commit: H100 artifacts and this file.

## Boxes

- L40S: NVIDIA L40S, driver 580.126.09, CUDA 12.4 image, catboost 1.2.10,
  lightgbm 4.7.0 (USE_CUDA=ON, probe ok), cuml 26.08.00, numpy 2.4.6;
  bench/results/trees_identical/l40s_2026-09-09/logs/{versions,gpu,setup}.txt.
- H100: NVIDIA H100 80GB HBM3, driver 580.126.09, same image and versions;
  bench/results/trees_identical/h100_2026-09-09/logs/{versions,gpu,setup}.txt.
  Source commit a9ba6818 on the box (setup.txt).

## Fingerprints (identity_break, IDENTICAL tier, 9 hostile fixtures x2 repeats)

Files: bench/results/trees_identical/{l40s,h100}_2026-09-09/ib/.
- baseline NVIDIA L40S vs Apple M4 JSON (bench/results/identity_break/
  apple-m4.identical.json, commit e616906e): 81/81 cells IDENTICAL for
  rf-clf, rf-reg, et-clf, et-reg, gbdt-symmetric, gbdt-depthwise,
  gbdt-lossguide, gbdt-rmse, kmeans (run locally, JSON diff only).
- fold (two_phase_halving_sum, base+gbdt rebuilt): 81/81 equal to baseline
  on both boxes (ib/diff.baseline.fold.txt).
- fused (`-D MOJOLEARN_2030_FUSED_EST_MOVE=1`): 36/36 equal (4 gbdt lanes)
  on both boxes (ib/diff.baseline.fused.txt).
- sp (fold + fused + the kernel_matrix patch): 36/36 equal on H100
  (h100 ib/diff.baseline.sp.txt). Note the patch's arm only runs above
  500,000 rows per leaf, which the 20,000-row fixtures never reach; the
  1M/2M prediction hashes below are the witness for that arm and they were
  NOT run (wind-down).
- rf2010 / rf2011 / rf2012: 18/18 equal each (rf-clf, rf-reg) on both boxes.
- rf_perf_candidates_check (reach gates, identical define, source and each
  candidate define): ALL ARMS GREEN on both boxes, logs/rfgate.*.log.

## Task 1: profile and reference table (IDENTICAL baseline, depth 6, 100 iters, lr 0.1, l2 1, 254 borders, no bagging, Plain)

Medians of the timed rounds, ms, Python surface wall (fit only, scoring
outside the timer). Logs under bench/results/trees_identical/<box>/speed/.

| box | cell | ours IDENTICAL | opponent (FAST) | log |
|---|---|---|---|---|
| L40S | symmetric Logloss higgs 1M, 7 rounds | 426 | catboost-gpu 781 | l40s .../speed/profile_cells_console.txt (console capture; the log files expired with the pod) |
| L40S | symmetric RMSE higgsreg 1M, 7 rounds | 318 | catboost-gpu 915 | same |
| H100 | symmetric Logloss higgs 1M, 7 rounds | 775 (697..1161) | catboost-gpu 900 (864..939) | h100 .../speed/baseline.gbdt-symmetric.higgs.r1000000.full.log |
| H100 | symmetric RMSE higgsreg 1M, 7 rounds | 806 (765..1313) | catboost-gpu 699 (680..759) | h100 .../speed/baseline.gbdt-symmetric.higgsreg.r1000000.full.log |
| H100 | rf higgs 1M, 5 rounds (100 trees, depth 16, sqrt, 128 bins, bootstrap) | 5762 (5116..6451) | cuml-rf-gpu 3314 (3257..3950); lightgbm-cuda 469654 (468691..471303, 3 rounds then budget) | h100 .../speed/baseline.rf.higgs.r1000000.full.log |

Quality on the same 500k test tail: Logloss ours 0.542067 / AUC 0.800716,
CatBoost GPU 0.542524 / 0.800431; RMSE ours 0.429173, CatBoost 0.429220.
Our hash is constant across rounds on both boxes (dac2cf366e219cec Logloss,
037aa3188d25c889 RMSE, the same on L40S and H100); CatBoost GPU's Logloss
hash changes every round.

H100 caveat: the Mojo fit wall inside those rounds is 251 ms (Logloss) and
169 ms (RMSE) per the stage clock, so 450-600 ms of every H100 round is
outside the Mojo fit (host-side Python/NumPy prep on a 208-core NUMA box
that prints tcmalloc mbind warnings); on the L40S the same gap is ~200 ms.
The opponent rows are measured on the same box in the same process, so
the ratio is still like for like, but the host tax is where the H100 rows
go before the device does.

Per-phase profile (MOJOLEARN_STAGE_TIMES=1, one untimed replicate, drains
per stage so it is not a benchmark), symmetric depth 6 higgs 1M:

| stage | L40S Logloss | H100 Logloss | H100 RMSE |
|---|---|---|---|
| sym.hist | 48.7 | 67.7 | 68.0 |
| sym.pstats | 9.7 | 16.0 | 16.0 |
| sym.score | 10.6 | 14.8 | 14.6 |
| sym.winner | 4.3 | 6.0 | 5.7 |
| sym.split | 39.2 | 45.5 | 45.3 |
| sym.drain | 1.0 | 1.5 | 1.5 |
| sym.leaves | - | - | 6.1 |
| est.move | 11.2 | 13.1 | - |
| est.approx | 15.8 | 16.6 | - |
| est.pstats | 26.5 | 22.8 | - |
| est.readback | 9.3 | 13.4 | - |
| accounted / fit wall | 176 / 212 | 217 / 251 | 157 / 169 |

Logs: <box>/speed/baseline.gbdt-symmetric.{higgs,higgsreg}.r1000000.stage.log
(L40S: profile_cells_console.txt). The Newton walker (est.*) is 63 ms of
251 on the H100; sym.hist and sym.split are the two largest phases.

RF 1M H100 quality (same 500k tail): ours logloss 0.538850 / AUC 0.809906,
cuml-rf-gpu 0.538814 / 0.809834, lightgbm-cuda 0.638510 / 0.754274 (rf
boosting with the 0.632 bagging LightGBM forces; 470 s per fit, so it
exhausts the 1800 s per-arm budget after round 3). Our hash is constant
(3ffa2951595422d4); cuML's is constant on this box too (a372de7ab27df595).
ET at every rung, RF 2M/5M: not run (wind-down; the ET 1M cell was killed
at its warm-up and its partial log was not kept).

H100 batch status at wind-down (logs/batchH.sh, logs/ab.txt): setup 16:37
to 16:44; PHASE_BUILDS (9 builds, 7 fingerprint sets, 4 reach gates) done
16:51; PHASE_TABLE_1M2M 5 of 8 cells done at 17:28 when the batch was
stopped (the lightgbm-cuda rf arm took 36 min of it); PHASE_SYM_AB (13
cells), PHASE_RF_AB (8 cells), the 5M rungs and the final stack (6 cells)
not started. Pod r8jua8lx0j0urj terminated 17:29 UTC, verified 404.

## Task 2: DEV 2030 fused walker move (`-D MOJOLEARN_2030_FUSED_EST_MOVE=1`)

Fingerprints equal (36/36, both boxes). Timing, L40S, ours only, medians
of 7 rounds at 1M and 5 at 2M (bench/results/trees_identical/
l40s_2026-09-09/speed/ was lost with the pod; the numbers are the FSPEED
lines read from the pod before it expired, transcribed here):
baseline 411 / 585, fold 336 / 570, fused (= fold + 2030) 322 / 557.
Fused is faster than fold at both rungs (4% / 2%), hashes equal
(dac2cf366e219cec at 1M on every set). H100 A/B: RUN 2026-09-10 (section above): fused 508/781 vs baseline
478/753 ms at 1M/2M, higher at both rungs, so FUSED_EST_MOVE_2030 STAYS
OFF on H100 evidence; the L40S 4%/2% did not reproduce.

## Task 3: halving_block_sum in three barriers

Landed (4ccccde6) and verified: 81/81 fingerprints equal on both boxes,
including kmeans through the base binding. L40S ours-only Logloss:
baseline 411 -> fold 336 ms at 1M, 585 -> 570 at 2M (same transcription
caveat as task 2). H100 timing not run.

## Task 4: RF candidates

Builds, fingerprints (18/18 each) and reach gates green on both boxes.
Timing at 1M/2M: RUN on the H100 2026-09-10 (section above): rf2011
(HIST_ITEMS_PER_THREAD 4) won both rungs and its default FLIPPED; rf2010
and rf2012 did not win.

## Task 5: structural phase

From the profile the largest phase that identity pays for is sym.split
(the 3-launch stable partition; 45 ms of 251 on the H100). The single-pass
decoupled-lookback partition (DEVIATION 1907) produces the same stable
permutation by construction; the patch under
bench/results/trees_identical/patches/ routes it under IDENTICAL on the
NVIDIA column only (Apple keeps the 3-launch path, AMD stays off). Built on
the H100 as set `sp` (fold + fused + patch): 36/36 fingerprints equal at
the 20k fixtures. 2026-09-10 (section above, define
`MOJOLEARN_IDENTICAL_SINGLE_PASS_PARTITION` replacing the patch): the 1M/2M
hashes equal baseline's (identity-safe) and the timing is higher at both
rungs (541/824 vs 478/753). Not landed; stays off.

## Rows for bench/OPPONENT_REFERENCE.md (orchestrator merges; do not create the file on this branch)

All on NVIDIA H100 80GB HBM3, driver 580.126.09, runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04,
HIGGS train = first N rows, test = last 500,000 rows, same process as our arm, ms median (min..max):

| opponent | version | lane, params | rows | median ms | log |
|---|---|---|---|---|---|
| catboost-gpu | 1.2.10 | SymmetricTree, Logloss, iters 100, depth 6, lr 0.1, l2 1, border_count 254, bootstrap No, Plain, seed 7 | 1M | 900 (864..939) | h100_2026-09-09/speed/baseline.gbdt-symmetric.higgs.r1000000.full.log |
| catboost-gpu | 1.2.10 | same, RMSE on the 0/1 label (higgsreg) | 1M | 699 (680..759) | h100_2026-09-09/speed/baseline.gbdt-symmetric.higgsreg.r1000000.full.log |
| cuml-rf-gpu | 26.08.00 | RandomForestClassifier, 100 trees, depth 16, sqrt features, 128 bins, bootstrap, seed 7, n_streams default | 1M | 3314 (3257..3950), logloss 0.538814 | h100_2026-09-09/speed/baseline.rf.higgs.r1000000.full.log |
| lightgbm-cuda | 4.7.0 (USE_CUDA=ON) | rf boosting, 100 trees, depth 16, 32768 leaves, bagging 0.632/1, feature_fraction sqrt, max_bin 255 | 1M | 469654 (468691..471303, 3 rounds), logloss 0.638510 | same |

L40S (same versions, driver 580.126.09): catboost-gpu 781 (771..788) Logloss 1M, 915 (849..941) RMSE 1M;
l40s_2026-09-09/speed/profile_cells_console.txt.

## RUN OWED on the Apple M4 (orchestrator)

The fold changed bytes in core/pinned_reduce.mojo and pointwise_targets.mojo,
and the RF constants moved behind defines. Apple identical fingerprints
must not move:

    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build.sh
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_gbdt.sh
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_rf.sh
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_trees.sh
    MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \
        --lanes rf-clf,rf-reg,et-clf,et-reg,gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse,kmeans \
        --vendor apple-m4 --json /tmp/apple_trees_lane.json
    python3 tools/identity_break.py --diff bench/results/identity_break/apple-m4.identical.json /tmp/apple_trees_lane.json
    python3 tools/identity_break.py --diff bench/results/trees_identical/h100_2026-09-09/ib/fold.json /tmp/apple_trees_lane.json
    tools/with_build_lock.sh pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 ensemble/checks/rf_perf_candidates_check.mojo
    pixi run check-fit-pointwise; pixi run check-logloss-train; pixi run check-ordered-boosting
    MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn --conformance   # or the usual identical card task

Expected: every cell IDENTICAL (the NVIDIA baseline already matched the
existing Apple JSON 81/81; the fold matched that baseline 81/81 on NVIDIA).

## Next commands for a fresh agent (in order)

    MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key tools/trees_leg.sh rent --gpu "NVIDIA H100 80GB HBM3" --minutes 90
    # then on the pod (see tools/trees_leg.sh ssh; the guard blocks compound ssh text, use `ssh <target> sh -s < script`):
    #   nohup sh tools/trees_identical_remote.sh > /root/trees_out/setup_console.log 2>&1 &   (about 7 min)
    #   scp the batch script; bench/results/trees_identical/h100_2026-09-09/logs/batchH.sh is the one that ran here:
    #   cut the lightgbm-cuda rf/et arms down (470 s per fit) by running rf/et cells with MOJOLEARN_SPEED_BUDGET_S=600, or --devices gpu with rounds 3
    #   remaining phases: PHASE_SYM_AB (baseline/fold/fused/spnf/sp, ours only, 1M x7 and 2M x5, higgsreg 1M), PHASE_RF_AB (baseline/rf2010/rf2011/rf2012, 1M/2M), et 1M/2M/5M full, rf 2M/5M full, sp full vs catboost at 1M/2M/5M
    #   the sp hash at 1M must equal dac2cf366e219cec (Logloss) and 037aa3188d25c889 (RMSE); 2M hashes must equal baseline's
    # flips, if the H100 confirms: FUSED_EST_MOVE_2030 default on (pointwise_oracle.mojo), the winning RF define into its comptime default, and the kernel_matrix patch handed to the kNN lane / orchestrator
    tools/trees_leg.sh reap

## Unfinished and why

- Every 2M/5M rung, ET at every rung, RF 2M/5M, the H100 A/B for fold,
  fused, sp and the RF candidates: the L40S pod expired (lease, the session
  was rate-limited mid-run) with the speed logs unfetched, and the H100 leg
  was wound down by the orchestrator while its first RF cell was in flight
  (lightgbm-cuda at 470 s per fit ate the phase).
- No flag flipped: nothing has an H100 timing with a log on disk.
