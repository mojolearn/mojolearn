# Lane forest-finish, H100 leg, 2026-09-11 night

The leg that BUILT and RAN what lane/forest-speed left as source: DEVIATION
2637 (RF and ET row-major staging into the pinned buffer across the host
pool), DEVIATION 2638 (the isolation forest lends X by address), and a new
DEVIATION 2663 trial (the ExtraTrees frontier batch width).

## The box

| fact | value |
|---|---|
| pod | `8gsem9f3thnhvu` (RunPod SECURE, reaped at the end of the leg) |
| GPU | NVIDIA H100 80GB HBM3, driver 580.126.09, `GPU-b645d4a6-3c99-a75a-6492-0b26a9929031` |
| CPU | Intel Xeon Platinum 8470, 208 logical CPUs visible, cgroup quota 22.1 CPUs, joblib `cpu_count` 23 |
| container | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| opponents | cuML 26.08.00, scikit-learn 1.9.1, numpy 2.4.6, CatBoost 1.2.10, LightGBM 4.7.0 |
| tier | IDENTICAL (`MOJOLEARN_NUMERIC_MODE=identical`) everywhere |
| rows | 1,000,000 training rows on both datasets, 100 trees, depth 16, sqrt features, 128 bins for RF, bootstrap for RF only, seed 7 |

## The binary sets (`/root/bins/<set>`, swapped into `python/mojolearn/identical/`)

| set | what it is |
|---|---|
| `baseline` | main `4dc4346a`, the commit this lane's merge took: `_mojolearn_rf.so`, `_mojolearn_trees.so`, `_mojolearn_svm.so` built in a second checkout (`/root/mojolearn_main`) on this pod. `_mojolearn.so` and `_mojolearn_gbdt.so` are byte-identical source on both sides and come from the setup build. |
| `rowmajor` | lane/forest-finish `600dcec0` (DEVIATIONS 2637 and 2638), all five extensions |
| `ctl` | `rowmajor` with `_mojolearn_trees.so` rebuilt from the DEVIATION 2663 source at its default width (4096), the A/B control |
| `stats` | `ctl` plus `-D MOJOLEARN_ET_CYCLE_STATS=1` (level cycles, searched nodes, DEVIATION 205 surveys) |
| `bw16k` / `bw32k` | `-D MOJOLEARN_ET_DEVICE_BATCH_16384=1` / `_32768=1` |

## The batches (`logs/batch{A,B,C,D}.sh`, run on the pod in that order)

- **A** builds both sets, runs `identity_break` on each and the diff, `check-if`
  under IDENTICAL, a non-finite refusal probe through the Python surface on
  both sets, then the same-pod speed cells: pair 1 full (opponent interleaved,
  main then lane), pair 2 ours-only in the reverse order (ABBA), taxi first and
  Istella-S after the download, then one untimed stage replicate per cell.
- **B** proves REACH (the `*_rowmajor` entries are called once for a C-order
  float32 fit and never for an F-order one) and splits the Python-side host
  time on Istella-S for both sets.
- **C** builds the DEVIATION 2663 sets, gates identity against `rowmajor`, runs
  `device_batched_check`, prints the cycle stats, and runs the rotated
  ours-only A/B on both datasets.
- **D** runs `tools/flip_verdict.py` for every switch.

## What DEVIATION 2663's switch reaches, and what a flip therefore owes

The batch width is set in `extratrees/estimator.mojo::resolve`, which BOTH the
ExtraTrees classifier and the ExtraTrees regressor take, so under the
one-default-per-switch rule (ENGINEERING_RULES.md section 9) its time gate is
the geometric mean over every (lane, dataset) cell it reaches, and quality must
not be worse in any of them. That is four cells, not two: `et` on taxi and
Istella-S (classification, `max_features='sqrt'`, 4 and 14 sampled columns) and
`et` on `taxireg` and `istellareg` (regression, `max_features=1.0`, so 11 and
220 sampled columns per node). The harness runs all four: `load_dataset` knows
the two `*reg` names and `our_et_arm` branches on `data.task`.

Istella-S regression samples every one of the 220 columns per node, so its
cells cost several times a classification fit. They are therefore run ONLY if
batch C's classification A/B puts a flip on the table; if the classification
geomean is not below 1 there is nothing to flip and nothing to spend the pod
time on. A flip claimed on the classification cells alone would be a switch
decided on half the lanes it reaches, which is exactly what section 9 forbids.

## Owed elsewhere, closed here

Setup's `tools/check_buffer_foreign_argtypes.py --real-cuml` exited 0 on this
pod at 21:58:54Z (`logs/buffer_foreign_argtypes_cuml.log` in the evidence
tarball): mojolearn imported beside real cuML and treelite, one small RF fit
and predict, no ctypes argtypes clash. That is the 0.8.1 check the release
notes still carried as owed on NVIDIA.

## Results: DEVIATIONS 2637 and 2638

Same pod, same process per cell, arms alternating, 1 warm-up plus 3 rounds,
ms median. BEFORE is main `4dc4346a` built on this pod, AFTER is this lane.

| family | dataset | opponent and device | opponent ms | before ms | after ms | after/before | ours/opponent after | quality |
|---|---|---|---|---|---|---|---|---|
| RandomForest | taxi | cuML 26.08.00, GPU | 1860 | 826 | 811 | 0.98 | 0.44x | logloss 0.525910 both |
| RandomForest | Istella-S | cuML 26.08.00, GPU | 3885 | 1977 | 1333 | 0.67 | 0.34x | logloss 0.145560 both |
| ExtraTrees | taxi | scikit-learn 1.9.1, CPU 23-core quota | 3295 | 1894 | 1831 | 0.97 | 0.56x | logloss 0.527541 both |
| ExtraTrees | Istella-S | scikit-learn 1.9.1, CPU 23-core quota | 15502 | 5782 | 4947 | 0.86 | 0.32x | logloss 0.188191 both |
| IsolationForest | taxi | cuML 26.08.00, GPU | 54.4 | 290 | 95.4 | 0.33 | 1.75x | proxy AUC 0.553631 both |
| IsolationForest | Istella-S | cuML 26.08.00, GPU | 1001 | 4456 | 155 | 0.035 | 0.16x | proxy AUC 0.821218 both |

Section 9 geometric means of after/before: RandomForest 0.81, ExtraTrees 0.91,
IsolationForest 0.11.
BOTH PASSES, because one pass is one measurement. Each cell was run twice: an
interleaved pass against the opponent (the table above) and an ours-only pass
in the reverse set order (ABBA). after/before per pass, interleaved then
ours-only: RF taxi 0.98 / 0.93, RF Istella-S 0.67 / 0.60, ET taxi 0.97 / 1.03,
ET Istella-S 0.86 / 0.86, iforest taxi 0.33 / 0.36, iforest Istella-S
0.035 / 0.034. Every cell holds its hash in both passes.

ET ON TAXI IS FLAT, AND THE TWO PASSES DISAGREE ON ITS SIGN (0.97 against
1.03), so it is noise around 1 and not a win: 16 columns of staging is not
where a taxi ExtraTrees fit spends its time. ET's geometric mean is below 1
either way (0.91 with the interleaved pass, 0.94 with the ours-only one)
because Istella-S carries it at 0.86 in both. RF's taxi cell is the same story
one notch milder (0.98 / 0.93).
 Quality is byte-equal before and after in every cell, so
all three pass the flip gate. These are not opt-in switches: 2637 and 2638 are
the shipped path on this branch, and these rows are what keeps them.

The isolation forest is where the staging mattered most, because its fit was
the one that copied the matrix three times in one thread: Istella-S 4456 ms to
155 ms is 3.5 percent of the old time, and it turns a cell we lost to cuML
(4456 against 1001) into one we win (155 against 1001). RandomForest's taxi
cell barely moves (0.98) because 16 columns of staging is not where its time
goes; Istella-S, at 220 columns and 880 MB per pass, is (0.67).

Hashes held one value in 3 of 3 rounds and are equal before and after: RF taxi
`d8f64dae01de00bd`, RF Istella-S `574b24d0d7af51d0`, ET taxi
`e683f121d11f59dd`, ET Istella-S `40b1c5b03ba40420`, iforest taxi
`6f68d48431290524`, iforest Istella-S `a1902225f8730abf`.

## Where the win comes from, line by line (Istella-S host split)

`tools/forest_host_split.py`, 3 repetitions after a warm-up, both sets, the
Python side of one fit wrapped call by call. Diagnostic clocks, not certifiable
timings; the FSPEED rows above are those.

| lane | set | `as_f32_colmajor` | inside the native call (`unwrapped`) | `fit_total` |
|---|---|---|---|---|
| RF | baseline (main) | 554..680 ms | 1484..1513 ms | 2072..2178 ms |
| RF | rowmajor (lane) | NOT CALLED | 1293..1303 ms | 1299..1309 ms |
| ET | baseline (main) | 558..571 ms | 5160..5193 ms | 5735..5748 ms |
| ET | rowmajor (lane) | NOT CALLED | 4945..4971 ms | 4950..4976 ms |

Two costs, both removed, and they add up to the FSPEED deltas rather than
merely accompanying them. First, `as_f32_colmajor` -- the one-thread Python
transpose of an 880 MB block -- is GONE from the lane's fits: the column-major
entry is never reached for a C-order float32 X, so there is no line to report
(the REACH table above is the other half of that statement). Second, the native
call itself drops about 200 ms on RF (1484..1513 to 1293..1303), which is the
second one-thread pass: the stage copy now runs across the host pool.

`encode_labels` (5 to 7 ms) and `export_fit_result` (24 to 48 ms) are unchanged
on both sets, which is what a staging change should leave alone.

## REACH: the row-major entries are the ones that ran

An unchanged model hash proves nothing if the new path was never taken, so the
probe wraps every `*_fit*` entry of the binding an estimator actually calls
(`_bind` is `_backend.binding(name, mode)`) and counts the calls. Lane build,
50,000 x 32 float32, one fit per estimator per layout:

| input layout | estimator | entry called (once) | predict hash |
|---|---|---|---|
| C-order | RandomForestClassifier | `rf_classifier_fit_rowmajor_export` | `0c098cc86bfc87d7` |
| C-order | ExtraTreesClassifier | `et_classifier_fit_rowmajor_export` | `fded7b6d0467eaef` |
| C-order | RandomForestRegressor | `rf_regressor_fit_rowmajor_export` | `ac736dfceab2e31b` |
| C-order | ExtraTreesRegressor | `et_regressor_fit_rowmajor_export` | `b29e30335566a2d9` |
| F-order | RandomForestClassifier | `rf_classifier_fit_export` | `0c098cc86bfc87d7` |
| F-order | ExtraTreesClassifier | `et_classifier_fit_export` | `fded7b6d0467eaef` |
| F-order | RandomForestRegressor | `rf_regressor_fit_export` | `ac736dfceab2e31b` |
| F-order | ExtraTreesRegressor | `et_regressor_fit_export` | `b29e30335566a2d9` |

Each row is exactly one call: the C-order fits never touch the column-major
entry and the F-order fits never touch the row-major one, which is the routing
`as_f32_forest_layout` promises. The predict hash is the SAME for both layouts
of the same estimator, so the two entries build the same model from the same
data in either layout. Batch E repeats this against the `baseline` set as the
control, where the row-major entries do not exist and both layouts must take
the plain ones.

## Where the ExtraTrees Istella-S fit actually spends its time

One untimed stage replicate on the lane's build (`speed rowmajor et <ds>
1000000 1 stage`, serialized by measurement, so the total runs long against an
untimed fit). Istella-S, then taxi:

| phase | Istella-S s | share | taxi s | share |
|---|---|---|---|---|
| score pass (init+score+finalize) | 2.258 | 45% | 0.656 | 34% |
| range pass (init+range+decode+nonconst) | 1.621 | 32% | 0.458 | 24% |
| stage + feature sampler | 0.668 | 13% | 0.448 | 23% |
| partition (4 kernels) | 0.315 | 6% | 0.246 | 13% |
| leaf pass | 0.058 | 1% | 0.041 | 2% |
| candidate + reduce + splits readback | 0.026 | 1% | 0.019 | 1% |
| host: queue push | 0.039 | 1% | 0.026 | 1% |
| host: split records + pop/assembly + setup | 0.025 | <1% | 0.014 | <1% |
| total (device loop) | 5.009 | | 1.907 | |

Outside the loop, `boundary_dataset_upload` is 0.501 s on Istella-S and 0.442 s
on taxi (that is DEVIATION 2637's staging, already the fast path here), and the
whole binding call is 6.44 s / 2.57 s under the clock.

THIS IS EVIDENCE AGAINST MY OWN DEVIATION 2663 HYPOTHESIS, recorded before its
A/B finished. The batch width was widened on the argument that a 4096-node
frontier runs many level cycles and each ends in a drain plus a host pass, so
fewer, wider cycles would pay. On this H100 that whole family of costs --
reduce readback, split records, pop and batch assembly, queue push -- is about
3 percent of the loop (0.09 s of 5.0 s on Istella-S). Two thirds of the time is
the range and score passes, which read the same cells whatever the batch width.
What a wider batch can still move is `stage + feature sampler` (13 percent),
which is per cycle, though it also makes each cycle's staging compare cover a
larger capacity. So the honest prior is a small effect, and the A/B decides it.

## Results: DEVIATION 2663

### Identity first: the width moves no bit

Each trial set is the `rowmajor` set with ONLY `_mojolearn_trees.so` rebuilt,
and the three binaries differ, so the defines reached the compiler rather than
producing one binary three times: `ctl` `ba521b20024795ca...`, `stats`
`237c8956dd3bdd0d...`, `bw16k` `6421f3c105f11ef2...` (`bw32k` built too).

`identity_break` rf-clf, rf-reg, et-clf, et-reg, iforest read 45 of 45 cells
stable on `ctl`, on `bw16k` and on `bw32k`, and each set's diff against
`rowmajor` carries 46 IDENTICAL rows with no DIVERGENT, MOVED or REFUSED. The
ExtraTrees fingerprints are equal at every width (et-clf/base
`c586b27a3b049614`, et-clf/wide `ee8b318d6bf698b2`, et-reg/base
`754d8c127ecfc04d`, et-reg/wide `b745e53515f59cac`), which is what the
scheduling-parameter argument predicted and what `device_batched_check`'s
"max_batch_size=3 must not move a tree" already guarded. So 2663 is purely a
speed question.

### Speed

Pending: batch C's A/B, batch D's verdict, and (only if a flip is on the table)
batch F's two regression cells.

Big logs are outside the repo in
`~/mojolearn-evidence/forest-finish-2026-09-11/`.
