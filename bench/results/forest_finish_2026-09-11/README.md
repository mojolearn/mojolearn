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

Pending: batch C's builds, identity and A/B, batch D's verdict, and (only if a
flip is on the table) batch F's two regression cells.

Big logs are outside the repo in
`~/mojolearn-evidence/forest-finish-2026-09-11/`.
