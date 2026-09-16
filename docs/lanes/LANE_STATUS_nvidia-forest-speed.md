# LANE nvidia-forest-speed

Goal as opened: make random forest and extratrees faster on NVIDIA at large
data, by porting LightGBM's histogram subtraction (build the smaller child's
histogram, derive the sibling as parent minus child) to the forest path.

**Verdict: do not build it. Three independent reasons, each measured.** The
lane also found that the comparison it was opened on is a category error, and
that our forest speed on NVIDIA was never missing.

No GPU was rented. **Spend for this lane: $0.00.** Section 6 says what a box
would still buy and what it would cost, so that call can be made separately.

---

## 1. The comparison the lane was opened on compares two different benchmarks

The brief read these two rows of `bench/OPPONENT_REFERENCE.md:36-47` as a
forest result:

```
LightGBM CUDA lossguide (num_leaves 2^depth, max_bin 255)   1313.7 ms
cuML RandomForest (100 trees, depth 16, sqrt, 128 bins)     3231.5 ms
```

They are not the same benchmark. The source logs settle it.

`bench/results/e1g/2026-08-28_030911-nvidia-speed-forest/remote/logs/` contains
`forest.gbdt-lossguide.r1000000.log` and `forest.et.r1000000.log` and **no
`forest.rf` log at all**. The 1313.705 ms cell is round 3 of that file:

```
FSPEED-HEADER family=forest lane=gbdt-lossguide arm=lightgbm-cuda mode=FAST device=NVIDIA_H100_80GB_HBM3 rounds=5
FSPEED lane=gbdt-lossguide arm=lightgbm-cuda shape=higgs-1000000x28 round=3 ms=1313.705
```

`lane=gbdt-lossguide` is **gradient boosting, 100 iterations at depth 6, lr
0.1** (`tools/speed_gbdt_arm.py::lane_config`). The cuML row beside it is
`lane=rf`, **100 trees at depth 16**. A depth-6 boosted ensemble builds about
100 x 2^6 nodes; a depth-16 forest builds up to 100 x 2^16. The two rows sit
next to each other in one table under one heading, which is how they came to
be read as one ladder.

**What LightGBM CUDA actually does on the forest lane**, five rounds
interleaved with ours and cuML's in a single process on one H100
(`bench/results/trees_identical/h100_2026-09-09/speed/baseline.rf.higgs.r1000000.full.log`):

| arm | rounds (ms) | median |
|---|---|---|
| ours (Sep 9 source) | 5116.3, 5762.3, 5551.5, 6425.8, 6451.3 | 5762 |
| cuml-rf-gpu | 3257.3, 3313.6, 3290.9, 3756.1, 3949.8 | 3314 |
| lightgbm-cuda | 471303.5, 469654.1, 468690.6 | **469654** |

LightGBM CUDA takes 470 seconds on the depth-16 forest lane, about 142x
cuML's cell. It has no fast forest number on this workload. So the premise
that subtraction is what makes LightGBM good at forests has the sign of the
evidence backwards: on the only forest cell we have measured for it,
LightGBM is the slowest arm in the table by two orders of magnitude.

The brief asked, if leaf-wise growth turned out to explain LightGBM's forest
number, to say so and stop rather than implement it. The answer is neither
leaf-wise nor subtraction: **there is no LightGBM forest number to explain.**

`bench/OPPONENT_REFERENCE.md` should carry the lane name on each row. That is
a one-line documentation fix and is listed in section 7 as not done here.

---

## 2. Our forest speed on NVIDIA was already measured

The brief said it "could find no forest speed row for OUR forest." The rows
exist under `bench/results/`; what is missing is that they were never copied
next to the opponents in `bench/OPPONENT_REFERENCE.md`.

HIGGS 1M x 28, 100 trees, depth 16, max_features sqrt, 128 bins, bootstrap,
seed 7, H100 80GB HBM3, our IDENTICAL arm:

| date | source state | median ms | file |
|---|---|---|---|
| 2026-09-09 | before native label encoding | 5762 | `trees_identical/h100_2026-09-09/speed/baseline.rf.higgs.r1000000.full.log` |
| 2026-09-10 | | 2475 | `h100_2026-09-10/speed/baseline.rf.higgs.r1000000.ours.log` |
| 2026-09-10b | native label encoding | 1516 | `h100_2026-09-10b/speed/baseline.rf.higgs.r1000000.ours.log` |
| 2026-09-11 | DEVIATION 2502 pure-node leaf | **1088** (1071..1108) | `h100_2026-09-11/speed/baseline.rf.higgs.r1000000.ours.pass1.log` |
| 2026-09-11 | same, second pass | 1081 (1051..1099) | `...ours.pass2.log` |

Our RF moved 5762 -> 1088 ms between Sep 9 and Sep 11 on committed changes.
The forest path has had six commits since Sep 11
(`36ec47446`, `71983ac0e`, `e032339f2`, `609954468`, `4d60d1b2f`,
`275143ac1`) and all six are Apple fence work, a mutex refactor or the
port-framing rename. None touches forest performance, so the Sep 11 profile
stands for current main.

---

## 3. Opponent context, each opponent's fastest cell, same process where we have it

Stating our number and theirs. The reader compares.

**The only same-process forest pairings we hold.** Ours interleaved with
cuML's round by round in one process on one H100, Istella-S 1M x 220 and 2M,
Sep 11, both arms IDENTICAL tier
(`trees_identical/h100_2026-09-11_istella/speed/baseline.rf.istella.r*.full.log`):

| shape | our median | cuML RandomForest median |
|---|---|---|
| Istella 1M | 3264.7 ms | 3500.0 ms |
| Istella 2M | 5103.0 ms | 5751.4 ms |
| HIGGS 1M (Sep 9 source) | 5762 ms | 3314 ms |

**Cross-session cells, which is a weaker pairing and is labelled as one.**
Our Sep 11 HIGGS 1M cell is 1088 ms; cuML's fastest committed HIGGS 1M cell
is 3231.5 ms (`e1g/2026-08-28_030908`, FAST-tier container) and 3314 ms
(Sep 9, same process as ours). Those are different containers on different
days, and `OPPONENT COLUMNS DRIFT POD TO POD` applies, so this pairing is
recorded and not leaned on.

**Every opponent's fastest cell on the forest lane, HIGGS 1M**, so that no
single opponent is quoted alone:

| opponent | fastest forest cell | source |
|---|---|---|
| cuML RandomForest 26.8.0 | 3231.5 ms | `e1g/2026-08-28_030908` |
| LightGBM CUDA 4.7.0 | 469654 ms | `h100_2026-09-09` rf lane |
| scikit-learn RandomForest (CPU) | 69837 ms | `lane=rf arm=sklearn-rf-cpu`, HIGGS 1M |
| XGBoost GPU | no forest cell measured | -- |
| CatBoost GPU | no forest cell measured | -- |

Every arm that has ever run the `rf` lane in this repository is in that
table: a grep across `bench/results/` for `FSPEED lane=rf` returns only
`ours`, `ours-ab`, `cuml-rf-gpu`, `cuml-rf-gpu-stream`, `lightgbm-cuda`,
`lightgbm-cpu` and `sklearn-rf-cpu`. XGBoost and CatBoost have never run it.
Their 617.3 ms and 846.1 ms cells in `OPPONENT_REFERENCE.md` are `gbdt-*`
lanes at depth 6, so they are recorded here as absent rather than borrowed.
The two CPU arms are listed for completeness and are not the comparison:
on a GPU box the harness itself declines them
(`FSPEED-NOTE ... a lane whose only opponent is a CPU library has NO legal
opponent here`).

ExtraTrees has no GPU opponent at all: `bench/speed/forest_speed_arm.py`
runs the `et` lane ours-only, and LightGBM's `extra_trees` arm is recorded
INVALID in `OPPONENT_REFERENCE.md` (181 s / 227 s single samples, plus a
build refusal). Our ET cells, FAST tier, Aug 28: 4349 ms at 1M, 7316 at 2M,
14826 at 5M.

---

## 4. The profile, and the histogram fraction

From `MOJOLEARN_STAGE_TIMES=1` (`ensemble/instruments.mojo`), H100, our
IDENTICAL arm, round 1, 100 trees at depth 16.

**Read this with the instrument's own caveat.** `stop()` drains the device
queue to close each stage, which serializes work that normally overlaps, so
a staged run is not a certifiable timing and the stage sum exceeds
`fit_total` (the table's `other` row is -3.9% on HIGGS and -5.7% on Istella).
The fractions below are therefore approximate. They are used to decide
whether an optimization is worth attempting, which is what they can carry;
they are not quoted as timings.

| | HIGGS 1M x28 | Istella 1M x220 |
|---|---|---|
| `fit_total` | 0.9407 s | 1.2332 s |
| `device_wait` (every GPU kernel, one bucket) | 0.4540 s, **48.3%** | 0.8568 s, **69.5%** |
| host, everything else | 0.5233 s, 55.6% | 0.4467 s, 36.2% |
| of which enqueue and queue management | 0.3271 s, **34.8%** | 0.2446 s, 19.8% |

Top host stages, HIGGS 1M: `host_enq_partition` 11.9%, `host_queue_push`
11.8%, `host_enq_hist` 11.1%, `leaf_values` 7.5%, `tree_copy` 3.7%.

`device_wait` is a single bucket; the stage table does not separate histogram
construction from split evaluation from partition inside it. The closest
per-kernel split we hold is the ExtraTrees rocprof run on MI300X
(`bench/results/e1/2026-08-29_211714-mojolearn-e2-amd/lanes/et_profile/rocprof_tpbshipped/et_tpbshipped_kernel_stats.csv`),
where two kernels are 97.8% of GPU time: 52.87% and 44.92%. Taking ~53% as
the scoring or histogram share of device time is the generous reading, and
it is a different vendor and a different estimator, so it is used only as an
upper-ish bound.

**Histogram construction is therefore about 0.483 x 0.53 = 25.6% of
`fit_total` on HIGGS 1M.** Its absolute ceiling, if histograms were every
kernel in `device_wait`, is 48.3%.

The headline the profile actually delivers is a different one: **on HIGGS 1M
more than a third of our forest fit is host-side enqueue and queue
management, not GPU work at all.** Section 6 returns to this.

---

## 5. Why subtraction does not apply, and the prediction stated before any build

### 5a. The arithmetic premise holds

Every forest histogram field is an integer, so `parent - child` would be
exact, with none of the drift LightGBM's float histograms take:

| struct | `ensemble/decisiontree/batched_levelalgo/bins.mojo` | fields |
|---|---|---|
| `ClassificationBin` | `:316` | `count: UInt32` |
| `WeightedClassificationBin` | `:412` | `count: UInt32`, `weight: Int32` |
| `RegressionBin` | `:500` | `label_sum: Int32`, `count: UInt32` |
| `WeightedRegressionBin` | `:605` | `label_sum: Int32`, `count: UInt32`, `weight: Int32` |

The atomics are integer on every path (`bins.mojo:379, 467, 558, 682`, all
`Atomic.fetch_add`). Checked per objective: gini, entropy, MSE, poisson,
gamma and inverse-gaussian all resolve to one of the four bins above; MAE is
not implemented on GPU (it falls to cuML's `default:` arm, deliberately).
**No forest objective uses a float accumulator.** Floats appear only after
the histogram, in the gain arithmetic and in dequantization on read.

ExtraTrees is a separate copy with no bin histogram at all: it draws one
random threshold per node and feature and accumulates a left/total pair
(`extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:1140`),
in Int32, with regression labels pre-quantized (DEVIATION 135). Its only
float accumulation path is behind `SCORE_SAB_FLOAT_ACCUM`, a sabotage hook,
never shipped.

So the brief's claim that subtraction would be bitwise identical for us
**is confirmed on every path**, for RF and ET alike. It is the next premise
that fails.

### 5b. Blocker one, measured: a parent does not hold its children's columns

The per-node feature sampler seeds on the node's index **in the tree**:

```
rng_seed = fnv1a32_hash(seed, treeid, nodeid)    # builder_kernels.cuh:88
```

with `nodeid = work_items[node_idx].idx`
(`ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels.mojo:211,
216-221`). A parent and its two children are three different tree indices, so
they draw three independent column samples. The histogram is indexed by the
node's own column **slot**, not a global feature id
(`kernels/builder_kernels_impl.mojo:2457`,
`column_samples[nid * n_sampled_cols + slot]`), so slot `j` of the parent and
slot `j` of a child are not even the same feature. ExtraTrees seeds the same
way (`extratrees/.../kernels/builder_kernels.mojo:304, 412`).

`ensemble/checks/sibling_column_overlap_check.mojo` measures the consequence
over 4096 parent/child triples. Output in
`bench/results/forest_subtraction/sibling_column_overlap.txt`:

```
  n_features  k=max_features  mean overlap  ceiling  independent-draw
   28           5              0.903            0.181       0.893
   28           6              1.290            0.215       1.286
   28           14             7.023            0.502       7.000
   54           7              0.902            0.129       0.907
   100          10             1.004            0.100       1.000
   784          28             0.999            0.036       1.000
   28           28             28.0             1.000       28.0
```

**On HIGGS's shape at max_features sqrt, a child shares 0.90 of its 5 columns
with its parent: 18.1%.** That is a hard ceiling on the fraction of a child's
histogram subtraction could ever supply. The measured overlaps track `k*k/n`,
the independent-draw expectation, on every row, which is the signature of
three unrelated samples.

The last row is the exception and is kept so this is recorded as a
conditional result: at `max_features = 1.0` every node draws a permutation of
the same full set, the overlap is total, and subtraction would be available
behind a column remap. That is not the forest default, not the benchmarked
configuration, and is the setting in which a forest decorrelates least.

**Sabotage, seen to fail.** `-D FOREST_OVERLAP_SAB_SHARED_SEED=1` drops
`nodeid` from the seed, which is exactly the counterfactual world in which
subtraction works. Every row moves to an overlap of exactly `k`, ceiling
1.000. The differing cells, not a count
(`bench/results/forest_subtraction/sibling_column_overlap.diff.txt`):

```
<    28           5              0.903076171875      0.180615234375
<    28           6              1.290283203125      0.21504720052083334
<    28           14             7.0228271484375     0.5016305106026786
<    54           7              0.9024658203125     0.12892368861607142
<    100          10             1.0037841796875     0.10037841796875
<    784          28             0.9991455078125     0.03568376813616071
---
>    28           5              5.0                 1.0
>    28           6              6.0                 1.0
>    28           14             14.0                1.0
>    54           7              7.0                 1.0
>    100          10             10.0                1.0
>    784          28             28.0                1.0
```

The check also carries a liveness assertion (the `k == n` row must be exactly
`k`, and the run raises otherwise), so a run that reached nothing refuses
rather than printing a clean zero.

### 5c. Blocker two: the parent histogram does not survive to the children

`N_BLKS_FOR_COLS = 10` (`builder.mojo:78`). Only ten columns are resident at
a time; the column loop advances `c += N_BLKS_FOR_COLS`
(`builder.mojo:2054-2061`) and **zeroes the live prefix before each round**
(`builder.mojo:1840-1852`). A parent's histogram is gone before its children
are built.

### 5d. Blocker three: the histogram is destroyed in place

`find_best_splits_kernel` runs `pdf_to_cdf` over the histogram buffer in the
same memory (`kernels/builder_kernels_impl.mojo:2846-2852`), converting the
PDF to a prefix sum. Even within one level there is no PDF left to subtract
from. Retaining one would mean a second buffer and a second write of every
histogram, which is new memory traffic charged against the saving.

Grep confirms no subtraction exists today anywhere under `ensemble/` or
`extratrees/`. The near-miss hits are `LabelSumMinus` (`bins.mojo:298-306`),
which is right-child-from-prefix-sum **within one node's own CDF**, and the
OOB set as the complement of a bootstrap draw. Neither is the trick.

### 5e. The prediction, stated before any build, and no build followed

Histogram construction is ~25.6% of `fit_total` (section 4). Subtraction
skips building only the **larger** sibling, which holds somewhat more than
half the parent's rows; take 0.6 in expectation. It works only on the 18.1%
of columns the parent shares.

```
predicted saving = 0.256 x 0.6 x 0.181 = 2.8% of fit_total
```

**Under 3%, before paying for the retained parent plane and for moving
`pdf_to_cdf` out of the histogram buffer, either of which could exceed it.**

Two sanity bounds around that number. The most generous possible accounting,
histograms as 100% of `device_wait` and overlap at 100%, is 0.483 x 0.6 =
29%; substituting the measured overlap collapses it to 2.8%. At
`max_features = 1.0`, where overlap is total, the prediction would be
0.256 x 0.6 x 1.0 = 15%, which is material but is not the forest default.

This project withdrew a speedup claim of one part in 870 today for being
unreplicated. A predicted 2.8% with three structural blockers in front of it
is not worth a build, so **no "after" measurement exists and no flag was
added.** That is the lane's answer.

---

## 6. Where forest speed work should go instead

Not a recommendation this lane acted on, and not measured beyond section 4.

On HIGGS 1M, `host_enq_partition` + `host_queue_push` + `host_enq_hist` are
**34.8% of `fit_total` and are pure host CPU**. `device_wait` is 48.3%. A
forest lane aimed at the host enqueue path is pointed at a larger share of
the clock than any histogram change could reach, and it is vendor-neutral.

The Istella column is the counterweight and is why this is phrased as a
direction and not a conclusion: at 220 features `device_wait` rises to 69.5%
and enqueue falls to 19.8%. The host cost scales with node count, not feature
count, so it dominates exactly where the trees are deep and narrow. A real
lane here needs the per-kernel split inside `device_wait` on NVIDIA, which
we do not have.

**What a rented box would still buy, and its price.** One H100 hour would
give the first `nsys` per-kernel forest profile on NVIDIA, separating
histogram from split from partition inside `device_wait`, on the live
datasets rather than retired HIGGS. `bench/speed/nvidia_identical_trees.py`
already has `--nvtx` and the documented `nsys` invocation.
`tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60` with
`MOJOLEARN_GEMM_LEG_EXTRA` is the runner that stages from R2, uses bincache
and terminates and verifies. Estimated $2 to $5. It was **not** rented,
because it cannot change this lane's verdict and the lane's stopping rule
was to report rather than spend. That call is left open.

---

## 7. What this lane does not cover

- **No GPU was rented and no new timing was taken.** Every number here is
  either committed evidence from H100 legs dated Sep 9 to Sep 11, or a host
  computation on the Mac. The claim that the Sep 11 profile stands for
  current main rests on reading six commits, not on re-running.
- **The histogram fraction inside `device_wait` is inferred, not measured on
  NVIDIA.** The ~53% split comes from an ExtraTrees rocprof run on MI300X.
  It is a different vendor and a different estimator. The lane's conclusion
  does not turn on it (the column-overlap ceiling alone is decisive), but the
  2.8% figure does.
- **The stage table is not a certifiable timing.** It drains per stage; its
  rows sum to more than `fit_total`.
- **`bench/OPPONENT_REFERENCE.md` is not fixed here.** Its forest section
  still lists `gbdt-lossguide` and `rf` cells under one heading with no lane
  column, which is what produced this lane's founding comparison. Our forest
  rows are still absent from it. Both are left for whoever owns that file.
- **`identity_break.py` was not run.** The brief's step 4 gates a feature
  flag that was never added; there is no arm to prove identical. The
  arithmetic claim it would have tested is instead settled at the type level
  in section 5a.
- **ExtraTrees was analysed but not profiled on NVIDIA.** Its ceiling is
  lower than RF's, not higher: it has no bin histogram to subtract, and its
  one random threshold per node and feature is redrawn per node, so there is
  nothing a parent could hand a child even if the columns matched.
- **One-box, one-vendor.** No AMD or Apple column was taken; none was needed,
  because the blocker is in the host-side algorithm and is vendor-neutral.

---

## Files this lane added

- `ensemble/checks/sibling_column_overlap_check.mojo` -- the overlap
  measurement, its sabotage arm and its liveness assertion.
- `bench/results/forest_subtraction/sibling_column_overlap.txt` -- clean arm.
- `bench/results/forest_subtraction/sibling_column_overlap.sabotage-shared-seed.txt`
  -- sabotage arm.
- `bench/results/forest_subtraction/sibling_column_overlap.diff.txt` -- the
  differing cells.

No shipping code was changed.
