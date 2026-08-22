# WINDOW 2026-08-22 — extratrees DEVIATION 211: the batch spans trees

One window, one box (Apple M4, 10 cores), arms alternated INSIDE one process
per rep — the discipline DEVIATION 208 paid for. The perf deferral on the
extratrees lane was lifted by the repo owner on 2026-08-22 ("please continue
improving performance for extra trees... I think we should parallelize to
use gpu"); this is the first window taken under it.

## What changed

`fit_classification_device` / `fit_regression_device` run the whole forest
through ONE merged frontier: every in-flight tree's level batch shares the
same launches, readbacks and synchronize points, and the tree id rides per
work item into the kernels. cuML's cross-tree parallelism is a CUDA stream
pool (`randomforest.cuh:336-341`); Metal has no streams, so the overlap is
expressed as a WIDER GRID instead. Full mechanism and gate:
`extratrees/DEVIATIONS.md` entry 211; `device_batched_check` holds the merged
forest to one-tree builds node for node and leaf bit for leaf bit.

## A/B: merged vs the per-tree loop (the old fit body), same binary

`extratrees/bench/batched_ab.mojo`. Both arms in one process, alternated,
3 reps, forest DIGESTS asserted identical on both arms and every rep — every
row below produced bit-identical forests. covtype, depth 12, seed fixed.

| config | serial ms | merged ms | ratio |
|---|---|---|---|
| clf 100,000 rows, 10 trees, sqrt | 572–633 | 140–142 | **4.06–4.46x** |
| clf 100,000 rows, 100 trees, sqrt | 5,553–5,707 | 1,062–1,085 | **5.12–5.37x** |
| clf 581,012 rows (SHIPPED SIZE), 10 trees, sqrt | 966–1,143 | 552–617 | **1.72–1.85x** |
| reg 100,000 rows, 10 trees, all | 422–625 | 239–264 | **1.76–2.37x** |
| reg 581,012 rows (SHIPPED SIZE), 10 trees, all | 1,399–1,509 | 1,229–1,235 | **1.13–1.23x** |

The shape is exactly the starvation diagnosis: the win is largest where the
grid was starved (small n, narrow `sqrt` grids, many trees) and smallest
where the kernels already filled the device (`all` at 581k). The ledger's
old ~11%-at-581k pricing understated it because merging also divides the
per-batch synchronize/readback floor, not just the root-level grid width.
The rows above 100,000 are the shipped size; the 100,000-row rows are
smaller runs, flagged as such per the large-data rule, included because
small-n is the regime this change targets.

## vs scikit-learn 1.9.0, interleaved in one process

`extratrees/bench/sklearn_interleaved.mojo`, unchanged harness: sklearn
`n_jobs=None` (the parameter-matched default), sklearn `n_jobs=-1` (all 10
cores), ours-gpu, alternating per rep. Train accuracy / MSE printed per rep
by the harness and in band on every row (ours-gpu accuracy was equal or
higher on most reps; node counts comparable). Ranges over 3 reps:

| config | vs sklearn 1 core | vs sklearn 10 cores |
|---|---|---|
| clf 100,000 rows, 10 trees, sqrt, d12 | 1.04–1.39x | 0.48–0.52x |
| clf 100,000 rows, 100 trees (sklearn default), sqrt, d12 | 2.33–2.99x | 0.72–0.86x |
| clf 581,012 rows, 10 trees, sqrt, d12 | 3.87–4.15x | **1.25–1.36x** |
| clf 581,012 rows, 100 trees (sklearn default), sqrt, d12 | 4.93–7.94x | **1.53–1.61x** |
| reg 581,012 rows, 10 trees, all, d12 | 9.21–9.44x | **3.17–3.22x** |

**The classification row FLIPPED.** Before this window the lane's standing
fact was 0.06–0.29x of sklearn's 10 cores at 100k rows and a loss at `sqrt`
at full size. At the shipped size, classification now beats all ten of
sklearn's cores — 1.25–1.36x at 10 trees, 1.53–1.61x at sklearn's default
100 trees — and at 100k it beats their single-core default arm and stands at
0.72–0.86x of ten cores at 100 trees. Ours amortizes with tree count
(per-tree cost FELL from 18.5 to 11.5 ms/tree at 100k as trees went 10 to
100) while sklearn's is flat per tree, so the sklearn-default configuration
is the flattering one for us and it is also simply THE default.

Regression at full size stands at 3.17–3.22x of ten cores in this window,
consistent with the 3.4x previously on the board (different window; the
number that carries is this window's).

## What this window did NOT measure

No cuML column (no NVIDIA hardware here), no inference, no depth sweep, and
nothing at other `max_features` counts — the sampled-column count still
decides the 100k classification story and that sweep is the obvious next
window. All ratios are fit-time only, as the harness defines.

## Addendum, same day: DEVIATION 212 (materialized score pass) — MEASURED AND DECLINED

The follow-on lever — the range pass stashing its gathered values (and
labels) in slot order so the score pass reads sequentially, recovering
cuML's read-once-per-level property without their bins — was built,
bit-identity gated, and measured in this same window, three modes alternated
inside one process, digests asserted identical every rep:

| config | values materialized | values + labels |
|---|---|---|
| clf 581,012 rows, sqrt, 10 trees | 0.90–1.01x | 0.86–1.11x |
| reg 581,012 rows, all, 10 trees | 0.74–0.92x | 0.89–0.94x |
| clf 100,000 rows, sqrt, 100 trees | 0.72–0.99x | 0.62–0.98x |

A wash at best, a loss at worst; reverted, full record and the corrected
reading of the "gather roofline" diagnosis in `extratrees/DEVIATIONS.md`
entry 212. The ledger's old "column materialization ~20%" estimate is
falsified by this table.

## Addendum 2: batch width (DEVIATION 213) and the Instruments profile

Sweeping the merged frontier's `max_batch_size` (4096 / 16384 / 32768,
alternated in one process, digests identical) at covtype 581k, 100 trees:
0.96-1.07x -- no effect. The Metal System Trace over one merged 100-tree fit
explains why and closes the schedule question: **GPU busy fraction 93.8%**,
the 10 longest dispatches hold 50.8% of busy time (the shallow levels' range
and score passes, where every row is active), and the **GPU performance
state sat at "Minimum M4" for 11.0 of 11.4 seconds** (thermal "Fair") -- the
heat-soaked-governor mechanism behind this box's standing 1.7x drift rule.
After 211 the host is not the cost, the kernels are the floor at this size,
and every absolute in this window was taken at whatever clock the window
got. Tools: `extratrees/bench/fit_once.mojo` + `profile_et_metal.py`.

## Addendum 3: the phase clock lands (DEVIATION 214), and the leaf tail is batched

The lane now has micro-step timing: `PhaseClock` in the forest trainers
(inert by default; `build/et_fit_once ... phases` prints nine phases with
the serialization distortion stated beside them). First attribution, shares
(the box was mid-drift; absolutes swung 2.7-10.2 s on one config within the
hour, so shares are the signal):

| phase | 581k x 10 trees | 100k x 100 trees |
|---|---|---|
| score pass | 41% | 30% |
| range pass | 28% | 25% |
| stage + feature sampler | 11% | 18% |
| candidate+reduce+readback | 10% | 9% |
| partition | 8% | 9% |
| leaf tail | 1% | 8.3% |

First fix it priced: the leaf tail allocated seven buffers and synchronized
once PER TREE; now one allocation set, one launch, one readback, one
synchronize per GROUP (the kernel was batch-ready per deviation 180, the
ranges already global per 211). Clocked leaf phase 735 -> 130 ms at 100
trees, bit-identity held by every device gate. Named next levers from the
table: stage+sampler and reduce at small n (per-batch fixed phases), the
two row passes everywhere (formulation floor per 212/213).

## Addendum 4: HIGGS, second dataset on the board

UCI HIGGS via the gbm-bench cache (`~/datasets/gbm-bench/higgs/`), converted
to the harness format as `higgs2m`: a 2,097,152-row prefix, because **the
full 11M rows are REFUSED by the classification device arm** -- DEVIATION
175's exact Int64 Gini pair is published only up to `SCORE_MAX_ROWS_EXACT =
2^21` rows, and 2,097,152 is exactly that bound. This is the first dataset
where the exactness cap binds a real request; widening the published key
(the check's own note says the Int64-pair bound is TIGHTER than the Int128
cross-multiply bound, so headroom exists) is now a named product item, not a
theoretical one. Regression has no such cap (DEVIATION 189).

2,097,152 x 28, 2 classes, depth 12, sqrt (5 of 28), interleaved one
process, 3 reps:

| config | vs sklearn 1 core | vs sklearn 10 cores | acc ours / theirs |
|---|---|---|---|
| 10 trees | 7.06-7.59x | **2.18-2.21x** | 0.635-0.639 / 0.642-0.657 |
| 100 trees (sklearn default) | 7.05-7.79x | **2.48-2.73x** | 0.646-0.652 / 0.659-0.662 |

Node counts comparable (ours 362-373k vs theirs 346-365k at 100 trees).
**OPEN QUESTION, stated rather than smoothed: our train accuracy runs
~1.0-1.5 points below sklearn's on higgs, consistently, with no overlap
across reps -- a gap covtype does not show (there ours was equal or
higher).** No bit-identity is claimed against sklearn (different RNG
designs), but a consistent one-directional gap on one dataset is a lead,
not noise: candidate suspects are the threshold-draw boundary handling,
the zero-gain gate's `<=`, and DEVIATION 151's stop-on-constant, none of
which covtype's feature distributions would exercise the same way. The
lane's quality-band check should gain a higgs fixture before this number
is quoted anywhere.

DEVIATION 211's A/B on higgs (serial vs merged, digests identical):
0.88-1.06x at 10 trees -- neutral, as the mechanism predicts: at 2M rows
the grid is never starved even one tree at a time, so the merge's win
(4-5x at 100k, 1.7-1.9x at 581k) tapers to nothing here. The absolute
numbers above rode the same drifting box as the rest of this window.

## Addendum 5: DEVIATION 215 -- cuML's sampler bias, found via higgs, fixed, confirmed

The higgs accuracy gap (addendum 4) traced to a cuML bug our port had
faithfully transcribed: when the excess sampler overshoots `k` uniques it
keeps the `k` SMALLEST column ids -- at `(n=28, k=5)` column 27 is sampled
at 0.38x column 0's rate. Higgs's most informative features are its
highest-indexed columns (starved); covtype's are its lowest (boosted) --
one mechanism, both signs. Fixed: the survivors are now the `k` smallest by
a keyed hash of `(tree, node, col)` -- exactly uniform by symmetry,
deterministic, host and device sharing one rule. Gated by a NEW
distribution assertion in `sampler_kernel_check` (4,000 nodes: fixed rule
670-768 around uniform 714; cuML's rule, kept as the required-RED sabotage
arm `SAMP_SAB_SMALLEST_K`: 273-836). No prior check could see it -- sets
valid, slots host-device identical -- which is the finding.

Confirmed, same harness, same seeds, interleaved (train acc, 3 reps):

| dataset | ours BEFORE | ours AFTER | sklearn |
|---|---|---|---|
| higgs2m, 10 trees | 0.635-0.639 (below every rep) | **0.640-0.652** (overlapping; one rep above) | 0.642-0.657 |
| covtype, 10 trees | 0.679-0.707 | 0.682-0.696 | 0.683-0.736 |

Higgs recovered ~+1.2 points to sklearn parity; covtype's average is
unchanged (0.690 -> 0.690) -- the bias's help there was marginal at
`(54, 7)`. Speedups in this window's clock state: higgs 2.8-3.1x their ten
cores, covtype 1.6-1.8x.

Also this round, the YEAR decomposition (`skl-et-cpu` run through the
peer's own gbm-bench harness, same split): sklearn ET scores MSE 96.25 at
the depth-8 parity config, so 3.4 of the 5.2-point ET gap vs LightGBM is
the MODEL FAMILY (depth-capped Geurts ET vs their leaf-wise binned
variant), not our code. Our remaining residual vs sklearn ET is 1.79 MSE
at `max_features=all` -- where the sampler never runs -- so it is a
SECOND, separate defect, still open.

## Addendum 6: the residual, cornered

Higgs at `max_features=all` (sampler out of the picture), interleaved, 10
trees, depth 12: ours 0.703-0.711 train acc vs sklearn's 0.708 -- ABOVE
theirs on two of three reps, at 2.6-4.4x their ten cores. So after
DEVIATION 215 there is NO our-vs-sklearn accuracy gap on higgs at either
`sqrt` or `all`. Combined with covtype regression (ours better than
sklearn on train MSE, addendum 1's window) and covtype classification
(parity), the year residual is now cornered to: TEST-set MSE, on year
specifically, regression at `all`, depth 8 -- 1.79 MSE vs sklearn ET, with
3.4 more being model family vs LightGBM's leaf-wise variant. Next
discriminators for it, in order: year TRAIN-vs-TEST MSE ours-vs-sklearn
(overfit-shape difference vs split-quality difference), then a read of
`regression_key`'s shift resolution at year's magnitudes. Not chased
further in this window.
