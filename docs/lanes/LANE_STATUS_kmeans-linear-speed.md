# lane/kmeans-linear-speed: k-means Lloyd update without atomics, and the sum scale from the device

Written 2026-09-17 evening for a reader with no memory, after the lane was hard-stopped
at ~21:00Z mid-identity-run with no resume note. Everything below the "State at the
pause" heading was reconstructed from the branch's commits and the pulled evidence in
`~/mojolearn-evidence/kmeans-linear-speed/final_pull_pause/kls_out` (203 files). The
"Resumed" section is kept current as the finishing pod runs.

## What changed (two candidates, both behind defines, both default OFF)

**DEVIATION 3080, `cluster/checks/reduce_by_key.mojo` and `cluster/impl/detail/kmeans.mojo`.**
`-D MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC=1`. The two atomic reductions of the Lloyd
update (`launch_accumulate_centroid_sums`, `launch_accumulate_weight_per_cluster`) are
replaced by a row-block table: rows are cut into blocks of 1024, thread `(block, feature)`
owns the `n_clusters` table cells `table[b][*][f]` and walks its block's rows in order
with plain Int32 loads and stores; a second launch gives every output cell one thread
that adds the block entries into `sums_i32`. No atomics, no shared memory.
WHY NO BIT MOVES: the addend is the same scalar fp32 expression the two existing kernels
use, `Int32(x * w * scale)`, once per `(row, feature)`; Int32 addition is associative and
commutative and `choose_scale` bounds every partial sum inside Int32, so any grouping of
the same addends gives the same total. This is the argument the privatized arm on main
already rests on. Sabotage `-D MOJOLEARN_KMEANS_BLOCK_ACC_SABOTAGE=1` drops the first
row of every block.

**DEVIATION 3081, `cluster/estimator.mojo`.** `-D MOJOLEARN_EXPERIMENTAL_KMEANS_DEVICE_SCALE=1`.
`kmeans_fit` used to walk the whole host matrix in float64 (`plan_sum_scale`) to pick the
Int32 quantization scale, then upload it. Now the scale comes from a device pass over the
uploaded matrix (`plan_sum_scale_certified`): per-column float32 sums of `abs(x)` with a
bounded reduction height, turned into an interval that provably contains the host's
`worst`; `choose_scale` is a step function, so if both ends of the interval give the same
scale it IS the host's scale. Any refusal (non-finite column, magnitude below 2^-40, an
interval straddling a power of two) falls back to the host pass, which stays the
definition. Sabotage `-D MOJOLEARN_KMEANS_DEVICE_SCALE_SABOTAGE=1` multiplies the device
magnitude by 4 before certification (two binades). NVIDIA only until other columns'
NaN propagation through `abs` and `+` is verified.

Nothing on the branch touches OLS, ridge, PCA or truncated SVD code; those lanes are in
the identity set because the same binding ships them, and the OLS/PCA stage probe
(`glm/ols_pca_stage_probe_main.mojo`) only measured.

Also on the branch: the pod body `tools/kmeans_linear_body.sh`, the Lloyd stage probe
`cluster/tools/kmeans_linear_stage_probe_main.mojo`, the one-check-per-process runner
`cluster/tools/kmeans_linear_checks_main.mojo` (on the RTX 4090 pod the second check of any
process that opens a fresh DeviceContext per check hangs in its first allocation), the
public fit profile `tools/kmeans_linear_profile.py`, and `check_blocked_accumulate` and
`check_plan_sum_scale_certified` wired into `cluster/kmeans_main.mojo`.

## The box

RunPod secure cloud, NVIDIA GeForce RTX 4090 (24,564 MiB, sm_89), driver 580.159.04, host
AMD EPYC 7642 (96 vCPU, 251 GiB), Mojo 1.0.0 (ed45d567), image
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04, CUDA 13.0 host. Datasets staged
from R2 (taxi_speed.npz, istella_speed.npz), blocks prepped by
`tools/classical_two_datasets.py prep`: taxi 4,000,000 x 11 and Istella-S 2,043,304 x 220,
k = 64, max_iter 20 (21 iterations reported), PCA 8 components, OLS with intercept.

First pod: hlcyyooxq1mp1x, 2026-09-17 19:38Z to 20:54Z (reaped by the orchestrator,
verified 404), $0.74/h, about 1.27 h, about $0.94.

## What was measured first (attribution, first pod, IDENTICAL, one Lloyd iteration split with a synchronize at every phase)

`cluster/tools/kmeans_linear_stage_probe_main.mojo`, k = 64, main's kernels (`base`):

| dataset | rows x cols | host `plan_sum_scale` | upload (pinned) | assign | sums (atomic) | weights (atomic) | finalize |
|---|---|---|---|---|---|---|---|
| taxi | 4,000,000 x 11 | 156 to 215 ms, once | 13 ms, once | 1.8 to 2.1 ms | 36 to 44 ms | 5.2 to 6.5 ms | 0.04 ms |
| Istella-S | 2,043,304 x 220 | 606 to 711 ms, once | 142 to 148 ms, once | 9.2 to 9.8 ms | 661 to 697 ms | 3.4 to 4.1 ms | 0.05 ms |

The update half of Lloyd, not the distance half, is the fit. Istella-S at 64 x 220 =
14,080 cells exceeds `PRIVATE_ACC_CELLS` (6,144) so main takes the DIRECT arm: 449.5
million global `Atomic.fetch_add` per iteration onto 14,080 cells. Taxi takes the
privatized arm and still pays one threadgroup atomic per element. On the RTX 4090 this is
far worse than on the H100 of the Sep 12 board (main's fit: taxi 194 ms on H100 against
1,216 ms here; Istella-S 1,261 ms against 14,167 ms), so the ratios below are 4090
ratios; the H100 column is owed at the next release.

Same probe with 3080 (`blocked`): sums 1.02 to 1.12 ms on taxi, 4.5 to 4.6 ms on Istella-S;
weights 0.63 to 0.67 ms and 0.54 to 0.60 ms. Every phase digest (labels, min_dist,
sums_i32, weight_i32, centroids, shift) equal to `base` for three iterations on both
datasets; the `sabotage` build moves sums_i32 and centroids from iteration 0.

OLS and PCA (`glm/ols_pca_stage_probe_main.mojo`, main's code, RTX 4090): OLS taxi entry
25 to 28 ms (upload 14, Gram 0.8, `A^T b` 2.8, rest 1 to 5); OLS Istella-S 668 to 746 ms
(upload 210, Gram 18, `A^T b` 6, REST 425 to 503, the device Jacobi and back-substitution);
PCA taxi 23 to 30 ms (upload 14, covariance 3.9); PCA Istella-S 318 to 389 ms (upload 134,
covariance 33, rest 150 to 221). `tools/kmeans_linear_profile.py` on taxi through the
public API: OLS total 329 ms of which `host._column_means` 165 ms and `native.ols_fit`
63 to 165 ms; PCA total 31 ms; k-means total 1,497 ms of which 1,485 native. The OLS
Python-side centering and the Istella-S Jacobi are the next targets; NOT touched here.

## State at the pause (first pod, all binaries mtime-checked against their sources)

Arms built on the pod (`arm-<name>/defines.txt`): `base` = main's source at 838ebfdf8
(no define); `off` = the branch source with no define; `blk` = 3080 only; `both` = 3080 +
3081; `sabo80` = both + BLOCK_ACC_SABOTAGE; `sabo81` = both + DEVICE_SCALE_SABOTAGE.
The binding sources did not change after abcb12a63 (later commits touch only the body
script and a probe), so `both` at abcb12a63 is the tip's binary.

Identity (`tools/identity_break.py`, fixtures base,ties,odd,dupes,wide, repeats 2), lanes
kmeans, kmeans-random, kmeans-array, kmeans-weighted, kmeans-sqrt, kmeans-classic-pp,
kmeans-cosine, pca, pca-whiten, pca-full-whiten, tsvd, ols, ridge, ols-no-intercept,
ols-weighted, ridge-no-intercept (80 fit cells, 160 infer/model cells, 80 batch cells):

| diff | verdict |
|---|---|
| cuda: base vs off vs both | IDENTICAL x3 on all 80 (`diff.cuda.base-off-both.txt`) |
| cpu (host bindings from each tree): base vs both | IDENTICAL on all 80, infer/model 150 IDENTICAL + 10 n/a (kmeans-cosine refuses a fit), batch 75 + 5 n/a |
| cuda vs cpu, before (base) | IDENTICAL 80 |
| cuda vs cpu, after (both) | IDENTICAL 80 |
| sabotage 3080 (both vs sabo80) | DIVERGENT 30 of the 30 reached k-means cells (6 fitting lanes x 5), the 50 non-k-means cells IDENTICAL |
| sabotage 3081 (both vs sabo81) | DIVERGENT 26 of 30; kmeans/ties, kmeans-array/ties, kmeans-random/ties, kmeans-classic-pp/ties IDENTICAL (reached but inert on that fixture; the likely reason, not verified, is that its addends quantize exactly at both scales so the two-binade change cancels in the division), kmeans-sqrt/ties diverges through `scales` only |

Checks (`kls_checks`, one per process): `check_blocked_accumulate` OK (2112 sum + 64 weight
cells bit-identical direct vs row-block at 3109 x 33 with a ragged tail, sabotage moved 99
cells); `check_plan_sum_scale_certified` OK (48 of 48 sweep members certified and equal to
the host scale, boundary pair, zero plane, NaN and inf refused); `check_privatized_accumulate`
OK.

Interleaved A/B (`tools/classical_two_datasets.py race`, arms alternated, 7 rounds, one
worker per arm, host-resident input, upload inside the clock), first pod:

| lane / dataset | before arm | before ms (min..max) | after arm | after ms (min..max) | ratio | digests |
|---|---|---|---|---|---|---|
| kmeans taxi 4M x 11 | base | 1216.1 (1203.8..1226.9, spread 1.019) | both | 99.9 (99.6..102.4, 1.027) | 12.17x | 89520efe99a08d5f both arms |
| kmeans Istella-S 2M x 220 | base | 14167.4 (14122.3..14261.3, 1.010) | both | 459.9 (456.4..464.3, 1.017) | 30.80x | 7f720b0b76896308 both arms |
| kmeans taxi | blk | 264.2 (248.8..267.2, 1.074) | both | 100.3 (100.0..102.1, 1.021) | 2.63x | equal |
| kmeans Istella-S | blk | 919.4 (861.1..976.0, 1.133) `u` | both | 457.6 (456.9..460.2, 1.007) | not quoted | equal |

The output digests equal the H100 board's (`bench/results/baseline_sweep_2026-09-12/board.tsv`
rows kmeans/taxi 89520efe99a08d5f and kmeans/istella 7f720b0b76896308), so the bits are
the same on the H100 record and on this box before and after.

Missing at the pause: (a) identity on the other lanes that reach `kmeans_fit` or
`kmeans_fit_main_traced` (spectral, spectral-precomputed, gmm, gmm-sample, ivf, ivf-euclidean,
ivf-extend, par-kmeans, par-gmm) was run on `off` with the spectral cells REFUSED (the
metrics binding was not built) and was 43 of 45 cells into `both` when stopped; `base` was
never run there; metrics, metrics-classification and par-graph-spectral, which also fit a
KMeans or a SpectralClustering, were not in the list at all; no cpu column for those lanes.
(b) A direct base vs blk A/B (3080 alone) and a blk vs both rerun with the Istella-S blk
arm inside the spread gate. (c) OLS and PCA A/B on both datasets (no code change, a
regression control). (d) This document.

## Resumed 2026-09-17 evening (second pod 0knlkeg0ni0y08)

RunPod secure RTX 4090 (driver 580.178.04, host 120 vCPU, two NUMA nodes; tcmalloc
prints `Unable to mbind memory` in every process), $0.74/h, rented 01:37Z. The branch was
first merged with `origin/main` at ac51feba4 (clean; main had moved only in
`tools/identity_break.py`, a two-line `bindings/_mojolearn_estimators_host.mojo` change
and three host oracles, nothing under `cluster/`, `glm/` or `decomposition/`), so `off`
is now main plus the lane's files exactly. `base` was built from origin/main's own
archive on the pod (`/root/mainsrc`); `diff -rq` of the two trees shows only the lane's
files. The whole sequence is `tools/kmeans_linear_resume_chain.sh`, detached, phase
markers under `/root/kls_out`; evidence pulled to
`~/mojolearn-evidence/kmeans-linear-speed/pod2_pull*/kls_out`. Every arm's binaries
postdate their sources (`arm-*/mtimes.txt`, `arm-*/source.txt`).

### Identity, cuda column, second pod (the 16 lanes, five fixtures, two repeats)

| diff | verdict |
|---|---|
| base vs off vs blk vs both | IDENTICAL x4 on all 80 fit cells; infer/model 150 IDENTICAL + 10 n/a; batch 75 + 5 n/a (`diff.cuda.base-off-blk-both.txt`) |
| sabotage 3080 (both vs sabo80) | DIVERGENT 30 of 30 reached cells, 50 IDENTICAL, the same cells and hashes as the first pod |
| sabotage 3081 (both vs sabo81) | DIVERGENT 26 of 30; the same four `ties` cells inert as on the first pod |

The three checks pass again on this pod (`checks/checks.txt`): blocked (2112 + 64 cells,
sabotage moved 99), scale (48 of 48 certified), privatized.

### Identity, cuda column, the other lanes that reach the Lloyd loop (second pod)

Lanes spectral, spectral-precomputed, gmm, gmm-sample, ivf, ivf-euclidean, ivf-extend,
par-kmeans, par-gmm, metrics, metrics-classification, par-graph-spectral (60 fit cells;
the spectral cells REFUSED on the first pod because `_mojolearn_metrics.so` was not built,
now built by `arm_extra`). Which seam each lane reaches, from the code: `ivf` calls
`kmeans_fit_main_traced` directly with its own scale (3080 only); GaussianMixture's
k-means init, SpectralClustering's assignment, the metrics lanes' `KMeans(n_clusters=4)`
and the parallel drivers go through `kmeans_fit` (3080 and 3081). Of those, only the
KMeans lanes and `par-kmeans` RECORD centroids; GMM, spectral and the metrics lanes
consume the labels alone.

| diff | verdict |
|---|---|
| base vs off vs blk vs both | IDENTICAL x4 on all 60 fit cells; infer/model 90 IDENTICAL + 30 n/a; batch 45 + 15 n/a (`diff.cuda2.base-off-blk-both.txt`) |
| sabotage 3080 (both vs sabo80) | DIVERGENT 45 of 60: gmm, gmm-sample, ivf, ivf-euclidean, ivf-extend, metrics, metrics-classification, par-gmm, par-kmeans, every fixture. IDENTICAL 15: spectral, spectral-precomputed, par-graph-spectral (reached; the dropped rows move the embedding's centroids without moving the recorded labels) |
| sabotage 3081 (both vs sabo81) | DIVERGENT 4 of 60: par-kmeans base, dupes, odd, wide (ties inert as in the KMeans lanes). IDENTICAL 56: every lane that records labels only, and ivf, which never takes the 3081 path |

So across both sets 3080 is seen DIVERGENT on 75 of the 90 cells that fit a k-means (30
of 30 in the first set, kmeans-cosine's five cells refuse a fit; 45 of 60 in the second)
and 3081 on 30 of the 35 that record centroids (the five `ties` cells inert); the rest are
reached by code path and inert on their recorded parts.

### Identity, cpu column, second pod (host bindings core, estimators, metrics, mixture, mixture_infer, ivf, ivf_search built from each tree; the pod's CPU is an EPYC 7543)

| diff | verdict |
|---|---|
| cpu base vs both, the 16 lanes | IDENTICAL 80 fit cells; infer/model 150 + 10 n/a; batch 75 + 5 n/a (`diff.cpu.base-both.txt`) |
| cpu base vs both, the other 12 lanes | IDENTICAL 45; the 15 par-* cells REFUSED on both sides ("no CPU implementation of the cooperative multi-GPU driver", a NotImplementedError by design) (`diff.cpu2.base-both.txt`) |
| cuda vs cpu, before (base) and after (both), the 16 lanes | IDENTICAL 80 each (`diff.before.cuda-vs-cpu.txt`, `diff.after.cuda-vs-cpu.txt`) |
| cuda vs cpu, before and after, the other 12 lanes | IDENTICAL 45 each, 15 one-column (the par-* cells) (`diff.before2.cuda-vs-cpu.txt`, `diff.after2.cuda-vs-cpu.txt`) |

### Interleaved A/B, second pod, 7 rounds, arms rotated (all output digests equal across arms)

| lane / dataset | before arm | before ms med (min..max, spread) | after arm | after ms med (min..max, spread) | note |
|---|---|---|---|---|---|
| kmeans taxi | base | 3397.5 (3312.1..3543.4, 1.070) | both | 101.5 (93.2..104.8, 1.124) `u` | after arm spread over 1.10 |
| kmeans Istella-S | base | 12416.6 (12202.6..12595.3, 1.032) | both | 414.3 (407.7..502.1, 1.232) `u` | first timed round 502, the rest 408..434 |
| kmeans taxi | base | 1297.6 (1277.9..1303.6, 1.020) | blk | 390.4 (360.9..403.6, 1.118) `u` | 3080 alone |
| kmeans Istella-S | base | 43778.2 (41709.6..43885.2, 1.052) | blk | 1011.5 (959.7..1040.0, 1.084) | 3080 alone, base 3.5x slower than 20 minutes earlier |
| kmeans taxi | blk | 580.5 (563.0..604.5, 1.074) | both | 103.0 (94.1..176.0, 1.869) `u` | 3081 given 3080; one 176 ms round |
| kmeans Istella-S | blk | 919.1 (888.9..1023.8, 1.152) `u` | both | 433.0 (417.8..442.9, 1.060) | 3081 given 3080 |
| ols taxi | base | 475.5 (469.7..526.4, 1.121) `u` | both | 547.5 (528.2..560.1, 1.060) | no OLS code differs between the trees |
| ols Istella-S | base | 1576.0 (1565.0..1617.3, 1.033) | both | 1585.5 (1563.7..1658.5, 1.061) | 0.99x, no code change |
| pca taxi | base | 65.9 (31.0..66.3, 2.134) `u` | both | 51.9 (21.6..53.8, 2.487) `u` | both arms bimodal (21 or 53 ms; 31 or 66 ms) |
| pca Istella-S | base | 291.9 (285.5..295.0, 1.033) | both | 267.6 (263.5..376.8, 1.430) `u` | rounds 1..4 at 264..268, then 377, 353, 310 |

READ THIS BEFORE QUOTING A RATIO FROM THIS POD. The atomic (before) arm is not stable
here: main's k-means read 12.4 s and then 43.8 s on Istella-S twenty minutes apart, 3.4 s
and then 1.3 s on taxi, with the same bits every time and no throttle reason reported by
`nvidia-smi` (P2, 2670 MHz, no active event reason; the cumulative SW power-capping
counter on this 29-day-old host reads 61 hours). The row-block arm did not move in the
same way (94..106 ms and 408..443 ms in every race on both pods). The OLS gap between two
trees whose OLS code is byte-identical (475 vs 548 ms) is this pod's noise floor for a
host-side-heavy fit. The first pod's A/B (both arms inside the 1.10 gate) remains the
quoted number; this pod's races are consistent with it in direction and larger in
magnitude. A k-means rerun on an idle GPU with more rounds follows the identity phases.
