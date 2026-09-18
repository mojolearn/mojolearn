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
magnitude.

### The k-means rerun on the idle GPU, 9 rounds (second pod, after every identity phase, `ab-*` dirs; the 7-round ones are `ab-*.r1`)

| lane / dataset | before arm | before ms med (min..max, spread) | after arm | after ms med (min..max, spread) |
|---|---|---|---|---|
| kmeans taxi | base | 3101.9 (2638.4..3364.6, 1.275) `u` | both | 99.0 (92.9..102.3, 1.101) |
| kmeans Istella-S | base | 12556.6 (12354.0..12884.6, 1.043) | both | 442.3 (431.8..497.4, 1.152) `u` |
| kmeans taxi | base | 2929.6 (2757.9..3461.9, 1.255) `u` | blk | 421.1 (346.8..484.2, 1.396) `u` |
| kmeans Istella-S | base | 34066.7 (30248.5..38238.1, 1.264) `u` | blk | 881.2 (834.3..930.5, 1.115) `u` |
| kmeans taxi | blk | 369.3 (172.1..594.8, 3.457) `u` | both | 99.7 (96.9..103.6, 1.070) |
| kmeans Istella-S | blk | 967.3 (922.5..1084.5, 1.176) `u` | both | 482.0 (440.5..563.9, 1.280) `u` |

The only arm that holds the gate on this box is `both` on taxi, the one fit with neither
a host pass nor an atomic. `blk` still runs the host scale pass and swings 172 to 595 ms
on taxi (the host is two NUMA nodes and tcmalloc cannot bind memory); `base` swings with
its atomics. This pod cannot give a gated before-arm, so no ratio is quoted from it.

### cuML on the same box, measured ONCE (bench/OPPONENT_REFERENCE.md's rule; no RTX 4090 row existed for k-means, OLS or PCA)

`opponent-both/`, `tools/classical_two_datasets.py race --arms ours,cuml-gpu`, 5 rounds,
cuML 26.8.0 (cupy 14.2.0, numpy 2.4.6 in the image's Python 3.11.10), RTX 4090 driver
580.178.04, cuML's inputs already on the device, ours uploaded inside the clock, k = 64,
20 iterations (ours reports 21), PCA 8 components, OLS with intercept. cuML's k-means
digests differ round to round (6 distinct in 5 rounds on both datasets); ours are one hash.

| lane / dataset | cuML ms med (min..max) | ours (3080 + 3081) ms med (min..max) | ours over cuML |
|---|---|---|---|
| kmeans taxi 4M x 11 | 129.5 (127.6..140.5) | 102.8 (96.2..122.9, spread 1.278 `u`) | 0.79 |
| kmeans Istella-S 2M x 220 | 321.4 (319.2..327.9) | 431.4 (426.4..446.4) | 1.34 |
| ols taxi | 23.2 (23.0..24.0) | 461.0 (436.2..476.6) | 19.9 |
| ols Istella-S | 73.6 (71.0..75.0), r2 -15111 (its eig solve on near-constant columns) | 1620.3 (1586.9..1703.8), r2 0.332 | 22.0 |
| pca taxi | 18.8 (18.5..20.4) | 32.8 (31.5..34.4) | 1.74 |
| pca Istella-S | 68.1 (66.8..70.2) | 285.2 (278.5..410.2, `u`) | 4.19 |

Rows for the table's RTX 4090 section (the orchestrator adds them; the log is
`~/mojolearn-evidence/kmeans-linear-speed/pod2_final/kls_out/opponent-both/`, pod
0knlkeg0ni0y08, 2026-09-18 03:00Z):

| lane | shape | parameters | opponent | inputs on device | median ms (min..max) | rounds |
|---|---|---|---|---|---|---|
| kmeans | 4,000,000 x 11 (taxi) | k 64, 20 iterations, shared init array | cuML 26.8.0 KMeans | yes | 129.5 (127.6..140.5) | 5 |
| kmeans | 2,043,304 x 220 (Istella-S) | k 64, 20 iterations, shared init array | cuML 26.8.0 KMeans | yes | 321.4 (319.2..327.9) | 5 |
| ols | 4,000,000 x 11 (taxi) | intercept | cuML 26.8.0 LinearRegression | yes | 23.2 (23.0..24.0) | 5 |
| ols | 2,043,304 x 220 (Istella-S) | intercept | cuML 26.8.0 LinearRegression | yes | 73.6 (71.0..75.0) | 5 |
| pca | 4,000,000 x 11 (taxi) | 8 components | cuML 26.8.0 PCA | yes | 18.8 (18.5..20.4) | 5 |
| pca | 2,043,304 x 220 (Istella-S) | 8 components | cuML 26.8.0 PCA | yes | 68.1 (66.8..70.2) | 5 |

## Decision

Both candidates pass every gate that can be run on one NVIDIA pod: bits unchanged on 140
fit cells and their infer, model and batch parts on cuda and on cpu, before against after
and cuda against cpu; every pinned reduction order untouched (the Int32 totals are the
same addends, and the scale is the host's scale or the host pass itself); sabotage seen
DIVERGENT on every cell that records the affected output; the A/B says flip with a large
margin on both datasets and no dataset regresses.

Ratio of record (first pod, both arms inside the 1.10 gate, 7 rounds, digests equal):
taxi 12.17x (1216.1 to 99.9 ms), Istella-S 30.80x (14167.4 to 459.9 ms), GEOMETRIC MEAN
19.4x. The second pod's races agree in direction and are larger (taxi 31x to 33x,
Istella-S 28x to 30x from the same both arm and a slower base) but their before arms
fail the spread gate and are not quoted. Split by change (the only gated pair is the first
pod's taxi blk to both, 2.63x): 3080 removes the atomics, 3081 removes the host pass; on
taxi after 3080 the host pass IS the fit (blk 264 to 580 ms against both's 100 ms), on
Istella-S 3080 alone is 15x and 3081 doubles it again.

Against cuML on the same RTX 4090, ours (IDENTICAL, one hash, input uploaded inside the
clock) reads 0.79x of cuML's time on taxi and 1.34x on Istella-S, geometric mean 1.03x;
main's k-means on that box read 1.3 to 3.4 s on taxi and 12.4 to 43.8 s on Istella-S,
that is 10x to 26x and 39x to 136x of cuML's time.

NOT DONE, ON PURPOSE. The defaults are still OFF. Flipping them for NVIDIA is one line
each (`KMEANS_BLOCK_ACC = is_defined[...]() or TARGET_COLUMN == COLUMN_NVIDIA` with a
`MOJOLEARN_KMEANS_BLOCK_ACC_OFF` opt-out, the same for 3081 in `cluster/estimator.mojo`),
but that is a binary this lane has not built or run, and unrun code stays unflipped.
Apple and AMD columns are owed at the next release for both (3081's certificate needs IEEE
NaN propagation through `abs` and `+` on the device, which only NVIDIA has been checked
for; 3080 is plain Int32 loads and stores and should be inert everywhere, unverified).

## Rejected

Nothing on the branch moved a bit. No candidate lost. The `blk` arm alone is not worth
shipping without 3081 on taxi-shaped data (the host pass dominates it), which is why the
two ship together.

## Owed and next

1. The NVIDIA default flip (above), then one identity run of the flipped binary.
2. Apple and AMD columns at the release record.
3. OLS: the Python-side `host._column_means` (165 ms of 329 ms on taxi through the public
   API) and the Istella-S device Jacobi (425 to 503 ms of 668 to 746 ms in the entry);
   PCA Istella-S likewise (150 to 221 ms of 318 to 389). Measured, untouched.
4. `PRIVATE_ACC_CELLS` (6,144) is now a dead constant on NVIDIA once 3080 is default;
   leave it for the other columns.

## Commands

Pod: `TREES_LEG_NAME=mojolearn-kmeans-linear-speed TREES_LEG_CUDA_VERSIONS=13.0
MOJOLEARN_STAGE_KEYS="gbm-bench/taxi/taxi_speed.npz gbm-bench/istella/istella_speed.npz"
sh tools/trees_leg.sh rent --gpu "NVIDIA GeForce RTX 4090" --minutes 150`; unpack
`git archive origin/main` into `/root/mainsrc` with its `SHIPPED_COMMIT.txt`; on the pod
`nohup setsid sh tools/kmeans_linear_resume_chain.sh > /root/kls_out/chain.log 2>&1 < /dev/null &`;
then `sh tools/kmeans_linear_body.sh ab both base 9 kmeans taxi,istella` (and blk base,
both blk) and `sh tools/kmeans_linear_body.sh opponent both 5 kmeans,ols,pca taxi,istella`;
`sh tools/trees_leg.sh pull /root/kls_out <dir>`; `sh tools/trees_leg.sh reap`.

Second pod 0knlkeg0ni0y08: 01:37Z to 03:04Z, 1.45 h at $0.74/h, $1.07; reaped, HTTP 404
verified. Both pods together about $2.0.
