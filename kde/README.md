# Kernel density estimation

GPU KernelDensity derived from cuML.

`NOT_IMPLEMENTED.tsv` lists kernels, metrics, and options that are deliberately
refused. `estimator.mojo` is the public Mojo surface.

## Verify

```bash
pixi run check-kde
```

Changes should test normalization, metric dispatch, bandwidth edge cases, and reduction order.
FAST-mode approximations must remain explicit and must not leak into IDENTICAL mode.

## FAST score pass (DEVIATION 2490, 2026-09-10)

The FAST tier scores queries in one fused kernel: distance, log-kernel,
log-weight and an online log-sum-exp per query thread, training rows
streamed through shared memory. No `n_query x n_train` matrix exists, so
memory is O(n_query + n_train) and 100k x 100k scores in about half a
second on an M4 where the staged path could not allocate. The staged
path (`kde.dists`, `kde.logk`, `kde.rowmax` stages, serial ascending fold)
is what IDENTICAL and DETERMINISTIC run, and what FAST runs whenever an
identity trace is requested, so the two arms compare in one process
(`python/mojolearn/tests/test_kde_fused_fast.py`). Register rows cover
`d <= 64`; wider rows take a feature-chunked kernel. Numbers and the
timing protocol: `bench/results/kde_fast_2026-09-10/`.

## Tiled IDENTICAL score pass (DEVIATION 2625, 2026-09-11)

IDENTICAL with no identity trace recording, for euclidean, l1 and
chebyshev, now scores through `kde_score_samples_tiled_identical`
(`kde/impl/neighbors/kernel_density.mojo`, block comment above
`KDE_TILED_FEAT`) with cuVS 26.08's tile geometry and 2D grid (query blocks x
train chunks) over the staged path's own per-cell arithmetic, the
log-kernel written straight into the matrix (no distance matrix), the row
max folded from chunk maxima, the `exp` terms computed one thread per cell,
and the staged serial ascending sum. It returns the staged path's bits.
A trace, cosine, sqeuclidean, minkowski, FAST and DETERMINISTIC still run
the staged (or fused FAST) path.

Identity on the H100 (RunPod pod `dn8er13wjuxtax`, 2026-09-11).
`check_kde_tiled_equals_staged` OK under IDENTICAL, 33,300 scores (6
kernels x 3 metrics x weighted/unweighted x 5 shapes straddling every tile
and chunk edge, entry dispatch and a q_tpb 32 / 100-row-chunk schedule) and
0 differ; every other `kde_check` gate OK. `kde_stage_profile.mojo` hashed
the scores of the staged path, the dispatch and nine tiled schedules equal
on both profile shapes (FNV 16594497303053393111 at 100,000 x 2,000 x 220,
17888536843391681998 at 100,000 x 2,000 x 11).

Where the time goes on the H100, synthetic fixture at the lane's shapes
(`kde/checks/kde_stage_profile.mojo`, 3 repetitions after a warm-up, the
range or the typical value).

| stage | 100,000 x 2,000 x 220 | 100,000 x 2,000 x 11 |
|---|---|---|
| binding's list copy of X (replayed) | 36 to 54 ms | 0.4 ms |
| host validation (DEVIATION 604) | 39 ms | 1.9 ms |
| pinned upload of X | 19 ms | 0.6 ms |
| staged distance matrix | 96 ms | 3.8 ms |
| staged log-kernel matrix | 0.8 ms | 0.8 ms |
| staged per-row logsumexp (lse_tpb 128) | 27.7 ms | 27.7 ms |
| device entry, staged | 124 ms | 32.3 ms |
| device entry, tiled (this deviation, defaults) | 118 ms | 18.9 ms |
| tiled, q_tpb 512, 1,024-row chunks | 108 ms | 18.5 ms |

The attribution (`kde/checks/kde_dist_attribution.mojo`) says the pins are
not the cost: at 500 queries the staged cell with `ftz` and `fma`, without
`ftz`, without `fma`, without the sqrt, and fully plain all take 23.8 to
24.4 ms. The scalar tiled accumulator loop is the cost (108 ms at 2,000 queries). The
same kernel with the 64 accumulators as one `SIMD[float32, 64]` value
(`t1_simd_logk_kernel` in that file, euclidean + gaussian, unweighted)
takes 30.6 ms and writes the same log-kernel matrix (FNV
9460907080735567157 both). The tiled lse steps after it take row max 0.02 ms,
terms 0.9 ms, serial sum 9.5 ms (7.1 ms at lse_tpb 16).

Taxi race on the same pod before this change (binding at 36ca51fd):
ours 39.2 ms against cuML 2.37 ms (16.55x), mean log-likelihood -9.58207
against -9.58205.

## SIMD accumulators and host staging (DEVIATIONS 2626 and 2660, 2026-09-11)

DEVIATION 2626. `kde_tiled_logk_kernel` holds the 64 accumulators of a
cell tile as one `SIMD[float32, 64]` value and advances a whole tile row
per feature, for all three tiled metrics (euclidean: `ftz_simd` of the
difference, `identical_mul_add_simd`, `ftz_simd`; l1: `ftz_simd(acc +
abs(ftz_simd(q - t)))`; chebyshev: `abs(ftz_simd(q - t))` and a strict
`>` select, row 39). Each lane is the scalar core it replaces, operation
for operation; weights, the six kernels and the epilog are untouched. This
is `t1_simd_logk_kernel` from `kde/checks/kde_dist_attribution.mojo`
generalized and shipped.

DEVIATION 2660. `kde_score_samples_binding` no longer copies X and the
queries into `List`s. `kde/estimator.mojo::kde_score_samples_host_ptr`
validates the caller's memory with `kde_validate_data_ptr` (16-value
blocks screened by one mask, the first positive block and the tail walked
by the original per-value loop, so the refusal is the same first value in
the same words), stages it once into pinned memory and writes the scores
to the caller's output from the pinned download. Gate:
`check_kde_host_ptr_equals_list` (host only, asserts in every tier).

The measurements, identity evidence and flip verdict for both are in the
section below this list and in `bench/results/kde_finish_2026-09-11/`.

RUN OWED from DEVIATION 2625's lane (items 1 and 4 are DEVIATIONS 2626 and
2660 above; the pod was reaped at the orchestrator's wind-down):
1. Port `t1_simd_logk_kernel`'s SIMD accumulator into
   `kde_tiled_logk_kernel` for all three metrics, weights and kernels, then
   on the H100 `pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1
   kde/checks/kde_check.mojo` (the tiled gate must stay 0 differ).
2. The section 9 race, before and after in one window, both datasets.
   Rebuild with `MOJOLEARN_NUMERIC_MODE=identical sh
   bindings/build_estimators.sh`, keep a copy of the 36ca51fd `python/` tree,
   then `MOJOLEARN_CTD_PHASES=races MOJOLEARN_CTD_LANES=kde
   MOJOLEARN_CTD_DATASETS=taxi,istella MOJOLEARN_CTD_ROUNDS=5
   MOJOLEARN_CTD_EXTRA_ARMS=ours-before MOJOLEARN_CTD_OURS_BEFORE_ROOT=<copy>
   sh tools/classical_two_datasets_leg.sh`, then `tools/flip_verdict.py`
   on ours / ours-before. The `ours-before` arm is new and unrun.
3. The same gate on the Apple M4 and the AMD MI300X (bits must not move,
   and the tiled pass is dispatched there too).
4. Host staging (about 75 to 90 ms of the Istella call): validate and stage
   the caller's float32 memory directly in `kde_score_samples_binding`
   instead of copying it into two Lists first. DONE, DEVIATION 2660.

## The before/after race and the flip verdict (2026-09-11, H100)

RunPod pod `ur95zh3h9qbx2p` (`kde-finish-2026-09-11_213144`), NVIDIA H100
80GB HBM3, driver 570.124.06, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, cuML 26.08
(`cuml-cu12` 26.8.0), scikit-learn 1.9.1, NumPy 2.4.6 in the image's Python
3.11. MAX requires driver 580 for its own PTX compiler, so both of our arms
and every gate ran with `MODULAR_NVPTX_COMPILER_PATH` pointing at the CUDA
12.9.86 wheel's `ptxas` (the image's 12.4 toolkit `ptxas` refuses PTX
`.version` 8.5); the escape is the one `bench/results/e1g/*/remote_body.sh`
already uses, it is recorded in `bench/results/kde_finish_2026-09-11/ptxas.txt`,
and it is the same for `ours` and `ours-base`.

BEFORE is `origin/main` at 2c64a778 (the staged path: no DEVIATION 2625,
2626 or 2660), built on this pod with `MOJOLEARN_GPU_ARCHS=sm_90a` and
raced as the harness's `ours-base` arm out of a second `python/` tree.
AFTER is this branch. Both are IDENTICAL builds, interleaved round by round
in one race, 1 warm-up plus 5 rounds, ms median (min..max):
`tools/classical_two_datasets.py race --arms ours,cuml-gpu,ours-base`.
Shapes are the classical lane's KDE block, 100,000 standardized fit rows x
2,000 queries, gaussian kernel, euclidean metric, Scott's bandwidth.

| dataset | shape | cuML ms | before ms | after ms | after / before | ours / cuML | score digest |
|---|---|---|---|---|---|---|---|
| taxi | 100,000 x 2,000 x 11 | 2.02 (1.98..2.06) | 35.69 (35.64..35.99) | 28.94 (28.92..28.97) | 0.811 | 14.30x | aa8ac4159ad2cbfa both arms |
| Istella-S | 100,000 x 2,000 x 220 | 6.95 (6.92..7.14) | 218.73 (217.91..221.95) | 67.94 (67.46..68.78) | 0.311 | 9.78x | 81d11ed7fcd9eb38 both arms |

**FLIP. Geometric mean of the two ratios 0.502, and quality is not worse on
either dataset**, so the three deviations stay ON by default. Mean
log-likelihood is EQUAL to every digit the harness prints between before and
after (taxi -9.582071959257126, Istella-S -212.1174684753418), and no query
lost its density; cuML reads -9.58205059337616 and -212.11746494293212 on
the same blocks. The cuML taxi median is the first race's, whose five rounds
held 1.98 to 2.06 ms; in the second race cuML's taxi arm took a 22.1 ms
round and read 3.56 ms median (ours 28.94 against it is 8.14x). Every one of
our rounds held one digest in both races.

The first race (this branch with DEVIATION 2626's SIMD accumulators but the
inherited q_tpb 256 / 1,024-row schedule) read taxi 29.10 ms and Istella-S
72.57 ms, that is 0.809 and 0.330, geometric mean 0.516; the schedule change
above took Istella-S a further 4.6 ms. Both races are in
`bench/results/kde_finish_2026-09-11/` (`race1_summary.tsv`,
`race2_summary.tsv`) and in `~/mojolearn-evidence/kde-finish-2026-09-11/`.

Where the time goes now (`kde/checks/kde_stage_profile.mojo`, same pod, 3
repetitions after a warm-up, the rep 2 value):

| stage | 100,000 x 2,000 x 220 | 100,000 x 2,000 x 11 |
|---|---|---|
| binding's List copy of X (2625's path) | 36.2 ms | 0.7 ms |
| host validation, List (DEVIATION 604) | 24.1 ms | 1.2 ms |
| host validation, pointer (DEVIATION 2660) | 9.3 ms | 0.2 ms |
| device entry, staged | 124.1 ms | 32.1 ms |
| device entry, tiled (2625 + 2626, defaults) | 41.0 ms | 27.3 ms |
| whole host entry, List (2625's path) | 83.3 ms | 28.9 ms |
| whole host entry, caller's memory (2660) | 63.6 ms | 28.0 ms |

The tiled Istella-S device entry was 118 ms with 2625's scalar accumulators
and is 41.0 ms with 2626's SIMD ones; the taxi shape is now dominated by the
serial log-sum-exp, which is the staged path's own order and cannot be
reassociated.

Identity on this pod, under IDENTICAL: the whole `kde_check` passes (15
checks), `check_kde_tiled_equals_staged` 33,300 scores and 0 differ,
`check_kde_host_ptr_equals_list` 984 scores and 0 differ with the same
refusal message on 15 planted cases, and the stage profile hashes the host
entry, the pointer entry, the staged replay, the forced-staged device entry,
the entry dispatch and nine tiled schedules EQUAL on both shapes: FNV
16594497303053393111 at 100,000 x 2,000 x 220 and 17888536843391681998 at
100,000 x 2,000 x 11, the same two values DEVIATION 2625 recorded. The race
digests are equal between before and after on both datasets, which is the
same statement at the binding's own boundary.

RUN OWED: `pixi run check-kde` under IDENTICAL on the Apple M4 and on an AMD
MI300X (Hot Aisle), where the tiled pass is dispatched too and the SIMD
accumulators and the pointer validator are new code. The commands are in the
lane's commit message.
