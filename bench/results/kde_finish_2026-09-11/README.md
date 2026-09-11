# kde-finish lane, 2026-09-11 (DEVIATIONS 2626 and 2660, finishing 2625)

RunPod pod `ur95zh3h9qbx2p` (`kde-finish-2026-09-11_213144`), NVIDIA H100
80GB HBM3, driver 570.124.06, kernel 6.8.0-58, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, cuML 26.08.00
(cuml-cu12 26.8.0, cupy 14.2.0), NumPy 2.4.6, scikit-learn 1.9.1, SciPy
1.17.1. Reaped at the end of the lane.

MAX wants driver 580 for its own PTX compiler and this box has 570, so every
gate and both of our arms ran with `MODULAR_NVPTX_COMPILER_PATH` at the CUDA
12.9.86 wheel's `ptxas` (`ptxas.txt`). The image's CUDA 12.4 `ptxas` refuses
PTX `.version` 8.5 and is NOT usable for the JIT gates; the AOT bindings ran
under either one. The escape is the one `bench/results/e1g/*/remote_body.sh`
already uses.

## What was measured

BEFORE is `origin/main` 2c64a778, the staged score pass, built on this pod
with `MOJOLEARN_GPU_ARCHS=sm_90a` and raced as the harness's `ours-base` arm
from a second `python/` tree (`python_base_sha.txt`). AFTER is
`lane/kde-finish`, that is DEVIATION 2625 (tiled IDENTICAL pass) plus 2626
(SIMD tile accumulators and the tiled pass's default schedule) plus 2660
(validate and stage the caller's memory in the binding). Both are IDENTICAL
builds, interleaved round by round in one race, 1 warm-up plus 5 rounds.
Shapes are the classical KDE block, 100,000 standardized fit rows x 2,000
queries, gaussian kernel, euclidean metric, Scott bandwidth.

| dataset | cuML ms | before ms | after ms | after / before | ours / cuML | digest before = after |
|---|---|---|---|---|---|---|
| taxi (d 11) | 2.02 (1.98..2.06) | 35.69 (35.64..35.99) | 28.94 (28.92..28.97) | 0.811 | 14.30x | aa8ac4159ad2cbfa |
| Istella-S (d 220) | 6.95 (6.92..7.14) | 218.73 (217.91..221.95) | 67.94 (67.46..68.79) | 0.311 | 9.78x | 81d11ed7fcd9eb38 |

Geometric mean of the two ratios 0.502, quality not worse on either dataset
(mean log-likelihood equal to every printed digit between the arms, taxi
-9.582071959257126 and Istella-S -212.1174684753418, no query without a
density), so under ENGINEERING_RULES.md section 9 this FLIPS and the three
deviations stay ON as the default.

`race1_summary.tsv` is the first race, with 2626's SIMD accumulators but
2625's inherited q_tpb 256 / 1,024-row schedule (taxi 29.10 ms, Istella-S
72.57 ms, ratios 0.809 and 0.330, geometric mean 0.516). `race2_summary.tsv`
is the second, with 2626's measured default schedule (q_tpb 128, 256-row
chunks), and is the table above. The cuML taxi median quoted above is race
1's; in race 2 cuML's taxi arm took a 22.1 ms round and read 3.56 ms median.

## Identity

* `kde_check_identical_race2.log` (and `kde_check_identical.log` for the
  first build): the whole IDENTICAL `kde_check`, 15 checks green, including
  `check_kde_tiled_equals_staged` 33,300 scores 0 differ and the new
  `check_kde_host_ptr_equals_list` 984 scores 0 differ over 6 metrics x 2
  kernels x weighted and unweighted, plus 15 planted refusal cases where the
  pointer validator gives the List validator's message word for word.
* `profile_hashes.txt`: the host entry, the pointer entry, the staged
  replay, the forced-staged device entry, the entry dispatch and nine tiled
  schedules all hash EQUAL on both shapes, FNV 16594497303053393111 at
  100,000 x 2,000 x 220 and 17888536843391681998 at 100,000 x 2,000 x 11.
  These are the two values DEVIATION 2625 recorded on pod dn8er13wjuxtax, so
  the bits did not move between that lane, this one and main's staged path.
* The race digests are equal between before and after on both datasets,
  which is the same statement at the binding's boundary.

RUN OWED: `pixi run check-kde` under IDENTICAL on the Apple M4 and on an AMD
MI300X. The tiled pass is dispatched on those vendors too, and the SIMD
accumulators and the pointer validator are new code there.

## Where the time goes (`profile_stages_rep2.txt`)

| stage | 100,000 x 2,000 x 220 | 100,000 x 2,000 x 11 |
|---|---|---|
| binding's List copy of X (2625's path) | 36.2 ms | 0.7 ms |
| host validation, List | 24.1 ms | 1.2 ms |
| host validation, pointer (2660) | 9.3 ms | 0.2 ms |
| device entry, staged | 124.1 ms | 32.1 ms |
| device entry, tiled (2625 + 2626) | 41.0 ms | 27.3 ms |
| whole host entry, List | 83.3 ms | 28.9 ms |
| whole host entry, caller's memory (2660) | 63.6 ms | 28.0 ms |

DEVIATION 2625's scalar tiled entry was 118 ms at the Istella-S shape; 2626's
SIMD accumulators take it to 41.0 ms. What remains at the taxi shape is
mostly the serial log-sum-exp, which is the staged path's own summation
order and cannot be reassociated without moving bits.

Big logs, both races' JSON with every round, the build logs and the console
captures are in `~/mojolearn-evidence/kde-finish-2026-09-11/`.
