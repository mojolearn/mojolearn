# kde-fused lane, 2026-09-12: two ways to delete the matrix, both refused

RunPod pod `ndscc544rcf8ek` (`kde-fused-2026-09-12_190726`), NVIDIA H100 80GB
HBM3, driver 580.126.20, kernel 6.8.0-107-generic, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, cuML 26.08.00,
scikit-learn 1.9.1, NumPy 2.4.6, Mojo 1.0.0 (ed45d567). The driver is above
Mojo's CUDA floor, so unlike the 2026-09-11 lane NO `ptxas` override was
needed anywhere. Datasets came from R2 (`tools/dataset_store.sh stage`) and
the box verified size and sha256 against
`bench/results/dataset_store/manifest.tsv`. Reaped at the end of the lane.

## The brief, and the answer

Close the KDE gap against cuML (9.78x on Istella-S, 14.30x on taxi as the
board stood) by removing the `n_query x n_train` log-kernel matrix, which
cuML never materializes and we write. Two ways to remove it were built,
gated and measured. **Both are slower than the matrix they delete, so no
default changed.** The lane ships the instruments, the two arms, and the
numbers that refuse them.

## 1. The premise did not survive the stage split

`kde/checks/kde_stage_profile.mojo`, 3 repetitions after a warm-up, rep 2,
100,000 fit rows x 2,000 queries. Every stage launched alone and drained, so
the device entry is attributed rather than assumed (`profile_stages.log`).

| stage | d = 220 | d = 11 |
|---|---|---|
| device entry, tiled matrix (the shipped default) | 36.9 ms | 26.9 ms |
| ... log-kernel matrix kernel (arithmetic + write) | 26.5 ms | 16.5 ms |
| ... the same kernel with the matrix write elided | 25.0 ms | 13.4 ms |
| ... row-max fold over chunk maxima | 0.05 ms | 0.05 ms |
| ... terms, `exp(logk - max)` over every cell | 0.92 ms | 0.92 ms |
| ... the serial ascending fold | 9.48 ms | 9.48 ms |
| host work in the binding's call | ~19.7 ms | ~0.6 ms |
| ... of which host validation (DEVIATION 604) | 7.28 ms | 0.21 ms |
| device entry, staged path (no tiling at all) | 124.2 ms | 32.3 ms |

**The matrix write is not the cost.** Eliding the store saves 1.5 ms of 36.9
at d = 220 and 3.1 ms of 26.9 at d = 11. The whole 800 MB round trip -- write,
terms pass, and the fold's re-read -- is about 11.9 ms of the 36.9 ms; the
other 25.0 ms is per-cell distance arithmetic that exists whether or not a
matrix does. That caps what any fusion can win here before it has cost
anything.

## 2. DEVIATION 2690, the fused pass: 2.5x slower at d = 220

`kde_fused_sum_kernel` never allocates, writes or reads a matrix. The
order-bound adds stay in ONE owner thread per query; the expensive per-cell
work is spread over `H` helper threads that hand terms through shared memory,
so the accumulator still sees `t_0, t_1, ... t_{n_train-1}` in that order.

| shape | tiled matrix | fused, default | fused, best of a 13-point sweep |
|---|---|---|---|
| d = 220 | 36.9 ms | 109.8 ms | 91.1 ms (`k_cells` 2, 512 threads) |
| d = 11 | 26.9 ms | 22.7 ms | 21.3 ms (`k_cells` 2, 512 threads) |

The cause is the grid, not the arithmetic. The row-max pass chunks train rows
across grid y (391 chunks x 16 query blocks = 6,256 blocks) because a max may
be folded from chunk maxima. The sum pass cannot be chunked that way -- the
fold is a dependent chain over all of `j` -- so it loses the grid-y dimension
and runs 125 blocks on 132 SMs. Measured alone it is 84.6 ms at d = 220
against the 36.9 ms of the three matrix stages it replaces. Its deficit
shrinks as `n_query` grows, so a caller scoring far more than 2,000 queries is
the case to re-measure; that is what the `fused_only` arm is for.

## 3. DEVIATION 2691, transposing the matrix: 4x slower

The fold reads `logk[q * n_train + j]`, one thread per query, so a warp reads
32 addresses `n_train * 4` bytes apart -- the textbook uncoalesced access,
worth 9.48 ms. Train-major (`j * n_query + q`) makes a warp read 128
contiguous bytes per step. The argument is clean; the measurement refuses it.

| stage | query-major (shipped) | train-major |
|---|---|---|
| serial fold, d = 220 | 9.48 ms | 37.74 ms |
| serial fold, d = 11 | 9.48 ms | 37.74 ms |
| whole tiled entry, d = 220 | 36.9 ms | 64.1 ms |
| whole tiled entry, d = 11 | 26.9 ms | 52.5 ms |

Counting transactions per warp-step misses what each layout does to a
THREAD's stream. Query-major gives each thread one long sequential run, so a
single 128-byte line serves 32 of its iterations. Train-major gives each
thread an `n_query * 4` byte stride, so each line is touched for 4 bytes and
dropped, and 2,000 strided streams thrash where 2,000 sequential ones
streamed.

## 4. The races: the lane changed nothing, and that is the claim

`tools/classical_two_datasets.py race`, arms interleaved round by round, 1
warm-up plus 5 rounds, ms median (min..max). `ours` is this branch, `ours-base`
is `origin/main` built on this same pod, so both are OURS and neither is an
opponent row. TWO FULL ROUNDS were run because round 1's Istella-S
`ours/ours-base` of 0.9182 looked like an effect and had no mechanism behind
it; it did not reproduce.

| dataset | round | cuML ms | ours ms | ours-base ms | ours / base | ours / cuML |
|---|---|---|---|---|---|---|
| taxi | 1 | 2.54 (2.15..2.64) | 28.96 (28.94..29.17) | 29.16 (28.98..29.26) | 0.9932 | 11.40x |
| taxi | 2 | 3.00 (2.61..3.23) | 29.39 (29.08..30.12) | 28.92 (28.77..29.28) | 1.0165 | 9.79x |
| Istella-S | 1 | 7.37 (7.31..7.79) | 64.30 (63.81..66.35) | 70.03 (68.95..70.50) | 0.9182 | 8.72x |
| Istella-S | 2 | 7.36 (7.19..7.44) | 70.49 (69.62..71.98) | 70.16 (69.28..72.20) | 1.0047 | 9.58x |

Parity on both datasets across the two rounds, which is the designed outcome:
the shipped path is main's. Quality is equal to every digit the harness prints
between `ours` and `ours-base` on both datasets (taxi -9.582071959257126,
Istella-S -212.1174684753418) with no query left without a density; cuML reads
-9.58205059337616 and -212.11746494293212 on the same blocks.

## Identity

Nothing in this lane moves a bit, and it is gated, not argued.

* `kde_check_identical.log`: the whole IDENTICAL `kde_check`, 15 checks green.
  `check_kde_tiled_equals_staged` now runs FIVE arms against the staged
  reference -- the entry's dispatch, the query-major and train-major matrices
  on one alternative schedule each, the fused pass on its default schedule,
  and the fused pass at `k_cells` 16 / `sum_tpb` 128 (4 helper threads per
  query, 32 queries a block, so a DIFFERENT thread computes each term while
  the SAME owner folds them in the same ascending order) -- 33,300 scores,
  0 differ.
* `profile_stages.log`: 150 schedule / layout / structure combinations hashed
  EQUAL on both shapes, at the two fingerprints this lane has carried since
  DEVIATION 2625: FNV 16594497303053393111 at d = 220 and
  17888536843391681998 at d = 11.
* The race digests are equal between `ours` and `ours-base` on both datasets
  (taxi `aa8ac4159ad2cbfa`, Istella-S `81d11ed7fcd9eb38`), and they are the
  same digests the 2026-09-11 lane recorded.

## RUN OWED

`pixi run check-kde` under IDENTICAL on the Apple M4 and on an AMD MI300X.
The fused kernel and the layout parameter are new code on those vendors too,
and the tiled pass is dispatched there.
