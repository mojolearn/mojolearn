# cnn-apple: progress (Apple Metal speed, FAST and IDENTICAL)

Branch `lane/cnn-apple` (worktree ~/mojolearn-wt/cnn-apple), merged with
origin/lane/merged at 31d27e477. Not merged to main (the orchestrator merges
the Apple branches into lane/apple-merged). Evidence:
~/mojolearn-evidence/cnn-apple/ (every steward stdout).

Tools (steward speed jobs, `--builds bindings/build_x_cnn.sh`):
- `tools/apple_speed_cnn/profile.sh`: `XCNN-SPEED` (the phase-4 bench:
  Conv2d N256 fwd/bwd at three shapes, CNNClassifier (32,64) batch 256 fit and
  predict_proba at 2048 and 8192 rows, median), `XCNN-ENTRY` (every binding
  call of one fit wrapped in a wall clock; each x_cnn entry synchronizes, so an
  entry's wall is its GPU time plus one wait), `XCNN-FLOOR` (one launch + one
  wait), `XCNN-DIGEST` (sha256 of weights, losses, probabilities, conv
  outputs: the before/after bit check on one column), and the GEMM plan sweep.
- `tools/apple_speed_cnn/gemm_plans.mojo`: every execution plan of the pinned
  GEMM at the x_cnn shapes, ms per call (GPU time: R calls, one wait) and the
  words that differ from the shipped dispatch.
- `tools/apple_speed_cnn/fastq.py`: the FAST paired quality set.

## Profile first (M4 m4-a, main 33917d8bb, IDENTICAL)

- Syncs are NOT the Apple cost here: one launch + one wait = 0.21 ms, a
  1-float download 0.03 ms; a fit of 2048 rows makes 252 binding calls.
- fit 2048 = 889 ms: conv block backward 500 ms (16 calls), conv block
  forward 336 ms (16 calls), everything else 54 ms.
- GEMM per trainer step (sweep): 4.6 (block 1 fwd, k=27) + 3.7 + 3.3 (block 1
  dW, db) + 2.1 (block 2 fwd, APPLE_MMA) + 24.0 (block 2 dW, x_cnn's forced
  SPLIT 64x64) + 1.9 (db) + 2.9 (dx) + ~1.3 (head) = ~44 ms of the ~111 ms
  step. The rest (~67 ms) was the element kernels: their index decoding is
  64-bit Int division (5 to 6 per element), which no GPU does in hardware.

## Changes (DEVIATION 5720, x_cnn/README.md)

1. 459e8dec8: `_ud`/`_um` 32-bit unsigned index division in im2col, conv
   output, dout rows, col2im, the batch gather and the max pool (every operand
   a non-negative index < 2^31: the same quotient). On Apple IDENTICAL, the
   weight/bias gradient GEMM plans from the M4 sweep: n == 1 -> SPLIT 16x16,
   wide n >= 64 -> APPLE_MMA (x_cnn's forced SPLIT plans were tuned on the RTX
   4090).
2. e0380cc28: ReLU + max pool forward in one launch (`relu_maxpool_fwd_at`),
   max pool backward + ReLU backward + GEMM row layout in one launch
   (`pool_relu_rows_bwd_at`), both built from the value functions the separate
   launches store (so the same words); small wide dW (m*n <= 32768, k <=
   131072) on TUNED 32x32 (14.0 -> 11.0 ms at 64x288x65536); `res_alloc` no
   longer waits (`res_free` does). Sabotage arms 5700 and 5706 regenerated for
   the new context (5706 now sits in the shared `_maxpool_fwd` body, so it
   bites the fused path and the plain one).
3. c78359e22: the FAST quality script. 31d27e477: merge of origin/lane/merged.
4. 37e50ac45: the fixed Apple dW/db rule of (1)-(2) was measured on the M4
   only and made the M3 Ultra's Conv2d 64->64 backward SLOWER (95.7 -> 114.2
   ms): the Ultra's 60 cores want the split plans (m3ultra sweep: 64x576x262144
   SPLIT 64x64 26.6 vs MMA 68.6 ms; 64x288x65536 SPLIT 64x64 4.4 vs TUNED
   9.3 vs MMA 17.6 ms), the M4's 10 cores want MMA/TUNED. Now
   `_apple_tuned_plan` times the candidates that were ever competitive (the
   default split plan, SPLIT 16x16, and for n >= 64 TUNED 32x32 and
   APPLE_MMA; for n == 1 SPLITK when it fits) once per shape per process
   (one run each, after a wait) and caches the fastest. Every candidate
   stores the same bits (contract 6.1: the execution plan may look at the
   device). The M4 numbers are unchanged by it; the Ultra's regression is
   gone.

## Results (median ms; before = 33917d8bb = main's x_cnn; digests before == after in every row)

M4 (m4-a), IDENTICAL, before -> after (e0380cc28):
| shape | before | after |
|---|---|---|
| Conv2d N256 3->64 32x32 fwd / bwd | 46.1 / 34.3 | 34.9 / 19.6 |
| Conv2d N256 64->64 32x32 fwd / bwd | 184.1 / 353.4 | 68.6 / 148.6 |
| Conv2d N256 64->128 16x16 fwd / bwd | 57.3 / 75.5 | 27.4 / 43.3 |
| CNNClassifier fit 2048 / 8192 rows | 889.2 / 3510.5 | 387.3 / 1499.4 (2.3x) |
| predict_proba 2048 / 8192 | 387.5 / 1730.4 | 134.2 / 535.0 (2.9-3.2x) |

M4 (m4-a), FAST, before -> after (c78359e22, the same GPU code):
| shape | before | after |
|---|---|---|
| Conv2d N256 3->64 32x32 fwd / bwd | 43.9 / 40.9 | 33.4 / 29.1 |
| Conv2d N256 64->64 32x32 fwd / bwd | 193.7 / 279.2 | 78.2 / 154.3 |
| Conv2d N256 64->128 16x16 fwd / bwd | 61.2 / 90.4 | 31.4 / 59.4 |
| CNNClassifier fit 2048 / 8192 rows | 724.5 / 2845.3 | 344.5 / 1330.9 (2.1x) |
| predict_proba 2048 / 8192 | 383.1 / 1627.7 | 130.2 / 537.9 (2.9-3.0x) |

M3 Ultra (m3ultra), IDENTICAL, before -> after (37e50ac45, the measured plan):
| shape | before | after |
|---|---|---|
| Conv2d N256 3->64 32x32 fwd / bwd | 34.8 / 10.0 | 30.2 / 6.6 |
| Conv2d N256 64->64 32x32 fwd / bwd | 60.1 / 95.7 | 41.2 / 73.8 |
| Conv2d N256 64->128 16x16 fwd / bwd | 24.6 / 36.4 | 19.0 / 30.8 |
| CNNClassifier fit 2048 / 8192 | 229.7 / 870.6 | 144.3 / 527.4 (1.6x) |
| predict_proba 2048 / 8192 | 97.8 / 399.4 | 51.8 / 194.1 (1.9-2.1x) |
M4 (m4-a) at 37e50ac45: the same as e0380cc28 within noise (fit 386.9 / 1500.9,
predict 134.2 / 580.1, Conv2d 64->64 bwd 148.6).

M3 Ultra (m3ultra), before -> after (e0380cc28, the fixed M4 rule; superseded for IDENTICAL):
| shape | IDENTICAL before -> after | FAST before -> after |
|---|---|---|
| Conv2d N256 3->64 32x32 fwd / bwd | 34.8 / 10.0 -> 30.0 / 7.3 | 33.9 / 10.7 -> 28.0 / 9.1 |
| Conv2d N256 64->64 32x32 fwd / bwd | 60.1 / 95.7 -> 39.6 / 114.2 (bwd SLOWER, see below) | 61.8 / 81.8 -> 41.7 / 59.3 |
| Conv2d N256 64->128 16x16 fwd / bwd | 24.6 / 36.4 -> 19.0 / 30.6 | 25.3 / 26.3 -> 20.1 / 20.8 |
| CNNClassifier fit 2048 / 8192 | 229.7 / 870.6 -> 195.5 / 707.7 | 198.3 / 745.2 -> 125.3 / 443.2 |
| predict_proba 2048 / 8192 | 97.8 / 399.4 -> 56.9 / 194.3 | 97.1 / 395.9 -> 50.2 / 189.7 |

FAST paired quality (m4-a, before 33917d8bb vs after c78359e22, `fastq.py`):
5 seeds x 2 seeded datasets (blobs, stripes; 2048 train / 1024 test,
CNNClassifier (32,64), 3 epochs) and 5 seeds x 2 Conv2d shapes against a
float64 reference. Every one of the 20 rows is BIT-IDENTICAL before and after
(same probability/output digests), so quality is unchanged by construction:
blobs acc 1.0000 x5 (log loss 0.0012-0.0026); stripes acc 0.9570, 0.9775,
0.9980, 0.9326, 0.9971 (log loss 0.094-0.272); conv max rel err 1.95e-7 to
2.62e-7 (3->64), 2.88e-7 to 3.29e-7 (64->64).

## Findings for other lanes

- PLAN_SPLITK (6) returns an untouched output on Metal when m*n*256 threads
  exceed 2^32 (m*n >= 16.7M: 262144x64, 65536x288, 65536x576): every word
  differs. The dispatcher never picks SPLITK there (it needs m*n <= 4096 or
  <= 24), so no shipped path reaches it; a forced-plan caller would.
- On Apple the IDENTICAL split/tuned GEMM plans run 2-4x slower than the same
  plans in FAST (block 2 dW 64x288x65536: 11.0 IDENTICAL vs 6.6 ms FAST): the
  Apple FMA repair in the pinned GEMM, the GEMM lane's territory.

M2 Pro (m2pro), IDENTICAL, before 33917d8bb -> after 37e50ac45 (digests equal in every row):
| shape | before | after |
|---|---|---|
| Conv2d N256 3->64 32x32 fwd / bwd | 57.5 / 47.3 | 47.7 / 20.9 |
| Conv2d N256 64->64 32x32 fwd / bwd | 183.2 / 512.5 | 73.1 / 199.3 |
| Conv2d N256 64->128 16x16 fwd / bwd | 58.4 / 90.3 | 30.0 / 54.6 |
| CNNClassifier fit 2048 / 8192 | 1129.5 / 4453.8 | 465.1 / 1774.3 (2.4-2.5x) |
| predict_proba 2048 / 8192 | 408.4 / 1705.3 | 171.1 / 688.6 (2.4-2.5x) |
M2 Pro dW sweep (1790584048397): its fastest weight-gradient plans match the M4's (TUNED 32x32 /
APPLE_MMA); its fastest bias-gradient plan is SPLITK, a candidate of `_apple_tuned_plan`.
