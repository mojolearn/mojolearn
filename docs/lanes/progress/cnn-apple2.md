# cnn-apple2: progress (Apple speed round 2, IDENTICAL and FAST)

Branch `lane/cnn-apple2` (worktree ~/mojolearn-wt/cnn-apple2), forked from
lane/apple-merged 037daa353. Brief: ~/mojolearn-evidence/apple2_speed_brief.md.
Evidence: ~/mojolearn-evidence/cnn-apple2/ (every steward stdout).

Family: Conv2d/Conv1d, pooling, BatchNorm, Dropout2d, adaptive pooling,
BasicBlock, GCNConv/SAGEConv, CNNClassifier (x_cnn). The measured surface is
round 1's: Conv2d N256 fwd/bwd at three shapes and CNNClassifier (32,64)
batch 256 fit/predict at 2048 and 8192 rows (`tools/apple_speed_cnn/profile.py`).

## Where round 1 left the Apple cost (37e50ac45, IDENTICAL)

M3 Ultra fit 2048 = 144 ms: conv block backward 65, forward 35, res_alloc 14
(35 calls), sgd 12 (48 calls, one wait each), gather 6.6. M4 fit 387 ms:
backward 217, forward 119. Per M4 step, block 2's weight gradient
(64 x 288 x 65536, OP_TN) alone is 11 ms (TUNED 32x32; the Ultra 4.4 ms on
SPLIT 64x64): the output is 18 small tiles, so APPLE_MMA ran 18 blocks each
walking all of k (14.7 ms), and the split plans ran on the FMA units.
predict 2048 on the Ultra: 28 ms of kernels, 20 ms of res_alloc (17 large
arrays, the gradient arrays included).

## Changes (A/B arms in ONE job per Mac: tools/apple_speed_cnn/ab.sh)

1. e2e93bf86 `PLAN_APPLE_MMA_SPLIT` (21, gemm/checks/gemm_identical.mojo,
   SHARED GEMM FILE): the APPLE_MMA kernel (GROUPED form) at the small tile
   over (tiles, aligned power-of-two leaf groups) on grid.y, each group's
   node to the workspace, then `_ksplit_fold_launch` (the SPLIT fold). The
   long-k group argument (Lemmas A and B). Group size: smallest power of two
   with tiles x groups <= 1024 (`MOJOLEARN_APPLE_MMA_SPLIT_BLOCKS`). Forced
   only: no dispatcher picks it, outside `range(GEMM_PLAN_COUNT)`. The
   ungrouped APPLE_MMA kernel gained one unread Int32 argument (0). x_cnn's
   `_apple_tuned_plan` times it among its OP_TN candidates (n >= 8).
   Before arm: `-D MOJOLEARN_XCNN_NO_MMA_SPLIT`. gemm_rtf_boundary_check
   now also runs the new plan where APPLE_MMA applies.
2. 62d6f9649 trainer plumbing (bits cannot move: the same launches):
   `x_cnn_sgd_r`/`x_cnn_adam_r` list forms (every parameter, one wait),
   `x_cnn_res_gather` pair form (batch and labels, one wait), a resident
   pool in `res_free`/`res_alloc` (<= 256 MB, zero filled on reuse;
   before arm `-D MOJOLEARN_XCNN_NO_RES_POOL`), predict_proba at most 2048
   rows per pass without the gradient arrays. Python before arm:
   `_LEGACY_STEP` (profile.py word `legacy`). Host twin has the list forms.
3. b59267dfa FAST: `APPLE_MMA_FAST` (SHARED GEMM FILE) compiles the APPLE_MMA
   kernels in FAST on Apple for callers that NAME the plan (every full
   window on the matrix path); x_cnn FAST times the OP_TN candidates and,
   elsewhere, the dispatcher's pick against APPLE_MMA once per shape.
   Before arm `-D MOJOLEARN_XCNN_NO_FAST_TUNE`. FAST bits may move: the
   paired quality set (fastq.py) runs per arm.

## Results

### m4-a (Apple M4), job 1790604402501, commit 3eead70b3, arms interleaved A B A B

The Mac was contended during this job (the 1-launch floor read 1.03 ms
against 0.20 ms in round 1), so every absolute number is about 2.4x round
1's; both arms saw the same box. Medians of the two rounds, ms.

IDENTICAL (base = `-D MOJOLEARN_XCNN_NO_MMA_SPLIT -D MOJOLEARN_XCNN_NO_RES_POOL`
+ `legacy`; mma = the split plan only; all = default):
| shape | base | mma | all |
|---|---|---|---|
| Conv2d 3->64 fwd / bwd | 40.0 / 46.3 | 39.9 / 40.5 | 39.5 / 33.0 |
| Conv2d 64->64 fwd / bwd | 144.7 / 262.9 | 143.5 / 240.8 | 141.8 / 239.9 |
| Conv2d 64->128 fwd / bwd | 48.3 / 95.9 | 48.6 / 93.5 | 48.6 / 93.9 |
| fit 2048 / 8192 | 935.1 / 3650.0 | 854.1 / 3336.5 | 782.5 / 3156.0 |
| predict 2048 / 8192 | 271.0 / 1095.9 | 272.5 / 1069.0 | 256.0 / 984.9 |
Digests equal in every row of every arm (fit weights d1bb646f6a0f113c /
66e7cc1839d709b9, proba 4c7b4f51477eeb51 / 138f69bae45669cb, conv y and bwd
as round 1). Calls per fit 252 -> 204; res_alloc 15.2 -> 3.7 ms.
Plan sweep (same job, every plan 0 mismatches): 64x288x65536 APPLE_MMA 13.3,
TUNED 32x32 23.7, APPLE_MMA_SPLIT 8.2 ms; 32x27x262144 SPLIT 32x32 8.7 ->
MMA_SPLIT 3.0; 64x576x262144 MMA 77.0 -> MMA_SPLIT 61.7; 64x27x262144
16.7 -> 4.8; 128x576x65536 36.9 -> 30.7.

FAST (base = `-D MOJOLEARN_XCNN_NO_FAST_TUNE -D MOJOLEARN_XCNN_NO_RES_POOL` + legacy):
| shape | base | all |
|---|---|---|
| Conv2d 3->64 fwd / bwd | 33.5 / 67.7 | 32.1 / 48.1 (31.0, 65.1: rounds differ) |
| Conv2d 64->64 fwd / bwd | 161.8 / 318.7 | 130.5 / 219.9 |
| Conv2d 64->128 fwd / bwd | 60.0 / 129.5 | 46.4 / 81.1 |
| fit 2048 / 8192 | 928.0 / 3610.1 | 744.0 / 2919.5 |
| predict 2048 / 8192 | 264.6 / 1043.9 | 206.5 / 799.5 |
FAST digests: base == all in every row (fit 2048 weights 9d12d28efad9d75e,
8192 d995f35c642015a0; conv y/bwd equal to IDENTICAL's). The paired quality
set (fastq.py, 20 rows) is identical row for row (same probability and
output digests): blobs acc 1.0 x5, stripes 0.9570 / 0.9775 / 0.9980 /
0.9326 / 0.9971, conv max rel err 1.95e-7 to 3.29e-7. FAST quality cannot
have moved: the words did not.

## Unproven

- 93cf24571 (PLAN_APPLE_MMA_SPLIT_BIG): job 1790606977177 (m4pro-a) pending.
- No M3 Ultra or M4 Pro number yet for any change.

## Shared code touched (the integration run must cover)

- gemm/checks/gemm_identical.mojo: PLAN_APPLE_MMA_SPLIT (21), the GROUPED
  APPLE_MMA kernel (one extra Int32 argument on the shipped APPLE_MMA
  launches), APPLE_MMA_FAST (`apple_mma_applies` answers in FAST on Apple;
  no dispatcher reads it there).
