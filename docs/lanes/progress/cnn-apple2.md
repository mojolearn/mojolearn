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

### m4pro-a (Apple M4 Pro), job 1790607883094, commit 5021979ab (a quiet box)

base = every lane/cnn-apple2 change off (`-D MOJOLEARN_XCNN_NO_MMA_SPLIT
-D MOJOLEARN_XCNN_NO_RES_POOL -D MOJOLEARN_XCNN_NO_TILED_LAYOUT` + legacy);
notile = all but the tiled layout; all = default. Median of two rounds, ms.

IDENTICAL:
| shape | base | notile | all |
|---|---|---|---|
| Conv2d 3->64 fwd / bwd | 29.8 / 12.6 | 30.0 / 11.2 | 28.9 / 8.5 |
| Conv2d 64->64 fwd / bwd | 51.7 / 122.0 | 52.0 / 81.5 | 50.8 / 80.0 |
| Conv2d 64->128 fwd / bwd | 21.3 / 28.8 | 21.3 / 28.9 | 20.7 / 28.3 |
| fit 2048 / 8192 | 226.3 / 864.0 | 166.7 / 659.7 | 162.4 / 648.1 (-28% / -25%) |
| predict 2048 / 8192 | 106.0 / 357.1 | 72.2 / 270.8 | 66.3 / 247.3 (-37% / -31%) |
Digests equal in every row of every arm (the same as m4-a and round 1).
Plan sweep: 64x288x65536 APPLE_MMA_SPLIT_BIG 1.90 ms (x_cnn's old pick
11.1); 64x576x262144 10.95 (72.7); 128x576x65536 5.78 (9.27).

FAST (base = `-D MOJOLEARN_XCNN_NO_FAST_TUNE -D MOJOLEARN_XCNN_NO_RES_POOL
-D MOJOLEARN_XCNN_NO_TILED_LAYOUT` + legacy):
| shape | base | all |
|---|---|---|
| Conv2d 3->64 fwd / bwd | 29.0 / 17.0 | 27.6 / 11.7 |
| Conv2d 64->64 fwd / bwd | 57.1 / 106.6 | 48.5 / 74.0 |
| Conv2d 64->128 fwd / bwd | 24.4 / 38.9 | 19.7 / 26.2 |
| fit 2048 / 8192 | 217.6 / 822.4 | 140.0 / 555.9 (-36% / -32%) |
| predict 2048 / 8192 | 100.7 / 354.6 | 53.6 / 197.8 (-47% / -44%) |
FAST digests base == all in every row; fastq.py 20 rows identical row for
row (the same values as on m4-a).

Stage profile (stages.mojo, M4 Pro, all arm, ms per launch; the layout
kernels here are still the one-thread-per-element ones): block 1 forward
GEMM (262144 x 32 x 27, TUNED) 2.62, pooled backward rows 1.44, bias
gradient 0.96, conv_out 0.89, im2col 0.77, weight gradient 0.70 (MMA
split); block 2 im2col 1.95, weight gradient 1.90, input gradient 1.75,
col2im 1.22, forward GEMM 1.45. In the notile arm the tuner had picked a
split plan for 32x27x262144 (2.30 ms) on a first-use run: fixed in
d3d5ce0e7 (one untimed run per candidate).

### m4pro-a (Apple M4 Pro), job 1790609769289, commit 39f622a4d

base = every change off (`-D MOJOLEARN_XCNN_NO_MMA_SPLIT -D
MOJOLEARN_XCNN_NO_RES_POOL -D MOJOLEARN_XCNN_NO_TILED_LAYOUT -D
MOJOLEARN_XCNN_NO_NT_TUNE -D MOJOLEARN_XCNN_NO_IM2COL_TAPS` + legacy);
mid = 5021979ab's set; all = default. Median of two rounds, ms.

IDENTICAL:
| shape | base | mid | all |
|---|---|---|---|
| Conv2d 3->64 fwd / bwd | 29.9 / 12.8 | 29.5 / 7.4 | 28.3 / 7.0 |
| Conv2d 64->64 fwd / bwd | 52.3 / 100.5 | 50.7 / 77.1 | 40.3 / 68.0 |
| Conv2d 64->128 fwd / bwd | 21.3 / 28.8 | 20.7 / 28.2 | 18.2 / 25.8 |
| fit 2048 / 8192 | 225.7 / 858.0 | 160.0 / 632.7 | 144.4 / 571.6 (-36% / -33%) |
| predict 2048 / 8192 | 103.2 / 353.5 | 66.4 / 247.7 | 52.5 / 192.5 (-49% / -46%) |
| BasicBlock 64 N64 H32 fwd / fwd+bwd | 96.7 / 201.8 | 94.2 / 194.1 | 92.2 / 184.5 |
| BatchNorm2d fwd / bwd | 23.9 / 19.6 | 23.3 / 19.0 | 24.4 / 19.3 (unchanged: no GEMM) |
| MaxPool2d fwd / bwd | 13.8 / 27.8 | 13.7 / 27.4 | 13.8 / 28.0 (unchanged) |
| GCNConv 100k/1M fwd / bwd | 503 / 37.9 | 490 / 34.0 | 491 / 33.5 |
| SAGEConv 100k/1M fwd / bwd | 470 / 63.8 | 481 / 57.0 | 469 / 56.8 |
Every digest line (Conv2d y/bwd, fit weights/losses/proba, BasicBlock,
GCNConv, SAGEConv) equal in all six runs.

FAST (base = `-D MOJOLEARN_XCNN_NO_FAST_TUNE -D MOJOLEARN_XCNN_NO_RES_POOL
-D MOJOLEARN_XCNN_NO_TILED_LAYOUT -D MOJOLEARN_XCNN_NO_IM2COL_TAPS` + legacy):
| shape | base | all |
|---|---|---|
| Conv2d 3->64 fwd / bwd | 29.2 / 17.0 | 26.4 / 6.5 |
| Conv2d 64->64 fwd / bwd | 57.5 / 110.6 | 38.2 / 59.5 |
| Conv2d 64->128 fwd / bwd | 24.3 / 38.9 | 17.0 / 23.5 |
| fit 2048 / 8192 | 215.4 / 822.2 | 117.7 / 465.6 (-45% / -43%) |
| predict 2048 / 8192 | 105.1 / 348.5 | 34.9 / 123.7 (-67% / -65%) |
| BasicBlock fwd / fwd+bwd | 92.8 / 201.3 | 80.3 / 162.9 |
| GCNConv / SAGEConv fwd | 490 / 471 | 494 / 480 |
FAST: every digest line and all 20 fastq.py rows equal between the arms.

Stages (all arm): im2col 0.75 -> 0.29 (block 1), 1.96 -> 0.70 (block 2);
conv_out tiled 0.89 -> 0.37, 0.44 -> 0.19; the TILED pooled backward rows
were SLOWER (1.44 -> 1.73, 0.75 -> 0.96): taken back to the element kernel
in 1e79c8b4c (opt-in `-D MOJOLEARN_XCNN_TILED_ROWS`). Block 1's forward
GEMM (262144 x 32 x 27) stayed 2.61 ms with the one-leaf MMA candidate.
GCNConv / SAGEConv forward is host NumPy (two lexsorts of 1.1M edges,
np.add.at): 1e79c8b4c builds the same CSR with one stable int64 argsort
per view and bincount (0.24 -> 0.09 s per order on the laptop, same order).

## Unproven

- 1e79c8b4c (rows back on the element kernel, GCN/SAGE host CSR):
  job 1790610432497 (m3ultra-b, commit f9dab4da4) pending.
- No M3 Ultra number yet for any change.

## Shared code touched (the integration run must cover)

- gemm/checks/gemm_identical.mojo: PLAN_APPLE_MMA_SPLIT (21), the GROUPED
  APPLE_MMA kernel (one extra Int32 argument on the shipped APPLE_MMA
  launches), APPLE_MMA_FAST (`apple_mma_applies` answers in FAST on Apple;
  no dispatcher reads it there), PLAN_APPLE_MMA_SPLIT_BIG (22),
  `apple_mma_applies_one_leaf` and the kernel's `wpl` for P == 1 (the
  same value wherever L % KB == 0, the only case the dispatchers send).
