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

4. 93cf24571 `PLAN_APPLE_MMA_SPLIT_BIG` (22, SHARED GEMM FILE): the split
   over the default 64x64 tile; x_cnn candidate for m, n >= 64.
5. 5021979ab / 1e79c8b4c layout changes (GEMM rows <-> NCHW) as 32x32
   threadgroup tiles: conv_out (`conv_out_val`) and Conv2d's dout_rows
   default on (`-D MOJOLEARN_XCNN_NO_TILED_LAYOUT`); the pooled backward's
   rows measured slower tiled and are opt-in (`-D MOJOLEARN_XCNN_TILED_ROWS`).
6. d3d5ce0e7 APPLE_MMA for ONE ragged leaf (`apple_mma_applies_one_leaf`,
   the kernel's `wpl` for P == 1, SHARED GEMM FILE), timed against the
   dispatcher in IDENTICAL too (`-D MOJOLEARN_XCNN_NO_NT_TUNE`); the MMA
   split for bias gradients; each tuning candidate runs once untimed first.
7. b4d13d73c im2col one thread per (row, channel) (`im2col_taps_at`,
   `-D MOJOLEARN_XCNN_NO_IM2COL_TAPS`).
8. 1e79c8b4c GCNConv/SAGEConv host CSR: one stable int64 argsort per view
   (lexsort's order exactly) and bincount (legacy arm keeps lexsort).
9. 985673313 / cf1ee40fb the DIRECT first-layer convolution (Apple only):
   k = C*KH*KW one leaf of at most 32 (the first block's 27): each cell the
   contract's chain `rtf_mul_add` from +0.0 over flushed operands (the
   GEMM's exact step), taps in registers, weights flushed in threadgroup
   memory, cols stored only when the backward reads them, the NCHW word
   `conv_out_val`'s. No GEMM, no y2, no conv_out launch
   (`-D MOJOLEARN_XCNN_NO_DIRECT_CONV`).
10. 4afd3f1ec max pool backward: where windows tile the input (kernel ==
   stride, no pad/dilation) the one visited window's step directly
   (`-D MOJOLEARN_XCNN_NO_POOL_BWD_TILE`; x_cnn/ops.mojo, host twin too).
11. d6e3cbd4a merge of origin/lane/apple-merged (no file in common).
12. 209bdc3b4 GCNConv/SAGEConv reuse the graph built for the same content
   key (n, edge_index bytes, edge_weight bytes, flags); DC_MAXW 2048.
13. df6386e7d BatchNorm per-channel folds through threadgroup memory (one
   threadgroup per channel; thread 0 folds in the same order;
   `-D MOJOLEARN_XCNN_NO_BN_BLOCK`). The mean's image order now reads
   `bn_mean_row` (x_cnn/ops.mojo) on host and device, and
   x_cnn/checks/sabotage/seam_5702_bn_fold_order.patch is REGENERATED to
   reverse it there (applies; bites both columns as before, UNPROVEN until
   the combined run).

Every x_cnn sabotage patch still applies (`git apply --check`, 23 of 23);
none touches a line these changes replaced.

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

### m3ultra-b (Apple M3 Ultra), job 1790610432497, commit cf1ee40fb

(The request was queued at f9dab4da4 and retargeted in place, before it
started, to cf1ee40fb: the job's CABRUN line prints the commit it ran.)
base = every change off (`-D MOJOLEARN_XCNN_NO_MMA_SPLIT -D
MOJOLEARN_XCNN_NO_NT_TUNE -D MOJOLEARN_XCNN_NO_RES_POOL -D
MOJOLEARN_XCNN_NO_TILED_LAYOUT -D MOJOLEARN_XCNN_NO_IM2COL_TAPS -D
MOJOLEARN_XCNN_NO_DIRECT_CONV -D MOJOLEARN_XCNN_NO_POOL_BWD_TILE` + legacy)
= round 1's code (its fit 2048 147.8 ms against round 1's 144.3); all =
default. Median of two rounds, ms.

| shape | IDENTICAL base | IDENTICAL all | FAST base | FAST all |
|---|---|---|---|---|
| Conv2d 3->64 fwd / bwd | 28.1 / 6.5 | 25.4 / 4.5 | 28.0 / 8.2 | 24.6 / 4.2 |
| Conv2d 64->64 fwd / bwd | 39.3 / 72.8 | 34.2 / 44.7 | 41.2 / 60.0 | 33.3 / 42.4 |
| Conv2d 64->128 fwd / bwd | 18.5 / 20.1 | 17.3 / 18.9 | 20.8 / 21.0 | 16.8 / 17.0 |
| CNNClassifier fit 2048 | 147.8 | 75.0 (2.0x) | 129.2 | 68.0 (1.9x) |
| CNNClassifier fit 8192 | 545.0 | 290.1 (1.9x) | 466.9 | 265.8 (1.8x) |
| predict_proba 2048 | 57.6 | 24.4 (2.4x) | 52.5 | 19.6 (2.7x) |
| predict_proba 8192 | 204.6 | 72.2 (2.8x) | 213.4 | 58.2 (3.7x) |
| BasicBlock 64 N64 H32 fwd / fwd+bwd | 78.3 / 174.1 | 80.2 / 153.5 | 78.3 / 162.7 | 77.2 / 144.1 |
| BatchNorm2d 64 N64 H32 fwd / bwd | 23.8 / 18.6 | 23.6 / 18.4 | 23.0 / 17.6 | 22.4 / 16.6 |
| MaxPool2d 2 N256 C64 H32 fwd / bwd | 11.3 / 28.2 | 11.6 / 27.2 | 11.7 / 28.9 | 11.1 / 27.0 |
| GCNConv 100k nodes 1M edges fwd / bwd | 520.3 / 31.8 | 247.7 / 30.9 | 519.5 / 30.4 | 247.6 / 30.4 |
| SAGEConv 100k / 1M fwd / bwd | 494.3 / 56.9 | 236.0 / 54.4 | 496.1 / 54.8 | 236.6 / 54.0 |

Bits: IDENTICAL, all 8 digest lines equal in all 4 runs (2 arms x 2
rounds), the same words as round 1 and the M4 / M4 Pro runs; FAST, all 8
digest lines equal in all 4 runs and all 20 fastq.py rows equal between
the arms (the same values as on the M4 and M4 Pro).
Stages (all vs base, ms): block 1 shipped forward 1.66 -> 0.98 (direct
conv), weight gradient 0.94 -> 0.35; block 2 weight gradient 4.59 -> 0.88
(APPLE_MMA_SPLIT_BIG), im2col 0.89 -> 0.30, conv_out 0.20 -> 0.11.
Plan sweep: 64x288x65536 4.68 -> 0.88 ms, 64x576x262144 26.6 -> 4.25 ms
(APPLE_MMA_SPLIT_BIG, 0 mismatches).

### m4pro-a (Apple M4 Pro), job 1790614006220, commit 2144a9548 (retargeted in place before it started)

base = every change off (all the defines above plus `-D
MOJOLEARN_XCNN_NO_BN_BLOCK`, + legacy) = round 1's code; all = default.
Median of two rounds, ms.

| shape | IDENTICAL base | IDENTICAL all | FAST base | FAST all |
|---|---|---|---|---|
| Conv2d 3->64 fwd / bwd | 29.8 / 12.7 | 29.0 / 7.0 | 29.6 / 17.0 | 27.3 / 6.5 |
| Conv2d 64->64 fwd / bwd | 51.7 / 102.0 | 40.6 / 66.9 | 57.7 / 107.1 | 38.5 / 60.5 |
| Conv2d 64->128 fwd / bwd | 21.4 / 28.8 | 18.2 / 25.8 | 24.2 / 38.8 | 17.1 / 23.6 |
| CNNClassifier fit 2048 | 225.9 | 125.1 (1.8x) | 215.1 | 108.1 (2.0x) |
| CNNClassifier fit 8192 | 855.5 | 494.4 (1.7x) | 821.0 | 427.6 (1.9x) |
| predict_proba 2048 | 105.1 | 43.9 (2.4x) | 101.2 | 34.8 (2.9x) |
| predict_proba 8192 | 350.1 | 157.9 (2.2x) | 340.5 | 121.8 (2.8x) |
| BasicBlock fwd / fwd+bwd | 96.3 / 197.2 | 72.7 / 152.9 | 89.2 / 198.8 | 67.0 / 138.8 |
| BatchNorm2d fwd / bwd | 22.9 / 16.4 | 16.1 / 12.9 | 19.3 / 15.3 | 14.1 / 10.9 |
| MaxPool2d fwd / bwd | 13.8 / 27.8 | 13.7 / 26.9 | 13.7 / 27.5 | 13.8 / 26.9 |
| GCNConv fwd / bwd | 499.9 / 38.1 | 50.2 / 34.7 | 490.0 / 34.3 | 49.6 / 34.0 |
| SAGEConv fwd / bwd | 468.4 / 63.1 | 58.9 / 56.6 | 469.4 / 58.5 | 58.7 / 57.0 |

Bits: IDENTICAL 9 digest lines (BatchNorm2d's added) equal in all 4 runs;
FAST 9 digest lines equal in all 4 runs and 20/20 fastq.py rows equal.

### m3ultra-b (Apple M3 Ultra), job 1790615497573, commit 2144a9548

base = round 1's code (every define above + legacy); all = default.
Median of two rounds, ms.

| shape | IDENTICAL base | IDENTICAL all | FAST base | FAST all |
|---|---|---|---|---|
| Conv2d 3->64 fwd / bwd | 28.5 / 6.5 | 25.5 / 4.5 | 27.9 / 8.2 | 26.2 / 4.3 |
| Conv2d 64->64 fwd / bwd | 39.3 / 73.0 | 34.2 / 44.8 | 41.2 / 60.0 | 33.3 / 42.5 |
| Conv2d 64->128 fwd / bwd | 18.9 / 20.2 | 17.5 / 18.9 | 20.7 / 21.0 | 16.8 / 17.0 |
| CNNClassifier fit 2048 | 148.2 | 74.5 (2.0x) | 127.2 | 68.3 (1.9x) |
| CNNClassifier fit 8192 | 545.6 | 294.9 (1.9x) | 461.2 | 267.1 (1.7x) |
| predict_proba 2048 | 58.4 | 23.2 (2.5x) | 54.0 | 20.5 (2.6x) |
| predict_proba 8192 | 206.5 | 72.0 (2.9x) | 212.0 | 58.5 (3.6x) |
| BasicBlock fwd / fwd+bwd | 79.4 / 174.8 | 65.2 / 131.3 | 79.2 / 160.9 | 59.7 / 120.8 |
| BatchNorm2d fwd / bwd | 21.9 / 17.1 | 14.4 / 11.0 | 20.6 / 15.2 | 12.4 / 9.4 |
| MaxPool2d fwd / bwd | 11.6 / 29.1 | 11.4 / 27.5 | 11.9 / 27.9 | 11.9 / 27.3 |
| GCNConv fwd / bwd | 520.7 / 31.5 | 49.0 / 29.4 | 533.0 / 30.4 | 48.6 / 29.1 |
| SAGEConv fwd / bwd | 495.1 / 56.8 | 58.8 / 53.4 | 493.9 / 55.4 | 59.3 / 53.5 |

Bits: IDENTICAL 9 digest lines equal in all 4 runs; FAST 9 digest lines
equal in all 4 runs and 20/20 fastq.py rows equal.

### m4-a (Apple M4), job 1790618938662, commit 93e40eaf2 (a quiet box: floor 0.18-0.22 ms)

base = round 1's code (every define + legacy); all = default. Median of
two rounds, ms.

| shape | IDENTICAL base | IDENTICAL all | FAST base | FAST all |
|---|---|---|---|---|
| Conv2d 3->64 fwd / bwd | 31.1 / 20.9 | 29.9 / 12.2 | 28.3 / 30.1 | 27.2 / 11.1 |
| Conv2d 64->64 fwd / bwd | 70.3 / 154.2 | 51.5 / 109.1 | 81.8 / 178.2 | 48.0 / 95.3 |
| Conv2d 64->128 fwd / bwd | 27.8 / 53.4 | 23.5 / 48.7 | 32.9 / 72.0 | 21.4 / 45.5 |
| CNNClassifier fit 2048 | 359.8 | 223.8 (1.6x) | 360.2 | 190.4 (1.9x) |
| CNNClassifier fit 8192 | 1393.0 | 885.4 (1.6x) | 1391.4 | 756.5 (1.8x) |
| predict_proba 2048 | 163.9 | 79.1 (2.1x) | 164.5 | 63.4 (2.6x) |
| predict_proba 8192 | 555.1 | 300.2 (1.8x) | 539.2 | 228.2 (2.4x) |
| BasicBlock fwd / fwd+bwd | 107.5 / 248.5 | 103.7 / 226.9 | 106.0 / 265.8 | 95.7 / 200.1 |
| BatchNorm2d fwd / bwd | 23.9 / 17.1 | 24.8 / 20.8 (SLOWER) | 20.1 / 15.2 | 22.0 / 16.9 (SLOWER) |
| MaxPool2d fwd / bwd | 14.1 / 28.4 | 13.8 / 25.1 | 13.8 / 29.0 | 13.8 / 25.0 |
| GCNConv fwd / bwd | 471.0 / 41.9 | 49.7 / 35.6 | 476.1 / 36.0 | 49.4 / 34.8 |
| SAGEConv fwd / bwd | 454.6 / 70.7 | 61.7 / 59.0 | 451.5 / 61.8 | 59.9 / 58.0 |

Bits: IDENTICAL 9 digest lines equal in all 4 runs; FAST 9 digest lines
equal in all 4 runs and 20/20 fastq.py rows equal.
The BatchNorm threadgroup folds are SLOWER on the 10-core M4 (and faster
on the M4 Pro and M3 Ultra): 770c1be66 reads eight staged words ahead of
the dependent adds; job 1790619765191 (m4-a) times it against
`-D MOJOLEARN_XCNN_NO_BN_BLOCK`.

## Unproven

- The regenerated seam 5702 sabotage patch: applies, not yet run.

## Shared code touched (the integration run must cover)

- gemm/checks/gemm_identical.mojo: PLAN_APPLE_MMA_SPLIT (21), the GROUPED
  APPLE_MMA kernel (one extra Int32 argument on the shipped APPLE_MMA
  launches), APPLE_MMA_FAST (`apple_mma_applies` answers in FAST on Apple;
  no dispatcher reads it there), PLAN_APPLE_MMA_SPLIT_BIG (22),
  `apple_mma_applies_one_leaf` and the kernel's `wpl` for P == 1 (the
  same value wherever L % KB == 0, the only case the dispatchers send).
