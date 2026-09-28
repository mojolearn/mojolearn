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

(pending: steward jobs 1790604402501 on m4-a)

## Unproven

Everything above until its A/B lands.

## Shared code touched (the integration run must cover)

- gemm/checks/gemm_identical.mojo: PLAN_APPLE_MMA_SPLIT (21), the GROUPED
  APPLE_MMA kernel (one extra Int32 argument on the shipped APPLE_MMA
  launches), APPLE_MMA_FAST (`apple_mma_applies` answers in FAST on Apple;
  no dispatcher reads it there).
