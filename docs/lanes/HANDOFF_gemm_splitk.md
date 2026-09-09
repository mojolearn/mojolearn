# GEMM split-K lane handoff (2026-09-09)

Branch `lane/gemm-splitk`, worktree `.claude/worktrees/agent-a68e7f0e68dc3bc9f`,
based on main `71d2ba71` (which already contains `lane/gemm-identical` and
its handoff `docs/lanes/HANDOFF_gemm.md`). Every build and run below was on
rented NVIDIA H100 pods through `tools/gemm_remote_leg.sh` (driver
580.126.09, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`,
Mojo 1.0.0), one pod at a time, each terminated by the script and verified
gone (HTTP 404) before the next. Nothing was built or run on the Mac. The
L40S pool answered "There are no instances currently available" at 13:43
EDT, so the iteration legs ran on the H100 too; no L40S number in this
file is new.

## Commits (`%h parent %p`)

- `5dd9c43b parent 71d2ba71` opponent reference: the 20 H100 cuBLAS rows
  (fp32 and tf32) transcribed from the Aug 25 logs, no re-run
- `c6b41018 parent 5dd9c43b` the gemm leg takes an extra body
  (`MOJOLEARN_GEMM_LEG_EXTRA`); `tools/gemm_identical_leg_extra.sh`
- `f6706e2d parent c6b41018` leg 1, H100 at the starting kernels (logs),
  and `tools/gemm_identical_table.py`
- `8ffebd44 parent f6706e2d` the four SPLIT plans, the register-stack fold,
  the outer-contiguous staging mapping
- `6960e4be parent 8ffebd44` leg 2 (logs)
- `2ada9f43 parent 6960e4be` block-per-cell fold for small outputs, batched
  loads in the stack fold, KS = 32 on the small split tiles
- `97fda692 parent 2ada9f43` leg 3 (logs)
- `645bd42c parent 97fda692` the 64x64 split plan; KS = 32 probes of the
  two wide tuned plans (plans 16, 17)
- `b2ceb403 parent 645bd42c` leg 4 (logs); short plan names in the table tool
- `540fcb14 parent b2ceb403` the 64x64 tuned plan steps K by 32 where two
  pages fit the column's shared limit (the measured flip)
- `82d360e4 parent 540fcb14` leg 5, the flipped dispatcher (logs)
- this file: the last commit on the branch

Note on `5dd9c43b`: the brief asked for the H100 cuBLAS transcription in
`bench/OPPONENT_REFERENCE.md` on this branch and that is what the commit
holds (transcription only, the H100 GEMM section and its owed item 2).
Main has moved since; the same rows are repeated in the marked section
"Opponent rows" below so the merge can take either the commit or the
section.

## What changed and why

**Which callers reach v1's SPLITK plan under IDENTICAL.** `choose_gemm_plan`
is reached only through `identical_gemm_into` / `identical_gemm`. Callers:
the bench drivers (`bench/speed/gemm_speed_main.mojo` arm v1,
`bench/gemm_card_main.mojo`, `bench/gemm_price_main.mojo`,
`bench/gemm_ladder_main.mojo`, `bench/lanes_price_main.mojo`), the gates
(`gemm_device_check`, `checks/batch_invariance_check.mojo`,
`solver/checks/profile_dot.mojo`, `gemm_backward_check`), the PUBLIC
`mojolearn.linalg` surface (`bindings/_mojolearn_linalg.mojo` ->
`gemm/host_entry.mojo::identical_gemm_host` -> `identical_gemm`), and
`core/gemm.mojo::gemm_tn` only for `m > 128` (OP_TN, never a SPLITK
shape). The estimators' small Gram shapes (OLS, PCA at <= 128 features)
go through `core/gram_splitk.mojo` and never see v1's dispatcher; that is
the "core identical" column below (0.49 ms at gram.32x32x1M), untouched
by this lane. So the SPLITK rows are the linalg surface's and the
drivers' cost, not the estimators'. `GEMM_IDENT_SWAP_537` was not touched.

**Why SPLITK was 50 ms on the H100.** `identical_gemm_leaf_kernel` gives
thread `gid` the pair `(cell = gid // P, t = gid % P)`, so adjacent lanes
read adjacent LEAVES of one cell: addresses `L` floats apart (NT) or
`L * m` apart (TN), one sector per lane per step, no reuse. The fold that
follows was never the cost.

**The SPLIT plans (11-15), `gemm/checks/gemm_identical.mojo`.** The tuned
register-blocked kernel `identical_gemm_tuned_kernel` gained a comptime
`SPLIT` parameter. Under it `block_idx.x` is the output tile and
`block_idx.y` the fold position `t`; the block walks that one leaf's
windows through the same two shared pages, the same `_tuned_step` seam,
the same `_tuned_window` / `_leaf_bounds` clamp, and at the leaf boundary
stores `ftz(acc)` (seam 5d) to the workspace at `t * m * n + cell`
(leaf-major, one legal layout of contract 7.2.2's level 0) instead of
pushing it into the thread-local stack. A second launch folds each cell's
`P` partials: one BLOCK per cell through `identical_gemm_fold_kernel[True]`
(the existing level-wise threadgroup fold, now with a `LEAF_MAJOR` layout
parameter) when `m * n <= 16384`, else one THREAD per cell through
`identical_gemm_fold_stack_kernel` (`_fold_push`, the FLAT plan's register
stack, loads issued eight at a time ahead of the pushes). Every sabotage
arm is mirrored (`SAB_NODE_ORDER` on the store, `SAB_FOLD_SERIAL` /
`SAB_PAD_PLUS_ZERO` / `SAB_FOLD_STRIDE` in the folds). Plans:

| id | plan | per-thread tile, thread grid | KS |
|---|---|---|---|
| 11 | `PLAN_SPLIT_32_2X2` | 2x2 cells, 16x16 threads: 32x32 tile | 32 |
| 12 | `PLAN_SPLIT_16_1X1` | 1x1, 16x16 threads: 16x16 tile | 32 |
| 13 | `PLAN_SPLIT_1X256` | 1x1, 1x256 threads: 1x256 tile (m == 1) | 16 |
| 14 | `PLAN_SPLIT_8X64` | 1x2, 8x32 threads: 8x64 tile (m <= 8) | 16 |
| 15 | `PLAN_SPLIT_64_4X4` | 4x4, 16x16 threads: 64x64 tile | 32 |
| 16 | `PLAN_TUNED_64_4X4_K32` | the 64x64 tuned plan at KS = 32 (probe) | 32 |
| 17 | `PLAN_TUNED_128_8X8_K32` | the 128x128 tuned plan at KS = 32 (probe) | 32 |

`choose_gemm_plan`, at `P >= 4` and within the 256 MB workspace bound
`PLAN_SPLITK` already honoured: `m <= 8 and n >= 256` -> 13 (m == 1) or
14; then `m, n >= 64 and m n <= 128 K` -> 15; `m, n >= 32` -> 11;
`m, n >= 16` -> 12; then the old SPLITK domain (`m n <= 4096`) -> 12.
`P >= 4` keeps the batch-invariance fixtures' `k = 300` shapes on their
documented plans. Plans 16 and 17 are never chosen; the forced-plan sweep
times them. `identical_gemm_workspace_floats` returns `m n P` for the
split plans; `GEMM_PLAN_COUNT = 18`, so every all-plans gate covers them.

**The outer-contiguous staging mapping (the TN variant).** `_tuned_g2r` and
the register-to-shared copy take a second slot-to-(row, column) expression
when an operand is contiguous along its OUTER index and not along `p`
(OP_TN's A, OP_NN's B): consecutive threads take consecutive rows at one
`p`, so the warp reads one line instead of `NTH` scattered words. Same
floats into the same page slots; which mapping ran is a property of two
strides. This serves every tuned and split plan at the TN rows.

**Task 4, the larger K step.** The forced sweep at leg 4 timed the two
wide tuned plans at KS = 32: on the 64x64 tile (two pages, 37 KB) it is
4% to 7% faster than KS = 16 at every row that tile serves (m = 64 rows
2.317 vs 2.500 ms, 8.05 vs 8.68, 8.09 vs 8.73, 9.12 vs 9.86; pca.wide
0.028 vs 0.029; kmeans 0.019 vs 0.020); on the 128x128 tile (37 KB a
PAGE, so one page and no prefetch overlap) it is 17% SLOWER (1.948 vs
1.659 at qkv). Bits equal for both by the launch-invariance gate. The
switch flipped for the 64x64 plan: `TUNED_64_KS` is `TUNED_KBLK` (32)
where `lib_smem_pages_for` gives two pages for the 64x64 pair (NVIDIA,
AMD) and 16 otherwise (Apple, 32 KB), read through the matrix so no
vendor is named. The 128x128 plan, which serves every t512 row, stays at
16; those rows are 4.4x to 4.9x and the 1.5x target was not reached (see
Unfinished).

**Leg tooling.** `tools/gemm_remote_leg.sh` runs a local sh file named in
`MOJOLEARN_GEMM_LEG_EXTRA` after the gemm payload's device check and card,
on the same box and lease; `tools/gemm_identical_leg_extra.sh` is this
lane's body (HOST_CAP card, tuned probe on the dispatcher and on every
forced plan, speed arms v1 and core, `core/gemm_identity_check`), all
IDENTICAL, no opponent. `tools/gemm_identical_table.py` turns a leg's
`remote/identical/` into the ratio table below; its opponent columns are
transcribed from `bench/OPPONENT_REFERENCE.md` and must stay equal to it.

## Bit evidence

Every leg, under `bench/results/e1g/<stamp>/remote/`:

| leg | tree | stamp | pod | gates | probe | capped card |
|---|---|---|---|---|---|---|
| 1 | c6b41018 (kernels as main) | `2026-09-09_132755-nvidia-h100-identical-gemm` | p1czk9ziexvyye | 7 green, 68 shapes, 11 plans | 20 match, 0 MOVED | not run (17 comparable rows of the uncapped card hash-equal) |
| 2 | 8ffebd44 | `2026-09-09_134404-nvidia-h100-identical-splitk` | 3ywuhoc9kp29yl | 7 green, 76 shapes, 15 plans | 20 match, 0 MOVED | byte-equal |
| 3 | 2ada9f43 | `2026-09-09_135640-nvidia-h100-identical-splitk2` | 0ierbcvr931ivq | 7 green, 76 shapes, 15 plans | 20 match, 0 MOVED | byte-equal |
| 4 | 645bd42c | `2026-09-09_140650-nvidia-h100-identical-splitk3` | ijous9yh9y68r9 | 7 green, 82 shapes, 18 plans | 20 match, 0 MOVED | byte-equal |
| 5 | 540fcb14 | `2026-09-09_141549-nvidia-h100-identical-splitk4` | 8l4bb0nxvmel1a | 7 green, 82 shapes, 18 plans | 20 match, 0 MOVED | byte-equal |

"gates" is `device_check.log` (`gemm device kernel + gates: all green
[IDENTICAL]`, `check_device_matches_oracle` over that many shapes,
`check_device_is_launch_invariant` 6 shapes x that many plans, batch
invariance). "probe" is `identical/probe.log`'s last line (untuned plan
vs the dispatcher, which sends 12 of the 20 rows to SPLIT plans).
"capped card" is `identical/capped.card` (`MOJOLEARN_GEMM_CARD_HOST_CAP=1`),
comments stripped, `cmp` against
`bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card`
(60 records). The leg's own `nvidia.card` is the UNCAPPED device arm and
is not diffable against the known-good card (43 of 60 rows hash a
different element count), which is why every leg's step 9 says
"DIVERGENT" and the leg exits 1; that verdict is structural, not a bit.
`identical/gemm_identity.log` is green on every leg
(`check_gemm_tn_over_capacity_takes_v1`, `check_pinned_gemm_is_batch_invariant`,
`check_pinned_gemv_matches_oracle`).

## H100 ratio tables, ours identical against cuBLAS FP32

Opponent: `bench/OPPONENT_REFERENCE.md`, H100 GEMM table (cuBLAS through
torch 2.4.1+cu124, `e1g/2026-08-25_155542`). Ours: medians of 5 rounds
from `remote/identical/speed_v1.log` and `speed_core.log` of the leg
named. A ratio above 1 is how many times slower the identical arm is
than cuBLAS; it is an internal cost, never a speed claim. "core" is
`core/gemm.mojo`'s route (the estimators'), unchanged by this lane.

Starting commit, leg 1 (`.../2026-09-09_132755-nvidia-h100-identical-gemm/remote/identical/`):

| shape | v1 identical ms (plan) | v1 / cublas-fp32 | core identical ms | core / cublas-fp32 | cublas-fp32 ms |
|---|---|---|---|---|---|
| gram.32x32x1M | 50.333 (SPLITK) | 209.7x | 0.494 | 2.1x | 0.240 |
| gram.32x32x64K | 0.944 (SPLITK) | 24.2x | 0.048 | 1.2x | 0.039 |
| gram.128sq.x100003 | 8.653 (TILE 16x16 KS=32) | 90.1x | 0.202 | 2.1x | 0.096 |
| ols.step1.16x16x64K | 0.185 (SPLITK) | 4.9x | 0.042 | 1.1x | 0.038 |
| pca.transform.8192x4x4 | 0.012 (TILE 32x8 KS=16) | 0.5x | 0.0094 | 0.4x | 0.024 |
| pca.transform.wide.8192x64x128 | 0.030 (TUNED 64x64 reg4x4) | 1.1x | 0.290 | 11.1x | 0.026 |
| kmeans.dist.4096x64x64 | 0.021 (TUNED 64x64 reg4x4) | 0.9x | 0.080 | 3.5x | 0.023 |
| ols.predict.gemv.64Kx16 | 0.031 (TILE 32x8 KS=16) | 1.5x | 0.012 | 0.6x | 0.020 |
| llama8b.qkv.t1 | 0.155 (SPLITK) | 3.7x | 0.578 | 13.8x | 0.042 |
| llama8b.qkv.t8 | 0.429 (TILE 8x32 KS=32) | 7.0x | 0.566 | 9.3x | 0.061 |
| llama8b.qkv.t512 | 1.629 (TUNED 128x128 reg8x8) | 4.4x | 34.226 | 91.5x | 0.374 |
| llama8b.mlp_up.t1 | 0.711 (TILE 8x32 KS=32) | 7.4x | 0.579 | 6.0x | 0.096 |
| llama8b.mlp_up.t8 | 0.729 (TILE 8x32 KS=32) | 4.6x | 2.198 | 13.7x | 0.160 |
| llama8b.mlp_up.t512 | 6.372 (TUNED 128x128 reg8x8) | 4.6x | 118.387 | 85.8x | 1.379 |
| llama8b.mlp_down.t1 | 0.495 (SPLITK) | 5.1x | 2.023 | 20.9x | 0.097 |
| llama8b.mlp_down.t8 | 1.520 (TILE 8x32 KS=32) | 7.7x | 1.967 | 9.9x | 0.198 |
| llama8b.mlp_down.t512 | 5.615 (TUNED 128x128 reg8x8) | 4.7x | 119.773 | 101.1x | 1.185 |
| llama8b.lm_head.t1 | 4.575 (TILE 8x32 KS=32) | 6.5x | 2.215 | 3.2x | 0.702 |
| llama8b.lm_head.t8 | 4.617 (TILE 8x32 KS=32) | 4.0x | 16.871 | 14.6x | 1.154 |
| llama8b.lm_head.t512 | 49.622 (TUNED 128x128 reg8x8) | 4.6x | skipped | - | 10.808 |

After this lane, leg 4 (`.../2026-09-09_140650-nvidia-h100-identical-splitk3/remote/identical/`).
Leg 5 (`.../2026-09-09_141549-nvidia-h100-identical-splitk4/remote/identical/`,
the final tree) differs only where the dispatcher now takes the 64x64
tuned plan at KS = 32: pca.transform.wide 0.030 ms (1.2x) and
kmeans.dist 0.021 (0.9x), every other row within noise of this table:

| shape | v1 identical ms (plan) | v1 / cublas-fp32 | core identical ms | core / cublas-fp32 | cublas-fp32 ms |
|---|---|---|---|---|---|
| gram.32x32x1M | 0.435 (SPLIT 32x32 reg2x2 KS=32) | 1.8x | 0.496 | 2.1x | 0.240 |
| gram.32x32x64K | 0.048 (SPLIT 32x32 reg2x2 KS=32) | 1.2x | 0.049 | 1.3x | 0.039 |
| gram.128sq.x100003 | 0.577 (SPLIT 64x64 reg4x4 KS=32) | 6.0x | 0.204 | 2.1x | 0.096 |
| ols.step1.16x16x64K | 0.031 (SPLIT 16x16 reg1x1 KS=32) | 0.8x | 0.042 | 1.1x | 0.038 |
| pca.transform.8192x4x4 | 0.013 (TILE 32x8 KS=16) | 0.5x | 0.010 | 0.4x | 0.024 |
| pca.transform.wide.8192x64x128 | 0.032 (TUNED 64x64 reg4x4 KS=16) | 1.2x | 0.292 | 11.2x | 0.026 |
| kmeans.dist.4096x64x64 | 0.022 (TUNED 64x64 reg4x4 KS=16) | 1.0x | 0.082 | 3.6x | 0.023 |
| ols.predict.gemv.64Kx16 | 0.031 (TILE 32x8 KS=16) | 1.6x | 0.013 | 0.6x | 0.020 |
| llama8b.qkv.t1 | 0.071 (SPLIT 1x256 reg1x1 KS=16) | 1.7x | 0.584 | 13.9x | 0.042 |
| llama8b.qkv.t8 | 0.125 (SPLIT 8x64 reg1x2 KS=16) | 2.1x | 0.572 | 9.4x | 0.061 |
| llama8b.qkv.t512 | 1.658 (TUNED 128x128 reg8x8 KS=16) | 4.4x | 34.485 | 92.2x | 0.374 |
| llama8b.mlp_up.t1 | 0.213 (SPLIT 1x256 reg1x1 KS=16) | 2.2x | 0.585 | 6.1x | 0.096 |
| llama8b.mlp_up.t8 | 0.378 (SPLIT 8x64 reg1x2 KS=16) | 2.4x | 2.219 | 13.9x | 0.160 |
| llama8b.mlp_up.t512 | 6.544 (TUNED 128x128 reg8x8 KS=16) | 4.7x | 119.286 | 86.5x | 1.379 |
| llama8b.mlp_down.t1 | 0.201 (SPLIT 1x256 reg1x1 KS=16) | 2.1x | 2.038 | 21.0x | 0.097 |
| llama8b.mlp_down.t8 | 0.393 (SPLIT 8x64 reg1x2 KS=16) | 2.0x | 1.985 | 10.0x | 0.198 |
| llama8b.mlp_down.t512 | 5.808 (TUNED 128x128 reg8x8 KS=16) | 4.9x | 120.658 | 101.8x | 1.185 |
| llama8b.lm_head.t1 | 1.535 (SPLIT 1x256 reg1x1 KS=16) | 2.2x | 2.236 | 3.2x | 0.702 |
| llama8b.lm_head.t8 | 3.186 (SPLIT 8x64 reg1x2 KS=16) | 2.8x | 16.995 | 14.7x | 1.154 |
| llama8b.lm_head.t512 | 51.232 (TUNED 128x128 reg8x8 KS=16) | 4.7x | skipped | - | 10.808 |

Read across: the SPLITK rows went from 4.9x-209.7x to 0.8x-1.8x
(gram.32x32x1M 50.3 -> 0.435 ms); the decode rows (m <= 8) from
3.7x-7.7x to 1.7x-2.8x, all inside the 3x target; the one TN row still
outside it is gram.128sq.x100003 at 6.0x (8.65 -> 0.577 ms); the t512
rows are unchanged at 4.4x-4.9x. The intermediate legs' tables print
from their directories with the table tool (`--plans` adds the forced
sweep; the probe halves `m`/`n` above 50 GMAC, so its lm_head.t512 row
is `m = 64`, not the speed row). Leg 2 (the first SPLIT plans, the
one-thread-per-cell fold) had the TN rows at 4.1x-12.1x because that
fold was latency-bound (0.19 ms at P = 512 whatever `m n`); leg 3's
block-per-cell fold and KS = 32 took gram.32x32x1M 0.974 -> 0.432 and
ols.step1 0.198 -> 0.030.

## Opponent rows (for bench/OPPONENT_REFERENCE.md; transcription, no re-run)

NVIDIA H100 80GB HBM3, driver 580.126.09, CUDA 12.4, cuBLAS through torch
2.4.1+cu124 (`torch.matmul`, `allow_tf32` False / True), medians of 5
rounds after 1 warm-up, from
`bench/results/e1g/2026-08-25_155542-nvidia-speed-gemmseq/remote/logs/gemm.gemm.cublas.log`.
The repeat leg `2026-08-25_160520` agrees within 0.002 ms on every row
except lm_head.t512 (10.820 / 1.387) and mlp_down.t512 tf32 (0.212).
These are the rows commit `5dd9c43b` put into the reference file.

| shape | cublas-fp32 | cublas-tf32 |
|---|---|---|
| gram.32x32x1M | 0.240 | 0.115 |
| gram.32x32x64K | 0.039 | 0.029 |
| gram.128sq.x100003 | 0.096 | 0.060 |
| ols.step1.16x16x64K | 0.038 | 0.028 |
| pca.transform.8192x4x4 | 0.024 | 0.019 |
| pca.transform.wide.8192x64x128 | 0.026 | 0.020 |
| kmeans.dist.4096x64x64 | 0.023 | 0.019 |
| ols.predict.gemv.64Kx16 | 0.020 | 0.019 |
| llama8b.qkv.t1 | 0.042 | 0.043 |
| llama8b.qkv.t8 | 0.061 | 0.048 |
| llama8b.qkv.t512 | 0.374 | 0.071 |
| llama8b.mlp_up.t1 | 0.096 | 0.097 |
| llama8b.mlp_up.t8 | 0.160 | 0.109 |
| llama8b.mlp_up.t512 | 1.379 | 0.187 |
| llama8b.mlp_down.t1 | 0.097 | 0.098 |
| llama8b.mlp_down.t8 | 0.198 | 0.110 |
| llama8b.mlp_down.t512 | 1.185 | 0.220 |
| llama8b.lm_head.t1 | 0.702 | 0.700 |
| llama8b.lm_head.t8 | 1.154 | 0.793 |
| llama8b.lm_head.t512 | 10.808 | 1.368 |

## Which plans are the default now, and which are not

Flipped (the dispatcher picks them, bits proven on every leg above):
the four split plans of `8ffebd44` and the 64x64 split plan of
`645bd42c` at the shapes listed under "What changed"; the 64x64 tuned
plan's K step of 32 on columns whose shared limit takes two 37 KB pages
(`540fcb14`; NVIDIA, AMD). Not flipped: the 128x128 tuned plan stays at
KS = 16 (KS = 32 measured 17% slower, one page); `PLAN_SPLITK` and
`PLAN_SPLITK_STAGED` remain as plans for the gates and are no longer
chosen at any shape with `P >= 4` that fits the workspace bound (a shape
with `P >= 4` that does NOT fit 256 MB of partials still falls to
SPLITK's old rule, which also required the fit, so in practice the
untuned fallback there is a TILE plan); `GEMM_IDENT_SWAP_537` untouched.

## L40S

No new L40S numbers: the pool had no instances at 13:43 EDT and the lane
iterated on the H100 instead. The previous lane's L40S table
(`docs/lanes/HANDOFF_gemm.md`, `2026-09-09_123601-nvidia-l40s-identical-gemm-merged`)
stands for the plans it measured; the SPLIT plans and the 64x64 KS = 32
flip have no L40S row. To get one: the leg command below with
`--gpu "NVIDIA L40S"` and `--reference l40s` on the table tool.

## RUN OWED on the Apple M4 (orchestrator)

    tools/with_identical_mode.sh pixi run mojo run -I . gemm/checks/gemm_device_check.mojo
    MOJOLEARN_GEMM_CARD_HOST_CAP=1 tools/gemm_card.sh device /tmp/apple.card
      then: grep -v '^#' /tmp/apple.card | cmp - <(grep -v '^#' bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card)
    tools/with_identical_mode.sh pixi run mojo run -I . gemm/checks/gemm_tuned_probe.mojo
    tools/with_identical_mode.sh pixi run check-gemm-identity
    tools/with_identical_mode.sh pixi run check-batch-invariance
    tools/with_identical_mode.sh pixi run mojo run -I . gemm/checks/gemm_backward_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . solver/checks/profile_dot.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . glm/ols_main.mojo
    MOJOLEARN_SPEED_GEMM_ARMS=v1 MOJOLEARN_SPEED_ROUNDS=5 tools/with_identical_mode.sh pixi run mojo run -I . bench/speed/gemm_speed_main.mojo

The Apple column runs the software seam; `TUNED_64_KS` resolves to 16
there (two pages of the 64x64 pair at 32 would be 37 KB) so the 64x64
tuned plan is the same kernel it was; the split plans' shared pages are
18 KB (32x32, KS 32), 37 KB (64x64, KS 32, ONE page on Apple), 9 KB
(16x16), 41 KB (1x256, KS 16, one page on Apple) and 12 KB (8x64). The
one-page arms of the 64x64 and 1x256 split plans are the two
instantiations no NVIDIA leg exercised (NVIDIA takes two pages there),
so the device check's launch-invariance gate on the M4 is the evidence
for them. The card must stay byte-equal; the probe must say 0 MOVED.

## Exact next commands for a fresh agent

One leg, gates plus timing, on either GPU (about 7 minutes of a lease on
the H100; the script terminates the pod and verifies it gone):

    MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_identical_leg_extra.sh \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-identical-gemm \
    sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
        --gpu "NVIDIA H100 80GB HBM3" \
        --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
    python3 tools/gemm_identical_table.py <leg>/remote/identical --reference h100 --plans
    grep -v '^#' <leg>/remote/identical/capped.card | cmp - <(grep -v '^#' bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card)
    tail -1 <leg>/remote/identical/probe.log      # must end "0 MOVED, 0 refused."
    grep 'all green' <leg>/remote/device_check.log

The leg exits 1 because its own step 9 diffs the UNCAPPED card against the
capped known-good card and reports a structural mismatch; the three lines
above are the verdict. Run the leg from a clean, committed tree (it ships
HEAD by `git archive`). `MOJOLEARN_GEMM_PROBE_PLANS="6 11 15"` narrows
the forced sweep; `MOJOLEARN_SPEED_ROUNDS` sets the rounds.

## Unfinished, and why

1. gram.128sq.x100003 (m = n = 128, k = 100003, OP_TN) is 6.0x, outside
   task 2's 3x. Its 64x64 split tile runs 4 tiles x 782 leaves at 2.8
   TMAC/s; the operand pair is loaded once per leaf-block and the block
   fold reads 782 partials per cell with stride `m n`. The next step is
   the dense kernel itself (register tile 4x4 at KS = 32 is what it
   already runs), or a 128x128 split tile (one tile x 782 leaves, 37 KB a
   page); neither was tried.
2. Task 4's 1.5x at the t512 rows was not reached (4.4x-4.9x). The 128x128
   tuned plan at KS = 32 is one page and lost 17%; the 64x64 plan at
   KS = 32 won 4% to 7% and flipped. Double buffering beyond two pages,
   or a 64x128 tile that keeps two pages at KS = 32, are the untried
   moves; the forced sweep (`--plans`) is the instrument.
3. No L40S row for the split plans or the flip (pool empty at 13:43 EDT);
   no AMD or Apple run of them (Apple is RUN OWED above; AMD is a
   DigitalOcean droplet, see the memory index).
4. `gemm_unpinned.mojo` mirrors `choose_gemm_plan_untuned` only, so the
   split plans have no unpinned counterpart (same standing as the tuned
   plans).
5. `ols.predict.gemv.64Kx16` (P = 1) and `pca.transform.8192x4x4` (P = 1)
   stay on their old plans by design (one leaf, nothing to split); they
   are 0.5x-1.6x already.
