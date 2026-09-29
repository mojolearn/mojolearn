# lowbit-mma-speed: the tuned integer unit kernel and the parallel quantizer

Lane D, branch `lane/lowbit-mma-speed`, worktree `~/mojolearn-wt/lowbit-mma-speed`,
forked from `lane/lowbit-units`. Brief `~/mojolearn-evidence/lowbit-units/brief.md`
(section "UPDATE 2026-09-29 ~03:30Z: Lane D" and every update after it). Lane files
`~/mojolearn-evidence/lowbit-mma-speed/`. Results in the repo:
`bench/results/lowbit_mma_speed/2026-09-29/`. Nothing here is dispatched and no
default moves: `identical_gemm_int8_into` still takes the reference plan.
`gemm/checks/gemm_int8_mma.mojo`, `gemm/checks/gemm_lowbit.mojo` and fp32.v1's
files are not edited by this lane.

Every number below is a time, a rate, or a time over fp32.v1's at the same row on
the same box in the same run. One run per table, five timed calls per arm, median.
Every timing run was built first, run once cold (times never read, digests kept and
compared with the run of record's), then run for the record, with nothing else of
the lane's on the box.

## What exists

| Piece | File |
|---|---|
| The tuned unit kernel: staged plans, the direct kernel and its probes, FOUR PRODUCTS ONE STAGING | `gemm/checks/gemm_int8_mma_tuned.mojo` |
| Its gates | `gemm/checks/gemm_int8_mma_tuned_check.mojo`, `gemm/checks/gemm_int8_pieces_tuned_check.mojo` |
| The parallel quantizer | `gemm/checks/quantize_int8_par.mojo` |
| Its gate | `gemm/checks/quantize_int8_par_check.mojo` |
| The arms in lane/lowbit-units' harness | `bench/gemm_lowbit_price_main.mojo` (`MOJOLEARN_LOWBIT_PRICE_ARMS` picks arms) |
| Job scripts, the PTX counter, the tables | `tools/lowbit_mma_speed/` |
| pixi tasks | `check-quantize-int8-par[-sabotage|-value-sabotage]`, `check-gemm-int8-mma-tuned[-sabotage|-value-sabotage|-unstated]`, `check-gemm-int8-pieces-tuned[-sabotage|-staging-sabotage|-value-sabotage]` |

ENTRY POINTS (all asynchronous, OP_NT):
- `identical_gemm_int8_mma_tuned_into(ctx, c, qa, ea, qb, eb, m, n, k)`: the
  reference's signature and its bits, on the plan `int8_tuned_dispatch` names.
- `identical_gemm_int8_pieces_tuned_into(ctx, s, ah, al, bh, bl, m, n, k)`: two int8
  planes of each operand in, `3 m n` Int32 out, cell-major: `s[3 (i n + j)]` HH,
  `+ 1` HL + LH, `+ 2` LL. No recombination and no float. `k <= 65536`, stated for
  LOW planes in [0, 127].
- `quantize_rows_int8_par_device(ctx, q, e, x, rows, cols)`:
  `quantize_rows_int8_device`'s signature and its codes.

## THE FINDING: what was NVIDIA-specific in the reference kernel

The reference's `_pack4` reads a fragment word with `unsafe_load[width=4]` on an
int8 pointer and states no alignment. On NVIDIA the compiler emits a
four-iteration LOOP of byte loads per word (PTX of the reference, job nvc3-0016
and nvc3-0020: `ld.global.b8`, `bfi.b32` and a branch; 8 words per k step; no
`ld.global.b32` of a code anywhere). The same load with `alignment=4` is one
`ld.global.b32`. Forced arm, the reference's schedule with only that changed
(`int8i32.v1.mma.direct.aligned-loads`), bits equal to the reference, the flat
plan and the oracle: qkv.t512 1.436 ms to 0.391 ms. The control arm (the
reference respelled in the new file) reads 1.436, the reference 1.433.
The orchestrator gave the edit of the reference file to lane/lowbit-int15.

The other suspects, each a forced arm at the training rows (run 4):
- the register pack re-spelled as a vector: nothing (0.391 to 0.393);
- the epilogue's seam: nothing (0.391 with it, 0.429 without, inside the noise of
  one run);
- the two m16n8 halves: one half alone 0.297 against 0.391 for two;
- the fragments loaded once and not per k step (a probe, wrong product on
  purpose): 0.036 ms. The unit steps are a tenth of the aligned kernel's time; the
  loads are the rest.

## Gate verdicts

| Gate | H100 (nvc3) | MI325X (do-amd) | M3 Ultra | M2 Pro |
|---|---|---|---|---|
| Parallel quantizer (`quantize_int8_par_check`): codes and exponents == host == reference device quantizer, both blocks, 36 + 4 cases | GREEN (nvc3-0015) | GREEN (1790653296353; again 1790656243553 at 464d5002a) | GREEN (1790653289781; again 1790656246400 at 464d5002a) | GREEN (1790653293863; again 1790656250587 at 464d5002a) |
| its sabotage arms SEEN failing: tree's first level skipped; value flip | 12 of 36 and 2 of 4 failed; 24 of 36 and 4 of 4 | the same counts | the same counts | the same counts |
| Tuned unit plans (`gemm_int8_mma_tuned_check`): tuned == reference unit plan == flat == oracle | GREEN: 486 quantized-fixture + 2016 planted cases, 13 staged plans and 3 direct plans (nvc3-0020, -0022, -0025, -0027) | GREEN: 378 + 1568 cases, 12 staged plans (1790656243553) | no integer unit | no integer unit |
| its sabotage arms SEEN failing: staging pad not written; value flip | 160 of 486 and 704 of 2016; all | planted 655 of 1568; all | | |
| the same gate with no alignment stated (byte path forced) | GREEN (nvc3-0025, -0027) | GREEN | | |
| FOUR PRODUCTS ONE STAGING (`gemm_int8_pieces_tuned_check`): every plan == reference device plan == host integers, k = 65536 planted, k = 65537 refused by name | GREEN: 200 + 235 cases, 4 plans (nvc3-0025); 7 plans (nvc3-0027) | GREEN: 200 + 235 cases, 4 plans | not run (no unit) | not run |
| its sabotage arms SEEN failing: middle sum takes HL twice; staging pad; value flip | 160 of 200 and 188 of 235; 72 and 24; all | seen, exit 1 each | | |

Digests across boxes, same extents (`bench/results/lowbit_mma_speed/2026-09-29/DIGESTS_ACROSS_BOXES.md`):
AGREE at every arm two boxes ran, 0 disagree: the quantizer's codes on four boxes,
12 of 12 rows; 12 tuned plans, 4 four-product plans and their reference device
plan, and the complete operations on the H100 and the MI325X, 12 of 12 rows each.
The harness's own digest arm (one bit of the last cell flipped) was seen at every
arm and shape of every timing run.

NOT GATED ANYWHERE: the two 32-warp plans on AMD (2048 threads a block there; not
run, which is not a pass). The wide 32-warp plan is launched nowhere (Failures 3).

## What each lever bought, H100

Plan names: `w` the warp's output tile, `b` the block's, `k` the k steps staged
per window, `l` the bytes of one staging load.

#### Training rows, run 4 (nvc3-0020): median ms / T MAC/s / over fp32.v1

| arm | qkv.t512 | mlp_up.t512 | mlp_down.t512 | lm_head.t512 |
|---|---:|---:|---:|---:|
| fp32.v1 (the shipped plan) | 1.0329 / 8.3 / 1.000 | 3.3427 / 9.0 / 1.000 | 3.3680 / 8.9 / 1.000 | 3.7312 / 9.0 / 1.000 |
| 0. the reference unit plan | 1.4326 / 6.0 / 1.387 | 4.6271 / 6.5 / 1.384 | 5.5446 / 5.4 / 1.646 | 5.1294 / 6.6 / 1.375 |
| control: the reference respelled in the new file | 1.4363 / 6.0 / 1.391 | 4.6268 / 6.5 / 1.384 | 5.5806 / 5.4 / 1.657 | 5.1196 / 6.6 / 1.372 |
| A. the fragment loads state their alignment (same schedule) | 0.3909 / 22.0 / 0.378 | 1.2510 / 24.0 / 0.374 | 1.3228 / 22.7 / 0.393 | 1.3284 / 25.3 / 0.356 |
| B. A, the register pack kept as scalars | 0.3932 / 21.8 / 0.381 | 1.2049 / 25.0 / 0.360 | 1.3206 / 22.8 / 0.392 | 1.3283 / 25.3 / 0.356 |
| 1. operands staged in shared memory (same tiles, 4-byte loads, 32 k steps) | 0.3811 / 22.5 / 0.369 | 1.0740 / 28.0 / 0.321 | 1.3956 / 21.5 / 0.414 | 1.2787 / 26.3 / 0.343 |
| 2. more fragments per warp: 32x32, block 64x64 | 0.3077 / 27.9 / 0.298 | 1.5394 / 19.5 / 0.461 | 1.2837 / 23.4 / 0.381 | 1.4248 / 23.6 / 0.382 |
| 2. more fragments per warp: 64x64, block 128x128 | 0.6809 / 12.6 / 0.659 | 1.5449 / 19.5 / 0.462 | 2.4927 / 12.1 / 0.740 | 1.5089 / 22.3 / 0.404 |
| 3. wider block: 128x256 (64x64 per warp, 8 warps) | 0.5619 / 15.3 / 0.544 | 1.3105 / 22.9 / 0.392 | 2.0251 / 14.8 / 0.601 | 1.2774 / 26.3 / 0.342 |
| 4. packed loads: 16 bytes (64x64 per warp, block 128x128) | 0.2642 / 32.5 / 0.256 | 0.6112 / 49.2 / 0.183 | 0.8977 / 33.5 / 0.267 | 0.5701 / 59.0 / 0.153 |
| 5. k loop: 64 steps a window | 0.2729 / 31.5 / 0.264 | 0.6451 / 46.6 / 0.193 | 1.0117 / 29.7 / 0.300 | 0.5887 / 57.1 / 0.158 |
| 5. k loop: 128 steps a window | 0.2597 / 33.1 / 0.251 | 0.6314 / 47.6 / 0.189 | 0.9568 / 31.4 / 0.284 | 0.5867 / 57.3 / 0.157 |
| 3+4+5. block 128x256, 16-byte loads, 64 steps | 0.2853 / 30.1 / 0.276 | 0.6192 / 48.6 / 0.185 | 0.9275 / 32.4 / 0.275 | 0.6034 / 55.7 / 0.162 |
| 1+4+5. 16x16 per warp, block 32x32 (the launcher's plan at m <= 16) | 0.1759 / 48.8 / 0.170 | 0.5278 / 57.0 / 0.158 | 0.6389 / 47.1 / 0.190 | 0.6296 / 53.4 / 0.169 |
| 2+4+5. 32x32 per warp, block 64x64 | 0.1291 / 66.5 / 0.125 | 0.6350 / 47.3 / 0.190 | 0.5225 / 57.5 / 0.155 | 0.5650 / 59.5 / 0.151 |
| 6. more warps per block: 16 warps of 32x32, block 128x128 (the launcher's plan at m > 16) | 0.1185 / 72.5 / 0.115 | 0.2941 / 102.2 / 0.088 | 0.4211 / 71.4 / 0.125 | 0.2935 / 114.5 / 0.079 |
| 6. more warps per block: 32 warps of 16x32, block 128x128 | 0.1271 / 67.6 / 0.123 | 0.5404 / 55.6 / 0.162 | 0.4559 / 65.9 / 0.135 | 0.5411 / 62.1 / 0.145 |
| the ROW plan written for the decode rows: 16x64 per warp, block 16x256 | 0.3973 / 21.6 / 0.385 | 1.4956 / 20.1 / 0.447 | 2.0552 / 14.6 / 0.610 | 1.7117 / 19.6 / 0.459 |
| PROBE (wrong product on purpose): fragments loaded once, the unit steps alone | 0.0363 / 236.8 / 0.035 | 0.0993 / 302.7 / 0.030 | 0.0916 / 328.4 / 0.027 | 0.1097 / 306.5 / 0.029 |
| PROBE: one m16n8 half per k step | 0.2965 / 29.0 / 0.287 | 0.9035 / 33.3 / 0.270 | 1.0368 / 29.0 / 0.308 | 1.0288 / 32.7 / 0.276 |
| PROBE: the epilogue without its seam | 0.4293 / 20.0 / 0.416 | 1.2132 / 24.8 / 0.363 | 1.3207 / 22.8 / 0.392 | 1.3708 / 24.5 / 0.367 |

#### Decode rows, run 4 (nvc3-0020): median ms / T MAC/s / over fp32.v1

| arm | qkv.t1 | qkv.t8 | mlp_up.t1 | mlp_up.t8 | mlp_down.t1 | mlp_down.t8 | lm_head.t1 | lm_head.t8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32.v1 (the shipped plan) | 0.0710 / 0.2 / 1.000 | 0.1048 / 1.3 / 1.000 | 0.1843 / 0.3 / 1.000 | 0.2763 / 1.7 / 1.000 | 0.1724 / 0.3 / 1.000 | 0.2923 / 1.6 / 1.000 | 1.2010 / 0.4 / 1.000 | 2.1551 / 2.0 / 1.000 |
| 0. the reference unit plan | 0.1582 / 0.1 / 2.228 | 0.1481 / 0.9 / 1.413 | 0.1919 / 0.3 / 1.041 | 0.2085 / 2.3 / 0.755 | 0.6522 / 0.1 / 3.783 | 0.6940 / 0.7 / 2.374 | 0.9557 / 0.5 / 0.796 | 1.3597 / 3.1 / 0.631 |
| control: the reference respelled in the new file | 0.1424 / 0.1 / 2.006 | 0.1476 / 0.9 / 1.408 | 0.1912 / 0.3 / 1.037 | 0.2069 / 2.3 / 0.749 | 0.6505 / 0.1 / 3.773 | 0.6990 / 0.7 / 2.391 | 0.9522 / 0.6 / 0.793 | 1.3856 / 3.0 / 0.643 |
| A. the fragment loads state their alignment (same schedule) | 0.0847 / 0.2 / 1.193 | 0.0833 / 1.6 / 0.795 | 0.1346 / 0.4 / 0.730 | 0.1342 / 3.5 / 0.486 | 0.4526 / 0.1 / 2.625 | 0.4437 / 1.1 / 1.518 | 0.3928 / 1.3 / 0.327 | 0.3983 / 10.6 / 0.185 |
| B. A, the register pack kept as scalars | 0.0650 / 0.3 / 0.915 | 0.0618 / 2.2 / 0.590 | 0.0900 / 0.7 / 0.488 | 0.0873 / 5.4 / 0.316 | 0.2878 / 0.2 / 1.669 | 0.2772 / 1.7 / 0.948 | 0.3064 / 1.7 / 0.255 | 0.3524 / 11.9 / 0.164 |
| 1. operands staged in shared memory (same tiles, 4-byte loads, 32 k steps) | 0.1838 / 0.1 / 2.589 | 0.1892 / 0.7 / 1.805 | 0.1657 / 0.4 / 0.899 | 0.1786 / 2.6 / 0.646 | 0.5859 / 0.1 / 3.398 | 0.6264 / 0.7 / 2.143 | 0.4928 / 1.1 / 0.410 | 0.5294 / 7.9 / 0.246 |
| 2. more fragments per warp: 32x32, block 64x64 | 0.2233 / 0.1 / 3.145 | 0.2280 / 0.6 / 2.176 | 0.2890 / 0.2 / 1.568 | 0.2967 / 1.6 / 1.074 | 1.0628 / 0.1 / 6.165 | 1.1110 / 0.4 / 3.801 | 1.0592 / 0.5 / 0.882 | 1.1130 / 3.8 / 0.516 |
| 2. more fragments per warp: 64x64, block 128x128 | 0.4146 / 0.0 / 5.839 | 0.4178 / 0.3 / 3.987 | 0.5690 / 0.1 / 3.087 | 0.5830 / 0.8 / 2.110 | 1.9526 / 0.0 / 11.326 | 2.0047 / 0.2 / 6.858 | 2.0755 / 0.3 / 1.728 | 2.1132 / 2.0 / 0.981 |
| 3. wider block: 128x256 (64x64 per warp, 8 warps) | 0.3722 / 0.0 / 5.242 | 0.3743 / 0.4 / 3.572 | 0.5449 / 0.1 / 2.957 | 0.5574 / 0.8 / 2.017 | 1.8547 / 0.0 / 10.758 | 1.8923 / 0.2 / 6.474 | 1.9640 / 0.3 / 1.635 | 1.9927 / 2.1 / 0.925 |
| 4. packed loads: 16 bytes (64x64 per warp, block 128x128) | 0.1548 / 0.1 / 2.180 | 0.1589 / 0.8 / 1.516 | 0.2069 / 0.3 / 1.123 | 0.2117 / 2.2 / 0.766 | 0.7534 / 0.1 / 4.370 | 0.8026 / 0.6 / 2.746 | 0.7786 / 0.7 / 0.648 | 0.8313 / 5.1 / 0.386 |
| 5. k loop: 64 steps a window | 0.1436 / 0.1 / 2.023 | 0.1439 / 0.9 / 1.373 | 0.2182 / 0.3 / 1.184 | 0.2226 / 2.1 / 0.806 | 0.7307 / 0.1 / 4.238 | 0.7593 / 0.6 / 2.598 | 0.8575 / 0.6 / 0.714 | 0.8758 / 4.8 / 0.406 |
| 5. k loop: 128 steps a window | 0.1296 / 0.1 / 1.825 | 0.1299 / 1.0 / 1.240 | 0.2054 / 0.3 / 1.114 | 0.2092 / 2.2 / 0.757 | 0.6678 / 0.1 / 3.874 | 0.6753 / 0.7 / 2.310 | 0.8019 / 0.7 / 0.668 | 0.8219 / 5.1 / 0.381 |
| 3+4+5. block 128x256, 16-byte loads, 64 steps | 0.1345 / 0.1 / 1.894 | 0.1357 / 1.0 / 1.295 | 0.2191 / 0.3 / 1.189 | 0.2200 / 2.1 / 0.796 | 0.6969 / 0.1 / 4.042 | 0.7166 / 0.7 / 2.452 | 0.8200 / 0.6 / 0.683 | 0.8371 / 5.0 / 0.388 |
| 1+4+5. 16x16 per warp, block 32x32 (the launcher's plan at m <= 16) | 0.0520 / 0.3 / 0.732 | 0.0537 / 2.5 / 0.512 | 0.0792 / 0.7 / 0.430 | 0.0857 / 5.5 / 0.310 | 0.2310 / 0.3 / 1.340 | 0.2555 / 1.8 / 0.874 | 0.2821 / 1.9 / 0.235 | 0.2962 / 14.2 / 0.137 |
| 2+4+5. 32x32 per warp, block 64x64 | 0.0769 / 0.2 / 1.083 | 0.0771 / 1.7 / 0.736 | 0.1225 / 0.5 / 0.665 | 0.1317 / 3.6 / 0.477 | 0.3883 / 0.2 / 2.252 | 0.4218 / 1.1 / 1.443 | 0.4596 / 1.1 / 0.383 | 0.5063 / 8.3 / 0.235 |
| 6. more warps per block: 16 warps of 32x32, block 128x128 (the launcher's plan at m > 16) | 0.0649 / 0.3 / 0.914 | 0.0651 / 2.1 / 0.621 | 0.0926 / 0.6 / 0.502 | 0.0997 / 4.7 / 0.361 | 0.2929 / 0.2 / 1.699 | 0.3024 / 1.6 / 1.035 | 0.3614 / 1.5 / 0.301 | 0.3776 / 11.1 / 0.175 |
| 6. more warps per block: 32 warps of 16x32, block 128x128 | 0.0646 / 0.3 / 0.910 | 0.0654 / 2.1 / 0.624 | 0.0904 / 0.6 / 0.491 | 0.0992 / 4.7 / 0.359 | 0.2873 / 0.2 / 1.666 | 0.3022 / 1.6 / 1.034 | 0.6396 / 0.8 / 0.533 | 0.6965 / 6.0 / 0.323 |
| the ROW plan written for the decode rows: 16x64 per warp, block 16x256 | 0.1711 / 0.1 / 2.410 | 0.1696 / 0.8 / 1.618 | 0.3257 / 0.2 / 1.767 | 0.3304 / 1.4 / 1.196 | 1.1218 / 0.1 / 6.507 | 1.1434 / 0.4 / 3.912 | 0.6554 / 0.8 / 0.546 | 0.6632 / 6.3 / 0.308 |
| PROBE (wrong product on purpose): fragments loaded once, the unit steps alone | 0.0147 / 1.1 / 0.207 | 0.0145 / 9.3 / 0.138 | 0.0155 / 3.8 / 0.084 | 0.0151 / 31.1 / 0.055 | 0.0227 / 2.6 / 0.132 | 0.0225 / 20.9 / 0.077 | 0.0355 / 14.8 / 0.030 | 0.0352 / 119.3 / 0.016 |
| PROBE: one m16n8 half per k step | 0.0566 / 0.3 / 0.797 | 0.0552 / 2.4 / 0.527 | 0.0753 / 0.8 / 0.409 | 0.0721 / 6.5 / 0.261 | 0.2470 / 0.2 / 1.433 | 0.2397 / 2.0 / 0.820 | 0.1760 / 3.0 / 0.147 | 0.2032 / 20.7 / 0.094 |
| PROBE: the epilogue without its seam | 0.0848 / 0.2 / 1.194 | 0.0813 / 1.7 / 0.776 | 0.1246 / 0.5 / 0.676 | 0.1276 / 3.7 / 0.462 | 0.4336 / 0.1 / 2.515 | 0.4294 / 1.1 / 1.469 | 0.3991 / 1.3 / 0.332 | 0.4062 / 10.3 / 0.188 |

#### The tall blocks, training rows, run 7 (nvc3-0027): median ms / T MAC/s / over fp32.v1

| arm | qkv.t512 | mlp_up.t512 | mlp_down.t512 | lm_head.t512 |
|---|---:|---:|---:|---:|
| fp32.v1 | 1.0321 / 8.3 / 1.000 | 3.3412 / 9.0 / 1.000 | 3.3859 / 8.9 / 1.000 | 3.7351 / 9.0 / 1.000 |
| block 128x128, 16 warps of 32x32 (the launcher's plan) | 0.0999 / 86.0 / 0.097 | 0.2928 / 102.7 / 0.088 | 0.4229 / 71.1 / 0.125 | 0.2953 / 113.9 / 0.079 |
| 7. tall block 256x64, 16 warps of 32x32 | 0.1146 / 75.0 / 0.111 | 0.6176 / 48.7 / 0.185 | 0.5178 / 58.1 / 0.153 | 0.6236 / 53.9 / 0.167 |
| 7. tall block 512x32, 16 warps of 32x32 | 0.1446 / 59.4 / 0.140 | 0.6725 / 44.7 / 0.201 | 0.7301 / 41.2 / 0.216 | 0.6796 / 49.5 / 0.182 |
| 7. tall block 512x16, 16 warps of 32x16 | 0.2407 / 35.7 / 0.233 | 0.9477 / 31.7 / 0.284 | 1.3041 / 23.1 / 0.385 | 1.0640 / 31.6 / 0.285 |

#### THE TARGET, training rows, run 6 (nvc3-0025): median ms / T MAC/s / over fp32.v1

| arm | qkv.t512 | mlp_up.t512 | mlp_down.t512 | lm_head.t512 |
|---|---:|---:|---:|---:|
| fp32.v1 | 1.0343 / 8.3 / 1.000 | 3.3386 / 9.0 / 1.000 | 3.3662 / 8.9 / 1.000 | 3.7345 / 9.0 / 1.000 |
| the reference unit plan, one product | 1.3540 / 6.3 / 1.309 | 4.6255 / 6.5 / 1.385 | 5.5520 / 5.4 / 1.649 | 5.1256 / 6.6 / 1.372 |
| one tuned product (block 128x128, 16 warps) | 0.1159 / 74.1 / 0.112 | 0.2930 / 102.6 / 0.088 | 0.4197 / 71.6 / 0.125 | 0.2975 / 113.0 / 0.080 |
| one tuned product (block 32x32, 4 warps) | 0.1812 / 47.4 / 0.175 | 0.5241 / 57.4 / 0.157 | 0.6407 / 46.9 / 0.190 | 0.6289 / 53.5 / 0.168 |
| parallel quantizer, activations | 0.0181 / 0.1 / 0.017 | 0.0168 / 0.1 / 0.005 | 0.0367 / 0.2 / 0.011 | 0.0182 / 0.1 / 0.005 |
| parallel quantizer, weights | 0.0598 / 0.3 / 0.058 | 0.1700 / 0.3 / 0.051 | 0.1804 / 0.3 / 0.054 | 0.1886 / 0.3 / 0.051 |
| quantize A + ONE tuned product | 0.1263 / 68.0 / 0.122 | 0.2999 / 100.3 / 0.090 | 0.4233 / 71.0 / 0.126 | 0.3018 / 111.4 / 0.081 |
| quantize A and B + ONE tuned product | 0.1824 / 47.1 / 0.176 | 0.4585 / 65.6 / 0.137 | 0.6054 / 49.7 / 0.180 | 0.4767 / 70.5 / 0.128 |
| quantize A + FOUR tuned products (four launches) | 0.4380 / 19.6 / 0.423 | 1.1551 / 26.0 / 0.346 | 1.6617 / 18.1 / 0.494 | 1.1563 / 29.1 / 0.310 |
| quantize A and B + FOUR tuned products (four launches) | 0.4954 / 17.3 / 0.479 | 1.3176 / 22.8 / 0.395 | 1.8426 / 16.3 / 0.547 | 1.3314 / 25.3 / 0.357 |
| four products, ONE staging: block 32x32, 4 warps of 16x16 | 0.5630 / 61.0 / 0.544 | 2.1874 / 55.0 / 0.655 | 1.9880 / 60.5 / 0.591 | 2.5523 / 52.7 / 0.683 |
| four products, ONE staging: block 64x128, 16 warps of 16x32 (the launcher's plan at m > 16) | 0.4344 / 79.1 / 0.420 | 1.5661 / 76.8 / 0.469 | 1.3561 / 88.7 / 0.403 | 1.7605 / 76.4 / 0.471 |
| four products, ONE staging: block 64x128, 8 warps of 32x32 | 0.5426 / 63.3 / 0.525 | 1.9240 / 62.5 / 0.576 | 1.7523 / 68.6 / 0.521 | 2.1257 / 63.3 / 0.569 |
| four products, ONE staging: block 64x64, 16 warps of 16x16 | 0.6860 / 50.1 / 0.663 | 2.5627 / 46.9 / 0.768 | 2.3535 / 51.1 / 0.699 | 2.8908 / 46.5 / 0.774 |
| quantize A + four products one staging + stand-in recombination | 0.4584 / 18.7 / 0.443 | 1.6238 / 18.5 / 0.486 | 1.3980 / 21.5 / 0.415 | 1.8189 / 18.5 / 0.487 |
| quantize A and B + four products one staging + stand-in recombination | 0.5125 / 16.8 / 0.496 | 1.7800 / 16.9 / 0.533 | 1.5760 / 19.1 / 0.468 | 1.9962 / 16.8 / 0.535 |

#### THE TARGET, decode rows, run 6 (nvc3-0025): median ms / T MAC/s / over fp32.v1

| arm | qkv.t1 | qkv.t8 | mlp_up.t1 | mlp_up.t8 | mlp_down.t1 | mlp_down.t8 | lm_head.t1 | lm_head.t8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32.v1 | 0.0697 / 0.2 / 1.000 | 0.1009 / 1.3 / 1.000 | 0.1781 / 0.3 / 1.000 | 0.2713 / 1.7 / 1.000 | 0.1689 / 0.3 / 1.000 | 0.2872 / 1.6 / 1.000 | 1.1967 / 0.4 / 1.000 | 2.1549 / 2.0 / 1.000 |
| the reference unit plan, one product | 0.1578 / 0.1 / 2.264 | 0.1474 / 0.9 / 1.461 | 0.1907 / 0.3 / 1.071 | 0.2079 / 2.3 / 0.766 | 0.6498 / 0.1 / 3.847 | 0.6903 / 0.7 / 2.404 | 0.9628 / 0.5 / 0.805 | 1.3640 / 3.1 / 0.633 |
| one tuned product (block 128x128, 16 warps) | 0.0707 / 0.2 / 1.014 | 0.0730 / 1.8 / 0.723 | 0.0928 / 0.6 / 0.521 | 0.0986 / 4.8 / 0.363 | 0.2904 / 0.2 / 1.719 | 0.3007 / 1.6 / 1.047 | 0.3601 / 1.5 / 0.301 | 0.3791 / 11.1 / 0.176 |
| one tuned product (block 32x32, 4 warps) | 0.0793 / 0.2 / 1.138 | 0.0844 / 1.6 / 0.836 | 0.0794 / 0.7 / 0.446 | 0.0841 / 5.6 / 0.310 | 0.2359 / 0.2 / 1.397 | 0.2589 / 1.8 / 0.901 | 0.2817 / 1.9 / 0.235 | 0.2995 / 14.0 / 0.139 |
| parallel quantizer, activations | 0.0139 / 0.0 / 0.199 | 0.0137 / 0.0 / 0.136 | 0.0140 / 0.0 / 0.079 | 0.0127 / 0.0 / 0.047 | 0.0204 / 0.0 / 0.121 | 0.0201 / 0.0 / 0.070 | 0.0133 / 0.0 / 0.011 | 0.0133 / 0.0 / 0.006 |
| parallel quantizer, weights | 0.0585 / 0.3 / 0.839 | 0.0580 / 0.3 / 0.575 | 0.1700 / 0.3 / 0.955 | 0.1703 / 0.3 / 0.628 | 0.1795 / 0.3 / 1.063 | 0.1794 / 0.3 / 0.625 | 1.3866 / 0.4 / 1.159 | 1.3850 / 0.4 / 0.643 |
| quantize A + ONE tuned product | 0.0578 / 0.3 / 0.829 | 0.0624 / 2.2 / 0.618 | 0.0813 / 0.7 / 0.456 | 0.0833 / 5.6 / 0.307 | 0.2406 / 0.2 / 1.425 | 0.2611 / 1.8 / 0.909 | 0.2845 / 1.8 / 0.238 | 0.2980 / 14.1 / 0.138 |
| quantize A and B + ONE tuned product | 0.1313 / 0.1 / 1.884 | 0.1350 / 1.0 / 1.338 | 0.2428 / 0.2 / 1.363 | 0.2440 / 1.9 / 0.899 | 0.4198 / 0.1 / 2.485 | 0.4397 / 1.1 / 1.531 | 1.6645 / 0.3 / 1.391 | 1.6784 / 2.5 / 0.779 |
| quantize A + FOUR tuned products (four launches) | 0.1909 / 0.1 / 2.739 | 0.1997 / 0.7 / 1.979 | 0.2954 / 0.2 / 1.659 | 0.3171 / 1.5 / 1.169 | 0.9176 / 0.1 / 5.433 | 1.0078 / 0.5 / 3.509 | 1.0981 / 0.5 / 0.918 | 1.1615 / 3.6 / 0.539 |
| quantize A and B + FOUR tuned products (four launches) | 0.2652 / 0.1 / 3.805 | 0.2732 / 0.5 / 2.708 | 0.4563 / 0.1 / 2.562 | 0.4786 / 1.0 / 1.764 | 1.0938 / 0.1 / 6.476 | 1.1855 / 0.4 / 4.128 | 2.4745 / 0.2 / 2.068 | 2.5406 / 1.7 / 1.179 |
| four products, ONE staging: block 32x32, 4 warps of 16x16 | 0.1088 / 0.6 / 1.561 | 0.1206 / 4.5 / 1.195 | 0.1410 / 1.7 / 0.792 | 0.1517 / 12.4 / 0.559 | 0.4333 / 0.5 / 2.565 | 0.4533 / 4.1 / 1.578 | 1.0703 / 2.0 / 0.894 | 1.1499 / 14.6 / 0.534 |
| four products, ONE staging: block 64x128, 16 warps of 16x32 (the launcher's plan at m > 16) | 0.1102 / 0.6 / 1.581 | 0.1188 / 4.5 / 1.177 | 0.1523 / 1.5 / 0.855 | 0.1639 / 11.5 / 0.604 | 0.4772 / 0.5 / 2.825 | 0.5200 / 3.6 / 1.811 | 1.1922 / 1.8 / 0.996 | 1.2685 / 13.3 / 0.589 |
| four products, ONE staging: block 64x128, 8 warps of 32x32 | 0.1465 / 0.5 / 2.102 | 0.1521 / 3.5 / 1.507 | 0.2241 / 1.0 / 1.258 | 0.2342 / 8.0 / 0.863 | 0.7398 / 0.3 / 4.380 | 0.8028 / 2.3 / 2.795 | 1.7541 / 1.2 / 1.466 | 1.8152 / 9.3 / 0.842 |
| four products, ONE staging: block 64x64, 16 warps of 16x16 | 0.0984 / 0.7 / 1.412 | 0.1076 / 5.0 / 1.066 | 0.2696 / 0.9 / 1.514 | 0.2869 / 6.5 / 1.058 | 0.4434 / 0.5 / 2.625 | 0.4795 / 3.9 / 1.670 | 2.1026 / 1.0 / 1.757 | 2.2332 / 7.5 / 1.036 |
| quantize A + four products one staging + stand-in recombination | 0.1013 / 0.2 / 1.453 | 0.1127 / 1.2 / 1.117 | 0.1483 / 0.4 / 0.833 | 0.1594 / 2.9 / 0.588 | 0.4497 / 0.1 / 2.663 | 0.4692 / 1.0 / 1.634 | 1.0708 / 0.5 / 0.895 | 1.1572 / 3.6 / 0.537 |
| quantize A and B + four products one staging + stand-in recombination | 0.1984 / 0.1 / 2.846 | 0.2069 / 0.6 / 2.051 | 0.3095 / 0.2 / 1.738 | 0.3233 / 1.5 / 1.192 | 0.6269 / 0.1 / 3.712 | 0.6454 / 0.7 / 2.247 | 2.4566 / 0.2 / 2.053 | 2.5371 / 1.7 / 1.177 |

Read plainly:
- THE ALIGNMENT OF THE LOADS is the largest single step: 1.433 to 0.391 ms at
  qkv.t512, the schedule unchanged.
- STAGING ALONE, the same tiles (lever 1): nothing, 0.391 to 0.381.
- MORE FRAGMENTS PER WARP with 4-byte staging loads: 32x32 bought time at two rows
  and cost it at two; 64x64 COST time at every row (0.681 at qkv.t512).
- WIDER BLOCK (128x256): less time than the 128x128 of 64x64 warps with 4-byte
  loads, the same with 16-byte loads.
- PACKED LOADS, 4 bytes to 16: 0.681 to 0.264 at the 64x64 warp tile.
- THE K LOOP, 32 to 64 to 128 steps a window: nothing either way at the 64x64 warp
  tile (0.264, 0.273, 0.260); with 16-byte loads and 64 steps the 16x16 and 32x32
  warp tiles read 0.176 and 0.129 where their 4-byte, 32-step forms read 0.381 and
  0.308 (the two levers were not separated at those tiles).
- MORE WARPS PER BLOCK: the 128x128 block cut into sixteen 32x32 warps is the plan
  that took the least time at all four training rows (0.10 to 0.12, 0.29, 0.42,
  0.29 ms; 72 to 115 T MAC/s). Cut into four 64x64 warps the same block reads
  0.27, 0.65, 1.01, 0.59. Cut into thirty-two 16x32 warps it reads 0.13, 0.54,
  0.46, 0.54.
- TALL BLOCKS (the weights read fewer times over): COST time at every row. The
  reading that the wide rows' time is the weights' traffic is REFUTED by this arm.
  It is confounded: the tall plans hold 116 registers a thread and one block per
  multiprocessor where the launcher's plan holds 63 and two (the PTX counter's
  `blocks_per_multiprocessor`, `h100/run7/ptx/probe.txt`).
- WHAT IS NOT ESTABLISHED: why the sixteen-warp 128x128 plan stands apart at the
  wide rows. It is the only plan with two 512-thread blocks resident per
  multiprocessor. No arm has yet held the tile fixed and changed only that.
- The unit steps alone (the probe) run at 237 to 328 T MAC/s; the best plan at 72
  to 115. The loads and the barriers are still most of the time.

## The quantizer, before and after

`parallel over reference` is a time over a time on the same box, same row, same
run. The digests of the two are equal at every row on every box.

| box | row | quantize activations, reference ms | parallel ms | parallel over reference | pack weights, reference ms | parallel ms | parallel over reference |
|---|---|---:|---:|---:|---:|---:|---:|
| H100 | qkv.t1 (1x4096, 4096x4096) | 0.4898 | 0.0136 | 0.0278 | 2.6680 | 0.0590 | 0.0221 |
| H100 | qkv.t512 (512x4096, 4096x4096) | 2.6269 | 0.0171 | 0.0065 | 2.4717 | 0.0610 | 0.0247 |
| H100 | mlp_up.t512 (512x4096, 14336x4096) | 2.6441 | 0.0162 | 0.0061 | 2.4763 | 0.1692 | 0.0683 |
| H100 | mlp_down.t512 (512x14336, 4096x14336) | 9.9635 | 0.0344 | 0.0035 | 9.9535 | 0.1803 | 0.0181 |
| H100 | lm_head.t8 (8x4096, 128256x4096) | 0.6078 | 0.0122 | 0.0201 | 12.1206 | 1.3759 | 0.1135 |
| H100 | lm_head.t512 (512x4096, 16032x4096) | 2.5223 | 0.0161 | 0.0064 | 2.6265 | 0.1868 | 0.0711 |
| MI325X | qkv.t1 (1x4096, 4096x4096) | 1.3449 | 0.0211 | 0.0157 | 3.8354 | 0.0415 | 0.0108 |
| MI325X | qkv.t512 (512x4096, 4096x4096) | 3.6619 | 0.0211 | 0.0058 | 3.7031 | 0.0455 | 0.0123 |
| MI325X | mlp_up.t512 (512x4096, 14336x4096) | 3.8893 | 0.0200 | 0.0051 | 4.1743 | 0.1132 | 0.0271 |
| MI325X | mlp_down.t512 (512x14336, 4096x14336) | 13.3351 | 0.0354 | 0.0027 | 13.6360 | 0.1229 | 0.0090 |
| MI325X | lm_head.t8 (8x4096, 128256x4096) | 1.4441 | 0.0210 | 0.0145 | 16.0919 | 0.8534 | 0.0530 |
| MI325X | lm_head.t512 (512x4096, 16032x4096) | 3.7850 | 0.0234 | 0.0062 | 4.3455 | 0.1220 | 0.0281 |
| M3 Ultra | qkv.t1 (1x4096, 4096x4096) | 1.2150 | 0.2340 | 0.1926 | 1.6220 | 0.3480 | 0.2145 |
| M3 Ultra | qkv.t512 (512x4096, 4096x4096) | 1.4690 | 0.2740 | 0.1865 | 1.5430 | 0.3520 | 0.2281 |
| M3 Ultra | mlp_up.t512 (512x4096, 14336x4096) | 1.4670 | 0.2790 | 0.1902 | 1.6290 | 0.7400 | 0.4543 |
| M3 Ultra | mlp_down.t512 (512x14336, 4096x14336) | 4.7610 | 0.3300 | 0.0693 | 5.1000 | 0.7170 | 0.1406 |
| M3 Ultra | lm_head.t8 (8x4096, 128256x4096) | 1.2550 | 0.2660 | 0.2120 | 10.2990 | 4.1590 | 0.4038 |
| M3 Ultra | lm_head.t512 (512x4096, 16032x4096) | 1.4750 | 0.2860 | 0.1939 | 1.8720 | 0.8020 | 0.4284 |
| M2 Pro | qkv.t1 (1x4096, 4096x4096) | 1.8350 | 0.1670 | 0.0910 | 3.8320 | 0.6700 | 0.1748 |
| M2 Pro | qkv.t512 (512x4096, 4096x4096) | 3.4900 | 0.2910 | 0.0834 | 3.8320 | 0.7030 | 0.1835 |
| M2 Pro | mlp_up.t512 (512x4096, 14336x4096) | 3.4910 | 0.3110 | 0.0891 | 13.7560 | 1.9550 | 0.1421 |
| M2 Pro | mlp_down.t512 (512x14336, 4096x14336) | 12.0070 | 0.4730 | 0.0394 | 13.8850 | 2.2180 | 0.1597 |
| M2 Pro | lm_head.t8 (8x4096, 128256x4096) | 1.3930 | 0.2230 | 0.1601 | 96.2280 | 15.7350 | 0.1635 |
| M2 Pro | lm_head.t512 (512x4096, 16032x4096) | 3.4880 | 0.3070 | 0.0880 | 13.8730 | 2.1640 | 0.1560 |

On the Macs the parallel quantizer's time is about one launch and wait (0.17 to
0.33 ms at every row but the weights of the head); nothing smaller can be read
from one call and one wait per sample there.

With the Apple exact-chunk probe (lane/lowbit-units', not retuned here) the
complete operation [quantize activations + probe] over fp32.v1 at the training
rows moved, M3 Ultra: 0.99, 0.70, 1.01, 0.68 with the reference quantizer to
0.66, 0.58, 0.63, 0.59 with the parallel one; M2 Pro: 0.67, 0.51, 0.64, 0.51 to
0.49, 0.46, 0.46, 0.46.

## THE TARGET, said plainly

Four products of the tuned kernel plus the conversions against ONE fp32.v1 product,
H100, measured as one operation with one wait (runs 5, 6 and 7 agree within 4%
but for qkv's four-launch arm, 0.38 to 0.42):

| operation | qkv.t512 | mlp_up.t512 | mlp_down.t512 | lm_head.t512 (n capped 16032) |
|---|---:|---:|---:|---:|
| fp32.v1, ms | 1.034 | 3.339 | 3.366 | 3.735 |
| parallel quantize A + FOUR tuned products, four launches: ms (over fp32.v1) | 0.438 (0.423) | 1.155 (0.346) | 1.662 (0.494) | 1.156 (0.310) |
| the same with the weights quantized per call | 0.495 (0.479) | 1.318 (0.395) | 1.843 (0.547) | 1.331 (0.357) |
| parallel quantize A + four products ONE STAGING + a stand-in recombination | 0.458 (0.443) | 1.624 (0.486) | 1.398 (0.415) | 1.819 (0.487) |
| the same with the weights quantized per call | 0.513 (0.496) | 1.780 (0.533) | 1.576 (0.468) | 1.996 (0.535) |

AT THE FOUR TRAINING ROWS THE TARGET IS MET ON THE H100: every form is under one
fp32.v1 product, between 0.31 and 0.55 of its time. After each lever, four
products alone (four times one product's median) over fp32.v1 at qkv.t512 read:
reference 5.55; loads aligned 1.51; staged 1.48; 16-byte loads and 64 steps, 16x16
warps 0.68; 32x32 warps 0.50; sixteen warps a block 0.46 (0.39 to 0.45 in the
later runs).

WHAT IS NOT IN THOSE NUMBERS. The fifteen-bit profile's own quantizer, split,
recombination and pinned seam are lane/lowbit-int15's. The quantizer here is the
int8 one (the same values read, one byte written where the fifteen-bit one writes
two). The recombination in the one-staging rows is a stand-in of the same work
with the backend's conversion, not the profile's seam. Four separate tuned
products cannot serve the profile as they stand: that kernel stores dequantized
float32, not Int32 sums, so the four-launch rows measure the count and the
one-staging rows are what the profile can call. lane/lowbit-int15 measured the
complete fifteen-bit call on the one-staging kernel at 0.44 to 0.56 of fp32.v1 at
the t512 rows on the H100 (its progress file has the run).

ONE STAGING AGAINST FOUR LAUNCHES (run 6, product alone, ms): 0.434 against 0.464
at qkv, 1.356 against 1.679 at mlp_down: less time. 1.566 against 1.172 at mlp_up,
1.761 against 1.190 at lm_head: MORE time. As written, one staging is not a gain
at the two wide rows on the H100. Its rate is flat, 76 to 82 T MAC/s over `4 m n k`,
where the one-product plan rises with the row's width; every four-product plan but
the smallest holds one block per multiprocessor (95 to 202 registers a thread).

THE DECODE ROWS ARE NOT MET. One tuned product is under fp32.v1 at six of eight
decode rows (0.14 to 0.90) and over it at qkv.t1 (1.14) and mlp_down.t1 (1.40).
Four products with the quantizer are over fp32.v1 at every decode row but the
head's two (0.54 and 0.92): 1.17 to 5.4. qkv.t1's numbers are launch and wait and
are UNDERPOWERED (0.058 ms for quantizer plus product in one wait against 0.079 for
the product alone in the same run); no conclusion is drawn from them.

MI325X, one run (request 1790656243553), the launcher's plans as chosen on the
H100: four launches plus the quantizer 0.41, 0.35, 0.51, 0.37 of fp32.v1; one
staging plus quantizer plus stand-in recombination 0.19, 0.25, 0.23, 0.30. One
tuned product's best plan per row 0.086, 0.050, 0.086, 0.075, and it is NOT the
H100's plan. From here AMD is lane/lowbit-amd-tuned's.

## What ran

| Box | Job | Commit | What | Verdict |
|---|---|---|---|---|
| H100 | nvc3-0011 | f301b3977 | lane/lowbit-units' gate and timing in this lane's tree (baseline) | GREEN, exit 0 |
| H100 | nvc3-0015 | 3c02e3529 | quantizer gate, quantizer timing | GREEN, exit 0 |
| H100 | nvc3-0016 | b645f0ef5+ | unit gate, PTX counter, unit timing (run 1) | gate GREEN, counter exit 0, timing RED (Failures 1) |
| H100 | nvc3-0018 | 720e7252c | the same, five more plans (run 2) | RED (Failures 2) |
| H100 | nvc3-0019 | ce9e400c6 | the same (run 3) | RED (Failures 3) |
| H100 | nvc3-0020 | 83c16987c | the same (run 4, THE LEVER TABLE) | GREEN, exit 0 |
| H100 | nvc3-0022 | e2ebca51e | unit gate, the target's timing (run 5) | GREEN, exit 0 |
| H100 | nvc3-0025 | 520406a38 | four-product gate, unit gate, the target's timing (run 6) | GREEN, exit 0 |
| H100 | nvc3-0027 | 229198668 | both gates, PTX counter, tall blocks at the training rows (run 7) | GREEN, exit 0 |
| MI325X | 1790653296353 | 3c02e3529 | quantizer gate and timing | PASS |
| MI325X | 1790656243553 | 464d5002a | unit gate, four-product gate, quantizer gate, every arm timed | PASS |
| M3 Ultra | 1790653289781, 1790656246400 | 3c02e3529, 464d5002a | quantizer gate and timing | PASS, PASS |
| M2 Pro | 1790653293863 | 3c02e3529 | quantizer gate and timing | PASS |
| M2 Pro | 1790656250587 | 464d5002a | quantizer gate and timing | gate GREEN, timing RED (Failures 5) |

The H100 tree is patch-synced: its `tree_head` is the merge base and the lane's
commit lies over it as a patch.

## Failures, each with its cause

1. nvc3-0016, timing RED at the ninth row. The probe `raw-store` stored the raw
   Int32 sum as a float; one cell of mlp_down.t512 was -987654, the harness's
   poison value, and the harness refused it as unwritten. Fixed: the probe adds a
   half, so no cell it stores is an integer. The nine rows that ran are kept as
   `h100/price_unit_run1_red`.
2. nvc3-0018, gate and timing RED. The H100 refused every launch of a 1024-thread
   plan (32x32 per warp, 4x8 warps) with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, and
   the gate ended at the first refusal without naming the plan. Fixed: the gate
   catches a refused launch, names the plan and goes on.
3. nvc3-0019, gate and timing RED. The plan put in its place (16x32 per warp, 4x8
   warps, 1024 threads) was refused as well. The PTX counter of nvc3-0020 read 85
   and 108 registers a thread for the two, above the 64 that 1024 threads leave of
   one multiprocessor's 65536. The plan is launched nowhere. After the staging
   helper changed (the alignment flag) the counter reads 43 and 62 for the same two
   instantiations; they were NOT retried, so whether they launch now is not known.
4. The reading that the wide rows' time is the weights' traffic was wrong: the
   tall blocks took more time at every row (run 7).
5. M2 Pro, request 1790656250587: the cold run exited 1, POISON SURVIVED at cell
   4352 of mlp_down.t512 in the arm [parallel quantizer + dispatched product]. The
   cell is the first of block 17 of the FLAT int8 kernel of `gemm_lowbit.mojo`
   (not this lane's). The run of record a minute later wrote every cell. The brief
   names this failure as known and open on that box; lane/lowbit-int15 is
   reproducing it. `m2pro/run2_quant_red/CAUSE.txt`.
6. One staging costs time at the two wide rows on the H100 (above). Not fixed.
7. The ROW plan written for the decode rows took two to four times the small
   plan's time there; the launcher does not take it.

## Owed

- The decode rows: the complete call is three launches against fp32.v1's one. A
  quantizer fused into the product's launch is the lever named by the orchestrator;
  not written.
- The four-product kernel at the wide rows: fewer registers a thread, or the next
  window's loads issued while the unit runs (two pages; `cp.async` on NVIDIA). Not
  written. A forced arm that holds the tile and changes only the blocks resident
  per multiprocessor has not run.
- The epilogue of the fifteen-bit profile inside the sums kernel's last step
  (lane/lowbit-int15's weak product, the weight gradient, 0.88 to 0.98): the
  interface is to be agreed through the orchestrator; nothing written.
- `ldmatrix` for the fragment loads on NVIDIA: not tried.
- Shapes between 17 and 511 rows: the launcher's choice there is not measured.
- More than one run per box before any ratio near 1 is read.
- The two refused 1024-thread plans, retried at their new register counts.
