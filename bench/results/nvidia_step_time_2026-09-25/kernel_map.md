# Kernel map of one T3 optimizer step on sm_90a (H100)

Shape: batch 4, length 2048 (M = 8,192 token rows a shard), d_model 768, 12
heads of 64, FF 2,048, 12 layers, vocabulary 50,257, K = 64 shards a step.
The phase structure is the one in
`/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/amd_step_time_2026-09-24/kernel_map.md`
(same source, same launches); this file records what differs on the NVIDIA
column and what the H100 measured. Per shard unless marked.

## What the NVIDIA column compiles differently (kernel-matrix rows)

| row | NVIDIA | AMD | what it selects |
|---|---|---|---|
| `lib_hardware_ftz_fma_for` | True | False | the step seam `fma.rn` + `mul.rn.ftz` by one (2 instructions a product); AMD `v_fmac` + class flush |
| `lib_gemm_stage_ftz_for` | True | True | operands flushed once at staging |
| `lib_gemm_kernel_body_for` | 1 (`kpack_hg`) | 1 | the packed body, gather staging, hardware fold flush (compiles out on AMD) |
| `lib_gemm_block_parallelism_for` | 132 | 110 | ksplit group rule S |
| `lib_gemm_leaf_split_for`, `lib_gemm_mfma_for`, `lib_postround_class_flush_for` | False | True | AMD-only GEMM paths (the AMD lane) |
| `GEMM_LAUNCH_BOUND` (shared) | 256 | 256 | `.maxntid 256`; on NVIDIA it changes nothing (255 registers with or without it) |
| `lib_gemm_window_admit_for` (this lane) | True | False | admitted windows run bare `fma.rn` |
| `lib_gemm_kpack_narrow_for` (this lane) | True | False | 128x64 kpack tile, 512 bound (128 registers, 2 blocks an SM) |
| `attn_default_arm_for` | `stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32_bswz` | same word | attention kernels `fwd_r2`, `zdot_estash_dres_pf`, `dq_tiled_pf`, `kvgrid_dkdv_pf` |
| `attn_fwd_kfull_for` (this lane, leg 4) | True | False | forward pass 0 stages a whole key block, 16-byte q/k loads |

## Kernels, measured (leg 1, origin/main, one lean B4 step, nsys)

| kernel (nsys id) | launches a shard | ms a shard | registers | shared bytes | grid x block |
|---|---|---|---|---|---|
| `identical_gemm_kpack_kernel` all leaves (2f61) | 171 | 300.6 | 255 | 33,792 | 384 x 256 (head: 25,152 tiles) |
| `identical_gemm_kpack_kernel` group (1c3b) | 84 | 92.9 | 255 | 33,792 | 96 x 8 x 256 |
| ksplit group folds (9e70, eb26) | 85 | 10.4 | 38, 46 | 0 | |
| attention forward `fwd_r2` (ea5e) | 12 | 68.3 | 96 | 15,360 | 3,072 x 256 |
| attention `zdot_estash_dres_pf` (02d8) | 12 | 33.9 | 63 | 10,240 | 6,144 x 256 |
| attention `dq_tiled_pf` (fb2c) | 12 | 29.8 | 128 | 8,192 | 1,536 x 256 |
| attention `kvgrid_dkdv_pf` (964c) | 12 | 25.2 | 63 | 12,288 | 3,072 x 256 |
| cross entropy (6 kernels) | 1 each | 14.1 | | | |
| nonfinite scans, norms, rope, residuals, silu, optimizer, embedding | | about 45 | | | |
| embedding run starts (`emb_run_begin_kernel`, 1 thread) | 1 | 2.05 | 20 | 0 | 1 x 1 |

Sum of kernel time about 606 ms a shard against a 611 ms lean step: the H100
step is kernel-bound; host gaps and launches are not where the time is.

GEMM resources (`ptxas -v`, CUDA 12.9, sm_90a, leg 1 `ptx/branch/`): 255
registers, a 4,096-byte local stack (the fold stack, 16 levels x 64 cells),
0 spill bytes (4 in the group kernel). With 255 registers only one 256-thread
block fits an SM (8 warps). With `-D MOJOLEARN_GEMM_NO_LAUNCH_BOUND=1` (1024)
the kernels fall to 32 to 64 registers and spill 2.6 to 7 KB: that define no
longer means "0.8.17's kernel" on NVIDIA.

## Where a GEMM window's time goes (DIAG subtraction at T3, leg 1 and 2)

Sum of the twelve calls weighted by their per-shard counts, main's kernel:
400 ms. Remove the flush multiply: 296 (-26 percent). Remove the staging
(stores, barrier, prefetch): 271 (-32). Remove the fold push: 342 (-15).
Remove the per-step shared loads: 391 (-2). Remove all four: 122 (the FMA
chain alone, -69). Finer (leg 2): the barrier alone -5 percent; the prefetch
alone or the staging stores alone -30 percent each (either removes the global
gather), so the exposed cost is the gather of the next window at one block
an SM.
