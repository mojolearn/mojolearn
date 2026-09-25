# NVIDIA step time at the GPT-3 Small shape, 2026-09-25

Branch `lane/nvidia-step-time` (from origin/main f2293183c, Release 0.8.18).
The problem: one optimizer step of the T3 run (162,147,840 parameters, batch
4, length 2048, vocabulary 50,257, K = 64 shards of 8,192 tokens) took 39.9 s
on one H100 (T1, 0.8.17). The lane's target: whatever identity allows.

Measured on RunPod pods with one NVIDIA H100 80GB HBM3 (driver 580.126.09,
CUDA 13.0 host, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`,
208 to 224 cores), four legs through `tools/gemm_remote_leg.sh nvidia --rent
--segment-lease 90 --dollar-cap 7` (dead-man armed before the create, every
pod verified deleted), every binding built from this branch's source on the
box with `MOJOLEARN_GPU_ARCHS=sm_90a`, `MOJOLEARN_TARGET_COLUMN=nvidia`,
IDENTICAL. The runner's own gates ran first on every leg
(`gemm_device_check`: 8 gates green).

## Result

| binding | what changed | s per optimizer step (steady) | lean B4 step | chain replay against the H100 chain |
|---|---|---|---|---|
| 0.8.17 GEMM source (leg 1) | the before | 38.83 | 0.611 s | PASS, 101..102 from ckpt 100 |
| origin/main (leg 1) | + the AMD lane's GEMM launch bound (256) | 38.83 (101..103), 38.79 (1999..2000) | 0.611 s | PASS, both |
| admission (leg 2) | + GEMM window admission | 33.80, 34.04 | 0.532 s | PASS, both |
| narrow (leg 3) | + 128x64 kpack tile under a 512 bound | 32.40, 32.52 | 0.508 s | PASS, both |
| **head (leg 4)** | + attention forward/dq launch bounds, one-block embedding scan | **30.67** (101..103), **30.77** (1999..2000) | **0.481 s** | PASS, both |

Same pods for the rows of a leg; `tools/lm_segment.py run --no-checkpoints
--expect-chain`, the published T3 checkpoints (`ckpt.sha256` in each leg:
80cd2126... and b8d98090...), the pinned FineWeb-Edu stream, recipe sha256
9f7f695b.... The first step of each replay carries setup (38.8 to 48.3 s).
Host hashing adds 6.5 to 7.4 s a step, outside the step seconds (it is the
same on every binding). The before matches T1's 39.9 s within the box.

Every replayed step's line equals the H100 run's (A/1 and A/2 chains): state
digest, gradient digest, the 64 shard losses and the learning-rate bits. State
digests at 101, 102, 103: abc8b816b5c3fb15, a9421f91b947f82c,
fcdb48b8ab51f2ef; at 1999 and 2000: dcb05e4e668a81e1, 0e39ed2bfe9bcbae, on
every binding of every leg. Every lean step on every binding wrote the same
witnesses (parameters 5516ffe5f550 / 77477af42588 / 4e439a8a9751, loss
676298dabb30 / afc46227a372 / 34fe4c49b0dd).

One H100 now takes 30.7 s a step against 38.8 s before, the same bits.

## Per kernel, before and after (nsys, one lean B4 step = one shard, ms)

`legs/leg1/nsys-branch/` (origin/main) and `legs/leg4/nsys-head/` (head).

| kernel | launches | before | after | why |
|---|---|---|---|---|
| `identical_gemm_kpack_kernel`, all leaves | 171 | 300.6 | 225.8 | admission (-20 percent), narrow tile (-7) |
| `identical_gemm_kpack_kernel`, group (long k) | 84 | 92.9 | 68.0 | the same |
| ksplit group folds | 85 | 10.4 | 10.4 | |
| attention forward `fwd_r2` | 12 | 68.3 | 42.8 | launch bound 1024: 96 -> 64 registers, 2 -> 4 blocks an SM |
| attention `zdot_estash_dres_pf` | 12 | 33.9 | 33.9 | not changed |
| attention `dq_tiled_pf` | 12 | 29.8 | 29.1 | launch bound 768: 128 -> 80 registers |
| attention `kvgrid_dkdv_pf` | 12 | 25.2 | 25.2 | not changed |
| embedding run starts | 1 | 2.05 | about 0.1 | one block instead of one thread |
| everything else (cross entropy, scans, norms, rope, residuals, optimizer) | | 41.8 | 41.6 | not changed |
| **sum of kernel time** | | **604.9** | **476.9** | |

The timed itemization (`item-timers.summary.txt`, phase timers on) agrees:
GEMM 401.7 -> 302.1 ms a shard, attention forward 68.5 -> 43.0.

## The changes (all NVIDIA rows; AMD and Apple compile what they compiled)

1. **GEMM window admission** (`lib_gemm_window_admit_for`,
   `checks/kernel_matrix.mojo`; `GEMM_WINDOW_ADMIT` in
   `gemm/checks/gemm_identical.mojo`). The contract step is
   `ftz(fma_rn(a, b, acc))`; NVIDIA spells it `fma.rn` then `mul.rn.ftz` by
   one, two instructions a product. While a 16-deep window is staged, each
   warp reduces the minimum biased exponent field of the NONZERO flushed A
   and B words it staged (Inf and NaN constrain nothing); after the staging
   barrier every thread reads the block's minima `Ea`, `Eb`. Every product the
   window forms is then a multiple of 2^(Ea + Eb - 300); if `Ea + Eb >= 174`
   that grain is at least 2^-126, and if every accumulator entering the
   window is a multiple of 2^-126 (true at a leaf start, +0.0, and after
   every admitted window), every exact step result and every rounded one is
   a multiple of 2^-126, so none is subnormal and the flush is the identity.
   Such a window runs `fma.rn` alone: the same instruction on the same
   operands in the same order, one rounding each. A window that fails, and
   every later window of its leaf, runs the two-instruction step. The
   decision is block-uniform. Revert `-D MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1`;
   sabotage `-D MOJOLEARN_GEMM_SABOTAGE_ADMIT_ALWAYS=1`.
2. **Narrow kpack tile** (`lib_gemm_kpack_narrow_for`): the kpack kernel's
   output tile 128x64 (a thread's cells 8x4 instead of 8x8) under a declared
   bound of 512 on its 256-thread launch (`GEMM_KPACK_LAUNCH_BOUND`): 128
   registers, two blocks an SM. A thread's cells, their chains, leaves and
   fold tree are unchanged. Revert `-D MOJOLEARN_GEMM_NO_KPACK_NARROW=1`.
3. **Attention launch bounds** (`attn_fwd_launch_bound_for` 1024,
   `attn_dq_launch_bound_for` 768 on NVIDIA; `ATTN_FWD_LAUNCH_BOUND`,
   `ATTN_DQ_LAUNCH_BOUND` in `transformer/impl/llama/fused_attention.mojo`).
   Register allocation only. Other columns declare 1024, the bound their
   backends assume without a declaration. Revert
   `-D MOJOLEARN_ATTN_NO_LAUNCH_BOUND=1`.
4. **Embedding run starts in one block** (`emb_run_begin_block_kernel`,
   `embedding/checks/embedding_identical.mojo`, every column): the exclusive
   integer prefix sum of the per-id counts by 256 threads instead of 1.
   Integer addition is exact, so every word is the serial kernel's. Revert
   `-D MOJOLEARN_EMB_SERIAL_RUN_BEGIN=1`.

## The bit proofs (head, leg 4; the same on leg 3 for changes 1 and 2)

1. GEMM A/B at the T3 shapes (`bench/gemm_excp_ab_main.mojo`, the twelve
   call kinds through the shipped entry points, three operand kinds: ordinary,
   one that drives subnormal step results, and a mixed one): the head's 36
   output hashes equal the reference build's (`-D MOJOLEARN_GEMM_NO_WINDOW_ADMIT=1
   -D MOJOLEARN_GEMM_NO_KPACK_NARROW=1`, main's GEMM) on all 36; the narrow
   tile alone equal on 36; the admission sabotage DIFFERS on all 12 cases of
   the subnormal-forcing kind (the check can fail) (`ab/*.hashes`).
2. The chain replays above (PASS, H100 chain lines, both windows).
3. `gemm_device_check` (runner, 8 gates), `gemm_backward_check` (10 gates),
   `gemm_workspace_check` (9 GEMMs, 4,608 cells bitwise equal to the host
   oracle): green.
4. `python -m mojolearn verify` over the 201 non-par lanes that reach
   `gemm/checks/gemm_identical.mojo` (`tools/nvidia_step_time/lanes_gemm_nonpar.txt`,
   the AMD lane's list), chunks of 25, against the shipped reference table
   (`verify/`): **181 lanes VERIFIED, 6,813 cell parts IDENTICAL, 0
   DIVERGENT, 0 OWED; 20 REFUSED**, every refusal a binding these legs did
   not build (the byte LM, tokenizer and saved-model HOST bindings, samba,
   the random forest family, arima, and a few gbdt/gp/metrics/ols variants),
   the same list as the AMD lane's legs; none a disagreement.
5. Lean B4 witnesses equal across every binding (above).

## Tried and not taken (all bit-identical where they ran)

- GEMM launch bound 512 on the 128x128 tile (128 registers, 560 B spilled):
  neutral. 80-register variants (bound 768: 128x64 and 64x64): slower. The
  64x64 tile at 97 registers: slower. Every older GEMM trial arm (`tuned128`,
  `lfold`, `half`, `quarter`, `kpack`, `kpack_wide`, `ksplit_leaf`, `kfoldv`,
  ...): slower than `kpack_hg` (`legs/leg1/arms.txt`).
- The gather of the next window issued before the staging barrier: slightly
  slower. ksplit block parallelism 264 (two blocks an SM): the weight
  gradient calls slower (0.542 -> 0.571 ms).
- Attention forward staging a whole key block with 16-byte q/k loads: slower
  (5.69 -> 5.93 ms a launch); dq at bound 1024: slower than 768. Attention
  arm words `_kvgrid_r64`: slower; `_zdefer`, `_zlag` with this word: not
  valid.
- `nvvm.minctasm` / `nvvm.maxnreg` metadata: refused or crashes the compiler
  as scalars and arrays (`tools/nvidia_step_time/minctasm_probe.mojo`); the
  register budget is set through `MAX_THREADS_PER_BLOCK_METADATA` instead.
- Tensor cores: not attempted. sm_90 has no instruction that forms ONE fp32
  product plus an fp32 accumulator rounded once to fp32: TF32 `mma`
  truncates the operands, FP64 `mma` rounds to fp64 and the second rounding
  to fp32 is a double rounding, and the fp32-accumulating `mma`/`wgmma` sum
  several products per instruction in their own order. No bit-equality
  argument like the AMD lane's K=1 MFMA exists here.

## Where the time goes now, and the gap, kernel by kernel

Per shard, head (476.9 ms of kernel time, 0.481 s lean):

- **GEMM 304 ms (64 percent)**: the kpack kernels 294 and the group folds 10;
  6.07 TFLOP a shard at about 20 TFLOP/s.
  The DIAG decomposition on main's kernel (`legs/leg1/diag_t3.log`,
  `legs/leg2/diag_t3.log`): the bare FMA chain alone is 122 ms. The flush
  multiply is gone on admitted windows (26 percent of main's kernel). What
  is left: the global gather of the next window (removing the prefetch or
  the staging stores alone saves 30 percent of main's kernel, the barrier
  alone 5), the fold push at every leaf (15 percent), and one or two blocks
  an SM. The next levers are asynchronous staging (`cp.async` into a deeper
  page ring, which needs dynamic shared memory above 48 KB and the admission
  minima read back from shared memory) and a fold stack that does not live
  in thread-local memory. Windows whose operands fail admission (tiny
  gradients) keep the two-instruction step; on the T3 data the admitted
  share is high enough that every GEMM call priced about 20 percent faster.
- **Attention 131 ms (28 percent)**: forward 42.8 (three passes over a
  global score stash; latency-bound, which is why occupancy moved it), zdot
  33.9, dq 29.1, dk/dv 25.2. Their chains still flush per product on the
  hardware seam; the same window admission applies to their dot chains (the
  q·k chains over the head dimension and the p·v chains over keys) and is
  the next lever there; the zdot and dk/dv kernels already sit at 63
  registers (four blocks an SM).
- **Everything else about 42 ms**: cross entropy 14 (memory-bound passes over
  the 8,192 x 50,257 logits), the nonfinite scans 5.8, norms, rope,
  residuals, the optimizer.
- **Host hashing, 6.5 to 7.4 s a step** (18 percent of the wall time of a
  30.7 s step) is outside the step seconds: the state (1.95 GB) and the
  summed gradient (649 MB) read back and hashed under the recipe's
  `sliced-sha256-8.v2` scheme after each step. It could overlap the next
  step's device work without changing a hash; not attempted (it is the
  segment runner's code, shared with the paused run).

## Owed

- AMD: the attention launch-bound rows now declare 1024 on the AMD kernels
  (their backend's default, so the same code is expected), the embedding
  scan kernel is shared by every column, and `gemm_identical.mojo` gained
  NVIDIA-only rows. The AMD byte LM binding builds for gfx942 from the head
  (`legs/leg4/builds/byte_lm.amd_gfx942.log`); AMD bits (the replays and the
  201 lanes the AMD README lists) and speed must be re-proven before a
  release. Apple: not compiled here (the embedding kernel reaches it).
- NVIDIA: the 20 refused lanes (their bindings) and the 59 par-* drivers (two
  devices); the two-H100 step time (19.5 s before).
- A release carrying the bindings.

## Costs

Leg 1 33.7 min ($1.96), leg 2 38.2 min ($2.22), leg 3 60.8 min ($3.54), leg
4 27.4 min ($1.59), at $3.49 an hour: $9.31 in all. No box of another lane,
the T3 run or the release was touched.

## Files

`RESUME.md` (the lane's running state), `kernel_map.md`,
`legs/leg{1,2,3,4}/` (each leg's `session.txt`, `ab/`, `ptx/`, `lean-*/`,
`replay-*.clean.log` and `replay-*/`, `item-timers.summary.*`, `nsys-*/`,
`diag_t3.log`, `verify/`, `runner.console.log`; larger raw files in
`~/mojolearn-evidence/nvidia-step-time/`). Tools:
`tools/nvidia_step_time/{session.sh,leg1.sh,leg2.sh,leg3.sh,leg4.sh,arms.sh,geom.sh,attn_arms.sh,verify_lanes.sh,probe_gemm_ptx.mojo,minctasm_probe.mojo,nsys_fold.py,lanes_gemm_nonpar.txt}`;
DIAG variants 6/7/8 in `bench/gemm_step_diag_main.mojo`; `-D
MOJOLEARN_GEMM_STEP_T3=1` in `gemm/checks/gemm_step_arms.mojo`.
