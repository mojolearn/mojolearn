# GEMM step lane, the IDENTICAL GEMMs of the byte LM training step (DEVIATIONS 2540 to 2544)

Source-only lane, September 11, 2026, IDENTICAL only. STATUS: BUILT, NOT
RUN (second session, same day; section 10). The first session was wound
down by Andrew's order before any code was written. This brief holds the
reading, the counted model, the arm designs and the identity argument for
every arm. No kernel, harness, check or leg exists yet (section 9 lists
what is owed). Nothing here ran on the Mac or on a GPU. Numbers not copied
from an evidence path are counted from source and say so.

Deciding column (coordinator, relaying Andrew, same day): **AMD Instinct
MI325X on DigitalOcean (HIP, `gfx942`)**. NVIDIA H100 is a confirmation
column. Geometry that differs by vendor is a kernel matrix row, never an
inline vendor branch. The ptxas register/stack/spill readback is an NVIDIA
instrument, not a gate.

Arithmetic boundary: [HANDOFF_speed_gemm_2026-09-10.md](HANDOFF_speed_gemm_2026-09-10.md)
(no tensor cores, no TF32, no reassociation, RN-FMA then FTZ multiply kept,
leaf boundaries and fold topology kept). The scalar shared-load trial was
negative and is not repeated. Layout of this brief follows
[BRIEF_attention_step_2026-09-11.md](BRIEF_attention_step_2026-09-11.md).

## 1. The entry points the LM step reaches (read from source)

Every GEMM of the lean step goes through ONE host function,
`gemm/checks/gemm_identical.mojo::identical_gemm_into`, which calls
`identical_gemm_with_plan(..., choose_gemm_plan(m, n, k))`.

- Head forward: `training/byte_lm.mojo::_byte_forward_loss` calls
  `identical_gemm_into(logits, residual2, lm_w, M, V, DM, OP_NT)`.
- Head backward: `_byte_step_device` calls
  `gemm/checks/gemm_backward.mojo::identical_gemm_backward_a_into` and
  `identical_gemm_backward_b_into`, both of which call `identical_gemm_into`.
- Block forward: `transformer/impl/llama/modeling_llama.mojo` calls
  `identical_gemm[False]` (q, k, v, o projections, gate, up, down), which
  allocates a 1-float workspace, synchronizes, and calls
  `identical_gemm_into[False]`.
- Block backward: `transformer/checks/transformer_backward.mojo::llama_decoder_layer_backward_device`
  calls `_route_a` / `_route_b` (DEVIATION 1428), which call the
  synchronizing `identical_gemm`, then `identical_gemm_into`. The RMSNorm
  weight gradient (`identical_gemm(ctx, dw_out, ones, dprod, 1, dm, m, OP_NN)`)
  takes a SPLIT plan and no arm touches it.

So a trial hook placed in `identical_gemm_into`, under a comptime define,
reaches every call the step makes, and the shipped build compiles nothing
new.

## 2. The calls at the target shape, their plan, and their block count

Target: 12 layers, DM 768, FF 2048, V 50257, L 2048, batch 1 (`TARGET_SHAPE`
in `tools/lm_step_memory_probe.py`). Backward shapes come from
`gemm_backward_a_call` (dA of OP_NT is OP_NN at `(m, k, n)`) and
`gemm_backward_b_call` (dB of OP_NT is OP_TN at `(n, k, m)`). Plans follow
`choose_gemm_plan` by hand; every row below resolves to
`PLAN_TUNED_128_8X8` (128x128 tile, register tile 8x8, 16x16 thread grid,
KS 16, two shared pages of 20,480 B, fold stack 16 x 64 floats in
thread-local memory).

| call | op | m x n x k | P | per step | blocks at 128x128 |
|---|---|---|---:|---:|---:|
| proj fwd (q, k, v, o) | NT | 2048 x 768 x 768 | 6 | 48 | 96 |
| proj dA | NN | 2048 x 768 x 768 | 6 | 48 | 96 |
| proj dB | TN | 768 x 768 x 2048 | 16 | 48 | 36 |
| gate/up fwd | NT | 2048 x 2048 x 768 | 6 | 24 | 256 |
| gate/up dA | NN | 2048 x 768 x 2048 | 16 | 24 | 96 |
| gate/up dB | TN | 2048 x 768 x 2048 | 16 | 24 | 96 |
| down fwd | NT | 2048 x 768 x 2048 | 16 | 12 | 96 |
| down dA | NN | 2048 x 2048 x 768 | 6 | 12 | 256 |
| down dB | TN | 768 x 2048 x 2048 | 16 | 12 | 96 |
| head fwd | NT | 2048 x 50257 x 768 | 6 | 1 | 6,288 |
| head dA | NN | 2048 x 768 x 50257 | 393 | 1 | 96 |
| head dB | TN | 50257 x 768 x 2048 | 16 | 1 | 2,358 |

## 3. What bounds these calls (a counted model, not a measurement)

Two pieces of retained evidence.

1. The static resources of the shipped 128x128 specialization on an H100
   (`bench/results/gemm_resources_2026-09-10/`): 255 registers per thread,
   4,144 local bytes per thread, 44 bytes of spill stores and loads, one
   active 256-thread block per SM, 2,048 maximum threads per SM.
2. The kNN lane's runtime attribute table on the same GPU model
   (`bench/results/e1g/2026-09-11_014151-nvidia/remote/knn-kernel-stats/driver_lines.txt`,
   `DeviceContext.compile_function[kern]()` then
   `get_attribute(Attribute.NUM_REGS)` and
   `occupancy_max_active_blocks_per_multiprocessor(256, 0)`): registers
   31, 40, 48, 56, 99, 107, 178 give 8, 6, 5, 4, 2, 2, 1 blocks per SM.
   Every row equals `min(8, floor(256 / registers))`.

The H100 has 132 SMs (the attention brief's issue-capacity count). So a
128x128 launch with at most one block per SM can occupy at most
`min(blocks, 132)` SMs. Fitting the measured H100 phase numbers of the task
(leg `bench/results/e1g/2026-09-11_113013-nvidia-h100-attention-step`,
earlier leg `2026-09-11_004220`) with "rate proportional to occupied SMs,
11.37 TFLOP/s at 132":

- head fwd, 6,288 blocks: predicted 11.37, measured 158.1 GFLOP / 13.9 ms =
  11.37 TFLOP/s; head dB, 2,358 blocks: 11.37 predicted, 11.29 measured;
  head dA, 96 blocks: 8.27 predicted, 8.59 measured.
- `attn.qkv_proj`, three proj fwd per layer at 96 blocks: 10.5 ms
  predicted, 10.1 measured. `attn.o_proj`: 3.5 predicted, 3.6 measured.
- `block.mlp_and_residuals`, gate and up at 256 blocks plus down at 96:
  23 ms of GEMM predicted beside 26.6 measured with the residuals.
- `bwd.after_attention`, four proj dA at 96 blocks and four proj dB at 36
  blocks per layer: 51 ms predicted, 40.5 measured (over-predicted; the TN
  dB may serve more per SM).

Reading: on the H100 the per-layer LM GEMMs are GRID-bound (36 to 96
blocks against 132 SMs) and occupancy-bound at one block per SM, and the
head is occupancy-bound. Two levers follow, and both are execution plan:
more blocks for the small outputs, and fewer registers per thread so more
than one block fits an SM. Removing the 44 bytes of spills is a third,
unquantified. The model is a fit to phase timers, not a proof; the leg's
resource readback and per-call price decide.

AMD: no SM count, no register budget and no occupancy query for the HIP
column exists anywhere in the repo. The leg will attempt the same
`DeviceFunction` attribute and occupancy calls on the MI325X and record
what they return or raise; until then the AMD column is priced, not
modeled.

## 4. Kernel matrix rows read (the AMD column)

`checks/kernel_matrix.mojo`: `column_shared_limit(COLUMN_AMD)` 64 KB
(NVIDIA 48 KB, Apple 32 KB); `column_lane_width(COLUMN_AMD)` 64 (NVIDIA and
Apple 32); `lib_block_size_for[K_LIB_GEMM_CONTRACTION, *]` 256 on every
column; `lib_hardware_ftz_fma_for` NVIDIA only, so AMD takes the software
`ftz(identical_mul_add(a, b, acc))` seam and `TUNED_STAGE_FTZ` is off (the
operand flush happens at use, not at staging); `gemm_wide_split_for` NVIDIA
only. No new vendor row is needed for the arms below: every vendor-varying
number is read through `lib_smem_pages_for` (shared limit) or
`lib_lane_width_for` (lane width), the idiom `TUNED_64_KS` already uses.

## 5. The chain every arm keeps (the identity argument, common part)

For output cell `(i, j)` the contract's arithmetic is: `(L, P)` from
`contract_partition(k)` alone; for each logical leaf `t` ascending, an
accumulator seeded `+0.0` takes one seam step per `p` ascending over
`[t L, min((t+1) L, k))`, the seam being `_tuned_step` (NVIDIA: RN-FMA then
RN-FTZ multiply by one; elsewhere `ftz(fma)`) on operands flushed as loaded
(`_tuned_loaded_operand` or staging ftz, a column row); the leaf partial is
`ftz(acc)`; partials enter the balanced tree by `_fold_push_local`'s merge
`ftz(ftz(slot_d) + ftz(val))` with the earlier leaves on the left; the root
is drained lowest level first by `_fold_drain_local`; the stored cell is
`ftz(root)`.

Execution-plan quantities, which no arm lets reach a leaf boundary or a
tree level: `RPT`, `CPT`, `TC` (hence `TR = 256 / TC`, `BM = RPT TR`,
`BN = CPT TC`), `KS`, `PAGES`, the grid, and the order in which a thread
visits its cells at a leaf boundary. The argument that they cannot move a
bit is `identical_gemm_tuned_kernel`'s own:

- Ownership is a bijection. Thread `tid` owns rows `accrow + u TR`
  (`accrow = tid / TC`, `u < RPT`) and columns `acccol + v TC`
  (`acccol = tid mod TC`, `v < CPT`); as `(accrow, u)` range over
  `[0, TR) x [0, RPT)` the rows cover `[0, BM)` exactly once, likewise the
  columns, and tiles are an exact bijection of the grid. So each cell has
  exactly one accumulator.
- `KS` only cuts the `p` range into windows. `_tuned_window[KS]` yields,
  for leaf `t`, windows at offsets `0, KS, 2 KS, ...` inside
  `[t L, min((t+1) L, k))`, each accumulating `chunk <= KS` steps
  ascending, trailing windows of a short leaf empty; the leaf boundary
  fires on the last window. Any `KS` that is a multiple of `VEC` gives the
  same step sequence per cell.
- `PAGES` and the staging expressions are the shipped ones: the slot to
  (row, column) map in `_tuned_g2r` and in the register-to-shared copy is
  the same expression the accumulate loop reads, for any `BM`, `BN`, `KS`
  (checked by hand for `TC` in {4, 8, 16, 32, 64}: `_tuned_g2r` never
  reads `TC`, and `ASLOTS = ceil(BM KV / 256)` covers `BM KV` slots).

## 6. The arms (designed; NO CODE YET)

All arms run only where `choose_gemm_plan` returns `PLAN_TUNED_128_8X8`,
exist only under `-D MOJOLEARN_GEMM_ARM_TRIAL=1`, and are selected by
`MOJOLEARN_GEMM_ARM` (`MOJOLEARN_GEMM_ARM_SABOTAGE=1` selects the sabotage
instantiation). The kernel is to be a trimmed copy of the non-SPLIT path of
`identical_gemm_tuned_kernel`, `identical_gemm_step_arm_kernel[RPT, CPT,
TC, KS, PAGES, LFOLD, SAB]`, reusing `_tuned_window`, `_tuned_g2r`,
`_tuned_step`, `_tuned_loaded_operand` unchanged; the shipped kernel is not
edited.

### 6.1 DEVIATION 2540, Arm A family: occupancy through live storage

`lfold` (128x128, reg 8x8, KS 16): the storage change alone.
`half` (64x128, reg 4x8, TC 16, KS 32 where two pages of 27,648 B fit the
shared limit, i.e. AMD, else 16): half the accumulators and half the
fold stack per thread, plus the cell-serial fold. `half_ks16`: the same at
KS 16 on every column (the AMD control for the K step). `quarter` (64x64,
reg 4x4, KS `TUNED_64_KS`): the smallest tile, aimed at the grid-bound
calls of section 3 (proj dB goes from 36 to 144 blocks). `quarter` repeats a
tile rejected at Llama t512 shapes (128 and 448 blocks); the new reasons
are the grid-bound LM outputs, the cell-serial fold, and an AMD column that
never priced it.

The storage change (`LFOLD = True`): at a leaf boundary the shipped kernel
builds `part = ftz(acc)` as an `NCELL`-wide register and merges all cells
per level with `NCELL`-wide loads and flushes, which is the likely register
peak (the scalar shared-load trial removed the 64 load registers of the
accumulate loop and the count stayed 255, so the peak is not there). The
arm instead computes the merge depth once from `occ` (the number of
trailing one bits, identical for every cell because every cell pushes the
same number of leaves), then walks cells in a RUNTIME loop:
`val = ftz(acc[e])`; for `d < depth`:
`val = ftz(ftz(stack[d NCELL + e]) + ftz(val))`; store `val` at slot
`depth`; then `occ += 1`. The drain walks cells in a runtime loop with
`_fold_drain_local`'s per-element expression and stores `ftz(root)`.

Identity argument for 2540. (1) Which chain: every cell's leaf chains and
fold are section 5's, with the same operands in the same slots. (2) Which
order: per cell the merges happen at levels `0 .. depth-1` in that order
with the slot as the left operand, exactly `_fold_push_local`'s sequence,
because that function also merges every occupied level from `d = 0`
upward and stores at the first free level, and `occ + 1` is the binary
counter its `occ - (2^t - 1) + 2^t` computes; the drain is element for
element `_fold_drain_local`. (3) Why unchanged: cells never share a float,
so visiting them one at a time instead of lane by lane cannot reorder any
cell's additions; tile and `KS` changes are section 5's execution-plan
quantities. The one divergence is fold overflow (`P >= 2^16`), which the
profile cap (`P <= 1024`) makes unreachable; the shipped function mutates
`occ` there and the arm returns without writing.

Counted storage: registers are not predictable from source; local fold
memory per thread is `16 NCELL` floats (4,096 B at 8x8, 2,048 at 4x8,
1,024 at 4x4). Shared pages per block: 40,960 B shipped; `half` 30,720
(KS 16) or 55,296 (KS 32, AMD); `quarter` 36,864 at KS 32.

### 6.2 DEVIATION 2541, Arm B: the head shape

`head`: only for calls with `max(m, n, k) >= 16,384` (the three head
calls). The tile spans the output's long axis: `32 x 256` when `n >= m`
(head fwd, 197 x 64 = 12,608 blocks) and `256 x 32` when `m > n` (head dA,
192 blocks instead of 96; head dB, 4,728). The thread grid is lane aligned
through `lib_lane_width_for`: long-axis thread columns equal the lane width
(AMD 64, NVIDIA 32), so one wavefront or warp owns consecutive cells of the
long axis. Register tile: N-wide AMD 8x4 (TC 64), NVIDIA 4x8 (TC 32);
M-wide AMD 4x8 (TC 4), NVIDIA 8x4 (TC 8); `NCELL` 32 everywhere; KS 16
(41,472 B a page at KS 32 fits nowhere twice); pages 2 x 23,040 B on AMD
and NVIDIA, one on Apple; cell-serial fold as 6.1. A lane width outside
8 .. 256 falls back to TC 16 (reg 2x16 / 16x2).
`half_head`: head calls take `head`, the other 128x128 calls take `half`.

Identity argument for 2541. (1) Which chain: section 5's per cell. (2)
Which order: `p` ascending within each leaf, leaves ascending, the 2540
cell-serial fold. (3) Why unchanged: only `RPT`, `CPT`, `TC`, the grid and
the applicability predicate (which reads `m`, `n`, `k` and returns a
geometry, never a partition) change, and contract 6.1 lets the execution
plan read those.

### 6.3 Reach (both arms)

`SAB = True` flips one ulp of the stored value of the first cell of thread
0 of every block (`+-0.0` becomes `+-2^-100`, otherwise the bit pattern plus
one), so a sabotage run must move at least one cell per block and the clean
run must restore the baseline bits. On a build without the trial define
the arm entry runs the shipped plan, the sabotage moves nothing, and the
check fails saying so.

## 7. Instruments designed (NO CODE YET)

- `gemm/checks/gemm_step_arms.mojo` (helpers), `gemm/checks/gemm_step_arms_check.mojo`:
  every geometry forced against `PLAN_TUNED_128_8X8` and `PLAN_FLAT` bit
  for bit on ragged controls (all three ops; `k` in {0, 1, 128, 129, 300,
  1000, 2049, 50257}; `m`, `n` off every tile multiple, including 1 x 3),
  reach per geometry, then the twelve LM calls of section 2 through the arm
  dispatch with reach per arm (`MOJOLEARN_GEMM_STEP_CHECK_LM=0` skips them
  on the M4).
- `bench/gemm_step_price_main.mojo`: per LM call, digest equality then two
  warmups and seven alternated rounds, `PRICE` and `TABLE` lines naming the
  plan or geometry that ran, and a `STEP` line weighting each call by its
  per-step count.
- `bench/gemm_step_resources_main.mojo`: `compile_function` of the shipped
  specialization and every arm geometry, printing registers, local,
  shared, max threads and blocks per SM, each in its own try (attempted on
  AMD too); NVIDIA only, `--emit asm` sidecars through
  `tools/gemm_cuda_resources.py` for the spill counts.
- `tools/gemm_step_leg.sh`: vendor from `MOJOLEARN_TARGET_COLUMN` or
  detection (`nvidia-smi -L`, else `/dev/kfd`), arch from
  `MOJOLEARN_GPU_ARCHS`, no CUDA command on the AMD path; builds, check,
  resources (instrument), price per arm, auto pick of the best `STEP`
  ratio, bindings with the trial define, both corpora, `lm-<arm>-<corpus>`
  bracketed by an opening and a closing baseline, `lmtiming-*`,
  `lm_summary.tsv` with `witnesses_equal_baseline` and the geometric-mean
  flip verdict; `status.tsv` one exit code per item, 300 s per process.
- `tools/lm_step_memory_probe.py`: record `gemm_arm` beside `attention_arm`.

## 8. What could not be resolved from source

- The register count of any arm geometry, and whether the fold is the
  register peak of the shipped kernel (the resources driver answers on
  NVIDIA).
- Anything about AMD occupancy: the SM count, a register budget, whether
  `DeviceFunction` attributes answer on HIP.
- Whether rate is proportional to occupied SMs on the AMD column, which is
  the premise of `quarter` and of the head's `256 x 32` dA tile.
- Whether `--emit asm` of a driver that only calls `compile_function`
  writes sidecars.

## 9. State at wind-down

Done: the reading (sections 1 to 4), the model (3), the arm designs and
identity arguments (5, 6), the instrument designs (7). Not done: every
file of section 7, the trial hook in `identical_gemm_into`, the arm kernel
and launchers, the probe field. DEVIATIONS 2542 (hook and selector), 2543
(check, price, resources) and 2544 (leg and probe field) are reserved for
those and not yet used.

## 10. Build lane (September 11, 2026, second session)

Source only, like the first session: nothing below ran on the Mac or on a
GPU. Section 10.1 and 10.2 were written before the kernel code they
describe; 10.3 onward after it.

### 10.1 The identity argument for the kernel as built

`gemm/checks/gemm_identical.mojo::identical_gemm_step_arm_kernel[RPT, CPT,
TC, KS, PAGES, LFOLD, SAB]` is `identical_gemm_tuned_kernel` with `SPLIT =
False`, `FS = TUNED_FOLD_SLOTS` and `SWIZZLE_NONE` fixed. Copied unchanged:
the prologue, `_tuned_g2r` for both operands, the register to shared copy
into page `w % PAGES` in both staging mappings, the prefetch, both
accumulate paths through `_tuned_loaded_operand` and `_tuned_step`, the
`PAGES == 1` barrier, and the leaf boundary test `win[2] == 1`. What
differs, and why no stored bit can move:

1. The LFOLD leaf boundary (2540). `depth` is the number of trailing one
   bits of `occ`, computed once per thread per leaf. If `depth >= FS` the
   thread returns: that is fold overflow, `P >= 2^16`, unreachable under
   the profile cap `P <= 1024`, and block uniform because `occ` is a
   function of the leaf count alone, so every thread of the block returns
   at the same window and no barrier is left waiting. Otherwise, for each
   cell `e` ascending, `val = ftz(acc[e])` (5d); for `d` in `0 .. depth-1`,
   `val = ftz(ftz(stack[d NCELL + e]) + ftz(val))` (5e, 5f, the slot on the
   left); `stack[depth NCELL + e] = val`; then `occ += 1`.
   `_fold_push_local` merges every cell at once at `d = 0, 1, ...` while
   bit `d` is set, stores at the first clear bit, and computes
   `occ - (2^depth - 1) + 2^depth`, which is `occ + 1`. Same operands, same
   order, same slot, per cell. Cells share no float, so walking them one at
   a time cannot reorder any cell's additions.
2. The LFOLD drain. Per cell, `_fold_drain_local`'s element expression
   (ascending `d`, the first occupied slot copied bit for bit, each later one
   `ftz(ftz(slot) + ftz(root))`), then the stored `ftz(root)` (5g). The arm
   drains only cells inside the output (`gi < m`, `gj < n`); the shipped
   kernel drains every lane and masks the store. A drain that is never
   stored moves no stored bit.
3. `LFOLD = False` keeps the shipped lane-wide `_fold_push_local` and
   `_fold_drain_local` calls. It is not an arm: it is the resources
   control, the trimmed copy at the shipped geometry.
4. No swizzle: `tile = raw`, the identity bijection, which is what the
   shipped plan passes (`SWIZZLE_NONE`).
5. The arm kernel names none of the file's global sabotage switches itself
   (the shared helpers it calls keep theirs). The arms check refuses to run
   on a build that defines one, because its baseline would be a sabotaged
   kernel.
6. `SAB = True` (reach, 6.3). At the store, thread 0's cell `(u, v) = (0,
   0)`, which is output cell `(i0, j0)` and always inside the output,
   stores `_gemm_step_arm_sabotage(value)`: `+-0.0` becomes `+-2^-100` (bit
   pattern `0x0D800000` with the sign kept, a normal no later seam
   flushes), any other pattern becomes the pattern plus one. The `k == 0`
   store does the same. So a sabotage launch moves exactly one cell per
   block, and the check requires `moved == blocks` of that geometry, which
   also names the geometry that ran (`half` at proj fwd is 192 blocks, the
   shipped 96).
7. The geometries (execution plan, contract 6.1). `lfold`: `(2 TUNED_RPT,
   2 TUNED_CPT, TUNED_TC, 16)`, 128x128. `half`: `(TUNED_RPT, 2 TUNED_CPT,
   TUNED_TC, GEMM_HALF_KS)`, 64x128, `GEMM_HALF_KS` 32 where
   `lib_smem_pages_for` answers 2 for the 27,648 B page at KS 32, else 16.
   `half_ks16`: the same at 16. `quarter`: `(TUNED_RPT, TUNED_CPT,
   TUNED_TC, TUNED_64_KS)`, 64x64. `head`: with `W =
   lib_lane_width_for[TARGET_COLUMN]` when `W` is a power of two in 8 ..
   256 and the block is 256 threads, else 16. N-wide (`n >= m`): `TC = W`,
   `RPT = 32 / (256 / W)`, `CPT = 256 / W`, tile 32x256. M-wide (`m > n`):
   `TC = 256 / W`, `RPT = 256 / W`, `CPT = 32 / (256 / W)`, tile 256x32.
   KS 16. AMD (`W = 64`): N-wide reg 8x4 TC 64, M-wide reg 4x8 TC 4.
   NVIDIA and Apple (`W = 32`): N-wide reg 4x8 TC 32, M-wide 8x4 TC 8.
   `PAGES` from `lib_smem_pages_for` at `(BM + BN)(KS + 4) 4` bytes, and a
   comptime assert that one page fits (`lib_smem_page_fits_for`). None of
   these reaches `leaf_in` or `p_in`, which the launcher takes from
   `contract_partition(k)` exactly as `identical_gemm_with_plan` does.
8. Applicability (2542). `gemm_step_arm_geometry(arm, m, n, k)` returns the
   shipped geometry unless `choose_gemm_plan(m, n, k)` is
   `PLAN_TUNED_128_8X8`; `head` also requires `max(m, n, k) >= 16,384`. It
   reads `m`, `n`, `k` and returns a geometry id, never a partition.

### 10.2 A claim of 6.2 the ownership rule does not support

6.2 says the head tile puts "one wavefront or warp" on "consecutive cells of
the long axis". That holds for the N-wide tile only. Thread `tid` owns rows
`tid / TC + u TR` and columns `tid mod TC + v TC`. With `TC = W`, a lane's
`W` threads share one `accrow` and take every `acccol`, so one lane owns
all 256 columns of the tile. The M-wide tile's long axis is rows, and a
lane of `W` threads spans only `W / TC = W^2 / 256` consecutive `accrow`
values (16 on AMD, 4 on NVIDIA), each strided by `TR = W`. Under this
kernel's ownership rule no M-wide geometry with `W < 256` puts a lane on
consecutive long-axis rows. The geometry is still legal execution plan and
is built as designed; a head dA or head dB price is not evidence about lane
alignment.

### 10.3 What was built, per deviation

Nothing ran. Nothing flips a default. In `identical_gemm_into` the shipped
dispatch line is unchanged, and the hook sits in front of it under `comptime
if GEMM_ARM_TRIAL`. The other edits to shipped files are three imports in
`gemm_identical.mojo` (`bitcast`, `getenv`, `lib_lane_width_for`) and one
recorded field in the probe.

- **2542 BUILT** (`gemm/checks/gemm_identical.mojo`, the section after
  `identical_gemm_into`). `GEMM_ARM_TRIAL`; the hook; `gemm_step_arm_from_env`
  (a shipped build reads no environment); `gemm_step_arm_parse` (raises on an
  unknown name, the empty name is `shipped`); `gemm_step_arm_name`;
  `gemm_step_arm_geometry` (the applicability predicate of 10.1 item 8);
  `gemm_step_geometry_tile`, `_blocks` and `_name`;
  `identical_gemm_step_geometry_into` (a forced geometry, clean or sabotage,
  for the harnesses; on a non-trial build every geometry runs the shipped
  plan); `_gemm_step_arm_hook`.
- **2540 BUILT.** `identical_gemm_step_arm_kernel[RPT, CPT, TC, KS, PAGES,
  LFOLD, SAB]`, `_launch_step_arm`, `_step_geometry_launch`; geometries
  `lfold`, `half`, `half_ks16`, `quarter`.
- **2541 BUILT.** Geometries `head_n` and `head_m` from the lane width;
  arms `head` and `half_head`.
- **2543 BUILT.** `gemm/checks/gemm_step_arms.mojo` holds the twelve LM
  calls, derived from each layer's forward `OP_NT` through
  `gemm_backward_a_call` and `gemm_backward_b_call`, plus the fill, poison,
  readback, compare, digest and median helpers.
  `gemm/checks/gemm_step_arms_check.mojo` runs in three parts. The selector
  part checks that names round-trip and that an unknown name raises from the
  parser and from `identical_gemm_into`. The ragged part runs all three ops;
  `k` in {0, 1, 128, 129, 300, 1000, 2049, 50257}; `m x n` of 1x3, 33x70,
  129x257, 300x129 and 257x520; and the subnormal-product kind at 129x257.
  There every geometry must match the shipped 128x128 plan and FLAT bit for
  bit, with reach as `moved == blocks`. The LM part sends the twelve calls,
  the three head calls among them, through `identical_gemm_into` under
  every arm by `setenv`, with bits and reach checked per call.
  `bench/gemm_step_price_main.mojo` prints BITS, SAMPLE, PRICE, TABLE and
  STEP lines, naming the plan and the geometry beside each timing.
  `bench/gemm_step_resources_main.mojo` prints registers, local, shared,
  const, max threads and blocks per SM for the shipped kernel, a trimmed
  control (`LFOLD = False` at 128x128) and every geometry. Each attribute
  is its own print, and each geometry its own try. It runs on any vendor
  and needs no ptxas.
- **2544 BUILT.** `tools/gemm_step_leg.sh` passes `sh -n` and `dash -n`.
  `tools/lm_step_memory_probe.py` records `gemm_arm` in the run mode, the
  setup event (`gemm_arm_requested`) and `result.json`.

Departures from the section 7 design:

- The arm kernel lives in `gemm_identical.mojo`, not
  `gemm_step_arms.mojo`: the hook dispatches it, so a separate module would
  be a circular import.
- The check refuses a build that defines a global GEMM sabotage.
- The leg brackets each corpus with `lm-shipped-<corpus>` first and
  `lm-shippedclose-<corpus>` last, and each ratio is taken against the
  mean of the two.
- `lmtiming-*` is opt-in (`MOJOLEARN_GEMM_STEP_LEG_LMTIMING=1`) so the leg
  fits the lease.
- Spill counts are not captured: the only instrument for them is
  NVIDIA-only.

### 10.4 RUN OWED, in order

M4, light, run by the orchestrator:

1. The gate on a trial build:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . gemm/checks/gemm_step_arms_check.mojo -o /tmp/gemm-step-check`
   then
   `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000 /tmp/gemm-step-check`.
   Expect PASS, and every `REACH ragged` line at N/N.
2. The same check on a build WITHOUT the trial define:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_step_arms_check.mojo -o /tmp/gemm-step-check-notrial`
   then
   `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000 /tmp/gemm-step-check-notrial`.
   Expect FAIL, naming the missing define (reach is not provable there).
3. The shipped path:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo -o /tmp/gemm-device-check && /tmp/gemm-device-check`
   stays green. On any shipped byte LM binding built from this commit,
   `nm -C _mojolearn_byte_lm.so | grep -c step_arm` prints 0.
4. Builds only, no run:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_price_main.mojo -o /tmp/gemm-step-price`
   and the same for `bench/gemm_step_resources_main.mojo`.

AMD, the deciding column:

5. The leg in 10.5, from a `git worktree add --detach` checkout of the
   commit that carries this lane.
6. Read these back:
   - `status.tsv`: every item should exit 0.
   - `step-check.log`: PASS, and an OK `LM` line for each of the 7 arms on
     each of the 12 calls.
   - `resources_lines.txt`: the repo's first AMD register and occupancy
     readback, or the raise that says HIP does not answer.
   - `price_step.txt` and `price_tables.txt`.
   - `lm_summary.tsv`: `witnesses_equal_baseline`, `ratio_vs_shipped` and
     the `verdict` lines.

   A FLIP verdict on the MI325X flips the arm as the default in the same
   session (ENGINEERING_RULES 8, 9 and 10). The H100 confirmation leg
   follows it (`tools/gemm_step_leg.sh` header).

### 10.5 The AMD leg

```sh
MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
MOJOLEARN_GPU_ARCHS=gfx942 \
MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_step_leg.sh \
MOJOLEARN_DO_EXTRA_ENV="MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,lfold,half,half_ks16,quarter,head,half_head MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=auto" \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi325x-gemm-step \
bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates
```

`LM_ARMS=auto` probes the arm with the lowest STEP ratio among the price
runs that exited 0. Name arms explicitly (`MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=
half,half_head`) when the lease allows more probes: each extra arm adds two
probes of one to two minutes.

### 10.6 Risks that need a build or a box

- **API use not compiled here.**
  - `bitcast` inside a kernel (`metrics/impl/binary_ranking.mojo` does it).
  - `setenv` in a check (`checks/e2_growth_cards.mojo` does it).
  - `HostBuffer.unsafe_ptr()` on an immutable argument, which the helpers'
    compare, digest and poison count use. `kmeans_check.mojo` takes a
    `HostBuffer` immutably; the build decides.
- **Whether a non-trial build elaborates any of the new section.**
  RUN OWED 3 checks with `nm`.
- **Compile time.** A trial build instantiates 12 arm kernels (6
  geometries, clean and sabotage) in the check, the price and the byte LM
  binding. HIP compile time for them is unknown and could crowd the
  60-minute lease. If it does, the knobs are
  `MOJOLEARN_GEMM_STEP_LEG_SKIP_RESOURCES=1` and a shorter
  `MOJOLEARN_GEMM_STEP_LEG_ARMS`.
- **1-D grid sizes.** At head fwd, `quarter` launches 25,152 blocks and
  `half` and `head` 12,608, against the shipped 6,288. A vendor grid limit
  would show as a launch error in the check.
- **Unknown until the resources driver prints.** Register counts of every
  arm, AMD occupancy, and whether `compile_function` attributes answer on
  HIP at all. Whether rate follows occupied SMs on AMD (section 8) stays
  open until the price lines exist.
- **Host memory.** The check and the price hold two 412 MB host buffers at
  head fwd.
- **The trial binding reads the environment on every GEMM call.** That is
  about 300 calls per step, the same per-call pattern as the attention
  lane's hook.

### 10.7 H100 leg result (2026-09-11, RunPod, commit cd086f67): NO FLIP

Run on NVIDIA first because the shared DigitalOcean GPU was held by the trees
lane (Andrew: "use runpod then and just do nvidia for now"). The AMD leg in
10.5 is still owed, so this is not a verdict for the deciding column.

Evidence: bench/results/e1g/2026-09-11_133216-nvidia-h100-gemm-step
(`NVIDIA H100 80GB HBM3`, driver 580.126.09, sm_90a, pod tr4o460ozelb4s
terminated and verified, 19 minutes on the pod). Command: 10.5 with
`tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 --gpu
"NVIDIA H100 80GB HBM3" --allow-concurrent` and the default arms. The leg's
own device card matched the M4 card generated at the same commit on all 60
stages.

- `status.tsv`: all 22 items exit 0. The Pile GitHub fetch took 426 s.
- `step-check.log`: PASS. Every geometry bit-equal to the shipped plan and to
  FLAT, reach proven per geometry, and an OK `LM` line for every arm on the
  twelve target-shape calls, including the three vocab-sized head calls the
  M4 skipped.
- `resources_lines.txt` (registers, local floats, blocks per 256-thread SM):
  shipped 255, 4144, 1; lfold 254, 4352, 1; half and half_ks16 226, 2176, 1;
  quarter 214, 1088, 1; head_n and head_m 254, 2176, 1. **No arm crosses to
  two blocks per SM.** The register cut is real but too small to change the
  block count, so each arm pays for its extra launches and gets nothing back.
- `price_step.txt` (GEMM sum per step, 12 calls weighted by count; shipped
  about 219 ms): quarter 1.270, head 1.277, half 1.842, half_ks16 1.844,
  half_head 1.878, lfold 2.862, shipped control 1.000. Only proj_dB (768 x
  768 x 2048) comes in under 1.0 for half (0.991), inside noise.
- `lm_summary.tsv` (`LM_ARMS=auto` picked quarter): lean step shipped
  0.4573 s (enwik8) and 0.4556 s (Pile GitHub), shipped close 0.4557 and
  0.4552, quarter 0.5145 and 0.5143. Witnesses equal on every step.
  `verdict quarter NO FLIP geomean=1.1282 enwik8=1.1271 pilegithub=1.1292`.

Reading: on the H100 the shipped 128x128 plan already fits one block per SM,
and the occupancy arms as built do not move that. They are declined on
NVIDIA. Whether HIP on the MI325X has a different register budget or SM
geometry is exactly what the AMD leg's `resources_lines.txt` answers; do not
build more occupancy arms before it has been read.

Open, not attributed: this leg's shipped lean step (0.4573 / 0.4556 s) is
slower than the same commit's shipped step on the concurrent attention leg
(`stash_tiled` 0.3835 / 0.3833 s, bench/results/e1g/2026-09-11_133041-nvidia-h100-attention-torch),
on a different H100 pod. The GEMM binding here is a `MOJOLEARN_GEMM_ARM_TRIAL`
build whose hook reads the environment on every GEMM call (10.6); the box
also differs. Nothing measured separates the two. Both the numerator and the
denominator of the NO FLIP ratio carry whatever it is, so the verdict stands,
but a shipped (non-trial) binding probe on the same box is owed before any
GEMM step time from a trial build is quoted.
