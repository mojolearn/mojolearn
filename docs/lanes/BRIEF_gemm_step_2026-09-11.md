# GEMM step lane, the IDENTICAL GEMMs of the byte LM training step (DEVIATIONS 2540 to 2544)

Source-only lane, September 11, 2026, IDENTICAL only. STATUS: WIP, wound
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
