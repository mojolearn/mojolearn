# GEMM long-k lane, the IDENTICAL GEMM calls whose blocks cannot fill the machine (DEVIATIONS 2590 to 2594)

Source-only lane, September 11, 2026, IDENTICAL only. STATUS: SOURCE
BUILT, NOTHING COMPILED OR RUN. The first session wound down after the
reading, the design and the identity argument (sections 1 to 8). A second
session built DEVIATIONS 2590 to 2594 from that design (section 9): the
group kernel, the two arms, the host and device checks, the harness lines
and the leg wrapper. Nothing here was built, compiled or timed on the Mac
or on a GPU; section 9.4 is the RUN OWED list and 9.5 the two legs.
Numbers not copied from an evidence path are counted from source or fitted
to the evidence table and say which. Sections 1 to 5 were written before
any code; sections 6 onward after it.

Arithmetic boundary is the one of
[HANDOFF_speed_gemm_2026-09-10.md](HANDOFF_speed_gemm_2026-09-10.md) and
[BRIEF_gemm_step_2026-09-11.md](BRIEF_gemm_step_2026-09-11.md) section 5.
The occupancy arms of DEVIATIONS 2540 to 2544 (that brief's sections 10
and 10.7) lost on the H100 and are not repeated. No tile, register tile,
K step or fold storage of the shipped plan changes here.

## 1. Purpose

Make the IDENTICAL GEMM of the byte LM training step cheaper where the H100
price table shows an outlier, without moving one output bit on any vendor,
and hand back a leg that decides it on the two LM corpora.

## 2. The measurement

Evidence: `bench/results/e1g/2026-09-11_133216-nvidia-h100-gemm-step/remote/gemm-step/price_tables.txt`
(commit cd086f67, `NVIDIA H100 80GB HBM3`, shipped plan `TUNED 128x128
reg8x8 KS=16 fold=16 local tpb=256 hwftz=True (1-D grid, no swizzle)`, the
`price-shipped` run). Useful TFLOP/s counts two flops per product step.

| call | op | m x n x k | per step | shipped blocks | shipped ms | TFLOP/s |
|---|---|---|---:|---:|---:|---:|
| proj_fwd | NT | 2048 x 768 x 768 | 48 | 96 | 0.329 | 7.34 |
| proj_dA | NN | 2048 x 768 x 768 | 48 | 96 | 0.336 | 7.19 |
| **proj_dB** | **TN** | **768 x 768 x 2048** | **48** | **36** | **0.887** | **2.72** |
| gateup_fwd | NT | 2048 x 2048 x 768 | 24 | 256 | 0.671 | 9.60 |
| gateup_dA | NN | 2048 x 768 x 2048 | 24 | 96 | 0.906 | 7.11 |
| gateup_dB | TN | 2048 x 768 x 2048 | 24 | 96 | 0.907 | 7.11 |
| down_fwd | NT | 2048 x 768 x 2048 | 12 | 96 | 0.889 | 7.25 |
| down_dA | NN | 2048 x 2048 x 768 | 12 | 256 | 0.683 | 9.43 |
| down_dB | TN | 768 x 2048 x 2048 | 12 | 96 | 0.916 | 7.03 |
| head_fwd | NT | 2048 x 50257 x 768 | 1 | 6,288 | 16.59 | 9.53 |
| head_dA | NN | 2048 x 768 x 50257 | 1 | 96 | 22.03 | 7.18 |
| head_dB | TN | 50257 x 768 x 2048 | 1 | 2,358 | 16.74 | 9.44 |

The weighted GEMM sum is about 219 ms of the 383 ms lean step (torch 2.4.1
eager TF32 runs the whole step in 38 ms on the same pod, eager FP32 62 ms).
proj_dB does the flops of proj_fwd in 2.7 times the time, about 27 ms per
step. The same table's `price-quarter` run matters to section 3: quarter
(64x64 tile, cell-serial fold) priced proj_dB at 144 blocks at 0.736 ms and
proj_fwd at 384 blocks at 0.436 ms.

## 3. The source reading

### 3.1 What one block of the shipped plan does

`identical_gemm_into` sends every call above to
`identical_gemm_with_plan(..., PLAN_TUNED_128_8X8)`, which launches ONE
specialization, `identical_gemm_tuned_kernel[8, 8, 16, 16, 16, PAGES]`,
through `_launch_tuned` on a 1-D grid of `ceil(m / 128) ceil(n / 128)`
blocks of 256 threads. The op never reaches the specialization; it arrives
as four runtime strides from `gemm_operand_strides`. So every call in the
table runs the same compiled kernel with the same registers, local and
shared sizes (`resources_lines.txt`: 255 registers, 4,144 local bytes, one
block per SM).

Inside one block, for `P = contract_leaf_count(k)` leaves of `L = 128`
steps and `KS = 16`, the thread loop runs `w_total = P * ceil(L / KS) =
8 P` windows. Every window stages `BM KS = 2,048` floats of A and 2,048 of
B from global memory into two shared pages, passes one barrier, and
accumulates 16 product steps into 64 cells on each of the 256 threads, so
262,144 product steps. The per-window work does not depend on `m`, `n`,
`k` or the op. What does depend on them is the number of blocks (`m` and
`n` only) and the number of windows one block walks in sequence (`k`
only, `8 P`).

| call | blocks | windows per block | block windows |
|---|---:|---:|---:|
| proj_fwd, proj_dA (k 768, P 6) | 96 | 48 | 4,608 |
| proj_dB (k 2048, P 16) | 36 | 128 | 4,608 |
| gateup_dA, gateup_dB, down_fwd, down_dB (P 16) | 96 | 128 | 12,288 |
| head_dA (P 393) | 96 | 3,144 | 301,824 |

proj_dB and proj_fwd issue the same 4,608 block windows. proj_fwd spreads
them over 96 blocks that run side by side and walk 48 windows each; proj_dB
puts them on 36 blocks that walk 128 each. The long `k` of a TN dB lands on
the axis a block walks in sequence, while its short output lands on the
axis blocks spread over.

### 3.2 Competing explanations, and what separates them

- **E1, block count against the machine (grid-bound).** With one block per
  SM (resources line) and 132 SMs on the H100 (the attention brief section
  3.1), a launch of `B` equal blocks finishes in about
  `ceil(B / S) W / (B r)`, with `W` the flops, `r` the flop rate of one
  block, and `S` the SMs. That FIT to this table, `r` taken from proj_fwd
  alone (76.5 GFLOP/s per block), gives proj_dB 0.877 ms (measured 0.887),
  gateup_dA 0.877 (0.906), gateup_fwd at 256 blocks, 2 rounds, 0.658
  (0.671), head_dA 21.5 (22.0), head_fwd at 48 rounds 15.8 (16.6), head_dB
  at 18 rounds 15.8 (16.7). With the quarter tile's per-block efficiency
  fitted from head_fwd alone (0.58), it predicts quarter proj_dB at 144
  blocks, 2 rounds, 0.756 ms (measured 0.736) and quarter proj_fwd at 384
  blocks, 3 rounds, 0.425 ms (0.436). A rate simply proportional to
  `min(B, S)` with no rounds predicts quarter proj_dB at 0.41 ms and does
  not fit. This is a fit to price lines, not a timer, and it is the working
  explanation, not a measured cause.
- **E2, the TN staging mapping.** A TN call takes the outer-contiguous
  mapping of `_tuned_g2r` for BOTH operands (8 scalar loads per thread per
  operand per window) where NT takes two 4-wide vector loads. The table
  already sets it against gateup_dB (TN, 96 blocks) at 7.11 TFLOP/s beside
  down_fwd (NT, same shape and blocks) at 7.25, so it can explain at most a
  few percent, not 2.7 times. A box separates it with the CONTROL line
  `ctl_nt_768x768x2048` (NT, 36 blocks, 128 windows): E2 predicts about 7
  TFLOP/s, E1 about 2.7.
- **E3, the leaf count and the window chain.** `P = 16` means 128 windows
  per block, 16 fold pushes and a deeper local stack. The table sets it
  against proj_dA (96 blocks, 48 windows) at 7.19 beside gateup_dA (96
  blocks, 128 windows) at 7.11. CONTROL `ctl_tn_768x768x768` (TN, 36
  blocks, 48 windows): E3 predicts about 7 TFLOP/s, E1 about 2.7.
- **E4, operand footprint and memory locality.** proj_dB's B is 2048 x 768
  where proj_fwd's is 768 x 768. CONTROL `ctl_tn_768x768x768` has the
  smallest operands of all: E4 predicts it faster per flop than proj_dB,
  E1 the same 2.7.
- **E5, a different register or occupancy picture for TN.** Ruled out from
  source: one specialization serves all three ops (3.1), so the resources
  line is the same for every row.
- **E6, a fixed per-call cost (host dispatch, the trial hook's environment
  read, the wait).** proj_fwd and proj_dB have the same `W`; a common
  intercept fitted from the two is about zero. CONTROL lines at equal
  blocks and different `W` (`ctl_nt_768x768x2048` against
  `ctl_tn_768x768x768`) expose any intercept.
- **E1 with rounds against E1 proportional.** CONTROL
  `ctl_nt_1536x1408x768` (132 blocks) against `ctl_nt_1664x1408x768` (143
  blocks): rounds predict the second about 1.85 times slower per flop, a
  proportional rate predicts the same per flop. That pair also reads `S`
  on any board, the AMD column included.

### 3.3 Which calls E1 says are affected

Every call with fewer than `S` blocks: the six 96-block calls at 7.0 to 7.3
TFLOP/s and proj_dB at 36. The 256-block and head calls sit at 9.4 to 9.6
because their block count is at or above `S`, where the fit says there is
at most a rounding loss (256 blocks is 2 rounds against 1.94). So one lever
serves all seven 7-ish calls and the outlier.

## 4. The design (DEVIATIONS 2590 and 2591)

More blocks without touching what a block does. The only axis left is the
one a block walks in sequence, `k`, and the contract fixes how `k` splits:
leaves of `L`, folded by the balanced tree. A block that owns a tile and a
GROUP of `2^g` consecutive leaves, instead of all `P`, runs exactly the
shipped per-window body, walks `8 * 2^g` windows, and hands one tree node
per cell to a workspace. A second launch folds the `G = ceil(P / 2^g)`
nodes per cell with the fold kernels the SPLIT plans already use. The
grid is `(tiles, G)`, so a call issues `tiles * G` blocks.

Predicted from the section 3.2 fit, before any fold cost `F`, and NOT
measured: proj_dB at `G = 16`, 576 blocks, 5 rounds, 0.274 ms against
0.887; the 96-block `P = 16` calls at `G = 8`, 768 blocks, 6 rounds, 0.658
against 0.906; proj_fwd and proj_dA at `G = 6`, 576 blocks, 0.274 against
0.33; head_dA at `G = 7` (six 64-leaf groups and a 9-leaf tail), about
17.5 against 22.0. Weighted, the GEMM sum goes from about 219 ms to about
162 ms plus `F` times about 180 calls. Whether `F` (the workspace, the
node stores, the fold launch, and the allocation per call) eats that is
exactly what the PHASE lines of section 6 print.

**The group rule** (`gemm_step_ksplit_group_leaves`, execution plan only,
reads `m`, `n`, `k` and a column row, returns a leaf count per group):

1. Applies only where `choose_gemm_plan` answers `PLAN_TUNED_128_8X8` and
   `P >= 2`.
2. The finest power-of-two group whose workspace `m n G` fits
   `SPLITK_MAX_WORKSPACE_FLOATS` (64 M floats, the cap `PLAN_SPLITK`
   honors). Declines when fewer than two groups fit.
3. `ksplit_leaf` stops there (at most `P` groups; reads no machine
   number). It is the S-free control and applies to every 128x128 call
   that fits, the 256-block ones included.
4. `ksplit` reads `S = lib_gemm_block_parallelism_for[TARGET_COLUMN]()`
   (a new kernel matrix row, NVIDIA 132, every other column 0 meaning no
   reading). With `S > 0` it declines calls with `tiles >= S`, then
   coarsens the group while the coarser split still issues at least `4 S`
   blocks, so the rounding loss stays within a quarter of the best a
   `tiles / S` bound allows. With `S = 0` it takes the finest split, like
   `ksplit_leaf`, but still only on calls under the cap.

At the step's calls, counted by hand from the rule: `ksplit` takes proj_fwd
and proj_dA at `G = 6` (1 leaf per group), proj_dB at `G = 16` (1 leaf),
the four 96-block `P = 16` calls at `G = 8` (2 leaves), head_dA at `G = 7`
(64 leaves); gateup_fwd, down_dA and the head forward and dB stay
shipped. `ksplit_leaf` takes the same calls at `G = P` (head_dA at `G = 25`,
16 leaves, the finest that fits the cap) and also gateup_fwd and down_dA
at `G = 6`. head_fwd (103 M cells) and head_dB (38.6 M cells) fit no split
under the cap and stay shipped under both arms.

## 5. The identity argument (written before the code)

Notation of the predecessor brief's section 5: for output cell `(i, j)`,
`(L, P) = contract_partition(k)`; leaf `t` is the accumulator seeded `+0.0`
and stepped by `_tuned_step` for `p` ascending over
`[t L, min((t + 1) L, k))` on operands flushed as loaded; the leaf partial
is `ftz(acc)`; partials enter the stack by `_fold_push_local`'s merge
`ftz(ftz(slot_d) + ftz(val))`, earlier leaves on the left; the root is
drained lowest level first by `_fold_drain_local`; the stored cell is
`ftz(root)`. `check_stack_fold_is_the_contract_tree` and
`check_tile_fold_is_the_contract_tree` in `gemm_device_check.mojo` prove
the push and drain spellings equal to `gemm_oracle.fold_balanced_tree`.

### 5.1 The leaf partials are the shipped ones, bit for bit

The group kernel (`identical_gemm_ksplit_kernel`) is the non-LFOLD path of
`identical_gemm_step_arm_kernel`, which is the non-SPLIT path of
`identical_gemm_tuned_kernel`, at the shipped geometry: `RPT = CPT = 8`,
`TC = 16`, `KS = 16`, `PAGES` from `lib_smem_pages_for` at the same page
bytes `_launch_tuned` computes. Copied unchanged: the prologue, both
`_tuned_g2r` mappings, the register to shared copy, the barrier, the
prefetch, both accumulate paths through `_tuned_loaded_operand` and
`_tuned_step`, the `PAGES == 1` barrier, the leaf boundary test and the
partial `ftz(acc)` pushed by `_fold_push_local`. The ONLY change in the
window loop is its range. Block `(tile, q)` starts at flat window
`w = q 2^g wpl` and stops before `w_end = min((q + 1) 2^g, P) wpl`, with
`wpl = _tuned_windows_per_leaf[KS](L)`. `_tuned_window` maps flat window `w` to
fold position `w // wpl`, so the block walks leaves `q 2^g` through
`min((q + 1) 2^g, P) - 1`, each with all of its windows, in ascending
order, and its accumulator is reset at each leaf boundary exactly as the
shipped loop resets it. `leaf_in` and `p_in` come from
`contract_partition(k)`; the group size is never passed to
`_tuned_window` and cannot reach a leaf boundary. So each leaf partial of
each cell is the shipped partial, from the same operands, in the same
step order.

### 5.2 A full group stores a node of the contract tree

Lemma A. Push `2^g` consecutive leaf partials into an empty stack. The
binary counter reaches `occ = 2^g`: one slot, level `g`, holds
`node(g, q)`, the balanced subtree over those leaves, and
`_fold_drain_local` returns that slot bit for bit (one occupied slot, no
addition). In the shipped kernel the same leaves enter a stack whose low
`g` bits of `occ` are zero when leaf `q 2^g` arrives (every earlier full
group has merged up past level `g - 1`), so the merges among these leaves
are the same expressions on the same operands in the same order, and the
value that reaches level `g` is the same `node(g, q)`.

### 5.3 A short tail group stores the remainder the drain would build

Lemma B. Let the last group hold `r < 2^g` leaves. In the shipped stack,
after all `P` pushes, the slots below level `g` are occupied exactly by
the tail's own merges (full groups leave the low bits zero, and `r < 2^g`
never carries into level `g`). `_fold_drain_local` walks levels ascending,
so after it has passed level `g - 1` its accumulator holds the value the
tail group's own drain returns, with `have` set. The group kernel stores
that value.

### 5.4 The fold of the group values is the contract tree

Lemma C. Push the `G` stored values, full groups ascending and the tail
(if any) last, into a fresh stack, and drain. For the full groups the
upper stack's level `d` is the shipped stack's level `d + g` by Lemma A
and induction on the merges (the same pairs merge, slot on the left). If
there is no tail, the upper drain is the shipped drain of levels `g` and
up. If there is a tail `T`, the shipped drain continues from `acc = T`
(Lemma B) over levels `g, g+1, ...` with `acc = ftz(ftz(slot) + ftz(acc))`
at every occupied level. Pushing `T` as the last value merges it with the
consecutive occupied low levels of the upper stack, slot on the left,
which is the same sequence of expressions, then places the result at the
first clear level, and the drain continues with the remaining occupied
levels in ascending order. Same operands, same order. The stored cell is
`ftz(root)` in both.

`identical_gemm_fold_stack_kernel` is that push and drain (`_fold_push`,
`_fold_drain`, register spelling, proven equal to
`fold_balanced_tree`). `identical_gemm_fold_kernel[True]` is the level-wise
`fold_balanced_tree` over its `P` inputs, proven equal to the push and
drain over the same inputs. Both are shipped, both take the count as their
`p_in` and nothing else from `k`, and the launch picks between them at
`SPLIT_BLOCK_FOLD_MAX_CELLS` exactly as `_launch_split` does. Passing `G`
as their count folds the `G` values by Lemma C.

Every value that enters the upper fold is already flushed (a partial is
`ftz(acc)`, a merge is `ftz(...)`, a carry inherits), so the extra
`ftz` on read in either fold kernel is the identity on these bits, as it
is on the SPLIT plans' partials.

Lemmas A to C are also checked on the host, exhaustively, before any
device run: `check_group_fold_is_the_contract_tree` (section 6) builds the
group values with the device's own `_fold_push_local` and
`_fold_drain_local`, folds them with `_fold_push` and `_fold_drain` and
with `fold_balanced_tree`, and requires the bits of
`fold_balanced_tree` over all `P` partials, for every `P` in 1 to 1,100
and every power-of-two group size from 1 to at least `2 P`.

### 5.5 What may vary and why it cannot move a bit

Execution plan only: the group size `2^g`, the grid `(tiles, G)`, which
fold kernel runs, whether the workspace is allocated per call, and the
column row `S`. None reaches `leaf_in`, `p_in`, a window's `p0` or
`chunk`, or a tree level: the group size selects which leaves a block
walks, never where a leaf starts, and Lemma C holds for every `g` and
every `P`, so a vendor that reads a different `S` picks a different `g`
and stores the same bits. Groups MUST be powers of two aligned at leaf 0;
`_ksplit_resolve_leaves` raises on any other size (a size at or above `P`
is one group, which is the whole tree and trivially exact).

Ownership, masking and barriers are the shipped kernel's: every thread of
block `(tile, q)` shares `q`, so every thread walks the same windows and
reaches every barrier; a block with `raw >= tiles` or `q >= G` returns
before any barrier (block uniform); a thread whose cell is outside the
output still stages and barriers, and only its store is masked.

`k == 0` (`P = 0`) takes no group launch: `identical_gemm_step_arm_kernel`
at the shipped geometry with `LFOLD = False` stores `+0.0` per cell
(contract section 8), the code its section 10.1 argues. `P = 1` is one
group of one leaf.

### 5.6 Reach, and the shipped path

`SAB = True` stores `1.0e30` in place of the node of ONE cell per block,
the cell of thread `q mod 256` at register cell `(q // 256) // 8,
(q // 256) mod 8`, so blocks `(tile, q)` of one tile sabotage distinct
cells for every `q < 16,384` (the ownership map is a bijection). `1.0e30`
dominates every sum these operands reach and is finite, so every
sabotaged node moves its output cell after the fold, and a sabotage
launch moves exactly the number of `(tile, q)` whose cell lies inside the
output (`gemm_step_ksplit_reach`, which also names the group count that
ran). On a build without `-D MOJOLEARN_GEMM_ARM_TRIAL=1`,
`identical_gemm_step_ksplit_into` runs the shipped plan, the sabotage
moves nothing, and the check fails saying so.

The shipped dispatch line in `identical_gemm_into` is not touched; the
hook in front of it is still the only reference to the arm section and
still compiles only under the trial define. The new kernel matrix row is
read only by the arm section.

## 6. State at wind-down (same day; Andrew asked every lane to stop)

SUPERSEDED by section 9, which records what the second session built. Kept
as the first session's record.

The lane was stopped after sections 1 to 5 and before the kernel. What
exists on the branch, all unbuilt:

- **DEVIATION 2591, row only.** `checks/kernel_matrix.mojo::lib_gemm_block_parallelism_for`
  (NVIDIA 132, every other column 0). Read by nothing yet. The import of it
  into `gemm/checks/gemm_identical.mojo` is in place and unused.
- **DEVIATION 2593, control calls only.** `gemm/checks/gemm_step_arms.mojo`
  gains `GEMM_STEP_CONTROL_CALLS`, `gemm_step_control_call_name` and
  `gemm_step_control_call` (the six CONTROL shapes of section 3.2), and its
  oracle import now names `OP_NN` and `OP_TN` beside `OP_NT`. No harness
  reads them yet.

**NOT BUILT (the whole arm):**

- 2590. `identical_gemm_ksplit_kernel[RPT, CPT, TC, KS, PAGES, SAB]` (the
  group kernel of 5.1 to 5.3), `_ksplit_groups_launch`,
  `_ksplit_fold_launch` (the `_launch_split` fold dispatch with `G` as the
  count), `_ksplit_resolve_leaves` (raises on a non-power-of-two group),
  `identical_gemm_step_ksplit_into` (allocates `m n G`, launches both,
  synchronizes; shipped plan on a non-trial build),
  `identical_gemm_step_ksplit_phase_into` and
  `identical_gemm_step_ksplit_workspace_floats`.
- 2591. Arms `ksplit` and `ksplit_leaf` (`GEMM_ARM_KSPLIT = 7`,
  `GEMM_ARM_KSPLIT_LEAF = 8`, geometries 7 and 8, `GEMM_ARM_COUNT` and
  `GEMM_GEOM_COUNT` 9), their parse, name, tile and geometry-name entries,
  `gemm_step_ksplit_group_leaves` (the section 4 rule with
  `GEMM_KSPLIT_SLACK = 4`), `gemm_step_ksplit_reach`,
  `gemm_step_geometry_reach` and `gemm_step_geometry_launched_blocks`, and
  the ksplit dispatch inside `identical_gemm_step_geometry_into`'s trial
  block.
- 2592. In `gemm/checks/gemm_step_arms_check.mojo`: the host check
  `check_group_fold_is_the_contract_tree` (section 5.4); expected reach
  from `gemm_step_geometry_reach` in the ragged and LM parts instead of
  `gemm_step_geometry_blocks`; a `RUN_KSPLIT` ragged loop over explicit
  group sizes {1, 2, 4, 16, 64}; `ksplit` and `ksplit_leaf` in `_arm_names`.
- 2593. `bench/gemm_step_price_main.mojo`: the CONTROL lines under
  `MOJOLEARN_GEMM_STEP_CONTROLS=1`, PHASE lines (group launch and fold
  launch timed apart, with a PHASEBITS equality pass), launched blocks
  and group leaves on PRICE and TABLE. `bench/gemm_step_resources_main.mojo`:
  a `ksplit_128x128` row.
- 2594. `tools/gemm_longk_leg.sh` (the POSIX wrapper: exports
  `MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,ksplit,ksplit_leaf`,
  `_LM_ARMS=ksplit,ksplit_leaf`, `_CHECK_ARMS=shipped,ksplit,ksplit_leaf`,
  `MOJOLEARN_GEMM_STEP_CONTROLS=1`, `_LEG_OUT=/root/gemm_leg_out/gemm-longk`,
  then `cd /root/mojolearn && exec sh tools/gemm_step_leg.sh`).

## 7. RUN OWED as it stood at wind-down

SUPERSEDED by sections 9.4 and 9.5. Kept as the first session's record.

Nothing to time: no arm exists. The two edits that do exist can only be
checked by a build, on the M4, run by the orchestrator:

1. `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . gemm/checks/gemm_step_arms_check.mojo -o /tmp/gemm-step-check`
   then `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000 /tmp/gemm-step-check`.
   Expect the 2540 to 2543 PASS unchanged (this lane changed no arm).
2. `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo -o /tmp/gemm-device-check && /tmp/gemm-device-check`
   stays green (the only shipped-file edits are an unused import and a new
   kernel matrix function).

The H100 leg is NOT READY. When the arm is built, the command is to be:

```sh
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=sm_90a \
MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_longk_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-gemm-longk \
sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" --local-card /tmp/gemm-longk-apple.card
```

with `/tmp/gemm-longk-apple.card` from `tools/gemm_card.sh device /tmp/gemm-longk-apple.card`
at the commit that carries the arm. Flip rule: geometric mean of the
enwik8 and Pile GitHub lean step ratios below 1, every step witness equal.

## 8. Risks only a build or a box settles

- The section 3.2 model is a fit to one price table; the CONTROL lines
  test it, and the rounds term (132 against 143 blocks) is its sharpest
  prediction.
- The fold launch, the node stores and a per-call workspace allocation
  (up to 100 MB at gateup_dA under `ksplit_leaf`, 24 times a step) are
  the cost `F` of section 4; nothing in the repo prices them at these
  output sizes. The PHASE lines were designed to.
- A 14-argument kernel and a `(tiles, G)` grid are untried on HIP and
  Metal.

## 9. Build lane (September 11, 2026, second session, worktree)

Source only. Nothing below was compiled, run or timed, on the Mac or on a
GPU. It builds sections 4 and 5 as designed; where the code departs from
section 6's NOT BUILT list, 9.2 says so. The identity argument is section 5,
unchanged; 9.1 names the lines it binds to.

### 9.1 The identity argument, bound to the code

- **The leaf partials (5.1).**
  `gemm/checks/gemm_identical.mojo::identical_gemm_ksplit_kernel[RPT, CPT,
  TC, KS, PAGES, SAB]` is `identical_gemm_step_arm_kernel` with `LFOLD =
  False`, line for line, except for three things. The block reads
  `raw = block_idx.x` and `q = block_idx.y`. The window loop runs from
  `lbeg * wpl` to `lend * wpl`, with `lbeg = q * gleaves` and
  `lend = min(lbeg + gleaves, P)`. The store is described next.
  `leaf_in` and `p_in` come from `contract_partition(k)` in
  `identical_gemm_step_ksplit_into`. `gleaves` never reaches
  `_tuned_window`.
- **The group node (Lemmas A and B).** Each block pushes its own leaves into
  a fresh `_fold_push_local` stack and stores `_fold_drain_local`'s value
  unflushed at `ws[q * m * n + cell]` for in-range cells. That is the SPLIT
  plans' leaf-major layout.
- **The fold of the nodes (Lemma C).** `_ksplit_fold_launch` is
  `_launch_split`'s fold dispatch with `G` as the count:
  `identical_gemm_fold_kernel[True]` at `m n <= SPLIT_BLOCK_FOLD_MAX_CELLS`,
  else `identical_gemm_fold_stack_kernel`. Both store `ftz(root)`.
- **Groups are powers of two aligned at leaf 0.** `_ksplit_resolve_leaves`
  raises on any other size, and on a size above `2^20`, because the size
  travels as an Int32.
- **`k == 0`** takes `_launch_step_arm[128x128, LFOLD = False]` and no group
  launch.
- **Barriers.** The two early returns (`raw >= n_tiles or q >= groups`, and
  an empty group) read only block-uniform values and come before any
  barrier. Every thread of block `(tile, q)` walks the same window range.
- **Reach (5.6).** `SAB = True` stores `1.0e30` in place of the node of the
  cell of thread `q mod 256`, register cell `q // 256`, when that cell is in
  the output. `gemm_step_ksplit_reach` counts exactly those cells, so it
  names the group count. At `k == 0` the step arm kernel's sabotage moves
  one cell per tile, and the reach says so.
- **Checked on the host.** Lemmas A to C are checked exhaustively by
  `check_group_fold_is_the_contract_tree` (9.3), with the device's own
  push, drain and fold functions.

### 9.2 What was built, per deviation

- **2590 BUILT** (`gemm/checks/gemm_identical.mojo`, the section "THE
  LONG-K GROUP ARMS" after `_step_geometry_launch`):
  - the group kernel of 9.1;
  - `_ksplit_groups_launch[RPT, CPT, TC, KS, SAB]`, with PAGES from
    `lib_smem_pages_for` at `_launch_tuned`'s page bytes, a comptime page
    fit assert, and grid `(tiles, G, 1)`;
  - `_ksplit_fold_launch` and `_ksplit_resolve_leaves`;
  - `identical_gemm_step_ksplit_into(ctx, c, a, b, ws, m, n, k, op,
    group_leaves, sabotage)`, which allocates `m n G` floats, synchronizes,
    launches both, synchronizes again, and keeps the buffer past the wait.
    On a non-trial build it runs PLAN_TUNED_128_8X8 and synchronizes;
  - `identical_gemm_step_ksplit_phase_into` (returns `(alloc_ns, group_ns,
    fold_ns)`, each phase host-synchronized);
  - `identical_gemm_step_ksplit_workspace_floats`.
  The shipped file gained one import (`perf_counter_ns`). The shipped
  dispatch line in `identical_gemm_into` is untouched, and the trial hook
  is still the only way in.
- **2591 BUILT.**
  - Ids: `GEMM_ARM_KSPLIT = 7`, `GEMM_ARM_KSPLIT_LEAF = 8`, geometries 7
    and 8, and both counts at 9.
  - Wiring: parse, name, tile (128x128) and geometry name entries
    (`_ksplit_geometry_name`, built from the bound constants). The ksplit
    branch sits in `gemm_step_arm_geometry` after the
    `choose_gemm_plan != PLAN_TUNED_128_8X8` test.
  - The rule is `gemm_step_ksplit_rule(m, n, k, s, read_s)`, section 4
    rules 1 to 4, with `GEMM_KSPLIT_SLACK = 4` and
    `gemm_step_ksplit_finest_leaves` for rule 2.
    `gemm_step_ksplit_group_leaves(geom, ...)` binds it: `ksplit` reads
    `GEMM_KSPLIT_S = lib_gemm_block_parallelism_for[TARGET_COLUMN]()`, and
    `ksplit_leaf` reads nothing.
  - Reach and blocks: `gemm_step_ksplit_reach`, `gemm_step_geometry_reach`
    and `gemm_step_geometry_launched_blocks`.
  - `gemm_step_geometry_group_leaves` gives the leaves a FORCED launch uses.
  - The dispatch is in `identical_gemm_step_geometry_into`'s trial block.
  - `checks/kernel_matrix.mojo::lib_gemm_block_parallelism_for`: NVIDIA
    132; **AMD 110 from a reading, not a measurement.** The attention brief
    section 11.1 transcribes 110 CUs (the MI250X) and resident blocks per
    CU as `min(8, 65536 // page bytes)`. The shipped 128x128 block holds two
    20,480 B pages, so one block per CU. The MI300X leg's CONTROL pair
    decides it. Every other column is 0.
- **2592 BUILT** (`gemm/checks/gemm_step_arms_check.mojo`):
  - `check_group_fold_is_the_contract_tree` (section 5.4, on the host):
    - covers every `P` in 1 to 1,100 and every power-of-two group size
      from 1 to the first at or above `2 P`, on 4 cells, with two partial
      kinds (the 13-bit significand generator, and the same with about one
      partial in five replaced by `-0.0`);
    - builds the nodes with `_fold_push_local` and `_fold_drain_local`;
    - folds them with `_fold_push`, `_fold_drain` and `ftz`, and with
      `fold_balanced_tree`;
    - requires both to equal `fold_balanced_tree` over all `P` partials.
  - `check_group_rule_hand_counts` (host): `gemm_step_ksplit_rule` at
    `S = 132`, and with no reading, at the twelve LM calls, against section
    4's hand counts in leaves per group. `ksplit`: 1, 1, 1, 0, 2, 2, 2, 0,
    2, 0, 64, 0. `ksplit_leaf`: 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 16, 0.
  - Ragged part: every geometry 1 to 8 forced, with expected reach from
    `gemm_step_geometry_reach`, plus a `RUN_KSPLIT` loop through
    `identical_gemm_step_ksplit_into` at group sizes {1, 2, 4, 16, 64},
    clean bits and reach per size, with its own REACH lines.
  - `ksplit` and `ksplit_leaf` are in `_arm_names`.
  - LM part: expected reach from `gemm_step_geometry_reach`, and each LM
    line prints group leaves and launched blocks.
- **2593 BUILT.**
  - `bench/gemm_step_price_main.mojo`: the per-call body is `_price_call`.
    LM calls keep their salts, and the STEP line format is unchanged (the
    step leg's auto pick parses it).
  - Under `MOJOLEARN_GEMM_STEP_CONTROLS=1`, the six `ctl_*` calls each
    print BITS, SAMPLE and one CONTROL line, never weighted into STEP.
  - Where the arm's geometry is ksplit: a PHASEBITS equality pass, then one
    PHASE line with the median allocation, group launch and fold launch.
  - PRICE gains `arm_launched_blocks=` and `group_leaves=` before `plan=[`.
    TABLE gains two trailing columns.
  - `bench/gemm_step_resources_main.mojo`: a `ksplit_128x128` row (the
    group kernel, clean, at the shipped geometry).
- **2594 BUILT.** `tools/gemm_longk_leg.sh` passes `sh -n` and `dash -n`
  (a parse, not a run). It exports:
  - `MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,ksplit,ksplit_leaf`;
  - `MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=ksplit,ksplit_leaf`;
  - `MOJOLEARN_GEMM_STEP_LEG_LMTIMING=1`;
  - `MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS=shipped,ksplit,ksplit_leaf`;
  - `MOJOLEARN_GEMM_STEP_CONTROLS=1`;
  - `MOJOLEARN_GEMM_STEP_LEG_OUT=/root/gemm_leg_out/gemm-longk`.

  A value the runner already exported wins. The wrapper writes
  `longk.txt`, refuses an LM arm list that names only `shipped` (the step
  leg drops it as the bracket, so it would resolve to none), changes into
  the source root and execs `sh tools/gemm_step_leg.sh`.
  `tools/gemm_step_leg.sh` changes by one line: `price_tables.txt` also
  collects CONTROL, PHASE and PHASEBITS lines.

Departures from section 6's list, and decisions it left open:

- **A forced ksplit launch at a shape the rule declines** uses the finest
  group under the cap (`gemm_step_geometry_group_leaves`), and 1 at
  `k == 0`. It raises when the output alone exceeds the cap. The ragged
  part forces ksplit geometries at shapes the rule declines (every ragged
  output under 128 K cells), and this keeps the group kernel under test
  there. The hook never forces: `gemm_step_arm_geometry` returns shipped
  wherever the rule declines.
- **The rule takes `S` as an argument** so that a host check can hold it to
  the hand counts on every column. `check_group_rule_hand_counts` is new;
  section 6 did not list it.
- **`identical_gemm_step_ksplit_phase_into` takes the caller's `ws`**, like
  the other entries, so its non-trial fallback can run the shipped plan.
- **The leg wrapper also exports** `_CHECK_ARMS`, `CONTROLS` and `_LEG_OUT`,
  the three knobs section 6 listed beyond the task's three.
- **No other arm, kernel or default changed.**

### 9.3 What the check proves when it passes, and what it cannot

- **When it passes on a trial build:**
  - the group fold is the contract tree for every `P` the profile allows
    and every group size;
  - the rule gives the section 4 hand counts;
  - every forced ksplit launch and every explicit group size stores the
    shipped plan's bits and FLAT's, on all three ops, ragged `m x n`,
    `k` in {0, 1, 128, 129, 300, 1000, 2049, 50257} and the subnormal
    kind;
  - every sabotage moves exactly its reach.
- **On a GPU box** the LM part adds the twelve target-shape calls through
  `identical_gemm_into` under `shipped`, `ksplit` and `ksplit_leaf`.
- **What it cannot prove:** the M4 run with `MOJOLEARN_GEMM_STEP_CHECK_LM=0`
  reaches only one shape where the rule applies: 257x520 at `k` of 129
  (`P = 2`) and 300 (`P = 3`). At `k` of 1000 that shape is 134 M flops,
  over the 50 M budget. The box's 400 M budget adds `k` of 1000 and 2049.
  So the LM calls on the box are the first applicable launches at step
  shapes.

### 9.4 RUN OWED, M4, light, run by the orchestrator one at a time

1. The gate on a trial build (the host fold check and the rule check run
   first in the same binary):
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . gemm/checks/gemm_step_arms_check.mojo -o /tmp/gemm-step-check`
   then
   `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000 /tmp/gemm-step-check`.
   Expect:
   - `check_group_fold_is_the_contract_tree: ... 0 disagree`;
   - twelve RULE lines and `check_group_rule_hand_counts: 0 failures`;
   - every `REACH ragged [...]` line at N/N for all eight geometries,
     `ksplit` and `ksplit_leaf` included, and for each of the five
     `ksplit group=` sizes;
   - PASS.
2. The same check on a build WITHOUT the trial define:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_step_arms_check.mojo -o /tmp/gemm-step-check-notrial`
   then
   `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000 /tmp/gemm-step-check-notrial`.
   Expect FAIL, and failure lines naming the missing define ("this build
   lacks -D MOJOLEARN_GEMM_ARM_TRIAL=1"). The two host checks still pass
   there.
3. The shipped path:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo -o /tmp/gemm-device-check && /tmp/gemm-device-check`
   stays green. Shipped-file edits: one import, the arm section, and the
   AMD value of a row the shipped build reads nowhere.
4. Builds only, no run:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_price_main.mojo -o /tmp/gemm-step-price`
   and
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_resources_main.mojo -o /tmp/gemm-step-resources`.

### 9.5 The legs

Both legs start from a `git worktree add --detach` checkout of the commit
that carries this section, after 9.4 is green.

NVIDIA H100, RunPod. The card comes from
`tools/gemm_card.sh device /tmp/gemm-longk-apple.card`, run at that commit.

```sh
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=sm_90a \
MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_longk_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-gemm-longk \
sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" --local-card /tmp/gemm-longk-apple.card
```

AMD MI300X, Hot Aisle. The runner `tools/hotaisle_leg.sh` belongs to
another lane and was a skeleton that exits 2 when this section was written.
Its intended body interface is `tools/do_extra_leg.sh`'s.

```sh
MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_longk_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-hotaisle-gemm-longk \
bash tools/hotaisle_leg.sh --rent --minutes 60 --skip-gates
```

Read back from `<leg out>/remote/gemm-longk/`:

- `status.tsv`: every item exits 0.
- `step-check.log`: PASS. An OK `LM` line for `shipped`, `ksplit` and
  `ksplit_leaf` on each of the twelve calls. The ksplit reach at head_dA
  is 672 (96 tiles, 7 groups) under `ksplit` on NVIDIA and 2,400 (25
  groups) under `ksplit_leaf`.
- `resources_lines.txt`: the `ksplit_128x128` row against the shipped row.
- `price_tables.txt` holds several readings:
  - PRICE and TABLE per arm;
  - the six CONTROL lines per arm. The shipped run's CONTROL lines are the
    section 3.2 readings: `ctl_nt_768x768x2048` against
    `ctl_tn_768x768x768` for E2, E3 and E4, and `ctl_nt_1536x1408x768`
    against `ctl_nt_1664x1408x768` for rounds and `S`;
  - the PHASE lines (allocation, group launch, fold launch) that price
    `F`.
- `price_step.txt`: the GEMM sum ratio per arm.
- `lm_summary.tsv`: `witnesses_equal_baseline`, `ratio_vs_shipped` and
  one verdict line per arm; `lmtiming-*` holds the component breakdowns.

**The flip rule** (ENGINEERING_RULES 9): the geometric mean of the enwik8
and pilegithub lean step ratios below 1, with every step witness equal to
shipped on both corpora. The verdict line computes it.

On AMD, the shipped run's CONTROL pair also reads `S` for
`lib_gemm_block_parallelism_for[COLUMN_AMD]`. That row follows the reading,
whatever the verdict.

### 9.6 Risks only a build or a box can settle

- **API and syntax never compiled:**
  - a 14-argument kernel on a 2-D grid of `(tiles, G)`;
  - `-Float32(0.0)` as the signed zero in the host check;
  - importing `_value` and the underscore fold helpers into the check (the
    device check does the same for the fold helpers);
  - a `List[Int]` borrowed into `_ragged_case`;
  - the phase function's tuple of `perf_counter_ns` differences.
- **Per-call cost on a trial build.** Every applicable GEMM of the LM step
  allocates its workspace and synchronizes twice inside
  `identical_gemm_into`, about 180 calls per step. That sits in the
  numerator of the LM ratio; the PHASE lines show how much of it is
  allocation. The workspace peaks near 100 MB per call (gateup_dA and the
  other `P = 16` calls under `ksplit_leaf`).
- **Section 3.2's model is a fit.** If the CONTROL lines refute E1, the
  arms have no mechanism to win, and the verdict will say so.
- **Compile time.** A trial build now instantiates the group kernel clean
  and sabotaged, plus both fold kernels, in the check, the price harness
  and the byte LM binding. HIP compile time for them is unknown. The knobs
  are `MOJOLEARN_GEMM_STEP_LEG_SKIP_RESOURCES=1` and a shorter arm list
  through the runner's environment.
- **Lease.** LMTIMING adds six probes to the eight LM probes, and the Pile
  GitHub fetch took 426 s on the last H100 leg. The previous leg used 19
  minutes of pod time; this one is expected longer and has not been timed.
- **AMD arch.** `tools/gemm_step_leg.sh` refuses to run on AMD without
  `MOJOLEARN_GPU_ARCHS`. The Hot Aisle runner must export the MI300X's
  arch; this lane does not know it.
- **The 0.457 s question** (BRIEF_gemm_step section 10.7) still stands: a
  trial binding's step time is not a shipped step time. The ratio is
  trial against trial, so the verdict holds either way.
