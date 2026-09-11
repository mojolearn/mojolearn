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

UPDATE, same day, third session. The H100 leg of 9.5 read `ksplit` FLIP
(lean step geometric mean 0.895, every step witness equal). Section 10
makes `ksplit` the shipped GEMM plan where the kernel matrix row enables it
(NVIDIA 132; AMD 0 until the MI300X leg), DEVIATION 2595, SOURCE ONLY, NOT
BUILT. Sections 5.6 and 9 describe the trial-only state before that flip and
are kept as the record.

UPDATE, same day, fourth session. Section 11 holds two follow-ups to the
flip, SOURCE ONLY. The first is the cause and fix of the M4 crash of the step
arms check built WITHOUT the trial define. The second is the A/B body
`tools/gemm_ksplit_classical_leg.sh`, which times the classical GEMM callers
under the new default against `tuned128`.

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

## 10. The flip, `ksplit` as the shipped GEMM plan where the kernel matrix row enables it (DEVIATION 2595)

September 11, 2026, third session, worktree lane. SOURCE ONLY. Nothing in
this section was built, compiled or run, on the Mac or on a GPU. 10.7 is the
RUN OWED list and 10.8 the two confirmation legs. There is no DEVIATIONS
register file in the repository (DEVIATIONS 2590 to 2594 are recorded only
here), so this section is the record of 2595.

### 10.1 The H100 leg that decides it

Evidence `bench/results/e1g/2026-09-11_152822-nvidia-h100-80gb-hbm3-gemm-longk/remote/gemm-longk/`
(NVIDIA H100 80GB HBM3, the 9.5 command). The numbers below are the
orchestrator's reading of that directory, which was not yet on this branch
when this section was written.

| corpus | shipped lean step s | ksplit lean step s | ratio | step witnesses |
|---|---:|---:|---:|---|
| enwik8 | 0.3833 | 0.3423 | 0.892 | equal on every step |
| Pile GitHub | 0.3836 | 0.3450 | 0.899 | equal on every step |

The geometric mean is 0.895, below 1, with every step witness equal on both
corpora, so ENGINEERING_RULES 9 flips the arm in the same session. The price
harness points the same way. The weighted GEMM sum per step went from
183.6 ms (shipped) to 142.6 ms (ksplit), and proj_dB, the outlier of section
2, from 0.746 ms to 0.265 ms. Every BITS and PHASEBITS line reads EQUAL. The
step check passed on the H100, the twelve LM calls through
`identical_gemm_into` included. Both sides of the ratio ran in one trial
binding on one pod, the only comparison BRIEF_gemm_step section 10.7
allows.

The AMD MI300X verdict (Hot Aisle) is still owed, so AMD stays on the old
plan in the shipped default until that leg reads it.

Not a claim against anyone. The torch step at the target shape is an owed
opponent row (bench/OPPONENT_REFERENCE.md); until it exists these are
internal before and after numbers.

### 10.2 How the dispatch decides

`identical_gemm_into` keeps its FAST branch (not compiled under IDENTICAL)
and its trial hook (compiled only under `-D MOJOLEARN_GEMM_ARM_TRIAL=1`). Its
last line is now `identical_gemm_shipped_into`, which reads one kernel matrix
row and no environment.

- **The switch.** `GEMM_KSPLIT_DEFAULT_S = lib_gemm_block_parallelism_for[TARGET_COLUMN]()`.
  Above 0 (`GEMM_KSPLIT_DEFAULT_ON`) the dispatch is
  `identical_gemm_shipped_at_row_into[False]` at that row. At 0 it is a
  `comptime` branch to the old line,
  `identical_gemm_with_plan(..., choose_gemm_plan(m, n, k))`, and the group
  kernel is not referenced.
- **The body.** `identical_gemm_shipped_at_row_into[SAB]` asks
  `gemm_default_ksplit_leaves_at(m, n, k, row)`, which answers 0 at any row
  at or below 0 and otherwise `gemm_step_ksplit_rule(m, n, k, row, True)`,
  the section 4 group rule (rules 1, 2 and 4, `GEMM_KSPLIT_SLACK = 4`). A
  positive answer runs `_ksplit_run[SAB]` at that group size; 0 runs the old
  line.
- **One body for the arm and the default.** `_ksplit_run[SAB]` is the trial
  arm's `P >= 1` path moved unchanged out of `identical_gemm_step_ksplit_into`
  (allocate `m n G` floats, synchronize, group launch, fold launch,
  synchronize, keep the buffer). The arm now calls it too.
- **The kernel matrix.** `lib_gemm_block_parallelism_for` is NVIDIA 132 (the
  value the H100 leg ran), AMD 0 with a docstring saying the MI300X leg
  decides the AMD value, every other column 0. The `ksplit` trial arm reads
  a second row, `lib_gemm_block_parallelism_trial_for`: the shipped row where
  it is above 0, AMD 110 (the section 9.2 reading), every other column 0. So
  the MI300X leg can still force `ksplit` at 110 while the AMD default is
  off. `gemm/checks/gemm_identical.mojo` reads no vendor name; both values
  live in the matrix.
- **Same calls, same groups as the H100 arm.** On NVIDIA the default and the
  trial arm read the same `S = 132`, so the default takes the calls and group
  sizes the leg timed (the section 4 hand counts). proj_fwd, proj_dA and
  proj_dB run at 1 leaf per group, the four `P = 16` 96-block calls at 2,
  head_dA at 64. gateup_fwd, down_dA, head_fwd and head_dB stay on the old
  plan. `check_group_rule_hand_counts` holds the default's rule to those
  counts at row 132 and to 0 at row 0.
- **A synchronize inside the entry.** Where the default takes a call,
  `identical_gemm_into` allocates the node workspace and synchronizes before
  it returns, and its docstring now says so. The wait is ordered on the
  caller's stream, so no caller's result moves. The H100 step time already
  carried the same allocation and waits.

### 10.3 How the old plan stays reachable

- `identical_gemm_with_plan(..., PLAN_TUNED_128_8X8)` is untouched, and every
  check still takes its reference from it by name.
- **The trial arm `tuned128`** (`GEMM_ARM_TUNED128 = 9`,
  `GEMM_GEOM_TUNED128 = 9`, both counts now 10) runs PLAN_TUNED_128_8X8 on
  every call `choose_gemm_plan` sends there. Other calls fall through to the
  shipped dispatch. The forced geometry runs the old plan on every build;
  through `identical_gemm_into` it needs the trial define like every arm. Its
  reach is 0 (the old plan has no sabotage instantiation), and the ragged
  part checks that per case.
- **The other arms.** `ksplit`, `ksplit_leaf` and the 2540 and 2541 arms are
  unchanged. A call an arm does not take now falls through to the shipped
  DEFAULT, so on NVIDIA the non-head calls of `head` and the calls `ksplit`
  declines run the default. That is what `shipped` means everywhere now.
- **Non-trial fallbacks.** On a build without the trial define,
  `identical_gemm_step_geometry_into`, `identical_gemm_step_ksplit_into` and
  `identical_gemm_step_ksplit_phase_into` run the shipped dispatch rather than
  the old plan by name, so their "the shipped default ran" labels are true.

### 10.4 Labels, so no line can confuse the two plans

- `gemm_step_geometry_name(GEMM_GEOM_SHIPPED)` is now
  `gemm_shipped_plan_summary()`, which says whether the ksplit default is on
  and at which row. `tuned128` names itself the old shipped plan.
- `gemm_shipped_dispatch_name(m, n, k)` names the plan the shipped dispatch
  ran at that call. Where ksplit ran it gives the group size, group count and
  launched blocks; elsewhere the old plan and the reason (row 0, or the rule
  declined).
- `gemm_step_arm_plan_label(arm)` is one line per arm, for example
  `shipped: default=ksplit(S=132) else tuned128` on NVIDIA and
  `shipped: default=tuned128 (block parallelism row 0)` on AMD and Apple.
- **Price harness.**
  - The reference is `identical_gemm_shipped_into`.
  - The header prints
    `DEFAULT gemm column=... block_parallelism_row=... ksplit_default=on|off trial_ksplit_S=... shipped=[...]`
    and `PLANLABEL arm=... label=...`.
  - PRICE and CONTROL gain `shipped_launched_blocks=`,
    `shipped_group_leaves=`, `choose_plan=[...]` and `shipped_plan=[...]`,
    which replace `plan=[...]`.
  - TABLE's shipped plan column is the plan that ran, its shipped blocks
    column the launched blocks, and a trailing column the shipped group
    leaves.
  - PHASE lines carry `phase_of=arm` or `phase_of=shipped_default` (the
    default's own phases on the `shipped` run).
  - `MOJOLEARN_GEMM_STEP_LABEL_ONLY=1` exits after the header, before a
    DeviceContext.
  - The STEP line is unchanged, because the leg's auto pick parses it.
- **Step arms check.** LM lines gain `shipped_plan=[...]` and
  `arm_plan=[...]`. DEFAULT lines carry `plan=[...]` per row.
- **Resources harness.** It gains a
  `GEMM_STEP_RESOURCES_GEOMETRY label=shipped_default` line.
- **`tools/gemm_step_leg.sh`.**
  - `plans.tsv` maps each price and LM arm to its PLANLABEL from the leg's
    own step-price binary, and `gate.txt` gains `shipped_plan=`.
  - Every probe runs with `MOJOLEARN_GEMM_PLAN_LABEL`, and
    `tools/lm_step_memory_probe.py` records it as `gemm_plan` beside
    `gemm_arm` (setup event and result.json).
  - `lm_summary.tsv` prints `gemm_plan=` per run, and every verdict line ends
    with `arm_plan=` and `shipped_plan=`.
  - `price_tables.txt` also collects DEFAULT and PLANLABEL lines.
- **`tools/gemm_longk_leg.sh`.** Its defaults are now the NVIDIA confirmation
  (10.8); the AMD leg narrows them through `MOJOLEARN_HOTAISLE_EXTRA_ENV`.

### 10.5 The checks

- **`gemm/checks/gemm_device_check.mojo`, GATE 4
  `check_device_default_dispatch`** (shipped build, no trial define).
  - Shapes: five over the 128 K-cell floor, where `choose_gemm_plan` answers
    PLAN_TUNED_128_8X8. 257x520x300 NT (P 3), 520x257x129 TN (P 2),
    257x520x129 NN (P 2), 257x520x1000 TN (P 8), and 257x520x128 NT (P 1,
    where the rule declines).
  - Each shape runs four ways, all required bit-identical. FLAT, the old
    plan by name, `identical_gemm_into` at the column's own row, and the
    shipped body at row 132 (`identical_gemm_shipped_at_row_into[False]`).
  - So the M4 exercises the default's body on Metal at four shapes while its
    own entry runs the old plan, and an NVIDIA box exercises the body through
    the entry.
  - Guards raise when the enabled row takes no shape, when a column whose
    row is above 0 takes none through its entry, and when a column whose row
    is 0 takes any.
- **`gemm/checks/gemm_step_arms_check.mojo`, new `check_default_dispatch`**
  (after the ragged part).
  - Shapes: 257x520 and 520x257, all three ops, `k` in {128, 129, 300, 1000}
    under the flop budget.
  - The shipped body at three rows (the column's, 0 and 132), clean and
    sabotaged, against the old plan. Clean moves nothing. Sabotaged moves
    exactly `gemm_step_ksplit_reach` at the default's group size where the
    row takes the call, and nothing where it does not.
  - Then the column's entry. `identical_gemm_shipped_into` clean, and
    `identical_gemm_into` with `MOJOLEARN_GEMM_ARM` unset and
    `MOJOLEARN_GEMM_ARM_SABOTAGE=1`, which the trial hook serves through
    `identical_gemm_shipped_at_row_into[True]` at the column's row. That run
    must move exactly the column's default reach.
  - On a column whose row is above 0 this proves the default's reach through
    the entry, a sabotage that moves exactly the group launch's cells. At row
    0, on any column, the sabotage moves nothing, which proves the old plan
    runs.
  - The same vacuity guards as the device gate. `REACH default` lines
    summarize per row and for the entry.
- **The LM part.** `gemm_step_geometry_reach(GEMM_GEOM_SHIPPED, ...)` is now
  the default's reach. `arm=shipped` expects the group launch's cells on
  NVIDIA (head_dA 672, 96 tiles by 7 groups) and 0 on AMD and Apple;
  `arm=tuned128` expects 0 everywhere.
- **Sabotage stays out of shipped builds.** The sabotaged instantiation of
  the shipped body is referenced only under the trial define (the hook and
  the check's `_run`), so a shipped build never compiles it.

### 10.6 Why every identity card is unchanged

- **Apple.** `tools/gemm_card.sh device` drives
  `bench/gemm_card_main.mojo::_device_product`, which calls
  `identical_gemm_into`. The Apple column's `lib_gemm_block_parallelism_for`
  row is 0, so `GEMM_KSPLIT_DEFAULT_ON` is False and
  `identical_gemm_shipped_into` compiles to the old line. Not one launch of
  the card changes, and the card must be byte-identical to one made at the
  parent commit.
- **AMD.** The row is 0, so the same argument holds and the card is unchanged
  launch for launch.
- **NVIDIA.** Card shapes the rule takes now run the group kernel and its
  fold instead of the TUNED 128x128 kernel. By section 5 (Lemmas A to C,
  checked exhaustively on the host by `check_group_fold_is_the_contract_tree`)
  every stored cell is the same bits, so the card is unchanged. The H100
  leg's `--local-card` diff against the Apple card at this commit confirms
  it, and GATE 4 checks the bits on the box first.

### 10.7 RUN OWED, M4, light, run by the orchestrator one at a time

1. **The shipped device gates**, no trial define:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo -o /tmp/gemm-device-check && /tmp/gemm-device-check`.
   Expect `gemm device kernel + gates: all green [IDENTICAL]  (8 gates, sabotage: none)`
   and `check_device_default_dispatch [IDENTICAL] OK: 5 shapes bit-identical to FLAT and the old plan; the shipped entry took ksplit at 0 (column row 0), the enabled row at 4`.
2. **The trial step check**:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . gemm/checks/gemm_step_arms_check.mojo -o /tmp/gemm-step-check`
   then
   `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000 /tmp/gemm-step-check`.
   Expect:
   - the 9.4 lines unchanged;
   - twelve RULE lines with `default(row=132)` equal to `ksplit(S=132)`,
     `default(row=0)=0` and `default(column row=0)=0`;
   - `REACH ragged [tuned128 (the OLD shipped plan, forced) ...] N/N`;
   - `REACH default [row=0 (this column's row)] 18/18` (took 0),
     `REACH default [row=0 (off)] 18/18` (took 0),
     `REACH default [row=132 (enabled, the H100 row)] 18/18` (the ksplit body
     took 12, every `P >= 2` case), and
     `REACH default [shipped entry, column row=0] 18/18` (ksplit took 0);
   - PASS.
3. **The same check without the trial define**:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_step_arms_check.mojo -o /tmp/gemm-step-check-notrial`
   then
   `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000 /tmp/gemm-step-check-notrial`.
   Expect FAIL, with failures naming the missing define. The host checks
   pass. In the default part only the row-132 sabotage runs fail (12 cases);
   row 0, the column row and the entry pass.
4. **The price harness build** (no timing run):
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_price_main.mojo -o /tmp/gemm-step-price`.
   Optional, no device work (it exits before a DeviceContext):
   `MOJOLEARN_GEMM_STEP_LABEL_ONLY=1 MOJOLEARN_GEMM_ARM=tuned128 /tmp/gemm-step-price`
   prints `DEFAULT gemm column=... block_parallelism_row=0 ksplit_default=off trial_ksplit_S=0 ...`
   and a `PLANLABEL arm=tuned128` line.
5. **The resources harness build** (no run):
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_resources_main.mojo -o /tmp/gemm-step-resources`.
6. **The byte LM binding, build only** (no import, no step), shipped
   defines:
   `MOJOLEARN_NUMERIC_MODE=identical python3 tools/macos_serial_guard.py --seconds 900 --rss-gib 12 -- sh bindings/build_byte_lm.sh`.
   On the built `_mojolearn_byte_lm.so`, `nm -C <binding> | grep -c ksplit`
   should print 0 (the Apple row compiles the old line).

### 10.8 The two confirmation legs

Both start from a `git worktree add --detach` checkout of the merge commit
that carries this section, after 10.7 is green.

**NVIDIA H100, RunPod: LM `tuned128` against the new default.** RunPod passes
no extra environment to the body, so `tools/gemm_longk_leg.sh`'s defaults are
this leg: price arms `shipped,tuned128,ksplit,ksplit_leaf`, LM arms
`tuned128`, check arms `shipped,tuned128,ksplit,ksplit_leaf`, CONTROL lines
and LM timing on. The Apple card comes first, at that commit:
`tools/gemm_card.sh device /tmp/gemm-ksplit-default-apple.card`.

```sh
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=sm_90a \
MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_longk_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-gemm-ksplit-default \
sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" --local-card /tmp/gemm-ksplit-default-apple.card
```

Read back:

- `device_check.log`: 8 gates green, GATE 4 with the shipped entry taking
  ksplit at 4 shapes (column row 132).
- The NVIDIA card equal to the Apple card on every stage.
- `step-check.log`: PASS, with `REACH default [row=132 (this column's row)]`
  and `[shipped entry, column row=132]` at N/N with ksplit taken. LM lines
  `arm=shipped` with the default's reach (head_dA 672) and
  `shipped_plan=[DEFAULT ksplit ...]`, and `arm=tuned128` with reach 0.
- `plans.tsv` and `gate.txt`: `shipped: default=ksplit(S=132) else tuned128`.
- `price_step.txt`: `STEP gemm arm=tuned128` above 1 if the 183.6 against
  142.6 ms reading repeats, and `STEP gemm arm=ksplit` near 1 (on NVIDIA the
  arm and the default are one body at one `S`).
- `lm_summary.tsv`: `verdict tuned128 NO FLIP` with a geometric mean above 1
  is the confirmation. `gemm_plan=` on every row says which plan ran.

**AMD MI300X, Hot Aisle: LM `ksplit` against `shipped`** (the AMD default is
still the old plan):

```sh
MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_longk_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-hotaisle-gemm-ksplit-default \
MOJOLEARN_HOTAISLE_EXTRA_ENV="MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,ksplit,ksplit_leaf MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=ksplit" \
bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates
```

Read back:

- `plans.tsv`: `shipped: default=tuned128 (block parallelism row 0)` and
  `ksplit(S=110) where its rule takes the call, else default=tuned128 ...`.
- `step-check.log`: PASS; `arm=shipped` reach 0 on every LM call.
- The shipped run's CONTROL pair (`ctl_nt_1536x1408x768` against
  `ctl_nt_1664x1408x768`) reads `S` for the MI300X.
- `lm_summary.tsv`: `verdict ksplit ...`.

A FLIP sets the AMD value of `lib_gemm_block_parallelism_for` in the same
session (the trial row then follows it). The value to set is 110, the one
that leg measured. If the CONTROL reading differs from 110, a different value
needs its own leg before it ships.

### 10.9 Risks only a build or a box can settle

- **Syntax never compiled.**
  - `comptime if GEMM_KSPLIT_DEFAULT_ON: ... else: ...` around calls in
    `identical_gemm_shipped_into` and in `identical_gemm_step_geometry_into`.
  - `comptime shipped = lib_gemm_block_parallelism_for[column]()` inside the
    new matrix row.
  - `_ksplit_run[SAB]` taking the callers' `mut` buffers.
  - The `comptime if GEMM_ARM_TRIAL` nested under a runtime `if` in the
    check's `_run`.
- **Every NVIDIA IDENTICAL caller, not only the LM step.** Any GEMM through
  `identical_gemm_into` with at least 128 K output cells, `P >= 2`, fewer than
  132 tiles and a workspace under 64 M floats now takes ksplit. That covers
  the transformer and byte LM, and also GP, kernel methods, Cholesky,
  mixture and Mamba backward where their shapes qualify. Each such call
  allocates up to 256 MB and synchronizes twice. Bits cannot move (section
  5); the time is measured only on the LM step. A smaller NVIDIA card may
  feel the per-call allocations (the 4090 pending-frees hang, DEVIATION
  2520, is the precedent).
- **FAST and DETERMINISTIC builds on NVIDIA** reach the default for the
  calls that fall through the vendor path (OP_TN, `allow_vendor=False`).
  Nothing has measured them there.
- **Compile time and binary size.** Every shipped NVIDIA binary that reaches
  `identical_gemm_into` now instantiates the group kernel (clean). The shipped
  device check also instantiates it on Metal through the row-132 run; the
  trial check already compiled it on the M4, the shipped check had not.
- **Global sabotage builds.** The `-D MOJOLEARN_GEMM_SABOTAGE_*` switches live
  in the tuned and SPLIT kernels, not in the group kernel. On an NVIDIA
  sabotage build, calls the default takes do not carry the sabotage, so GATE
  4 would read OK there while other gates fail. The recorded sabotage
  evidence is Apple's (row 0), so nothing already on file changes.
- **Lease.** The NVIDIA confirmation prices four arms with CONTROL lines and
  probes one LM arm plus the brackets on two corpora, with timing probes.
  The Pile GitHub fetch alone took 426 s on an earlier leg.

## 11. Follow-ups to the flip: the no-trial check crash and the classical caller A/B

September 11, 2026, fourth session, worktree branch `lane/ksplit-followups`
off the 2595 merge (f3705577). SOURCE ONLY. Nothing in this section was
built, compiled, parsed or run, on the Mac or on a GPU. No kernel's
arithmetic and no kernel matrix value changed. No new DEVIATION number is
taken; both jobs follow DEVIATION 2595.

### 11.1 Job 1, the step arms check built without the trial define crashed on the M4

**The reading (orchestrator, Apple M4).** The build was
`pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_step_arms_check.mojo`
(no trial define), run with `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000`.
It printed the host fold check, the RULE table,
`check_group_rule_hand_counts: 0 failures` and the selector line, then
segfaulted inside Metal (IOGPU frames) in `check_ragged_controls`. Under
`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` the same run raised
`Failed to create compute pipeline state (GPU machine code generation): Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPT`.
The trial build passed (reach 102/102). The same no-trial gate at 3612e17d
printed its expected FAIL lines.

**The cause, a hypothesis from source, not yet confirmed by a run.** It is
not a comptime parameter that goes degenerate without the define.
`GEMM_KSPLIT_S` and `GEMM_KSPLIT_DEFAULT_S` read the same rows on both
builds, and every kernel specialization the no-trial build references, the
trial build references too. What changed is WHICH kernel the no-trial ragged
part launches, and INTO WHICH WORKSPACE.

- 2595 changed the non-trial fallback of
  `identical_gemm_step_geometry_into` (every arm geometry but `tuned128`)
  and of `identical_gemm_step_ksplit_into`. Before, it was
  `identical_gemm_with_plan(..., PLAN_TUNED_128_8X8)`. After, it is
  `identical_gemm_shipped_into`, which on the Apple column (row 0) compiles
  to `identical_gemm_with_plan(..., choose_gemm_plan(m, n, k))`.
- `_ragged_case` sizes `dw` as the larger of FLAT's and the old 128x128
  plan's workspace. Both are 0, so `dw` holds ONE float (4 bytes).
- At a ragged output with `P >= 4` whose `m n P` fits 64 M floats,
  `choose_gemm_plan` answers a SPLIT plan. `_launch_split` then writes
  `m n P` partials into `dw` and folds them out of it. Under the M4's 50 M
  flop budget the ragged part reaches these shapes, each on all three ops:

| shape | P | plan | leaf kernel, then fold | floats written into the 1-float `dw` |
|---|---:|---|---|---:|
| 1x3x1000 | 8 | PLAN_SPLIT_16_1X1 | `identical_gemm_tuned_kernel[1, 1, 16, 32, 1, PAGES, True]`, `identical_gemm_fold_kernel[True]` | 24 |
| 1x3x2049 | 17 | PLAN_SPLIT_16_1X1 | same | 51 |
| 1x3x50257 | 393 | PLAN_SPLIT_16_1X1 | same | 1,179 |
| 33x70x1000 | 8 | PLAN_SPLIT_32_2X2 | `identical_gemm_tuned_kernel[2, 2, 16, 32, 1, PAGES, True]`, `identical_gemm_fold_kernel[True]` | 18,480 |
| 33x70x2049 | 17 | PLAN_SPLIT_32_2X2 | same | 39,270 |
| 129x257x1000 (ordinary and subnormal) | 8 | PLAN_SPLIT_64_4X4 | `identical_gemm_tuned_kernel[4, 4, 16, 32, 1, PAGES, True]`, `identical_gemm_fold_stack_kernel` | 265,224 |
| 300x129x1000 | 8 | PLAN_SPLIT_64_4X4 | same | 309,600 |

  Every such case launches the plan 26 times (8 arm geometries and 5 group
  sizes, clean and "sabotaged"). A GPU write of up to 1.2 MB past a 4-byte
  buffer corrupts neighboring GPU-mapped allocations, and a segfault inside
  IOGPU frames fits that.
- The first such launch in the process is 1x3x1000 NN under `lfold`. It is
  also the first pipeline creation of
  `identical_gemm_tuned_kernel[1, 1, 16, 32, 1, PAGES, True]` in the run.
  The XPC error under shader validation is consistent with the validation
  layer's instrumented compile failing at that first new pipeline. It is not
  evidence that Metal cannot build the kernel: the shipped device check
  launches the SPLIT plans on the M4 with a correctly sized workspace.
- The trial build never launches `choose_gemm_plan`'s plan in the ragged
  part. It launches the arm kernels (no workspace), the old plan and FLAT by
  name, and the ksplit body (which allocates its own workspace). At 3612e17d
  the no-trial fallback was the old plan by name, with workspace 0, so the
  same run only failed on reach.

**The fix** (`gemm/checks/gemm_step_arms_check.mojo`, the check only). A
build without the trial define now launches nothing it cannot reach.

- `_ragged_case` launches no arm geometry but `tuned128` (PLAN_TUNED_128_8X8
  on every build), and no explicit group size. The references (the old plan
  and FLAT) and `tuned128` still run on every case.
  `check_ragged_controls` fails each of the 8 arm geometries and 5 group
  sizes ONCE, by name, and prints their REACH lines as `0/N cases NOT LAUNCHED`.
- `check_default_dispatch` runs the shipped body CLEAN at every row
  (column row, 0 and 132) and the entry clean, both through
  `identical_gemm_shipped_into` and through `identical_gemm_into`. All must
  store the old plan's bits. No sabotage runs: a clean run that moves nothing
  proves no reach. Each row and the entry fail once for the unproven reach.
- `check_lm_calls` runs the shipped entry clean once per call and fails each
  selected arm once.
- `_trial_hint()` still starts with "this build lacks -D MOJOLEARN_GEMM_ARM_TRIAL=1".
- The trial build's launches, lines and verdict are unchanged.
- `gemm/checks/gemm_identical.mojo` changes in docstrings only. They record
  that the non-trial fallbacks of `identical_gemm_step_geometry_into` and
  `identical_gemm_step_ksplit_into` read `ws`, so a caller must size it with
  `identical_gemm_workspace_max_floats`. The price harness already does.

**Expected no-trial output** (M4, `MOJOLEARN_GEMM_STEP_CHECK_LM=0`, `MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000`):

- the host checks pass as before (0 disagree, 0 failures);
- the selector line, and one selector failure naming the define;
- `REACH ragged [<geometry>] 0/102 cases NOT LAUNCHED (no trial define: the arm kernel is not compiled)`
  for lfold, half, half_ks16, quarter, the two head geometries, ksplit and
  ksplit_leaf. The library's geometry name still ends in "(NOT RUN: ... the
  shipped default ran)"; in this check nothing ran for them;
- `REACH ragged [tuned128 (the OLD shipped plan, forced) ...] 102/102`;
- `REACH ragged [ksplit group=N] 0/102 cases NOT LAUNCHED` for N in 1, 2, 4,
  16 and 64;
- `ragged controls: 102 cases x 9 geometries and 5 explicit group sizes, 8 (m n k) over the 50000000 budget skipped, 13 failures`;
- 18 `DEFAULT ... sabotage=NOT RUN (no trial define) ... CLEAN OK` lines per
  row, and `REACH default [row=0 (this column's row)] 18/18 clean launches stored the old plan's bits; reach NOT PROVEN (no trial define)`.
  The ksplit body took 0 of the cases at row 0 (both lines) and 12 at row 132;
- `REACH default [shipped entry, column row=0] 18/18 clean calls stored the old plan's bits; reach NOT PROVEN (no trial define); ksplit took 0 of them`;
- `default dispatch: 18 cases x 3 rows and the shipped entry, 2 (m n k) over the budget skipped, column row=0 [...], 4 failures`;
- 18 FAIL lines, then the raise `gemm_step_arms_check: 18 failures`: 1
  selector, 13 ragged and 4 default. No segfault.

### 11.2 Job 2, the classical GEMM caller A/B

ENGINEERING_RULES section 9: a shared kernel's flip must hold for every lane
it reaches. On NVIDIA the ksplit default reaches every classical estimator
that calls `identical_gemm_into`.

**When the default can take a call** (`gemm_step_ksplit_rule` at `S = 132`).
Four conditions must all hold:

- `choose_gemm_plan` answers TUNED 128x128. That needs `m >= 128`,
  `n >= 128`, at least 131,072 output cells, and no SPLITK base.
- `P >= 2`, that is `k >= 129`.
- At least two groups fit the 64 M float workspace.
- Fewer than 132 tiles (`ceil(m/128) ceil(n/128)`), which bounds the output
  near 2.1 M cells.

**The classical call sites, read from source** (IDENTICAL on NVIDIA and AMD):

| family | call site | GEMM (op, m x n x k) | default takes it? |
|---|---|---|---|
| OLS (`LinearRegression`) | `glm/estimator.mojo::ols_fit_host` -> `lstsq_eig` -> `core/gemm.mojo::gemm_tn` -> `gemm_tn_identical_v1` | TN d x d x rows | no: d x d is 121 cells (taxi) or 48,400 (Istella-S), under the 128 K floor, for any d up to 362 |
| PCA (`covariance_eigh`) and tSVD | `decomposition/estimator.mojo::pca_fit_host` -> `pca_fit` -> `gemm_tn` | TN d x d x rows | no, same shape |
| GP posterior mean | `gaussian_process/estimator.mojo` | TN n_star x 1 x n_train | no: n = 1 |
| Cholesky trailing update (GP fit, Cholesky, KRR fit) | `cholesky/checks/potrf.mojo` | NT n_trail x n_trail x 32 | no: k = `CHOL_NB_PINNED` = 32, so P = 1 |
| GP Gram and cross covariance | `gaussian_process/checks/kernels.mojo` | elementwise kernels, no GEMM | not reached |
| kernel matrices (SVC kernel rows, KRR and kernel PCA) | `svm/impl/distance/kernel_matrices.mojo::kernel_op` | NT m x n x d | YES where d >= 129, both sides >= 128, at least 131,072 cells and under 132 tiles |
| GMM E-step | `mixture/checks/estep.mojo` | NN rows x d x d | YES at d >= 129 (at d = 220, per-call rows up to 8,320) |
| GMM M-step | `mixture/checks/mstep.mojo` | TN ncomp x d x n, TN d x d x n | no below d = 363 |
| Nystroem | `kernel_methods/estimator.mojo` (697, 851) | NT q x q x q; rows x q x q | YES at q from 363 to 1,408 (the first); q >= 129 with rows under the tile bound (the second) |
| RBFSampler transform | `kernel_methods/estimator.mojo` (1017) | NN rows x D x d | YES at d >= 129, D >= 128, rows under the tile bound |
| KRR predict | `kernel_methods/estimator.mojo` (368) | NN n_query x t x n | only with t >= 128 targets |
| spectral Lanczos | `spectral/.../lanczos.mojo` | k x n x ncv, n x 1 x ... | no: m is the component count or n = 1 |
| kNN, k-means, KDE, QN GLMs | `core/gemm.mojo::gemm_nt` | the pinned NT kernel, NOT `identical_gemm_into` (unless `-D MOJOLEARN_537_GEMM_IDENT_SWAP`) | not reached |
| public linalg GEMM | `gemm/host_entry.mojo` -> `identical_gemm` | any user shape | wherever the rule takes it |

Section 9 declares classical shapes only for kNN (400,000 index rows, 4,000
queries) and for k-means, OLS and PCA (4,000,000 rows). kNN and k-means do
not reach the entry. So OLS and PCA are the only classical callers that
reach `identical_gemm_into` at a section 9 shape, and GP (no declared shape)
completes the set asked for. **At those shapes the rule declines every GEMM
the three issue, on both datasets.** Both arms therefore run the same
launches, and the A/B is expected to read a ratio of 1 within noise with
equal bits. The leg measures that instead of assuming it, and `dispatch.txt`
records the Mojo dispatch's own answer per caller shape.

The families the default CAN take are the kernel matrices, the GMM E-step,
Nystroem, RBFSampler and the public linalg GEMM. None of them has a section 9
shape yet, so their A/B is OWED and needs a declared shape first (a lane
decision, not this brief's).

**The entry points.** `bench/speed/classical_speed_main.mojo` is the
classical lanes' timing driver. It times `ols` and `pca` on a splitmix64
generator (4,000,000 x 32) and `gp` on the 12 x 3 correctness fixture, and
none of its lanes reads taxi or Istella-S. The only real-data classical
harness in the tree is the kNN gate, which runs through a public estimator of
a trial binding. The new driver `tools/gemm_ksplit_classical_ab.py` follows
that gate's pattern:

- It runs the public `LinearRegression`, `PCA(n_components=8, svd_solver="covariance_eigh")`
  and `GaussianProcessRegressor(kernel=RBF(sqrt(d)), alpha=0.1)` of IDENTICAL
  bindings built with `-D MOJOLEARN_GEMM_ARM_TRIAL=1`.
- It prints the FSPEED format of the classical driver and
  `tools/speed_cuml_arm.py`, so `tools/flip_verdict.py` reads its logs as
  they are.
- It loads data through `tools/speed_gbdt_arm.py::load_taxi` and
  `load_istella` (`regression=True`). Taxi takes its 11 `TAXI_NUMERIC`
  columns and the fare target. Istella-S takes its 220 features and the
  grade.
- OLS and PCA take the first 4,000,000 train rows. Istella-S's train split
  holds 2,043,304, so it takes them all.
- GP takes 4,000 train rows (a declared rung of the GP ladder) and 1,000 test
  rows.
- Quality is RMSE (OLS, GP) and reconstruction MSE (PCA) on the harness's
  test rows, outside the timed region.
- `--prep standardize` (the default) applies the training rows' float64
  column mean and deviation to both splits. It is the same bytes for both
  arms, and it is needed: raw Istella-S float32-max sentinels overflow a
  float32 Gram.

Around the driver:

- `bindings/build_estimators.sh` and `bindings/build_gp.sh` gained the
  `MOJOLEARN_BUILD_EXTRA_DEFINES` hook of `bindings/build.sh`. It is an
  empty expansion when unset.
- `bench/gemm_step_price_main.mojo`'s label mode gained one host-only
  DISPATCH line per caller shape (`MOJOLEARN_GEMM_STEP_LABEL_M`, `_N`, `_K`,
  `_CALLER`).

**The record and the verdict.**

- Each timed process is one lane, one dataset, one arm and one block:
  `<lane>.<dataset>.<tuned128|default>.<block>.log`.
- Each log carries `FSPEED-HEADER`, then `FSPEED-NOTE gemm_arm=... gemm_plan=<PLANLABEL>`
  (which plan ran), `FSPEED-GEMM` (the caller's GEMM shapes), one warm-up and
  3 rounds, and `FSPEED-ACC`.
- `hash` is FNV-1a64 over the outputs.
- Blocks run ABBA, 2 per arm.
- `verdict` runs `tools/flip_verdict.py` per caller, with the `tuned128`
  logs as BEFORE and the default logs as AFTER. FLIP there means the default
  is faster.
- Then it prints `caller=<lane> verdict=...`. A caller REGRESSES when the
  geometric mean of its two default/tuned128 median ratios exceeds 1 plus the
  measured noise. The noise is the largest block-to-block spread of either
  arm's block medians on either dataset. A caller also REGRESSES when
  flip_verdict marks a quality metric WORSE. Otherwise it HOLDS.
- A caller whose output hashes differ across arms or blocks is an
  IDENTITY-BREAK. A missing dataset is UNMEASURED.

**How a regressing caller would be fenced out, never with a vendor branch.**
Nothing is fenced without the measurement. There are two ways.

1. **A caller flag.** Add a comptime `allow_ksplit: Bool = True` parameter
   to `identical_gemm_into`, the `allow_vendor` pattern, and thread it to
   `identical_gemm_shipped_into`. There `comptime if GEMM_KSPLIT_DEFAULT_ON and allow_ksplit`
   keeps the default and anything else compiles the old line. The regressing
   caller's call site passes `False`. This fences one caller on every column.
2. **A shape bound in the rule.** Add a bound the measurement names (for
   example a minimum `k`, or a maximum output cells to tiles ratio) to
   `gemm_step_ksplit_rule`, and so to `gemm_default_ksplit_leaves_at`. This
   fences every caller at that shape on every column, and
   `check_group_rule_hand_counts` must still give the section 4 hand counts
   at the LM calls.

Both are execution plan (contract 6.1) and move no bit.

### 11.3 RUN OWED, M4, light, run by the orchestrator one at a time

0. **Optional, the discriminator for 11.1's hypothesis**, on a no-trial
   binary built at f3705577 (before this fix). With
   `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=2000` the
   ragged part reaches only 1x3 at k up to 300, all FLAT and no SPLIT plan.
   Expect no crash; the check still fails, and the default part reports that
   no case ran. With `MOJOLEARN_GEMM_STEP_CHECK_FLOPS=3000000` it reaches
   33x70x1000, where PLAN_SPLIT_32_2X2 writes 18,480 floats into the 1-float
   `dw`. Expect the crash. If the 2000 run also crashes, the hypothesis is
   wrong.
1. **The same check without the trial define, at this commit**:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_step_arms_check.mojo -o /tmp/gemm-step-check-notrial`
   then
   `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000 /tmp/gemm-step-check-notrial`.
   Expect the 11.1 lines, no segfault, and a nonzero exit with
   `gemm_step_arms_check: 18 failures`.
2. **The trial check**:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . gemm/checks/gemm_step_arms_check.mojo -o /tmp/gemm-step-check`
   then
   `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000 /tmp/gemm-step-check`.
   Expect PASS, unchanged from 10.7 item 2 (every ragged REACH at 102/102,
   the default rows at 18/18).
3. **The shipped device gates**:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo -o /tmp/gemm-device-check && /tmp/gemm-device-check`.
   Expect 10.7 item 1 unchanged (gemm_identical.mojo changed in docstrings only).
4. **The price harness, build and host label mode** (no device work):
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_price_main.mojo -o /tmp/gemm-step-price`
   then
   `MOJOLEARN_GEMM_STEP_LABEL_ONLY=1 MOJOLEARN_GEMM_ARM=tuned128 MOJOLEARN_GEMM_STEP_LABEL_CALLER=ols.gram.istella.TN MOJOLEARN_GEMM_STEP_LABEL_M=220 MOJOLEARN_GEMM_STEP_LABEL_N=220 MOJOLEARN_GEMM_STEP_LABEL_K=2043304 /tmp/gemm-step-price`.
   Expect the DEFAULT and PLANLABEL lines, then
   `DISPATCH caller=ols.gram.istella.TN m=220 n=220 k=2043304 arm=tuned128 ksplit_default_takes=no default_leaves_per_group=0 choose_plan=[TILE 16x16 ...] shipped_plan=[...] arm_geometry=[shipped ...]`,
   then `LABEL ONLY`.
5. **The Apple card the leg compares against**, at this commit:
   `tools/gemm_card.sh device /tmp/gemm-ksplit-classical-apple.card`.

The leg body is POSIX sh that nobody has parsed.
`tools/gemm_remote_leg.sh` runs `sh -n` on it before it ships it, so a syntax
error stops the leg before any rental.

### 11.4 The H100 leg for job 2

RunPod passes no extra environment to the body, so the defaults of
`tools/gemm_ksplit_classical_leg.sh` are this leg. Run it from a
`git worktree add --detach` checkout of the merge commit that carries this
section, after 11.3 is green.

```sh
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=sm_90a \
MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_ksplit_classical_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-gemm-ksplit-classical \
sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" --local-card /tmp/gemm-ksplit-classical-apple.card
```

Read back from `<leg out>/remote/ksplit-classical/`:

- `status.tsv`: every build, the two downloads (or `cached`), `smoke`, the
  24 timed items (3 lanes x 2 datasets x 2 arms x 2 blocks), `dispatch` and
  `verdicts`.
- `plans.tsv` and `gate.txt`: `shipped` labeled
  `shipped: default=ksplit(S=132) else tuned128`, and `tuned128` labeled the
  old plan.
- `dispatch.txt`: one DISPATCH line per caller shape per arm. Section 11.2
  predicts `ksplit_default_takes=no` for every line (ols.gram and pca.cov at
  11x11x4000000 and 220x220x2043304, gp.mean at 1000x1x4000,
  gp.chol_trailing at 3968x3968x32). Any `yes` means the default does reach
  the caller, and the verdict is then the measurement that matters.
- `verdicts.log`: flip_verdict's block per caller, then
  `caller=gp|ols|pca verdict=...` and `CLASSICAL KSPLIT A/B gp=... ols=... pca=...`.
  The expected reading is HOLDS for all three, with witnesses equal and ratios
  near 1.

### 11.5 Risks only a build or a box can settle

- **Never compiled or run.** The risky spots are these.
  - A comptime Bool in a runtime `if not GEMM_ARM_TRIAL` with `continue`
    inside loops, in three check functions.
  - The price harness's DISPATCH block.
  - The Python driver: `numpy.asarray` on mojolearn Arrays,
    `GaussianProcessRegressor` accepting `alpha=0.1`, and `fit` returning the
    estimator.
  - The sh body.
- **The crash cause is a hypothesis.** 11.3 item 0 separates it from a Metal
  compiler fault. If the fix still crashes, the next suspect is the row-132
  clean ksplit launch in `check_default_dispatch`, which the trial build also
  runs.
- **Lease.** The cap is 60 minutes (`MINUTES_CAP`). On the 15:28 H100 leg
  the price harness built in 38 s and the base binding in 66 s.
  - The Istella-S download (472 MB plus a decode of minutes) runs in the
    background during the builds.
  - The Istella-S OLS and PCA Grams (220x220x2,043,304, about 99 GFLOP on the
    TILE 16x16 plan) dominate the timing, 16 fits each with warm-ups. Their
    identical throughput at that shape is unmeasured.
  - Taxi runs first for every lane and GP first in each dataset, so a lease
    kill leaves Istella-S's OLS and PCA UNMEASURED rather than the whole leg.
  - The knobs are `MOJOLEARN_CLASSICAL_AB_ROUNDS`, `_BLOCKS` and `_ROWS`, but
    RunPod cannot pass them. Narrowing means editing the defaults in the
    checkout the leg ships.
- **pip in the pixi environment.** numpy and pyarrow are installed with
  `pixi run python -m pip`, as `tools/gemm_step_leg.sh` does for numpy. A
  failure shows in `numpy.log`, `pyarrow.log` and a red `download-taxi` row.
- **Standardization is a prep choice.** It is the same bytes for both arms
  and moves no GEMM shape. With `--prep raw` the Istella-S sentinels make
  float32 Grams overflow, and flip_verdict refuses a non-finite quality.
- **What the verdict can and cannot say.** At these shapes both arms run the
  same launches. A HOLDS here says the default costs these callers nothing;
  it does not price ksplit on classical work. That price is owed on the
  families 11.2 names, once they have section 9 shapes.

## 12. The classical caller A/B on AMD: `tuned128` against the shipped default on the Hot Aisle MI300X

September 11, 2026, fifth session, worktree branch `lane/ksplit-classical-amd`
off origin/main at 190fb7a4 (the AMD row set to 110). The branch merges
`origin/lane/classical-hotaisle-run`, which brings in
`tools/classical_two_datasets.py`. That merge had one conflict, in
`tools/trees_leg.sh`, and took main's side, which already contains the
branch's pod-prefix change. The branch then merged origin/main again at
033fadb1, which carries the classical MI300X rows and the same
`tools/classical_two_datasets.py`. Two coordinator messages shaped this
section. One added kmeans on Istella-S and asked for PCA's on-box DISPATCH
line. The other corrected the first brief: KDE and kmeans may not reach
`identical_gemm_into`, so each caller's path is confirmed from source here,
and the callers that really reach it run first. SOURCE ONLY. Nothing in this section was built,
compiled or run, on the Mac or on a GPU. The syntax checks in 12.5 are the
only commands run. No kernel, no kernel matrix value and no shipped binding
build changed, so no DEVIATION number is taken.

### 12.1 Why

190fb7a4 set the AMD value of `lib_gemm_block_parallelism_for` to 110. The
ksplit default therefore runs on AMD wherever the group rule takes a call
through `identical_gemm_into`, at `S = 110`: fewer than 110 tiles, where
NVIDIA allows 132. Section 11 wrote the H100 A/B for OLS, PCA and GP.
ENGINEERING_RULES 9 requires every lane a shared kernel's flip reaches to
hold, and rule 10 makes AMD the tuning column. The classical lane runs KDE
and SVC on this box on Istella-S at d = 220, above the `k >= 129` floor, so
every number it has for them predates ksplit.

### 12.2 The callers, their shapes, and what the rule predicts

The rule at `S = 110` is section 11.2's with 110 in place of 132:

- `choose_gemm_plan` answers TUNED 128x128: `m >= 128`, `n >= 128`, at least
  131,072 output cells, and no SPLIT plan chosen first.
- `P >= 2`, that is `k >= 129` (`CONTRACT_K_LEAF_MIN = 128`).
- Two groups fit the 64 M float workspace.
- Fewer than 110 tiles of 128x128.

The `tuned128` arm forces the old plan only on calls `choose_gemm_plan` sends
to TUNED 128x128, and every other call falls through to the default. The two
arms of a caller therefore differ exactly on the calls the default takes.
Everything below is read from source at this commit. The DISPATCH lines the
leg writes on the box are the measurement.

The cells run in this order (`MOJOLEARN_CLASSICAL_AB_ORDER`). Each cell is
2 blocks per arm, ABBA.

| order | cells | caller | shape and entry | GEMM through `identical_gemm_into` (op, m x n x k) | `choose_gemm_plan` by this reading | default takes it |
|---|---|---|---|---|---|---|
| 1 | svc:istella, svc:taxi | SVC fit (`C=1`, RBF, `gamma=1/d`) | `svc` block: 10,000 fit rows, 10,000 eval rows, standardized | square tile NT 1024 x 1024 x d once per outer SMO iteration; batch tile NT nnz_da x 10,000 x d | TUNED 128x128 on both datasets | Istella-S: the square tile YES (1 leaf per group, 2 groups, 64 tiles); the batch tile only when nnz_da = 128 (79 tiles). Taxi: never (k = 11, P = 1) |
| 2 | kmeans:istella | kmeans fit (k 64, 20 iterations, the block's shared init) | `big` block: Istella-S 2,043,304 x 220, raw columns, sentinel cleaned | none. Recorded for DISPATCH only: the unfused tile NT 32768 x 64 x 220, `reach=not_called` | that tile: PLAN_TUNED_64_4X4 | NOT REACHED (the tile would be declined too, n = 64) |
| 3 | pca:istella | PCA fit (8 components, `covariance_eigh`) | the same `big` block | covariance TN 220 x 220 x 2,043,304 | PLAN_SPLIT_64_4X4 | no |
| 4 | kde:istella | KDE `score_samples` (fit before the clock) | `kde` block: 100,000 fit rows, 2,000 queries, standardized | none | - | NOT REACHED |
| 5 | kmeans:taxi, pca:taxi, kde:taxi | the same three callers on taxi (d = 11, `big` block 4,000,000 x 11) | the same entries | kmeans tile 32768 x 64 x 11 (not called); PCA TN 11 x 11 x 4,000,000; KDE none | PLAN_TUNED_64_4X4; PLAN_SPLIT_16_1X1; - | no |
| 6 | gp:taxi, gp:istella | GP fit and predict | 4,000 train and 1,000 test rows, trees loader, standardized, `alpha` 2^-20 (the H100 rung; 12.3) | mean TN 1000 x 1 x 4000; Cholesky trailing NT 3968 x 3968 x 32 | mean PLAN_SPLIT_16_1X1; trailing TUNED 128x128 (both read on the H100 box) | no (n = 1; P = 1) |
| 7 | ols:taxi, ols:istella | OLS fit | `big` block, both datasets | Gram TN d x d x rows | taxi PLAN_SPLIT_16_1X1; Istella-S PLAN_SPLIT_64_4X4 | no |

**Which callers reach the entry, from source.** The coordinator's
correction and section 11.2's table agree with this reading. SVC's
`kernel_op` is the only one of the six that calls `identical_gemm_into` at a
shape the rule takes, so it runs first. KDE and kmeans never call the entry.
PCA, OLS and GP call it only below the floors. Every caller after SVC is a
control and must read HOLDS at a ratio near 1.

**KDE never calls `identical_gemm_into`.** Under IDENTICAL,
`kde/impl/distance/distance.mojo` runs the L2Expanded arm as
`pinned_distance_tile_kernel` (IDENTITY_PATHS row 24).
`core/gemm.mojo::gemm_nt` is the pinned NT kernel, and nothing under `kde/`
calls the entry. Its logs carry `FSPEED-NOTE ... gemm_entry=none` in place
of a DISPATCH line. A KDE ratio beyond noise is box drift, not a plan effect.

**kmeans never calls `identical_gemm_into`.** In
`cluster/impl/detail/min_cluster_distance_compute.mojo`, the Lloyd
assignment for L2Expanded is the fused SIMT distance kernel
(`fused_distance_nn/simt_kernel.mojo`), which writes no distance tile.
Nothing under `cluster/` calls the entry. The one `gemm_nt` in
`cluster/impl/detail/kmeans.mojo` is the k-means++ candidate product. That is
the pinned kernel under IDENTICAL, and the classical cell does not run it
(`init="array"`). The unfused arm's tile is `min(batch_samples, rows) x k x d`,
32768 x 64 x 220 at the default `batch_samples = 1 << 15`. Its line is printed
with `reach=not_called` so the box records the rule's answer there. That
answer is a decline, because n = k = 64 is under the tuned floor, so no kmeans
distance product at k = 64 can cross the rule whatever d is.

**PCA's DISPATCH line is recorded whether its cells run or not.** The body
runs `tools/gemm_ksplit_classical_ab.py shapes` before any timed cell. That
prints every caller's FSPEED-GEMM lines at the declared section 9 shapes, so
`pca.cov.istella.TN` at 220x220x2,043,304 reaches `dispatch.txt` even if the
deadline cuts its cells.

**SVC on Istella-S is the one real question.** Where the default takes the
square tile, each call allocates 2 x 1024 x 1024 floats (8 MB) and
synchronizes twice (10.2), and SMO issues that tile once per outer iteration.
The batch tile is taken only at nnz_da = 128. Below 128, m is under the
tuned floor. From 129 to 1024 the tiles run from 158 to 632, and the rule
declines. How many outer iterations reach nnz_da = 128 is data-dependent,
and the host has no count of it. Taxi SVC runs the same launches on both
arms. The same square tile is also taken on NVIDIA (64 tiles is under 132),
and section 11's H100 leg does not run SVC.

**A correction to 11.3 item 4.** That item expected `choose_plan=[TILE 16x16 ...]`
for `ols.gram.istella` at 220x220x2,043,304. By this reading it is
PLAN_SPLIT_64_4X4. P is 1024 (L = 1996), the SPLITK workspace of 48,400 x
1024 = 49,561,600 floats fits the 64 M cap, and 48,400 cells is under
128x1024. TILE 16x16 is `choose_gemm_plan_untuned`'s answer. Either plan is
outside TUNED 128x128, so neither arm changes the call. The H100 leg's
on-box DISPATCH lines settle the label on NVIDIA
(`bench/results/e1g/2026-09-11_170957-nvidia-h100-80gb-hbm3-gemm-ksplit-classical`,
merged at 523cba80). Both `ols.gram.istella` and `pca.cov.istella` print
`choose_plan=[SPLIT 64x64 reg4x4 KS=32 ...]` with
`ksplit_default_takes=no`. The AMD lines come from this leg; the AMD wide
split row is off, and 220 is not a multiple of 128 in any case.

Predicted verdicts: SVC measured, with taxi a null and Istella-S the
reading. kmeans, PCA, KDE, GP and OLS HOLDS with ratios near 1. Every
witness is equal on every caller.

### 12.3 What was built

- **`tools/gemm_ksplit_classical_ab.py`**, shared with the H100 leg.
  - `time --source ctd` makes `kmeans`, `ols`, `pca`, `kde` and `svc` run on
    the classical lane's blocks under `--data`, through
    `classical_two_datasets.BUILDERS[(lane, "ours")]`, that harness's
    `_digest` and its `quality()`. The timed region, the rows, the cleaning
    and the quality function are the classical lane's own. `gp` has no block,
    so it keeps the trees loader.
  - New lanes `kde`, `svc` and `kmeans`, refused without `--source ctd`.
    Shape tags under ctd end in `-ctd`.
  - FSPEED-ACC lines are written for every float metric: `inertia` for
    kmeans, r2 and rmse for OLS, `explained_variance_ratio_sum` for PCA,
    `mean_log_likelihood` for KDE, `accuracy` for SVC. Counts, flags and
    `*_over_ours` ratios become NOTE lines.
  - Under ctd every FSPEED-GEMM line ends in `reach=entry` or
    `reach=not_called`. KDE and kmeans print `gemm_entry=none`, and kmeans
    also prints its unfused tile as `reach=not_called`. SVC prints three
    lines: the square tile, and the batch tile at nnz_da 128 and at n_ws.
    `SMO_WS_SIZE = 1024` (`svm/impl/workingset.mojo`) and
    `KMEANS_BATCH_SAMPLES = 1 << 15` (`cluster/impl/kmeans_params.mojo`)
    are transcribed for those records only.
  - A new `shapes` subcommand prints each lane's FSPEED-GEMM lines at the
    declared section 9 shapes, with no timing and no device work. Its
    constants come from `tools/classical_two_datasets.py`, and it says so
    when it falls back to transcribed values.
  - `smoke --lanes`, a `verdict` that tells flip_verdict the direction of
    the three classical metrics it does not know, and an optional
    `MOJOLEARN_CLASSICAL_AB_CONTEXT` NOTE line.
  - **Fixes from the H100 leg's evidence (523cba80), which ran this driver.**
    `GP_ALPHA` is now 2^-20. NUMERIC_IDENTICAL accepts only 0 and 2^-20 (the
    Cholesky profile's pinned jitter, DEVIATIONS 1751 and 1752), and the
    driver's 0.1 got every GP cell and the GP smoke refused by name. The
    H100 leg's taxi cells all refused on `No module named 'pyarrow'`, because
    the pixi environment has no pip. The AMD body's uv Python 3.12 venv
    decodes and preps, and every timed process under `--source ctd` loads
    NumPy blocks and never imports pyarrow. The GP trees loader also reads
    the NumPy cache, and imports pyarrow only in the decode.
  - Defaults are unchanged: `time` defaults to `--source trees`, and `smoke`
    and `verdict` default to `gp,ols,pca`. Every line the H100 body's calls
    print is the text they printed before.
- **`tools/gemm_ksplit_classical_amd_leg.sh`**, the new `tools/hotaisle_leg.sh`
  body. Its header lists every step.
  - At t=0: the Istella-S tarball (curl, resumable, sha256-checked,
    untarred) and the taxi months (sha256-checked), in the background.
  - numpy and pyarrow in the pixi python. A uv Python 3.12 venv takes over
    the decode and the prep if pyarrow will not import there.
  - `shapes`: every caller's FSPEED-GEMM lines at the declared shapes,
    before anything can be cut.
  - Per dataset, in the background: `tools/speed_gbdt_arm.py --download`,
    then `tools/classical_two_datasets.py prep` for the lanes with a block.
  - The label binary (`bench/gemm_step_price_main.mojo`, IDENTICAL plus
    trial), outside the fetched tree.
  - The IDENTICAL bindings the lanes import, each with the trial define:
    build.sh (which carries kmeans), build_svm.sh, build_estimators.sh,
    build_gp.sh.
  - `smoke`, then a wait until the data is ready or until
    `MOJOLEARN_CLASSICAL_AB_TIMING_FLOOR` seconds (1500) are left.
  - The ABBA blocks, cell by cell in `MOJOLEARN_CLASSICAL_AB_ORDER` (the
    12.2 order by default; any cell of LANES x DATASETS the order leaves
    out runs after it). Items that cannot start in time are
    SKIPPED_DEADLINE, items whose data failed SKIPPED_NODATA.
  - `dispatch.txt` from the box's own label binary, then `verdicts`.
  - The deadline is the Hot Aisle runner's work bound, read from the
    container's PID 1 `timeout` and `started=`, less 150 s.
- **`bindings/build_svm.sh`** gains the `MOJOLEARN_BUILD_EXTRA_DEFINES` hook
  that build.sh, build_estimators.sh and build_gp.sh already have. It expands
  to nothing when unset, so a release build is unchanged.
- **`tools/gemm_ksplit_classical_leg.sh`**, the H100 body, is unchanged.

**Why a second body.** RunPod passes a body no environment, so the H100
body's defaults are the H100 leg itself. This leg needs a different lane set
and order, the classical prep, the SVM binding, the Hot Aisle deadline and a
resumable fetch. Putting those behind vendor branches in the H100 body would
change the file that leg ships while its run is still owed. The driver is
the shared, vendor-agnostic part.

### 12.4 The record and the verdict

- Everything from 11.2 holds.
- Logs are `<lane>.<dataset>.<tuned128|default>.<block>.log` under
  `<leg out>/remote/ksplit-classical-amd/`.
- `status.tsv` also carries `fetch-*`, `untar-istella`, `shapes`,
  `download-*`, `prep-*`, `wait-data`, and the SKIPPED rows.
- `shapes.log` holds the declared FSPEED-GEMM lines. `gemm_shapes.txt`
  merges them with the lines from the timed logs, and `dispatch.txt` has two
  DISPATCH lines per shape, one for `shipped` and one for `tuned128`. A
  caller name ending in `_not_called` (kmeans) records only the rule's
  answer; that caller never makes the call.
- A timed process that ran beside a decode or a prep carries
  `FSPEED-NOTE ... context=bg=data_work_running`.
- `gate.txt` carries the box's DEFAULT line. On this commit it should read
  `block_parallelism_row=110 ksplit_default=on trial_ksplit_S=110`.
- `plans.tsv` should read `shipped: default=ksplit(S=110) else tuned128`.
- `bindings.sha256` shows each binding's `gfx942` string count, which must
  be above 0.

### 12.5 RUN OWED, Mac, no cost, run by the orchestrator

1. `sh -n tools/gemm_ksplit_classical_amd_leg.sh && dash -n tools/gemm_ksplit_classical_amd_leg.sh`
2. `sh -n bindings/build_svm.sh && dash -n bindings/build_svm.sh`, and
   `sh -n tools/gemm_ksplit_classical_leg.sh` (unchanged, still parses).
3. `python3 -m py_compile tools/gemm_ksplit_classical_ab.py tools/classical_two_datasets.py`
4. One dry run, from a `git worktree add --detach` checkout of the merge
   commit that carries this section:
   `MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_ksplit_classical_amd_leg.sh bash tools/hotaisle_leg.sh amd`.
   Without a mode flag the runner reads no key, calls no API and creates
   nothing. Expect `ok the extra body is valid sh`,
   `ok the extra body environment: 0 export(s)`, the bundle under its cap,
   and `DRY RUN: GREEN. Nothing rented.` A BLOCK for a dirty tree is about
   the checkout, not the body.

This lane ran items 1 to 3 on its own worktree. `sh -n` and `dash -n`
(/bin/dash) passed on all three shell files, `py_compile` passed on both
Python files, and a grep of the new body for bashisms found none. The
orchestrator repeats them at the merge commit, and the item 4 dry run is
still owed. The trial SVM binding has never been built anywhere. Its first
build is on the box, and a red `build-binding-svm` row names it.

### 12.6 The legs

The lease cap is 60 minutes. Two earlier MI300X legs measure most of the
budget:

- **The runner.** On the 164818 Hot Aisle leg the body got
  `work_seconds=3204` after a 106 s image pull. The price harness built in
  31 s and the base binding in 46 s.
- **The classical setup and fits**, on the RunPod MI300X pod (033fadb1,
  `bench/results/classical_hotaisle_2026-09-11/runpod_mi300x/`). The
  bindings built in 41 to 50 s each. The Istella-S fetch took 344 s, the
  untar 16 s, the decode 160 s and the prep 6 s, so the data was ready about
  9 minutes in. Ours per fit, Istella-S then taxi: kmeans 5239 and 129 ms,
  OLS 1828 and 221 ms, PCA 686 and 38 ms, SVC 241 and 1559 ms, KDE about
  0.7 s (608 to 771 ms over the rounds that ran) and 62.6 ms. GP at 4,000
  rows has no row there.

All six callers are 48 timed processes (6 lanes x 2 datasets x 2 arms x 2
blocks), each one warm-up plus 3 rounds. The fits alone sum to about 4
minutes. The rest is per-process overhead: the pixi Python start, the
IDENTICAL import, a block load of up to 2 GB, and quality outside the clock.
The H100 leg that ran this driver measured it (523cba80). Its Istella-S OLS
and PCA processes took 25 to 87 s each, and its Istella-S download step took
1,372 s on that RunPod pod. At those numbers, 48 processes and a slow
Istella-S setup do not fit a 53-minute work bound with margin. So **the set
is split into two legs**. They can run at the same time on two VMs (the team
limit is 2):

- **Leg 1, 32 processes: svc, kmeans, pca, kde.** The entry caller first,
  then the Istella-S controls the coordinator named, then the taxi halves.
- **Leg 2, 16 processes: gp, ols.**

Each caller's two arms always run on one VM, so no ratio mixes VMs.

Leg 1:

```sh
MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_ksplit_classical_amd_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-hotaisle-gemm-ksplit-classical \
MOJOLEARN_HOTAISLE_EXTRA_ENV=MOJOLEARN_CLASSICAL_AB_LANES=svc,kmeans,pca,kde \
bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates
```

Leg 2:

```sh
MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_ksplit_classical_amd_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-hotaisle-gemm-ksplit-classical-gp-ols \
MOJOLEARN_HOTAISLE_EXTRA_ENV=MOJOLEARN_CLASSICAL_AB_LANES=gp,ols \
bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates
```

A caller either leg returns UNMEASURED gets a new lease with only that
caller in `MOJOLEARN_CLASSICAL_AB_LANES`, never a longer one.

Read back from each leg's `<leg out>/remote/ksplit-classical-amd/`:

- `status.tsv`: fetches, `shapes`, decodes, preps, builds, `smoke`,
  `wait-data`, the 32 or 16 timed items, `dispatch`, `verdicts`.
- `gate.txt` and `plans.tsv`: the row-110 DEFAULT line and the labels from
  12.4.
- `dispatch.txt`: 12.2 predicts `ksplit_default_takes=yes` on
  `svc.square_tile.istella.NT` and `svc.batch_tile_nnz128.istella.NT`, and
  `no` on every other line, including
  `kmeans.lloyd_unfused_tile_not_called.istella.NT` and
  `pca.cov.istella.TN`.
- `verdicts.log`: `CLASSICAL KSPLIT A/B svc=... kmeans=... pca=... kde=...`
  (leg 1) and `CLASSICAL KSPLIT A/B gp=... ols=...` (leg 2).

A REGRESSES verdict on SVC is fenced by one of the two ways in 11.2, never
with a vendor branch. A control that does not read HOLDS near 1 points at the
box or the harness, not the plan, because its launches are the same on both
arms.

### 12.7 Risks only a build or a box can settle

- **Never built or run.** The trial define in the SVM binding. The trial hook
  reading the environment on every SVC kernel call. The driver's ctd path:
  runner constructors, `outputs()` every round, `quality()` on a 2 GB block,
  and `shapes`. The body: the ORDER cell list, the deadline read from
  `/proc/1/cmdline`, `date -d`, and the `rocminfo` Marketing Name parse.
- **Python in the pixi environment.** Pixi's Python is 3.14 and has numpy
  (the H100 leg's Istella-S cells ran) but no pip, so the body's `pip install
  pyarrow` there fails by design. The uv 3.12 venv then runs the decode and
  the prep. That venv recipe ran on the RunPod MI300X classical leg, and it
  needs astral.sh and PyPI reachable from the VM.
- **GP at the pinned ridge.** 2^-20 is the larger of the two ridges
  IDENTICAL accepts. A 4,000-row RBF kernel matrix on standardized
  Istella-S, with its near-duplicate rows, may still refuse at a pivot.
  The driver then raises `the kernel matrix did not factor`, and GP is
  UNMEASURED, not guessed.
- **Data time on Hot Aisle is unmeasured.** The MI300X pod had the data
  ready about 9 minutes in. The H100 pod's Istella-S download step took
  1,372 s. The first cell is `svc:istella`, and a cell waits for its
  dataset, so a slow Istella-S fetch holds every cell behind it. If the
  timing floor arrives first, logs of cells that run beside the decode say
  so. A leg that expects a slow fetch can put the taxi cells first through
  `MOJOLEARN_CLASSICAL_AB_ORDER`.
- **Per-process overhead on the MI300X is unmeasured.** On the H100 it was
  25 to 87 s for the Istella-S big-block lanes. SVC's per-round digest calls
  `predict`, which is untimed but still spends lease time.
- **SVC noise on Istella-S.** The pod read 241 ms with a spread of 138 to
  502 ms, wider than any plan effect of a few milliseconds per call, so the
  block band can make HOLDS trivially wide. If the SVC band comes back above
  10 percent, rerun SVC alone with `MOJOLEARN_CLASSICAL_AB_ROUNDS=9`.
- **How often SVC hits nnz_da = 128 is data-dependent**, and nothing on the
  host counts it. SVC `predict` may be taken too (batch x n_support x 220
  under 110 tiles), but it is untimed and outside the verdict.
- **kmeans.** The binding refuses `tol=0`, and the classical runner falls
  back to 1e-7 (21 iterations on the pod). Both arms get the same fallback.
  Its `inertia` quality is a float64 pass over 2,043,304 x 220 rows, outside
  the clock.
- **Noise and controls.** kmeans, PCA, KDE, GP, OLS and taxi SVC are nulls
  by the rule. If any moves beyond its block band, the band belongs to the
  box, and the Istella-S SVC ratio has to be read against it.
- **The NVIDIA confirmation is owed** (rule 10). The H100 default takes the
  same SVC square tile, and section 11.4's leg does not run SVC. The body is
  vendor-agnostic, but its deadline comes from the Hot Aisle runner. On
  RunPod, which passes no environment, `MOJOLEARN_CLASSICAL_AB_BODY_SECONDS`
  would have to be set in the checkout's defaults.
- **Evidence size.** Build logs over 100 KB stay outside the repository.
