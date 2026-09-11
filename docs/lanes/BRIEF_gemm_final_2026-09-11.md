# GEMM final lane, the fold launch of the ksplit default (DEVIATIONS 2640 to 2642)

Source-only lane, September 11, 2026, IDENTICAL only, branch
`lane/gemm-final-h100` off `origin/main` 36ca51fd. Sections 1 to 5 (the
reading, the design and the identity argument) were written and committed
before any code. Sections 6 onward were written after it. Nothing here was
built, compiled, parsed or timed on the Mac or on a GPU. Numbers not copied
from an evidence path are counted from source or fitted to retained lines,
and say so. There is no DEVIATIONS register file in the repository, so this
brief is the record of 2640 to 2642.

Arithmetic boundary: the one in
[BRIEF_gemm_long_k_2026-09-11.md](BRIEF_gemm_long_k_2026-09-11.md) section 5
and [BRIEF_gemm_kernel_2026-09-11.md](BRIEF_gemm_kernel_2026-09-11.md)
section 5. No tensor cores, no TF32, no reassociation, no change to the leaf
partition, the per-leaf chain, the fold tree or any seam. The mechanisms that
already lost on NVIDIA are not repeated: fewer accumulators per thread,
half and quarter tiles, the cell-serial leaf fold and the head geometries
(2540 to 2544), and per-step loads from a packed shared page (2599).

## 1. Purpose, and whether anything is worth building

Andrew called this the last GEMM pass and asked for a stop with the
arithmetic if no arm is likely to save 1 percent of the step (about 3 ms).
The reading below finds one line that is not at a floor and is large enough:
the FOLD LAUNCH of the shipped ksplit default costs 13.3 ms per step, and the
retained fold lines say it is bound by per-thread fold work at one thread per
output cell, not by the arithmetic of the product. The group launches that do
the products are at the packing floor of 132 blocks side by side on every
call but one (head_dA). This lane builds two trial arms that keep the shipped
group launches and change only the fold (and, in the second arm, the group
size the long-k rule already allows). Section 4.4 gives the expected saving
with its range: about 0 to 9 ms for `kfoldv` and about minus 2 to plus 14 ms
for `kfoldv_leaf`, where the low end is the case the fold is DRAM traffic
bound, which no retained line separates. That is enough to justify a leg,
not a prediction that either flips.

## 2. The measurement

Evidence, H100 80GB HBM3 pods at 1980 MHz:

- `bench/results/e1g/2026-09-11_191151-nvidia-h100-80gb-hbm3-gemm-kernel/remote/gemm-kernel/price-shipped.log`
  (commit e6ffb6f4): per-call price of the shipped default, with PHASE lines
  (allocation, group launch, fold launch, each host-synchronized apart).
- `bench/results/e1g/2026-09-11_152822-nvidia-h100-80gb-hbm3-gemm-longk/remote/gemm-longk/price_tables.txt`:
  the same PHASE lines for `ksplit` (the rule the default flipped to) and
  `ksplit_leaf` (the finest group under the workspace cap), plus CONTROL
  shapes.
- `bench/results/e1g/2026-09-11_190725-nvidia-h100-80gb-hbm3-step-breakdown/remote/step-breakdown/breakdown.tsv`:
  GEMM 144.9 ms per step (48 percent of the 299.8 ms timed envelope; lean
  step 0.2958 s enwik8, 0.2947 s Pile GitHub).

### 2.1 The fold launch per call and per step (shipped default, kernel leg)

| call | per step | groups G | cells m n | fold ms | fold ms per step |
|---|---:|---:|---:|---:|---:|
| proj_fwd | 48 | 6 | 1,572,864 | 0.0604 | 2.90 |
| proj_dA | 48 | 6 | 1,572,864 | 0.0600 | 2.88 |
| proj_dB | 48 | 16 | 589,824 | 0.0506 | 2.43 |
| gateup_dA | 24 | 8 | 1,572,864 | 0.0696 | 1.67 |
| gateup_dB | 24 | 8 | 1,572,864 | 0.0698 | 1.68 |
| down_fwd | 12 | 8 | 1,572,864 | 0.0700 | 0.84 |
| down_dB | 12 | 8 | 1,572,864 | 0.0701 | 0.84 |
| head_dA | 1 | 7 | 1,572,864 | 0.0670 | 0.07 |
| | | | | **sum** | **13.30** |

The allocation phase is 3.0 µs per call (0.65 ms per step over 217 calls).
The group phase is the rest.

### 2.2 The group phase per leaf-round

A leaf-round is one 128-step leaf walked by 132 blocks side by side. The long-k
brief's rounds model (section 3.2, E1) says a launch of equal blocks takes
`ceil(blocks / 132)` rounds of one block's walk. Its CONTROL pair confirmed
the rounding: `ctl_nt_1536x1408x768` at 792 one-leaf blocks (6 rounds) priced
0.2518 ms and `ctl_nt_1664x1408x768` at 858 (7 rounds) 0.2876 ms, a ratio of
1.142 against 7/6 = 1.167 and against a cell ratio of 1.083. Per leaf-round,
from the group phase of each call:

| call and plan | blocks x leaves per block | leaf-rounds | group ms | µs per leaf-round |
|---|---|---:|---:|---:|
| gateup_fwd, `ksplit_leaf` | 1536 x 1 | 12 | 0.4703 | 39.2 |
| down_dA, `ksplit_leaf` | 1536 x 1 | 12 | 0.4739 | 39.5 |
| gateup_dA, `ksplit_leaf` | 1536 x 1 | 12 | 0.4816 | 40.1 |
| proj_fwd, shipped | 576 x 1 | 5 | 0.2064 | 41.3 |
| gateup_dA, shipped | 768 x 2 | 12 | 0.5082 | 42.3 |
| proj_dB, shipped | 576 x 1 | 5 | 0.2159 | 43.2 |
| head_dA, `ksplit_leaf` (25 groups of 16) | 2400 x 16 (one tail of 9) | 290 | 12.843 | 44.3 |
| head_dA, shipped (7 groups of 64) | 672 x 64 (one tail of 9) | 320 | 14.341 | 44.8 |
| gateup_fwd, shipped (tuned 128x128) | 256 x 6 | 12 | 0.5578 | 46.5 |
| head_fwd, shipped (tuned) | 6288 x 6 | 288 | 13.913 | 48.3 |
| head_dB, shipped (tuned) | 2358 x 16 | 288 | 14.034 | 48.7 |
| ctl 1536x1408x768, tuned | 132 x 6 | 6 | 0.2899 | 48.3 |
| ctl 1536x1408x768, `ksplit_leaf` | 792 x 1 | 6 | 0.2518 | 42.0 |

The head_dA leaf-round counts are the first-in first-out schedule of its
blocks, counted in section 3.3. Two facts come out of this table. One-leaf
walks run 13 to 19 percent faster per leaf than the tuned all-leaves walks,
on the same kernel body (cause not measured). And head_dA spends 320
leaf-rounds where 288 are enough.

## 3. The source reading

### 3.1 What the fold launch does per cell

`_ksplit_fold_launch` (gemm/checks/gemm_identical.mojo) sends every step call
(`m n` above `SPLIT_BLOCK_FOLD_MAX_CELLS` = 16,384) to
`identical_gemm_fold_stack_kernel`: ONE THREAD PER OUTPUT CELL, blocks of
`FLAT_TPB` = 256 threads on a 1-D grid (6,144 blocks at 1.57 M cells). Each
thread loads its `G` group nodes from `ws[q m n + cell]`, eight at a time
(`FOLD_BATCH`, none batched when `G < 8`), pushes each into `_fold_push`'s
16-slot register stack, drains with `_fold_drain`, and stores `ftz(root)`.
`_fold_push` and `_fold_drain` unroll `GEMM_FOLD_LEVELS` = 12 levels at
compile time, each with an occupancy test and a `placed` guard, so a push
costs about 12 level tests plus at most one merge (a merge is three `ftz` and
one add). Counted from source: about 80 instructions per push, so about 550
per cell at `G` = 6 and 1,340 at `G` = 16, nearly all of it control, not
arithmetic.

### 3.2 A fit that every retained fold line follows

The resources README (`bench/results/gemm_resources_2026-09-10/README.md`)
transcribes 2,048 resident threads per SM on the H100. A one-thread-per-cell
kernel with no shared memory is then capped at 2,048 x 132 = 270,336 cells
per round. Fitting `fold = 8 µs + rounds x t(G)` with
`rounds = ceil(m n / 270,336)`:

| shape | G | cells | rounds | fold ms | (fold - 8 µs) per round |
|---|---:|---:|---:|---:|---:|
| ctl 768x768 | 6 | 589,824 | 3 | 0.0275 | 6.5 µs |
| proj_fwd | 6 | 1,572,864 | 6 | 0.0604 | 8.7 µs |
| ctl 1536x1408 | 6 | 2,162,688 | 8 | 0.0803 | 9.0 µs |
| ctl 1664x1408 | 6 | 2,342,912 | 9 | 0.0866 | 8.7 µs |
| gateup_fwd (`ksplit_leaf`) | 6 | 4,194,304 | 16 | 0.1465 | 8.7 µs |
| head_dA | 7 | 1,572,864 | 6 | 0.0670 | 9.8 µs |
| gateup_dA | 8 | 1,572,864 | 6 | 0.0696 | 10.3 µs |
| proj_dB | 16 | 589,824 | 3 | 0.0506 | 14.2 µs |
| ctl 1024x1024 | 16 | 1,048,576 | 4 | 0.0825 | 18.6 µs |
| gateup_dA (`ksplit_leaf`) | 16 | 1,572,864 | 6 | 0.1158 | 18.0 µs |
| head_dA (`ksplit_leaf`) | 25 | 1,572,864 | 6 | 0.1628 | 25.8 µs |

At fixed `G` the time per round is constant across a 7x range of cells, and
per round it grows about 1 µs per group. `G` = 7 (seven serial load-push
pairs) and `G` = 8 (one batch of eight loads, then eight pushes) cost the
same per round, so the loads' latency does not set it. What sets it, on this
reading, is the per-thread push work times the rounds the thread cap forces.
This is a fit to price lines, not a timer. The competing cause is DRAM
traffic: 1.57 M cells times 7 words is 44 MB in 60 µs, about 730 GB/s. No
retained line separates the two, and section 4.4 carries both.

### 3.3 What is already at a floor

With 132 blocks side by side and blocks that walk whole leaves (a group
boundary must be a leaf boundary, long-k Lemmas A to C), a call cannot finish
in fewer leaf-rounds than `ceil(tiles x P / 132)`. First-in first-out
schedules of the shipped plans, counted by hand:

| call | tiles x P | floor | shipped | finest group under the cap |
|---|---:|---:|---:|---:|
| proj_fwd, proj_dA | 96 x 6 | 5 | 5 (576 one-leaf blocks) | 5 (same) |
| proj_dB | 36 x 16 | 5 | 5 | 5 (same) |
| gateup_fwd, down_dA | 256 x 6 | 12 | 12 (tuned, 2 x 6) | 12 (1536 one-leaf blocks) |
| gateup_dA, gateup_dB, down_fwd, down_dB | 96 x 16 | 12 | 12 (768 x 2) | 12 (1536 x 1) |
| head_fwd | 6288 x 6 | 286 | 288 | not under the cap |
| head_dB | 2358 x 16 | 286 | 288 | not under the cap |
| head_dA | 96 x 393 | 286 | 320 | 290 (25 groups of 16) |

head_dA's 320: 576 blocks of 64 leaves finish four full rounds at 256, the
last 48 run to 320, and the 96 nine-leaf tail blocks fit in the gaps. At 16
leaves: 2,304 blocks of 16 finish 17 rounds at 272, the last 60 end at 288,
the tail ends at 290. So no group size or tile count can buy a leaf-round on
any call except head_dA (320 to 290, about 1.3 ms, and `ksplit_leaf` MEASURED
it: 12.99 against 14.32 ms on the long-k pod). What remains is (a) the fold
launch, 13.3 ms per step, (b) head_dA's schedule, and (c) the faster
per-leaf rate of one-leaf walks, which only a cheaper fold can buy: on the
long-k pod `ksplit_leaf` lost 0.058 ms per gateup_fwd call because its fold
(0.1465 ms) ate its group gain (0.556 to 0.470 ms).

### 3.4 Candidates declined from the reading

- **Merging calls that share an operand** (q, k and v forward share `x`; their
  dB calls share `x` too). Cell for cell the merged product is the same
  chain, so bits hold. But the merge lives in the step glue and the attention
  file, which other lanes own, and at 288 tiles the merged forward is 14
  leaf-rounds against three calls of 5 each, about 0.04 ms per merged call.
  A dA merge would add inside the chain and change bits.
- **Folding inside the last group's launch.** The last group's blocks would
  load `G - 1` stored nodes per cell after their walk, in the heavy kernel's
  threads. That is serial non-arithmetic work in the 255-register kernel,
  the kind E-e (2540 `lfold`) priced at 2.76x, and on the 2-leaf calls it
  adds a leaf-round (672 blocks in the first launch is 6 rounds, not 5).
- **Wider tiles to cut tiles per round.** 128 cells per thread was 12 percent
  slower per flop inside the 2599 family, and the tile axis is where 2540 and
  2599 already lost.
- **A hardware FTZ multiply in the fold.** It would be a new seam spelling on
  one vendor for signaling payloads; the fold keeps the software `ftz` the
  shipped fold uses.
- **Dropping the synchronize after the workspace allocation.** At most 3 µs
  per call, 0.65 ms per step, and it touches the shipped call structure.

## 4. The design (DEVIATIONS 2640 to 2642)

### 4.1 The lane fold kernel (2640)

`identical_gemm_kfold_lanes_kernel[W, FS, VB, SAB]` replaces one thread per
cell with one thread per `W` CONSECUTIVE cells of the row-major output.
Block `b`, thread `t` owns cells `base + e`, `e < W`, with
`base = (b TPB + t) W`. Per thread:

1. Group nodes are loaded `W` wide: `ws.unsafe_load[width=W](q m n + base)`
   when the run is inside the output, else `W` masked scalar loads (the last
   thread of a ragged output). `VB` group vectors load before their pushes,
   then the tail one at a time, `q` ascending.
2. Each group vector is pushed into a REGISTER stack of `FS` levels by
   `W` lanes, `_fold_push_lanes[W, FS]`, which is `_fold_push` with every
   float replaced by a lane and the level loop unrolled at compile time. The
   drain is `_fold_drain_lanes[W, FS]`, `_fold_drain` the same way.
3. The stored cell is `ftz(root)` per lane, one `W`-wide store, or masked
   scalar stores on the ragged thread.

Constants, the same on every column (no vendor reads anything here):
`GEMM_KFOLD_W` 16, `GEMM_KFOLD_FS` 8, `GEMM_KFOLD_VB` 2,
`GEMM_KFOLD_TPB` = `FLAT_TPB` (256). Register widths: stack 128, a load batch
32, a lane vector 16, all powers of two. `FS` 8 covers `G <= 255`
(`GEMM_KFOLD_MAX_GROUPS`); a call with more groups keeps the shipped fold.
Counted from source, the lane fold issues about 99 instructions per cell at
`G` = 6 (against about 550) and 267 at `G` = 16 (against about 1,340),
because the level control is paid once per 16 cells, and it launches 16 times
fewer threads (384 blocks at 1.57 M cells, one round under the 2,048-thread
cap for every step call).

### 4.2 `kfoldv` (2640): the shipped rule, the lane fold

On every call where `gemm_step_ksplit_rule(m, n, k, GEMM_KSPLIT_S, True)` (the
shipped default's rule on NVIDIA, S = 132) takes the call: allocate `m n G`,
synchronize, `_ksplit_groups_launch` (the shipped group kernel, unchanged),
the lane fold, synchronize. Every other call runs the shipped dispatch. It
isolates the fold: the group launches, group sizes, allocations and waits are
the shipped ones.

### 4.3 `kfoldv_leaf` (2641): the finest group under the cap, the lane fold

The same lines at `gemm_step_ksplit_rule(m, n, k, 0, False)`, the `ksplit_leaf`
rule already priced with bits equal on the H100: one-leaf groups on proj_*,
gateup_*, down_* (gateup_fwd and down_dA included, which the shipped rule
declines at 256 tiles), 16-leaf groups on head_dA (25 groups). head_fwd and
head_dB stay shipped (not under the cap). It buys the one-leaf walk rate and
head_dA's schedule, which only pay once the fold is cheap.

### 4.4 Expected saving, with the arithmetic

Model A (the fold is per-thread-work bound, section 3.2): lane fold
`= 8 µs + (scalar fold - 8 µs) / 5`, from the source counts in 3.1 and 4.1.
Model B: half of that gain. Model C (the fold is DRAM traffic bound): no gain.

`kfoldv`, fold µs saved per call times calls per step, model A:
proj_fwd 41.9 x 48 = 2.01 ms; proj_dA 41.6 x 48 = 2.00; proj_dB 34.1 x 48 =
1.64; gateup_dA 49.3 x 24 = 1.18; gateup_dB 49.4 x 24 = 1.19; down_fwd
49.6 x 12 = 0.60; down_dB 49.7 x 12 = 0.60; head_dA 0.05. **9.3 ms (3.1
percent of the step). Model B 5.8 ms. Model C 0.**

`kfoldv_leaf`, model A, against the shipped price on the long-k pod (group
phase from `ksplit_leaf`'s PHASE line plus the model A lane fold):
gateup_fwd (0.5618 to 0.4703 + 0.0357) 55.8 µs x 24 = 1.34 ms; down_dA
(0.5705 to 0.5095) x 12 = 0.73; gateup_dA (0.5765 to 0.5112) x 24 = 1.57;
gateup_dB (0.5890 to 0.5190) x 24 = 1.68; down_fwd (0.5714 to 0.5036) x 12 =
0.81; down_dB (0.5870 to 0.5152) x 12 = 0.86; head_dA (14.321 to 12.882) 1.44;
proj_* as `kfoldv` 5.65. **14.1 ms (4.8 percent). Model C: the measured
`ksplit_leaf` against `ksplit`, a LOSS of 2.2 ms** (gateup_fwd and down_dA
lose 2.0 ms, the 2-leaf calls lose 1.5, head_dA gains 1.3).

The leg reads which model holds from the PHASE lines (`fold_ms` of both arms
against the shipped default on the same pod) and decides by the LM step.

## 5. The identity argument (written before the code)

### 5.1 The value every stored cell must hold

For output cell `c`, the shipped default stores
`ftz(_fold_drain(S_G))` where `S_0` is the empty 12-level stack and
`S_{q+1} = _fold_push(S_q, node(q, c))` for `q = 0 .. G - 1` ascending
(`identical_gemm_fold_stack_kernel`, or at `m n <= 16,384`
`identical_gemm_fold_kernel[True]`, which
`check_group_fold_is_the_contract_tree` holds equal to it and to
`fold_balanced_tree` over all `P` leaf partials for every `P` in 1 to 1,100
and every power-of-two group). `node(q, c)` is the word the group launch
stored at `ws[q m n + c]`.

### 5.2 The group launch and the group sizes are the shipped ones

Both arms call `_ksplit_groups_launch` with `identical_gemm_ksplit_kernel`
unchanged, so `node(q, c)` is the shipped group node for the group size the
arm's rule names. `kfoldv`'s rule is the shipped default's rule at the same
row. `kfoldv_leaf`'s rule is `ksplit_leaf`'s. Long-k Lemmas A to C hold for
every power-of-two group size aligned at leaf 0 and every tile, so for either
rule the fold of the `G` nodes by 5.1 is the contract tree over the `P`
leaves. The group size is execution plan (contract 6.1).

### 5.3 Every cell has exactly one lane, and lanes never mix

`c -> (b, t, e) = (c // (256 W), (c // W) mod 256, c mod W)` is a bijection
from `[0, m n)` onto the lanes with `base + e < m n`. A thread with
`base >= m n` returns before any load or store. A lane with
`base + e >= m n` loads nothing (its register lane stays `+0.0`) and stores
nothing. Every expression in the kernel, the push and the drain reads and
writes lane `e` of one register from lane `e` of another, or loads lane `e`
from `ws[q m n + base + e]` (the `W`-wide load's element `e` is that word,
and the masked scalar load reads the same word). No expression combines two
lanes. So each stored cell is a function of its own `G` nodes only.

### 5.4 One lane performs `_fold_push` and `_fold_drain` exactly

`occ` is one integer per thread. It starts at 0 and changes only in
`_fold_push_lanes`, by the same `occ - (1 << d)` and `occ + (1 << d)` as
`_fold_push`, so after `q` pushes it equals the `occ` the scalar kernel holds
after `q` pushes of any cell. Every branch in the push and the drain reads
only `occ`, `d` and `placed`, so every lane of a thread takes the branches the
scalar kernel takes for its cell. In a taken merge, lane `e` computes
`ftz(ftz(stack[d W + e]) + ftz(val[e]))`, which is `_fold_push`'s
`ftz(ftz(stack[d]) + ftz(val))` with `stack[d]` read from the same level. In
a store it copies `val[e]` to `stack[d W + e]`. The drain is `_fold_drain`
lane by lane: the lowest occupied level copies, each higher occupied level
joins on the left through the same expression. The stored value is
`ftz(root[e])`. The load batches change when a node is loaded, never which
node a push receives or the order of pushes (`q` ascending in both loops).

### 5.5 Eight levels instead of twelve

Before push `q + 1`, `occ = q <= G - 1 <= 254`. Some bit of `q` among bits 0
to 7 is clear, so `placed` becomes True at a level below 8, which is the
level `_fold_push` places at, having made the same merges below it. After
the last push `occ = G <= 255`, so the drain's occupied levels are among 0 to
7, the same levels `_fold_drain` visits (levels 8 to 11 are never occupied
there either). `_fold_push_lanes` returns False only when all eight levels
are occupied, which needs `occ = 255` before a push, that is `G >= 256`; the
launcher refuses `G > 255` and keeps the shipped fold. The host check proves
the bound both ways.

### 5.6 What varies, and reach

`W`, `FS`, `VB` and the block size are execution plan: which thread owns a
cell, how many loads serve it, how many registers hold its stack. None of
them reaches a leaf boundary, a group boundary, a tree level or a seam. They
are comptime constants equal on every column. `k == 0` takes no group launch
in either arm; it runs the step arm kernel at the shipped geometry
(`identical_gemm_step_ksplit_into`'s own `P == 0` line) and stores `+0.0`.

`SAB = True` is compiled only under `-D MOJOLEARN_GEMM_ARM_TRIAL=1`. The group
launch runs its existing sabotage (node `1.0e30` for thread `q mod 256`,
register cell `q // 256`, per `(tile, q)`), and the lane fold moves lane 0 of
thread 0 of every fold block through `_gemm_step_arm_sabotage`. A cell hit by
both still differs (a `1.0e30` root with one bit added). So the reach is the
group launch's cells plus the fold blocks minus the cells in both, counted on
the host by `gemm_step_kfold_reach`; where `G > 255` the shipped fold runs
and the reach is the group cells alone. The check requires exact equality,
which names the rule, the group count, `W` and the block size that ran.

## 6. State at wind-down (same day; Andrew asked every lane to stop)

Written after the code. NOTHING WAS BUILT, COMPILED, PARSED OR RUN, on the Mac
or on a GPU.

**Built (source only).**

- `gemm/checks/gemm_identical.mojo`, trial arm sections only: arms `kfoldv`
  (12) and `kfoldv_leaf` (13) and geometries 12 and 13 (counts now 14), with
  parse, name, geometry, tile, geometry name and plan label entries;
  `identical_gemm_step_geometry_into` dispatches them under the trial define;
  `gemm_step_geometry_group_leaves` and `gemm_step_geometry_reach` answer for
  them. A new section before `_fast_vendor_gemm`: the constants
  (`GEMM_KFOLD_W` 16, `_FS` 8, `_VB` 2, `_TPB` = `FLAT_TPB`, `_MAX_GROUPS`
  255), `_fold_push_lanes`, `_fold_drain_lanes`,
  `identical_gemm_kfold_lanes_kernel[W, FS, VB, SAB]`, `gemm_kfold_blocks`,
  `_kfold_fold_launch`, `_kfold_run`, `identical_gemm_step_kfold_into`,
  `identical_gemm_step_kfold_phase_into`, `_kfold_ksplit_geometry`,
  `gemm_step_kfold_rule`, `gemm_step_kfold_leaves`, `gemm_step_kfold_reach`,
  `_kfold_geometry_name`. No shipped kernel, dispatch line or kernel matrix
  value changed; the new kernel is referenced only under
  `comptime if GEMM_ARM_TRIAL`.
- `gemm/checks/gemm_step_arms_check.mojo`: both names in `_arm_names()`, so
  the selector round-trips them, the ragged part forces geometries 12 and 13
  at every case (bits against the old plan and FLAT, exact reach) and the LM
  section runs both through `identical_gemm_into`. A no-trial build fails
  them by name like every other arm.
- `bench/gemm_step_price_main.mojo`: PHASEBITS and PHASE lines with
  `phase_of=arm_kfold` where either arm takes a call.
- `tools/gemm_final_leg.sh`: the leg body (section 4.4's arms).

**Finished after the resume (branch `lane/gemm-final-checks` off main, which
already carries the arms at d368b6d8; no kernel or dispatch line changed).**

1. `check_kfold_lanes_is_the_stack_fold` (host): for every `G` in 1 to 255 and
   three node kinds (ordinary, `-0.0` mixed, about a third scaled by `1e-38`
   so subnormals reach every `ftz`), 16 lanes pushed with `_fold_push_lanes`
   and drained with `_fold_drain_lanes` must store in every lane the bits
   `_fold_push`, `_fold_drain` and `ftz` store for that lane's cell alone,
   and `fold_balanced_tree`'s bits for the first two kinds; no overflow below
   256 groups, `occ` ends at `G`, 255 pushes fit and the 256th overflows.
2. `check_kfold_rule_hand_counts` (host): at the twelve LM calls, `kfoldv` at
   S = 132 equals `gemm_step_ksplit_rule(.., 132, True)` and the hand counts
   [1,1,1,0,2,2,2,0,2,0,64,0]; `kfoldv_leaf` equals the ksplit_leaf rule and
   [1,1,1,1,1,1,1,1,1,0,16,0]; groups [6,6,16,0,8,8,8,0,8,0,7,0] and
   [6,6,16,6,16,16,16,6,16,0,25,0], none above 255; fold blocks
   [384,384,144,1024,384,384,384,1024,384,25129,384,9424].
3. Resources rows `fold_stack_shipped` and `kfold_lanes` in
   `bench/gemm_step_resources_main.mojo`.
4. Docstring paragraphs in the check, the price harness and the resources
   harness; the check's banner and PASS lines name 2640 to 2642.

## 7. RUN OWED, M4, light, run by the orchestrator one at a time

1. `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . gemm/checks/gemm_step_arms_check.mojo -o /tmp/gemm-final-check`
   then `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000 /tmp/gemm-final-check`.
   Expect PASS with `REACH ragged [kfoldv ...] N/N` and
   `REACH ragged [kfoldv_leaf ...] N/N` beside the existing lines, and before
   any device work
   `check_kfold_lanes_is_the_stack_fold: 12240 lane folds (3 kinds, G 1..255, W=16, FS=8), 0 disagree; overflow bound fit=True next_placed=False`,
   twelve `RULE_KFOLD` lines and `check_kfold_rule_hand_counts: 0 failures`.
   The no-trial build (item 2) must print the same two host lines with 0
   failures.
5. The resources harness build (no run on the M4 is needed):
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_resources_main.mojo -o /tmp/gemm-final-resources`.
   On a GPU box it prints `fold_stack_shipped` and `kfold_lanes` rows.
2. The same without `-D MOJOLEARN_GEMM_ARM_TRIAL=1` (`-o /tmp/gemm-final-check-notrial`),
   same env: expect FAIL naming the define for both new geometries.
3. `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo -o /tmp/gemm-device-check && /tmp/gemm-device-check`:
   `all green [IDENTICAL]  (8 gates, sabotage: none)`, unchanged.
4. `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_price_main.mojo -o /tmp/gemm-final-price`,
   then `MOJOLEARN_GEMM_STEP_LABEL_ONLY=1 MOJOLEARN_GEMM_ARM=kfoldv /tmp/gemm-final-price`.

## 8. The H100 leg

From a detached worktree of the commit carrying this section, after section
7 is green, with the Apple card first
(`tools/gemm_card.sh device /tmp/gemm-final-apple.card`):

```sh
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=sm_90a \
MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_final_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-gemm-final \
sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" --local-card /tmp/gemm-final-apple.card
```

About 25 minutes on the pod (the 2599 leg with the same shape took 23).
Read `price_tables.txt` PHASE `fold_ms` for `arm_kfold` against
`shipped_default` (which of models A to C holds) and `lm_summary.tsv`
verdicts.

## 9. Risks only a build or a box can settle

- `SIMD[DType.float32, FS * W]` as a `mut` parametric argument, 16-wide global
  loads and a 16-wide global store in a kernel (the width-4 load is the only
  precedent), and a 128-lane register stack: compile and register count.
- Register spill: stack 128 + batch 32 + vectors; a spill to local memory
  would pay the traffic section 3.4 avoids.
- Model C (DRAM traffic bound fold): `kfoldv` flat and `kfoldv_leaf` about
  2 ms slower.

## 8. H100 leg results (2026-09-11 21:15Z to 21:40Z, RunPod NVIDIA H100 80GB HBM3 at 1980 MHz, commit d368b6d8, run by the neural session mojolearn-d1): NO FLIP, both arms

Evidence `bench/results/e1g/2026-09-11_211539-nvidia-h100-80gb-hbm3-gemm-final`
(`remote/gemm-final/` holds the on-box files). Gates: `status.tsv` every
item 0 (builds, step-check, resources, four prices, both bindings, both
corpora, 16 LM probes); `diff_apple_vs_nvidia.txt` `RESULT: IDENTICAL`;
`lm_summary.tsv` `witnesses_equal_baseline=True` for both arms on both
corpora and for the closing shipped run. The M4 gates of section 7 were
green before the leg (trial arms check PASS with `REACH ragged` 102/102 for
both arms, no-trial run FAIL naming the define, device check all green,
price and resources builds). The host checks of section 6 were not in this
commit; mojolearn-df added them on main afterwards (2c64a778, kernels
unchanged), so the leg measured exactly these kernels.

Lean step, steady median seconds, one pod:

| run | enwik8 | Pile GitHub | ratio over shipped | verdict |
|---|---:|---:|---:|---|
| shipped (`ksplit(S=132) else tuned128`) | 0.2940 | 0.2924 | 1 | reference |
| shipped, closing run | 0.2935 | 0.2906 | 0.998 / 0.994 | drift bracket |
| `kfoldv` (2640) | 0.3151 | 0.3167 | 1.0727 / 1.0864, geomean 1.0795 | NO FLIP |
| `kfoldv_leaf` (2641) | 0.3329 | 0.3318 | 1.1331 / 1.1382, geomean 1.1357 | NO FLIP |

GEMM sum per step (`price_step.txt` STEP lines, per-call medians weighted
by per-step counts): shipped 141.4 ms, `ksplit_leaf` 143.9 (1.016),
`kfoldv` 162.9 (1.143), `kfoldv_leaf` 181.4 (1.263); `moved=0` everywhere.

The fold launch per call (`price_tables.txt` PHASE lines, `fold_ms`,
milliseconds; groups in parentheses):

| call | shipped fold | `kfoldv` fold | `kfoldv_leaf` fold | `ksplit_leaf` fold (shipped kernel, finer rule) |
|---|---:|---:|---:|---:|
| proj_fwd, proj_dA (2048 x 768 x 768) | 0.060 (6) | 0.155 (6) | 0.154 (6) | 0.060 (6) |
| proj_dB | 0.050 (16) | 0.115 (16) | 0.116 (16) | 0.051 (16) |
| gateup, down (2048 x 3072 x 768 and back) | 0.069 (8) | 0.191 (8) | 0.334 (16) | 0.115 (16) |
| head_dA (2048 x 768 x 50257) | 0.067 (7) | 0.162 (7) | 0.478 (25) | 0.163 (25) |

The reading. The lane fold (16 cells per thread through an 8-level
register stack) takes 2.3x to 2.8x the shipped fold's time at the same
group count on every call, and the shipped fold at 16 groups costs the same
as at 6 (0.050 against 0.060 ms) where the lane fold scales with the groups
it loads (0.115 at 16, 0.155 at 6, 0.334 at 16 on the wider calls). So the
fold's cost was never the per-thread push instructions section 3.2 fitted;
the shipped one-thread-per-cell fold is already at its traffic and launch
floor at these sizes, and a fold that reads every group into one thread's
lanes pays its loads serially. Section 3.2's fit read launch-count scaling
as instruction count. `kfoldv_leaf` adds the finer rule's extra groups on
top. `ksplit_leaf`, the price-only control (the shipped fold under the finer
rule), takes head_dA from 14.31 to 12.97 ms per call (its 25 groups against
7) and loses about 0.03 ms on each of the 20 small calls, so its GEMM sum is
1.016x shipped: not a flip either, and not an LM arm here.

No register readback for the fold kernels exists in this leg (the resources
rows of section 6 item 3 landed after it, 2c64a778); a later leg at that
commit prints them. Nothing else is owed on these arms: the fold is bound by
traffic, and the trial arms stay as trial arms (`MOJOLEARN_GEMM_ARM_TRIAL`
only), never a default.
