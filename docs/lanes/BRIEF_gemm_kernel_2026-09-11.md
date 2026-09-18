# GEMM kernel lane, the base identical kernel's own throughput on the H100 (DEVIATION 2599)

Source-only lane, September 11, 2026, IDENTICAL only, branch
`lane/gemm-kernel-h100` off `origin/main` 5cc3b8df. STATUS: SOURCE BUILT,
NOTHING COMPILED OR RUN. Sections 1 to 5 (the reading, the design and the
identity argument) were written before any code. Sections 6 onward were
written after it. Nothing here was built, compiled, parsed or timed on the Mac
or on a GPU. Numbers not copied from an evidence path are counted from source
and say so. There is no DEVIATIONS register file in the repository, so this
brief is the record of 2599.

Arithmetic boundary: the one in
[HANDOFF_speed_gemm_2026-09-10.md](../../archive/docs/lanes/HANDOFF_speed_gemm_2026-09-10.md),
[BRIEF_gemm_step_2026-09-11.md](BRIEF_gemm_step_2026-09-11.md) section 5 and
[BRIEF_gemm_long_k_2026-09-11.md](BRIEF_gemm_long_k_2026-09-11.md) section 5.
No tensor cores, no TF32, no reassociation, RN-FMA then the FTZ multiply
kept, leaf boundaries and fold topology kept. The occupancy arms of
DEVIATIONS 2540 to 2544 (fewer accumulators per thread) are not repeated.

## 1. Purpose

GEMM is about 143 ms of the 295 ms shipped LM step on the H100. The shipped
ksplit default (DEVIATION 2595) already gives the per-layer calls enough
blocks, and its CONTROL pairs read 1.00 where a call was already one block
per SM. What is left is the throughput of one block of the base identical
kernel. This lane reads that kernel from source, names what can bound it
and what each existing line already says, and builds two trial arms under
`-D MOJOLEARN_GEMM_ARM_TRIAL=1` that change how a block spends its time
without moving one stored bit on any vendor. It hands back an H100 leg that
decides them on the two LM corpora.

## 2. The measurement

Evidence: `bench/results/e1g/2026-09-11_163310-nvidia-h100-80gb-hbm3-gemm-ksplit-default/remote/gemm-longk/price_tables.txt`
(`price-shipped` run, H100 80GB HBM3, `clocks.current.sm` 1980 MHz at the
start). Useful TFLOP/s counts two flops per product step.

| call | op | m x n x k | per step | shipped ran | launched blocks | ms | TFLOP/s |
|---|---|---|---:|---|---:|---:|---:|
| proj_fwd | NT | 2048 x 768 x 768 | 48 | ksplit 1 leaf | 576 | 0.2655 | 9.10 |
| proj_dA | NN | 2048 x 768 x 768 | 48 | ksplit 1 leaf | 576 | 0.2683 | 9.00 |
| proj_dB | TN | 768 x 768 x 2048 | 48 | ksplit 1 leaf | 576 | 0.2653 | 9.11 |
| gateup_fwd | NT | 2048 x 2048 x 768 | 24 | TUNED 128 | 256 | 0.5579 | 11.55 |
| gateup_dA | NN | 2048 x 768 x 2048 | 24 | ksplit 2 leaves | 768 | 0.5757 | 11.19 |
| gateup_dB | TN | 2048 x 768 x 2048 | 24 | ksplit 2 leaves | 768 | 0.5871 | 10.97 |
| down_fwd | NT | 2048 x 768 x 2048 | 12 | ksplit 2 leaves | 768 | 0.5696 | 11.31 |
| down_dA | NN | 2048 x 2048 x 768 | 12 | TUNED 128 | 256 | 0.5661 | 11.38 |
| down_dB | TN | 768 x 2048 x 2048 | 12 | ksplit 2 leaves | 768 | 0.5822 | 11.07 |
| head_fwd | NT | 2048 x 50257 x 768 | 1 | TUNED 128 | 6,288 | 13.876 | 11.39 |
| head_dA | NN | 2048 x 768 x 50257 | 1 | ksplit 64 leaves | 672 | 14.379 | 11.00 |
| head_dB | TN | 50257 x 768 x 2048 | 1 | TUNED 128 | 2,358 | 14.016 | 11.28 |

STEP (weighted GEMM sum): 142.5 ms. The lean step on the same pod is 0.343 s
(enwik8) and 0.346 s (Pile GitHub).

**A correction to the task's framing, fixed here as found.** The "about 9.4
to 9.6 TFLOP/s" for gateup_fwd, down_dA, head_fwd and head_dB is the earlier
table (`2026-09-11_133216-nvidia-h100-gemm-step`, 1590 MHz pod, long-k brief
section 2). On the 1980 MHz pod the same four calls read 11.28 to 11.55. The
ratio of the two readings (about 1.20) is close to the ratio of the two SM
clock readings (1.245). The per-call milliseconds the task quotes are the
1980 MHz table's. The ceiling on this pod is therefore about 11.4 TFLOP/s,
and every model below is written per SM clock so the two pods can be read
with it. proj_* sit near 9.1 because the default's one-leaf groups carry a
fold launch (PHASE `fold_ms` 0.060 of 0.270 ms) and 576 blocks run in about
five rounds.

torch 2.4.1 eager fp32 runs the whole step in 62 ms on that pod family. It
is an owed opponent row, not a claim, and it is not used below.

## 3. The source reading

### 3.1 One block of the body every step call runs

Both paths of the shipped dispatch run the same per-window body:
`identical_gemm_tuned_kernel[8, 8, 16, 16, 16, 2]` (the TUNED 128x128 plan)
and `identical_gemm_ksplit_kernel[8, 8, 16, 16, 2]` (the ksplit group
kernel, the same window loop over a leaf range). On NVIDIA,
`lib_hardware_ftz_fma_for` is true, so `TUNED_STAGE_FTZ` is on (operands
flushed once at staging) and `_tuned_step` is
`llvm.nvvm.mul.rn.ftz.f(llvm.nvvm.fma.rn.f(a, b, acc), 1.0)`.

Geometry: `TUNED_TPB` 256 threads; `TC` 16 thread columns, so `TR` 16 thread
rows; register tile `RPT = CPT = 8`, so `NCELL` 64 accumulators per thread
and a 128 x 128 output tile. Thread `tid` owns rows `accrow + u * 16`
(`accrow = tid // 16`, `u < 8`) and columns `acccol + v * 16`
(`acccol = tid mod 16`, `v < 8`). `KS` 16, `VEC` 4, `KV` 4, `SSTRIDE` 20.
`PAGES` 2 (page 20,480 B, two fit 49,152). `L` 128 gives 8 windows per leaf.

Counted per thread per window, full window:

| work | count | notes |
|---|---:|---|
| DRAM to registers, A | 8 words | NT/NN: 2 guarded 4-wide loads; TN: 8 scalar loads |
| DRAM to registers, B | 8 words | NT: 2 4-wide loads; NN/TN: 8 scalar loads |
| registers to shared | 16 words | 4 vector stores (p-contiguous) or 16 scalar |
| barrier | 1 | one per window at `PAGES == 2` |
| prefetch of window w+1 | 16 words | overlaps this window's arithmetic |
| shared to registers | 64 loads, 256 words | per `kc` (4): 8 A rows and 8 B lines, 4 wide |
| register copies | about 512 | `ra` 32 and `rb` 32 per `kc`, `bfl` 8 and `afl` 8 per step |
| FMA | 1,024 | 64 cells x 16 steps |
| FTZ multiply | 1,024 | one per FMA |
| leaf boundary (1 in 8 windows) | 64 `ftz`, 64 local stores, 64 local loads and 192 `ftz` per merge level | `_fold_push_local`, lane-wide |

Per block per window that is 4,096 staged DRAM words, 4,096 shared words
stored, 65,536 shared words loaded and 524,288 arithmetic instructions.

### 3.2 The per-cell inner loop and its dependency chain

Per cell per product step: one fused multiply-add and one FTZ multiply, two
instructions, in that order, on one accumulator. Contract 7.1 fixes one
accumulator per cell walking `p` ascending, so the chain per cell is
`2 L = 256` dependent instructions per leaf. The fold adds `P - 1` additions
per cell at depth `ceil(log2 P)` (under 1 percent of the arithmetic,
contract 13.3). Nothing in v1 allows a second accumulator inside a leaf. The
chain's length per cell is the contract's and no plan can shorten it.
Only the number of chains that progress at once can change (cells per
thread times threads times blocks per SM times SMs), and so can the
instructions a thread spends on things other than the chain.

The contract's arithmetic ceiling, counted. The attention brief's issue
model (`BRIEF_attention_step_2026-09-11.md` section 3.1) gives the H100 128
FMA lanes per SM. At 132 SMs and 1.98 GHz that is 33.5 T lane-instructions
per second. The step costs two instructions for its two useful flops, so
the ceiling for this contract is about 33.5 TFLOP/s at 1.98 GHz (half of the
67 TFLOP/s nominal FMA peak the GEMM handoff cites). The measured 11.4 is
34 percent of that. On the 1590 MHz pod, 9.5 against 26.9 is 35 percent.

### 3.3 Memory access

Contract section 3's strides (`gemm_operand_strides`):

| op | A_eff[i, p] | B_eff[p, j] | read along `p` |
|---|---|---|---|
| NT | `a[i k + p]`, row `i` of an `m x k` buffer | `b[j k + p]`, row `j` of an `n x k` buffer | both contiguous |
| NN | `a[i k + p]` | `b[p n + j]`, column `j` of a `k x n` buffer | B column-major through a row-major buffer |
| TN | `a[p m + i]`, column `i` of a `k x m` buffer | `b[p n + j]` | both column-major through row-major buffers |

The staging hides the stride from the inner loop. `_tuned_g2r` takes the
p-contiguous mapping (4 threads load one 4-word run of one line) or the
outer-contiguous mapping (consecutive threads take consecutive outer lines
at one `p`, so a warp loads a contiguous run). The DRAM words a block needs
are the same under both: `(BM + BN) k` per block walk, 128 A lines and 128 B
lines times `k`. Per useful flop that is `(BM + BN) / (2 BM BN) = 1/128` at
128 x 128. At the 1980 MHz ceiling that is about 89 G staged words per
second, about 356 GB/s of DRAM staging traffic, the same at gateup_fwd (n
768) and head_fwd (n 50,257), as any bound constant per flop would be.

Inside the block the page is a pack of the window, laid out by TILE line
(`row * SSTRIDE + step`, with 4 slack words per line the kernel never reads).
The accumulate loop reads it by OWNERSHIP: a thread's 8 A rows are 16 page
rows (320 words) apart and its 8 B lines likewise, in 64 four-word loads per
window. The 16 threads that share `accrow` load the same A rows, and the
16 that share `acccol` load the same B lines. So the page is packed, but not
in the order a thread reads it. Each thread copies 4 steps of 8 lines into
32-wide registers per `kc`, then copies again into `bfl` and `afl` per step.

### 3.4 Thread, block and resource geometry

Retained H100 readbacks: 255 registers per thread, 4,144 local bytes (4,096
of them the fold stack, 16 levels x 64 floats), 40,960 shared bytes, 44 bytes
of spill stores and loads, 256 maximum threads per block, one 256-thread
block per SM, 2,048 maximum resident threads per SM, 12.5 percent
theoretical thread occupancy
(`bench/results/gemm_resources_2026-09-10/README.md`,
`...gemm-ksplit-default/.../resources_lines.txt`). The ksplit group kernel
reads 255, 4,096, 40,960, one block per SM. Across the 2540 arms the count
moved with accumulators far less than one per cell: 64 cells 254 or 255, 32
cells 226, 16 cells 214. 255 against a 256-register budget reads like a
compiler cap on a demand above it, not a count of what the kernel needs.
That is a reading, not a measurement.

### 3.5 What the retained lines already say

| # | line | what changed | reading |
|---|---|---|---|
| E-a | gateup_dB TN 10.97 against down_fwd NT 11.31 (same shape and blocks) | staging mapping only | 2 to 3 percent |
| E-b | scalar shared-load trial (`gemm_resources_2026-09-10`) | 4 times the shared load transactions, 64 fewer load registers | flat within 0.6 percent; 255 registers unchanged |
| E-c | plan 16 against plan 9 (64 x 64, `KS` 32 against 16, two pages each) | half the windows per step | 4 to 7 percent faster |
| E-d | plan 17 against plan 10 (128 x 128, `KS` 32 at one page against 16 at two) | half the windows but no prefetch overlap | 17 percent slower |
| E-e | `lfold` at head_fwd (`...133216.../price-lfold.log`) | only the leaf-boundary fold, spelled as a runtime per-cell loop; same tile, same staging, same inner loop, same DRAM words | 2.76 times slower (3.45 against 9.53 TFLOP/s) |
| E-f | `half` 2.03, `quarter` 1.63 at head_fwd | fewer cells per thread, plus the `lfold` spelling | confounded |
| E-g | pod clocks 1590 against 1980 MHz | the box | TFLOP/s ratio about the clock ratio |

E-e matters most. The fold at a leaf boundary fires once per 8 windows. A
spelling change there, with the arithmetic and the DRAM traffic untouched,
cost more than all of the arithmetic of the leaf. So the time of a block is
not set by the arithmetic chain alone. Serialized non-arithmetic work
inside a thread can dominate it.

### 3.6 Competing causes, and what a line would show for each

None of these is measured as THE cause. Each names the price or resource
line that would separate it.

- **C1, arithmetic issue capacity.** The step is two instructions, and the
  block is at 34 percent of the lane ceiling. It predicts TFLOP/s following
  the SM clock (E-g, consistent). It predicts no change from anything that
  leaves the two instructions per step alone, which E-e contradicts. A
  price line that moves under an arm that keeps the step unchanged rules it
  out as the sole bound.
- **C2, DRAM staging words per flop** (`1/128` at 128 x 128). It predicts a
  constant staged-words rate across 128 x 128 calls (consistent), E-a flat
  (consistent), E-c flat (roughly), and E-e flat, which is contradicted.
  Separated by an arm that changes `(BM + BN) / (BM BN)` while keeping the
  per-window body: the price moves by at most that ratio on saturated calls
  (head_fwd, head_dB, gateup_fwd, down_dA).
- **C3, per-window non-arithmetic work in the thread.** That is the 512
  register copies, 64 shared loads and staging stores per window, the
  window bookkeeping and the barrier. It predicts sensitivity to windows per
  step (E-c, consistent) and to spelling outside the arithmetic (E-e,
  consistent in kind). E-b only partly separates it: more transactions and
  fewer registers were flat, while the copies stayed. Separated by an arm
  that removes the copies and most shared transactions at the same tile,
  the same staging words and the same arithmetic. The price line moves, and
  the resources line shows fewer registers or the same 255.
- **C4, the register cap and occupancy** (255, one block per SM). It
  predicts no change in price unless `blocks_per_sm_256` moves. At 64
  cells per thread no arm here can plausibly reach 127 registers (16 cells
  read 214). The resources line of any arm names it.
- **C5, local-memory traffic of the fold stack at leaf boundaries.** E-e
  shows a serial spelling there is expensive. The shipped lane-wide spelling
  issues 64 local stores per push and 64 local loads per merge level. No
  retained line separates the shipped push cost. A ksplit group of one leaf
  (proj_fwd PHASE, 576 blocks, `group_ms` 0.206) and the TUNED walk of 48
  windows (tuned128 proj_fwd, 96 blocks, 0.284 ms) both come to about 5 to 6
  µs per window with the same pushes per window.

## 4. The design (DEVIATION 2599)

Two trial arms, named in the arm framework beside `tuned128`, `ksplit` and
`ksplit_leaf`, compiled only under `-D MOJOLEARN_GEMM_ARM_TRIAL=1`, applying
only where `choose_gemm_plan` answers `PLAN_TUNED_128_8X8` (every call of the
step). Each arm runs the ksplit group structure where its group rule takes
the call (so it is compared against the shipped default on equal terms) and
the whole leaf range in one launch where it does not. Both use one new
kernel, `identical_gemm_kpack_kernel[RPT, CPT, TC, KS, FS, PAGES, GROUP,
SAB]`, a copy of the shipped per-window body with the page and the
accumulate loop changed as below. No shipped kernel, no shipped dispatch
line and no kernel matrix value changes.

### 4.1 `kpack`: the page packed in the order the accumulate reads it (targets C3)

Same tile (128 x 128), register tile (8 x 8), thread grid, `KS` 16, group
rule (`gemm_step_ksplit_rule` itself, so the same group sizes as the
default), fold, staging mappings (`_tuned_g2r` unchanged) and DRAM words.
Three changes, all in where operand words sit and how they are loaded:

1. **The packed page.** Line `r` (a tile row of A or a tile column of B) at
   window step `c` is stored at
   `gemm_kpack_addr(r, c, G, R, KS) = (r mod G) KS R + c R + (r div G)`,
   with `(G, R) = (TR, RPT)` for A and `(TC, CPT)` for B. A thread's lines
   are `g + u G` with `g` its own `accrow` (A) or `acccol` (B), so its `R`
   lines at step `c` are `R` consecutive words at `g KS R + c R`.
2. **Per-step loads.** Per step, one `R`-wide load of A and one `C`-wide load
   of B straight into the operand registers the step reads. There are no
   `ra`/`rb` copies and no per-element inserts, and 2 loads per step
   replace 16 per 4 steps (32 loads per window against 64).
3. **No slack words.** The page is `lines x KS` floats (16,384 B for both
   operands). `PAGES` comes from `lib_smem_pages_for` at that size plus a
   1,024-byte guard (cuobjdump reported 41,984 shared bytes for the shipped
   40,960-byte allocation), so two pages on NVIDIA and AMD and one on Apple.

The staging store writes the word `_tuned_g2r` put in slot `(s, e)` to the
packed address of the `(line, step)` that slot holds, `VEC` scalar stores
per slot where the shipped kernel issued one vector store. What a price
line would show: under C3, `kpack` faster on every call with regs at or
below 255; under C1, C2 or C4 alone, flat.

### 4.2 `kpack_wide`: more chains per thread, fewer staged words per flop (targets C2, reads C4)

`kpack` with the register tile widened to 8 x 16 (`RPT` 8, `CPT` 16, `TC`
16, so a 128 x 256 output tile), 128 accumulators per thread, and a local
fold stack of `GEMM_FOLD_LEVELS` (12) levels, 6,144 local bytes against
the 12,800 that 16 levels would take. Staged words per flop
`(128 + 256) / (2 x 128 x 256) = 1/171` against `1/128`, 1.33 times fewer.
`KS` is read through the kernel matrix like `TUNED_64_KS`: 16 where two
guarded pages of the 128 x 256 pack at `KS` 16 fit the column (AMD, 2 x
25,600 B within 64 KB), otherwise 12 (NVIDIA: 2 x 19,456 B within 48 KB, with
11 windows per 128-step leaf, the last one 8 steps on the ragged path;
Apple, where one page is used either way). Its group rule is section 4 of
the long-k brief counted on the 128 x 256 tile (`gemm_step_kpack_rule`):

| call | 128 x 128 (default, `kpack`) | 128 x 256 (`kpack_wide`) |
|---|---:|---:|
| proj_fwd, proj_dA | 1 leaf (96 tiles, 576 blocks) | 1 leaf (48 tiles, 288 blocks) |
| proj_dB | 1 leaf (36 tiles, 576 blocks) | 1 leaf (18 tiles, 288 blocks) |
| gateup_fwd, down_dA | declines (256 tiles) | 1 leaf (128 tiles, 768 blocks) |
| gateup_dA, gateup_dB, down_fwd, down_dB | 2 leaves (96 tiles, 768 blocks) | 1 leaf (48 tiles, 768 blocks) |
| head_fwd, head_dB | declines | declines (3,152 and 1,179 tiles) |
| head_dA | 64 leaves (7 groups, 672 blocks) | 32 leaves (13 groups, 624 blocks) |

What a price line would show: under C2 up to 1.33 times the TFLOP/s of
`kpack` at head_fwd and head_dB, less the extra barriers of `KS` 12 on
NVIDIA (E-c puts windows per step at a few percent). Under C4 binding, the
resources line reads regs 255 with more local spill bytes and the price
reads slower than `kpack`. `kpack_wide` against `kpack` isolates the wider
tile; `kpack` against shipped isolates the pack.

### 4.3 Candidates declined from the reading

- **A transposing pack of the operand in global memory** (materialize
  `A_eff` and `B_eff` p-contiguous before the launch). The staged DRAM words
  do not change, the pre-launch doubles the traffic, head_dB would copy
  about 412 MB per call, and E-a bounds what the mapping can be worth at 2
  to 3 percent.
- **Fewer registers at 64 cells to reach two blocks per SM.** 16 cells read
  214 registers (C4); nothing in source suggests 127 at 64.
- **A single hardware FTZ FMA per step.** Removing the round-then-flush seam
  already failed numerical admission (GEMM handoff).

## 5. The identity argument (written before the code)

### 5.1 The chain every stored cell must equal

For output cell `(i, j)`: `(L, P) = contract_partition(k)`. For leaf `t`
ascending, `acc = +0.0`, then for `p` ascending over `[t L, min((t+1) L, k))`,
`acc = S(A'[i, p], B'[p, j], acc)`, where `S` is `_tuned_step` (NVIDIA
`mul.rn.ftz(fma.rn(a, b, acc), 1.0)`, elsewhere `ftz(fma(a, b, acc))`) and
`A'`, `B'` are the operand words flushed as loaded (`_tuned_g2r`'s staging
`ftz` where `TUNED_STAGE_FTZ` is on, `_tuned_loaded_operand` at use where it
is off). The leaf partial is `ftz(acc)`. Partials enter `_fold_push_local`
in `t` order, the root comes from `_fold_drain_local`, and the stored cell
is `ftz(root)`. In group mode the group's node is stored unflushed and the
shipped fold kernels fold the `G` nodes (long-k brief Lemmas A to C, checked
exhaustively on the host). Every add of the contract is inside `S`, a fold
merge `ftz(ftz(slot) + ftz(val))` or a drain step, and each of them reads and
writes registers or thread-local memory of ONE thread for ONE cell.

### 5.2 Why packing cannot reorder a sum

The shared page holds operand words and nothing else. No float written into
it is a function of more than one operand word, because every store is a
copy of one register slot `_tuned_g2r` filled from one global word. No
partial sum, node or accumulator ever enters the page. So the layout of the
page can change which address an operand word occupies. It cannot change
which additions happen or in what order, because the additions are not in
the page.

What must hold is that every read returns the word the shipped kernel's
read returns. `gemm_kpack_addr(r, c) = (r mod G) KS R + c R + (r div G)` is a
bijection from `[0, R G) x [0, KS)` onto `[0, R G KS)`. Euclidean division
gives a unique `(r mod G, r div G)` with `r div G < R` because the tile has
`R G` lines. So distinct `(line, step)` pairs never share an address. The
store writes slot `(s, e)` at the address of the `(line, step)` that slot
holds under `_tuned_g2r`'s own mapping (the p-contiguous
`(idx // KV, (idx mod KV) VEC + e)` with `idx = tid + s 256`, or the
outer-contiguous `(idx0 mod lines, idx0 // lines)` with
`idx0 = tid + (s VEC + e) 256`, the same two expressions the shipped
register-to-shared copy uses). A thread reads its line `g + u G` at step `c`
from `g KS R + c R + u`, which is `gemm_kpack_addr(g + u G, c)` because
`g < G`. Every `(line, step)` a thread reads was staged by exactly one
thread of the block before the barrier, the same coverage argument as the
shipped kernel (the slot maps cover `[0, lines KV)` and `[0, lines KS)`).
The host check `check_kpack_page_is_a_bijection` walks all 256 threads,
every slot, both mappings, both operands and both geometries, and requires
exactly one store per address and the expected `(line, step)` at every read.

The page drops the 4 slack words per line. The shipped kernel never read
them: its highest full-path read is `kc VEC + VEC <= KS`, its ragged read is
`cc < chunk <= KS`, and its stores end below `KS`. Page bytes and `PAGES`
are execution plan (a kernel matrix row read). Which page a window uses,
`w mod PAGES`, and the barrier and prefetch placement are the shipped lines.

### 5.3 Why per-step loads and the wider register tile cannot reorder a sum

A load is not an operation on an accumulator. Per cell the accumulate is
`acc[cell] = S(afl, bfl[v], acc[cell])` with `afl = load(A', line, step)` and
`bfl[v] = load(B', line, step)`. The shipped full path takes steps in the
order `kc` ascending, `e` ascending, which is `c = kc VEC + e` ascending
over `[0, KS)`. The arm takes `c` ascending over `[0, KS)` directly. The
ragged path takes `cc` ascending over `[0, chunk)` in both. So each
accumulator sees the same `S` inputs, operand values by 5.2, in the same
order. Loading 8 words per step instead of 32 per 4 steps changes when
operand registers are filled, not what `S` reads or the order `S` runs on
one accumulator.

The wider register tile (`kpack_wide`) changes which thread owns a cell.
Thread `tid` owns rows `accrow + u 16` (`accrow < 16`, `u < 8`), covering
`[0, 128)` once, and columns `acccol + v 16` (`acccol < 16`, `v < 16`),
covering `[0, 256)` once, and tiles are an exact bijection of the grid. So
every cell has exactly one accumulator, seeded `+0.0` at each leaf boundary
and stepped once per `p`. Cells share no float, so updating 128 of them in
one thread instead of 64 cannot reorder any cell's additions (the
predecessor brief's section 5 argument, with `CPT` 16).

`KS` only cuts `p` into windows. `_tuned_window[KS]` yields windows at
offsets `0, KS, 2 KS, ...` inside each leaf, and each accumulates
`chunk <= KS` steps ascending. At `KS` 12 and `L` 128 there are 11 windows,
the last of 8 steps on the ragged path, and the leaf boundary fires on the
last. Any `KS` that is a `VEC` multiple gives the same step sequence per
cell.

### 5.4 The fold stack and the groups

`FS` is the stack's slot count. `_fold_push_local[NCELL, FS]` and
`_fold_drain_local[NCELL, FS]` are the shipped functions at every `FS`. A
push merges while bit `d` of `occ` is set and stores at the first clear bit.
`occ` counts the leaves pushed, at most `P <= CONTRACT_MAX_LEAVES = 1024`,
which needs slots 0 to 10, so `FS = 12` never overflows. The drain's extra
unoccupied levels do nothing. The same merges run on the same operands in
the same order. `check_group_fold_is_the_contract_tree` now also builds
every group node at `FS` 12 and requires its bits to equal `FS` 16's at every
`P` in 1 to 1,100 and every power-of-two group size.

The group rule reads `m`, `n`, `k` and a column row and returns a group size
(contract 6.1). Groups are powers of two aligned at leaf 0
(`_ksplit_resolve_leaves`), and Lemmas A to C hold for every group size
and every tile, because the tile appears nowhere in them. The group kernel's
window range is the ksplit kernel's (`[q 2^g wpl, min((q+1) 2^g, P) wpl)`),
and its node store, `_ksplit_fold_launch` and the synchronizes are the
ksplit path's lines.

### 5.5 What varies by vendor, and reach

`PAGES` and `kpack_wide`'s `KS` are read through `lib_smem_pages_for`, the
kernel matrix row `TUNED_64_KS` already reads. No vendor is named. A column
that resolves a different `KS` or `PAGES` stores the same bits by 5.2 and
5.3.

`SAB = True` is compiled only under the trial define. In all-leaves mode it
moves the stored cell of thread 0, cell (0, 0) of every block
(`_gemm_step_arm_sabotage`, the 2540 rule), so reach is one cell per tile.
In group mode it stores `1.0e30` for thread `q mod 256`, register cell
`q // 256` of block `(tile, q)` (the ksplit rule, on the arm's tile), so
reach is the number of `(tile, q)` whose cell is in the output.
`gemm_step_kpack_reach` counts both, and the checks require exact equality,
which also names which mode and tile ran.

## 6. What was built (written after the code)

Nothing ran. Nothing flips a default. The shipped dispatch is byte for byte
the lines it was. Every hunk in `gemm/checks/gemm_identical.mojo` sits in the
trial arm sections: the arm comment list after `identical_gemm_into`, the arm
and geometry ids, the arm selector functions, and a new section before
`_fast_vendor_gemm`. `identical_gemm_into`'s body, `identical_gemm_shipped_into`,
`choose_gemm_plan`, every shipped kernel and every kernel matrix value are
unchanged. The new kernel is referenced only under `comptime if
GEMM_ARM_TRIAL` and from the trial harnesses.

- **`gemm/checks/gemm_identical.mojo`.**
  - Arms `kpack` (10) and `kpack_wide` (11), geometries 10 and 11, both
    counts now 12, with parse, name, tile, geometry name and plan label
    entries. `gemm_step_arm_geometry` maps both arms to their geometry on
    every call `choose_gemm_plan` sends to the 128x128 plan, and
    `identical_gemm_step_geometry_into` dispatches them under the trial
    define. `gemm_step_geometry_group_leaves` and `gemm_step_geometry_reach`
    answer for them, so the launched-blocks and reach functions the
    harnesses print already cover them.
  - Constants: `GEMM_KPACK_PAGE_GUARD_BYTES` 1024; `GEMM_KPACK_*` (8, 8, KS
    16, FS 16); `GEMM_KPACKW_*` (8, 16, FS 12, BM 128, BN 256), with
    `GEMM_KPACKW_KS` 16 or 12 read through `lib_smem_pages_for`.
  - `gemm_kpack_addr`, `gemm_kpack_stage_p` and `gemm_kpack_stage_outer`:
    the one spelling of the packed address and of `_tuned_g2r`'s slot
    content, called by the kernel and by the host check.
  - `identical_gemm_kpack_kernel[RPT, CPT, TC, KS, FS, PAGES, GROUP, SAB]`,
    the body of sections 4 and 5.
  - Host side: `_kpack_launch`, `_kpack_run` (all leaves, or allocate,
    group launch, `_ksplit_fold_launch` and synchronize),
    `_kpack_geometry_run`, `identical_gemm_step_kpack_into`,
    `identical_gemm_step_kpack_phase_into`.
  - Rule and reach: `gemm_step_kpack_rule` (tile-parameterized),
    `gemm_step_kpack_leaves` (`kpack` calls `gemm_step_ksplit_rule`
    itself), `gemm_step_kpack_reach` and `_kpack_geometry_name`.
- **`gemm/checks/gemm_step_arms_check.mojo`.** `kpack` and `kpack_wide` are
  in the arm list, so the LM section runs both through `identical_gemm_into`
  with bits against the old plan and reach per call. The ragged part forces
  geometries 10 and 11 at every case with bits against the old plan and FLAT
  and exact reach, as it does every geometry. `check_group_fold_is_the_contract_tree`
  adds the FS 12 against FS 16 node parity. Two new host checks,
  `check_kpack_page_is_a_bijection` and `check_kpack_rule_hand_counts`, run
  before any device work. The banner names both geometries.
- **`bench/gemm_step_price_main.mojo`.** PRICE and TABLE lines for the arms
  come from the existing arm loop. Where an arm's rule takes a call, PHASEBITS
  and PHASE lines with `phase_of=arm_kpack` time the allocation, the packed
  group launch and the fold launch apart.
- **`bench/gemm_step_resources_main.mojo`.** Rows `kpack_all`, `kpack_group`,
  `kpack_wide_all` and `kpack_wide_group`, with registers, local, shared,
  const, max threads and blocks per SM.
- **`tools/gemm_kernel_leg.sh`.** The POSIX wrapper (passes `sh -n`), shown
  in section 9.

Departures from sections 4 and 5: none in the arithmetic. On the NVIDIA
column (`TUNED_STAGE_FTZ` on), the full-window accumulate reads the two load
registers directly. Elsewhere it copies B through `_tuned_loaded_operand`
per step, as the shipped kernel does, because that function is the operand
flush there.

## 7. What the checks prove, and what they cannot

- **Host, every column, before any device work.** The pack address is a
  bijection for both geometries, both operands, both staging mappings and
  both `kpack_wide` K steps. The kernel's staging helper names exactly
  `_tuned_g2r`'s slot content. Every read address holds the pair its reader
  expects. The 128x128 rule equals the shipped group rule at the twelve LM
  calls, and the 128x256 counts match section 4.2. Group nodes at FS 12
  equal FS 16 at every `P` in 1 to 1,100.
- **Device, the M4 run** (`MOJOLEARN_GEMM_STEP_CHECK_LM=0`, 50 M flop
  budget).
  - Both geometries store the old plan's bits and FLAT's on all three ops,
    ragged `m x n`, `k` in {0, 1, 128, 129, 300, 1000, 2049, 50257} under
    the budget, and the subnormal kind.
  - Reach is exact. The Apple trial row `S` is 0, so the rule takes the
    finest split. At 257x520 with `k` 129 and 300 the group launch runs;
    everywhere else the all-leaves launch runs, `k == 0` included.
  - Metal runs `kpack` on one page (34,816 B guarded exceeds 32 KB for two)
    and `kpack_wide` at KS 12 on one page.
- **Device, the H100 check** adds the 400 M budget and the twelve LM calls
  through the entry: two pages, `kpack_wide` at KS 12, group mode at the
  step shapes.
- **What no check proves.** That either arm is faster. That the kernel
  matrix's page count holds at the real driver limit with the guard (the
  launch would fail loudly). Anything about AMD at KS 16 for `kpack_wide`,
  which no box here runs.

## 8. RUN OWED, M4, light, run by the orchestrator one at a time

1. **The trial step check.**
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . gemm/checks/gemm_step_arms_check.mojo -o /tmp/gemm-step-check`
   then
   `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000 /tmp/gemm-step-check`.
   Expect:
   - `check_group_fold_is_the_contract_tree: ... 0 disagree; ... group nodes at FS 12 against FS 16, 0 differ`;
   - twelve `check_kpack_page_is_a_bijection [...] ... disagreements=0` lines and `check_kpack_page_is_a_bijection: 12 cases (...) 0 failures`;
   - twelve `RULE_KPACK` lines and `check_kpack_rule_hand_counts: 0 failures`;
   - `REACH ragged [kpack 128x128 ...] N/N` and `REACH ragged [kpack_wide 128x256 ...] N/N`, beside the existing geometries' lines unchanged;
   - PASS.
2. **The same check without the trial define.**
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_step_arms_check.mojo -o /tmp/gemm-step-check-notrial`
   then
   `MOJOLEARN_GEMM_STEP_CHECK_LM=0 MOJOLEARN_GEMM_STEP_CHECK_FLOPS=50000000 /tmp/gemm-step-check-notrial`.
   Expect FAIL, with `REACH ragged [kpack ...]: NOT RUN in N cases, bits
   and reach unproven (this build lacks -D MOJOLEARN_GEMM_ARM_TRIAL=1 ...)`
   and the same for `kpack_wide` beside the existing failure lines. The four
   host checks still pass there.
3. **The shipped device gates.**
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo -o /tmp/gemm-device-check && /tmp/gemm-device-check`.
   Expect `all green [IDENTICAL]  (8 gates, sabotage: none)`, unchanged
   (this lane touches no shipped path).
4. **The price harness build** (no timing run):
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_price_main.mojo -o /tmp/gemm-step-price`.
   Optional, host only, no device work:
   `MOJOLEARN_GEMM_STEP_LABEL_ONLY=1 MOJOLEARN_GEMM_ARM=kpack_wide /tmp/gemm-step-price`
   prints a `PLANLABEL arm=kpack_wide label=kpack_wide: packed page 128x256 reg8x16 KS=12 ...` line.
5. **The resources harness build** (no run):
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_resources_main.mojo -o /tmp/gemm-step-resources`.

## 9. The H100 leg

From a `git worktree add --detach` checkout of the commit that carries this
section, after section 8 is green. The Apple card comes first, at that commit:
`tools/gemm_card.sh device /tmp/gemm-kernel-apple.card`.

```sh
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_GPU_ARCHS=sm_90a \
MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_kernel_leg.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-gemm-kernel \
sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" --local-card /tmp/gemm-kernel-apple.card
```

The wrapper's body, `tools/gemm_kernel_leg.sh`, exports
`MOJOLEARN_GEMM_STEP_LEG_ARMS=shipped,kpack,kpack_wide`,
`MOJOLEARN_GEMM_STEP_LEG_LM_ARMS=kpack,kpack_wide`,
`MOJOLEARN_GEMM_STEP_LEG_LMTIMING=1` and
`MOJOLEARN_GEMM_STEP_LEG_CHECK_ARMS=shipped,kpack,kpack_wide` when unset,
writes under `/root/gemm_leg_out/gemm-kernel`, refuses an LM arm list of
`shipped` alone, then runs `sh tools/gemm_step_leg.sh`.

Read back from `<leg out>/remote/gemm-kernel/`:

- `status.tsv`: every item exits 0; `gate.txt` and `plans.tsv` name
  `shipped: default=ksplit(S=132) else tuned128` and both arm labels.
- `device_check.log` green and the NVIDIA card equal to the Apple card
  (no shipped path changed).
- `step-check.log`: PASS; `REACH ragged` N/N for both geometries; OK `LM`
  lines for `shipped`, `kpack` and `kpack_wide` on all twelve calls. The
  `kpack` group reach at head_dA is 672 (the shipped default's), and
  `kpack_wide`'s 624 (48 tiles by 13 groups). `kpack_wide` at head_fwd has
  all-leaves reach 3,152.
- `resources_lines.txt`: `kpack_all` and `kpack_group` against
  `shipped_128x128` and `ksplit_128x128` (C3: registers at or below 255);
  `kpack_wide_all` and `kpack_wide_group` (C4: registers, local bytes
  beyond the 6,144-byte stack, blocks per SM).
- `price_tables.txt` and `price_step.txt`: BITS EQUAL on every call and
  PRICE per arm. The saturated calls (head_fwd, head_dB, and gateup_fwd and
  down_dA for `kpack`) read the kernel's own TFLOP/s against section 2.
  PHASE lines with `phase_of=arm_kpack` give the group and fold phases where
  the rule took a call.
- `lm_summary.tsv`: `witnesses_equal_baseline`, `ratio_vs_shipped` and one
  verdict line per arm; `lmtiming-*` for the component breakdown.

**The flip rule** (ENGINEERING_RULES 9): the geometric mean of the enwik8
and pilegithub lean step ratios against the shipped default below 1, on the
same pod, with every step witness equal on both corpora. A flip changes only
the NVIDIA row. No such row exists yet. The flip would add one SCHEDULING row
(for example `lib_gemm_kernel_pack_for`, NVIDIA on, every other column off)
that `identical_gemm_shipped_into` reads to run the winning body in place of
the tuned and ksplit kernels on the calls they serve, with the same group
rule. That flip is its own commit after the verdict, not part of this lane.

Reading the two arms together:

| `kpack` | `kpack_wide` against `kpack` | what it says |
|---|---|---|
| faster | faster | C3 and C2 both bind; flip the faster one |
| faster | slower, regs 255 with spills | C3 binds; the wide tile runs into C4 |
| flat | faster | C2 binds, C3 does not |
| flat | flat or slower | none of C2, C3 set the rate at these shapes; C1 or C4 remain |

## 10. Risks only a build or a box can settle

- **Syntax and API never compiled.**
  - A Tuple returned by an inlined helper and reassigned inside a kernel
    (`var sla = ...; if ...: sla = ...`).
  - `unsafe_load[width=16]` from shared memory in a kernel (the width-16
    precedent in the repo is host code).
  - `SIMD[DType.float32, 128]` accumulators, and 128 inlined `_tuned_step`
    calls per step times 12 steps in the `kpack_wide` full path (compile
    time and PTX size).
  - `var` declarations inside `comptime if GROUP` blocks.
  - `_group_node[FS]` with a runtime `stack_allocation[FS * GROUP_NC, ...]`
    on the host.
- **Shared-memory limit at the edge.** `kpack` uses 32,768 B and `kpack_wide`
  36,864 B on NVIDIA, both under the 48 KB row with the 1,024-byte guard per
  page. The driver's real accounting is known only from one cuobjdump figure.
- **Registers and local memory of `kpack_wide`.** 128 accumulators per
  thread push past the 255 the shipped kernel reads. The compiler may
  serialize or spill, and a spilled accumulator would pay local-memory
  traffic per step (E-e is what that kind of traffic cost). The per-thread
  local limit on the H100 is not in the repository.
- **`kpack_wide` at KS 12 on NVIDIA.** 11 windows per 128-step leaf against 8,
  with the last window on the ragged path (a runtime step loop). E-c prices
  windows per step at a few percent; the ragged path's cost is unpriced.
- **Compile time on the pod.** A trial build now instantiates the packed
  kernel at two geometries, group and all-leaves, clean and sabotaged (8
  specializations) in the check, the price harness and the byte LM binding,
  plus 4 in the resources harness. If the lease is short, the knobs are
  `MOJOLEARN_GEMM_STEP_LEG_SKIP_RESOURCES=1` and
  `MOJOLEARN_GEMM_STEP_LEG_LMTIMING=0` through the runner, or a narrower arm
  list.
- **Per-call waits.** Where an arm's rule takes a call it allocates and
  synchronizes twice inside `identical_gemm_into`, exactly as the shipped
  default does on the same calls. `kpack_wide` takes gateup_fwd and down_dA
  too (36 more calls per step than the default), so its LM ratio carries
  36 more allocations and waits than its GEMM price does.
- **The section 3 model is a reading.** C1 to C5 are competing causes named
  from source and retained lines. None is measured as the cause, and the
  arms were chosen to separate them, not on a prediction that either wins.

## 11. First build, a width-12 register, and the fix (branch `lane/gemm-kernel-h100-fix`)

**What failed.** The M4 trial build of the step check, merged onto main,
stopped in the offload pass with
`SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<12, f32>'`.
The no-trial build compiled, and the check had built with the earlier trial
arms, so the lane diff was the source.

**The cause.** `identical_gemm_kpack_kernel` called
`_tuned_g2r[BSLOTS, VEC, KV, BN, NTH]` (the prologue, the two-page prefetch
and the one-page refill), and `_tuned_g2r` returns
`SIMD[DType.float32, SLOTS * VEC]`. For `kpack_wide` at KS 12, B has 256
lines and KV is 3, so `BSLOTS = (256 x 3 + 255) // 256 = 3` and the register
is 12 wide. Every column where two guarded 128x256 pages at KS 16 do not fit
takes KS 12, which is Apple and NVIDIA, so the H100 build would have failed
the same way. The other instantiations were already powers of two (A is 2
slots, 8 wide, at KS 12 and 16; `kpack` is 8 and 8; AMD at KS 16 gives B 4
slots, 16 wide). No other width in the lane can be 12. `FS` 12 sizes only
thread-local memory walked by runtime loops (`_fold_push_local`,
`_fold_drain_local`, `_group_node`), `KS` 12 feeds only integer arithmetic
(`_tuned_window`, `_tuned_windows_per_leaf`, page sizes) and runtime or
comptime loop bounds, and the loads are `RPT` 8 and `CPT` 16 wide.

**The fix.** `gemm_kpack_register_slots(slots)` gives the least power of two
at or above the slot count. The kernel instantiates `_tuned_g2r` at
`AREG = gemm_kpack_register_slots(ASLOTS)` and
`BREG = gemm_kpack_register_slots(BSLOTS)` (B at KS 12 is now 4 slots, 16
wide), asserts `VEC` is a power of two and that the padding covers the
slots, and leaves every other line alone. The staging stores still walk
`ASLOTS` and `BSLOTS`. `_tuned_g2r` itself, `KS`, the page, the loads, the
accumulate and the fold are unchanged, as is every shipped kernel and
dispatch line.

**Why no bit moves.** A slot's content in `_tuned_g2r` is a function of
`tid`, its own index, `VEC`, `KV`, the line count, the strides and the
window, never of the instantiated slot count, so slots below `BSLOTS` hold
the same words as before. A padded slot `s >= BSLOTS` is never staged, since
`idx = tid + s 256 >= BSLOTS 256 >= lines KV` under the p-contiguous mapping
and `idx0 >= BSLOTS 4 256 >= lines KV 4` under the outer-contiguous one. So
`_tuned_g2r` leaves it `+0.0`, no store visits it (the store loops stop at
`BSLOTS`) and no read touches it (the accumulate reads only the page). The
page therefore holds the same words at the same addresses, and section 5
holds unchanged. `check_kpack_page_is_a_bijection` now walks the padded
slots of every case and fails if one would stage a pair. Its lines add
`register_slots=`, which reads 4 for `kpack_wide B KS=12` and equals `slots`
everywhere else. The padded slots are dead in the kernel, so the compiler
should drop them. If the resources line for `kpack_wide` at KS 12 shows 4
more registers than expected, this is where they come from.

**Still unsettled by source reading.** The section 10 compile risks other
than this one (a Tuple reassigned inside a kernel, width-16 shared loads in
a kernel, 128-cell accumulators) are all power-of-two or front-end
constructs that the failed build had already type-checked. None is clearly
a compile failure, so they are unchanged and the next M4 build settles them.

## 12. The H100 leg (2026-09-11, measured)

Leg `bench/results/e1g/2026-09-11_191151-nvidia-h100-80gb-hbm3-gemm-kernel`,
commit e6ffb6f4 (section 11's fix merged), one RunPod H100 80GB HBM3 pod at
1980 MHz, run by `tools/gemm_remote_leg.sh` with `tools/gemm_kernel_leg.sh`
as the body, the pod terminated and verified gone (HTTP 404). Every
`status.tsv` item exits 0. The NVIDIA card equals the Apple card
(`RESULT: IDENTICAL`), `device_check.log` is green (8 gates), and
`step-check.log` passes with `REACH ragged` 117/117 for both `kpack` and
`kpack_wide`. Every call's price line reads `BITS ... EQUAL`. The attention
arm on this commit is still the previous NVIDIA default
`stash_tiled_fgrid_r32_qres_pf` for every arm, so the ratios isolate GEMM.

**Lean LM step**, steady medians in seconds, ratio against the shipped
default bracketed before and after on the same pod:

| arm | enwik8 | Pile GitHub | geomean | witnesses | verdict |
|---|---|---|---|---|---|
| shipped (open, close) | 0.2951, 0.2939 | 0.2942, 0.2939 | | reference | |
| `kpack` | 0.3657 (1.2418) | 0.3662 (1.2453) | 1.2436 | equal | NO FLIP |
| `kpack_wide` | 0.4110 (1.3954) | 0.4120 (1.4010) | 1.3982 | equal | NO FLIP |

**GEMM per step** (`price_step.txt`, per-call medians weighted by per-step
counts, a GEMM sum and not a step time): shipped against itself 142.5 ms
(ratio 1.0006), `kpack` 216.2 ms against 143.6 (1.505), `kpack_wide` 256.8 ms
against 143.3 (1.793). The `kpack` LM step grows by about 71 ms and its GEMM
sum by about 73 ms, so the whole LM loss is inside GEMM.

**Kernel rate on the saturated calls**, TFLOP/s shipped to arm (same-round
pairs, 7 rounds, medians):

| call | `kpack` | `kpack_wide` |
|---|---|---|
| head_fwd | 11.31 to 7.23 | 11.33 to 6.35 |
| head_dB | 11.29 to 8.23 | 11.26 to 7.01 |
| gateup_fwd | 11.33 to 7.23 | 11.37 to 6.00 |
| down_dA | 11.14 to 7.61 | 11.20 to 6.27 |

Every other call moves the same way. The one-leaf proj_* groups read 1.42 to
1.57 under `kpack` and 1.86 to 1.97 under `kpack_wide`. At head_dA, where the
rule takes the call, the `kpack` PHASE line puts the loss in the group phase
(`group_ms` 22.40 against 14.34), with the fold unchanged (0.068 ms).

**Resources** (`resources_lines.txt`): `kpack_all` and `kpack_group` read 255
registers, local 4,200 and 4,232 B, shared 32,768 B; `kpack_wide_all` and
`kpack_wide_group` read 255 registers, local 6,976 B, shared 36,864 B. Every
label, the shipped and ksplit controls included, reads
`blocks_per_sm_256=1`.

**Reading, against section 9's table.** No row of the table fits. Both arms
are slower, so neither C2 nor C3 is the rate limit in the direction the arms
could exploit. `kpack` removes the per-window register copies at the same
tile, staging words and arithmetic, and the rate falls by a third with the
same 255 registers and one block per SM, so C4 did not move either. The arms
replace per-window register copies with per-step loads from a packed shared
page (`load=per-step` in both labels), and `kpack_wide` also doubles the B
page and takes KS 12. A plausible reading is that a shared load per product
step costs the H100 more than the register copies it replaces, and that the
wider page and extra windows add to that. This is a reading of the price
lines and is not measured as the cause.

**Decision.** No flip. The NVIDIA GEMM plan stays
`shipped: default=ksplit(S=132) else tuned128`, no scheduling row is added,
and both arms remain trial only behind `MOJOLEARN_GEMM_ARM_TRIAL`. What is
left of section 3.6 after this leg is C1 (arithmetic issue capacity, which
tracks the SM clock) and C5 (fold stack traffic at leaf boundaries). Any
next GEMM arm should keep register staging per window and change something
else. It is owed, not started.

## 13. The page was bank-conflicted: `kpack_pad` (DEVIATION 2700, 2026-09-13, branch `lane/gemm-kpack-pad`)

SOURCE BUILT, NOT YET RUN ON A GPU. Section 12 read the `kpack` loss as "a
shared load per product step costs the H100 more than the register copies
it replaces". That reading skipped the address arithmetic. This section
counts it, from source.

### 13.1 The counting

`gemm_kpack_addr(r, c, G, R, KS) = (r mod G) KS R + c R + (r div G)`, and
the accumulate loop reads B at `bbase = acccol KS CPT`, one `CPT`-wide load
per step at `bbase + c CPT`. At the 128x128 geometry `KS CPT = 128` words.
`acccol = tid mod 16`, so the 32 threads of one warp hold 16 distinct
`acccol` values (twice each, at two `accrow`s that share A). Their B
addresses at step `c` are `acccol 128 + c 8`: sixteen addresses 128 words
apart. Shared memory on the H100 (and the MI300X, and Apple) is interleaved
over 32 four-byte banks, a period of 32 words, and `128 mod 32 = 0`, so all
sixteen land on the same bank group. A 16-byte load is served eight threads
per phase; eight distinct addresses on one bank group is an eight-way
conflict on every phase of every B load of every step. The A load is not
conflicted: the two `accrow`s of a warp are two addresses, broadcast to 16
threads each.

Per thread per window that is 16 B loads (two 16-byte phases each) at
eight-way conflict against the shipped kernel's 64 unconflicted 4-wide
loads. The measured rate fell by a third (section 12) with the arithmetic,
the DRAM words and the registers unchanged, which is the size of thing a
serialized shared load in the step loop does. `kpack_wide` at `CPT 16`
puts its groups `KS 16 = 192` (KS 12) or `256` words apart, both `0 mod
32`, the same conflict on a wider load, and lost more.

This is a count from source, not a measurement. What separates it from
section 12's reading is one arm that keeps everything of `kpack` and moves
only the group stride off the bank period.

### 13.2 The arm

`kpack_pad`: `identical_gemm_kpack_kernel` with a new parameter `PAD`
(default 0, so `kpack` and `kpack_wide` compile to exactly what they were).
Each line group is `KS R + PAD` words instead of `KS R`; the last `PAD`
words of every group are never stored or read. `GEMM_KPACK_PAD = VEC = 4`
makes the stride 132, `4 mod 32`, so the eight threads of a load phase take
eight distinct bank groups (`4 g mod 32` for `g < 8`, and again for `8 <= g
< 16`). Page bytes grow from 16,384 to 16,640 per operand pair; two guarded
pages still fit 49,152. Nothing else changes: same tile, same register
tile, same `_tuned_g2r` staging, same rule (`gemm_step_ksplit_rule` at
`GEMM_KSPLIT_S`), same fold, same `_tuned_step` in the same order. A
placement of the same words, so no bit can move (section 5's argument holds
verbatim, with `total = G (KS R + PAD)` in place of `G KS R`).

Plumbing: arm 14 `GEMM_ARM_KPACK_PAD`, geometry 14 `GEMM_GEOM_KPACK_PAD`,
name `kpack_pad`, in every place `kpack` is named (`gemm_identical.mojo`,
the arms check, the price and resources mains); `check_kpack_page_is_a_bijection`
takes a `pad` argument, walks the padded page for `kpack_pad` A and B under
both mappings (16 cases now), and requires every pad word to receive zero
stores and every data word exactly one. `tools/gemm_kernel_leg.sh` prices
`shipped,kpack,kpack_pad` (so the unpadded page is a same-pod CONTROL) and
runs the LM probe on `kpack_pad` alone.

### 13.3 What the leg decides

Same rule as section 9: the geometric mean of the enwik8 and Pile GitHub
lean step ratios against the shipped default on the same pod, every witness
equal. Three readings are possible on the price lines, and each says
something about section 3.6:

- `kpack_pad` at or below `shipped` on the saturated calls (head_fwd,
  head_dB, gateup_fwd, down_dA): the conflict was the whole `kpack` loss,
  C3 (per-window copies) is back on the table, and the LM verdict decides
  the flip.
- `kpack_pad` between `kpack` and `shipped`: the conflict was part of it;
  the remainder is the per-step shared load itself (section 12's reading),
  and the next arm keeps the padded page and moves the load out of the
  step (one `RPT`-wide and one `CPT`-wide load per `VEC` steps into
  registers, the shipped `kc` shape, on the packed page).
- `kpack_pad` equal to `kpack`: the count above is wrong for this hardware,
  and section 12's reading stands.

RUN OWED: the H100 leg of section 9 with this branch's commit.

### 13.4 The ceiling, corrected (this belongs in the plan, and is)

`docs/lanes/PLAN_next_2026-09-13.md` section 1 wrote the kernel at "about
15% of peak" against a 67 TFLOP/s fp32 FMA peak. Section 3.2 above counted
the contract's ceiling at 33.5 TFLOP/s at 1.98 GHz: the seam is TWO issued
instructions per product step (`fma.rn` then `mul.rn.ftz`), and the single
`fma.rn.ftz` was measured wrong at the smallest-normal boundary
(`archive/docs/lanes/HANDOFF_performance_followup_2026-09-09.md`, 262,144 adversarial
triples on L40S and H100). The shipped 11.4 is 34% of what this contract
can reach on this box, and a kernel rewrite under this contract is bounded
at about 3x, realistically 2x. The only lever on the ceiling itself is the
seam, which is a contract question (whether the three columns' native
flush-to-zero FMAs agree with one another at the boundary), not a kernel
one. Nobody has run that probe.

### 13.5 The H100 leg (2026-09-13, measured): the conflict WAS the `kpack` loss; `kpack_pad` NO FLIP at 1.014

Evidence: `bench/results/e1g/2026-09-13_150005-nvidia-h100-gemm-kpack-pad/remote/gemm-kernel/`
(RunPod `NVIDIA H100 80GB HBM3`, sm_90a, commit 8e4d4539, pod jqy0074cwzbngu
terminated and verified, 21 minutes of work on the box). `status.tsv`: all
26 items exit 0. `step-check.log`: PASS, the padded page 16 bijection cases 0
failures, every LM call `clean_moved=0`. The box's device card matched the
M4 card generated at the same commit at every stage.

**LM verdict** (`lm_summary.tsv`, lean step, 2 shipped brackets, every step
witness equal to shipped on both corpora):

| arm | enwik8 | Pile GitHub | geomean | verdict |
|---|---:|---:|---:|---|
| `kpack` (same pod, the unpadded CONTROL) | 1.3107 | 1.3056 | 1.3081 | NO FLIP |
| `kpack_pad` | 1.0133 | 1.0150 | 1.0142 | NO FLIP |

Shipped lean step 0.2317 / 0.2313 s; `kpack` 0.3044 / 0.3025; `kpack_pad`
0.2353 / 0.2352.

**GEMM sum per step** (`price_step.txt`): shipped against itself 1.000
(141.6 ms); `kpack` 1.502 (214.0 against 142.5); `kpack_pad` 1.023 (145.7
against 142.5). Per call, `kpack_pad` against shipped:

| call | ran | ratio |
|---|---|---:|
| head_dB | all leaves, 2,358 blocks | 0.955 |
| down_dA | all leaves, 256 | 0.972 |
| gateup_fwd | all leaves, 256 | 0.986 |
| head_fwd | all leaves, 6,288 | 0.996 |
| gateup_dB, down_dB, proj_dB | group launch | 1.024 to 1.034 |
| head_dA | group launch, 64 leaves | 1.043 |
| proj_dA, gateup_dA, proj_fwd, down_fwd | group launch | 1.052 to 1.066 |

`kpack` on the same pod: 1.37 to 1.63 on every call.

**Resources** (`resources_lines.txt`): `kpack_pad_all` 255 registers, local
4,200 B, shared 33,792 B, one block per SM; `kpack_pad_group` 255, 4,240,
33,792, one. The same registers and block count as `kpack` and as shipped.

**Reading.** Section 13.3's first case. Padding the stride by four words
took the `kpack` loss from 1.31 to 1.01 on the step and from 1.50 to 1.02
on the GEMM sum, with nothing else changed, so the eight-way bank conflict
was the `kpack` loss and section 12's "a shared load per product step costs
the H100 more than the register copies it replaces" is withdrawn: with the
conflict gone, a per-step load from shared costs about what the per-window
register copies cost, within a few percent either way. On the four
all-leaves calls the copy-free body is 0.4 to 4.5 percent FASTER than
shipped; on the group launches it is 2 to 7 percent slower, which is the
kpack kernel's scalar staging stores (16 per thread per window against four
vector stores) weighing more where a block walks 8 or 16 windows than where
it walks 48 or more. Neither is a flip.

What this settles about section 3.6: C3 is not the lever. The per-window
copies and the per-step shared loads are interchangeable at this rate, so
the block's time is not set by either. With C2 (DRAM words per flop, ruled
by the same-shape flat lines), C3 and C4 (the register cap, unchanged at
255 through every arm) off the table, what is left is C1 read as LATENCY
rather than issue: an extra shared load per step hid completely, which a
kernel at its issue limit could not have absorbed. The block runs eight
warps, two per scheduler, and each cell's chain is two dependent
instructions per step. The arm that separates latency-bound from
issue-bound is one that raises the resident warps per SM, which means a
second block per SM, which means at most 128 registers, and the 2540
resources lines say cutting accumulators alone does not reach it (16 cells
read 214). Where the other 100 or so registers go is a resources question
before it is an arm: the `_tuned_g2r` staging registers, the fold stack's
local addressing, or the 64-wide `acc` vector's spill pattern. Owed, not
started.

**Decision.** NO FLIP. `kpack_pad` stays trial only behind
`MOJOLEARN_GEMM_ARM_TRIAL`, beside `kpack` and `kpack_wide`. No shipped
line, no matrix row changes. The lane branch merges because its checks pass
on two vendors and its docs correct a wrong reading.

## 14. The seam itself: what each column's native FMA does at the boundary (DEVIATION 2701, 2026-09-13, branch `lane/gemm-seam-probe`)

### 14.1 Why this is the only lever on the ceiling

Section 3.2 counted the contract's ceiling at 33.5 TFLOP/s on the H100
because the seam `ftz(fma_rn(a, b, acc))`, round-then-flush, is two issued
instructions per product step on NVIDIA. The single `fma.rn.ftz` was
rejected on 2026-09-09 because at a=0x3f7fffff, b=0x00800000, acc=+0 it
returns 0 where round-then-flush returns 0x00800000: it flushes BEFORE
rounding (bench/results/attention_exact_fma_2026-09-09/README.md). The
same day's kNN audit found Apple's native FMA does the same pre-round
flush at that triple (bench/results/knn/2026-09-09-selector-final/apple/BOUNDARY.md),
repaired it for kNN only at a 39% cost, and left "other Apple FMA
consumers" as an open audit. The GEMM seam on Apple is one of those
consumers.

So the question is not "can NVIDIA be made to round-then-flush in one
instruction" (it cannot) but "what does every column's native FMA actually
do at the boundary, and do they agree with EACH OTHER". If they agree, a
contract whose seam is the native instruction is one instruction per step
on every column and the Apple gap closes with it.

### 14.2 The probe

`gemm/checks/gemm_seam_probe.mojo`: the 262,144 adversarial triples of
`transformer/checks/attention_fma_boundary_check.mojo` (22 edge words, 42
LCG words, full Cartesian product, operands and accumulator flushed as the
production path flushes them), through four lanes on the device: the
shipped seam (`_tuned_step`), the native FMA with no flush
(`identical_mul_add`), the hardware ftz FMA where the column has one
(NVIDIA's `llvm.nvvm.fma.rn.ftz.f`), and the software round-then-flush
spelling. Each lane is hashed (FNV-1a 64 over the words in triple order).
`tools/gemm_seam_probe_reference.py` recomputes `a b + acc` EXACTLY on the
host (rationals, RN-even to binary32, signed zeros, overflow) under three
semantics, round-then-flush (`rtf`, the contract), flush-before-round
(`fbr`) and no flush (`none`), hashes each, and names which one every
device lane equals. The device never sees the reference; a lane that equals
none of the three is reported as such. 315 of the 262,144 triples separate
`rtf` from `fbr`. `tools/gemm_seam_probe_leg.sh` is the on-box body; it
fetches no dataset.

### 14.3 Apple M4, measured (the same session, local)

Every lane, the SHIPPED seam included, hashes to `fbr`
(`f269fc70e5625987`), not to `rtf` (`62a6b5621e27c707`): all 262,144
triples, 0 mismatches between lanes. So on Apple the shipped GEMM seam is
flush-before-round at all 315 boundary triples, the contract's
round-then-flush is not what the Apple column computes, and the identity
cards agree across vendors only because no card and no LM witness has
landed on one of those 315 patterns. The host reference is validated by the
same line: its `fbr` hash equals the device's, byte for byte, over 262,144
words.

### 14.4 NVIDIA H100 and AMD MI300X, measured (2026-09-13, both legs the same hour)

Evidence: `bench/results/e1g/2026-09-13_161417-nvidia-h100-seam-probe/remote/seam-probe/`
(RunPod H100 80GB HBM3, pod odis4zq0db7yme terminated and verified) and
`bench/results/e1g/2026-09-13_161422-amd-mi300x-hotaisle-seam-probe/remote/seam-probe/`
(Hot Aisle MI300X gfx942, VM 71667ca0 deleted and verified). Both probes
built and ran in under ten seconds; neither fetched a dataset.

| column | shipped | fma (native, no flush) | hwftz | swrtf |
|---|---|---|---|---|
| Apple M4 | `fbr` | `fbr` | (= fma) `fbr` | `fbr` |
| NVIDIA H100 | `rtf` | `none` | **NONE OF THE THREE** | `rtf` |
| AMD MI300X | `rtf` | `none` | (= fma) `none` | `rtf` |

Hashes: `rtf` 62a6b5621e27c707, `fbr` f269fc70e5625987, `none`
aed7498f99e07f49; NVIDIA's `hwftz` lane eb76eb53d65e0007. Mismatch counts
against the native no-flush lane: `rtf` 1,120 (every subnormal result,
flushed), `fbr` 1,435 (those plus the 315 boundary triples), NVIDIA `hwftz`
1,170. So `fma.rn.ftz` flushes the 1,120 subnormal results and 50 of the
315 boundary triples, not all of them (a count inferred from the three
totals, not a listing; the 24 printed diffs are all subnormal flushes and
the literal a=0x3f7fffff b=0x00800000 acc=+0 case). Its rule is neither
round-then-flush nor flush-before-round on the exact value; it is something
internal to the unit. The literal boundary triple reads 0x00800000 on the
NVIDIA and AMD shipped lanes and 0x00000000 on Apple's.

### 14.5 Verdict: no single-instruction contract exists

The three native FMAs disagree with one another: Apple flushes before
rounding at every boundary triple, NVIDIA's hardware ftz FMA flushes at 50
of 315 by a rule of its own, AMD's default mode flushes nothing. There is
no semantics that every column computes in one instruction, so the seam
cannot be made one instruction per step on NVIDIA by changing the contract.
The ceiling of section 3.2 stands at 33.5 TFLOP/s. This closes the lever
the plan named "the only lever on the ceiling itself".

What the probe found instead, and what it is worth:

1. **Apple's shipped GEMM seam does not implement the contract.** It is
   `fbr` at all 315 boundary patterns of the triple set (14.3), the
   contract is `rtf`, and NVIDIA and AMD compute `rtf`. Cross-vendor
   identity on Apple is therefore conditional on no product landing in
   `[2^-126 - 2^-150, 2^-126)`, an event no card or witness has produced
   but which the contract promises against. The kNN repair of 2026-09-09
   is the known fix and cost 39% there. Whether GEMM on Apple takes the
   same repair, or the contract records the exception, is Andrew's call;
   it is a correctness item, not a speed one.
2. **AMD pays a software flush after every step.** Its native FMA keeps
   subnormals and the shipped seam spells `ftz()` as bit operations
   (bitcast, and, compare, and, compare, select) after the FMA. A
   two-instruction spelling (`v_cmp_class_f32` for the subnormal class,
   `v_cndmask_b32` to the signed zero) or a wave-mode flush at kernel
   entry, IF AMD's output flush is post-round (`rtf`), would cut the AMD
   seam from about six issued instructions to two or one. Unmeasured; the
   MI300X lean step is 1.198 s against the H100's 0.232, and this is one
   named piece of that gap. The mode probe is the same harness with one
   more lane.
3. **NVIDIA stays at two instructions**, and the kernel work of section 13
   (latency-bound, register census) is the remaining lever there.

## 15. The census: every "vector" shared load is scalar (DEVIATIONS 2702 and 2703, 2026-09-13, branch `lane/gemm-census`)

### 15.1 What the PTX says, read locally

Section 13.5 asked where 255 registers go. Before the H100 census
(15.3), the kernels were cross-compiled on the M4 to sm_90a PTX
(`mojo build --emit asm --target-triple x86_64-unknown-linux-gnu
--target-cpu x86-64-v3 --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA
-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I .
bench/gemm_step_resources_main.mojo`, the form the swizzle lane used on
2026-09-10; the shipped kernel's sidecar hashes to the SAME `f9dade9e`
module as the retained `bench/results/gemm_swizzle_2026-09-10/h100-current-128.ptx.gz`,
so the shipped kernel has not changed since). Per basic block of the
shipped 128x128 kernel, the accumulate window is

| block | instructions | fma.rn | mul.rn.ftz | ld.shared.b32 | ld.shared.v4 |
|---|---:|---:|---:|---:|---:|
| the full window | 2,323 | 1,024 | 1,024 | 256 | 0 |
| the ragged window | 149 | 64 | 64 | 16 | 0 |

and the whole module has **no vector memory operation of any kind**: 272
`ld.shared.b32`, 32 `st.shared.b32`, 96 `ld.global.b32`, 127
`st.global.b32`, zero `.v4` or `.v2`. The two shared pages are declared
`.shared .align 4`. So the "two 4-wide vector loads" and "4 vector stores"
of section 3.1 are scalar in the code the H100 runs: 256 shared load
instructions per thread per window against 2,048 arithmetic ones, 64
per `kc` where the source spells 16. `kpack` and `kpack_pad` are the same:
their `RPT`-wide loads lower to 8 scalar loads each, 256 per window.

Two retained readings change under this:

- E-b (section 3.5), the "scalar shared-load trial", compared scalar
  loads against a baseline that was ALREADY scalar, and read flat. It
  measured nothing.
- Section 3.3's shared traffic count (64 four-word loads per window) is
  256 transactions per thread per window, four times the count.

The cause is alignment, not the instruction count in source: a
`stack_allocation` in shared memory defaults to the element alignment (4
bytes), and NVPTX cannot emit `ld.shared.v4.f32` (16-byte aligned by
definition) from a load it cannot prove aligned, so LLVM splits every
vector load into scalars. The pages are 16-byte aligned in fact (the
runtime allocates them so) and every load start in the padded page is a
multiple of 4 words, but neither is stated.

The fold blocks are as section 3.1 counted: 1,485 and 1,472 instructions
per merge level (64 `ld.local`, 384 `selp`, 576 `and`, 384 `setp`, 64
`add`), a runtime loop over 16 levels with a 64-wide vector carried
across iterations. That loop is where the software `ftz` mass lives, not
the step, and it runs once per leaf. It is also the block with the most
64-wide temporaries alive at once, which is the register question 15.3
answers.

### 15.2 The arm: `kpack_padv` (DEVIATION 2703)

`identical_gemm_kpack_kernel` gains `ALIGN: Int = 4` (bytes): the two
pages are `stack_allocation[..., alignment=ALIGN]` and the per-step loads
are `load[width=RPT, alignment=ALIGN]`. `kpack_padv` = `kpack_pad` at
`ALIGN` 16 (`GEMM_KPACK_ALIGN`); arm 15, geometry 15, plumbed everywhere
`kpack_pad` is. Cross-compiled the same way, its sidecars carry 68
`ld.shared.v4.f32` and 0 `ld.shared.b32`, `.shared .align 16`: 32 loads
per window per operand pair where 2599's body issued 256. Same words, same
order, no bit can move; the M4 arms check and the H100 step check say so
before any price is read. The leg prices `shipped,kpack_pad,kpack_padv`
and runs the LM probe on `kpack_pad` and `kpack_padv`, so the alignment is
a same-pod A/B against the same page.

What it can and cannot show. The shared-load instructions drop 8x; if the
block was bound by issue slots or the LSU, the rate moves. If the block is
latency-bound at two warps per scheduler (section 13.5's reading), the
loads were already hiding and this reads flat, which is itself the
separation the census wanted. The global loads stay scalar in this arm:
`k = 50,257` rows are not 16-byte aligned, so a vector global load needs a
block-uniform alignment branch, a later arm if this one moves.

### 15.3 The H100 census (DEVIATION 2702)

`tools/gemm_kernel_census_leg.sh`: a generated driver compiles the shipped
tuned kernel, the ksplit group kernel, `kpack` and `kpack_pad` (all-leaves
and group) through `compile_function[kern, dump_asm, _dump_sass,
_ptxas_info_verbose=True]` and prints the runtime's own `NUM_REGS`, local,
shared and blocks-per-SM, then `ptxas --verbose` for spills and `cuobjdump
--dump-resource-usage` / `--dump-sass` per PTX. Launches nothing. The SASS
is what names the registers: the maximum register index used inside the
accumulate loop against inside the fold loop says whether the 255 is the
loop's or the fold's. RUN OWED at the time of writing (launched the same
hour as the `kpack_padv` build).

There is no register cap or launch-bounds knob in this Mojo: an
`@__llvm_metadata` annotation with `nvvm.minctasm` or `nvvm.maxnreg`
crashes the compiler (exit 139, tried locally), the stdlib carries no
`MAX_THREADS_PER_BLOCK_METADATA` symbol, and the PTX has no `.maxntid`. A
second block per SM can only come from fewer live values in source, which
is why the census comes before any occupancy arm.

### 15.4 The H100 leg (2026-09-13, measured): 8x fewer shared-load instructions moved the rate 1 percent

Evidence: `bench/results/e1g/2026-09-13_165544-nvidia-h100-gemm-padv-census/remote/gemm-kernel/` (RunPod H100 80GB HBM3, commit
b12f2a83, pod c4agpfxgw1hbla terminated and verified). `status.tsv`: all
26 items exit 0; `step-check.log` PASS with `kpack_padv` bit-equal on
every LM call; the box's device card matched the M4 card.

**LM verdict** (`lm_summary.tsv`, every step witness equal to shipped on both
corpora, 2 brackets):

| arm | enwik8 | Pile GitHub | geomean | verdict |
|---|---:|---:|---:|---|
| `kpack_pad` (scalar loads, same page) | 1.0115 | 1.0131 | 1.0123 | NO FLIP |
| `kpack_padv` (aligned, `ld.shared.v4`) | 1.0122 | 1.0112 | 1.0117 | NO FLIP |

**GEMM sum** (`price_step.txt`): shipped 1.000, `kpack_pad` 1.025,
`kpack_padv` 1.016. Per call `kpack_padv` is 0.3 to 1.5 percent faster than
`kpack_pad` everywhere, and against shipped it reads 0.966 (head_dB), 0.972
(down_dA), 0.977 (gateup_fwd), 0.990 (head_fwd) on the all-leaves calls and
1.014 to 1.058 on the group launches, the same shape as 13.5.

**Resources**: `kpack_padv_all` and `_group` 255 registers, one block per
SM, like every kernel in this brief.

**Reading.** Cutting the shared-load instructions from 256 to 32 per thread
per window, with the arithmetic, the DRAM words, the page and the registers
unchanged, bought about one percent. So the block is not bound by
shared-load issue slots or by the LSU either. Section 15.1's finding stands
as a fact about the code (every "vector" load is scalar) and as a small,
real gain, and it does not explain the 34 percent. What is left of section
3.6 is C1 read as the SCHEDULE of the two-instruction chain and the
occupancy that hides it, and C5, and both need the SASS, which this leg did
not get: the runtime's `_dump_sass` wrote empty files, and the image's
ptxas 12.4 refuses Mojo's PTX 8.5 ("Unsupported .version 8.5; current
version is '8.4'"), the failure the 2026-09-10 lane hit and fixed with
`pip install nvidia-cuda-nvcc-cu12==12.6.85` (bench/results/gemm_resources_2026-09-10/README.md).
The census body takes that ptxas next, and asks Nsight Compute for the
stall breakdown where the box allows it, which is the one instrument that
names latency against issue directly instead of by elimination.

**Decision.** NO FLIP. `kpack_padv` stays trial only, beside `kpack_pad`.
The aligned-page spelling is the right one for any future kernel body (it
is free and it is 1 percent), and the shipped kernel's own scalar loads are
a shipped-path change to make once a kernel-body arm is worth flipping.

### 15.5 The SASS census (2026-09-13, measured): ptxas re-vectorized the loads, the loop is clean, and the 34 percent is not in the loop's instruction count

Evidence: `bench/results/e1g/2026-09-13_171857-nvidia-h100-gemm-census2/remote/gemm-census/` (RunPod H100, commit 8c8a500c; PTX
and SASS per kernel in `dumps/`, `ptxas --verbose` 12.6.85 from the pip
wheel, the 12.4 cuobjdump reading its cubins). `tools/gemm_census_split.py`
reads it. The R2 staging of DEVIATION 2704 ran on this leg for the first
time: two corpora verified on the box in 21 s.

**Registers and spills (ptxas 12.6.85, sm_90a):**

| kernel | registers | spill bytes | stack | accumulate loop: insns / FFMA / FMUL / LDS.128 / max register |
|---|---:|---:|---:|---|
| shipped tuned 128 | 255 | 44 | 4,144 | 2,060 / 1,024 / 981 / 55 / R220 |
| ksplit group | 255 | 0 | 4,096 | 2,037 / 1,024 / 960 / 53 / R252 |
| kpack (all leaves) | 255 | 104 | 4,200 | 2,044 / 1,024 / 960 / 60 / R160 |
| kpack_pad (all leaves) | 255 | 104 | 4,200 | 2,044 / 1,024 / 960 / 60 / R160 |

Three things the SASS settles:

1. **ptxas re-vectorized the shared loads.** The PTX carries 256 scalar
   `ld.shared.b32` per window (15.1); the SASS carries 55 `LDS.128` in the
   loop and no scalar shared load. The assembler proved the alignment the
   front end could not. That is why `kpack_padv` bought one percent (15.4):
   at the SASS level there was nothing left to vectorize. Section 15.1's
   finding stands as a fact about the PTX and as a guarantee the aligned
   spelling makes explicit, not as a lever.
2. **The accumulate loop is clean.** 2,060 instructions for 2,048 math
   operations, 55 loads, no local-memory traffic, no barrier, no branch.
   The dependent pair is scheduled in groups of seven cells (shipped:
   seven FFMA, seven FMUL on the same registers, distance 8) or deeply
   interleaved (kpack: FMUL 109 instructions after its FFMA); the shared
   load reaches its first consumer after 31 (shipped) to 46 (ksplit)
   instructions. Nothing in the loop's own text is a 3x.
3. **The 255 registers are not the loop's.** The loop's highest register
   is R220 in the shipped kernel and R160 in the kpack kernels; R252 is
   used outside it, in the fold and the staging. So the register cap is set
   by the once-per-leaf fold spelling, which is the place a second block
   per SM would have to be won, and even then the kpack loop's R160 says
   the loop itself would need re-allocation to fit 128.

**What is NOT settled, and cannot be from static text.** Nsight Compute on
the RunPod container refuses the performance counters (`ERR_NVGPUCTRPERM`,
`ncu.log`), so the stall breakdown is unavailable there. Per block-window
the measured time is about 11,960 cycles against 4,120 issue cycles for
the loop's instructions on four schedulers; where the other 7,800 go
(barrier imbalance across the eight warps, the two-warp-per-scheduler
occupancy against the 4-cycle dependent latency, operand-bank stalls,
the staging head) is exactly what the counters would name. The next
instrument is a DIAGNOSTIC arm set behind its own define, never bit-checked
and never shipped: the loop with the FMUL removed, with the shared loads
replaced by register reuse, and with the barrier and staging removed, each
priced against the shipped kernel so the 7,800 cycles are decomposed by
subtraction. That is one H100 hour and the last question before the
kernel-body decision.

## 16. The diagnostic decomposition (DEVIATION 2705, 2026-09-13, branch `lane/gemm-diag`)

### 16.1 Why by subtraction

Section 15.5 left about 7,800 of the 11,960 cycles a block spends per
window unexplained by the loop's own text, and Nsight Compute is refused
in the RunPod container. So the window is decomposed by REMOVING one thing
at a time from the `kpack_padv` body (the cleanest body: R160 in the loop,
aligned vector loads, the padded page) and pricing what is left against
the unmodified body, same pod, same twelve LM calls, same group rule:

| variant | what is removed | what its price isolates |
|---|---|---|
| `base` | nothing (DIAG 0, the `kpack_padv` body) | the reference |
| `nomul` | the flush multiply: one instruction per step | the second instruction's issue cost, and whether the FFMA/FMUL chain is the stall |
| `noload` | the per-step shared loads: step-0 operands for all 16 steps | shared-memory latency and LSU time inside the loop |
| `nostage` | the staging stores, the barrier and the prefetch | the barrier's warp imbalance and the window head |
| `nofold` | the fold push at the leaf boundary | the once-per-leaf local-memory work |
| `floor` | all four | the FMA chain alone at two warps per scheduler: the latency floor |

`bench/gemm_step_diag_main.mojo` and `tools/gemm_diag_leg.sh`. EVERY VARIANT
BUT `base` COMPUTES WRONG BITS BY DESIGN; the variants exist only under
`-D MOJOLEARN_GEMM_DIAG=1` (a comptime assert refuses them otherwise), they
are not arms, geometries or candidates, the arms check never builds with
the define, and nothing here can reach a shipped line. Cross-compiled
sidecars confirm each variant drops exactly what it says (no `mul.rn.ftz`
in `nomul`, 8 instead of 68 `ld.shared.v4` in `noload`, no `bar.sync` in
`nostage`).

### 16.2 How to read it

Let `t` be `base`'s time and `t_x` a variant's. `t - t_x` is the cost of
the removed thing IN THIS SCHEDULE, upper-bounded (removing work also
frees issue slots). Readings:

- `floor` near `base`: the FMA chain itself is the time; the block is
  latency-bound at two warps per scheduler and only occupancy (a second
  block per SM, which means the fold's registers) or a shorter dependent
  chain per cell can move it. Under the contract the chain is fixed, so
  occupancy is the whole remaining lever.
- `nomul` far below `base` (near half): the two-instruction seam is the
  time and section 14 already closed that lever; GEMM on NVIDIA is done.
- `nostage` far below `base`: the barrier's imbalance across eight warps
  is the time, and a deeper prefetch or a split barrier is the next arm.
- `noload` far below `base`: the loop's shared loads are exposed after
  all, and software-pipelining the operand registers is the next arm.
- `nofold` far below `base`: the once-per-leaf fold is the time (E-e's
  reading), and the fold spelling is the next arm.

RUN OWED at the time of writing (launched the same hour, `--minutes 30`,
no dataset needed).

### 16.3 Measured (2026-09-13, two H100 legs): the staging phase is a third of the window and the fold is a fifth

Evidence: `bench/results/e1g/2026-09-13_174125-nvidia-h100-gemm-diag2/remote/gemm-diag/diag.txt` (RunPod H100, 1980 MHz before
and after, commit f66ab6e5; the first leg,
`bench/results/e1g/2026-09-13_173653-nvidia-h100-gemm-diag`, is retained
with its `nofold` and `floor` lines INVALID: resetting the accumulator
with no consumer let the compiler delete the whole FMA loop, which the
cross-compiled sidecars showed as zero FMAs; the rerun keeps the
accumulator live and stores it). Seven rounds, two warmups, medians;
`base` reproduces the priced `kpack_padv` line (145 against 145 ms).

**GEMM sum per step, each variant against `base`:**

| variant | removed | ratio | cost of the removed thing |
|---|---|---:|---:|
| `nomul` | the flush multiply | 0.824 | 18 percent |
| `noload` | the per-step shared loads | 0.982 | 2 percent |
| `nostage` | staging stores, barrier, prefetch | 0.666 | **33 percent** |
| `nofold` | the fold push at the leaf boundary | 0.799 | **20 percent** |
| `floor` | all four | 0.312 | 69 percent |

Per call the pattern holds everywhere: `nostage` 0.58 to 0.72, `nofold`
0.71 to 0.88 (0.74 on the three head calls, which fold 6 to 393 leaves),
`nomul` 0.79 to 0.85, `noload` 0.975 to 0.993. The four costs are
close to additive (73 percent summed against 69 measured).

**Reading.**

1. **The loop is not the time.** The shared loads are 2 percent and the
   second instruction 18, where an issue-bound loop would pay 50 for it.
   `floor`, the bare FMA chain, runs the sum in 45 ms, about 34 TFLOP/s,
   half the 67 one-instruction peak: that is the latency floor of two
   warps per scheduler, and it says a second block per SM would be worth
   up to 2x on the loop ITSELF, if the registers ever allowed it.
2. **The staging phase is a third.** Sixteen scalar shared stores per
   thread per window at a four-way bank conflict (the packed page puts the
   four threads that stage one line's four steps on one bank), 16 scalar
   global loads, and the barrier that holds all eight warps in the same
   phase so the FMA pipe idles while the LSU works. A staging that stores
   one 8-wide vector per thread, conflict-free under both operand
   mappings (each thread gathers the 8 lines `g + 16 u` at one step; the
   p-contiguous operand maps `g = tid div 16, step = tid mod 16`, the
   outer-contiguous one the transpose, so the loads coalesce and the
   stores land on distinct bank groups), cuts the stores 8x and the
   conflict to none. Bounded by the 33.
3. **The fold is a fifth**, from once per eight windows. Its per-level
   block is 1,485 instructions for 64 cells because `ftz` is spelled as
   six integer operations and the merge spells three of them per cell
   (`ftz(ftz(a) + ftz(b))`). On the NVIDIA column the hardware flush the
   step seam already uses, `mul.rn.ftz` by one, is one instruction and
   was measured equal to the software `ftz` on the 262,144 words
   (2026-09-09). The same spelling in the push, the drain and the leaf
   partial takes the per-level block from about 23 instructions per cell
   to about 5. Bounded by the 20. NVIDIA only; every other column keeps
   the software spelling, so no bit can move anywhere.

Both are placement and spelling, both keep every term and every order,
and together they bound at about 1.35x on the GEMM sum (142 to about 105
ms, 37 ms of the 343 ms step). They are the kernel-body work the plan
asked to size before opening, and they are the next arms (section 17).

## 17. The two arms the decomposition named (DEVIATION 2706, 2026-09-13, branch `lane/gemm-hfgs`)

### 17.1 `kpack_hf`: the hardware flush in the fold

`_fold_flush[HW]`: on the NVIDIA column with `HW`, `ftz(x)` is spelled as
`mul.rn.ftz(x, 1.0)`, the instruction the step seam has used since
2026-09-09. Identity: `x` is a stored binary32, so the multiply by one is
exact and its only effect is the flush of a subnormal to its signed zero,
which is `ftz`'s definition; the 2026-09-09 gate measured the pair equal on
262,144 words. `_fold_push_local`, `_fold_drain_local`, the leaf partial and
the output flush take `HW` from the kernel's `HWFOLD`; every other caller
(the shipped tuned kernel included) passes the default and keeps the
six-operation software spelling, as does every other column, so no bit can
move anywhere. Cross-compiled: the fold's `selp` count falls from 1,024 to
128 and `mul.rn.ftz` rises by the flushes. Bounded by section 16.3's 20
percent.

### 17.2 `kpack_gs`: gather staging

Under `GATHER`, thread `tid` stages line group `g` at window step `c`,
`(g, c) = (tid div 16, tid mod 16)` for a p-contiguous operand and `(tid
mod 16, tid div 16)` for an outer-contiguous one; it gathers the eight
lines `g + 16 u` at step `c` from DRAM (`_kpack_gather`: zero past the
matrix or the window, flushed at staging as `_tuned_g2r` flushes) and
stores them as ONE 8-wide vector at `g stride + c 8`, which is
`gemm_kpack_addr(g + 16 u, c)` for `u = 0..7`. The same words at the same
addresses as the slot mapping, so the accumulate loop and every existing
check are untouched; `check_kpack_gather_covers_the_page` walks the 256
threads under both mappings and requires every data word stored exactly
once and no pad word at all. Per thread per window: 16 scalar shared
stores at a four-way conflict become two `st.shared.v4` that are
conflict-free under both mappings (p-contiguous: a warp's 16 lanes at
consecutive steps write 128 contiguous words; outer-contiguous: consecutive
`g` at stride 132 words hit distinct bank groups). The global loads stay
scalar and coalesce under both mappings (16 consecutive `p` of one line,
or 16 consecutive lines at one `p`). Bounded by section 16.3's 33 percent,
of which the barrier's own skew is not touched.

### 17.3 `kpack_hg`: both

The two are independent placements and compose. The leg prices
`shipped, kpack_padv, kpack_hf, kpack_gs, kpack_hg` and runs the LM probe
on the three, so each is read against its base (`kpack_padv`) and against
shipped on the same pod. M4 arms check: all three 117/117 ragged reach,
every LM call bit-equal to shipped, the gather coverage 4 cases 0 failures.
RUN OWED at the time of writing.

### 17.4 The H100 leg (2026-09-13, measured): all three FLIP; `kpack_hg` takes the lean step 0.232 to 0.211 s

Evidence: `bench/results/e1g/2026-09-13_175602-nvidia-h100-gemm-hfgs/remote/gemm-kernel/` (RunPod H100 80GB HBM3, commit
851a0090, pod 4n8lu2a0jn38lx terminated and verified; corpora staged from
R2 in 16 s). `status.tsv`: every item exit 0; `step-check.log` PASS with
all three arms bit-equal on every LM call; the box's device card matched
the M4 card.

**LM verdict** (`lm_summary.tsv`, lean step, every step witness equal to
shipped on both corpora, 2 brackets; shipped 0.2319 / 0.2320 s):

| arm | enwik8 | Pile GitHub | geomean | verdict |
|---|---:|---:|---:|---|
| `kpack_hf` (hardware fold flush) | 0.9695 | 0.9697 | **0.9696** | FLIP |
| `kpack_gs` (gather staging) | 0.9462 | 0.9442 | **0.9452** | FLIP |
| `kpack_hg` (both) | 0.9101 | 0.9070 | **0.9085** | FLIP |

**GEMM sum per step** (`price_step.txt`, against shipped on the same pod):
`kpack_padv` 1.015 (the base, as before), `kpack_hf` 0.951, `kpack_gs`
0.913, `kpack_hg` **0.852** (143.2 to 121.9 ms). Per call, `kpack_hg`
against shipped: head_dB 0.740, down_dA 0.773, head_fwd 0.780, gateup_fwd
0.789, head_dA 0.813, the 2-leaf group calls 0.867 to 0.909, the 1-leaf
proj calls 0.916 to 0.951. The two arms compose almost exactly (0.951 x
0.913 = 0.868 against 0.852 measured). Resources: 255 registers, one
block per SM, unchanged.

**Reading.** The decomposition was right about where the time was and
about what a placement-only change could recover: the staging phase gave
back 9 of its 33 points on the sum and the fold 5 of its 20, with the
loop, the seam and the page untouched. This is the first kernel-body win
under the contract and the first flip since `ksplit`. What remains of the
staging phase is the barrier's skew and the scalar global loads; what
remains of the fold is its local-memory traffic and the level loop. Neither
is the 3x the plan once hoped for; both are further arms of the same kind.

**Decision.** FLIP `kpack_hg` on the NVIDIA column: the shipped dispatch
routes every call the TUNED 128x128 plan serves through the `kpack_hg`
body (group launch where the ksplit rule takes the call, all leaves
otherwise), selected by a kernel-matrix row that is 0 on every other
column, so Apple and AMD compile exactly what they compile today and the
row is the switch. Section 18 is the ship.

## 18. The ship (DEVIATION 2707, 2026-09-13, branch `lane/gemm-hg-flip`)

### 18.1 The switch

`checks/kernel_matrix.mojo::lib_gemm_kernel_body_for[column]`: 0 is the
2595 dispatch as it stood, 1 is the `kpack_hg` body. NVIDIA 1, every other
column 0, so Apple and AMD compile exactly the line they compiled before
and the AMD row waits for its own MI300X leg. In `gemm_identical.mojo`,
`GEMM_BODY_KPACK_HG` reads the row; `identical_gemm_shipped_into` runs
`_shipped_body_kpack_hg[False]` where it is on: every call
`choose_gemm_plan` sends to the TUNED 128x128 plan runs
`identical_gemm_kpack_kernel` with the padded aligned page, gather staging
and the hardware fold flush, the group launch plus the fold at the ksplit
row's group size where the rule takes the call and all leaves otherwise;
every other call keeps its plan. The trial hook's shipped sabotage runs
the same body with `SAB`, and `gemm_step_geometry_reach(SHIPPED)` names
that body's reach, so the arms check's shipped-entry case holds the new
default to exactly `gemm_step_kpack_reach` where the body runs and to
nothing where it does not. `identical_gemm_shipped_at_row_into` is
untouched: it remains the ksplit body at a named row, which is what the
check's three-row probe reads, and the row 0 case still proves the old
plan. The plan summary, the per-call dispatch name and the arm plan label
all say the body, so no probe row can confuse the two defaults.

Without `-D MOJOLEARN_GEMM_ARM_TRIAL=1` the shipped NVIDIA build now
compiles the kpack kernel (cross-compiled: two sidecars carry
`st.shared.v4`), and the row at 0 compiles none of it: 0 is the revert.

### 18.2 The gate

- M4: the arms check (row 0 on Apple: the shipped path is the old one and
  must stay bit-equal to FLAT; the new code must compile and the
  `kpack_hg` arm must still reach 117/117).
- H100, one leg: the gemm payload's device check and card, diffed against
  the M4 card (the new shipped path against Apple's, three-vendor identity
  through the two cards), then `tools/gemm_kernel_leg.sh` with `shipped`
  (the new body), `ksplit` (EXACTLY the 2595 default it replaced, the
  same-pod OLD reference) and `kpack_hg` (the arm it was flipped from,
  which must now read 1.00 against shipped: the proof the shipped path IS
  that body), price and LM on both corpora, every witness equal.
- AMD: nothing changes on the AMD column; its row stays 0 until an MI300X
  leg prices the gather staging there (the fold flush is NVIDIA's
  instruction and does not apply).

RUN OWED at the time of writing.

### 18.3 The gate (2026-09-13, measured): the shipped path is the body, the card agrees with Apple, the old default reads 1.045

Evidence: `bench/results/e1g/2026-09-13_183737-nvidia-h100-gemm-hg-flip/` (RunPod H100 80GB HBM3, commit 07707794, pod
xnyxf5gvysup87 terminated and verified; corpora from R2 in 16 s).

- **Card**: the gemm payload's device check on the NEW shipped path and its
  card against the M4 card: NO DIVERGENCE at any matched stage
  (`diff_apple_vs_nvidia.txt`). Three-vendor identity of the shipped body
  through the two cards (AMD's row is 0 and unchanged).
- **Step check** (`step-check.log`): PASS; the default dispatch section
  reads `column row=132 [shipped DEFAULT kpack_hg body ...]`, the
  shipped-entry case 24 OK 0 FAIL at the body's own reach, the three-row
  probe unchanged.
- **LM** (`lm_summary.tsv`, every step witness equal on both corpora):
  shipped 0.2106 / 0.2112 s (the previous pods' shipped read 0.2318 /
  0.2320); `kpack_hg` against the new shipped **1.0005** (the shipped
  path IS that body); `ksplit`, the 2595 default's own group launch with
  the tuned plan elsewhere, 1.0449 against the new shipped on this pod.
- **GEMM sum** (`price_step.txt`): `kpack_hg` 0.999; `ksplit` 1.081
  (131.6 against 121.8 ms). The `ksplit` ARM prices 131.6 where the old
  shipped default priced 143 on every pod of the day: the arm and the old
  default share the kernels but not every launch line, so the same-pod OLD
  reference here is a lower bound of the gain, and the cross-pod shipped
  numbers (0.232 to 0.211 on the step, 143 to 122 on the sum, a pod-to-pod
  drift under 1 percent all day) are the measurement.
- **One harness defect, fixed after**: the shipped noise-control price run
  (`price-shipped`) exited 1 after its first call because the PHASE line's
  second call site still passed the shipped geometry id to the kpack phase
  timer ("geometry 0 is not a kpack geometry"). `bench/gemm_step_price_main.mojo`
  now resolves the phase geometry once; `tools/gemm_price_shipped_leg.sh`
  reruns only that control. The verdict does not rest on it.

**SHIPPED.** The NVIDIA kernel body row is 1 on main; 0 is the revert.

### 18.4 The AMD row (2026-09-13, measured and gated on a Hot Aisle MI300X)

Two legs, both on the 8core MI300X VM in the rocm/dev-ubuntu-22.04 container, one heat
window each, commit bfde0442 for the price and this branch for the gate.

The price (`bench/results/e1g/2026-09-13_193949-amd-mi300x-hotaisle-gemm-amd-row`, on main):
`kpack_gs` FLIP geomean 0.9570 (enwik8 0.9559, Pile GitHub 0.9580) and `kpack_hg` FLIP 0.9585
(0.9573, 0.9597) lean step against shipped, GEMM sum 592 to 561 ms (0.947) and 559 ms (0.944),
every step witness equal to shipped on both corpora, step check PASS. On this column the body's
hardware fold flush compiles out (`comptime if HW and TUNED_HW_FTZ_FMA`, NVIDIA only), so
`kpack_hg` here IS the gather staging alone and the two arms price the same to 0.15 percent.
The row (`lib_gemm_kernel_body_for`) is 1 on AMD as on NVIDIA, one value and one code path.

The gate (`bench/results/e1g/2026-09-13_201129-amd-mi300x-hotaisle-gemm-amd-row-gate`), the same shape as 18.3 with the row flipped: the card under the new
shipped path diffed against the M4's (`identity_trace_diff.py`: IDENTICAL, same stage
sequence, same counts, same hashes), the step check PASS with the body's reach, and the
kernel leg with `shipped` (the new body), `ksplit` (exactly the row-0 dispatch at S=110 it
replaced) and `kpack_hg` (the arm it was flipped from): `kpack_hg` reads 1.0006 against
shipped (enwik8 1.0019, Pile GitHub 0.9993; GEMM sum 0.9994), the proof that the shipped path
is that body; `ksplit` reads 1.0342 (1.0366, 1.0318; GEMM sum 1.041), the old default slower
by the amount the price leg promised; every witness equal on both corpora.

## 19. The Apple seam is STILL LIVE, rechecked on 2026-09-17 (lane `lane/gemm-next`)

### 19.1 The recheck, measured

Section 14.3 measured the Apple GEMM seam as flush-before-round on
2026-09-13 and section 14.5 called it "a correctness item, not a speed
one". PLAN_next_2026-09-13.md repeats it. Nothing in the repository has
addressed it since: `git grep` for the `fbr` hash `f269fc70e5625987` finds
it in this brief alone, and the four files that name the item at all
(this brief, that plan, `gemm/checks/gemm_seam_probe.mojo`,
`tools/gemm_seam_probe_reference.py`) carry no fix.

So the probe was rebuilt and rerun, unchanged, at `origin/main`
(712eedd16) on the M4, one Metal slot, 0.294 s of GPU time, no dataset.
Evidence:
`bench/results/e1g/2026-09-17_162800-apple-m4-seam-probe-recheck/`.

```
SEAM_HASH lane=shipped fnv1a64=f269fc70e5625987
SEAM_HASH lane=fma     fnv1a64=f269fc70e5625987
SEAM_HASH lane=hwftz   fnv1a64=f269fc70e5625987
SEAM_HASH lane=swrtf   fnv1a64=f269fc70e5625987
SEAM_BOUNDARY a=3f7fffff b=00800000 acc=00000000 shipped=00000000 ...
```

and `tools/gemm_seam_probe_reference.py`, which recomputes the three
semantics exactly on the host and never sees the device:

```
reference rtf  fnv1a64=62a6b5621e27c707
reference fbr  fnv1a64=f269fc70e5625987
lane shipped   fnv1a64=f269fc70e5625987 -> fbr
boundary i=400 a=3f7fffff b=00800000 acc=00000000 rtf=00800000 fbr=00000000
```

All 262,144 triples, 0 mismatches between lanes, 315 boundary triples
separating `rtf` from `fbr`. The contract is `rtf`; the NVIDIA and AMD
legs of 14.4 read `rtf` on their shipped lanes from this same reference
script, so the check CAN fail and has been seen to come out the other way
on two columns. **The Apple shipped GEMM seam does not implement the
contract at those 315 patterns. STILL LIVE at 712eedd16.**

### 19.2 It is not a GEMM defect; GEMM is one consumer

`_tuned_step` off NVIDIA is `ftz(identical_mul_add(a, b, acc))`, and
`checks/numerics.mojo::identical_mul_add` is `std.math.fma`, the column's
native instruction. The outer `ftz` cannot restore a value the hardware
already flushed to zero inside the FMA, so every Apple IDENTICAL consumer
of `identical_mul_add` has the same exposure. 245 Mojo files call it. The
kNN audit of 2026-09-09 repaired ITS consumer only, and the repair is
still in the tree and is generic:
`neighbors/checks/zero_fma_boundary.mojo::repair_zero_fma(a, b, c, zero)`,
integer-only, no floating-point operation in the decision, called only on
the branch where the flushed FMA returned zero.

So the shape of a GEMM fix is known and small: on the columns whose
native FMA is `fbr`, spell the seam as the FMA plus a zero-result test
that calls `repair_zero_fma`. The cost is the per-step compare and branch
(the repair itself is `@no_inline` and almost never taken), not the 39
percent the kNN inner loop paid. It has not been priced here.

### 19.3 Why it is not being fixed in this lane without a decision

Changing the Apple seam changes Apple's bits at those 315 patterns and
therefore, in principle, every Apple reference cell. In practice no card
and no LM witness has ever landed on one of them (14.5), so the fix may
move zero published bits; that is checkable but only by a run of the
Apple column, which `lane/reference-regen` is already doing. Reported to
Andrew with the hashes rather than fixed silently, and to be sequenced
with that lane. `[[a-floor-in-prose-is-not-a-floor]]`: this paragraph is
not the record. The record is the probe and the reference script, which a
change walks past.

### 19.4 Two corrections to what the lane brief said before

- **The AMD kernel-body row is NOT 0.** Section 18.4 flipped it to 1 on
  2026-09-13 (gather staging, measured and gated on a Hot Aisle MI300X,
  GEMM sum 592 -> 559 ms). `checks/kernel_matrix.mojo::lib_gemm_kernel_body_for`
  returns 1 for NVIDIA and for AMD on main today. Apple's row is the only
  one still 0, and the fold flush would compile out there anyway, so what
  Apple is owed is the gather staging alone. Any handoff still saying "the
  Apple AND AMD rows are 0" is stale.
- **The one AMD lever section 14.5 named is still unmeasured**: the
  roughly six-instruction software `ftz` per product step where two would
  do, and the wave-mode flush question (whether AMD's output flush is
  post-round). That is untouched by the gather-staging flip.

## 20. What is actually left, sized from the record and from the compiler (2026-09-17, lane `lane/gemm-next`)

### 20.1 The AMD lever is the big one, and it is not the one the plan named

Section 18.4 already flipped AMD's kernel-body row to 1 on 2026-09-13, so the
gather staging IS on AMD and the "Apple and AMD rows stay 0" line of 18.1 is
true only of Apple. Any plan still saying AMD never got the kernel-body work is
reading section 18.1 and not 18.4.

What AMD did NOT get, and still has not got, is its seam. Section 14.5 point 2
called it "about six issued instructions" and unmeasured. It is now counted, off
the emitted gfx942 GCN, on the M4, with no GPU and no rental
(`bench/results/e1g/2026-09-17_163900-apple-m4-amd-seam-instruction-count`, and
445148a4c):

| column | issued instructions per product step | spelling |
|---|---:|---|
| NVIDIA | 2 | `fma.rn` then `mul.rn.ftz` |
| **AMD, shipped** | **8** | `v_fmac_f32` then `v_and`, `v_and`, `v_cmp_ne`, `v_cmp_eq`, `s_and`, `v_and`, `v_cndmask` |
| AMD, wave mode | **1** | `v_fmac_f32`, with one `s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0` per KERNEL |

The MI300X lean step was 1.198 s against the H100's 0.232 (14.5) and its GEMM
sum 559 ms after the gather staging (18.4), so about half the AMD step is GEMM
and every product step of it carries eight instructions where two would do on
NVIDIA and one would do here. **That is the largest single structural
difference between the two columns that anyone has counted**, and it is a
property of `ftz()` in `checks/numerics.mojo`, not of the GEMM kernel.

**It is an instruction count and not a time, and nothing here says otherwise.**
Section 15.5 measured 11,960 cycles per block window against 4,120 issue cycles
for the loop ON NVIDIA, i.e. that column's loop is not issue-bound, and section
16.3's bare FMA chain read 0.31 of the body. Whether AMD's loop is issue-bound
is UNMEASURED. But AMD's loop carries four times NVIDIA's instruction count per
step, so the two columns' answers cannot be assumed to be the same one, and
`lane/attention-speed` section 21.1 is the standing warning: the same arm read
0.8207 on one column and 0.9716 on the other.

### 20.2 The one question that decides it, and it is ten seconds

`s_setreg` is worth nothing unless AMD's hardware output flush is
ROUND-THEN-FLUSH. If AMD flushes before rounding, as Apple's FMA does (14.3,
19.1), the arm computes the wrong value at the 315 boundary triples and is a
DEFECT, not an optimization. Nothing about the instruction count changes that.

The harness is the existing seam probe with one more lane (e0b6582b0): a
separate `mode_seam_kernel` with its own buffer, launched after the four
existing lanes are downloaded, because `s_setreg` changes the mode for the rest
of the wave and setting it inside `seam_kernel` would silently change the lanes
computed after it. The containment is read out of the GCN and can fail: the
`s_setreg` appears in `gfx942_mode_seam_kernel.amdgcn` and is ABSENT from
`gfx942_seam_kernel.amdgcn`. Off AMD the lane names itself
`modeftz_NOT_SET_ON_THIS_COLUMN`. The M4 gate says the four existing lanes still
hash `f269fc70e5625987`, so the edit moved nothing.

`tools/gemm_seam_probe_column_leg.sh` (059d92c03) exists because
`tools/gemm_remote_leg.sh` exports `MOJOLEARN_GPU_ARCHS` to the box but NOT
`MOJOLEARN_TARGET_COLUMN`, while `tools/hotaisle_leg.sh` exports both. The lane
is `comptime MODE_LANE = TARGET_COLUMN == COLUMN_AMD`, decided at BUILD time, so
an unnamed column would have compiled the lane out and printed a line that looks
like an answer.

### 20.3 On NVIDIA the leftovers are small and the instrument for them is known

Section 15.5's arithmetic is the sizing, and it does not need redoing: about
11,960 cycles per block window against 4,120 issue cycles for the loop, so about
7,800 cycles are not the loop. Section 16.3 decomposed them against the
`kpack_padv` body (staging about a third, fold about a fifth) and section 17.4's
`kpack_hg` took both, for 0.852 on the GEMM sum. What 18.3 left named is the
BARRIER SKEW and the SCALAR GLOBAL LOADS in the staging head.

Neither can be sized from static text: 15.5 says so in its own words, and Nsight
Compute is refused in the RunPod container (`ERR_NVGPUCTRPERM`). Sizing them
means DIAG variants against the CURRENT shipped body, and
`bench/gemm_step_diag_main.mojo`'s base is `kpack_padv`, the body kpack_hg
replaced. So the NVIDIA leftover work is: re-base the diagnostic on the shipped
body, add a no-barrier and a vectorized-global-load variant, one H100 hour, and
only then decide whether an arm is worth building. That is a SMALLER and LATER
item than 20.1, and it is not a ninth scheduling arm: every variant computes
wrong bits by design, is behind `-D MOJOLEARN_GEMM_DIAG=1`, and never ships.

### 20.4 There is no dominant GEMM cell, and that ARGUES FOR the kernel

`lane/attention-speed` section 22.6 reads the per-leaf detail: nine of the
thirteen `gemm.<kind>` leaves sit between 13.0 and 14.4 ms and together are
123.1 ms of the 144.9. A reader could take that as bad news. It is the opposite,
because all thirteen kinds are ONE KERNEL at different shapes:
`identical_gemm_backward_a_into` and `identical_gemm_backward_b_into`
(`gemm/checks/gemm_backward.mojo:449` and `:503`) reorder operands and forward
to a single `identical_gemm_into`, whose own docstring says "Everything
numerical happens inside `identical_gemm_into`". So a one percent KERNEL win is
worth about 1.45 ms of the step where a one percent win on the largest single
cell is worth 0.14 ms, and the breadth is exactly what makes a kernel-body
change pay. It is also the reason the plan's "do NOT open a ninth scheduling
arm" is right: a scheduling arm can only move one shape's dispatch, and there is
no shape worth moving.

## 21. The itemization re-run, and the AMD answer (2026-09-17, both measured)

### 21.1 The step, itemized fresh: GEMM is 57.7 percent of it

Evidence: `bench/results/e1g/2026-09-17_203224-nvidia-h100-step-breakdown-after-flips/remote/step-breakdown/`
(RunPod H100 80GB HBM3, commit b2f4eea71, pod e8rs6eq63pwejl terminated and
verified gone, corpora staged from R2, every phase in `status.tsv` exit 0,
`witnesses.tsv` `all bits_identical True`). No new code: this is
`tools/step_breakdown_leg.sh` (DEVIATION 2630) re-run.

**Shares are of the REAL step, not the envelope.** The envelope is the timed
figure and it is 3.48 percent larger than the shipped step because of the
timers; every component below is rescaled onto the untimed step so the
percentages add to the thing being optimized.

| | enwik8 | Pile GitHub |
|---|---:|---:|
| timed envelope | 214.42 ms | 213.67 ms |
| **REAL step (untimed, shipped build)** | **207.21 ms** | **207.87 ms** |
| instrumentation | 7.21 ms (3.48%) | 5.80 ms (2.79%) |

| component | ms of the real step | share |
|---|---:|---:|
| **GEMM (every GEMM call)** | **119.50** | **57.7%** |
| attention kernels, launchers, regime scans | 60.12 | 29.0% |
| remainders (timer prints, host between ticks) | 4.35 | 2.1% |
| refusal and validation scans | 3.97 | 1.9% |
| RMSNorm forward | 2.96 | 1.4% |
| softmax, cross entropy, loss readback | 2.63 | 1.3% |
| embedding forward and backward | 2.49 | 1.2% |
| RMSNorm backward | 2.33 | 1.1% |
| attention rope and kv cache | 2.06 | 1.0% |
| ids upload, weight unpack, gradient pack | 1.97 | 0.9% |
| AdamW and its shadow copy | 1.67 | 0.8% |
| residual adds, copies, fan-in | 1.67 | 0.8% |
| swiglu backward / forward | 1.01 / 0.68 | 0.5% / 0.3% |
| binding admission, final wait, publish | 0.03 | 0.01% |

Against 2026-09-11 (144.90 / 116.23 of a 299.78 ms envelope), GEMM fell 21.3 ms
and attention 54.0 ms, so attention's flips have been the larger half of the
step's improvement and GEMM's share ROSE from 48.3 to 57.7 percent.

### 21.2 The ceiling each component implies, which is the sentence that matters

GEMM does 1.5180 TFLOP per step (two flops per product step, the twelve LM
call shapes at their per-step counts). In 118.18 ms that is **12.84 TFLOP/s,
38.3 percent of the 33.5 TFLOP/s contract ceiling** of section 3.2. On
2026-09-11 the same arithmetic read 10.68 TFLOP/s and 31.9 percent, so
`kpack_hg` and the wait removal moved the rate by a fifth.

- **GEMM at its contract ceiling: 45.31 ms, which takes 72.87 ms off the step,
  35.2 percent.** That is the absolute bound and it is not reachable.
- GEMM at 60 to 70 percent of the ceiling, which is what section 3.2 bounds a
  clean SIMT rewrite at: 75.5 to 64.7 ms, taking **42.7 to 53.5 ms off the
  step, 20.6 to 25.8 percent**.
- Attention halved would take 30.1 ms off the step, 14.5 percent.
- Everything that is neither, halved, would take 13.8 ms off, 6.7 percent.

So GEMM is both the largest component and the one with the largest measured
headroom, and no other single component can be worth more than about 15 percent
even if it went to zero. **The lane does not stand down.**

### 21.3 Where GEMM's own time is NOT evenly spread, and this is new

Section 20.4 and `lane/attention-speed` 22.6 read "no dominant cell" off the
2026-09-11 leg. At this commit the leaves have separated by RATE, and that is a
different reading:

| kind | ms | TFLOP/s |
|---|---:|---:|
| `proj_fwd` / `proj_dA` / `proj_dB` | 11.83 / 11.79 / 11.43 | **9.80 / 9.83 / 10.14** |
| `gateup_fwd` / `_dA` / `_dB` | 10.29 / 12.02 / 11.89 | 15.03 / 12.87 / 13.00 |
| `down_fwd` / `_dA` / `_dB` | 6.02 / 5.14 / 5.97 | 12.84 / 15.05 / 12.94 |
| `head_fwd` / `_dA` / `_dB` | 10.49 / 11.30 / 10.01 | 15.07 / 13.99 / 15.79 |

The three `proj_*` kinds are **35.05 ms at about 9.9 TFLOP/s while the other
nine run 12.8 to 15.8**. They are 144 of the step's GEMM calls, the shape is
2048 x 768 x 768, and the gap is the largest one left inside GEMM: at 13.0
TFLOP/s they would be 26.8 ms and at 15.0 they would be 23.2, i.e. **8.3 to 11.9
ms of the step, 4.0 to 5.7 percent**, without touching the kernel's arithmetic.

**THE CAUSE IS NOT ISOLATED AND IS NOT CLAIMED HERE.** Section 2 recorded these
calls as running the `ksplit` one-leaf path rather than the TUNED 128 plan the
kernel-body row serves, and attributed their 9.1 TFLOP/s to a fold launch on a
one-leaf group plus 576 blocks over 132 SMs in about five ragged rounds. That
would explain why `kpack_hg` did not move them. But the launch and sync counters
in this leg's own `breakdown.tsv` are not usable as a check (`gemm.down_dA`
reads 0.0 launches for 12 calls, `gemm.head_fwd` 0.0 for one, which cannot be
literally true), so the attribution is a HYPOTHESIS. What settles it is the
price harness's `PHASE` lines, which give `alloc_ms`, `group_ms` and `fold_ms`
per call and already exist: `tools/gemm_kernel_leg.sh`, one H100 hour, no new
code. That is the next NVIDIA measurement and it should be taken before any arm.

### 21.4 The AMD wave-mode lever is DEAD, and the negative found something larger

Evidence: `bench/results/e1g/2026-09-17_205022-amd-mi300x-hotaisle-seam-mode-probe-b/remote/seam-probe/`
(Hot Aisle MI300X VF, gfx942, `column=amd` written by the wrapper before the
build, build and run in 7 s, no dataset, VM 7a1b7c34 deleted and verified gone
1 s after, $0.10).

```
SEAM_HASH lane=shipped  62a6b5621e27c707 -> rtf
SEAM_HASH lane=fma      aed7498f99e07f49 -> none
SEAM_HASH lane=modeftz  eb76eb53d65e0007 -> NONE OF THE THREE
SEAM_MODE column=amd mode_set=True boundary=00000000
```

Setting `MODE.FP_DENORM` does NOT give round-then-flush. At the literal boundary
triple the mode-flushed FMA returns `00000000` where the contract requires
`00800000`. **So the eight-instructions-to-one lever of section 20.1 is a
DEFECT, not an optimization, and that question is closed.** It cost one minute
of MI300X to close, which is what the probe was for.

The negative carries the larger finding. `eb76eb53d65e0007` is EXACTLY the hash
NVIDIA's `fma.rn.ftz` lane produced on 2026-09-13
(`bench/results/e1g/2026-09-13_161417-nvidia-h100-seam-probe/remote/seam-probe/seam_probe.log`
line 7). **AMD's wave-mode flush and NVIDIA's hardware ftz FMA agree bit for bit
over all 262,144 triples.** Section 14.5 said the three native FMAs disagree,
which is true of the DEFAULTS; what it could not see is that two of the three
converge on ONE semantics once AMD's mode register is set, and that the odd
column is Apple (`f269fc70e5625987`, flush-before-round, section 19).

That is a CONTRACT observation and it is Andrew's, not this lane's: a
one-instruction seam exists on NVIDIA and AMD and computes the same bits on
both, and what stands between it and a doubled ceiling is Apple and the
315 boundary triples. **No contract change is proposed here.**

### 21.5 What AMD is still owed

The seam lever is dead but AMD's SPEED is still unmeasured at this commit. Its
kernel-body row has been 1 since 18.4, its step was 1.198 s against the H100's
0.232, and no step itemization has ever been taken on that column. The same
instrument used in 21.1 is vendor-agnostic and derives its own column, so an
AMD itemization is one `tools/step_breakdown_leg.sh` run on an MI300X with
`MOJOLEARN_STAGE_STRICT=1`. That is the AMD measurement worth buying next, and
it is a measurement, not an arm.

## 22. The `proj` fold, isolated; and the AMD column, itemized for the first time (2026-09-18)

### 22.1 RETRACTED 2026-09-18. THE CONCLUSION OF THIS SECTION IS WRONG. See 23.

**`group_leaves` IS NOT `P`.** This section read the `PHASE` line's
`group_leaves=1` as the contract's `P == 1` and concluded that the `proj` fold
performs no arithmetic and can be fused away for 4.0 percent of the step. That
is false. `_ksplit_resolve_leaves(group_leaves, p_count)` returns
`(group_leaves, ceil(p_count / group_leaves))`, so `group_leaves = 1` means ONE
LEAF PER GROUP, which is the MAXIMUM number of groups, the exact opposite of
`P == 1`. For `proj` at `k = 768` the contract's own `contract_leaf_count` gives
`P = 6`, which is what the same `PHASE` line prints as `groups=6`, and
`proj_dB` at `k = 2048` gives `P = 16`, printed as `groups=16`. The fold sums
six (or sixteen) real partials per cell through
`identical_gemm_fold_kernel`'s level-by-level tree. **Removing it would move
bits.** No call in the step has a `P == 1` fold, so contract 7.3 never applies
here and the proposed change has zero applicability.

The measurements below are correct and are kept. Only the reading of them was
wrong. Section 23 has the right one.

#### 22.1 as written (measurements correct, conclusion retracted)

Evidence: `bench/results/e1g/2026-09-18_013251-nvidia-h100-gemm-proj-phase/remote/gemm-kernel/price_tables.txt`
(RunPod H100 80GB HBM3, pod 20usq6mvqsq2u5 terminated and verified gone,
`MOJOLEARN_STAGE_STRICT=1`). No new code: `tools/gemm_kernel_leg.sh`, whose
`PHASE` lines decompose each call into workspace allocation, group launch and
fold launch, each host-synchronized, medians over 7 rounds.

| call | `group_leaves` | `group_ms` | `fold_ms` | `sum_ms` | fold share |
|---|---:|---:|---:|---:|---:|
| `proj_fwd` | **1** | 0.1939 | 0.0606 | 0.2575 | **23.5%** |
| `proj_dA` | **1** | 0.1939 | 0.0602 | 0.2576 | **23.4%** |
| `proj_dB` | **1** | 0.1955 | 0.0512 | 0.2500 | **20.5%** |
| `gateup_dA` | 2 | 0.4430 | 0.0702 | 0.5161 | 13.6% |
| `gateup_dB` | 2 | 0.4404 | 0.0698 | 0.5131 | 13.6% |
| `down_fwd` | 2 | 0.4459 | 0.0704 | 0.5196 | 13.6% |
| `head_dA` | 64 | 11.5375 | 0.0672 | 11.6074 | 0.6% |

**The fold is a roughly CONSTANT 0.05 to 0.07 ms per call whatever the leaf
count**, from one leaf to sixty-four. It is a fixed launch cost, not work. On a
0.25 ms `proj` call that fixed cost is a quarter of the call; on the 11.6 ms
`head_dA` it is six parts in a thousand. That is the whole of the `proj` rate
gap and it is not a property of the shape or of the kernel body.

**And every `proj` call has `group_leaves = 1`.** `gemm/IDENTICAL_FP32_CONTRACT.md`
section 7.3 is titled "`P == 1` performs NO fold addition" and says: "The tree
over one node has no internal node. The single leaf partial reaches the output
through seam 5g and through nothing else."

~~At 144 `proj` calls per step the fold costs 8.3 ms, 4.0 percent of the step,
and it performs no arithmetic.~~ **RETRACTED: it performs five additions per
cell at `proj` and fifteen at `proj_dB`. See 23.**

~~The change is to have the group kernel apply 5g and write `c` directly when
`leaves == 1`.~~ **RETRACTED. There is no such call. See 23.**

### 22.2 AMD, itemized: the step is 81 percent GEMM

Evidence: `bench/results/e1g/2026-09-18_013257-amd-mi300x-hotaisle-step-breakdown/remote/step-breakdown/`
(Hot Aisle MI300X gfx942, commit ce29b6185, `column=amd` derived on the box,
every phase in `status.tsv` exit 0, `witnesses.tsv` `all bits_identical True`,
`MOJOLEARN_STAGE_STRICT=1`). **No AMD step has ever been itemized before.** The
instrument of 21.1 is vendor-agnostic and derives its own column, so this is the
same `tools/step_breakdown_leg.sh` with nothing added.

| | AMD MI300X | NVIDIA H100 (21.1) |
|---|---:|---:|
| REAL step (untimed) | 687.97 ms | 207.21 ms |
| instrumentation | 6.94 ms (**1.01%**) | 7.21 ms (3.48%) |
| **GEMM** | **559.22 ms, 81.3%** | **119.50 ms, 57.7%** |
| attention kernels and scans | 91.57 ms, 13.3% | 60.12 ms, 29.0% |
| everything else | 37.2 ms, 5.4% | 27.6 ms, 13.3% |
| GEMM rate, same 1.5180 TFLOP/step | **2.72 TFLOP/s** | 12.84 TFLOP/s |

The two columns are running the same kernel over the same shapes and AMD is
**4.7x slower at it**. GEMM is therefore a much larger target on AMD than on
NVIDIA: 81 percent of that step against 58 percent of this one.

**The 33.5 TFLOP/s contract ceiling is an H100 figure** (section 3.2 derives it
from that part's issue model at 1.98 GHz). It MUST NOT be applied to the AMD
column, and no AMD ceiling is stated here because none has been derived.

### 22.3 The new discriminating fact is the FLATNESS, and it is a contrast, not an isolation

| | AMD spread | NVIDIA spread |
|---|---|---|
| per-kind TFLOP/s over the twelve LM calls | 2.51 to 2.85, **1.13x** | 9.80 to 15.79, **1.61x** |

On NVIDIA the rate varies by shape by more than half, and 22.1 has just shown
why: a fixed per-call launch cost that a short call cannot amortize. On AMD the
rate is very nearly **shape-independent**, which is what a fixed cost PER
PRODUCT STEP looks like rather than per call. AMD's seam is eight issued
instructions per product step where NVIDIA's is two (20.1).

**That is a consistent contrast and NOT an isolation.** Clock, CU occupancy,
LDS bandwidth and the memory path are not excluded, and `[[measure-before-attributing]]`
forbids naming the seam as the cause on this evidence. What it does do is raise
the value of the contract-preserving 8-to-5 spelling of 19b6a9c7f, because the
thing it would touch is 81 percent of the AMD step rather than 58 percent of the
NVIDIA one. The isolation is available and is the same shape as 22.1: price the
seam spelling against shipped on the twelve calls with the existing harness.

### 22.4 The ranking after both legs

1. ~~`proj` fold fusion, NVIDIA, about 4.0 percent of the step.~~ **RETRACTED
   2026-09-18, see 23: there is no `P == 1` fold anywhere in the step and the
   `proj` gap is structural, not waste. DO NOT BUILD IT.**
2. **The AMD seam spelling, 8 slots to 5**, on 81 percent of the AMD step. Needs
   the device proof of 19b6a9c7f first (the probe lane hashing `rtf`), which is
   ten seconds, and then a price leg.
3. The NVIDIA barrier skew and staging-head global loads, still unsized, still
   needing DIAG variants re-based on the shipped body.
4. The Apple seam repair (section 19), which is Andrew's call and
   `lane/reference-regen`'s table.

## 23. The `proj` fold is REAL, the gap is STRUCTURAL, and 22.1 is retracted (2026-09-18)

### 23.1 The error, named exactly

Section 22.1 read `group_leaves=1` off a `PHASE` line as the contract's
`P == 1` and concluded the `proj` fold adds nothing. **They are different
numbers.**

`gemm/checks/gemm_identical.mojo::_ksplit_resolve_leaves(group_leaves, p_count)`
returns `(group_leaves, ceil(p_count / group_leaves))`. `group_leaves` is
LEAVES PER GROUP. `group_leaves = 1` is therefore the MAXIMUM number of groups,
which is the opposite of one leaf overall. `P` is `contract_leaf_count(k)`, and
at the contract's `L = 128`:

| call | `k` | `P = ceil(k/L)` | the `PHASE` line's `groups` |
|---|---:|---:|---:|
| `proj_fwd`, `proj_dA` | 768 | **6** | 6 |
| `proj_dB` | 2048 | **16** | 16 |

The printed `groups` IS `P` whenever `group_leaves = 1`, which is the arithmetic
that should have been done before the claim. `identical_gemm_fold_kernel` then
folds those six (or sixteen) partials level by level through contract 7.2's
tree, so the `proj` fold performs **five real additions per cell**, and fifteen
at `proj_dB`. Contract 7.3 never applies: **no call in the step has `P == 1`.**
Removing or fusing that fold WOULD MOVE BITS. The proposed change has zero
applicability and must not be built.

`[[absence-in-one-file-is-not-never-measured]]` and
`[[measure-before-attributing]]` both cover this: the reading came from one
printed token and not from the function that produces it.

### 23.2 What the same evidence actually says, and it is a complete explanation

The `PHASE` lines split every call into its group phase and its fold phase.
Three arms of the same leg, medians over 7 rounds:

| call | output cells | `P` | `group_ms` | `fold_ms` | fold share | **group-only TFLOP/s** | whole-call TFLOP/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `proj_fwd` | 1,572,864 | 6 | 0.1939 | 0.0606 | 23.6% | **12.46** | 9.38 |
| `proj_dA` | 1,572,864 | 6 | 0.1939 | 0.0602 | 23.4% | **12.46** | 9.38 |
| `proj_dB` | 589,824 | 16 | 0.1955 | 0.0512 | 20.5% | **12.36** | 9.66 |
| `gateup_dA` | 1,572,864 | 8 | 0.4430 | 0.0703 | 13.6% | **14.54** | 12.48 |
| `gateup_dB` | 1,572,864 | 8 | 0.4404 | 0.0698 | 13.6% | **14.63** | 12.56 |
| `down_fwd` | 1,572,864 | 8 | 0.4459 | 0.0704 | 13.6% | **14.45** | 12.40 |
| `down_dB` | 1,572,864 | 8 | 0.4355 | 0.0696 | 13.7% | **14.79** | 12.68 |
| `head_dA` | 1,572,864 | 7 | 11.5375 | 0.0672 | 0.6% | **13.70** | 13.62 |

**Two facts, and together they close the question 21.3 opened.**

1. **`fold_ms` tracks the OUTPUT SIZE and not `P`.** Every call with 1,572,864
   cells folds in 0.067 to 0.070 ms whether it has 6, 7 or 8 partials, and
   `proj_dB`, which has the MOST partials (16) and the FEWEST cells (589,824),
   is the FASTEST fold of the eight at 0.0512 ms. The fold is an output-sized
   pass, near-independent of the partial count. That is why it looked
   "constant" in 22.1 and it is not evidence of a launch that does nothing.
2. **With the fold excluded, the rate is UNIFORM across every kind:
   12.36 to 14.80 TFLOP/s, including `proj`.** The `proj` kernel is not slow.

So the `proj` "rate gap" of 21.3 is **entirely the fold's share, and the fold's
share is large for `proj` because `proj` does the least arithmetic per output
cell of any call in the step**: `k = 768` into 1,572,864 cells, against `k =
2048` into the same cells for `gateup` and `down`, and `k = 768` into 102.9
million cells for `head_fwd`. Every call pays a fold sized by its OUTPUT; `proj`
amortizes that fixed cost over 2.7x less work than `gateup` does. **That is
structural, not waste, and there is nothing to remove.**

### 23.3 The one thing still worth checking, and why the 8.3 ms was an upper bound anyway

`fold_ms` is measured with the phase HOST-SYNCHRONIZED, as the `PHASE` line's
own trailer says. So it is a launch plus a full host round trip, and an H100
launch-plus-sync is itself in the tens of microseconds. The fold's traffic is
about 44 MB for a 1.57M-cell, 6-partial call, which at HBM speed is roughly
15 us against the 60 us measured. **So a large part of the measured `fold_ms`
may be the instrumentation's own sync rather than the shipped path's cost**, and
22.1's 8.3 ms was an upper bound in two independent ways before its reasoning
was even wrong.

Whether the shipped path pays a real 0.06 ms per call is answerable and is NOT
answerable from `PHASE` lines, which sync by construction. It needs the step
breakdown's `launches_per_step` and `syncs_per_step` for the `gemm.*` leaves,
which this lane already found unusable in the 2026-09-17 leg (`gemm.down_dA`
reads 0.0 launches for 12 calls). Fixing that counter is a smaller and more
honest item than any arm, and it is what a future lane should do first.

### 23.4 The ranking, corrected

1. **The AMD seam spelling, 8 issue slots to 5** (19b6a9c7f), on the 81.3
   percent of the AMD step that is GEMM (22.2). Device proof first: the probe
   lane hashing `62a6b5621e27c707`, ten seconds of MI300X.
2. The NVIDIA barrier skew and staging-head global loads, still unsized, needing
   DIAG variants re-based on the shipped `kpack_hg` body.
3. The Apple seam repair (19), Andrew's call, `lane/reference-regen`'s table.
4. Fix the `gemm.*` launch and sync counters so a fold's real cost can be read
   without a synchronizing harness (23.3).

**There is no cheap certain NVIDIA win left.** 22.1 appeared to be one for a few
hours and was not. The honest statement of where the NVIDIA column stands is
21.2's: GEMM is 57.7 percent of the step at 38.3 percent of its contract
ceiling, the ceiling is worth 35.2 percent of the step and is unreachable, and a
clean rewrite bounded at 60 to 70 percent of it is worth 20.6 to 25.8 percent.
Nothing smaller than that is now known to be available on this column.
