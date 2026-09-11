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
[HANDOFF_speed_gemm_2026-09-10.md](HANDOFF_speed_gemm_2026-09-10.md),
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
