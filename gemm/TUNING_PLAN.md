# The tuning plan for `mojolearn.identical.gemm.fp32.v1`

How to make the identical GEMM fast without moving one bit.

Written 2026-08-25 alongside `gemm/original/gemm_identical_tuned.mojo`.
**Neither file has been compiled and neither has been executed.** The lane
that wrote them had no execution rights of any kind -- no `mojo`, no `pixi`,
no build, no test, no benchmark. Everything in section 6 is a PREDICTION
recorded before measurement, in the discipline of `IDENTITY_COST_PLAN.md`
Part 1 and of `gemm/UNPINNED_CONTROL.md` section 5, and a prediction that
turns out wrong is the most useful thing this document can produce.

DEVIATIONS 1250 through 1264. The lane's range is 1250 through 1289 and the
unused numbers 1265 through 1289 are reserved for whoever builds, gates and
runs it.

**`gemm/original/gemm_identical.mojo` WAS NOT TOUCHED.** It is the shipped
profile, it is gated on three vendors, and it stays untouched precisely so
that it remains the reference the tuned file is compared against. Every edit
this round wanted to make to it is in `OWED, AND WHY I DID NOT DO IT HERE`.

---

## 1. The number this attacks, and the part of it that is already priced

`mojolearn.identical.gemm.fp32.v1` runs at 2 to 5 percent of FP32 peak on
Apple, on an H100 and on an MI325X. Vendor libraries reach 62 to 94 percent
at the same precision. That is 12x to 25x.

`gemm/UNPINNED_CONTROL.md`'s single-variable arm split part of it, measured
today on an M4 at the tiled llama8b `t512` rows.

| ratio | value | what it is |
|---|---|---|
| pinned / unpinned | 1.55x | the whole cost of the fold-order pin |
| pinned / strict (`NACC = 1`) | 1.25x | the SEAMS alone -- `ftz` plus refusing FMA contraction |
| unpinned / strict | 1.24x | the instruction-level parallelism the pin forbids |
| the fold tree | ~0x | `P == 1` rows show the same 1.26x to 1.55x with NO tree present |

**So the pin is 1.55x of a 12x to 25x gap, and the remaining 8x to 16x is
kernel engineering.** This document is the kernel engineering, and it is
also -- section 4 -- the discovery that one part of the remaining gap is not
engineering at all but a second, unpriced cost of the pin.

### 1.1 One inference from those numbers that shapes every prediction below

Count the pinned inner step. Two shared loads, two operand flushes, one
`fma`, one accumulator flush. `gemm/UNPINNED_CONTROL.md` section 5.1 prices
a flush at roughly four scalar operations, so the pinned step is about one
FMA plus fourteen scalar operations and the strict step is about one FMA
plus two loads. **The pinned kernel issues roughly five times the
instructions and takes 1.25x the time.**

A kernel whose instruction count can quintuple for a 25 percent slowdown is
not instruction-issue bound. It is stalled on something else -- shared
memory latency, the dependent FMA chain, or DRAM -- and roughly eighty
percent of its issued work is being hidden inside those stalls.

That is the single most useful fact this round has, and it cuts both ways.
It says every technique below that only removes INSTRUCTIONS (the operand
flush hoist, the vectorized loads) will disappoint relative to its op count.
It says every technique that attacks the STALL (register blocking, double
buffering, the wider tile) is where the round's money is. Section 6's
predictions are written to that shape and can be falsified as a group if the
run comes back the other way around.

---

## 2. The one constraint, stated as an operational test

**The tuned kernel must produce output bit-identical to `gemm_oracle` at
every shape, exactly as `gemm_identical.mojo` does today.**

The pin fixes WHICH partial sums combine and IN WHAT ORDER. It says nothing
about when bytes arrive or where they sit. So the test each technique below
has to pass is one question.

> Does this change the multiset of values entering any accumulator, or the
> order in which they enter it, or the set of `ftz` applications any value
> passes through?

If the answer is no at every shape including the ragged and degenerate ones,
the technique is available however violently it rearranges memory. If the
answer is yes anywhere, it is a v2 of the contract and not a tuning knob,
and no measurement licenses taking it (contract 13.7's last bullet).

### 2.1 The four things that are NOT available, and were not taken

| forbidden | contract | what a tuner reaches for, and why it is refused |
|---|---|---|
| changing the k partition | 6, 6.1, 6.2 | Split-k selection is the one autotuning axis the pin genuinely removes. `contract_partition`, `contract_leaf_size` and `contract_leaf_count` are IMPORTED by the tuned file and never re-derived; `K_LEAF_MIN` and `MAX_LEAVES` are profile constants and changing either is a revision of the contract. |
| reassociating within a leaf | 7.1 | Four `Veclen` lanes accumulated into four registers and added at the end is a balanced tree of depth 2 hidden inside one leaf. The tuned kernel vectorizes LOADS and never ACCUMULATORS -- see 3.3. |
| dropping `ftz`, or letting the compiler contract | 4, 4.1, 5 | These cost 1.25x and they are the contract, not an inefficiency. Section 8.3 records where a seam MIGHT be provably inert and why this round did not act on it. |
| changing the fold tree or its carries | 7.2, 7.2.2 | `_fold_push_tile` is a lane-by-lane replica of the shipped `_fold_push`, same expression, same `occ` arithmetic, same merge condition. DEVIATION 1263 says it is argued and not yet proven. |

---

## 3. The techniques, one section each

Each carries what it changes, why it cannot move a bit argued from the
contract's own clauses, the expected speedup, and how it could go wrong.

### 3.1 Register blocking -- one thread computes `RPT x CPT` output cells

**DEVIATION 1251. This is the single biggest lever and it is also the one
the contract most nearly forbids, so the argument is worth getting exactly
right.**

WHAT IT CHANGES. `identical_gemm_tiled_kernel` gives one thread one output
cell. Each `k` step it issues two threadgroup loads, two operand flushes,
one `fma` and one accumulator flush. The tuned kernel gives one thread
`RPT * CPT` cells arranged as a rectangle of the block's tile, so the `RPT`
A values and `CPT` B values it loads per `k` step serve `RPT * CPT` FMAs
instead of one.

WHY IT CANNOT MOVE A BIT. Contract 7.1 forbids sub-partitioning ONE leaf of
ONE cell into several accumulators, and names register tiling as the clause's
likeliest violator. It does not, anywhere, restrict how many CELLS a thread
owns. The distinction is exact and it is the whole argument.

    forbidden (7.1)   one cell, four accumulators over disjoint k ranges,
                      summed at the end -- a depth-2 tree inside a leaf
    taken here        sixteen cells, one accumulator each, every one of them
                      the contract's serial ascending chain over the WHOLE
                      leaf, seeded +0.0, flushed after every step

Cell `(i, j)`'s accumulator sees exactly the products `A_eff[i,p] *
B_eff[p,j]` for `p` ascending through its leaf, one `identical_mul_add` per
step, and nothing else. The other fifteen cells in the thread are fifteen
other output cells and no value ever moves between them. Contract 0.3 and
6.1 say the arithmetic for cell `(i, j)` may not depend on how many other
cells shared the launch; register blocking changes exactly that number and
nothing else.

**AND IT BUYS BACK, LEGALLY, THE TERM THE UNPINNED ARM BOUGHT ILLEGALLY.**
The unpinned control measured `unpinned / strict = 1.24x` and section 2 of
that document attributes it to `NACC = 4` independent accumulator lanes --
instruction-level parallelism, bought by breaking 7.1. A 4x4 register tile
puts SIXTEEN independent FMA chains in flight per thread without touching
7.1, because they are sixteen different cells. The pin forbids ILP in `k`.
It does not forbid ILP in `m` and `n`.

EXPECTED SPEEDUP. Two channels, and they are different.

- Shared-memory reads per FMA fall from 2.0 to `(RPT + CPT) / (RPT * CPT)`,
  which is 1.0 at 2x2, 0.75 at 4x2 and 0.5 at 4x4 -- a 2x, 2.67x and 4x
  reduction in threadgroup traffic per flop. With `VEC = 4` shared reads the
  LOAD INSTRUCTION count per FMA falls from 2.0 to 0.125 at 4x4, a factor of
  16.
- The dependent-chain stall is covered by `RPT * CPT` independent chains.

Predicted 1.4x to 2.2x on its own at 4x2, more at 4x4 where the tile fits.

HOW IT COULD GO WRONG.

1. **The accumulators land in memory instead of registers.**
   `bench/results/GEMM_ROUND_2026-08-19.md` is this repository doing exactly
   that -- `stack_allocation` with no address space is thread-local MEMORY,
   so every accumulator access became a load and a store and the register
   tiling came out SLOWER than the naive 16x16 kernel across the board.
   Every accumulator in the tuned file is a lane of a `SIMD` value indexed
   by a `comptime for` bound, which is the spelling that fixed that round
   and the spelling `cluster/derived/distance/fused_distance_nn/simt_kernel.
   mojo` ships today. It is not a guarantee. Section 7's first gate is a
   register or occupancy readout, and without one this failure mode looks
   exactly like "the technique did not help".
2. **The tile starves the machine.** A 64x64 tile at `n = 64` produces
   `ceil(m/64)` blocks where a 16x16 tile produces sixteen times as many. On
   an M4 that is fine and on an H100 with 132 SMs it may not be. Predicted
   as a possible REGRESSION at the narrow-`n` rows, section 6.4.
3. **The fold stack eats the register budget.** Section 4. This is the one
   that actually bit.

### 3.2 Double buffering the k-window staging

**DEVIATION 1256.**

WHAT IT CHANGES. `identical_gemm_tiled_kernel` stages a `KS`-wide window
into one threadgroup page, barriers, accumulates, barriers, and only then
begins the DRAM read for the next window. Two barriers per window and no
overlap between the memory system and the ALUs. The tuned kernel allocates
`PAGES` pages and runs

    R->S window w into page w%2 ; barrier ; G->R window w+1 ; compute page w%2

so the DRAM read for window `w + 1` is in flight while window `w` is
computed, and there is ONE barrier per window instead of two. The write of
page `(w+1)%2` at the top of the next iteration can only conflict with the
compute of iteration `w - 1`, and every thread finished that before the
barrier at the top of iteration `w`, which is what makes one barrier
sufficient.

WHY IT CANNOT MOVE A BIT. A staging copy is a bit-exact copy, which is
`identical_gemm_tiled_kernel`'s own stated argument -- threadgroup memory
carries OPERANDS and never partial sums. WHEN the copy happens changes no
value. The accumulation loop reads the same floats from the same page in the
same order, and `contract_partition` is untouched. Contract 13.5's second
escape is explicit that a block may own a tile and all of its `k` leaves and
schedule the reads however it likes.

**`PAGES` IS NOT CHOSEN IN THE KERNEL (DEVIATION 1260).** It comes from
`kernel_matrix.mojo::lib_smem_pages_for[TARGET_COLUMN, page_bytes]`, whose
docstring already carries the finding -- *"2 for RAFT's double buffer, 1
where it will not fit. Apple's 32 KB is the only column that forces 1 at
Policy4x4."* `[[ALWAYS GPU-agnostic]]`: there is no `if apple` anywhere in
the tuned file and the Apple divergence is a matrix row one directory over.

EXPECTED SPEEDUP. Halving the barrier count on a kernel that barriers once
per 16 or 32 `k` steps, plus DRAM latency overlap on a kernel that currently
has none. Predicted 1.10x to 1.40x, and larger at `KS = 16` than at
`KS = 32` because the barrier count per `k` element is twice as high there.

HOW IT COULD GO WRONG.

1. **The pages do not fit and `PAGES` resolves to 1**, in which case the
   technique is absent and the kernel degrades to the shipped two-barrier
   shape. At `KS = 32` with a 64x64 tile that is exactly what happens on
   Apple -- 18 KB per page, 36 KB doubled, against a 32 KB limit. Section
   6.3 predicts a specific consequence of it.
2. **The doubled footprint costs a resident block.** 20 KB of Apple's 32 KB
   admits ONE block per core, so the latency the double buffer hides may be
   latency that a second resident block was hiding for free. This is the
   most likely way a correct double buffer measures as a loss, and section
   6.5 records it as a falsifiable prediction rather than a caveat.
3. **A missing barrier that usually works.** A ping-pong race produces the
   right answer most of the time, which is the worst possible failure. The
   gate for it is bit equality against the oracle at every shape AND launch
   invariance against `PLAN_FLAT` and `PLAN_SPLITK_STAGED`, repeated, since
   a race that survives one launch will not survive sixty-two.

### 3.3 Vectorized global loads, and why they are not a reassociation

**DEVIATION 1257.**

WHAT IT CHANGES. `_tuned_g2r` issues one `unsafe_load[width=VEC]` per `VEC`
operand elements when the operand is contiguous along `p` and the window is
full, and `VEC` scalar strided loads otherwise. Shared stores are always
`VEC` wide.

WHY IT CANNOT MOVE A BIT. **`VEC` is a LOAD WIDTH here and an ACCUMULATION
GEOMETRY in `simt_kernel.mojo`, and that difference is the entire safety
argument.** RAFT accumulates a `Veclen` chunk per accumulator, so there the
width decides a summation order and `kernel_matrix.mojo` correctly labels
`PINNED_VECLEN` a NUMERIC row. In the tuned kernel the inner
`comptime for e in range(VEC)` walks the loaded chunk ascending into the ONE
accumulator that cell already had, so a `VEC` of 1, 2 or 4 accumulates the
same terms into the same accumulator in the same order. Contract 7.1's
no-sub-partition clause is satisfied by construction and not by care.

Beyond that, the two load paths write the same floats into the same shared
slots, so a machine or an orientation where the guard never fires is slower
and never different.

EXPECTED SPEEDUP. It depends on the ORIENTATION and not on the machine, and
that is a property of contract section 3's strides.

| op | rows in `bench/gemm_shapes.mojo` | `a_sp` | `b_sp` | vectorizes |
|---|---|---|---|---|
| `OP_NT` | 8 through 19, every transformer row, plus 4 through 7 | 1 | 1 | BOTH operands |
| `OP_TN` | 0 through 3, the Gram and OLS rows | `m` | `n` | NEITHER |

Predicted 1.00x to 1.20x at the `OP_NT` rows and EXACTLY 1.00x at
`gram.128sq.x100003`, which is the one tiled `OP_TN` row. A technique that
helps eighteen rows and not two is a finding about the strides.

HOW IT COULD GO WRONG. An unaligned vector load that faults or silently
splits. The spelling used is `unsafe_load[width=VEC](offset)` with no
alignment parameter, which is what `simt_kernel.mojo:360` ships and runs in
this tree; if the toolchain wants an explicit `alignment=`, that is a
compile error and not a wrong answer, which is the good kind of failure.

### 3.4 The operand flush hoist

**DEVIATION 1252, and it is the only technique here that touches a contract
seam rather than a memory access, so it gets the longest argument for its
size.**

WHAT IT CHANGES. The shipped inner loop reads a staged A value and a staged
B value and flushes both at the point of use, once per FMA. The tuned inner
loop flushes each staged value ONCE, into a register, and uses it `CPT` or
`RPT` times.

WHY IT CANNOT MOVE A BIT. `ftz` is a pure function of one float and it is
idempotent -- `ftz(ftz(x)) == ftz(x)` -- so the value multiplied is
identical whether the flush happened at the load or at each use. Contract
section 5's table says the seam is *"each `A_eff[i,p]` AS LOADED"* and
*"each `B_eff[p,j]` AS LOADED"*, which is one flush per loaded value, which
is exactly what this does. The hoist makes the code MORE literally the
contract's spelling, not less.

**SEAM 5c IS NOT HOISTED AND CANNOT BE.** It is the accumulator after every
`fma` step, one per cell per `k` step, and contract section 5 names it the
largest single cost item in the profile. Nothing in this round touches it,
and section 5.3 explains why that puts a ceiling on the whole exercise.

EXPECTED SPEEDUP. Per `k` step per thread, a 4x4 tile does `RPT + CPT = 8`
operand flushes where the unhoisted spelling would do `2 * RPT * CPT = 32`.
That removes 24 of 32 operand flushes, and operand flushes are two of the
three flushes per FMA, so roughly half of all flush work disappears. If ALL
flushes are worth the measured 1.25x, half of them is worth about 1.11x --
but section 1.1 says the flushes are largely hidden inside stalls, so the
honest prediction is the low end. Predicted 1.03x to 1.15x.

HOW IT COULD GO WRONG. It cannot go wrong bitwise. It can go wrong by
raising register pressure -- `RPT + CPT` live flushed operands per `k` chunk
on top of the accumulators and the fold stack -- and on a kernel already
close to a spill boundary that is a net loss. Section 4's table budgets it.

### 3.5 The padded threadgroup stride

**DEVIATION 1258.**

WHAT IT CHANGES. Both operands are staged `[outer][k]` with `k` innermost at
a row stride of `KS + VEC` rather than `KS`. `simt_kernel.mojo`'s
`smem_stride = kblk + veclen`, and that file's comment verbatim -- padding,
not a rounding.

WHY IT CANNOT MOVE A BIT. It is a shared-memory ADDRESS change. The value at
logical `(row, kk)` is the same float at either stride, the accumulation
reads the same floats in the same order, and no value is created or
destroyed by the pad.

EXPECTED SPEEDUP. Insurance, not a technique. A power-of-two row stride puts
every thread of a group that differs only in its row on the same bank; the
pad breaks that. Predicted 1.00x to 1.15x, and it is the term most likely to
measure as exactly 1.00x because with `VEC`-wide reads the hardware is
already splitting each access into phases.

HOW IT COULD GO WRONG. It costs `BM * VEC + BN * VEC` floats of threadgroup
memory -- 512 floats, 2 KB, at a 64x64 tile -- which at `PAGES = 2` is 4 KB
against Apple's 32 KB. If that 4 KB is what pushes the plan from two
resident blocks to one, the padding is a net loss and the fix is an XOR
swizzle at the same footprint. Owed, section 8.

### 3.6 The wider output tile

WHAT IT CHANGES. The shipped default at large shapes is `TILE_16_16_32`, a
16x16 output tile with 256 threads. The tuned wide plan is 64x64 with the
same 256 threads, which is only possible BECAUSE of 3.1.

WHY IT CANNOT MOVE A BIT. The tile decides which thread owns which cell.
Contract 6.1 puts output tiling on the execution-plan side explicitly, and
13.5's second escape is exactly this shape -- one block owns an output tile
and all of its `k` leaves.

EXPECTED SPEEDUP. This is the DRAM roofline term and section 5 has the
numbers. Predicted 1.0x to 2.0x at the `t512` rows and possibly negative at
narrow `n`.

HOW IT COULD GO WRONG. Fewer, fatter blocks. Section 6.4.

### 3.7 The flattened window sequence

**DEVIATION 1255.** Not a speedup in itself. It is what makes 3.2 writable.

The shipped kernel nests the staging `while` inside the leaf `for` and
clamps each window to the leaf's end, so the window that must be PREFETCHED
while window `t` computes may belong to leaf `t + 1`. The tuned kernel maps
flat window `w` to fold position `w // wpl` and offset `(w % wpl) * KS`,
with `wpl = ceil(L / KS)` -- a pure function of `L`, `P`, `k` and the
comptime `KS`, and of nothing in the launch.

WHY IT CANNOT MOVE A BIT. `_leaf_at` and `_leaf_bounds` are IMPORTED, so the
leaf boundary is still `[t*L, min((t+1)*L, k))` and still reads `leaf` and
`k` alone. The push into the fold tree fires at `w % wpl == wpl - 1`, once
per leaf, whatever the raggedness.

The cost is at most `wpl - 1` EMPTY windows on the short last leaf, which
stage zeros and accumulate nothing. **A staged zero that is never read is
staging hygiene and not operand padding**, which is the shipped kernel's own
distinction -- the accumulation loop runs `chunk` steps and never `KS`, so
no `0.0 * 0.0` product ever enters an accumulator and contract section 8's
ban on padded operands is not touched. The benefit is a block-uniform trip
count, so every thread provably reaches every barrier, which the previous
shape guaranteed only by an argument about clamps.

---

## 4. The finding of the round -- the fold stack bounds the register tile

**DEVIATION 1253, and it is a cost of the pin that no arm has priced.**

`gemm/UNPINNED_CONTROL.md` confound C1 records that the pinned kernel
carries `GEMM_FOLD_SLOTS = 16` float lanes of fold stack per thread whether
or not `P` needs them, calls it a cost OF the pin, and correctly says it acts
through occupancy rather than through arithmetic. At one cell per thread
that is a footnote worth sixteen registers.

**At `RPT * CPT` cells per thread it stops being a footnote, because the
stack is per CELL.** A 4x4 register tile at a fold depth of 8 needs 128 float
lanes of fold stack beside its 16 accumulators. No column in
`kernel_matrix.mojo` has that many architectural registers per thread to
spare at a useful occupancy.

An unpinned kernel has no such term at all. It carries ONE accumulator set
for the whole `k` range, so it can take the widest register tile the machine
supports. **The pinned kernel cannot, and the lever this round most wanted is
the lever the pin most restricts.**

### 4.1 The mitigation, which is a mitigation and not a fix

**DEVIATION 1254.** `_fold_push`'s `occ` is a binary counter over the leaves
pushed so far, so after `t` pushes it equals `t` and the highest slot index
ever touched while pushing `P` leaves is `floor(log2 P)`. A stack of `FS`
slots therefore covers every `P <= 2^FS - 1`, exactly -- and not `2^FS`,
because at `P = 2^FS` the last push merges through every slot and needs one
more. That off-by-one has the same shape as contract 7.2.2's warning that
`fold_node_total(P)` is not bounded by `2P`, and it is checked rather than
trusted, by `_fold_push_tile`'s overflow return and by a host-side refusal.

So `FS` is a comptime kernel parameter and the host dispatches it from `P`.

**A DISPATCH ON `P` IS A DISPATCH ON `k` AND ON NOTHING ELSE.** Contract 6.1
is the batch-invariance clause and it permits the NUMERICAL plan to see `k`;
this is the EXECUTION plan seeing a function of `k`, which is a weaker thing
still. Every variant computes the same `(d, q)` nodes over the same `P`
leaves in the same pairing. What varies is how many registers the compiled
kernel reserves for a tree it computes identically either way. Contract
13.5's last sentence forbids choosing the NUMERICAL TREE at a performance
dispatch boundary and nothing here does -- there is no
`if P < 32 use the serial fold`, and there never may be.

### 4.2 The register budget, in floats per thread

Live SIMD lanes, counted from the source. Not measured, and section 7's
first gate is the readout that would replace this table with facts.

| plan | cells | acc | fold stack | A frag | B frag | prefetch | total |
|---|---|---|---|---|---|---|---|
| shipped `TILE 16x16 KS=32` | 1 | 1 | 16 | -- | -- | -- | ~17 |
| tuned 64x64 reg4x4 `FS = 1` | 16 | 16 | 16 | 16 | 16 | 8 | ~72 |
| tuned 64x32 reg4x2 `FS = 8` | 8 | 8 | 64 | 16 | 8 | 8 | ~104 |
| tuned 32x32 reg2x2 `FS = 16` | 4 | 4 | 64 | 8 | 8 | 8 | ~92 |
| a hypothetical unpinned 64x64 reg4x4 | 16 | 16 | **0** | 16 | 16 | 8 | ~56 |

Read the last two rows together. **The unpinned kernel gets the widest tile
for 56 registers. The pinned kernel needs 104 to get half of it.** That is
the price, it is structural, and it is invisible to any arm that times two
kernels at one cell per thread.

### 4.3 What this costs at the shapes in the table

| shape | `k` | `P` | tuned plan | tile | what the pin cost |
|---|---|---|---|---|---|
| `kmeans.dist.4096x64x64` | 64 | 1 | reg4x4 `FS = 1` | 64x64 | nothing -- contract 7.3, no tree, no stack |
| `pca.transform.wide.8192x64x128` | 128 | 1 | reg4x4 `FS = 1` | 64x64 | nothing |
| `llama8b.*.t512`, `k = 4096` | 4096 | 32 | reg4x2 `FS = 8` | 64x32 | half the register tile |
| `llama8b.mlp_down.t512` | 14336 | 112 | reg4x2 `FS = 8` | 64x32 | half the register tile |
| `gram.128sq.x100003` | 100003 | 782 | reg2x2 `FS = 16` | 32x32 | three quarters of it |

**The cost of the pin on the tuning is a rising function of `P`, which is a
rising function of `k`.** That is a shape of cost nothing in
`IDENTICAL_FP32_CONTRACT.md` section 13 anticipates -- 13.3 argues the fold
is under one percent of the ARITHMETIC at every legal `k`, which is true and
is about a different resource. Section 8's OWED item 8 asks for that
sentence to be qualified rather than left to be read as covering the whole
question.

---

## 5. The roofline reasoning

Three rooflines, and the current kernel is probably not on the one people
assume.

### 5.1 Flops per byte staged into threadgroup memory

The number `gemm/UNPINNED_CONTROL.md` section 5.1 quotes. At
`TILE 16x16 KS=32` one staging window moves `TM*KS + KS*TN = 1024` floats,
4096 bytes, and performs `TM*TN*KS = 8192` FMAs against them, 16384 flops.

    16384 flops / 4096 bytes  =  4.0 flops per staged byte

Register blocking does not change this number at all, because it changes
neither the tile nor `KS`. Widening the tile does.

| tile | staged floats per `KS = 16` window | FMAs | flops per staged byte |
|---|---|---|---|
| 16x16 | 512 | 4096 | 4.0 |
| 32x32 | 1024 | 16384 | 8.0 |
| 64x32 | 1536 | 32768 | 10.7 |
| 64x64 | 2048 | 65536 | 16.0 |

### 5.2 Flops per byte read OUT of threadgroup memory

**This is the one the current kernel is on, and it is the one register
blocking moves.** One block owns its tile's whole `k` range, so per FMA the
shipped kernel issues two threadgroup loads, eight bytes, for two flops.

| plan | shared loads per FMA | flops per shared byte | vector load instructions per FMA at `VEC = 4` |
|---|---|---|---|
| shipped, 1 cell per thread | 2.0 | 0.25 | 2.0 |
| tuned reg2x2 | 1.0 | 0.50 | 0.25 |
| tuned reg4x2 | 0.75 | 0.67 | 0.19 |
| tuned reg4x4 | 0.5 | 1.00 | 0.125 |

**A 4x-to-16x improvement in the resource the kernel is most plausibly
limited by is the whole reason register blocking is the first technique in
section 3.**

### 5.3 Flops per byte read from DRAM, and the machine balance

Per block, over the whole `k` range, `(BM + BN) * k * 4` bytes in and
`2 * BM * BN * k` flops, so the intensity is `BM*BN / (2*(BM+BN))` and
depends on the tile alone.

| tile | flops per DRAM byte |
|---|---|
| 16x16 | 4.0 |
| 32x32 | 8.0 |
| 64x32 | 10.7 |
| 64x64 | 16.0 |
| 128x128 (not implemented, section 8) | 32.0 |

**An M4's balance is nearer 15 flops per byte, so the shipped 16x16 tile is
3.75x on the bandwidth side of the line and the 64x64 tile is the first one
that crosses it.** That is the argument for the wider tile and it is also
the reason 3.3, 3.4 and 3.5 are predicted small -- a technique that does not
move this ratio cannot help a bandwidth-bound kernel much, and three of the
six do not move it at all.

**AND THE 15 IS THE NUMBER THIS DOCUMENT IS LEAST SURE OF.** It is the
figure the lane was handed. If the M4's real balance is nearer 30 flops per
byte, then even the 64x64 tile is on the bandwidth side, section 6's
predictions are too optimistic across the board, and the next lever is a
128x128 tile at 8x8 outputs per thread -- which section 4 says is available
only at `P == 1` and is unavailable at every transformer row. Measuring the
balance on the actual box, with a stream kernel, before believing any ratio
in section 6, is OWED item 12.

### 5.4 The ceiling contract 5c puts on the whole exercise

**DEVIATION 1261, and it is analysis and not measurement.**

Seam 5c is one accumulator flush per cell per `k` step and it is
irreducible -- no tiling, no vectorization and no staging changes the count.
If a flush lowers to `F` scalar operations, then the best any realization of
this profile can do on a machine with one FMA pipe and no separate flush
unit is

    FMA issue efficiency  <=  1 / (1 + F)

    F = 4    20% of scalar FMA peak
    F = 2    33%
    F = 1    50%
    F = 0    100%   (the flush compiled away entirely)

**So vendor libraries at 62 to 94 percent are not a reachable target for the
IDENTICAL arm at any level of engineering, and the honest ambition for this
round is 2-5 percent to somewhere between 8 and 15 percent.** Saying so
before the run is the point of writing it down.

Two things could falsify this bound and both are owed, not assumed. `F` has
never been counted from a disassembly on any vendor. And on a backend whose
denormal flushing is architectural -- Metal, where contract section 5 says
`ftz` is bitwise a no-op -- a sufficiently informed compiler could fold the
flush away and reach `F = 0` legally, which would mean the 1.25x
`pinned / strict` measurement is telling us the compiler does NOT do that
today. Section 8.3 says why this round did not go looking for that and did
not act on it.

---

## 6. The predictions, recorded before any run

Written on op counts, on the four measured ratios in section 1, and on the
rooflines above. Nothing here is measured. If the run disagrees, the run is
right and this section is deleted rather than defended
(`[[mojotrees code is NOT the source of truth]]`).

### 6.1 Per technique, at the `llama8b.*.t512` tiled rows

Each ratio is `shipped / tuned` with only that technique enabled, which is
NOT how the file is built -- it ships as one kernel with all of them on. The
per-technique numbers are falsifiable only by an ablation build and OWED
item 5 asks for one, precisely because a combined number that lands inside
the combined range tells you nothing about which term was right.

| # | technique | predicted | most likely to be wrong because |
|---|---|---|---|
| 3.1 | register blocking, 4x2 | 1.40x -- 2.20x | it is the term section 1.1 says should be biggest, and section 4 says the pin already halved it |
| 3.2 | double buffering | 1.10x -- 1.40x | the doubled footprint may cost a resident block and pay the gain back |
| 3.3 | vectorized global loads | 1.00x -- 1.20x | `OP_NT` only; exactly 1.00x on the two `OP_TN` tiled rows |
| 3.4 | operand flush hoist | 1.03x -- 1.15x | section 1.1 says removed instructions are mostly hidden |
| 3.5 | padded shared stride | 1.00x -- 1.15x | most likely exactly 1.00x |
| 3.6 | wider tile | 1.00x -- 2.00x | entirely contingent on the machine balance in 5.3 being 15 and not 30 |

### 6.2 Combined

**`shipped / tuned` at the `llama8b.*.t512` tiled rows lands between 1.8x
and 3.0x, and the single most likely value is 2.2x.**

The terms are NOT independent and their product is not the answer -- they
all attack the same stall, and a naive product of the midpoints above gives
roughly 3.5x, which this document does not predict. If the measured number
is above 3.0x, the section 1.1 inference is wrong and the kernel was
instruction-issue bound after all.

**In absolute terms that is 2 to 5 percent of peak becoming 4 to 12 percent,
against vendor libraries at 62 to 94 percent.** This round does not close the
gap. It takes perhaps a third of it in log terms and leaves 5x to 15x, of
which section 5.4 argues 3x to 5x is a hard ceiling of contract 5c and the
rest is work nobody has done. Any write-up that quotes a speedup from this
round without that sentence is `[[IDENTITY IS NOT FREE]]`'s failure mode
with the sign flipped.

### 6.3 The `KS` sweep

DEVIATION 1259 gives the file six tuned plans and a DELEGATE fallback; two
of the six are never dispatched and exist for this sweep, exactly as
`PLAN_SPLITK_STAGED` does in the shipped file.

`TUNED_PLAN_R4C4_K16_S1` and `TUNED_PLAN_R4C4_K32_S1` differ in `KS` alone,
and `lib_smem_pages_for` resolves them to `PAGES = 2` and `PAGES = 1`
respectively on the Apple column and to `PAGES = 2` on both for NVIDIA and
AMD.

**Prediction. On Apple the `KS = 16` plan wins, because it is the only one
of the two that gets a double buffer at all. On NVIDIA and AMD the `KS = 32`
plan wins, because both get the double buffer and 32 halves the barrier
count.** If that comes out the same way on all three columns, one of the two
mechanisms is not doing what section 3.2 says it does.

### 6.4 Where a REGRESSION is predicted

A prediction that names nothing it can lose is not a prediction.

1. **`kmeans.dist.4096x64x64`.** `n = 64` and the tuned wide plan makes
   `ceil(4096/64) * 1 = 64` blocks where `TILE_16_16_32` makes 1024. With
   `k = 64` the whole GEMM is 16.8 million FMAs and is dominated by launch
   and by tail effects, not by the inner loop. **Predicted 0.8x to 1.1x --
   flat to a mild loss on Apple, and a larger loss on an H100 with 132 SMs
   where 64 blocks cannot fill the machine.**
2. **`pca.transform.8192x4x4` and `ols.predict.gemv.64Kx16`.** `n = 4` and
   `n = 1`. Both DELEGATE -- they are below the narrow tile's threshold --
   so both are predicted at exactly 1.00x, and if either moves, the
   dispatcher is not doing what `tuned_gemm_plan_name` says it is.
3. **`gram.128sq.x100003`.** `P = 782` forces `FS = 16` and the 32x32 tile,
   and `OP_TN` means neither operand vectorizes. Predicted 1.0x to 1.4x, the
   weakest tiled row in the table, and the row that most directly measures
   DEVIATION 1253.

### 6.5 The prediction most likely to be wrong

**`TUNED_PLAN_R2C2_K16_S1` beats `TUNED_PLAN_R4C4_K16_S1` on Apple at the
`P == 1` rows, despite having a quarter of the register-tile reuse.**

At `KS = 16` and `VEC = 4` the 64x64 double-buffered plan claims about 20 KB
of Apple's 32 KB, which admits ONE resident block per core; the 32x32 plan
claims about 10 KB and admits three. If the kernel is latency bound -- which
section 1.1 argues it is -- three resident blocks may hide more than four
times the register reuse exposes.

This is written down because it is the prediction whose failure teaches the
most. If 4x4 wins comfortably, the kernel is reuse bound and the next round
should go WIDER at the cost of occupancy. If 2x2 wins, the kernel is
occupancy bound, `lib_smem_pages_for` returning 2 is a trap on the Apple
column, and the next round should go NARROWER and cut the footprint.

### 6.6 The prediction about the fold

**The `P == 1` rows and the `P > 1` rows will show the same relative
improvement from techniques 3.2 through 3.6 and a DIFFERENT one from 3.1,
and the difference will be DEVIATION 1253 and not the fold's arithmetic.**

Contract 13.3 is right that the fold is under one percent of the arithmetic
at every legal `k`, and `gemm/UNPINNED_CONTROL.md` prediction 4 confirmed it
by measuring the same 1.26x to 1.55x at rows with no tree at all. Neither
statement covers the register budget. If the tuned `P == 1` rows improve by
2.5x and the tuned `P = 32` rows improve by 1.6x on the same box in the same
window, the difference is the register tile the fold stack refused to allow,
and that is a number contract section 13 currently prices at zero.

---

## 7. What would make a speedup FAKE

A tuned GEMM is unusually prone to looking faster while being wrong.
`[[verify reach, not output]]`, `[[reached-but-inert]]`. Every one of these
must be excluded before a single ratio is quoted.

1. **The kernel does not write its output.** A masked store with an
   off-by-one in `gi < m and gj < n`, or a grid that leaves most tiles
   unlaunched, is the fastest GEMM ever written. The gate is a POISONED
   output buffer before an untimed warm-up, read back and digested, exactly
   as the existing arms in `bench/gemm_price_main.mojo` do. A kernel that
   launches without writing must not be allowed to turn in the best time in
   the table.
2. **The kernel reads uninitialized threadgroup memory.** The ragged path
   and the empty windows of DEVIATION 1255 stage zeros into slots the
   accumulation loop then does not read; if the `chunk == KS` fast path were
   ever taken at `chunk < KS`, it would read them and the answer would be
   silently wrong at exactly the shapes with a ragged last leaf. The gate is
   the oracle at `gram.128sq.x100003` (35-element last leaf) and
   `gram.32x32x1M` (`L = 977`, a leaf that is not a `KS` multiple at either
   `KS`), per cell, not per digest.
3. **A ping-pong race that usually works.** DEVIATION 1256 removes a
   barrier. A missing barrier gives the right answer most of the time, and
   the failure is timing dependent, so ONE passing run proves nothing. The
   gate is bit equality repeated, plus launch invariance against `PLAN_FLAT`
   and `PLAN_SPLITK_STAGED`, whose realizations of the same DAG share no
   memory with this one.
4. **The gain vanishes at another shape.** All twenty rows of
   `bench/gemm_shapes.mojo`, in the order they are in, nothing dropped for
   being slow or unflattering (`[[never build to datasets]]`). A tuned
   kernel that wins at `t512` and loses at `t1` has moved the problem.
5. **The timing loop skips a synchronize.** `tuned_gemm_into` is
   ASYNCHRONOUS, exactly as `identical_gemm_into` is. A loop that enqueues
   `REPEATS` launches and synchronizes once at the end measures enqueue
   throughput, and a kernel with more launches will look better at it.
6. **The tuned arm DELEGATED and the bench reported it as tuned.** Six of
   the twenty rows are predicted to delegate. `tuned_gemm_plan_name` must be
   printed per row and a delegated row must be excluded from any headline
   ratio, because it measured `identical_gemm_into` twice.
7. **The accumulators spilled and the kernel is fast for the wrong reason,
   or slow for a reason nobody diagnosed.** This is the
   `bench/results/GEMM_ROUND_2026-08-19.md` failure and it is invisible to
   timing. A register-count or occupancy readout per kernel per vendor is
   the only instrument that separates it from "the technique did not help".
8. **The arms were measured in different thermal windows.** The M4 drifts
   1.7x in twenty minutes under heat and the mechanism is measured
   (`[[the M4 drifts 1.7x in 20 minutes]]`). The tuned arm and the shipped
   arm live in ONE binary specifically so they can alternate call by call.
   A run that alternates block by block is discarded.
9. **The two arms were built in different numeric modes.** `tuned` in a FAST
   binary against `device` in an IDENTICAL one is not a comparison of
   kernels. Read the mode back from the run and never from the source
   (`[[the shared checkout's mode flip]]`); `tuned_gemm_banner` exists for
   that and prints the resolved kernel-matrix column beside the mode,
   because a simulated column changes `PAGES`.
10. **A sabotaged build.** In a `-D MOJOLEARN_GEMM_SABOTAGE_*` build the
    shipped kernel's arms compile in and the two are not comparable. No
    timing may be taken from one. `SAB_NODE_ORDER` is the exception in the
    other direction -- it cannot reach the tuned kernel at all, because it
    sabotages a split-K workspace address and the tuned file has no
    workspace, so a gate should assert that a `SAB_NODE_ORDER` build of the
    tuned kernel is bit-identical to an unsabotaged one, and that assertion
    is a statement about coverage rather than a pass.
11. **Quoting the result as "the speed of identical GEMM".** It is one
    machine, one dtype, one shape family, on a kernel that has never been
    compiled. Section 6.2's second paragraph is the sentence that has to
    travel with the number.

---

## 8. OWED, AND WHY I DID NOT DO IT HERE

This lane could write exactly two paths,
`gemm/original/gemm_identical_tuned.mojo` and this file, and had no
execution rights. Everything below is required to turn a proposal into a
measurement and none of it was done here.

### 8.1 Before anything may route to the tuned kernel

1. **`gemm/original/gemm_device_check.mojo` -- extend it to the tuned
   plans.** Per-cell bits against `gemm_oracle` at all 62 shapes, launch
   invariance requiring every tuned plan to equal `PLAN_FLAT` and
   `PLAN_SPLITK_STAGED` bit for bit, batch invariance across a dispatch
   boundary and across composition, and all six sabotages shown to FAIL
   (five of them; see item 10 of section 7 for the sixth). **Until this
   passes, the tuned file is a proposal about how to go faster and not a
   realization of the profile**, and its own header says so.
2. **`check_tuned_stack_fold_is_the_contract_tree`.** DEVIATION 1263.
   `_fold_push_tile` and `_fold_drain_tile` against
   `gemm_oracle.fold_balanced_tree` for every `P` in `1 .. 2049` on hashed
   partials, at every `FS` the file compiles, requiring bit equality at
   every one. The scalar `_fold_push` has exactly this proof and this
   spelling has only an argument.
3. **`check_tuned_fold_depth_covers_the_plan`.** That
   `tuned_plan_max_leaves` equals `2^FS - 1` and not `2^FS`, by running
   `_fold_push_tile` at both and requiring the overflow return to flip; and
   that the deepest plan covers `CONTRACT_MAX_LEAVES`
   (`tuned_gemm_fold_depth_covers_the_profile` is the predicate, unasserted).
4. **`bench/gemm_shapes.mojo` -- a shape with `P` between 256 and 1024 and
   an `m n` large enough to tile.** The table's only such row is
   `gram.128sq.x100003` at `P = 782`, so the `FS = 16` plan has one fixture
   and `[[uniform test data hides permutation]]`'s sibling problem applies --
   one fixture is not coverage.

### 8.2 To get a number at all

5. **`bench/gemm_price_main.mojo` -- add the `tuned` arm.**
   `_timed_shape_with_device` needs `tuned` beside `device`, `vendor`,
   `pinned`, `unpinned` and `unpinned1`, with a poisoned warm-up, a digest,
   and its call inside the same `for _ in range(REPEATS)` loop so the arms
   alternate call by call. `tuned_gemm_into` mirrors `identical_gemm_into`
   argument for argument specifically so this is an addition and not a
   refactor. It must print `tuned_gemm_plan_name` per row. **And it should
   carry per-technique ABLATION builds**, because section 6.1's six
   predictions are unfalsifiable from a combined number alone.
6. **`tools/gemm_price.sh` -- add `-D MOJOLEARN_GEMM_TUNED_ARM=1`** to the
   IDENTICAL leg's build line, and print `tuned_gemm_banner()` in the mode
   witness block.
7. **A register-count and occupancy readout per kernel per vendor.** Section
   7 item 7 and DEVIATION 1253. Without it the failure mode that killed the
   2026-08-19 round is indistinguishable from a technique that did not work,
   and section 4.2's table stays a count of source lanes rather than a
   measurement.
8. **Measure the machine balance on the box.** Section 5.3 rests on 15 flops
   per byte and every prediction in section 6 rests on section 5.3. A stream
   kernel and a peak-FMA kernel in the same thermal window is a morning's
   work and it decides whether the 64x64 tile is over the line or under it.

### 8.3 Edits to files this lane was not permitted to touch

9. **`gemm/original/gemm_identical.mojo` -- possibly make `_leaf_bounds`,
   `_leaf_at`, `_tile_grid` and the `SAB_*` aliases public.** The tuned file
   imports all of them by their underscore names, as
   `gemm/original/gemm_unpinned.mojo` already does, and there is precedent
   in the tree (`glm/derived/glm/qn/glm_linear.mojo` imports `_read_scalar`).
   If the compiler refuses, rename them without the underscore and update
   every call site in the SAME commit. **A COPY of any of them into the
   tuned file is not an acceptable fix** -- a copied constant drifts, and a
   drifted leaf boundary would make the two kernels differ in more than the
   tuning.
10. **`original/kernel_matrix.mojo` -- a `lib_acc_th_cols` row.**
    `TUNED_TC = 16` is the one scheduling number the tuned file names
    itself, and the matrix's own header calls a constant restated in a
    kernel the failure the table exists to prevent. The matrix pins the
    per-thread tile and the block size and has no row for the thread grid
    that connects them, so a vendor measurement of it has nowhere to land.
11. **`gemm/README.md` -- register DEVIATIONS 1250 through 1264.** They are
    defined in the two files this lane wrote and nowhere else, and a
    deviation that lives only in a docstring is a deviation the next lane
    renumbers over.
12. **`gemm/IDENTICAL_FP32_CONTRACT.md` section 13 -- two edits.** An eighth
    entry in 13.6 for the tuned arm. And a qualification of 13.3, which says
    the fold is under one percent of the arithmetic at every legal `k` --
    true, and about a different resource than DEVIATION 1253, which is the
    register budget. The sentence is not wrong and it is being read as
    covering more than it does.
13. **`gemm/UNPINNED_CONTROL.md` confound C1 -- a pointer to DEVIATION
    1253.** C1 calls the fold stack an occupancy effect that timing cannot
    separate from an arithmetic one. That is right and incomplete: at more
    than one cell per thread it is not an occupancy effect at all, it is a
    hard bound on the largest available tile. `[[fix-docs-on-discovery]]`
    would have this fixed in the same commit that lands the finding.

### 8.4 Techniques considered and deliberately NOT implemented

14. **DEVIATION 1262. Matrix and tensor instructions -- `simdgroup_matrix`,
    `mma.sync`, MFMA.** **Almost certainly REFUSED by this contract, and
    that is the
    largest single unclaimed component of the 12x to 25x.** An MMA
    instruction performs an entire small matrix product in hardware; the
    order in which it accumulates its own partial products is not specified
    by any vendor, there is no per-step seam at which contract 5c's
    accumulator flush could be applied, and on NVIDIA the FP32 path is TF32
    (`kernel_matrix.column_vendor_fp32_matmul_is_tf32` already models this
    and `VENDOR_TF32_PRODUCT_REL_BOUND` already prices it), which is not the
    same arithmetic at all. This is written down as ANALYSIS. It is not
    implemented, it is not owed as an implementation, and the honest
    statement is that the identical profile forgoes the matrix pipelines and
    that this is a cost of the pin nobody has quantified.
15. **Spilling the high fold-stack levels to a global workspace**, so a 4x4
    tile survives at `P > 1`. Level `d` is touched once per `2^d` leaves and
    a leaf is at least `K_LEAF_MIN = 128` `k` steps, so keeping levels 0
    through 2 in registers and spilling levels 3 and up costs
    `m * n * (P/8) * 2` floats of traffic -- roughly 6 percent extra DRAM at
    `llama8b.qkv.t512` -- and would buy back the full register tile.
    Bit-safe by the same argument as DEVIATION 1254 (it changes where a node
    is stored and not which node it is), and NOT implemented, because it
    ends the tuned file's "no global scratch at any shape" property and this
    round had no way to measure whether the 6 percent is worth it.
16. **Eliding `ftz` on a backend whose flushing is architectural.** Contract
    section 5 states that `ftz` is bitwise a no-op on an FTZ backend, so on
    Metal it is provably inert. **This lane did not act on that and the
    brief is explicit that it may not.** It is recorded here as an OWED
    EXPERIMENT with a specific shape: disassemble the shipped tiled kernel
    on Metal, count whether the flush survives codegen, and report `F` from
    section 5.4. If `F` is already 0 on Apple, then the measured
    `pinned / strict = 1.25x` is not the flush and the whole attribution in
    `gemm/UNPINNED_CONTROL.md` prediction 5 needs redoing. **A source-level
    elision would be a per-vendor arithmetic difference and is forbidden
    whatever the disassembly says** -- the contract's value is that one
    source computes one answer everywhere.
17. **A 128x128 tile at 8x8 outputs per thread**, 32 flops per DRAM byte,
    the next rung past 64x64. Available at `P == 1` only, for DEVIATION
    1253's reason, which is worth stating plainly: the pin does not merely
    slow the identical GEMM down, it makes the next tuning rung unreachable
    at every shape with more than one leaf.
18. **A tile swizzle chosen for DRAM locality.** All three `SWIZZLE_*`
    permutations are imported and the tuned dispatcher uses `SWIZZLE_NONE`
    everywhere. A sweep is a cheap experiment and a bijection cannot move a
    bit, so this is owed rather than argued.
19. **`KS = 64` or `KS = 128`, so a `K_LEAF_MIN` leaf is one or two staging
    windows.** The footprint does not fit doubled on any column at a 64x64
    tile, so it trades the double buffer for the barrier count and the two
    effects have opposite signs. Owed as a sweep, not predicted here.
20. **A second and a third vendor.** This is one machine and no machine has
    run it. `[[IDENTITY IS NOT FREE]]` is explicit that an Apple number is
    not an NVIDIA or an AMD number, and this file's `PAGES` row resolves
    DIFFERENTLY on Apple than on the other two at `KS = 32`, so the tuned
    kernel is not even the same kernel across the three columns.

---

## Appendix. What the author is least confident compiles

The tuned kernel has never been built. In descending order of doubt.

1. **`comptime for` over a bound that came from a function PARAMETER**,
   with a SIMD lane write inside, nested inside a `while` loop. Every one is
   written against a `comptime` LOCAL (`comptime NS = FS`,
   `comptime NR = RPT`, `comptime NSLOT = SLOTS`) rather than against the
   parameter directly, because `simt_kernel.mojo` uses exactly that shape
   and ships. `gemm/UNPINNED_CONTROL.md`'s appendix rates the direct form
   its number-one doubt, so the indirection is deliberate insurance. If it
   still fails, the fallback is a hand-written unroll per plan, which costs
   a maintenance hazard and no correctness.
2. **`SIMD[DType.float32, 128]`** -- the `FS = 8`, 4x4 fold stack of
   `TUNED_PLAN_R4C4_K16_S8`. The widest SIMD anywhere in this tree today is
   far narrower. If the type is rejected or the value is spilled wholesale,
   that plan is the one to drop; it is a sweep-only plan the dispatcher
   never returns, so dropping it costs no shape.
3. **The underscore imports `_leaf_bounds`, `_leaf_at`, `_tile_grid` and the
   `SAB_*` comptime aliases** from `gemm_identical.mojo`. See OWED item 9.
   The `SAB_*` names are module-level `comptime` values rather than
   functions and nothing in this tree imports one across modules yet.
4. **`stack_allocation` with a comptime count derived from a
   `lib_smem_pages_for` call inside a parametric launcher.** `PAGES` is
   comptime and `PAGES * APAGE` is comptime, but the chain from a
   kernel-matrix accessor through a function parameter list into an
   allocation size is longer than anything precedented here.
5. **`as_.unsafe_load[width=VEC](offset)` on a SHARED-address-space
   pointer.** `simt_kernel.mojo:383` does exactly this and ships, so the
   doubt is low, but it is doing it at `veclen` from a `comptime` local in
   the same file and this does it at a value threaded from another module.
6. **`comptime assert FS >= 1 and (FS & (FS - 1)) == 0, "..."`** inside a
   parametric `def`. Precedented in `simt_kernel.mojo` at
   `comptime assert (tc & (tc - 1)) == 0 and tc <= 32`, but that is a
   comptime local and this is a parameter.
7. **`comptime kern = tuned_gemm_tiled_kernel[RPT, CPT, TC, KS, FS, PAGES]`
   inside a parametric `def`**, where five of the six parameters arrive from
   the enclosing function's parameter list. `gemm_unpinned.mojo` does the
   four-parameter version and its appendix lists the fourth-parameter case
   as its own item 4, so this is the same doubt one step further out.
8. **Forward references** -- `tuned_gemm_plan_name` calls
   `tuned_plan_fold_slots`, which is defined later in the file.
   `identical_gemm` does the same with
   `identical_gemm_workspace_max_floats`, so this should be fine, and it is
   listed because "should be fine" is how the list above got written.

None of these is a correctness risk. Every one is a compile error, which is
the failure mode you want from a file that has never been built.
