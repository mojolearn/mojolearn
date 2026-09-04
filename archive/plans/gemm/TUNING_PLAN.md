# The tuning plan for `mojolearn.identical.gemm.fp32.v1`

How to make the identical GEMM fast without moving one bit.

## STATUS

**NOTHING IN THIS LANE HAS BEEN COMPILED, RUN OR MEASURED.** Written
2026-08-25 alongside `gemm/checks/gemm_identical_tuned.mojo`, which exists on
disk and has never been through a compiler.

    gemm/checks/gemm_identical_tuned.mojo   EXISTS, never compiled
    gemm/checks/gemm_tuned_probe.mojo       EXISTS, never run
    tools/e2_remote_leg.sh                  registers a `tuned-probe` arm
    pixi.toml                               NO task for either file
    bench/gemm_price_main.mojo              NO `tuned` arm
    bench/results/                          NO tuned number on any column

`gemm/checks/gemm_identical.mojo` WAS NOT TOUCHED. It is the shipped profile,
it is gated on three vendors, and it stays untouched so that it remains the
reference the tuned file is compared against.

Everything in section 5 is a PREDICTION recorded before measurement. A
prediction that turns out wrong is the most useful thing this document can
produce, and a wrong one is deleted rather than defended.

DEVIATIONS 1250 through 1264 are defined here and in the tuned file and
NOWHERE ELSE; `gemm/README.md` does not register them (checked). 1265 through
1289 are reserved for whoever builds, gates and runs this.

---

## 1. The number this attacks, and the part already priced

`mojolearn.identical.gemm.fp32.v1` runs at 2 to 5 percent of FP32 peak on
Apple, on an H100 and on an MI325X. Vendor libraries reach 62 to 94 percent
at the same precision, so the gap is 12x to 25x.

`gemm/UNPINNED_CONTROL.md`'s single-variable arm split part of it, MEASURED
on an M4 at the tiled llama8b `t512` rows.

| ratio | value | what it is |
|---|---|---|
| pinned / unpinned | 1.55x | the whole cost of the fold-order pin |
| pinned / strict (`NACC = 1`) | 1.25x | the SEAMS alone, `ftz` plus refusing FMA contraction |
| unpinned / strict | 1.24x | the instruction-level parallelism the pin forbids |
| the fold tree | ~0x | `P == 1` rows show the same 1.26x to 1.55x with NO tree present |

**So the pin is 1.55x of a 12x to 25x gap and the remaining 8x to 16x is
kernel engineering.** Section 4 is the finding that one part of the remainder
is not engineering at all but a second, unpriced cost of the pin.

### 1.1 The inference that shapes every prediction below

The pinned inner step is two shared loads, two operand flushes, one `fma` and
one accumulator flush. `gemm/UNPINNED_CONTROL.md` 5.1 prices a flush at about
four scalar operations, so the pinned step is roughly one FMA plus fourteen
scalar operations against the strict step's one FMA plus two loads. **The
pinned kernel issues roughly five times the instructions and takes 1.25x the
time.**

A kernel whose instruction count can quintuple for a 25 percent slowdown is
not instruction-issue bound. It is stalled on shared-memory latency, the
dependent FMA chain, or DRAM, and about eighty percent of its issued work is
hidden inside those stalls. Every technique that only removes INSTRUCTIONS
should disappoint relative to its op count; every technique that attacks the
STALL is where the money is. Section 5's predictions are written to that
shape and are falsifiable as a group.

---

## 2. The one constraint, stated as an operational test

**The tuned kernel must produce output bit-identical to `gemm_oracle` at
every shape, exactly as `gemm_identical.mojo` does today.** The pin fixes
WHICH partial sums combine and IN WHAT ORDER. It says nothing about when
bytes arrive or where they sit, so the test each technique has to pass is one
question.

> Does this change the multiset of values entering any accumulator, or the
> order in which they enter it, or the set of `ftz` applications any value
> passes through?

If the answer is no at every shape including the ragged and degenerate ones,
the technique is available however violently it rearranges memory. If the
answer is yes anywhere it is a v2 of the contract and no measurement licenses
taking it (contract 13.7).

| forbidden | contract | why it is refused |
|---|---|---|
| changing the k partition | 6, 6.1, 6.2 | split-k selection is the one autotuning axis the pin genuinely removes. `contract_partition`, `contract_leaf_size` and `contract_leaf_count` are IMPORTED and never re-derived; `K_LEAF_MIN` and `MAX_LEAVES` are profile constants |
| reassociating within a leaf | 7.1 | four `Veclen` lanes into four registers added at the end is a depth-2 tree hidden inside one leaf. The tuned kernel vectorizes LOADS and never ACCUMULATORS |
| dropping `ftz`, or letting the compiler contract | 4, 4.1, 5 | these cost 1.25x and they are the contract, not an inefficiency. Section 7 item 16 records where a seam MIGHT be provably inert and why this round did not act on it |
| changing the fold tree or its carries | 7.2, 7.2.2 | `_fold_push_tile` is a lane-by-lane replica of the shipped `_fold_push`. DEVIATION 1263 says it is ARGUED and not yet PROVEN |

---

## 3. The techniques, one section each

Each row's middle column is the seam decision, which is the part of this
document that is a record rather than a proposal.

| # | DEV | technique | why it cannot move a bit |
|---|---|---|---|
| 3.1 | 1251 | **register blocking**, one thread owns `RPT x CPT` output cells | Contract 7.1 forbids sub-partitioning ONE leaf of ONE cell into several accumulators. It does not restrict how many CELLS a thread owns. Sixteen cells with one accumulator each are sixteen contract-shaped serial ascending chains and no value moves between them. **The pin forbids ILP in `k`; it does not forbid ILP in `m` and `n`**, so a 4x4 tile buys back legally the term the unpinned arm bought illegally |
| 3.2 | 1256, 1260 | **double buffering** the k-window staging, `PAGES` pages, one barrier per window instead of two | A staging copy is bit exact and threadgroup memory carries OPERANDS, never partial sums. WHEN the copy happens changes no value. `PAGES` comes from `kernel_matrix.mojo::lib_smem_pages_for`, so there is no `if apple` anywhere in the tuned file |
| 3.3 | 1257 | **vectorized global loads**, `unsafe_load[width=VEC]` when the operand is contiguous along `p` | `VEC` is a LOAD WIDTH here and an ACCUMULATION GEOMETRY in `simt_kernel.mojo`, and that difference is the whole safety argument. The inner `comptime for e in range(VEC)` walks the chunk ascending into the ONE accumulator the cell already had |
| 3.4 | 1252 | **the operand flush hoist**, flush each staged value once into a register and use it `CPT` or `RPT` times | `ftz` is idempotent, `ftz(ftz(x)) == ftz(x)`. Contract section 5 says the seam is each operand AS LOADED, which is one flush per loaded value. **Seam 5c is NOT hoisted and cannot be**, being one accumulator flush per cell per `k` step |
| 3.5 | 1258 | **the padded threadgroup stride**, `KS + VEC` instead of `KS` | A shared-memory ADDRESS change. The value at logical `(row, kk)` is the same float at either stride |
| 3.6 | -- | **the wider output tile**, 64x64 at 256 threads instead of 16x16 | The tile decides which thread owns which cell. Contract 6.1 puts output tiling on the execution-plan side explicitly and 13.5's second escape is exactly this shape |
| 3.7 | 1255 | **the flattened window sequence**, flat window `w` maps to fold position `w // wpl`, offset `(w % wpl) * KS`, `wpl = ceil(L / KS)` | `_leaf_at` and `_leaf_bounds` are IMPORTED, so the leaf boundary still reads `leaf` and `k` alone. The cost is at most `wpl - 1` EMPTY windows on a short last leaf, which stage zeros the accumulation loop never reads. **A staged zero that is never read is staging hygiene and not operand padding**, so contract section 8's ban on padded operands is untouched |

---

## 4. The finding of the round, the fold stack bounds the register tile (DEVIATION 1253)

**This is the finding of the round and it is a cost of the pin that no arm
has priced.**

`gemm/UNPINNED_CONTROL.md` confound C1 records that the pinned kernel carries
`GEMM_FOLD_SLOTS = 16` float lanes of fold stack per thread whether or not
`P` needs them, and correctly says it acts through occupancy. At one cell per
thread that is a footnote. **At `RPT * CPT` cells per thread it stops being
one, because the stack is per CELL.** A 4x4 tile at fold depth 8 needs 128
float lanes of fold stack beside its 16 accumulators, and no column in
`kernel_matrix.mojo` has that many architectural registers to spare at a
useful occupancy.

An unpinned kernel carries ONE accumulator set for the whole `k` range and
has no such term, so it can take the widest tile the machine supports. **The
lever this round most wanted is the lever the pin most restricts.**

**The mitigation, DEVIATION 1254.** `_fold_push`'s `occ` is a binary counter
over the leaves pushed so far, so the highest slot index ever touched while
pushing `P` leaves is `floor(log2 P)`. A stack of `FS` slots therefore covers
every `P <= 2^FS - 1`, exactly, and NOT `2^FS`, because at `P = 2^FS` the last
push merges through every slot and needs one more. That off-by-one has the
same shape as contract 7.2.2's warning that `fold_node_total(P)` is not
bounded by `2P`, and it is checked by `_fold_push_tile`'s overflow return and
a host-side refusal rather than trusted. `FS` is a comptime kernel parameter
and the host dispatches it from `P`.

**A dispatch on `P` is a dispatch on `k` and on nothing else.** Contract 6.1
permits the NUMERICAL plan to see `k`; this is the EXECUTION plan seeing a
function of `k`, which is weaker still. Every variant computes the same
`(d, q)` nodes over the same `P` leaves in the same pairing. Contract 13.5's
last sentence forbids choosing the numerical TREE at a performance dispatch
boundary, and nothing here does; there is no `if P < 32 use the serial fold`
and there never may be.

**The register budget, counted from source, NOT measured.** Live SIMD lanes
per thread.

| plan | cells | acc | fold stack | A frag | B frag | prefetch | total |
|---|---|---|---|---|---|---|---|
| shipped `TILE 16x16 KS=32` | 1 | 1 | 16 | -- | -- | -- | ~17 |
| tuned 64x64 reg4x4 `FS = 1` | 16 | 16 | 16 | 16 | 16 | 8 | ~72 |
| tuned 64x32 reg4x2 `FS = 8` | 8 | 8 | 64 | 16 | 8 | 8 | ~104 |
| tuned 32x32 reg2x2 `FS = 16` | 4 | 4 | 64 | 8 | 8 | 8 | ~92 |
| hypothetical UNPINNED 64x64 reg4x4 | 16 | 16 | **0** | 16 | 16 | 8 | ~56 |

**The unpinned kernel gets the widest tile for 56 registers; the pinned one
needs 104 to get half of it.** That is structural and it is invisible to any
arm that times two kernels at one cell per thread.

What it costs at the shapes in `bench/gemm_shapes.mojo`.

| shape | `k` | `P` | tuned plan | what the pin cost |
|---|---|---|---|---|
| `kmeans.dist.4096x64x64` | 64 | 1 | reg4x4 `FS = 1`, 64x64 | nothing, contract 7.3, no tree |
| `pca.transform.wide.8192x64x128` | 128 | 1 | reg4x4 `FS = 1`, 64x64 | nothing |
| `llama8b.*.t512` | 4096 | 32 | reg4x2 `FS = 8`, 64x32 | half the register tile |
| `llama8b.mlp_down.t512` | 14336 | 112 | reg4x2 `FS = 8`, 64x32 | half the register tile |
| `gram.128sq.x100003` | 100003 | 782 | reg2x2 `FS = 16`, 32x32 | three quarters of it |

**The cost of the pin on the tuning is a rising function of `P`, which is a
rising function of `k`.** Contract 13.3 argues the fold is under one percent
of the ARITHMETIC at every legal `k`, which is true and is about a different
resource. Section 7 item 12 asks for that sentence to be qualified.

---

## 5. The roofline reasoning

### 5.1 The three rooflines

Flops per byte STAGED into threadgroup memory, at one `KS = 16` window.

| tile | staged floats | FMAs | flops per staged byte |
|---|---|---|---|
| 16x16 | 512 | 4096 | 4.0 |
| 32x32 | 1024 | 16384 | 8.0 |
| 64x32 | 1536 | 32768 | 10.7 |
| 64x64 | 2048 | 65536 | 16.0 |

Flops per byte read OUT of threadgroup memory. **This is the one the current
kernel is on and the one register blocking moves.**

| plan | shared loads per FMA | flops per shared byte | vector load instructions per FMA at `VEC = 4` |
|---|---|---|---|
| shipped, 1 cell per thread | 2.0 | 0.25 | 2.0 |
| tuned reg2x2 | 1.0 | 0.50 | 0.25 |
| tuned reg4x2 | 0.75 | 0.67 | 0.19 |
| tuned reg4x4 | 0.5 | 1.00 | 0.125 |

Flops per DRAM byte is `BM*BN / (2*(BM+BN))` and depends on the tile alone,
so 4.0 at 16x16, 8.0 at 32x32, 10.7 at 64x32, 16.0 at 64x64 and 32.0 at a
128x128 that is not implemented.

**An M4's balance is nearer 15 flops per byte, so the shipped 16x16 tile is
3.75x on the bandwidth side of the line and 64x64 is the first tile that
crosses it. AND THE 15 IS THE NUMBER THIS DOCUMENT IS LEAST SURE OF.** It is
the figure the lane was handed. If the real balance is nearer 30, every
prediction below is too optimistic and the next lever is a 128x128 tile,
which section 4 says is available only at `P == 1`. Measuring the balance on
the box with a stream kernel is owed.

### 5.4 The ceiling contract 5c puts on the whole exercise (DEVIATION 1261)

Seam 5c is one accumulator flush per cell per `k` step and it is irreducible.
No tiling, vectorization or staging changes the count. If a flush lowers to
`F` scalar operations, the best any realization of this profile can do on a
machine with one FMA pipe and no separate flush unit is

    FMA issue efficiency  <=  1 / (1 + F)
    F = 4  ->  20% of scalar FMA peak     F = 1  ->  50%
    F = 2  ->  33%                        F = 0  -> 100%

**So vendor libraries at 62 to 94 percent are not a reachable target for the
IDENTICAL arm at any level of engineering, and the honest ambition for this
round is 2-5 percent becoming 8 to 15 percent.** Two things could falsify the
bound and both are owed. `F` has never been counted from a disassembly on any
vendor. And on a backend whose denormal flushing is architectural, contract
section 5 says `ftz` is bitwise a no-op, so a sufficiently informed compiler
could reach `F = 0` legally, which would mean the 1.25x `pinned / strict`
measurement is telling us the compiler does NOT do that today.

## 6. The predictions, recorded before any run

Each ratio is `shipped / tuned` with only that technique enabled, which is
NOT how the file is built. Per-technique numbers are falsifiable only by an
ablation build, which is owed.

| # | technique | predicted at the `llama8b.*.t512` tiled rows | most likely to be wrong because |
|---|---|---|---|
| 3.1 | register blocking, 4x2 | 1.40x to 2.20x | it is the term 1.1 says should be biggest, and section 4 says the pin already halved it |
| 3.2 | double buffering | 1.10x to 1.40x | the doubled footprint may cost a resident block and pay the gain back |
| 3.3 | vectorized global loads | 1.00x to 1.20x | `OP_NT` only; **exactly 1.00x** at `gram.128sq.x100003`, the one tiled `OP_TN` row, because neither operand vectorizes there |
| 3.4 | operand flush hoist | 1.03x to 1.15x | 1.1 says removed instructions are mostly hidden. A 4x4 tile does 8 operand flushes where the unhoisted spelling does 32 |
| 3.5 | padded shared stride | 1.00x to 1.15x | most likely exactly 1.00x |
| 3.6 | wider tile | 1.00x to 2.00x | entirely contingent on the machine balance being 15 and not 30 |

**Combined, `shipped / tuned` lands between 1.8x and 3.0x, single most likely
value 2.2x.** The terms are NOT independent and their product is not the
answer; a naive product of the midpoints gives about 3.5x, which this
document does not predict. Above 3.0x means section 1.1's inference is wrong and
the kernel was instruction-issue bound after all.

**In absolute terms 2 to 5 percent of peak becomes 4 to 12 percent, against
vendor libraries at 62 to 94 percent. This round does not close the gap.** It
takes perhaps a third of it in log terms and leaves 5x to 15x, of which 5.2
argues 3x to 5x is a hard ceiling of contract 5c and the rest is work nobody
has done. Any write-up quoting a speedup from this round without that
sentence is `[[IDENTITY IS NOT FREE]]`'s failure mode with the sign flipped.

Four named ways to be wrong, each of which teaches something.

1. **The `KS` sweep.** `lib_smem_pages_for` gives Apple `PAGES = 2` at
   `KS = 16` and `PAGES = 1` at `KS = 32`, and gives both to NVIDIA and AMD.
   Predicted: on Apple `KS = 16` wins because it is the only one that gets a
   double buffer at all; on NVIDIA and AMD `KS = 32` wins because both get
   one and 32 halves the barrier count. If all three columns come out the
   same way, one of the two mechanisms is not doing what 3.2 says.
2. **Regressions.** `kmeans.dist.4096x64x64` at `n = 64` makes 64 blocks
   where `TILE_16_16_32` makes 1024; predicted 0.8x to 1.1x, worse on an H100
   with 132 SMs. `pca.transform.8192x4x4` and `ols.predict.gemv.64Kx16` both
   DELEGATE, so both must be exactly 1.00x, and if either moves the
   dispatcher is not doing what `tuned_gemm_plan_name` says.
   `gram.128sq.x100003` is predicted 1.0x to 1.4x, the weakest tiled row, and
   it is the row that most directly measures DEVIATION 1253.
3. **The prediction most likely to be wrong.** `TUNED_PLAN_R2C2_K16_S1` beats
   `TUNED_PLAN_R4C4_K16_S1` on Apple at the `P == 1` rows. The 64x64
   double-buffered plan claims about 20 KB of Apple's 32 KB and admits ONE
   resident block per core; the 32x32 plan claims about 10 KB and admits
   three. If 4x4 wins comfortably the kernel is reuse bound and the next
   round goes WIDER; if 2x2 wins the kernel is occupancy bound,
   `lib_smem_pages_for` returning 2 is a trap on Apple, and the next round
   goes NARROWER.
4. **The fold.** The `P == 1` rows and the `P > 1` rows will show the same
   relative improvement from 3.2 through 3.6 and a DIFFERENT one from 3.1,
   and the difference will be DEVIATION 1253 and not the fold's arithmetic.
   If tuned `P == 1` improves by 2.5x and tuned `P = 32` by 1.6x on the same
   box in the same window, the difference is the register tile the fold stack
   refused to allow, which contract section 13 currently prices at zero.

---

## 7. What would make a speedup FAKE

`[[verify reach, not output]]`, `[[reached-but-inert]]`. Every one of these
must be excluded before a single ratio is quoted.

1. **The kernel does not write its output.** A masked store with an
   off-by-one in `gi < m and gj < n`, or a grid that leaves most tiles
   unlaunched, is the fastest GEMM ever written. POISON the output buffer
   before an untimed warm-up, read back and digest, as
   `bench/gemm_price_main.mojo`'s existing arms do.
2. **The kernel reads uninitialized threadgroup memory.** If the
   `chunk == KS` fast path were ever taken at `chunk < KS` it would read
   3.7's empty-window zeros. Gate at `gram.128sq.x100003` (35-element last
   leaf) and `gram.32x32x1M` (`L = 977`, not a `KS` multiple at either `KS`),
   PER CELL and not per digest.
3. **A ping-pong race that usually works.** 3.2 removes a barrier and a
   missing barrier gives the right answer most of the time, so ONE passing
   run proves nothing. Repeat bit equality, plus launch invariance against
   `PLAN_FLAT` and `PLAN_SPLITK_STAGED`, which share no memory with this one.
4. **The gain vanishes at another shape.** All twenty rows of
   `bench/gemm_shapes.mojo`, in order, nothing dropped for being unflattering
   (`[[never build to datasets]]`).
5. **The timing loop skips a synchronize.** `tuned_gemm_into` is
   ASYNCHRONOUS. Enqueue `REPEATS` launches and synchronize once and you have
   measured enqueue throughput.
6. **The tuned arm DELEGATED and the bench reported it as tuned.** Six of
   twenty rows are predicted to delegate. Print `tuned_gemm_plan_name` per
   row and exclude a delegated row from any headline ratio.
7. **The accumulators spilled.** `bench/results/GEMM_ROUND_2026-08-19.md` is
   this repository doing exactly that: `stack_allocation` with no address
   space is thread-local MEMORY, so every accumulator access became a load
   and a store and register tiling came out SLOWER than the naive 16x16
   kernel across the board. Every accumulator in the tuned file is a lane of
   a `SIMD` value indexed by a `comptime for` bound, the spelling that fixed
   that round. **A register or occupancy readout per kernel per vendor is the
   only instrument that separates a spill from "the technique did not help".**
8. **Different thermal windows.** The M4 drifts 1.7x in twenty minutes
   (`[[the M4 drifts 1.7x in 20 minutes]]`). The tuned and shipped arms live
   in ONE binary so they can alternate CALL BY CALL; a run that alternates
   block by block is discarded.
9. **Different numeric modes.** Read the mode back from the run and never
   from the source (`[[the shared checkout's mode flip]]`).
   `tuned_gemm_banner` prints the resolved kernel-matrix column beside the
   mode, because a simulated column changes `PAGES`.
10. **A sabotaged build.** In a `-D MOJOLEARN_GEMM_SABOTAGE_*` build the
    shipped kernel's arms compile in and the two are not comparable.
    `SAB_NODE_ORDER` is the exception in the other direction, since it
    sabotages a split-K workspace address and the tuned file has no
    workspace; a gate should assert a `SAB_NODE_ORDER` build of the tuned
    kernel is bit-identical to an unsabotaged one, and report that as a
    statement about COVERAGE rather than as a pass.
11. **Quoting the result as "the speed of identical GEMM".** One machine, one
    dtype, one shape family, on a kernel that has never been compiled.

---

## 8. OWED, AND WHY I DID NOT DO IT HERE

Nothing on this list has been done.

**Before anything may route to the tuned kernel.**

1. Extend `gemm/checks/gemm_device_check.mojo` to the tuned plans. Per-cell
   bits against `gemm_oracle` at all 62 shapes, launch invariance requiring
   every tuned plan to equal `PLAN_FLAT` and `PLAN_SPLITK_STAGED`, batch
   invariance across a dispatch boundary and across composition, and the five
   reachable sabotages shown to FAIL. **Until this passes the tuned file is a
   proposal about how to go faster and not a realization of the profile.**
2. `check_tuned_stack_fold_is_the_contract_tree`, DEVIATION 1263.
   `_fold_push_tile` and `_fold_drain_tile` against
   `gemm_oracle.fold_balanced_tree` for every `P` in `1 .. 2049` on hashed
   partials, at every `FS` the file compiles. The scalar `_fold_push` has
   this proof; the tiled spelling has only an argument.
3. `check_tuned_fold_depth_covers_the_plan`. That `tuned_plan_max_leaves`
   equals `2^FS - 1` and not `2^FS`, by running `_fold_push_tile` at both and
   requiring the overflow return to flip, and that the deepest plan covers
   `CONTRACT_MAX_LEAVES`.
4. A `bench/gemm_shapes.mojo` row with `P` between 256 and 1024 and an `m n`
   large enough to tile. `gram.128sq.x100003` at `P = 782` is the only such
   row today, so the `FS = 16` plan has ONE fixture.

**To get a number at all.**

5. A `tuned` arm in `bench/gemm_price_main.mojo` beside `device`, `vendor`,
   `pinned`, `unpinned` and `unpinned1`, with a poisoned warm-up, a digest,
   and its call inside the same `for _ in range(REPEATS)` loop so the arms
   alternate call by call. Plus per-technique ABLATION builds, because 5.3's
   six predictions are unfalsifiable from a combined number.
6. `-D MOJOLEARN_GEMM_TUNED_ARM=1` on `tools/gemm_price.sh`'s IDENTICAL leg,
   and `tuned_gemm_banner()` in the mode witness block. A pixi task for the
   probe; `tools/e2_remote_leg.sh` already carries a `tuned-probe` arm and
   nothing else does.
7. A register-count and occupancy readout per kernel per vendor. Section 6
   item 7 and DEVIATION 1253.
8. Measure the machine balance on the box. Every prediction in 6 rests on
   5.1's 15 flops per byte.

**Edits to files this lane was not permitted to touch.**

9. Possibly make `_leaf_bounds`, `_leaf_at`, `_tile_grid` and the `SAB_*`
   aliases public in `gemm_identical.mojo`. **A COPY of any of them into the
   tuned file is not an acceptable fix**, because a drifted leaf boundary
   would make the two kernels differ in more than the tuning.
10. A `lib_acc_th_cols` row in `checks/kernel_matrix.mojo`. `TUNED_TC = 16`
    is the one scheduling number the tuned file names itself, and the matrix
    has no row for the thread grid, so a vendor measurement of it has nowhere
    to land.
11. Register DEVIATIONS 1250 through 1264 in `gemm/README.md`. They are
    defined in this file and the tuned file and nowhere else (checked).
12. Two edits to `gemm/IDENTICAL_FP32_CONTRACT.md` section 13, an entry for
    the tuned arm and a qualification of 13.3, which is true about the
    arithmetic and is being read as covering the register budget.
13. A pointer from `gemm/UNPINNED_CONTROL.md` confound C1 to DEVIATION 1253.
    C1 calls the fold stack an occupancy effect; at more than one cell per
    thread it is a hard bound on the largest available tile.

**Considered and deliberately NOT implemented.**

14. **DEVIATION 1262. Matrix and tensor instructions**, `simdgroup_matrix`,
    `mma.sync`, MFMA. **Almost certainly REFUSED by this contract and the
    largest single unclaimed component of the 12x to 25x.** An MMA
    instruction accumulates its own partial products in an order no vendor
    specifies, there is no per-step seam at which contract 5c's flush could
    apply, and on NVIDIA the FP32 path is TF32
    (`kernel_matrix.column_vendor_fp32_matmul_is_tf32`), which is not the
    same arithmetic. The honest statement is that the identical profile
    forgoes the matrix pipelines and nobody has quantified that cost.
15. Spilling the high fold-stack levels to a global workspace so a 4x4 tile
    survives at `P > 1`. Bit-safe by DEVIATION 1254's argument, costs roughly
    6 percent extra DRAM at `llama8b.qkv.t512`, and ends the tuned file's "no
    global scratch at any shape" property.
16. Eliding `ftz` on a backend whose flushing is architectural. Contract
    section 5 says `ftz` is provably inert on Metal. **This lane did not act
    on that and the brief is explicit that it may not.** The owed EXPERIMENT
    is to disassemble the shipped tiled kernel on Metal and report `F`; if
    `F` is already 0 on Apple then the measured `pinned / strict = 1.25x` is
    not the flush and `gemm/UNPINNED_CONTROL.md` prediction 5 needs redoing.
    **A source-level elision is forbidden whatever the disassembly says.**
17. A 128x128 tile at 8x8 outputs per thread. Available at `P == 1` only, for
    DEVIATION 1253's reason: the pin does not merely slow the identical GEMM
    down, it makes the next tuning rung unreachable at every shape with more
    than one leaf.
18. A tile swizzle chosen for DRAM locality. All three `SWIZZLE_*`
    permutations are imported and the dispatcher uses `SWIZZLE_NONE`; a
    bijection cannot move a bit, so this is a cheap owed sweep.
19. `KS = 64` or `KS = 128`. The footprint does not fit doubled on any column
    at a 64x64 tile, so it trades the double buffer for the barrier count and
    the two effects have opposite signs.
20. A second and a third vendor. `PAGES` resolves DIFFERENTLY on Apple than
    on the other two at `KS = 32`, so the tuned kernel is not even the same
    kernel across the three columns.

**What the author is least confident compiles**, in descending order of
doubt, kept because the file has never been built and a compile error is the
failure mode you want.

1. `comptime for` over a bound that came from a function PARAMETER with a
   SIMD lane write inside, nested in a `while`. Every one is written against
   a `comptime` LOCAL because `simt_kernel.mojo` uses that shape and ships.
2. `SIMD[DType.float32, 128]`, the `FS = 8` 4x4 fold stack of
   `TUNED_PLAN_R4C4_K16_S8`. It is a sweep-only plan the dispatcher never
   returns, so dropping it costs no shape.
3. The underscore imports from `gemm_identical.mojo`. The `SAB_*` names are
   module-level `comptime` values and nothing in this tree imports one across
   modules yet.
4. `stack_allocation` with a comptime count derived from a
   `lib_smem_pages_for` call inside a parametric launcher.
5. `as_.unsafe_load[width=VEC](offset)` on a SHARED-address-space pointer at
   a `VEC` threaded from another module.
6. `comptime assert` on a parameter rather than on a comptime local.
7. `comptime kern = tuned_gemm_tiled_kernel[RPT, CPT, TC, KS, FS, PAGES]`
   inside a parametric `def`, five of six parameters from the enclosing list.
8. Forward references, `tuned_gemm_plan_name` calling `tuned_plan_fold_slots`.
