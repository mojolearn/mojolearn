# The unpinned control arm

The single-variable arm that prices `mojolearn.identical.gemm.fp32.v1`.

Written 2026-08-25 alongside `gemm/checks/gemm_unpinned.mojo`. **Neither
file has been compiled and neither has been executed.** The lane that wrote
them had no execution rights of any kind. Everything in section 5 is a
PREDICTION recorded before measurement, in the discipline of
`IDENTITY_COST_PLAN.md` Part 1, and a prediction that turns out wrong is the
most useful thing this document can produce.

DEVIATIONS 1130 through 1136. The lane's range is 1130 through 1149 and the
unused numbers 1137 through 1149 are reserved for whoever builds and runs it.

---

## 1. The question, and why no existing arm answers it

`bench/gemm_price_main.mojo::_timed_shape_with_device` times five arms and
DEVIATION 1092 says outright that none of them isolates the pin.

| tempting comparison | what it actually varies | why it is not the experiment |
|---|---|---|
| `device` vs `pinned` | tiling, staging, plan selection AND the pin | `pinned_gemm_nt_kernel` is FLAT, one thread per cell, serial whole-`k` loop, no shared memory. Measured on the M4 at full llama8b shapes on 2026-08-25, `device` BEATS `pinned` by 1.6x to 1.8x. That says tiling beats not-tiling. |
| `device` vs `vendor` | our kernel engineering, Modular's kernel engineering AND the pin | Two terms move at once, and on an H100 MAX's matmul measured 200 TFLOP/s against a 67 TFLOP/s FP32 non-tensor peak, so it is not even the same arithmetic. |
| `device` IDENTICAL vs `device` FAST | seams 4 and 5 only | The right shape of experiment and it prices two of the six clauses. `GLOBAL_NUMERIC_MODE` is comptime, so the two sides are two BINARIES in two thermal windows, and the partition, the tree and the fold-stack registers are IDENTICAL in both, by that mode switch's own design. |

The arm that answers it is the identical kernel's own execution plans with
the numerical plan of `IDENTICAL_FP32_CONTRACT.md` removed and nothing else
touched. That is `gemm/checks/gemm_unpinned.mojo`.

**One consequence is worth stating before anything else.** The unpinning is
written out longhand in a separate kernel rather than reached through
`NUMERIC_FAST`, so the pinned arm and the unpinned arm live in ONE binary and
alternate call by call inside one thermal window. The M4 drifts 1.7x in
twenty minutes and the shell has to work around that at the mode level; this
arm defeats it outright at the arm level. DEVIATION 1136.

---

## 2. The pin is a conjunction. Clause by clause, dropped or held

The pin is six clauses, not one, and an arm that drops them together can only
report a sum. The instrument drops all six and exposes the largest of them as
a parameter so the sum can be split in two.

| # | clause | contract | dropped? | how, in the instrument |
|---|---|---|---|---|
| 1 | the leaf size and leaf count rule as a pure function of `k` alone | 6, 6.1, 6.2 | **as arithmetic, YES. As schedule, NO** | `contract_partition(k)` is still imported and still called, and `leaf` and `p_count` are still passed to every kernel. The accumulators live ACROSS leaf boundaries, so no leaf boundary is a fold boundary any more. See section 3, item 1 -- this split is the whole reason the arm is single-variable. |
| 2 | serial ascending accumulation within a leaf, no sub-partition | 7.1 | **YES** | `NACC` independent accumulator lanes in the `k` direction, combined at the very end. This is verbatim the construction 7.1 names and forbids. `NACC = 4` in `unpinned_gemm_into`, `NACC = 1` in `unpinned_gemm_into_one_acc`. |
| 3 | `identical_mul_add` rather than a contracted FMA | 4, 4.1, 4.2 | **YES** | The source reads `acc + x * y`. `identical_mul_add` is not imported. The backend contracts it or does not, per backend, per context, which is exactly the freedom the pin removes. |
| 4 | `ftz` at the seven named seams 5a..5g | 5 | **YES, all seven** | `ftz` is not imported. Operands unflushed (5a, 5b), accumulator unflushed after every step (5c), leaf partial unflushed (5d), fold reads and results unflushed (5e, 5f), output store unflushed (5g). Section 5 of the contract calls 5c the largest single cost item in the profile, so this row is the one to watch. |
| 5 | the fixed balanced adjacent-pair fold tree with bit-for-bit carries | 7.2, 7.2.2 | **YES** | FLAT and all five TILE plans have NO fold at all -- there is one accumulator set for the whole `k`. SPLITK folds by STRIDE (`buf[q] + buf[q + half]`), which is `pinned_block_sum`'s shape and fixture F8's defect, with no carry rule. |
| 6 | `+0.0` fold seeding and the unconditional fold at `P == 1` | 7.3, 9.2 | **YES, and the clause dissolves** | With one accumulator set per cell rather than one per leaf, the leaf seed of 9.2(a) is a single seed for the whole product and `P == 1` is not a case. `-0.0` behavior is therefore unspecified in this arm and no gate may assert on it. |

Two smaller pins go with them and are recorded so nobody counts them twice.
`gemm_operand_strides` is KEPT, so the three orientations are still one
numerical implementation and `OP_TN` still reads its `k x m` operand
correctly. The `k == 0` store of `+0.0` (contract 8) is kept implicitly, in
that the FLAT fallback still runs and still stores, but nothing asserts it.

---

## 3. What is held fixed, and the two places the holding is delicate

Held bit for bit, and reached BY IMPORT rather than by copy, because a copied
constant is a constant that drifts and a drifted tile constant would put the
tiling back into the experiment.

- `choose_gemm_plan` -- **the same shape picks the same plan**. This is the
  single most important thing held, and `unpinned_gemm_plan_name` exists so a
  run can print both arms' plans and prove it.
- `contract_partition`, `gemm_operand_strides`, `_leaf_bounds`, `_tile_grid`.
- `FLAT_TPB`, `SPLITK_LEAF_TPB`, `SPLITK_FOLD_TPB`, `SWIZZLE_NONE`,
  `SWIZZLE_REVERSE`, `SWIZZLE_TRANSPOSE`, `CONTRACT_MAX_LEAVES`.
- the staging loops, the `barrier()` placement, the `TM * KS` and `KS * TN`
  threadgroup allocations, the tile bijection, the out-of-range store mask.
- the workspace. `unpinned_gemm_workspace_floats` is a call-through to
  `identical_gemm_workspace_floats`, equal by construction, and the SPLITK
  arm uses the same `stride = P` layout so the same allocation serves both.

**Delicate holding, item 1. The leaf loop survives as a schedule.**
DEVIATION 1131. The leaf partition does two jobs in the pinned kernel and
only one of them is arithmetic. The other is that the staging window is
clamped to the leaf's end, so wherever the leaf length is not a multiple of
`KS` the pinned kernel pays an extra short staging round trip and an extra
pair of barriers at the boundary. Delete the leaf loop and the arm would move
the traffic as well as the arithmetic, and the confound would be back. So the
outer `for t in range(p_count)` loop and the `_leaf_bounds` clamp are still
there, doing nothing arithmetic, purely to keep the barrier count and the
transaction count identical at every shape including the ragged tail. At
`k = 1,000,000` the leaf is 977 elements against a `KS` of 32, so this is not
a hypothetical.

**Delicate holding, item 2. The split-K fold keeps the ping-pong.**
DEVIATION 1133. A stride fold can run in place, which would halve the
threadgroup footprint and change occupancy on all three vendors. The
instrument allocates `2 * CONTRACT_MAX_LEAVES` floats and uses half, wasting
threadgroup memory on purpose because it is the cheap way to hold an
occupancy input still. The level count is `ceil(log2 P)` under both pairings,
so the barrier count is held too.

`PLAN_SPLITK_STAGED` is REFUSED rather than unpinned (DEVIATION 1134). It is
a realization OF the tree, one global launch per logical level at
`fold_level_base` addresses, and with the tree gone it has nothing to
realize. `choose_gemm_plan` never returns it, so no shape is lost.

---

## 4. The confounds that remain, named

A confound you name is a result. A confound you hide is a lie. These are the
ones the instrument could not remove, in descending order of how much damage
they can do to a headline ratio.

**C1. The fold stack's registers.** The pinned kernel carries
`GEMM_FOLD_SLOTS = 16` float lanes of fold stack plus an `occ` mask per
thread, at every plan, whether or not `P` needs them. The unpinned kernel
carries `NACC` lanes -- 4, or 1 in the strict arm. So the strict arm uses
FEWER registers per thread than the pinned one and may reach a higher
occupancy for that reason alone. This cost is ATTRIBUTABLE to the pin, since
the fold stack exists only because the tree does, but it acts through
occupancy and not through arithmetic, so a large measured gap may be an
occupancy effect wearing an arithmetic effect's clothes. **Nothing in the
timing can separate the two.** A register-count or occupancy readout from the
toolchain would, and it is not in this instrument.

**C2. The staging loop is a copy, not a shared body.** Mojo has no way to
share a loop body between two kernels that differ in their middle, so
`unpinned_gemm_tiled_kernel`'s staging half is a character-for-character copy
of `identical_gemm_tiled_kernel`'s. **If one is edited and the other is not,
the arms differ in the traffic and every number from the run is worthless,
silently.** This is the most likely way this experiment rots.

**C3. The five tile tuples are transcribed, not imported.** The
`(TM, TN, KS, swizzle, two_d)` arguments at the five `_launch_tiled` call
sites are call-site arguments and not named constants, so
`unpinned_gemm_with_plan` transcribes them. Same rot risk as C2, smaller
blast radius, and `unpinned_gemm_plan_name` catches only a plan mismatch and
not a tuple mismatch.

**C4. The split-K pairing changes the bank pattern.** Adjacent-pair reads
`2q, 2q+1` and stride reads `q, q+half` hit threadgroup banks differently.
That is inherent -- the pairing IS the variable -- but it means the split-K
fold row prices "adjacent pairing plus its access pattern" and not the
pairing alone.

**C5. `NACC > 1` raises register pressure.** That is a cost the unpinned arm
pays for a benefit the pin forbids it to take, so it is part of the clause-2
measurement rather than beside it. It is listed because a reader comparing
register counts across the three arms will otherwise think something is
wrong.

**C6. Compiler variance between two distinct kernel functions.** Two
functions can be scheduled and register-allocated differently for reasons
unrelated to the source difference. Not visible without a disassembly.

**C7. The ceiling.** An unpinned kernel wearing the identical kernel's tiling
is still not a TUNED kernel. No tensor cores, no double buffering, no
register blocking in `m` or `n`, no vectorized shared loads, no `KS` chosen
for the machine. **The number this arm enables is what the pin costs at OUR
level of engineering, not what the pin costs in principle**, and it is a
LOWER bound on the pin's cost rather than an upper one. `device / unpinned`
must never be quoted without that sentence attached.

**C8. Five of the twenty shapes have `P == 1`.** At `P == 1` there is no fold
and clause 5 is untested by construction, so those rows price clauses 2, 3
and 4 only. That is a feature if it is reported and a trap if it is not.

**C9. Sabotage builds are not comparable.** The instrument carries no
`SAB_*` switches, because it makes no identity claim and there is nothing to
falsify. In an unsabotaged build the pinned kernel's `comptime if SAB_*`
branches compile out and the two arms are comparable; in a sabotaged build
they are not, and no timing may be taken from one.

---

## 5. The prediction, recorded before the measurement

Written on op counts and machine balance, in the discipline of
`IDENTITY_COST_PLAN.md` Part 1 -- counting is free and it is not measurement.
If the run disagrees, the run is right and this section is deleted rather
than defended.

### 5.1 The op count the prediction rests on

The pinned inner step is two loads, two operand flushes, one `fma`, one
accumulator flush. A flush is a magnitude compare, a zero compare, an and,
and a select, so call it roughly four scalar operations. **The pinned step is
about one FMA plus about twelve scalar operations. The unpinned step is one
FMA, or a multiply and an add.** On an ALU-bound kernel that is a factor of
several; on a bandwidth-bound kernel it is nothing. Which regime each shape
is in is the prediction.

`TILE 16x16 KS=32` stages 1024 floats per window and performs 8192 FMAs
against them, so about 4 flops per byte moved into threadgroup memory,
against a machine balance nearer 15 on an M4. **The tiled plans in this
repository are on the bandwidth side of the line**, which is the single fact
that shapes every number below.

### 5.2 The predictions, each falsifiable on its own

1. **`unpinned / device` at the `t512` tiled rows lands between 1.10x and
   1.45x.** Not the 3x to 5x the raw op count suggests, because the tiles are
   bandwidth-limited and the removed operations are mostly hidden.
2. **The strict arm and the `NACC = 4` arm land within 10% of each other on
   every GPU row.** A GPU hides a dependency chain with thread-level
   parallelism, not with instruction-level parallelism, so contract 7.1's
   no-sub-partition clause -- the clause that looks most expensive on paper
   -- should be nearly free here. **This is the prediction most likely to be
   wrong**, and if it is wrong it is wrong in the direction of the ILP arm
   pulling away at the large tiles, which would mean the identical kernel is
   more latency-bound than the balance argument says.
3. **The `t1` rows come in at 1.00x to 1.06x.** With `m = 1` the weights are
   streamed once with no reuse and the shape is pure bandwidth, so the pin
   should be invisible. `llama8b.qkv.t1` runs SPLITK (`m*n = 4096`, which is
   exactly the dispatcher's threshold) and `llama8b.lm_head.t1` runs
   `TILE_8_32_32`, so the row pair also checks that the plan split is being
   reported.
4. **The fold contributes under 2% of the gap at every shape in the table.**
   Contract 13.3 argues the fold is under 1% of the arithmetic at every legal
   `k`, and the five `P == 1` rows are the control for it -- they have no fold
   at all, so if their gap resembles the `P > 1` rows' gap, the fold is not
   where the cost is. If a `P > 1` row shows a much larger gap than a
   comparable `P == 1` row, contract 13.3 is wrong and that is the most
   valuable result the run can return.
5. **The largest single term is the accumulator flush, 5c.** Contract section
   5 predicts this in words and this arm is the first thing that can
   contradict it. If the total gap is small AND the strict arm equals the
   pinned arm, the flush is being hidden or elided, and the next question is
   whether the toolchain emits it at all.
6. **`gram.32x32x1M` shows the largest gap of the twenty rows**, because it
   is the only row where a single thread walks `L = 977` elements at a leaf
   with `m*n = 1024` cells of parallelism -- the one shape where there is not
   enough work to hide anything. Predicted 1.3x to 1.8x.
7. **On Apple, at every `P == 1` row, the two arms produce IDENTICAL BITS
   while differing in time.** Metal through MAX contracts by default and
   flushes denormals by default, so at `P == 1` with normal data the pin is
   bit-inert and only the instruction count separates the arms. That is not a
   failure of the arm; on a non-contracting backend those same rows should
   separate in bits, and the difference between the two vendors is itself the
   measurement. **Do not use bit equality at a `P == 1` row as evidence the
   arm did not run** -- section 6 gives the reach check that does work.

### 5.3 The one prediction about the paper

If predictions 1 and 3 both hold, the honest headline is that the pin costs
between nothing and roughly a third at transformer shapes on this hardware,
that the cost is concentrated in the denormal flush rather than in the fold
or the partition, and that the fold -- which is the part of the contract that
took the most argument to specify -- is the cheapest clause in it. Standing
rule `[[IDENTITY IS NOT FREE]]` binds the writing of it. Conforming costs on
every vendor, this arm measures one, and a small number here is not a licence
to say identity is free anywhere.

---

## 6. How it will be judged

### 6.1 The comparison

Three arms, in ONE binary, built with `-D MOJOLEARN_NUMERIC_IDENTICAL=1` AND
`-D MOJOLEARN_GEMM_UNPINNED_ARM=1`, alternating call by call inside one
thermal window.

| arm | call | what it is |
|---|---|---|
| `device` | `identical_gemm_into` | the pinned v1 kernel, unchanged |
| `unpinned` | `unpinned_gemm_into` | the same plans, all six clauses dropped, `NACC = 4` |
| `unpinned1` | `unpinned_gemm_into_one_acc` | the same, `NACC = 1`, the strict control |

Report `device / unpinned` and `device / unpinned1` per shape. The gap
between the two ratios is contract 7.1's clause; `device / unpinned1` is
everything else.

Build it in the IDENTICAL binary. In a FAST binary the `device` arm is
already unpinned at clauses 3 and 4 and the comparison degenerates to a
measurement of the partition and the fold alone.

### 6.2 The shapes

All twenty rows of `bench/gemm_shapes.mojo`, in the order they are in.
**Nothing is dropped for being slow, unflattering or uninteresting**
(`[[never build to datasets]]`). The rows that carry the argument are

- `llama8b.*.t512`, `TILE_16_16_32`, `P = 32` or `P = 112` -- the flagship;
- `llama8b.*.t1`, split between SPLITK and `TILE_8_32_32` -- the
  bandwidth-bound control for prediction 3;
- `kmeans.dist.4096x64x64`, `pca.transform.wide`, `ols.predict.gemv`,
  `pca.transform.8192x4x4` -- `P == 1`, no fold, the control for prediction 4;
- `gram.32x32x1M`, SPLITK at `P = 1024` with `L = 977` -- prediction 6, and
  the only row where the ragged staging window is not aligned to `KS`;
- `gram.128sq.x100003`, `TILE_16_16_32` at `P = 782` with a 35-element last
  leaf -- the ragged-tail row.

Contract 13.6.1 also asks for a `P` sweep at fixed small `m n` over
`1, 2, 3, 8, 32, 128, 1018, 1024`, with `P = 1018` in it because it is the
carrying case and the carry path is otherwise priced at zero. That sweep is a
separate run and is owed.

### 6.3 The reach check, before any timing is believed

`[[verify reach, not output]]`, `[[reached-but-inert]]`. Three gates, all of
which must pass before a single ratio is quoted.

1. **The banner.** `unpinned_gemm_banner(nacc)` must print ENABLED. A binary
   without the define raises rather than silently returning the pinned
   answer, so a missing define cannot become a fast number.
2. **The digests must DIFFER.** At every shape with `P > 1`, the unpinned
   arm's output digest must differ from the pinned arm's. Equal digests at a
   `P > 1` shape mean the unpinning did not happen and the row is void. At
   `P == 1` on Apple they are expected to AGREE (prediction 7) and agreement
   there proves nothing either way.
3. **The poison check.** Each arm gets an untimed warm-up against a poisoned
   output buffer, read back and digested, exactly as the existing four arms
   do. A kernel that launches without writing its output must not be allowed
   to turn in the best time in the table.

### 6.4 What would make the result VACUOUS

- **Different plans.** If `unpinned_gemm_plan_name` and the pinned arm's plan
  differ at any row, that row is not the experiment. It should be
  structurally impossible since both call the same imported function, and it
  is checked anyway because impossible things happen when a file is edited.
- **Equal digests at `P > 1`.** See 6.3, gate 2.
- **A gap inside the noise.** If `device / unpinned` sits within the run to
  run spread of `device / device`, the arm has measured nothing and must say
  so rather than report a ratio near 1.00 as a finding. The spread has to be
  measured in the same window, not assumed.
- **A gap at EVERY shape of exactly 1.00.** Two very different things produce
  this and they must not be merged. Either the toolchain elided the pinned
  arm's flushes, in which case the pin is free at the instruction level and
  the finding is about the toolchain and needs a disassembly, or every shape
  in the table is bandwidth-bound and the instrument cannot see the pin at
  all, in which case the finding is that the pin is invisible AT THESE SHAPES
  and a compute-bound shape is owed.
- **A sabotaged build.** See C9.
- **Arms measured in different thermal windows.** The whole design of the
  instrument is that this is not necessary. If a run alternates block by
  block instead of call by call, discard it.
- **Quoting it as the price of identity in general.** See C7. It is a lower
  bound at our level of engineering, on one machine, for FP32 GEMM.

---

## 7. What the run should print

One line per shape, and the plan name from the IMPORTED `gemm_plan_name` on
both arms, so a mismatch is visible rather than inferred. `L`, `P` and
whether the last leaf is ragged, because prediction 4 is read off the `P`
column. Both ratios. Both digests. The banner, once, at the top, with the
numeric mode witness the rest of the file already prints -- `[[the shared
checkout's mode flip]]`, read the mode back from the run and never from the
source.

---

## OWED, AND WHY I DID NOT DO IT HERE

This lane could write exactly two paths, `gemm/checks/gemm_unpinned.mojo`
and this file. Everything below is required to turn the instrument into a
measurement and none of it was done here.

1. **`bench/gemm_price_main.mojo` -- add the two arms.**
   `_timed_shape_with_device` needs `unpinned` and `unpinned1` beside
   `device`, `vendor` and `pinned`, each with a poisoned warm-up, a digest,
   and its call inside the same `for _ in range(REPEATS)` loop so the arms
   alternate call by call. `unpinned_gemm_into` mirrors
   `identical_gemm_into`'s signature argument for argument specifically so
   this is an addition and not a refactor. Two new `mut ns_*` counters and
   two new columns.
2. **`bench/gemm_price_main.mojo` -- DEVIATION 1092's text is now FALSE.** It
   says the arm "does not exist anywhere in this repository". Once
   `gemm_unpinned.mojo` lands, that sentence is wrong, and
   `[[fix-docs-on-discovery]]` requires the false sentence deleted in the
   SAME commit rather than annotated. The replacement should say the arm
   exists, name this file, and say it has not yet run.
3. **`tools/gemm_price.sh` -- add `-D MOJOLEARN_GEMM_UNPINNED_ARM=1`** to the
   IDENTICAL leg's build line only, and print the banner in the mode witness
   block. The FAST leg must NOT get the define, per section 6.1.
4. **`gemm/IDENTICAL_FP32_CONTRACT.md` section 13.6 -- a seventh
   measurement.** The six listed measurements do not include an arm that
   isolates the pin, which is why 1092 had to be written at all. The new item
   should point at this file and carry section 6.4's vacuity list.
5. **`gemm/README.md` -- register DEVIATIONS 1130 through 1136.** They are
   defined in the two files this lane wrote and nowhere else, and a deviation
   that lives only in a docstring is a deviation the next lane will renumber
   over.
6. **`gemm/checks/gemm_identical.mojo` -- possibly make `_leaf_bounds` and
   `_tile_grid` public.** The instrument imports both by their underscore
   names. There is precedent for that in the tree
   (`glm/impl/glm/qn/glm_linear.mojo` imports `_read_scalar`), so it is
   expected to work, but if the compiler refuses the fix is to rename them
   without the underscore and update both call sites in the same commit. A
   COPY of either function into the instrument is not an acceptable fix, for
   the reason in C2.
7. **The 13.6.1 `P` sweep.** A separate small-`m n` driver over
   `P = 1, 2, 3, 8, 32, 128, 1018, 1024`, all three arms, which is the only
   measurement that can support or refute contract 13.4 and the only one that
   prices the carry path at anything other than zero.
8. **`IDENTITY_COST_PLAN.md` -- write the static estimate for GEMM down
   FIRST.** Part 1's five terms (transcendental, flush, fold depth, refused
   hardware, occupancy) are countable from the contract without running
   anything, and the whole point of that plan is that the measurement
   FALSIFIES the count rather than sources it. Counting after the run is
   worth nothing.
9. **A register-count or occupancy readout.** Confound C1 cannot be resolved
   by timing alone. Whatever the toolchain will report per kernel, on each
   vendor, is what separates an arithmetic gap from an occupancy gap.
10. **A second vendor.** This is one machine. `[[IDENTITY IS NOT FREE]]` is
    explicit that the Apple number is not the NVIDIA or AMD number, and on a
    non-contracting backend clause 3 stops being bit-inert and may cost
    differently.

---

## Appendix. What the author is least confident compiles

The instrument has never been built. In descending order of doubt.

1. **`comptime for u in range(NACC)` where `NACC` is a function parameter,
   with a SIMD lane write `accs[u] = ...` inside, NESTED INSIDE A `while`
   LOOP.** `gemm_identical.mojo` uses `comptime for` only over a module-level
   `comptime` bound and only at the top level of a function body. Both the
   parametric bound and the nesting are new here. If it fails, the fallback
   is a hand-written unroll for the two `NACC` values the file actually uses,
   which costs a maintenance hazard and no correctness.
2. **The underscore imports `_leaf_bounds` and `_tile_grid`.** See OWED 6.
3. **`comptime assert (NACC & (NACC - 1)) == 0, "..."`** inside a parametric
   `def`. `comptime assert` must be in a function body, which it is, but this
   is the only place in the file that evaluates a bitwise expression on a
   comptime parameter inside an assert.
4. **`comptime kern = unpinned_gemm_tiled_kernel[TM, TN, KS, NACC]` inside a
   parametric `def`.** The pinned file does the three-parameter version at
   the top level of a parametric launcher, so the shape is precedented, but
   the fourth parameter arriving from the enclosing function's parameter list
   is not.
5. **`SIMD[DType.float32, 1]` in the strict arm**, and the `comptime for u in
   range(1, NACC)` combine loop over an EMPTY range when `NACC == 1`.
