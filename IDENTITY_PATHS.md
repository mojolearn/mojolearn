# Every pathway that can move a bit, and what `IDENTICAL` does about it

Opened 2026-08-21, when Andrew asked the question this document exists to
answer: *will the identical column actually be identical across every GPU?*

The answer was **no**, and the reason was not the histogram — the histogram is
the one part that was already right. The reason is that the enumeration was
incomplete, so pathways with no pinning at all sat outside a mechanism that
was designed correctly to hold them.

**This file is the enumeration.** It is the thing that makes the claim
provable rather than hopeful, and it is the artifact the paper is about.

## The rule

For a NUMERIC pathway, `IDENTICAL` has exactly three legitimate moves:

| move | when | example |
|---|---|---|
| **PIN** | the operation is deterministic given a parameter, and the parameter is machine-derived | the partition-stats chunk count |
| **REPLACE** | the operation is order-dependent in itself | float atomic → fixed-point integer accumulator |
| **REFUSE** | neither is available yet | a column that misses the identity floor |

There is no fourth move, and in particular there is no "usually fine". A
toggle that silently returns a non-identical model is worse than no toggle,
because it converts a checkable property into a belief.

## The ledger

| # | pathway | order-dependent? | what `IDENTICAL` does | status |
|---|---|---|---|---|
| 1 | global histogram flush | float `atomicAdd` | REPLACE — fixed-point Int32 | **closed** |
| 2 | hist_2 shared accumulation | float adds into per-warp slices | REPLACE — shared Int32 atomics | **closed** |
| 3 | histogram block size / replication factor | sets how many partials combine | PIN — the frozen floor bounds it | **closed — RE-OPENED AND RE-CLOSED 2026-08-22**: the fan-out audit found the closure lived only in `spec_for` (the runtime REPORT) while `block_size_for`, the comptime accessor the kernels compile against, read the device column unconditionally (NVIDIA IDENTICAL would have built 768-thread float replicas against the pinned 512 geometry; Apple's 32 KB coincidence hid it). Gated at the accessor (reads `GLOBAL_NUMERIC_MODE`, every caller inherits), `501d000`. Replication factor itself pinned through `partition_chunks_sm_for` — DEVIATION 354. |
| 4 | replication lane count | a summation order | PIN — 32 on every vendor | **closed** |
| 5 | reduce stage width | a summation order | PIN — 512 | **closed** |
| 6 | library cross-lane folds (`PINNED_LIB_REDUCE_LANES`) | a summation order | PIN — 32 | **closed** |
| 7 | **partition stats → LEAF VALUES** | **chunk count = f(core count)** | **PIN — `partition_chunks_sm_for`** | **closed 2026-08-21 — and the CLASS swept 2026-08-22**: three more members found and pinned through the SAME function — `std_dev_blocks` (doc-parallel noise, DEVIATION 252), `target_variance_blocks` (greedy noise, DEVIATION 353, live on every `random_strength != 0` fit), histogram `replication_for` (DEVIATION 354). One pin to audit. RF/ET checkouts swept clean: NO machine-derived count anywhere in ensemble/ or extratrees/ (cuML's own `4*SM*256` cross-GPU bug was NOT inherited). |
| 8 | **fixed-point scale magnitude** (and `functionValue`, same reduce) | ~~device float reduce + float atomic~~ **the atomic half CLOSED 2026-08-21** (the "DETERMINISM FIX" block, `pointwise_targets.mojo:607`): every block STORES its partial, `deterministic_sum_lanes_kernel` folds them in one fixed 256-thread shape, and the partial COUNTS are pure `f(n_rows)` — `mse_blocks`, `multilogit_blocks`, `bootstrap_grid_blocks` all verified machine-independent 2026-08-22, no row-7-class hazard. What REMAINS: the per-block partial itself is MAX's `block.sum`, whose internal cross-lane fold follows the hardware warp width — 32 on Apple and NVIDIA, **64 on AMD** — so the partial's bits differ on the AMD column only | REPLACE the remainder — `pinned_block_sum` (pointwise_targets.mojo, DEVIATION 251): FAST arm IS the library call, IDENTICAL arm the `deterministic_sum_lanes_kernel` shared-memory shape | **DEVICE column CLOSED; AMD residue CLOSED at 14 of 16 producer sites 2026-08-22** — pointwise + cross-entropy + multilogit (12 sites, DEVIATION 251), `std_dev_partials_kernel` (251's 13th), `compute_target_variance_kernel` (`4e395b4`); bootstrap was already a fixed shared fold. REMAINING two: `gbdt/methods/kernel/exact_estimation.mojo:228` (`block_sum[1024]`) and `gbdt/gpu_util/partitions_reduce.mojo:156,209` (leaf-values path — resolve together with the parked DEVIATION 250 patch, which rewrites that file). Same fix: import `pinned_block_sum`. |
| 9 | **FMA contraction** | codegen decision | **CONSTRUCTION LANDED 2026-08-21**: `numerics.identical_mul_add` -- explicit `fma` under IDENTICAL (one rounding, identical on Metal/PTX/AMDGPU), the naive chain under FAST. Apple's FAST baseline measured UNFUSED (`check-ieee-arith`: fused 0 / unfused 1,046,394 of 2^20), so IDENTICAL differs from FAST on Apple BY DESIGN and is the same bits everywhere. | **helper landed; checklist SWEPT on the greedy score path 2026-08-22** (`4e395b4`: noise multiply-subtract both bodies, `_add_leaf` leafwise intermediates; lossguide selection audited — ZERO float arithmetic in the selection file). STILL OPEN: the symmetric arm's `_add_leaf` calls at compute_scores.mojo:420-460 (`pin_mul_add=False`), cursor update, estimator derivations, leaf rescale, `std_dev_partials_kernel`'s `w*tmp*tmp` chain. Each site converted must cite this row. |
| 10 | division and `sqrt` in scoring | IEEE-correct on normals everywhere measured; the REAL hazard is DENORMAL POLICY | **CONSTRUCTION LANDED 2026-08-21**: `numerics.ftz` -- flush-to-signed-zero under IDENTICAL, comptime no-op under FAST. MEASURED, not designed: the identical model reproduced ALL 53,041 observed Metal divergences bit for bit (`check-ieee-arith`'s ftz-model arm), so on FTZ backends the helper is bitwise inert and on denormal-honoring backends it aligns them to Metal. | **Apple column CLOSED (IEEE+FTZ, no fast-math); helper landed; checklist SWEPT 2026-08-22 across the histogram family and greedy score path** (`a195a6b`: quantizer operand+product, subtracts, scans, dequantize, cancellations, quotients, gains flushed BEFORE the argmax compare, winner stores, variance chain), the ET seams (DEVIATION 453: threshold draw, gain cancellation, leaf dequantization) and RF quantiles (DEVIATION 403). STILL OPEN: part-stats producers (`partitions_reduce`/`pointwise_targets` stores — a denormal `part_stat` re-enters through `part_stat - sum_left`), the symmetric arm's quotient/cancellation seams. Run `check-ieee-arith` first on every new backend column. |
| 11 | k-NN tie handling (`select_radix`) | `atomicAdd` placement, no index tie-break | REFUSE — out of scope for GBDT | documented in `UNWIRED.md` |
| 12 | **transcendentals in DEVICE loss kernels** — `std.math.exp`/`log` in `gbdt/targets/kernel/pointwise_targets.mojo` (Poisson/Quantile/Logloss derivatives, ~15 call sites) | vendor implementation choice: the stdlib may lower these to per-target intrinsics (PTX fast paths on NVIDIA, OCML on AMD, Metal's own on Apple), so `exp(x)`'s last bit can differ per vendor even with every reduction pinned | **AUDIT then REPLACE if divergent** — determine what each backend actually compiles (probe like `check-ieee-arith`: same 2^20 inputs, compare bits per backend); if it is not one polynomial from one source, port a single polynomial evaluated through `identical_mul_add` (row 9) under IDENTICAL | **OPEN — row was MISSING until 2026-08-22.** The host-side libm rule (`optimal_const_for_loss.mojo`, `binarization.mojo`) covers the HOST column only; nothing covered the device column. Found while answering the "is LCD sufficient" question — the enumeration's own incompleteness class (see opening paragraph). THE CLOSURE NOW HAS THREE CONSUMERS: (a) these loss kernels; (b) ET's device algo-L sampler `log`/`exp` (refused on Apple by DEVIATION 199, must close before any NVIDIA↔AMD claim covering that regime); (c) ET's HOST sampler dispatch `n_parallel_samples_for` (`builder_kernels.mojo:312-345`, `ceil(log/log)` via libm — macOS vs glibc last bits flip the draw count and, at the 128/9216 boundaries, WHICH KERNEL runs). Build ONE portable log/exp artifact (polynomial through `identical_mul_add`) and route all three. FOURTH INSTANCE, RESOLVED BY REFUSE (DEVIATION 405): RF's log criteria (Entropy/Poisson/Gamma) raise BY NAME under IDENTICAL — the first wired REFUSE outside k-NN — while Gini/MSE/InverseGaussian (no transcendental) run identical; the portable artifact later upgrades that REFUSE to a run. |

| 13 | **ET range-pass min/max fold** — `node_feature_range_kernel` folded float min/max partials with `max.gpu.primitives.block.min/max` | float min/max are bit-exact selections EXCEPT `-0.0` vs `+0.0`, which compare equal — so which zero survives is decided by fold ORDER, and the sign reaches the model through the `threshold == max -> min` guard | REPLACE — under IDENTICAL the fold runs over `range_key` UInt32 total-order keys (same bits at any width; the cross-block atomics already used these keys since DEVIATION 204) | **closed 2026-08-22 — DEVIATION 452** |
| 14 | **ET rescue-path staging rewrite** — `stage_batch` at the retry call site rewrote host staging with no drain behind it | a MEASURED within-device race: two runs of identical source grew DIFFERENT device trees (node counts 587/459/495/581 vs 605/461/465/597) — an identity failure before any second vendor; DEVIATION 450's every-rewrite-behind-a-drain invariant held inside `search_batch` but not at this outside caller | REPLACE — one `ctx.synchronize()` after the retry-path `stage_batch`, both trainers; rescue-free cycles keep 450's single-drain shape | **closed 2026-08-22 — DEVIATION 455; device == host EXACTLY post-fix; shipped `_mojolearn_trees.so` rebuilt (`d50d0bf`)** |
| 15 | **tie-breaks: leaf selection and split resolution** | the class where atomic arrival order could pick winners | AUDITED, NOT CHANGED — lossguide leaf argmin is strict `<` seeded `+MAX` (tie → lowest leaf index, their `greedy_search_helper.cpp:296-309`); NaN gains lose every compare by IEEE rules on every vendor (new gate P8 pins this against a `max()`/sort rewrite); depthwise cross-block reduce is a sequential host fold under `(gain, feature_id, bin_id)` total order; device in-block argmax has the smaller-bin-feature tie-break, no atomics; smaller-child tie → right child (their `:1318`); ET reduction is a total lexicographic order with DEVIATION 194 guaranteeing the float metric never decides between scored candidates | **verified clean 2026-08-22, both non-symmetric arms + ET** |
| 16 | **RF draw paths end to end** — bootstrap rows (Philox), per-tree seed, feature sampling, ET threshold draws | cuML's own versions ARE machine-dependent (grid from `4*SMCount`, seed truncated 64→32) — their cross-GPU irreproducibility bug | FIXED, not ported: stride frozen at 110592 with launch geometry from `n` only (DEVIATIONS 184/185), seed high half folded when nonzero (DEVIATION 400, bit-identical below 2³²), feature sample = pure hash held to CCCL's compiled oracle, ET thresholds pure `(seed, tree, node, feature)` with exact int→float construction | **closed 2026-08-22 — every forest draw is a pure function of position on every vendor** |
| 17 | **RF in-block best-split reduction** — warp rotate-reduce at `WARP_SIZE` | 64-wide AMD groups differently, and `Split::update`'s range-merge is non-associative in the equal-gain-equal-colid tie class (DEVIATION 105); cross-block mutex merge needs NO pin — audited: one block per (node, sampled column) means merged candidates always differ in `colid`, where `update` is a plain total-order max | PIN — width 32 under IDENTICAL, shuffle through shared memory (DEVIATION 404); reach proven by two sabotages (group-drop inert at width 32, moves 4/4 at 128) | **closed 2026-08-22** |
| 18 | **host-libm knife edges in DISPATCH** — RF `compute_max_features` (`log2` via libm, then `max(1, IdxT(ratio*n_cols))` truncation) and ET `n_parallel_samples_for` (`ceil(log/log)`) | macOS libm and glibc differ in last bits; near an integer boundary the TRUNCATION flips a column count or WHICH KERNEL runs — cross-vendor is cross-HOST here | REPLACE — exact integer arithmetic for the `sqrt`/`log2` string arms; the ET dispatch rides row 12's portable-log artifact | **OPEN (measured safe for n_cols 2..4096 on THIS host only)** |

Eleven closed, five open, one verified-clean, one refused-out-of-scope
(2026-08-22 five-lane fan-out round: rows 3 and 8 re-closed deeper, 13-18
added; the forest checkouts came back clean of row-7-class hazards and float
atomics entirely — see rows 7 and 16). Still open: row 8's last two producer
sites (rides the parked DEVIATION 250 patch), rows 9/10's named symmetric-arm
and part-stats seams, row 12's four-consumer transcendental closure (ET's
algo-L and RF's log-criteria REFUSE both upgrade when it lands), and row 18's
host-libm dispatch knife edges. **That ratio is the honest state of the
guarantee** — and the remaining wall between it and the cross-GPU CLAIM is
E1: no second vendor's hardware has run yet, so everything above the Apple
column is construction plus transcription, certified by gates on ONE device.
The `check_identity_paths` gate (below) and E1 remain the exit criteria.

## The three findings that produced this file

### 1. The toggle was not reachable

`NUMERIC_IDENTICAL` existed, was documented, and had a well-argued design.
**It could not be selected.** Five kernel files each declared their own
`comptime BUILD_MODE = NUMERIC_FAST`, so building the identical arm meant
editing five files and knowing which five. Every statement anyone had made
about "the IDENTICAL mode" was a statement about a configuration that had
never been built.

Fixed: `mojo_only/numerics.GLOBAL_NUMERIC_MODE` is one line, every site reads
it, the default is unchanged bit for bit, and flipping it rebuilds the tree in
the other mode.

### 2. A scheduling row was feeding a float sum

`partition_stats_chunks` is CatBoost's `CeilDivide(2 * SMCount(), statCount)`.
It decides how many chunks a leaf's rows are split into before `block.sum`
reduces each one **in float**. So the machine's core count decides the last
bits of every per-leaf stat, and per-leaf stats **are the leaf values**.

Apple's 10 cores and an A100's 108 SMs give different chunk counts and
therefore different models — not rarely, not probabilistically, but on every
tree. `hardware_matrix.gpu_cores_for` declares itself SCHEDULING and is right
about every other reader it has.

`numerics.mojo` opens by warning about exactly this: *"a block count is a
summation order."* The mistake was one indirection away from that sentence and
survived a matrix, a check and two audits.

Fixed by pinning inside `partition_stats_chunks` itself — the single place the
formula lives — because the launch AND the `stat_partials` buffer sizing both
read it, and pinning only the launch would have sized a buffer from one count
and indexed it with another.

### 3. The scale is derived from a float atomic

Item 8. `choose_scale` snapping to a power of two contains it to
roughly 1e-6 per boosting round; when it fires the whole histogram shifts by a
factor of two. Small probability, large blast radius, and no reason to keep it:
the fix is the accumulator the histogram already uses.

**Update 2026-08-22:** the atomic was replaced the same day this file was
written (per-block partial stores + `deterministic_sum_lanes_kernel`) and the
ledger row lagged the code. The audit that caught the stale row also verified
the partial counts are machine-independent. The residue is narrower than the
row ever said: MAX's `block.sum` folds at hardware warp width inside each
block, which diverges only on AMD's 64-wide wavefront. See the row for the
named fix.

## What has to be true before the claim is made

1. Items 8, 9, 10 and 12 closed.
2. `check_identity_paths` — a test that FAILS if a new float reduction appears
   without a matrix row. This ledger is a document today, which means it rots.
3. The cross-vendor run (E1), which is now the LAST step rather than the first.

Running E1 before 1 would produce a failure that teaches nothing: we already
know what it would find.
