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
| 8 | **fixed-point scale magnitude** (and `functionValue`, same reduce) | ~~device float reduce + float atomic~~ **the atomic half CLOSED 2026-08-21** (the "DETERMINISM FIX" block, `pointwise_targets.mojo:607`): every block STORES its partial, `deterministic_sum_lanes_kernel` folds them in one fixed 256-thread shape, and the partial COUNTS are pure `f(n_rows)` — `mse_blocks`, `multilogit_blocks`, `bootstrap_grid_blocks` all verified machine-independent 2026-08-22, no row-7-class hazard. What REMAINS: the per-block partial itself is MAX's `block.sum`, whose internal cross-lane fold follows the hardware warp width — 32 on Apple and NVIDIA, **64 on AMD** — so the partial's bits differ on the AMD column only | REPLACE the remainder — `pinned_block_sum` (pointwise_targets.mojo, DEVIATION 251): FAST arm IS the library call, IDENTICAL arm the `deterministic_sum_lanes_kernel` shared-memory shape | **DEVICE column CLOSED; AMD residue CLOSED at 14 of 16 producer sites 2026-08-22** — pointwise + cross-entropy + multilogit (12 sites, DEVIATION 251), `std_dev_partials_kernel` (251's 13th), `compute_target_variance_kernel` (`4e395b4`); bootstrap was already a fixed shared fold. REMAINING two: `gbdt/methods/kernel/exact_estimation.mojo:228` (`block_sum[1024]`) and `gbdt/gpu_util/partitions_reduce.mojo:156,209` (leaf-values path — resolve together with the parked DEVIATION 250 patch, which rewrites that file). Same fix: import `pinned_block_sum`. **E1 VERIFIED 2026-08-23**: after `c077a22` pinned those last two producer folds, the 20-tree RMSE fit's full 302-stage card is bit-identical Apple(M4/Metal)↔AMD(MI325X/HIP) at `39a0d88`. |
| 9 | **FMA contraction** | codegen decision | **CONSTRUCTION LANDED 2026-08-21**: `numerics.identical_mul_add` -- explicit `fma` under IDENTICAL (one rounding, identical on Metal/PTX/AMDGPU), the naive chain under FAST. **APPLE'S FAST BASELINE IS FUSED, CORRECTED 2026-08-23 — the sentence that used to stand here was an artifact of the check that produced it.** It read "Apple's FAST baseline measured UNFUSED (`check-ieee-arith`: fused 0 / unfused 1,046,394 of 2^20), so IDENTICAL differs from FAST on Apple BY DESIGN". That verdict came off 2^20 HASHED patterns of which **ZERO separate a fused `a*b+c` from an unfused one** — random exponents put the product and the addend so far apart that both spellings round the same way — and the tie arm was written `if got == unfused` FIRST, so every one of those 1,046,394 ties was counted as evidence of UNFUSED. A backend that contracts everything scored exactly the same. `check-ieee-arith` now carries a BUILT-TO-SEPARATE arm (`a`, `b` with half-width mantissas and `c = -fl(a*b)`, so unfused is exactly +0.0 and fused is the rounding error): Metal through MAX reports **FUSED on 1,629 of 1,629** separating patterns, and an isolated two-kernel probe agrees (naive `a*a+c` returns the fma bits). INDEPENDENTLY CORROBORATED by the GBDT lane the same day: `f8044fe`'s message records that routing `exp`/`log` through a plain wrapper MOVED FAST BITS because "the stdlib exp/log lower to Mojo polynomials whose mul-add chains contract PER CONTEXT" — i.e. that lane was already watching Apple FAST contract while this row said it did not. CONSEQUENCES: (a) `identical_mul_add` is BIT-INERT on Apple, exactly as `ftz` is on an FTZ backend, so IDENTICAL does NOT differ from FAST on Apple at these seams and the pins this lane swept moved no shipped bits — verified, the three unsupervised suites print byte-identical output at HEAD and after; (b) the pin's value is the backend that does NOT contract by default, where an unpinned `acc += x * y` rounds twice against Metal's once; (c) any reasoning anywhere that rests on "Apple is unfused" is unsound and must be re-derived. | **helper landed; checklist SWEPT on the greedy score path 2026-08-22** (`4e395b4`: noise multiply-subtract both bodies, `_add_leaf` leafwise intermediates; lossguide selection audited — ZERO float arithmetic in the selection file). STILL OPEN: the symmetric arm's `_add_leaf` calls at compute_scores.mojo:420-460 (`pin_mul_add=False`), cursor update, estimator derivations, leaf rescale, `std_dev_partials_kernel`'s `w*tmp*tmp` chain. Each site converted must cite this row. **E1 EVIDENCE 2026-08-23**: the cursor update CLOSED (`39a0d88` `identical_mul_add`; full-card Apple↔AMD agreement through all 302 RMSE stages) — and the remaining open sites are no longer hypothetical: NVIDIA H100 at the same commit FIRST DIVERGES at `tree001.winners.scores` (RMSE) / `tree000.winners.scores` (Logloss) while Apple↔AMD agree bit for bit, the measured proof that TWO backends agreeing closes nothing. The RMSE prediction hash still matched all three vendors because the argmax survived last-bit score wiggle — an output identity WITHOUT a certificate, which is exactly the difference the cards exist to catch. DEVIATION 253 (in flight) pins the compute_scores sites. |
| 10 | division and `sqrt` in scoring | IEEE-correct on normals everywhere measured; the REAL hazard is DENORMAL POLICY | **CONSTRUCTION LANDED 2026-08-21**: `numerics.ftz` -- flush-to-signed-zero under IDENTICAL, comptime no-op under FAST. MEASURED, not designed: the identical model reproduced ALL 53,041 observed Metal divergences bit for bit (`check-ieee-arith`'s ftz-model arm), so on FTZ backends the helper is bitwise inert and on denormal-honoring backends it aligns them to Metal. | **Apple column CLOSED (IEEE+FTZ, no fast-math); helper landed; checklist SWEPT 2026-08-22 across the histogram family and greedy score path** (`a195a6b`: quantizer operand+product, subtracts, scans, dequantize, cancellations, quotients, gains flushed BEFORE the argmax compare, winner stores, variance chain), the ET seams (DEVIATION 453: threshold draw, gain cancellation, leaf dequantization) and RF quantiles (DEVIATION 403). STILL OPEN: part-stats producers (`partitions_reduce`/`pointwise_targets` stores — a denormal `part_stat` re-enters through `part_stat - sum_left`), the symmetric arm's quotient/cancellation seams. Run `check-ieee-arith` first on every new backend column. |
| 11 | **k-NN tie handling** — `select_radix`'s output placement AND the fused arm's queue | RAFT places every output with `atomicAdd` and has no index tie-break, so WHICH of several equidistant neighbours is returned, and WHERE it lands, are both arrival order; the fused arm's FAISS comparator (`Comparators.cuh:17`) compares the DISTANCE ONLY and its cross-block merge is mutex-ordered | **REPLACE + PIN, and the REFUSE was retired 2026-08-23.** TILED: `neighbors/mojo_only/select_radix_identical.mojo` runs the radix passes over a 64-bit COMPOSITE `(twiddle_in(distance) << 32) | index` — a total order in which the tie class does not exist (DEVIATION 500) — and rewrites every output slot from a rank rather than an atomic arrival (DEVIATION 501). FUSED: `grid_x` pinned to 1 so no merge decides a tie, which makes its (unnamed) tie set a pure function of `(m, n, k)` and the pinned policy (DEVIATION 502); the ARM itself is pinned to cuVS's own dispatch in the same deviation, because two arms with two tie rules chosen BY SHAPE make the answer depend on the caller's query count | **closed 2026-08-23** — `neighbors/mojo_only/knn_identity_check.mojo`: four equidistant candidates and two slots return indices 5 and 100 (the two lowest) under IDENTICAL; the same fixture under FAST has been observed returning a DIFFERENT neighbour for two identical queries in one process, which is the hazard, measured |
| 12 | **transcendentals in DEVICE loss kernels** — `std.math.exp`/`log` in `gbdt/targets/kernel/pointwise_targets.mojo` (Poisson/Quantile/Logloss derivatives, ~15 call sites) | vendor implementation choice: the stdlib may lower these to per-target intrinsics (PTX fast paths on NVIDIA, OCML on AMD, Metal's own on Apple), so `exp(x)`'s last bit can differ per vendor even with every reduction pinned | **AUDIT then REPLACE if divergent** — determine what each backend actually compiles (probe like `check-ieee-arith`: same 2^20 inputs, compare bits per backend); if it is not one polynomial from one source, port a single polynomial evaluated through `identical_mul_add` (row 9) under IDENTICAL | **OPEN — row was MISSING until 2026-08-22.** The host-side libm rule (`optimal_const_for_loss.mojo`, `binarization.mojo`) covers the HOST column only; nothing covered the device column. Found while answering the "is LCD sufficient" question — the enumeration's own incompleteness class (see opening paragraph). THE CLOSURE NOW HAS THREE CONSUMERS: (a) these loss kernels; (b) ET's device algo-L sampler `log`/`exp` (refused on Apple by DEVIATION 199, must close before any NVIDIA↔AMD claim covering that regime); (c) ET's HOST sampler dispatch `n_parallel_samples_for` (`builder_kernels.mojo:312-345`, `ceil(log/log)` via libm — macOS vs glibc last bits flip the draw count and, at the 128/9216 boundaries, WHICH KERNEL runs). Build ONE portable log/exp artifact (polynomial through `identical_mul_add`) and route all three. FOURTH INSTANCE, RESOLVED BY REFUSE (DEVIATION 405): RF's log criteria (Entropy/Poisson/Gamma) raise BY NAME under IDENTICAL — the first wired REFUSE outside k-NN — while Gini/MSE/InverseGaussian (no transcendental) run identical; the portable artifact later upgrades that REFUSE to a run. **CORE LANDED 2026-08-23 (`ed0fe5d`)**: `portable_expf`/`portable_logf` in `mojo_only/numerics.mojo` — the Cephes single-precision tables through `identical_mul_add`'s fma with row 10's flush policy baked in unconditionally — plus mode-gated `identical_exp`/`identical_log`. Gate: `check-portable-translog` (measured on Apple: max 1 ulp vs float64 stdlib over 2^20 hashed inputs per function, device bits == host bits on every lane, sabotage arm fails the compare on 1,084,122 lanes). Its printed device hash — `8705486125800438413` on Apple — is the cross-vendor certificate line: run it FIRST on every new column. Consumers now being routed: DEVIATION 406 (RF criteria REFUSE→run), 456/457 (ET device sampler + host dispatch, absorbing row 18's ET half), gbdt loss kernels + multilogit next; cross-vendor verification of the routed consumers PENDING. |

| 13 | **ET range-pass min/max fold** — `node_feature_range_kernel` folded float min/max partials with `max.gpu.primitives.block.min/max` | float min/max are bit-exact selections EXCEPT `-0.0` vs `+0.0`, which compare equal — so which zero survives is decided by fold ORDER, and the sign reaches the model through the `threshold == max -> min` guard | REPLACE — under IDENTICAL the fold runs over `range_key` UInt32 total-order keys (same bits at any width; the cross-block atomics already used these keys since DEVIATION 204) | **closed 2026-08-22 — DEVIATION 452** |
| 14 | **ET rescue-path staging rewrite** — `stage_batch` at the retry call site rewrote host staging with no drain behind it | a MEASURED within-device race: two runs of identical source grew DIFFERENT device trees (node counts 587/459/495/581 vs 605/461/465/597) — an identity failure before any second vendor; DEVIATION 450's every-rewrite-behind-a-drain invariant held inside `search_batch` but not at this outside caller | REPLACE — one `ctx.synchronize()` after the retry-path `stage_batch`, both trainers; rescue-free cycles keep 450's single-drain shape | **closed 2026-08-22 — DEVIATION 455; device == host EXACTLY post-fix; shipped `_mojolearn_trees.so` rebuilt (`d50d0bf`)** |
| 15 | **tie-breaks: leaf selection and split resolution** | the class where atomic arrival order could pick winners | AUDITED, NOT CHANGED — lossguide leaf argmin is strict `<` seeded `+MAX` (tie → lowest leaf index, their `greedy_search_helper.cpp:296-309`); NaN gains lose every compare by IEEE rules on every vendor (new gate P8 pins this against a `max()`/sort rewrite); depthwise cross-block reduce is a sequential host fold under `(gain, feature_id, bin_id)` total order; device in-block argmax has the smaller-bin-feature tie-break, no atomics; smaller-child tie → right child (their `:1318`); ET reduction is a total lexicographic order with DEVIATION 194 guaranteeing the float metric never decides between scored candidates | **verified clean 2026-08-22, both non-symmetric arms + ET** |
| 16 | **RF draw paths end to end** — bootstrap rows (Philox), per-tree seed, feature sampling, ET threshold draws | cuML's own versions ARE machine-dependent (grid from `4*SMCount`, seed truncated 64→32) — their cross-GPU irreproducibility bug | FIXED, not ported: stride frozen at 110592 with launch geometry from `n` only (DEVIATIONS 184/185), seed high half folded when nonzero (DEVIATION 400, bit-identical below 2³²), feature sample = pure hash held to CCCL's compiled oracle, ET thresholds pure `(seed, tree, node, feature)` with exact int→float construction | **closed 2026-08-22 — every forest draw is a pure function of position on every vendor** |
| 17 | **RF in-block best-split reduction** — warp rotate-reduce at `WARP_SIZE` | 64-wide AMD groups differently, and `Split::update`'s range-merge is non-associative in the equal-gain-equal-colid tie class (DEVIATION 105); cross-block mutex merge needs NO pin — audited: one block per (node, sampled column) means merged candidates always differ in `colid`, where `update` is a plain total-order max | PIN — width 32 under IDENTICAL, shuffle through shared memory (DEVIATION 404); reach proven by two sabotages (group-drop inert at width 32, moves 4/4 at 128) | **closed 2026-08-22** |
| 18 | **host-libm knife edges in DISPATCH** — RF `compute_max_features` (`log2` via libm, then `max(1, IdxT(ratio*n_cols))` truncation) and ET `n_parallel_samples_for` (`ceil(log/log)`) | macOS libm and glibc differ in last bits; near an integer boundary the TRUNCATION flips a column count or WHICH KERNEL runs — cross-vendor is cross-HOST here | REPLACE — exact integer arithmetic for the `sqrt`/`log2` string arms; the ET dispatch rides row 12's portable-log artifact | **OPEN (measured safe for n_cols 2..4096 on THIS host only)** |

### The unsupervised sections (2026-08-23)

Rows 1-18 were written by and for the GBDT and forest lanes, and nothing in
them covers `cluster/`, `neighbors/` or `dbscan/` — which between them had
ZERO identity constructions before this round: not one `ftz`, not one
`identical_mul_add`, not one pinned fold. The enumeration's own incompleteness
class again (see the opening paragraph), found by asking whether the identical
column covers the algorithms this repository actually wins on.

| # | pathway | order-dependent? | what `IDENTICAL` does | status |
|---|---|---|---|---|
| 19 | **the L2 distance accumulators** — `cluster/.../simt_kernel` (k-means assignment), `neighbors/.../fused_l2_knn`, `dbscan/.../epsilon_neighborhood`, `neighbors/.../ball_cover/common`, `core/expand_distances`, `core/row_norms` | `acc += x * y` (or `+= diff * diff`) is one rounding or two AT THE CODEGEN'S WHIM, and the per-cell k order is fixed but the CONTRACTION is not. In DBSCAN the consequence is discrete, not a drift: one ULP moves a point across `<= eps` and an adjacency BIT flips | PIN — every accumulate and every expanded-L2 epilogue through `identical_mul_add` / `identical_mul_add_simd`, every seam through `ftz` / `ftz_simd` (DEVIATIONS 503, 506) | **closed 2026-08-23.** Reach is not provable by BITS on Apple, because Apple's FAST codegen contracts too (row 9's correction), so `check_fused_contraction_pin` compares against a DEVICE oracle carrying both spellings and reports which arm the backend took rather than pretending to a difference it cannot see. On a non-contracting backend the same check becomes a bit-level reach proof with no edit. |
| 20 | **the unsupervised float folds** — `core/row_norms`, `cluster/mojo_only/reduce_by_key` (inertia AND the centroid SHIFT), `cluster/mojo_only/plus_plus` (candidate cost + the three-stage scan) | `max.gpu.primitives.block.sum` folds across lanes at the HARDWARE width: 32 on Apple and NVIDIA, **64 on AMD's wavefront**. Row 8's residue, in a section that had no equivalent of row 8's fix | REPLACE — `core/pinned_reduce.pinned_block_sum`, a halving tree with no lane primitive in it, under IDENTICAL; the library call unchanged under FAST (DEVIATION 504) | **closed 2026-08-23** — `check_pinned_fold_shape` requires the device to equal a HOST halving tree bit for bit and first proves the fixture separates that shape from a sequential one (584535.1 vs 584535.25). The GBDT lane's `pinned_block_sum` is a same-shape twin in a file this lane may not edit; the merge is named in `core/pinned_reduce.mojo`, not hidden. |
| 21 | **`lib_block_size_for`, the library block-size row** | the matrix labels it SCHEDULING. For `K_LIB_ROW_NORM`, `K_LIB_REDUCE_BY_KEY` and `K_LIB_PLUS_PLUS` the block size is the WIDTH OF A FLOAT FOLD, and for the third it also sets the scan's chunk count, which decides WHICH SAMPLE k-means++ draws. Row 7's class, one table over | PIN at the comptime accessor — `lib_block_bounds_a_float_fold` names the three and the accessor resolves them to `COLUMN_BIT_IDENTICAL` under IDENTICAL (DEVIATION 508) | **closed 2026-08-23, and BIT-INERT TODAY: every column carries 128 for all three.** The gate is for the next measurement — the table's own docstring invites a vendor number to land there "WITHOUT touching a kernel", and on these three rows that would have split the vendors silently. `check_float_fold_rows_are_pinned` also asserts the three INTEGER rows stay free. |
| 22 | **k-means assignment geometry** — grid shape, `veclen`, policy | a different core count reaches the kernel as a different grid; `fused_veclen_for` reads POINTER ALIGNMENT, which is an allocator's business | AUDITED, NOT CHANGED — the argmin carries `raft::argmin_op`'s `(value, key)` total order in BOTH arms (ours in the fused one, where THEIRS compares the value only), a row's owner lane is `row mod Mblk` at every `grid_y`, and the k order is ascending at every `veclen` | **verified clean 2026-08-23** — `check_assignment_geometry_invariance`: 512 assignments bit-identical at `grid_y` 1/2/5/8 with 16 of them decided by the tie-break, on a fixture with duplicated centroids planted to force ties. Holds in BOTH modes, which is the point: this one was already right. |
| 23 | **the fused k-NN queue** — `faiss_select::WarpSelect` and its cross-block merge | the comparator is the DISTANCE ONLY, so a tie is resolved by the queue's feed order and, at `grid_x > 1`, by which block won the mutex; the network is 32 lanes wide by construction (`WARP_LANES`, `lid = tid % 32`, `NumWarpQ / 32`) | PIN `grid_x = 1` and pin the ARM (DEVIATION 502) so the tie set is a pure function of `(m, n, k)` and the policy; REFUSE the arm outright where the lane width is not 32, because there the network addresses the wrong half of the wavefront and is not merely non-identical but WRONG | **closed on the 32-lane columns 2026-08-23; REFUSED on a 64-lane one.** The refusal raises at the entry rather than compiling a silently different answer; the closure is a width-parameterized network, unported. |
| 24 | **the tiled k-NN arm's distance step** | `core/gemm.gemm_nt` is `linalg.matmul`, a CLOSED vendor library: its tile shape and k-split are per-vendor and a k-split IS a summation order. cuVS is worse off here — their distance GEMM defaults to `CUBLAS_COMPUTE_32F_FAST_TF32`, ten mantissa bits — but inheriting their design does not oblige us to inherit that | REPLACE — `neighbors/mojo_only/pinned_distance_tile.mojo` computes product and epilogue in one kernel, one thread per cell, feature axis ascending, so the order is a pure function of `k` (DEVIATION 505) | **closed 2026-08-23, and it is the one place in this lane where identity buys a slower kernel rather than a different rounding: 2.85x on the priced arm** (`tools/price_unsupervised_identity.sh`). Deliberately the simplest correct shape: a staged or split version would be a second thing to pin. |
| 25 | **DBSCAN's propagation caps** — `weak_cc_batched` and `merge_labels`, `max_iterations` default 200 | `atomicMin` propagation reaches a UNIQUE fixed point whatever order the updates land in — that is why DBSCAN needs no pin downstream of the distance — but ONLY at the fixed point. A run cut off at the cap returns a SNAPSHOT of the atomic order on that machine, and upstream returns it silently | REFUSE — raise under IDENTICAL when the loop exits without `changed == 0` (DEVIATION 507); FAST keeps upstream's truncation | **closed 2026-08-23** — `check_dbscan_refuses_truncated_propagation` shows the cost of the refusal as well as the refusal: at `max_iterations = 1` the FAST arm returns a labelling that differs from the converged one and says nothing. |
| 26 | **DBSCAN's batch count** — `max_mbytes_per_batch = 0` derives it from the DEVICE'S FREE MEMORY | a decomposition of the graph, chosen by a hardware number, with `merge_labels` folding the pieces | AUDITED, NOT CHANGED — each batch's `weak_cc` canonicalizes to a component minimum before the merge, so the fold is over the same lattice whatever the decomposition | **verified clean 2026-08-23 on a BORDER-HEAVY fixture** — `check_dbscan_batch_count_invariance`: 1,020 labels bit-identical at 2/12/6/2 batches, and the check REFUSES to pass unless the fixture actually produced border points (6) and noise (14). Every pre-existing DBSCAN check used separated blobs, where the case that could move does not exist. |

Also enumerated and NOT closed, because it belongs to another lane and this
one may not quietly take it: **`core/column_stats.mojo`'s float `block.sum`
pair** (`K_LIB_COLUMN_STATS`), which `decomposition/` reaches through PCA and
OLS. It is the same defect as row 20 in the same directory, one import away
from the same fix, and it is left named rather than done.

Rows 1-18 (the GBDT and forest lanes): eleven closed, five open, one
verified-clean. Rows 19-26 (the unsupervised sections, 2026-08-23): six
closed, two verified-clean, one refusal scoped to a 64-lane column and one
row NAMED FOR ANOTHER LANE (`column_stats`). **Row 11 is no longer a
refusal**, and row 9's Apple sentence was wrong and is corrected above.

The pre-existing summary, kept because its reasoning still holds for the
rows it describes: eleven closed, five open, one verified-clean, one
refused-out-of-scope
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

**That last sentence is now STALE and the E1 lane owns the rewrite:** row 9
already records a full-card Apple<->AMD agreement and an NVIDIA H100
divergence at `tree001.winners.scores`, so a second and a third vendor HAVE
run. Left standing rather than rewritten because the paragraph is the GBDT
lanes' summary of their own rows; flagged here because a reader who stops at
this paragraph would take away the opposite of what row 9 measured. The
unsupervised rows 19-26 have NOT been to a second vendor and say so.

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
