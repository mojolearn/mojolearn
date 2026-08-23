# hierarchy: single-linkage agglomerative clustering, from cuML / cuVS / RAFT

Mirrors `cuml/cpp/src/hierarchy/linkage.cu` (the entry), which forwards to
`cuvs/cpp/src/cluster/detail/{single_linkage,connectivities,mst,agglomerative}.cuh`,
whose MST is RAFT's Boruvka `raft/sparse/solver/{mst,mst_solver}.cuh` +
`detail/{mst_solver_inl,mst_kernels,mst_utils}.cuh`. **COPY, DO NOT
IMPROVE**, with five declared deviations (620, 621, 622; 623 and 624 from
the row-39 audit). File-for-file map in `PORTED_MAP.tsv`; what is not here
and why, parameter by parameter, in `UNPORTED.tsv`. DEVIATION numbers
620-624 spent; 625-629 free.

## Status

**CERTIFIED Apple M4 <-> NVIDIA H100 at leg 11 (commit 144aa5b, judged by `tools/e3_round_judge.sh` section 7 on 2026-08-23): the IDENTICAL card is bit-identical across the two vendors, 8 stages; the FAST cards differ, recorded, the shipped arm makes no cross-vendor claim; AMD MI325X is OWED (that leg was not run).** Rung 1 -- `single_linkage` on dense Float32 with `L2SqrtExpanded`
(cuML's `metric="euclidean"`), `connectivity="pairwise"`, `n_clusters`
given, `children` + `labels` out -- is ported, wired from cuML's entry down
to RAFT's kernels, and gated bit for bit against a host oracle in BOTH
numeric modes on one M4 (2026-08-23). Rung 2 (`connectivity="knn"`, cuML's
PYTHON default) is NOT ported and is refused by name; see UNPORTED. No
timing was measured and none is published.

    linkage_check mode=IDENTICAL ... ALL OK      (13 checks, 5 fixtures + 2 planted)
    linkage_check mode=FAST      ... ALL OK      (distance gates REPORT; vendor-shaped
                                                  assertions RECORDED, see ROW 39 AUDIT)

## Commands

    tools/with_build_lock.sh     pixi run mojo run -I . hierarchy/mojo_only/linkage_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . hierarchy/mojo_only/linkage_check.mojo

    tools/with_build_lock.sh     pixi run mojo run -I . hierarchy/linkage_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/linkage.card \
        tools/with_identical_mode.sh pixi run mojo run -I . hierarchy/linkage_main.mojo
    python3 tools/identity_trace_diff.py /tmp/linkage.apple.card /tmp/linkage.other.card

The pixi tasks are `check-linkage` and `linkage-main` (FAST via `pixi run
check-linkage`; IDENTICAL via `tools/with_identical_mode.sh pixi run
check-linkage`; no `*-identity` task exists by design). Every printed line
carries the mode the binary COMPILED in.

## The identity content: why the MST, the dendrogram and the labels are a pure function of the input bits

Single linkage is four stages. For each, what is a reduction, and why it is
either order-free or pinned:

| stage | their spelling | ours | reduction? | why reproducible |
|---|---|---|---|---|
| distances `z[i,j]` | `cuvs::distance<L2SqrtExpanded>`: `\|\|x_i\|\|^2 + \|\|x_j\|\|^2 - 2 x_i.x_j`, norms precomputed, clamp, sqrt (`connectivities.cuh:157-160`) | FAST: `core/row_norms` + `core/gemm.gemm_nt` (MAX matmul) + `core/expand_distances`. IDENTICAL: `core/row_norms` (pinned fold) + `neighbors/mojo_only/pinned_distance_tile` (DEVIATION 505) | YES, two float folds: the norm and the dot product | IDENTICAL: the norm folds `NORM_TPB` strided partials then a halving tree (no lane primitive); the dot product is one thread walking the feature axis ascending through `identical_mul_add` + `ftz`; the epilogue is one `fma`; sqrt is `identical_sqrt`. Every one of those is a pure function of `d` and `NORM_TPB`. The host oracle replicates that arithmetic and the device matches it BYTE FOR BYTE on 5 fixtures (`check_linkage_distances_match_host_pinned`). `z` is exactly SYMMETRIC because both operands are `X`. FAST: not pinned (a MAX matmul's k-split is the vendor's), reported not asserted |
| self-loop | diagonal <- `FLT_MAX` (`:162-175`) | `self_loop_max_kernel` | no | elementwise |
| MST edge set | RAFT Boruvka with a cuRAND per-vertex ALTERATION of the weights to make them distinct, then per-row min (warp), per-color `atomicMin` on the altered double, "vertices added each other" test, label propagation by `atomicMin` (`mst_solver_inl.cuh`, `mst_kernels.cuh`) | the same kernels, the same launches, with the comparison on the TOTAL ORDER `(weight_order_key(w), min(u,v), max(u,v))` and NO alteration (DEVIATION 620); the key puts `-0.0` (key -1) below `+0.0` (key 0) and every NaN on one key, so no hardware `min`/`max` ever sees a (+0, -0) pair (IDENTITY_PATHS row 39) | YES, but every one is a MIN over a total order or an integer count: per-row min (32-lane halving fold), per-color min (three integer `atomicMin` phases), `next_color` min (`atomicMin`), edge count (`atomicAdd` of ones) | a min over a total order is associative and commutative with no rounding, so the fold shape, the lane count and the atomic landing order cannot move it; an integer add is exact. Their alteration made weights distinct with a RANDOM draw and is the one thing that is NOT a function of the input; replacing it with the undirected-index tie-break makes the per-round picks, the colors, the orientation `(src = the vertex that added, dst = its neighbor)` and therefore the edge set deterministic. Under a total order the MST is UNIQUE, so a serial Kruskal must agree with Boruvka exactly -- and does, on every fixture (`check_linkage_mst_matches_kruskal`). The edge WEIGHTS are copied from `z`, so every weight byte is the pinned distance byte |
| sort by weight | `thrust::sort_by_key` on weight alone, unstable (`sort.h:101`) | host merge sort on the same total order (DEVIATION 621) | a sort = a total order | two distinct MST edges never tie under the triple, so the sorted list is a pure function of the edge set |
| dendrogram | `build_dendrogram_host`: union-find over the sorted list, row `i` = `(find(src_i), find(dst_i))` (`agglomerative.cuh:134-150`) | the same, with a textbook `find` (DEVIATION 622) | no | serial host walk of a fixed list; the orientation of each row is Boruvka's, which is deterministic (above) |
| labels | `extract_flattened_clusters`: levels, the `n_clusters` smallest of the last `2(k-1)` children as roots labelled by descending position, `inherit_labels` walks up (`:238-326`) | the same kernels; the descending sort of `2(k-1)` ints on the host | the root selection is a sort (total order on distinct ints); `inherit_labels` races only on writes of the SAME value | labels are orientation-free (a child maps to its merge row; roots are a set), so a host serial transliteration matches the device BITWISE (`check_linkage_dendrogram_and_labels`) |

So: **under IDENTICAL the distance bytes are pinned; the MST edge set,
its sorted order, the dendrogram and the labels are pure functions of those
bytes; and the whole chain is launch-invariant** -- `check_linkage_launch_
invariance` holds every stage's bytes fixed across tile block 256/64/32,
MST block 256/128/1024, extract block 256/64/512, two input paddings
(37 floats of 1e30, 1024 floats of NaN) and two output paddings/poisons
(0x7fffffff, 0xdeadbeef), on the 203-row hashed fixture and on the
tie fixture; `check_linkage_batch_composition` holds each of the 37x37
cells of a small launch equal to the same cell inside the 203x203 one;
`check_linkage_card_is_stable` holds two cards from two launch shapes
record-identical. Under FAST the MST / dendrogram / labels gates still
hold GIVEN the device's weights, and the distance gates print a REPORT.

One FAST observation worth recording so nobody misreads it: on these
fixtures (d <= 8) the FAST arm's distance bytes happened to equal the
host pinned arithmetic (0 of 41209 cells differ). That is a coincidence of
fold shapes at tiny `d` (the library warp fold and the halving tree pair
the same partials when only 2..8 lanes hold data, and MAX's matmul walks
8 products in order), NOT a pin, and the check says REPORT for that
reason.

## The five deviations

**DEVIATION 620 -- no random alteration; a total order breaks ties**
(`hierarchy/mojo_only/edge_order.mojo`, the block). RAFT draws `v` cuRAND
uniforms and adds `max * (rand[row] + rand[col])` to every weight in
double (`mst_solver_inl.cuh:212-238`, `mst_kernels.cuh:289-307`) so the
per-supervertex min is unique; which of two equal-weight edges wins is
then a function of the RNG, not the input. Ours compares `(weight_order_
key(w), min(u,v), max(u,v))` everywhere RAFT compared the altered weight,
and takes the per-color min in three integer `atomicMin` phases (key, then
`min(u,v)` among key-ties, then `max(u,v)` among those) because Metal has
no 64-bit atomic. The order MUST be on the UNDIRECTED edge: with the
directed CSR index as tie-break, supervertices `{0,3} {1,5} {2,6}` with
only `{0,5} {1,6} {2,3}` tied at the minimum pick a 3-cycle (worked in the
block). MEASURED: `LINK_SAB_RANDOM_ALTERATION` (a hashed per-vertex draw
standing in for cuRAND, applied to ties only, exactly as theirs is) on the
duplicate-point / lattice fixture moves **38 of 47** sorted MST slots and
the children; on the tie-free blobs it moves nothing, which is the point.

**DEVIATION 621 -- the MST sort is total** (`hierarchy/ported/sparse/op/
sort.mojo`). `coo_sort_by_weight` is `thrust::sort_by_key` on the weight
alone, NOT stable, so equal-weight MST edges come out in an implementation-
chosen order, and `build_dendrogram_host` walks that order. Ours sorts on
the same triple. MEASURED: `LINK_SAB_SORT_WEIGHT_ONLY` (weight only, ties
in reverse discovery order -- an order Thrust may return) moves **47 of
47** children rows and the labels on the lattice fixture.

**DEVIATION 622 -- `UnionFind::find` reads `parent[-1]`** (`hierarchy/
ported/cluster/detail/agglomerative.mojo`, the block). Traced line by line
in the block: on every `find` of a root their compression loop reads one
`int` before the vector (UB) and writes `parent[n_indices-1]`; the root it
returns is still right. Ours is the textbook compression. MEASURED:
`check_linkage_union_find_matches_a_naive_one` runs the ported struct and
a compression-free union-find over every fixture's sorted MST and gets
identical roots and sizes row for row (a Mojo `List` would have trapped
on the `-1`, which is how it was found).

**DEVIATION 623 -- a NaN distance is refused by name before any stage**
(`hierarchy/mojo_only/nan_guard.mojo`, the block; called from
`connectivities.mojo::pairwise_distances` after the self-loop transform).
Theirs hands whatever `cuvs::distance` wrote to the MST. A NaN cell arises
from a non-finite input row or from two rows whose squared norms overflow
Float32 (`||x||^2 + ||y||^2 - 2 x.y` is `inf - inf`), and a COMPUTED NaN
carries the vendor's payload (row 39: Apple 0x7fc00000, NVIDIA 0x7fffffff,
AMD 0xffc00000), so it can never sit in `linkage.dists` or
`linkage.mst.weights`. `weight_order_key` already makes the MST's DECISIONS
payload-free; the raw weight `temp_weights` copies out is not. One integer
atomic count of NaN cells, raise with the count. `+inf` is not refused (one
bit pattern everywhere). MEASURED: `check_linkage_nan_distances_refused`,
both modes, three plantings refused with counts 2 / 56 / 14 of 64; with the
guard skipped (`LINK_SAB_SKIP_NAN_GUARD`) the MST completes and 7 of 7
`mst.weights` are NaN with payload 0x7fc00000 on this M4. The blobs_dups
card is byte-identical before and after (diff below).

**DEVIATION 624 -- the host packing of the weight key orders a negative
key as the device does** (`hierarchy/mojo_only/edge_order.mojo`, the block
over `pack_edge_key`). Found by the row-39 fixture: the device compares
keys as SIGNED Int32 (`triple_less`, `Atomic.min`), so `-0.0` (key -1)
sorts before `+0.0` (key 0); the first `pack_edge_key` put the raw key bits
in the top of the UInt64, so the HOST sort and the oracle's Kruskal put
`-0.0` LAST. Boruvka took the `-0.0` edge first; `coo_sort_by_weight` then
put it last. MEASURED before the fix (IDENTICAL, M4): `check_linkage_signed_
zero_mst [IDENTICAL] order A (-0.0 first, +0.0 second): the -0.0 edge did
not win by key; slot 0 (0,2,0x00000000) slot 1 (1,6,0x3f93c680), want slot
0 (0,1,0x80000000) slot 1 (0,2,0x00000000)`. The fix flips the key's sign
bit into the UInt64 (`^ 0x80000000`); for every non-negative key (every
clamped distance, FLT_MAX, +inf, the NaN key) the packed order is
unchanged, so no fixture or card bit moves; only a `-0.0` or negative
weight fed straight into `build_sorted_mst` is ordered differently, and now
the same way on host and device.

## Checks (all green, both modes, one M4)

| check | what it asserts | IDENTICAL | FAST |
|---|---|---|---|
| `check_linkage_distances_match_host_pinned` | per cell vs the host pinned arithmetic; symmetric; FLT_MAX diagonal | 9216/2304/4096/41209/10404 cells bitwise equal | REPORT (0 cells differ on these fixtures, see above) |
| `check_linkage_mst_matches_kruskal` | device Boruvka edge SET and ORDER == host Kruskal on the device's weights, weights bitwise; `m-1` edges | OK x5 (rounds 5/3/5/4/5) | OK x5 (asserted only when the FAST matrix is symmetric; RECORDED otherwise, row 39 audit) |
| `check_linkage_dendrogram_and_labels` | children rows (unordered pairs) == oracle; labels BITWISE == serial extract; partition == cut | OK x5 | OK x5 |
| `check_linkage_entry_matches_stages` | cuML's `single_linkage` entry returns the staged run's bytes (reach) | OK x5 | OK x5 (a mismatch is RECORDED under FAST: two launches of a vendor matmul) |
| `check_linkage_union_find_matches_a_naive_one` | DEVIATION 622 | OK x5 | OK x5 |
| `check_linkage_launch_invariance` | the headline, above | every stage's bytes identical | children/labels/rounds identical; distances and MST reported stable |
| `check_linkage_batch_composition` | 37-row launch cells == the 203-row launch's | 1369/1369 | REPORT (0 differ) |
| `check_linkage_float64_reference` | MST total within 1e-4 of a Float64 direct-form Kruskal | rel 1.3e-8 / 0 / 4.6e-5 / 1.6e-8 / 1.3e-8 | same numbers; a miss is RECORDED under FAST (vendor distances), asserted under IDENTICAL |
| `check_linkage_refusals` | `use_knn=True`, `metric=L1`, `n_clusters > n_rows`, `n_rows > 46340` raise by name | OK | OK |
| `check_linkage_sabotages` | the table below | OK | OK |
| `check_linkage_card_is_stable` | two cards, two launch shapes, 6 records identical | OK | REPORT: identical |
| `check_linkage_signed_zero_mst` | ROW 39: a planted 7-vertex graph into `build_sorted_mst` with `w(0,1)`/`w(0,2)` = `-0.0`/`+0.0` in BOTH lane orders; sorted slot 0 is the `-0.0` edge (bits 0x80000000), slot 1 the `+0.0` edge (0x00000000), written from the order's definition; device == Kruskal bitwise; children == host dendrogram; MST block 256 == 64 | OK x2 | OK x2 (integer keys; no vendor float op, so it asserts in both modes) |
| `check_linkage_nan_distances_refused` | DEVIATION 623: two overflowing rows / every row overflowing / one NaN input cell refused by name with counts 2 / 56 / 14 of 64 BEFORE any stage; guard skipped -> 7 of 7 `mst.weights` NaN, payload RECORDED | OK x3 + RECORDED 0x7fc00000 | OK x3 + RECORDED 0x7fc00000 |

## Sabotage table

| arm | fixture | required | result (IDENTICAL build) |
|---|---|---|---|
| `LINK_SAB_RANDOM_ALTERATION` | dups_lattice | MUST FAIL the MST gate | `FAILED the MST gate as required: 38 of 47 sorted edge slots differ from Kruskal; children differ` |
| `LINK_SAB_RANDOM_ALTERATION` | blobs (tie-free) | report | `MST equal -- a random tie-break only moves ties` |
| `LINK_SAB_ROTATE_CONTRACTION` | hashed203 | MUST FAIL the distance gate | `FAILED the distance gate as required: 4553 of 41209 cells moved` (FAST: reported 4553 too) |
| `LINK_SAB_STD_SQRT` | hashed203 | report | `0 of 41209 cells differ from identical_sqrt's` -- Apple's sqrt is correctly rounded, so this arm CANNOT fail on this device; it is the arm that moves cells on NVIDIA's approximate sqrt (DEVIATION 258). THAT IS THE DOCUMENTED APPLE LIMIT of this check |
| `LINK_SAB_SORT_WEIGHT_ONLY` | dups_lattice | MUST FAIL the dendrogram gate | `FAILED the dendrogram gate as required: 47 of 47 children rows moved; labels MOVED` |
| `LINK_SAB_SKIP_NAN_GUARD` | every row overflowing (planted) | report what the guard prevents | `the MST completed in 2 rounds and 7 of 7 mst.weights are NaN with THIS DEVICE'S payload 0x7fc00000` (both modes) |
| key on `abs(w)` (manual edit of `weight_order_key`, restored) | planted signed-zero graph | order A INERT, order B MUST FAIL | order A: `OK ... slot 0 (0,1,0x80000000)` (the (lo,hi) tie-break agrees with the key); order B: `the -0.0 edge did not win by key; slot 0 (0,1,0x00000000) slot 1 (0,2,0x80000000), want slot 0 (0,2,0x80000000) slot 1 (0,1,0x00000000)` |
| pre-624 `pack_edge_key` (the raw key bits, measured once before the fix) | planted signed-zero graph | MUST FAIL | order A: `slot 0 (0,2,0x00000000) slot 1 (1,6,0x3f93c680), want slot 0 (0,1,0x80000000) ...` |

## ROW 39 AUDIT (2026-08-23): signed zero, NaN payloads, FAST-shaped assertions

IDENTITY_PATHS row 39, measured the same day on Apple M4, NVIDIA H100 and
AMD MI325X: `max(+0.0, -0.0)` is -0.0 on Apple (the second operand) and
+0.0 on NVIDIA/AMD; NaN payloads are 0x7fc00000 / 0x7fffffff / 0xffc00000.
Every site in `hierarchy/` that orders, clamps or records a float was read
against those facts.

**Sites reviewed.**

| site | what | can +-0 / NaN reach it | verdict |
|---|---|---|---|
| `mojo_only/edge_order.mojo::weight_order_key` | the float-to-Int32 total-order key every MST comparison uses | YES by construction: any weight fed to `build_sorted_mst` | `-0.0` -> key -1, `+0.0` -> key 0, DISTINCT, `-0.0` sorts first; every NaN payload -> ONE key (`WEIGHT_KEY_NAN`). The KEY decides, never a hardware min, so the answer is the same on every vendor. Docstring now says so; fixture below |
| `mojo_only/edge_order.mojo::triple_less`, `edge_lo`/`edge_hi` | Int32 lexicographic compare; integer min/max of two vertex ids | integers | vendor-free, kept |
| `mojo_only/edge_order.mojo::pack_edge_key` | the host UInt64 packing of the key for `coo_sort_by_weight` and the oracle's Kruskal | a negative key (`-0.0`, negative weight) | DEFECT, fixed: host put a negative key LAST, device put it FIRST. DEVIATION 624 |
| `ported/sparse/solver/detail/mst_kernels.mojo:164,196,228` `Atomic.min` (key / lo / hi), `:377,380` (colors) | integer atomic mins | integers | vendor-free, kept |
| `ported/sparse/solver/detail/mst_kernels.mojo` per-lane `<` and the 32-lane halving fold | both through `triple_less` on keys | keys only | kept |
| `ported/sparse/op/sort.mojo::coo_sort_by_weight` | host merge sort on `pack_edge_key` | via the key | kept (624 fixes the packing it calls) |
| `ported/cluster/detail/agglomerative.mojo:283` `if c > max_child` | Int32 max over children ids | integers | kept; no float compare decides a merge or a cut (the cut is by COUNT, `n_clusters`, not by a distance threshold; `out_delta` is copied, never compared) |
| `ported/cluster/detail/connectivities.mojo` distance step: the clamp inside `neighbors/mojo_only/pinned_distance_tile.mojo:106` (IDENTICAL) and `core/expand_distances.mojo:50` (FAST), and the oracle's `host_pinned_distance` | `if dist <= 0.0: dist = 0.0` | a cancelling expanded identity can be negative or `-0.0`; `-0.0 <= 0.0` is TRUE | an IEEE compare, not a hardware max: returns `+0.0` for `-0.0` and every negative on every vendor; `sqrt(+0.0)` is `+0.0` through `identical_sqrt` and the stdlib alike. So NO `-0.0` leaves `pairwise_distances`; PROVEN UNREACHABLE for the pairwise path (those two files are other lanes'; read, not edited) |
| `core/row_norms.mojo` (called) | sum of squares seeded `+0.0` | squares are `>= +0.0`, `(-0.0)*(-0.0)` is `+0.0` | never `-0.0` |
| `ported/cluster/detail/connectivities.mojo::self_loop_max_kernel` | diagonal <- FLT_MAX | elementwise store | kept |
| `mojo_only/nan_guard.mojo` (NEW) | integer count of NaN cells, refusal by name | the NaN is the thing counted | DEVIATION 623 |
| `mojo_only/linkage_check.mojo:787` `rel = abs(total32 - total64) / denom` | a Float64 host tolerance on the MST total | host Float64, no `-0.0` question | FACT 3: the Float32 total is a vendor product's under FAST -> RECORDED under FAST, asserted under IDENTICAL |

**What was changed.** (1) `weight_order_key`'s docstring states the signed-
zero and NaN mapping and why it is vendor-free (no bit moved). (2) DEVIATION
624: `pack_edge_key` flips the key's sign bit into the UInt64 so the host
order equals the device's on negative keys; `unpack_edge_wk` inverts it
(no bit of any existing fixture or card moved; measured, the blobs_dups
card diffs IDENTICAL). (3) DEVIATION 623: `pairwise_distances` scans the
matrix after the self-loop transform and refuses any NaN cell by name
(`hierarchy/mojo_only/nan_guard.mojo`). (4) `pairwise_distances` routes
the sabotage-tile copy only for the two tile arms (`ROTATE_CONTRACTION`,
`STD_SQRT`); the tie-break, sort and NaN-guard arms leave the distance
step on its production path in both modes. (5) `LINK_SAB_SKIP_NAN_GUARD`
added.

**NaN audit per recorded stage (`linkage_main.mojo`, 8 stages).**
`linkage.x`: the input bytes (a NaN there is the caller's, same bits
everywhere). `linkage.norms`: `+inf` on overflow (one bit pattern), NaN
only from a NaN input, and in that case `linkage.dists` holds NaN too, so
the DEVIATION 623 refusal fires inside `pairwise_distances` BEFORE EITHER
is recorded (`linkage_main.mojo` records both after the call returns).
`linkage.dists`: guarded by 623. `linkage.mst.rounds`, `linkage.mst.edges`,
`linkage.children`, `linkage.labels`: integers. `linkage.mst.weights`:
copies of `linkage.dists` cells, so guarded by 623. The check's NaN
PADDING poison (`check_linkage_launch_invariance`, 1024 floats of NaN
after the input) never enters a recorded or compared buffer: every copy
back and every record uses the used length (`_copy_f32(..., nnz)`,
`record_list_f32(run.dists)`), verified by reading the calls. The
planted-NaN gate: `check_linkage_nan_distances_refused` (3 refusals with
counts, both modes) and the skip arm showing the 0x7fc00000 payload that
would otherwise sit in `mst.weights`.

**The -0.0 fixture.** `check_linkage_signed_zero_mst`: a 7-vertex dense
symmetric graph fed straight into `build_sorted_mst` (the only entry a
`-0.0` weight can reach; see the clamp row above), hashed distinct positive
weights except `w(0,1)`, `w(0,2)` = `-0.0`/`+0.0` (order A) and
`+0.0`/`-0.0` (order B; vertex 0's CSR row visits 1 before 2, so the lane
sees the two zeros in both orders) and `w(1,2) = +0.0` so a wrong order
changes the edge SET, not only the sort. Asserts: sorted slot 0 is the
`-0.0` edge with bits 0x80000000 and slot 1 the `+0.0` edge with
0x00000000 (expected values written from the order's definition), device
Boruvka == host Kruskal bitwise, children == host dendrogram, MST block
256 == 64. Integer keys throughout, so it ASSERTS IN BOTH MODES. Results
(M4, IDENTICAL and FAST): `order A ... slot 0 (0,1,0x80000000) slot 1
(0,2,0x00000000)`; `order B ... slot 0 (0,2,0x80000000) slot 1
(0,1,0x00000000)`.

**Sabotage, and which is inert.** There is NO hardware `max`/`min` on this
path to swap, so the row-39 "swap the operands" sabotage has no site here;
the equivalent is to make the KEY blind to the sign of zero. Keying on
`abs(w)` (manual edit, restored): order A is INERT -- the `(lo,hi)` tie-
break happens to agree with the key, on every vendor, which is exactly why
one order is not a test -- and order B FAILS: `the -0.0 edge did not win by
key; slot 0 (0,1,0x00000000) slot 1 (0,2,0x80000000), want slot 0
(0,2,0x80000000) slot 1 (0,1,0x00000000)`. Had the per-lane min been a
hardware `min` on the float, Apple would return the SECOND operand (order
A -> `+0.0` wins, order B -> `-0.0` wins) and NVIDIA/AMD the IEEE minimum
(`-0.0` in both orders): the two vendors would disagree on order A. The
key spelling is what removes that disagreement, and the fixture's both-
orders assertion is what would catch a future float-min spelling on any
vendor (on Apple in order A, on NVIDIA/AMD nowhere -- which is why the
abs-key sabotage, not a swap, is the one recorded). The pre-624 packing
was measured once before the fix (the table above).

**FAST demotions (FACT 3).** `check_linkage_float64_reference` (a miss ->
`RECORDED [FAST]`); `check_linkage_mst_matches_kruskal` (an ASYMMETRIC FAST
matrix -> RECORDED: Kruskal reads the upper triangle, Boruvka both
directions, so the comparison is defined only on a symmetric matrix, and a
vendor matmul's (i,j) vs (j,i) bytes are not pinned); `check_linkage_entry_
matches_stages` (two launches of a vendor matmul -> RECORDED on mismatch);
`check_linkage_sabotages` arms `RANDOM_ALTERATION` and `SORT_WEIGHT_ONLY`
(they move TIES, and whether a vendor's FAST matrix holds exact ties is the
vendor's -> RECORDED if inert). Kept asserting under FAST: refusals by name,
the host union-find, the dendrogram/labels given the run's own MST, launch
invariance of children/labels given a stable MST, the signed-zero MST gate
and the NaN refusal (integer arithmetic / refusal by name). No IDENTICAL
assertion was weakened. On this M4 every demoted line still prints OK in
both modes (the FAST matrix is symmetric and tie-bearing here).

**Check results after the audit (M4, 2026-08-23).** IDENTICAL: 13 checks,
`linkage_check mode=IDENTICAL ALL OK`. FAST: `linkage_check mode=FAST ALL
OK`, distance gates REPORT (0 cells differ, asymmetric 0), the two new
gates OK in both. Card driver: `linkage_main mode=IDENTICAL ... boruvka_
rounds=5 n_connected_components=1`, card identical to the pre-audit card.

## What is ported, refused, not ported

Ported and gated: cuML `single_linkage` (PAIRWISE), cuVS
`build_dist_linkage`, `single_linkage`, the PAIRWISE `distance_graph_impl`,
`build_sorted_mst` (loop shape), `UnionFind`, `build_dendrogram_host`,
`extract_flattened_clusters` and its two kernels, RAFT `mst`, `MST_solver`
and every kernel in `mst_kernels.cuh` except `alteration_kernel`.

Refused by name: `use_knn=True` / `connectivity="knn"` (rung 2), metrics
other than L2SqrtExpanded / L2Expanded, `n_clusters > n_rows`, `n_clusters
< 1`, `n_rows < 2`, `n_rows > 46340`.

Not ported: the KNN_GRAPH connectivity, `connect_knn_graph` /
`cross_component_nn` / `merge_msts`, `build_mr_linkage` (HDBSCAN's),
`alteration` (620). Ported but UNREACHED: `add_reverse_edge`
(`symmetrize_output=true`, which single linkage never passes) -- a row for
`UNWIRED.md`, handed off below.

Honesty about the Python default: cuML's `AgglomerativeClustering` defaults
`connectivity="knn"` (`agglomerative.pyx:123`); its C++ entry and
scikit-learn default to the full pairwise graph. Rung 1 is the latter.

## The card

`hierarchy/linkage_main.mojo` writes eight stages on the blobs_dups fixture:
`linkage.x`, `linkage.norms`, `linkage.dists`, `linkage.mst.rounds` (an
INTEGER stage read first: a card that differs there disagreed about how
much work to do), `linkage.mst.edges`, `linkage.mst.weights`,
`linkage.children`, `linkage.labels`. Apple IDENTICAL card, 2026-08-23:

    0  linkage.x            f32  510    51b79ed6ed4a230a
    1  linkage.norms        f32  102    f24adbf8590470a5
    2  linkage.dists        f32  10404  3925db759e392721
    3  linkage.mst.rounds   i32  1      2d401a55eec16520
    4  linkage.mst.edges    i32  202    f4b887b7110d85a0
    5  linkage.mst.weights  f32  101    daf76b3d0cef46ee
    6  linkage.children     i32  202    0fb95fdc640e4824
    7  linkage.labels       i32  102    1ff4250dd4aab576

Re-emitted after the row-39 audit (DEVIATIONS 623, 624 in the path):
`tools/identity_trace_diff.py` reports `RESULT: IDENTICAL. Same stage
sequence, same counts, same hashes.` against the card above.

## ROW TEXT FOR THE IDENTITY LANE

| 43 | **single-linkage agglomerative clustering, END TO END** -- `hierarchy/ported/hierarchy/linkage.mojo` (cuML `linkage.cu`) -> cuVS `single_linkage.cuh`/`connectivities.cuh`/`mst.cuh`/`agglomerative.cuh` -> RAFT Boruvka `mst_solver_inl.cuh`/`mst_kernels.cuh`: pairwise L2 distances, Boruvka MST, sort, host union-find dendrogram, device label extraction | the distance tile is row 24's (DEVIATION 505) and the row norm row 19's, unchanged; NEW and vendor-dependent in THEIR spelling: (a) RAFT breaks MST weight ties with a cuRAND per-vertex ALTERATION (`mst_solver_inl.cuh:212-238`), so on duplicate points or equal distances the edge SET is a function of the RNG stream, not the input; (b) `coo_sort_by_weight` is an unstable `thrust::sort_by_key` on weight alone (`sort.h:101`), so equal-weight MST edges reach the dendrogram in an implementation-chosen order; (c) the per-color min is `atomicMin` on a double (Metal has no 64-bit atomic) | (a) DEVIATION 620: the total order `(weight_order_key(w), min(u,v), max(u,v))` everywhere RAFT compared the altered weight, per-color min in three integer `atomicMin` phases; (b) DEVIATION 621: a host merge sort on the same triple; (c) folded into 620. A min over a total order and an integer count are the only reductions in the MST and both are order-free, so the edge set, the dendrogram and the labels are pure functions of the distance bytes; gated: edge set == serial Kruskal (unique MST under a total order), labels BITWISE == serial extract, launch invariance across tile/MST/extract block sizes and two paddings/poisons, 37x37 cells == the same cells of 203x203, two cards identical; sabotages 620 (38/47 slots moved) and 621 (47/47 rows moved) fail as required, rotate-contraction moves 4553/41209 cells, std-sqrt moves 0 (Apple limit); row-39 audit 2026-08-23: `-0.0` < `+0.0` by integer key (no hardware min/max on the path; planted both lane orders, key-on-abs sabotage fails order B), a NaN distance refused by name before any stage (DEVIATION 623), host key packing made to agree with the device on negative keys (DEVIATION 624) | **construction + Apple gates; no second vendor has run it.** Open: rung 2's `connect_knn_graph` host overload picks a RANDOM vertex per component (`std::mt19937(std::random_device())`, `mst.cuh:167-190`) and must be pinned when ported |

## HAND-OFF TO THE IDENTITY LANE

1. `IDENTITY_PATHS.md`: row 43 carries the row text above; when this
   README's row text changes, the ledger's row 43 is to be refreshed from
   it (the row-39 audit sentence is the current delta).
2. `UNWIRED.md`, NOT wired table: `hierarchy/ported/sparse/solver/detail/
   mst_kernels.mojo::add_reverse_edge` | nothing (single linkage passes
   `symmetrize_output=false`, `cuvs mst.cuh:297-298`) | a caller wanting a
   symmetric MST edge list; transliterated and gated by nothing.
3. `pixi.toml`: `check-linkage` and `linkage-main` exist (FAST via
   `pixi run check-linkage`; IDENTICAL via `tools/with_identical_mode.sh
   pixi run check-linkage`). Nothing further asked.
4. `neighbors/mojo_only/pinned_distance_tile.mojo` is CALLED from
   `hierarchy/ported/cluster/detail/connectivities.mojo` (IDENTICAL arm)
   with `X` as both operands through a sub-buffer view; no change to that
   file. `hierarchy/mojo_only/sabotage_tile.mojo` is a COPY of its kernel
   with two sabotage arms, reached only by the check -- the duplication is
   recorded here so a change to the tile's arithmetic is mirrored (or the
   sabotage arms are moved into the tile behind a `sabotage` argument,
   which is the neighbors lane's call).
5. `hierarchy/ported/sparse/op/sort.mojo::merge_sort_u64_with_index` and
   the oracle's `merge_sort_f64_with_index` are host merge sorts written
   here because no host sort primitive with a stated stability was found in
   the tree; if `core/` grows one, these two should call it.
6. The Python surface (rung 1's estimator), for whoever owns `python/` and
   `bindings/`: scikit-learn's `AgglomerativeClustering(n_clusters=2,
   metric="euclidean", linkage="single", connectivity=None,
   compute_full_tree="auto", distance_threshold=None, compute_distances=
   False)` is the target (NOT Array-API supported, `ROADMAP.md`'s leak
   table). Map: `n_clusters` -> `n_clusters`; `metric="euclidean"` ->
   L2SqrtExpanded (1) and refuse every other value by name; `linkage` must
   be `"single"` (refuse others by name, as cuML does at
   `agglomerative.pyx:157`); `connectivity` must be `None` (the dense
   graph; cuML's `"knn"` is rung 2, refuse by name); `distance_threshold`
   is not ported (would need `out_delta`, which `build_dendrogram_host`
   already produces -- `n_clusters` = 1 + the number of merges whose
   delta > threshold); `compute_distances` likewise from `out_delta`.
   Outputs: `labels_` (partition equal to sklearn's `_hc_cut`; the NUMBERING
   is cuML's -- descending root index -- not sklearn's, say so), `children_`
   (sklearn's rows are `(find(row), find(col))` from scipy's MST, ours are
   Boruvka-oriented; same unordered pairs when the MST is tie-free and the
   sort agrees; under ties sklearn's mergesort on `mst.data` keeps scipy's
   coo order, which is NOT our `(min,max)` order -- document, do not claim
   row equality), `n_leaves_ = n_rows`, `n_connected_components_ = 1`.
   Entry: `hierarchy/ported/hierarchy/linkage.mojo::single_linkage(ctx, x,
   n_rows, n_cols, n_clusters, metric, children, labels)` with `children`
   an Int32 device buffer of `(n_rows-1)*2` and `labels` of `n_rows`.

## What is left

- Rung 2: the KNN_GRAPH connectivity + `connect_knn_graph` (both
  overloads; pin the host overload's random vertex choice) + `merge_msts`
  + `cross_component_nn`. Everything it needs below it (the MST with
  `initialize_colors=false`, the loop shape) is here.
- AMD MI325X (NVIDIA H100 is closed at leg 11); the NVIDIA risk
  is the one the std-sqrt sabotage names (approximate sqrt at the distance
  seam, routed through `identical_sqrt`, unverified there), and the
  `atomicMin` phases are integer and should be inert.
- `distance_threshold` / `compute_distances` on the Python surface (cheap:
  `out_delta` exists).
- A caller that feeds `build_sorted_mst` a precomputed graph (rung 2's
  k-NN graph) owns its own NaN guard: DEVIATION 623 scans only the
  pairwise matrix, and a NaN weight fed in directly reaches `mst.weights`
  with its payload (the `LINK_SAB_SKIP_NAN_GUARD` line shows exactly that).
