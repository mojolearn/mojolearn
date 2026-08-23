# hierarchy: single-linkage agglomerative clustering, from cuML / cuVS / RAFT

Mirrors `cuml/cpp/src/hierarchy/linkage.cu` (the entry), which forwards to
`cuvs/cpp/src/cluster/detail/{single_linkage,connectivities,mst,agglomerative}.cuh`,
whose MST is RAFT's Boruvka `raft/sparse/solver/{mst,mst_solver}.cuh` +
`detail/{mst_solver_inl,mst_kernels,mst_utils}.cuh`. **COPY, DO NOT
IMPROVE**, with three declared deviations (620, 621, 622). File-for-file map
in `PORTED_MAP.tsv`; what is not here and why, parameter by parameter, in
`UNPORTED.tsv`. DEVIATION numbers 620-622 spent; 623-629 free.

## Status

**CONSTRUCTION plus one Apple device's gates; no second vendor has run
this.** Rung 1 -- `single_linkage` on dense Float32 with `L2SqrtExpanded`
(cuML's `metric="euclidean"`), `connectivity="pairwise"`, `n_clusters`
given, `children` + `labels` out -- is ported, wired from cuML's entry down
to RAFT's kernels, and gated bit for bit against a host oracle in BOTH
numeric modes on one M4 (2026-08-23). Rung 2 (`connectivity="knn"`, cuML's
PYTHON default) is NOT ported and is refused by name; see UNPORTED. No
timing was measured and none is published.

    linkage_check mode=IDENTICAL ... ALL OK      (11 checks, 5 fixtures)
    linkage_check mode=FAST      ... ALL OK      (distance gates REPORT)

## Commands

    tools/with_build_lock.sh     pixi run mojo run -I . hierarchy/mojo_only/linkage_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . hierarchy/mojo_only/linkage_check.mojo

    tools/with_build_lock.sh     pixi run mojo run -I . hierarchy/linkage_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/linkage.card \
        tools/with_identical_mode.sh pixi run mojo run -I . hierarchy/linkage_main.mojo
    python3 tools/identity_trace_diff.py /tmp/linkage.apple.card /tmp/linkage.other.card

Every printed line carries the mode the binary COMPILED in. No pixi task
is registered (pixi.toml is not this lane's); the two lines for it are
under HAND-OFF.

## The identity content: why the MST, the dendrogram and the labels are a pure function of the input bits

Single linkage is four stages. For each, what is a reduction, and why it is
either order-free or pinned:

| stage | their spelling | ours | reduction? | why reproducible |
|---|---|---|---|---|
| distances `z[i,j]` | `cuvs::distance<L2SqrtExpanded>`: `\|\|x_i\|\|^2 + \|\|x_j\|\|^2 - 2 x_i.x_j`, norms precomputed, clamp, sqrt (`connectivities.cuh:157-160`) | FAST: `core/row_norms` + `core/gemm.gemm_nt` (MAX matmul) + `core/expand_distances`. IDENTICAL: `core/row_norms` (pinned fold) + `neighbors/mojo_only/pinned_distance_tile` (DEVIATION 505) | YES, two float folds: the norm and the dot product | IDENTICAL: the norm folds `NORM_TPB` strided partials then a halving tree (no lane primitive); the dot product is one thread walking the feature axis ascending through `identical_mul_add` + `ftz`; the epilogue is one `fma`; sqrt is `identical_sqrt`. Every one of those is a pure function of `d` and `NORM_TPB`. The host oracle replicates that arithmetic and the device matches it BYTE FOR BYTE on 5 fixtures (`check_linkage_distances_match_host_pinned`). `z` is exactly SYMMETRIC because both operands are `X`. FAST: not pinned (a MAX matmul's k-split is the vendor's), reported not asserted |
| self-loop | diagonal <- `FLT_MAX` (`:162-175`) | `self_loop_max_kernel` | no | elementwise |
| MST edge set | RAFT Boruvka with a cuRAND per-vertex ALTERATION of the weights to make them distinct, then per-row min (warp), per-color `atomicMin` on the altered double, "vertices added each other" test, label propagation by `atomicMin` (`mst_solver_inl.cuh`, `mst_kernels.cuh`) | the same kernels, the same launches, with the comparison on the TOTAL ORDER `(weight_order_key(w), min(u,v), max(u,v))` and NO alteration (DEVIATION 620) | YES, but every one is a MIN over a total order or an integer count: per-row min (32-lane halving fold), per-color min (three integer `atomicMin` phases), `next_color` min (`atomicMin`), edge count (`atomicAdd` of ones) | a min over a total order is associative and commutative with no rounding, so the fold shape, the lane count and the atomic landing order cannot move it; an integer add is exact. Their alteration made weights distinct with a RANDOM draw and is the one thing that is NOT a function of the input; replacing it with the undirected-index tie-break makes the per-round picks, the colors, the orientation `(src = the vertex that added, dst = its neighbor)` and therefore the edge set deterministic. Under a total order the MST is UNIQUE, so a serial Kruskal must agree with Boruvka exactly -- and does, on every fixture (`check_linkage_mst_matches_kruskal`). The edge WEIGHTS are copied from `z`, so every weight byte is the pinned distance byte |
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

## The three deviations

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

## Checks (all green, both modes, one M4)

| check | what it asserts | IDENTICAL | FAST |
|---|---|---|---|
| `check_linkage_distances_match_host_pinned` | per cell vs the host pinned arithmetic; symmetric; FLT_MAX diagonal | 9216/2304/4096/41209/10404 cells bitwise equal | REPORT (0 cells differ on these fixtures, see above) |
| `check_linkage_mst_matches_kruskal` | device Boruvka edge SET and ORDER == host Kruskal on the device's weights, weights bitwise; `m-1` edges | OK x5 (rounds 5/3/5/4/5) | OK x5 |
| `check_linkage_dendrogram_and_labels` | children rows (unordered pairs) == oracle; labels BITWISE == serial extract; partition == cut | OK x5 | OK x5 |
| `check_linkage_entry_matches_stages` | cuML's `single_linkage` entry returns the staged run's bytes (reach) | OK x5 | OK x5 |
| `check_linkage_union_find_matches_a_naive_one` | DEVIATION 622 | OK x5 | OK x5 |
| `check_linkage_launch_invariance` | the headline, above | every stage's bytes identical | children/labels/rounds identical; distances and MST reported stable |
| `check_linkage_batch_composition` | 37-row launch cells == the 203-row launch's | 1369/1369 | REPORT (0 differ) |
| `check_linkage_float64_reference` | MST total within 1e-4 of a Float64 direct-form Kruskal | rel 1.3e-8 / 0 / 4.6e-5 / 1.6e-8 / 1.3e-8 | same |
| `check_linkage_refusals` | `use_knn=True`, `metric=L1`, `n_clusters > n_rows`, `n_rows > 46340` raise by name | OK | OK |
| `check_linkage_sabotages` | the table below | OK | OK |
| `check_linkage_card_is_stable` | two cards, two launch shapes, 6 records identical | OK | REPORT: identical |

## Sabotage table

| arm | fixture | required | result (IDENTICAL build) |
|---|---|---|---|
| `LINK_SAB_RANDOM_ALTERATION` | dups_lattice | MUST FAIL the MST gate | `FAILED the MST gate as required: 38 of 47 sorted edge slots differ from Kruskal; children differ` |
| `LINK_SAB_RANDOM_ALTERATION` | blobs (tie-free) | report | `MST equal -- a random tie-break only moves ties` |
| `LINK_SAB_ROTATE_CONTRACTION` | hashed203 | MUST FAIL the distance gate | `FAILED the distance gate as required: 4553 of 41209 cells moved` (FAST: reported 4553 too) |
| `LINK_SAB_STD_SQRT` | hashed203 | report | `0 of 41209 cells differ from identical_sqrt's` -- Apple's sqrt is correctly rounded, so this arm CANNOT fail on this device; it is the arm that moves cells on NVIDIA's approximate sqrt (DEVIATION 258). THAT IS THE DOCUMENTED APPLE LIMIT of this check |
| `LINK_SAB_SORT_WEIGHT_ONLY` | dups_lattice | MUST FAIL the dendrogram gate | `FAILED the dendrogram gate as required: 47 of 47 children rows moved; labels MOVED` |

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

## ROW TEXT FOR THE IDENTITY LANE

| 40 | **single-linkage agglomerative clustering, END TO END** -- `hierarchy/ported/hierarchy/linkage.mojo` (cuML `linkage.cu`) -> cuVS `single_linkage.cuh`/`connectivities.cuh`/`mst.cuh`/`agglomerative.cuh` -> RAFT Boruvka `mst_solver_inl.cuh`/`mst_kernels.cuh`: pairwise L2 distances, Boruvka MST, sort, host union-find dendrogram, device label extraction | the distance tile is row 24's (DEVIATION 505) and the row norm row 19's, unchanged; NEW and vendor-dependent in THEIR spelling: (a) RAFT breaks MST weight ties with a cuRAND per-vertex ALTERATION (`mst_solver_inl.cuh:212-238`), so on duplicate points or equal distances the edge SET is a function of the RNG stream, not the input; (b) `coo_sort_by_weight` is an unstable `thrust::sort_by_key` on weight alone (`sort.h:101`), so equal-weight MST edges reach the dendrogram in an implementation-chosen order; (c) the per-color min is `atomicMin` on a double (Metal has no 64-bit atomic) | (a) DEVIATION 620: the total order `(weight_order_key(w), min(u,v), max(u,v))` everywhere RAFT compared the altered weight, per-color min in three integer `atomicMin` phases; (b) DEVIATION 621: a host merge sort on the same triple; (c) folded into 620. A min over a total order and an integer count are the only reductions in the MST and both are order-free, so the edge set, the dendrogram and the labels are pure functions of the distance bytes; gated: edge set == serial Kruskal (unique MST under a total order), labels BITWISE == serial extract, launch invariance across tile/MST/extract block sizes and two paddings/poisons, 37x37 cells == the same cells of 203x203, two cards identical; sabotages 620 (38/47 slots moved) and 621 (47/47 rows moved) fail as required, rotate-contraction moves 4553/41209 cells, std-sqrt moves 0 (Apple limit) | **construction + Apple gates; no second vendor has run it.** Open: rung 2's `connect_knn_graph` host overload picks a RANDOM vertex per component (`std::mt19937(std::random_device())`, `mst.cuh:167-190`) and must be pinned when ported |

## HAND-OFF TO THE IDENTITY LANE

1. `IDENTITY_PATHS.md`: append the row above as row 40 (or the next free
   number).
2. `UNWIRED.md`, NOT wired table: `hierarchy/ported/sparse/solver/detail/
   mst_kernels.mojo::add_reverse_edge` | nothing (single linkage passes
   `symmetrize_output=false`, `cuvs mst.cuh:297-298`) | a caller wanting a
   symmetric MST edge list; transliterated and gated by nothing.
3. `pixi.toml` tasks:

       check-linkage = "mojo run -I . hierarchy/mojo_only/linkage_check.mojo"
       linkage-main  = "mojo run -I . hierarchy/linkage_main.mojo"

   and `check-linkage` into whichever aggregate the unsupervised identity
   gate runs in both modes.
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
- A second vendor. Nothing here has run on NVIDIA or AMD; the NVIDIA risk
  is the one the std-sqrt sabotage names (approximate sqrt at the distance
  seam, routed through `identical_sqrt`, unverified there), and the
  `atomicMin` phases are integer and should be inert.
- `distance_threshold` / `compute_distances` on the Python surface (cheap:
  `out_delta` exists).
