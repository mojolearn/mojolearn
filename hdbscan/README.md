# hdbscan: HDBSCAN, from cuML / cuVS / RAFT

Mirrors `cuml/cpp/src/hdbscan/runner.h` (the entry), which forwards to
`cuvs/cpp/src/cluster/detail/single_linkage.cuh::build_mr_linkage` and
back into `cuml/cpp/src/hdbscan/detail/{condense,stabilities,select,
extract}.cuh`, whose MST and dendrogram are the ones `hierarchy/` already
ports. **COPY, DO NOT IMPROVE**, with the deviations listed below.
File-for-file map in `DERIVATION_MAP.tsv`; what is not here and why,
parameter by parameter, in `NOT_IMPLEMENTED.tsv`. DEVIATION numbers 1600-1613
spent; 1614-1629 free.

## Status

## OPEN DEFECT, and the gate is RED because of it. DEVIATION 1614.

**`check_condensed_tree_vs_oracle` FAILS on `blobs96` in both modes and the
lane is committed red rather than green.** Nothing downstream of the
condensed tree should be believed until this closes.

What is established, by the orchestrator's run on 2026-08-25:

    condense diff [FAST] blobs96 first at edge 0 of 100:
      device (p=96, c=97, sz=32)   oracle (p=96, c=97, sz=64)

The edge COUNTS agree, 100 edges over the same cluster count, and the first
divergence is at edge 0 where both describe the SAME split of the 96-point
root. Device emits the size-32 child first, the oracle emits the size-64
child first, and 32 + 64 = 96. So this is not a different tree. **It is a
LEFT/RIGHT ORDER difference, and it propagates: 30 parents, 34 children and
6 sizes differ over the whole tree.**

It is not in `condense`. Both implementations read
`left = children[2i]`, `right = children[2i+1]` and emit left first, which
is `condense.cuh`'s own order. The disagreement is one stage earlier, in
what the DENDROGRAM put in those two slots.

**The suspect, named precisely.** `checks/hdbscan_oracle.mojo::oracle_dendrogram`
appends `(find(lo[i]), find(hi[i]))`, taking its child order from the
CANONICALIZED edge pair, `lo` being `min(u, v)`. The device path goes
through `hierarchy/impl/cluster/detail/agglomerative.mojo::build_dendrogram_host`
fed by the device MST, which carries the edge in the orientation Boruvka
stored it. At any edge where the MST stored the higher index first, left and
right swap.

**Which one is right is a question for their source and has not been
answered.** `agglomerative.cuh:104-155` is what decides it, together with
what `mst_solver` actually stores in `src` and `dst`. Do not "fix" the
oracle to match the device or the device to match the oracle without reading
that, because the two are equally easy to change and only one of them is
faithful. `hierarchy`'s DEVIATION 621 replaced their sort with one keyed on
`(weight_order_key, min(u,v), max(u,v))`, and whether that canonicalization
was meant to reach the stored ORIENTATION or only the sort ORDER is the
crux.

Everything ahead of this stage passes. Refusals, core distances, the mutual
reachability matrix and its symmetry, and the signed-zero max all gate green
in both modes.


**CONSTRUCTION PLUS WRITTEN GATES. NOTHING HAS BEEN RUN.** No build, no
check, no driver and no benchmark has been executed against this
directory. **There is no performance number in this lane and there is no
cross-vendor claim in this lane, because nothing has been run.** The
thirteen checks below are written, not passed; the sabotage table states
what each arm MUST move, not what it did move. The orchestrator builds,
runs and records; every "OK" line in this file is a line the code is
written to print, and until a run prints it the honest reading of this
lane is "a port and a suite exist".

Rung 1 is `HDBSCAN.fit` on dense Float32 with `L2SqrtExpanded`
(cuML's `metric="euclidean"`), the DENSE mutual reachability graph
(DEVIATION 1600), Excess of Mass selection, `labels_` and `core_dists`
out. Rung 2 -- the sparse k-NN mutual reachability graph, which is
cuML's own dispatch -- is NOT ported and is refused by name; see
`NOT_IMPLEMENTED.tsv` for the two walls.

## Commands

    tools/with_build_lock.sh     pixi run mojo run -I . hdbscan/checks/hdbscan_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . hdbscan/checks/hdbscan_check.mojo
    tools/with_build_lock.sh     pixi run mojo run -I . hdbscan/hdbscan_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/hdbscan.card \
        tools/with_identical_mode.sh pixi run mojo run -I . hdbscan/hdbscan_main.mojo
    python3 tools/identity_trace_diff.py /tmp/hdbscan.apple.card /tmp/hdbscan.other.card

## The upstream pins

Read 2026-08-25 with `git -C <checkout> rev-parse --short HEAD`:

| upstream | checkout | pin | what this lane reads |
|---|---|---|---|
| cuML | `upstream/cuml-v26.08.00` | `265b9da` | `cpp/src/hdbscan/**`, `cpp/include/cuml/cluster/hdbscan.hpp` |
| cuVS | `upstream/cuvs` | `94c2819` | `cpp/src/cluster/detail/single_linkage.cuh`, `cpp/src/neighbors/detail/reachability.cuh`, `cpp/include/cuvs/cluster/agglomerative.hpp` |
| RAFT | `upstream/raft` | `661a3b8` | `sparse/convert/csr.cuh` (the CSR scan) |

**The cuVS pin AGREES with `hierarchy/DERIVATION_MAP.tsv`** (`94c2819`), and
the brief's line numbers for `build_mr_linkage` (`:50-118`) are that
tree's. `upstream/cuml` (`00094f7`, 25.08) is NOT used: it is the wrong
tree for `hdbscan/`, exactly as `[[mojolearn-upstream-pinned-trees]]`
warns.

**A discrepancy found while reading, recorded rather than resolved.**
`upstream/cuvs-v26.08.00` (`6ba2ce2`) and `upstream/raft-v26.08.00`
(`ebf9268`) also exist in the checkout directory. cuML `265b9da`'s
`runner.h:66-113` fills `linkage_params.all_neighbors_params`, a field
that exists in cuVS `6ba2ce2` (`agglomerative.hpp:141-147`) and NOT in
cuVS `94c2819`, whose `mutual_reachability_params` carries `min_samples`
and `alpha` only. **So the two trees this lane reads do not compile
against each other.** It does not change a line of the port: every
function ported here is identical in body between the two cuVS trees,
and the one place they differ -- how `build_mr_linkage` obtains the
graph -- is DEVIATION 1600's refused half in both. Line citations
throughout this lane are `94c2819`'s, matching `hierarchy/`.

## WHAT THIS LANE REUSES RATHER THAN REWRITES

Most of HDBSCAN's substrate was already ported. Nothing below was copied
into `hdbscan/`, re-derived, or edited; every one is an `import`, and
`hierarchy/NOT_IMPLEMENTED.tsv` line 8's claim that the MST and agglomerative
code "would reuse this lane's `mst_solver.mojo` and `agglomerative.mojo`
unchanged" **held, with one qualification stated at the end.**

| file | what this lane takes from it | reached from |
|---|---|---|
| `hierarchy/impl/sparse/solver/mst_solver.mojo` | RAFT's Boruvka `MST_solver`, `Graph_COO`, the `mst()` entry | `build_sorted_mst` |
| `hierarchy/impl/sparse/solver/detail/mst_kernels.mojo` | every Boruvka kernel, including DEVIATION 620's three integer `atomicMin` phases | same |
| `hierarchy/impl/cluster/detail/mst.mojo` | `build_sorted_mst`, `get_n_components`, and the `connect_knn_graph` refusal this lane depends on being loud | `build_mr_linkage` |
| `hierarchy/impl/cluster/detail/agglomerative.mojo` | `UnionFind` (DEVIATION 622) and `build_dendrogram_host` | `build_mr_linkage` |
| `hierarchy/impl/cluster/detail/connectivities.mojo` | the PAIRWISE distance graph: `pairwise_distances`, `fill_indices2`, `indptr_sequence_kernel`, `self_loop_max_kernel`, `FLOAT32_MAX`, `PAIRWISE_MAX_ROWS`, `DISTANCE_L2_SQRT_EXPANDED` | `build_mr_linkage` |
| `hierarchy/impl/sparse/op/sort.mojo` | `coo_sort_by_weight` (DEVIATION 621) AND `merge_sort_u64_with_index`, which is also the condensed tree's sort (DEVIATION 1611) | `build_sorted_mst`, `condense()`, `select_parent_csr` |
| `hierarchy/checks/edge_order.mojo` | the **total order**: `weight_order_key`, `pack_edge_key`, `triple_less`, `edge_lo`/`edge_hi`. **No second order was invented.** It orders MST edges, the segment minimum of the lambdas (DEVIATION 1604), the parent-lambda max in `do_labelling_on_host`, `max_lambda_of`, and the oracle's `(distance, index)` k-NN sort | everywhere a float is compared |
| `hierarchy/checks/nan_guard.mojo` | DEVIATION 623's NaN refusal on the distance matrix, called from inside `pairwise_distances`. This lane's DEVIATION 1607 covers the arrays 623 does not reach and does not duplicate it | `pairwise_distances` |
| `hierarchy/checks/linkage_oracle.mojo` | the oracle's substrate: `host_row_norms_pinned`, `host_pinned_distance`, `host_kruskal`, `NaiveUnionFind`, `merge_sort_f64_with_index`, `partitions_agree`, and the fixture hash `_hash_unit` | `hdbscan_oracle.mojo`, `hdbscan_fixture.mojo` |
| `neighbors/estimator.mojo` | `knn_search_traced` -- the k-NN, its tile planning, its arm pin (DEVIATION 509), and its `(distance, index)` host sort, which is what makes the k-th order statistic well defined | `compute_knn` |
| `neighbors/checks/pinned_distance_tile.mojo` | the pinned distance arithmetic (DEVIATION 505), through `hierarchy`'s call, not this lane's | the distance step |
| `neighbors/checks/select_radix_identical.mojo` | the composite `(distance, index)` top-k key (DEVIATION 500), through `knn_search` | the k-NN |
| `core/identity_trace.mojo` | `IdentityTrace`, `record_device`, `record_host`, `record_list_*`, `first_divergence`. **Every stage hash in this lane goes through it** | the whole fit |
| `core/row_norms.mojo`, `core/gemm.mojo`, `core/expand_distances.mojo` | the FAST distance arm, through `hierarchy`'s call | the distance step |
| `checks/numerics.mojo` | `ftz`, `identical_mul_add`, `identical_mul`, `identical_div`, `identical_sqrt`, and **`identical_fmax` / `portable_fmaxf`**, which is the total-order max DEVIATION 1601 needs. It already existed (DEVIATION 825, for softmax) and was not re-derived | the three-way max, every seam |

**Not re-implemented, and it is worth naming the negative explicitly**:
the MST solver, the agglomerative labeling, the union-find, the edge
total order, k-NN, the top-k selection, the distance kernels, GEMM, the
reductions, the transcendentals and the identity tracing. None of them
appears in `hdbscan/`.

**Where the `hierarchy/NOT_IMPLEMENTED.tsv` line-8 claim needed a
qualification.** It said HDBSCAN's linkage "would reuse this lane's
`mst_solver.mojo` and `agglomerative.mojo` unchanged". Both DO carry
over unchanged and are imported as written. What the sentence does not
say, and what this lane found, is that **the GRAPH does not carry over**:
`build_mr_linkage` feeds `build_sorted_mst` a SPARSE k-NN graph whose MST
is a forest, and the fix-up that forest needs is the part
`hierarchy/DERIVATION_MAP.tsv` records as NOT PORTED and raises by name. So
the reusable half is reusable and the half above it is not, which is
DEVIATION 1600 and is the whole reason this lane ships a dense graph. The
claim is not false; it is narrower than it reads.

## The identity table

Same shape as `IDENTITY_PATHS.md`'s ledger. The three rows the brief
names are H1, H2 and H3.

| # | pathway | order-dependent? | what `IDENTICAL` does | status |
|---|---|---|---|---|
| **H1** | **the core distance, a k-TH ORDER STATISTIC** -- `reachability.cuh:60-62`, `knn_dists[row * n_neighbors + (min_samples - 1)]` over `knn_search`'s result | **the VALUE is not; the NEIGHBOUR IDENTITY is.** The k-th smallest element of a multiset is a pure function of the multiset, so it does not depend on which of several equal elements the selector picked. WHICH neighbour supplied it does: upstream's `select_k` comparator is the distance only (IDENTITY_PATHS row 11) and their tie is arrival order | **PIN, and the pin is one directory over.** The value inherits `neighbors/`: DEVIATION 505 pins the distances, DEVIATION 500 ranks on the composite `(distance, index)` key so the SET is the lowest-index one, DEVIATION 509 pins the ARM to tiled on every column, and `knn_search`'s host sort on `(distance, index)` makes slot `k-1` the k-th smallest rather than an arbitrary member of the top-k set. This lane adds DEVIATION 1602: it READS from the sorted result and states the two claims apart, and it never reads the index at all. **`check_core_distances_vs_oracle` asserts the value per cell, bit for bit, against a host oracle that takes the k-th order statistic by FULLY SORTING each row rather than by a top-k, and prints how many rows have a TIED k-th neighbour so a green line cannot be mistaken for a claim about the neighbour** | construction; gate written, not run |
| **H2** | **mutual reachability, a THREE-WAY MAX** of `core[row]`, `core[col]` and `alpha * d(row,col)` -- `reachability.cuh:127-130`, the CUDA `max` | **yes, at the zeros and at NaN.** IDENTITY_PATHS row 39, measured on three columns: `max(+0.0, -0.0)` is `-0.0` on Apple (the second operand) and `+0.0` on NVIDIA and AMD; `min(-0,+0)` splits the same way; a computed NaN's payload is `0x7fc00000` / `0x7fffffff` / `0xffc00000`. A three-way max is a two-level fold, so the operand ORDER decides a zero's sign, and that sign lands in the graph as a WEIGHT | **PIN — DEVIATION 1601.** `identical_fmax` (`numerics.mojo`, DEVIATION 825's primitive, imported not re-derived): NaN canonicalized BEFORE the compare, operands flushed, selection on `_total_order_key`, an INTEGER map with `-0.0` strictly below `+0.0`. No hardware max and no float compare on the path. Being a total order, the max is commutative and associative, so `mr(a,b) == mr(b,a)` BIT FOR BIT -- which is what makes `hierarchy`'s undirected edge order well defined on this graph and is asserted first in the gate. **Reachable?** Not through the distances (both arms clamp `if dist <= 0.0: dist = 0.0`, an IEEE compare that maps every negative residue and `-0.0` to `+0.0` on every vendor), so the pin is INERT on the default path and `check_mutual_reachability_ties` PLANTS `-0.0` into a core-distance array and a distance cell and drives the kernel with them. `HDB_SAB_HW_MAX` is the arm | construction; gate written, not run |
| **H3a** | **the condensed tree: a TRAVERSAL that is a NUMBERING** -- `condense.cuh:37-66` BFS, `:156-160` `next_label++` in visit order | **not a float order, but a total one, and everything downstream is indexed by it.** `stabilities[c]`, `is_cluster[c]`, `births[c]`, the CSR segment boundaries and the final label numbering are all keyed on `relabel`, which is assigned in BFS order | **PINNED BY BEING THEIRS, AND SAID SO.** The traversal is transcribed level by level from `:46-64`; there is no atomic, no thread and no float in it, so it is reproducible for the reason a `for` loop is. What could break it is a rewrite to a different traversal, so `HDB_SAB_CONDENSE_DFS` swaps the queue for a stack and `check_condensed_tree_vs_oracle` compares NODE FOR NODE (parents, children, sizes exactly; lambdas bitwise) rather than comparing a summary | construction; gate written, not run |
| **H3b** | **the stability sum, a FLOAT SUM OVER A TREE TRAVERSAL** -- `kernels/stabilities.cuh:39-44`, a float `atomicAdd` per condensed edge into a per-cluster cell | **yes, and run to run rather than merely vendor to vendor.** Rows 1, 8 and 36's defect. Two more folds sit beside it: the per-parent minimum lambda (`cub::DeviceSegmentedReduce::Min`, a library fold shape, with row 39's `min(+0,-0)` split inside it) and Excess of Mass's per-node subtree sum (`thrust::transform_reduce`, a device tree reduction whose result decides a BOOLEAN and therefore a CLUSTER -- row 7's class) | **REPLACE, three times.** DEVIATION 1603: one thread per CLUSTER, walking that cluster's contiguous CSR segment ASCENDING through `ftz`/`identical_mul_add`; no atomic, no lane primitive, no block fold, no cross-thread communication. DEVIATION 1604: the segment minimum is a serial ascending scan comparing `weight_order_key`, with `FLT_MAX` written explicitly as the empty-segment identity where theirs leaves index 0 uninitialized. DEVIATION 1605: the Excess-of-Mass loop runs on the host over ONE download, its subtree sum serial and ascending, and the `stability[node] = subtree_stability` write-back kept IN PLACE because the loop runs leaves-to-root and an ancestor reads the updated children. **WHICH ORDER IS PINNED: the condensed tree's `(parent, child)` order, which is a TOTAL order because every node of a tree has exactly one parent, so `child` is unique across the array and no tie is possible inside a segment.** Gates: `check_stabilities_vs_oracle` per cluster bit for bit; arms `HDB_SAB_STABILITY_DESCENDING` and `HDB_SAB_EOM_NO_UPDATE` | construction; gates written, not run |
| H4 | the dense mutual reachability graph itself | one thread per cell, three loads, one store; no fold | PIN by construction; `check_launch_invariance` holds the cells fixed across two block sizes and across a 37x37 launch versus the same cells of the full one | construction |
| H5 | `lambda = 1 / distance` (`condense.cuh:149`), on the host | a host divide; IDENTITY_PATHS row 18's class (a host libm difference is a cross-HOST difference) | DEVIATION 1606: `identical_div` (row 49's seam). Their `distance > 0.0` guard and `FLT_MAX` sentinel kept unchanged. Arm `HDB_SAB_LAMBDA_STD_DIV` | construction |
| H6 | `cluster_sizes[0] += size` in `excess_of_mass` (`select.cuh:178-179`) | **a data race in THEIR kernel**: a non-atomic read-modify-write into one cell from every thread whose parent is the root | DEVIATION 1613: a serial host fold, exact integer sum. `[[assume-our-code-is-broken]]` says fix their bugs rather than port them, numbered and checked; read off their source, NOT observed (we have no CUDA build to race) | construction |
| H7 | a NaN or infinite core distance / mutual reachability cell / delta / lambda | a computed NaN's payload is the vendor's (row 39), and every one of those arrays is a recorded card stage | REFUSE — DEVIATION 1607, by name, with the count and the first index. `+inf` IS refused here where `hierarchy` keeps it, and the reason is written in the block: `1/inf` is `+0.0`, which is indistinguishable at the stability seam from `1/FLT_MAX`'s underflow -- two causes collapsing onto one value inside a certified stage | construction |
| H8 | the whole pipeline's REPRODUCIBILITY UNDER A ROW PERMUTATION | **yes, and it is the algorithm's, not this port's.** The MST tie-break is `(weight key, min(u,v), max(u,v))` and it READS THE INDEX, so under an exact tie a permutation can select a different (equally minimal) MST. **Mutual reachability makes ties ENDEMIC**: `mr(a,b)` collapses to a CORE DISTANCE whenever the two points are closer than their cores, and many pairs then share one value | **NAMED, NOT CLOSED.** `check_permutation_invariance` counts the tied pairs first and ASSERTS the partition where there are none, RECORDS the outcome with the counts where there are. Upstream's answer under the same tie is a cuRAND draw (`hierarchy`'s DEVIATION 620), so ours is strictly more reproducible and still not permutation-invariant. Closing it needs a tie-break that does not read an index, and this lane does not have one | **OPEN, and stated rather than hidden** |

## The deviations

**DEVIATION 1600 -- the mutual reachability graph is the COMPLETE
PAIRWISE one, not their sparse k-NN COO**
(`hdbscan/checks/mutual_reachability_dense.mojo`, the block). Two
walls, both other lanes': `connect_knn_graph`/`cross_component_nn`/
`merge_msts` are NOT PORTED in `hierarchy/` and a k-NN mutual
reachability graph over separated clusters is a forest; and
`mutual_reachability_knn_l2` needs a `DistanceEpilogue` template
`neighbors/` does not carry, which is not optional because the epilogue
is not monotone in the distance alone. The complete graph is connected,
so Boruvka finishes in one component and the fix-up loop is never
entered. The MST of the COMPLETE mutual reachability graph is the
reference answer their sparse path accelerates, so where their fix-up
would have found the same edges the two agree and where it would not,
ours is the exact one. The price is `m * m` cells, which is why
`PAIRWISE_MAX_ROWS` (46340) is a harder bound here than in `hierarchy`.
NOT MEASURED: that sentence is an array-size statement.

**DEVIATION 1601 -- the three-way max is a total-order selection**
(same file, second block; the helper is
`checks/hdbscan_sabotage.mojo::mr_max3`). Row H2 above.

**DEVIATION 1602 -- the core distance is read from a `(distance,
index)`-SORTED k-NN result, and the two claims are stated apart**
(`impl/hdbscan/detail/reachability.mojo`, the block). Row H1 above.

**DEVIATION 1603 -- the stability sum is a per-cluster serial ascending
fold, not a float `atomicAdd`** (`impl/hdbscan/detail/stabilities.mojo`,
the block). Row H3b.

**DEVIATION 1604 -- the per-parent minimum lambda is a total-order scan,
not `cub::DeviceSegmentedReduce::Min`** (same file, second block). Row
H3b. Note the second reason in the block, which a reader will not guess:
their `births_parent_min[0]` is never written and never read, so its
`rmm::device_uvector` contents are uninitialized memory their code is
careful not to touch; ours writes the identity explicitly.

**DEVIATION 1605 -- the Excess-of-Mass loop runs on the host over one
download, and its subtree sum is serial and ascending**
(`impl/hdbscan/detail/select.mojo`, the block). Row H3b. Their loop is
already on the host; what moved is one scalar readback and one
`thrust::transform_reduce` per node.

**DEVIATION 1606 -- `lambda = 1 / distance` through `identical_div`**
(`impl/hdbscan/detail/condense.mojo`, the block). Row H5.

**DEVIATION 1607 -- a non-finite value is refused by name before it can
reach a recorded stage** (`checks/mutual_reachability_dense.mojo`,
third block). Row H7.

**DEVIATION 1609 -- `TreeUnionFind::find` is iterative**
(`impl/hdbscan/detail/extract.mojo`, the block). Theirs recurses; same
root, same fully compressed array, no unbounded stack. `hierarchy`'s
DEVIATION 622 is the sibling with the opposite cause (there the recursion
was fine and the INDEXING was out of bounds).

**DEVIATION 1610 -- `get_probabilities` is DEFERRED and `probabilities_`
is refused by name** (`estimator.mojo::hdbscan_probabilities_host`).
Returning zeros or ones would put a number nobody computed into a field a
caller will plot. Closing it is DEVIATION 1604's fold with the comparison
reversed; `NOT_IMPLEMENTED.tsv` has the row.

**DEVIATION 1611 -- the condensed-tree sort is a host stable merge sort
on a packed `(parent, child)` key**
(`impl/hdbscan/condensed_hierarchy.mojo`, the block). The merge sort is
`hierarchy`'s, imported. `pack_edge_key` is NOT reused for it and the
block says why: that packing has 16-bit vertex fields sized for an
undirected edge of the distance graph, and these are 32-bit tree node
ids.

**DEVIATION 1613 -- `cluster_sizes[0]` is a data race in their kernel and
a serial fold in ours** (`impl/hdbscan/detail/select.mojo`, second
block). Row H6.

(1608 and 1612 were reserved during drafting and released; they are not
used. 1614-1629 are free.)

## The checks

Thirteen, all in `checks/hdbscan_check.mojo`, each printing
`check_<name> OK [<mode>]: <what it established>`. **These are written,
not run.**

| check | what it asserts | IDENTICAL | FAST |
|---|---|---|---|
| `check_hdbscan_refusals` | 15 arms and parameters raise BY NAME, and the message NAMES what was refused | assert | assert |
| `check_core_distances_vs_oracle` | per cell vs the oracle's k-th order statistic (a full row sort, not a top-k); prints the count of rows with a TIED k-th neighbour | assert, bitwise | REPORT |
| `check_mutual_reachability_ties` | `mr` exactly SYMMETRIC; per cell vs the host three-way max; a REVERSED row order gives the same partition; a PLANTED `(+0, -0)` resolves to `+0.0` | assert | symmetry and reversal assert; the cell and planted comparisons REPORT |
| `check_condensed_tree_vs_oracle` | node for node: parents, children, sizes exactly, lambdas bitwise, plus the shape | assert | structure asserts, lambdas REPORT |
| `check_stabilities_vs_oracle` | per cluster, bit for bit, POST-selection (so the write-back is covered) | assert | REPORT |
| `check_labels_vs_oracle` | labels vs the oracle; the OUTLIER COUNT separately; and on the four fixtures that plant an assignment: ASSERTS that every planted-noise point returns `-1` and that no two planted clusters are MERGED, RECORDS how far any planted cluster was SPLIT and how many of its points came back noise | oracle asserts; planted asserts | oracle RECORDS; **planted still asserts** |
| `check_permutation_invariance` | same points, different row order, same partition and same noise set | assert where the mutual reachability has NO tied pair; RECORD with counts where it has | same |
| `check_launch_invariance` | labels, core distances and stabilities across tile 256/64, MST 256/128, mr 256/64, core 256/64, stability 256/64, select 256/64, a 1024-float NaN padding and a 37-float 1e30 padding; plus 37x37 mutual reachability cells alone vs inside the 96-row launch | assert | assert |
| `check_card_is_emitted` | at least 20 stages recorded, and two cards from two launch shapes record-for-record identical | assert | assert |
| `check_hdbscan_signed_zero_inputs` | a `-0.0` coordinate on every row moves no core distance, no stability and no label against the `+0.0` twin fixture | assert | assert |
| `check_hdbscan_selection_leaf` | the LEAF arm matches the oracle's leaf rule; RECORDS on how many fixtures it differs from EOM (a zero is an owed fixture, not a failure) | assert | assert |
| `check_hdbscan_float64_reference` | the Float32 mutual-reachability MST total within 1e-4 relative of a Float64 DIRECT-FORM one | assert | RECORD |
| `check_hdbscan_sabotages` | the table below | assert / record per arm | same |

## SABOTAGES TO PERFORM

Every named path has an arm. `[[reached-but-inert]]`: a path that runs is
not a path that is gated, so each MUST FAIL line raises if the gate does
NOT move.

| check | sabotage | exactly what it corrupts | what must move |
|---|---|---|---|
| `check_mutual_reachability_ties` | `HDB_SAB_MR_TWO_WAY` | drops `core_dists[col]` from the three-way max, so `mr(a,b)` becomes `max(core[a], alpha*d)` | **MUST FAIL**: `mr` stops being symmetric. Raises if 0 of the `m(m-1)/2` pairs became asymmetric on `dups_lattice48` |
| `check_mutual_reachability_ties` | `HDB_SAB_HW_MAX` | the three-way max becomes the STDLIB `max` on a PLANTED `(+0.0, -0.0)` core-distance and distance array | **RECORDED with the moved-cell count and the first `bits -> bits` pair.** Row 39: the stdlib max returns the second operand on Apple and the IEEE-2019 maximum on NVIDIA and AMD, so 0 here would be a fact about LLVM folding `maxnum` into a compare-select (the `HW_MAX_CLAMP` lesson), not about the pin |
| `check_core_distances_vs_oracle` | `HDB_SAB_CORE_KTH_PLUS_ONE` | `core_distances` reads column `min_samples` instead of `min_samples - 1` | **MUST FAIL**: raises if 0 of the 96 core distances moved on `blobs96` |
| `check_condensed_tree_vs_oracle` | `HDB_SAB_CONDENSE_DFS` | `bfs_from_node`'s level queue becomes a DEPTH-FIRST stack: same node set, different `next_label` assignment | **MUST FAIL**: raises if the condensed `children` array and the edge count are both unchanged on `blobs96` |
| `check_stabilities_vs_oracle` | `HDB_SAB_STABILITY_DESCENDING` | the per-cluster fold walks its segment DESCENDING | **RECORDED with the moved count.** A summation order; a fixture whose segments are all one or two terms cannot separate the two directions, and that is a property of the fixture, not of the pin |
| `check_labels_vs_oracle` (via the selection) | `HDB_SAB_EOM_NO_UPDATE` | drops `stability[node] = subtree_stability` on a deselected node (`select.cuh:227`) | **MUST FAIL**: SWEPT over all six fixtures, raises if the selection flags, the labels and the stabilities are ALL unchanged on every one of them. A per-fixture assertion would be an assertion about the fixture (whether its tree deselects a node at all); the sweep is an assertion about the pin |
| `check_hdbscan_refusals` | `HDB_SAB_SKIP_GUARDS` | skips DEVIATION 1607's non-finite refusals | **RECORDED**: how many core distances came back NaN and with WHICH PAYLOAD (Apple `0x7fc00000` / NVIDIA `0x7fffffff` / AMD `0xffc00000`) -- the byte the guard keeps out of a card stage |
| `check_condensed_tree_vs_oracle` (lambdas) | `HDB_SAB_LAMBDA_STD_DIV` | `1 / distance` through the hardware divide instead of `identical_div` | **RECORDED**: 0 is the expected answer on Apple (its divide is correctly rounded). This is the arm that moves on a column whose divide is approximate -- the DEVIATION 258 shape, one operation over, and the documented Apple limit of this check |
| (manual, not an arm) | key on `abs(lambda)` in `weight_order_key` | makes the lambda min/max blind to the sign of zero | the `hierarchy` lane already records this manual sabotage for its own gate; it is named here because DEVIATIONS 1604 and `do_labelling_on_host`'s max ride on the same function, and a change to it must be re-run against BOTH lanes |

## WHAT THE ORCHESTRATOR MUST WIRE

**`pixi.toml`.** Two lines, in the file's existing format, to be added in
the alphabetical block that currently runs from `check-kde` to
`check-metrics-trust` (around line 978):

    check-hdbscan = "mojo run -I . hdbscan/checks/hdbscan_check.mojo"
    hdbscan-main = "mojo run -I . hdbscan/hdbscan_main.mojo"

FAST is `pixi run check-hdbscan`; IDENTICAL is
`tools/with_identical_mode.sh pixi run check-hdbscan`. No `*-identity`
task, by the same design decision `hierarchy/README.md` records.

**`UNWIRED.md`**, NOT wired table, two rows:

    hdbscan/estimator.mojo::hdbscan_fit_host | nothing | bindings/ and a
      python/mojolearn HDBSCAN class; see HAND-OFF below
    hdbscan/impl/hdbscan/detail/extract.mojo::do_labelling_on_host's
      allow_single_cluster branch | nothing | a fixture with
      allow_single_cluster=True; see WHAT IS OWED

**`IDENTITY_PATHS.md`**: this lane's rows are H1-H8 above and want a
ledger row of their own (the next free number after 51). The row text is
the H2/H3b pair plus DEVIATION 1600's refusal, and it should be written
from this README rather than beside it, as `hierarchy/README.md`'s
hand-off item 1 asks.

**Nothing else.** No file outside `hdbscan/` was created or edited by
this lane, including `hierarchy/` and `neighbors/`, which it depends on.
The two changes those lanes MIGHT want are named under WHAT IS OWED and
were deliberately not made.

## HAND-OFF: the Python surface

For whoever owns `python/` and `bindings/`. The target is
`hdbscan.HDBSCAN` (scikit-learn-contrib) as cuML mirrors it:

- `min_samples=5`, `min_cluster_size=5`, `max_cluster_size=0`,
  `alpha=1.0`, `allow_single_cluster=False`,
  `cluster_selection_method="eom"`, `metric="euclidean"` -> their
  defaults, `hdbscan.hpp:132-140` and `:197-198`.
- `cluster_selection_epsilon` must be `0.0` -- refuse any other value by
  name (rung 2, `NOT_IMPLEMENTED.tsv`).
- `metric` must be `"euclidean"` or `"l2"` -> `L2SqrtExpanded` (1);
  refuse every other value by name, as their own `RAFT_EXPECTS` does.
- Outputs: `labels_` (`n_rows` int32, `-1` for noise, numbered by
  ascending condensed cluster id -- cuML's rule and
  scikit-learn-contrib's, not an invention of this surface),
  `n_clusters_`, and `core_distances_`. `probabilities_` must raise
  `NotImplementedError` by name (DEVIATION 1610); so must
  `outlier_scores_`, `condensed_tree_`, `single_linkage_tree_` and
  `approximate_predict` (soft clustering, `NOT_IMPLEMENTED.tsv`).
- Entry: `hdbscan/estimator.mojo::hdbscan_fit_host(ctx, x_ptr,
  labels_ptr, core_dists_ptr, info_ptr, n_rows, n_cols, ...)` with
  `info_ptr` an Int32 buffer of FOUR.
- **A Python fit through this entry DOES emit the identity card**, unlike
  `hierarchy`'s, because `fit_hdbscan` threads the trace itself.

## WHAT IS OWED

**The second and third vendor legs. Neither has been run, and neither has
the first.** Nothing in this directory has been built or executed.

1. **Apple M4, both modes.** The first thing owed is the FIRST leg: build
   `check-hdbscan` under FAST and under
   `tools/with_identical_mode.sh`, record every printed line, and fill
   the sabotage table's "what actually moved" column, which is currently
   empty by design. Until that runs, the "MUST FAIL" claims are
   predictions.
2. **NVIDIA H100.** The named risks, in the order a card diff would meet
   them: (a) `sqrt` at the distance seam -- NVIDIA's is APPROXIMATE
   (DEVIATION 258) and the path is routed through `identical_sqrt`, but
   unverified there; that is `hierarchy`'s owed item inherited whole.
   (b) The three-way max: `max(+0.0, -0.0)` is `+0.0` on NVIDIA and
   `-0.0` on Apple, so `HDB_SAB_HW_MAX` should move DIFFERENT cells on
   the two columns and the pinned arm should move none. (c) NaN payloads
   at `HDB_SAB_SKIP_GUARDS`. (d) `identical_div` in DEVIATION 1606 --
   `numerics.mojo` says row 49 is "NOT cross-vendor certified until a leg
   re-prints `check-division`'s certificate hash".
3. **AMD MI325X.** Additionally: the 64-wide wavefront. This lane
   launches no lane primitive and no block fold of its own, so the
   exposure is entirely through `neighbors/`'s k-NN (DEVIATION 509 pins
   AUTO to the tiled arm on every column precisely because the fused
   arm's 32-lane network does not exist there) and through `hierarchy/`'s
   MST, whose folds are integer.
4. **The deferred upstream files**, each with its row in `NOT_IMPLEMENTED.tsv`:
   `soft_clustering.cuh`, `prediction_data.cu`, `detail/predict.cuh`,
   `detail/membership.cuh` (DEVIATION 1610, small), the epsilon search in
   `detail/select.cuh`, the NN-descent graph builder, and the SPARSE
   mutual reachability graph with its cross-component fix-up (DEVIATION
   1600, which is the rung-2 headline).
5. **Two switch sides with no check.** `allow_single_cluster=True` and
   `max_cluster_size != 0` are both HONORED end to end and both UNRUN by
   the suite: every fixture uses their defaults. That is PORTING_RULES
   rule 8's exact shape -- the file has a caller, the suite is green,
   and one side of the switch never runs. Two named checks are owed, and
   `allow_single_cluster=True` in particular is the only thing that
   reaches `do_labelling_on_host`'s root branch (`extract.cuh:141-160`),
   which is currently dead code in this port.
6. **Permutation invariance is OPEN and named** (row H8). The gate
   records rather than asserts wherever the mutual reachability has a
   tied pair, which on these fixtures is most of them. Closing it needs a
   tie-break that does not read a row index, and no such order exists in
   this tree.
7. **A change `neighbors/` may want, NOT MADE.**
   `neighbors/impl/neighbors/detail/knn_brute_force.mojo` has no
   `DistanceEpilogue` parameter, and that single gap is the whole of
   DEVIATION 1600's wall (b). If the neighbors lane threads an epilogue
   through `tiled_brute_force_knn` -- their template parameter, so it is
   a port and not an invention -- rung 2's sparse graph becomes
   reachable. This lane did not touch that file.
8. **A change `hierarchy/` may want, NOT MADE.** Wall (a) is
   `connect_knn_graph` / `cross_component_nn` / `merge_msts`, which that
   lane records as NOT PORTED. Its host overload picks one RANDOM vertex
   per component with `std::mt19937(std::random_device())`
   (`mst.cuh:167-190`) and must be pinned when ported -- the lowest
   vertex index per component is the obvious pin, and it is already named
   in `hierarchy/NOT_IMPLEMENTED.tsv`. Separately, `hierarchy`'s
   `sorted_coo_to_csr`-shaped counting scan now exists in three places
   (`spectral/impl/sparse/op/coo_ops.mojo`,
   `hdbscan/impl/hdbscan/detail/utils.mojo`, and RAFT's original); if
   `core/` ever grows a container-free CSR scan, all of them should call
   it. This lane did not touch `hierarchy/`.
