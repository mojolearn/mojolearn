# lane/apple-fast-hdbscan2: HDBSCAN under FAST on Apple (board lane hdbscan, AFC_FAMILY=classical, taxi and istella)

Six independent build defines plus one that turns them all on, each compiled only under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()` (`hdbscan/impl/detail/fast_apple.mojo`),
default OFF; IDENTICAL, every build off Apple and a FAST Apple build without the define compile main's code
unchanged. No environment read was added. The profile the defines answer is `docs/apple-fast/notes/hdbscan.md`.
Request lines: `docs/apple-fast/ab/hdbscan2.txt` (taxi lines first; a define that cannot change a dataset's
route has no line for it, see "where it applies").

Which route each dataset takes on main (m = 100,000, k = 11): taxi (d = 11) takes the matrix-unit k-NN and
hierarchy's host-driven FAST Boruvka (`fast_euclidean_mst`); istella (d = 220) takes the chunked matrix-unit
k-NN and the device sparse MR Boruvka (`sparse_mr_mst_device`) with the per-pair search kernel under a 2^30-MAC
launch bound and a wait after every launch.

| define | where it applies | site |
|---|---|---|
| `-D MOJOLEARN_HDB_SMR_TILED` | istella (the sparse arm: d > 64 or m > 46,340) | `hdbscan/impl/cluster/detail/sparse_mr_mst.mojo` `SMR_TILED`, `_search` |
| `-D MOJOLEARN_HDB_CORE_TILE` | taxi (d <= 64 and k <= 16); istella falls back to the k-NN | `hdbscan/impl/detail/core_tile.mojo`, `reachability.mojo::compute_core_dists` |
| `-D MOJOLEARN_HDB_DEV_BORUVKA` | taxi (the d <= 64 FAST arm) | `hdbscan/impl/cluster/detail/fast_mr_mst_device.mojo`, `single_linkage.mojo::build_mr_linkage` |
| `-D MOJOLEARN_HDB_ONE_SYNC` | both | `hdbscan/impl/detail/extract.mojo::extract_clusters`, `runner.mojo::fit_hdbscan`, `tree_device.mojo::td_stage_*` |
| `-D MOJOLEARN_HDB_LINKAGE_DEVICE` | both | `hdbscan/impl/cluster/detail/dendrogram_union.mojo`, `single_linkage.mojo` (dendrogram call), `tree_device.mojo::_condensed_two_reads` |
| `-D MOJOLEARN_HDB_SELECT_DEVICE` | both (epsilon 0) | `hdbscan/impl/detail/extract.mojo::_extract_one_read`, `label_points_sel_kernel` |
| `-D MOJOLEARN_HDBSCAN2_ALL` | both | all six |

## MOJOLEARN_HDB_SMR_TILED (istella)

Mechanism. On main the tiled search kernel `sparse_mr_search_tiled_kernel` (a block owns 64 listed points and
walks 64-point tiles of j, both sides staged 16 features at a time in threadgroup memory, 4 x 4 cells per
thread in registers) is compiled for NVIDIA and AMD only; Apple runs the per-pair kernel, which reads 2d words
per pair with no reuse, under a 2^30 multiply-add launch bound WITH A WAIT AFTER EVERY LAUNCH, because macOS
silently cuts a command buffer that holds the GPU for seconds. On istella round 1 that is 48 columns per
launch: ~2,084 launches, each memset + search + fold + merge + synchronize, for the first phase alone. The
define runs the tiled kernel on Apple under a 2^34-MAC bound (the tiled kernel does 16 FMAs per 8 threadgroup
reads, so a launch of that size is milliseconds) and waits every 8 launches; round 1 becomes ~128 launches and
16 waits. The poison memset and the device fold stay, so a cut launch is still refused by name.

Expected effect. The istella MST is the fit's dominant stage on main (thousands of drained launches in round 1
alone); expect the whole fit to drop by most of that stage. Taxi is unchanged (it never takes the sparse arm).

Bits. The tiled kernel computes the same chain as the per-pair kernel (`ftz(identical_mul_add(...))` in feature
order, then `mr_edge_weight`), and the per-point minimum is over the total order (key, j), so the tree is the
same edge for edge: digests should match main's FAST istella digests exactly.

Risk. The tiled kernel has never been compiled for Metal (2-D block of 16 x 16, eight threadgroup arrays,
~9 KB); the M3 build is the compile check. The 2^34 bound assumes the kernel runs near its NVIDIA efficiency;
if a launch were cut, the poison is refused by name (a failed fit, not a wrong one), and the bound
`SMR_APPLE_TILED_MACS` is the one constant to lower.

## MOJOLEARN_HDB_CORE_TILE (taxi)

Mechanism. HDBSCAN reads one float per row from the k-NN: the k-th smallest distance, self included. Main's
route copies X twice, computes two norm vectors, runs the matrix-unit k-NN into an m x k distance and index set
nobody reads, sorts every row, narrows the indices to Int32 and reads slot k - 1 (eleven buffers, about six
waits). The define runs ONE kernel: each thread keeps its row in registers (d <= 64, zero padded), walks every
row in 64-row threadgroup tiles, accumulates the squared differences in feature order, and keeps the 16 smallest
values seen in a sorted register array (a candidate not below the current k-th costs one compare; an admitted
one, one branchless swap pass). Core distance = sqrt of the k-th slot; a non-finite distance writes NaN so the
existing `refuse_nonfinite_device` refuses it by name. The k-NN buffers shrink to one cell.

Expected effect. Taxi only (istella's d = 220 falls back to the k-NN route). The m^2 d pass is scalar here
against the matrix unit there, so the arithmetic may be slower while the plumbing (two X copies, norms, row
sort, index narrowing, ~5 waits, 10 buffers) is gone; the A/B decides.

Bits. The k-th smallest of a multiset does not depend on which equal element supplied it; the only difference
is the distance arithmetic: direct `(x - y)^2` sums here (the chain the FAST Boruvka search already uses for
pair distances) against the matrix unit's expanded `|x|^2 + |y|^2 - 2 x.y` there. They agree to rounding; a
mutual reachability weight can move by an ulp where rounding breaks a near tie differently. Partition and
quality are unaffected; a digest may differ from main's FAST taxi digest at that level.

Risk. Register pressure at DMAX = 64 (64 row floats + 16 top-k + accumulators) may lower occupancy; taxi runs
the DMAX = 16 instance. Fits gate: 16 KB threadgroup tile, checked at compile time.

## MOJOLEARN_HDB_DEV_BORUVKA (taxi)

Mechanism. Main's d <= 64 FAST arm runs the search on the device (the matrix-unit `MmaBoruvka`, else
`fb_nearest_other_kernel`) but drives the rounds from the host: per round an upload of the component labels,
two m-word readbacks (four when points were deferred), two waits, about eight host passes over m with a host
union-find, and at the end a host sort of the edges and three uploads. The define keeps the SAME search kernels
(same arithmetic, same lowest-index tie rule) and replaces everything around them with the sparse arm's device
round (`sparse_mr_mst.mojo`'s kernels, imported): classify / drop / phase A pick / compaction (the exact
pruning plan of the host code), each component's cheapest edge under (key, lo, hi) by integer minimums,
hooking, pointer jumping and relabel; rank + scatter emit the edges sorted and oriented. Per round three status
words come back; no m-word transfer, no host loop. `best_j` is poisoned before every search so a cut launch is
refused by name (main's route had no such guard).

Expected effect. Taxi only. Removes ~17 x (8 host passes over 100k + union-find + 4 x 400 KB readbacks + 1
upload) and the host sort of 100k edges; adds ~30 small launches per round and one extra status wait per
round. Expect a moderate drop; the search launches themselves are unchanged.

Bits. The edge SET is identical (Boruvka under one strict total order has one answer; the pruning never
changes which edge a component takes). The edge ORDER among exactly equal weights becomes (lo, hi), the dense
and sparse arms' order, where main's FAST route emitted discovery order; equal weights are common in mutual
reachability (a point's edges all read its core distance), so the dendrogram's merge order on such plateaus
can change. The partition and the stabilities on a plateau are the same (an intermediate cluster born and
dying at one lambda has zero excess of mass); the condensed-tree NUMBERING, and so the cluster numbers in
`labels_`, can move on plateaus where main's FAST labels differed from IDENTICAL's. Quality is unaffected.

Risk. Imports private names (`_compact`, `_read_status`) from `sparse_mr_mst.mojo`; the adapter kernel maps
the search's -1 / -2 / -3 markers to the sparse round's. Composes with CORE_TILE (different stages).

## MOJOLEARN_HDB_ONE_SYNC (both)

Mechanism. `extract_clusters` ends with seven downloads and one scalar read, each creating a host buffer,
waiting and copying; the runner then downloads the condensed tree's four arrays and the core distances the
same way: fourteen waits for the outputs. The define stages every copy into a host buffer with no wait
(`td_stage_*`, whole buffers, no sub-buffer views), waits once per site (two in all) and takes the lists.

Expected effect. Twelve fewer waits (~0.2 ms each on Apple plus the per-live-buffer cost at ~50 live
buffers): a few milliseconds on either dataset. Small, safe, composes with everything.

Bits. None: the same device words are read.

Risk. None beyond compile (whole-buffer `enqueue_copy` into a host buffer of `len(buf)` words).

## MOJOLEARN_HDB_LINKAGE_DEVICE (both) -- compile owed: peer

What main already does. The single linkage tree and the condense are ALREADY on the device on main (no host
linkage loop exists any more): both FAST MST arms emit the edges sorted by (weight key, lo, hi), so no sort is
needed; `hierarchy/.../dendrogram_device.mojo::build_dendrogram_device` builds the dendrogram by a divide and
conquer over the edge index with parallel union-find per level; `tree_device.mojo::build_condensed_device`
condenses with min_cluster_size by pointer jumping and radix sorts. What is left is WAITS: the dendrogram's
17 levels each loop (memset flag, hook, jump, 1-word readback, synchronize) until no hook fires (~40 waits at
m = 100,000), and the condense waits eight times (delta refusal, two MST-check reads, n_split, a wait after
the rank, max_cdepth, lambda refusal, closing synchronize).

Mechanism. (1) Dendrogram (`dendrogram_union.mojo`): each level's hook loop becomes ONE launch of a lock-free
union-find, one thread per lower-half edge: find both roots by relaxed atomic loads, CAS the larger root to the
smaller, retry from the roots when another thread's hook won. Roots are only ever hooked under smaller roots, so
the final root of every component is its smallest label, exactly where main's converged min-label hooking ends;
main's jump / top / size / relabel kernels then run unchanged. 7 launches per level, no flag, no readback, no
closing wait. (2) Condense (`_condensed_two_reads`): the same kernels in the same order; the delta refusal, the
MST check and n_split come back in ONE 4-word read (the refusals raised in main's order with main's messages and
the same first index, an integer Atomic.min), and max_cdepth with the lambda refusal in a second 2-word read;
the split-flag kernel bounds-checks children so an invalid MST is refused before anything reads out of range.

Expected effect. ~46 fewer waits per fit on either dataset (each ~0.2 ms plus the per-live-buffer cost),
plus ~30 fewer flag launches. A few to ~15 ms; the A/B decides.

Bits. None: integer work only, the same dendrogram and condensed tree as main's FAST route, so labels and
probabilities are identical.

Risk. The CAS loop is the first lock-free union in hdbscan on Metal (Atomic.compare_exchange on Int32 device
memory is already used on Apple by core/device_mutex.mojo); a compile or pipeline failure shows at the peer's
build. Not compiled here (compile owed: peer).

## MOJOLEARN_HDB_SELECT_DEVICE (both, epsilon 0) -- compile owed: peer

What main already does. Stabilities (fixed blocked fold), EOM (one launch per cluster-tree level), the
deselection (pointer jumping), the label maps (scan), the labels (pointer jumping), the stability scores and the
probabilities all run on the device on main. What is left: a synchronize after the stabilities, after the
negation, after the selection and after the labelling, the selected count read back (it feeds the labelling's
root rule and two output sizes), the stability scores downloaded on their own, then the eight output downloads
(one wait under ONE_SYNC).

Mechanism. `_extract_one_read`: the same kernels in the same order with no wait between them. The selected count
stays on the device (`off[n_clusters]` after the scan); the labelling's root rule (`n_selected == 1 and
allow_single_cluster`) is read there by `label_points_sel_kernel`; the scores land in an n_clusters-long buffer
(the kernel writes only slots < n_selected); every output plus the count comes back in ONE readback (one
synchronize for the whole extract). Leaf selection is supported (one `leaf_kernel` launch). With
cluster_selection_epsilon != 0 main's route runs (the epsilon search decides on the host whether it runs from
the selected count; the board runs epsilon 0).

Why EOM stays level by level. The EOM recurrence is a tree DP whose float sum must keep its order
(0 + stab[k0] + stab[k1], `identical_mul_add`): a scan or pointer-jumping form reassociates the sums and moves
bits, and a single-launch "last child to arrive continues" form needs a device-scope release/acquire between
threadgroups, which Metal does not order (`fence` fails at pipeline creation on Apple, core/device_mutex.mojo).
So the closest correct parallel form is main's: one launch per level, all cluster-tree nodes of a level in
parallel, with no wait between levels. That is what runs here.

Expected effect. ~6 to 13 fewer waits per fit (13 against main, 6 against ONE_SYNC), and one host buffer per
output instead of a staged copy chain. Small, a few ms.

Bits. None: the same device words, so labels, probabilities, stabilities and scores are main's.

Risk. Low. Composes with ONE_SYNC (this path does its own single readback; ONE_SYNC's runner staging still
applies). Not compiled here (compile owed: peer).

## MOJOLEARN_HDBSCAN2_ALL

All six at once (the switches touch different stages: k-NN, MST, dendrogram + condense, extract, outputs). On
taxi: CORE_TILE + DEV_BORUVKA + LINKAGE_DEVICE + SELECT_DEVICE + ONE_SYNC; on istella: SMR_TILED +
LINKAGE_DEVICE + SELECT_DEVICE + ONE_SYNC. Tags `hdbscan2-all-*` are the four-define ALL of head 6631889a3;
`hdbscan2-all6-*` are the six-define ALL (compile owed: peer). Where ALL beats the best single define, the stages compose; where
it does not, read the single lines.

## Compile record (2026-10-03, head 6631889a3, compile only, nothing run)
FAST + `-D MOJOLEARN_HDBSCAN2_ALL` rc=0; FAST + `HDB_CORE_TILE` rc=0; FAST + `HDB_DEV_BORUVKA` rc=0;
FAST + `HDB_ONE_SYNC` rc=0; FAST + `HDB_SMR_TILED` rc=0; FAST, no define, rc=0; IDENTICAL rc=0.
LINKAGE_DEVICE and SELECT_DEVICE (added after cac2b2a53): NOT compiled in this lane (Andrew, 2026-10-03: compile
slots jammed, the M3 peer compiles). Compile owed: peer -- FAST + each new define, FAST ALL, FAST off, IDENTICAL.
