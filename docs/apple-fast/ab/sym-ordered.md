# lane/apple-fast-sym-ordered: Ordered boosting (gbdt-ordered) under FAST on Apple, round 2

Branch `lane/apple-fast-sym-ordered` (cut from origin/main 8897404da). Board lane `gbdt-ordered`, binding `gbdt`,
datasets taxi (deciding) then istella; M3 Ultra FAST baseline taxi 56.1 s, istella 75.4 s. Builds on the
2026-10-02 lane (`ordered.md`: `ORDERED_BATCH_EST` is the FAST default, `ORDERED_FOLD_DERIVS` dropped). Profile and the
candidates not taken: `docs/apple-fast/notes/sym-ordered.md`.

Every switch lives in `gbdt/methods/ordered_fast_switches.mojo` and is False unless the build is FAST on an Apple GPU
and its define is passed; IDENTICAL compiles main's statements unchanged (ordered IDENTICAL bits are a published lane).
Default OFF. `MOJOLEARN_ORD_ALL` turns on all four. Request lines: `sym-ordered.txt` (taxi for every define first, then
istella; run `all` before the singles on istella).

## `-D MOJOLEARN_ORD_FOLD_BINS_ONE` (`ORD_FOLD_BINS_ONE`)

Mechanism. Every tree's structure search starts by writing the fold layout's initial bins: partition `p` (fold `f`'s
learn prefix at `2f`, its quality slice at `2f + 1`) gets bin `p` over its range of the concatenated document array.
Main does it per partition (`write_fold_based_initial_bins`, `oblivious_tree_fold_tasks.mojo:50-88`): a fresh staging
buffer, a fill, a copy into the bins, and one drain at the end, 2F allocations and 4F launches a tree (F = 14 folds at
1M rows: 28 fresh Metal buffers, 56 launches, 1 drain). The switch uploads the partition starts once a fit
(`fast_part_off`, `ordered_boosting.mojo` CreateState) and the pooled tree writes every bin in one launch
(`fold_bins_from_table_kernel`: position -> last start at or below it, by binary search over 2F + 1 entries,
`oblivious_tree_doc_parallel_structure_searcher.mojo`). The first tree (the pool's construction) keeps main's path.
Expected effect: fewer launches, no per-tree allocations before the ~150 launches of the level loop (on Apple every
live buffer taxes every launch), one drain fewer a tree. Bits: the same bins, so none from this switch alone.
Risk: none to quality; an empty partition shares its start with the next and is never chosen (the search takes the
LAST start at or below the position).

## `-D MOJOLEARN_ORD_STD_PARALLEL` (`ORD_STD_PARALLEL`)

Mechanism. The score-noise sum over the quality slices and the fixed-point scale's two magnitudes
(`sum |w|`, `sum |g|`) are one fixed-order fold a tree: `_ord_std_lanes_kernel` runs 8 blocks of 32 lanes, each lane
walking every 256th of the ~2n concatenated positions (`ordered_boosting.mojo`, step 3), then `_ord_std_combine_kernel`
folds the 256 lane sums. On the L40S that kernel was the tree's largest (13 ms a tree at 4.1M rows, 12% of the fit);
on the M3 Ultra eight threadgroups occupy a fraction of the GPU. The switch replaces the lanes kernel by
`_ord_std_wide_kernel`: 256 blocks x 256 threads (65,536 lanes, four strides' loads in flight per thread), each
block folded by the library block sum, the same combine kernel over the block sums. It also drops the per-tree
`getenv("MOJOLEARN_ORD_STD_SPLIT")` (a host step a tree). Expected effect: the score-std stage shrinks to a memory-bound
pass over the two planes. Bits: FAST bits change (a different fold order of the same terms); the identity column is
untouched. Risk: none to quality beyond float32 summation order; the drain that reads the three sums stays (the scale
and the std are host scalars the searcher takes as arguments).

## `-D MOJOLEARN_ORD_FOLD_INDEX` (`ORD_FOLD_INDEX`)

Mechanism. Main's levels gather the document id of every concatenated position each level
(`launch_gather_with_mask_u32` over ~2n positions, one launch a level) and the histogram and split kernels then read
the compressed index at those ids, a random access per (position, feature). The fold-order index
(`ordered_fold_index()`, lane/neural-pass124, today an env read per tree and OFF by default) gathers the compressed
index into fold order once per learn permutation (`fold_cindex_gather_kernel`, cached in the pool next to the doc ids)
and lets every level read it at the position: no gather launch a level, sequential reads. The switch makes it the
compiled FAST + Apple default (`fold_order = fold_count > 1`, no env read) and shrinks the unused `d_observations` to
one cell. Measured elsewhere: M4, taxi 1M, 10 trees, `pw.hist` 269 -> 197 ms (pr116 notes). Expected effect: the
histogram stage (38% of the fit on the L40S) and the split stage faster; D = 8 launches and one 8 MB allocation fewer
a tree. Bits: the histogram quantizer's dither keys on the row id it is handed (`hist2_dither`), which is now the fold
position rather than the document, so FAST bits change; the dither is a uniform hash either way and the quantization
error it dithers is the same magnitude, so held-out quality should sit inside run-to-run spread. Memory: one
`n_cols x 2n` u32 copy of the compressed index per learn permutation (3), resident for the fit. Risk: quality (check
auc/logloss against the run-to-run spread); memory on small boxes (not the M3 Ultra).

## `-D MOJOLEARN_ORD_TREE_LEAN` (`ORD_TREE_LEAN`)

Mechanism. Main allocates per tree: the two search planes (`sw`, `sg`, ~2n floats each), the score-noise partials,
the tree's bins (n u32) in the fit, and the observation scratch (~2n u32) in the searcher, ~30 MB of fresh Metal
buffers a tree, each a new live buffer for the rest of the tree. The switch allocates them once a fit (`fast_planes`,
`fast_bins`, `fast_obs`), hands the searcher the scratch (`obs_scratch`) and passes the fit's handles where the tree
allocated; every cell is rewritten each tree before it is read, and the in-order queue puts the previous tree's reads
first. It also passes `readback=False` to `_PermPartition.enqueue_sort` when the batched estimation is on: the three
host copies per permutation (12 a tree) that `settle` would read are dead under it. Expected effect: no allocation churn
in the tree loop, fewer live buffers at every launch, 12 fewer blits a tree. Bits: none (the same kernels over the same
values). Risk: none expected; the one thing to watch is a buffer read after a rewrite, which the queue order rules
out.

## `-D MOJOLEARN_ORD_ALL`

All four together; they touch disjoint stages and compose. Taxi first, then istella.

## Not changed

The fold derivatives (2F launches; the one-launch arm `MOJOLEARN_ORDERED_FOLD_DERIVS` measured +1.8% and stays
opt-in), the structure search's level kernels (shared with the Plain fit), the device partition sorts (7 launches per
permutation), the batched estimation's grid, the learn-loss and held-out drains. The brief's candidates 2 (permutations
in one launch) and 5 (derivatives fused into the apply) are declined in the notes file with the reason.
