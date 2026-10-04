# Ordered boosting (gbdt-ordered) on the Apple GPU: the per-tree profile

Lane af-sym-ordered (branch `lane/apple-fast-sym-ordered`, 2026-10-03). Read from the code at main 8897404da;
nothing here was timed by this lane. Board lane `gbdt-ordered`: SymmetricTree, `boosting_type='Ordered'`, 500 trees,
depth 8, learning_rate 0.1, border_count 254, `bootstrap_type='No'`, random_strength 1.0, 4 permutations (3 learn +
1 estimation), fold growth 2, min_fold_size 100. M3 Ultra FAST: taxi 56.1 s (CatBoost CPU 99.3 s), istella 75.4 s;
plain symmetric on the same taxi data 11.25 s, so an Ordered tree costs ~5x a plain one (~112 ms vs ~22 ms a tree).

Notation: `n` rows, `F` folds (`int_log2(ceil(n / 100))`, 14 at 1M rows), `P = 4` permutations, `L = 3` learn
permutations, `total` = concatenated fold layout = sum of the fold right edges ~ `2n` (fold `f` is `[off_f, off_f +
R_f)`, learn prefix first, quality slice after), `doc_count = total`, `D = 8` levels, leaves 256.

## What main does per tree (FAST + Apple default: ORDERED_BATCH_EST on, FOLD_DERIVS off)

| step | launches | drains | host copies | allocations (Metal buffers) |
|---|---|---|---|---|
| 1. learn permutation, tree seed | host RNG | 0 | 0 | 0 |
| 2. fold derivatives, `ordered_boosting.mojo:2254-2300` | `2F` (`launch_approximate` + `_ord_scatter_planes_kernel` per fold) | 0 | 0 | 2 (`sw`, `sg`, `total` floats each, ~8 MB at 1M rows) |
| 3. score std + magnitudes (`fused_sums`), `:2318-2355` | 2 (`_ord_std_lanes_kernel`: 8 blocks x 32 lanes; `_ord_std_combine_kernel`) | **1** (`h_sums`) | 1 | 1 (`part`) + a `getenv("MOJOLEARN_ORD_STD_SPLIT")` per tree |
| 4. bootstrap | 0 on the board | 0 | 0 | 0 |
| scale, snap, `:2470-2478` | 1 (`enqueue_snap_plane`) | 0 | 0 | 0 |
| 5. structure, pooled path (`oblivious_tree_doc_parallel_structure_searcher.mojo:454-466`) | `reset_subsets` (~3), `write_fold_based_initial_bins` (**fill + copy per partition = 4F**), `update_subsets_stats` (~2) | **1** (inside `write_fold_based_initial_bins`, `oblivious_tree_fold_tasks.mojo:87`) | 0 | **2F staging buffers** + `d_observations` (`doc_count` u32) + `d_fold_cindex` (1 u32) |
| 5. levels x D, `:636-820` | per level: 1 doc-id gather (`launch_gather_with_mask_u32`, skipped under fold order), `compute_hist2` (per policy helper, ~1-3), score (1), fused resolve+pack+bins (1), `split_subsets_from_desc` (~3-4) | 0 | 0 | 0 |
| 5. tree drain, `:826-834` | 0 | **1** (winner ids + scores) | 2 | 0 |
| bins, `:2495-2512` | `compute_bins_for_model` (~1) + `P` x 7 (`_PermPartition.enqueue_sort`) | 0 | **3P dead readbacks** (`h_seg/h_tot/h_snap`, never read under `fast_on`) | 1 (`bins`, `n` u32) |
| 6. batched estimation, `:1855-1924` | `L + 1` reduce, 1 leaves, `L + 1` apply | 0 | 1 (leaves, read after step 7's drain) | 0 |
| 7. learn loss, `:2596-2610` | 2 | **1** | 1 | 0 |
| 8. held-out cursor + loss | ~3 | 1 | 1 | 0 |

At F = 14, P = 4, D = 8: ~28 + 2 + 1 + 5 + 56 + ~90 + 29 + 9 + 2 + 3 = **~225 launches**, **5 drains**, ~20 host
copies (12 dead), **~33 fresh Metal allocations (~30 MB) per tree**, two env reads per tree (`MOJOLEARN_ORD_STD_SPLIT`,
`MOJOLEARN_ORDERED_FOLD_INDEX`). On Apple every launch is encoded against every live buffer and every allocation is a
new live buffer for the rest of the tree, so the 28 staging buffers of the fold bins tax the ~150 launches that
follow them within the tree.

Measured elsewhere (not Apple): L40S stage times, main 0.8.34, taxi 4.1M rows, 20 trees, 2098 ms: `pw.hist` 797 ms
(38%), `ord.score_std` 261 ms (12%, the 8-block lanes sum: 13 ms a tree), `pw.split` 247 ms (12%),
`ord.fold_estimates` 209 ms (10%, before BATCH_EST). M4, taxi 1M, 10 trees: the fold-order index took `pw.hist` 269
-> 197 ms (`~/mojolearn-evidence/runs/handoff-2026-10-02/scratchpad/pr116.md`).

## Where the fold / permutation structure is a host loop of launches

- Step 2: `for f in range(n_folds)`: two launches per fold (prefix `[0, R_f)` of the learn permutation). The one-launch
  form exists (`MOJOLEARN_ORDERED_FOLD_DERIVS`, opt-in) and measured +1.8% on the M3; not retried here.
- Step 5 setup: `write_fold_based_initial_bins`: `for p in range(2F)`: allocate, fill, copy. The histogram pass itself
  is NOT a per-fold loop: CatBoost's feature-parallel searcher keys the fold id into the low bits of every document's
  bin, so `compute_hist2` and the dynamic scorer already cover every fold of the permutation in one launch set per
  level (brief candidate 1 is already main's design).
- Bins: `for p in range(perm_count)`: 7 launches + 3 readbacks per permutation (`_PermPartition.enqueue_sort`).
- Step 6: `for lp in range(learn_count)`: one reduce and one apply per learn permutation (tasks of one permutation are
  already batched by BATCH_EST).

## The experiments (each its own define, see `docs/apple-fast/ab/sym-ordered.md`)

| define | removes per tree | adds per tree |
|---|---|---|
| `MOJOLEARN_ORD_FOLD_BINS_ONE` | 4F launches, 2F allocations, 1 drain (step 5 setup) | 1 launch (`fold_bins_from_table_kernel`) |
| `MOJOLEARN_ORD_STD_PARALLEL` | the 8-block lanes sum (256 lanes over `total` positions), 1 env read | a 256-block x 256-thread reduce (same combine) |
| `MOJOLEARN_ORD_FOLD_INDEX` | D doc-id gathers, random-access compressed-index reads, `d_observations`, 1 env read | a one-time fold-order gather of the compressed index per learn permutation (cached in the pool) |
| `MOJOLEARN_ORD_TREE_LEAN` | 5 allocations (~30 MB), 3P dead readbacks | 0 |
| `MOJOLEARN_ORD_ALL` | all of the above | |

Not done, and why: batching the permutations into one reduce / apply / sort launch (brief candidate 2) needs either
every permutation's cursor, target, weight and partition buffers contiguous in one parent (the partitions are arena
views with no exposed offsets) or ~30 pointer arguments on one kernel (Metal binds at most 31 buffers); fusing the
next tree's fold derivatives into this tree's apply (candidate 5) needs the next learn permutation drawn a tree early
and the planes allocated ahead, for at most the 2F launch overheads that the FOLD_DERIVS arm already showed are not
the cost; deferring the score-std / scale drain needs `choose_scale` and the std on the device through the searcher's
and calcer's scalar arguments (a signature change across shared Plain code).
