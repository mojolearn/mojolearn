# sym-hist: profile of the oblivious (symmetric) per-level split search on Apple FAST

Lane lane/apple-fast-sym-hist (2026-10-03). Read by code, not measured. Board lanes gbdt-symmetric and
gbdt-symmetric-1000 (taxi, istella) run `fit_oblivious_tree_structure_traced`
(gbdt/methods/oblivious_tree_doc_parallel_structure_searcher.mojo) with fold_count 1 (Plain boosting), one
`PolicyScoreHelper` per feature grouping policy present (binary, half-byte, one-byte; istella and taxi are
one-byte dominated at border_count 254), max_depth levels per tree.

## What is already in main (do not redo)

- Sibling subtraction (brief candidate 1): CatBoost's partial pass is in. Depth 0 is a full pass; every later
  level computes only the SMALLER child of each pair (`shift_part_and_bin_sums_ptr`, `compute_histogram`
  n=2 loop) into the right-child slot and `update_pointwise_histograms_kernel` recovers the sibling as
  parent - child (`histograms_helper.mojo`, `pointwise_kernels.mojo:compute_hist2`).
- Score + argmax on the device (brief candidate 3): `find_optimal_split_single_fold_kernel` writes one
  (feature, bin, score, gain) record per 128-thread block; `pw_resolve_pack_bins_kernel` (DEVIATION 3111,
  `PW_FUSED_SEARCH`) folds the records, packs the split descriptor, writes the level's winner record and
  applies the split to the bins in ONE launch. No host read per level: one drain per tree (DEVIATION 207).
- Split apply fused with the winner fold (brief candidate 4): same kernel as above (`bins_done=True`).
- Half-byte packing (brief candidate 5): the grouping policy decides it at dataset build (`blocks_for`);
  features with <= 15 borders already go through `compute_hist2_half_byte`. On Apple FAST every one-byte
  width routes through the single 8-bit fixed-point accumulator (`pointwise_one_byte_fixed_for[APPLE,
  False]` is True), so there is ONE one-byte launch per level, not four.

## Launches per level (fold_count 1, P = 2^depth parts, N rows, F features, BF bin features)

| # | kernel (file) | grid x block | notes |
|---|---|---|---|
| 1 | `compute_split_properties_nb_kernel[8, full, m]` (kernel/pointwise_hist2_one_byte_templ.mojo) | (ceil(F/4) * m, P or P/2, 1) x 256 | m = CatBoost doc multiplier `pw_block_multiplier` (sm_count * 2 * 1.25 blocks wanted, cap 64, >= 10k rows per block); 32 KB threadgroup memory per block (one block per Apple core); m > 1 writes back with global float atomics (FAST only) |
| 2 | `scan_pointwise_histograms_kernel` (kernel/split_properties_helpers.mojo) | (ceil(F/256), P or P/2, 1) x 256 | one THREAD per (feature, part): serial inclusive scan over the feature's folds, both stats |
| 3 | `update_pointwise_histograms_kernel[2]` (pointwise_kernels.mojo) | (ceil(BF/256), P/2, 1) x 256 | partial pass only: sibling = parent - child, elementwise over BF |
| 4 | `find_optimal_split_single_fold_kernel` (kernel/pointwise_scores.mojo) | ceil(BF/128) x 128 | thread per bin feature loops P leaves; block argmin; one record per block (`result_blocks`) |
| 5 | `pw_resolve_pack_bins_kernel` (kernel/pointwise_split_resolve.mojo) | min(ceil(N/256), 65535) x 256 | EVERY THREAD folds all result records of all helpers (`_fold_helper` x3) before its grid-stride bin update; thread (0,0) writes the winner, score_before and the descriptor |
| 6-9 | `scan_key_bit_kernel`, `scan_block_sums_parallel_kernel` (1 block), `add_block_carry_kernel`, `reorder_one_bit_u32_kernel` (gpu_util/kernel/radix_sort.mojo) | ceil(N/512) x 512 | one radix pass over the new bit: bins and indices land in `tmp_*` |
| 10-11 | `copy_u32_kernel` x2 (gpu_util/copy.mojo) | ceil(N/256) x 256 | the ping-pong flip after an odd pass count (always 1 pass here): tmp -> bins, tmp -> indices, 2 x N x 4 B read + write |
| 12-13 | `update_partition_offsets_kernel`, `update_partition_sizes_kernel` (pointwise_optimization_subsets.mojo) | ceil(N/256) x 256 | both grid-stride over sorted bins comparing bins[i] with bins[i-1]; sizes needs offsets written first |
| 14-15 | `gather_with_mask_f32_kernel` x2 (gpu_util/kernel/transform.mojo) | ceil(N/256) x 256 | weights and targets gathered by `indices`; both read `indices` |
| 16 | `partition_update_kernel[1024]` (kernel/pointwise_scores.mojo) | P x 1024 | ONE BLOCK PER PARTITION reduces weight and target over the partition's rows: at depth 0 one threadgroup reads all N rows (8 MB at 1M rows), at depth 1 two |

Plus one of rows 1 to 4 per extra policy present (binary skips the scan). So 16 launches per level at one
policy, ~96 per 6-level tree, none of them waiting. Host waits: none inside the level loop; one
`enqueue_copy` x2 + `synchronize` per tree (`pw.drain`). Device allocations per level: none (the pool
holds every buffer: `PointwiseTreeWorkspace`, `TOptimizationSubsets`, the helper's `d_hist`). Host
readbacks per level: none.

## Where the time can go (by mechanism, not measured)

- Launch count: 16 per level x 6 levels; symmetric-1000 is 1000 trees, so ~100k launches per fit before
  the leaf estimation. The sort's two copies and the two partition-dimension passes are pure launch and
  bandwidth overhead (rows 10 to 13), the two gathers read `indices` twice (rows 14 to 15).
- Redundant work: row 5 folds `result_blocks` records in EVERY thread. istella: ~220 features x ~254 bins
  = ~56k bin features -> ~440 records x 4 loads, per thread, for ~N threads (the grid is N/256 blocks).
- Serial shape: row 16 at shallow depth is one or two threadgroups over all rows.
- Hist occupancy: row 1 wants `sm_count * 2 * 1.25` blocks; with 32 KB threadgroup memory per block Apple
  holds one block per core, so the tail of the last wave is a larger fraction on 80 cores than on an SM
  count the heuristic was written for.

## The defines (all FAST + Apple only, default OFF; see docs/apple-fast/ab/sym-hist.md)

MOJOLEARN_SYM_SORT_SWAP, MOJOLEARN_SYM_RESOLVE_BLOCK, MOJOLEARN_SYM_GATHER_FUSED,
MOJOLEARN_SYM_PART_STATS_PAR, MOJOLEARN_SYM_SCAN_SUB_FUSED, MOJOLEARN_SYM_HIST_MULT, and
MOJOLEARN_SYM_HIST_ALL (all six). Switches live in gbdt/methods/kernel/sym_fast.mojo.
