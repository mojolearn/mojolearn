# lane/apple-fast-sym-hist: the oblivious (symmetric) per-level split search, fewer launches and less redundant work

Board lanes gbdt-symmetric and gbdt-symmetric-1000 (taxi, istella). Binding gbdt. Every switch is compiled under
FAST + Apple only (`gbdt/methods/kernel/sym_fast.mojo`) and defaults OFF; IDENTICAL compiles main's code unchanged.
Request lines: `sym-hist.txt` (istella first, then taxi, per switch; one symmetric-1000 istella line for ALL).
Profile of the level pass (16 launches per level, no host wait inside the tree): `docs/apple-fast/notes/sym-hist.md`.

Three of the brief's candidates are already main's code and were not redone: sibling subtraction (CatBoost's
partial pass, `histograms_helper.mojo`), score + argmax on the device with the split applied in the same launch
(`pw_resolve_pack_bins_kernel`, DEVIATION 3111 / 207) and half-byte packing (the grouping policy at dataset
build; Apple FAST already routes every one-byte width through the one 8-bit fixed-point kernel). The switches
below target what remains: launch count, redundant per-thread work, a serial reduce and hist occupancy.

| switch | site | what changes under FAST on Apple |
|---|---|---|
| `-D MOJOLEARN_SYM_SORT_SWAP` | `pointwise_optimization_subsets.mojo::split_subsets_from_desc` | the one-bit radix pass lands in `tmp_bins` / `tmp_indices` and the handles are swapped; the two `copy_u32_kernel` launches per level (and their 2 x N x 4 B read + write) are gone |
| `-D MOJOLEARN_SYM_RESOLVE_BLOCK` | `kernel/pointwise_split_resolve.mojo::_fold_block` | `pw_resolve_pack_bins_kernel` folds the per-block score records once per threadgroup (strided fold, shared-memory tree reduce) instead of once per THREAD |
| `-D MOJOLEARN_SYM_GATHER_FUSED` | `pointwise_optimization_subsets.mojo::gather_pair_sizes_kernel`, `update_subsets_stats` | partition sizes + weight gather + target gather in one launch after the offsets kernel (3 launches become 1; `indices` read once) |
| `-D MOJOLEARN_SYM_PART_STATS_PAR` | `kernel/pointwise_scores.mojo::partition_update_chunked_kernel` | per-partition weight / target sums on a (partitions x chunks) grid with a float atomic per block after a fill, instead of one 1024-thread block per partition |
| `-D MOJOLEARN_SYM_SCAN_SUB_FUSED` | `kernel/split_properties_helpers.mojo::scan_sub_pointwise_histograms_kernel`, `pointwise_kernels.mojo::compute_hist2` | on the partial pass the fold scan and the sibling subtraction are one launch (per non-binary policy) |
| `-D MOJOLEARN_SYM_HIST_MULT` | `pointwise_kernels.mojo::pw_block_multiplier` | CatBoost's document multiplier is asked for 2x the SM count's worth of histogram blocks (cap 64 and the 10k-rows-per-block floor stand) |
| `-D MOJOLEARN_SYM_HIST_ALL` | `sym_fast.mojo` | all six |

## SYM_SORT_SWAP

Mechanism. `launch_radix_sort_bins` over one bit is one `_radix_pass` (4 launches) whose output sits in the temp
pair, followed by two `copy_u32_kernel` launches to move it back (its ping-pong flip). The switch runs the same pass
and swaps the `DeviceBuffer` handles in `TOptimizationSubsets` instead; every consumer takes `subsets.bins` /
`subsets.indices` by name afterwards (the level loop re-reads `subsets.indices.copy()` each level, `reset_subsets`
refills both), and the temp pair is scratch of the same length. Expected: 2 launches and 2 x N x 8 B of traffic
fewer per level (12 launches per 6-level tree; at 1000 trees that is 12k launches). Bits: none, the sorted arrays
are the same values. Risk: none known; the fold arm (`write_fold_based_initial_bins`) writes into whichever handle
`bins` is at reset, which is fine.

## SYM_RESOLVE_BLOCK

Mechanism. `pw_resolve_pack_bins_kernel` runs on an N/256-block grid and every thread folded all score records
(`result_blocks` per helper, ~440 for istella's ~56k bin features) before its row loop: ~440 x 4 loads x N
threads per level. The switch folds once per threadgroup: threads stride the records from the sentinel, then a
shared-memory tree reduce under `_record_less`, which is a strict total order on (gain, feature, bin) where a
tie means identical records, so the winner is the sequential fold's winner. Expected: the level's resolve launch
drops from O(N x records) to O(N + records x blocks) loads; biggest on istella, near zero on taxi (18 records).
Bits: none. Risk: none known; the shared arrays are `SPLIT_BLOCK_SIZE` wide and the launch block is exactly that.

## SYM_GATHER_FUSED

Mechanism. After the sort, `update_subsets_stats` launched offsets, sizes, gather(weights), gather(targets), stats.
The sizes kernel and the two gathers all grid-stride over the N rows; `gather_pair_sizes_kernel` does the three
in one pass (sizes needs offsets, which still run first). Expected: 2 launches fewer per level and one read of
`indices` instead of two. Bits: none (same stores). Risk: none known; the empty-row case keeps main's path.

## SYM_PART_STATS_PAR

Mechanism. `partition_update_kernel` is one 1024-thread block per partition, so at depth 0 ONE threadgroup reads
every row's weight and target (8 MB at 1M rows) and at depth 1 two do. The switch fills `partition_stats` with
zeros, then launches (partitions x chunks) blocks of 256 threads, `chunks` sized so the grid holds ~8 blocks per
SM (at most 64 per partition); each block reduces its chunk with the same `_compute_sum` + `_block_reduce_sum`
and adds its two partials with a global float atomic; chunk 0 writes Count = size. Expected: the stats step
stops being a few-threadgroup serial pass at shallow depths. Bits: FAST only; the atomics make the per-partition
sums order-dependent across blocks (the m > 1 histogram writeback already is on FAST). Quality: the sums feed
the score's `part_weight - weight_left`; same values up to float summation order. Risk: the fill is one more
queue command (a blit, not a kernel), so on taxi the net could be flat.

## SYM_SCAN_SUB_FUSED

Mechanism. On a partial pass each non-binary policy runs the fold scan (one thread per feature and child part)
and then the elementwise sibling subtraction (`parent - child`, which slot by partition size). The thread that
scans a feature's folds owns exactly those cells, so it does the subtraction for them in the same launch, with
the same arithmetic in the same order per cell. Only at `fold_count == 1` over the whole bin-feature line (the
board's Plain boosting); folds or slices keep the two launches. Expected: 1 launch fewer per policy per level
at depth >= 1 and one pass over the histogram instead of two. Bits: none. Risk: a one-hot or single-fold
feature skips the scan but must still be subtracted; the kernel does that (`do_scan` only gates the running sum).

## SYM_HIST_MULT

Mechanism. `pw_block_multiplier` wants `sm_count * 2 * 1.25` histogram blocks and splits the document axis to get
there. Each block takes 32 KB of threadgroup memory, which is one block per Apple core, so the heuristic written
for an SM's two resident blocks underfills Apple's cores and leaves a larger last-wave tail. The switch asks the
same ladder for twice the blocks (cap 64, >= 10k rows per block unchanged). Expected: more, shorter histogram
blocks at shallow depths (where there are few parts); nothing at deep levels where parts already fill the grid.
Bits: FAST only (more document blocks, more float atomic partials per cell). Risk: more atomics per cell; could
lose on istella where the feature axis already supplies ~55 blocks per part.

## SYM_HIST_ALL

All six. They touch disjoint launches of the level pass and compose without interaction; the ALL line on
gbdt-symmetric-1000 istella measures the summed launch-count effect where it matters most (1000 trees).
