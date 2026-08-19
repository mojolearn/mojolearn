# Built here, not yet reached

**Two categories exist in this tree and only two:** `ported/` is a port of
a real file of theirs, and `mojo_only/` is something they never needed. There
is no third category of "good idea worth adopting" -- if it is in their
source it is simply PORTED or NOT PORTED YET, and this file tracks the
second.

mojotrees accumulated four fully implemented, tested, documented stages that
no default fit could reach, and it took a day to find them. This file exists
so this tree cannot repeat that quietly. **Audited by grep, not by memory.**

## Wired and driving, with a guard

| row | read by | guard |
|---|---|---|
| `block_size_for` | all four histogram kernel files | `probe_main` RAISES if a kernel's `BLOCK_SIZE` disagrees with the table |
| `hist_floats_per_thread_for` | same | same |
| `reduce_width_for` | `point_hist_half_byte_template` | same |
| `requires_uniform_iteration_for` | `hist_binary`, `hist_half_byte` | kernel refuses at comptime if the row ever says lane-sync |

## NOT wired

| thing | read by | why not, and what would change it |
|---|---|---|
| `deterministic_flush` (matrix row) | **nothing** but a doc line | No kernel does a MULTI-BLOCK flush. One block per (leaf, feature-group) owns its output slot outright, so no atomic is needed and the histograms are correct without it. The row starts mattering the moment replication across blocks exists. |
| `mojo_only/fixed_point.mojo` | **nothing** but its own check | Same reason. It is verified in isolation (overflow bound tight at 268,435,453 of 268,435,455; forward and reverse accumulation exact) and used by no kernel. |
| `column_lane_width` | `spec_for` only | Nothing needs a lane width while `sync_granularity` is `SYNC_BLOCK` everywhere. |
| pinned host partitions (`partsCpu`) | the kernel writes them | `update_partitions_after_split_kernel` takes `host_offset` / `host_size` and writes them, so the device half is ported. The driver must allocate those as PINNED host memory rather than ordinary device buffers, which is where the trick pays: the host then learns every leaf's size with no device-to-host copy, and the next level's smaller-sibling decision needs no readback in the critical path. |

**Consequence to state plainly, because it is the honest answer to "does the
matrix drive the fixed-point path": IT DOES NOT, YET.** Under `FAST` the
design says NVIDIA and AMD keep float atomics and only `IDENTICAL` pins them
to the integer flush, while Apple is forced regardless. None of that is in
effect, because the row has no reader.

## The rule

A row that nothing reads is indistinguishable from a row something reads.
When the multi-block flush lands, `deterministic_flush` must be consulted at
that site and this table must move it upstairs in the same commit.


## Placement audit, 2026-08-19

Andrew asked whether things had been put in `mojo_only/` that belong in
`ported/`. Two had:

- **The stable partition** replaces ONE CALL inside their `split_points.cu`
  (`cub::DeviceRadixSort::SortPairs`, `:658-689`). Splitting it out left the
  reorder incomplete in the file that owns it, so it now lives in
  `ported/.../split_points.mojo` under an explicit DEVIATION BLOCK banner, so
  a reviewer diffing against their file knows exactly where the port stops
  being literal. It is the one place in `ported/` allowed to be better than
  CatBoost, because there is no CatBoost code there to be faithful to.
- **The level driver** is a port of `ComputeOptimalSplits` plus `SplitLeaves`
  (`greedy_search_helper.cpp:398`, `:534`) and was sitting in `mojo_only/`.
  Moved to `ported/methods/greedy_subsets_searcher/greedy_search_helper.mojo`.

What is correctly in `mojo_only/`: `numerics` and `kernel_matrix` (they ship
one vendor and need no columns), `fixed_point` (they use a hardware
instruction Metal lacks), and the harnesses.

**The rule the audit produced:** a replacement for a step of THEIR file
belongs in that file, marked. `mojo_only/` is for what CatBoost never had to
write at all.


## Multi-level tree: runs, conserves rows, does NOT fully split yet

`run_tree` grows a whole oblivious tree and the row-conservation invariant
holds at every depth: 4096 rows across 2, 8 and 64 partitions, none negative,
exactly `2^depth` leaves. But only the first two levels split for real.
Depth 6 produces 64 partitions of which **4 hold rows**.

Two causes found and fixed, one remains:

- **FIXED: the wrong histogram variant.** CatBoost has
  `ComputeSplitPropertiesDirectLoadsImpl` AND
  `ComputeSplitPropertiesGatherImpl`; only the direct one was ported. Direct
  reads `bins[position]`, gather reads `cindex[indices[position]]`. The
  compressed index is never permuted, so position stops naming a row after
  the first reorder. Took depth 3 from 2 to 4 non-empty.
- **FIXED: the stat gather was ported and never called.** The histogram reads
  bins THROUGH the index and stats BY POSITION, so leaving the stat columns
  unpermuted pairs the right rows' bins with the wrong rows' gradients.
- **OPEN: something still stops splitting after depth 1.** Not yet isolated.

**Why this matters more than the bug does:** every isolated kernel check
passed, the end-to-end depth-0 level passed against a host calculation, and
the row-conservation invariant passed at every depth. The only thing that
caught it was counting how many leaves actually hold rows. A tree that
conserves every row and splits nothing looks exactly like healthy plumbing.
