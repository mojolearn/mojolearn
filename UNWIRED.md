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
| `deterministic_flush` (matrix row) | **WIRED**: `hist_binary.mojo` branches on `deterministic_flush_for[TARGET_COLUMN, ...]` at comptime | Forced true on apple because Metal has no float atomic; follows the mode on nvidia and amd. The multi-block path sums Int32 partials through an integer atomic and converts back in `fixed_to_float_kernel`. Correct and exercised. `hist_replicas` defaults to 1 because 1 and 32 are INDISTINGUISHABLE interleaved at this shape, not because 32 is slower: the earlier 'slower' reading was cross-run noise. |
| `mojo_only/fixed_point.mojo` | its own check, and the flush now uses the same scheme inline | Same reason. It is verified in isolation (overflow bound tight at 268,435,453 of 268,435,455; forward and reverse accumulation exact) and used by no kernel. |
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


## Multi-level tree: RESOLVED

`run_tree` grows a whole oblivious tree correctly. Depth 3 gives 8 leaves all
non-empty; depth 6 gives 64 of which 46 hold rows, which is what 4096 rows
over 64 leaves looks like with real data. Row conservation holds at every
depth.

**The final cause was the TEST DATA, not the port**, after three rounds of
looking like a porting bug. The bins were `((r // (f+1)) + f) % 2`, which
makes features near-duplicates: every leaf came out with the same
distribution, every feature tied on score, the argmax deterministically
re-picked the same one, and re-splitting on an already-used feature produced
empty children. Independent xorshift bins fixed it.

**What resolved it was reading bytes, not reasoning.** A probe built the same
histogram twice, once as one leaf of 1024 rows and once as two leaves of 512,
and got 512 against 256 + 256. Once the histogram was proven to track the
partition, the remaining suspects collapsed to the data. Three inferences
failed; the measurement took one attempt.

**Three real bugs were found while chasing it and all are fixed:**

- only `ComputeSplitPropertiesDirectLoadsImpl` was ported, not the gather
  variant, so below depth 0 the histogram read unrelated rows' bins
- `gather_in_leaves_kernel` was ported, verified, and never called, so the
  stat columns stayed unpermuted while the bins were permuted
- `enqueue_copy(dst_buf=..., src_ptr=device_ptr)` silently did nothing; Mojo
  has no device-to-device form, and the gathered index and stats were never
  written back. Now `ported/gpu_util/copy.mojo`

So the hunt paid even though the final cause was elsewhere.
