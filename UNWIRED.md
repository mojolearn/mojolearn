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
| pinned host partitions (`partsCpu`) | the kernel writes them | `update_partitions_after_split_kernel` takes `host_offset` / `host_size` and writes them, so the device half is ported. **MEASURED 2026-08-19: on this stack the trick does NOT pay and the sentence that used to sit here was wrong.** A kernel handed an `enqueue_create_host_buffer` pointer writes NOTHING, silently, all zeros (64 of 64 wrong) -- another silent no-op in the family of `enqueue_copy(dst_buf=, src_ptr=device)`. The working route is `DeviceBuffer.map_to_host()`, which does see kernel output (0 of 64 wrong), but 54 of them cost 18-29 ms against 8-14 ms for 54 `enqueue_copy` + `synchronize`, so it is 2x SLOWER than the copy it was supposed to replace. CatBoost gets this for free because `PartitionsCpu` is `EPtrType::CudaHost` and `TSplitPointsKernel` dereferences it on the host as a plain pointer; we have no equivalent. Keep the copies, cut their COUNT. |

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


## Mixed-width trees: WORKING

`run_tree_layout` grows, splits and conserves every row over a dataset mixing
binary, half-byte and one-byte features:

    depth 4 -> 16 of 16 leaves non-empty
    depth 6 -> 44 of 64 leaves non-empty   (uniform binary gets 46)

For most of its life it grew, conserved every row and produced `2^depth`
partitions while splitting NOTHING: 1 non-empty leaf of 64. Two defects, and
seven real bugs found while chasing them.

### The cause: `TPointHistOneByte::Reduce` is two stages

`hist_one_byte.cu:177-230` reduces in two:

1. fold the per-warp copies at stride `warpHistSize = 1024` into
   `Histogram[1024 + start]`
2. UN-SCRAMBLE. The slot for feature `f` bin `b` is
   `((b & 31) << 5) + 4 * (b >> 5) + f + innerHistStart`, so each feature's
   `8 >> InnerHistBitsCount` sub-copies have to be gathered:
   `sum[i] += src[i + block * blockSize]`, then
   `Histogram[histSize * i + fold] = sum[i]` with `i` the FEATURE

The port had collapsed both into one fold at stride `4 * histSize`, which
sums the wrong set of cells.

### Why every check passed anyway

**A single strided fold returns the RIGHT answer whenever every cell holds
the same value.** `check_one_byte_bits` assigned bins `(r + f) % n_folds`,
which gives consecutive rows consecutive bins and leaves every cell holding
exactly `rows_per_fold`. Summing the wrong set of equal cells still lands on
the right number. The check could not see a permutation.

`check_one_byte_bits` now takes `scattered`, which hashes the bin instead.
Same kernel, same parameters, 2048 rows, 4 features, 64 folds, 6 bits, 2
stats:

    uniform   (r + f) % 64      0 wrong of 512
    scattered hashed          490 wrong of 512

That reproduces the mixed-path failure with no dispatcher, no feature blocks
and no bridge, and fold 0 reads `38.0 / want 41` in both.

### The second defect: the test data

`check_mixed_tree` planted bins as `x % folds[f]`, but `Folds` is the BORDER
count and a feature takes bins `0..Folds`. `% 1` made all 8 binary features
the constant 0, and a split resolved onto a constant feature puts every row
on one side.

### The assertion that was missing

Conservation cannot see a tree that never splits. Sending every row to one
side of every split conserves rows exactly and still produces `2^depth`
partitions, so "rows sum to `n_rows`" and "leaf count is `2^depth`" both
pass. Both tree checks now also require `nonempty >= 2`.


## One-byte in the mixed path: the exclusion trail

Ruled out by measurement, each in one attempt:

1. compressed-index base vs feature-block stride (real bug, fixed)
2. missing `WriteReducesHistograms` (real omission, ported)
3. duplicated block-to-flat bridge (real bug, fixed; took 8 wrong slices to 2)
4. the one-byte bit width (width-matched scores WORSE, so not the cause)
5. `stat_count = 2` (standalone passes with two planes)
6. the row count (standalone passes at 2048 rows)
7. cross-block interference (one-byte ALONE fails)
8. unzeroed per-block scratch (real bug, fixed; not the cause)
9. the bridge itself (PRE-bridge `block_hist` is identically wrong)

Nine exclusions, six of them real bugs. Every one came from a measurement and
none from reasoning; each of the three inferences that "found" something
found a real bug that was not the cause.

**The lesson is about the checks, not the kernel.** Exclusion 5 and exclusion
6 both reported the kernel correct at exactly the parameters that were
failing, because the check's data pattern made the defect invisible. A
histogram check whose expected value is the same in every cell verifies the
total and nothing about placement. Plant scattered bins and compare against a
host tally per cell.


## The control plane: NOT PORTED AT ALL

`PORTED_MAP.tsv` has entries from `gpu_data/`, `cuda_util/`, `methods/` and
`options/`. It has **zero entries from `catboost/cuda/cuda_lib/`**, which is
their entire control plane: 1694 lines across `cuda_manager`,
`gpu_single_worker`, `single_device`, `task`, `stream_section_tasks_launcher`
and the task queues.

We ported the `.cu` files and HAND-WROTE the driver. The driver is the part
that is slow. Measured Aug 19: ~34 of our 50 ms/tree is fixed and
row-independent, and it lives here.

### Three differences, read out of their source

**1. The split descriptor is a by-value kernel argument, not five buffers.**
`split_points.h:76-90`, `TSplitPointsSingleLeafKernel`, holds
`TCFeature Feature; ui32 FeatureBin; ui32 LeafIdToSplit;
ui32 RightLeafIdAfterSplit;` as plain members. Ours broadcasts exactly those
five scalars into five device buffers with five `enqueue_copy` and a
`synchronize`, every level (`greedy_search_helper.mojo:1474-1479`).

**2. One task is many launches on one stream with NO host sync inside.**
`split_points.cpp:232-280`, `TSplitPointsSingleLeafKernel::Run`, issues
`SplitAndMakeSequenceInLeaf`, `SortByFlagsInLeaf`, `CopyLeaf`, `GatherLeaf`
and the partition update back to back on the same `stream`, and blocks
nowhere. Ours syncs after nearly every stage.

**3. They read leaf sizes off a host pointer.** `PartitionsCpu` is
`EPtrType::CudaHost` (`split_properties_helper.h:49`), so
`partitionsCpuPtr[LeafIdToSplit].Size` is a dereference. See the row above
for why this one does NOT transfer to our stack.

Their per-level blocking reads for an oblivious tree are **two**:
`bestProps.Read(propsCpu)` (`greedy_search_helper.cpp:517`) and the pinned
partition read. `ComputeTargetStdDev`'s `ReadReduce` is per TREE, not per
level.

**Closed 2026-08-19 for all three drivers.** Every drain that was pure
ordering is gone, and every remaining one goes through
`TCudaManager::WaitComplete` under a budget that RAISES when exceeded, so a
fourth cannot reappear quietly.

| driver | drains before | drains now | budget |
|---|---|---|---|
| `run_one_level` | 8 | 2 | `2` |
| `run_tree` | 14 | 3 per level + 1 | `3 * max_depth + 1` |
| `run_tree_layout` | 2 per level + 1 stray setup drain | 2 per level + 1 | `2 * max_depth + 1` |

`run_tree_layout` already had its LOOP right. What it still had was a raw
`ctx.synchronize()` in its SETUP, before the manager was constructed, which
the budget therefore never saw. An uncounted drain inside a function that
advertises a budget is worse than a counted one.

`run_tree` is the only driver still above their two, and the third drain is
annotated at its site. It is not theirs: CatBoost's left child keeps the
PARENT's slot, so `UpdatePartitionsAfterSplit` (`split_points.cu:387`) is
reached with no host arithmetic in front of it, while `run_tree` scatters
leaf `i` to slot `2i` and zeroes `2i+1` on the host, and needs `p_off`/`p_sz`
back to do it. `run_tree_layout` in the same file already adopted their slot
convention. The permutation depends on NO host value, so closing it is a
kernel or a slot convention, not a drain. **This is the one OPEN item.**

**One raw `ctx.synchronize()` remains in the file and it stays.** It is in
`upload_blocks`, and it is LIFETIME, not ordering: the per-block host buffers
are locals that die at return while the copies still read them. It runs once
per tree build, not per level, and removing it would be a use-after-free that
no digest would catch.

### What the difference is worth, measured

Trivial kernel, 64 elements, three interleaved trials:

    54 launches, sync after each      7.7 / 8.9 / 9.8 ms
    54 launches, ONE sync at the end  2.1 / 1.2 / 1.2 ms

and correctness holds with the syncs gone: 54 chained increments behind a
single drain came back exact, so **stream ordering is enough and 7 of our 9
per-level syncs are pure ordering we are paying for by hand.**


## `cluster/` (cuVS k-means), added 2026-08-19

**Everything in this section is UNREACHED.** Not "untested": unreached. No
kernel in `cluster/` has been enqueued, and this tree's rule is that a kernel
is not ported until it has been (`PORTING.md 9`). Compiling is not evidence.

| thing | read by | why not, and what would change it |
|---|---|---|
| the whole `fit` path | **REACHED AND PASSING**, `cluster/kmeans_main.mojo` | `row_norms`, `gemm`, `reduce_min`, the fixed-point accumulate, `finalize_centroids`, `centroid_shift`, `finish_sum` and `check_convergence_kernel` all run in `check_kmeans_fit`. Reach proved by two sabotages predicting different movements, not by a digest. |
| `kmeans_plus_plus` and its four sampling kernels | **STILL UNREACHED** | The check inits with `INIT_ARRAY` on purpose, because a check that also depends on the draw cannot say which half failed. `chunk_sums`, `select_chunk`, `select_index_in_chunk`, `gather_rows`, `candidate_cost` and `adopt_candidate_min` have compiled and never run. They need their own check with a planted distance profile whose argmax is known. |
| `init_random` | STILL UNREACHED | Same reason. |
| `use_fused` (`cluster/ported/cluster/detail/kmeans_common.mojo`) | nothing | Ported for the evidence it carries, not for its answer. We have no CUTLASS counterpart, so on every backend this tree targets the answer is False and the unfused path is the only path. It is a row that documents a decision of theirs rather than one that drives ours, and it should stay that way unless a fused kernel ever exists here. |
| `sampling_probability` (k-means\|\| step 3) | nothing | `initScalableKMeansPlusPlus` is NOT PORTED, and it is cuVS's DEFAULT init at `oversampling_factor = 2.0`. `kmeans_fit_main` RAISES on the default rather than substituting classic k-means++, which is the whole point: a substituted initialization that still reports an inertia is exactly the silent deviation this file exists to prevent. |
| `init_size`, `device_buffer_samples` (`KMeansParams`) | nothing | Both belong to cuVS's host-resident arm, which is out of scope here. Carried so the params struct is theirs field for field; they will stay dead unless a host arm appears. |
| `mojo_only/fixed_point.mojo` | **NOW READ**: `cluster/mojo_only/reduce_by_key.mojo` accumulates cluster sums through the same Int32 scheme | Moved upstairs from the NOT WIRED table. Its overflow argument transferred with one noun changed, leaf to cluster, which is the strongest evidence so far that the shared substrate is genuinely shared and not tree-shaped. Still unreached until the check runs. |

**The sabotage the first run has to do**, because a digest cannot tell a
working change from a no-op and this repository has been bitten by exactly
that: corrupt `centroid_norm` before the reduction and watch inertia move. If
inertia does not move, the reduction is not reading it, and the whole
assignment step is doing something other than what it says.

### `cluster/` follow-up, 2026-08-19

| thing | state | why it matters |
|---|---|---|
| `kmeans_plus_plus` host sampling | **FIXED in the same commit that found it** | It used to copy all `n_samples` distances to the host per accepted centroid. Now a two-level device search (`chunk_sums` / `select_chunk` / `select_index_in_chunk`) plus a one-launch `gather_rows_kernel`. No transfer in the fit scales with rows any more. |
| the remaining device-to-host reads | audited, all O(1) or O(candidates) | `h_done` one Int32 per Lloyd iteration, `h_cost` one Float32 per restart, `host_cost` `n_trials` floats per accepted centroid. cuVS reads the last of those too. |
| `check_convergence` (host version) | kept, unused by the fit | The device kernel beside it is what runs. The host one documents the three details in prose and is what a bring-up harness wants. Not dead by accident. |


## Sibling subtraction: WIRED AND WORKING

`run_tree_layout` builds the SMALLER child of each pair and derives the
larger as `parent - smaller`, which is what CatBoost does and what this port
had been skipping. Measured, mixed dataset:

    depth 0   live 1   built 1   derived 0
    depth 1   live 2   built 1   derived 1
    depth 2   live 4   built 2   derived 2
    depth 3   live 8   built 4   derived 4

**Half the histogram work, and the tree is IDENTICAL to the full rebuild**:
16 of 16 non-empty at depth 4, 48 of 64 at depth 6, both matching the
rebuild exactly. That equality is the correctness evidence. Conservation is
not: children summing to the parent is an identity under subtraction and
cannot fail.

It took three attempts because two unrelated bugs sat underneath it, and
both were invisible to every check that existed:

1. the histogram writeback used the LOOKED-UP leaf id where CatBoost uses
   the DENSE `blockIdx.y`, which only matters for a non-identity id list
2. the one-byte kernel was replicated with no atomic to sum the partials,
   which only matters at `replicas > 1`

Neither had anything to do with subtraction. Both were reaching real
datasets.

### Timing: RESOLVED, and the answer is that it does not matter

`bench_subtraction` alternates the two arms inside ONE process. 800k x 100
one-byte, depth 6, five interleaved repeats:

    subtraction ON     39.94 ms median   (39.37 - 41.19)
    subtraction OFF    40.16 ms median   (39.89 - 42.28)

    INDISTINGUISHABLE, ranges overlap
    leaf-size mismatches between arms: 0

**Halving the histogram build does not move the wall clock at this shape.**
That is not a defect in the subtraction; it is a statement about where the
time goes. About 34 of 40 ms is fixed control-plane cost that does not scale
with rows, so halving the part that does scale moves a small share of a small
share.

It is kept ON because CatBoost does it and this is a port, and because at
larger row counts the histogram share grows. But **the honest conclusion is
that kernel-side work is not where the remaining time is**, which is now the
second independent measurement saying so.

The zero mismatch count is the other half of the result. The two arms compute
the tree by different amounts of work and agree leaf for leaf, which is
correctness evidence that conservation could never give.

## Top-bin aliasing at narrow fold counts, OPEN

With 64-fold features the permuted check showed about one bin of rows landing
in bin 0 of every one-byte feature: cells 40, 104, 168, 232, `got` above
`want` by roughly `size / 65`. `Folds` is the BORDER count, so a feature takes
bins `0..Folds`, and at 64 folds the top bin appears to alias onto bin 0. At
the real border count of 254 it does not happen, which is why the shipped
check uses 254. Not explained, and it is a correctness question for any
dataset binned below 8 bits.


### `neighbors/` (cuVS brute force + RAFT radix select), 2026-08-19

| thing | state | note |
|---|---|---|
| `tiled_brute_force_knn` and `radix_topk_one_block_kernel` | **REACHED AND PASSING**, `neighbors/knn_main.mojo` | Truth computed on the host in Float64 with the DIRECT formula, so the GPU's expanded-identity answer is checked against an independent computation. Reach proved by two sabotages predicting different movements. |
| tie handling in `select_radix` | REACHED, and NON-REPRODUCIBLE BY DESIGN | RAFT places every output with `atomicAdd` and has no index tie-break, so which of several equidistant neighbors is returned is not reproducible. Not fixed here: fixing it is an improvement on the upstream. An `IDENTICAL` column cannot cover k-NN indices until it is. |
| index-axis tiling | NOT PORTED | Only the query axis is tiled, so one query tile's distances against the whole index must fit. Splitting the index needs a merge of partial top-k lists, which is where the correctness trap is. |

### `decomposition/` (RAFT PCA), 2026-08-19

| thing | state | note |
|---|---|---|
| `pca_fit`, `pca_transform`, `covariance_kernel`, `column_mean_kernel`, `shift_columns_kernel`, `jacobi_eigh` | **REACHED AND PASSING**, `decomposition/pca_main.mojo` | Reach for the centering path is proved by an INVARIANT rather than a corruption: a +1000 column shift must change nothing, which is impossible unless both the mean and the shift kernels run. |
| `pca_transform` | built, and its output is NOT yet checked | `pca_fit` is checked three ways; `pca_transform` compiles and is called by nothing in the checks. It reuses `core/gemm.mojo`, which is exercised elsewhere, but that is an argument and not evidence. Next check to write. |
| `whiten`, `pca_inverse_transform` | NOT PORTED | see `decomposition/UNPORTED.tsv` |
