# Built here, not yet reached

**Two categories exist in this tree and only two:** `gbdt/` is a port of
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
| `hardware_matrix` occupancy rows (`gpu_cores_for`, `max_threads_per_core_for`, `threadgroup_limit_for`, `smem_statically_partitioned_for`, `smem_per_core_for`, `max_active_blocks_for`) | `pairwise_distance_base` (`TARGET_GPU_CORES`, `max_active_blocks_per_core`, hence `launch_config_generator`), `core/gram_splitk` (`gram_splitk_chunk_count`, `gram_splitk_applies`), and through the first `fused_l2_knn_grid` + the k-NN AUTO default | `check_hardware_matrix` (runs in `knn_main` AND `pca_main`) RAISES if any reader disagrees with the table, pins the Apple column to the pre-keying constants bit-for-bit, and resolves the nvidia/amd columns on the host (structural, NOT hardware validation) |
| `gram_splitk_is_target_arm` | `gram_splitk_applies` (comptime gate: non-Apple targets return False at every shape) | same check: apple must route 32x32x4M to split-K; nvidia/amd columns must answer False, because MAX's own split-K arms are comptime-gated `not has_apple_gpu_accelerator()` (LANE_gram-splitk finding 1) |

## NOT wired

| thing | read by | why not, and what would change it |
|---|---|---|
| `deterministic_flush` (matrix row) | **WIRED**: `hist_binary.mojo` branches on `deterministic_flush_for[TARGET_COLUMN, ...]` at comptime | Forced true on apple because Metal has no float atomic; follows the mode on nvidia and amd. The multi-block path sums Int32 partials through an integer atomic and converts back in `fixed_to_float_kernel`. Exercised, and correct ONLY WHEN THE CALLER BOUNDS THE SCALE: `doc_parallel_boosting.fit` passed `0.0` for both magnitudes, which makes `choose_scale` return its LARGEST value, and every replicated half-byte flush overflowed Int32 from the moment `launch_histograms_for_blocks` started replicating that kernel. Fixed 2026-08-19; `fit` now reads the stats plane back and passes the two sums of magnitudes. `hist_replicas` defaults to 1 because 1 and 32 are INDISTINGUISHABLE interleaved at this shape, not because 32 is slower: the earlier 'slower' reading was cross-run noise. |
| `mojo_only/fixed_point.mojo` | **WIRED**: `greedy_search_helper.run_tree` and `run_tree_layout` both call `choose_scale`, and `cluster/mojo_only/reduce_by_key.mojo` accumulates through the same scheme | It is also verified in isolation (overflow bound tight at 268,435,453 of 268,435,455; forward and reverse accumulation exact). The tree path used to INLINE the derivation instead of calling it, and the copy drifted: on an all-zero input `choose_scale` returns a scale of 1.0 while the inlined copy returned 268,435,455, the largest scale the type admits. The isolated check passed the whole time, because what was wrong was the wiring and not the unit. |
| `column_lane_width` | `spec_for` only | Nothing needs a lane width while `sync_granularity` is `SYNC_BLOCK` everywhere. |
| the device FeatureFreq calcer: `TWeightedBinFreqCalcerGpu` + `compute_simple_ctrs_device` (`gbdt/ctrs/ctr_calcers.mojo`), over `gpu_util/kernel/partitions.mojo` and `gpu_util/kernel/segmented_reduce.mojo` | `mojo_only/freq_ctr_device_check.mojo` only (`pixi run check-freq-ctr-device`, BIT-equal vs the host driver + independent tally, sabotage-verified) | The port half of PORTING.md 52 is closed; what remains is ONE LINE of wiring in a file this lane does not own: `train()`'s permutation-independent call swaps `compute_simple_ctrs(...)` for `compute_simple_ctrs_device(ctx, ...)` (`gbdt/train.mojo:455` at time of writing). Handoff note in the driver's docstring; delete this row in the commit that lands the line. |
| the category-hash stack: `gbdt/digest/city.mojo` (`city_hash_64`), `gbdt/cat_feature/cat_feature.mojo` (`calc_cat_feature_hash`), `gbdt/models/hash.mojo` (`calc_hash`, `cat_hash_chain_element`) | `mojo_only/cityhash_check.mojo` only (`pixi run check-cityhash`, gated against their own `city.cpp` compiled by `tools/cityhash_oracle/`) | Training needs NO hash -- tree-CTR combination bins are dense-bin reindexing through `TCtrBinBuilder::AddCompressedBins` (`gpu_data/oblivious_tree_bin_builder.cpp:123-186`), and simple CTRs run on dense codes whose labels cannot change any count. The hash becomes load-bearing at exactly two seams, both named in RECON_CTRS.md step 6: tree-CTR tables in a model file (keyed by the `CalcHash` fold over sign-extended category hashes, `ctr_provider.h:94-122`) and raw strings arriving at `train()`/`predict()` without a prep script. Wire it when either lands; until then the check is its only reader. |
| pinned host partitions (`partsCpu`) | the kernel writes them | `update_partitions_after_split_kernel` takes `host_offset` / `host_size` and writes them, so the device half is ported. **MEASURED 2026-08-19: on this stack the trick does NOT pay and the sentence that used to sit here was wrong.** A kernel handed an `enqueue_create_host_buffer` pointer writes NOTHING, silently, all zeros (64 of 64 wrong) -- another silent no-op in the family of `enqueue_copy(dst_buf=, src_ptr=device)`. The working route is `DeviceBuffer.map_to_host()`, which does see kernel output (0 of 64 wrong), but 54 of them cost 18-29 ms against 8-14 ms for 54 `enqueue_copy` + `synchronize`, so it is 2x SLOWER than the copy it was supposed to replace. CatBoost gets this for free because `PartitionsCpu` is `EPtrType::CudaHost` (`split_properties_helper.h:49`, allocated by `cudaHostAlloc` at `cuda_base.cpp:6`) and `TSplitPointsKernel` dereferences it on the host as a plain pointer (`split_points.cpp:56-62`, `split_points.cu:667`). **The gap is REAL on the pin and is CLOSING upstream.** `HostBuffer` does not conform to `DevicePassable`, which is why the kernel writes nothing, and no allocator on `DeviceContext` other than `enqueue_create_buffer`, `create_buffer_sync` and `enqueue_create_host_buffer` exists to try. But MAX **nightly** adds `DeviceBuffer.unsafe_host_ptr()`, which "on devices with unified memory (Apple silicon) returns a CPU-addressable pointer to the buffer, so the host can read a kernel's output after `DeviceContext.synchronize()` without an `enqueue_copy` round trip", and says it "suits small control records rather than bulk readback" -- a partition array being exactly that. It is NOT in the `26.5` reference and NOT in the shipped release notes, so `mojo 1.0.0` / `max 26.5.0` cannot call it. When the pin moves, `hp_off` / `hp_size` become the read side and the per-level `p_sz` copy goes away. Until then, keep the copies, cut their COUNT. Full search and ordering rules in the DEVIATION BLOCK in `gbdt/gpu_util/gpu_data/partitions.mojo`. |

**Consequence to state plainly, because it is the honest answer to "does the
matrix drive the fixed-point path": IT DOES NOT, YET.** Under `FAST` the
design says NVIDIA and AMD keep float atomics and only `IDENTICAL` pins them
to the integer flush, while Apple is forced regardless. None of that is in
effect, because the row has no reader.

## The rule

A row that nothing reads is indistinguishable from a row something reads.
When the multi-block flush lands, `deterministic_flush` must be consulted at
that site and this table must move it upstairs in the same commit.


## The CTR block's device primitives: WIRED 2026-08-21, one arm excepted

The entry that stood here said `gbdt/gpu_util/kernel/segmented_scan.mojo`
and `gbdt/gpu_util/kernel/radix_sort.mojo` were ported, gated and reached by
nothing. **Both now have the caller their own file named**, so that entry is
deleted rather than annotated:

- `ReorderBins` is called by `TCtrBinBuilderGpu.add_cat_feature_bins`
  (`gbdt/ctrs/ctr_bins_builder.mojo`), at `first_bit = 0` and
  `last_bit = IntLog2(uniqueValues)`, which is theirs
  (`ctr_bins_builder.h:216-219`).
- `SegmentedScanAndScatterNonNegativeVector` is called twice by
  `THistoryBasedCtrCalcerGpu` (`gbdt/ctrs/ctr_calcers.mojo`), at their
  `ctr_calcers.h:93` and `:137`.
- Both are reached from `train(cat_features=...)` under CatBoost's GPU
  `simple_ctr` default, and `pixi run check-ctr-device` runs that path.

ONE ARM IS STILL UNREACHED, and it is a whole entry point rather than a
detail:

| file | symbol | theirs | what would call it |
|---|---|---|---|
| `gbdt/gpu_util/kernel/segmented_scan.mojo` | `launch_segmented_scan_vector` | `cuda_util/segmented_scan.h:8` -> `segmented_scan.cu:22` | `THistoryBasedCtrCalcer::VisitFloatFeatureMeanCtrs` (`ctr_calcers.h:182`), the ONLY call site of it in all of `catboost/cuda`. That method computes `FloatTargetMeanValue` CTRs, which appear in no default description and have no calcer here (`ctr_calcers.mojo` records `set_float_sample` as unported), so nothing this port can configure reaches it. |

Its shape is exercised only by `pixi run check-segscan`, on both the
inclusive and the exclusive arm, and the flag-mask argument a real caller
would pass has still never been supplied by anything but a check. The
scatter form beside it is now driven by real callers at real sizes, which
is the difference.

Neither primitive has a timing number and neither should be quoted with one.
The radix sort's pass count is `bits` where CUB's is about `bits/5`, and
that is a priced difference, not a measured one.

Also added 2026-08-21 and reached from the first commit:
`gbdt/gpu_util/kernel/transform.mojo` (`GatherWithMask`, `ScatterWithMask`)
and `gbdt/gpu_util/kernel/scan.mojo` (`ScanVector<ui32>`), both called by
`TCtrBinBuilderGpu` and both gated directly by `check-ctr-device` section 0
as well as through it. Section 0 is not redundant: deleting the scan's
device-wide block carry moves 3489 of 4001 cells there and leaves the
builder's own section green, because a SIMPLE ctr feeds `ComputeCurrentBins`
an array whose only end flag is the last one and every block carry is
therefore zero. That half of the scan stays unreached until tree CTRs land.


## Placement audit, 2026-08-19

Andrew asked whether things had been put in `mojo_only/` that belong in
`gbdt/`. Two had:

- **The stable partition** replaces ONE CALL inside their `split_points.cu`
  (`cub::DeviceRadixSort::SortPairs`, `:658-689`). Splitting it out left the
  reorder incomplete in the file that owns it, so it now lives in
  `gbdt/.../split_points.mojo` under an explicit DEVIATION BLOCK banner, so
  a reviewer diffing against their file knows exactly where the port stops
  being literal. It is the one place in `gbdt/` allowed to be better than
  CatBoost, because there is no CatBoost code there to be faithful to.
- **The level driver** is a port of `ComputeOptimalSplits` plus `SplitLeaves`
  (`greedy_search_helper.cpp:398`, `:534`) and was sitting in `mojo_only/`.
  Moved to `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo`.

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
  written back. Now `gbdt/gpu_util/copy.mojo`

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

**When first written, everything in this section was UNREACHED.** Not
"untested": unreached. No kernel in `cluster/` had been enqueued, and this
tree's rule is that a kernel is not ported until it has been (`PORTING.md
9`). Compiling is not evidence. The rows below carry the per-item status as
it changed.

| thing | read by | why not, and what would change it |
|---|---|---|
| the whole `fit` path | **REACHED AND PASSING**, `cluster/kmeans_main.mojo` | `row_norms`, `gemm`, `reduce_min`, the fixed-point accumulate, `finalize_centroids`, `centroid_shift`, `finish_sum` and `check_convergence_kernel` all run in `check_kmeans_fit`. Reach proved by two sabotages predicting different movements, not by a digest. |
| `kmeans_plus_plus` and its sampling kernels | **NOW REACHED AND PASSING** | `check_kmeans_plus_plus_init` runs a full fit through the k-means++ path with `oversampling_factor = 0` and recovers 4/4 centroids as a permutation. `check_device_inclusive_scan` checks the three-stage scan alone against a host scan at 20,000 elements, exact. The other checks still use `INIT_ARRAY` on purpose so a failure cannot hide in the draw; this one exists precisely because that left the whole initialization dark. |
| `init_random` | **NOW REACHED** (2026-08-20) | `check_scalable_supplement_branch` starves the k-means\|\| rounds with `oversampling_factor = 1e-9`, which forces the fewer-than-k supplement arm (`detail/kmeans.cuh:755-777`) and that arm runs `init_random` for the missing centroids. The `init = Random` entry path itself still has no dedicated check. |
| `use_fused` (`cluster/gbdt/cluster/detail/kmeans_common.mojo`) | nothing | Ported for the evidence it carries, not for its answer. We have no CUTLASS counterpart, so on every backend this tree targets the answer is False and the unfused path is the only path. It is a row that documents a decision of theirs rather than one that drives ours, and it should stay that way unless a fused kernel ever exists here. |
| `sampling_probability` (k-means\|\| step 3) | nothing -- documentation row | `initScalableKMeansPlusPlus` is **PORTED AND REACHED** (2026-08-20): the default init runs, and both arms of their `oversampling_factor == 0` selection are checked (`check_scalable_kmeans_plus_plus_init`, `check_kmeans_plus_plus_init`). This Float64 helper itself is still called by nothing: Apple silicon has no device Float64, so the shipped predicate is the Float32 `scalable_keep` in `cluster/mojo_only/scalable_init.mojo`, which the check replays on the host element for element. The helper stays as the readable statement of `SamplingOp` (`kmeans_common.cuh:73-81`). The `== k` copy-out arm of the selection tail (`detail/kmeans.cuh:778-784`) is ported but UNREACHED -- no deterministic fixture pins the candidate count to exactly k. |
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
| `pca_fit`, `pca_transform`, `covariance_kernel`, `column_mean_kernel`, `shift_columns_kernel`, `jacobi_eigh` | **REACHED AND PASSING**, `decomposition/pca_main.mojo` | Reach for the centering path is proved by an INVARIANT rather than a corruption: a +1000 column shift must change nothing, which is impossible unless the mean kernel and the centered read both run (the FUSED `x - mu` tile load on the split-K arm, DEVIATION 42; `shift_columns_kernel` on the fallback arm, where `check_covariance_fused_and_fallback_restore` also proves the center + restore pair by sabotage-shaped sentinels). |
| `pca_transform` | built, and its output is NOT yet checked | `pca_fit` is checked three ways; `pca_transform` compiles and is called by nothing in the checks. It reuses `core/gemm.mojo`, which is exercised elsewhere, but that is an argument and not evidence. Next check to write. |
| `whiten`, `pca_inverse_transform` | NOT PORTED | see `decomposition/UNPORTED.tsv` |


## FIXED: any policy needing more than ONE COLUMN produced an EMPTY histogram

This is the largest correctness bug found in this port, and every check in
the repository passed while it was live.

CatBoost's grid x carries TWO factors multiplied together
(`hist_one_byte.cu:290-291`):

    numBlocks.x  = ceil(fCount / GroupSize);          // feature GROUPS
    numBlocks.x *= CeilDivide(maxActiveBlocks, ...);  // row replication

and the kernels invert it as `maxBlocksPerPart = gridDim.x / featureBlocks`.
The port launched with grid x = REPLICATION ALONE and dropped the
feature-group factor. With one group the two agree. With two or more,
`maxBlocksPerPart` is `1 / 2 == 0`, the kernel divides by zero, and the
histogram comes back ENTIRELY EMPTY rather than partial.

`GroupSize` is how many features share a `UInt32`, so the wall is exactly:

    binary      33 features
    half-byte    9 features
    one-byte     5 features

**Every check in this repository used a single column per policy.** The mixed
check is 8 binary, 4 half-byte, 4 one-byte: one column each. That is why 16
features passed and 17 would not have.

Downstream, an empty histogram makes every score zero, `best_bin` stays at
its `0xFFFFFFFF` sentinel, and the driver's clamp turns that into
bin-feature 0. So the visible symptom was a tree that always split on
bin-feature 0, which is what led here.

### What it invalidates

**Every timing measured before this commit**, including the 1.66x-behind
CatBoost comparison and the "subtraction is indistinguishable" result. The
800k x 100 one-byte benchmark is 25 columns, so it was timing a tree built
from an empty histogram. Corrected numbers are in the commit.

The `mixed_hist_probe` sweep now passes at 32 one-byte, 100 binary, 16
half-byte and a mixed 40/24/20, none of which it could do before.

## The boosting check regressed 0.66 -> 14.79: RESOLVED, the fixed-point scale

**`doc_parallel_boosting.fit` passed `0.0, 0.0` for `total_weight` and
`total_gradient`.** Those are the two numbers the Int32 histogram flush is
bounded by. Zero is not "unknown" to `choose_scale`'s inlined twin in
`greedy_search_helper`: it fell back to `mag = 1.0`, which is a SCALE of
268,435,455, the largest the type admits and therefore the one that
overflows soonest. `Int32(val * fixed_scale)` wrapped on any histogram cell
holding more than about eight rows' worth of weight, and at 8,192 rows over
240 bin-features with four replicated blocks a cell carries about 136.

It had been latent since the flush was written. What made it live was
replication reaching the half-byte kernel: before, `launch_histograms_for_blocks`
launched half-byte at `grid.x = groups`, so `maxBlocksPerPart` was 1,
`activeBlockCount` was 1, and the writeback took the plain float store on
both branches. `boosting_check` is sixteen half-byte features and nothing
else, so it never touched the accumulator. Once the grid became
`groups * replicas` the top two levels of every tree (depth 0 at four active
blocks, depth 1 at two) started scoring a wrapped histogram; depth 2 and
below fall back to one block per partition and were unaffected.

**Why every assertion still passed.** Leaf values come from
`compute_partition_stats`, which never reads the histogram. Only the SPLITS
were corrupt. A tree with arbitrary splits and correct leaf values still
reduces the loss, still reduces it monotonically, and still beats predicting
the mean, so the check's three assertions were all blind to it. It also
explains why swapping Cosine for L2 changed nothing to the last digit: both
calcers were ranking the same garbage.

Fixed by making `fit` read the stats plane back and pass `sum of abs` per
plane, and by deleting both inlined copies of the scale derivation in favour
of `choose_scale`, whose zero-input answer is a scale of 1.0 rather than the
maximum. `boosting_check` now asserts a LEVEL (2% of the mean baseline) and
that most splits land on a feature the target depends on.

**The reusable part.** `probe_main.check_fixed_point` verified `choose_scale`
in isolation and passed throughout. The unit was never wrong; the caller
never called it. A safety derivation that is inlined at its call sites has as
many versions as call sites, and the copies drift toward the unsafe branch
because the unsafe branch is the one nothing exercises.

**The arithmetic, because a second agent bisected to a different conclusion.**
That arm removed the `* replicas` from the half-byte launch and recorded
"our fixed-point Int32 stand-in for that atomic is WRONG in this kernel",
citing the depth-0 weight histogram reading -1.5e-08, -3.0e-08, -4.5e-08,
-6.0e-08 where it should read 550, 1104, 1651, 2237. Those four numbers name
the cause outright:

    -4  / 268435455 = -1.4901e-08
    -8  / 268435455 = -2.9802e-08
    -12 / 268435455 = -4.4703e-08
    -16 / 268435455 = -5.9605e-08

268,435,455 is `fixed_scale` at `mag = 1.0`. A 137-row partial times that
scale is 3.7e10, the conversion saturates at INT32_MAX, and the FOUR active
blocks of an 8,192-row partition sum to 4 * 2147483647 = 8,589,934,588,
which as Int32 is exactly -4; the prefix scan walks it out to -8, -12, -16.
The kernel summed its four partials correctly. The scale it was handed was
the largest the type admits. So `* replicas` is back, `choose_scale` bounds
the scale by construction, and CatBoost's own grid formula
(`hist_half_byte.cu:80-81`) is not given up.

Measured with replication back on and the scale bounded: boosting 0.61 mse,
better than the 0.66 that stood before any of this landed, with the mixed
tree at 16/16 leaves at depth 4 and every slice and permuted-id check at 0
wrong.

`mojo_only/replicated_half_byte_check.mojo` is the check that arm asked for
-- a replicated half-byte histogram against a host tally -- and it is wired
into `probe_main`. Its third arm hands the kernel an unbounded scale ON
PURPOSE and requires the result to move, so the file fails rather than passes
if it ever stops reaching the flush it exists to cover. **That file has not
been run**; it was written after the fix was measured, so it has never seen
the failure it was designed around and its thresholds are derived, not
observed.


## The boosting check's trees always split on bin-feature 0: RESOLVED

Cause was the column bug above. After the fix the boosting check falls from
83.62 to 0.66 mse over twelve trees against a mean baseline of 66.46, where
it used to stall at 57.26, and its trees use 8 of 16 leaves instead of 2.

What follows is the original entry, kept because the reasoning that narrowed
it is the reusable part.


`boosting_check` learns and the model round-trips exactly, but every tree it
grows picks bin-feature 0 (feature 0, bin 0) at every level and at every
boosting iteration, no matter how the residuals move. So the model is a
sequence of stumps on one split and the loss stalls at 57.26 against a
mean-baseline of 66.46: it learns, but far less than it should.

**It is not the argmax.** The mixed-tree path through the SAME
`compute_optimal_splits_kernel` picks varied splits on the same run:

    bf 37 -> feature 11 bin 5
    bf 22 -> feature  9 bin 6
    bf  8 -> feature  8 bin 0
    bf 211 -> feature 14 bin 43

So the reduction, the coverage loop and the skip mask all work. The
difference is the data or the path, and the two candidates are:

1. `boosting_check` uses SIXTEEN half-byte features at 15 folds and nothing
   else, so `hist_cells_per_leaf` is 240 and every feature shares one policy
   block. The mixed check spans three policies.
2. Its weight plane is uniformly 1.0, where the other checks plant varied
   weights. A degenerate weight plane could make the score's denominator
   nearly constant and flatten the comparison.

### Attack it in this order

Run the boosting check's exact dataset through
`mojo_only/mixed_hist_probe.mojo`, which compares each policy's histogram
slice against a host tally. If the slice is right, the histogram is not at
fault and the score kernel gets the same treatment on the same data. Do NOT
reason about which split "should" win; hand-arithmetic on this got the wrong
answer twice already this session.

### What it does NOT invalidate

The stop-on-repeat rule, the model round-trip and the monotone loss are all
real and all measured. A model made of stumps still trains, stores and
replays correctly, which is what those three assert.


## Where the time goes, RE-MEASURED after the column fix

The earlier row sweep was taken on empty histograms and said about 34 of 40
ms was fixed control-plane cost, so kernels were not the bottleneck. **That
conclusion is dead.** With the kernels actually doing work, 800k x 100
one-byte, depth 6:

    100k rows    42.8 ms/tree
    200k rows    50.6
    400k rows    66.6
    800k rows   129.2

Eight times the rows costs 3.0x the time, and the increments are superlinear
at the top: 400k to 800k is 1.94x for 2x the rows. Extrapolating the upper
segment puts fixed cost around 30 ms, which is close to the old figure in
ABSOLUTE terms and completely different in SHARE: it was ~85% of a 40 ms
tree, it is now ~23% of a 129 ms tree.

**So kernel-side work is now the majority of a tree at scale, and the two
earlier measurements saying otherwise were both measuring nothing.** Against
CatBoost's 30.1 ms/tree on the same shape we are about 4.3x behind, and the
gap is in the histogram, not the control plane.

The subtraction is worth 1.67x of that and is already on. What remains
unexamined is whether replication (`replicas_for`) helps now that there is
real work to replicate; every measurement of it so far was taken on an empty
histogram and none of them mean anything.



## THE 4.3x: we load one element where CatBoost loads FOUR

Found by reading their kernel rather than profiling ours, 2026-08-19.

`hist_one_byte.cu:42-49`:

    static constexpr ELoadSize LoadSize() {
        #if __CUDA_ARCH__ < 500
        return ELoadSize::OneElement;
        #else
        return ELoadSize::FourElements;
        #endif
    }

**CatBoost's one-byte histogram uses 4-wide vector loads on any GPU newer
than Maxwell.** Every one of our three histogram kernels runs `LOAD_SIZE = 1`.
The histogram is bandwidth bound, so this is four times the load instructions
for the same bytes, on the exact path that is 4.3x behind.

Their `FourElements` body (`compute_hist_loop_two_stats.cuh:366`):

    uint4  localBins[N];
    float4 localStats1[N];
    float4 localStats2[N];
    for k in N:  localBins[k]   = Ldg((uint4*)  bins,  warpSize * k);
    for k in N:  localStats1[k] = Ldg((float4*) stats, warpSize * k);
                 localStats2[k] = Ldg((float4*)(stats + statsLineSize), ...);
    hist.AddPoints<loadSize * N>(...);

with `N = 4` on `__CUDA_ARCH__ >= 700` (`hist_one_byte.cu:30-37`), so a thread
takes SIXTEEN points per iteration where ours takes two.

### Why our loop can adopt it almost directly

Our striping already matches theirs line for line: `entries_per_warp`,
`global_warp_id`, `stripe_size`, `local_idx = (tid & 31) * LOAD_SIZE`, and
`iter_count`. All of it was ported with `LOAD_SIZE` as a symbol. Raising it
to 4 changes the constant and the load width and nothing structural.

### The one thing that must come with it

`AlignMemoryAccess` (`:57-108`), which our port skipped with the note "at
LOAD_SIZE 1 there is nothing to align". It peels the unaligned HEAD and TAIL
of the partition on block 0, with scalar `AddPoint`, so the main loop is
guaranteed 4-aligned and fully in range. That is what lets their inner loop
carry NO per-element bounds test. Ours tests every element, which is both the
scalar cost and a branch in the hot loop.

Adopt one without the other and the loop reads past the partition.

### Verified portable

A 25-line probe confirms Metal takes `(ptr + i).load[width=4]()` for both
`UInt32` and `Float32`. So `FourElements` is not blocked by the toolchain; it
was simply not ported.

### `select_warpsort`, 2026-08-19: COMPILES, NEVER RUN

`neighbors/gbdt/matrix/detail/select_warpsort.mojo` is a port of RAFT's
`select_warpsort.cuh`, the family this repository ruled UNTRANSLATABLE on the
claim that Mojo has no warp primitives. That claim was false and the port
exists now.

**It compiles and it has never executed.** By the four tiers in
`VENDOR_LIBRARIES.md` it is at COMPILES, not at RUNS ON DEVICE, and those are
different things: `linalg.transpose` compiles too and signals on device
pointers.

Nothing calls it. `knn_brute_force.mojo` still selects between the ported
radix select and `nn.topk.top_k`. Wiring it in is deliberately a separate
step, and it must go in BESIDE those two rather than replacing either, so
that `check_vendor_topk_matches_ported` can be extended to a three-way
agreement. That is this repo's rule for any new selection implementation and
it is the only thing that would catch a wrong answer here.

Why it is worth finishing: RAFT's own dispatch (`select_k-inl.cuh:38`) sends
`2 < k <= 256` to this family and only `k > 256` to radix, so every k a k-NN
user actually asks for goes here. We currently run their second choice across
the whole practical range.

**AND HERE IS WHERE IT STOPS, 2026-08-19.** Wiring it into
`knn_brute_force.mojo` behind a `use_warpsort` flag **crashes the Mojo
compiler**. Not a type error, not a constraint failure: `mojo build` dies and
the crash handler runs.

    UniversalExceptionRaise: (os/kern) failure (5)
    crash_report_exception_handler.cc:257

Isolated by bisection: the file COMPILES on its own
(`mojo build --emit=object` is clean, after 15 mechanical `Self.`-qualifier
fixes). It is instantiating `warpsort_topk_block_kernel[16, True, 8]` at a
launch site that kills the compiler. The wiring is reverted; the port stays.

So warpsort is not at COMPILES either, by the four tiers in
`VENDOR_LIBRARIES.md`. It is at "compiles alone, cannot be instantiated".

The suspect is the recursive parametric bitonic network: `bitonic_merge` and
`bitonic_sort` recurse on a comptime `SIZE` over `mut SIMD` arguments, and a
`@parameter if` that fails to prune the dead branch recurses forever. That
would explain a compiler death rather than a diagnostic. **Unverified** —
narrowing it means bisecting the network with a standalone harness, which is
the next step and is not this session's.
