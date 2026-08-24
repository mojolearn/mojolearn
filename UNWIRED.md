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
| `gbdt/methods/dynamic_boosting_folds.mojo` and `gbdt/methods/oblivious_tree_fold_tasks.mojo` -- rung 3's fold machinery and the fold layout | `check-dynamic-boosting-folds`, `check-fold-tasks`, and now **`check-ordered-boosting`, which drives them through the subsets, the split and the histogram at `FoldCount = 12`**. | **NO LONGER INERT BELOW THE SEARCHER.** `create_fold_based_subsets` places 1,463 documents into 12 partitions per cell, two splits carry the fold id in the low bits through `FoldBits = 4`, and `compute_hist2` is exact over 12,288 cells at 4 leaves x 12 folds with the partition array strided by 16 and the histogram by 12 -- **the first time a fold axis has run a kernel in this tree**. What is still missing is one caller and four literals: `PolicyScoreHelper` hard-codes `foldCount = 1` at three sites (DEVIATION 126), so no TREE can grow at `fold_count > 1` and the searcher refuses rather than growing one a fold axis short; and the arm is wired into the DOC-PARALLEL searcher, which upstream cannot have folds at all (DEVIATION 127) -- it belongs in `oblivious_tree_structure_searcher.mojo`, which now exists. Above the searcher, `TDynamicBoosting` is unported: per-permutation datasets, the per-fold approx CURSORS, `ComputeWeakTarget`'s fold arm, per-fold leaf estimation and model averaging (`dynamic_boosting.h:230-470`). That last is the bulk of rung 3's 1,353 lines. Until all three, no fit can select `boosting_type=Ordered` and this repository cannot claim parity with CatBoost AS SHIPPED. |
| `gbdt/methods/oblivious_tree_structure_searcher.mojo` and `gbdt/methods/oblivious_tree_bin_builder.mojo` -- rung 2's feature-parallel searcher and the `docBins` chain under it | **NOTHING but its own check** (`check-feature-parallel-identity`). Audited by grep 2026-08-21: the only importer of either outside themselves is that file. | It is CatBoost's `TFeatureParallelObliviousTreeSearcher` at `FoldBits == 0` and one device, gated as an IDENTITY against the doc-parallel searcher at two row counts (`PORTING.md` 120): same splits, same order, and 16,434 per-document leaf ids exact. It is the searcher CatBoost runs for single-target symmetric trees at their DEFAULT `boosting_type=Ordered` (`PORTING.md` 91 F), and it is the only one of the three that can carry folds or tree CTRs -- so rung 4's front half is waiting on it too. **What would change it**: a third value on `doc_parallel_boosting.fit`'s searcher selector -- the arguments already match `fit_oblivious_tree_structure` exactly, and the extra return value (`docBins`) is what the feature-parallel leaves estimator will want, so the wiring is one call site plus a discard. Until then no fit reaches one line of either file. |
| ~~**the whole POINTWISE FAMILY**~~ **WIRED 2026-08-21** -- histograms, host launch layer (`gbdt/methods/pointwise_kernels.mojo`), scorer (`gbdt/methods/kernel/pointwise_scores.mojo`) and the `histograms_helper` state machine. `gbdt/methods/kernel/` minus `exact_estimation` and `linear_solver`: `split_properties_helpers`, `compute_point_hist2_loop`, the four one-byte accumulators, `pointwise_hist2_half_byte_template`, and the three drivers (`..._one_byte_templ`, `..._binary`, `..._half_byte`) | **NOTHING.** Audited by grep 2026-08-21: outside `gbdt/methods/kernel/` itself, the only importers are the seven `mojo_only/pointwise_*_check.mojo` files. No fit reaches one line of it. | It is CatBoost's OTHER histogram family, the one both of their oblivious searchers share and the one they run for single-target symmetric trees (`PORTING.md` 91 F). It is gated hard in isolation -- nine checks, every gate per cell, every sabotage verified to move one -- and that is exactly the state this file exists to make impossible to forget. **What would change it: the SEARCHER.** `oblivious_tree_doc_parallel_structure_searcher` + `doc_parallel_pointwise_oblivious_tree.h` are what call these kernels, and they need `pointwise_hist2.cu` (the dispatcher), `pointwise_scores.cu` + `score_calcers.cuh` (the scorer), `pointwise_kernels.{h,cpp}` (host wrappers), `histograms_helper`, `pointwise_optimization_subsets` and `pointwise_scores_calcer` first -- about 1,017 more kernel lines and 2,184 host lines of theirs. `NEXT_TWO.md` rung 1 is the plan and its order. **RESOLVED the same day.** `gbdt/methods/oblivious_tree_doc_parallel_structure_searcher.mojo` is the caller, through `pointwise_scores_calcer.mojo`, and `pixi run check-pointwise-vs-greedy` grows a tree with it and requires every split to match the greedy-subsets searcher's. The row is kept rather than deleted because the state it describes lasted a day and the entry is what made it visible. **FULLY WIRED the same day**: `doc_parallel_boosting.fit` takes `use_pointwise_searcher`, and `pixi run check-fit-pointwise` fits the same data both ways and requires the loss curves to agree iteration for iteration. The DEFAULT IS FALSE -- not for correctness (the arms are bit-identical over twenty iterations) but because the pointwise arm pays a host round trip per tree in `split_stat_planes`, and flipping a default is a measurement's job. |
| `deterministic_flush` (matrix row) | **WIRED**: `hist_binary.mojo` branches on `deterministic_flush_for[TARGET_COLUMN, ...]` at comptime | Forced true on apple because Metal has no float atomic; follows the mode on nvidia and amd. The multi-block path sums Int32 partials through an integer atomic and converts back in `fixed_to_float_kernel`. Exercised, and correct ONLY WHEN THE CALLER BOUNDS THE SCALE: `doc_parallel_boosting.fit` passed `0.0` for both magnitudes, which makes `choose_scale` return its LARGEST value, and every replicated half-byte flush overflowed Int32 from the moment `launch_histograms_for_blocks` started replicating that kernel. Fixed 2026-08-19; `fit` now reads the stats plane back and passes the two sums of magnitudes. `hist_replicas` defaults to 1 because 1 and 32 are INDISTINGUISHABLE interleaved at this shape, not because 32 is slower: the earlier 'slower' reading was cross-run noise. |
| `mojo_only/fixed_point.mojo` | **WIRED**: `greedy_search_helper.run_tree` and `run_tree_layout` both call `choose_scale`, and `cluster/mojo_only/reduce_by_key.mojo` accumulates through the same scheme | It is also verified in isolation (overflow bound tight at 268,435,453 of 268,435,455; forward and reverse accumulation exact). The tree path used to INLINE the derivation instead of calling it, and the copy drifted: on an all-zero input `choose_scale` returns a scale of 1.0 while the inlined copy returned 268,435,455, the largest scale the type admits. The isolated check passed the whole time, because what was wrong was the wiring and not the unit. |
| `column_lane_width` | `spec_for` only | Nothing needs a lane width while `sync_granularity` is `SYNC_BLOCK` everywhere. |
| the device FeatureFreq calcer: `TWeightedBinFreqCalcerGpu` + `compute_simple_ctrs_device` (`gbdt/ctrs/ctr_calcers.mojo`), over `gpu_util/kernel/partitions.mojo` and `gpu_util/kernel/segmented_reduce.mojo` | `mojo_only/freq_ctr_device_check.mojo` only (`pixi run check-freq-ctr-device`, BIT-equal vs the host driver + independent tally, sabotage-verified) | The port half of PORTING.md 52 is closed; what remains is ONE LINE of wiring in a file this lane does not own: `train()`'s permutation-independent call swaps `compute_simple_ctrs(...)` for `compute_simple_ctrs_device(ctx, ...)` (`gbdt/train.mojo:846` as of 2026-08-22; it was :455 when this row was written, which is why a line number in this file is a hint and not an address). Handoff note in the driver's docstring; delete this row in the commit that lands the line. |
| the category-hash stack: `gbdt/digest/city.mojo` (`city_hash_64`), `gbdt/cat_feature/cat_feature.mojo` (`calc_cat_feature_hash`), `gbdt/models/hash.mojo` (`calc_hash`, `cat_hash_chain_element`) | `mojo_only/cityhash_check.mojo` only (`pixi run check-cityhash`, gated against their own `city.cpp` compiled by `tools/cityhash_oracle/`) | Training needs NO hash -- tree-CTR combination bins are dense-bin reindexing through `TCtrBinBuilder::AddCompressedBins` (`gpu_data/oblivious_tree_bin_builder.cpp:123-186`), and simple CTRs run on dense codes whose labels cannot change any count. The hash becomes load-bearing at exactly two seams, both named in RECON_CTRS.md step 6: tree-CTR tables in a model file (keyed by the `CalcHash` fold over sign-extended category hashes, `ctr_provider.h:94-122`) and raw strings arriving at `train()`/`predict()` without a prep script. Wire it when either lands; until then the check is its only reader. |
| pinned host partitions (`partsCpu`) | the kernel writes them | `update_partitions_after_split_kernel` takes `host_offset` / `host_size` and writes them, so the device half is ported. **MEASURED 2026-08-19: on this stack the trick does NOT pay and the sentence that used to sit here was wrong.** A kernel handed an `enqueue_create_host_buffer` pointer writes NOTHING, silently, all zeros (64 of 64 wrong) -- another silent no-op in the family of `enqueue_copy(dst_buf=, src_ptr=device)`. The working route is `DeviceBuffer.map_to_host()`, which does see kernel output (0 of 64 wrong), but 54 of them cost 18-29 ms against 8-14 ms for 54 `enqueue_copy` + `synchronize`, so it is 2x SLOWER than the copy it was supposed to replace. CatBoost gets this for free because `PartitionsCpu` is `EPtrType::CudaHost` (`split_properties_helper.h:49`, allocated by `cudaHostAlloc` at `cuda_base.cpp:6`) and `TSplitPointsKernel` dereferences it on the host as a plain pointer (`split_points.cpp:56-62`, `split_points.cu:667`). **The gap is REAL on the pin and is CLOSING upstream.** `HostBuffer` does not conform to `DevicePassable`, which is why the kernel writes nothing, and no allocator on `DeviceContext` other than `enqueue_create_buffer`, `create_buffer_sync` and `enqueue_create_host_buffer` exists to try. But MAX **nightly** adds `DeviceBuffer.unsafe_host_ptr()`, which "on devices with unified memory (Apple silicon) returns a CPU-addressable pointer to the buffer, so the host can read a kernel's output after `DeviceContext.synchronize()` without an `enqueue_copy` round trip", and says it "suits small control records rather than bulk readback" -- a partition array being exactly that. It is NOT in the `26.5` reference and NOT in the shipped release notes, so `mojo 1.0.0` / `max 26.5.0` cannot call it. When the pin moves, `hp_off` / `hp_size` become the read side and the per-level `p_sz` copy goes away. Until then, keep the copies, cut their COUNT. Full search and ordering rules in the DEVIATION BLOCK in `gbdt/gpu_util/gpu_data/partitions.mojo`. |

**CORRECTED 2026-08-22. The paragraph that stood here said "does the matrix
drive the fixed-point path: IT DOES NOT, YET ... because the row has no
reader." That was false, and it was contradicted by the WIRED table three
rows above it in this same file.** `deterministic_flush_for` has EIGHTEEN
references across six `gbdt/` files -- `hist_binary`, `hist_half_byte`,
`hist_one_byte`, `hist_2_one_byte_base`, `point_hist_half_byte_template` and
`doc_parallel_boosting`. Under `FAST` the design keeps float atomics on
NVIDIA and AMD and only `IDENTICAL` pins them to the integer flush, while
Apple is forced regardless -- and that IS in effect, at comptime, in every
histogram kernel.

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

**CORRECTED 2026-08-22: THE ROW ABOVE IS FALSE AND IS KEPT ONLY TO RECORD
THAT IT WAS.** `launch_segmented_scan_vector` HAS a production caller and
always did on this side -- `gbdt/methods/leaves_estimation/
leaves_estimation_helper.mojo:36` imports it and `:210` calls it, inside
`compute_weighted_quantile`, the port of their
`ComputeWeightedQuantile` (`leaves_estimation_helper.h:64-146`). It scans
the sorted per-bin weights into `weights_prefix_sum`, which
`compute_weighted_quantile_kernel` then binary-searches for the alpha
quantile.

It is reachable from a real entry point, and not marginally:
`compute_weighted_quantile` <- `compute_exact_approx` <-
`pointwise_oracle` <- `doc_parallel_boosting` under
`leaf_estimation_method == LEAF_ESTIMATION_EXACT`, which is what CATBOOST'S
OWN DEFAULTS select for MAE, MAPE and Quantile. So it does not merely have a
caller; it runs at their defaults for three losses.

What was true is the narrow part: OUR caller is not THEIRS.
`VisitFloatFeatureMeanCtrs` is still unported, so the `FloatTargetMeanValue`
path never reaches it, and the flag-mask argument that caller would supply
has still only ever come from a check. That is a statement about one
argument, not about the entry point.

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


## The control plane: PORTED, ONE FILE UNREACHED

**CORRECTED 2026-08-22. This section was headed "NOT PORTED AT ALL" and
claimed `PORTED_MAP.tsv` has "zero entries from `catboost/cuda/cuda_lib/`".
Both were false.** `PORTED_MAP.tsv` has EIGHT `cuda_lib` rows, and
`gbdt/gpu_lib/` is 2,774 lines: `gpu_manager`, `gpu_single_worker`, `task`,
`tasks_queue/single_host_task_queue`, `gpu_base`, `gpu_profiler`, `slice`,
`device_id`, `fwd` and `mapping`. All but `mapping.mojo` are REACHED from a
real entry point, through `TCudaManager` in `greedy_search_helper.mojo`.

What is genuinely not connected here is narrower and worth keeping:

- `gbdt/gpu_lib/mapping.mojo` (505 lines) -- multi-device mappings. Nothing
  needs a mapping at one device, which is the only case this port targets.
- FIVE of the ten `ECommandType` payloads in `gbdt/gpu_lib/task.mojo` are
  never constructed: `stream_kernel_command`, `allocate_memory_command`,
  `free_memory_command`, `reset_command`, `free_stream_command`.

The performance sentence below stands and is why this section exists at all.
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
| tie handling in `select_radix` | REACHED, and NON-REPRODUCIBLE BY DESIGN **UNDER `NUMERIC_FAST`; CLOSED UNDER `NUMERIC_IDENTICAL` 2026-08-23** | RAFT places every output with `atomicAdd` and has no index tie-break, so which of several equidistant neighbors is returned is not reproducible. The ported file KEEPS that, because fixing a thing upstream does not do is an improvement and improvements live outside `ported/`. The improvement now exists beside it: under IDENTICAL the tiled arm runs `neighbors/mojo_only/select_radix_identical.mojo`, whose 64-bit `(distance, index)` composite key has no tie class at all and whose output slots come from a rank (DEVIATIONS 500/501), and the fused arm's queue is pinned to `grid_x = 1` with the ARM itself pinned to cuVS's dispatch (DEVIATION 502). **The sentence that used to end this row -- "an `IDENTICAL` column cannot cover k-NN indices until it is" -- is no longer true and is deleted.** See IDENTITY_PATHS row 11 and `UNSUPERVISED_IDENTITY.md`. |
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

### `select_warpsort`: RUNS ON DEVICE, AND IS IN NO SHIPPING PATH

`neighbors/ported/matrix/detail/select_warpsort.mojo` is a port of RAFT's
`select_warpsort.cuh`, the family this repository ruled UNTRANSLATABLE on the
claim that Mojo has no warp primitives. That claim was false and the port
exists now.

**THE PARAGRAPHS THAT STOOD HERE SAID "COMPILES, NEVER RUN", "Nothing calls
it" AND "compiles alone, cannot be instantiated". All three are false and are
deleted, not softened.** They were written on 2026-08-19 and the same day's
lane round fixed the two compiler crashes under them and ran the kernel on
device. The record is `bench/results/LANE_warpsort_2026-08-19.md`: two
independent compiler crashes (one of them a bare `while` loop with no GPU,
no SIMD and no parametric anything in it, which also RETIRES this file's old
suspicion of the recursive parametric bitonic network), then

    $ /tmp/warpsort_probe_main.bin
    check_warpsort_matches_radix: OK
    check_warpsort_reach_by_sabotage: OK
    check_warpselect_matches_oracle: OK
    check_warpselect_reach_by_sabotage: OK

four device checks, each proved non-vacuous by a negative control, comparing
`warpsort_topk_block_kernel` three ways against the ported radix select and
an independent Float64 host oracle on a scattered hashed fixture
(`neighbors/mojo_only/warpsort_check.mojo`, driven by
`neighbors/warpsort_probe_main.mojo`). By the four tiers in
`VENDOR_LIBRARIES.md` it is at RUNS ON DEVICE.

**What is still true, and is why the file stays in UNWIRED.** Three things,
each checked against the tree on 2026-08-24 rather than inherited:

1. **No shipping path reaches it.** `knn_brute_force.mojo` imports
   `radix_topk_one_block_kernel`, `radix_topk_identical_kernel` and
   `nn.topk.top_k` and nothing else; its own comment at `:50` still says
   "`select_radix.mojo` is ported and is the selector here". Wiring warpsort
   in must go in BESIDE those rather than replacing either, so that
   `check_vendor_topk_matches_ported` becomes a three-way agreement. That is
   this repo's rule for any new selection implementation and it is the only
   thing that would catch a wrong answer here.
2. **Nothing automated runs the checks.** `pixi.toml` registers no task for
   `neighbors/warpsort_probe_main.mojo`; the run above was a hand build to
   `/tmp`. So the four checks are evidence that the kernel worked once on one
   Apple M4, and they are not a gate: no CI, no `check-*` task, and nothing
   re-runs them when the file or its dependencies change.
3. **No leg has ever carried it.** Phase 8 of `tools/e1_bootstrap.sh` names
   gemm, cd, kde, linkage, svm, mamba and metrics, and every
   `bench/results/e1/*/lanes/` directory holds exactly those. Warpsort has
   never run on an NVIDIA or AMD box.

Why it is worth finishing: RAFT's own dispatch (`select_k-inl.cuh:38`) sends
`2 < k <= 256` to this family and only `k > 256` to radix, so every k a k-NN
user actually asks for goes here. We currently run their second choice across
the whole practical range. The handoff signature for the fused path is in
`bench/results/LANE_warpsort_2026-08-19.md` section 7.

## 2026-08-21: what the loss-breadth round left unreached

**`gbdt/estimator.mojo` -> the CPython extension. RESOLVED 2026-08-22, and
the paragraph that stood here was false in both of its claims.** It said
`mojo build --emit shared-lib` "embeds no compiled Metal code" and that
`mojolearn.GradientBoosting` "is listed as a named absence". Neither was
true: `GradientBoosting` was imported and in `__all__` the whole time, and
the shipped `.so` carried 85 gbdt AIR blobs and ran KMeans GPU kernels
fine. What a user actually hit was a stale artifact -- `TypeError: takes 6
positional arguments but 8 were given` -- because the build could no longer
be replaced.

The cause was NOT the basename lottery running out of tickets, which is what
`2cb82ac` recorded -- **and it was not the module outgrowing a budget either,
which is what this paragraph said next.** The decline it quoted, 85 (shipped)
-> 73 (`2cb82ac~1`) -> 58 (`9ab10bc`) -> 56 (HEAD), is real as a series of
measurements and wrong as an explanation: those counts were read out of a
COMPILER CACHE, not produced by those builds.

The actual cause, measured with the cache cleared before each build:
`MACOSX_DEPLOYMENT_TARGET` set in the environment -- at ANY value -- makes
`mojo build` write an empty 134-byte metallib for every kernel and embed
nothing. `$MODULAR_HOME/cache/.mojo_cache` does not key on the deployment
target, so one such build poisons those keys for every later build whatever
its flags. The "basename lottery" was which names happened to hash to keys
still holding real metallibs from an older configuration; the "continuous
decline" was those surviving entries being invalidated one source change at a
time. Full account and the single-variable table: PORTING.md 70. The fix is to
pass the floor to the linker instead, `-Xlinker -platform_version`, which
keeps `minos 11.0` and every kernel at once.

GBDT still has its own extension, `bindings/_mojolearn_gbdt.mojo` +
`build_gbdt.sh` -- **not for the capacity reason it was commissioned under,
which does not exist, but for the one `_mojolearn_estimators` already had:**
an independently changing binding should not be a merge point. It carries
**141 gbdt blobs, 41 of them greedy_subsets, against 85 and 37 in the last
artifact that ever worked.** RMSE, Logloss, MAE and MultiClass all fit and
predict from Python; `MultiClassOneVsAll` refuses by name, for a reason that
is now about `predict_proba`'s missing sigmoid route rather than about
kernels.

**Per-row weights through `train()`** -- WIRED 2026-08-21. `train` takes
`sample_weight` and multiplies it with `class_weights`, which is their own
combination at pool build (`target/data_providers.cpp:168`). Their
group-weight factor is absent because this port carries no `group_id`.
`sample_weight` of all ones is BIT-IDENTICAL to passing none.

**`FilterZeroEntries` for Bernoulli and Poisson.** DEVIATION 69: the draws
are ported and the zero-weight rows are not filtered out, which is
output-identical and costs the histogram time on rows that contribute
nothing. Their `BootstrapAndFilter` compacts (`gpu_data/bootstrap.h:104-122`)
and returns `isContinuousIndices = false`, which selects a different
histogram arm on their side.

**`min_data_in_leaf`.** Not wired in the searcher, and DEVIATION 69 depends
on it staying that way: it is the one score-side test that counts ROWS
rather than summing WEIGHTS, so wiring it makes the unfiltered bootstrap
stop being output-identical.

**`NumErrors`.** In their target kernel's switch
(`pointwise_targets.cu:497-501`) and deliberately not ported: their own
`Init` (`pointwise_target_impl.h:259-299`) has no case for it, so training
cannot reach it. It is a metric that borrows the target kernel, and this
port has no metric path for it to arrive by.

**`GammaBootstrapImpl`.** In their `bootstrap.cu:21-33` and not ported,
because the only call to it is COMMENTED OUT in their own file (`:82`).

## `IDENTITY_PROFILE` is not written into the serialized model

Added 2026-08-21 with the vendor columns. `kernel_matrix.IDENTITY_PROFILE`
names the version of the bit-identity guarantee: the frozen floor (32 KB
threadgroup, 32 logical replication lanes, block 512, threadgroup Int32
atomics) that every `IDENTICAL` model is produced under. A vendor that cannot
meet the floor is refused by name and runs `FAST`; the floor is never
lowered to fit one, because lowering it changes which partial sums combine
and therefore every model already produced.

**The model file does not record which profile produced it.** So a profile-1
model and a future profile-2 model are distinguishable only by provenance,
which is precisely the thing that goes missing. The whole point of freezing
the floor is that a user can hold two model files and know whether they are
comparable; without the field they have to know when each was trained.

It is a header field and about an hour of work, and the deadline is real
rather than notional: it must land BEFORE a second profile exists, because
after that there are unlabelled models in the field and no way to label them
retroactively. `VENDOR_COLUMNS.md` carries the same warning where a reader
adding a vendor will hit it.


## Three kernel-matrix rows are declared and nothing reads them

Audited by grep 2026-08-21, on the day the vendor columns were added.

`column_max_block_size` is load-bearing: the identity-floor gate and all
three block resolvers call it, and it caught a real defect the day it landed.
These three do not have a reader outside the matrix, its check, and the
report:

| row | who should eventually read it |
|---|---|
| `column_has_dedicated_shared_memory` | whoever sizes replication on a column that answers `False` -- the replication factor is a bet on scratchpad speed and that column loses it |
| `column_spec_guarantees_onchip_shared` | the same decision, for a target with no vendor to ask |
| `column_lane_width_is_fixed` | any kernel that ever indexes by HARDWARE lane. Nothing does today, because `sync_granularity_for` is `SYNC_BLOCK` everywhere and left no reason to |

They are declared facts for a bring-up, not inputs to a decision the code
makes today. Recorded here rather than described as load-bearing, because
this file exists to stop exactly that inflation.

## The identity claim's scale hazard

Found 2026-08-21 by reading the training path to answer "will the identical
column actually be identical across GPUs".

The histogram accumulation is exact integer arithmetic and cannot depend on
arrival order. **The scale that quantizes floats into it can.**
`doc_parallel_boosting` derives `fixed_scale` from a device reduction that
accumulates in Float32 through block sums and a float atomic, and that is
order-dependent by construction.

`choose_scale` snapping to a power of two contains it: a perturbation of a few
parts in 1e6 moves the snapped scale only when `raw` lands that close to a
power-of-two boundary. Order 1e-6 per round, order 1e-4 per hundred-tree fit.
When it does fire, the entire histogram shifts by a factor of two and near-tied
splits can flip.

**Fix before any cross-vendor identity claim is published:** make the magnitude
reduction order-independent -- the same fixed-point accumulator the histogram
already uses, or a deterministic tree reduce with a fixed shape. Then the
probability is zero rather than small, and the construction argument covers the
whole path instead of most of it.

Two siblings, same audit, not fixed either: FMA contraction is a codegen
decision no matrix row can reach, and `compute_scores`' division and `sqrt`
are IEEE-correct only if no backend substitutes a fast approximation. See
`VENDOR_COLUMNS.md`.

## `gbdt/methods/batch_feature_tensor_builder.mojo`

Ported, gated by `pixi run check-feature-tensor`, and **called by nothing**.

`TFeatureTensor` plus `TBatchFeatureTensorBuilder` are the front half of tree
CTRs. The only producer of the `baseTensorIndices` this builder consumes is
`TFeatureTensorTracker` (`gpu_data/oblivious_tree_bin_builder.h:32-90`), which
is driven by `TFeatureParallelObliviousTreeSearcher` -- rung 2 of
`NEXT_TWO.md`, unported. Tree CTRs live nowhere else in their tree.

**Not reachable by flipping a flag.** `TCatFeatureParams.check()` in
`gbdt/options/catboost_options.mojo` raises on `max_ctr_complexity` above 1,
and that guard should come off LAST. What has to exist first, in order:

1. a `TBinarizedFeaturesManager` carrying the tensor -> feature-id map
   (`InverseCtrs`, `binarizations_manager.h:80-100`), which is what makes
   `IsTreeCtr(featureId)` answerable at all;
2. `tree_ctrs_dataset.{h,cpp}` + `tree_ctrs.{h,cpp}` -- the per-device dataset
   cache keyed by this tensor's hash, plus the memory estimator;
3. `ctr_from_tensor_calcer.h`, the real `TFeatureTensorVisitor`;
4. `TFeatureTensorTracker`;
5. the feature-parallel searcher.

Until 1-5 exist the builder is correct and inert, which is the state this file
exists to record. Its hash is a CACHE KEY, so getting it wrong later is a
silent wrong-dataset bug rather than a crash -- hence the 800-tensor collision
sweep in the check.


## The DEPTHWISE lane, 2026-08-22

`gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo` and
the four files under it (`models/non_symmetric_tree.mojo`,
`greedy_subsets_searcher/model_builder.mojo`,
`greedy_subsets_searcher/structure_searcher_options.mojo`, plus the
`ComputeOptimalSplitsRegion` arm of `kernel/compute_scores.mojo` and the
`ComputeNonSymmetricDecisionTreeBinsImpl` arm of
`models/kernel/add_bin_values.mojo`).

**WIRED 2026-08-23 (DEVIATION 259).** This section used to read "reached by
`check-depthwise` and by NOTHING ELSE ... the boosting driver would need: a
policy on the searcher call, a `TNonSymmetricTree` arm on `AppendModels`
(their `add_non_symmetric_tree_doc_parallel.{h,cpp}`, unported), and the
`TAdditiveModel` / model-text writer taught a second tree shape", and that
refusal was the honest state until all three landed in one session:
`doc_parallel_boosting.fit_with_test(grow_policy=, max_leaves=,
min_data_in_leaf=)` dispatches on the policy per tree, `gbdt/models/
add_non_symmetric_tree_doc_parallel.mojo` is their apply, `TAdditiveModel`
carries `non_symmetric_models` (their `std::variant` of the two
instantiations, `train.cpp:436-455`), `model_text` writes `ntree`/`node`
records, and `catboost_options.check()`, `train()` and
`python/mojolearn/ensemble.py` accept the two policies with CatBoost's own
refusals beside them. `pixi run check-grow-policy` is the gate. The paragraph
is kept because the state it describes lasted a day and the entry is what
made it visible.

### Written and NOT written, so the difference is on the record

| written, reached only by the gate | NOT written, recorded instead |
|---|---|
| `TNonSymmetricTreeStructure.visit_bins` (their `VisitBins`) | `GetHash` -- a `THashMap` key this port replaces with the parent LEAF ID |
| `TFlatTreeBuilder`, BOTH duplicate policies | `BuildTreeLikeModel<TObliviousTreeModel>` -- the symmetric lane returns a split list directly and never folds paths back |
| `compute_non_symmetric_decision_tree_bins_kernel` | `BuildTreeLikeModel<TRegionModel>` -- `EGrowPolicy::Region`, no lane |
| `TTreeStructureSearcherOptions.check()` | `TNonSymmetricTree::ShiftLeafValues` / `UpdateWeights` -- `Rescale` and `UpdateLeaves` are the boosting loop's two lines (values replaced by the estimator's, then scaled; 2026-08-23); the estimator's `WriteWeights` is not ported for either tree shape, so the non-symmetric leaf weights stay the searcher's |
| the `sm_count_override` test knob (claim 6) | `SortPath` / `SortUniquePath` -- the Region model builder's |
| | `TTreeStructureSearcherOptions.BootstrapOptions` -- the boosting driver bootstraps one level up in this port, so the field would be a second unread copy |
| | `TTreeStructureSearcherOptions.FixedBinarySplits` -- fed only by `fixed_binary_splits`, an option refused by name |
| | the CPU `GreedyTensorSearchDepthwise` (`private/libs/algo/greedy_tensor_search.cpp:1467`) -- read and cited in `DEPTHWISE.md`, deliberately not ported; the thesis is GPU access and porting their CPU learner would produce a second CPU learner |

### The one thing that IS wired into a shared file

`points_subsets.TLeaf` carried `depth: Int` with the note *"when the model
builder lands, this becomes the path"*. The model builder landed, so it is now
`path: TLeafPath` with a `get_depth()`. The symmetric lane never constructed a
`TLeaf` outside that file (audited by grep), so nothing else moved.
