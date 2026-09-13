# BRIEF: gbdt-feature-freq diverges on every vendor pair (2026-09-13)

DEVIATION 2710. Branch `lane/feature-freq-divergence` off main `321ba4947`.
American English, no em-dashes. Every "identical" below prints its hash.

## 1. The finding

`tools/identity_break.py` lane `gbdt-feature-freq` (`ExperimentalTwoLevelFeatureFreq`,
`python/mojolearn/ensemble.py:1632`, one tree, depth two, `sources=[0, 1]`,
RMSE, `random_state=7`, over `_coded(X)`) is the only lane of 46 whose cells
differ between vendors. Train hashes, 2026-09-13, three boxes, one source
(`bench/results/identity_break/2026-09-13_46-lanes/` on main):

| fixture | Apple M4 (Metal) | NVIDIA H100 | AMD MI325X (gfx942) |
|---|---|---|---|
| base | MOVED (bfc0e037ac48e0f5 / 7d9c56b51213cb42) | 944e48bb | 459e5ba5892267e5 |
| ties | 57ff9964d1af1d4a | 23f66dd4 | 404084f594d869ec |
| hashed | 9d97a55431b6f8e0 | b99acd4c | b99acd4c05cb3998 |
| wide | c6d7fccba2483372 | 317b4b6b | c4fc5f42cfd1bd84 |
| denormal | e7f1da14d1a9794b | 6da3311a | 7530ab136b23266a |
| denormal_ftz | e7f1da14d1a9794b | 6da3311a | 7530ab136b23266a |
| dupes | e8c61a9407503755 | 196c1b4b | 2384b2ef8d37ac05 |
| odd | c481331145bfb998 | 79388cbe | 8ed47117009415f4 |
| negative | 75028913a14bb938 | 802f50a1 | 47777b8ece73b417 |

Apple differs from both other boxes on all nine fixtures, and NVIDIA
differs from AMD on eight of nine, agreeing on `hashed` only. The other five GBDT lanes
(`gbdt-symmetric`, `gbdt-depthwise`, `gbdt-lossguide`, `gbdt-rmse`,
`gbdt-ordered-rmse`) are IDENTICAL x2 across the three boxes on all 45 cells
and all 90 infer and model cells. Within one box the lane is stable, except
one MOVED `base` cell on the M4 during the full 46-lane run under load;
thirteen later runs never reproduced it. No cross-vendor evidence for this
estimator existed before this run. `docs/CROSS_VENDOR_FEATURE_IDENTITY_AUDIT.md:63`
reads "no distinct current3way evidence located" for
`ExperimentalTwoLevelFeatureFreq`, `docs/TREE_ALPHA_FEATURE_STATUS.md:21`
lists it as an experimental bounded route, and there is no
`docs/SUPPORT_MATRIX.md` in this tree (`find . -iname "SUPPORT_MATRIX*"`
returns nothing; the tree-only AMD versus NVIDIA fixtures the
`OrderedRMSE` docstring calls certified are the ordered lane's, not this one's).

## 2. The path, read end to end

Python. `ExperimentalTwoLevelFeatureFreq.fit` (`python/mojolearn/ensemble.py:1656-1737`)
does NOT call `GradientBoosting.fit`. It validates the dense codes on the
host and calls `gbdt_fit_two_level_feature_freq` directly with
`[n_rows, n_features, n_weights, n_sources, learning_rate, l2_leaf_reg, random_state]`.
`numeric_mode` is not a parameter of the class; the tier is the build
define (`bindings/build_gbdt.sh:196-218`, `-D MOJOLEARN_NUMERIC_IDENTICAL=1`
lands the binary in `python/mojolearn/identical/`), so this path runs under
`GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL` like every other lane.

Binding. `bindings/_mojolearn_gbdt.mojo:499-532`
`gbdt_fit_two_level_feature_freq_binding`, one `DeviceContext`, GIL released.

Mojo entry. `gbdt/estimator.mojo:106-235` `gbdt_fit_two_level_feature_freq`:
source columns become one-hot code columns (folds = cardinality), numeric
columns get Uniform-3 borders (`compute_ctr_borders`,
`gbdt/ctrs/ctr_binarization.mojo:119-165`, float64 host arithmetic narrowed
to float32, theirs literally), the FeatureFreq table is built ON THE HOST
(`build_feature_freq_tensor_table`, `gbdt/models/tensor_ctr_value_table.mojo:167-232`,
integer counts over a mixed-radix key), its per-row values are computed ON
THE HOST (`estimator.mojo:212-214`, `value_for_key`
`tensor_ctr_value_table.mojo:113-119`, one Float32 IEEE division
`(count + 0) / (n_rows + 1)`), binarized on the host
(`materialize_tensor_candidate` `:480-504`), and staged as column 10 with a
pinned fold capacity of 3 (`stage_tensor_candidate_host` `:373-417`). Then
`fit_two_level_feature_freq_tree` (`gbdt/methods/doc_parallel_boosting.mojo:443-535`)
uploads rows and two stat planes (weight, weighted target), and calls
`run_sequential_two_level_feature_freq_tree`
(`gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:4439-4540`).

Device driver. That function constructs a `TSynchronizedSymmetricLevelState`
(`greedy_search_helper.mojo:3960`) and runs `run_synchronized_symmetric_level`
(`:4384-4426`) twice, regenerating the tensor column between levels from
the level-one winner (`stage_next_feature_freq_after_winner`,
`tensor_ctr_value_table.mojo:1126-1196`, again host integer work). The level
driver's launches are the baseline's own kernels:
`launch_histograms_for_blocks` (`:4166-4192`), `scan_histograms_kernel`,
`compute_partition_stats`, `compute_optimal_splits_kernel[SCORE_FUNCTION_COSINE]`
(`kernel/compute_scores.mojo:89-222`, `identical_mul_add` and
`identical_sqrt` under IDENTICAL, `:20`, `:46-87`, `:200`),
`enqueue_symmetric_level_winner`, `launch_stable_partition_routed[IDENTICAL]`.
Leaf values are a host loop (`two_level_weighted_leaf_value`,
`doc_parallel_boosting.mojo:424-440`), unit weights, so `1.0 * y` is exact
and no contraction can move a bit.

## 3. The three candidate classes

(a) A kernel-matrix rule that gives Apple a different path. The rows this
driver reads are the baseline's, `HIST2_SMEM_MODE = hist_smem_mode_for[TARGET_COLUMN, IDENTICAL]`
(`kernel/hist_2_one_byte_base.mojo:164-167`, "2-warp-shared Int32 fixed
point on Apple and under NUMERIC_IDENTICAL everywhere"),
`hist2_block_size_for[column, smem_mode]` (`checks/kernel_matrix.mojo:606-614`),
`partition_chunks_sm_for[identical]` (pinned). Printed grep:

    $ grep -n "is_defined\|kernel_matrix\|IDENTICAL" gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo | head
    39:from std.sys.compile import is_defined
    145:from checks.kernel_matrix import (
    326:comptime IDENTICAL_DRAIN_SCHEDULE = HIST_BUILD_MODE == NUMERIC_IDENTICAL
    852:    launch_stable_partition_routed[HIST_BUILD_MODE == NUMERIC_IDENTICAL](

Nothing keyed on the column is specific to this driver, and the same
kernels give `gbdt-symmetric` IDENTICAL x2 on all three vendors. Not (a).

(c) A float path outside `checks/numerics.mojo`. The host side has two
divisions, `value_for_key` (one IEEE Float32 division) and
`uniform_borders` (float64, one multiply, one divide, one add, narrowed;
their `double` arithmetic literally), no sqrt, no log, no libm; the device
score uses `identical_mul_add` and `identical_sqrt`. A divergent float path
would also not explain a lone same-box mover. Not (c).

(b) FOUND. `run_sequential_two_level_feature_freq_tree` constructs the
workspace with a literal `False` for `acc_live`
(main `321ba4947`, `greedy_search_helper.mojo:4478-4481`):

    var state = TSynchronizedSymmetricLevelState(
        ctx, layout.copy(), blocks^, initial_device^,
        n_rows, stat_count, 2, False,
    )

and `run_bounded_synchronized_tensor_tree` does the same (`:4590-4593`).
`TSynchronizedSymmetricLevelState.__init__` forwards it to `TTreeWorkspace`
(`:3994-3997`), which sizes the fixed-point accumulator by it
(`:3487-3489`, "one placeholder cell when no kernel can touch it"):

    self.acc_i32 = ctx.enqueue_create_buffer[DType.int32](
        hist_cells if acc_live else 1
    )

and `initialize_tree` zeroes it only when the flag is set (`:4028-4029`).
The baseline passes the comptime truth instead (`:4733-4746`):

    comptime _ACC_LIVE = acc_i32_is_live[hist2_smem_mode]()
    ...
    ws.append(TTreeWorkspace(ctx, layout, blocks, n_rows, stat_count, max_depth, _ACC_LIVE))

`acc_i32_is_live` (`:3262-3297`) is True whenever the flush is fixed point,
which under `NUMERIC_IDENTICAL` is "everywhere" (`:3270-3271`). The
histogram kernels' accumulator branch is comptime, not keyed on the
workspace flag (`:3282-3287`, in the docstring's own words "the histogram
kernels' accumulator branch is comptime-dead" only when the comptime truth
is False). So under IDENTICAL every one-byte histogram launch of this
driver writes `(leaf, stat, bin)` fixed-point cells, `hist_cells =
max_leaves * stat_count * hist_cells_per_leaf = 4 * 2 * 33 = 264` Int32
cells for the identity_break layout, into a FOUR-BYTE buffer, and
`write_reduces_from_fixed_kernel` (`:3181-3200`, the fused bridge at
`:2784-2830`) reads them back and re-zeroes them. Both the bridge's
correctness argument ("`acc_i32` is all-zero between levels" because "the per-tree
memset at the tree top establishes it", `:5127-5132`) and its bounds are
void on this path.

What lands after the four bytes is the device allocator's business, which
is why three vendors gave three answers, why each box is stable with itself
(same process, same allocation order), and why the M4 could move once under
load (a different neighbour page that one time). The checks that exercise
this state (`checks/tensor_sync_state_check.mojo:21`,
`checks/tree_ctr_slice_check.mojo:406,417`) pass the same literal `False`.

The class is (b), an uninitialized AND undersized accumulator buffer written by
the fixed-point histogram fold; no atomic, no wave-size rule, no fold order.

## 4. Proof on the Mac

The calcer itself is not on the device. Its counts are host integers
(`tensor_ctr_value_table.mojo:213-224`) and its values one host division;
the persisted table never reaches the model on these fixtures because the
tensor column (feature 10) wins neither level on the M4 (the `base` model
text carries `split 0 0 5 1` and `split 0 1 0 0 split_type take_bin`,
features 5 and 0). So the oracle that can judge a side is one of the level
WINNERS, not of the table.

`bench/results/identity_break/2026-09-13_feature_freq_2710/ff_winner_oracle.py`
is a float64 host oracle of both level winners. It mirrors
`compute_optimal_splits_kernel[SCORE_FUNCTION_COSINE]`. The left mass is
the histogram cell (a prefix over folds for a numeric feature after
`scan_histograms_kernel`, the fold's own count for a one-hot feature), the
right mass is the partition total minus it, each leaf adds `sum * mu` to
the score and `weight * mu * mu` to the denominator with
`mu = sum / (weight + lambda)`, the final score is `score / sqrt(denum)`,
ties go to the lowest bin-feature index; the tensor column is rebuilt on
the host exactly as the fit builds it (count/(n+1), Uniform-3 borders in
float64 narrowed to float32, bins = borders strictly below, level two
doubles the key with the level-one winner's split bit). Device versus host
on all nine fixtures (`winner_oracle_after.txt`, binding
`f04fe201c07213ad`, which equals the shipped binding's bytes on every
column, section 5):

| fixture | level 1 device | host argmax | margin | level 2 device | host argmax | margin |
|---|---|---|---|---|---|---|
| base | f5 fold1 | f5 fold1 AGREE | 4.0e+01 | f0 fold0 | f0 fold0 AGREE | 0 (fold 0 and 1 of a binary one-hot are the same partition) |
| ties | f5 fold1 | AGREE | 5.3e+00 | f6 fold1 | AGREE | 3.5e+00 |
| hashed | f5 fold1 | AGREE | 2.6e+01 | f0 fold0 | f0 fold1, 2.8e-14 apart | tie of the same partition; the kernel breaks it to the lowest index |
| wide | f5 fold1 | AGREE | 1.9e+03 | f6 fold1 | AGREE | 2.6e+02 |
| denormal | f5 fold1 | AGREE | 3.9e+01 | f0 fold0 | AGREE | 0 (same partition) |
| denormal_ftz | f5 fold1 | AGREE | 3.9e+01 | f0 fold0 | AGREE | 0 (same partition) |
| dupes | f5 fold1 | AGREE | 1.7e+01 | f6 fold1 | AGREE | 5.4e-01 |
| odd | f5 fold1 | AGREE | 3.7e+01 | f0 fold0 | AGREE | 0 (same partition) |
| negative | f5 fold2 | AGREE | 2.7e+00 | f6 fold2 | AGREE | 3.6e-02 |

18 of 18 device winners are the host argmax or its exact-tie twin; the
smallest real margin is 3.6e-02, far above any fixed-point rounding. On the
M4 the four-byte accumulator's out-of-bounds cells evidently land in
private, zeroed page slack (Metal allocations are page granular; 1056
bytes fit in one page), so its tree is the correct one and its column is
the reference. By the lane's rule, AGREE on the Mac puts the AMD and
NVIDIA sides in the wrong, and their fix needs a leg on those boxes
(section 6).

## 5. The fix, and before/after on the Mac

`tensor_acc_live_for[hist2_smem_mode]()` (`greedy_search_helper.mojo`, next
to `acc_i32_is_live`) returns the baseline's rule, or the literal `False`
under `-D MOJOLEARN_2710_TENSOR_ACC_DEAD=1`; both tensor entries and both
checks call it. Printed grep after the edit:

    $ git diff | grep -n "^[-+].*False\|^[-+].*tensor_acc_live_for"
    -        ctx, layout^, blocks^, cindex^, 8, 2, 2, False
    +        tensor_acc_live_for[HIST2_SMEM_MODE](),
    -        ctx, sync_layout^, sync_blocks^, pinned_device^, 6, 2, 2, False
    +        tensor_acc_live_for[HIST2_SMEM_MODE](),
    -        6, 2, 2, False,
    +        6, 2, 2, tensor_acc_live_for[HIST2_SMEM_MODE](),
    -        n_rows, stat_count, 2, False,
    +        n_rows, stat_count, 2, tensor_acc_live_for[hist2_smem_mode](),
    -        n_rows, stat_count, depth, False,
    +        n_rows, stat_count, depth, tensor_acc_live_for[hist2_smem_mode](),

One IDENTICAL build of `bindings/build_gbdt.sh` on the M4
(`MOJOLEARN_COMPILE_JOBS=1`, `nice -n 19`) took the binding from
`6f5c661559ef8fe5` (the shipped `python/mojolearn/identical/_mojolearn_gbdt.so`
of 2026-09-13 15:45) to `f04fe201c07213ad`.

`gbdt-feature-freq` on the M4, `--repeats 2`, train / infer / model:

| fixture | before | after |
|---|---|---|
| base | 7d9c56b51213cb42 / 95b976d85efc7f2d / b2959e177eb6e088 | 7d9c56b51213cb42 / 95b976d85efc7f2d / b2959e177eb6e088 |
| ties | 57ff9964d1af1d4a / 41371bafcf6397a3 / a7417c31e1b422b4 | same |
| hashed | 9d97a55431b6f8e0 / cf308a4c656bfbdc / bf3222bbcdf1f476 | same |
| wide | c6d7fccba2483372 / 2a4274996f907f13 / 6b7d9eddfa4b79b6 | same |
| denormal | e7f1da14d1a9794b / 973775798c10da25 / d6c30648533bdf45 | same |
| denormal_ftz | e7f1da14d1a9794b / 973775798c10da25 / d6c30648533bdf45 | same |
| dupes | e8c61a9407503755 / 3a9423e4f0d23cad / 531909ad2026b783 | same |
| odd | c481331145bfb998 / 711887d930537c47 / 53a4160101e38573 | same |
| negative | 75028913a14bb938 / 5a5944c8b884be8c / 0794be87633fd36b | same |

`--diff ff_before.json ff_after.json` prints `summary: IDENTICAL=9` and
`summary (infer/model): IDENTICAL=18`. Against the Sep 13 Apple column
(`apple-m4.json`) it prints `IDENTICAL=8, MOVED=1`, the MOVED being that run's own
`base` cell, whose second value is exactly today's `7d9c56b51213cb42`.

So on the Mac the change is INERT BY BYTES, which is what section 4
predicts, since the box that was already computing the right histogram keeps it.
This is therefore not yet a fix by the repo's rule (no before/after that
moves); it is a candidate whose flip is owed on the boxes that were wrong.
Reach on the Mac is by reading, not by bytes. The constructor sizes
`acc_i32` by the flag (`:3487-3489`) and `initialize_tree` zeroes it by the
flag (`:4028-4029`); with the shipped literal, the M4 wrote 264 cells past a
4-byte allocation on every histogram launch and happened to get away with it.

The five other GBDT lanes on the after binding print `cells=45 stable=45
moved=0 refused=0`, `infer: stable=45`, `model: stable=45`, and
`--diff apple-m4.json gbdt5_after.json` prints `summary: IDENTICAL=45` and
`summary (infer/model): IDENTICAL=90` on those lanes (the report's one
MOVED row is the reference JSON's own `gbdt-feature-freq/base`, not run on
this side). No other lane's bytes moved.

The evidence sits in `bench/results/identity_break/2026-09-13_feature_freq_2710/`
(untracked, beside the other identity results) as `ff_before.json`,
`ff_after.json`, `gbdt5_after.json`, `winner_oracle_after.txt`,
`binding_sha256.txt`, the two oracle scripts.

## 6. Owed

1. AMD MI325X and NVIDIA H100 legs, NOT rented under this lane's cap.
   `tools/feature_freq_identity_leg.sh` is the POSIX body (the shape of
   `bench/results/e1g/2026-09-13_195718-amd-mi325x-do-identity-three-columns/extra_body.sh`):
   build every IDENTICAL binding, run the six GBDT lanes on the shipped
   source (ARM fixed), rebuild gbdt with `-D MOJOLEARN_2710_TENSOR_ACC_DEAD=1`
   and rerun (ARM dead), rebuild and rerun the shipped source (the bracket),
   and diff in-box. The expectation is that ARM dead reproduces that vendor's 2026-09-13
   `gbdt-feature-freq` hashes and leaves the five other lanes IDENTICAL
   (the switch flips exactly the divergent lane), ARM fixed equals the Apple
   column above cell for cell (`--diff apple-m4.json <leg>.2710.json` at
   home). Only that diff earns the word fixed.
2. `checks/tensor_sync_state_check.mojo` and `checks/tree_ctr_slice_check.mojo`
   were edited to the same rule and NOT rerun (each compiles the whole
   gbdt package; the cap allowed one build). `pixi run` them on the leg or
   on the next free box before merging.
3. `docs/CROSS_VENDOR_FEATURE_IDENTITY_AUDIT.md:63` still reads "no
   distinct current3way evidence located" for this estimator. After the
   leg it gets a row with the three columns and the hashes; until then
   the estimator stays experimental and outside any identity claim, and
   any earlier same-vendor-pair agreement taken with the literal `False`
   (two boxes corrupting the same way) is not evidence.
4. `hashed` is the one fixture where NVIDIA and AMD agreed with each other
   and not with Apple (`b99acd4c05cb3998` versus `9d97a55431b6f8e0`); the
   host oracle says Apple's tree is the argmax there, so the leg should
   show both moving to `9d97a55431b6f8e0`. If ARM fixed on AMD still reads
   `b99acd4c`, the diagnosis is incomplete and section 3 reopens.

## The two owed legs (2026-09-13 evening): AMD and NVIDIA both move to the Apple column

AMD MI325X (`bench/results/e1g/2026-09-13_214941-amd-mi325x-do-feature-freq-2710`) and NVIDIA
H100 (`bench/results/e1g/2026-09-13_221244-nvidia-h100-feature-freq-2710`), each on one box from one source, three arms: the dead arm
(`-D MOJOLEARN_2710_TENSOR_ACC_DEAD=1`) reproduces that vendor's Sep 13 hashes exactly (AMD base
459e5ba5892267e5, NVIDIA base 944e48bbb96131b5, both hashed b99acd4c05cb3998); the fixed arm
equals the Apple column cell for cell on all nine fixtures in train, infer and model (base
7d9c56b51213cb42, ties 57ff9964d1af1d4a, hashed 9d97a55431b6f8e0, wide c6d7fccba2483372,
denormal and denormal_ftz e7f1da14d1a9794b, dupes e8c61a9407503755, odd c481331145bfb998,
negative 75028913a14bb938); fixed_again equals fixed (`summary: IDENTICAL=54`); the other five
GBDT lanes are IDENTICAL in every arm (45 cells). On the Mac the diff of the committed Apple
column against each vendor's fixed arm reads `IDENTICAL=53` on the six lanes plus the committed
Apple base mover, which is the pre-fix run's own single event. Three vendors, one answer.
