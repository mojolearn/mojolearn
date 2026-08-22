# The DEPTHWISE lane

Opened 2026-08-22 on Andrew's instruction: *mirror CatBoost's CPU and GPU
algorithms for **depthwise growth**, reuse what the symmetric-tree and forest
lanes already built, do not go as feature-complete as elsewhere, and get far
enough to show **bitwise identity across GPUs** for this growth policy.*

`EGrowPolicy::Depthwise`. The LOSSGUIDE (leafwise) lane is a separate session
and owns `EGrowPolicy::Lossguide`; its plan is `LOSSGUIDE.md` and the lane
boundary table lives there, agreed rather than restated here.

## What depthwise IS, in CatBoost

It is **not a different learner.** `TGreedyTreeLikeStructureSearcher<TModel>`
is one class templated on the model, `TGreedySearchHelper` is one class with
`switch (Options.Policy)` in three of its methods, and the histogram /
subtraction / split-points machinery is shared by all four policies verbatim.
The distance between SymmetricTree and Depthwise is exactly five things:

| # | what | their line |
|---|---|---|
| 1 | `numScoreBlocks = leavesToVisit.size()` instead of 1, so the score kernel is `ComputeOptimalSplitsRegion` with the leaf on grid **y** | `greedy_search_helper.cpp:428-432`, `:470-488` |
| 2 | the host reduces **one winner per leaf**, not one for the level | `:546-553` |
| 3 | `MinLeafSize` goes LIVE — the size test is guarded by `Policy != SymmetricTree` | `:685` |
| 4 | `UpdateFeatureWeightsForBestSplits` / `model_size_reg` is **not** called | `:466` (inside the SymmetricTree branch) |
| 5 | the model is `TNonSymmetricTree`, rebuilt from leaf **paths** | `model_builder.cpp:278` |

`SelectLeavesToSplit` is **shared** with SymmetricTree — both take every leaf
with `BestSplit.Defined() && BestSplit.Score < 0` (`:355-364`). Lossguide takes
the single best leaf and has no sign test; Region takes the shallower of the
last pair. Those two are the other lane's and Region is nobody's.

### The CPU side, and why the GPU side is what got ported

CatBoost's CPU depthwise is a genuinely different algorithm:
`GreedyTensorSearchDepthwise` (`private/libs/algo/greedy_tensor_search.cpp:1467`)
walks `curLevelLeafs` in pairs, applies the subtract trick through a
`TQueue<TVector<TBucketStats>>` of parent statistics, and gates on
`gain < 1e-9` where the GPU gates on `Score < 0`. It also carries
`MinDataInLeaf` as a **candidate filter** (`:1528`, skip the leaf entirely)
rather than as a terminal mark.

That is read, cited and **not ported**, for the same reason the symmetric lane
ported `catboost/cuda/` and not `catboost/private/libs/algo/`: the thesis is
*GPU access*, the measuring comparison is our GPU against their CPU on the
same Mac, and porting their CPU learner would produce a second CPU learner
rather than a reachable GPU one. Where their CPU code IS the shipped path —
the quantizer's border building, the 100k subsample — this repository has
ported it and says so. Growth is not one of those places.

The two do not agree tree-for-tree and CatBoost does not claim they do.

## What landed

All of it untracked-then-committed in this lane; none of it touches a file the
symmetric or lossguide lanes were editing.

| file | ports | status |
|---|---|---|
| `gbdt/data/leaf_path.mojo` | `cuda/data/leaf_path.h` | written by the lossguide lane, consumed here |
| `gbdt/gpu_data/gpu_structures.mojo` | `TTreeNode`, `gpu_structures.h:167` | appended at the foot |
| `gbdt/models/non_symmetric_tree.mojo` | `TNonSymmetricTreeStructure` + `VisitBins` + `TNonSymmetricTree` | new |
| `gbdt/methods/greedy_subsets_searcher/model_builder.mojo` | `TFlatTreeBuilder` + `BuildTreeLikeModel<TNonSymmetricTree>` | new |
| `gbdt/methods/greedy_subsets_searcher/structure_searcher_options.mojo` | `TTreeStructureSearcherOptions` | new |
| `.../kernel/compute_scores.mojo` | `ComputeOptimalSplitsRegion` (`:303`) | appended at the foot |
| `gbdt/models/kernel/add_bin_values.mojo` | `ComputeNonSymmetricDecisionTreeBinsImpl` (`add_model_value.cu:353`) | appended at the foot |
| `.../greedy_search_helper_depthwise.mojo` | the Depthwise arms + `FitImpl` + `MakeSplit`'s multi-leaf arm | new |
| `mojo_only/depthwise_check.mojo` | — | the gate, 7 claims |
| `gbdt/methods/greedy_subsets_searcher/points_subsets.mojo` | `TLeaf.Path` | `depth: Int` became the real `TLeafPath`, closing that field's own TODO |

### What was REUSED rather than rewritten, which is most of it

The level machinery is policy-blind and the port already had it. `MakeSplit`'s
kernel takes a **per-leaf-slot** `TCFeature` and bin because THEIRS does
(`split_properties_helper.cpp:856-880`) — the symmetric caller just fills
every slot with the same value. So the depthwise driver calls, unchanged:

`launch_histograms_for_blocks`, `scan_histograms_kernel`,
`substract_histograms_kernel`, `zero_histograms_kernel`,
`copy_histograms_kernel`, `split_and_make_sequence_kernel`,
`launch_stable_partition`, `launch_reorder_in_leaves`,
`update_partitions_after_split_kernel`, `compute_partition_stats`,
`build_necessary_histograms` / `non_zero_leaves`, `choose_scale`,
`best_split_properties_less`, `resolve_split`, and the whole `TTreeWorkspace`
pool. The lane's own `TDepthwiseWorkspace` holds **four** buffers the
symmetric pool cannot serve, all of them because a symmetric level has one
winner and a dense id list where a depthwise level has neither.

Nothing was reused from `ensemble/` or `extratrees/`. Those are cuML ports
with cuML's names and cuML's node layout; the instruction was to use
CatBoost's names and mirror CatBoost's folders, and the CatBoost-shaped
pieces all already existed inside `gbdt/`. What DID transfer from those lanes
is method, not code: the fingerprint-style bit gate and the
sabotage-before-you-believe-it discipline.

## The control plane is THEIRS, deliberately

`HOST_AND_DEVICE.md` rule two: cut our host waits down to their count, not
below it, not yet. Theirs blocks the host **twice per level** —
`bestProps.Read` (`greedy_search_helper.cpp:517`) and `RebuildLeavesSizes`
(`split_properties_helper.cpp:802`). This loop blocks at exactly those two
points and nowhere else.

The symmetric lane's DEVIATION 94/95/207 (blind level enqueue, device-derived
scale, one drain per tree) are **not** replayed here. They were worth ~13 ms
of a 50 ms tree and they were taken AFTER that lane had produced a number.
This lane has no number yet, and a control plane better than theirs cannot
answer whether THEIR design is fast on Metal. That replay is the first
performance item, priced and named, not forgotten.

## Bitwise identity across GPUs

**This lane adds no new row to `IDENTITY_PATHS.md`,** and that is the claim.

Every float that decides anything comes from machinery already enumerated
there: the histogram flush (rows 1-6), the partition-stats chunk count
(row 7, PINNED), the fixed-point scale (row 8, open on both lanes). The two
host reductions this lane adds are order-independent by construction rather
than by pinning:

* the per-leaf cross-block argmin is a sequential fold in block order under a
  **total order** on `(gain, feature_id, bin_id)` — `best_split_properties_less`,
  which is a transcription of their `TBestSplitProperties::operator<`. The
  block count cannot move it.
* `build_necessary_histograms` groups siblings in **leaf-id order** where
  theirs walks a `THashMap` (`split_properties_helper.cpp:1306`). Hash order
  does not move bits — the pairs are independent — but id order is
  reproducible and theirs is not, so this can only narrow the gap.

`depthwise_check` **claim 6** is the run that says so: the same tree is grown
three times in one process with core counts `{this device, 108, 1}` and the
three models must be bit-identical — nodes, split types and leaf-value bit
patterns. The core count is the only machine-dependent input the algorithm
has, and it is the input that was silently building different models on a
10-core M4 and a 108-SM A100 before row 7 was pinned. A `sm_count_override`
parameter exists on `fit_depthwise_tree` for exactly this and for nothing
else, per RESUME.md's rule that a configuration which cannot be varied inside
one process cannot be measured here.

What claim 6 does **not** prove is that Metal and CUDA agree. Denormal policy
and FMA contraction are rows 9 and 10 and are tested by `check-ieee-arith`,
which is the first thing to run on any new backend column. The NVIDIA and AMD
columns are `tools/remote_gpu.sh` and RunPod, and running them is billed and
awaits Andrew's explicit word — same standing as every other lane.

### Contraction on the shared leafwise kernel, from the lossguide lane

The Depthwise and Lossguide scorers are **one body** (their DEVIATION 302), so
what is true of row 9 there is true here. Their measurement, 2026-08-22:

* the pin **reaches Metal**, verified in emitted code rather than inferred —
  diffing a FAST build's AIR sidecar against an IDENTICAL one turns
  `fmul contract` + `fadd contract` into `call contract float @llvm.fma.f32`,
  10 sites to 22, with the L2 kernel module byte-identical as the control. So
  `numerics.identical_mul_add` works and row 9's construction is sound.
* on the one seam observed discriminating (`DenumSqr += weight*mu*mu`),
  **Apple's own FAST codegen already fuses**. FAST and IDENTICAL can therefore
  agree on Apple, and that agreement transfers to no other backend.
* this CONTRADICTS the generalization in `IDENTITY_PATHS.md` row 9, which
  records `check-ieee-arith` measuring Apple unfused, 0 of 2^20. Both can be
  right — different expressions, different kernels — and the transferable
  lesson is that **contraction is measured per seam and never inherited from a
  probe**. Row 9 as written reads as a device-wide property and should grow
  that caveat; it predates both lanes and neither has edited it.

An earlier version of that finding ("15 naive / 3 fused in one build") was
retracted by its own lane within the hour as a defect in their gate's bucket
ordering, before it reached this file. It is recorded here because the
retraction is the useful part: a three-way `if/elif/else` tally silently files
"the two walks agree" under the first bucket, and every `stat_count == 2`
fixture is structurally blind to contraction anyway, since `fma(a,b,0) ==
a*b + 0` exactly.

## Running the two arms, and the stage ladder

`check-depthwise` gates whichever numeric mode the build carries.
`GLOBAL_NUMERIC_MODE` is a `comptime` in `mojo_only/numerics.mojo` and the
shipped default is `NUMERIC_FAST`, so the `IDENTICAL` arm needs a build with
that one line flipped. **Do not flip it in the shared checkout** — three
sessions compile against it. Use a detached worktree:

```sh
WT=/tmp/wt-identical
git worktree add --detach "$WT" HEAD
sed -i '' 's/^comptime GLOBAL_NUMERIC_MODE = NUMERIC_FAST$/comptime GLOBAL_NUMERIC_MODE = NUMERIC_IDENTICAL/' \
  "$WT/mojo_only/numerics.mojo"
pixi run mojo run -I "$WT" "$WT/mojo_only/depthwise_check.mojo"
pixi run mojo run -I "$WT" "$WT/mojo_only/depthwise_trace_probe.mojo"
git worktree remove "$WT"
```

`check-depthwise-trace` is the STAGE LADDER, and **it is the lossguide lane's
`core/identity_trace.mojo`, not a second instrument.** This lane briefly had
its own (`mojo_only/stage_digest.mojo`, commit `e5cef46`) because both lanes
built the same thing inside the same hour without knowing. It is deleted.
Theirs is a strict superset — generic over `DType` where mine was four
hand-written methods, `create_sub_buffer` for a short read where mine needed
two lengths, plus raw dumps, enforced tag uniqueness, a format version, and a
reader: `tools/identity_trace_diff.py` aligns two traces on tag SEQUENCES and
classifies each differing cell (DENORMAL-vs-ZERO / SIGN / NAN-payload /
ULP<=n / LARGE). **That classification is the diagnosis; a hash is only the
location.** An all-denormal divergence is row 10 and a mode difference; a
scattered 1-ULP divergence is a summation order; those send you to different
files.

What this lane added to that file, at the foot: `read_trace_lines` and
`first_divergence`, so a MOJO check that produces two traces in one process
can raise on the difference. `mojo_only/identity_trace_check.mojo` had
already written that loop inline and can now drop its copy.

The probe answers two questions locally and leaves both traces on disk for
the cross-machine path:

- **control** — the same configuration twice. Run-to-run noise and machine
  dependence are indistinguishable without it.
- **localization** — this device's core count against 108.

Two things the ladder taught about itself, both the hard way. It **needs the
control** (it was added after the fact). And it **must never hash a
machine-sized scratch**: `sflags` and `gmap` are sized `n_rows` while a level
writes only the splitting leaves' rows, so hashing the plane digests HISTORY
and not the stage — it named a false first-divergence at `d3.flags` while
claim 6 correctly said the models were bit-identical. Dropped; `row_index`,
`stats` and the two partition planes are the complete live-region
description. The lossguide lane had reached the same rule independently and
it is in their file's docstring.

Order on a new backend: `check-ieee-arith`, then `check-depthwise`, then the
ladder. Running the ladder first tells you a stage disagrees without telling you
whether the arithmetic was ever going to agree.

**Measured, M4, 2026-08-22:**

| | check-depthwise | trace records | reproducible | identical across {device, 108, 1} SMs |
|---|---|---|---|---|
| FAST | 7/7 | 99 | yes | yes — a platform accident, Metal has no float threadgroup atomics |
| IDENTICAL | 7/7, claim 6 **gated** | 94 | yes | **yes** |

## Deviation numbers

**350-399 are this lane's**, agreed with the lossguide lane (which took
300-349) so that the 2026-08-20 collision — two lanes claiming 42
concurrently — cannot repeat. Spent so far: 350 (`TTreeNode`'s `ui16` fields
raise where theirs narrow silently), 351 (the bin-feature lookup table),
352 (partition stats recomputed per level rather than updated inside the
split).

## What is NOT done, and is not pretended to be

* **Not wired into training.** `doc_parallel_boosting` grows symmetric trees
  only, and `catboost_options.check()` still refuses `grow_policy=Depthwise`
  by name. That refusal is honest until the boosting driver takes a policy;
  relaxing it before then would make an option that is accepted and dropped,
  which is what `PORTING_RULES.md` rule 3 forbids. `UNWIRED.md` carries the
  order of what has to land first.
* **No CatBoost differential.** There is no `oracle` row for depthwise. The
  gate compares against host tallies and against itself, which is why claim 7
  is a sabotage and not a digest.
* **No number.** Nothing has been benchmarked. The first performance question
  is the control-plane replay above; the second is whether a depthwise level
  still launches `O(1)` kernels in the leaf count, which is CatBoost's own
  design property (`compute_by_blocks_helper.h:87-92`) and is worth
  confirming rather than assuming now that the leaf frontier is ragged.
* **Region and Lossguide** are absent rather than stubbed. Lossguide is the
  other lane's; Region is nobody's.
* `Rescale` / `ShiftLeafValues` / `UpdateLeaves` / `UpdateWeights` on
  `TNonSymmetricTree`, `BuildTreeLikeModel<TObliviousTreeModel>`, `GetHash`,
  `SortPath` / `SortUniquePath`, `BootstrapOptions`, `FixedBinarySplits`:
  recorded in `UNWIRED.md`, not written.
