# The Lossguide lane

Opened 2026-08-22. **This lane ports `EGrowPolicy::Lossguide` -- leafwise,
best-first growth -- out of CatBoost's own GPU learner and into mojolearn.**
A sibling lane ports `EGrowPolicy::Depthwise`. The two share a spine; the
boundary is written down below so the spine gets built once.

The charter is unchanged and is `STANDING_ORDERS.md` plus `PORTING_RULES.md`.
Copy, do not improve. Their file tree is our file tree. Every claim cites
`file.cpp:line` against CatBoost `54a8143a` at `/private/tmp/catboost-src`.

## What we are NOT doing

**Not feature complete.** Andrew's scope, 2026-08-22 verbatim in effect: enough
Lossguide to show a **bit-identical model across GPU backends**, and no more.
So this lane does NOT port, and will refuse rather than approximate:

* `EGrowPolicy::Region` (`greedy_search_helper.cpp:325-350`) -- the third
  non-symmetric policy, and a different leaf-selection rule again.
* `FixedBinarySplits` (`:388`, `:545`) -- it reroutes Lossguide onto the
  Depthwise kernel mid-tree, so it is a policy mixer and out of scope.
* Multi-device anything. `AllReduceThroughMaster`, `TStripeBuffer` device
  loops, `RebuildLeavesSizes`' `dev` sum. Device count is 1 and their loops
  collapse; every collapse is noted at its site.
* Multiclass (`MultiLogitOptimization`) on the leafwise path. The symmetric
  arm has it; this lane's first gate is single target.
* `ModelSizeReg` / `UpdateFeatureWeightsForBestSplits`. **Read their code
  before assuming it applies**: that call sits inside the
  `Policy == SymmetricTree` branch only (`:466`), so Lossguide never runs it.
  This is a real asymmetry in their source, not a shortcut of ours.

## What Lossguide IS, read off their source

The whole of the policy is four decisions. Everything else on the leafwise
path is shared with Depthwise or with the symmetric searcher we already have.

### 1. Which leaf gets split -- `SelectLeavesToSplit` (`greedy_search_helper.cpp:319-324`)

```cpp
if (Options.Policy == EGrowPolicy::Lossguide) {
    TMaybe<ui32> leafToSplit = FindBestLeafToSplit(subsets);
    ...push_back(leaf) for that one leaf only
}
```

`FindBestLeafToSplit` (`:296-309`) is a plain argmin of `BestSplit.Score`
over every leaf that has one, `<` strict, so **the FIRST leaf wins a tie**
and leaf ids are creation order. One leaf per iteration, and the tree grows
one split at a time.

**The difference from Depthwise that is easy to miss and changes models:**
Depthwise and SymmetricTree take every leaf whose `BestSplit.Score < 0`
(`:355-359`). Lossguide has **no `< 0` test at all**. A leaf whose best split
makes the objective worse is still split, if it is the least-bad leaf in the
tree. Lossguide is bounded by `MaxLeaves` and by `IsTerminalLeaf`, never by
the sign of the gain. Do not "fix" this.

### 2. Which kernel scores it -- `ComputeOptimalSplits` (`:398-533`)

`numScoreBlocks = leavesToVisit.size()` for all three non-symmetric policies
(`:441-446`), then the dispatch splits three ways (`:465`, `:490`, `:513`):

| policy | kernel | shape |
|---|---|---|
| SymmetricTree | `TComputeOptimalSplitsKernel` | one score summed over all leaves |
| Depthwise (+ any fixed splits) | `TComputeOptimalSplitsLeafwiseKernel` -> `ComputeOptimalSplitsRegion` | `partIds[blockIdx.y]`, `partCount` leaves |
| **Lossguide** | **`TComputeOptimalSplitLeafwiseKernel` -> `ComputeOptimalSplit`** | **`CB_ENSURE(leavesToVisit.size() <= 2)`, two scalar part ids** |

The two non-symmetric kernel bodies in `kernel/compute_scores.cu` are
**identical line for line** (`:303-385` vs `:393-475`) except for one
statement: Region reads `partIds += blockIdx.y; thisPartId = partIds[0]`,
Lossguide takes `partId` / `maybeSecondPartId` as scalars and picks with
`blockIdx.y == 0 ? partId : maybeSecondPartId`. That is the entire delta, and
it is why the depthwise lane and this one can share one Mojo kernel body with
the part id resolved by a comptime arm. Recorded as **DEVIATION 317**.

Why `<= 2`: a Lossguide iteration splits exactly one leaf, which creates
exactly two leaves without a `BestSplit`, and `SelectLeavesToVisit`
(`:697-710`) returns exactly the leaves that lack one. The root iteration
returns 1. So the leafwise scorer never sees more than two.

### 3. How the rows move -- `MakeSplit(leafId, ...)` (`split_properties_helper.cpp:952-1027`)

`MakeSplit` on a `TVector` of one leaf **delegates to a whole separate
single-leaf overload** (`:840-842`), which launches
`TSplitPointsSingleLeafKernel` instead of `TSplitPointsKernel`. Under it are
six single-leaf kernels in `kernel/split_points.cu` that this repository has
never ported, because the symmetric path only ever splits every leaf at once:

`SplitAndMakeSequenceInLeaf`, `SortByFlagsInLeaf`, `UpdatePartitionAfterSplit`,
`GatherInplaceSingleLeaf`, `GatherLeaf`, `CopyLeaf` (`kernel/split_points.cuh:34`,
`:46`, `:64`, `:108`, `:117`, `:127`).

This is the half nobody else needs: the depthwise lane splits many leaves per
level and reuses the multi-leaf kernels already in `kernel/split_points.mojo`.

**The sentence that used to sit here, "this is the lane's real porting work",
is WRONG and is deleted.** Their own comment calls it a "fast path for
lossguide learning" (`split_properties_helper.cpp:944`) and our multi-leaf
`split_and_make_sequence_kernel` already takes per-leaf-slot features, bins
and ids -- so a one-element call grows a correct Lossguide tree without any
of the six. They are a PERFORMANCE arm. See finding 3.

Their leaf sizes are then refreshed by `FastUpdateLeavesSizes` (`:815-828`),
which reads back **two** partitions, not all of them -- the `RebuildLeavesSizes`
full read (`:800-812`) is O(leaves) per split and Lossguide splits O(leaves)
times, so theirs is O(leaves^2) if you take the wrong branch. Take theirs.

### 4. When it stops -- `ShouldTerminate` / `IsTerminalLeaf` (`:668-690`)

```cpp
if (leafCount >= Options.MaxLeaves) return true;         // the Lossguide bound
const bool checkLeafSize = Options.Policy != EGrowPolicy::SymmetricTree;
flag = (checkLeafSize && leaf.Size <= Options.MinLeafSize)
       || leaf.Path.GetDepth() >= Options.MaxDepth;
```

`MinLeafSize` becomes LIVE here. `catboost_options.mojo:289` already carries
the note that CatBoost ignores `min_data_in_leaf` under SymmetricTree and that
it "cannot become silently live the day Lossguide lands" -- **this is that
day**, and the option's docstring is corrected in the same commit that wires
it, per the fix-docs-on-discovery rule.

`MaxLeaves` likewise. `catboost_options.mojo:21` records that CatBoost
overwrites the constructed 31 with `1 << MaxDepth` for **every policy but
Lossguide**; Lossguide is the one policy where the user's `max_leaves` is the
number that runs.

## The model that comes out is not an oblivious tree

A Lossguide tree has a different split per node, so `TObliviousTreeStructure`
(one split per level) cannot hold it. Their answer is
`TNonSymmetricTreeStructure` (`cuda/models/non_symmetric_tree.h:11`), a flat
preorder array of `TTreeNode { FeatureId, Bin, LeftSubtree, RightSubtree }`
(`gpu_data/gpu_structures.h:167`), built by `TFlatTreeBuilder`
(`model_builder.cpp:135-283`) out of the per-leaf `TLeafPath`s the search
returns, and applied by `ComputeNonSymmetricDecisionTreeBinsImpl`
(`models/kernel/add_model_value.cu:353-395`) -- a per-row walk down the flat
array with no recursion and no stack.

That apply kernel is 25 lines and is the whole prediction path. It is shared
with the depthwise lane verbatim.

## The lane boundary, settled with the depthwise lane

**Settled 2026-08-22 by direct message with `cascadeprojects-9f`, the
depthwise lane, and accepted by both.** The first draft of this section had
this lane building the shared spine. Reality overtook it inside twenty
minutes: that lane was already writing the spine in the same checkout. So the
spine is THEIRS and this lane consumes it.

**Theirs (shared, built once):** `gbdt/data/leaf_path.mojo` (`TLeafPath`),
`gbdt/models/non_symmetric_tree.mojo` (`TNonSymmetricTreeStructure` and its
`VisitBins` walk), `gbdt/methods/greedy_subsets_searcher/model_builder.mojo`
(`TFlatTreeBuilder`, `BuildTreeLikeModel<TNonSymmetricTree>`),
`structure_searcher_options.mojo`, `TTreeNode` in `gpu_data/gpu_structures.mojo`,
`ComputeNonSymmetricDecisionTreeBins` in `models/kernel/add_bin_values.mojo`,
and the `ComputeOptimalSplitsRegion` arm.

**Mine alone:** the six single-leaf kernels in `kernel/split_points.mojo`
(`TSplitPointsSingleLeafKernel`), the `ComputeOptimalSplit` arm of
`kernel/compute_scores.mojo`, `split_properties_helper.mojo`'s single-leaf
`make_split` and `fast_update_leaves_sizes`, and the Lossguide arms of
`select_leaves_to_split` / `should_terminate`.

**Convergent files** -- `kernel/compute_scores.mojo`, `kernel/split_points.mojo`,
`gbdt/options/catboost_options.mojo`, `pixi.toml`, `gbdt/gpu_data/gpu_structures.mojo`.
Additions at the FOOT behind their own headers; neither lane rewrites the
other's arm.

### A HAZARD ALREADY REALIZED, recorded so it is not repeated

**I overwrote that lane's `gbdt/data/leaf_path.mojo` at 09:57.** It was
untracked, so git held no copy and their file was destroyed. The cause was a
directory listing I had taken twenty minutes earlier and then trusted. The
rule this pays for: **in a shared checkout, `ls` the target directory in the
same tool call as the write, or check `git status` for untracked files,
before creating any file whose name another lane might plausibly want.** The
damage was recoverable only because their `non_symmetric_tree.mojo` compiled
clean against my replacement and they had lost nothing they could not
restate.

### Files this lane may not touch

Mid-edit by other sessions at lane open:
`gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo`,
`gbdt/methods/doc_parallel_boosting.mojo`,
`gbdt/methods/kernel_add_model_value.mojo`,
`gbdt/methods/leaves_estimation/pointwise_oracle.mojo`, `ensemble/*`,
`bindings/*`, `packaging/*`.

`greedy_search_helper.mojo` is the natural home for `TGreedySearchHelper`'s
Lossguide methods and it is DIRTY. So they land in
`greedy_search_helper_lossguide.mojo` in the same directory, with a merge note
at its head and a `PORTED_MAP.tsv` row pointing at the same upstream file.
**That split is temporary and is itself DEVIATION 316.** The depthwise lane is
doing the same with `greedy_search_helper_depthwise.mojo`.

## Deviation numbers

Rule 3 says assign them before parallel work begins. The gbdt lane is at 144,
the forest lanes are at 215.

* **315-349 is this lane, Lossguide.**
* **350-399 is the depthwise lane's.** Do not spend them.
* **300-314 is the ENSEMBLE lane's and always was.** See the renumbering
  below.

### RENUMBERED 2026-08-22: this lane opened on a block that was already taken

The block agreed with the depthwise lane this morning was 300-349. **It
collided.** The ensemble lane had committed 305, 306, 309, 311, 313 and 314 in
`ensemble/randomforest.mojo` at `7ce6db4`, 2026-08-21 15:06 -- **a day before
the agreement**. Verified against git before accepting the ruling, not taken
on assertion: the commit, the date and six live citations are all there, and
`core/philox.mojo:872` carries their 306 as well.

First-landed-keeps. The ensemble lane holds 300-314 and this lane moved to
315-349, on the perf lane's ruling.

**THE MAP, because five commits already in history cite the OLD numbers and
cannot be rewritten -- peers have built on them:**

| old | new | what |
|---|---|---|
| 300 | **315** | the lane's own header row (reserved, never cited in code) |
| 301 | **316** | the three-file split of one CatBoost file |
| 302 | **317** | one shared body for both leafwise score kernels |
| 303 | **318** | the multiply-add pin on the Cosine calcer |
| 304 | **319** | the cross-lane predicate import (CLOSED) |

Commits `97df3d8`, `790728a`, `77ca6ab`, `8321474`, `41717a7` and `bc2739a`
say 301-304 in their messages and mean 316-319. The CODE is renumbered;
history is not, and this table is the only thing that connects them.

**WHY IT HAPPENED, since rule 3 exists to prevent exactly this.** Two lanes
agreed a block between themselves without checking what was already committed
in a THIRD lane's directory. An agreement between two parties is not a
namespace. Deviation blocks now route through the perf lane, which is the
right answer and was not in place this morning.

Taken so far (the depthwise lane holds 350-399 and has spent 350-352):

| # | what | status |
|---|---|---|
| 315 | reserved -- the lane's own header row | -- |
| 316 | `greedy_search_helper_lossguide.mojo` is a second file for one of theirs, forced by a dirty checkout | open |
| 317 | one shared body serves `ComputeOptimalSplit` and `ComputeOptimalSplitsRegion`; the part-id read is the only difference | **LANDED**, and it caught a block-size defect in its first hour |
| 318 | the Cosine calcer's two multiply-adds routed through `numerics.identical_mul_add` on the leafwise path, behind a defaulted comptime parameter so the symmetric arm's source is unchanged character for character | landed and verified in the AIR |

**DEVIATION 317 LANDED, after a false start worth keeping.** I wrote the
shared `_leafwise_scan_part` + `_leafwise_argmax_write` helpers so the
Depthwise kernel would be four lines on top of them. The depthwise lane
landed `compute_optimal_splits_region_kernel` in the same file at the same
hour with the body written out again, so `compute_scores.mojo` briefly
carried it TWICE -- exactly the drift surface 317 exists to remove.

**They deleted their copy and folded onto the helpers, and it immediately
paid for itself:** their standalone body used `SCORE_BLOCK_SIZE` (128), the
SYMMETRIC launcher's block, where both leafwise launchers use 256
(`compute_scores.cu:484`, `:557`). Nothing but a fidelity read catches that
-- the block argmax is a total order and the scan is grid-strided, so every
correctness gate passes at either size. One body, one defect found, in its
first hour.

The sentence that used to sit here -- "the fix is not urgent enough to do
while both files are hot" -- is deleted. It was wrong: the other lane did it
the same hour and found a bug doing it.

## The gates, in the order they get built

Correctness gates on analytic answers and on their output, never on a real
dataset (rule 4).

1. `check-leaf-path` -- `TLeafPath` host algebra. `PreviousSplit`,
   `IsSorted`, `HasDuplicates`, sibling paths sharing a parent key.
2. `check-non-symmetric-model` -- `VisitBins` against the paths that built
   the tree. Their traversal is an explicit unwind stack with two subtree
   counters and it is the easiest thing in this file to get subtly wrong;
   the check is a round trip, `paths -> TFlatTreeBuilder -> VisitBins ->
   paths`, on hand-built ragged trees including the single-leaf tree
   (`Nodes.empty()`, `:64-67`).
3. `check-leafwise-scores` -- the leafwise scorer against a host
   recomputation on a planted histogram, with ties planted, plus the
   `blockIdx.y == 1` arm reached by sabotage (rule: reach is per branch).
4. `check-split-single-leaf` -- the six single-leaf kernels against the
   multi-leaf path on the SAME split. They must agree cell for cell; the
   multi-leaf path is already gated, so it is a real oracle.
5. `check-lossguide-tree` -- one whole tree, host replay of the search
   against the device, split for split and leaf value for leaf value.
6. **`check-lossguide-identity`** -- the deliverable. Under
   `NUMERIC_IDENTICAL`, perturb every machine-derived scheduling row that
   `IDENTITY_PATHS.md` enumerates (block counts, replication, chunk counts,
   grid x) and assert the model is **byte for byte the same**. A backend
   change is a scheduling change; if no scheduling change can move a bit,
   the cross-backend claim has a mechanism behind it rather than a hope.
   Rows 8, 9 and 10 of that ledger are OPEN and this lane must not claim
   past them.

## What was found, 2026-08-22

### 1. `Score` and `Gain` are the SAME NUMBER on the leafwise kernels, and that is why this port's collapse is safe HERE

`TBestSplitProperties` carries two floats. The cross-block host reduce keys on
`Gain` (`operator<`, `gpu_structures.h:80-93`); **Lossguide's leaf argmin keys
on `Score`** (`greedy_search_helper.cpp:300`). This port collapsed the two
into one number long ago, documenting it as "out_score carries their Gain".
That collapse would silently change WHICH LEAF Lossguide splits if the two
fields ever differed on this path.

They do not. The symmetric kernel assigns `bestScore = score` and
`bestGain = gain` -- two different values (`:142-144`). **Both leafwise
kernels assign `bestScore = gain` and `bestGain = gain`** (`:353-357`,
`:468-472`). So on the Lossguide path the argmin over `Score` IS an argmin
over the weighted gain, and the collapse is exactly right. It would NOT have
been right on the symmetric arm.

Checked rather than assumed, because assuming it is how a lane ships a tree
that splits the wrong leaf and still passes every histogram gate.

### 2. RETRACTED AND REPLACED: the contraction pin WORKS, and my gate was the thing that was broken

**What this section said first, and what commit `97df3d8` says in its
message, is FALSE and is deleted:** "15 shapes match the unfused chain and
three match `fma`, in one kernel, in one build ... MAX on Metal contracts
these two lines on some instantiations and not others", together with the
conclusion that `numerics.identical_mul_add` "moved not one bit".

The 15/3 split was a defect in `check-leafwise-scores`. Its tally had three
buckets tested as `if naive: elif fused: else neither`, so **every fixture on
which the two host walks AGREE was counted as "naive"**. Running the gate's
own `host_best` against ITSELF on the CPU, with no device involved at all,
reproduces the same 15/3 split: it is a property of the FIXTURES. And the
reason so many fixtures agree is exact: `fma(a, b, 0) == a * b + 0`, so every
`stat_count == 2` shape -- one iteration of the stat loop, both accumulators
still at their seeds -- is structurally blind to contraction, and nine of the
eighteen shapes were that.

Corollary the gate could not have survived: its `if cos_naive != 0: FAIL`
assertion under IDENTICAL could never have passed, however correct the pin
was.

**WHAT IS ACTUALLY TRUE.**

1. **The pin reaches Metal.** Verified in emitted code rather than inferred.
   `mojo build --emit asm` writes a per-kernel AIR sidecar; diffing a FAST
   build against an IDENTICAL one turns `fmul contract` + `fadd contract`
   into `call contract float @llvm.fma.f32`, **10 sites to 22 -- exactly the
   +12 that six pinned `_add_leaf` calls times two lines predicts.** The L2
   kernel's module is byte-identical between the two builds, which is the
   control. IDENTITY_PATHS row 9's construction is sound.

2. **On the one seam that discriminates, Metal's own FAST codegen already
   fuses.** The three non-blind shapes match the `fma` walk under FAST too.
   **That contradicts the generalization in IDENTITY_PATHS row 9**, which
   records `check-ieee-arith` measuring Apple UNFUSED, fused 0 of 2^20. Both
   measurements can be right -- different expressions in different kernels --
   and the transferable lesson is the one this lane is putting on the record:
   **contraction must be measured PER SEAM and never inherited from a probe.**
   On Apple the pin therefore buys nothing here and buys everything on a
   backend whose codegen chooses the other way.

3. **Honest limit.** The discriminating fixtures only exercise
   `DenumSqr += weight * mu * mu`. A model that fuses ONLY `Score += sum * mu`
   is bitwise equal to the naive chain on every fixture here, so the device
   matching a both-fused model establishes the `DenumSqr` seam and leaves the
   `Score` seam undetermined. The gate now buckets four ways -- fused / naive
   / TIE / neither -- runs past `stat_count == 3`, and asserts
   `cos_fused != 0` so a fixture set that discriminates nothing cannot pass
   quietly.

4. **THE CORRECTED TALLY, run on the device in both modes, M4, 2026-08-22:**

   | mode | shapes | fused | naive | tie (blind) | neither |
   |---|---|---|---|---|---|
   | `NUMERIC_FAST` | 36 | **6** | **0** | 30 | 0 |
   | `NUMERIC_IDENTICAL` | 36 | **6** | **0** | 30 | 0 |

   Identical, and that is the expected result rather than a null one: Apple
   already fuses this seam, so **the pin is bitwise inert on Apple and its
   whole value is aligning a backend that chooses otherwise.** Zero naive
   and zero neither in both modes; six shapes discriminate, so the gate is
   not blind. L2 is bit-identical to the host walk at every shape in both
   modes, which is the control -- its accumulation has no multiply to fuse.

**THE LESSON, which is the reusable part.** A tally is not a measurement
until the buckets are disjoint and the blind cases are named. I read a number
off my own gate and wrote a conclusion about a vendor's compiler from it,
and the number was about my fixtures. `[[gate-against-a-real-accumulator]]`
is the same class: a gate that cannot distinguish two hypotheses will still
report one of them.

### 3. The six single-leaf kernels are a PERFORMANCE arm, not a correctness prerequisite

Pointed out by the depthwise lane and consistent with their own comment,
"fast path for lossguide learning" (`split_properties_helper.cpp:944`): our
already-ported multi-leaf `split_and_make_sequence_kernel` takes per-leaf-slot
split features, bins and ids, so calling it with a one-element list grows a
correct Lossguide tree. **Order of work re-planned accordingly**: get a
correct tree on the multi-leaf path and gate it, THEN port the single-leaf
kernels and price them as a measured deviation. That is the right order
anyway -- their fast path exists because Lossguide splits one leaf O(leaves)
times, and a speed claim about it needs a slow arm to be measured against.

Being verified independently against their source before it is relied on.

## What is not yet known

* Whether the `Score += sum * mu` seam discriminates at all on any backend.
  The `DenumSqr` seam does; this one has not been caught doing so, and an
  unexercised pin is a pin nobody has tested.
* What Apple's AIR-to-binary stage does with `llvm.fma.f32`. The AIR diff
  proves the pin survives to AIR, not to metal machine code. Low risk,
  unmeasured, and labelled as such.

## The stage-hash instrument, and where the tags go

Added 2026-08-22 on Andrew's ask: **hash tags per section, so a cross-GPU
difference has an address.** A bit-identity claim that can only be checked on
the final model is a claim nobody can debug -- two model files that differ
tell you nothing about where the first bit moved.

* `core/identity_trace.mojo` -- the writer. `MOJOLEARN_IDENTITY_TRACE=<path>`
  makes a fit emit `<seq> <tag> <dtype> <count> <fnv1a64>` per checkpoint,
  over RAW BIT PATTERNS in index order.
  `MOJOLEARN_IDENTITY_TRACE_DUMP=<substring>` also writes the raw elements so
  the comparison can go to cells.
* `tools/identity_trace_diff.py` -- the reader. Aligns two traces on their
  TAG SEQUENCES first (a differing stage set is a bigger finding than any
  hash), then names the FIRST diverging stage, verifies both dumps re-hash to
  their records, and classifies each differing cell:
  `DENORMAL-vs-ZERO` / `SIGN` / `NAN-vs-NUMBER` / `DIFFERENT-NAN-PAYLOAD` /
  `ULP<=n` / `LARGE`.
* `pixi run check-identity-trace` -- gates BOTH HALVES TOGETHER. A writer and
  a reader tested only apart are two programs, not one instrument.

**Why the classification is the point, not a garnish.** An all-`DENORMAL-vs-ZERO`
divergence is IDENTITY_PATHS row 10 -- Metal flushes subnormals, CUDA does
not -- and it is a MODE difference, not an accuracy problem. A scattered
1-ULP divergence is a summation order. Those two findings send a reader to
completely different files, and an instrument that reported only "the
histogram differs" would not separate them. The end-to-end gate plants
`+0.0` against the smallest denormal and REQUIRES the differ to name that
class.

### The four rules a checkpoint has to obey

1. Bit patterns, never decimal text (`[[mojo-string-float-roundtrip]]`).
2. **Tags must be machine-independent AND unique within a trace.** Alignment
   is by tag sequence, so a tag carrying an SM count produces two disjoint
   tag sets, and a repeated tag lets the aligner pair tree 3's record against
   tree 7's. Uniqueness is enforced in the writer, not left as a convention.
3. Hash the LOGICAL buffer, never a machine-sized scratch. `stat_partials` is
   sized from the core count (row 7); two backends legitimately hold
   different partials that reduce to the same answer. Checkpoint the reduce.
4. **A traced run is not a measurement.** Every record drains and copies.

### The stage list for the Lossguide fit

**STATE 2026-08-22, second pass.** The sentence that used to sit here --
"wired so far: `fixture.*` and `score.*`; the rest land with the fit loop" --
is STALE and is replaced: the merged driver now carries the ladder end to end
(`d<iter>.leaves`, `plan.*`, `hist.scanned`, `scores.gain/bin`, `visit`,
`best.*`, `split.*`, `rowindex`, `stats`, `parts.*`, `final.*`, `model.*`),
and `check-lossguide` L6 gates its REACH on this policy's path (94 records on
the fixture fit, traced tree identical to untraced). The table below is the
intent; the driver spells the per-iteration prefix `d<iter>.` -- the
`treeNN.` layer lands when the boosting loop carries the trace across trees.

**Two stages the driver's ladder could NOT supply are now this lane's
`select_leaves_to_split_traced`** (`greedy_search_helper_lossguide.mojo`):
`queue.feature/bin/gain` -- the per-leaf `BestSplit` records over ALL leaves,
verbatim stored fields, which IS the priority queue the argmin reads (the
driver's `best.*` covers only the <= 2 VISITED leaves) -- and
`selected_leaf`, the policy's integer choice. Gated by `check-lossguide-policy`
P7, whose tooth is a gain edit on an unselected, unvisited leaf: `queue.gain`
must move while `selected_leaf` and `queue.feature` stand still. **Driver
wiring OWED**: the one-line swap at the `lossguide_select_leaves_to_split`
call site to the traced variant with `trace` and `d_tag` -- the depthwise
lane's file, edit requested through the orchestrator.

| tag | what it pins |
|---|---|
| `treeNN.target.stats` | `StochasticDer`'s gradient and weight planes, before any tree work |
| `treeNN.root.part_stats` | the root reduce -- if this moves, nothing downstream is interpretable |
| `treeNN.iterII.leafKK.hist.built` | the histogram actually accumulated |
| `treeNN.iterII.leafKK.hist.subtracted` | the sibling derivation, which is where a float subtraction re-rounds |
| `treeNN.iterII.leafKK.hist.scanned` | the prefix scan the scorer reads |
| `treeNN.iterII.best_gain` / `.best_bin` | the REDUCED winner, never the per-block scratch (rule 3) |
| `treeNN.iterII.selected_leaf` | which leaf Lossguide chose -- an integer, so a difference here is structural and not numeric |
| `treeNN.iterII.partitions` | offsets and sizes after the split |
| `treeNN.iterII.row_index` | the permutation, which no float can explain |
| `treeNN.structure.nodes` | the flat non-symmetric tree |
| `treeNN.leaf_values` | the values, after estimation |
| `model.cursor` | the applied prediction, per tree |

The ordering is deliberate: **integers before floats at every level.** If
`selected_leaf` or `row_index` differs, the divergence is structural and no
amount of ULP analysis on the histograms will explain it. If they agree and
only the float stages move, it is a numeric pathway and IDENTITY_PATHS is the
file to open.

### Stage wall timers: BUILT ONCE, BY THE OTHER LANE, AND CONSUMED HERE

The orchestrator's ask (env-gated `MOJOLEARN_STAGE_TIMES=1`, stage -> seconds
at fit end, env read once) was ALREADY BEING BUILT by the depthwise lane when
this lane went to write it: `depthwise_stage_times.mojo` appeared untracked
in the target directory, found by the `ls`-in-the-same-breath rule the 09:57
overwrite paid for. **No second instrument was written** -- the
instrument-built-twice pattern from the dedup audit, caught BEFORE the
duplicate this time. Their `StageTimes` drains on both edges of every stage
(so a stage-timed run is NOT a timing, same clause as trace rule 4), prints
integer-math milliseconds, and its `begin`/`end` pairs already bracket the
shared driver's stages, which ARE the Lossguide stages -- one driver, four
policy branches.

**REACH VERIFIED ON THIS POLICY'S BRANCH, 2026-08-22** (their wiring
committed at `045afa3`, including the `report()` this lane's first look
predated -- that observation was a mid-edit snapshot):
`MOJOLEARN_STAGE_TIMES=1 pixi run check-lossguide` prints the full 14-stage
table on every Lossguide fit, headed `lossguide fit: rows=... leaves=...
iterations=...`. Fixture-scale shape (4096 rows, 9 leaves -- TRIAGE, drains
per stage, never a benchmark): hist.* ~46%, split.chain ~16%, score.* ~13%.
The dataset-scale table that prices the single-leaf fast path is the
orchestrator's queued run; the port itself is GATED on that table by ruling.

### FOUND 2026-08-22: the driver's reorder never takes the inplace fast arm

`launch_reorder_in_leaves` dispatches on `max_leaf_rows` exactly as theirs
does on `maxLeafSize` -- the largest SPLITTING leaf's partition size,
computed on the host over the leaves being split (`split_points.cpp:60-63`),
`<= 1024` taking `GatherInplaceLeqSize`, ONE launch, no scratch. The
SYMMETRIC helper passes the real number (`max_live_rows`,
`greedy_search_helper.mojo:1278`, `:3342`). **The merged non-symmetric driver
passes `n_rows`** (`greedy_search_helper_depthwise.mojo:1562`), so past 1024
TOTAL rows the fast arm is unreachable however small the split leaf is, and
every split pays the slow arm's `2*ceil(statCount/8) + 2` launches plus
scratch round-trip. Lossguide is the policy this taxes hardest: it splits
O(leaves) times and its late splits are exactly the small leaves the fast arm
exists for, so at `stat_count = 2` the tree pays ~4x the reorder launches
CatBoost pays, into the launch-tax budget the covtype record already names.
The fix is one hoisted host max -- `max over to_split of leaves[id].size`,
fresh because sizes are rebuilt from `p_sz` each iteration, exactly their
`partitionsCpuPtr[cpuLeafIdsPtr[leaf]].Size` -- passed instead of `n_rows`.
NOT a deviation: it RESTORES their dispatch. The depthwise lane's file, so
relayed through the orchestrator rather than edited here.

## One item that is bigger than this lane, for Andrew

**`IDENTITY_PATHS.md` row 9 records "Apple's FAST baseline measured UNFUSED
(fused 0 / unfused 1,046,394 of 2^20)" as a general property of the Apple
column.** This lane has a seam where that generalization is FALSE: Metal's
FAST codegen fuses the cosine calcer's `DenumSqr` accumulation, measured on
the device, six discriminating shapes, both modes.

Both measurements are correct. They are different expressions in different
kernels, and that is exactly the point: **contraction is a per-seam property
and a probe cannot license a claim about a kernel it did not compile.** Row 9
is the artifact the paper rests on and it currently reads as though the probe
settled the column.

The depthwise lane independently reached the same conclusion and neither of
us has edited that file, because it predates both lanes and is shared. **The
caveat belongs in row 9 itself, not only in two lane records.** Raising it
rather than doing it.
* Whether `FindBestLeafToSplit`'s HOST argmin adds an identity pathway that
  `IDENTITY_PATHS.md` does not enumerate. It compares scores read back from
  the device, and the argmin's tie-break is `<` strict on leaf id, so the
  ORDER is pinned; what is not yet established is whether the scores reaching
  it can differ across backends for a reason other than finding 2. Open.
* Whether `MinLeafSize` going live changes any existing gate. It should not
  -- SymmetricTree never reads it (`:685`) -- but "should not" is not a
  measurement.

## Where the lane stands

| piece | state |
|---|---|
| `ComputeOptimalSplit`, the Lossguide scorer | **PORTED AND GATED**, `check-leafwise-scores`, five teeth |
| leaf selection (`FindBestLeafToSplit`, no sign test, argmin) | **PORTED AND GATED**, `check-lossguide-policy`, six claims incl. a device sign-convention gate |
| the fit loop | **PORTED AND GATED**, `check-lossguide`, six claims -- ONE DRIVER with four policy branches, not a second driver |
| the non-symmetric model + apply | the depthwise lane's, consumed |
| the stage-hash instrument | **LANDED**, `check-identity-trace`, writer and reader gated together; both lanes on it |
| the queue + selected-leaf checkpoints | **LANDED** in `select_leaves_to_split_traced`, gated by `check-lossguide-policy` P7; the one-line driver swap is OWED (depthwise lane's file, requested via orchestrator) |
| stage wall timers | the depthwise lane's `depthwise_stage_times.mojo`, CONSUMED not duplicated; lossguide-branch reach VERIFIED against their committed `045afa3` |
| DEVIATION 318, the multiply-add pin | landed, verified in the emitted AIR |
| single-leaf split kernels | not started -- a PERFORMANCE arm, correctly ordered after the tree |
| CatBoost's own Lossguide values as an oracle | **OWED**, and it cannot be run on this box: their GPU learner does not start on Apple silicon. The NVIDIA column's job. |

**What "done" does NOT yet mean.** Every claim gated here is about the POLICY
being the policy their source describes -- exact `max_leaves`, no sign test,
the `<=` size boundary, a structure that differs from Depthwise on identical
input. **None of it compares a value against CatBoost's own output**, because
CatBoost's GPU arm cannot run here at all. That comparison is the NVIDIA
column's and it is the honest gap in front of any accuracy claim.

## Duplication, after Andrew asked "are we recreating anything?"

A repo-wide audit ran. What it found about THIS lane, and what was done:

* **The stage-hash instrument was built TWICE, by two lanes, inside one
  hour.** Resolved the right way and quickly: the depthwise lane deleted
  theirs and moved onto `core/identity_trace.mojo`, priced on the merits.
  That is the pattern.
* **A docstring of mine claimed a gate that does not exist** -- that the
  symmetric kernel's inlined block argmax and the factored
  `_leafwise_argmax_write` were gated against each other. They are not; the
  check never launches the symmetric kernel. Claim deleted, gap labelled.
  **A comment asserting a gate exists is as load-bearing as the gate, and
  nothing checks comments.**
* Two drivers became one (see above). `splitmix`/`hashed` imported rather
  than copied. `bits`/`bits_f32` collapsed. DEVIATION 319 closed.

**Refuted as duplication:** `TDepthwiseWorkspace` holds only what
`TTreeWorkspace` cannot serve; `greedy_search_helper_lossguide.mojo` is 190
lines, three functions, one import.

**Open, and NOT this lane's to fix:** `estimation_bench.mojo`'s splitmix
carries a different multiplier under the same name (one of thirteen copies);
`core/segmented_sort.mojo` is a declared fork of the gbdt one and has
diverged; and **the vec4 histogram fast paths are symmetric-only, so both
non-symmetric policies silently miss a measured 5.9x** -- the largest single
item on the list.

