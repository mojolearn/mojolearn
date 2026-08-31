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
at its head and a `DERIVATION_MAP.tsv` row pointing at the same upstream file.
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
intent; the driver spells the per-iteration prefix `d<iter>.`, and since
2026-08-23 the boosting loop hands it the `treeNNN.` layer (`tag_prefix`, one
card per fit across every tree; the single-tree gates pass no prefix and
their cards are unchanged).

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
**THE FIX AS FIRST SPECIFIED HERE WAS WRONG, and the depthwise lane caught
it landing it (their commit `bf43e7b`; their board has the record).** The
literal form -- "host max over `to_split` of `leaves[id].size` before the
call" -- reads ZERO at that point: the split-prep loop has already
overwritten the split slot with the LEFT CHILD, whose size their own
`SplitLeaf` resets to 0 (`split_properties_helper.cpp:790`). That form would
take the fast arm ALWAYS and overrun `GatherInplace`'s shared-memory bound.
The landed fix takes the max from the PARENT snapshots inside the split-prep
loop -- the value their `partitionsCpuPtr[cpuLeafIdsPtr[leaf]].Size` read
actually sees -- passed instead of `n_rows`. NOT a deviation: it RESTORES
their dispatch. The finding was real; the call-site spec was the part only
the owner's read could check, which is what single-writer files are for.

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
| the boosting loop and the Python surface | **ON THE SURFACE SINCE 2026-08-23** (DEVIATION 259): `train(grow_policy="Lossguide", max_leaves=, min_data_in_leaf=)` and `GradientBoosting(grow_policy="Lossguide", ...)`, through `doc_parallel_boosting.fit_with_test`'s non-symmetric arm, their `NeedEstimation` estimation, `add_non_symmetric_tree_doc_parallel` apply and `ntree` model text; CatBoost's own GPU default score function there (NewtonL2, `catboost_options.cpp:980-991`) is the Python default; `check-grow-policy` gates it, E2 cells `gbdt_rmse_lossguide`, `_leaves8`, `_cosine` carry the cards, `gbdt_multiclass_lossguide` is REFUSED as their registry refuses it |
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


## The split-cost round, 2026-08-27: DEVIATIONS 1901-1903 (fast-speed series)

Two source recons (LightGBM CUDA, XGBoost GPU hist) converge on the same
verdict about the 1M-2M H100 gap (lossguide 2.42x/2.83x, depthwise 1.91x):
the loss is the cost of ONE SPLIT, multiplied by `max_leaves - 1` sequential
splits per lossguide tree. Three deviations answer it. All of them are
FAST-arm only: 1901 and 1903 behind `SPLIT_COST_IDENTICAL` in
`greedy_search_helper_depthwise.mojo`, 1902 behind its own kernel-matrix
row `ridx_only_splits_for` (False under IDENTICAL on every column). Either
way the IDENTICAL column executes the pre-round path unchanged, so the
merge gate's byte-compare against main passes by construction.

### DEVIATION 1901 -- partition stats propagate from the split record. LANDED

* **Mechanism.** `update_partition_stats_from_split_kernel`
  (`kernel/split_points.mojo`) writes both children's `part_stats` rows at
  split time from the parent's row and the winner's SCANNED histogram cell
  (ordered: cell = left, complement = right; one-hot: inverted, because the
  flag test is equality and one-hot bins are never scanned). The per-level
  `compute_partition_stats` sweep over `d_all_ids` (DEVIATION 352) runs only
  at iteration 1 on the FAST arm, to seed the root. The end-of-tree sweep
  stays in BOTH modes, so leaf values still come from the exact reduction.
* **Price.** O(max_leaves x n_rows x stat_count) per tree becomes
  O(max_leaves) plus one root pass -- at 1M rows x 64 leaves x 2 stats,
  ~128M row-stat reads per tree deleted. In exchange, the gains a level
  scores with ride on re-associated float sums (the histogram-subtraction
  tradeoff applied to partition stats), so FAST trees can differ from
  IDENTICAL trees in low-bit tie regions.
* **Citations.** recon_lightgbm_cuda.md mechanism e2 / borrow 1
  (`cuda_data_partition.cu:798-903`, `cuda_best_split_finder.cu:434-443`);
  our sweep at `greedy_search_helper_depthwise.mojo` (DEVIATION 352 block).
* **Gate note for the orchestrator.** The one-hot inversion and the
  folds<=1 (binary-feature) prefix identity are the two places a sign error
  hides; both are exercised only by fixtures with one-hot / binary winners.
  A cheap probe: FAST vs IDENTICAL part_stats trace rows (`dN.partstats`)
  on a 4k-row fixture should differ by ULPs, not by sides.

### DEVIATION 1903 -- histogram aliasing: no per-split copy, no blanket zero. LANDED

* **Mechanism.** LightGBM's `cuda_hist_pool_` aliases the parent's slot to
  the larger child and zeroes the arena once per tree
  (`cuda_data_partition.cu:825-831,895-898`,
  `cuda_histogram_constructor.cpp:76-80` -- recon mechanism a3 / borrow 3).
  This port's slots are leaf-id-indexed by every kernel from the build to
  the scorer, so a pointer pool proper would touch the other lane's files;
  the FAST arm instead exploits the aliasing the id scheme already gives:
  the left child KEEPS the parent's id, so the parent's histogram already
  sits where an in-place `big == left` subtraction needs it. The
  unconditional split-time `copy_histograms` launch is deleted on the FAST
  arm; a copy runs at PLAN time only for `big == right` pairs (left
  strictly smaller -- ties derive the left sibling, PORTING.md 136, so
  balanced splits copy NOTHING). The per-level zero pass runs only over
  compute slots that were ever written this tree (`hist_slot_dirty`);
  fresh right-child slots still hold the once-per-tree memset's zeros.
* **Price.** Deleted per split on the FAST arm: one full-histogram copy
  (`hist_cells x stat_count x 4 B` read + write) for every tied/left-heavy
  pair, and one full-histogram zero for every fresh slot -- x ~63 splits
  per lossguide tree. Bit-inert by construction (same bytes at every read
  as the old schedule), so unlike 1901 this deviation cannot move FAST
  trees; it only removes launches and traffic.
* **Citations.** recon_lightgbm_cuda.md mechanism a3;
  `copy_histograms[_vec4]_kernel` / `zero_histograms_kernel`
  (`kernel/histogram_utils.mojo`, unmodified -- only the launch schedule in
  `greedy_search_helper_depthwise.mojo` moved).
* **Gate note.** The `hist_slot_dirty` proof rests on two invariants: leaf
  ids are never reused within a tree, and the arena memset at
  `CreateInitialSubsets` runs once per fit call. Both hold today in this
  driver; a future slot-reuse scheme must revisit the elision.

### DEVIATION 1902 -- move only the row index at a split. BUILT 2026-08-28, WIRING OWED

**Status: every kernel-side and launcher-side edit is in the tree, on
`integrate/dev1902-ridx-only-splits`; the three call-site binds in the
non-symmetric driver are OWED** -- `greedy_search_helper_depthwise.mojo`
is another agent's this round, so the binds are spelled below for the
orchestrator, the same hand-off shape as the traced-select swap. Until
they land, every built arm is instantiable and dead and NOTHING in any
build changes: all new comptime parameters default to the old body.

* **The mechanism, as built.** The stat planes stay in the order the
  objective writes them for the life of the fit -- exactly as the
  compressed index always has -- and only `row_index` (4 B/row) is
  permuted at a split. Every stat reader on the non-symmetric path
  gathers `stats[stat * line + row_index[pos]]` through the SAME index
  register those kernels already load for the compressed-index gather.
  LightGBM (`cuda_data_partition.cu:288-334, 679-783, 907-944`, hist
  gather `cuda_histogram_constructor.cu:53-55`) and XGBoost
  (`row_partitioner.cuh:112-201`, gather `histogram.cu:186,212`) are both
  this design; CatBoost moves the stat columns, which is why this is a
  numbered deviation and not a port.
* **What was built, by file.**
  1. `kernel/split_points_ridx.mojo` (NEW): `launch_reorder_index_only`
     -- the index half of `TSplitPointsKernel::Run`, their `> 1024`
     dispatch kept. Fast arm: `gather_inplace_kernel` at `grid.x = 1`
     (was `1 + statCount`), ONE launch. Slow arm: the index copy/gather
     pair alone, TWO launches (was `2 * ceil(statCount / 8) + 2`);
     `new_stats` scratch is dead on this route. Kernels imported from
     `split_points.mojo` UNCHANGED.
  2. The five hist GATHER kernels (`hist_one_byte`, `hist_2_one_byte_base`,
     `hist_2_one_byte_8bit`, `hist_binary`, `hist_half_byte`): defaulted
     comptime `ridx_stats: Bool = False`. True redirects ONLY the stat
     loads -- peel reads `stats_p + hrow`, the main loop trades the
     4-wide contiguous stat load for per-element gathers through the
     `vi[e]` the bin gather already holds. Dither keys stay
     position-based. Direct (depth-0) kernels untouched: the root is
     pre-split and identity-indexed.
  3. `greedy_search_helper.mojo` launchers (`launch_histograms_for_blocks`,
     `launch_hist2_8bit`, `launch_one_byte`, `launch_hist2_one_byte`):
     the same defaulted `ridx_stats`, bound onto every gather enqueue.
  4. `gpu_util/kernel/partition_stats_gather.mojo` (NEW):
     `compute_partition_stats_gather` -- phase 1 transcribed with the one
     gathered load (cross-referenced both ways with its twin), phase 2
     and the pinned chunk formula IMPORTED from `partitions_reduce.mojo`.
  5. `original/kernel_matrix.mojo`: `ridx_only_splits_for[column,
     identical]` -- the named row. True on apple/nvidia/amd/amd-rdna
     under FAST; False under IDENTICAL on EVERY column, so the identical
     route keeps the stat-moving path byte for byte and the merge gate's
     byte-compare passes by construction. Unlike 1907's row there is no
     vendor guarantee to decline (plain loads, no spin), so Apple is in.
  6. Design point 5 (the writers) needed NO edit: the objective writes
     the planes fresh in document order at tree start and `row_index` is
     re-seeded to the identity -- the depth-0 DIRECT kernels already
     depend on exactly that invariant, so it is enforced, not assumed.
* **THE OWED WIRING, three binds in `greedy_search_helper_depthwise.mojo`**
  (plus one comptime at its head:
  `comptime RIDX_ONLY_SPLITS = ridx_only_splits_for[TARGET_COLUMN,
  GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL]()`):
  1. Hist build (`launch_histograms_for_blocks[hist2_smem_mode](` at
     ~:1371) becomes
     `launch_histograms_for_blocks[hist2_smem_mode, RIDX_ONLY_SPLITS](`.
  2. Split apply (~:2154): under `comptime if RIDX_ONLY_SPLITS:` call
     `launch_reorder_index_only(ctx, n_split, max_split_rows, d_left,
     p_off, p_sz, row_index, new_index, gmap, sm_count=sm_count)` in
     place of `launch_reorder_in_leaves(...)` (else-arm unchanged); the
     returned launch count feeds the same `mgr.stream_kernel()` loop.
  3. End-of-tree leaf-value sweep (~:2302): under the same guard call
     `compute_partition_stats_gather(ctx, len(leaves), n_rows,
     stat_count, n_rows, d_ids, p_off, p_sz, stats, row_index,
     stat_partials, part_stats, sm_count=sm_count)`.
  The iteration-1 root seed sweep (~:1530) needs NO bind: it runs before
  any split, over the identity index, and the contiguous read is already
  exact. `compute_target_std_dev` (~:1122) likewise: pre-split, full
  plane. The symmetric driver is NOT converted -- 1902 is the
  non-symmetric policies' deviation and the defaults keep it byte for
  byte.
* **Bit-exactness, argued once (`split_points_ridx.mojo` banner) and
  relied on everywhere:** the old schedule's permuted plane satisfies
  `P[stat][pos] == D[stat][ridx[pos]]` at every point of the fit (true at
  tree start; preserved because splits permute `P` and `ridx` with the
  same `gather_map`, and a permutation of floats moves bytes, never
  re-rounds). Converted readers load the SAME BITS at the same loop
  iteration on the same thread, in the same accumulation order, under the
  same position-keyed dither. So the FAST model pre/post wiring must be
  BYTE-IDENTICAL -- 1902 is bit-inert like 1903, not re-associating like
  1901 -- and that claim is itself the cheapest gate (below). Guarded
  FAST-only by the row anyway, per the round's orders: the argument
  becomes a measurement only when the A/B byte-compare runs. One trace
  row legitimately moves: `dN.stats` now hashes a STATIONARY plane, so a
  pre/post trace diff must show every other row identical and only
  `dN.stats` differing -- a differing `rowindex`, `hist.*`, `partstats`
  or `model.*` row falsifies the invariant and names the stage.
* **What the orchestrator must gate, in order.**
  1. IDENTICAL byte-compare against main (passes by construction; run it
     anyway -- it is the claim the guards exist for).
  2. FAST model byte-compare, wired vs unwired, same commit, same device,
     lossguide AND depthwise fixtures -- the transcription-exactness
     claim. Include a fixture with one-hot/binary winners (sub-byte
     kernels' arms) and one over 1024 rows per leaf at some level (slow
     arm) and one under (inplace arm), or an arm ships ungated.
  3. The uniform-data trap ([[uniform-test-data-hides-permutation]]):
     the byte-compare fixtures must carry HASHED per-row stat values --
     a uniform plane makes a dropped permutation invisible, and 1902 is
     EXACTLY a permutation-accounting change.
  4. The price: 1M/2M lossguide rungs intra-leg (FAST, routed vs not, one
     thermal window per [[the-M4-drifts]] on the Mac leg). The hist
     gather's lost coalescing on the stat stream is the risk; both
     competitors eat it and win, but if it costs more than the reorder
     saves at 255 bins x 2 stats, the row flips to False on that column
     and the deviation dies on the bench, not in review.
* **Interaction with the round's other deviations.** 1901: its
  `update_partition_stats_from_split_kernel` reads histograms and
  `part_stats`, never row stats -- untouched; it also already cut the
  per-level sweeps to two per tree, which is why the gathered sweep's
  marginal cost is small. 1903: histogram aliasing, no stat-plane
  contact -- orthogonal. 1904 (`leaf_winner_fold_kernel`): reads the
  scanned histogram and the split record -- untouched. 1907: produces
  the `gather_map`; under 1902 the SAME map permutes one column instead
  of `stat_count + 1`, so 1907's single-pass partition and 1902 compose
  multiplicatively on the over-500k levels. Land order 1901 -> 1903 ->
  1902 held.
* **The bill today.** On any splitting leaf over 1024 rows,
  `launch_reorder_in_leaves` (`kernel/split_points.mojo:751-931` before this
  round's insertions) runs the slow arm: `2 * ceil(stat_count / 8) + 2`
  launches that move every stat column PLUS the row index out to scratch and
  gather them back -- `(stat_count + 1) * 4` bytes per row, twice. At 1M-5M
  rows the early, expensive splits are ALL over 1024 rows, and lossguide
  pays the arm ~63 sequential times per tree. Both recons converge on the
  alternative: LightGBM moves 4 bytes/row of the split leaf in ~4 lean
  kernels and NEVER moves a gradient (`cuda_data_partition.cu:288-334,
  679-783, 907-944`; its hist kernel gathers through `data_indices_in_leaf`,
  `cuda_histogram_constructor.cu:53-55`); XGBoost's `RowPartitioner`
  is the same design (`row_partitioner.cuh:112-201`, gather at
  `histogram.cu:186,212`).
* **The mechanism, on this port's planes.** Stop permuting `stats` at a
  split; keep permuting `row_index` (4 B/row, the existing partition +
  index copy/gather pair, or the inplace arm). Every consumer that today
  reads `stats` CONTIGUOUSLY over a partition range `[offset, offset+size)`
  instead gathers `stats[row_index[offset + i] + stat * line_size]` -- the
  same indirection those kernels ALREADY perform for the compressed index
  through `load_indices`. `stats` then holds document order for the life of
  the fit, exactly as the compressed index does, and the split's payload
  drops from `(stat_count + 1)` columns twice to one column twice (or once,
  on the inplace arm).
* **The edits, by file and owner.**
  1. `kernel/hist_2_one_byte_base.mojo` + the `hist_2_one_byte_{5,6,7,8}bit`
     twins and `hist_binary` / `hist_half_byte`: the stat loads become
     gathers through the already-loaded row index (they load
     `indices[p_offset + pe]` for the cindex TODAY -- the same register
     feeds the stat address). OTHER LANE (hist_2_* is named out of bounds
     for this round).
  2. `greedy_search_helper.mojo` (`launch_histograms_for_blocks`, the
     symmetric driver's reorder call sites, `compute_target_std_dev` if it
     reads per-partition): plumb the flag/remove the stat reorder launches.
     OTHER LANE.
  3. `gpu_util/partitions_reduce.mojo` (`compute_partition_stats`): gather
     instead of contiguous read. Under 1901's FAST arm this runs twice per
     tree (root seed + leaf values), so its marginal cost is small. OTHER
     LANE'S FILE by this round's boundary.
  4. `kernel/split_points.mojo` + the non-symmetric driver (THIS lane):
     delete the stat copy/gather launches from `launch_reorder_in_leaves`'s
     slow arm and the stat blocks from `gather_inplace_kernel`'s grid
     (`grid.x` drops from `1 + statCount` to 1); `new_stats` scratch dies.
  5. The estimation/bootstrap writers (`doc_parallel_boosting` and the
     objective): they must WRITE the stat planes in the order the readers
     now assume (document order), not the permuted order. This is the one
     place the design can silently break: today the planes are re-permuted
     every split so writer and reader agree by construction.
* **Staging plan (why this is LAST).** Behind the same
  `SPLIT_COST_IDENTICAL` gate, one consumer at a time, each staged against
  the uniform-data trap ([[uniform-test-data-hides-permutation]]): plant
  HASHED per-row stat values and compare per cell, because a uniform plane
  makes a dropped permutation invisible. The hist gather is the risky read
  (coalescing changes from contiguous to indirect on exactly the hot
  kernel); the recons' answer is that both competitors eat that gather and
  win anyway, but it must be MEASURED, not assumed -- if the gather costs
  more than the reorder saves at 255 bins x 2 stats, the deviation dies on
  the bench, not in review.
* **Interaction with 1901/1903.** Independent of both: 1901 removes the
  per-level stats sweep (fewer contiguous readers to convert), 1903 does
  not touch the stat planes at all. Land order 1901 -> 1903 -> 1902 stands.

## The quantized-histogram round, 2026-08-28: DEVIATIONS 1911-1914

The recons' remaining histogram borrow, landed as ONE new kernel family
plus its launcher: XGBoost's integer-quantized gradients with ONE
shared-memory histogram per block (recon_xgboost_gpu.md a / borrows 2, 3,
5) and LightGBM's packed pair + grid floor (recon_lightgbm_cuda.md b2 and
the `min_grid_dim_y` note). All FAST-only: the routing row
(`greedy_quantized_hist_for`, `original/kernel_matrix.mojo`) is comptime
False under IDENTICAL, so the IDENTICAL column never elaborates the branch
and its schedule -- and the merge gate's byte-compare -- are untouched by
construction. DEV 1915 was reserved for this round and is UNUSED.

**Files:** `kernel/hist_quantized_shared.mojo` (new: the kernels),
`quantized_hist_launcher.mojo` (new: shape test, grid math, launch chain),
`greedy_search_helper_depthwise.mojo` (workspace planes + dispatch),
`original/kernel_matrix.mojo` (two rows). NOTE FOR BOTH LANES: the
non-symmetric driver serves Depthwise AND Lossguide, so this round moves
the Lossguide FAST histogram arm too -- deliberately; the leaf-choice
variance candidate below is a Lossguide finding.

### DEVIATION 1911 -- fixed-point gradient pairs, packed one word per row. LANDED

* **Scheme, precisely.** Per level, `quantize_pair_kernel` converts both
  stat planes of every row in the partitions being built to fixed point:
  `q = hist2_quantize(stat, fixed_scale, hist2_dither(position))` -- the
  package's standing dithered-floor quantizer (this port's measured
  stand-in for XGBoost's rounding constant; truncation and
  round-to-nearest both failed measurably, its docstring has the numbers)
  at the standing per-tree scale (`choose_scale` of the plane magnitudes
  -- XGBoost derives its scale per round from clipped magnitude sums,
  same role, `quantiser.cuh` / `histogram.cu:90-115`). The two Int32 land
  in one UInt64 via SIMD bitcast (lane 0 = plane 0/weight-hess, lane 1 =
  plane 1/gradient -- their grad-high/hess-low), no shifts anywhere, so
  the sign-extension pack trap cannot occur. The pack is a LOAD format
  only: accumulation stays per-stat Int32, so there is no cross-half
  carry and none of LightGBM's width-ladder overflow machinery.
* **Why per level, not per round.** XGBoost quantizes once per round and
  never moves gradients; this port re-permutes the stat planes at every
  split, so the packed plane is rebuilt per level over exactly the rows
  being built (one 8 B/row write+read per level). DEVIATION 1902 is what
  makes it per-round; until then this cost is stated, not hidden.
* **Bit relationship to the standing arms.** The addends are bit-for-bit
  the values the shared-Int32 / fused-8-bit arms compute inline (same
  positions, same dither draws, same scale), and integer addition is
  associative -- so per-cell integer totals equal that arm's exactly.

### DEVIATION 1912 -- ONE shared histogram per block, integer atomics, one flush. LANDED

* **Mechanism.** `qh_hist[_gather]_kernel`: one Int32 histogram copy per
  thread block in threadgroup memory (layout `[feature][bin][stat]`,
  interleaved pair -- LightGBM's), every thread accumulating with
  threadgroup integer `Atomic.fetch_add` (relaxed, DEVIATION 1898's
  standing note), one global flush per block into a per-leaf Int32
  accumulator (`d_qacc`, dense-leaf-major), `qh_write_hist_kernel`
  dequantizing accumulator -> flat float histogram with the bridge's
  exact expression (`Float32(Int(q)) / fixed_scale`, ftz'd) and zeroing
  what it reads (self-cleaning; the one blanket memset is at pool
  allocation). Dequantize-on-flush: scan/subtract/score/split read the
  flat histogram unchanged and cannot tell which arm built it.
* **Why this is THE portable shared-memory design.** CatBoost's
  warp-private float slices cost 128 B of shared memory PER THREAD;
  the shared-copy design costs bytes PER BIN -- but a float shared copy
  needs threadgroup FLOAT atomics, which Metal does not have. Metal DOES
  have threadgroup INTEGER atomics (every column does,
  `column_has_threadgroup_int_atomics`), which is exactly why the
  quantized integer histogram is the one shared-copy design that ships
  from one source on every vendor. Lane-agnostic by construction (no
  lane-indexed slices, block barriers only): 32-lane and 64-lane columns
  compile the same file -- the DEVIATION 1906/1910 refusals do not apply.
* **Expected FAST numeric effect, and why accuracy holds.** On the
  NVIDIA/AMD FAST columns the histogram arithmetic changes from
  CatBoost's order-nondeterministic float atomics to dithered fixed-point
  Int32 -- the SAME arithmetic the Apple FAST arm
  (`HIST_SMEM_SHARED2_I32`) and the IDENTICAL column already run, whose
  accuracy is already gated (254-border oracle reproduces CatBoost mse to
  7 decimals under the row-count-aware scale; XGBoost ships quantized
  histograms as its ONLY GPU mode). Zero-mean dithered cell error grows
  as sqrt(rows); overflow is impossible under `choose_scale`'s bound
  (every cell is a partial over a subset of all rows; stated once in the
  kernel file's banner). Side effect worth gating FOR: the FAST histogram
  becomes deterministic run to run, which deletes candidate 1 of the
  recon's 2.6x lossguide round-to-round variance list (float-atomic ULP
  wiggle flipping the argmin's leaf choice).

### DEVIATION 1913 -- feature groups sized to the COLUMN's threadgroup memory. LANDED

* **Mechanism.** `quantized_hist_group_features_for` (kernel-matrix row):
  group = `column_shared_limit // 2 KB-per-feature`, floored to whole
  cindex words -- Apple/32 KB -> 16 features per block, NVIDIA/48 KB ->
  24, AMD/64 KB -> 32, never NVIDIA's 227 KB opt-in and never a
  hardcoded vendor number. Grid x = feature groups x row-replicas
  (XGBoost's 2D grid, `histogram.cu:420`); each group's blocks read the
  packed-pair plane once for the whole group, so full passes over the
  gradients drop from `ceil(F/4)` (one cindex word per block today) to
  `ceil(F/G)`.
* **Shape refusals, at the launcher.** `quantized_hist_shape_ok`: ONE-BYTE
  policy blocks only (the packed decode is 4 one-byte features per word;
  half-byte/binary words would read garbage bins -- DEVIATION 1910's own
  semantics argument) and exactly TWO stat planes (the packed word holds
  two; multiclass keeps the PASS route). Refused shapes and unclaimed
  columns run `launch_histograms_for_blocks` byte for byte.

### DEVIATION 1914 -- grid floors. LANDED

* **Mechanism.** Two cited constants in `qh_replicas` /
  `active_block_count`: XGBoost's `kMinItemsPerBlock = 8192`
  (`histogram.cu:405-419`) caps replicas at what the rows can feed --
  host-side by the mean partition, in-kernel exactly per partition (idle
  replicas return before touching shared memory) -- and LightGBM's
  `min_grid_dim_y_ = 160` (`cuda_histogram_constructor.hpp:152`) floors
  the total launch so late small-leaf levels stay device-filling instead
  of launch-bound. The occupancy target between them is CatBoost's own
  `2 * SMCount` (gather arm doubled), restated because this family's grid
  has no stat axis. FAST-only by reachability, so the device's own
  `sm_count` is correct (no IDENTICAL pin needed).

### What the orchestrator must gate (this round)

1. **IDENTICAL byte-compare against main** -- must pass by construction
   (comptime row False; verified locally: the IDENTICAL build compiles
   with the branch folded away).
2. **FAST cell-for-cell histogram check**: quantized arm vs the
   shared-Int32/fused-8-bit arm on the same fixture must be BIT-identical
   per cell (same addends, associative adds) -- the check that catches an
   addressing error in the new group/flush arithmetic. A hashed-value
   fixture, per [[uniform-test-data-hides-permutation]].
3. **FAST accuracy vs the float-atomic arm** on NVIDIA/AMD (mse parity
   class, the 254-border oracle shape) -- quantization legally moves FAST
   numerics; expected magnitude is the Apple arm's precedent (ULP-class
   split ties, mse parity to several decimals).
4. **Speed A/B** depthwise + lossguide on H100/MI325X at 1M/2M, and the
   Mac row -- the routing row is where this flips back per column if the
   number disagrees.
5. **Determinism probe**: two same-process FAST lossguide runs should now
   produce identical trees on NVIDIA (they could not under float
   atomics); if the 6.0-15.8 s spread survives, candidate 1 is dead and
   candidates 2/3 take over.

### Unfinished, precisely

* No register-blocked ILP in the row loop (XGBoost stages 8 items/thread
  in registers before the atomics, `histogram.cu:197-233`); the first
  profile decides whether it is needed.
* The quantize pass is per-level (see 1911); per-round arrives with 1902.
* One-cell placeholder planes still allocate when the family is dead;
  `d_qstats`/`d_qacc` cost `8 B/row + 4 B/cell` when live (~8 MB + ~3 MB
  at 1M rows x 100 features x 254 folds x 2 stats).
* Multi-stat (multiclass) and sub-byte shapes stay on the standing arms
  by refusal; a packed multi-stat design was not attempted.

### THE AUG 28 ROUND'S GATE RECORD (DEV 1902 wiring + 1911-1914 + 1921-1923, orchestrator, one consolidated pass)

* Parse compile of `original/depthwise_check.mojo`: 0 errors on Apple
  FAST, Apple IDENTICAL, NVIDIA FAST, AMD 64-lane FAST.
* Seeded A/B byte-compare vs main `3d65d70`, same device, same window
  (200k x 24 rng(11), 20 trees depth 6 seed 7): SymmetricTree, Depthwise,
  Lossguide all BYTE-IDENTICAL -- 1902's transcription-exactness and
  1911's addend-equality claims hold on the shipped binding, quantized
  route live.
* Extended fixture per the 1902 gate spec (120k x 20: 8 continuous + 6
  binary + 6 twelve-value columns, so qh REFUSES the shape and the
  STANDING launchers run the five inline `ridx_stats` gather arms; depth 4
  = over-1024 slow reorder arm, depth 10 = inplace arm): Depthwise d4/d10,
  Lossguide d10 all BYTE-IDENTICAL vs a main-built binding.
* Reach, per branch. Sabotage A (`qh_write_hist_kernel` dequantize
  `Int(q) + 1`): Depthwise DIFFERS (max 2.87e-1, all rows), Symmetric
  BYTE-IDENTICAL (correct -- its driver is not converted). Lossguide
  stayed byte-identical UNDER SABOTAGE while a host print probe showed
  `quantized_built=True` on EVERY lossguide iteration -- the route is
  reached; the poison (+1/fixed_scale on nonzero cells only, the `if q !=
  0` self-cleaning branch) sat below every argmax flip threshold on the
  fixture. SABOTAGE-DESIGN NOTE for the next round: an additive
  1-quantum poison is too weak for well-separated lossguide gains; use a
  multiplicative poison. Sabotage B (`launch_reorder_index_only` guard
  `<= 0` -> `<= 1`, skipping single-split reorders -- code only the NEW
  route executes): Depthwise DIFFERS (max 8.5e-1), Lossguide DIFFERS
  (max 1.44), Symmetric BYTE-IDENTICAL. 1902 live on both non-symmetric
  policies.
* `check-depthwise`: all 7 claims OK on FAST and on
  `-D MOJOLEARN_NUMERIC_IDENTICAL=1`.
* The 1902 x 1911 composition seam (found and closed at integration):
  `qh_hist_gather_kernel` loads the packed pair POSITIONALLY, correct only
  while the stat plane is permuted alongside. Under RIDX_ONLY_SPLITS the
  quantize pass gathers instead -- value through `indices[pos]`, dither
  still keyed on the storage position -- so `q_stats[pos]` holds
  bit-for-bit the pair the permuted plane would have held and the gather
  kernel stays untouched.
* NOT measured here: speed. The 1M/2M/5M price rungs on all three vendors
  are the measuring lane's, per the round's division of labor; every row
  above names its flip-back point.
