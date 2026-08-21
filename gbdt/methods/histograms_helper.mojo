"""When to rebuild a level's histograms from scratch, and how big they are.

PORT OF `catboost/cuda/methods/histograms_helper.h` at CatBoost `54a8143a`,
`TComputeHistogramsHelper`. Transliterated. Do not improve.

This is the object the doc-parallel oblivious searcher holds one of per
FEATURE GROUPING POLICY -- binary, half-byte, one-byte -- and drives once per
depth. It owns almost no arithmetic. What it owns is a four-field state
machine, and that state machine decides the single most consequential bit in
the whole histogram family:

    **BuildFromScratch is the FULL-PASS flag.**

`Compute` passes it to `ComputeHistogram2` as `buildFromScratch`, which
becomes `fullPass` in every kernel below. A full pass computes every part; a
partial pass computes only the SMALLER child of each sibling pair and files
it under the right one, leaving the subtraction to recover the other. Getting
the flag wrong does not crash and does not obviously corrupt: it produces a
histogram that is right for one depth and quietly wrong afterwards.

## The state machine, and the `++` that looks like a bug

    ++CurrentBit;
    if (static_cast<ui32>(CurrentBit) != newSubsets.CurrentDepth
        || CurrentBit == 0) {
        BuildFromScratch = true;
        CurrentBit = newSubsets.CurrentDepth;
    }

`CurrentBit` starts at **-1**, so the first call increments it to 0 and the
`|| CurrentBit == 0` arm fires -- the first level is always a full pass, and
it is that clause and not the inequality that makes it one. Read the
inequality alone and depth 0 looks like a partial pass, because `-1 + 1 == 0`
already equals `newSubsets.CurrentDepth`.

Thereafter the helper ADVANCES ITS OWN COUNTER and checks whether the subsets
agree. They agree exactly when the caller went down one level since the last
call. Any other transition -- a new tree, a re-entry at the same depth, a
skipped level -- disagrees, and the answer is to rebuild.

**So the helper does not need to be told a new tree started.** That is the
design: the only input is the subsets' own depth, and every discontinuity
resolves to a full pass on its own. A port that added a `reset()` call would
work and would be a different program.

## Two different sizes, and they are not the same expression

    ResetHistograms   (1 << MaxDepth)    * FoldCount * binFeatures * 2
    Compute           (1 << CurrentBit)  * FoldCount * binFeatures * 2

The allocation is for the DEEPEST level the tree can reach; the view handed
to the scorer is for the level actually built. Sizing the allocation with
`CurrentBit` works for every level except the ones that grow, which is the
kind of defect that survives a shallow fixture.

DEVIATION: the buffer management is expressed as SIZES here rather than as
`TCudaBuffer::Reset` calls. Their `Reset` on a stripe mapping reallocates
only when the size grows and is otherwise a view change; ours returns the
size and lets the caller own the buffer, because this repository has no
`TCudaBuffer` and a helper that allocated on its own would hide the growth
`ResetHistograms` exists to make explicit.

NOT PORTED HERE: `EnsureHistCompute`, which synchronizes a
`TComputationStream`. There are no streams on Metal (`metal-hardware-gaps`),
so every launch is already ordered on the one queue and the flag it guards
has nothing to guard.
"""


#: `EFeaturesGroupingPolicy`, the three grids a compressed index carries.
#: One `ComputeHistogramsHelper` per policy, which is why `Policy` is a field
#: rather than an argument.
comptime POLICY_BINARY = 0
comptime POLICY_HALF_BYTE = 1
comptime POLICY_ONE_BYTE = 2


def policy_name(policy: Int) raises -> String:
    if policy == POLICY_BINARY:
        return String("BinaryFeatures")
    if policy == POLICY_HALF_BYTE:
        return String("HalfByteFeatures")
    if policy == POLICY_ONE_BYTE:
        return String("OneByteFeatures")
    raise Error("unknown grouping policy " + String(policy))


@fieldwise_init
struct HistogramPlan(Copyable, ImplicitlyCopyable, Movable):
    """What one `Compute` call resolved to, before anything is launched.

    Returned rather than acted on so the decision can be gated without a
    GPU: every field here is a pure function of the call sequence, and the
    sequence is what goes wrong.
    """

    var current_bit: Int
    """`CurrentBit` after the update. The number of splits already in the
    tree, so `1 << current_bit` is the part count at this level."""

    var build_from_scratch: Bool
    """**THE FULL-PASS FLAG.** True computes every part; False computes the
    smaller sibling of each pair and files it under the right one."""

    var part_count: Int
    """`1 << CurrentBit`, their argument `numStats`/`partCount`."""


struct ComputeHistogramsHelper(Copyable, Movable):
    """`TComputeHistogramsHelper` (`histograms_helper.h:18-168`)."""

    var policy: Int
    var fold_count: Int
    var max_depth: Int

    var current_bit: Int
    """`int CurrentBit = -1;` -- and the -1 is load-bearing. See the module
    docstring."""

    var build_from_scratch: Bool
    """`bool BuildFromScratch = true;`"""

    def __init__(
        out self, policy: Int, fold_count: Int, max_depth: Int
    ) raises:
        """Their constructor (`:25-42`) plus the two field initialisers."""
        if fold_count < 1:
            raise Error("fold_count must be at least 1")
        if max_depth < 1:
            raise Error("max_depth must be at least 1")
        self.policy = policy
        self.fold_count = fold_count
        self.max_depth = max_depth
        self.current_bit = -1
        self.build_from_scratch = True

    def plan(mut self, subsets_current_depth: Int) -> HistogramPlan:
        """The first eleven lines of `Compute` (`:43-56`), copied.

        Split out from the launch so the DECISION is a pure function of the
        call sequence and can be gated on the host. The launch itself adds
        no state beyond clearing `BuildFromScratch`, which this does.
        """
        self.current_bit += 1
        if self.current_bit != subsets_current_depth or self.current_bit == 0:
            self.build_from_scratch = True
            self.current_bit = subsets_current_depth

        var was_from_scratch = self.build_from_scratch
        var part_count = 1 << self.current_bit

        # their `BuildFromScratch = false;` after the launch (`:79`). It is
        # inside `if (DataSet->GetGridSize(Policy))` upstream, so a policy
        # with no features never clears it -- transcribed in
        # `clear_from_scratch`, which the caller invokes only when it
        # actually launched.
        return HistogramPlan(self.current_bit, was_from_scratch, part_count)

    def clear_from_scratch(mut self):
        """`BuildFromScratch = false;` (`:79`).

        SEPARATE FROM `plan`, and that is theirs rather than tidiness: the
        assignment sits inside `if (DataSet->GetGridSize(Policy))`, so a
        policy whose grid is empty keeps `BuildFromScratch` set and rebuilds
        on the next call that does have features. Folding it into `plan`
        would clear the flag for a level that never computed anything.
        """
        self.build_from_scratch = False

    def histogram_view_size(self, current_bit: Int, bin_features: Int) -> Int:
        """`(1 << CurrentBit) * FoldCount * features.Size() * 2` (`:57-62`).

        The slice the scorer reads: sized for the level actually built.
        """
        return (1 << current_bit) * self.fold_count * bin_features * 2

    def histogram_alloc_size(self, bin_features: Int) -> Int:
        """`(1 << MaxDepth) * FoldCount * histograms.Size() * 2` (`:147-151`).

        The ALLOCATION, sized for the deepest level the tree can reach. Not
        the same expression as the view -- see the module docstring.
        """
        return (1 << self.max_depth) * self.fold_count * bin_features * 2
