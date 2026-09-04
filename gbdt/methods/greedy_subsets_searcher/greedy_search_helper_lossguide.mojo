# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Which ONE leaf Lossguide splits next, and when it stops.

PORT OF the `EGrowPolicy::Lossguide` arms of
`catboost/cuda/methods/greedy_subsets_searcher/greedy_search_helper.cpp` at
CatBoost `54a8143a`. Transliterated. Do not improve.

============================ DEVIATION 316 ============================
**Theirs is ONE file and ours is three.** `TGreedySearchHelper` holds every
policy's arm in `greedy_search_helper.cpp`; here the symmetric arm is in
`greedy_search_helper.mojo`, the Depthwise arm is in
`greedy_search_helper_depthwise.mojo` and this is the Lossguide arm.

The reason is a checkout state, not a design: `greedy_search_helper.mojo` was
mid-edit by a third session when both non-symmetric lanes opened, and a lane
that edits another lane's live file trades a merge conflict for a silent
overwrite. **This split folds back into `greedy_search_helper.mojo` when that
file is quiet**, and `DERIVATION_MAP.tsv` points all three rows at the same
upstream file so the fold is a rename and not an archaeology exercise.
=======================================================================

===================== DEVIATION 319, CLOSED =====================
**This file used to import `IsTerminalLeaf`, `ShouldTerminate` and
`SelectLeavesToVisit` from the DEPTHWISE lane's file**, to avoid a second
copy of predicates whose whole job is to agree. It no longer does, and the
reason is not tidiness: the driver merge made the depthwise file import THIS
one, and the pair became a CYCLE.

The cycle compiled. It was resolved anyway, by direction: the shared driver
depends on each policy's arm, and no policy arm depends on the driver. This
file therefore imports nothing from it, and the three shared predicates stay
where the driver can reach them. Checks may import from both; a check is a
leaf of the dependency graph and cannot create a cycle.

Their true home is still `greedy_search_helper.mojo`, with everything else
`TGreedySearchHelper` owns -- see DEVIATION 316.
=======================================================================

================= CROSS-VENDOR IDENTITY AUDIT, 2026-08-22 =================
Audited under `IDENTITY_PATHS.md`'s three-move discipline (PIN / REPLACE /
REFUSE) for anything that could move a bit across GPU vendors. **The verdict
is that this file needs NO move**, and the reasons are worth one paragraph
each because each names a hazard class that is ABSENT rather than fixed:

* **No order-dependent operation.** Both scans below iterate a `List` in
  index order -- Mojo `List` iteration is defined order, and no `Dict` or
  hash container appears anywhere on the selection path (a `Dict` here
  would be an iteration-order hazard; there is none to pin).
* **Ties are already deterministic.** The argmin's strict `<` gives an
  exact float tie to the LOWEST leaf index, which is creation order --
  CatBoost's own shape (`greedy_search_helper.cpp:302`), gated by the
  policy check's P3 in both tie positions.
* **NaN is deterministic too, and INVISIBLE.** `gain < best_gain` is
  IEEE-false when `gain` is NaN, on every vendor, so a NaN-gain leaf loses
  every comparison exactly like an undefined one -- their `Score <
  bestScore` verbatim. A rewrite through `max()`/sort could change that
  (max's NaN rule is not `<`'s), so the policy check's P8 pins it.
* **No float arithmetic at all.** Zero multiply-add seams (row 9 audit:
  nothing to route through `identical_mul_add`) and zero cross-kernel
  stores (row 10 audit: nothing to flush through `ftz`). The only float
  OPERANDS here are `best_split.gain` values, PRODUCED upstream -- the
  score kernel (`kernel/compute_scores.mojo`) and the driver's host reduce
  -- so their cross-vendor identity is those files' rows (9/10/12 and the
  target-variance rows), not this one's.

Given bit-identical inputs, this file's outputs are bit-identical across
vendors in BOTH numeric modes; it carries no `BUILD_MODE` branch because it
has nothing to branch on.
============================================================================

WHAT LOSSGUIDE ACTUALLY IS. Four decisions, and only two of them are in this
file because the other two are shared:

  1. `SelectLeavesToSplit`  -> the single best leaf, HERE      (`:319-324`)
  2. `ComputeOptimalSplits` -> `ComputeOptimalSplit`, the kernel
     (`kernel/compute_scores.mojo`'s `compute_optimal_split_kernel`)
  3. `MakeSplit`            -> their single-leaf fast path, or the multi-leaf
     kernels with a one-element list (see `archive/research/LOSSGUIDE.md` finding 3)
  4. `ShouldTerminate`      -> shared, and `MaxLeaves` is where Lossguide
     differs from every other policy in what the number MEANS
"""

from core.identity_trace import IdentityTrace
from gbdt.methods.greedy_subsets_searcher.points_subsets import TLeaf


def find_best_leaf_to_split(leaves: List[TLeaf]) raises -> Int:
    """`FindBestLeafToSplit` (`greedy_search_helper.cpp:296-309`).

        double bestScore = std::numeric_limits<double>::infinity();
        TMaybe<ui32> bestLeaf;
        for (size_t i = 0; i < subsets.Leaves.size(); ++i) {
            if (subsets.Leaves[i].BestSplit.Defined()) {
                if (subsets.Leaves[i].BestSplit.Score < bestScore) {
                    bestScore = subsets.Leaves[i].BestSplit.Score;
                    bestLeaf = i;
                }
            }
        }

    Returns their `TMaybe<ui32>` as `-1` for "nothing" -- every caller here
    tests it, and the one caller that matters is `select_leaves_to_split`
    below, which returns an empty list rather than splitting anything.

    THREE THINGS THAT ARE DECISIONS AND NOT FORMALITIES:

    **The comparison is STRICT `<`, so on a tie the FIRST leaf wins**, and
    leaf ids are creation order (`MakeSplit` keeps the parent's id for the
    left child and appends the right at `leavesCount + i`,
    `split_properties_helper.cpp:872-873`). So a tie between two leaves is
    broken by which was created first, which on a Lossguide tree means the
    one nearer the root of the split ORDER, not of the tree. Reversing the
    comparison to `<=` grows a different tree from the same data, and ties
    are not exotic: two leaves that saw the same rows on a constant feature
    tie exactly.

    **There is NO SIGN TEST.** Depthwise and SymmetricTree take only leaves
    whose `BestSplit.Score < 0`, i.e. only splits that improve
    (`:355-359`). Lossguide takes the best leaf whatever its sign. **A
    Lossguide tree therefore keeps splitting after every remaining split
    makes the objective worse**, until `MaxLeaves` or `IsTerminalLeaf`
    stops it. That is their design and it is what `max_leaves` is FOR. A
    port that "helpfully" added `score < 0` here would grow smaller trees
    than CatBoost on every dataset and would look like a tuning difference
    rather than a defect.

    **`Score`, not `Gain`.** Their argmin reads `BestSplit.Score` while the
    cross-block host reduce keys on `operator<`, which is `Gain`
    (`gpu_structures.h:80-93`). On this path the two fields hold the SAME
    number -- `ComputeOptimalSplit` assigns `bestScore = gain; bestGain =
    gain` (`compute_scores.cu:468-472`) where the OBLIVIOUS kernel assigns
    `bestScore = score; bestGain = gain` (`:142-144`) -- so this port,
    which carries one number, is correct here and would NOT be if these
    records ever came from the oblivious kernel. Checked against their
    source, not assumed.

    ============ THE SIGN, AND IT IS NOT THIS PORT'S ============
    **A `TLeaf.best_split` HOLDS THEIR SIGN, NOT OURS.** The kernel's gain is
    theirs negated (see `kernel/compute_scores.mojo`'s sign block), but the
    HOST REDUCE negates it back before storing:

        var cand = TBestSplitProperties(f, b, -our_gain, -our_gain)
        if best_split_properties_less(cand, best): best = cand

    -- and `best_split_properties_less` is a transcription of their
    `operator<`, which is `Gain <` with LOWER BETTER, over a default record
    whose gain is `Float32.MAX` so that an undefined candidate loses every
    comparison. Both of those only make sense on their sign.

    So the record that reaches this function is CatBoost's own number and
    their code ports VERBATIM: an ARGMIN with strict `<`, seeded at `+inf`.

    **THIS FUNCTION WAS WRITTEN AS AN ARGMAX AND WAS WRONG.** I reasoned "our
    kernel's sign is flipped, so flip the comparison" and never traced the
    value to where it is STORED. The flip happens twice -- once in the kernel
    and once in the reduce -- so it cancels, and a Lossguide tree built on the
    argmax splits the WORST leaf available at every iteration. The lesson is
    the one [[read-their-source-against-ours]] keeps paying for: a sign is a
    property of a VALUE AT A PLACE, and the place is the store, not the
    nearest comment about the kernel.

    Pinned empirically, not by argument: `check-lossguide-policy`'s S1 runs
    the real kernel on a fixture whose best split improves, applies the real
    reduce, and asserts the stored gain is NEGATIVE.
    ==============================================================
    """
    var best_leaf = -1
    var best_gain = Float32.MAX
    for i in range(len(leaves)):
        if leaves[i].best_split.defined:
            # their `< bestScore`, VERBATIM: strictly less wins, so the
            # FIRST leaf keeps a tie exactly as theirs does.
            if leaves[i].best_split.gain < best_gain:
                best_gain = leaves[i].best_split.gain
                best_leaf = i
    return best_leaf


def select_leaves_to_split(leaves: List[TLeaf]) raises -> List[Int]:
    """`SelectLeavesToSplit`'s Lossguide arm (`greedy_search_helper.cpp:319-324`).

        TMaybe<ui32> leafToSplit = FindBestLeafToSplit(subsets);
        for (ui32 leaf = 0; leaf < subsets.Leaves.size(); ++leaf) {
            if (leafToSplit.Defined() && leaf == *leafToSplit) {
                leavesToSplit->push_back(leaf);
            }
        }

    Their loop is a one-element filter written the long way; it produces
    either `[bestLeaf]` or `[]`. Kept as the same two outcomes rather than
    as their loop, because the loop cannot produce a third.

    **AT MOST ONE LEAF PER ITERATION, and that is what bounds the scorer.**
    One split makes two new leaves, both without a `BestSplit`, so the next
    `SelectLeavesToVisit` returns exactly those two -- which is why their
    `CB_ENSURE(leavesToVisit.size() <= 2)` (`:511`) holds and why
    `compute_optimal_split_kernel` takes two scalar leaf ids instead of a
    buffer. The bound is a consequence of THIS function, not an independent
    assumption, and breaking this one breaks that kernel's contract.
    """
    var out = List[Int]()
    var best = find_best_leaf_to_split(leaves)
    if best >= 0:
        out.append(best)
    return out^


def select_leaves_to_split_traced(
    leaves: List[TLeaf],
    mut trace: IdentityTrace,
    tag_prefix: StringSlice,
) raises -> List[Int]:
    """`select_leaves_to_split` with the LEAF QUEUE and the WINNER on the
    identity-trace ladder. Same decision, verbatim -- it delegates.

    NOT A PORT, like everything on the ladder: CatBoost ships one GPU backend
    and needs no cross-backend address for a diverging bit. The two stages
    this adds are the ones `archive/research/LOSSGUIDE.md`'s stage table owed and the driver's
    existing records cannot supply:

    * `<prefix>queue.*` -- the per-leaf `BestSplit` records over ALL leaves,
      which IS Lossguide's priority queue. The driver's `best.*` records
      cover only the <= 2 leaves VISITED this iteration; the argmin below
      reads every leaf, so a stale or clobbered record on an unvisited leaf
      is visible here and nowhere else.
    * `<prefix>selected_leaf` -- which leaf the policy chose. An INTEGER, so
      a cross-backend difference here is structural, and per the ladder's
      ordering rule it is read before any float stage is worth reading.

    THE QUEUE IS RECORDED VERBATIM, STORED FIELDS ONLY. An undefined slot
    holds the default record -- `feature_id = -1` (their own `Defined()`
    sentinel), `bin_id = 0`, `gain = Float32.MAX` (`gpu_structures.h:64`) --
    which both backends construct identically, so no synthesis is needed and
    none is done: what is hashed is the host queue state the argmin actually
    reads. Rule 3 (hash the LOGICAL buffer) is satisfied for free -- the
    queue is host state, not machine-sized scratch.

    `tag_prefix` is the caller's `dN.` iteration prefix; the trace enforces
    tag uniqueness, so passing a constant prefix from a loop raises on the
    second iteration rather than mis-aligning two traces.
    """
    if trace.enabled:
        var qf = List[Int32]()
        var qb = List[Int32]()
        var qg = List[Float32]()
        for i in range(len(leaves)):
            ref bs = leaves[i].best_split
            qf.append(bs.feature_id)
            qb.append(bs.bin_id)
            qg.append(bs.gain)
        trace.record_list_i32(String(tag_prefix) + "queue.feature", qf)
        trace.record_list_i32(String(tag_prefix) + "queue.bin", qb)
        trace.record_list_f32(String(tag_prefix) + "queue.gain", qg)
    var out = select_leaves_to_split(leaves)
    if trace.enabled:
        var sel = List[Int32]()
        if len(out) > 0:
            sel.append(Int32(out[0]))
        else:
            sel.append(Int32(-1))
        trace.record_list_i32(String(tag_prefix) + "selected_leaf", sel)
    return out^


def find_max_depth(leaves: List[TLeaf]) raises -> Int:
    """`FindMaxDepth` (`greedy_search_helper.cpp:311-317`).

    Read twice per iteration by their `ComputeOptimalSplits` and
    `SplitLeaves`, both times only to index `FixedBinarySplits` and to
    number a log line -- neither of which this lane ports. It is here
    because their loop is here and because the Lossguide log line's
    `iteration` is `subsets.Leaves.size()` and NOT this
    (`:596-600`), which is a distinction a reader will otherwise have to
    re-derive from their source.
    """
    var depth = 0
    for i in range(len(leaves)):
        var d = leaves[i].get_depth()
        if d > depth:
            depth = d
    return depth
