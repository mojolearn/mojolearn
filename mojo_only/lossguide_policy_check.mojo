"""Which leaf Lossguide splits, proved against Depthwise on the same input.

    pixi run check-lossguide-policy

NO CATBOOST COUNTERPART: a gate, so `mojo_only/`. HOST ONLY -- it creates no
`DeviceContext`, launches nothing, and runs in milliseconds, so it is safe to
run inside another lane's timing window.

WHAT IS UNDER TEST. `find_best_leaf_to_split` and `select_leaves_to_split`
from `greedy_search_helper_lossguide.mojo`, this repository's port of
`greedy_search_helper.cpp:296-324`, plus the two shared predicates the
Lossguide loop depends on for its termination.

THE GATE THAT MATTERS IS DIFFERENTIAL, not analytic. Every property below is
stated as *Lossguide against Depthwise on the identical leaf list*, using the
depthwise lane's own `select_leaves_to_split` as the contrast. That is worth
more than a transcription check: the two arms are twenty lines apart in
CatBoost's source and the ways a port goes wrong are exactly the ways one arm
quietly acquires the other's rule.

  P1  AT MOST ONE LEAF, ALWAYS. Lossguide splits one leaf per iteration
      (`:319-324`) and Depthwise splits every improving leaf (`:355-359`).
      On a list with many improving leaves, Lossguide returns 1 and
      Depthwise returns many. **This is also the invariant
      `compute_optimal_split_kernel` depends on**: one split makes two new
      leaves, so their `CB_ENSURE(leavesToVisit.size() <= 2)` (`:511`) is a
      CONSEQUENCE of this function and not a separate assumption.

  P2  NO SIGN TEST. This is the property most likely to be "fixed" by a
      well-meaning reader. On a list where EVERY leaf's best split makes the
      objective worse, **Depthwise returns nothing and Lossguide still
      returns one leaf.** CatBoost grows Lossguide trees past the point of
      improvement and stops on `MaxLeaves`, not on the sign of the gain.

  P3  THE TIE GOES TO THE FIRST LEAF. Their comparison is strict `<`
      (`:302`), so among equal scores the lowest leaf id wins, and leaf ids
      are creation order. Planted as an exact float tie, then re-planted at
      a different position to prove the answer follows the ID and not the
      position of the maximum in the scan.

  P4  UNDEFINED LEAVES ARE INVISIBLE. `BestSplit.Defined()` guards the
      argmin (`:299`), and a leaf whose scorer wrote the poison record has
      an undefined split. A port that treated the poison sentinel as a very
      good score would split the one leaf that has no usable split at all.
      Includes the all-undefined case, which must select NOTHING rather
      than leaf 0.

  P5  TERMINATION. `ShouldTerminate`'s `MaxLeaves` bound is the one that
      makes `max_leaves` mean something under Lossguide and nothing under
      every other policy, and `IsTerminalLeaf`'s size test is `<=`, so
      `min_data_in_leaf = 1` makes a ONE-ROW leaf terminal. Both boundaries
      are checked on both sides.

  P6  THE `<= 2` CONTRACT, REPLAYED, AND THE STATE THAT BREAKS IT. One
      Lossguide iteration is replayed from the loop's real precondition --
      every leaf already carrying a best split -- and `select_leaves_to_visit`
      must return EXACTLY two afterwards. This is the property the score
      kernel's two-scalar signature rests on.

      The same replay from a tree holding one UNDEFINED leaf returns THREE,
      and that is not a fixture artifact: a leaf whose scorer wrote the
      poison record keeps an undefined split, `MarkTerminal` re-evaluates
      only the two new children (`:657-659`), and so **CatBoost's own
      `CB_ENSURE(leavesToVisit.size() <= 2)` (`:511`) would fire.** The gate
      asserts the count IS three, so the fragility is pinned rather than
      papered over, and the Lossguide caller must raise where theirs raises
      instead of clamping three leaves into a two-scalar kernel.

THE SABOTAGES. P2 and P3 are applied to the EXPECTATION as well: the file
asserts that the Depthwise arm DOES reject the all-worse list and that it
returns more than one leaf on the many-improving list. If those two
assertions ever pass trivially, the contrast has gone dead and the
differential gates are measuring nothing.
"""

from gbdt.methods.greedy_subsets_searcher.greedy_search_helper_depthwise import (
    is_terminal_leaf,
    select_leaves_to_split as depthwise_select_leaves_to_split,
    select_leaves_to_visit,
    should_terminate,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper_lossguide import (
    find_best_leaf_to_split,
    find_max_depth,
    select_leaves_to_split,
)
from gbdt.methods.greedy_subsets_searcher.points_subsets import (
    TBestSplitProperties,
    TLeaf,
)
from gbdt.methods.greedy_subsets_searcher.structure_searcher_options import (
    TTreeStructureSearcherOptions,
)
from gbdt.options.catboost_options import GROW_LOSSGUIDE, GROW_SYMMETRIC


def leaf(gain: Float32, defined: Bool, size: Int = 100) raises -> TLeaf:
    """One leaf carrying a planted best split. `gain` is in THIS PORT'S
    sign, where larger is better and their `Score < 0` is our `gain > 0`."""
    var l = TLeaf()
    l.size = size
    if defined:
        l.update_best_split(
            TBestSplitProperties(Int32(7), Int32(3), gain, gain)
        )
    return l^


def undefined_leaf(size: Int = 100) raises -> TLeaf:
    return leaf(Float32(0.0), False, size)


def lossguide_options() raises -> TTreeStructureSearcherOptions:
    var o = TTreeStructureSearcherOptions()
    o.policy = GROW_LOSSGUIDE
    o.max_leaves = 8
    o.max_depth = 6
    o.min_leaf_size = Float64(1.0)
    return o^


def check_lossguide_policy() raises:
    var failures = 0

    print("-- P1: at most one leaf, where Depthwise takes many --")
    var many = List[TLeaf]()
    # four leaves that all improve (our sign: gain > 0), one that does not
    many.append(leaf(Float32(0.5), True))
    many.append(leaf(Float32(2.5), True))
    many.append(leaf(Float32(1.5), True))
    many.append(leaf(Float32(-0.25), True))
    many.append(leaf(Float32(0.75), True))

    var lg = select_leaves_to_split(many)
    var dw = depthwise_select_leaves_to_split(many)
    if len(lg) != 1:
        print("  FAIL Lossguide selected", len(lg), "leaves, want exactly 1")
        failures += 1
    elif lg[0] != 1:
        print("  FAIL Lossguide picked leaf", lg[0], "want 1 (gain 2.5)")
        failures += 1
    elif len(dw) < 2:
        # THE CONTRAST MUST BE LIVE. If Depthwise also returned one leaf
        # this comparison would be measuring nothing.
        print(
            "  FAIL the contrast is dead: Depthwise returned", len(dw),
            "leaves on a list built to give it four",
        )
        failures += 1
    else:
        print(
            "  ok   Lossguide 1 leaf (id", lg[0], "), Depthwise", len(dw),
            "leaves, same input",
        )

    print()
    print("-- P2: NO SIGN TEST, where Depthwise refuses outright --")
    var all_worse = List[TLeaf]()
    all_worse.append(leaf(Float32(-3.0), True))
    all_worse.append(leaf(Float32(-0.5), True))
    all_worse.append(leaf(Float32(-1.75), True))

    var lg2 = select_leaves_to_split(all_worse)
    var dw2 = depthwise_select_leaves_to_split(all_worse)
    if len(dw2) != 0:
        print(
            "  FAIL the contrast is dead: Depthwise split", len(dw2),
            "leaves whose splits all make the objective worse",
        )
        failures += 1
    elif len(lg2) != 1:
        print(
            "  FAIL Lossguide returned", len(lg2),
            "on an all-worse list; it must still split the least-bad leaf",
        )
        failures += 1
    elif lg2[0] != 1:
        print("  FAIL Lossguide picked leaf", lg2[0], "want 1 (gain -0.5)")
        failures += 1
    else:
        print(
            "  ok   every split makes it worse: Depthwise 0, Lossguide 1"
            " (the least-bad leaf, id", lg2[0], ")",
        )

    print()
    print("-- P3: an exact tie goes to the FIRST leaf --")
    var tie_a = List[TLeaf]()
    tie_a.append(leaf(Float32(1.0), True))
    tie_a.append(leaf(Float32(4.0), True))
    tie_a.append(leaf(Float32(4.0), True))
    tie_a.append(leaf(Float32(2.0), True))
    if find_best_leaf_to_split(tie_a) != 1:
        print(
            "  FAIL tie at ids 1,2 went to",
            find_best_leaf_to_split(tie_a), "want 1",
        )
        failures += 1
    else:
        # MOVE THE TIE. If the port were keeping the LAST maximum, or the
        # one nearest the end of the scan, the first case could pass by
        # luck; the winner must track the ID.
        var tie_b = List[TLeaf]()
        tie_b.append(leaf(Float32(4.0), True))
        tie_b.append(leaf(Float32(1.0), True))
        tie_b.append(leaf(Float32(4.0), True))
        if find_best_leaf_to_split(tie_b) != 0:
            print(
                "  FAIL tie at ids 0,2 went to",
                find_best_leaf_to_split(tie_b), "want 0",
            )
            failures += 1
        else:
            print("  ok   ties at (1,2) -> 1 and at (0,2) -> 0")

    print()
    print("-- P4: undefined leaves are invisible to the argmin --")
    var mixed = List[TLeaf]()
    mixed.append(undefined_leaf())
    mixed.append(leaf(Float32(-9.0), True))
    mixed.append(undefined_leaf())
    if find_best_leaf_to_split(mixed) != 1:
        print(
            "  FAIL undefined leaves were considered; picked",
            find_best_leaf_to_split(mixed), "want 1",
        )
        failures += 1
    else:
        var none_defined = List[TLeaf]()
        none_defined.append(undefined_leaf())
        none_defined.append(undefined_leaf())
        if find_best_leaf_to_split(none_defined) != -1:
            print("  FAIL an all-undefined list reported a best leaf")
            failures += 1
        elif len(select_leaves_to_split(none_defined)) != 0:
            print("  FAIL an all-undefined list selected something to split")
            failures += 1
        else:
            print(
                "  ok   a single defined leaf at gain -9 still wins;"
                " all-undefined selects nothing"
            )

    print()
    print("-- P5: termination, both boundaries --")
    var o = lossguide_options()
    # MaxLeaves: their test is `leafCount >= Options.MaxLeaves`.
    var seven = List[TLeaf]()
    for _ in range(7):
        seven.append(leaf(Float32(1.0), True))
    var eight = List[TLeaf]()
    for _ in range(8):
        eight.append(leaf(Float32(1.0), True))
    if should_terminate(seven, o):
        print("  FAIL terminated at 7 leaves with max_leaves 8")
        failures += 1
    elif not should_terminate(eight, o):
        print("  FAIL did not terminate at 8 leaves with max_leaves 8")
        failures += 1
    else:
        print("  ok   max_leaves 8: 7 continues, 8 stops")

    # IsTerminalLeaf's size test is `<=`, and it is LIVE for Lossguide
    # where it is dead for SymmetricTree.
    var one_row = leaf(Float32(1.0), True, 1)
    var two_rows = leaf(Float32(1.0), True, 2)
    if not is_terminal_leaf(one_row, o):
        print(
            "  FAIL min_data_in_leaf=1 must make a ONE-ROW leaf terminal"
            " (their test is `<=`, not `<`)"
        )
        failures += 1
    elif is_terminal_leaf(two_rows, o):
        print("  FAIL a two-row leaf is terminal at min_data_in_leaf=1")
        failures += 1
    else:
        var sym = lossguide_options()
        sym.policy = GROW_SYMMETRIC
        if is_terminal_leaf(one_row, sym):
            print(
                "  FAIL the size test fired under SymmetricTree, where"
                " CatBoost guards it off (`:685`)"
            )
            failures += 1
        else:
            print(
                "  ok   `<=` boundary: 1 row terminal, 2 rows not, and the"
                " test is dead under SymmetricTree"
            )

    print()
    print("-- P6: the `leavesToVisit.size() <= 2` contract, replayed --")
    # THE PRECONDITION IS PART OF THE CONTRACT, and the first version of
    # this gate got it wrong: it replayed from trees carrying UNDEFINED
    # leaves and reported six leaves to visit. That was the fixture, not the
    # code. At the top of a Lossguide iteration every leaf already HAS a
    # best split, because `ComputeOptimalSplits` scored every leaf
    # `SelectLeavesToVisit` returned and wrote the result back
    # (`greedy_search_helper.cpp:552-556`). The invariant is therefore
    # "one split turns an all-defined tree into exactly two undefined
    # leaves", not "at most two leaves are ever undefined".
    var worst = 0
    for n_leaves in range(1, 7):
        var tree = List[TLeaf]()
        for i in range(n_leaves):
            tree.append(
                leaf(Float32(i) * Float32(0.5) + Float32(0.1), True)
            )
        var picked = select_leaves_to_split(tree)
        if len(picked) != 1:
            print("  FAIL an all-defined tree of", n_leaves, "selected", len(picked))
            failures += 1
            continue
        # ONE ITERATION of their loop: the chosen leaf becomes the left
        # child IN PLACE and a right child is appended, and BOTH lose their
        # best split -- `SplitLeaf` calls `BestSplit.Reset()`
        # (`split_properties_helper.cpp:795`).
        var left_id = picked[0]
        var right = TLeaf()
        right.size = tree[left_id].size // 2
        var left = TLeaf()
        left.size = tree[left_id].size - right.size
        tree[left_id] = left^
        tree.append(right^)

        var to_visit = select_leaves_to_visit(tree)
        if len(to_visit) > worst:
            worst = len(to_visit)
        if len(to_visit) != 2:
            print(
                "  FAIL a tree of", n_leaves,
                "after one split needs", len(to_visit),
                "leaves visited; their CB_ENSURE allows at most 2",
            )
            failures += 1
    if worst != 2:
        print(
            "  FAIL the replay never reached two leaves to visit (worst =",
            worst, "), so the bound was not actually tested",
        )
        failures += 1
    else:
        print("  ok   six tree sizes, exactly 2 to visit after every split")

    # ============ AND THE ONE STATE THAT WOULD BREAK IT ============
    # A leaf whose scorer wrote the POISON record keeps an undefined best
    # split, and `MarkTerminal` only re-evaluates the two NEW children
    # (`:657-659`), so nothing clears it. The next iteration's
    # `SelectLeavesToVisit` therefore returns that leaf PLUS the two new
    # ones, and **their own `CB_ENSURE(leavesToVisit.size() <= 2)`
    # (`:511`) would fire.**
    #
    # This is a fragility in CatBoost, recorded rather than repaired: the
    # port must not "fix" it, and a Mojo caller must raise where theirs
    # raises. It is close to unreachable -- a poison record needs EVERY
    # bin feature marked `SkipInScoreCount`, since a degenerate candidate
    # scores a gain of 0 and still beats the sentinel -- but "close to"
    # is not "never", and a caller that clamped instead of raising would
    # feed the score kernel three leaves through a two-scalar signature.
    #
    # Asserted in the direction that keeps it honest: this file states
    # that the count IS three, so if a future change makes it two the
    # assertion fails and someone re-reads this block.
    var poisoned = List[TLeaf]()
    poisoned.append(leaf(Float32(1.0), True))
    poisoned.append(undefined_leaf())
    poisoned.append(leaf(Float32(2.0), True))
    var pick2 = select_leaves_to_split(poisoned)
    var pid = pick2[0]
    var pr = TLeaf()
    pr.size = poisoned[pid].size // 2
    var pl = TLeaf()
    pl.size = poisoned[pid].size - pr.size
    poisoned[pid] = pl^
    poisoned.append(pr^)
    var n_visit = len(select_leaves_to_visit(poisoned))
    if n_visit != 3:
        print(
            "  FAIL the poisoned-leaf state gives", n_visit,
            "leaves to visit, want 3 -- re-read the block above, their"
            " CB_ENSURE behaviour may have changed",
        )
        failures += 1
    else:
        print(
            "  ok   a poisoned leaf makes it 3, which is where THEIR"
            " CB_ENSURE fires; the caller must raise, not clamp"
        )

    print()
    print("-- find_max_depth --")
    var depths = List[TLeaf]()
    depths.append(leaf(Float32(1.0), True))
    if find_max_depth(depths) != 0:
        print("  FAIL a root-only tree has max depth", find_max_depth(depths))
        failures += 1
    else:
        print("  ok   a root-only tree has depth 0")

    if failures != 0:
        raise Error(
            "lossguide policy check: " + String(failures) + " failures"
        )
    print()
    print("lossguide policy check: PASS")


def main() raises:
    check_lossguide_policy()
