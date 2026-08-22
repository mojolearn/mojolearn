"""Which leaf Lossguide splits, proved against Depthwise on the same input.

    pixi run check-lossguide-policy

NO CATBOOST COUNTERPART: a gate, so `mojo_only/`. The sentence that used to
sit here, "HOST ONLY -- it creates no `DeviceContext`, launches nothing",
became FALSE the day S1 landed: S1 launches the real score kernel, precisely
because a host-only fixture could not see the sign-convention defect it pins.
Every claim but S1 is host algebra in milliseconds; the file as a whole is a
device check and is NOT safe inside another lane's timing window.

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

  P7  THE QUEUE CHECKPOINT SEES UNVISITED LEAVES. The identity-trace ladder's
      `queue.*` / `selected_leaf` stages (`select_leaves_to_split_traced`)
      must: emit exactly four records in order; change nothing about the
      decision; emit NOTHING when disabled; and -- the tooth -- a gain edit
      on a leaf that is NOT selected and NOT visited must move the
      `queue.gain` hash while `selected_leaf` and `queue.feature` stand
      still. The driver's `best.*` records cover only the visited pair, so
      an instrument claiming to hash the whole priority queue has to be
      shown a difference only the whole queue can carry.

THE SABOTAGES. P2 and P3 are applied to the EXPECTATION as well: the file
asserts that the Depthwise arm DOES reject the all-worse list and that it
returns more than one leaf on the many-improving list. If those two
assertions ever pass trivially, the contrast has gone dead and the
differential gates are measuring nothing.
"""

from max.gpu.host import DeviceContext
from mojo_only.leafwise_scores_check import bits

from gbdt.methods.greedy_subsets_searcher.kernel.compute_scores import (
    LEAFWISE_SCORE_BLOCK_SIZE,
    compute_optimal_split_kernel,
)
from gbdt.methods.helpers import best_split_properties_less
from gbdt.models.oblivious_model import BIN_SPLIT_TAKE_GREATER
from gbdt.options.catboost_options import SCORE_FUNCTION_COSINE

from gbdt.methods.greedy_subsets_searcher.greedy_search_helper_depthwise import (
    is_terminal_leaf,
    select_leaves_to_split as depthwise_select_leaves_to_split,
    select_leaves_to_visit,
    should_terminate,
)
from core.identity_trace import IdentityTrace, read_trace_lines
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper_lossguide import (
    find_best_leaf_to_split,
    find_max_depth,
    select_leaves_to_split,
    select_leaves_to_split_traced,
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
    """One leaf carrying a planted best split.

    **`gain` IS IN CATBOOST'S SIGN, LOWER IS BETTER**, because that is what a
    `TLeaf.best_split` actually holds. The kernel's gain is theirs negated,
    and the host reduce negates it BACK before storing
    (`cand = TBestSplitProperties(f, b, -our_gain, -our_gain)`), so the two
    flips cancel and the stored record is CatBoost's own number. S1 below
    pins that empirically against the real kernel rather than by argument --
    this file asserted the opposite convention for its first hour, and every
    gate still passed, because a self-consistent fixture cannot see a
    convention error."""
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


def check_sign_convention(ctx: DeviceContext) raises -> Int:
    """S1. THE ONLY DEVICE GATE IN THIS FILE, and the one that would have
    caught the defect the rest of it could not.

    `find_best_leaf_to_split` was written as an ARGMAX because the score
    kernel's sign is this port's, flipped from CatBoost's. It is flipped
    TWICE: once in the kernel and once again in the host reduce, which
    stores `-our_gain` so that `best_split_properties_less` -- a
    transcription of their `operator<`, lower-is-better, over a default
    record whose gain is `+MAX` -- means what it says. The two flips cancel
    and the stored record is CatBoost's own number.

    **Every host gate in this file passed under the wrong convention**,
    because a fixture written in one sign and asserted in the same sign is
    self-consistent whichever sign that is. Only a gate that spans the
    BOUNDARY can see it. So this one runs the real kernel, applies the real
    reduce, and asserts the identity rather than any fixture's property:

        stored_gain == -our_gain,  bit for bit
        find_best_leaf_to_split picks the leaf whose OUR-SIGN gain is LARGEST

    An identity, not a fixture property, so it holds whatever the planted
    data does and cannot be satisfied by re-planting.
    """
    var failures = 0
    comptime N_BF = 16
    comptime STAT_COUNT = 2
    comptime N_LEAVES = 3

    var n_cells = N_LEAVES * STAT_COUNT * N_BF
    var h_hist = ctx.enqueue_create_host_buffer[DType.float32](n_cells)
    var h_ps = ctx.enqueue_create_host_buffer[DType.float32](
        N_LEAVES * STAT_COUNT
    )
    var h_skip = ctx.enqueue_create_host_buffer[DType.uint8](N_BF)
    var h_fid = ctx.enqueue_create_host_buffer[DType.uint32](N_BF)
    var h_fw = ctx.enqueue_create_host_buffer[DType.float32](N_BF)
    for b in range(N_BF):
        h_skip.unsafe_ptr().unsafe_store(b, UInt8(0))
        h_fid.unsafe_ptr().unsafe_store(b, UInt32(b))
        h_fw.unsafe_ptr().unsafe_store(b, Float32(1.0))
    for l in range(N_LEAVES):
        var w = Float32(20.0) + Float32(l) * Float32(11.0)
        h_ps.unsafe_ptr().unsafe_store(l * STAT_COUNT, w)
        h_ps.unsafe_ptr().unsafe_store(
            l * STAT_COUNT + 1, Float32(4.0) - Float32(l) * Float32(1.3)
        )
        for b in range(N_BF):
            var f = Float32(b + 1) / Float32(N_BF + 2)
            h_hist.unsafe_ptr().unsafe_store(l * STAT_COUNT * N_BF + b, w * f)
            h_hist.unsafe_ptr().unsafe_store(
                l * STAT_COUNT * N_BF + N_BF + b,
                (Float32(4.0) - Float32(l) * Float32(1.3)) * f * f,
            )

    var argmax_blocks = 2
    var d_hist = ctx.enqueue_create_buffer[DType.float32](n_cells)
    var d_ps = ctx.enqueue_create_buffer[DType.float32](
        N_LEAVES * STAT_COUNT
    )
    var d_skip = ctx.enqueue_create_buffer[DType.uint8](N_BF)
    var d_fid = ctx.enqueue_create_buffer[DType.uint32](N_BF)
    var d_fw = ctx.enqueue_create_buffer[DType.float32](N_BF)
    var d_score = ctx.enqueue_create_buffer[DType.float32](argmax_blocks * 2)
    var d_bin = ctx.enqueue_create_buffer[DType.uint32](argmax_blocks * 2)
    ctx.enqueue_copy(dst_buf=d_hist, src_ptr=h_hist.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ps, src_ptr=h_ps.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_skip, src_ptr=h_skip.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_fid, src_ptr=h_fid.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_fw, src_ptr=h_fw.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[compute_optimal_split_kernel[SCORE_FUNCTION_COSINE]](
        d_skip.unsafe_ptr(), Int32(N_BF), d_fid.unsafe_ptr(),
        d_fw.unsafe_ptr(), d_hist.unsafe_ptr(), d_ps.unsafe_ptr(),
        Int32(STAT_COUNT), Int32(0), Int32(2), Int32(0),
        Float32(1.5), Float32(0.0), UInt64(0),
        d_score.unsafe_ptr(), d_bin.unsafe_ptr(),
        grid_dim=(argmax_blocks, 2, 1),
        block_dim=(LEAFWISE_SCORE_BLOCK_SIZE, 1, 1),
    )
    var h_s = ctx.enqueue_create_host_buffer[DType.float32](argmax_blocks * 2)
    var h_b = ctx.enqueue_create_host_buffer[DType.uint32](argmax_blocks * 2)
    ctx.enqueue_copy(dst_ptr=h_s.unsafe_ptr(), src_buf=d_score)
    ctx.enqueue_copy(dst_ptr=h_b.unsafe_ptr(), src_buf=d_bin)
    ctx.synchronize()

    # THE REAL REDUCE, transcribed from the depthwise driver's own loop so
    # this gate cannot pass against a private variant of it.
    var leaves = List[TLeaf]()
    var our_gain = List[Float32]()
    for row in range(2):
        var best = TBestSplitProperties()
        var best_ours = -Float32.MAX
        for i in range(argmax_blocks):
            var slot = row * argmax_blocks + i
            var bf = h_b.unsafe_ptr().unsafe_load(slot)
            if bf == UInt32(0xFFFFFFFF):
                continue
            var g = h_s.unsafe_ptr().unsafe_load(slot)
            var cand = TBestSplitProperties(
                Int32(Int(bf)), Int32(0), -g, -g
            )
            if best_split_properties_less(cand, best):
                best = cand
                best_ours = g
        var lf = TLeaf()
        lf.size = 100
        lf.update_best_split(best)
        leaves.append(lf^)
        our_gain.append(best_ours)

    if len(leaves) != 2 or not leaves[0].best_split.defined:
        print("  FAIL the kernel produced no usable record; S1 is BLIND")
        return failures + 1

    for i in range(2):
        if bits(leaves[i].best_split.gain) != bits(-our_gain[i]):
            print(
                "  FAIL leaf", i, "stored", leaves[i].best_split.gain,
                "but the kernel's gain was", our_gain[i],
                "-- the stored record is NOT the kernel's gain negated",
            )
            failures += 1
    if failures == 0:
        var want = 0 if our_gain[0] > our_gain[1] else 1
        var got = find_best_leaf_to_split(leaves)
        if got != want:
            print(
                "  FAIL find_best_leaf_to_split picked", got, "but the leaf"
                " with the LARGER our-sign gain is", want,
                "( gains", our_gain[0], our_gain[1], ")",
            )
            failures += 1
        else:
            print(
                "  ok   stored gain is the kernel's negated, bit for bit,"
                " and the argmin picks the better leaf (", got, ")",
            )
    return failures


def check_lossguide_policy(ctx: DeviceContext) raises:
    var failures = 0

    print("-- S1: the SIGN CONVENTION, across the kernel/host boundary --")
    failures += check_sign_convention(ctx)
    print()

    print("-- P1: at most one leaf, where Depthwise takes many --")
    var many = List[TLeaf]()
    # four leaves that IMPROVE (their sign: gain < 0), one that does not
    many.append(leaf(Float32(-0.5), True))
    many.append(leaf(Float32(-2.5), True))
    many.append(leaf(Float32(-1.5), True))
    many.append(leaf(Float32(0.25), True))
    many.append(leaf(Float32(-0.75), True))

    var lg = select_leaves_to_split(many)
    var dw = depthwise_select_leaves_to_split(many)
    if len(lg) != 1:
        print("  FAIL Lossguide selected", len(lg), "leaves, want exactly 1")
        failures += 1
    elif lg[0] != 1:
        print("  FAIL Lossguide picked leaf", lg[0], "want 1 (gain -2.5)")
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
    all_worse.append(leaf(Float32(3.0), True))
    all_worse.append(leaf(Float32(0.5), True))
    all_worse.append(leaf(Float32(1.75), True))

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
        print("  FAIL Lossguide picked leaf", lg2[0], "want 1 (gain +0.5)")
        failures += 1
    else:
        print(
            "  ok   every split makes it worse: Depthwise 0, Lossguide 1"
            " (the least-bad leaf, id", lg2[0], ")",
        )

    print()
    print("-- P3: an exact tie goes to the FIRST leaf --")
    var tie_a = List[TLeaf]()
    tie_a.append(leaf(Float32(-1.0), True))
    tie_a.append(leaf(Float32(-4.0), True))
    tie_a.append(leaf(Float32(-4.0), True))
    tie_a.append(leaf(Float32(-2.0), True))
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
        tie_b.append(leaf(Float32(-4.0), True))
        tie_b.append(leaf(Float32(-1.0), True))
        tie_b.append(leaf(Float32(-4.0), True))
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
    mixed.append(leaf(Float32(9.0), True))
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
                "  ok   a single defined leaf at gain +9 (a WORSE split)"
                " still wins; all-undefined selects nothing"
            )

    print()
    print("-- P5: termination, both boundaries --")
    var o = lossguide_options()
    # MaxLeaves: their test is `leafCount >= Options.MaxLeaves`.
    var seven = List[TLeaf]()
    for _ in range(7):
        seven.append(leaf(Float32(-1.0), True))
    var eight = List[TLeaf]()
    for _ in range(8):
        eight.append(leaf(Float32(-1.0), True))
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
    var one_row = leaf(Float32(-1.0), True, 1)
    var two_rows = leaf(Float32(-1.0), True, 2)
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
                leaf(-(Float32(i) * Float32(0.5) + Float32(0.1)), True)
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
    poisoned.append(leaf(Float32(-1.0), True))
    poisoned.append(undefined_leaf())
    poisoned.append(leaf(Float32(-2.0), True))
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
    print("-- P7: the queue checkpoint is wired, inert, and sees UNVISITED leaves --")
    # The ladder stage the driver cannot supply: `best.*` covers only the
    # visited pair, `queue.*` is the whole priority queue. Four sub-claims,
    # and the tooth is the last one -- an edit only the whole queue can see.
    var q = List[TLeaf]()
    q.append(leaf(Float32(-1.0), True))
    q.append(leaf(Float32(-3.0), True))  # the winner: lowest, THEIR sign
    q.append(undefined_leaf())
    q.append(leaf(Float32(2.0), True))  # defined, worse, never selected

    var plain = select_leaves_to_split(q)
    var tr_off = IdentityTrace.disabled()
    var picked_off = select_leaves_to_split_traced(q, tr_off, "p7.")
    if len(picked_off) != 1 or len(plain) != 1 or picked_off[0] != plain[0]:
        print("  FAIL the traced variant changed the DECISION under disabled()")
        failures += 1
    elif tr_off.seq != 0:
        print("  FAIL a disabled trace emitted", tr_off.seq, "records")
        failures += 1
    else:
        var pa = String("/tmp/mojolearn_lossguide_p7_a.trace")
        var tra = IdentityTrace.to_path(pa)
        var picked_a = select_leaves_to_split_traced(q, tra, "p7.")
        var lines_a = read_trace_lines(pa)
        # A gain edit on leaf 3: DEFINED, never selected, and -- because it
        # is not one of the two children a split just made -- never VISITED.
        var q2 = q.copy()
        q2[3] = leaf(Float32(2.5), True)
        var pb = String("/tmp/mojolearn_lossguide_p7_b.trace")
        var trb = IdentityTrace.to_path(pb)
        var picked_b = select_leaves_to_split_traced(q2, trb, "p7.")
        var lines_b = read_trace_lines(pb)
        if picked_a[0] != plain[0] or picked_b[0] != plain[0]:
            print("  FAIL tracing (or the leaf-3 edit) moved the decision itself")
            failures += 1
        elif len(lines_a) != 4 or len(lines_b) != 4:
            print(
                "  FAIL want exactly 4 records, got", len(lines_a), "and",
                len(lines_b),
            )
            failures += 1
        elif (
            _tag(lines_a[0]) != "p7.queue.feature"
            or _tag(lines_a[1]) != "p7.queue.bin"
            or _tag(lines_a[2]) != "p7.queue.gain"
            or _tag(lines_a[3]) != "p7.selected_leaf"
        ):
            print(
                "  FAIL tag order is", _tag(lines_a[0]), _tag(lines_a[1]),
                _tag(lines_a[2]), _tag(lines_a[3]),
            )
            failures += 1
        elif lines_a[2] == lines_b[2]:
            # THE TOOTH. Identical hashes across a real queue difference is
            # the worst failure this instrument can have (its silence would
            # read as cross-backend agreement), so it is asserted, not hoped.
            print(
                "  FAIL queue.gain hash did not move across a gain edit on"
                " an unselected leaf -- the checkpoint is not hashing the"
                " whole queue"
            )
            failures += 1
        elif lines_a[0] != lines_b[0] or lines_a[3] != lines_b[3]:
            print(
                "  FAIL queue.feature or selected_leaf moved when only a"
                " gain changed -- the records are not independent"
            )
            failures += 1
        else:
            print(
                "  ok   4 records, decision untouched, disabled emits none,"
                " and the queue.gain hash MOVES on an unvisited leaf's edit"
            )

    print()
    print("-- find_max_depth --")
    var depths = List[TLeaf]()
    depths.append(leaf(Float32(-1.0), True))
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


def _tag(line: String) raises -> String:
    """Field 2 of a trace record, `<seq>\\t<tag>\\t<dtype>\\t<count>\\t<hash>`."""
    var parts = line.split("\t")
    if len(parts) < 2:
        return String("")
    return String(parts[1])


def main() raises:
    var ctx = DeviceContext()
    check_lossguide_policy(ctx)
