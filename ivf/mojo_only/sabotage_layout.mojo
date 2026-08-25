"""The layout and the merge with one pin BROKEN AT A TIME, for the check.

NOT A PORT, NOT REACHED by `ivf/estimator.mojo`, `ivf/ivf_main.mojo` or
anything under `ivf/ported/`. `ivf/mojo_only/ivf_check.mojo` selects an arm
and nothing else can, exactly the arrangement
`hierarchy/mojo_only/sabotage_tile.mojo` uses: the production functions in
`ivf/mojo_only/list_layout.mojo` carry no sabotage parameter at all, so the
shipped bits cannot depend on this file.

WHY THE COPIES EXIST RATHER THAN A FLAG ON THE REAL ONES
---------------------------------------------------------
`PORTING_RULES.md` rule 8 is the reason a sabotage exists (a path that runs
is not a path that is gated) and `[[reached-but-inert]]` is the reason a
sabotage has to be a SEPARATE spelling: a flag threaded into the production
function is a branch in the production function, and the day someone passes
it by accident the shipped answer moves. These are copies, and the un-
sabotaged arm of each is never called -- `ivf_check.mojo` calls the real
function when it wants the real answer.

THE DISTANCE-TILE SABOTAGE IS NOT COPIED HERE.
`hierarchy/mojo_only/sabotage_tile.mojo::sabotage_distance_tile_kernel` is
already a copy of `neighbors/mojo_only/pinned_distance_tile.mojo`'s kernel
with a rotated-contraction arm and an `std.math.sqrt` arm, which is exactly
what this lane needs and for the same reason (the distance tile is the same
kernel in both lanes). It is IMPORTED by `ivf_check.mojo` rather than
written a third time. A third spelling of one sabotage is a third thing to
get subtly wrong, and `ivf/README.md` records the dependency.

THE ARMS
---------
  IVF_SAB_NONE               the real behaviour, so a check can run both
                             sides through one call site
  IVF_SAB_CARRY_POSITION     `gather_candidate_indices` returns the
                             WITHIN-LIST POSITION instead of the original
                             row id. THE CLASSIC IVF BUG: it compiles, it
                             returns k neighbours, the distances are right
                             to the last bit, and the identities are some
                             other rows'. `check_index_carry` MUST FAIL on
                             this arm.
  IVF_SAB_LIST_ARRIVAL_ORDER the within-list order is a scramble standing
                             in for their `atomicAdd` arrival order
                             (`ivf_flat_build.cuh:131`), so `list_indices`
                             no longer ascends within a list and the merge
                             below it no longer emits ascending original
                             indices. DEVIATION 1783's gate.
  IVF_SAB_MERGE_PROBE_ORDER  the candidates are concatenated PROBE BY
                             PROBE instead of merged, so an equidistant
                             pair from two lists is resolved by which list
                             was probed first. DEVIATION 1786's gate, and
                             the one that breaks the reduction to brute
                             force without breaking anything a distance
                             comparison would see.
  IVF_SAB_PROBE_TIE_HIGH     the probe order breaks a tie in the coarse
                             distance toward the HIGHER list id.
                             DEVIATION 1788's gate.
  IVF_SAB_EMPTY_COUNTS_ONE   the chunk scan counts an empty list as
                             holding one vector, so `n_samples` overcounts
                             and the selector is handed a row longer than
                             the one that was filled. DEVIATION 1782's
                             empty-list gate; `check_empty_list` MUST see
                             the answer move (or the refusal fire).
"""

from ivf.mojo_only.list_layout import ListLayout


comptime IVF_SAB_NONE = 0
comptime IVF_SAB_CARRY_POSITION = 1
comptime IVF_SAB_LIST_ARRIVAL_ORDER = 2
comptime IVF_SAB_MERGE_PROBE_ORDER = 3
comptime IVF_SAB_PROBE_TIE_HIGH = 4
comptime IVF_SAB_EMPTY_COUNTS_ONE = 5


def sabotage_name(arm: Int) -> String:
    if arm == IVF_SAB_NONE:
        return String("IVF_SAB_NONE")
    if arm == IVF_SAB_CARRY_POSITION:
        return String("IVF_SAB_CARRY_POSITION")
    if arm == IVF_SAB_LIST_ARRIVAL_ORDER:
        return String("IVF_SAB_LIST_ARRIVAL_ORDER")
    if arm == IVF_SAB_MERGE_PROBE_ORDER:
        return String("IVF_SAB_MERGE_PROBE_ORDER")
    if arm == IVF_SAB_PROBE_TIE_HIGH:
        return String("IVF_SAB_PROBE_TIE_HIGH")
    if arm == IVF_SAB_EMPTY_COUNTS_ONE:
        return String("IVF_SAB_EMPTY_COUNTS_ONE")
    return String("<unknown sabotage arm ") + String(arm) + ">"


def sabotage_build_list_layout(
    labels: List[UInt32],
    x: List[Float32],
    n_rows: Int,
    dim: Int,
    n_lists: Int,
    sabotage: Int,
) raises -> ListLayout:
    """`build_list_layout` with `IVF_SAB_LIST_ARRIVAL_ORDER` available.

    The arm walks the rows in a stride-coprime order rather than ascending,
    which stands in for their atomic's arrival order without needing a
    race to produce one: the point is only that the within-list sequence
    stops being the ascending original index. A deterministic scramble is
    the RIGHT stand-in here, because a check whose sabotage is itself
    nondeterministic cannot state what it proved.
    """
    var sizes = List[Int32]()
    for _ in range(n_lists):
        sizes.append(Int32(0))
    for i in range(n_rows):
        var l = Int(labels[i])
        if l < 0 or l >= n_lists:
            raise Error(
                "sabotage_build_list_layout: row "
                + String(i)
                + " carries label "
                + String(l)
            )
        sizes[l] = sizes[l] + Int32(1)

    var offsets = List[Int32]()
    offsets.append(Int32(0))
    var running = Int32(0)
    for l in range(n_lists):
        running = running + sizes[l]
        offsets.append(running)

    var cursor = List[Int32]()
    for l in range(n_lists):
        cursor.append(offsets[l])

    var list_indices = List[UInt32]()
    for _ in range(n_rows):
        list_indices.append(UInt32(0))
    var list_data = List[Float32]()
    for _ in range(n_rows * dim):
        list_data.append(Float32(0.0))

    # The visit order. ASCENDING is the production rule and is what makes
    # each list ascend; the sabotage visits `(t * 7 + 3) % n_rows`, which
    # is a permutation for any `n_rows` coprime with 7 and is the arrival
    # order the atomic would have produced, minus the nondeterminism.
    for t in range(n_rows):
        var i = t
        if sabotage == IVF_SAB_LIST_ARRIVAL_ORDER and n_rows % 7 != 0:
            i = (t * 7 + 3) % n_rows
        var l = Int(labels[i])
        var slot = Int(cursor[l])
        cursor[l] = cursor[l] + Int32(1)
        list_indices[slot] = UInt32(i)
        for f in range(dim):
            list_data[slot * dim + f] = x[i * dim + f]

    return ListLayout(
        n_lists, n_rows, dim, offsets^, list_indices^, list_data^
    )


def sabotage_merge_probed_lists(
    layout: ListLayout,
    probe_list_ids: List[UInt32],
    n_probes: Int,
    sabotage: Int,
) raises -> List[Int32]:
    """`merge_probed_lists` with `IVF_SAB_MERGE_PROBE_ORDER` available.

    The arm concatenates the probed lists end to end in PROBE ORDER, which
    is what their own candidate numbering is
    (`ivf_common.cuh:53-56`'s chunk description) and what a straightforward
    IVF port would do. It is not wrong for their scan, because their scan
    carries the database id alongside every candidate and postprocesses it.
    It is wrong HERE, because the selector's key is the candidate's
    POSITION and nothing else, so this arm silently substitutes "the
    earlier-probed list wins" for "the lower original index wins".
    """
    if sabotage != IVF_SAB_MERGE_PROBE_ORDER:
        raise Error(
            "sabotage_merge_probed_lists: only IVF_SAB_MERGE_PROBE_ORDER is"
            " implemented here; call the real merge_probed_lists for"
            " anything else, so the un-sabotaged answer never comes out of"
            " this file."
        )
    var out = List[Int32]()
    for p in range(n_probes):
        var l = Int(probe_list_ids[p])
        var s = Int(layout.offsets[l])
        while s < Int(layout.offsets[l + 1]):
            out.append(Int32(s))
            s += 1
    return out^


def sabotage_gather_candidate_indices(
    layout: ListLayout, slots: List[Int32], sabotage: Int
) raises -> List[UInt32]:
    """`gather_candidate_indices` with `IVF_SAB_CARRY_POSITION` available.

    The arm returns the WITHIN-LIST POSITION of each candidate instead of
    its original row id -- `slot - offsets[list_of(slot)]`, the value
    their `inlist_id` holds. That is a number the layout genuinely knows
    and that looks like an index, which is exactly why this bug survives
    review: every distance is right, the array has the right length and
    dtype, and the answer names other people's rows.
    """
    if sabotage != IVF_SAB_CARRY_POSITION:
        raise Error(
            "sabotage_gather_candidate_indices: only"
            " IVF_SAB_CARRY_POSITION is implemented here; call the real"
            " gather_candidate_indices for anything else."
        )
    var out = List[UInt32]()
    for c in range(len(slots)):
        var s = Int(slots[c])
        var l = 0
        while l + 1 < len(layout.offsets) and Int(layout.offsets[l + 1]) <= s:
            l += 1
        out.append(UInt32(s - Int(layout.offsets[l])))
    return out^


def sabotage_sort_probe_slots(
    mut dist: List[Float32], mut idx: List[UInt32], base: Int, n: Int
):
    """The probe order with the tie broken toward the HIGHER list id.

    `IVF_SAB_PROBE_TIE_HIGH`. The only change from
    `sort_slots_by_distance_then_index` is `ib >= iv` where the real one
    has `ib <= iv`, which is the whole of a tie rule: one character, and it
    decides which of two equidistant lists a query probes first and
    therefore (on the sabotaged merge, and only there) which of two
    equidistant vectors it returns.
    """
    for a in range(1, n):
        var dv = dist[base + a]
        var iv = idx[base + a]
        var b = a - 1
        while b >= 0:
            var db = dist[base + b]
            var ib = idx[base + b]
            if db < dv or (db == dv and ib >= iv):
                break
            dist[base + b + 1] = db
            idx[base + b + 1] = ib
            b -= 1
        dist[base + b + 1] = dv
        idx[base + b + 1] = iv


def sabotage_calc_chunk_indices(
    list_sizes: List[Int32],
    probe_list_ids: List[UInt32],
    n_probes: Int,
    sabotage: Int,
) raises -> List[Int32]:
    """`calc_chunk_indices` with `IVF_SAB_EMPTY_COUNTS_ONE` available.

    The arm gives an EMPTY probed list a size of one, so `n_samples`
    overcounts by the number of empty lists probed. Nothing crashes: the
    selector is simply handed a row length longer than the number of cells
    the distance kernel filled, and it selects over uninitialized memory --
    a wrong answer that is different every run, which is the worst kind and
    the reason `check_empty_list` exists.
    """
    if sabotage != IVF_SAB_EMPTY_COUNTS_ONE:
        raise Error(
            "sabotage_calc_chunk_indices: only IVF_SAB_EMPTY_COUNTS_ONE is"
            " implemented here; call the real calc_chunk_indices for"
            " anything else."
        )
    var out = List[Int32]()
    var total = Int32(0)
    for p in range(n_probes):
        var l = Int(probe_list_ids[p])
        var sz = list_sizes[l]
        if sz == Int32(0):
            sz = Int32(1)
        total = total + sz
        out.append(total)
    return out^
