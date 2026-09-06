# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CSR-shaped inverted lists, the index carry, and the probe merge.

NOT A PORT. cuVS lays an IVF-FLAT index out as `n_lists` separately
allocated arrays of INTERLEAVED GROUPS (`ivf_flat.hpp:96-115`,
`ivf_flat_build.cuh:137-146`), because that is the shape
`ivfflat_interleaved_scan` issues its vectorized loads against. This port
does not run that scan (DEVIATION 1785), so it does not need that layout,
and it needs a different property that layout does not have.

THE THREE THINGS THIS FILE IS RESPONSIBLE FOR
----------------------------------------------

**1. The layout is CSR (DEVIATION 1782).** One offsets array of
`n_lists + 1`, one contiguous `list_indices` of `n_rows`, one contiguous
`list_data` of `n_rows * dim`. An empty list is `offsets[l+1] ==
offsets[l]`, which is a normal state and not a special case anywhere below.

**2. The ORIGINAL index travels (DEVIATION 1784), and the within-list
order is ASCENDING ORIGINAL INDEX (DEVIATION 1783).** Their
`build_index_kernel` takes its slot from
`atomicAdd(list_sizes_ptr + list_id, 1)` (`ivf_flat_build.cuh:131`), so the
within-list order is arrival order: reproducible in NEITHER value nor
placement across runs, let alone across vendors. The count that atomic
produces is exact (it is an integer atomic, and integer addition is
order-free), but the SLOT each vector lands in is not, and the slot is what
a selection keyed on position would read. Replaced here with a stable
counting sort by `(label, original index)`, which makes the layout a pure
function of the assignment and nothing else.

**3. The probe merge preserves ascending original index
(DEVIATION 1786), and that is the whole reason the reduction gate can
exist.** The identical selector
(`neighbors/checks/select_radix_identical.mojo`) keys on
`(twiddle_in(distance) << 32) | POSITION_IN_THE_ROW`, and it has no input
index array to key on instead. So the candidate row's POSITION ORDER *is*
the tie-break. If candidates were concatenated probe-by-probe, an
equidistant pair from two different lists would be resolved by WHICH LIST
WAS PROBED FIRST -- and brute force resolves it by the lower original
index, so `n_probe == n_lists` would not reduce to brute force and the
lane's strongest gate would be false. Merging the probed lists ASCENDING BY
CARRIED ORIGINAL INDEX makes the candidate row's position order agree with
the original index order, so the selector's key is `(distance, original
index)` restricted to the candidates. At `n_probe == n_lists` the candidate
row IS the index in original order and the two computations are the same
one.

This is the classic IVF identity bug and its fix, and
`ivf/checks/sabotage_layout.mojo` carries both wrong versions as arms.

WHY THE LAYOUT AND THE MERGE ARE ON THE HOST
---------------------------------------------
**DEVIATION 1800**, and it is a real departure from `PORTING_RULES.md`
rule 2, which says a control-plane decision they make on the device is one
we make on the device. Theirs is a device kernel; this is a host counting
sort and a host k-way merge.

The reason is that the deterministic device spelling needs a SEGMENTED RANK
-- `rank[i] = |{j < i : label[j] == label[i]}|` -- which is a multi-block
scan, and neither spelling has ever been measured against the other.
Writing an unmeasured multi-block scan to replace an unmeasured host loop
is inventing, which is
what `PORTING_RULES.md` 0c is about. Closure condition, stated so it is not
mistaken for done: a segmented exclusive scan over the label histogram,
plus a scatter that reads its rank rather than an atomic. It changes no
bit -- the host sort already produces the ordering the device one would --
so it is a speed change and belongs behind a measurement.
"""


@fieldwise_init
struct ListLayout(Movable):
    """The CSR triple, plus the assignment it was built from.

    `list_indices[s]` is the ORIGINAL row id of the vector stored at slot
    `s`, and `list_data[s * dim + f]` is that vector's feature `f`. The pair
    is the index carry: a slot is meaningless without both halves, which is
    why they are built together here and never separately.
    """

    var n_lists: Int
    var n_rows: Int
    var dim: Int
    var offsets: List[Int32]
    var list_indices: List[UInt32]
    var list_data: List[Float32]


def build_list_layout(
    labels: List[UInt32],
    x: List[Float32],
    n_rows: Int,
    dim: Int,
    n_lists: Int,
) raises -> ListLayout:
    """The CSR build: histogram, exclusive scan, stable scatter.

    Their `build_index_kernel` (`ivf_flat_build.cuh:109-150`) in three host
    passes with the arrival order removed:

      1. `sizes[l] = |{i : labels[i] == l}|` -- their `atomicAdd` counts
         this exactly and so does this loop; an integer atomic's TOTAL is
         order-free and it is only the slot it hands back that is not.
      2. `offsets[0] = 0; offsets[l+1] = offsets[l] + sizes[l]` -- ascending
         over `l`, integers, no float anywhere in this function.
      3. one ASCENDING pass over `i`, appending row `i` to its list. The
         pass order is the original index order, so each list comes out
         ascending in the original index by construction. That is the
         stability property DEVIATION 1783 needs and it is free: it is what
         a single ascending pass does.

    Raises on a label outside `[0, n_lists)` rather than writing past a
    list, because a bad label here silently corrupts a neighbouring list
    and the corruption is invisible in every aggregate.
    """
    if len(labels) != n_rows:
        raise Error(
            "build_list_layout: labels has "
            + String(len(labels))
            + " entries, expected n_rows = "
            + String(n_rows)
        )
    if len(x) != n_rows * dim:
        raise Error(
            "build_list_layout: x has "
            + String(len(x))
            + " values, expected n_rows * dim = "
            + String(n_rows * dim)
        )

    var sizes = List[Int32]()
    for _ in range(n_lists):
        sizes.append(Int32(0))
    for i in range(n_rows):
        var l = Int(labels[i])
        if l < 0 or l >= n_lists:
            raise Error(
                "build_list_layout: row "
                + String(i)
                + " carries label "
                + String(l)
                + ", outside [0, "
                + String(n_lists)
                + ")"
            )
        sizes[l] = sizes[l] + Int32(1)

    var offsets = List[Int32]()
    offsets.append(Int32(0))
    var running = Int32(0)
    for l in range(n_lists):
        running = running + sizes[l]
        offsets.append(running)

    # The write cursor per list, starting at the list's offset. This is the
    # counter their kernel keeps in `list_sizes_ptr` and increments with an
    # atomic; here it is advanced by one ascending host pass, so the slot a
    # row gets is a function of the row index and never of arrival.
    var cursor = List[Int32]()
    for l in range(n_lists):
        cursor.append(offsets[l])

    var list_indices = List[UInt32]()
    for _ in range(n_rows):
        list_indices.append(UInt32(0))
    var list_data = List[Float32]()
    for _ in range(n_rows * dim):
        list_data.append(Float32(0.0))

    for i in range(n_rows):
        var l = Int(labels[i])
        var slot = Int(cursor[l])
        cursor[l] = cursor[l] + Int32(1)
        # THE CARRY. `list_index[inlist_id] = source_ix`,
        # `ivf_flat_build.cuh:135`, with `source_ix = i` because this port
        # has no `source_ixs` gather arm (their `gather_src` template
        # parameter serves `fill_refinement_index`, which is not ported).
        list_indices[slot] = UInt32(i)
        for f in range(dim):
            list_data[slot * dim + f] = x[i * dim + f]

    return ListLayout(
        n_lists, n_rows, dim, offsets^, list_indices^, list_data^
    )


def merge_probed_lists(
    layout: ListLayout, probe_list_ids: List[UInt32], n_probes: Int
) raises -> List[Int32]:
    """The candidate SLOTS of `n_probes` lists, ordered by carried index.

    Returns slot numbers into `layout.list_indices` / `layout.list_data`,
    ordered so that `layout.list_indices[out[c]]` ASCENDS in `c`. Because
    each list's slice already ascends, this is a k-way merge over
    `n_probes` sorted runs and it needs no sort and no comparison of
    anything but integers.

    The linear scan over the `n_probes` heads is `O(n_candidates *
    n_probes)`. That is not the shape a fast implementation has and it is
    not defended as one; the alternatives have never been measured against
    it and `ivf/README.md` says so. A heap, or the device segmented
    scan of DEVIATION 1800, is the same ORDER of output and therefore the
    same bits.

    An EMPTY probed list contributes an empty run and falls out of the
    merge on the first `head == end` test. `check_empty_list` gates that
    the whole path -- merge, gather, distance, selection -- is unmoved by
    one, which is the degenerate case that has to be boring.
    """
    if len(probe_list_ids) < n_probes:
        raise Error(
            "merge_probed_lists: probe_list_ids has "
            + String(len(probe_list_ids))
            + " entries, expected at least n_probes = "
            + String(n_probes)
        )

    var head = List[Int32]()
    var end = List[Int32]()
    var total = 0
    for p in range(n_probes):
        var l = Int(probe_list_ids[p])
        if l < 0 or l >= layout.n_lists:
            raise Error(
                "merge_probed_lists: probe "
                + String(p)
                + " names list "
                + String(l)
                + ", outside [0, "
                + String(layout.n_lists)
                + ")"
            )
        head.append(layout.offsets[l])
        end.append(layout.offsets[l + 1])
        total += Int(layout.offsets[l + 1]) - Int(layout.offsets[l])
        # A LIST MAY NOT BE PROBED TWICE. The coarse selector returns a set
        # (its key is a total order over distinct list ids), so a repeat is
        # a defect upstream of here and it would double-count a vector into
        # the candidate row, where the selector's distinct-key invariant
        # would then be false.
        for q in range(p):
            if probe_list_ids[q] == probe_list_ids[p]:
                raise Error(
                    "merge_probed_lists: list "
                    + String(l)
                    + " appears at probe "
                    + String(q)
                    + " and probe "
                    + String(p)
                    + ". The coarse selection's key is a total order over"
                    " distinct list ids, so a repeat means the selector"
                    " returned a multiset."
                )

    var out = List[Int32]()
    for _ in range(total):
        var best = -1
        var best_idx = UInt32(0)
        for p in range(n_probes):
            if head[p] >= end[p]:
                continue
            var cand = layout.list_indices[Int(head[p])]
            if best < 0 or cand < best_idx:
                best = p
                best_idx = cand
        if best < 0:
            raise Error(
                "merge_probed_lists: ran out of heads with "
                + String(len(out))
                + " of "
                + String(total)
                + " candidates emitted; the offsets and the list contents"
                " disagree"
            )
        out.append(head[best])
        head[best] = head[best] + Int32(1)
    return out^


def gather_candidate_vectors(
    layout: ListLayout, slots: List[Int32]
) raises -> List[Float32]:
    """`[len(slots), dim]` row-major, the probed vectors in merge order.

    A PERMUTATION. No arithmetic is performed on a value in this function,
    which is why it can be host-side without being a numeric pathway.
    """
    var out = List[Float32]()
    for c in range(len(slots)):
        var s = Int(slots[c])
        for f in range(layout.dim):
            out.append(layout.list_data[s * layout.dim + f])
    return out^


def gather_candidate_indices(
    layout: ListLayout, slots: List[Int32]
) raises -> List[UInt32]:
    """The ORIGINAL row id of each candidate, in merge order.

    This is the array `check_index_carry` sabotages. Carrying the WITHIN-
    LIST POSITION here instead of the original id is the classic IVF bug:
    it compiles, it returns k neighbours, the distances are right, and the
    identities are somebody else's rows.
    """
    var out = List[UInt32]()
    for c in range(len(slots)):
        out.append(layout.list_indices[Int(slots[c])])
    return out^


def gather_candidate_norms(
    slots: List[Int32], list_norm: List[Float32]
) raises -> List[Float32]:
    """The squared norm of each candidate, in merge order.

    A GATHER OF A DEVICE-PRODUCED ARRAY, not a recomputation.
    `list_norm` is `core/row_norms.mojo::row_norm_kernel` run over
    `list_data` -- the PERMUTED matrix -- and it is indexed here by SLOT,
    which is what `list_norm` is indexed by.

    THAT PERMUTATION IS BIT-EXACT AND NOT MERELY CLOSE, which is the whole
    reason the norms can be taken over the permuted copy at all:
    `row_norm_kernel` is one block per row and reads only that row, so
    permuting the rows permutes the outputs and changes no float. The norm
    a candidate is scored against is therefore the same float a brute force
    over the unpermuted matrix scores that row against, and
    `check_nprobe_equals_nlists_is_brute_force` depends on it.
    """
    var out = List[Float32]()
    for c in range(len(slots)):
        var s = Int(slots[c])
        if s < 0 or s >= len(list_norm):
            raise Error(
                "gather_candidate_norms: slot "
                + String(s)
                + " is outside the norm array of "
                + String(len(list_norm))
                + " entries"
            )
        out.append(list_norm[s])
    return out^
