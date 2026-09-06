# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The planted point sets, and what each one exists to separate.

NOT A PORT. cuML checks HDBSCAN against scikit-learn-contrib in Python on
`make_blobs` data with a random seed; this tree checks a port against a
host computation that is bit-for-bit predictable, so every fixture has to
be a PURE FUNCTION of its index -- no RNG, no floating-point host
arithmetic that a second machine's libm could answer differently, and
values that differ per cell so a permutation is visible
(`[[uniform-test-data-hides-permutation]]`).

Each fixture exists to separate ONE thing, and a fixture that separates
nothing is a fixture whose green line means nothing:

    HFIX_BLOBS       three well-separated blobs, hashed jitter, 96 x 4.
                     The cluster of every row is KNOWN BY CONSTRUCTION
                     (`row % 3`), so the label gate compares against a
                     planted answer and not only against an oracle that
                     could share a bug with the port.
    HFIX_GRADIENT    a dense core inside a sparse halo, 90 x 2, with the
                     halo's spacing rising with the index. Excess of Mass
                     has a real choice here: the core's stability and the
                     union's stability are close and the selection can go
                     either way, which is what makes
                     HDB_SAB_EOM_NO_UPDATE able to move it.
    HFIX_DUPS        a 6 x 6 integer lattice plus 12 EXACT DUPLICATES of
                     lattice points, 48 x 2. Distance 0 pairs and equal
                     distances by the hundred, so the mutual reachability
                     three-way max ties and the MST's total order is
                     actually exercised rather than merely present.
    HFIX_OUTLIER     two tight blobs of 20 plus ONE point far from both,
                     41 x 2. The outlier must come back as label -1,
                     which is the one output HDBSCAN has that single
                     linkage does not. Two blobs rather than one because
                     `allow_single_cluster = False` (their default) makes
                     a one-cluster tree return all noise; see
                     `hfixture_value`.
    HFIX_SIGNED_ZERO blobs whose feature 2 is EXACTLY `-0.0` on every row
                     (bit pattern 0x80000000), 48 x 3.
    HFIX_POS_ZERO    the SAME fixture with `+0.0` in that column. The pair
                     is the point: `check_hdbscan_signed_zero_inputs`
                     requires the two to produce byte-identical cards, so
                     a `-0.0` in the INPUT cannot reach an output. The
                     `-0.0` that CAN reach the three-way max is planted
                     directly into a core-distance array by
                     `check_mutual_reachability_ties`, because no input
                     coordinate produces one (`(-0.0)*(-0.0)` is `+0.0`
                     and the distance clamp maps every negative residue to
                     `+0.0`).
    HFIX_NESTED      SIX blobs of 8 on a designed distance ladder, 48 x 2,
                     whose CONDENSED CLUSTER TREE BRANCHES TWICE ON BOTH
                     SIDES OF ITS ROOT. It exists for exactly one arm,
                     `HDB_SAB_CONDENSE_DFS`, and the next block says why
                     the other five fixtures cannot host it.

======================================================================
WHY HFIX_NESTED HAD TO BE ADDED: THE OTHER FIVE CANNOT SEPARATE A
TRAVERSAL ORDER, AND THAT IS ARITHMETIC, NOT A GUESS.
======================================================================
`condense.mojo` assigns `next_label` in `bfs_from_node`'s visit order,
and `condensed_hierarchy.condense()` then SORTS the four output arrays on
`(parent, child)` (DEVIATION 1611). The sort erases the order in which
the edges were APPENDED, so the ONLY way a traversal order can reach the
condensed tree is through the VALUES in `relabel`, and `relabel` is
written at exactly one place: the `left_count >= min_cluster_size and
right_count >= min_cluster_size` branch, their case 1. Call the
dendrogram nodes that take that branch the SPLIT NODES; they are the
internal nodes of the condensed CLUSTER tree, and the label a cluster
gets is decided by the position of its parent split node in the walk.

Breadth first and depth first order two split nodes DIFFERENTLY only when
the pair is INCOMPARABLE -- neither an ancestor of the other -- and the
one in the LEFT branch of their common ancestor is STRICTLY DEEPER.
Depth first descends the left branch to the bottom first; breadth first
takes the shallower node first whichever branch it is in. On an
ancestor/descendant pair BOTH walks visit the ancestor first, always.

A cluster tree with k leaf clusters has k - 1 split nodes, and with
k <= 3 every pair of them is an ancestor/descendant pair. So a fixture
with three or fewer clusters CANNOT MOVE `HDB_SAB_CONDENSE_DFS`, and
`blobs96` is measured at 100 edges and 5 clusters -- 96 leaf edges plus
FOUR cluster edges, so `next_label` was incremented four times, so it has
exactly TWO split nodes, so they are nested. The arm was only ever
pointed at `blobs96`, so that alone settles why it was inert; the other
five fixtures are two-blob sets, a core-plus-halo and a lattice, and
their condensed `n_clusters` has never been PRINTED, so this file does
not claim a number for them -- what it claims is that none of them was
BUILT to branch and that a fixture which is built to branch is the only
honest way to gate a traversal.

WHAT HFIX_NESTED PLANTS INSTEAD, in numbers. Six blobs of 8, centres on
one line at x = 0, 20, 72, 320, 346, 406, jitter +-0.6 per coordinate.
The five nearest-neighbour gaps between consecutive centres are 20, 52,
248, 26 and 60, and no non-consecutive pair is closer than the
consecutive ones between them, so the MST over the blobs is that path and
Kruskal joins them in the order 20, 26, 52, 60, 248:

    root {L, R};  L {L01, b2};  L01 {b0, b1};  R {R34, b5};  R34 {b3, b4}

FIVE split nodes at three depths: root at 0, L and R at 1, L01 and R34 at
2. Whichever of L and R the dendrogram puts on the left, that side's
depth-2 split node is LEFT of the other side's depth-1 split node and
deeper than it, so the violating pair exists in BOTH orientations and the
fixture does not depend on DEVIATION 1614's choice coming out one way.
Breadth first labels in the order root, L, R, L01, R34; depth first in
the order root, L, L01, R, R34. With 48 points, `relabel[root] = 48` and
the two walks disagree on FOUR of the ten cluster ids (the pair born at
the other side's depth-1 node and the pair born at the first side's
depth-2 node trade places), which moves every condensed edge naming one
of them.

THE MARGINS ARE WIDE ON PURPOSE. With jitter +-0.6 the closest pair
across a gap `g` lies in `[g - 1.2, g]`, so the five merge heights fall
in the disjoint intervals [18.8, 20], [24.8, 26], [50.8, 52], [58.8, 60]
and [246.8, 248]: the ladder cannot reorder. Blob size 8 is at least
`min_cluster_size` = 5, so every one of the five merges IS a split node;
and 8 is BELOW `2 * min_cluster_size` = 10, so no blob can split into two
children that are both large enough, which is what keeps the number of
split nodes at exactly five rather than at "five and whatever the jitter
adds". Feature 1 carries jitter only and no per-blob offset, because a
second separating axis would turn every merge height into a square root
this file cannot certify by hand, and a fixture whose shape is asserted
rather than derived is the defect this fixture exists to repair.

IT PLANTS NO LABELS. `hfixture_planted_label` returns `-2` here.
Which of the ten clusters Excess of Mass selects is a modelling outcome
of the stability arithmetic, and planting a guess would be planting an
answer this file cannot derive; the label gate falls back to the oracle,
as it already does on `gradient90` and `dups_lattice48`.
======================================================================

THE HASH IS `hierarchy/checks/linkage_oracle.mojo::_hash_unit`,
IMPORTED. One fixture hash in the tree rather than two: a second
generator is a second thing to get wrong, and two lanes whose fixtures
"look similar" but hash differently cannot have their cards compared at
all. Only the ASSEMBLY below is this lane's.
"""

from std.memory import bitcast

from max.gpu.host import DeviceContext, HostBuffer

from hierarchy.checks.linkage_oracle import _hash_unit


comptime HFIX_BLOBS = 0
comptime HFIX_GRADIENT = 1
comptime HFIX_DUPS = 2
comptime HFIX_OUTLIER = 3
comptime HFIX_SIGNED_ZERO = 4
comptime HFIX_POS_ZERO = 5
comptime HFIX_NESTED = 6
comptime HFIX_COUNT = 7

comptime NEG_ZERO_BITS: UInt32 = 0x80000000


def hfixture_name(fix: Int) -> String:
    if fix == HFIX_BLOBS:
        return String("blobs96")
    if fix == HFIX_GRADIENT:
        return String("gradient90")
    if fix == HFIX_DUPS:
        return String("dups_lattice48")
    if fix == HFIX_OUTLIER:
        return String("blob_plus_outlier41")
    if fix == HFIX_SIGNED_ZERO:
        return String("signed_zero48")
    if fix == HFIX_NESTED:
        return String("nested_ladder48")
    return String("pos_zero48")


def hfixture_n(fix: Int) -> Int:
    if fix == HFIX_BLOBS:
        return 96
    if fix == HFIX_GRADIENT:
        return 90
    if fix == HFIX_DUPS:
        return 48
    if fix == HFIX_OUTLIER:
        return 41
    if fix == HFIX_NESTED:
        return 48
    return 48


def hfixture_d(fix: Int) -> Int:
    if fix == HFIX_BLOBS:
        return 4
    if fix == HFIX_GRADIENT:
        return 2
    if fix == HFIX_DUPS:
        return 2
    if fix == HFIX_OUTLIER:
        return 2
    if fix == HFIX_NESTED:
        return 2
    return 3


def hfixture_min_samples(fix: Int) -> Int:
    """`min_samples` for each fixture. Small enough that the core distance
    is a LOCAL density and not the whole cloud, large enough that a single
    duplicate pair does not set it to zero for a whole blob."""
    if fix == HFIX_DUPS:
        return 4
    if fix == HFIX_OUTLIER:
        return 4
    if fix == HFIX_NESTED:
        # 5, so the 5 nearest neighbours of every row lie inside its own
        # blob of 8; the next blob is at least 18.8 away and the whole
        # blob fits in a 1.2 x 1.2 box, so the core distance is a LOCAL
        # density on this fixture by construction.
        return 5
    return 5


def hfixture_min_cluster_size(fix: Int) -> Int:
    if fix == HFIX_DUPS:
        return 4
    if fix == HFIX_OUTLIER:
        return 5
    if fix == HFIX_NESTED:
        # 5 is the value the split-node count is DERIVED from, not a
        # taste: 8 >= 5 makes every blob merge a split node, and
        # 8 < 2 * 5 makes it impossible for a blob to contribute one of
        # its own. Change it and this file's header stops being true.
        return 5
    return 5


def hfixture_value(fix: Int, i: Int, f: Int) -> Float32:
    """The value at cell `(i, f)`, a PURE FUNCTION of the three integers.

    No host `exp`, `log`, `sqrt` or `pow` appears anywhere below -- only
    additions and multiplications of exactly representable constants with
    hashed values -- so this table is the same on every host libm
    (IDENTITY_PATHS row 18's class, avoided rather than pinned).
    """
    if fix == HFIX_BLOBS:
        # Three blobs at 0, 12, -12 in feature 0, with a hashed jitter of
        # +-0.75 and a per-blob offset in feature 1 so the blobs are
        # separated in two coordinates rather than one.
        var blob = i % 3
        var center = Float32(0.0)
        if blob == 1:
            center = Float32(12.0)
        elif blob == 2:
            center = Float32(-12.0)
        var v = center + (_hash_unit(i, f, 11) - Float32(0.5)) * Float32(1.5)
        if f == 1:
            v = v + Float32(blob) * Float32(6.0)
        return v

    if fix == HFIX_GRADIENT:
        # Rows 0..39: a DENSE core, jitter +-0.15 about the origin.
        # Rows 40..89: a halo whose radius grows with the index, so the
        # local density falls off smoothly and the core-vs-union stability
        # comparison is close rather than lopsided.
        if i < 40:
            return (_hash_unit(i, f, 21) - Float32(0.5)) * Float32(0.3)
        var step = Float32(i - 40)
        var radius = Float32(1.5) + step * Float32(0.11)
        var angle_u = _hash_unit(i, 0, 22)
        # A hashed point on a square ring rather than a circle: a circle
        # would need cos/sin, which is a host transcendental and row 12's
        # class. The ring's four sides are chosen by the low bits of the
        # index, and the position along a side by the hash.
        var side = i % 4
        var t = (angle_u - Float32(0.5)) * Float32(2.0) * radius
        if f == 0:
            if side == 0:
                return radius
            if side == 1:
                return -radius
            return t
        if side == 0 or side == 1:
            return t
        if side == 2:
            return radius
        return -radius

    if fix == HFIX_DUPS:
        # 36 lattice points (0..5 x 0..5) then 12 EXACT duplicates of
        # lattice points. Integer coordinates, so equal distances are
        # exactly equal and the tie path is reached rather than
        # approached.
        var src = i
        if i >= 36:
            var dup_of = [0, 7, 14, 21, 28, 35, 0, 7, 5, 30, 17, 17]
            src = dup_of[i - 36]
        if f == 0:
            return Float32(src % 6)
        return Float32(src // 6)

    if fix == HFIX_OUTLIER:
        # TWO tight blobs of 20, then one point far from both.
        #
        # WHY TWO AND NOT ONE. With `allow_single_cluster = False` --
        # their default (`hdbscan.hpp:138`) and this suite's -- Excess of
        # Mass deselects the ROOT unconditionally (`select.cuh:191`), so a
        # dataset whose condensed tree has ONE cluster returns every point
        # as noise. That is upstream's behavior and not a defect, but a
        # fixture built on it would plant an assignment the algorithm is
        # not asked to produce. Two blobs give the root a real split, and
        # the outlier is still the only planted noise point.
        if i == 40:
            if f == 0:
                return Float32(40.0)
            return Float32(40.0)
        var side = Float32(0.0)
        if i >= 20:
            side = Float32(11.0)
        return side + (_hash_unit(i, f, 31) - Float32(0.5)) * Float32(1.0)

    if fix == HFIX_NESTED:
        # SIX blobs of 8 on one line, centres at 0, 20, 72, 320, 346, 406.
        # Every centre is an integer and therefore exact in Float32, so
        # the five gaps below are exact too: 20, 52, 248, 26, 60 between
        # consecutive centres. Kruskal joins them 20, 26, 52, 60, 248,
        # which is the two-sided nesting this file's header derives.
        #
        # Feature 1 is jitter and NOTHING ELSE, deliberately: a per-blob
        # offset would make each merge height a square root and the
        # ladder would stop being an arithmetic fact.
        var nb = i % 6
        var cx = Float32(0.0)
        if nb == 1:
            cx = Float32(20.0)
        elif nb == 2:
            cx = Float32(72.0)
        elif nb == 3:
            cx = Float32(320.0)
        elif nb == 4:
            cx = Float32(346.0)
        elif nb == 5:
            cx = Float32(406.0)
        var jitter = (_hash_unit(i, f, 51) - Float32(0.5)) * Float32(1.2)
        if f == 0:
            return cx + jitter
        return jitter

    # HFIX_SIGNED_ZERO / HFIX_POS_ZERO: two blobs in features 0 and 1,
    # feature 2 a planted zero of one sign or the other on EVERY row.
    if f == 2:
        if fix == HFIX_SIGNED_ZERO:
            return bitcast[DType.float32](NEG_ZERO_BITS)
        return Float32(0.0)
    var blob2 = i % 2
    var center2 = Float32(0.0)
    if blob2 == 1:
        center2 = Float32(9.0)
    var v2 = center2 + (_hash_unit(i, f, 41) - Float32(0.5)) * Float32(1.2)
    return v2


def hfixture_planted_label(fix: Int, i: Int) -> Int32:
    """The cluster this row belongs to BY CONSTRUCTION, or `-2` where the
    fixture does not plant one.

    `-1` means "must be noise" and is a planted assertion like any other;
    `-2` means "the fixture does not say", and a gate that reads it must
    fall back to the oracle. The two are different claims and collapsing
    them is how a planted gate quietly becomes an oracle gate.
    """
    if fix == HFIX_BLOBS:
        return Int32(i % 3)
    if fix == HFIX_OUTLIER:
        if i == 40:
            return Int32(-1)
        if i < 20:
            return Int32(0)
        return Int32(1)
    if fix == HFIX_SIGNED_ZERO or fix == HFIX_POS_ZERO:
        return Int32(i % 2)
    return Int32(-2)


def hfixture_n_planted_clusters(fix: Int) -> Int:
    """How many clusters the fixture plants, or 0 where it plants none."""
    if fix == HFIX_BLOBS:
        return 3
    if fix == HFIX_OUTLIER:
        return 2
    if fix == HFIX_SIGNED_ZERO or fix == HFIX_POS_ZERO:
        return 2
    return 0


def hfixture_as_list(fix: Int) -> List[Float32]:
    var n = hfixture_n(fix)
    var d = hfixture_d(fix)
    var out = List[Float32](capacity=n * d)
    for i in range(n):
        for f in range(d):
            out.append(hfixture_value(fix, i, f))
    return out^


def hfixture_permutation(fix: Int) -> List[Int]:
    """A fixed permutation of `0 .. n-1` for
    `check_hdbscan_permutation_invariance`, built from the hash so it is
    NOT a rotation or a reversal -- a structured permutation can preserve
    an index-order tie-break by accident and prove nothing.

    Fisher-Yates with an INTEGER-hashed swap partner, walked ascending,
    which is a pure function of `n`. The partner is drawn with integer
    arithmetic only -- no float is converted to an index anywhere -- so
    the permutation is the same on every host.
    """
    var n = hfixture_n(fix)
    var perm = List[Int](capacity=n)
    for i in range(n):
        perm.append(i)
    for i in range(n - 1):
        var span = n - i
        var j = i + Int(_hash_int(i, 91) % UInt64(span))
        var tmp = perm[i]
        perm[i] = perm[j]
        perm[j] = tmp
    return perm^


def _hash_int(i: Int, salt: Int) -> UInt64:
    """A splitmix-style integer mix. Fixture plumbing, not an algorithm:
    the only property asked of it is that it is a pure function of its two
    integers and that consecutive `i` do not give consecutive outputs."""
    var h = UInt64(i + 3) * UInt64(0x9E3779B97F4A7C15) + UInt64(
        salt + 7
    ) * UInt64(0x94D049BB133111EB)
    h = h ^ (h >> UInt64(30))
    h = h * UInt64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> UInt64(27))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(31))


def hfixture_permuted_as_list(fix: Int, perm: List[Int]) -> List[Float32]:
    """The fixture with its ROWS reordered by `perm`: new row `r` holds
    old row `perm[r]`. The VALUES are copied bit for bit, so a difference
    in the answer is an order dependence and never a re-hash."""
    var n = hfixture_n(fix)
    var d = hfixture_d(fix)
    var out = List[Float32](capacity=n * d)
    for r in range(n):
        var src = perm[r]
        for f in range(d):
            out.append(hfixture_value(fix, src, f))
    return out^


def build_hfixture(
    ctx: DeviceContext, fix: Int
) raises -> HostBuffer[DType.float32]:
    var n = hfixture_n(fix)
    var d = hfixture_d(fix)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    ctx.synchronize()
    for i in range(n):
        for f in range(d):
            host.unsafe_ptr().unsafe_store(i * d + f, hfixture_value(fix, i, f))
    return host^
