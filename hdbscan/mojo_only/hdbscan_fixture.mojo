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

THE HASH IS `hierarchy/mojo_only/linkage_oracle.mojo::_hash_unit`,
IMPORTED. One fixture hash in the tree rather than two: a second
generator is a second thing to get wrong, and two lanes whose fixtures
"look similar" but hash differently cannot have their cards compared at
all. Only the ASSEMBLY below is this lane's.
"""

from std.memory import bitcast

from max.gpu.host import DeviceContext, HostBuffer

from hierarchy.mojo_only.linkage_oracle import _hash_unit


comptime HFIX_BLOBS = 0
comptime HFIX_GRADIENT = 1
comptime HFIX_DUPS = 2
comptime HFIX_OUTLIER = 3
comptime HFIX_SIGNED_ZERO = 4
comptime HFIX_POS_ZERO = 5
comptime HFIX_COUNT = 6

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
    return 3


def hfixture_min_samples(fix: Int) -> Int:
    """`min_samples` for each fixture. Small enough that the core distance
    is a LOCAL density and not the whole cloud, large enough that a single
    duplicate pair does not set it to zero for a whole blob."""
    if fix == HFIX_DUPS:
        return 4
    if fix == HFIX_OUTLIER:
        return 4
    return 5


def hfixture_min_cluster_size(fix: Int) -> Int:
    if fix == HFIX_DUPS:
        return 4
    if fix == HFIX_OUTLIER:
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
