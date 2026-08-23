"""The TOTAL ORDER on MST edges, and why the MST needs one.

NOT A PORT. RAFT's Boruvka (`raft/sparse/solver/detail/mst_solver_inl.cuh`,
`mst_kernels.cuh`) never needed this file because it breaks ties with a
random alteration of the weights (`alteration()`, `:212-238`). This file is
what replaces that alteration. Read the DEVIATION BLOCK below before
touching either.

======================================================================
DEVIATION BLOCK -- DEVIATION 620. NO RANDOM ALTERATION; A TOTAL ORDER
ON (weight bits, min(u,v), max(u,v)) BREAKS EVERY TIE.
======================================================================

WHAT THEIRS DOES. `MST_solver::solve` (`mst_solver_inl.cuh:121`) calls
`alteration()` when `e > 1`. `alteration_max()` (`:186-207`) sorts and
uniques every weight and takes HALF the smallest gap between distinct
weights (or 1 when all weights are identical). `alteration()` (`:212-238`)
draws `v` uniform values from cuRAND (`CURAND_RNG_PSEUDO_DEFAULT`, seed
1234567) and `alteration_kernel` (`mst_kernels.cuh:289-307`) writes
`altered_weights[e] = weights[e] + max * (rand[row] + rand[col])` in
`alteration_t = double`. Every later comparison -- the per-lane `<` at
`:63`, the lane fold at `:78`, the per-color `atomicMin` at `:94`, the
`min_edge_color[...] == vertex_weight` test at `:127` -- is on the ALTERED
double, and `temp_weights` carries the original float back out (`:148`).

The purpose is stated at `:117-120`: Boruvka adds, per supervertex, EVERY
vertex whose min edge weight equals the supervertex's min weight
(`min_edge_per_supervertex`, `:127`). If two vertices of one supervertex
tie, BOTH edges are added in one round and the result can be a cycle, or
an edge count above `v - 1` (the `RAFT_EXPECTS` at `:143`). The alteration
makes all weights distinct so that cannot happen.

WHY IT CANNOT BE PORTED AS-IS. Which of two equal-weight edges wins is
decided by `rand[row] + rand[col]`, a cuRAND XORWOW stream. There is no
cuRAND on Metal or HIP, and even a faithful XORWOW port would make the MST
a function of the RNG implementation rather than of the input: two
equal-weight edges are ordered by two random draws, and a second vendor
with a second generator picks the other edge. The edge SET then differs,
so the dendrogram differs, so the labels differ -- on EXACTLY the inputs a
clustering library sees most: duplicate points (weight 0 ties) and data on
a grid (equal distances). `hierarchy/mojo_only/linkage_check.mojo`'s
duplicate-point fixture is the measurement: under the sabotage
`LINK_SAB_RANDOM_ALTERATION` (a pseudo-random draw per vertex standing in
for cuRAND's) the MST edge set differs from the oracle's on the first run.

WHAT OURS DOES. An edge is compared by the lexicographic triple

    (weight_order_key(w), min(u, v), max(u, v))

and NOTHING ELSE. `weight_order_key` maps the Float32 weight onto an Int32
whose integer order is the float order (the `range_key` map of DEVIATION
204, shifted into the signed range), so `-0.0 < +0.0`, and NaN is sent to
a single key just below the sentinel. Two DISTINCT undirected edges have
distinct `(min, max)`, so the triple is a TOTAL order on undirected edges;
the two directions of one edge carry the SAME triple, which is the property
theirs buys with `rand[row] + rand[col]` being symmetric (`:235`,
"keeping Wuv == Wvu"), and it is what makes the "vertices added each
other" test at `mst_kernels.cuh:139-141` work.

WHY THE ORDER MUST BE ON THE UNDIRECTED EDGE, NOT THE DIRECTED INDEX.
Boruvka's cycle-freedom proof needs ONE order that every supervertex
reads the same way. With the directed CSR index `u*m + v` as tie-break,
supervertices A = {0, 3}, B = {1, 5}, C = {2, 6} with only the three edges
{0,5}, {1,6}, {2,3} tied at the minimum pick A->B (key 0*m+5), B->C
(1*m+6), C->A (2*m+3): three edges, three supervertices, a cycle. With
`(min, max)` all three read {0,5} < {1,6} < {2,3} the same way and the
standard argument holds.

HOW A MIN OVER THIS ORDER IS REPRODUCIBLE. A min over a total order is
associative and commutative with NO rounding, so it does not care which
lane, block, or atomic lands first. The per-color min is taken in three
integer `atomicMin` phases (weight key, then `min(u,v)` among weight ties,
then `max(u,v)` among those), because Metal has no 64-bit atomic and each
phase is an order-free integer min; see `mst_kernels.mojo`.

WHAT THIS COSTS THEM, AND US, IN FIDELITY. Their MST on ties is a
different (equally minimal) tree, so on tie-free inputs the two agree on
the edge set and weights exactly; on inputs with ties ours is the
lexicographically-least MST and theirs is a random one. The MST WEIGHT
multiset is identical either way (all MSTs share it); the edge set, the
dendrogram, and therefore the labels at a cut through a tie, are not.
======================================================================
"""

from std.memory import bitcast


comptime WEIGHT_KEY_SENTINEL: Int32 = 0x7FFFFFFF
"""`std::numeric_limits<alteration_t>::max()`'s role: what `min_edge_color`
is filled with before each round (`mst_solver_inl.cuh:282-283`). No real
weight maps here: NaN is sent to `WEIGHT_KEY_NAN`, one below."""

comptime WEIGHT_KEY_NAN: Int32 = 0x7FFFFFFE
"""Every NaN weight, so a NaN edge sorts LAST but is still an edge."""

comptime EDGE_SENTINEL: Int32 = 0x7FFFFFFF
"""`std::numeric_limits<edge_t>::max()`: `new_mst_edge[v]` when `v` found no
outgoing edge this round (`:284`, `:153`)."""

comptime VERTEX_SENTINEL: Int32 = 0x7FFFFFFF
"""`std::numeric_limits<vertex_t>::max()`: an unused `temp_src` slot
(`:314`), which `kernel_count_new_mst_edges` and the compaction skip."""


@always_inline
def weight_order_key(w: Float32) -> Int32:
    """A Float32 as an Int32 whose INTEGER order is the float's order.

    DEVIATION 204's `range_key` (non-negative: set the sign bit; negative:
    invert every bit), then the top bit flipped so the UNSIGNED order
    becomes the SIGNED one `Atomic.min` on `Int32` sees. For a
    non-negative float that is its raw bit pattern, which is why a
    distance's key reads like its bits in a trace. NaN -> `WEIGHT_KEY_NAN`
    (all NaN payloads to one key, so `NaN == NaN` for the equality test
    in `min_edge_per_supervertex`). Exactly invertible for every non-NaN
    input: `weight_order_unkey(weight_order_key(x))` is `x` bit for bit.

    SIGNED ZERO (IDENTITY_PATHS row 39, audited 2026-08-23): `-0.0` and
    `+0.0` map to DISTINCT keys, `-1` and `0`, so `-0.0` sorts FIRST and an
    edge of weight `-0.0` beats an edge of weight `+0.0` in every min this
    file's order decides -- on every vendor, because the INTEGER key
    decides, not a hardware `min`/`max` whose answer on a (+0, -0) pair is
    the vendor's (Apple returns the second operand, NVIDIA and AMD the
    IEEE-2019 minimum). Every NaN payload maps to ONE key, so the MST
    never sees a payload; the raw float is still what `temp_weights`
    carries out, which is why `pairwise_distances` refuses a NaN distance
    before it reaches a recorded stage (DEVIATION 623). `hierarchy/mojo_only/
    linkage_check.mojo::check_linkage_signed_zero_mst` plants both zeros.
    """
    if w != w:
        return WEIGHT_KEY_NAN
    var b = rebind[UInt32](w.to_bits())
    var k: UInt32
    if (b & UInt32(0x80000000)) != 0:
        k = ~b
    else:
        k = b | UInt32(0x80000000)
    return bitcast[DType.int32](k ^ UInt32(0x80000000))


@always_inline
def weight_order_unkey(k: Int32) -> Float32:
    """The exact inverse of `weight_order_key` away from NaN."""
    var u = bitcast[DType.uint32](k) ^ UInt32(0x80000000)
    if (u & UInt32(0x80000000)) != 0:
        return bitcast[DType.float32](u ^ UInt32(0x80000000))
    return bitcast[DType.float32](~u)


@always_inline
def edge_lo(u: Int32, v: Int32) -> Int32:
    return u if u < v else v


@always_inline
def edge_hi(u: Int32, v: Int32) -> Int32:
    return v if u < v else u


@always_inline
def triple_less(
    wk_a: Int32, lo_a: Int32, hi_a: Int32,
    wk_b: Int32, lo_b: Int32, hi_b: Int32,
) -> Bool:
    """`(wk, lo, hi)_a < (wk, lo, hi)_b`, lexicographic. The ONE comparison
    every arm of the MST uses; there is no second spelling."""
    if wk_a != wk_b:
        return wk_a < wk_b
    if lo_a != lo_b:
        return lo_a < lo_b
    return hi_a < hi_b


# ======================================================================
# DEVIATION BLOCK -- DEVIATION 624. THE HOST PACKING OF THE WEIGHT KEY
# ORDERS A NEGATIVE KEY (-0.0, ANY NEGATIVE WEIGHT) THE WAY THE DEVICE DOES.
# ======================================================================
#
# NOT A DEPARTURE FROM RAFT (theirs has no key; DEVIATION 620 introduced
# it) but a departure from this file's first draft, found by the row-39
# audit (2026-08-23) when `check_linkage_signed_zero_mst` planted a `-0.0`
# weight. The device compares keys as SIGNED Int32 (`triple_less`,
# `Atomic.min`), so `-0.0` (key -1) sorts before `+0.0` (key 0). The first
# draft of `pack_edge_key` placed the key's RAW 32 bits in the top half of
# the UInt64, so on the host `-0.0` packed as 0xFFFFFFFF... and sorted LAST:
# Boruvka took the `-0.0` edge first, `coo_sort_by_weight` then put it
# last, and the oracle's Kruskal rejected it altogether -- device and host
# disagreed on the MST of a graph holding one `-0.0`. MEASURED before the
# fix, IDENTICAL build, Apple M4, 2026-08-23:
#     check_linkage_signed_zero_mst [IDENTICAL] order A (-0.0 first, +0.0
#     second): the -0.0 edge did not win by key; slot 0 (0,2,0x00000000)
#     slot 1 (1,6,0x3f93c680), want slot 0 (0,1,0x80000000) slot 1
#     (0,2,0x00000000)
# (Boruvka had taken the -0.0 edge; the host sort put it LAST). The fix
# flips the key's sign bit on the way into the UInt64 (`^ 0x80000000`), the
# standard signed-to-unsigned order map, and `unpack_edge_wk` flips it
# back. For every NON-NEGATIVE key (every clamped L2 distance, FLT_MAX,
# +inf, the NaN key) the packed order is UNCHANGED, so no bit of any
# existing fixture or card moves; only a graph fed straight into
# `build_sorted_mst` with a `-0.0` or negative weight is ordered
# differently, and now the same way on the host as on the device.
# ======================================================================


def pack_edge_key(wk: Int32, lo: Int32, hi: Int32) -> UInt64:
    """The triple as ONE UInt64 whose integer order is the triple's order,
    for HOST sorts (Kruskal in the oracle, `coo_sort_by_weight`). The key's
    sign bit is flipped so the SIGNED Int32 order the device uses becomes
    the UNSIGNED order of the top 32 bits (DEVIATION 624; `-0.0`'s key -1
    lands below `+0.0`'s key 0, as on the device). `lo` and `hi` are below
    2^16 because `hierarchy` refuses `n_rows > 46340` (their `value_idx nnz
    = m * m` overflows Int32 there, `connectivities.cuh:145`); the masks
    truncate, they do not assert, so a caller past 16 bits must refuse
    before packing, as `pairwise_distances` does."""
    var w = UInt64(bitcast[DType.uint32](wk) ^ UInt32(0x80000000)) & UInt64(0xFFFFFFFF)
    var l = UInt64(bitcast[DType.uint32](lo)) & UInt64(0xFFFFFFFF)
    var h = UInt64(bitcast[DType.uint32](hi)) & UInt64(0xFFFFFFFF)
    return (w << UInt64(32)) | (l << UInt64(16)) | h


def unpack_edge_lo(key: UInt64) -> Int32:
    return Int32(Int((key >> UInt64(16)) & UInt64(0xFFFF)))


def unpack_edge_hi(key: UInt64) -> Int32:
    return Int32(Int(key & UInt64(0xFFFF)))


def unpack_edge_wk(key: UInt64) -> Int32:
    """The inverse of `pack_edge_key`'s key half (DEVIATION 624)."""
    return bitcast[DType.int32](
        UInt32(Int((key >> UInt64(32)) & UInt64(0xFFFFFFFF))) ^ UInt32(0x80000000)
    )


# ======================================================================
# SABOTAGE ARMS. Selected by `linkage_check.mojo` ONLY; every driver and
# the Python-facing entry pass `LINK_SAB_NONE`. Each arm breaks one pin so
# the check can SHOW the gate fail (COMMON_BRIEF rule 6), then restores.
# ======================================================================

comptime LINK_SAB_NONE = 0

comptime LINK_SAB_RANDOM_ALTERATION = 1
"""Re-enable a RAFT-style random tie-break: among equal weights, order by
a per-vertex pseudo-random draw `hash(lo) + hash(hi)` instead of
`(lo, hi)`. This is their `rand[row] + rand[col]` with a hash for cuRAND;
it moves ONLY ties, exactly as theirs does (`alteration_max` keeps the
perturbation below the smallest weight gap). Must FAIL on the duplicate
point / equal distance fixture and MAY pass on tie-free data."""

comptime LINK_SAB_ROTATE_CONTRACTION = 2
"""Start the distance dot product at feature `block_idx % d` and wrap,
instead of at 0. A different summation order per block; must move bits
in the distance matrix, so the MST weights differ from the oracle's."""

comptime LINK_SAB_STD_SQRT = 3
"""`std.math.sqrt` in place of `identical_sqrt` at the distance seam. On
Apple both are correctly rounded, so this arm is EXPECTED NOT TO FAIL on
this device and the check REPORTS rather than asserts; it is the arm
that would fail on NVIDIA's approximate sqrt (DEVIATION 258)."""

comptime LINK_SAB_SORT_WEIGHT_ONLY = 4
"""`coo_sort_by_weight` on the weight alone with ties left in REVERSE
discovery order, i.e. one of the orders `thrust::sort_by_key` is allowed
to return. Must FAIL the dendrogram gate on the equal-distance fixture
(DEVIATION 621)."""


comptime LINK_SAB_SKIP_NAN_GUARD = 5
"""Skip DEVIATION 623's NaN refusal in `pairwise_distances`, so the check
can show a planted NaN distance reaching the MST weights (with the
vendor's payload) when the guard is absent. Touches nothing else."""


@always_inline
def sabotage_vertex_draw(v: Int32) -> Int32:
    """The stand-in for cuRAND's per-vertex uniform draw under
    `LINK_SAB_RANDOM_ALTERATION`: a 31-bit hash of the vertex index."""
    var h = UInt32(Int(v) & 0x7FFFFFFF) * UInt32(0x9E3779B1)
    h = h ^ (h >> UInt32(15))
    h = h * UInt32(0x85EBCA77)
    h = h ^ (h >> UInt32(13))
    return Int32(Int(h & UInt32(0x3FFFFFFF)))


@always_inline
def sabotaged_lo_hi(
    sabotage: Int32, lo: Int32, hi: Int32
) -> Tuple[Int32, Int32]:
    """Under `LINK_SAB_RANDOM_ALTERATION` the secondary key is the summed
    draw (symmetric in u, v, as theirs is), the tertiary the draw of `lo`
    alone; otherwise `(lo, hi)`. Called at EVERY triple construction in
    the MST kernels so the sabotage is one consistent (wrong) order."""
    if sabotage == LINK_SAB_RANDOM_ALTERATION:
        var a = sabotage_vertex_draw(lo)
        var b = sabotage_vertex_draw(hi)
        return (a + b, a)
    return (lo, hi)
