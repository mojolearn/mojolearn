# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Planted index and query sets for the IVF gates, ASSEMBLED FROM BITS.

NOT A PORT. `IDENTITY_PATHS` row 32's lesson applies here verbatim: a host
`target += v * w` chain is a contraction decision, so a fixture built by
host arithmetic can hand two machines different inputs before the first
kernel runs. Every value here is either a splitmix64 hash turned into a
float32 by BITCAST or an EXACT literal bit pattern. **No floating-point
operation is performed in this file.**

`mix64` and `bits_value` are IMPORTED from
`kde/mojo_only/kde_fixture.mojo` rather than written again. They are the
hashed-fixture primitive this repository already has; a second splitmix64
with the same constants and a different last line is a thing to get wrong
for nothing, and the fixtures of two lanes drawing from one stream is a
feature -- the values are non-uniform and non-repeating per cell, so a
permutation of rows or columns changes the answer
(`[[uniform-test-data-hides-permutation]]`).

THE FIVE FIXTURES THE BRIEF REQUIRES, AND WHAT EACH ONE REACHES
----------------------------------------------------------------

1. **`ivf_index_fixture` + `ivf_query_fixture`, correct neighbours known by
   construction.** Every EVEN query row is an EXACT COPY of index row
   `(q * 7) % n_index`. Expanded L2 of a row against itself is a
   cancellation clamped at zero (`unfused_distance_nn.cuh:80-81`, kept in
   `pinned_distance_tile.mojo`), so that row is the query's nearest
   neighbour at the minimum possible distance and NOTHING about the
   quantizer, the probe count or the selector had to be trusted to know
   it. The odd rows are fresh hashed rows with no such guarantee.

2. **`ivf_duplicate_fixture`, so the tie path is REACHED.** Rows are laid
   out in groups of three EXACT duplicates, so every distance from any
   query is attained by three distinct original indices at once. Under
   `IDENTICAL` the composite key resolves that to the lowest of the three
   and `check_search_vs_oracle` compares which; under `FAST` the ported
   selector's back-fill decides it by atomic arrival and the same
   comparison is a REPORT. A gate that never reaches the tie class is a
   gate about a class that does not exist -- `check_assignment_ties`
   REFUSES ITSELF if no tie occurs, and so does the duplicate arm.

3. **`ivf_equidistant_centers` / `ivf_equidistant_points`, a point exactly
   equidistant from two centroids.** Centre 0 is `+1` in column 0 and zero
   elsewhere; centre 1 is `-1` in column 0. The point is `0` in column 0
   and `+0.5` in column 1. Every coordinate is an exact power of two and
   every squared difference is exactly representable, so the two distances
   are EQUAL IN FLOAT and not merely close: `(0-1)^2 = 1` and
   `(0-(-1))^2 = 1`, bit for bit, in any summation order. Planting a tie
   that only ALMOST ties would be a fixture that quietly tests the
   non-tie path, which is the failure `[[uniform-test-data-hides-
   permutation]]` names in its own domain.

4. **`ivf_planted_labels`, a list that comes out EMPTY.** The labels skip
   one list id entirely, so `list_offsets[l+1] == list_offsets[l]` for that
   `l` and every probe of it contributes nothing. Planted rather than
   coaxed out of k-means, because a degenerate case that only appears when
   the quantizer happens to produce it is a case the check cannot state it
   ran.

5. **`ivf_signed_zero_fixture`, signed zeros in a coordinate.**
   `IDENTITY_PATHS` row 39. Column 0 alternates the bit patterns
   `0x00000000` and `0x80000000` by row, so rows differing ONLY in the sign
   of a zero exist in the index. They are numerically identical points
   (`+0.0 == -0.0`), so they are an exact duplicate pair whose two members
   a distance comparison can never separate and only the index tie-break
   can order.
"""

from std.memory import bitcast

from kde.mojo_only.kde_fixture import bits_value, mix64


comptime BITS_POS_ZERO: UInt32 = 0x00000000
comptime BITS_NEG_ZERO: UInt32 = 0x80000000
comptime BITS_ONE: UInt32 = 0x3F800000
comptime BITS_NEG_ONE: UInt32 = 0xBF800000
comptime BITS_HALF: UInt32 = 0x3F000000


def ivf_index_fixture(n: Int, d: Int, salt: Int) -> List[Float32]:
    """`n x d` hashed rows, `|v|` in `[0.5, 2)`, signed."""
    var out = List[Float32]()
    for i in range(n):
        for f in range(d):
            out.append(bits_value(mix64(i, f, salt), True))
    return out^


def ivf_query_fixture(
    index: List[Float32], n_index: Int, n_query: Int, d: Int, salt: Int
) -> List[Float32]:
    """Even rows: an EXACT COPY of index row `(q * 7) % n_index`, so that
    row is the known-by-construction nearest neighbour. Odd rows: fresh
    hashed rows."""
    var out = List[Float32]()
    for q in range(n_query):
        for f in range(d):
            if q % 2 == 0:
                out.append(index[((q * 7) % n_index) * d + f])
            else:
                out.append(bits_value(mix64(q, f, salt + 7), True))
    return out^


def ivf_query_source_row(q: Int, n_index: Int) -> Int:
    """Which index row an even query row copies. -1 for an odd row.

    Exported so a check states the expected answer from the same
    expression the fixture built it with, rather than from a second copy of
    `(q * 7) % n_index` that can drift out of step with this file.
    """
    if q % 2 != 0:
        return -1
    return (q * 7) % n_index


def ivf_duplicate_fixture(n: Int, d: Int, salt: Int) -> List[Float32]:
    """`n x d` rows in groups of three EXACT duplicates.

    Row `i` carries the hash of `i // 3`, so rows `3g`, `3g+1` and `3g+2`
    are bit-identical. Every distance is therefore attained by three
    distinct original indices and the tie class is reached on every query.
    """
    var out = List[Float32]()
    for i in range(n):
        for f in range(d):
            out.append(bits_value(mix64(i // 3, f, salt + 11), True))
    return out^


def ivf_duplicate_group(i: Int) -> Int:
    """The duplicate group row `i` belongs to. Its members are `3g`, `3g+1`
    and `3g+2`, and the lowest-index member is `3g`."""
    return i // 3


def ivf_signed_zero_fixture(n: Int, d: Int, salt: Int) -> List[Float32]:
    """`n x d` hashed rows whose column 0 is `+0.0` on even rows and
    `-0.0` on odd ones, and whose remaining columns repeat by PAIRS.

    Row `2g` and row `2g+1` therefore differ in NOTHING except the sign of
    a zero, which no arithmetic can separate: `+0.0 == -0.0` compares
    equal, `(+0.0)^2` and `(-0.0)^2` are both `+0.0`, and the expanded L2
    between them is exactly the clamped zero. They are an exact duplicate
    pair that only the index half of the key can order (row 39).
    """
    var out = List[Float32]()
    for i in range(n):
        for f in range(d):
            if f == 0:
                if i % 2 == 0:
                    out.append(bitcast[DType.float32](BITS_POS_ZERO))
                else:
                    out.append(bitcast[DType.float32](BITS_NEG_ZERO))
            else:
                out.append(bits_value(mix64(i // 2, f, salt + 13), True))
    return out^


def ivf_equidistant_centers(d: Int) raises -> List[Float32]:
    """Two centres, `+1` and `-1` in column 0, zero elsewhere.

    EXACT POWERS OF TWO. `(0 - 1)^2` and `(0 - (-1))^2` are both exactly
    `1.0` in float32, and the expanded form `||p||^2 + ||c||^2 - 2 p.c`
    gives `0.25 + 1 - 0` on both sides with every term exactly
    representable, so the tie is a real tie in any summation order on any
    backend. A fixture whose "tie" is two floats a few ulps apart is a
    fixture that tests the non-tie path.
    """
    if d < 2:
        raise Error(
            "ivf_equidistant_centers: needs d >= 2 so the point can sit off"
            " the axis the two centres differ on; got " + String(d)
        )
    var out = List[Float32]()
    for c in range(2):
        for f in range(d):
            if f == 0:
                if c == 0:
                    out.append(bitcast[DType.float32](BITS_ONE))
                else:
                    out.append(bitcast[DType.float32](BITS_NEG_ONE))
            else:
                out.append(bitcast[DType.float32](BITS_POS_ZERO))
    return out^


def ivf_equidistant_centers_reversed(d: Int) raises -> List[Float32]:
    """`ivf_equidistant_centers` with the two rows swapped.

    The point of the reversal is that it changes WHICH CENTROID carries
    list id 0 without changing the geometry at all, so a tie rule that
    reads the id gives the same LIST and a tie rule that reads arrival
    order, a pointer, or the argmin's fold shape does not.
    """
    var fwd = ivf_equidistant_centers(d)
    var out = List[Float32]()
    for f in range(d):
        out.append(fwd[d + f])
    for f in range(d):
        out.append(fwd[f])
    return out^


def ivf_equidistant_points(n: Int, d: Int, salt: Int) raises -> List[Float32]:
    """`n` points, every one of them EXACTLY equidistant from the two
    centres above: column 0 is `+0.0`, column 1 is `+0.5`, the rest zero.

    All `n` rows are identical on purpose. The check needs the tie to occur
    on every row so that "no tie occurred" cannot be mistaken for "the rule
    held", and it refuses itself if the two device distances are not bit-
    equal.
    """
    if d < 2:
        raise Error(
            "ivf_equidistant_points: needs d >= 2; got " + String(d)
        )
    _ = salt
    var out = List[Float32]()
    for _ in range(n):
        for f in range(d):
            if f == 1:
                out.append(bitcast[DType.float32](BITS_HALF))
            else:
                out.append(bitcast[DType.float32](BITS_POS_ZERO))
    return out^


def ivf_planted_labels(
    n: Int, n_lists: Int, empty_list: Int
) raises -> List[UInt32]:
    """Labels covering every list except `empty_list`, round-robin.

    The round robin is over the `n_lists - 1` non-empty ids in ascending
    order, so the resulting layout is a pure function of `(n, n_lists,
    empty_list)` and the check can state the expected offsets rather than
    read them back and agree with itself.
    """
    if n_lists < 2:
        raise Error(
            "ivf_planted_labels: needs n_lists >= 2 to leave one empty; got"
            " " + String(n_lists)
        )
    if empty_list < 0 or empty_list >= n_lists:
        raise Error(
            "ivf_planted_labels: empty_list "
            + String(empty_list)
            + " is outside [0, "
            + String(n_lists)
            + ")"
        )
    var live = List[UInt32]()
    for l in range(n_lists):
        if l != empty_list:
            live.append(UInt32(l))
    var out = List[UInt32]()
    for i in range(n):
        out.append(live[i % len(live)])
    return out^


def ivf_planted_centers(n_lists: Int, d: Int, salt: Int) -> List[Float32]:
    """Hashed centres for a hand-planted index.

    Used only where the LAYOUT is planted rather than fitted
    (`check_empty_list`, `check_index_carry`): the search's coarse step
    still needs a centre per list, and these are hashed for the same
    permutation-sensitivity reason every other fixture here is.
    """
    var out = List[Float32]()
    for l in range(n_lists):
        for f in range(d):
            out.append(bits_value(mix64(l, f, salt + 17), True))
    return out^
