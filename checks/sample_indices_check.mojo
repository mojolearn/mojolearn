# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The border-sample draw against their `SampleIndices`.

WHAT THIS GATES. `gbdt/train.mojo`'s `sample_indices_for_borders`, which
is the sampling half of CatBoost's `GetSubsetForBuildBorders`
(`libs/data/quantization.cpp:118-141`) over `SampleIndices<ui32>`
(`libs/helpers/sample.h:20-43`). It IMPORTS that function rather than
re-typing it: a check that builds its own copy of the thing it checks
cannot catch the copy drifting, and this repository has been bitten by
exactly that (DEVIATION 115).

WHY IT EXISTS. Until 2026-08-22 this port drew 100,000 rows per float
column WITH REPLACEMENT and INDEPENDENTLY PER FEATURE, citing
`GetSampleSizeForBorderSelectionType` and `SampleArray` -- a real pair of
functions on the wrong code path, reached from `NCB::BuildBorders` rather
than from the quantizer. DEVIATION 135.

    pixi run check-sample-indices
"""

from std.math import log2
from gbdt.train import sample_indices_for_borders


def _distinct_and_repeats(
    idx: List[UInt32], nrr: Int
) raises -> Tuple[Int, Int, Int]:
    """(distinct, repeats, out-of-range) over the drawn indices."""
    var seen = List[Bool]()
    seen.resize(nrr, False)
    var dup = 0
    var oob = 0
    var distinct = 0
    for i in range(len(idx)):
        var v = Int(idx[i])
        if v < 0 or v >= nrr:
            oob += 1
            continue
        if seen[v]:
            dup += 1
        else:
            seen[v] = True
            distinct += 1
    _ = seen^
    return (distinct, dup, oob)


def _cell(name: String, nrr: Int, sn: Int, sd0: UInt64) raises -> Int:
    var idx = sample_indices_for_borders(nrr, sn, sd0)
    var fails = 0
    var t = _distinct_and_repeats(idx, nrr)
    var distinct = t[0]
    var dup = t[1]
    var oob = t[2]

    # S1 SIZE. Their `sampleSize` is exact, not approximate.
    if len(idx) != sn:
        print(
            "  FAIL " + name + " S1 size: " + String(len(idx))
            + " wanted " + String(sn)
        )
        fails += 1
    # S2 NO REPETITION. Their file says "Sample k element indices without
    # repetition"; the old draw repeated, and this is the gate that
    # separates the two.
    if dup != 0:
        print("  FAIL " + name + " S2 repetition: " + String(dup))
        fails += 1
    # S3 IN RANGE.
    if oob != 0:
        print("  FAIL " + name + " S3 out of [0,n): " + String(oob))
        fails += 1
    print(
        "  " + name + ": n=" + String(nrr) + " k=" + String(sn)
        + " -> drew " + String(len(idx)) + ", distinct " + String(distinct)
        + ", repeats " + String(dup) + ", out-of-range " + String(oob)
    )
    return fails


def check_sample_indices() raises:
    print("SAMPLE INDICES (their sample.h:20-43), DEVIATION 135:")
    var fails = 0

    # WHICH BRANCH each cell takes is THEIR predicate `k > 1 && k > (n /
    # log2(k))`, computed here so the cells are chosen on purpose rather
    # than by luck, and so a future change to the predicate shows up as a
    # branch label moving.
    #   464809 / 200000 -> 200000 > 26410  TRUE  -> partial Fisher-Yates
    #                      (covtype's row count at the shipped cap)
    #   1000000 / 50    -> 50 > 178571     FALSE -> rejection into a set
    #   1000 / 1000     -> n == k                -> iota
    print(" branch: partial Fisher-Yates")
    fails += _cell("covtype-shaped", 464809, 200000, UInt64(0))
    print(" branch: rejection into a set")
    fails += _cell("sparse draw", 1000000, 50, UInt64(0))
    print(" branch: n == k")
    fails += _cell("full", 1000, 1000, UInt64(0))

    # S4 DETERMINISM, PER POSITION. Comparing multisets would pass on a
    # permutation, which is the failure mode this repository has hit
    # before, so this compares placement.
    var a = sample_indices_for_borders(464809, 200000, UInt64(7))
    var b = sample_indices_for_borders(464809, 200000, UInt64(7))
    var moved = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            moved += 1
    if moved != 0:
        print("  FAIL S4 determinism: " + String(moved) + " positions differ")
        fails += 1
    else:
        print("  S4 determinism: " + String(len(a)) + " positions identical")

    # S5 THE SEED REACHES THE DRAW. Without this, S4 would also pass on a
    # function that ignored its seed entirely.
    var c = sample_indices_for_borders(464809, 200000, UInt64(8))
    var moved2 = 0
    for i in range(len(a)):
        if a[i] != c[i]:
            moved2 += 1
    if moved2 == 0:
        print("  FAIL S5 seed reach: a second seed drew the same sample")
        fails += 1
    else:
        print(
            "  S5 seed reach: " + String(moved2) + " of " + String(len(a))
            + " positions moved at a second seed"
        )

    # S6 THE COVERAGE THE OLD DRAW LOST, stated as an ANALYTIC identity
    # rather than a tally of ours. Drawing k times WITH replacement from
    # n covers n*(1 - (1-1/n)^k) distinct rows in expectation. At
    # covtype's 464,809 rows the old code's 100,000 draws reach about
    # 90,030 distinct rows; theirs reaches exactly 200,000. This gate
    # asserts only the part that is ours to hold: the new draw's distinct
    # count IS its k, with no shortfall at all.
    var d = sample_indices_for_borders(464809, 200000, UInt64(0))
    var t2 = _distinct_and_repeats(d, 464809)
    if t2[0] != 200000:
        print(
            "  FAIL S6 coverage: distinct " + String(t2[0])
            + " of 200000 drawn"
        )
        fails += 1
    else:
        print("  S6 coverage: 200000 of 200000 drawn rows are distinct")

    # S7 WHAT THIS GATE DOES NOT COVER, said out loud so nobody reads it
    # as covering more than it does: that ONE subset is shared by every
    # float column is a property of the CALLER, checked by reading
    # `gbdt/train.mojo` -- `sample_idx` is built once before `_draw_task`
    # and every column gathers `src[sidx[i]]`. And the SET is not
    # CatBoost's, because their engine is `TRestorableFastRng64` and ours
    # is `TRandom`; only the semantics match. Both are DEVIATION 135.

    if fails != 0:
        raise Error(String(fails) + " sample-index gate(s) failed")
    print("  all gates pass")


def main() raises:
    check_sample_indices()
