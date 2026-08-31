# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SOAK the cell that produced DEVIATION 134's intermittent.

One sweep run in roughly a hundred came back with its FIRST cell wrong --
depth 1, 8 features -- on BOTH searchers, and the two searchers DISAGREED
WITH EACH OTHER:

    greedy    2/12 splits, first divergence tree 2, mse 4.679346084594727
    pointwise 8/12 splits, first divergence tree 8, mse 4.383606433868408
    CatBoost                                        4.133252577078921
    correct, every other run: 12/12 and 12/12, both arms 4.1332526206970215

`check-fit-pointwise` requires the two arms to be BIT-IDENTICAL. On that run
they were not. That is the observation this file exists to chase.

WHY A SOAK AND NOT ANOTHER GREEN RUN. The lane that found it re-ran the same
cell about a hundred times and never saw it again. A hundred clean runs of a
one-in-a-hundred event is not evidence of absence -- it is roughly a coin
flip. So this driver runs the cell hundreds of times per invocation, in ONE
process, and reports the DISTRIBUTION rather than a pass/fail: how many runs
disagreed with CatBoost, how many runs had the two arms disagree with EACH
OTHER, and every distinct loss value either arm produced.

The last of those is the discriminating one. If a run is ever wrong, the set
of distinct losses has more than one element, and the elements themselves say
whether the drift is one ULP (a float reduction reordering) or several
percent (a different tree). DEVIATION 134's observation was the latter.

WHAT IT DOES NOT DO: raise. A soak that stops at its first anomaly reports
one sample of the thing it was built to characterise. It exits non-zero at
the end if anything was seen, and prints everything either way.

    pixi run soak-determinism            # the default cell, 200 reps
    MOJOLEARN_SOAK_REPS=1000 pixi run soak-determinism
    MOJOLEARN_SOAK_FIXTURE=bench/oracle_d2_f8.txt pixi run soak-determinism

THE OTHER HALF OF THE EXPERIMENT is a build, not a flag: flip
`GLOBAL_NUMERIC_MODE` to `NUMERIC_IDENTICAL` in `mojo_only/numerics.mojo`
and run this again. Under IDENTICAL the greedy family's multi-block flush
sums through a fixed-point integer accumulator instead of a float
`atomicAdd`, so if the float atomic is the whole story the greedy arm goes
silent. **If the pointwise arm still drifts under IDENTICAL, there is a
second defect and it is not the histogram flush** -- which matters, because
the pointwise arm's flush is a plain store at this row count and no
mechanism for its half of that run is known.
"""

from std.os import getenv

from mojo_only.oracle_check import tree_structure_diff


def main() raises:
    var reps = 200
    var reps_s = getenv("MOJOLEARN_SOAK_REPS")
    if reps_s.byte_length() > 0:
        reps = Int(reps_s)
    var path = String("bench/oracle_d1_f8.txt")
    var path_s = getenv("MOJOLEARN_SOAK_FIXTURE")
    if path_s.byte_length() > 0:
        path = path_s

    print("SOAK of", path, "--", reps, "reps, both searchers, one process")
    print(
        "  chasing DEVIATION 134: a run where the two arms disagreed with"
        " each other"
    )
    print()

    # every distinct loss either arm produced, with a count, so a drift of
    # one ULP and a drift of several percent are told apart by reading them
    var seen_g = List[Float64]()
    var seen_g_n = List[Int]()
    var seen_p = List[Float64]()
    var seen_p_n = List[Int]()

    var bad_vs_catboost = 0
    var arms_disagreed = 0
    var first_bad = -1
    var their_mse = 0.0

    for r in range(reps):
        var g = tree_structure_diff(path, False, False, False)
        var p = tree_structure_diff(path, True, False, False)
        their_mse = g.their_mse

        var found = False
        for i in range(len(seen_g)):
            if seen_g[i] == g.our_mse:
                seen_g_n[i] += 1
                found = True
                break
        if not found:
            seen_g.append(g.our_mse)
            seen_g_n.append(1)

        found = False
        for i in range(len(seen_p)):
            if seen_p[i] == p.our_mse:
                seen_p_n[i] += 1
                found = True
                break
        if not found:
            seen_p.append(p.our_mse)
            seen_p_n.append(1)

        var g_off = g.matched != g.compared
        var p_off = p.matched != p.compared
        if g_off or p_off:
            bad_vs_catboost += 1
            if first_bad < 0:
                first_bad = r
            print(
                "  REP", r, "DISAGREES WITH CATBOOST:",
                "greedy", g.matched, "/", g.compared,
                "( first div", g.first_divergence, ")",
                " pointwise", p.matched, "/", p.compared,
                "( first div", p.first_divergence, ")",
            )
            print(
                "        greedy mse", g.our_mse,
                " pointwise mse", p.our_mse,
                " CatBoost", g.their_mse,
            )
        if g.our_mse != p.our_mse:
            arms_disagreed += 1
            print(
                "  REP", r,
                "THE TWO ARMS DISAGREE WITH EACH OTHER: greedy", g.our_mse,
                "pointwise", p.our_mse,
                "-- this is what check-fit-pointwise forbids",
            )

    print()
    print("  reps                       ", reps)
    print("  runs disagreeing with CatBoost", bad_vs_catboost)
    print("  runs where the ARMS disagreed ", arms_disagreed)
    print("  CatBoost's loss               ", their_mse)
    print("  distinct greedy losses:")
    for i in range(len(seen_g)):
        print("       ", seen_g[i], " x", seen_g_n[i])
    print("  distinct pointwise losses:")
    for i in range(len(seen_p)):
        print("       ", seen_p[i], " x", seen_p_n[i])

    if bad_vs_catboost == 0 and arms_disagreed == 0 and len(seen_g) == 1 and len(seen_p) == 1:
        print()
        print(
            "  CLEAN over", reps,
            "reps: one loss on each arm, both matching CatBoost.",
        )
        print(
            "  THIS IS NOT A PROOF OF ABSENCE. At the observed rate of about"
            " one in a hundred,"
        )
        print(
            "  a clean run of", reps, "carries roughly a",
            Int(100.0 * (0.99 ** Float64(reps))), "percent chance of having"
            " missed it.",
        )
        return

    print()
    print(
        "  REPRODUCED. first bad rep", first_bad,
        "-- DEVIATION 134 is live and this is the sample to work from.",
    )
    raise Error("determinism soak reproduced the intermittent")
