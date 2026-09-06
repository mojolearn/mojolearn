# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the DBSCAN checks."""

from dbscan.checks.dbscan_check import (
    check_dbscan,
    check_dbscan_batching_agrees,
    check_dbscan_duplicate_equals_weight_two,
    check_dbscan_eps_sensitivity,
    check_dbscan_manhattan_changes_the_labels,
    check_dbscan_manhattan_neighborhood,
    check_dbscan_manhattan_refused_on_the_ball_cover,
    check_dbscan_max_mbytes_moves_the_batch,
    check_dbscan_rbc_matches_brute,
    check_dbscan_rbc_two_loop_arms,
    check_dbscan_tiny_budget_agrees,
    check_dbscan_uniform_weight_matches_unweighted,
    check_dbscan_weighted_degree_matches_host_oracle,
    check_dbscan_weighted_fold_is_pinned,
    check_exclusive_scan_beyond_the_old_cap,
    check_fused_eps_agrees_with_materialized,
)


def main() raises:
    check_fused_eps_agrees_with_materialized()
    check_dbscan()
    check_dbscan_eps_sensitivity()
    check_exclusive_scan_beyond_the_old_cap()
    check_dbscan_batching_agrees()
    check_dbscan_rbc_matches_brute()
    check_dbscan_rbc_two_loop_arms()
    check_dbscan_max_mbytes_moves_the_batch()
    check_dbscan_tiny_budget_agrees()
    # The metric arm (DEVIATION 27). Cheapest first: the fold pin is a
    # comptime assertion plus a host fold, the neighborhood is one kernel,
    # and the label check is two fits.
    #
    # THE FOUR WEIGHTED GATES WERE BLOCKED ON THE TOOLCHAIN, NOT DROPPED,
    # AND THE BLOCK IS NOW CLEARED. Building any of them used to raise
    # `DeadArgumentElimination surveyUse failed`, an LLVM pass assertion,
    # and take the whole dbscan build down, which is why `sample_weight`
    # shipped IMPLEMENTED AND UNGATED.
    #
    # THE CURE IS THE OPTIMIZATION LEVEL, NOT THE SOURCE. `pixi run
    # check-dbscan` passes `-O1`. MEASURED on an Apple M4, 2026-09-01: -O3 and -O2 both assert, -O1 and -O0 both build, and `DeadArgumentElimination` is an -O2-and-above pass, so
    # nothing in this file was ever the trigger. Four candidate source
    # rewrites were tried first, on the theory that some construct here
    # surveyed badly, and EVERY ONE OF THEM STILL ASSERTED AT -O3:
    # replacing `vertex_deg_dispatch` with the parametric `vertex_deg_run`,
    # splitting `_fit_weighted`'s conditionally-dead argument into two
    # functions, hoisting a ternary out of an append, and copying a
    # `List[List[Int]]` element before passing it. The first of those has
    # been REVERTED, because it took the dispatcher out from under the gate
    # for no benefit.
    #
    # THE BISECT THAT FOUND IT, kept because the handle is reusable. The
    # IMPORT above is what pulls a gate into codegen, so an import and its
    # call must BOTH be commented to disable one. Enabling them one at a
    # time showed TWO independent triggers, not one: the fold gate alone
    # asserts, and the three fit gates alone assert. That is why a
    # single-candidate build that still crashed would have retired a good
    # fix, and it is why nothing was retired on one build.
    #
    # WHAT IT COST, in the past tense at last: from the day `sample_weight`
    # landed until 2026-09-01 nothing had asserted that a uniform weight
    # reproduces the unweighted labels, that duplicating a point equals
    # weight two, or that the weighted degree's fold is the pinned width.
    # All four now print OK. Apple M4 only; a three-vendor leg is owed.
    check_dbscan_weighted_fold_is_pinned()
    check_dbscan_manhattan_neighborhood()
    check_dbscan_manhattan_changes_the_labels()
    check_dbscan_manhattan_refused_on_the_ball_cover()
    # The weighted core-point test (DEVIATION 28) and its wiring
    # (DEVIATION 29).
    check_dbscan_weighted_degree_matches_host_oracle()
    check_dbscan_uniform_weight_matches_unweighted()
    check_dbscan_duplicate_equals_weight_two()
