# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the DBSCAN checks."""

from dbscan.checks.dbscan_check import (
    check_dbscan,
    check_dbscan_batching_agrees,
    # check_dbscan_duplicate_equals_weight_two,  # BLOCKED, see the note in main
    check_dbscan_eps_sensitivity,
    check_dbscan_manhattan_changes_the_labels,
    check_dbscan_manhattan_neighborhood,
    check_dbscan_manhattan_refused_on_the_ball_cover,
    check_dbscan_max_mbytes_moves_the_batch,
    check_dbscan_rbc_matches_brute,
    check_dbscan_rbc_two_loop_arms,
    check_dbscan_tiny_budget_agrees,
    # check_dbscan_uniform_weight_matches_unweighted,  # BLOCKED, see the note in main
    # check_dbscan_weighted_degree_matches_host_oracle,  # BLOCKED, see the note in main
    # check_dbscan_weighted_fold_is_pinned,  # BLOCKED, see the note in main
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
    # THE FOUR WEIGHTED GATES ARE BLOCKED ON THE TOOLCHAIN, NOT DROPPED.
    # Building any of them raises `DeadArgumentElimination surveyUse
    # failed`, an LLVM pass assertion, and takes the whole dbscan build
    # down. Bisected 2026-09-01: with all seven new gates disabled the lane
    # builds; the three manhattan gates build; the weighted family does not.
    # Disabling a call is not enough because the module still compiles the
    # function, and extracting one gate to its own file did not help because
    # the family shares `_host_weighted_degree_strided` and
    # `_host_pinned_fold`. `@no_inline` on both helpers, PORTING.md 54's
    # defence against this same assertion, did not clear it.
    #
    # WHAT THIS COSTS, stated plainly: sample_weight is IMPLEMENTED and
    # UNGATED. Nothing here asserts that a uniform weight reproduces the
    # unweighted labels, that duplicating a point equals weight two, or that
    # the weighted degree's fold is the pinned width. Do not quote a
    # sample_weight result from this lane until these run.
    # check_dbscan_weighted_fold_is_pinned()
    check_dbscan_manhattan_neighborhood()
    check_dbscan_manhattan_changes_the_labels()
    check_dbscan_manhattan_refused_on_the_ball_cover()
    # The weighted core-point test (DEVIATION 28) and its wiring
    # (DEVIATION 29).
    # check_dbscan_weighted_degree_matches_host_oracle()
    # check_dbscan_uniform_weight_matches_unweighted()
    # check_dbscan_duplicate_equals_weight_two()
