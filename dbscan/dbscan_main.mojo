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
    check_dbscan_weighted_fold_is_pinned()
    check_dbscan_manhattan_neighborhood()
    check_dbscan_manhattan_changes_the_labels()
    check_dbscan_manhattan_refused_on_the_ball_cover()
    # The weighted core-point test (DEVIATION 28) and its wiring
    # (DEVIATION 29).
    check_dbscan_weighted_degree_matches_host_oracle()
    check_dbscan_uniform_weight_matches_unweighted()
    check_dbscan_duplicate_equals_weight_two()
