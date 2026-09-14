# SPDX-License-Identifier: Apache-2.0
"""Existing DBSCAN oracle/branch gates with distributed neighborhoods enabled."""
from dbscan.checks.dbscan_check import (
    check_dbscan,
    check_dbscan_batching_agrees,
    check_dbscan_tiny_budget_agrees,
    check_dbscan_rbc_two_loop_arms,
    check_dbscan_manhattan_changes_the_labels,
    check_dbscan_weighted_degree_matches_host_oracle,
    check_dbscan_weighted_fold_is_pinned,
    check_dbscan_uniform_weight_matches_unweighted,
    check_dbscan_duplicate_equals_weight_two,
)


def main() raises:
    check_dbscan()
    check_dbscan_batching_agrees()
    check_dbscan_tiny_budget_agrees()
    check_dbscan_rbc_two_loop_arms()
    check_dbscan_manhattan_changes_the_labels()
    check_dbscan_weighted_degree_matches_host_oracle()
    check_dbscan_weighted_fold_is_pinned()
    check_dbscan_uniform_weight_matches_unweighted()
    check_dbscan_duplicate_equals_weight_two()
    print("DBSCAN neighborhood gates PASS")
