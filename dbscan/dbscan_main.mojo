# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the DBSCAN checks."""

from dbscan.mojo_only.dbscan_check import (
    check_dbscan,
    check_dbscan_eps_sensitivity,
    check_dbscan_batching_agrees,
    check_dbscan_max_mbytes_moves_the_batch,
    check_dbscan_rbc_matches_brute,
    check_dbscan_rbc_two_loop_arms,
    check_dbscan_tiny_budget_agrees,
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
