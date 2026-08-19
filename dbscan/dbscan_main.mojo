"""Entry point for the DBSCAN checks."""

from dbscan.mojo_only.dbscan_check import (
    check_dbscan,
    check_dbscan_eps_sensitivity,
    check_dbscan_batching_agrees,
    check_dbscan_rbc_matches_brute,
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
