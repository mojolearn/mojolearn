"""Entry point for the DBSCAN checks."""

from dbscan.mojo_only.dbscan_check import (
    check_dbscan,
    check_dbscan_eps_sensitivity,
    check_dbscan_batching_agrees,
    check_exclusive_scan_beyond_the_old_cap,
)


def main() raises:
    check_dbscan()
    check_dbscan_eps_sensitivity()
    check_exclusive_scan_beyond_the_old_cap()
    check_dbscan_batching_agrees()
