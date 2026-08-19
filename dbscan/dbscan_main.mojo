"""Entry point for the DBSCAN checks."""

from dbscan.mojo_only.dbscan_check import (
    check_dbscan,
    check_dbscan_eps_sensitivity,
)


def main() raises:
    check_dbscan()
    check_dbscan_eps_sensitivity()
