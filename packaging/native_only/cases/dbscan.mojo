# SPDX-License-Identifier: Apache-2.0
# Native DBSCAN fit graph/propagation/final relabel and consumed planted partition/noise labels.
from dbscan.checks.dbscan_check import check_dbscan


def main() raises:
    check_dbscan()
    print("NATIVE_ONLY_PASS dbscan")
