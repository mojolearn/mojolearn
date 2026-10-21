# SPDX-License-Identifier: Apache-2.0
# Native whole HDBSCAN fit on7 fixtures of41–96 rows; consumed labels/outlier counts versus scalar oracle and independently planted partitions.
from hdbscan.checks.hdbscan_check import check_labels_vs_oracle as check_0


def main() raises:
    check_0()
    print("NATIVE_ONLY_PASS hdbscan")
