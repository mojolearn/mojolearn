# SPDX-License-Identifier: Apache-2.0
# Native single-linkage fit on5 fixtures; consumed dendrogram/labels versus independent Kruskal/cut partition plus actual entry/staged equality. Other linkage methods not exercised.
from hierarchy.checks.linkage_check import check_linkage_dendrogram_and_labels as check_0
from hierarchy.checks.linkage_check import check_linkage_entry_matches_stages as check_1


def main() raises:
    check_0()
    check_1()
    print("NATIVE_ONLY_PASS agglomerative")
