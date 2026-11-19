# SPDX-License-Identifier: Apache-2.0
# Standalone wrapper of `neighbors/checks/knn_identity_check_suite.mojo`: the checks, fixtures and helpers live there
# (a module with main() cannot be imported; split by lane fx-checks-orphans, 2026-11-18).
# Import from the suite, never from this file.
from neighbors.checks.knn_identity_check_suite import (
    check_knn_tie_set_is_lowest_index,
    check_knn_tiled_is_query_tile_invariant,
    check_knn_fused_tie_set_is_geometry_invariant,
    check_knn_auto_arm_is_pinned,
)


def main() raises:
    check_knn_tie_set_is_lowest_index()
    check_knn_tiled_is_query_tile_invariant()
    check_knn_fused_tie_set_is_geometry_invariant()
    check_knn_auto_arm_is_pinned()
