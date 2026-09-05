# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Refuse experimental layout activation outside IDENTICAL mode.

Build in FAST and DETERMINISTIC, with neither and both experimental defines.
The effective flags must remain false. Run the existing public estimator
host-reference, alternate-method and query-tile checks in each build.
This gate deliberately refuses IDENTICAL, whose four arms have a separate
bitwise qualification driver in knn_layout_dispatch_check.mojo.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from neighbors.impl.neighbors.detail.knn_brute_force import EXPERIMENTAL_SMALLK_IDENTICAL, EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL
from neighbors.checks.estimator_check import (
    check_knn_search_arms_agree,
    check_knn_search_matches_host,
    check_plan_query_tile,
)


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        raise Error("layout mode isolation gate requires FAST or DETERMINISTIC")
    if EXPERIMENTAL_SMALLK_IDENTICAL or EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL:
        raise Error("IDENTICAL k-NN experiment activated in another numeric mode")
    print("LAYOUT_MODE_FLAGS", "mode", numeric_mode_name(),
          "selector", Int(EXPERIMENTAL_SMALLK_IDENTICAL),
          "transpose", Int(EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL))
    check_plan_query_tile()
    check_knn_search_matches_host()
    check_knn_search_arms_agree()
    print("KNN LAYOUT MODE ISOLATION PASS")
