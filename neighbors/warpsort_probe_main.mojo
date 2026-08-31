# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the warp-sort top-k checks.

Separate from `knn_main.mojo` on purpose: warpsort is not wired into
`knn_brute_force.mojo` yet, and this main is what makes the port REACHABLE
without touching a file another lane is in.
"""

from neighbors.original.warpsort_check import (
    check_warpselect_matches_oracle,
    check_warpselect_reach_by_sabotage,
    check_warpsort_matches_radix,
    check_warpsort_reach_by_sabotage,
)


def main() raises:
    check_warpsort_matches_radix()
    check_warpsort_reach_by_sabotage()
    check_warpselect_matches_oracle()
    check_warpselect_reach_by_sabotage()
