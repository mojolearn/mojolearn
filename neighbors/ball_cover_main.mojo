# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the random ball cover checks."""

from neighbors.mojo_only.ball_cover_check import (
    check_ball_cover,
    check_ball_cover_at_scale,
    check_ball_cover_dense_and_max_k,
    check_ball_cover_max_k_wiring,
    check_ball_cover_order_is_load_bearing,
    check_ball_cover_reach_by_sabotage,
)


def main() raises:
    check_ball_cover()
    check_ball_cover_dense_and_max_k()
    check_ball_cover_reach_by_sabotage()
    check_ball_cover_order_is_load_bearing()
    check_ball_cover_at_scale()
    check_ball_cover_max_k_wiring()
