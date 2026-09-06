# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the ball cover's k-NN query and its metric arms.

ORDER IS DELIBERATE. `check_rbc_knn_prunes_work` runs FIRST, because every
equality arm after it is meaningless if the index is not actually pruning:
an exhaustive comparison against a disguised full scan passes and proves
nothing. The refusals run next, because an admitted non-metric would make
the exactness claims false rather than merely unmeasured. Then the
exhaustive arm, then the three sabotage arms that show it can fail.
"""

from neighbors.checks.ball_cover_knn_check import (
    check_rbc_eps_metric_is_reached,
    check_rbc_eps_metrics_match_host,
    check_rbc_knn_matches_brute_force,
    check_rbc_knn_prune_is_load_bearing,
    check_rbc_knn_prunes_work,
    check_rbc_knn_radius_is_read,
    check_rbc_knn_slack_costs_no_answer,
    check_rbc_metric_refusals,
)


def main() raises:
    check_rbc_knn_prunes_work()
    check_rbc_metric_refusals()
    check_rbc_knn_matches_brute_force()
    check_rbc_eps_metrics_match_host()
    check_rbc_eps_metric_is_reached()
    check_rbc_knn_radius_is_read()
    check_rbc_knn_slack_costs_no_answer()
    check_rbc_knn_prune_is_load_bearing()
