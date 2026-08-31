# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the k-means checks.

Separate from the root `probe_main.mojo` on purpose. That file is the
boosting side's and is under active edit by another session in this shared
checkout; a second section adding imports to it would collide for no benefit.
"""

from cluster.checks.estimator_check import (
    check_kmeans_fit_recovers_planted,
    check_kmeans_fit_rejects_bad_shapes,
    check_kmeans_fit_weight_arms_agree,
    check_plan_sum_scale,
)
from cluster.checks.kmeans_check import (
    check_accumulate_veclen_dispatch,
    check_assignment_arm_dispatch,
    check_assignment_arms_match_oracle,
    check_device_inclusive_scan,
    check_fused_policy_dispatch,
    check_fused_reduction_across_lanes,
    check_kmeans_fit,
    check_kmeans_plus_plus_init,
    check_privatized_accumulate,
    check_reach_by_sabotage,
    check_scalable_kmeans_plus_plus_init,
    check_scalable_sampling_selection,
    check_scalable_supplement_branch,
)


def main() raises:
    check_reach_by_sabotage()
    check_kmeans_fit()
    check_device_inclusive_scan()
    check_kmeans_plus_plus_init()
    check_scalable_sampling_selection()
    check_scalable_kmeans_plus_plus_init()
    check_scalable_supplement_branch()
    check_fused_reduction_across_lanes()
    check_assignment_arm_dispatch()
    check_assignment_arms_match_oracle()
    check_fused_policy_dispatch()
    check_privatized_accumulate()
    check_accumulate_veclen_dispatch()

    # The caller-facing surface, last: it is the only thing here a user can
    # reach, and it is worth nothing if the kernels above are wrong.
    check_plan_sum_scale()
    check_kmeans_fit_recovers_planted()
    check_kmeans_fit_weight_arms_agree()
    check_kmeans_fit_rejects_bad_shapes()
