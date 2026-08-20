"""Entry point for the k-means checks.

Separate from the root `probe_main.mojo` on purpose. That file is the
boosting side's and is under active edit by another session in this shared
checkout; a second section adding imports to it would collide for no benefit.
"""

from cluster.mojo_only.kmeans_check import (
    check_assignment_arm_dispatch,
    check_device_inclusive_scan,
    check_fused_policy_dispatch,
    check_fused_reduction_across_lanes,
    check_kmeans_fit,
    check_kmeans_plus_plus_init,
    check_privatized_accumulate,
    check_reach_by_sabotage,
)


def main() raises:
    check_reach_by_sabotage()
    check_kmeans_fit()
    check_device_inclusive_scan()
    check_kmeans_plus_plus_init()
    check_fused_reduction_across_lanes()
    check_assignment_arm_dispatch()
    check_fused_policy_dispatch()
    check_privatized_accumulate()
