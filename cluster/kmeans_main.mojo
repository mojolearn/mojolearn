"""Entry point for the k-means checks.

Separate from the root `probe_main.mojo` on purpose. That file is the
boosting side's and is under active edit by another session in this shared
checkout; a second section adding imports to it would collide for no benefit.
"""

from cluster.mojo_only.kmeans_check import (
    check_device_inclusive_scan,
    check_kmeans_fit,
    check_kmeans_plus_plus_init,
    check_reach_by_sabotage,
)


def main() raises:
    check_reach_by_sabotage()
    check_kmeans_fit()
    check_device_inclusive_scan()
    check_kmeans_plus_plus_init()
