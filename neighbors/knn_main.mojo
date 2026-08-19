"""Entry point for the brute-force k-NN checks.

The first three check their FALLBACK, `tiled_brute_force_knn`. The last three
check the path their dispatch actually takes for k <= 64 on row-major L2,
which is `fusedL2Knn` (`knn_brute_force.cuh:443`).
"""

from neighbors.mojo_only.knn_check import (
    check_dispatch_takes_fused,
    check_fused_l2_knn,
    check_fused_reach_by_sabotage,
    check_knn,
    check_knn_reach_by_sabotage,
    check_vendor_topk_matches_ported,
)


def main() raises:
    check_knn()
    check_knn_reach_by_sabotage()
    check_vendor_topk_matches_ported()
    check_fused_l2_knn()
    check_fused_reach_by_sabotage()
    check_dispatch_takes_fused()
