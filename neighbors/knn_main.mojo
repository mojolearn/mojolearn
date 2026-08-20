"""Entry point for the brute-force k-NN checks.

The first three check their FALLBACK, `tiled_brute_force_knn`. The rest check
the path their dispatch actually takes for k <= 64 on row-major L2, which is
`fusedL2Knn` (`knn_brute_force.cuh:443`) with
`faiss_select::WarpSelect` in registers as its selector.
"""

from mojo_only.hardware_matrix_check import check_hardware_matrix
from neighbors.mojo_only.knn_check import (
    check_dispatch_takes_fused,
    check_fused_edge_shapes,
    check_fused_griddimx_merge,
    check_fused_griddimx_one_capped_y,
    check_fused_k_ceiling,
    check_fused_l2_knn,
    check_fused_queue_reach_by_sabotage,
    check_fused_reach_by_sabotage,
    check_knn,
    check_knn_reach_by_sabotage,
    check_launch_config_values,
    check_vendor_topk_matches_ported,
)


def main() raises:
    check_hardware_matrix()
    check_knn()
    check_knn_reach_by_sabotage()
    check_vendor_topk_matches_ported()
    check_fused_l2_knn()
    check_fused_edge_shapes()
    check_fused_reach_by_sabotage()
    check_fused_queue_reach_by_sabotage()
    check_fused_k_ceiling()
    check_dispatch_takes_fused()
    check_launch_config_values()
    check_fused_griddimx_merge()
    check_fused_griddimx_one_capped_y()
