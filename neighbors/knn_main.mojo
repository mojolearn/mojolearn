"""Entry point for the brute-force k-NN checks.

The first three check their FALLBACK, `tiled_brute_force_knn`. The rest check
the path their dispatch actually takes for k <= 64 on row-major L2, which is
`fusedL2Knn` (`knn_brute_force.cuh:443`) with
`faiss_select::WarpSelect` in registers as its selector.

Then the caller-facing surface, and LAST the k-NN classifier and regressor
(`neighbors/mojo_only/knn_classify_check.mojo`, `knn_regress_check.mojo`,
2026-08-23), which vote and average over what `knn_search` returns and are
worth nothing if it is wrong. `pixi run check-knn` runs all of it; each of
those two files also has its own `main` for the loop while working on one:
`mojo run -I . neighbors/mojo_only/knn_classify_check.mojo`.
"""

from mojo_only.hardware_matrix_check import check_hardware_matrix
from neighbors.mojo_only.estimator_check import (
    check_knn_search_arms_agree,
    check_knn_search_matches_host,
    check_plan_query_tile,
)
from neighbors.mojo_only.knn_classify_check import (
    check_getuniquelabels,
    check_knn_classify_matches_host_transcription,
    check_knn_classify_multi_output_layout,
    check_knn_classify_reach_by_sabotage,
    check_knn_classify_run_twice_identical,
    check_knn_classify_ties_go_to_lowest_class,
)
from neighbors.mojo_only.knn_regress_check import (
    check_knn_regress_matches_host_transcription,
    check_knn_regress_multi_output_layout,
    check_knn_regress_reach_by_sabotage,
    check_knn_regress_run_twice_identical,
)
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
    # The caller-facing surface, last: it is the only thing here a user
    # can reach, and it is worth nothing if the kernels above are wrong.
    check_plan_query_tile()
    check_knn_search_matches_host()
    check_knn_search_arms_agree()
    # The classifier and regressor over that surface.
    check_getuniquelabels()
    check_knn_classify_matches_host_transcription()
    check_knn_classify_ties_go_to_lowest_class()
    check_knn_classify_reach_by_sabotage()
    check_knn_classify_multi_output_layout()
    check_knn_classify_run_twice_identical()
    check_knn_regress_matches_host_transcription()
    check_knn_regress_reach_by_sabotage()
    check_knn_regress_multi_output_layout()
    check_knn_regress_run_twice_identical()
