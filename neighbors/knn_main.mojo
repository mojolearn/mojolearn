# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the brute-force k-NN checks.

The first three check their FALLBACK, `tiled_brute_force_knn`. The rest check
the path their dispatch actually takes for k <= 64 on row-major L2, which is
`fusedL2Knn` (`knn_brute_force.cuh:443`) with
`faiss_select::WarpSelect` in registers as its selector.

Then the caller-facing surface, and LAST the k-NN classifier and regressor
(`neighbors/original/knn_classify_check.mojo`, `knn_regress_check.mojo`,
2026-08-23), which vote and average over what `knn_search` returns and are
worth nothing if it is wrong. `pixi run check-knn` runs all of it; each of
those two files also has its own `main` for the loop while working on one:
`mojo run -I . neighbors/original/knn_classify_check.mojo`.
"""

from original.hardware_matrix_check import check_hardware_matrix
from original.kernel_matrix import TARGET_COLUMN, lib_lane_width_for
from neighbors.original.estimator_check import (
    check_knn_search_arms_agree,
    check_knn_search_matches_host,
    check_plan_query_tile,
)
from neighbors.original.knn_classify_check import (
    check_getuniquelabels,
    check_knn_classify_matches_host_transcription,
    check_knn_classify_multi_output_layout,
    check_knn_classify_reach_by_sabotage,
    check_knn_classify_run_twice_identical,
    check_knn_classify_ties_go_to_lowest_class,
)
from neighbors.original.knn_regress_check import (
    check_knn_regress_matches_host_transcription,
    check_knn_regress_multi_output_layout,
    check_knn_regress_reach_by_sabotage,
    check_knn_regress_run_twice_identical,
)
from neighbors.original.knn_check import (
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
    # THE FUSED ARM EXISTS ONLY ON A 32-LANE COLUMN (IDENTITY_PATHS row
    # 23): `fused_l2_knn` refuses at entry on a 64-wide wavefront, by
    # design, and the eight checks below call it by name. On the MI325X
    # (2026-08-23, FAST leg) `check_fused_l2_knn` raised that refusal and
    # took the whole gate -- and the IDENTICAL pass the script runs after
    # it -- down with it, for an arm the column never ships. The refusal
    # is the measurement; record it once and run everything else.
    if lib_lane_width_for[TARGET_COLUMN]() == 32:
        check_fused_l2_knn()
        check_fused_edge_shapes()
        check_fused_reach_by_sabotage()
        check_fused_queue_reach_by_sabotage()
        check_fused_k_ceiling()
        check_dispatch_takes_fused()
        check_launch_config_values()
        check_fused_griddimx_merge()
        check_fused_griddimx_one_capped_y()
    else:
        print(
            "fused-arm checks RECORDED as REFUSED on this column: lane"
            " width "
            + String(lib_lane_width_for[TARGET_COLUMN]())
            + " != 32, so `fused_l2_knn` refuses at entry (row 23) and"
            " AUTO never selects it here (DEVIATION 512 under FAST, 509"
            " under IDENTICAL); the tiled arm above and the surface below"
            " are this column's whole k-NN."
        )
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
