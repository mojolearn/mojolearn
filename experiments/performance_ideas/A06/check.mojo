# SPDX-License-Identifier: Apache-2.0
"""AMD exact selector qualification: forced fused merge and caller batching.

The first gates force the real fused selector at multiple column grids and
check exact ordered (distance,index) ties. The full public knn_search sweeps
exercise neighboring low-dimensional widths and just outside that region.
"""
from neighbors.checks.knn_check import check_fused_griddimx_merge
from neighbors.checks.knn_identity_check import check_knn_fused_tie_set_is_geometry_invariant
from neighbors.checks.query_batch_check import _case
from checks.numerics import GLOBAL_NUMERIC_MODE,NUMERIC_IDENTICAL


# A06 experiment: NEVER RUN — PENDING MEASUREMENT; incumbent defaults retained.
# A06 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Explicit selector/low-dimensional public caller qualification; incumbent dispatch retained.
def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    check_fused_griddimx_merge()
    check_knn_fused_tie_set_is_geometry_invariant()
    var dims: List[Int] = [2,3,4,8,16]
    var ks: List[Int] = [1,8,17]
    for di in range(len(dims)):
        for ki in range(len(ks)):
            _case(769,65,dims[di],ks[ki])
    print("A06_PASS fused_multiblock_ties_and_15_public_query_cases")
