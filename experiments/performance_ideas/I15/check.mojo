# SPDX-License-Identifier: Apache-2.0
"""Exercise public exact queries and forced hierarchical candidate merges.
Every distance word and original index is checked; duplicate tie sets and
geometries must agree. Existing certified bound-compaction gate is also
run, so inconclusive bound behavior retains exact candidate semantics."""
from neighbors.checks.query_batch_check import _case
from neighbors.checks.knn_check import check_fused_griddimx_merge
from neighbors.checks.knn_identity_check import check_knn_fused_tie_set_is_geometry_invariant
from neighbors.checks.knn_selector_bound_compact_check import main as bound_check

def main() raises:
    for n in [255, 257, 1031]:
        for k in [1, 17, 129]:
            _case(n, 37, 19, k)
    check_fused_griddimx_merge()
    check_knn_fused_tie_set_is_geometry_invariant()
    bound_check()
    print("I15 PASS public_queries=9 hierarchical_merge certified_bound_compaction")
