# SPDX-License-Identifier: Apache-2.0
"""Retained 256-row accumulator is already promoted; test real fit interaction.

Three profiles isolate 256 (current), 1024 (old control), and4096 (existing
wide-feature alternative). Exact integer accumulator and bounds checks run
before complete assignment/fit traces and convergence interaction arms.
"""
from cluster.checks.kmeans_check import check_blocked_accumulate,check_assignment_arms_match_oracle
from cluster.checks.kmeans_identity_check import check_assignment_geometry_invariance,check_kmeans_trace_localizes
from cluster.checks.reduce_by_key import BLOCK_ACC_ROWS


# A08 experiment: NEVER RUN — PENDING VALIDATION; incumbent defaults retained.
# A08 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Incumbent promoted256 default retained; alternate chunk/poll arms require defines.
def main() raises:
    check_blocked_accumulate()
    check_assignment_arms_match_oracle()
    check_assignment_geometry_invariance()
    check_kmeans_trace_localizes()
    print("A08_PASS accumulator_rows="+String(BLOCK_ACC_ROWS)+" exact_accumulator_assignment_and_fit_trace")
