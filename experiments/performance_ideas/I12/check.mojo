# SPDX-License-Identifier: Apache-2.0
"""Focused independent coordinates, host iterate replay, launch geometry,
coefficient prediction, refusal names, and optimization quality gates.
Gram and speculative line-search profiles are intentionally independent
pending complete trial-vector evaluation/first-acceptance integration."""
from solver.checks.cd_check import check_cd_refuses_by_name, check_cd_recovers_the_planted_support, check_cd_device_equals_oracle, check_cd_is_launch_invariant, check_cd_elasticnet_arms_reach, check_cd_predict_matches_host

def main() raises:
    check_cd_refuses_by_name()
    check_cd_recovers_the_planted_support()
    check_cd_device_equals_oracle()
    check_cd_is_launch_invariant()
    check_cd_elasticnet_arms_reach()
    check_cd_predict_matches_host()
    print("I12 PASS accepted_coordinate_iterates prediction launch_and_arm_reach")
